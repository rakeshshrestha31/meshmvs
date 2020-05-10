#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import open3d as o3d
import logging
import os
import shutil
import time
import numpy as np
import tqdm

import detectron2.utils.comm as comm
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.logger import setup_logger
from fvcore.common.file_io import PathManager
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import save_obj

from shapenet.config import get_shapenet_cfg
from shapenet.data import build_data_loader, register_shapenet
from shapenet.evaluation import evaluate_split, evaluate_test, evaluate_test_p2m

# required so that .register() calls are executed in module scope
from shapenet.modeling import MeshLoss, build_model
from shapenet.modeling.heads.depth_loss import adaptive_berhu_loss
from shapenet.modeling.mesh_arch import VoxMeshMultiViewHead, VoxMeshDepthHead
from shapenet.solver import build_lr_scheduler, build_optimizer
from shapenet.utils import Checkpoint, Timer, clean_state_dict, default_argument_parser
from shapenet.utils.depth_backprojection import get_points_from_depths
from meshrcnn.utils.metrics import compare_meshes

logger = logging.getLogger("shapenet")
P2M_SCALE = 0.57
NUM_PRED_SURFACE_SAMPLES = 6466
NUM_GT_SURFACE_SAMPLES = 6466


def copy_data(args):
    data_base, data_ext = os.path.splitext(os.path.basename(args.data_dir))
    assert data_ext in [".tar", ".zip"]
    t0 = time.time()
    logger.info("Copying %s to %s ..." % (args.data_dir, args.tmp_dir))
    data_tmp = shutil.copy(args.data_dir, args.tmp_dir)
    t1 = time.time()
    logger.info("Copying took %fs" % (t1 - t0))
    logger.info("Unpacking %s ..." % data_tmp)
    shutil.unpack_archive(data_tmp, args.tmp_dir)
    t2 = time.time()
    logger.info("Unpacking took %f" % (t2 - t1))
    args.data_dir = os.path.join(args.tmp_dir, data_base)
    logger.info("args.data_dir = %s" % args.data_dir)


def get_dataset_name(cfg):
    if cfg.DATASETS.TYPE.lower() == "multi_view":
        return "MeshVoxMultiView"
    elif cfg.DATASETS.TYPE.lower() == "depth":
        return "MeshVoxDepth"
    elif cfg.DATASETS.TYPE.lower() == "single_view":
        return "MeshVox"
    else:
        print("unrecognized dataset type", cfg.DATASETS.TYPE)
        exit(1)


def main_worker_eval(worker_id, args):

    device = torch.device("cuda:%d" % worker_id)
    cfg = setup(args)

    # build test set
    test_loader = build_data_loader(
        cfg, get_dataset_name(cfg), "test", multigpu=False
    )
    logger.info("test - %d" % len(test_loader))

    # load checkpoing and build model
    if cfg.MODEL.CHECKPOINT == "":
        raise ValueError("Invalid checkpoing provided")
    logger.info("Loading model from checkpoint: %s" % (cfg.MODEL.CHECKPOINT))
    cp = torch.load(PathManager.get_local_path(cfg.MODEL.CHECKPOINT))
    state_dict = clean_state_dict(cp["best_states"]["model"])
    model = build_model(cfg)
    model.load_state_dict(state_dict)
    logger.info("Model loaded")
    model.to(device)

    prediction_dir = os.path.join(
        cfg.OUTPUT_DIR, "predictions", "eval", "predict", "0"
    )
    save_predictions(model, test_loader, prediction_dir)
    exit(0)

    if args.eval_p2m:
        evaluate_test_p2m(model, test_loader)
    else:
        evaluate_test(model, test_loader)


def main_worker(worker_id, args):
    distributed = False
    if args.num_gpus > 1:
        distributed = True
        dist.init_process_group(
            backend="NCCL", init_method=args.dist_url, world_size=args.num_gpus, rank=worker_id
        )
        torch.cuda.set_device(worker_id)

    device = torch.device("cuda:%d" % worker_id)

    cfg = setup(args)

    # data loaders
    loaders = setup_loaders(cfg)
    for split_name, loader in loaders.items():
        logger.info("%s - %d" % (split_name, len(loader)))

    # build the model
    model = build_model(cfg)
    model.to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[worker_id],
            output_device=worker_id,
            check_reduction=True,
            broadcast_buffers=False,
            # find_unused_parameters=True,
        )

    optimizer = build_optimizer(cfg, model)
    cfg.SOLVER.COMPUTED_MAX_ITERS = cfg.SOLVER.NUM_EPOCHS * len(loaders["train"])
    scheduler = build_lr_scheduler(cfg, optimizer)

    loss_fn_kwargs = {
        "chamfer_weight": cfg.MODEL.MESH_HEAD.CHAMFER_LOSS_WEIGHT,
        "normal_weight": cfg.MODEL.MESH_HEAD.NORMALS_LOSS_WEIGHT,
        "edge_weight": cfg.MODEL.MESH_HEAD.EDGE_LOSS_WEIGHT,
        "voxel_weight": cfg.MODEL.VOXEL_HEAD.LOSS_WEIGHT,
        "gt_num_samples": cfg.MODEL.MESH_HEAD.GT_NUM_SAMPLES,
        "pred_num_samples": cfg.MODEL.MESH_HEAD.PRED_NUM_SAMPLES,
        "upsample_pred_mesh": cfg.MODEL.MESH_HEAD.UPSAMPLE_PRED_MESH,
    }
    loss_fn = MeshLoss(**loss_fn_kwargs)

    checkpoint_path = "checkpoint.pt"
    checkpoint_path = os.path.join(cfg.OUTPUT_DIR, checkpoint_path)
    cp = Checkpoint(checkpoint_path)
    if len(cp.restarts) == 0:
        # We are starting from scratch, so store some initial data in cp
        iter_per_epoch = len(loaders["train"])
        cp.store_data("iter_per_epoch", iter_per_epoch)
    else:
        logger.info("Loading model state from checkpoint")
        model.load_state_dict(cp.latest_states["model"])
        optimizer.load_state_dict(cp.latest_states["optim"])
        scheduler.load_state_dict(cp.latest_states["lr_scheduler"])

    training_loop(cfg, cp, model, optimizer, scheduler, loaders, device, loss_fn)


def training_loop(cfg, cp, model, optimizer, scheduler, loaders, device, loss_fn):
    Timer.timing = False
    iteration_timer = Timer("Iteration")

    # model.parameters() is surprisingly expensive at 150ms, so cache it
    if hasattr(model, "module"):
        params = list(model.module.parameters())
    else:
        params = list(model.parameters())
    loss_moving_average = cp.data.get("loss_moving_average", None)
    while cp.epoch < cfg.SOLVER.NUM_EPOCHS:
        if comm.is_main_process():
            logger.info("Starting epoch %d / %d" % (cp.epoch + 1, cfg.SOLVER.NUM_EPOCHS))

        # When using a DistributedSampler we need to manually set the epoch so that
        # the data is shuffled differently at each epoch
        for loader in loaders.values():
            if hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(cp.epoch)

        for i, batch in enumerate(loaders["train"]):
            if i == 0:
                iteration_timer.start()
            else:
                iteration_timer.tick()
            batch = loaders["train"].postprocess(batch, device)

            num_infinite_params = 0
            for p in params:
                num_infinite_params += (torch.isfinite(p.data) == 0).sum().item()
            if num_infinite_params > 0:
                msg = "ERROR: Model has %d non-finite params (before forward!)"
                logger.info(msg % num_infinite_params)
                return

            model_kwargs = {}
            if cfg.MODEL.VOXEL_ON and cp.t < cfg.MODEL.VOXEL_HEAD.VOXEL_ONLY_ITERS:
                model_kwargs["voxel_only"] = True

            module = model.module if hasattr(model, "module") else model
            if type(module) in [VoxMeshMultiViewHead, VoxMeshDepthHead]:
                model_kwargs["intrinsics"] = batch["intrinsics"]
                model_kwargs["extrinsics"] = batch["extrinsics"]
            if type(module) == VoxMeshDepthHead:
                model_kwargs["masks"] = batch["masks"]
            with Timer("Forward"):
                model_outputs = model(batch["imgs"], **model_kwargs)
                voxel_scores = model_outputs["voxel_scores"]
                meshes_pred = model_outputs["meshes_pred"]
                merged_voxel_scores = model_outputs.get(
                    "merged_voxel_scores", None
                )

            num_infinite = 0
            for cur_meshes in meshes_pred:
                cur_verts = cur_meshes.verts_packed()
                num_infinite += (torch.isfinite(cur_verts) == 0).sum().item()
            if num_infinite > 0:
                logger.info("ERROR: Got %d non-finite verts" % num_infinite)
                return

            loss, losses = None, {}
            if num_infinite == 0:
                loss, losses = loss_fn(
                    voxel_scores, merged_voxel_scores,
                    meshes_pred, batch["voxels"],
                    (batch["points"], batch["normals"])
                )

            skip = loss is None
            if loss is None or (torch.isfinite(loss) == 0).sum().item() > 0:
                logger.info("WARNING: Got non-finite loss %f" % loss)
                skip = True
            # depth losses
            elif "depths" in batch:
                if "pred_depths" in model_outputs:
                    depth_loss = adaptive_berhu_loss(
                        batch["depths"], model_outputs["pred_depths"],
                        batch["masks"]
                    )
                    if not torch.any(torch.isnan(depth_loss)):
                        loss = loss \
                             + (depth_loss * cfg.MODEL.MVSNET.PRED_DEPTH_WEIGHT)
                    losses["pred_depth_loss"] = depth_loss
                if "rendered_depths" in model_outputs \
                        and not model_kwargs.get("voxel_only", False):
                    pred_depths = model_outputs["pred_depths"]
                    masks = batch["masks"]
                    all_ones_masks = torch.ones_like(masks)
                    resized_masks = F.interpolate(
                        masks.view(-1, 1, *(masks.shape[2:])),
                        pred_depths.shape[-2:], mode="nearest"
                    ).view(*(masks.shape[:2]), *(pred_depths.shape[-2:]))
                    masked_depths = pred_depths * resized_masks
                    for depth_idx, rendered_depth in \
                            enumerate(model_outputs["rendered_depths"]):
                        rendered_depth_loss = adaptive_berhu_loss(
                            masked_depths, rendered_depth, all_ones_masks
                        )
                        if not torch.any(torch.isnan(rendered_depth_loss)):
                            loss = loss \
                                 + (rendered_depth_loss \
                                    * cfg.MODEL.MVSNET.RENDERED_DEPTH_WEIGHT)
                        losses["rendered_depth_loss_%d" % depth_idx] \
                                = rendered_depth_loss

                        # rendered vs GT depth loss, only for debug
                        rendered_gt_depth_loss = adaptive_berhu_loss(
                            batch["depths"], rendered_depth, all_ones_masks
                        )
                        losses["rendered_gt_depth_loss_%d" % depth_idx] \
                                = rendered_gt_depth_loss

            if model_kwargs.get("voxel_only", False):
                for k, v in losses.items():
                    if "voxel" not in k:
                        losses[k] = 0.0 * v

            if loss is not None and cp.t % cfg.SOLVER.LOGGING_PERIOD == 0:
                if comm.is_main_process():
                    cp.store_metric(loss=loss.item())
                    str_out = "Iteration: %d, epoch: %d, lr: %.5f," % (
                        cp.t,
                        cp.epoch,
                        optimizer.param_groups[0]["lr"],
                    )
                    for k, v in losses.items():
                        str_out += "  %s loss: %.4f," % (k, v.item())
                    str_out += "  total loss: %.4f," % loss.item()

                    # memory allocaged
                    if torch.cuda.is_available():
                        max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                        str_out += " mem: %d" % max_mem_mb

                    if len(meshes_pred) > 0:
                        mean_V = meshes_pred[-1].num_verts_per_mesh().float().mean().item()
                        mean_F = meshes_pred[-1].num_faces_per_mesh().float().mean().item()
                        str_out += ", mesh size = (%d, %d)" % (mean_V, mean_F)
                    logger.info(str_out)

            if loss_moving_average is None and loss is not None:
                loss_moving_average = loss.item()

            # Skip backprop for this batch if the loss is above the skip factor times
            # the moving average for losses
            if loss is None:
                pass
            elif loss.item() > cfg.SOLVER.SKIP_LOSS_THRESH * loss_moving_average:
                logger.info("Warning: Skipping loss %f on GPU %d" % (loss.item(), comm.get_rank()))
                cp.store_metric(losses_skipped=loss.item())
                skip = True
            else:
                # Update the moving average of our loss
                gamma = cfg.SOLVER.LOSS_SKIP_GAMMA
                loss_moving_average *= gamma
                loss_moving_average += (1.0 - gamma) * loss.item()
                cp.store_data("loss_moving_average", loss_moving_average)

            if skip:
                logger.info("Dummy backprop on GPU %d" % comm.get_rank())
                loss = 0.0 * sum(p.sum() for p in params)

            # Backprop and step
            scheduler.step()
            optimizer.zero_grad()
            with Timer("Backward"):
                loss.backward()

            # When training with normal loss, sometimes I get NaNs in gradient that
            # cause the model to explode. Check for this before performing a gradient
            # update. This is safe in mult-GPU since gradients have already been
            # summed, so each GPU has the same gradients.
            num_infinite_grad = 0
            for p in params:
                if p.grad is not None:
                    num_infinite_grad += (torch.isfinite(p.grad) == 0).sum() \
                                                                      .item()
            if num_infinite_grad == 0:
                optimizer.step()
            else:
                msg = "WARNING: Got %d non-finite elements in gradient; skipping update"
                logger.info(msg % num_infinite_grad)
            cp.step()

        cp.step_epoch()
        eval_and_save(
            model, loaders, optimizer, scheduler, cp,
            cfg.SOLVER.EARLY_STOP_METRIC
        )

    if comm.is_main_process():
        logger.info("Evaluating on test set:")
        test_loader = build_data_loader(
            cfg, get_dataset_name(cfg), "test", multigpu=False
        )
        evaluate_test(model, test_loader)


def eval_and_save(
    model, loaders, optimizer, scheduler, cp, early_stop_metric
):
    # NOTE(gkioxari) For now only do evaluation on the main process
    if comm.is_main_process():
        logger.info("Evaluating on training set:")
        train_metrics, train_preds = evaluate_split(
            model, loaders["train_eval"], prefix="train_", max_predictions=1000
        )
        eval_split = "val"
        if eval_split not in loaders:
            logger.info("WARNING: No val set!!! Computing metrics on test set!")
            eval_split = "test"
        logger.info("Evaluating on %s set:" % eval_split)
        test_metrics, test_preds = evaluate_split(
            model, loaders[eval_split], prefix="%s_" % eval_split, max_predictions=1000
        )
        str_out = "Results on train: "
        for k, v in train_metrics.items():
            str_out += "%s %.4f " % (k, v)
        logger.info(str_out)
        str_out = "Results on %s: " % eval_split
        for k, v in test_metrics.items():
            str_out += "%s %.4f " % (k, v)
        logger.info(str_out)

        # The main process is responsible for managing the checkpoint
        # TODO(gkioxari) revisit these stores
        """
        cp.store_metric(**train_preds)
        cp.store_metric(**test_preds)
        """
        cp.store_metric(**train_metrics)
        cp.store_metric(**test_metrics)
        cp.early_stop_metric = eval_split + "_" + early_stop_metric

        cp.store_state("model", model.state_dict())
        cp.store_state("optim", optimizer.state_dict())
        cp.store_state("lr_scheduler", scheduler.state_dict())
        cp.save()

    # Since evaluation and checkpointing only happens on the main process,
    # make all processes wait
    if comm.get_world_size() > 1:
        dist.barrier()


@torch.no_grad()
def save_predictions(model, loader, output_dir):
    """
    This function is used save predicted and gt meshes
    """
    # Note that all eval runs on main process
    assert comm.is_main_process()
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    device = torch.device("cuda:0")
    os.makedirs(output_dir, exist_ok=True)

    for batch in tqdm.tqdm(loader):
        batch = loader.postprocess(batch, device)
        model_kwargs = {}
        module = model.module if hasattr(model, "module") else model
        if type(module) in [VoxMeshMultiViewHead, VoxMeshDepthHead]:
            model_kwargs["intrinsics"] = batch["intrinsics"]
            model_kwargs["extrinsics"] = batch["extrinsics"]
        if type(module) == VoxMeshDepthHead:
            model_kwargs["masks"] = batch["masks"]
        model_outputs = model(batch["imgs"], **model_kwargs)

        # TODO: debug only
        # save_debug_predictions(batch, model_outputs)

        pred_mesh = model_outputs["meshes_pred"][-1]
        gt_mesh = batch["meshes"]
        pred_mesh = pred_mesh.scale_verts(P2M_SCALE)
        gt_mesh = gt_mesh.scale_verts(P2M_SCALE)

        pred_points = sample_points_from_meshes(
            pred_mesh, NUM_PRED_SURFACE_SAMPLES, return_normals=False
        )
        gt_points = sample_points_from_meshes(
            gt_mesh, NUM_GT_SURFACE_SAMPLES, return_normals=False
        )

        pred_points = pred_points.cpu().detach().numpy()
        gt_points = gt_points.cpu().detach().numpy()

        batch_size = pred_points.shape[0]
        for batch_idx in range(batch_size):
            label, label_appendix = batch["id_strs"][batch_idx].split("-")[:2]
            pred_filename = os.path.join(
                output_dir, "{}_{}_predict.xyz".format(label, label_appendix)
            )
            gt_filename = os.path.join(
                output_dir, "{}_{}_ground.xyz".format(label, label_appendix)
            )

            np.savetxt(pred_filename, pred_points[batch_idx])
            np.savetxt(gt_filename, gt_points[batch_idx])

            # pred_filename = pred_filename.replace(".xyz", ".obj")
            # gt_filename = gt_filename.replace(".xyz", ".obj")

            # pred_verts, pred_faces = pred_mesh[batch_idx] \
            #                             .get_mesh_verts_faces(0)
            # gt_verts, gt_faces = gt_mesh[batch_idx] \
            #                         .get_mesh_verts_faces(0)
            # save_obj(pred_filename, pred_verts, pred_faces)
            # save_obj(gt_filename, gt_verts, gt_faces)

            # metrics = compare_meshes(
            #     pred_mesh[batch_idx], gt_mesh[batch_idx],
            #     num_samples=NUM_GT_SURFACE_SAMPLES, scale=1.0,
            #     thresholds=[0.01, 0.014142], reduce=True
            # )
            # print("%s_%s: %r" % (label, label_appendix, metrics))


@torch.no_grad()
def save_debug_predictions(batch, model_outputs):
    """
    save voxels and depths
    """
    from shapenet.modeling.voxel_ops import cubify
    from shapenet.modeling.mesh_arch import save_depths

    batch_size = len(batch["id_strs"])
    for view_idx, voxels in enumerate(model_outputs["voxel_scores"]):
        cubified = cubify(voxels, 48, 0.2)
        for batch_idx in range(batch_size):
            label, label_appendix = batch["id_strs"][batch_idx].split("-")[:2]
            save_obj(
                "/tmp/{}_{}_{}_multiview_vox.obj".format(
                    label, label_appendix, view_idx
                ),
                cubified[batch_idx].verts_packed(),
                cubified[batch_idx].faces_packed()
            )
    merged_voxels = cubify(model_outputs["merged_voxel_scores"], 48, 0.2)
    for batch_idx in range(batch_size):
        label, label_appendix = batch["id_strs"][batch_idx].split("-")[:2]
        save_obj(
            "/tmp/{}_{}_merged_vox.obj".format(
                label, label_appendix
            ),
            merged_voxels[batch_idx].verts_packed(),
            merged_voxels[batch_idx].faces_packed()
        )

    for stage_idx, pred_mesh in enumerate(model_outputs["meshes_pred"]):
        for batch_idx in range(batch_size):
            label, label_appendix = batch["id_strs"][batch_idx].split("-")[:2]
            view_weights = model_outputs["view_weights"][stage_idx][batch_idx]
            view_weights = F.normalize(view_weights, dim=-1).squeeze(1)
            save_obj("/tmp/{}_{}_{}_pred_mesh.obj".format(
                label, label_appendix, stage_idx
            ), pred_mesh[0].verts_packed(), pred_mesh[0].faces_packed())
            point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
                pred_mesh[0].verts_packed().cpu().numpy()
            ))
            point_cloud.colors = o3d.utility.Vector3dVector(
                view_weights.detach().cpu().numpy()
            )
            o3d.io.write_point_cloud("/tmp/{}_{}_{}_pred_cloud.ply".format(
                label, label_appendix, stage_idx
            ), point_cloud)

    if "pred_depths" in model_outputs:
        masks = F.interpolate(
            batch["masks"], model_outputs["pred_depths"].shape[-2:],
            mode="nearest"
        )
        masked_depths = model_outputs["pred_depths"] * masks
        # TODO: the labels won't be corrent when batch size > 1. Fix it
        save_depths(
            masked_depths,
            "pred_{}_{}".format(label, label_appendix), (137, 137)
        )
        save_backproj_depths(masked_depths, batch["id_strs"], "depth_cloud")

    if "rendered_depths" in model_outputs:
        # TODO: the labels won't be corrent when batch size > 1. Fix it
        for stage_idx, depth in enumerate(model_outputs["rendered_depths"]):
            save_depths(
                depth,
                "rendered_{}_{}_{}".format(label, label_appendix, stage_idx),
                (137, 137)
            )
            save_backproj_depths(
                depth, batch["id_strs"], "rendered_cloud_{}".format(stage_idx)
            )


@torch.no_grad()
def save_backproj_depths(depths, id_strs, prefix):
    depths = F.interpolate(depths, (224, 224))
    dtype = depths.dtype
    device = depths.device
    intrinsics = torch.tensor([
        [248.0, 0.0, 111.5], [0.0, 248.0, 111.5], [0.0, 0.0, 1.0]
    ], dtype=dtype, device=device)
    depth_points = get_points_from_depths(depths, intrinsics)

    for batch_idx in range(len(depth_points)):
        label, label_appendix = id_strs[batch_idx].split("-")[:2]
        for view_idx in range(len(depth_points[batch_idx])):
            points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
                depth_points[batch_idx][view_idx].detach().cpu().numpy()
            ))
            o3d.io.write_point_cloud("/tmp/{}_{}_{}_{}.ply".format(
                prefix, label, label_appendix, view_idx
            ), points)


def setup_loaders(cfg):
    loaders = {}
    loaders["train"] = build_data_loader(
        cfg, get_dataset_name(cfg), "train", multigpu=comm.get_world_size() > 1
    )

    # Since sampling the mesh is now coupled with the data loader, we need to
    # make two different Dataset / DataLoaders for the training set: one for
    # training which uses precomputd samples, and one for evaluation which uses
    # more samples and computes them on the fly. This is sort of gross.
    loaders["train_eval"] = build_data_loader(
        cfg, get_dataset_name(cfg), "train_eval", multigpu=False
    )

    loaders["val"] = build_data_loader(
        cfg, get_dataset_name(cfg), "val", multigpu=False
    )
    return loaders


def setup(args):
    """
    Create configs and setup logger from arguments and the given config file.
    """
    cfg = get_shapenet_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # register dataset
    data_dir, splits_file = register_shapenet(cfg.DATASETS.NAME)
    cfg.DATASETS.DATA_DIR = data_dir
    cfg.DATASETS.SPLITS_FILE = splits_file
    # if data was copied the data dir has changed
    if args.copy_data:
        cfg.DATASETS.DATA_DIR = args.data_dir
    cfg.freeze()

    colorful_logging = not args.no_color
    output_dir = cfg.OUTPUT_DIR
    if comm.is_main_process() and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    comm.synchronize()

    logger = setup_logger(
        output_dir, color=colorful_logging, name="shapenet", distributed_rank=comm.get_rank()
    )
    logger.info(
        "Using {} GPUs per machine. Rank of current process: {}".format(
            args.num_gpus, comm.get_rank()
        )
    )
    logger.info(args)

    logger.info("Environment info:\n" + collect_env_info())
    logger.info(
        "Loaded config file {}:\n{}".format(args.config_file, open(args.config_file, "r").read())
    )
    logger.info("Running with full config:\n{}".format(cfg))
    if comm.is_main_process() and output_dir:
        path = os.path.join(output_dir, "config.yaml")
        with open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(os.path.abspath(path)))
    return cfg


def shapenet_launch():
    args = default_argument_parser()

    # Note we need this only for pretrained models with torchvision.
    os.environ["TORCH_HOME"] = args.torch_home

    if args.copy_data:
        # if copy data is 1 then you need to provide args.data_dir
        # from which to copy data
        if args.data_dir == "":
            raise ValueError("You need to provide args.data_dir")
        copy_data(args)

    if args.eval_only:
        main_worker_eval(0, args)
        return

    if args.num_gpus > 1:
        mp.spawn(main_worker, nprocs=args.num_gpus, args=(args,), daemon=False)
    else:
        main_worker(0, args)


if __name__ == "__main__":
    shapenet_launch()
