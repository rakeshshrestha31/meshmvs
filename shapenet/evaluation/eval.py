# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
from collections import defaultdict
import tqdm
import os
import cv2

import detectron2.utils.comm as comm
import torch
import torch.nn.functional as F
import torchvision
from detectron2.evaluation import inference_context
from pytorch3d.ops import sample_points_from_meshes

from meshrcnn.utils.metrics import compare_meshes

import shapenet.utils.vis as vis_utils
from shapenet.data.utils import image_to_numpy, imagenet_deprocess
from shapenet.modeling.mesh_arch import \
    VoxMeshMultiViewHead, VoxMeshDepthHead, VoxDepthHead
from shapenet.modeling.heads.depth_loss import \
    adaptive_berhu_loss, interpolate_multi_view_tensor
from shapenet.modeling.heads.mesh_loss import MeshLoss
from shapenet.modeling.mesh_arch import cubify

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_test(model, data_loader, vis_preds=False):
    """
    This function evaluates the model on the dataset defined by data_loader.
    The metrics reported are described in Table 2 of our paper.
    """
    # Note that all eval runs on main process
    assert comm.is_main_process()
    deprocess = imagenet_deprocess(rescale_image=False)
    device = torch.device("cuda:0")
    # evaluation
    class_names = {
        "02828884": "bench",
        "03001627": "chair",
        "03636649": "lamp",
        "03691459": "speaker",
        "04090263": "firearm",
        "04379243": "table",
        "04530566": "watercraft",
        "02691156": "plane",
        "02933112": "cabinet",
        "02958343": "car",
        "03211117": "monitor",
        "04256520": "couch",
        "04401088": "cellphone",
    }

    num_instances = {i: 0 for i in class_names}
    chamfer = {i: 0 for i in class_names}
    normal = {i: 0 for i in class_names}
    f1_01 = {i: 0 for i in class_names}
    f1_03 = {i: 0 for i in class_names}
    f1_05 = {i: 0 for i in class_names}

    num_batch_evaluated = 0
    for batch in data_loader:
        batch = data_loader.postprocess(batch, device)
        sids = [id_str.split("-")[0] for id_str in batch["id_strs"]]
        for sid in sids:
            num_instances[sid] += 1

        with inference_context(model):
            model_kwargs = {}
            module = model.module if hasattr(model, "module") else model
            if isinstance(module, VoxMeshMultiViewHead):
                model_kwargs["intrinsics"] = batch["intrinsics"]
                model_kwargs["extrinsics"] = batch["extrinsics"]
            if isinstance(module, VoxMeshDepthHead):
                model_kwargs["masks"] = batch["masks"]

            model_outputs = model(batch["imgs"], **model_kwargs)
            voxel_scores = model_outputs["voxel_scores"]
            meshes_pred = model_outputs["meshes_pred"]

            cur_metrics = compare_meshes(meshes_pred[-1], batch["meshes"], reduce=False)
            cur_metrics["verts_per_mesh"] = meshes_pred[-1].num_verts_per_mesh().cpu()
            cur_metrics["faces_per_mesh"] = meshes_pred[-1].num_faces_per_mesh().cpu()

            for i, sid in enumerate(sids):
                chamfer[sid] += cur_metrics["Chamfer-L2"][i].item()
                normal[sid] += cur_metrics["AbsNormalConsistency"][i].item()
                f1_01[sid] += cur_metrics["F1@%f" % 0.1][i].item()
                f1_03[sid] += cur_metrics["F1@%f" % 0.3][i].item()
                f1_05[sid] += cur_metrics["F1@%f" % 0.5][i].item()

                if vis_preds:
                    img = image_to_numpy(deprocess(batch["imgs"][i]))
                    vis_utils.visualize_prediction(
                        batch["id_strs"][i], img, meshes_pred[-1][i], "/tmp/output"
                    )

            num_batch_evaluated += 1
            logger.info("Evaluated %d / %d batches" % (num_batch_evaluated, len(data_loader)))

    vis_utils.print_instances_class_histogram(
        num_instances,
        class_names,
        {"chamfer": chamfer, "normal": normal, "f1_01": f1_01, "f1_03": f1_03, "f1_05": f1_05},
    )


@torch.no_grad()
def evaluate_test_p2m(model, data_loader):
    """
    This function evaluates the model on the dataset defined by data_loader.
    The metrics reported are described in Table 1 of our paper, following previous
    reported approaches (like Pixel2Mesh - p2m), where meshes are
    rescaled by a factor of 0.57. See the paper for more details.
    """
    assert comm.is_main_process()
    device = torch.device("cuda:0")
    # evaluation
    class_names = {
        "02828884": "bench",
        "03001627": "chair",
        "03636649": "lamp",
        "03691459": "speaker",
        "04090263": "firearm",
        "04379243": "table",
        "04530566": "watercraft",
        "02691156": "plane",
        "02933112": "cabinet",
        "02958343": "car",
        "03211117": "monitor",
        "04256520": "couch",
        "04401088": "cellphone",
    }

    num_instances = {i: 0 for i in class_names}
    chamfer = {i: 0 for i in class_names}
    normal = {i: 0 for i in class_names}
    f1_1e_4 = {i: 0 for i in class_names}
    f1_2e_4 = {i: 0 for i in class_names}

    num_batch_evaluated = 0
    for batch in data_loader:
        batch = data_loader.postprocess(batch, device)
        sids = [id_str.split("-")[0] for id_str in batch["id_strs"]]
        for sid in sids:
            num_instances[sid] += 1

        with inference_context(model):
            model_kwargs = {}
            module = model.module if hasattr(model, "module") else model
            if isinstance(module, VoxMeshMultiViewHead):
                model_kwargs["intrinsics"] = batch["intrinsics"]
                model_kwargs["extrinsics"] = batch["extrinsics"]
            if isinstance(module, VoxMeshDepthHead):
                model_kwargs["masks"] = batch["masks"]

            model_outputs = model(batch["imgs"], **model_kwargs)
            voxel_scores = model_outputs["voxel_scores"]
            meshes_pred = model_outputs["meshes_pred"]

            # NOTE that for the F1 thresholds we take the square root of 1e-4 & 2e-4
            # as `compare_meshes` returns the euclidean distance (L2) of two pointclouds.
            # In Pixel2Mesh, the squared L2 (L2^2) is computed instead.
            # i.e. (L2^2 < τ) <=> (L2 < sqrt(τ))
            cur_metrics = compare_meshes(
                meshes_pred[-1], batch["meshes"], scale=0.57, thresholds=[0.01, 0.014142], reduce=False
            )
            cur_metrics["verts_per_mesh"] = meshes_pred[-1].num_verts_per_mesh().cpu()
            cur_metrics["faces_per_mesh"] = meshes_pred[-1].num_faces_per_mesh().cpu()

            for i, sid in enumerate(sids):
                chamfer[sid] += cur_metrics["Chamfer-L2"][i].item()
                normal[sid] += cur_metrics["AbsNormalConsistency"][i].item()
                f1_1e_4[sid] += cur_metrics["F1@%f" % 0.01][i].item()
                f1_2e_4[sid] += cur_metrics["F1@%f" % 0.014142][i].item()

            num_batch_evaluated += 1
            logger.info("Evaluated %d / %d batches" % (num_batch_evaluated, len(data_loader)))

    vis_utils.print_instances_class_histogram_p2m(
        num_instances,
        class_names,
        {"chamfer": chamfer, "normal": normal, "f1_1e_4": f1_1e_4, "f1_2e_4": f1_2e_4},
    )


@torch.no_grad()
def evaluate_split(
    model, loader, max_predictions=-1, num_predictions_keep=10, prefix="", store_predictions=False
):
    """
    This function is used to report validation performance during training.
    """
    # Note that all eval runs on main process
    assert comm.is_main_process()
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    device = torch.device("cuda:0")
    num_predictions = 0
    num_predictions_kept = 0
    predictions = defaultdict(list)
    metrics = defaultdict(list)
    deprocess = imagenet_deprocess(rescale_image=False)
    for batch in loader:
        batch = loader.postprocess(batch, device)
        model_kwargs = {}
        module = model.module if hasattr(model, "module") else model
        if isinstance(module, VoxMeshMultiViewHead):
            model_kwargs["intrinsics"] = batch["intrinsics"]
            model_kwargs["extrinsics"] = batch["extrinsics"]
        if isinstance(module, VoxMeshDepthHead):
            model_kwargs["masks"] = batch["masks"]
            if module.cfg.MODEL.USE_GT_DEPTH:
                model_kwargs["depths"] = batch["depths"]
        model_outputs = model(batch["imgs"], **model_kwargs)
        meshes_pred = model_outputs["meshes_pred"]
        voxel_scores = model_outputs["voxel_scores"]
        merged_voxel_scores = model_outputs.get(
            "merged_voxel_scores", None
        )

        # Only compute metrics for the final predicted meshes, not intermediates
        cur_metrics = compare_meshes(meshes_pred[-1], batch["meshes"])
        if cur_metrics is None:
            continue
        for k, v in cur_metrics.items():
            metrics[k].append(v)

        voxel_losses = MeshLoss.voxel_loss(
            voxel_scores, merged_voxel_scores, batch["voxels"]
        )
        # to get metric negate loss
        for k, v in voxel_losses.items():
            metrics[k].append(-v.item())

        # Store input images and predicted meshes
        if store_predictions:
            N = batch["imgs"].shape[0]
            for i in range(N):
                if num_predictions_kept >= num_predictions_keep:
                    break
                num_predictions_kept += 1

                img = image_to_numpy(deprocess(batch["imgs"][i]))
                predictions["%simg_input" % prefix].append(img)
                for level, cur_meshes_pred in enumerate(meshes_pred):
                    verts, faces = cur_meshes_pred.get_mesh(i)
                    verts_key = "%sverts_pred_%d" % (prefix, level)
                    faces_key = "%sfaces_pred_%d" % (prefix, level)
                    predictions[verts_key].append(verts.cpu().numpy())
                    predictions[faces_key].append(faces.cpu().numpy())

        num_predictions += len(batch["meshes"])
        logger.info("Evaluated %d predictions so far" % num_predictions)
        if 0 < max_predictions <= num_predictions:
            break

    # Average numeric metrics, and concatenate images
    metrics = {"%s%s" % (prefix, k): np.mean(v) for k, v in metrics.items()}
    if store_predictions:
        img_key = "%simg_input" % prefix
        predictions[img_key] = np.stack(predictions[img_key], axis=0)

    return metrics, predictions


@torch.no_grad()
def evaluate_vox(model, loader, prediction_dir=None, max_predictions=-1):
    """
    This function is used to report validation performance of voxel head output
    """
    # Note that all eval runs on main process
    assert comm.is_main_process()
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    if prediction_dir is not None:
        for prefix in ["merged", "vox_0", "vox_1", "vox_2"]:
            output_dir = pred_filename = os.path.join(
                prediction_dir, prefix, "predict", "0"
            )
            os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda:0")
    metrics = defaultdict(list)
    deprocess = imagenet_deprocess(rescale_image=False)
    for batch_idx, batch in tqdm.tqdm(enumerate(loader)):
        if max_predictions >= 1 and batch_idx > max_predictions:
            break
        batch = loader.postprocess(batch, device)
        model_kwargs = {}
        module = model.module if hasattr(model, "module") else model
        if isinstance(module, VoxMeshMultiViewHead):
            model_kwargs["intrinsics"] = batch["intrinsics"]
            model_kwargs["extrinsics"] = batch["extrinsics"]
        if isinstance(module, VoxDepthHead):
            model_kwargs["masks"] = batch["masks"]
            if module.cfg.MODEL.USE_GT_DEPTH:
                model_kwargs["depths"] = batch["depths"]
        model_outputs = model(batch["imgs"], **model_kwargs)
        voxel_scores = model_outputs["voxel_scores"]
        transformed_voxel_scores = model_outputs["transformed_voxel_scores"]
        merged_voxel_scores = model_outputs.get(
            "merged_voxel_scores", None
        )

        # NOTE that for the F1 thresholds we take the square root of 1e-4 & 2e-4
        # as `compare_meshes` returns the euclidean distance (L2) of two pointclouds.
        # In Pixel2Mesh, the squared L2 (L2^2) is computed instead.
        # i.e. (L2^2 < τ) <=> (L2 < sqrt(τ))
        if "meshes_pred" in model_outputs:
            meshes_pred = model_outputs["meshes_pred"]
            cur_metrics = compare_meshes(
                meshes_pred[-1], batch["meshes"],
                scale=0.57, thresholds=[0.01, 0.014142]
            )
            for k, v in cur_metrics.items():
                metrics["final_" + k].append(v)

        voxel_losses = MeshLoss.voxel_loss(
            voxel_scores, merged_voxel_scores, batch["voxels"]
        )
        # to get metric negate loss
        for k, v in voxel_losses.items():
            metrics[k].append(-v.detach().item())

        # save meshes
        if prediction_dir is not None:
            # cubify all the voxel scores
            merged_vox_mesh = cubify(
                merged_voxel_scores, module.voxel_size, module.cubify_threshold
            )
            # transformed_vox_mesh = [cubify(
            #     i, module.voxel_size, module.cubify_threshold
            # ) for i in transformed_voxel_scores]
            vox_meshes = {
                "merged": merged_vox_mesh,
                # **{
                #     "vox_%d" % i: mesh
                #     for i, mesh in enumerate(transformed_vox_mesh)
                # }
            }

            gt_mesh = batch["meshes"].scale_verts(0.57)
            gt_points = sample_points_from_meshes(
                gt_mesh, 9000, return_normals=False
            )
            gt_points = gt_points.cpu().detach().numpy()

            for mesh_idx in range(len(batch["id_strs"])):
                label, label_appendix \
                        = batch["id_strs"][mesh_idx].split("-")[:2]
                for prefix, vox_mesh in vox_meshes.items():
                    output_dir = pred_filename = os.path.join(
                        prediction_dir, prefix, "predict", "0"
                    )
                    pred_filename = os.path.join(
                        output_dir,
                        "{}_{}_predict.xyz".format(label, label_appendix)
                    )
                    gt_filename = os.path.join(
                        output_dir,
                        "{}_{}_ground.xyz".format(label, label_appendix)
                    )

                    pred_mesh = vox_mesh[mesh_idx].scale_verts(0.57)
                    pred_points = sample_points_from_meshes(
                        pred_mesh, 6466, return_normals=False
                    )
                    pred_points = pred_points.squeeze(0).cpu() \
                                             .detach().numpy()

                    np.savetxt(pred_filename, pred_points)
                    np.savetxt(gt_filename, gt_points[mesh_idx])

            # find accuracy of each cubified voxel meshes
            for prefix, vox_mesh in vox_meshes.items():
                vox_mesh_metrics = compare_meshes(
                    vox_mesh, batch["meshes"],
                    scale=0.57, thresholds=[0.01, 0.014142]
                )

                if vox_mesh_metrics is None:
                    continue
                for k, v in vox_mesh_metrics.items():
                    metrics[prefix + "_" + k].append(v)

    # Average numeric metrics, and concatenate images
    metrics = {k: np.mean(v) for k, v in metrics.items()}
    return metrics


@torch.no_grad()
def evaluate_split_depth(
    model, loader, max_predictions=-1, num_predictions_keep=10,
    prefix="", store_predictions=False
):
    """
    This function is used to report validation performance during training.
    """
    # Note that all eval runs on main process
    assert comm.is_main_process()
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    deprocess = imagenet_deprocess(rescale_image=False)

    device = torch.device("cuda:0")
    num_predictions = 0
    num_predictions_kept = 0
    total_l1_err = 0.0
    total_gt_depth = 0.0
    min_gt_depth = np.inf
    max_gt_depth = -np.inf
    num_pixels = 0.0
    predictions = defaultdict(list)
    metrics = defaultdict(list)
    deprocess = imagenet_deprocess(rescale_image=False)
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        model_kwargs = {}
        model_kwargs["extrinsics"] = batch["extrinsics"]
        model_outputs = model(batch["imgs"], **model_kwargs)
        pred_depths = model_outputs["depths"]
        loss = adaptive_berhu_loss(
            batch["depths"], pred_depths, batch["masks"]
        ).item()

        depth_gt = interpolate_multi_view_tensor(
            batch["depths"], pred_depths.shape[-2:]
        )
        mask = interpolate_multi_view_tensor(
            batch["masks"], pred_depths.shape[-2:]
        )
        masked_pred_depths = pred_depths * mask

        total_l1_err += \
                torch.sum(torch.abs(masked_pred_depths - depth_gt)).item()
        total_gt_depth += torch.sum(torch.abs(depth_gt)).item()
        min_gt_depth = min(min_gt_depth, torch.min(depth_gt[depth_gt > 1e-7]).item())
        max_gt_depth = max(max_gt_depth, torch.max(depth_gt).item())
        num_pixels += torch.sum(mask).item()
        cur_metrics = {"depth_loss": loss, "negative_depth_loss": -loss}

        if cur_metrics is None:
            continue
        for k, v in cur_metrics.items():
            metrics[k].append(v)

        viz_image_size = (224, 224)
        viz_masks = interpolate_multi_view_tensor(
            batch["masks"], viz_image_size
        ).detach().cpu()

        # debug: viz images
        def apply_colormap(tensor):
            nonlocal viz_image_size

            tensor = F.interpolate(
                tensor.unsqueeze(0).unsqueeze(0),
                viz_image_size, mode="nearest"
            ).squeeze(0).squeeze(0)

            nonzeros = (tensor > 1e-7).unsqueeze(0).expand(3, -1, -1) \
                        .detach().cpu().float()

            img = cv2.applyColorMap(
                tensor.detach().cpu().mul(255).byte().numpy(),
                cv2.COLORMAP_JET
            )
            tensor = torch.from_numpy(img).permute(2, 0, 1).float().div(255) \
                        * nonzeros
            tensor = torch.cat((tensor, nonzeros[0].unsqueeze(0)), dim=0)
            return tensor

        for idx in range(3):
            imgs = F.interpolate(
                batch["imgs"][idx], viz_image_size, mode="bilinear"
            ).detach().cpu()

            imgs = [
                torch.cat((deprocess(img), viz_masks[idx, i].unsqueeze(0)))
                for i, img in enumerate(imgs.unbind(0))
            ]

            depth_gt_colored = [apply_colormap(i) for i in depth_gt[idx].unbind(0)]
            depth_pred_colored = [
                apply_colormap(i) for i in masked_pred_depths[idx].unbind(0)
            ]

            grid = torch.stack((
                imgs[0], depth_gt_colored[0], depth_pred_colored[0],
                imgs[1], depth_gt_colored[1], depth_pred_colored[1],
                imgs[2], depth_gt_colored[2], depth_pred_colored[2],
            ), dim=0)

            img_idx = num_predictions + idx
            base_path = "/projects/mesh_mvs/meshmvs/output_debug/depths"
            img_names = [
                os.path.join(base_path, "{}_input_view_{}.png".format(img_idx, 0)),
                os.path.join(base_path, "{}_gt_view_{}.png".format(img_idx, 0)),
                os.path.join(base_path, "{}_pred_view_{}.png".format(img_idx, 0)),
                os.path.join(base_path, "{}_input_view_{}.png".format(img_idx, 1)),
                os.path.join(base_path, "{}_gt_view_{}.png".format(img_idx, 1)),
                os.path.join(base_path, "{}_pred_view_{}.png".format(img_idx, 1)),
                os.path.join(base_path, "{}_input_view_{}.png".format(img_idx, 2)),
                os.path.join(base_path, "{}_gt_view_{}.png".format(img_idx, 2)),
                os.path.join(base_path, "{}_pred_view_{}.png".format(img_idx, 2)),
            ]
            for img, filename in zip(grid.unbind(0), img_names):
                torchvision.utils.save_image(img, filename)
            # torchvision.utils.save_image(
            #     grid,
            #     "/projects/mesh_mvs/meshmvs/output_debug/{}.png".format(
            #         num_predictions + idx
            #     ),
            #     nrow=3
            # )

        # Store input images and predicted meshes
        if store_predictions:
            N = batch["imgs"].shape[0]
            for i in range(N):
                if num_predictions_kept >= num_predictions_keep:
                    break
                num_predictions_kept += 1

                img = image_to_numpy(deprocess(batch["imgs"][i]))
                depth = image_to_numpy(batch["depths"][i].unsqueeze(0))
                pred_depth = image_to_numpy(pred_depths[i].unsqueeze(0))
                predictions["%simg_input" % prefix].append(img)
                predictions["%sdepth_input" % prefix].append(depth)
                predictions["%sdepth_pred" % prefix].append(pred_depth)

        num_predictions += len(batch["imgs"])
        logger.info(
                "Evaluated %d predictions so far: avg err: %f, avg depth: %f, min depth: %f, max_depth: %f" \
                    % (
                        num_predictions,
                        total_l1_err / num_pixels,
                        total_gt_depth / num_pixels,
                        min_gt_depth,
                        max_gt_depth
                    )
        )
        if 0 < max_predictions <= num_predictions:
            break

    # Average numeric metrics, and concatenate images
    metrics = {"%s%s" % (prefix, k): np.mean(v) for k, v in metrics.items()}
    if store_predictions:
        keys = ["%simg_input", "%sdepth_input", "%sdepth_pred"]
        keys = [i % prefix for i in keys]
        predictions = {
            k: np.stack(predictions[k], axis=0) for k, v in predictions.items()
        }

    return metrics, predictions
