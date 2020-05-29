# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
from collections import defaultdict
import detectron2.utils.comm as comm
import torch
from detectron2.evaluation import inference_context

from meshrcnn.utils.metrics import compare_meshes

import shapenet.utils.vis as vis_utils
from shapenet.data.utils import image_to_numpy, imagenet_deprocess
from shapenet.modeling.mesh_arch import VoxMeshMultiViewHead, VoxMeshDepthHead
from shapenet.modeling.heads.depth_loss import adaptive_berhu_loss
from shapenet.modeling.heads.mesh_loss import MeshLoss

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
            if type(module) in [VoxMeshMultiViewHead, VoxMeshDepthHead]:
                model_kwargs["intrinsics"] = batch["intrinsics"]
                model_kwargs["extrinsics"] = batch["extrinsics"]
            if type(module) == VoxMeshDepthHead:
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
            if type(module) in [VoxMeshMultiViewHead, VoxMeshDepthHead]:
                model_kwargs["intrinsics"] = batch["intrinsics"]
                model_kwargs["extrinsics"] = batch["extrinsics"]
            if type(module) == VoxMeshDepthHead:
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
        if type(module) in [VoxMeshMultiViewHead, VoxMeshDepthHead]:
            model_kwargs["intrinsics"] = batch["intrinsics"]
            model_kwargs["extrinsics"] = batch["extrinsics"]
        if type(module) == VoxMeshDepthHead:
            model_kwargs["masks"] = batch["masks"]
            if module.mvsnet is None:
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

    device = torch.device("cuda:0")
    num_predictions = 0
    num_predictions_kept = 0
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
        cur_metrics = {"depth_loss": loss, "negative_depth_loss": -loss}

        if cur_metrics is None:
            continue
        for k, v in cur_metrics.items():
            metrics[k].append(v)

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
        logger.info("Evaluated %d predictions so far" % num_predictions)
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
