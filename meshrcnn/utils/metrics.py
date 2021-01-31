# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.ops import knn_gather, knn_points, sample_points_from_meshes

logger = logging.getLogger(__name__)


@torch.no_grad()
def compare_meshes_p2m(
    pred_meshes, gt_meshes, thresholds=[0.00005, 0.00010, 0.00015, 0.00020]
):
    len_gt_points = 9000
    len_pred_points = 6466

    gt_meshes = gt_meshes.scale_verts(0.57)
    gt_points = sample_points_from_meshes(
        gt_meshes, len_gt_points, return_normals=False
    )

    pred_meshes = pred_meshes.scale_verts(0.57)
    pred_points = sample_points_from_meshes(
        pred_meshes, len_pred_points, return_normals=False
    )

    return compare_points_p2m(pred_points, gt_points)

# TODO: handle batch size > 1
@torch.no_grad()
def compare_points_p2m(
    pred_points, gt_points, thresholds=[0.00005, 0.00010, 0.00015, 0.00020]
):
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)

    precisions  = []
    recalls  = []
    f_scores = []
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists2 <= t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists2 <= t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + 1e-6)

        precisions.append(precision.item())
        recalls.append(recall.item())
        f_scores.append(f1.item())

    # TODO: Handle batch_size, here dims expanded to fake batch size 1
    return {
        "precisions": np.expand_dims(np.asarray(precisions), 0),
        "recalls": np.expand_dims(np.asarray(recalls), 0),
        "f_scores": np.expand_dims(np.asarray(f_scores), 0),
    }


@torch.no_grad()
def compare_voxel_scores(pred_vox_scores, gt_vox_binary, prob_threshold):
    """
    the scores are in logits
    """
    def vox_scores_to_binary(vox_scores):
        nonlocal prob_threshold
        vox_probs = vox_scores.sigmoid()
        return vox_probs > prob_threshold

    pred_vox_binary = vox_scores_to_binary(pred_vox_scores)
    return compare_voxel_binaries(pred_vox_binary, gt_vox_binary.bool())


@torch.no_grad()
def compare_voxel_binaries(pred_vox_binary, gt_vox_binary):
    # batchwise counts
    tp = torch.sum((pred_vox_binary == 1) & (gt_vox_binary == 1), [1, 2, 3]).float()
    tn = torch.sum((pred_vox_binary == 0) & (gt_vox_binary == 0), [1, 2, 3]).float()
    fp = torch.sum((pred_vox_binary == 1) & (gt_vox_binary == 0), [1, 2, 3]).float()
    fn = torch.sum((pred_vox_binary == 0) & (gt_vox_binary == 1), [1, 2, 3]).float()
    precisions = tp / (tp + fp + 1e-6)
    recalls = tp / (tp + fn + 1e-6)
    f_scores = 2 * precisions * recalls / (precisions + recalls + 1e-6)
    return {
        "precisions": precisions.detach().cpu().numpy(),
        "recalls": recalls.detach().cpu().numpy(),
        "f_scores": f_scores.detach().cpu().numpy()
    }

@torch.no_grad()
def compare_meshes(
    pred_meshes, gt_meshes, num_samples=10000, scale="gt-10", thresholds=None, reduce=True, eps=1e-8
):
    """
    Compute evaluation metrics to compare meshes. We currently compute the
    following metrics:

    - L2 Chamfer distance
    - Normal consistency
    - Absolute normal consistency
    - Precision at various thresholds
    - Recall at various thresholds
    - F1 score at various thresholds

    Inputs:
        - pred_meshes (Meshes): Contains N predicted meshes
        - gt_meshes (Meshes): Contains 1 or N ground-truth meshes. If gt_meshes
          contains 1 mesh, it is replicated N times.
        - num_samples: The number of samples to take on the surface of each mesh.
          This can be one of the following:
            - (int): Take that many uniform samples from the surface of the mesh
            - 'verts': Use the vertex positions as samples for each mesh
            - A tuple of length 2: To use different sampling strategies for the
              predicted and ground-truth meshes (respectively).
        - scale: How to scale the predicted and ground-truth meshes before comparing.
          This can be one of the following:
            - (float): Multiply the vertex positions of both meshes by this value
            - A tuple of two floats: Multiply the vertex positions of the predicted
              and ground-truth meshes by these two different values
            - A string of the form 'gt-[SCALE]', where [SCALE] is a float literal.
              In this case, each (predicted, ground-truth) pair is scaled differently,
              so that bounding box of the (rescaled) ground-truth mesh has longest
              edge length [SCALE].
        - thresholds: The distance thresholds to use when computing precision, recall,
          and F1 scores.
        - reduce: If True, then return the average of each metric over the batch;
          otherwise return the value of each metric between each predicted and
          ground-truth mesh.
        - eps: Small constant for numeric stability when computing F1 scores.

    Returns:
        - metrics: A dictionary mapping metric names to their values. If reduce is
          True then the values are the average value of the metric over the batch;
          otherwise the values are Tensors of shape (N,).
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    pred_meshes, gt_meshes = _scale_meshes(pred_meshes, gt_meshes, scale)

    if isinstance(num_samples, tuple):
        num_samples_pred, num_samples_gt = num_samples
    else:
        num_samples_pred = num_samples_gt = num_samples

    pred_points, pred_normals = _sample_meshes(pred_meshes, num_samples_pred)
    gt_points, gt_normals = _sample_meshes(gt_meshes, num_samples_gt)
    if pred_points is None:
        logger.info("WARNING: Sampling predictions failed during eval")
        return None
    elif gt_points is None:
        logger.info("WARNING: Sampling GT failed during eval")
        return None

    if len(gt_meshes) == 1:
        # (1, S, 3) to (N, S, 3)
        gt_points = gt_points.expand(len(pred_meshes), -1, -1)
        gt_normals = gt_normals.expand(len(pred_meshes), -1, -1)

    if torch.is_tensor(pred_points) and torch.is_tensor(gt_points):
        # We can compute all metrics at once in this case
        metrics = _compute_sampling_metrics(
            pred_points, pred_normals, gt_points, gt_normals, thresholds, eps
        )
    else:
        # Slow path when taking vert samples from non-equisized meshes; we need
        # to iterate over the batch
        metrics = defaultdict(list)
        for cur_points_pred, cur_points_gt in zip(pred_points, gt_points):
            cur_metrics = _compute_sampling_metrics(
                cur_points_pred[None], None, cur_points_gt[None], None, thresholds, eps
            )
            for k, v in cur_metrics.items():
                metrics[k].append(v.item())
        metrics = {k: torch.tensor(vs) for k, vs in metrics.items()}

    if reduce:
        # Average each metric over the batch
        metrics = {k: v.mean().item() for k, v in metrics.items()}

    return metrics


def _scale_meshes(pred_meshes, gt_meshes, scale):
    if isinstance(scale, float):
        # Assume scale is a single scalar to use for both preds and GT
        pred_scale = gt_scale = scale
    elif isinstance(scale, tuple):
        # Rescale preds and GT with different scalars
        pred_scale, gt_scale = scale
    elif scale.startswith("gt-"):
        # Rescale both preds and GT so that the largest edge length of each GT
        # mesh is target
        target = float(scale[3:])
        bbox = gt_meshes.get_bounding_boxes()  # (N, 3, 2)
        long_edge = (bbox[:, :, 1] - bbox[:, :, 0]).max(dim=1)[0]  # (N,)
        scale = target / long_edge
        if scale.numel() == 1:
            scale = scale.expand(len(pred_meshes))
        pred_scale, gt_scale = scale, scale
    else:
        raise ValueError("Invalid scale: %r" % scale)
    pred_meshes = pred_meshes.scale_verts(pred_scale)
    gt_meshes = gt_meshes.scale_verts(gt_scale)
    return pred_meshes, gt_meshes


def _sample_meshes(meshes, num_samples):
    """
    Helper to either sample points uniformly from the surface of a mesh
    (with normals), or take the verts of the mesh as samples.

    Inputs:
        - meshes: A MeshList
        - num_samples: An integer, or the string 'verts'

    Outputs:
        - verts: Either a Tensor of shape (N, S, 3) if we take the same number of
          samples from each mesh; otherwise a list of length N, whose ith element
          is a Tensor of shape (S_i, 3)
        - normals: Either a Tensor of shape (N, S, 3) or None if we take verts
          as samples.
    """
    if num_samples == "verts":
        normals = None
        if meshes.equisized:
            verts = meshes.verts_batch
        else:
            verts = meshes.verts_list
    else:
        verts, normals = sample_points_from_meshes(meshes, num_samples, return_normals=True)
    return verts, normals


def _compute_sampling_metrics(pred_points, pred_normals, gt_points, gt_normals, thresholds, eps):
    """
    Compute metrics that are based on sampling points and normals:

    - L2 Chamfer distance
    - Precision at various thresholds
    - Recall at various thresholds
    - F1 score at various thresholds
    - Normal consistency (if normals are provided)
    - Absolute normal consistency (if normals are provided)

    Inputs:
        - pred_points: Tensor of shape (N, S, 3) giving coordinates of sampled points
          for each predicted mesh
        - pred_normals: Tensor of shape (N, S, 3) giving normals of points sampled
          from the predicted mesh, or None if such normals are not available
        - gt_points: Tensor of shape (N, S, 3) giving coordinates of sampled points
          for each ground-truth mesh
        - gt_normals: Tensor of shape (N, S, 3) giving normals of points sampled from
          the ground-truth verts, or None of such normals are not available
        - thresholds: Distance thresholds to use for precision / recall / F1
        - eps: epsilon value to handle numerically unstable F1 computation

    Returns:
        - metrics: A dictionary where keys are metric names and values are Tensors of
          shape (N,) giving the value of the metric for the batch
    """
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)
    if gt_normals is not None:
        pred_normals_near = knn_gather(gt_normals, knn_pred.idx, lengths_gt)[..., 0, :]  # (N, S, 3)
    else:
        pred_normals_near = None

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    if pred_normals is not None:
        gt_normals_near = knn_gather(pred_normals, knn_gt.idx, lengths_pred)[..., 0, :]  # (N, S, 3)
    else:
        gt_normals_near = None

    # Compute L2 chamfer distances
    chamfer_l2 = pred_to_gt_dists2.mean(dim=1) + gt_to_pred_dists2.mean(dim=1)
    metrics["Chamfer-L2"] = chamfer_l2

    # Compute normal consistency and absolute normal consistance only if
    # we actually got normals for both meshes
    if pred_normals is not None and gt_normals is not None:
        pred_to_gt_cos = F.cosine_similarity(pred_normals, pred_normals_near, dim=2)
        gt_to_pred_cos = F.cosine_similarity(gt_normals, gt_normals_near, dim=2)

        pred_to_gt_cos_sim = pred_to_gt_cos.mean(dim=1)
        pred_to_gt_abs_cos_sim = pred_to_gt_cos.abs().mean(dim=1)
        gt_to_pred_cos_sim = gt_to_pred_cos.mean(dim=1)
        gt_to_pred_abs_cos_sim = gt_to_pred_cos.abs().mean(dim=1)
        normal_dist = 0.5 * (pred_to_gt_cos_sim + gt_to_pred_cos_sim)
        abs_normal_dist = 0.5 * (pred_to_gt_abs_cos_sim + gt_to_pred_abs_cos_sim)
        metrics["NormalConsistency"] = normal_dist
        metrics["AbsNormalConsistency"] = abs_normal_dist

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics
