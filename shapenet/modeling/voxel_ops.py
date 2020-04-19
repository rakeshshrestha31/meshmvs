import torch
import pytorch3d.ops
from pytorch3d.structures import Meshes

import time

from shapenet.utils.coords import \
        custom_padded_grid_sample, voxel_grid_coords, \
        world_coords_to_voxel, voxel_coords_to_world, \
        voxel_to_world, transform_meshes, transform_verts


def logit(x, eps=1e-5):
    """
    logit or log-odds. Inverse of sigmoid
    """
    if type(x) != torch.Tensor:
        x = torch.tensor(x)
    odds = (x / (1 - x).clamp(min=eps)).clamp(min=eps)
    return torch.log(odds)


def dummy_mesh(N, device):
    verts_batch = torch.randn(N, 4, 3, device=device)
    faces = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]
    faces = torch.tensor(faces, dtype=torch.int64)
    faces_batch = faces.view(1, 4, 3).expand(N, 4, 3).to(device)
    return Meshes(verts=verts_batch, faces=faces_batch)


def add_dummy_meshes(meshes):
    N = len(meshes)
    dummies = dummy_mesh(N, meshes.device)
    verts_list = meshes.verts_list()
    faces_list = meshes.faces_list()
    for i in range(N):
        if faces_list[i].shape[0] == 0:
            # print('Adding dummmy mesh at index ', i)
            vv, ff = dummies.get_mesh(i)
            verts_list[i] = vv
            faces_list[i] = ff
    return Meshes(verts=verts_list, faces=faces_list)


def cubify(voxel_scores, voxel_size, cubify_threshold):
    V = voxel_size
    N = voxel_scores.shape[0]
    voxel_probs = voxel_scores.sigmoid()
    active_voxels = voxel_probs > cubify_threshold
    voxels_per_mesh = (active_voxels.view(N, -1).sum(dim=1)).tolist()
    start = V // 4
    stop = start + V // 2
    for i in range(N):
        if voxels_per_mesh[i] == 0:
            voxel_probs[i, start:stop, start:stop, start:stop] = 1
    meshes = pytorch3d.ops.cubify(voxel_probs, cubify_threshold)

    meshes = add_dummy_meshes(meshes)
    meshes = voxel_to_world(meshes)
    return meshes


def merge_multi_view_voxels(
    voxel_scores, extrinsics,
    voxel_size, cubify_threshold, max_non_occupied_score
):
    """
    Merge multive voxel scores
    Inputs:
    - voxel_scores: tensor of shape (batch, view, d, h, w)
    - extrinsics: tensors  of size (batch, view, 4, 4)
    - voxel_size: float
    - cubify_threshold: float [0, 1]. Occupancy threshold
    - max_non_occupied_score: max logit (log-odds) score for non-occupied voxels
    Returns:
    - list of float tensor of shape (batch, d, h, w).
        Merged voxel scores (first element) along with the score of each view
        in the same coordinate frame
    """
    batch_size = voxel_scores.shape[0]
    device = voxel_scores.device
    voxel_scores = voxel_scores.unbind(dim=1)
    T_ref_world = extrinsics[:, 0]
    T_world_ref = torch.inverse(T_ref_world)
    timestamp = int(time.time() * 1000)

    # compute grid points
    grid_shape = list(voxel_scores[0].shape[-3:])
    norm_coords = voxel_grid_coords(grid_shape)
    grid_points = voxel_coords_to_world(norm_coords.view(-1, 3)) \
                        .view(1, -1, 3).expand(batch_size, -1, -1) \
                        .to(device)

    transformed_voxel_scores = []

    for view_idx, voxel_scores_view in enumerate(voxel_scores):
        T_view_world = extrinsics[:, view_idx]
        T_view_ref = T_view_world.bmm(T_world_ref)

        # transform to view frame to find corresponding normalized coords
        grid_points_view = transform_verts(grid_points, T_view_ref)[:, :, :3]
        transformed_norm_coords = world_coords_to_voxel(grid_points_view) \
                                        .view(batch_size, *grid_shape, 3)
        # makes sure that the out of bound voxels
        # have proper (non-occupied) scores
        voxel_scores_ref = custom_padded_grid_sample(
            voxel_scores_view.unsqueeze(1), transformed_norm_coords,
            pad_value=max_non_occupied_score,
            mode="bilinear", align_corners=True
        ).squeeze(0)
        transformed_voxel_scores.append(voxel_scores_ref)
        # save_voxel_grids_view(
        #     view_idx, timestamp, voxel_scores_view,
        #     voxel_scores_ref, T_view_ref, voxel_size, cubify_threshold
        # )

    # adding the scores (which are actually the log-odds)
    # is equivalent to Bayesian update of occupancy probabilities
    merged_voxel_scores = torch.sum(
        torch.stack(transformed_voxel_scores, dim=0), dim=0
    )
    # save_merged_voxel_grids(
    #     timestamp, merged_voxel_scores, voxel_size, cubify_threshold
    # )
    return [merged_voxel_scores, *transformed_voxel_scores]


@torch.no_grad()
def save_merged_voxel_grids(
    file_prefix, voxel_scores, voxel_size, cubify_threshold
):
    """
    save merged voxel grids for debugging purpose
    """
    from pytorch3d.io import save_obj
    import open3d as o3d
    meshes = cubify(voxel_scores, voxel_size, cubify_threshold)
    for batch_idx, mesh in enumerate(meshes):
        filename = "/tmp/cube_mesh_{}_{}_merged.obj" \
                        .format(file_prefix, batch_idx)
        save_obj(filename, mesh.verts_packed(), mesh.faces_packed())


@torch.no_grad()
def save_voxel_grids_view(
    view_idx, file_prefix,
    voxel_scores_view, voxel_scores_ref, T_view_ref,
    voxel_size, cubify_threshold
):
    """
    save a view's voxel grids for debugging purpose
    """
    from pytorch3d.io import save_obj
    import open3d as o3d

    batch_size = voxel_scores_view.shape[0]
    device = voxel_scores_view.device
    T_ref_view = torch.inverse(T_view_ref)

    voxel_probs = voxel_scores_view.sigmoid()
    active_voxels = voxel_probs > cubify_threshold
    cubified_meshes = cubify(voxel_scores_view, voxel_size, cubify_threshold)
    cubified_meshes = transform_meshes(cubified_meshes, T_ref_view)

    cubified_meshes_ref = cubify(voxel_scores_ref, voxel_size, cubify_threshold)

    # compute grid points
    grid_shape = list(voxel_scores_view.shape[-3:])
    norm_coords = voxel_grid_coords(grid_shape)
    grid_points_flat = voxel_coords_to_world(norm_coords.view(-1, 3)) \
                            .view(1, -1, 3).expand(batch_size, -1, -1) \
                            .to(device)

    # transform to ref frame
    grid_points_ref = transform_verts(
        grid_points_flat, T_ref_view
    )[:, :, :3].view(batch_size, *grid_shape, 3)

    for batch_idx in range(len(cubified_meshes)):
        points = grid_points_ref[batch_idx][active_voxels[batch_idx]]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
            points.view(-1, 3).cpu().detach().numpy()
        ))
        o3d.io.write_point_cloud(
            "/tmp/cube_mesh_{}_{}_{}_voxels.ply".format(
                file_prefix, batch_idx, view_idx
            ),
            pcd
        )

        def save_mesh(filename, mesh):
            save_obj(filename, mesh.verts_packed(), mesh.faces_packed())

        save_mesh(
            "/tmp/cube_mesh_{}_{}_{}.obj" \
                    .format(file_prefix, batch_idx, view_idx),
            cubified_meshes[batch_idx]
        )
        save_mesh(
            "/tmp/cube_mesh_{}_{}_{}_ref.obj" \
                    .format(file_prefix, batch_idx, view_idx),
            cubified_meshes_ref[batch_idx]
        )
