import torch
import torch.nn as nn

import numpy as np

## batch x view x *img_size -> batch_view x * img_size
def flatten_batch_view(tensor):
    return tensor.view(-1, *tensor.size()[2:])

## batch_view x * img_size-> batch x view x *img_size
def unflatten_batch_view(tensor, batch_size):
    return tensor.view(batch_size, -1, *tensor.size()[1:])

def get_bearing_vectors(width, height, intrinsics, dtype, device):
    # form tensor with image (u, v) coordinates
    u_indices = torch.linspace(0, width - 1, width,
                               dtype=dtype, device=device)
    v_indices = torch.linspace(0, height - 1, height,
                               dtype=dtype, device=device)
    grid_v, grid_u = torch.meshgrid(v_indices, u_indices)
    grid_uv = torch.stack((grid_u, grid_v), dim=-1)

    # find bearing vectors
    principal_point = intrinsics[:2, 2].type(dtype).to(device)
    focal_lengths = torch.tensor([intrinsics[0, 0], intrinsics[1, 1]],
                                 dtype=dtype, device=device)
    bearing_vectors = (grid_uv - principal_point) / focal_lengths
    return bearing_vectors

def bearing_depth_to_xyz(bearing_vectors, depths):
    # expand multiple batches and views
    bearing_vectors = bearing_vectors.view(1, 1, *(bearing_vectors.size())) \
                                     .expand(*(depths.size()[:2]), -1, -1, -1)
    depths_repeated = depths.unsqueeze(-1)\
                            .expand(*[-1 for _ in range(len(depths.size()))], 2)
    xy_local = bearing_vectors * depths_repeated
    xyz_local = torch.cat((xy_local, depths.unsqueeze(-1)), dim=-1)

    # transform it to ShapeNet (East-Up-South) from DTU (East-Down-North)
    xyz_local[:, :, :, :, 1:3] *= -1
    return xyz_local

## transform to world coordinate frame
def batch_transform(xyz_cam, T_world_cam):
    batch_size = xyz_cam.size(0)
    height, width = xyz_cam.size()[2:4]
    ones = torch.ones((*(xyz_cam.size()[:-1]), 1),
                      dtype=xyz_cam.dtype, device=xyz_cam.device)
    xyzw_cam = torch.cat((xyz_cam, ones), dim=-1)

    # batch x view x height x width x 4 -> batch_view x height x width x 4
    xyzw_flattened = flatten_batch_view(xyzw_cam)
    # flatten batch_view x height x width x 4 -> batch_view x height_width x 4
    xyzw_flattened = xyzw_flattened.view(xyzw_flattened.size(0), -1,
                                         xyzw_flattened.size(-1))

    T_world_cam_flattened = flatten_batch_view(T_world_cam)
    xyzw_world = torch.bmm(xyzw_flattened,
                           T_world_cam_flattened.transpose(-2, -1))

    # batch_view x height_width x 4 -> flatten batch_view x height x width x 4
    xyzw_world = xyzw_world.view(xyzw_world.size(0), height, width, 4)
    # batch_view x height x width x 4 -> batch x view x height x width x 4
    xyzw_world = xyzw_world.view(batch_size, -1, height, width, 4)

    return xyzw_world[:, :, :, :, :3]

##
#  @param xyz_coords batch x view x height x width x 3
#  @return list of size batch x view. Each element num_points x 3
def group_3dpoints(batched_xyz, valid_points_mask):
    batch_size = batched_xyz.size(0)
    num_views = batched_xyz.size(1)
    grouped_points = [
        [None for _ in range(num_views)]
        for _ in range(batch_size)
    ]
    for batch_idx in range(batch_size):
        for view_idx in range(num_views):
            xyz_subset = batched_xyz[batch_idx, view_idx]
            mask_subset = valid_points_mask[batch_idx, view_idx]
            valid_points = xyz_subset[mask_subset]
            grouped_points[batch_idx][view_idx] = valid_points
    return grouped_points

##
# @param extrinsics T_cam_world
def get_points_from_depths(depths, intrinsics, extrinsics=None):
    device = depths.device
    dtype = depths.dtype
    height, width = depths.size()[-2:]
    batch_size = depths.size(0)
    T_cam_world = extrinsics
    if extrinsics is not None:
        T_world_cam = unflatten_batch_view(
            torch.inverse(flatten_batch_view(extrinsics)), batch_size
        )
    else:
        T_world_cam = None
    bearing_vectors = get_bearing_vectors(width, height, intrinsics,
                                          dtype, device)
    xyz_cam = bearing_depth_to_xyz(bearing_vectors, depths)
    valid_points_mask = xyz_cam[:, :, :, :, 2] < -1e-3
    if T_world_cam is not None:
        xyz_world = batch_transform(xyz_cam, T_world_cam)
    else:
        xyz_world = xyz_cam
    grouped_points = group_3dpoints(xyz_world, valid_points_mask)
    return grouped_points
