# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.utils.registry import Registry
from pytorch3d.ops import cubify
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere

import time

from shapenet.modeling.backbone import build_backbone
from shapenet.modeling.heads import MeshRefinementHead, VoxelHead
from shapenet.utils.coords import get_blender_intrinsic_matrix, voxel_to_world
from shapenet.utils.coords import transform_meshes, transform_verts
from shapenet.utils.coords import world_coords_to_voxel, voxel_coords_to_world
from shapenet.utils.coords import voxel_grid_coords

MESH_ARCH_REGISTRY = Registry("MESH_ARCH")


@MESH_ARCH_REGISTRY.register()
class VoxMeshHead(nn.Module):
    def __init__(self, cfg):
        super(VoxMeshHead, self).__init__()

        self.setup(cfg)
        # backbone
        self.backbone, feat_dims = build_backbone(cfg.MODEL.BACKBONE)
        # voxel head
        cfg.MODEL.VOXEL_HEAD.COMPUTED_INPUT_CHANNELS = feat_dims[-1]
        self.voxel_head = VoxelHead(cfg)
        # mesh head
        cfg.MODEL.MESH_HEAD.COMPUTED_INPUT_CHANNELS = sum(feat_dims)
        self.mesh_head = MeshRefinementHead(cfg)

    def setup(self, cfg):
        # fmt: off
        self.cubify_threshold   = cfg.MODEL.VOXEL_HEAD.CUBIFY_THRESH
        self.voxel_size         = cfg.MODEL.VOXEL_HEAD.VOXEL_SIZE
        # fmt: on

        self.register_buffer("K", get_blender_intrinsic_matrix())

    def _get_projection_matrix(self, N, device):
        return self.K[None].repeat(N, 1, 1).to(device).detach()

    def _dummy_mesh(self, N, device):
        verts_batch = torch.randn(N, 4, 3, device=device)
        faces = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]
        faces = torch.tensor(faces, dtype=torch.int64)
        faces_batch = faces.view(1, 4, 3).expand(N, 4, 3).to(device)
        return Meshes(verts=verts_batch, faces=faces_batch)

    def cubify(self, voxel_scores):
        V = self.voxel_size
        N = voxel_scores.shape[0]
        voxel_probs = voxel_scores.sigmoid()
        active_voxels = voxel_probs > self.cubify_threshold
        voxels_per_mesh = (active_voxels.view(N, -1).sum(dim=1)).tolist()
        start = V // 4
        stop = start + V // 2
        for i in range(N):
            if voxels_per_mesh[i] == 0:
                voxel_probs[i, start:stop, start:stop, start:stop] = 1
        meshes = cubify(voxel_probs, self.cubify_threshold)

        meshes = self._add_dummies(meshes)
        meshes = voxel_to_world(meshes)
        return meshes

    def _add_dummies(self, meshes):
        N = len(meshes)
        dummies = self._dummy_mesh(N, meshes.device)
        verts_list = meshes.verts_list()
        faces_list = meshes.faces_list()
        for i in range(N):
            if faces_list[i].shape[0] == 0:
                # print('Adding dummmy mesh at index ', i)
                vv, ff = dummies.get_mesh(i)
                verts_list[i] = vv
                faces_list[i] = ff
        return Meshes(verts=verts_list, faces=faces_list)

    def forward(self, imgs, voxel_only=False):
        N = imgs.shape[0]
        device = imgs.device

        img_feats = self.backbone(imgs)
        # voxel scores from one view only
        voxel_scores = [self.voxel_head(img_feats[-1])]
        # add view dimension (single view)
        img_feats = [i.unsqueeze(1) for i in img_feats]
        P = [self._get_projection_matrix(N, device)]

        if voxel_only:
            dummy_meshes = self._dummy_mesh(N, device)
            dummy_refined = self.mesh_head(img_feats, dummy_meshes, P)
            return voxel_scores, dummy_refined

        cubified_meshes = self.cubify(voxel_scores[0])
        refined_meshes = self.mesh_head(img_feats, cubified_meshes, P)
        return voxel_scores, refined_meshes

@MESH_ARCH_REGISTRY.register()
class VoxMeshMultiViewHead(VoxMeshHead):
    def __init__(self, cfg):
        nn.Module.__init__(self)

        self.setup(cfg)
        # backbone
        self.backbone, feat_dims = build_backbone(cfg.MODEL.BACKBONE)
        # voxel head
        cfg.MODEL.VOXEL_HEAD.COMPUTED_INPUT_CHANNELS = feat_dims[-1]
        self.voxel_head = VoxelHead(cfg)
        # mesh head
        # times 3 cuz multi-view (mean, avg, std) features will be used
        cfg.MODEL.MESH_HEAD.COMPUTED_INPUT_CHANNELS = sum(feat_dims) * 3
        self.mesh_head = MeshRefinementHead(cfg)

    def setup(self, cfg):
        VoxMeshHead.setup(self, cfg)
        self.cubify_threshold_logit = self.logit(self.cubify_threshold)

    @staticmethod
    def logit(x, eps=1e-5):
        """
        logit or log-odds. Inverse of sigmoid
        """
        if type(x) != torch.Tensor:
            x = torch.tensor(x)
        odds = (x / (1 - x).clamp(min=eps)).clamp(min=eps)
        return torch.log(odds)

    def forward(self, imgs, intrinsics, extrinsics, voxel_only=False):
        """
        Args:
        - imgs: tensor of shape (B, V, 3, H, W)
        - intrinsics: tensor of shape (B, V, 4, 4)
        - extrinsics: tensor of shape (B, V, 4, 4)
        """
        batch_size = imgs.shape[0]
        num_views = imgs.shape[1]
        device = imgs.device

        # flatten the batch and views
        flat_imgs = imgs.view(-1, *(imgs.shape[2:]))
        img_feats = self.backbone(flat_imgs)
        voxel_scores = self.voxel_head(img_feats[-1])
        # unflatten the batch and views
        voxel_scores = voxel_scores.view(
            batch_size, num_views, *(voxel_scores.shape[1:])
        )
        img_feats = [
            i.view(batch_size, num_views, *(i.shape[1:])) for i in img_feats
        ]

        K = self._get_projection_matrix(batch_size, device)
        rel_extrinsics = self.relative_extrinsics(extrinsics)
        P = [K.bmm(T) for T in rel_extrinsics.unbind(dim=1)]
        voxel_scores = self.merge_multi_view_voxels(voxel_scores, extrinsics)

        if voxel_only:
            dummy_meshes = self._dummy_mesh(batch_size, device)
            dummy_refined = self.mesh_head(img_feats, dummy_meshes, P)
            return voxel_scores, dummy_refined

        cubified_meshes = self.cubify(voxel_scores[0])
        refined_meshes = self.mesh_head(img_feats, cubified_meshes, P)
        return voxel_scores, refined_meshes

    @staticmethod
    def relative_extrinsics(extrinsics):
        """
        Converts extrinsics in world frame to reference frame

        Args:
        - extrinsics: tensors of size (batch, view, 4, 4)

        Returns:
        - tensors of size (batch, view, 4, 4)
        """
        T_ref_world = extrinsics[:, 0]
        T_world_ref = torch.inverse(T_ref_world)
        rel_extrinsics = torch.stack([
            T_view_world.bmm(T_world_ref)
            for T_view_world in  extrinsics.unbind(dim=1)
        ], dim=1)
        return rel_extrinsics

    def grid_sample_voxel_scores(self, voxel_scores, grid):
        """
        wrapper around torch.nn.functional.grid_sample
        makes sure that the out of bound voxels have proper (non-occupied) scores
        Inputs:
        - voxel_scores: FloatTensor of shape (N, D, H, W)
        - grid: FloatTensor of shape (N, D, H, W)
        Returns:
        - FloatTensor of shape (N, D, H, W)
        """
        # logit score that makes a cell non-occupied
        non_occupied_score = self.cubify_threshold_logit - 1e-1
        # simulate padding by non_occupied_score by
        # subtracting it before grid_sample and then adding it back later
        voxel_scores = voxel_scores - non_occupied_score
        return F.grid_sample(
            voxel_scores.unsqueeze(1), grid,
            mode="bilinear", padding_mode="zeros", align_corners=True
        ).squeeze(1) + non_occupied_score

    def merge_multi_view_voxels(self, voxel_scores, extrinsics):
        """
        Merge multive voxel scores
        Inputs:
        - voxel_scores: tensor of shape (batch, view, d, h, w)
        - extrinsics: tensors  of size (batch, view, 4, 4)
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
            grid_points_view = transform_verts(
                grid_points, T_view_ref
            )[:, :, :3]
            transformed_norm_coords = world_coords_to_voxel(grid_points_view) \
                                            .view(batch_size, *grid_shape, 3)
            voxel_scores_ref = self.grid_sample_voxel_scores(
                voxel_scores_view, transformed_norm_coords,
            )
            transformed_voxel_scores.append(voxel_scores_ref)
            # self.save_voxel_grids_view(
            #     view_idx, timestamp, voxel_scores_view,
            #     voxel_scores_ref, T_view_ref
            # )

        # adding the scores (which are actually the log-odds)
        # is equivalent to Bayesian update of occupancy probabilities
        merged_voxel_scores = torch.sum(
            torch.stack(transformed_voxel_scores, dim=0), dim=0
        )
        # self.save_merged_voxel_grids(timestamp, merged_voxel_scores)
        return [merged_voxel_scores, *transformed_voxel_scores]

    def save_merged_voxel_grids(self, file_prefix, voxel_scores):
        """
        save merged voxel grids for debugging purpose
        """
        from pytorch3d.io import save_obj
        import open3d as o3d
        meshes = self.cubify(voxel_scores)
        for batch_idx, mesh in enumerate(meshes):
            filename = "/tmp/cube_mesh_{}_{}_merged.obj" \
                            .format(file_prefix, batch_idx)
            save_obj(filename, mesh.verts_packed(), mesh.faces_packed())

    def save_voxel_grids_view(
        self, view_idx, file_prefix,
        voxel_scores_view, voxel_scores_ref, T_view_ref
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
        active_voxels = voxel_probs > self.cubify_threshold
        cubified_meshes = self.cubify(voxel_scores_view)
        cubified_meshes = transform_meshes(cubified_meshes, T_ref_view)

        cubified_meshes_ref = self.cubify(voxel_scores_ref)

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

        for batch_idx, mesh in enumerate(cubified_meshes):
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

@MESH_ARCH_REGISTRY.register()
class SphereInitHead(nn.Module):
    def __init__(self, cfg):
        super(SphereInitHead, self).__init__()

        # fmt: off
        backbone                = cfg.MODEL.BACKBONE
        self.ico_sphere_level   = cfg.MODEL.MESH_HEAD.ICO_SPHERE_LEVEL
        # fmt: on

        self.register_buffer("K", get_blender_intrinsic_matrix())
        # backbone
        self.backbone, feat_dims = build_backbone(backbone)
        # mesh head
        cfg.MODEL.MESH_HEAD.COMPUTED_INPUT_CHANNELS = sum(feat_dims)
        self.mesh_head = MeshRefinementHead(cfg)

    def _get_projection_matrix(self, N, device):
        return self.K[None].repeat(N, 1, 1).to(device).detach()

    def forward(self, imgs):
        N = imgs.shape[0]
        device = imgs.device

        img_feats = self.backbone(imgs)
        # add view dimension (single view)
        img_feats = [i.unsqueeze(1) for i in img_feats]

        P = [self._get_projection_matrix(N, device)]

        init_meshes = ico_sphere(self.ico_sphere_level, device).extend(N)
        refined_meshes = self.mesh_head(img_feats, init_meshes, P)
        return None, refined_meshes


@MESH_ARCH_REGISTRY.register()
class Pixel2MeshHead(nn.Module):
    def __init__(self, cfg):
        super(Pixel2MeshHead, self).__init__()

        # fmt: off
        backbone                = cfg.MODEL.BACKBONE
        self.ico_sphere_level   = cfg.MODEL.MESH_HEAD.ICO_SPHERE_LEVEL
        # fmt: on

        self.register_buffer("K", get_blender_intrinsic_matrix())
        # backbone
        self.backbone, feat_dims = build_backbone(backbone)
        # mesh head
        cfg.MODEL.MESH_HEAD.COMPUTED_INPUT_CHANNELS = sum(feat_dims)
        self.mesh_head = MeshRefinementHead(cfg)

    def _get_projection_matrix(self, N, device):
        return self.K[None].repeat(N, 1, 1).to(device).detach()

    def forward(self, imgs):
        N = imgs.shape[0]
        device = imgs.device

        img_feats = self.backbone(imgs)
        # add view dimension (single view)
        img_feats = [i.unsqueeze(1) for i in img_feats]

        P = [self._get_projection_matrix(N, device)]

        init_meshes = ico_sphere(self.ico_sphere_level, device).extend(N)
        refined_meshes = self.mesh_head(img_feats, init_meshes, P, subdivide=True)
        return None, refined_meshes


def build_model(cfg):
    name = cfg.MODEL.MESH_HEAD.NAME
    return MESH_ARCH_REGISTRY.get(name)(cfg)
