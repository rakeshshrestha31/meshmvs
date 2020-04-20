# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.utils.registry import Registry
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere

import time
import cv2

from shapenet.modeling.backbone import build_backbone, build_custom_backbone
from shapenet.modeling.heads import MeshRefinementHead, VoxelHead, MVSNet
from shapenet.modeling.voxel_ops import \
        dummy_mesh, add_dummy_meshes, cubify, merge_multi_view_voxels, logit
from shapenet.utils.coords import \
        get_blender_intrinsic_matrix, relative_extrinsics
from shapenet.data.utils import imagenet_deprocess

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
            dummy_meshes = dummy_mesh(N, device)
            dummy_refined = self.mesh_head(img_feats, dummy_meshes, P)
            return voxel_scores, dummy_refined

        cubified_meshes = cubify(
            voxel_scores[0], self.voxel_size, self.cubify_threshold
        )
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
        self.cubify_threshold_logit = logit(self.cubify_threshold)

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

        # debug only
        # timestamp = int(time.time() * 1000)
        # save_images(imgs, timestamp)

        K = self._get_projection_matrix(batch_size, device)
        rel_extrinsics = relative_extrinsics(extrinsics, extrinsics[:, 0])
        P = [K.bmm(T) for T in rel_extrinsics.unbind(dim=1)]
        voxel_scores = merge_multi_view_voxels(
            voxel_scores, extrinsics, self.voxel_size, self.cubify_threshold,
            # logit score that makes a cell non-occupied
            self.cubify_threshold_logit - 1e-1
        )

        if voxel_only:
            dummy_meshes = dummy_mesh(batch_size, device)
            dummy_refined = self.mesh_head(img_feats, dummy_meshes, P)
            return voxel_scores, dummy_refined

        cubified_meshes = cubify(
            voxel_scores[0], self.voxel_size, self.cubify_threshold
        )
        refined_meshes = self.mesh_head(img_feats, cubified_meshes, P)
        return voxel_scores, refined_meshes


@MESH_ARCH_REGISTRY.register()
class VoxMeshDepthHead(VoxMeshMultiViewHead):
    def __init__(self, cfg):
        nn.Module.__init__(self)
        VoxMeshMultiViewHead.setup(self, cfg)

        self.contrastive_depth_input = cfg.MODEL.CONTRASTIVE_DEPTH_INPUT
        self.mvsnet = MVSNet(cfg.MODEL.MVSNET)
        if cfg.MODEL.RGB_FEATURES_INPUT:
            self.rgb_cnn, rgb_feat_dims \
                    = build_backbone(cfg.MODEL.BACKBONE)
        else:
            self.rgb_cnn, rgb_feat_dims = None, [0]

        self.pre_voxel_depth_cnn, pre_voxel_depth_feat_dims \
            = build_custom_backbone(cfg.MODEL.DEPTH_BACKBONE, 1)

        if self.contrastive_depth_input:
            self.post_voxel_depth_cnn, post_voxel_depth_feat_dims \
                = build_custom_backbone(cfg.MODEL.DEPTH_BACKBONE, 2)
        else:
            # can reuse same features used for voxel prediction
            # self.post_voxel_depth_cnn = self.pre_voxel_depth_cnn
            post_voxel_depth_feat_dims = pre_voxel_depth_feat_dims

        # voxel head
        cfg.MODEL.VOXEL_HEAD.COMPUTED_INPUT_CHANNELS = \
                rgb_feat_dims[-1] + pre_voxel_depth_feat_dims[-1]
        self.voxel_head = VoxelHead(cfg)
        # mesh head
        # times 3 cuz multi-view (mean, avg, std) features will be used
        # TODO: attention-based stuffs
        cfg.MODEL.MESH_HEAD.COMPUTED_INPUT_CHANNELS \
                = (sum(rgb_feat_dims) + sum(post_voxel_depth_feat_dims)) * 3
        self.mesh_head = MeshRefinementHead(cfg)

    def predict_depths(self, imgs, extrinsics):
        """
        Gets predicted depths and depth features

        Args:
        - imgs: tensor of shape (B, V, 3, H, W)
        - intrinsics: tensor of shape (B, V, 4, 4)
        - extrinsics: tensor of shape (B, V, 4, 4)

        Returns:
        - depths: tensor of shape (B, V, 1, H, W)
        - depth features: list of tenosrs of shape (B*V, C, H, W)
        """
        mvsnet_output = self.mvsnet(imgs, extrinsics)
        # flatten batch/size and add channel dimension: (B*V, 1, H, W)
        depths = mvsnet_output["depths"].unsqueeze(2)
        depths = depths.view(-1, *(depths.shape[2:]))
        depths = F.interpolate(
            depths, imgs.shape[-2:], mode="bilinear", align_corners=False
        )
        # features shape: (B*V, C, H, W)
        depth_feats = self.pre_voxel_depth_cnn(depths)
        batch_size, num_views = mvsnet_output["depths"].shape[:2]
        # (B, V, 1, H, W)
        depth = depths.view(batch_size, num_views, *(depths.shape[1:]))
        return depth, depth_feats

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

        if self.rgb_cnn is not None:
            # features shape: (B*V, C, H, W)
            img_feats = self.rgb_cnn(imgs.view(-1, *(imgs.shape[2:])))
        else:
            img_feats = []

        depths, depth_feats = self.predict_depths(imgs, extrinsics)

        # debug only
        # timestamp = int(time.time() * 1000)
        # save_images(imgs, timestamp)
        # save_depths(depths, timestamp)

        # merge RGB and depth features
        if img_feats:
            rgbd_feats = [
                torch.cat((i, j), dim=1) for i, j in zip(img_feats, depth_feats)
            ]
        else:
            rgbd_feats = depth_feats

        voxel_scores = self.voxel_head(rgbd_feats[-1])
        # unflatten the batch and views
        voxel_scores = voxel_scores.view(
            batch_size, num_views, *(voxel_scores.shape[1:])
        )
        rgbd_feats = [
            i.view(batch_size, num_views, *(i.shape[1:])) for i in rgbd_feats
        ]

        K = self._get_projection_matrix(batch_size, device)
        rel_extrinsics = relative_extrinsics(extrinsics, extrinsics[:, 0])
        P = [K.bmm(T) for T in rel_extrinsics.unbind(dim=1)]
        voxel_scores = merge_multi_view_voxels(
            voxel_scores, extrinsics, self.voxel_size, self.cubify_threshold,
            # logit score that makes a cell non-occupied
            self.cubify_threshold_logit - 1e-1
        )

        if voxel_only:
            dummy_meshes = dummy_mesh(batch_size, device)
            dummy_refined = self.mesh_head(rgbd_feats, dummy_meshes, P)
            return voxel_scores, dummy_refined

        cubified_meshes = cubify(
            voxel_scores[0], self.voxel_size, self.cubify_threshold
        )
        refined_meshes = self.mesh_head(rgbd_feats, cubified_meshes, P)
        return voxel_scores, refined_meshes


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


@torch.no_grad()
def save_images(imgs, file_prefix):
    """
    Args:
    - imgs: tensor of shape (B, V, C, H, W)
    - file_prefix: prefix to use in the filename to distinguish batches
    """
    transform = imagenet_deprocess(False)
    for batch_idx in range(imgs.shape[0]):
        for view_idx in range(imgs.shape[1]):
            img = imgs[batch_idx, view_idx]
            img = transform(img) * 255
            img = img.type(torch.uint8).cpu().detach() \
                     .permute(1, 2, 0).numpy()
            # white background
            img[img == 0] = 255
            filename = "/tmp/image_{}_{}_{}.png" \
                            .format(file_prefix, batch_idx, view_idx)
            cv2.imwrite(filename, img)


@torch.no_grad()
def save_depths(depths, file_prefix):
    """
    Args:
    - depths: tensor of shape (B, V, 1, H, W)
    - file_prefix: prefix to use in the filename to distinguish batches
    """
    for batch_idx in range(depths.shape[0]):
        for view_idx in range(depths.shape[1]):
            depth = depths[batch_idx, view_idx, 0] * 255
            depth = depth.type(torch.uint8).cpu().detach().numpy()
            filename = "/tmp/depth_{}_{}_{}.png" \
                            .format(file_prefix, batch_idx, view_idx)
            cv2.imwrite(filename, depth)
