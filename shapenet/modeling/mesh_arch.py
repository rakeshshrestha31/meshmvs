# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.utils.registry import Registry
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere

import time
import cv2
import functools

from shapenet.modeling.backbone import build_backbone, build_custom_backbone
from shapenet.modeling.heads import \
        MeshRefinementHead, VoxelHead, MVSNet, DepthRenderer, \
        MultiHeadAttentionFeaturePooling, SimpleAttentionFeaturePooling
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

    @staticmethod
    def extract_img_features(meshes, img_feats):
        """returns img features regardless of the meshes"""
        return {"img_feats": img_feats}

    def forward(self, imgs, voxel_only=False):
        N = imgs.shape[0]
        device = imgs.device

        img_feats = self.backbone(imgs)
        # voxel scores from one view only
        voxel_scores = [self.voxel_head(img_feats[-1])]
        # add view dimension (single view)
        img_feats = [i.unsqueeze(1) for i in img_feats]
        P = [self._get_projection_matrix(N, device)]
        feats_extractor = functools.partial(
            self.extract_img_features, img_feats=img_feats
        )

        if voxel_only:
            dummy_meshes = dummy_mesh(N, device)
            dummy_refined, _, _ = self.mesh_head(
                feats_extractor, dummy_meshes, P
            )
            return {
                "voxel_scores": voxel_scores, "meshes_pred": dummy_refined,
            }

        cubified_meshes = cubify(
            voxel_scores[0], self.voxel_size, self.cubify_threshold
        )
        refined_meshes, _, _ = self.mesh_head(
            feats_extractor, cubified_meshes, P
        )
        return {
            "voxel_scores": voxel_scores, "meshes_pred": refined_meshes,
        }


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

        # multi-view feature fusion
        prefusion_feat_dims = sum(feat_dims)
        postfusion_feat_dims = self.init_feature_fusion(
            cfg, prefusion_feat_dims
        )

        # mesh head
        # times 3 cuz multi-view (mean, avg, std) features will be used
        cfg.MODEL.MESH_HEAD.COMPUTED_INPUT_CHANNELS = postfusion_feat_dims
        self.mesh_head = MeshRefinementHead(cfg, self.fuse_multiview_features)

    def setup(self, cfg):
        VoxMeshHead.setup(self, cfg)
        self.cubify_threshold_logit = logit(self.cubify_threshold)

    @staticmethod
    def extract_feature_stats(features):
        """
        Extract feature statistics (max, mean, std) from multi-view features

        Args:
        - vert_aligned_feats: tensors of shape (V, N, C)

        Returns:
        - feature stats: tensor of shape (N, C*3)
        - weights: tensor of shape (N, V, 1)
        """
        max_features = torch.max(features, dim=0)[0]
        mean_features = torch.mean(features, dim=0)
        var_features = torch.var(features, dim=0, unbiased=False)
        # calculating std using torch methods give NaN gradients
        # var will have different unit that mean/max, hence std desired
        std_features = torch.sqrt(var_features + 1e-8)
        stats_features = torch.cat(
            (max_features, mean_features, std_features), dim=-1
        )

        # dummy weights (all views have the same weights)
        num_views, num_points = features.shape[:2]
        weights = torch.ones(
            (num_points, num_views, 1),
            dtype=features.dtype, device=features.device
        ) / num_views
        return stats_features, weights

    def init_feature_fusion(self, cfg, prefusion_feat_dims):
        """
        Returns:
        - feature dimensions after fusion
        """
        if cfg.MODEL.FEATURE_FUSION_METHOD == "multihead_attention":
            return self.init_multihead_attention(cfg, prefusion_feat_dims)
        elif cfg.MODEL.FEATURE_FUSION_METHOD == "simple_attention":
            return self.init_simple_attention(cfg, prefusion_feat_dims)
        elif cfg.MODEL.FEATURE_FUSION_METHOD == "stats":
            return self.init_stats_fusion(cfg, prefusion_feat_dims)
        else:
            raise ValueError(
                "Unsupported feature fusion method %s" \
                    % cfg.MODEL.FEATURE_FUSION_METHOD
            )

    def init_multihead_attention(self, cfg, prefusion_feat_dims):
        if cfg.MODEL.MULTIHEAD_ATTENTION.FEATURE_DIMS <= 0:
            postfusion_feat_dims = prefusion_feat_dims
        else:
            postfusion_feat_dims = cfg.MODEL.MULTIHEAD_ATTENTION.FEATURE_DIMS

        self.feature_fusion_head = MultiHeadAttentionFeaturePooling(
            prefusion_feat_dims, postfusion_feat_dims,
            num_heads = cfg.MODEL.MULTIHEAD_ATTENTION.NUM_HEADS,
            use_stats_query=False
        )
        return postfusion_feat_dims

    def init_simple_attention(self, cfg, prefusion_feat_dims):
        self.feature_fusion_head = \
                SimpleAttentionFeaturePooling(prefusion_feat_dims)
        # features size doesn't change cuz its weighted sum
        return prefusion_feat_dims

    def init_stats_fusion(self, cfg, prefusion_feat_dims):
        self.feature_fusion_head = self.extract_feature_stats
        # *3 for max, mean and std features
        return prefusion_feat_dims * 3

    def fuse_multiview_features(self, features):
        """
        Args:
        - features: list of tensors of shape (B, N, C_in). List len = num_views
        Returns:
        - fused feature: tensor of shape (B, N, C_out)
        - weights: tensor of shape (B, N, V, 1)
        """
        # shape (B, V, N, C_in)
        joint_features = torch.stack(features, dim=1)
        fused_features = []
        fusion_weights = []
        # each batch needs to be treated separately
        # as they'll have different view weights for attention method
        # this doesn't matter for stats fusion, but let's keep it uniform
        for batch_features in joint_features.unbind(0):
            feats, weights = self.feature_fusion_head(batch_features)
            fused_features.append(feats)
            fusion_weights.append(weights)

        fused_features = torch.stack(fused_features, dim=0)
        fusion_weights = torch.stack(fusion_weights, dim=0)
        return fused_features, fusion_weights

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
        feats_extractor = functools.partial(
            self.extract_img_features, img_feats=img_feats
        )

        # debug only
        # timestamp = int(time.time() * 1000)
        # save_images(imgs, timestamp)

        K = self._get_projection_matrix(batch_size, device)
        rel_extrinsics = relative_extrinsics(extrinsics, extrinsics[:, 0])
        P = [K.bmm(T) for T in rel_extrinsics.unbind(dim=1)]
        merged_voxel_scores, transformed_voxel_scores = merge_multi_view_voxels(
            voxel_scores, extrinsics, self.voxel_size, self.cubify_threshold,
            # logit score that makes a cell non-occupied
            self.cubify_threshold_logit - 1e-1
        )
        # separate views into list items
        voxel_scores = voxel_scores.unbind(1)

        if voxel_only:
            dummy_meshes = dummy_mesh(batch_size, device)
            dummy_refined, _, view_weights = self.mesh_head(
                feats_extractor, dummy_meshes, P
            )
            return {
                "voxel_scores": voxel_scores, "meshes_pred": dummy_refined,
                "merged_voxel_scores": merged_voxel_scores,
                "transformed_voxel_scores": transformed_voxel_scores,
                "view_weights": view_weights
            }

        cubified_meshes = cubify(
            merged_voxel_scores, self.voxel_size, self.cubify_threshold
        )
        refined_meshes, _, view_weights = self.mesh_head(
            feats_extractor, cubified_meshes, P
        )
        return {
            "voxel_scores": voxel_scores, "meshes_pred": refined_meshes,
            "merged_voxel_scores": merged_voxel_scores,
            "transformed_voxel_scores": transformed_voxel_scores,
            "view_weights": view_weights
        }


@MESH_ARCH_REGISTRY.register()
class VoxMeshDepthHead(VoxMeshMultiViewHead):
    def __init__(self, cfg):
        nn.Module.__init__(self)
        VoxMeshMultiViewHead.setup(self, cfg)

        self.contrastive_depth_input = cfg.MODEL.CONTRASTIVE_DEPTH_INPUT
        self.mvsnet_image_size = torch.tensor(cfg.MODEL.MVSNET.INPUT_IMAGE_SIZE)

        self.mvsnet = MVSNet(cfg.MODEL.MVSNET) if not cfg.MODEL.USE_GT_DEPTH \
                        else None
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

        # depth renderer
        self.depth_renderer = DepthRenderer(cfg)

        # multi-view feature fusion
        prefusion_feat_dims = sum(rgb_feat_dims) \
                            + sum(post_voxel_depth_feat_dims)
        postfusion_feat_dims = self.init_feature_fusion(
            cfg, prefusion_feat_dims
        )

        # mesh head
        # times 3 cuz multi-view (mean, avg, std) features will be used
        # TODO: attention-based stuffs
        cfg.MODEL.MESH_HEAD.COMPUTED_INPUT_CHANNELS = postfusion_feat_dims
        self.mesh_head = MeshRefinementHead(cfg, self.fuse_multiview_features)

        print({
            "rgb_feat_dims": rgb_feat_dims,
            "pre_voxel_depth_feat_dims": pre_voxel_depth_feat_dims,
            "post_voxel_depth_feat_dims": post_voxel_depth_feat_dims,
            "vox_head_input": cfg.MODEL.VOXEL_HEAD.COMPUTED_INPUT_CHANNELS,
            "prefusion_feat_dims": prefusion_feat_dims,
            "postfusion_feat_dims": postfusion_feat_dims
        })
        print("Using GT depth input:", cfg.MODEL.USE_GT_DEPTH)

        if cfg.MODEL.VOXEL_HEAD.FREEZE:
            self.freeze_voxel_head()

    def freeze_voxel_head(self):
        modules_to_freeze = [
            self.voxel_head, self.rgb_cnn, self.pre_voxel_depth_cnn
        ]
        for module_to_freeze in modules_to_freeze:
            if module_to_freeze is None:
                continue
            for param in module_to_freeze.parameters():
                param.requires_grad = False

    def extract_rgbd_features(
        self, meshes, rgbd_feats, extrinsics
    ):
        """
        returns rgbd features regardless of the meshes

        Args:
        - meshes (Meshes)
        - rgbd_feats (tensor):
            Tensor of shape (B, V, C, H, W) giving image features,
            or a list of such tensors.
        - extrinsics (list of tensors): list of (B, 4, 4) transformations
        - img_shape (array like): 2D array of (H, W)
        Returns:
        - feats (tensor): Tensor of shape (B, V, C, H, W) giving image features,
                              or a list of such tensors.
        - rendered_depths (tensor): shape (B, V, H, W)
        """
        rendered_depths = self.depth_renderer(
            meshes.verts_padded(), meshes.faces_padded(),
            extrinsics, self.mvsnet_image_size
        )
        return {
            "img_feats": rgbd_feats,
            "rendered_depths": rendered_depths
        }

    def extract_contrastive_features(
        self, meshes, pred_depths, extrinsics
    ):
        """
        contrastive depth feature extractor

        Args:
        - meshes (Meshes)
        - pred_depths (tensor): shape (B, V, H, W)
        - extrinsics (list of tensors): list of (B, 4, 4) transformations
        Returns:
        - feats (tensor): Tensor of shape (B, V, C, H, W) giving image features,
                              or a list of such tensors.
        - rendered_depths (tensor): shape (B, V, H, W)
        """
        batch_size, num_views = pred_depths.shape[:2]
        rendered_depths = self.depth_renderer(
            meshes.verts_padded(), meshes.faces_padded(),
            extrinsics, self.mvsnet_image_size
        )
        pred_depths = F.interpolate(
            pred_depths, rendered_depths.shape[-2:], mode="nearest"
        )
        # (B, V, 2, H, W)
        contrastive_input = torch.stack((pred_depths, rendered_depths), dim=2)
        # flattened batch/views (BxV, 2, H, W)
        contrastive_input = contrastive_input \
                                .view(-1, *(contrastive_input.shape[2:]))
        # list of (B*V, C, H, W)
        feats = self.post_voxel_depth_cnn(contrastive_input)
        # unflatten batch/views: (B, V, C, H, W)
        feats = [
            i.view(batch_size, num_views, *(i.shape[1:])) for  i in feats
        ]
        return {
            "img_feats": feats,
            "rendered_depths": rendered_depths
        }

    def predict_depths(self, imgs, masks, extrinsics, **kwargs):
        """
        Gets predicted depths and depth features

        Args:
        - imgs: tensor of shape (B, V, 3, H, W)
        - intrinsics: tensor of shape (B, V, 4, 4)
        - extrinsics: tensor of shape (B, V, 4, 4)

        Returns:
        - depths: tensor of shape (B, V, H, W)
        - depth features: list of tenosrs of shape (B*V, C, H, W)
        """
        # flatten batch/size and add channel dimension: (B*V, 1, H, W)
        def interpolate(tensor, size):
            """ (B, V, H1, W1) -> (B*V, 1, H1, W1)
            """
            # (B*V, 1, H, W)
            tensor = tensor.view(-1, 1, *(tensor.shape[2:]))
            # (B*V, 1, H, W)
            return F.interpolate(tensor, size, mode="nearest")

        if self.mvsnet is None:
            if "depths" in kwargs:
                gt_depth = kwargs["depths"]
                gt_depth = interpolate(gt_depth, (56, 56)) \
                                .view(*(gt_depth.shape[:2]), 56, 56)
                mvsnet_output = {"depths": gt_depth}
            else:
                print("both gt depth and mvsnet unavailable")
                exit(1)
        else:
            mvsnet_output = self.mvsnet(imgs, extrinsics)
        depths = interpolate(mvsnet_output["depths"], imgs.shape[-2:])
        masks = interpolate(masks, imgs.shape[-2:])
        depths = depths * masks

        # features shape: (B*V, C, H, W)
        depth_feats = self.pre_voxel_depth_cnn(depths)
        batch_size, num_views = mvsnet_output["depths"].shape[:2]
        # (B, V, 1, H, W)
        depths = depths.view(batch_size, num_views, *(depths.shape[-2:]))

        return mvsnet_output["depths"], depth_feats

    def forward(self, imgs, intrinsics, extrinsics, masks, voxel_only=False, **kwargs):
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

        depths, depth_feats = self.predict_depths(imgs, masks, extrinsics, **kwargs)
        masks_resized = F.interpolate(masks, depths.shape[-2:], mode="nearest")
        masked_depths = depths * masks_resized

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
        if img_feats:
            img_feats = [
                i.view(batch_size, num_views, *(i.shape[1:])) for i in img_feats
            ]

        K = self._get_projection_matrix(batch_size, device)
        rel_extrinsics = relative_extrinsics(extrinsics, extrinsics[:, 0])
        P = [K.bmm(T) for T in rel_extrinsics.unbind(dim=1)]
        merged_voxel_scores, transformed_voxel_scores = merge_multi_view_voxels(
            voxel_scores, extrinsics, self.voxel_size, self.cubify_threshold,
            # logit score that makes a cell non-occupied
            self.cubify_threshold_logit - 1e-1
        )
        # separate views into list items
        voxel_scores = voxel_scores.unbind(1)

        if self.contrastive_depth_input:
            def feats_extractor(*args, **kwargs):
                nonlocal masked_depths, rel_extrinsics
                feats = self.extract_contrastive_features(
                    *args, **kwargs,
                    pred_depths=masked_depths, extrinsics=rel_extrinsics,
                )
                feats["img_feats"] += img_feats
                return feats
            # add image features
        else:
            feats_extractor = functools.partial(
                self.extract_rgbd_features, rgbd_feats=rgbd_feats,
                extrinsics=rel_extrinsics
            )

        if voxel_only:
            dummy_meshes = dummy_mesh(batch_size, device)
            dummy_refined, mesh_features, view_weights = self.mesh_head(
                feats_extractor, dummy_meshes, P
            )
            rendered_depths = [i["rendered_depths"] for i in mesh_features]
            return {
                "voxel_scores": voxel_scores, "meshes_pred": dummy_refined,
                "merged_voxel_scores": merged_voxel_scores,
                "transformed_voxel_scores": transformed_voxel_scores,
                "pred_depths": depths, "rendered_depths": rendered_depths,
                "view_weights": view_weights
            }

        cubified_meshes = cubify(
            merged_voxel_scores, self.voxel_size, self.cubify_threshold
        )

        refined_meshes, mesh_features, view_weights = self.mesh_head(
            feats_extractor, cubified_meshes, P
        )

        rendered_depths = [i["rendered_depths"] for i in mesh_features]
        # add rendered depth of the final mesh
        rendered_depths.append(self.depth_renderer(
            refined_meshes[-1].verts_padded(),
            refined_meshes[-1].faces_padded(),
            rel_extrinsics, self.mvsnet_image_size
        ))

        # debug only
        # timestamp = int(time.time() * 1000)
        # save_images(imgs, timestamp)
        # save_depths(masked_depths, timestamp)
        # save_depths(masks, str(timestamp)+"_mask")
        # for i, rendered_depth in enumerate(rendered_depths):
        #     save_depths(rendered_depth, "%d_rendered_%d" % (timestamp, i))
        # exit(0)

        return {
            "voxel_scores": voxel_scores, "meshes_pred": refined_meshes,
            "merged_voxel_scores": merged_voxel_scores,
            "transformed_voxel_scores": transformed_voxel_scores,
            "pred_depths": depths, "rendered_depths": rendered_depths,
            "view_weights": view_weights
        }


@MESH_ARCH_REGISTRY.register()
class VoxDepthHead(VoxMeshDepthHead):
    """
    voxel prediction with depth features
    """
    def __init__(self, cfg):
        nn.Module.__init__(self)
        VoxMeshMultiViewHead.setup(self, cfg)

        self.single_view_voxel_prediction = cfg.MODEL.VOXEL_HEAD.SINGLE_VIEW
        self.mvsnet_image_size = torch.tensor(cfg.MODEL.MVSNET.INPUT_IMAGE_SIZE)
        self.mvsnet = MVSNet(cfg.MODEL.MVSNET) if not cfg.MODEL.USE_GT_DEPTH \
                        else None
        if cfg.MODEL.RGB_FEATURES_INPUT:
            self.rgb_cnn, rgb_feat_dims \
                    = build_backbone(cfg.MODEL.BACKBONE)
        else:
            self.rgb_cnn, rgb_feat_dims = None, [0]

        self.pre_voxel_depth_cnn, pre_voxel_depth_feat_dims \
            = build_custom_backbone(cfg.MODEL.DEPTH_BACKBONE, 1)

        # voxel head
        cfg.MODEL.VOXEL_HEAD.COMPUTED_INPUT_CHANNELS = \
                rgb_feat_dims[-1] + pre_voxel_depth_feat_dims[-1]
        self.voxel_head = VoxelHead(cfg)

        print({
            "rgb_feat_dims": rgb_feat_dims,
            "pre_voxel_depth_feat_dims": pre_voxel_depth_feat_dims,
            "vox_head_input": cfg.MODEL.VOXEL_HEAD.COMPUTED_INPUT_CHANNELS,
        })
        print("Using GT depth input:", cfg.MODEL.USE_GT_DEPTH)

        if cfg.MODEL.VOXEL_HEAD.FREEZE:
            self.freeze_voxel_head()

    def forward(
        self, imgs, intrinsics, extrinsics, masks,
        voxel_only=False, **kwargs
    ):
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

        depths, depth_feats = self.predict_depths(imgs, masks, extrinsics, **kwargs)
        masks_resized = F.interpolate(masks, depths.shape[-2:], mode="nearest")
        masked_depths = depths * masks_resized

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
        if self.single_view_voxel_prediction:
            merged_voxel_scores = voxel_scores[:, 0]
            transformed_voxel_scores = merged_voxel_scores
            voxel_scores = [voxel_scores[:, 0]]
        else:
            K = self._get_projection_matrix(batch_size, device)
            rel_extrinsics = relative_extrinsics(extrinsics, extrinsics[:, 0])
            P = [K.bmm(T) for T in rel_extrinsics.unbind(dim=1)]
            merged_voxel_scores, transformed_voxel_scores \
                = merge_multi_view_voxels(
                    voxel_scores, extrinsics,
                    self.voxel_size, self.cubify_threshold,
                    # logit score that makes a cell non-occupied
                    self.cubify_threshold_logit - 1e-1
                )
            # separate views into list items
            voxel_scores = voxel_scores.unbind(1)

        return {
            "voxel_scores": voxel_scores,
            "merged_voxel_scores": merged_voxel_scores,
            "transformed_voxel_scores": transformed_voxel_scores,
            "pred_depths": depths,
        }


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
        return {
            "voxel_scores":None, "meshes_pred": refined_meshes,
        }


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
        return {
            "voxel_scores":None, "meshes_pred": refined_meshes,
        }


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
def save_depths(depths, file_prefix, size=None):
    """
    Args:
    - depths: tensor of shape (B, V, H, W)
    - file_prefix: prefix to use in the filename to distinguish batches
    - size: size to the depth image to resize to (optional)
    """
    if size is not None:
        depths = F.interpolate(
            depths.view(-1, 1, *(depths.shape[-2:])), size, mode='nearest'
        ).view(*(depths.shape[:2]), *size)
    for batch_idx in range(depths.shape[0]):
        for view_idx in range(depths.shape[1]):
            depth = depths[batch_idx, view_idx] / 2.5 * 255
            depth = depth.type(torch.uint8).cpu().detach().numpy()
            filename = "/tmp/depth_{}_{}_{}.png" \
                            .format(file_prefix, batch_idx, view_idx)
            cv2.imwrite(filename, depth)
