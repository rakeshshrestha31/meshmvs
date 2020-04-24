import json
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from torch.utils.data import Dataset

import torchvision.transforms as T
from PIL import Image
import cv2
from shapenet.data.utils import imagenet_preprocess
from shapenet.utils.coords import SHAPENET_MAX_ZMAX, SHAPENET_MIN_ZMIN, project_verts
from .mesh_vox_multi_view import MeshVoxMultiViewDataset

logger = logging.getLogger(__name__)

# 0.57 is the scaling used by the 3D-R2N2 dataset
# 1000 is the scale applied for saving depths as ints
DEPTH_SCALE = 0.57 * 1000


class MeshVoxDepthDataset(MeshVoxMultiViewDataset):
    def __init__(
        self,
        data_dir,
        normalize_images=True,
        split=None,
        return_mesh=False,
        voxel_size=32,
        num_samples=5000,
        sample_online=False,
        in_memory=False,
        return_id_str=False,
        depth_only=False,
    ):
        MeshVoxMultiViewDataset.__init__(
            self, data_dir, normalize_images=normalize_images,
            split=split, return_mesh=return_mesh, voxel_size=voxel_size,
            num_samples=num_samples, sample_online=sample_online,
            in_memory=in_memory, return_id_str=return_id_str
        )
        self.set_depth_only(depth_only)

    def set_depth_only(self, value):
        self.depth_only = value

    @staticmethod
    def read_depth(data_dir, sid, mid, iid):
        depth_file = os.path.join(
            data_dir, sid, mid, "rendering_depth", str(iid).zfill(2) + ".png"
        )
        if os.path.isfile(depth_file):
            depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
            depth = depth.astype(np.float32) / DEPTH_SCALE
            depth = torch.from_numpy(depth)
            return depth
        else:
            print('depth file not found:', depth_file)
            exit(1)

    @staticmethod
    def read_mask(data_dir, sid, mid, img_path):
        img_path = os.path.join(data_dir, sid, mid, "images", img_path)
        rgbda_img = cv2.imread(img_path, -1)
        mask = rgbda_img[:, :, -1]
        mask = mask > 1e-7
        return torch.from_numpy(mask).float()

    def __getitem__(self, idx):
        sid = self.synset_ids[idx]
        mid = self.model_ids[idx]
        metadata = self.read_camera_parameters(self.data_dir, sid, mid)

        depths = []
        masks = []
        for iid in self.image_ids:
            img_path = metadata["image_list"][iid]
            depths.append(self.read_depth(self.data_dir, sid, mid, iid))
            masks.append(self.read_mask(self.data_dir, sid, mid, img_path))

        depths = torch.stack(depths, dim=0)
        masks = torch.stack(masks, dim=0)
        masks = F.interpolate(
            masks.view(-1, 1, *(masks.shape[1:])),
            depths.shape[-2:], mode="bilinear", align_corners=False
        ).view(*(depths.shape))

        if self.depth_only:
            # depths, masks, images and camera parameters
            K = metadata["intrinsic"]
            imgs = torch.stack([
                self.transform(self.read_image(
                    self.data_dir, sid, mid, metadata["image_list"][iid]
                ))
                for iid in self.image_ids
            ], dim=0)
            extrinsics = torch.stack(
                [metadata["extrinsics"][iid] for iid in self.image_ids], dim=0
            )
            return {
                "depths": depths, "masks": masks, "imgs": imgs,
                "intrinsics": K, "extrinsics": extrinsics
            }
        else:
            return {
                **MeshVoxMultiViewDataset.__getitem__(self, idx),
                "depths": depths, "masks": masks
            }

