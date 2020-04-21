import json
import logging
import os
import numpy as np
import torch
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
    ):
        MeshVoxMultiViewDataset.__init__(
            self, data_dir, normalize_images=normalize_images,
            split=split, return_mesh=return_mesh, voxel_size=voxel_size,
            num_samples=num_samples, sample_online=sample_online,
            in_memory=in_memory, return_id_str=return_id_str
        )

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

    def __getitem__(self, idx):
        sid = self.synset_ids[idx]
        mid = self.model_ids[idx]

        depths = []
        for iid in self.image_ids:
            depths.append(self.read_depth(self.data_dir, sid, mid, iid))

        depths = torch.stack(depths, dim=0)
        # TODO: get mask from the alpha channel of RGB instead
        masks = (depths > 1e-7).float()

        return {
            **MeshVoxMultiViewDataset.__getitem__(self, idx),
            "depths": depths, "masks": masks
        }

