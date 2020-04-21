import json
import logging
import os
import torch
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from torch.utils.data import Dataset

import torchvision.transforms as T
from PIL import Image
from shapenet.data.utils import imagenet_preprocess
from shapenet.utils.coords import SHAPENET_MAX_ZMAX, SHAPENET_MIN_ZMIN, project_verts
from .mesh_vox import MeshVoxDataset

logger = logging.getLogger(__name__)


class MeshVoxMultiViewDataset(MeshVoxDataset):
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
        # call the PyTorch Dataset interface in this way
        # since the immediate parent is MeshVoxDataset
        Dataset.__init__(self)
        if not return_mesh and sample_online:
            raise ValueError("Cannot sample online without returning mesh")

        self.data_dir = data_dir
        self.return_mesh = return_mesh
        self.voxel_size = voxel_size
        self.num_samples = num_samples
        self.sample_online = sample_online
        self.return_id_str = return_id_str

        self.synset_ids = []
        self.model_ids = []
        self.mid_to_samples = {}
        # TODO: get the image ids from parameters
        self.image_ids = [0, 6, 7]

        self.transform = self.get_transform(normalize_images)

        summary_json = os.path.join(data_dir, "summary.json")
        with open(summary_json, "r") as f:
            summary = json.load(f)
            for sid in summary:
                logger.info("Starting synset %s" % sid)
                allowed_mids = None
                if split is not None:
                    if sid not in split:
                        logger.info("Skipping synset %s" % sid)
                        continue
                    elif isinstance(split[sid], list):
                        allowed_mids = set(split[sid])
                    elif isinstance(split, dict):
                        allowed_mids = set(split[sid].keys())
                for mid, num_imgs in summary[sid].items():
                    if allowed_mids is not None and mid not in allowed_mids:
                        continue
                    if not sample_online and in_memory:
                        samples_path = os.path.join(data_dir, sid, mid, "samples.pt")
                        samples = torch.load(samples_path)
                        self.mid_to_samples[mid] = samples
                    self.synset_ids.append(sid)
                    self.model_ids.append(mid)

    def __getitem__(self, idx):
        sid = self.synset_ids[idx]
        mid = self.model_ids[idx]
        ref_iid = self.image_ids[0]

        metadata = self.read_camera_parameters(self.data_dir, sid, mid)
        K = metadata["intrinsic"]

        imgs = []
        extrinsics = []
        for iid in self.image_ids:
            img_path = metadata["image_list"][iid]
            img = self.read_image(self.data_dir, sid, mid, img_path)
            img = self.transform(img)
            imgs.append(img)
            extrinsics.append(metadata["extrinsics"][iid])

        imgs = torch.stack(imgs, dim=0)
        extrinsics = torch.stack(extrinsics, dim=0)
        RT = extrinsics[0]

        # Maybe read mesh
        verts, faces = None, None
        if self.return_mesh:
            verts, faces = self.read_mesh(self.data_dir, sid, mid, RT)

        # Maybe use cached samples
        points, normals = None, None
        if not self.sample_online:
            points, normals = self.sample_points_normals(
                self.data_dir, sid, mid, RT
            )

        voxels, P = None, None
        if self.voxel_size > 0:
            voxels, P = self.read_voxels(
                self.data_dir, sid, mid, ref_iid, K, RT
            )

        id_str = "%s-%s-%02d" % (sid, mid, ref_iid)
        return {
            "imgs": imgs, "verts": verts, "faces": faces, "points": points,
            "normals": normals, "voxels": voxels, "Ps": P, "id_str": id_str,
            "intrinsics": K, "extrinsics": extrinsics
        }
