#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import open3d as o3d
import logging
import os
import shutil
import time
import numpy as np
import tqdm
from pathlib import Path
from PIL import Image
import cv2
import copy

os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import pyrender
import trimesh

import detectron2.utils.comm as comm
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.logger import setup_logger
from fvcore.common.file_io import PathManager
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import save_obj
from pytorch3d.transforms import Transform3d

from shapenet.config import get_shapenet_cfg
from shapenet.data import build_data_loader, register_shapenet
from shapenet.data.mesh_vox import MeshVoxDataset
from shapenet.evaluation import \
        evaluate_split, evaluate_test, evaluate_test_p2m, evaluate_vox

# required so that .register() calls are executed in module scope
from shapenet.modeling import MeshLoss, build_model
from shapenet.modeling.heads.depth_loss import adaptive_berhu_loss
from shapenet.modeling.mesh_arch import VoxMeshMultiViewHead, VoxMeshDepthHead
from shapenet.solver import build_lr_scheduler, build_optimizer
from shapenet.utils import Checkpoint, Timer, clean_state_dict, default_argument_parser
from shapenet.utils.depth_backprojection import get_points_from_depths
from meshrcnn.utils.metrics import compare_meshes
from shapenet.utils.coords import relative_extrinsics, get_blender_intrinsic_matrix

from tools.train_net_shapenet import get_dataset_name, setup, save_debug_predictions

logger = logging.getLogger("shapenet")


def main_worker_eval(worker_id, args):

    device = torch.device("cuda:%d" % worker_id)
    cfg = setup(args)

    # build test set
    test_loader = build_data_loader(
        cfg, get_dataset_name(cfg), "test", multigpu=False
    )
    logger.info("test - %d" % len(test_loader))

    # load checkpoing and build model
    if cfg.MODEL.CHECKPOINT == "":
        raise ValueError("Invalid checkpoing provided")
    logger.info("Loading model from checkpoint: %s" % (cfg.MODEL.CHECKPOINT))
    cp = torch.load(PathManager.get_local_path(cfg.MODEL.CHECKPOINT))

    if args.eval_latest_checkpoint:
        logger.info("using latest checkpoint weights")
        state_dict = clean_state_dict(cp["latest_states"]["model"])
    else:
        logger.info("using best checkpoint weights")
        state_dict = clean_state_dict(cp["best_states"]["model"])

    # build test set
    test_loader = build_data_loader(
        cfg, get_dataset_name(cfg), "test", multigpu=False
    )

    logger.info("test - %d" % len(test_loader))

    batch = read_batch()
    # batch = next(iter(test_loader))
    print('batch keys:', list(batch.keys()))

    batch = test_loader.postprocess(batch, device)
    print('intrinsics conf', cfg.MODEL.MVSNET.FOCAL_LENGTH, cfg.MODEL.MVSNET.PRINCIPAL_POINT)
    print('intrinsics', batch["intrinsics"])

    # cfg.MODEL.MVSNET.FOCAL_LENGTH = (
    #     batch["intrinsics"][0, 0, 0].item(), batch["intrinsics"][0, 1, 1].item()
    # )
    # cfg.MODEL.MVSNET.PRINCIPAL_POINT = (
    #     batch["intrinsics"][0, 0, 2].item(), batch["intrinsics"][0, 1, 2].item()
    # )
    # print(cfg.MODEL.MVSNET.FOCAL_LENGTH, cfg.MODEL.MVSNET.PRINCIPAL_POINT)

    model = build_model(cfg)
    model.load_state_dict(state_dict)
    logger.info("Model loaded")
    model.to(device)

    val_loader = build_data_loader(
        cfg, get_dataset_name(cfg), "test", multigpu=False
    )
    logger.info("val - %d" % len(val_loader))
    test_metrics, test_preds = evaluate_split(
        model, val_loader, prefix="val_", max_predictions=100
    )
    str_out = "Results on test"
    for k, v in test_metrics.items():
        str_out += "%s %.4f " % (k, v)
    logger.info(str_out)

    from shapenet.utils.coords import voxel_coords_to_world, voxel_grid_coords
    from shapenet.modeling.voxel_ops import cubify
    import open3d as o3d

    MIN_X = -1.1161
    MIN_Y = -1.1161
    MIN_Z = -2.4414
    MAX_X = 1.1161
    MAX_Y = 1.1161
    MAX_Z = -0.6030

    # batch = test_loader.dataset[2748]
    # batch = test_loader.dataset.collate_fn([batch])

    renderer = setup_renderer()

    for batch in tqdm.tqdm(test_loader):
    # if True:
        sid, mid = batch["id_strs"][0].split('-')[0:2]
        # if sid != "03001627" or mid != "cc8fe2000b1471b2a85f7c85e000fc79":
        #     continue

        if '_'.join([sid, mid]) not in [
            "03001627_df55d3e445f11f909a8ef44e1d2c5b75",
            "03001627_faef9e4cff5fa61987be36ce60737655",
            "03001627_eb7c250519101dc22f21cf17406f1f25",
            "02691156_da58b3e055132c9f6afab9f956f15ea",
            "02691156_df6aae66a8c378ae9029a69fa5fc9ad",
            "02691156_e31da3ac74fa3c0c23db3adbb2f1dce",
            "02691156_e02485f093835f45c1b64d86df61366a",
            "02691156_ebe0d0bfa6ec36edd88eab18f1be033b",
            "02691156_fbf6917bdd86d5862df404314e891e08",
            "02691156_f7160900b6ce7bc4e63e266a803d9270",
            "04379243_fc7d921df59e86e6beedb4c8fd29e2d1",
            "04379243_fb50672ad3f7b196cae684aee7caa8d9",
            "04379243_e750a8adb862c9f654f948e69de0f232",
            "04379243_ddeb44a5621da142aa29e9f0529e8ef7",
            "04379243_dd4f28a0e0d3f93c614a26402360d21a",
            "04379243_d0b38b27495542461b02cde7e81f0fc3",
            "02828884_e2be5da815f914f22250bf58700b4d8f",
        ]:
            continue

        # if Path('/tmp/grid_{}_{}.png'.format(sid, mid)).exists():
        #     print(sid, '_', mid, 'already done')
        #     continue

        img1 = get_image(test_loader.dataset.data_dir, sid, mid, '00.png')
        img2 = get_image(test_loader.dataset.data_dir, sid, mid, '06.png')
        img3 = get_image(test_loader.dataset.data_dir, sid, mid, '07.png')

        if None in [img1, img2, img3]:
            continue

        batch = test_loader.postprocess(batch, device)
        model_kwargs = {
            key: batch[key]
            for key in ['intrinsics', 'extrinsics', 'masks', 'depths']
        }
        model_outputs = model(batch["imgs"].to(device), **model_kwargs)

        extrinsics = batch["extrinsics"]
        rel_extrinsics  = relative_extrinsics(extrinsics, extrinsics[:, 0])
        rendered_images = {}
        for voxel_name in ["transformed_voxel_scores", "merged_voxel_scores"]: # , "voxel_scores"]:
            voxel_views = model_outputs[voxel_name]
            if voxel_name == "merged_voxel_scores":
                voxel_views = [voxel_views]
            for view_idx, voxels in enumerate(voxel_views):
                ## new grid based on Euclidean distance
                # norm_grid = voxel_grid_coords(voxels.shape[-3:])
                # grid_points = voxel_coords_to_world(norm_grid.view(-1, 3)) \
                #                 .view(*(voxels.shape[-3:]), 3)
                # grid_points = grid_points[voxels[0] > 0.2]

                # # grid coords (0-47) in uniform Euclidean interval
                # # convert East-Down-North to East-North-Up
                # new_grid_coords = torch.stack((
                #     (grid_points[:, 0] - MIN_X) / (MAX_X - MIN_X) * (voxels.shape[-1] - 1),
                #     (grid_points[:, 2] - MIN_Z) / (MAX_Z - MIN_Z) * (voxels.shape[-3] - 1),
                #     (grid_points[:, 1] - MIN_Y) / (MAX_Y - MIN_Y) * (voxels.shape[-2] - 1),
                # ), dim=-1).long()

                # new_bool_grid = torch.zeros_like(norm_grid[..., 0]).bool()
                # for grid_coord in new_grid_coords.unbind(0):
                #     x, y, z = grid_coord.tolist()
                #     new_bool_grid[x, y, z] = 1

                # print(grid_points.shape, torch.nonzero(new_bool_grid).shape)
                # np.save(
                #     "/tmp/{}_{}.npy".format(voxel_name, view_idx),
                #     new_bool_grid.detach().cpu().numpy()
                # )

                # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
                #     grid_points.view(-1, 3).cpu().detach().numpy()
                # ))
                # o3d.io.write_point_cloud(
                #     "/tmp/{}_{}.ply".format(voxel_name, view_idx), pcd
                # )

                ## colored voxel grid
                cubified = cubify(voxels, 48, 0.2)
                mesh = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(cubified[0].verts_packed().detach().cpu().numpy()),
                    o3d.utility.Vector3iVector(cubified[0].faces_packed().detach().cpu().numpy())
                )
                colored_mesh = color_mesh(mesh)
                mesh_path = '/tmp/colored_{}_{}_{}_{}.ply'.format(
                    voxel_name, sid, mid, view_idx
                )
                o3d.io.write_triangle_mesh(mesh_path, colored_mesh)

                # mesh_node = add_mesh(renderer, mesh_path)

                # for sub_view_idx, extrinsic in enumerate(rel_extrinsics[0].unbind(0)):
                #     rendered_img = render_scene(
                #         renderer, np.linalg.inv(extrinsic.detach().cpu().numpy())
                #     )

                #     key = "{}_{}_{}".format(voxel_name, view_idx, sub_view_idx)
                #     rendered_images[key] = torch.from_numpy(rendered_img) \
                #                             .float().permute(2, 0, 1) / 255.0

                # remove_node(renderer, mesh_node)

        # image_grid = torch.stack((
        #     img1,
        #     rendered_images["transformed_voxel_scores_0_0"],
        #     rendered_images["transformed_voxel_scores_1_0"],
        #     rendered_images["transformed_voxel_scores_2_0"],
        #     rendered_images["merged_voxel_scores_0_0"],
        #     img2,
        #     rendered_images["transformed_voxel_scores_0_1"],
        #     rendered_images["transformed_voxel_scores_1_1"],
        #     rendered_images["transformed_voxel_scores_2_1"],
        #     rendered_images["merged_voxel_scores_0_1"],
        #     img3,
        #     rendered_images["transformed_voxel_scores_0_2"],
        #     rendered_images["transformed_voxel_scores_1_2"],
        #     rendered_images["transformed_voxel_scores_2_2"],
        #     rendered_images["merged_voxel_scores_0_2"],
        # ), dim=0)

        # torchvision.utils.save_image(
        #     image_grid,
        #     '/tmp/grid_{}_{}.png'.format(sid, mid),
        #     nrow=5
        # )


    renderer["renderer"].delete()

    exit(0)


    eval_scannet(model, batch)


def get_image(data_dir, sid, mid, img):
    img_path = Path(data_dir) / sid / mid / 'images' / img
    img = cv2.imread(str(img_path))
    if img is None:
        print(str(img_path), 'invalid')
        return None
    img = img[..., ::-1]
    img = torch.from_numpy(np.copy(img)).float().permute(2, 0, 1) / 255.0
    img = F.interpolate(img.unsqueeze(0), (224, 224)).squeeze(0)
    return img


def setup_renderer():
    scene = pyrender.Scene(
        ambient_light=np.array([0.35, 0.35, 0.35, 1.0])
    )

    cam = pyrender.IntrinsicsCamera(fx=248, fy=248, cx=111.5, cy=111.5)

    r = pyrender.OffscreenRenderer(
        viewport_width=224, viewport_height=224
    )

    return {
        "renderer": r,
        "scene": scene,
        "cam": cam,
    }


def add_mesh(renderer, mesh_path):
    trimesh_mesh = trimesh.load(mesh_path)
    # trimesh_mesh.apply_transform(T_world_object)
    pyrender_mesh = pyrender.Mesh.from_trimesh(trimesh_mesh)

    # the default material has metallic property and is unnecessarily reflective
    # this removes the reflectiveness
    pyrender_mesh.primitives[0].material.baseColorFactor \
        = np.ones(4).astype(np.float32)

    mesh_node = renderer["scene"].add(pyrender_mesh)

    return mesh_node


def remove_node(renderer, node):
    renderer["scene"].remove_node(node)


def render_scene(renderer_dict, pose):
    scene = renderer_dict['scene']
    r = renderer_dict['renderer']
    cam = renderer_dict['cam']

    cam_node = scene.add(cam, pose=pose)
    r.render(scene)

    rendered_color, rendered_depth = r.render(scene)
    rendered_color = cv2.cvtColor(rendered_color, cv2.COLOR_BGR2RGB)

    # remove the current cam node for the next one to be added
    scene.remove_node(cam_node)

    return rendered_color


@torch.no_grad()
def color_mesh(mesh):
    vertices = np.asarray(mesh.vertices)

    x_max = np.max(vertices[:, 0])
    x_min = np.min(vertices[:, 0])
    y_max = np.max(vertices[:, 1])
    y_min = np.min(vertices[:, 1])
    z_max = np.max(vertices[:, 2])
    z_min = np.min(vertices[:, 2])

    min_coord = np.min(mesh.vertices)

    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.stack((
            (vertices[:, 0] - x_min) / (x_max - x_min),
            (vertices[:, 1] - y_min) / (y_max - y_min),
            (vertices[:, 2] - z_min) / (z_max - z_min),
        ), axis=-1)
    )

    return mesh

@torch.no_grad()
def eval_scannet(model, batch):
    """
    This function is used save predicted and gt meshes
    """
    # Note that all eval runs on main process
    assert comm.is_main_process()
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module


    device = torch.device("cuda:0")

    model_kwargs = {
        key: batch[key]
        for key in ['intrinsics', 'extrinsics', 'masks', 'depths']
    }

    model.mvsnet = None

    # print('imgs', batch['imgs'].shape)
    # print('input', model.mvsnet.input_image_size)
    # print('intrinsics', model.mvsnet.intrinsics)
    # print('extrinsics', batch['extrinsics'])

    model_outputs = model(batch["imgs"].to(device), **model_kwargs)

    save_debug_predictions(batch, model_outputs)

    gcn_stages = range(len(model_outputs["meshes_pred"]))
    for gcn_stage in gcn_stages:
        pred_mesh = model_outputs["meshes_pred"][gcn_stage]

        batch_size = batch["imgs"].shape[0]
        for batch_idx in range(batch_size):
            pred_filename = "/tmp/scannet_out_{}.obj".format(gcn_stage)
            pred_verts, pred_faces = pred_mesh[batch_idx] \
                                        .get_mesh_verts_faces(0)
            save_obj(pred_filename, pred_verts, pred_faces)


def pose_to_opengl(cam_pose):
    # use OpenGL convention (East-Up-South) instead of East-Down-North
    rotation_correction = np.asarray([
        [1, 0, 0], [0, -1, 0], [0, 0, -1]
    ])
    cam_pose[:3, :3] = np.dot(cam_pose[:3, :3], rotation_correction)
    return cam_pose


def read_batch():
    data_root = Path('/datasets/scannet/scans')
    scene = Path('scene0001_00')
    img_path = data_root / scene / 'segmented-color'
    depth_path = data_root / scene / 'segmented-depth'
    pose_path = data_root / scene / 'poses'

    # intrinsics_path = data_root / scene / 'intrinsics' / 'intrinsic_color.txt'
    # intrinsics = np.loadtxt(str(intrinsics_path))[:3, :3].astype(float)
    # intrinsics = torch.from_numpy(intrinsics)
    intrinsics = get_blender_intrinsic_matrix()

    # resized_img_size = (960, 1280)
    # resized_img_size = (224, 224)
    scale = 0.7

    img_transform = MeshVoxDataset.get_transform(True)

    images = []
    depths = []
    masks = []
    poses = []

    for img_file in img_path.iterdir():
        img_id = img_file.name.split('.')[0]
        print('image_id', img_id)

        with open(str(img_file), "rb") as f:
            img = Image.open(f).convert("RGB")

        mask = cv2.imread(str(img_file), -1)[:, :, -1] > 1e-7
        masks.append(torch.from_numpy(mask.astype(float)))

        depth_file = depth_path / (img_id + '.png')
        print(depth_file)
        depth = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH).astype(float)
        depth *= (scale / 1e3)
        depth = torch.from_numpy(depth).float()
        # depth *= masks[-1]
        # depth *= (depth < 2.0).float()
        # depth = F.interpolate(
        #     depth.unsqueeze(0).unsqueeze(0), size=resized_img_size, mode='nearest'
        # ).squeeze(0).squeeze(0)
        depths.append(depth)

        print(torch.max(depth))

        nonzero_depth = depth[depth > 1e-7]
        print(
            'depth', depth.shape,
            'min', torch.min(nonzero_depth),
            'max', torch.max(nonzero_depth),
        )

        # depths.append(torch.zeros((224, 224), dtype=torch.float32))

        img = img_transform(img)
        original_img_size = img.shape[-2:]

        # img = F.interpolate(
        #     img.unsqueeze(0), size=resized_img_size,
        #     mode='bilinear', align_corners=False
        # ).squeeze(0)

        images.append(img)

        pose_file = pose_path / (img_id + '.txt')
        pose = np.loadtxt(pose_file).astype(float)
        pose = pose_to_opengl(pose)
        pose[:3, -1] *= scale
        pose = np.linalg.inv(pose)
        poses.append(torch.from_numpy(pose))

    images = torch.stack(images)
    depths = torch.stack(depths)
    masks = torch.stack(masks).float()
    poses = torch.stack(poses).float()

    print('img min', torch.min(images))
    print('img max', torch.max(images))
    print('img mean', torch.mean(images))
    print('img size', images.shape)

    batch = {
        'imgs': images,
        'depths': depths,
        'intrinsics': intrinsics,
        'extrinsics': poses,
        'masks': masks,
        'points': torch.tensor([1]),
        'normals': torch.tensor([1]),
        'voxels': None,
        'id_strs': ['scene0706-00']
    }

    # add batch size
    batch = {
        key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }

    return batch


def scannet_launch():

    args = default_argument_parser()

    # Note we need this only for pretrained models with torchvision.
    os.environ["TORCH_HOME"] = args.torch_home

    if args.copy_data:
        # if copy data is 1 then you need to provide args.data_dir
        # from which to copy data
        if args.data_dir == "":
            raise ValueError("You need to provide args.data_dir")
        copy_data(args)

    main_worker_eval(0, args)


if __name__ == "__main__":
    scannet_launch()
