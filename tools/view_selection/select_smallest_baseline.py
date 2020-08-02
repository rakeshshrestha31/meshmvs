#!/usr/bin/env python

# standard library imports
import os
import sys
import cv2
import numpy as np
from skimage import io, transform
import itertools
import json
import tqdm

# custom imports
from shapenet.data import register_shapenet
from shapenet.data.mesh_vox import MeshVoxDataset

# torch imports
import torch
import torchvision

MIN_BASELINE = 0.2

def main(cfg, debug=False):
    # dictionary from scene to best subset
    best_subsets = {}
    data_ids = load_data_ids(cfg)
    for sid, mid in tqdm.tqdm(data_ids):
        metadata = MeshVoxDataset.read_camera_parameters(cfg['data_dir'], sid, mid)
        T_world_cams = torch.inverse(metadata['extrinsics'])
        best_subset = find_best_subset(T_world_cams)
        best_subsets['%s_%s'%(sid, mid)] = best_subset

        if debug and len(best_subsets) > 5:
            output_debug_images(best_subsets, cfg, metadata)
            break

    with open('/tmp/best_subsets.json', 'w') as f:
        json.dump(best_subsets, f, indent=4)


def output_debug_images(best_subsets, cfg, metadata):
    def read_image(sid, mid, iid):
        nonlocal cfg, metadata
        return MeshVoxDataset.read_image(
            cfg['data_dir'], sid, mid,
            metadata['image_list'][iid]
        )
    image_transform = MeshVoxDataset.get_transform(False)
    best_images = []
    original_images = []
    for sid_mid in best_subsets:
        sid, mid = sid_mid.split('_')
        best_images.append(torch.stack([
            image_transform(read_image(sid, mid, iid))
            for iid in best_subsets[sid_mid]
        ], 0))
        original_images.append(torch.stack([
            image_transform(read_image(sid, mid, iid))
            for iid in [0, 6, 7]
        ], 0))
    save_image_grids('/tmp/best_images.png', best_images, 3)
    save_image_grids('/tmp/original_images.png', original_images, 3)


def save_image_grids(filename, images, nrow):
    images_tensor = torch.stack(images, 0)
    images_tensor = images_tensor.view(-1, *images_tensor.shape[-3:])
    grid_images_tensor \
            = torchvision.utils.make_grid(images_tensor, nrow=nrow)
    grid_images_np = (grid_images_tensor.permute(1, 2, 0) * 255).numpy() \
                            .astype(np.uint8)
    cv2.imwrite(filename, grid_images_np)


def load_data_ids(cfg):
    model_ids = []
    summary_json = os.path.join(cfg['data_dir'], 'summary.json')
    with open(summary_json, 'r') as f:
        summary = json.load(f)
    with open(cfg['splits_file'], 'r') as f:
        splits = json.load(f)
    for sid, split_name in itertools.product(summary, ['train', 'test', 'val']):
        print("Starting synset (%s, %s)" % (sid, split_name))
        split = splits[split_name]
        allowed_mids = None
        if sid not in split:
            continue
        if isinstance(split[sid], list):
            allowed_mids = set(split[sid])
        elif isinstance(split[sid], dict):
            allowed_mids = set(split[sid].keys())
        else:
            sys.exit('Allowed mids not found')

        model_ids.extend([
            (sid, mid) for mid, _ in summary[sid].items()
            if mid in allowed_mids
        ])
    return model_ids


def find_best_subset(T_world_cams):
    global MIN_BASELINE
    # batched processing
    # n x 3. 3 = index of subsets
    subsets = list(itertools.combinations(range(len(T_world_cams)), 3))

    # n x 2. 2 = pairs
    pairs = list(itertools.combinations(range(len(T_world_cams)), 2))
    pairs_indices = {pair: i for i, pair in enumerate(pairs)}
    pairs = torch.tensor(pairs, dtype=torch.long)

    # m x 3. m = num pairs, 3 = position vectors of each cam in a pair
    positions0 = T_world_cams[pairs[:, 0], :3, 3]
    positions1 = T_world_cams[pairs[:, 1], :3, 3]
    # m x 1. m = num pairs, 1 = baseline of each pair
    pair_baselines = torch.norm(positions0 - positions1, dim=1)
    pair_baselines = pair_baselines + ((pair_baselines < MIN_BASELINE) * 1e16)

    # n X 3 x 2. 3 = subset 2 = pairs within subsets
    subsets_with_pairs = torch.tensor([
        list(itertools.combinations(subset, 2))
        for subset in subsets
    ], dtype=torch.long)

    # n x 3. contains index of a pair
    subsets_with_pairs_indices = [
        pairs_indices[tuple(i.tolist())]
        for i in subsets_with_pairs.view(-1, 2).unbind(0)
    ]
    subsets_with_pairs_indices \
            = torch.tensor(subsets_with_pairs_indices, dtype=torch.long).view(-1, 3)

    # n x 3 X 1. 1 = baseline
    subsets_baselines = sum([
        pair_baselines[subsets_with_pairs_indices[:, i]] for i in range(3)
    ])

    min_idx = torch.argmin(subsets_baselines)
    return subsets[min_idx]

    # unbatched processing
    # subsets = list(itertools.combinations(range(len(T_world_cams)), 3))
    # subset_baselines = []
    # pose0s = []
    # pose1s = []
    # for subset_indices in subsets:
    #     total_baseline = 0
    #     for subset_pair_indices in \
    #             itertools.combinations(subset_indices, 2):
    #         pose0 = T_world_cams[subset_pair_indices[0]]
    #         pose1 = T_world_cams[subset_pair_indices[1]]

    #         baseline_vector = pose1[:3, 3] - pose0[:3, 3]
    #         baseline = torch.norm(baseline_vector)
    #         baseline = float('inf') if baseline < MIN_BASELINE \
    #                                 else baseline.item()
    #         total_baseline += baseline
    #     subset_baselines.append(total_baseline)


    # min_subset_idx = np.argmin(subset_baselines)
    # return subsets[min_subset_idx]


def get_cfg():
    data_dir, splits_file = register_shapenet('shapenet')
    return {'data_dir': data_dir, 'splits_file': splits_file}

if __name__ == '__main__':
    cfg = get_cfg()
    main(cfg, debug=False)

