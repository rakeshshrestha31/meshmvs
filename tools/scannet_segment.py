from pathlib import Path
import numpy as np
import cv2

import matplotlib.pyplot as plt
import open3d as o3d


def pose_to_opengl(cam_pose):
    # use OpenGL convention (East-Up-South) instead of East-Down-North
    rotation_correction = np.asarray([
        [1, 0, 0], [0, -1, 0], [0, 0, -1]
    ])
    cam_pose[:3, :3] = np.dot(cam_pose[:3, :3], rotation_correction)
    return cam_pose


if __name__ == '__main__':
    data_root = Path('/datasets/scannet/scans')
    scene = 'scene0001_00'
    color_path = data_root / scene / 'color'
    depth_path = data_root / scene / 'depth'
    label_path = data_root / scene / 'label-filt'
    instance_path = data_root / scene / 'instance-filt'

    segmented_color_path = data_root / scene / 'segmented-color'
    segmented_color_path.mkdir(exist_ok=True)

    segmented_depth_path = data_root / scene / 'segmented-depth'
    segmented_depth_path.mkdir(exist_ok=True)

    intrinsics_color_path = data_root / scene / 'intrinsics' / 'intrinsic_color.txt'
    intrinsics_color = np.loadtxt(str(intrinsics_color_path))[:3, :3]

    intrinsics_depth_path = data_root / scene / 'intrinsics' / 'intrinsic_depth.txt'
    intrinsics_depth = np.loadtxt(str(intrinsics_depth_path))[:3, :3]

    new_intrinsics = np.asarray([
        [248, 0, 111.5],
        [0, 248, 111.5],
        [0, 0, 1]
    ])
    new_size = (224, 224)

    mesh_path = data_root / scene / '{}_vh_clean_2.labels.ply'.format(scene)
    scene_mesh = o3d.io.read_triangle_mesh(
        str(mesh_path)
    )

    geometry_list = [scene_mesh]

    # for img_file in color_path.iterdir():
    #     img_id = img_file.name.split('.')[0]
    for img_id in [362, 369, 376]:
        img_id = str(img_id)

        pose_path = data_root / scene / 'poses' / (img_id + '.txt')
        pose = np.loadtxt(pose_path)
        pose = pose_to_opengl(pose)
        geometry_list.append(
            o3d.geometry.TriangleMesh.create_coordinate_frame(0.5) \
                        .transform(pose)
        )

        img_file = color_path / (img_id + '.jpg')
        depth_file = depth_path / (img_id + '.png')
        label_file = label_path / (img_id + '.png')
        instance_file = instance_path / (img_id + '.png')

        img = cv2.imread(str(img_file))

        labels = cv2.imread(str(label_file), cv2.IMREAD_ANYDEPTH)
        unique_labels = np.unique(labels)
        # print('unique_labels', unique_labels)

        instances = cv2.imread(str(instance_file), cv2.IMREAD_ANYDEPTH)
        unique_instances = np.unique(instances)
        # print('unique_instances', unique_instances)

        # selected_instances = instances * (labels == 2).astype(np.uint16)
        # if np.unique(selected_instances).shape[0] > 1:
        #     print(np.unique(selected_instances))
        # continue

        # segmentation
        foreground = ((labels == 4) & (instances == 6)).astype(np.uint8)
        # foreground = ((labels == 4)).astype(np.uint8)
        unique_foreground = np.unique(instances * foreground)
        if unique_foreground.shape[0] < 2:
            continue
        print('instances', unique_foreground)

        foreground = np.repeat(foreground[:, :, np.newaxis], 3, axis=-1)
        background = 1 - foreground

        segmented_img = (img * foreground) + (background * 255)
        segmented_img_with_alpha = np.zeros(
            (*(segmented_img.shape[:2]), 4), dtype=segmented_img.dtype
        )
        segmented_img_with_alpha[:, :, :3] = segmented_img
        segmented_img_with_alpha[:, :, 3] = (foreground[:, :, 0] * 255)

        # make shapenet compatible
        map1, map2 = cv2.initUndistortRectifyMap(
            intrinsics_color, newCameraMatrix=new_intrinsics,
            distCoeffs=None, size=new_size, R=None, m1type=cv2.CV_32FC1
        )
        new_img = cv2.remap(segmented_img_with_alpha, map1, map2, cv2.INTER_LINEAR)
        cv2.imwrite(str(segmented_color_path / (img_id + '.png')), new_img)
        # cv2.imwrite(str(segmented_color_path / (img_id + '_{}.png'.format(instance))), new_img)
        # cv2.imwrite(str(segmented_color_path / (img_id + '.png')), img)

        # depth
        print(depth_file)
        depth = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH)
        # make shapenet compatible
        map1, map2 = cv2.initUndistortRectifyMap(
            intrinsics_depth, newCameraMatrix=new_intrinsics,
            distCoeffs=None, size=new_size, R=None, m1type=cv2.CV_32FC1
        )
        new_depth = cv2.remap(depth, map1, map2, cv2.INTER_NEAREST)
        print(np.max(depth), depth.shape, depth.dtype)
        print(np.max(new_depth), new_depth.shape, new_depth.dtype)
        cv2.imwrite(str(segmented_depth_path / (img_id + '.png')), new_depth)

        # plt.imshow(new_img)
        # plt.imshow(segmented_img_with_alpha[...,::-1])
        # plt.imshow(segmented_img[...,::-1])
        # plt.imshow(img[...,::-1])
        # plt.imshow(labels, cmap='gray', alpha=0.6)
        # plt.imshow(instances, cmap=cmap, alpha=0.6)
        # plt.show()

    exit(0)
    o3d.visualization.draw_geometries(geometry_list)

