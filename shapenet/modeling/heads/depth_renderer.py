import torch
import torch.nn as nn
import torch.nn.functional as F

import neural_renderer as nr

import numpy as np
import cv2

class DepthRenderer(nn.Module):
    def __init__(self, cfg):
        super(DepthRenderer, self).__init__()
        min_depth = cfg.MODEL.MVSNET.MIN_DEPTH
        max_depth = min_depth + (cfg.MODEL.MVSNET.DEPTH_INTERVAL \
                                    * cfg.MODEL.MVSNET.NUM_DEPTHS)
        self.renderer = nr.Renderer(
            camera_mode='projection',
            near=min_depth, far=max_depth,
            anti_aliasing=False
        )
        fx, fy = cfg.MODEL.MVSNET.FOCAL_LENGTH
        cx, cy = cfg.MODEL.MVSNET.PRINCIPAL_POINT
        self.camera_k = torch.tensor(
            [[fx, 0, cx],
             [0, fy, cy],
             [0, 0, 1]],
            dtype=torch.float32
        )
        self.dist_coeffs = torch.zeros(5, dtype=torch.float32)

        # conversion from shapenet convention (East-Up_South)
        # to renderer convention (East-Down-North)
        # final rotation: R_renderer_shapenet * extrinsics
        # inverse y and z, equivalent to inverse x, but gives positive z
        rvec = np.array([np.pi, 0., 0.], dtype=np.float32)
        R = cv2.Rodrigues(rvec)[0]
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        self.T_renderer_shapenet = torch.from_numpy(T)
        self.T_shapenet_renderer = torch.inverse(self.T_renderer_shapenet)

    def transform_to_renderer_frame(self, T_view_world):
        """
        Args:
        - T_view_world: (batch x 4 x 4) transformation
                        in shapenet coordinates (East-Up-South)
        Returns:
        - (batch x 4 x 4) transformation in renderer frame (East-Down-North)
        """
        batch_size = T_view_world.size(0)
        device = T_view_world.device

        self.T_renderer_shapenet = self.T_renderer_shapenet.to(device)
        self.T_shapenet_renderer = self.T_shapenet_renderer.to(device)

        # change to correct shape (batched)
        T_renderer_shapenet = self.T_renderer_shapenet \
                                  .unsqueeze(0) .expand(batch_size, -1, -1)
        T_shapenet_renderer = self.T_shapenet_renderer \
                                  .unsqueeze(0).expand(batch_size, -1, -1)

        # inverse y and z, equivalent to inverse x, but gives positive z
        T_view_world = torch.bmm(T_renderer_shapenet, T_view_world)
        return T_view_world

    def forward(self, coords, faces, extrinsics, image_shape):
        """
        Multi-view rendering
        Args:
        - pred_coords: (batch x vertices x 3) tensor
        - faces: (batch x faces x 3) tensor
        - image_shape: shape of the depth image to be rendered
        - extrinsics: (batch x view x 2 x 4 x 4) tensor
        Returns:
        - depth tensor batch x view x height x width
        """
        batch_size = extrinsics.size(0)
        num_views = extrinsics.size(1)
        # augment views: size = (batch x view x vertices x 3)
        coords_augmented = coords.unsqueeze(1).expand(-1, num_views, -1, -1) \
                                    .contiguous()
        # size = (batch x view x faces` x 3)
        faces_augmented = faces.unsqueeze(1).expand(-1, num_views, -1, -1) \
                                    .contiguous()

        depth_flattened = self.render_depth(
            _flatten_batch_view(coords_augmented),
            _flatten_batch_view(faces_augmented),
            _flatten_batch_view(extrinsics),
            image_shape
        )
        return _unflatten_batch_view(depth_flattened, batch_size)

    def render_depth(self, coords, faces, T_view_world, image_shape):
        """
        renders a batch of depths
        Args:
        - pred_coords: (batch x vertices x 3) tensor
        - faces: (batch x faces x 3) tensor
        - image_shape shape of the depth image to be rendered
        - T_view_world: (batch x 4 x 4) transformation
                        in shapenet coordinates (EUS)
        Returns:
        - depth tensors of shape (batch x h x w)
        """
        image_size = image_shape.max()
        # This is not thread safe!
        self.renderer.image_size = image_size
        batch_size, num_points = coords.size()[:2]

        # move to correct device
        device = coords.device
        self.camera_k = self.camera_k.to(device)
        self.dist_coeffs = self.dist_coeffs.to(device)
        faces = faces.type(torch.int32).to(device)

        # change to correct shape (batches)
        dist_coeffs = self.dist_coeffs.unsqueeze(0).expand(batch_size, -1)

        # transformation stuffs
        T_view_world = self.transform_to_renderer_frame(T_view_world)
        R = T_view_world[:, :3, :3]
        t = T_view_world[:, :3, 3].unsqueeze(1)
        depth = self.renderer(
            vertices=coords, faces=faces, mode='depth',
            K=self.camera_k.unsqueeze(0), dist_coeffs=dist_coeffs,
            R=R, t=t, orig_size=image_size
        )
        depth[depth <= self.renderer.near] = 0
        depth[depth >= self.renderer.far] = 0
        return depth


## Private utility functions
def _flatten_batch_view(tensor):
    return tensor.view(-1, *(tensor.size()[2:]))


def _unflatten_batch_view(tensor, batch_size):
    return tensor.view(batch_size, -1, *(tensor.size()[1:]))

