import torch
import torch.nn as nn
from torch.nn import MaxPool1d
import torch.nn.functional as F

import numpy as np

# transformation to align with Pixel2Mesh (shapenet) coordinate frame
T_shapenet_dtu = np.asarray([
    [1.0,  0.0,  0.0, 0.0],
    [0.0, -1.0,  0.0, 0.0],
    [0.0,  0.0, -1.0, 0.0],
    [0.0,  0.0,  0.0, 1.0]
])


def project_pixel_coords(x, y, depth_values, src_proj, ref_proj, batch):
    """project pixel coords (x, y) to another image frame for each possible depth_values"""
    num_depth = depth_values.shape[1]
    transform = torch.matmul(src_proj[:, 0], torch.inverse(ref_proj[:, 0]))

    device = x.get_device()
    # transform the extrinsics from shapenet frame to DTU
    transform_shapenet_dtu = torch.tensor(T_shapenet_dtu, dtype=torch.float32).unsqueeze(0)
    if device >= 0:
        transform_shapenet_dtu = transform_shapenet_dtu.cuda(device)

    src_transform = torch.matmul(
        torch.inverse(transform_shapenet_dtu),
        torch.matmul(src_proj[:, 0], transform_shapenet_dtu)
    )
    ref_transform = torch.matmul(
        torch.inverse(transform_shapenet_dtu),
        torch.matmul(ref_proj[:, 0], transform_shapenet_dtu)
    )

    src_proj_new = src_proj[:, 0].clone()
    src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_transform[:, :3, :4])
    ref_proj_new = ref_proj[:, 0].clone()
    ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_transform[:, :3, :4])
    proj = torch.matmul(src_proj_new, torch.inverse(ref_proj_new))
    rot = proj[:, :3, :3]  # [B,3,3]
    trans = proj[:, :3, 3:4]  # [B,3,1]

    xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
    xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
    rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
    rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                        -1)  # [B, 3, Ndepth, H*W]
    proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
    proj_xy = proj_xyz[:, :2, :, :] / (proj_xyz[:, 2:3, :, :] + 0.0001)  # [B, 2, Ndepth, H*W]

    return proj_xy


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        # proj = torch.matmul(src_proj, torch.inverse(ref_proj))

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        proj_xy = project_pixel_coords(x, y, depth_values, src_proj, ref_proj, batch)

        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea


class MVSNet(nn.Module):
    def __init__(self, cfg):
        super(MVSNet, self).__init__()

        self.freeze_cv = cfg.FREEZE
        self.input_image_size = cfg.INPUT_IMAGE_SIZE
        self.depth_values = cfg.MIN_DEPTH \
                          + (torch.arange(cfg.NUM_DEPTHS, dtype=torch.float32)
                                * cfg.DEPTH_INTERVAL)
        self.intrinsics = torch.tensor([
            [cfg.FOCAL_LENGTH[0], 0, cfg.PRINCIPAL_POINT[0], 0],
            [0, cfg.FOCAL_LENGTH[1], cfg.PRINCIPAL_POINT[1], 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0]
        ], dtype=torch.float32)

        self.feature = VGG16P2M()
        self.cost_regularization = CostRegNet(cfg.FEATURES_LIST)

        # self.features_dim = 960# + 191
        # self.features_dim = 384
        # self.features_dim = 960 # 120
        self.features_dim = np.sum(self.cost_regularization.features_list)

        if cfg.CHECKPOINT:
            print("==> Loading MVSNet checkpoint:", cfg.CHECKPOINT)
            state_dict = torch.load(cfg.CHECKPOINT)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            self.load_state_dict(state_dict)

        for param in self.parameters():
            param.requires_grad = not self.freeze_cv
        print("==> cost volume weight require_grad is:", not self.freeze_cv)
        print("==> number of cost volume features:",
              self.cost_regularization.features_list, self.features_dim)

    ## Newer batched method of getting features
    #  @detail Even with all the mucking around with tensors and lists shapes,
    #           it's still 2.5x faster than old get_features_unbatched
    #  @param imgs tensor of size batch x views x channel x height x width
    #  @return 2D list of size num_views x num_features.
    #           each item is batch x channels x h x w tensor
    def get_features_batched(self, imgs):
        debug = False
        batches, views = imgs.size()[:2]
        flattened_imgs = imgs.view(batches * views, *(imgs.size()[2:]))
        # list of (batches*views) x channels x h x w tensors
        flattened_features = self.feature(flattened_imgs)
        # list of views x batches x channels x h x w tensors
        features = [i.view(batches, views, *(i.size()[1:])).transpose(1, 0)
                    for i in flattened_features]
        # list of num_features x num_views tensors.
        # Each item is batch x channels x h x w tensors
        features = [feature.unbind(0) for feature in features]
        # transpose: dim = num_views x num_features
        features = [
            [ features[i][j] for i in range(len(features)) ]
            for j in range(len(features[0]))
        ]

        if debug:
            features_unbatched = self.get_features_unbatched(imgs)
            # this method and get_features_unbatched get same results
            # but this method is 2.5x faster
            print('eq:',
                [[torch.all(torch.abs(k -l) < 1e-4).item() for k, l in zip(i, j)]
                 for i, j in zip(features, features_unbatched)]
            )

        return features

    ## Older unbatched method of getting features
    #  @param imgs tensor of size batch x views x channel x height x width
    #  @return 2D list of size batch x num_features
    def get_features_unbatched(self, imgs):
        imgs_list = imgs.unbind(1)
        features = [self.feature(img) for img in imgs_list]
        return features

    # scale intrinsics based on the feature size
    @staticmethod
    def rescale_proj_matrices(proj_matrices, img_size, feature_size):
        feature_resize_factor = (
            img_size[0] / feature_size[0],
            img_size[1] / feature_size[1]
        )
        # avoid mutating the original proj_matrices argument
        new_proj_matrices = proj_matrices.clone()
        new_proj_matrices[:, :, 1, 0, :3] /= feature_resize_factor[0]
        new_proj_matrices[:, :, 1, 1, :3] /= feature_resize_factor[1]
        return new_proj_matrices

    def forward(self, imgs, extrinsics):
        batch_size = imgs.size(0)
        num_views = imgs.size(1)
        depth_values = self.depth_values.to(imgs).unsqueeze(0) \
                                        .expand(batch_size, -1)
        self.intrinsics = self.intrinsics.to(imgs)
        # flatten batch and view before image interpolation
        imgs = F.interpolate(
            imgs.view(-1, *(imgs.shape[2:])), size=self.input_image_size,
            mode='bilinear', align_corners=False
        ).view(batch_size, num_views, imgs.shape[2], *(self.input_image_size))

        proj_matrices = torch.stack((
            extrinsics,
            self.intrinsics.unsqueeze(0).unsqueeze(1) \
                           .expand(batch_size, num_views, -1, -1)
        ), dim=2)

        # all the views should be treated as ref_features iteratively
        view_lists = [np.roll(range(num_views), -i) for i in range(num_views)]
        # for 3 views you'd expect this view_lists
        # view_lists = ((0, 1, 2), (1, 2, 0), (2, 0, 1))

        # imgs = torch.unbind(imgs, 1)
        # proj_matrices = torch.unbind(proj_matrices, 1)
        # assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"
        img_height, img_width = imgs.shape[-2], imgs[0].shape[-1]

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        features = self.get_features_batched(imgs)

        # scale intrinsics based on the feature size
        proj_matrices = self.rescale_proj_matrices(
            proj_matrices, imgs.size()[-2:], features[0][0].size()[-2:]
        )

        costvolume_outputs = {'features': [], 'depths': []}

        for view_list in view_lists:
            reordered_features = [features[i] for i in view_list]
            reordered_proj_matrices = [proj_matrices[:, i] for i in view_list]
            costvolume_output = self.compute_costvolume_depth(
                reordered_features, reordered_proj_matrices, depth_values
            )
            # costvolume_outputs['features'].append(costvolume_output['features'])
            costvolume_outputs['depths'].append(costvolume_output['depth'])

        costvolume_outputs['depths'] = torch.stack(costvolume_outputs['depths'],
                                                   dim=1)
        return costvolume_outputs

    def compute_costvolume_depth(self, features, proj_matrices, depth_values):
        num_views = len(features)
        num_depth = depth_values.shape[1]

        ref_feature = features[0][0]
        src_features = []
        for i in range(len(features[1:3])):
            src_features.append(features[i][0])

        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume.clone() ** 2
        # volume_sq_sum = ref_volume.pow_(2)
        del ref_volume
        for src_fea, src_proj in zip(src_features, src_projs):
            # warpped features
            warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_values)
            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
            del warped_volume
        # aggregate multiple feature volumes by variance
        if num_views > 1:
            volume_variance = \
                    volume_sq_sum.div_(num_views) \
                                 .sub_(volume_sum.div_(num_views).pow_(2))
        else:
            volume_variance = volume_sq_sum

        # step 3. cost volume regularization
        cost_agg = self.cost_regularization(volume_variance)
        cost_reg = cost_agg["x"]
        cost_agg_feature = cost_agg["x_agg"]

        cost_reg = cost_reg.squeeze(1)
        prob_volume = F.softmax(cost_reg, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_values)
        # add cost aggregated feature
        return {"features": cost_agg_feature, "depth": depth}


class VGG16P2M(nn.Module):
    def __init__(self, n_classes_input=3, pretrained=False):
        super(VGG16P2M, self).__init__()

        self.features_dim = 960

        self.conv0_1 = nn.Conv2d(n_classes_input, 16, 3, stride=1, padding=1)
        self.conv0_2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.conv1_1 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # 224 -> 112
        self.conv1_2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv1_3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 112 -> 56
        self.conv2_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2_3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self._initialize_weights()  # not load the pre-trained model

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, img):
        img = F.relu(self.conv0_1(img))
        img = F.relu(self.conv0_2(img))
        # img0 = torch.squeeze(img) # 224

        img = F.relu(self.conv1_1(img))
        img = F.relu(self.conv1_2(img))
        img = F.relu(self.conv1_3(img))
        # img1 = torch.squeeze(img) # 112

        img = F.relu(self.conv2_1(img))
        img = F.relu(self.conv2_2(img))
        img = F.relu(self.conv2_3(img))
        img2 = img

        return [img2]


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class CostRegNet(nn.Module):
    def __init__(self, features_list):
        super(CostRegNet, self).__init__()
        # self.features_list = [32, 64, 128, 256]
        # self.features_list = [64, 128, 256, 512]
        # self.features_list = [100, 200, 400, 800]
        self.features_list = features_list
        self.conv0 = ConvBnReLU3D(64, self.features_list[0])

        self.conv1 = ConvBnReLU3D(self.features_list[0], self.features_list[1], stride=2)
        self.conv2 = ConvBnReLU3D(self.features_list[1], self.features_list[1])

        self.conv3 = ConvBnReLU3D(self.features_list[1], self.features_list[2], stride=2)
        self.conv4 = ConvBnReLU3D(self.features_list[2], self.features_list[2])

        self.conv5 = ConvBnReLU3D(self.features_list[2], self.features_list[3], stride=2)
        self.conv6 = ConvBnReLU3D(self.features_list[3], self.features_list[3])

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(self.features_list[3], self.features_list[2], kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(self.features_list[2]),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(self.features_list[2], self.features_list[1], kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(self.features_list[1]),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(self.features_list[1], self.features_list[0], kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(self.features_list[0]),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(self.features_list[0], 1, 3, stride=1, padding=1)

    def forward(self, x):
        x_agg = []
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x_agg.append(x)

        x = conv4 + self.conv7(x)
        x_agg.append(x)

        x = conv2 + self.conv9(x)
        x_agg.append(x)

        x = conv0 + self.conv11(x)
        x_agg.append(x)

        x = self.prob(x)
        return {"x":x, "x_agg":x_agg}


# p: probability volume [B, D, H, W]
# depth_values: discrete depth values [B, D]
def depth_regression(p, depth_values):
    depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)
    return depth
