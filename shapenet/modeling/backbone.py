# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1, conv3x3


class ResNetBackbone(nn.Module):
    def __init__(self, net):
        super(ResNetBackbone, self).__init__()
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.stage1 = net.layer1
        self.stage2 = net.layer2
        self.stage3 = net.layer3
        self.stage4 = net.layer4

    def forward(self, imgs):
        feats = self.stem(imgs)
        conv1 = self.stage1(feats)  # 18, 34: 64
        conv2 = self.stage2(conv1)
        conv3 = self.stage3(conv2)
        conv4 = self.stage4(conv3)

        return [conv1, conv2, conv3, conv4]


class ResNetVarChannel(nn.Module):
    """
    Copy of torchvision.model.ResNet with parameterized input channels
    """
    def __init__(self, block, layers, input_channels=3, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetVarChannel, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def resnet18(**kwargs):
    return ResNetVarChannel(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNetVarChannel(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNetVarChannel(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNetVarChannel(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    return ResNetVarChannel(Bottleneck, [3, 8, 36, 3], **kwargs)


_FEAT_DIMS = {
    "resnet18": (64, 128, 256, 512),
    "resnet34": (64, 128, 256, 512),
    "resnet50": (256, 512, 1024, 2048),
    "resnet101": (256, 512, 1024, 2048),
    "resnet152": (256, 512, 1024, 2048),
}


_CUSTOM_RESNET_FUNCS = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
}


class VGG16P2M(nn.Module):
    def __init__(self, n_classes_input=3):
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

        self.conv3_1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # 56 -> 28
        self.conv3_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(128, 256, 5, stride=2, padding=2)  # 28 -> 14
        self.conv4_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(256, 512, 5, stride=2, padding=2)  # 14 -> 7
        self.conv5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, 3, stride=1, padding=1)

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
        img = F.relu_(self.conv0_1(img))
        img = F.relu_(self.conv0_2(img))
        # img0 = torch.squeeze(img) # 224

        img = F.relu_(self.conv1_1(img))
        img = F.relu_(self.conv1_2(img))
        img = F.relu_(self.conv1_3(img))
        # img1 = torch.squeeze(img) # 112

        img = F.relu_(self.conv2_1(img))
        img = F.relu_(self.conv2_2(img))
        img = F.relu_(self.conv2_3(img))
        img2 = img

        img = F.relu_(self.conv3_1(img))
        img = F.relu_(self.conv3_2(img))
        img = F.relu_(self.conv3_3(img))
        img3 = img

        img = F.relu_(self.conv4_1(img))
        img = F.relu_(self.conv4_2(img))
        img = F.relu_(self.conv4_3(img))
        img4 = img

        img = F.relu_(self.conv5_1(img))
        img = F.relu_(self.conv5_2(img))
        img = F.relu_(self.conv5_3(img))
        img = F.relu(self.conv5_4(img))
        img5 = img

        return [img2, img3, img4, img5]


def build_backbone(name, pretrained=True):
    resnets = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    if name in resnets and name in _FEAT_DIMS:
        cnn = getattr(torchvision.models, name)(pretrained=pretrained)
        backbone = ResNetBackbone(cnn)
        feat_dims = _FEAT_DIMS[name]
        return backbone, feat_dims
    else:
        raise ValueError('Unrecognized backbone type "%s"' % name)


def build_custom_backbone(name, input_channels):
    if name in _CUSTOM_RESNET_FUNCS:
        cnn = _CUSTOM_RESNET_FUNCS[name](input_channels=input_channels)
        backbone = ResNetBackbone(cnn)
        feat_dims = _FEAT_DIMS[name]
        return backbone, feat_dims
    if name == "vgg":
        backbone = VGG16P2M(n_classes_input=input_channels)
        feat_dims = [64, 128, 256, 512]
        return backbone, feat_dims
    else:
        raise ValueError('Unrecognized backbone type "%s"' % name)
