import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from collections import namedtuple

from shapenet.modeling.heads.depth_loss import interpolate_multi_view_tensor

class PerceptualLoss(nn.Module):
    """based on https://github.com/oneTaken/pytorch_fast_style_transfer"""
    def __init__(self, requires_grad=False, pretrained=True):
        super(PerceptualLoss, self).__init__()
        self.vgg16 = torchvision.models.vgg16(pretrained=True)
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        vgg_pretrained_features = self.vgg16.features
        self.img_size = (224, 224)

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = requires_grad

    def features(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

    @staticmethod
    def gram_matrix(y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram

    @staticmethod
    def multiview_depth_to_rgb(img, img_size):
        """
        Simulate RGB image from depths by repeating 3 channels
        :param img: (B, V, H, W)
        :return (B*V, 3, H, W)
        """
        img = interpolate_multi_view_tensor(img, img_size)
        B, V, H, W = img.shape
        img = img.unsqueeze(2).expand(-1, -1, 3, -1, -1)
        img = img.view(B*V, 3, H, W)
        return img

    def forward(self, predicted_img, content_img, style_img=None):
        predicted_img = self.multiview_depth_to_rgb(predicted_img, self.img_size)
        content_img = self.multiview_depth_to_rgb(content_img, self.img_size)

        features_pred = self.features(predicted_img)
        features_content = self.features(content_img)

        loss_content = self.mse_loss(features_pred[1], features_content[1])

        loss_style = 0.0
        if style_img is not None:
            style_img = self.multiview_depth_to_rgb(style_img, (224, 224))
            features_style = self.features(style_img)
            for ft_y, ft_s in zip(features_pred, features_style):
                gm_y = self.gram_matrix(ft_y)
                gm_s = self.gram_matrix(ft_s)
                loss_style = loss_style + self.mse_loss(gm_y, gm_s)

        return {
            'loss_content': loss_content,
            'loss_style': loss_style
        }

def get_perceptual_losses(
        cfg, perceptual_loss_model,
        rendered_depth, predicted_depth, gt_depth
):
    # perceptual loss
    content_img = None
    style_img = None

    if cfg.CONTENT_IMAGE == "gt_depth":
        content_img = gt_depth
    elif cfg.CONTENT_IMAGE == "predicted_depth":
        content_img = predicted_depth

    if cfg.STYLE_IMAGE == "gt_depth":
        style_img = gt_depth
    elif cfg.STYLE_IMAGE == "predicted_depth":
        style_img = predicted_depth

    if content_img is not None:
        perceptual_losses = perceptual_loss_model(
            predicted_img=rendered_depth,
            content_img=content_img,
            style_img=style_img
        )
        return perceptual_losses


