from collections import OrderedDict
import itertools
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F


class VoxelRefineHead(nn.Module):
    def __init__(self, cfg):
        super(VoxelRefineHead, self).__init__()

        self.num_layers = cfg.MODEL.VOXEL_REFINE_HEAD.NUM_LAYERS

        self.encoder_layers = OrderedDict()
        self.decoder_layers = OrderedDict()

        # dim progression: 1->8->16->32->...
        prev_dim = 1
        curr_dim = 8

        for layer_idx in range(self.num_layers):
            layer_name = "ConvBlock{}".format(layer_idx)
            conv_block = ResBlock(
                prev_dim, curr_dim
            )
            self.encoder_layers[layer_name] = conv_block
            prev_dim = curr_dim
            curr_dim *= 2
            self.add_module(layer_name, conv_block)

        for layer_idx in range(self.num_layers-1, -1, -1):
            layer_name = "DeconvBlock{}".format(layer_idx)
            encoder_layer = list(self.encoder_layers.values())[layer_idx]

            # deconv channels opposite of conv
            in_channels = encoder_layer.conv1.out_channels
            out_channels = encoder_layer.conv1.in_channels

            # no concatenation for the last (deepest) layer
            if layer_idx != self.num_layers-1:
                in_channels *= 2

            deconv_block = nn.Sequential(
                nn.ConvTranspose3d(
                    in_channels, out_channels,
                    kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                # nn.BatchNorm3d(out_channels),
                nn.ReLU(True)
            )
            self.decoder_layers[layer_name] = deconv_block
            self.add_module(layer_name, deconv_block)

        self.final_deconv = nn.ConvTranspose3d(
            2, 1, kernel_size=1, stride=1
        )

        def init_weights(m):
            if type(m) in [nn.Conv3d, nn.ConvTranspose3d]:
                weight_init.c2_msra_fill(m)

        enc_dec_layers = itertools.chain(self.encoder_layers.values(),  self.decoder_layers.values())
        for block in enc_dec_layers:
            block.apply(init_weights)

        # use normal distribution initialization for voxel prediction layer
        nn.init.normal_(self.final_deconv.weight, std=0.001)
        if self.final_deconv.bias is not None:
            nn.init.constant_(self.final_deconv.bias, 0)

    def forward(self, x):
        # the input is also used for concatenation
        enc_outputs = [x]
        for encoder_layer in self.encoder_layers.values():
            x = encoder_layer(x)
            enc_outputs.append(x)

        # decoder with concatenated skip connections from encoder
        for decoder_idx, decoder_layer in \
                enumerate(self.decoder_layers.values()):
            encoder_idx = len(self.decoder_layers) - decoder_idx - 1
            x = decoder_layer(x)
            x = torch.cat((x, enc_outputs[encoder_idx]), dim=1)

        x = self.final_deconv(x)

        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        # self.bn1 = nn.BatchNorm3d(out_channels)
        self.maxpool1 = nn.MaxPool3d(2)
        self.relu1 = nn.ReLU(True)

        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        # self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(True)

        self.downsample = nn.Conv3d(
            in_channels, out_channels, kernel_size=1, stride=2
        )

    def forward(self, inp):
        x = self.conv1(inp)
        # x = self.bn1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        # x = self.bn2(x)

        # residual connection
        x = x + self.downsample(inp)

        x = self.relu2(x)

        return x
