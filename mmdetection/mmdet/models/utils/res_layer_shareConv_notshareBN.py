# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule, Sequential
from torch import nn as nn


class DownSample(nn.Module):
    def __init__(self, conv_cfg, inplanes, planes, kernel_size, stride, bias,
                 norm_cfg):
        super(DownSample, self).__init__()
        self.conv = build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=bias)
        self.norm = build_norm_layer(norm_cfg, planes)[1]
        self.norm_notShare = build_norm_layer(norm_cfg, planes)[1]

    def forward(self, x):
        rgb_x, lwir_x = x
        rgb_x = self.conv(rgb_x)
        lwir_x = self.conv(lwir_x)
        rgb_x, lwir_x = self.norm(rgb_x), self.norm_notShare(lwir_x)
        return tuple([rgb_x, lwir_x])


class AvgPool2d_Two_Inputs(nn.Module):
    def __init__(self, kernel_size, stride, ceil_mode=True, count_include_pad=False):
        super(AvgPool2d_Two_Inputs, self).__init__()
        self.pool = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad)

    def forward(self, x):
        rgb_x, lwir_x = x
        return tuple([self.pool(rgb_x), self.pool(lwir_x)])


class ResLayer_shareConv_notShareBN(Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
    """

    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 downsample_first=True,
                 **kwargs):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(
                    AvgPool2d_Two_Inputs(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.append(
                DownSample(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False,
                    norm_cfg=norm_cfg))
            downsample = nn.Sequential(*downsample)

        layers = []
        if downsample_first:
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
            inplanes = planes * block.expansion
            for _ in range(1, num_blocks):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))

        else:  # downsample_first=False is for HourglassModule
            for _ in range(num_blocks - 1):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=inplanes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        super(ResLayer_shareConv_notShareBN, self).__init__(*layers)

