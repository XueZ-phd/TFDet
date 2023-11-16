# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, ModuleList, force_fp32
from mmdet.models.losses import accuracy
from mmdet.models.backbones.resnet import Bottleneck, BasicBlock
from mmdet.models.builder import HEADS
from .bbox_head import BBoxHead
from .double_bbox_head import BasicResBlock


class AdaptiveAvgPool2d(BaseModule):
    def __init__(self):
        super(AdaptiveAvgPool2d, self).__init__()
        self.avg_pool2d = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.avg_pool2d(x).flatten(1)
        return x


@HEADS.register_module()
class FocusConvFCBBoxHead(BBoxHead):

    def __init__(self,
                 num_loc_convs=0,
                 num_fc_convs=2,
                 conv_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=dict(
                     type='Normal',
                     override=[
                         dict(type='Normal', name='fc_cls', std=0.01),
                         dict(type='Normal', name='fc_reg', std=0.001)]),
                 **kwargs):
        kwargs.setdefault('with_avg_pool', True)
        super(FocusConvFCBBoxHead, self).__init__(init_cfg=init_cfg, **kwargs)
        assert self.with_avg_pool
        assert num_loc_convs > 0
        assert num_fc_convs > 0
        self.num_loc_convs = num_loc_convs
        self.conv_out_channels = conv_out_channels
        self.num_fc_convs = num_fc_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # increase the channel of input features
        self.res_block = BasicResBlock(self.in_channels,
                                       self.conv_out_channels)

        # add conv heads
        self.conv_branch = self._add_conv_branch()

        # add fc heads
        self.fc_cls_branch = ModuleList()
        self.fc_cls_branch.append(BasicResBlock(self.in_channels, self.conv_out_channels))
        for _ in range(self.num_fc_convs):
            self.fc_cls_branch.append(Bottleneck(self.conv_out_channels, self.conv_out_channels//4,
                                                 conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))

        self.adaptive_avg_pool = AdaptiveAvgPool2d()

        out_dim_reg = 4 if self.reg_class_agnostic else 4 * self.num_classes
        self.fc_reg = nn.Linear(self.conv_out_channels, out_dim_reg)
        self.fc_cls = nn.Linear(self.conv_out_channels, self.num_classes + 1)

    def _add_conv_branch(self):
        """Add the fc branch which consists of a sequential of conv layers."""
        branch_convs = ModuleList()
        for i in range(self.num_loc_convs):
            branch_convs.append(
                Bottleneck(
                    inplanes=self.conv_out_channels,
                    planes=self.conv_out_channels // 4,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        return branch_convs

    def forward(self, x_cls, x_reg):
        # conv head
        x_conv = self.res_block(x_reg)

        for conv in self.conv_branch:
            x_conv = conv(x_conv)

        if self.with_avg_pool:
            x_conv = self.avg_pool(x_conv)

        x_conv = x_conv.view(x_conv.size(0), -1)
        bbox_pred = self.fc_reg(x_conv)

        # fc head
        for conv in self.fc_cls_branch:
            x_cls = conv(x_cls)
        cls_score = self.fc_cls(self.adaptive_avg_pool(x_cls))

        return cls_score, bbox_pred
