from mmcv.cnn.bricks.non_local import _NonLocalNd
from typing import Dict
import torch.nn as nn
import torch

class zxCrossNonLocal2d(_NonLocalNd):

    def __init__(self,
                 sub_sample: bool = False,
                 conv_cfg: Dict = dict(type='Conv2d'),
                 **kwargs):
        super().__init__(in_channels=1, conv_cfg=conv_cfg, reduction=1, **kwargs)

        self.sub_sample = sub_sample

        if sub_sample:
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            self.g = nn.Sequential(self.g, max_pool_layer)
            if self.mode != 'gaussian':
                self.phi = nn.Sequential(self.phi, max_pool_layer)
            else:
                self.phi = max_pool_layer



    def forward(self, rgb_x: torch.Tensor, lwir_x: torch.Tensor) -> torch.Tensor:
        lwir_expectation = torch.sum(torch.softmax(lwir_x, dim=1) * lwir_x, dim=1, keepdim=True)
        rgb_expectation = torch.sum(torch.softmax(rgb_x, dim=1) * rgb_x, dim=1, keepdim=True)

        n = rgb_x.size(0)
        g_x = self.g(lwir_expectation).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        if self.mode == 'gaussian':
            theta_x = rgb_expectation.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample:
                phi_x = self.phi(lwir_expectation).view(n, self.in_channels, -1)
            else:
                phi_x = lwir_expectation.view(n, self.in_channels, -1)
        elif self.mode == 'concatenation':
            theta_x = self.theta(rgb_expectation).view(n, self.inter_channels, -1, 1)
            phi_x = self.phi(lwir_expectation).view(n, self.inter_channels, 1, -1)
        else:
            theta_x = self.theta(rgb_expectation).view(n, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(lwir_expectation).view(n, self.inter_channels, -1)

        pairwise_func = getattr(self, self.mode)
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # NonLocal1d y: [N, H, C]
        # NonLocal2d y: [N, HxW, C]
        # NonLocal3d y: [N, TxHxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        # NonLocal1d y: [N, C, H]
        # NonLocal2d y: [N, C, H, W]
        # NonLocal3d y: [N, C, T, H, W]
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels,
                                                    *lwir_expectation.size()[2:])

        output = lwir_x + self.conv_out(y)

        return output