from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
import warnings
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from mmdet.core import bbox2result
from mmcv.runner import auto_fp16
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
from ..utils.csp_layer import CSPLayer
from ..backbones.resnet import Bottleneck


@DETECTORS.register_module()
class RetinaNet_RGBT(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 share_weights,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RetinaNet_RGBT, self).__init__(init_cfg)

        for _k in ['backbone']:
            assert _k in list(share_weights.keys())
        self.share_weights = share_weights

        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if not self.share_weights['backbone']:
            self.lwir_backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        '''zx fusion'''
        back_norm = backbone.get('norm_cfg', None)
        self.taf = TargetAwareFusion(neck.in_channels,
                                     norm_cfg=neck.norm_cfg if back_norm is None else back_norm,
                                     act_cfg=neck.act_cfg)

    def extract_feat(self, rgb_img, lwir_img, gt_masks=None):
        """Directly extract features from the backbone+neck."""
        rgb_x = self.backbone(rgb_img)

        if self.share_weights['backbone']:
            assert not hasattr(self, 'lwir_backbone')
            lwir_x = self.backbone(lwir_img)
        else:
            lwir_x = self.lwir_backbone(lwir_img)

        if gt_masks is not None:
            loss_mask, x = self.taf(rgb_x, lwir_x, gt_masks=gt_masks)
        else:
            x = self.taf(rgb_x, lwir_x, gt_masks=None)

        if self.with_neck:
            x = self.neck(x)

        if gt_masks is not None:
            return loss_mask, x

        return x

    def forward_dummy(self, img):
        raise NotImplementedError

    def forward_train(self,
                      rgb_img,
                      lwir_img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,):

        loss_mask, x = self.extract_feat(rgb_img, lwir_img, gt_masks)

        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        losses.update(dict(loss_mask=loss_mask))

        return losses

    async def async_simple_test(self, rgb_img, lwir_img, img_meta, proposals=None, rescale=False):
        raise NotImplementedError

    def simple_test(self, rgb_img, lwir_img, img_metas, rescale=False):

        feat = self.extract_feat(rgb_img, lwir_img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

    def onnx_export(self, img, img_metas, with_nms=True):
        raise NotImplementedError

    def forward_test(self, rgb_imgs, lwir_imgs, img_metas, **kwargs):
        for var, name in [(rgb_imgs, 'rgb_imgs'), (lwir_imgs, 'lwir_imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        assert len(rgb_imgs) == len(lwir_imgs)
        num_augs = len(rgb_imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(rgb_imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        for rgb_img, lwir_img, img_meta in zip(rgb_imgs, lwir_imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                assert rgb_img.size() == lwir_img.size()
                img_meta[img_id]['batch_input_shape'] = tuple(rgb_img.size()[-2:])

        if num_augs == 1:
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(rgb_imgs[0], lwir_imgs[0], img_metas[0], **kwargs)
        else:
            assert lwir_img[0].size(0) == rgb_img[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{rgb_img[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(rgb_imgs, lwir_imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('rgb_img', 'lwir_img'))
    def forward(self, rgb_img, lwir_img, img_metas, return_loss=True, **kwargs):
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(rgb_img[0], lwir_img[0], img_metas[0])

        if return_loss:
            assert kwargs['gt_masks'] is not None
            return self.forward_train(rgb_img, lwir_img, img_metas, **kwargs)
        else:
            assert 'gt_masks' not in list(kwargs.keys())
            return self.forward_test(rgb_img, lwir_img, img_metas, **kwargs)



'''zzzzzzzzzzzzzzzzzzzzzzzxxxxxxxxxxxxxxxxxxxxxxxxxx simple fusion'''


class zxCSPLayer(CSPLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expand_ratio=0.5,
                 num_blocks=1,
                 add_identity=True,
                 use_depthwise=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=None,
                 final_act_cfg=None):
        super(zxCSPLayer, self).__init__(
            in_channels,
            out_channels,
            expand_ratio,
            num_blocks,
            add_identity,
            use_depthwise,
            conv_cfg,
            norm_cfg,
            act_cfg,
            init_cfg)

        mid_channels = int(out_channels * expand_ratio)
        self.final_conv = ConvModule(
            2 * mid_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=final_act_cfg)

    def forward(self, x):
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)
        return self.final_conv(x_final)


class zxBottleneck(Bottleneck):
    expansion = 1
    def __init__(self, *args, **kwargs):
        super(zxBottleneck, self).__init__(*args, **kwargs)

        downsample = []
        downsample.extend([
            build_conv_layer(
                self.conv_cfg,
                self.inplanes,
                self.planes,
                kernel_size=1,
                stride=self.stride,
                bias=False),
            build_norm_layer(self.norm_cfg, self.planes)[1]
        ])

        self.downsample = nn.Sequential(*downsample)

    def forward(self, x, fused_x):
        """Forward function."""
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        # if self.with_plugins:
        #     out = self.forward_plugin(out, self.after_conv1_plugin_names)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        # if self.with_plugins:
        #     out = self.forward_plugin(out, self.after_conv2_plugin_names)

        out = self.conv3(out)
        out = self.norm3(out)

        # if self.with_plugins:
        #     out = self.forward_plugin(out, self.after_conv3_plugin_names)

        identity = self.downsample(x)

        out = out + identity + fused_x

        out = self.relu(out)

        return out


class DiceBCELoss(nn.Module):
    def __init__(self, weight=1.0, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE * self.weight


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale


class TargetAwareFusion(nn.Module):
    def __init__(self,
                 in_channels,
                 norm_cfg,
                 act_cfg,):
        super(TargetAwareFusion, self).__init__()

        self.diceBCELoss = DiceBCELoss(2.0)
        self.fusion_layers = []
        self.downsample_layers = []
        self.mask_layers = []
        self.spatial_att_layers = []
        for idx, in_c in enumerate(in_channels):
            layer_idx = idx+1
            # fusion layers
            self.add_module(f'fusionLayer{layer_idx}', zxCSPLayer(in_c*2,
                                                                  in_c,
                                                                  expand_ratio=0.5,
                                                                  norm_cfg=norm_cfg,
                                                                  act_cfg=act_cfg,
                                                                  final_act_cfg=act_cfg if layer_idx == 1 else None))
            self.fusion_layers.append(f'fusionLayer{layer_idx}')

            # mask layers
            self.add_module(f'maskLayer{layer_idx}',
                            nn.Sequential(*[ConvModule(in_c, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg),
                                            ConvModule(1, 1, 1, norm_cfg=norm_cfg, act_cfg=dict(type='Sigmoid'))]))
            self.mask_layers.append(f'maskLayer{layer_idx}')

            # spatial attention layers
            self.add_module(f'spatialAttLayer{layer_idx}', SpatialGate())
            self.spatial_att_layers.append(f'spatialAttLayer{layer_idx}')

            if layer_idx < len(in_channels):
                self.add_module(f'bottomUpLayer{layer_idx}', zxBottleneck(in_c, in_channels[layer_idx], 2,))
                self.downsample_layers.append(f'bottomUpLayer{layer_idx}')

    def get_batch_mask(self, masks):
        batch_mask = []
        for mask in masks:
            mask1img = np.clip(np.sum(mask.masks, 0, keepdims=True), 0, 1)
            batch_mask.append(mask1img)
        return torch.from_numpy(np.asarray(batch_mask, np.float32))

    def forward(self, rgb_x, lwir_x, gt_masks):
        rgb_x = rgb_x if isinstance(rgb_x, tuple) else [rgb_x]
        lwir_x = lwir_x if isinstance(lwir_x, tuple) else [lwir_x]

        if gt_masks is not None:
            batch_gt_masks = self.get_batch_mask(gt_masks)
            loss_mask = 0

        fused_results = []
        bu_results = []
        bu_results_wSpaAtt = []
        assert len(rgb_x) == len(lwir_x) == len(self.fusion_layers) == len(self.spatial_att_layers) == len(self.mask_layers)
        for layer_idx, (tmp_rx, tmp_lx, tmp_fusion_name, tmp_mask_name, tmp_spaAtt_name) in enumerate(zip(rgb_x, lwir_x, self.fusion_layers, self.mask_layers, self.spatial_att_layers)):

            tmp_fusion_layer = getattr(self, tmp_fusion_name)
            tmp_fused_res = tmp_fusion_layer(torch.cat([tmp_rx, tmp_lx], 1))
            fused_results.append(tmp_fused_res)

            # bootom-up layer
            if layer_idx == 0:
                bu_results.append(tmp_fused_res)

            if layer_idx > 0:
                bottomupLayer = getattr(self, f'bottomUpLayer{layer_idx}')
                bu_results.append(bottomupLayer(bu_results[-1], fused_results[-1]))

            # mask supervision
            tmp_mask_layer = getattr(self, tmp_mask_name)
            pred_mask = tmp_mask_layer(bu_results[-1])

            assert tmp_rx.shape == tmp_lx.shape
            bs, c, h, w = tmp_rx.shape
            if gt_masks is not None:
                gt_mask_1level = F.interpolate(batch_gt_masks, (h, w), mode='nearest').to(pred_mask.device)
                loss_mask_ = self.diceBCELoss(pred_mask, gt_mask_1level)
                loss_mask += loss_mask_

            # spatial attention
            tmp_spaAtt_layer = getattr(self, tmp_spaAtt_name)
            bu_results_wSpaAtt.append(tmp_spaAtt_layer(bu_results[-1]))

        if gt_masks is not None:
            return loss_mask/len(self.mask_layers), bu_results_wSpaAtt
        return bu_results_wSpaAtt
