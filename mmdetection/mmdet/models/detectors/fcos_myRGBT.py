# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import torch

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck, build_loss
from .base import BaseDetector
import torch.nn as nn
from mmcv.runner import auto_fp16
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
import collections
from ..utils.csp_layer import CSPLayer
import torch.nn.functional as F
from ..backbones.resnet import Bottleneck

@DETECTORS.register_module()
class FCOS_RGBT(BaseDetector):
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
        super(FCOS_RGBT, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        '''backbone'''
        self.rgb_backbone = build_backbone(backbone)
        if share_weights['backbone']:
            self.lwir_backbone = self.rgb_backbone
        else:
            self.lwir_backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.taf = TargetAwareFusion(neck.in_channels, backbone.norm_cfg, dict(type='ReLU'))

    def extract_feat(self, rgb_img, lwir_img, gt_masks=None):
        """Directly extract features from the backbone+neck."""
        rgb_x = self.rgb_backbone(rgb_img)
        lwir_x = self.lwir_backbone(rgb_img)

        if gt_masks is not None:
            loss_mask, x = self.taf(rgb_x, lwir_x, gt_masks=gt_masks)
        else:
            x = self.taf(rgb_x, lwir_x, gt_masks=None)

        if self.with_neck:
            x = self.neck(x)

        if gt_masks is not None:
            return loss_mask, x

        return x

    def forward_dummy(self, rgb_img, lwir_img, gt_masks=None):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        if gt_masks is not None:
            loss_mask, x = self.extract_feat(rgb_img, lwir_img, gt_masks)
        else:
            x = self.extract_feat(rgb_img, lwir_img, None)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      rgb_img,
                      lwir_img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):

        assert tuple(rgb_img[0].size()[-2:])==tuple(lwir_img[0].size()[-2:])
        batch_input_shape = tuple(rgb_img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

        if gt_masks is not None:
            loss_mask, x = self.extract_feat(rgb_img, lwir_img, gt_masks)
        else:
            x = self.extract_feat(rgb_img, lwir_img, None)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        if gt_masks is not None:
            losses.update(dict(loss_mask=loss_mask))
        return losses

    def simple_test(self, rgb_img, lwir_img, img_metas, rescale=False):
        feat = self.extract_feat(rgb_img, lwir_img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, rgb_imgs, lwir_imgs, img_metas, rescale=False):

        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(rgb_imgs, lwir_imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, rgb_img, lwir_img, img_metas, with_nms=True):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(rgb_img, lwir_img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(rgb_img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)
        # TODO Can we change to `get_bboxes` when `onnx_export` fail
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            *outs, img_metas, with_nms=with_nms)

        return det_bboxes, det_labels

    def forward_test(self, rgb_imgs, lwir_imgs, img_metas, **kwargs):
        for var, name in [(rgb_imgs, 'rgb_imgs'), (lwir_imgs, 'lwir_imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        assert len(rgb_imgs) == len(lwir_imgs)
        num_augs = len(rgb_imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(rgb_imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for rgb_img, lwir_img, img_meta in zip(rgb_imgs, lwir_imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(rgb_img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(rgb_imgs[0], lwir_imgs[0], img_metas[0], **kwargs)
        else:
            assert rgb_imgs[0].size(0) == lwir_imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{rgb_imgs[0].size(0)}'
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

    def forward(self, x, mask):
        mask = mask.to(x.device)
        x_short = self.short_conv(x)

        x_main = self.main_conv(x * mask)
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
    def __init__(self, weight=1, size_average=True):
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


class TargetAwareFusion(nn.Module):
    def __init__(self,
                 in_channels,
                 norm_cfg,
                 act_cfg):
        super(TargetAwareFusion, self).__init__()

        self.diceBCELoss = DiceBCELoss()
        self.fusion_layers = []
        self.downsample_layers = []
        self.mask_layers = []
        for layer_idx, in_c in enumerate(in_channels):
            layer_idx = layer_idx+1
            self.add_module(f'fusionLayer{layer_idx}', zxCSPLayer(in_c*2, in_c,
                                                                  norm_cfg=norm_cfg,
                                                                  act_cfg=act_cfg,
                                                                  final_act_cfg=act_cfg if layer_idx==1 else None))
            self.fusion_layers.append(f'fusionLayer{layer_idx}')

            self.add_module(f'maskLayer{layer_idx}',
                            nn.Sequential(*[ConvModule(2*in_c, 1, 1, act_cfg=act_cfg),
                                            ConvModule(1, 1, 1, norm_cfg=norm_cfg, act_cfg=dict(type='Sigmoid'))]))
            self.mask_layers.append(f'maskLayer{layer_idx}')

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
        assert len(rgb_x) == len(self.fusion_layers) == len(lwir_x)

        if gt_masks is not None:
            batch_gt_masks = self.get_batch_mask(gt_masks)
            loss_mask = 0

        results = []
        assert len(rgb_x) == len(lwir_x) == len(self.fusion_layers)
        for idx, (tmp_rx, tmp_lx, tmp_fusion_name, tmp_mask_name) in enumerate(zip(rgb_x, lwir_x, self.fusion_layers, self.mask_layers)):
            tmp_mask_layer = getattr(self, tmp_mask_name)
            pred_mask = tmp_mask_layer(torch.cat([tmp_rx, tmp_lx], 1))

            tmp_fusion_layer = getattr(self, tmp_fusion_name)
            assert tmp_rx.shape == tmp_lx.shape
            bs, c, h, w = tmp_rx.shape
            if gt_masks is not None and pred_mask is not None:
                gt_mask_1level = F.interpolate(batch_gt_masks, (h, w), mode='nearest').to(pred_mask.device)
                loss_mask_ = self.diceBCELoss(pred_mask, gt_mask_1level)
                loss_mask += loss_mask_

            # tmp_res = tmp_fusion_layer(torch.cat([tmp_rx, tmp_lx], 1), pred_mask)

            pred_mask=torch.ones_like(pred_mask)
            tmp_res = tmp_fusion_layer(torch.cat([tmp_rx, tmp_lx], 1), pred_mask)
            results.append(tmp_res)

        # bottom-up
        results_bu = [results[0]]
        for level in range(1, len(results)):
            bottomupLayer = getattr(self, f'bottomUpLayer{level}')
            results_bu.append(bottomupLayer(results_bu[-1], results[level]))

        if gt_masks is not None:
            return loss_mask/len(self.mask_layers), results_bu
        return results_bu







