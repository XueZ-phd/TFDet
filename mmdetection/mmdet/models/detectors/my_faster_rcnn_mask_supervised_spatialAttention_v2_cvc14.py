import shutil
import warnings

import cv2
from PIL import Image
from glob import glob
import os
import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from mmcv.runner import auto_fp16
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer, DepthwiseSeparableConvModule
from ..utils.csp_layer import CSPLayer
from ..builder import DETECTORS, build_backbone, build_head, build_neck, build_loss
from .base import BaseDetector
from ..backbones.resnet import Bottleneck

import matplotlib.pyplot as plt


@DETECTORS.register_module()
class FasterRCNN_RGBTwMask_wSpaAttV2_CVC14(BaseDetector):
    def __init__(self,
                 share_weights,
                 backbone,
                 taf_cfg,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(FasterRCNN_RGBTwMask_wSpaAttV2_CVC14, self).__init__(init_cfg=init_cfg)

        for _k in ['backbone']:
            assert _k in list(share_weights.keys())
        self.share_weights = share_weights

        ''''backbone'''
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained

        self.backbone = build_backbone(backbone)
        if not self.share_weights['backbone']:
            self.lwir_backbone = build_backbone(backbone)

        '''neck'''
        if neck is not None:
            self.neck = build_neck(neck)

        '''dense head'''
        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        '''roi head'''
        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        '''zx fusion'''
        print('dice_weight:', taf_cfg['dice_weight'])
        print('neg_entropy_weight:', taf_cfg['neg_entropy_weight'])

        self.taf = TargetAwareFusion(taf_cfg['in_channels'],
                                     norm_cfg=taf_cfg['norm_cfg'],
                                     act_cfg=taf_cfg['act_cfg'],
                                     dice_weight=taf_cfg['dice_weight'],
                                     neg_entropy_weight=taf_cfg['neg_entropy_weight'])

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

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


    def forward_dummy(self, rgb_img, lwir_img, gt_masks=None):
        outs = ()
        # backbone
        if gt_masks is not None:
            loss_mask, x = self.extract_feat(rgb_img, lwir_img, gt_masks)
        else:
            x = self.extract_feat(rgb_img, lwir_img, None)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(rgb_img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs


    def forward_train(self,
                      rgb_img,
                      lwir_img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        assert gt_masks is not None
        losses = dict()

        loss_mask, x = self.extract_feat(rgb_img, lwir_img, gt_masks)
        losses.update(dict(loss_mask=loss_mask))

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            # forward_train 目的是训练RPNHead.其包括RPNHead的前向和损失函数。
            # 前向过程接收FPN的特征，分别产生cls和bbox预测
            # 反向过程在计算loss之前，需要产生anchor，分配正负样本，采样正负样本防止类别不平衡，计算损失
            # 由预测的cls和bbox后处理，产生最后的proposals。
            # 产生proposal的过程中，get_bbox函数可以传入score_factors,用于与预测的cls联合做后处理
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self, rgb_img, lwir_img, img_meta, proposals=None, rescale=False):
        raise NotImplementedError

    def simple_test(self, rgb_img, lwir_img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(rgb_img, lwir_img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, rgb_img, lwir_img, img_metas, rescale=False):
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

    def onnx_export(self, rgb_img, lwir_img, img_metas):
        assert rgb_img.shape == lwir_img.shape
        img_shape = torch._shape_as_tensor(rgb_img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x = self.extract_feat(rgb_img, lwir_img)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        if hasattr(self.roi_head, 'onnx_export'):
            return self.roi_head.onnx_export(x, proposals, img_metas)
        else:
            raise NotImplementedError(
                f'{self.__class__.__name__} can not '
                f'be exported to ONNX. Please refer to the '
                f'list of supported models,'
                f'https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx'  # noqa E501
            )


'''zzzzzzzzzzzzzzzzzzzzzzzxxxxxxxxxxxxxxxxxxxxxxxxxx simple fusion'''

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


class SpatialGate(nn.Module):
    def __init__(self, norm_cfg):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = ConvModule(2, 1, kernel_size, 1, (kernel_size-1)//2, bias=False, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale


class TargetAwareFusion(nn.Module):
    def __init__(self,
                 in_channels,
                 norm_cfg,
                 act_cfg,
                 dice_weight,
                 neg_entropy_weight):
        super(TargetAwareFusion, self).__init__()

        self.diceBCELoss = DiceBCELoss(dice_weight)
        self.neg_entropy_weight = neg_entropy_weight
        self.cos = torch.nn.functional.cosine_similarity
        self.cosV_conv_layers = []
        self.sigmoid = torch.nn.Sigmoid()
        self.fusion_layers = []
        self.mask_layers = []
        self.spatial_att_layers = []
        for idx, in_c in enumerate(in_channels):
            layer_idx = idx+1
            # fusion layers
            self.add_module(f'fusionLayer{layer_idx}', CSPLayer(in_c*2,
                                                                in_c,
                                                                expand_ratio=0.5,
                                                                norm_cfg=norm_cfg,
                                                                act_cfg=act_cfg))
            self.fusion_layers.append(f'fusionLayer{layer_idx}')

            # mask layers
            self.add_module(f'maskLayer{layer_idx}',
                            nn.Sequential(*[ConvModule(in_c, 64, 3, norm_cfg=norm_cfg, act_cfg=act_cfg, bias=False, padding=1),
                                            ConvModule(64, 64, 3, norm_cfg=norm_cfg, act_cfg=act_cfg, bias=False, padding=1),
                                            ConvModule(64, 1, 1, bias=False, act_cfg=dict(type='Sigmoid'))]))
            self.mask_layers.append(f'maskLayer{layer_idx}')
            self.add_module(f'cosVConvLayer{layer_idx}',
                            nn.Sequential(*[ConvModule(in_c, in_c//16, 1, bias=False, act_cfg=act_cfg),
                                            ConvModule(in_c//16, in_c, 1, bias=False, act_cfg=None)]))
            self.cosV_conv_layers.append(f'cosVConvLayer{layer_idx}')

            # spatial attention layers
            self.add_module(f'spatialAttLayer{layer_idx}', SpatialGate(norm_cfg))
            self.spatial_att_layers.append(f'spatialAttLayer{layer_idx}')


    def get_batch_mask(self, masks):
        batch_mask = []
        for mask in masks:
            mask1img = np.clip(np.sum(mask.masks, 0, keepdims=False), 0, 1)
            batch_mask.append(mask1img)
        return np.asarray(batch_mask, np.float32)

    def _neg_entropy(self, logits):
        logits = logits.squeeze(-1).squeeze(-1)
        probs = F.softmax(logits, -1)
        return torch.mean(torch.sum(probs * F.log_softmax(logits, -1), -1), 0)

    def forward(self, rgb_x, lwir_x, gt_masks):
        rgb_x = rgb_x if isinstance(rgb_x, tuple) else [rgb_x]
        lwir_x = lwir_x if isinstance(lwir_x, tuple) else [lwir_x]

        if gt_masks is not None:
            batch_gt_masks = self.get_batch_mask(gt_masks)
            loss_mask = 0

        bu_results_wSpaAtt = []
        assert len(rgb_x) == len(lwir_x) == len(self.fusion_layers) == len(self.spatial_att_layers) \
               == len(self.mask_layers)

        # np_masks = np.empty((len(self.mask_layers), 512, 640))

        for layer_idx, (tmp_rx, tmp_lx, tmp_fusion_name, tmp_mask_name, tmp_spaAtt_name) in \
                enumerate(zip(rgb_x, lwir_x, self.fusion_layers, self.mask_layers, self.spatial_att_layers)):

            tmp_fusion_layer = getattr(self, tmp_fusion_name)
            tmp_fused_res = tmp_fusion_layer(torch.cat([tmp_rx, tmp_lx], 1))

            # mask supervision
            tmp_mask_layer = getattr(self, tmp_mask_name)
            pred_mask = tmp_mask_layer(tmp_fused_res)

            # np_masks[layer_idx, ...] = np.array(Image.fromarray(pred_mask.squeeze(0).squeeze(0).detach().cpu().numpy()).resize((640, 512), resample=Image.NEAREST))

            assert tmp_rx.shape == tmp_lx.shape
            bs, c, h, w = tmp_rx.shape
            if gt_masks is not None:
                assert bs == len(batch_gt_masks)
                resized_gt_mask = np.empty((bs, 1, h, w), np.float32)
                for mask_idx, tmp_gt_mask in enumerate(batch_gt_masks):
                    resized_gt_mask[mask_idx, ...] =\
                        np.expand_dims(np.array(Image.fromarray(tmp_gt_mask).resize((w, h), resample=Image.NEAREST)), 0)
                gt_mask_1level = torch.from_numpy(resized_gt_mask).to(pred_mask.device)
                assert gt_mask_1level.requires_grad is False, 'the ground-truth mask should not be updated'
                loss_mask_ = self.diceBCELoss(pred_mask, gt_mask_1level)
                loss_mask += loss_mask_

            # refine the fused result by the cosine similarity between the pred_mask and the first fused result
            similarity_value = self.cos(pred_mask.flatten(2), tmp_fused_res.flatten(2), 2).unsqueeze(-1).unsqueeze(-1)
            tmp_cosV_cov_layer = getattr(self, self.cosV_conv_layers[layer_idx])
            similarity_logits = tmp_cosV_cov_layer(similarity_value)
            similarity_value = self.sigmoid(similarity_logits)
            refined_tmp_fused_res = similarity_value * tmp_fused_res

            if gt_masks is not None:
                loss_mask += self._neg_entropy(similarity_logits) * self.neg_entropy_weight

            tmp_spaAtt_layer = getattr(self, tmp_spaAtt_name)
            bu_results_wSpaAtt.append(tmp_spaAtt_layer(refined_tmp_fused_res))

        # save_path = '/home/zx/cross-modality-det/code/mmdetection/runs/FasterRCNN_vgg16_w_mask_SpaAttV2_ROIFocalLoss5_CIOU20_cosineSE_notDetach_negEntropy1/predMasks'
        # next_id = len(glob(save_path + '/*.png')) + 1
        # plt.imsave(f'{save_path}/{next_id:05d}.png', np_masks.mean(0), vmin=0.0, vmax=1.0, cmap='gray')

        if gt_masks is not None:
            return loss_mask/len(self.mask_layers), bu_results_wSpaAtt
        return bu_results_wSpaAtt







