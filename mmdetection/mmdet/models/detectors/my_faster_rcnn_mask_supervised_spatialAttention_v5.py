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
from mmcv.runner import BaseModule
from mmcv.runner import auto_fp16
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer, DepthwiseSeparableConvModule
from ..utils.csp_layer import CSPLayer
from ..builder import DETECTORS, build_backbone, build_head, build_neck, build_loss
from .base import BaseDetector
from ..backbones.resnet import Bottleneck
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
import matplotlib.pyplot as plt

ren = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
attn = lambda features: torch.sum(torch.softmax(features, 1) * features, dim=1)[0].detach().cpu().numpy()
# attn = lambda features: torch.mean(features, dim=1)[0].detach().cpu().numpy()

@DETECTORS.register_module()
class FasterRCNN_RGBTwMask_wSpaAttV5(BaseDetector):
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
        super(FasterRCNN_RGBTwMask_wSpaAttV5, self).__init__(init_cfg=init_cfg)

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

        # save_dir = '/home/zx/cross-modality-det/code/mmdetection/runs/FasterRCNN_vgg16_channelRelation_dscSEFusion_similarityMax_1/rgbt_feature_cosine_similarities'
        # save_dir_s0 = os.path.join(save_dir, 'scale0')
        # save_dir_s1 = os.path.join(save_dir, 'scale1')
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir_s0, exist_ok=True)
        #     os.makedirs(save_dir_s1, exist_ok=True)
        # # scale 0
        # x0, x1 = rgb_x[0], lwir_x[0]
        # cosine = F.cosine_similarity(x0.flatten(2).unsqueeze(1), x1.flatten(2).unsqueeze(2), dim=3).squeeze().detach().cpu().numpy()
        # n_npf = len(glob(os.path.join(save_dir_s0, '*.npy')))
        # np.save(os.path.join(save_dir_s0, f'{n_npf:04d}.npy'), cosine)
        # # scale 1
        # x0, x1 = rgb_x[1], lwir_x[1]
        # cosine = F.cosine_similarity(x0.flatten(2).unsqueeze(1), x1.flatten(2).unsqueeze(2),
        #                              dim=3).squeeze().detach().cpu().numpy()
        # n_npf = len(glob(os.path.join(save_dir_s1, '*.npy')))
        # np.save(os.path.join(save_dir_s1, f'{n_npf:04d}.npy'), cosine)

        if gt_masks is not None:
            loss_mask, x = self.taf(rgb_x, lwir_x, gt_masks=gt_masks)
        else:
            x = self.taf(rgb_x, lwir_x, gt_masks=None)

        # # save feature visualization
        # save_root = '/home/zx/cross-modality-det/code/mmdetection/runs/FasterRCNN_vgg16_AblationScratch_onlyFFM_channelRelation_dscFusion_se/feature_visualization'
        # # save_root = '/home/zx/cross-modality-det/code/mmdetection/runs/FasterRCNN_vgg16_channelRelation_dscSEFusion_similarityMax_1/feature_visualization'
        # if not os.path.exists(save_root):
        #     os.makedirs(save_root, exist_ok=False)
        # num = len(glob(os.path.join(save_root, '*_rgb.png')))
        #
        # plt.imshow(ren(rgb_img[0].permute([1, 2, 0]).detach().cpu().numpy()))
        # plt.imshow(cv2.resize(attn(x[0]), (640, 512)), alpha=0.5)
        # plt.axis('off')
        # plt.savefig(os.path.join(save_root, f'{num:04d}_rgb.png'), dpi=150, bbox_inches='tight', pad_inches=0.0)
        # plt.close(plt.gcf())
        #
        # plt.imshow(ren(lwir_img[0].permute([1, 2, 0]).detach().cpu().numpy()))
        # plt.imshow(cv2.resize(attn(x[0]), (640, 512)), alpha=0.5)
        # plt.axis('off')
        # plt.savefig(os.path.join(save_root, f'{num:04d}_lwir.png'), dpi=150, bbox_inches='tight', pad_inches=0.0)
        # plt.close(plt.gcf())



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
            outs = outs + (rpn_outs,)
        proposals = torch.randn(1000, 4).to(rgb_img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs,)
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

        # # save bboxes
        # save_root = '/home/zx/cross-modality-det/code/mmdetection/runs/FasterRCNN_vgg16_AblationScratch_onlyFFM_channelRelation_dscFusion_se/feature_visualization'
        # # save_root = '/home/zx/cross-modality-det/code/mmdetection/runs/FasterRCNN_vgg16_channelRelation_dscSEFusion_similarityMax_1/feature_visualization'
        # rgb_img_np = cv2.imread(img_metas[0]['rgb_filename'])
        # lwir_img_np = cv2.imread(img_metas[0]['lwir_filename'])
        #
        # gt_box_file = os.path.join(
        #     '/home/zx/cross-modality-det/datasets/zx-sanitized-kaist-keepPerson-fillNonPerson/annotations/test',
        #     os.path.basename(img_metas[0]['rgb_filename']).replace('.png', '.txt')
        # )
        # assert os.path.isfile(gt_box_file)
        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore')
        #     gt_bboxes = np.loadtxt(gt_box_file, skiprows=1, usecols=[1, 2, 3, 4], dtype=int, ndmin=2)
        # for g_bbox in gt_bboxes:
        #     gx0, gy0, gw, gh = g_bbox
        #     gx1, gy1 = gx0 + gw, gy0 + gh
        #     cv2.rectangle(rgb_img_np, (gx0, gy0), (gx1, gy1), (0, 255, 0), 1)
        #     cv2.rectangle(lwir_img_np, (gx0, gy0), (gx1, gy1), (0, 255, 0), 1)
        #
        # bboxes = self.roi_head.simple_test(
        #     x, proposal_list, img_metas, rescale=rescale)
        # for box in bboxes[0][0]:
        #     x0, y0, x1, y1 = list(map(int, box[:4]))
        #     score = box[-1]
        #     if score < 0.3:
        #         continue
        #     cv2.rectangle(rgb_img_np, (x0, y0), (x1, y1), (0, 0, 255), 1)
        #     cv2.rectangle(lwir_img_np, (x0, y0), (x1, y1), (0, 0, 255), 1)
        # num = len(glob(os.path.join(save_root, '*_rgb.png'))) - 1
        # cv2.imwrite(os.path.join(save_root, f'{num:04d}_rgb_label.png'), rgb_img_np)
        # cv2.imwrite(os.path.join(save_root, f'{num:04d}_lwir_label.png'), lwir_img_np)
        # return bboxes

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
            # print(img_meta[0]['lwir_filename'])
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
                f'https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx'
                # noqa E501
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
        # dice
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = dice_loss + BCE

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


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AdaptiveModalitySel(BaseModule):
    def __init__(self, in_channel, norm_cfg, act_cfg,
                 init_cfg=[dict(type='Kaiming', layer='Conv2d'),
                           dict(type='Constant', layer='_BatchNorm', val=1.0)]):
        super(AdaptiveModalitySel, self).__init__(init_cfg=init_cfg)
        ks = 7
        self.conv_out = nn.Sequential(
            ConvModule(2 * in_channel, 2, 1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(2, 2, ks, padding=ks//2, act_cfg=None, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, xa, xb):
        prob = self.sigmoid(self.conv_out(torch.cat([xa, xb], dim=1)))
        a, b = torch.chunk(prob, 2, 1)
        xa = a * xa + xa
        xb = b * xb + xb
        return xa, xb


class GlobalChannelCommunication(BaseModule):
    def __init__(self, channel, k_size,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv1d', 'Linear']),
                     dict(
                         type='Constant',
                         layer='LayerNorm',
                         val=1)]):
        super(GlobalChannelCommunication, self).__init__(init_cfg=init_cfg)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.norm = nn.LayerNorm(channel)
        self.act = nn.GELU()
        self.fc = nn.Linear(channel, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)    # [b, c, 1, 1]
        y = self.norm(self.conv(y.squeeze(-1).transpose(-1, -2))).transpose(-1, -2).unsqueeze(-1)  # [b, c, 1, 1]
        y = self.act(y).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class TargetAwareFusion(BaseModule):
    def __init__(self,
                 in_channels,
                 norm_cfg,
                 act_cfg,
                 dice_weight,
                 neg_entropy_weight,
                 init_cfg=[dict(type='Kaiming', layer='Conv2d'),
                           dict(type='Constant', layer='_BatchNorm', val=1.0)]):
        super(TargetAwareFusion, self).__init__(init_cfg=init_cfg)
        self.diceBCELoss = DiceBCELoss(dice_weight)
        self.neg_entropy_weight = neg_entropy_weight
        self.cos = torch.nn.functional.cosine_similarity
        self.sigmoid = torch.nn.Sigmoid()
        for idx, in_c in enumerate(in_channels):
            # deformable layer
            self.add_module(f'rgb_deformable_layer_{idx}',
                            build_conv_layer(dict(type='DCNv2', deform_groups=1), in_c, in_c, kernel_size=3, padding=0,
                                             bias=False))
            self.add_module(f'lwir_deformable_layer_{idx}',
                            build_conv_layer(dict(type='DCNv2', deform_groups=1), in_c, in_c, kernel_size=3, padding=0,
                                             bias=False))
            # self.add_module(f'rgb_conv_layer_{idx}',
            #                 ConvModule(in_c, in_c, 3, padding=1, act_cfg=None, bias=False))
            # self.add_module(f'lwir_conv_layer_{idx}',
            #                 ConvModule(in_c, in_c, 3, padding=1, act_cfg=None, bias=False))
            # fusion layers
            self.add_module(f'group_conv3d_layer_{idx}',
                            nn.Sequential(
                                nn.Conv3d(in_c, in_c, (2, 1, 1), groups=in_c, bias=False),
                                nn.BatchNorm3d(in_c),
                                nn.GELU()))
            nn.init.constant_(getattr(self, f'group_conv3d_layer_{idx}')[0].weight, 1.0)
            nn.init.constant_(getattr(self, f'group_conv3d_layer_{idx}')[1].weight, 1.0)
            nn.init.constant_(getattr(self, f'group_conv3d_layer_{idx}')[1].bias, 0.0)
            # # channel weight layer
            self.add_module(f'dsc_layer_{idx}', ConvModule(in_c, in_c, 1, groups=in_c, bias=False,
                                                           norm_cfg=norm_cfg, act_cfg=act_cfg))
            # self.add_module(f'conv1x1_layer_{idx}', ConvModule(in_c, in_c, 1, norm_cfg=norm_cfg, act_cfg=act_cfg))
            self.add_module(f'se_layer_{idx}', SELayer(in_c))
            # mask layers
            self.add_module(f'mask_layer_{idx}',
                            nn.Sequential(*[ConvModule(in_c, 64, 3, norm_cfg=norm_cfg, act_cfg=act_cfg, bias=False, padding=1),
                                            ConvModule(64, 64, 3, norm_cfg=norm_cfg, act_cfg=act_cfg, bias=False, padding=1),
                                            ConvModule(64, 1, 1, bias=False, act_cfg=dict(type='Sigmoid'))]))

            self.add_module(f'mlp_layer_{idx}',
                            nn.Sequential(*[ConvModule(in_c, in_c//16, 1, bias=False, act_cfg=act_cfg),
                                            ConvModule(in_c//16, in_c, 1, bias=False, act_cfg=dict(type='Sigmoid'))]))

            # spatial attention layers
            self.add_module(f'spatial_layer_{idx}', SpatialGate(norm_cfg=norm_cfg))

            # # cat conv
            # self.add_module(f'conv{idx}', nn.Sequential(
            #     ConvModule(2*in_c, in_c, 1, act_cfg=act_cfg, norm_cfg=norm_cfg, bias=False),
            #     CSPLayer(in_c, in_c, act_cfg=act_cfg, norm_cfg=norm_cfg)))


    def get_batch_mask(self, masks):
        batch_mask = []
        for mask in masks:
            mask1img = np.clip(np.sum(mask.masks, 0, keepdims=False), 0, 1)
            batch_mask.append(mask1img)
        return np.asarray(batch_mask, np.float32)

    def forward(self, rgb_x, lwir_x, gt_masks):
        rgb_x = rgb_x if isinstance(rgb_x, tuple) else [rgb_x]
        lwir_x = lwir_x if isinstance(lwir_x, tuple) else [lwir_x]

        if gt_masks is not None:
            batch_gt_masks = self.get_batch_mask(gt_masks)
            # loss_mask = 0.0
            loss_mask = torch.tensor(0.0).to(rgb_x[0].device)

        results = []
        for idx in range(len(rgb_x)):
            tmp_rx = rgb_x[idx]
            tmp_lx = lwir_x[idx]
            # deformable layer
            tmp_rx = getattr(self, f'rgb_deformable_layer_{idx}')(tmp_rx)
            tmp_lx = getattr(self, f'lwir_deformable_layer_{idx}')(tmp_lx)
            # tmp_rx = getattr(self, f'rgb_conv_layer_{idx}')(tmp_rx)
            # tmp_lx = getattr(self, f'lwir_conv_layer_{idx}')(tmp_lx)
            # group conv3d fusion
            tmp_fused_res = getattr(self, f'group_conv3d_layer_{idx}')(torch.cat([tmp_rx.unsqueeze(2), tmp_lx.unsqueeze(2)], 2))
            tmp_fused_res = tmp_fused_res.squeeze(2)
            # self-attention
            tmp_fused_res = getattr(self, f'dsc_layer_{idx}')(tmp_fused_res)
            # tmp_fused_res = getattr(self, f'conv1x1_layer_{idx}')(tmp_fused_res)
            tmp_fused_res = getattr(self, f'se_layer_{idx}')(tmp_fused_res)
            # mask supervision
            pred_mask = getattr(self, f'mask_layer_{idx}')(tmp_fused_res)
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
            v = self.cos(pred_mask.flatten(2), tmp_fused_res.flatten(2), 2).unsqueeze(-1).unsqueeze(-1)
            s = getattr(self, f'mlp_layer_{idx}')(v)
            tmp_fused_res = s * tmp_fused_res

            if gt_masks is not None:
                loss_mask += F.binary_cross_entropy(s, torch.ones_like(s)) * self.neg_entropy_weight

            tmp_fused_res = getattr(self, f'spatial_layer_{idx}')(tmp_fused_res)
            results.append(tmp_fused_res)

            # # cat conv
            # x = torch.cat([tmp_rx, tmp_lx], 1)
            # x = getattr(self, f'conv{idx}')(x)
            # results.append(x)

        if gt_masks is not None:
            return loss_mask/len(rgb_x), results
        return results









