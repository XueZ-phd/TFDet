import warnings

import numpy as np
import torch.nn as nn
import torchvision.ops
from mmcv.cnn import ConvModule, ContextBlock, DepthwiseSeparableConvModule
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from mmcv.runner import auto_fp16
import torch


@DETECTORS.register_module()
class FasterRCNN_RGBT(BaseDetector):
    def __init__(self,
                 share_weights:[bool, dict],
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(FasterRCNN_RGBT, self).__init__(init_cfg=init_cfg)

        if isinstance(share_weights, bool):
            self.share_weights = {}
            self.share_weights['backbone'] = share_weights
            self.share_weights['neck'] = share_weights
            self.share_weights['dense_head'] = share_weights

        elif isinstance(share_weights, dict):
            for _k in ['backbone', 'neck', 'dense_head']:
                assert _k in list(share_weights.keys())
            self.share_weights = share_weights

        ''''backbone'''
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained

        self.rgb_backbone = build_backbone(backbone)
        if self.share_weights['backbone']:
            self.lwir_backbone = self.rgb_backbone
        else:
            self.lwir_backbone = build_backbone(backbone)

        '''neck'''
        if neck is not None:
            self.rgb_neck = build_neck(neck)
            if self.share_weights['neck']:
                self.lwir_neck = self.rgb_neck
            else:
                self.lwir_neck = build_neck(neck)

        '''dense head'''
        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)

            self.rgb_rpn_head = build_head(rpn_head_)
            if self.share_weights['dense_head']:
                self.lwir_rpn_head = self.rgb_rpn_head
            else:
                self.lwir_rpn_head = build_head(rpn_head_)
            # rpn head of rgb-t features
            # self.rgbT_rpn_head = build_head(rpn_head_)

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

        self.probEn = ProbEn()


    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rgb_rpn_head') and self.rgb_rpn_head is not None \
               and hasattr(self, 'lwir_rpn_head') and self.lwir_rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'rgb_neck') and self.rgb_neck is not None \
               and hasattr(self, 'lwir_neck') and self.lwir_neck is not None

    def extract_feat(self, rgb_img, lwir_img):
        """Directly extract features from the backbone+neck."""
        rgb_x = self.rgb_backbone(rgb_img)
        lwir_x = self.lwir_backbone(lwir_img)
        if self.with_neck:
            rgb_x = self.rgb_neck(rgb_x)
            lwir_x = self.lwir_neck(lwir_x)

        rgbT_x = self.fuse_rgbt_feature(rgb_x, lwir_x)

        return rgb_x, lwir_x, rgbT_x


    def fuse_rgbt_feature(self, rgb_x, lwir_x):
        x = []
        for _rgb_x, _lwir_x in zip(rgb_x, lwir_x):
            # rgbT_x = torch.cat([_rgb_x, _lwir_x], 1)
            # dw_x = self.fusion_net_dw(rgbT_x)
            # ct_x = self.fusion_net_ct(rgbT_x)
            # conv_x = self.fusion_conv1(rgbT_x)
            # rgbT_x = dw_x+ct_x+conv_x+_rgb_x+_lwir_x
            # x.append(self.fusion_conv2(rgbT_x))
            rgbT_x = _rgb_x + _lwir_x
            x.append(rgbT_x)
        return x

    def merge_proposals(self, rgb_proposal_list, lwir_proposal_list):
        proposals_list = []
        for rgb_p, lwir_p in zip(rgb_proposal_list, lwir_proposal_list):
            dets = torch.cat([rgb_p, lwir_p], 0)
            props = self.probEn.nms_bayesian(dets[:, :4], dets[:, -1], 0.5,)
            proposals_list.append(props.to(rgb_p.device))
        return proposals_list

    def forward_dummy(self, rgb_img, lwir_img):

        raise NotImplementedError

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

        rgb_x, lwir_x, rgbT_x = self.extract_feat(rgb_img, lwir_img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            # forward_train 目的是训练RPNHead.其包括RPNHead的前向和损失函数。
            # 前向过程接收FPN的特征，分别产生cls和bbox预测
            # 反向过程在计算loss之前，需要产生anchor，分配正负样本，采样正负样本防止类别不平衡，计算损失
            # 由预测的cls和bbox后处理，产生最后的proposals。
            # 产生proposal的过程中，get_bbox函数可以传入score_factors,用于与预测的cls联合做后处理
            rgb_rpn_losses, rgb_proposal_list = self.rgb_rpn_head.forward_train(
                rgb_x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            _rgb_rpn_losses = {}
            for _k in list(rgb_rpn_losses.keys()):
                _rgb_rpn_losses[f'rgb_{_k}'] = rgb_rpn_losses[_k]
            losses.update(_rgb_rpn_losses)

            lwir_rpn_losses, lwir_proposal_list = self.lwir_rpn_head.forward_train(
                lwir_x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            _lwir_rpn_losses = {}
            for _k in list(lwir_rpn_losses.keys()):
                _lwir_rpn_losses[f'lwir_{_k}'] = lwir_rpn_losses[_k]
            losses.update(_lwir_rpn_losses)

            rgbT_proposal_list = self.merge_proposals(rgb_proposal_list, lwir_proposal_list)

        else:
            rgb_proposal_list = proposals
            lwir_proposal_list = proposals

        roi_losses = self.roi_head.forward_train(rgbT_x, img_metas, rgbT_proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                rgb_img, lwir_img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        raise NotImplementedError

    def simple_test(self, rgb_img, lwir_img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        rgb_x, lwir_x, rgbT_x = self.extract_feat(rgb_img, lwir_img)
        if proposals is None:
            rgb_proposal_list = self.rgb_rpn_head.simple_test_rpn(rgb_x, img_metas)
            lwir_proposal_list = self.lwir_rpn_head.simple_test_rpn(lwir_x, img_metas)

            rgbT_proposal_list = self.merge_proposals(rgb_proposal_list, lwir_proposal_list)
        else:
            rgb_proposal_list = proposals
            lwir_proposal_list = proposals


        return self.roi_head.simple_test(
            rgbT_x, rgbT_proposal_list, img_metas, rescale=rescale)

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
            return self.forward_train(rgb_img, lwir_img, img_metas, **kwargs)
        else:
            return self.forward_test(rgb_img, lwir_img, img_metas, **kwargs)

    def onnx_export(self, rgb_img, lwir_img, img_metas):
        raise NotImplementedError(f'{self.__class__.__name__} does '
                                  f'not support ONNX EXPORT')



class ProbEn():

    def __init__(self):
        pass

    def bayesian_fusion(self, match_score_vec):
        log_positive_scores = np.log(match_score_vec)
        log_negative_scores = np.log(1 - match_score_vec)
        fused_positive = np.exp(np.sum(log_positive_scores))
        fused_negative = np.exp(np.sum(log_negative_scores))
        fused_positive_normalized = fused_positive / (fused_positive + fused_negative)
        return fused_positive_normalized

    def weighted_box_fusion(self, bbox, score):
        weight = score / np.sum(score)
        out_bbox = np.zeros(4)
        for i in range(len(score)):
            out_bbox += weight[i] * bbox[i]
        return out_bbox

    def nms_bayesian(self, dets, scores, thresh):
        dets = dets.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()

        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        match_scores = []
        match_bboxs = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            match = np.where(ovr > thresh)[0]
            match_ind = order[match + 1]

            match_score = list(scores[match_ind])
            match_bbox = list(dets[match_ind][:, :4])

            original_score = scores[i].tolist()
            original_bbox = dets[i][:4]

            # If some boxes are matched
            if len(match_score) > 0:
                match_score += [original_score]
                # Try with different fusion methods
                final_score = self.bayesian_fusion(np.asarray(match_score))
                match_bbox += [original_bbox]
                final_bbox = self.weighted_box_fusion(match_bbox, match_score)

                match_scores.append(final_score)
                match_bboxs.append(final_bbox)
            else:
                match_scores.append(original_score)
                match_bboxs.append(original_bbox)

            order = order[inds + 1]

        assert len(keep) == len(match_scores)
        assert len(keep) == len(match_bboxs)

        match_dets = np.hstack([np.array(match_bboxs), np.array(match_scores,).reshape(-1, 1)])
        match_dets = torch.from_numpy(match_dets.astype(np.float32))
        return match_dets

