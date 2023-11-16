from .base import BaseDetector
from ..builder import DETECTORS, build_backbone, build_head, build_neck
import warnings
from mmcv.runner import auto_fp16
import torch
import torch.nn.functional as F

@DETECTORS.register_module()
class SparseRCNN_RGBT(BaseDetector):
    r"""Implementation of `Sparse R-CNN: End-to-End Object Detection with
    Learnable Proposals <https://arxiv.org/abs/2011.12450>`_"""

    def __init__(self,
                 share_weights: [bool, dict],
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SparseRCNN_RGBT, self).__init__(init_cfg)

        if isinstance(share_weights, bool):
            self.share_weights = {}
            self.share_weights['backbone'] = share_weights
            self.share_weights['neck'] = share_weights

        elif isinstance(share_weights, dict):
            for _k in ['backbone', 'neck']:
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

        self.cosSim = torch.nn.CosineSimilarity(-1)

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

    def filter_boxes_features(self, rgb_proposal_boxes, rgb_proposal_features,
                              lwir_proposal_boxes, lwir_proposal_features):
        num_img, num_proposals, num_feature = rgb_proposal_features.shape
        cosSim_PF = self.cosSim(F.softmax(rgb_proposal_features, -1), F.softmax(lwir_proposal_features, -1))
        max_idxes = torch.topk(cosSim_PF, num_proposals//2, -1, )[-1]
        device = rgb_proposal_boxes.device

        proposal_boxes = torch.empty((num_img, num_proposals, 4), device=device)
        proposal_features = torch.empty((num_img, num_proposals, num_feature), device=device)
        for img_id in range(num_img):
            tmp_boxes = torch.cat([rgb_proposal_boxes[img_id][max_idxes[img_id]], lwir_proposal_boxes[img_id][max_idxes[img_id]]], 0)
            tmp_PF = torch.cat([rgb_proposal_features[img_id][max_idxes[img_id]], lwir_proposal_features[img_id][max_idxes[img_id]]], 0)
            proposal_boxes[img_id] = tmp_boxes
            proposal_features[img_id] = tmp_PF

        return proposal_boxes, proposal_features

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
        """Forward function of SparseR-CNN and QueryInst in train stage.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (List[Tensor], optional) : Segmentation masks for
                each box. This is required to train QueryInst.
            proposals (List[Tensor], optional): override rpn proposals with
                custom proposals. Use when `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        assert proposals is None, 'Sparse R-CNN and QueryInst ' \
            'do not support external proposals'

        rgb_x, lwir_x, rgbT_x = self.extract_feat(rgb_img, lwir_img)

        rgb_proposal_boxes, rgb_proposal_features, imgs_whwh = \
            self.rgb_rpn_head.forward_train(rgb_x, img_metas)

        lwir_proposal_boxes, lwir_proposal_features, lwir_imgs_whwh = \
            self.lwir_rpn_head.forward_train(lwir_x, img_metas)

        assert (imgs_whwh == lwir_imgs_whwh).all()
        proposal_boxes, proposal_features = self.filter_boxes_features(rgb_proposal_boxes, rgb_proposal_features,
                                                                       lwir_proposal_boxes, lwir_proposal_features)
        roi_losses = self.roi_head.forward_train(
            rgbT_x,
            proposal_boxes,
            proposal_features,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_masks=gt_masks,
            imgs_whwh=imgs_whwh)
        return roi_losses

    def simple_test(self, rgb_img, lwir_img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        rgb_x, lwir_x, rgbT_x = self.extract_feat(rgb_img, lwir_img)
        rgb_proposal_boxes, rgb_proposal_features, imgs_whwh = \
                self.rgb_rpn_head.simple_test_rpn(rgb_x, img_metas)
        lwir_proposal_boxes, lwir_proposal_features, lwir_imgs_whwh = \
                self.lwir_rpn_head.simple_test_rpn(lwir_x, img_metas)
        assert (imgs_whwh == lwir_imgs_whwh).all()
        proposal_boxes, proposal_features = self.filter_boxes_features(rgb_proposal_boxes, rgb_proposal_features,
                                                                       lwir_proposal_boxes, lwir_proposal_features)
        results = self.roi_head.simple_test(
            rgbT_x,
            proposal_boxes,
            proposal_features,
            img_metas,
            imgs_whwh=imgs_whwh,
            rescale=rescale)
        return results


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