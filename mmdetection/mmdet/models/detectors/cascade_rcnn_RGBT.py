# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .my_faster_rcnn_mask_supervised_spatialAttention import FasterRCNN_RGBTwMask_wSpaAtt
from .my_faster_rcnn_mask_supervised_spatialAttention import DiceBCELoss
import torch.nn as nn


@DETECTORS.register_module()
class CascadeRCNN_RGBT(FasterRCNN_RGBTwMask_wSpaAtt):
    r"""Implementation of `Cascade R-CNN: Delving into High Quality Object
    Detection <https://arxiv.org/abs/1906.09756>`_"""

    def __init__(self,
                 share_weights,
                 backbone,
                 dice_weight,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CascadeRCNN_RGBT, self).__init__(
            share_weights,
            backbone=backbone,
            dice_weight=dice_weight,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def show_result(self, data, result, **kwargs):
        """Show prediction results of the detector.

        Args:
            data (str or np.ndarray): Image filename or loaded image.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).

        Returns:
            np.ndarray: The image with bboxes drawn on it.
        """
        if self.with_mask:
            ms_bbox_result, ms_segm_result = result
            if isinstance(ms_bbox_result, dict):
                result = (ms_bbox_result['ensemble'],
                          ms_segm_result['ensemble'])
        else:
            if isinstance(result, dict):
                result = result['ensemble']
        return super(CascadeRCNN_RGBT, self).show_result(data, result, **kwargs)
