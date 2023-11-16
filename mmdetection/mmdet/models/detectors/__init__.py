# Copyright (c) OpenMMLab. All rights reserved.
from .atss import ATSS
from .autoassign import AutoAssign
from .base import BaseDetector
from .cascade_rcnn import CascadeRCNN
from .centernet import CenterNet
from .cornernet import CornerNet
from .deformable_detr import DeformableDETR
from .detr import DETR
from .fast_rcnn import FastRCNN
from .faster_rcnn import FasterRCNN
from .fcos import FCOS
from .fovea import FOVEA
from .fsaf import FSAF
from .gfl import GFL
from .grid_rcnn import GridRCNN
from .htc import HybridTaskCascade
from .kd_one_stage import KnowledgeDistillationSingleStageDetector
from .lad import LAD
from .mask2former import Mask2Former
from .mask_rcnn import MaskRCNN
from .mask_scoring_rcnn import MaskScoringRCNN
from .maskformer import MaskFormer
from .nasfcos import NASFCOS
from .paa import PAA
from .panoptic_fpn import PanopticFPN
from .panoptic_two_stage_segmentor import TwoStagePanopticSegmentor
from .point_rend import PointRend
from .queryinst import QueryInst
from .reppoints_detector import RepPointsDetector
from .retinanet import RetinaNet
from .rpn import RPN
from .scnet import SCNet
from .single_stage import SingleStageDetector
from .solo import SOLO
from .sparse_rcnn import SparseRCNN
from .tood import TOOD
from .trident_faster_rcnn import TridentFasterRCNN
from .two_stage import TwoStageDetector
from .vfnet import VFNet
from .yolact import YOLACT
from .yolo import YOLOV3
from .yolof import YOLOF
from .yolox import YOLOX

from .my_faster_rcnn import FasterRCNN_RGBT
# from .detr_RGBT import DETR_RGBT
# from .sparse_rcnn_RGBT import SparseRCNN_RGBT
# from .retinanet_RGBT import RetinaNet_RGBT
# from .fcos_myRGBT import FCOS_RGBT
# from .my_faster_rcnn_mask_supervised import FasterRCNN_RGBTwMask
from .my_faster_rcnn_mask_supervised_spatialAttention import FasterRCNN_RGBTwMask_wSpaAtt
from .my_faster_rcnn_mask_supervised_spatialAttention_v2 import FasterRCNN_RGBTwMask_wSpaAttV2
from .my_faster_rcnn_mask_supervised_spatialAttention_v3 import FasterRCNN_RGBTwMask_wSpaAttV3
from .my_faster_rcnn_mask_supervised_spatialAttention_v2_llvip import FasterRCNN_RGBTwMask_wSpaAttV2_LLVIP
from .my_faster_rcnn_mask_supervised_spatialAttention_v2_onlyFFM import FasterRCNN_RGBTwMask_wSpaAttV2_LLVIP_onlyFFM
from .my_faster_rcnn_mask_supervised_spatialAttention_v2_FFMxMask_noFRM_noLnegentropy import FasterRCNN_RGBTwMask_wSpaAttV2_FFMxMask
from .my_faster_rcnn_mask_supervised_spatialAttention_v2_cvc14 import FasterRCNN_RGBTwMask_wSpaAttV2_CVC14
from .my_faster_rcnn_mask_supervised_spatialAttention_v2_dicebceLovasz_Sigmoid import FasterRCNN_RGBTwMask_wSpaAttV2_Lovasz_Sigmoid
from .my_faster_rcnn_mask_supervised_spatialAttention_v2_replaceFRMwithCBAM import FasterRCNN_RGBTwMask_wSpaAttV2_replaceFRMwithCBAM
from .my_faster_rcnn_mask_supervised_spatialAttention_v2_replaceFFMwithCatConv import FasterRCNN_RGBTwMask_wSpaAttV2_replaceFFMwithCatConv
from .yolox_RGBT import YOLOX_RGBT
from .my_faster_rcnn_mask_supervised_spatialAttention_v4 import FasterRCNN_RGBTwMask_wSpaAttV4
from .my_faster_rcnn_mask_supervised_spatialAttention_v5 import FasterRCNN_RGBTwMask_wSpaAttV5
# from .cascade_rcnn_RGBT import CascadeRCNN_RGBT

__all__ = [
    'ATSS', 'BaseDetector', 'SingleStageDetector', 'TwoStageDetector', 'RPN',
    'KnowledgeDistillationSingleStageDetector', 'FastRCNN', 'FasterRCNN',
    'MaskRCNN', 'CascadeRCNN', 'HybridTaskCascade', 'RetinaNet', 'FCOS',
    'GridRCNN', 'MaskScoringRCNN', 'RepPointsDetector', 'FOVEA', 'FSAF',
    'NASFCOS', 'PointRend', 'GFL', 'CornerNet', 'PAA', 'YOLOV3', 'YOLACT',
    'VFNet', 'DETR', 'TridentFasterRCNN', 'SparseRCNN', 'SCNet', 'SOLO',
    'DeformableDETR', 'AutoAssign', 'YOLOF', 'CenterNet', 'YOLOX',
    'TwoStagePanopticSegmentor', 'PanopticFPN', 'QueryInst', 'LAD', 'TOOD',
    'MaskFormer', 'Mask2Former',

    'FasterRCNN_RGBT',
    # 'FasterRCNN_RGBTwMask', 'DETR_RGBT', 'SparseRCNN_RGBT',
    # 'RetinaNet_RGBT', 'FCOS_RGBT',
]
