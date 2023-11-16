import os.path

_base_ = [
    # '../_base_/models/faster_rcnn_r50_fpn.py',
    # '../_base_/datasets/coco_detection.py',
    '../_base_/default_runtime.py']

# 因为我是按照lwir图片的文件名索引的rgb的图片文件名。所以：
# 当读取一对rgbt图像时，modality应该是lwir。
# 当读取一张图像是，modality可以是lwir或者visible，但是要注意，将pipeline修改成单图的pipeline

modality = 'lwir'
load_RGBT = True

data_root = '/home/zx/cross-modality-det/datasets/LLVIP/LLVIP/coco_format'
assert os.path.isdir(data_root)

# load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

base_lr = 1e-2
bs = 3    # bs=3x4与bs=10差不多,但是bs=3x4速度是bs=10的3倍左右.

# `img_scale` can either be a tuple (single-scale) or a list of tuple (multi-scale), There are 3 multiscale modes.
# Please see my_load_rgbt_pipeline.ResizeRGBT for more details.
img_scale = (1280, 1024)    # Target size (w, h)

lwir_mean = [74.319, 74.319, 74.319]
lwir_std = [48.458, 48.458, 48.458]

rgb_mean = [50.059, 48.529, 31.771]
rgb_std = [49.817, 49.766, 46.235]
classes = ['person']  # mmdet/datasets/coco.py 171行显示，需要将所有类别传入，否则将忽略其它类别的标签

norm_cfg = dict(type='SyncBN', requires_grad=True)
# norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='FasterRCNN_RGBTwMask_wSpaAttV5',
    share_weights=dict(backbone=True),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        norm_cfg=norm_cfg,
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),

    taf_cfg=dict(
        in_channels=[512, 1024, 2048],
        out_channels=256,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        dice_weight=2.0,    # best 2.0
        neg_entropy_weight=1.0,),   # best 1.0

    neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        num_outs=4),

    # RPNHead
    rpn_head=dict(
        type='RPNHead', # RPNHead前向：(FPN feats)->rpn_conv(in_c, feat_c)->relu->rpn_cls. rpn_cls:conv(feat_c, 3*1, 1)
        #                                                                     \->rpn_reg. rpn_rgb:conv(feat_c, 3*4, 1)
        in_channels=256,
        feat_channels=256,  # feat_channels表示RPNHead前向传播时rpn_conv的输出通道数
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8], # default 8
            ratios=[1.701, 2.0, 2.406],# 因为只有一个person类别，msds-rcnn设置anchor_ratio=0.41
            strides=[8, 16, 32, 64],  # 在不同的FPN尺度上生成multi_level_anchors, 每一尺度面积为 (scales * strides)^2
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), # use_sigmoid=True, rpn_cls的cls_out_channels=1
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
    ),

    # StandardRoIHead_RGBT
    roi_head=dict(
        type='StandardRoIHead',
        # bbox_roi_extractor 表示基于Proposals提取Features中相对应的特征
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8, 16, 32]
        ),
        # bbox_head 表示基于上述ROI特征进行预测
        bbox_head=dict(
            # Shared2FCBBoxHead 的结构为            /->cls: fc(1024, n_cls+1)
            # roi->fc(256*7*7, 1024)->fc(1024, 1024)
            #                                      \->reg: fc(1024, (4 if reg_class_agnostic else 4 * self.num_classes))
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=len(classes),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',  #  encodes bbox (x1, y1, x2, y2) into delta (dx, dy, dw, dh)
                                            # and decodes delta (dx, dy, dw, dh) back to original bbox (x1, y1, x2, y2).
                target_means=[0.0, 0.0, 0.0, 0.0],  # 与FasterRCNN论文一致，将anchor与gt的坐标变换结果减去均值，除以方差。
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,   # default False, 表示每个类别都回归各自对应的框
            # loss_cls=dict(
            #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            # loss_bbox=dict(type='L1Loss', loss_weight=1.0)

            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=5.0),

            reg_decoded_bbox=True,  # 对于 `IouLoss`, `GIouLoss`,回归损失是直接回归原始框的，因此要reg_decoded_bbox=True
            # 具体说，在调用get_target时，会将anchors用bbox_coder编码成与gt的相对变换。GIOULoss需要回归原始Anchors，因此要decode
            loss_bbox=dict(type='CIoULoss', loss_weight=20.0)
        )),

    train_cfg=dict(
        rpn=dict(
            # assign positive / negative anchors
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,    # iou(gt, anchor)>0.7的anchor定义为正样本，分配大于1的标签
                neg_iou_thr=0.3,    # iou(gt, anchor)<0.3的anchor定义为负样本，分配标签0
                min_pos_iou=0.3,    # 0.3<=iou(gt, anchor)<0.7的anchor定义为-1，忽略它们，在采样时不采样它们。
                match_low_quality=True,
                ignore_iof_thr=-1), # default -1
                # ignore_iof_thr=0.3), # 当定义了ignore gt bbox时，需要指定忽略bbox与ignore gt之间的iof阈值. iof表示interaction over foreground
            # sample targets from assigned anchors
            sampler=dict(
                type='RandomSampler',   # Sampler控制了正负样本比例，有效避免了class imbalance
                num=256,    # default 256. Number of samples。从一张图的所有FPN层Anchor中采样256个样本
                pos_fraction=0.5,   # Fraction of positive samples。
                                    # sampled positive anchors个数 = min(positive anshors, 128)
                                    # negative anchors个数 = 256-sampled postive anchors
                neg_pos_ub=-1,  # Upper bound number of negative and positive samples.
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,  # default -1. if pos_weight<=0: label_weights[pos_inds]=1; else: label_weights[pos_inds]=pos_weight
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
                # ignore_iof_thr=0.5),
            sampler=dict(
                type='RandomSampler',
                num=512, # default 512
                pos_fraction=0.25,   # default 0.25
                neg_pos_ub=-1,
                add_gt_as_proposals=True    # add_gt_as_proposals可以保证ground-truth作为 positive proposals
            ),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,       # default 1000
            max_per_img=1000,   # default 1000
            nms=dict(type='nms', iou_threshold=0.7),    # default 0.7
            min_bbox_size=0),  # default 0
        rcnn=dict(
            score_thr=0.05,  # default 0.1
            nms=dict(type='nms', iou_threshold=0.5),   # best 0.45
            max_per_img=100)),  # default=100
)


dataset_type = 'CocoDataset'

train_pipeline = [
        dict(type='LoadRGBTFromFile'),
        dict(type='LoadRGBTAnnotations', with_bbox=True, with_mask=True),
        dict(type='ResizeRGBT', img_scale=img_scale, keep_ratio=True),
        dict(type='RandomFlipRGBT', flip_ratio=0.5),
        dict(
            type='NormalizeRGBT',
            rgb_mean=rgb_mean,
            rgb_std=rgb_std,
            lwir_mean=lwir_mean,
            lwir_std=lwir_std,
            to_rgb=True),   # to_rgb：先从bgr转换成rgb，再归一化
        dict(type='PadRGBT', size_divisor=32),
        dict(type='DefaultFormatBundleRGBT'),
        dict(type='CollectRGBT', keys=['rgb_img', 'lwir_img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        # dict(type='CollectRGBT', keys=['rgb_img', 'lwir_img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_masks'])
    ]

test_pipeline = [
    dict(type='LoadRGBTFromFile'),
    dict(
        type='MultiScaleFlipAugRGBT',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='ResizeRGBT', keep_ratio=True),
            dict(type='RandomFlipRGBT'),
            dict(
                type='NormalizeRGBT',
                rgb_mean=rgb_mean,
                rgb_std=rgb_std,
                lwir_mean=lwir_mean,
                lwir_std=lwir_std,
                to_rgb=True),
            dict(type='PadRGBT', size_divisor=32),
            dict(type='RGBTImageToTensor', keys=['lwir_img', 'rgb_img']),
            dict(type='CollectRGBT', keys=['lwir_img', 'rgb_img'])
        ])
]

data = dict(
    samples_per_gpu=bs,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=f'{data_root}/{modality}_train.json',
        classes=classes,
        img_prefix='',
        pipeline=train_pipeline,
        filter_empty_gt=False),
    val=dict(
        type=dataset_type,
        ann_file=f'{data_root}/{modality}_test.json',
        classes=classes,
        img_prefix='',
        pipeline=test_pipeline,
        filter_empty_gt=False),
    test=dict(
        type=dataset_type,
        ann_file=f'{data_root}/{modality}_test.json',
        classes=classes,
        img_prefix='',
        pipeline=test_pipeline,
        filter_empty_gt=False))
evaluation = dict(interval=1, metric='bbox', classwise=True)

optimizer = dict(type='SGD', lr=base_lr, momentum=0.9, weight_decay=0.0005)

optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])

runner = dict(type='EpochBasedRunner', max_epochs=12)