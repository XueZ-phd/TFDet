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

# server configuration
server = '3090'
# check server
assert server in ['2070', '3060', '3090']
if server == '3060':
    data_root = '/UsrFile/Usr3/zx/KAIST-Sanitized/coco_format'
elif server == '2070':
    data_root = '/home/ivlab/new_home/zx/cross-modality-det/datasets/KAIST-Sanitized/coco_format'
elif server == '3090':
    data_root = '/home/zx/cross-modality-det/datasets/coco_format'
assert os.path.isdir(data_root)

# load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

base_lr = 1e-2
# base_lr = 3e-3
# base_lr = 5e-3 # UniDif_FasterRCNN_RGBT lr=1e-2时，loss=nan

# `img_scale` can either be a tuple (single-scale) or a list of tuple (multi-scale), There are 3 multiscale modes.
# Please see my_load_rgbt_pipeline.ResizeRGBT for more details.

bs = 3    # bs=3x4与bs=10差不多,但是bs=3x4速度是bs=10的3倍左右.
# bs = 1
# bs = 4
img_scale = (640, 512)

lwir_mean = [44.712, 42.793, 44.712]
lwir_std = [28.165, 28.041, 28.165]     # [28.165, 28.041, 28.165]

rgb_mean = [91.392, 84.984, 75.076]
rgb_std = [67.841, 64.853, 65.535]  # [67.841, 64.853, 65.535]
classes = ['person', 'people', 'person?', 'cyclist', 'person?a']  # mmdet/datasets/coco.py 171行显示，需要将所有类别传入，否则将忽略其它类别的标签

norm_cfg = dict(type='SyncBN', requires_grad=True)
# norm_cfg = dict(type='BN', requires_grad=True)


model = dict(
    type='RetinaNet_RGBT',
    share_weights=dict(backbone=True),
    backbone=dict(
        type='SSDVGG',
        depth=16,
        with_last_pool=False,
        ceil_mode=True,
        out_indices=(3, 4),
        out_feature_indices=(22, 34),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://vgg16_caffe')),

    neck=dict(
        type='FPN',
        in_channels=[512, 1024],
        out_channels=256,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        num_outs=2),

    bbox_head=dict(
        type='RetinaHead',
        num_classes=len(classes),
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=8,    # default 4
            scales_per_octave=3,
            ratios=[1.701, 2.0, 2.406],
            strides=[8, 16]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=0.1),    # default -1
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,   # default 1000
        min_bbox_size=0,
        score_thr=0.00, # default 0.05
        nms=dict(type='nms', iou_threshold=0.45),   # default 0.5
        max_per_img=100))

dataset_type = 'CocoDataset'

train_pipeline = [
        dict(type='LoadRGBTFromFile'),
        dict(type='LoadRGBTAnnotations', with_bbox=True, with_mask=True),
        dict(type='FillRGBTIgnoredBoxRegionsByMeanValues', rgb_mean=rgb_mean, lwir_mean=lwir_mean),
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
        dict(type='CollectRGBT', keys=['rgb_img', 'lwir_img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_masks'])
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
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=f'{data_root}/{modality}_test.json',
        classes=classes,
        img_prefix='',
        pipeline=test_pipeline))

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

checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
