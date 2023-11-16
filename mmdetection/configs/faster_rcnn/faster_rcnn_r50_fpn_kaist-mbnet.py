_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py']

# 因为我是按照lwir图片的文件名索引的rgb的图片文件名。所以：
# 当读取一对rgbt图像时，modality应该是lwir。
# 当读取一张图像是，modality可以是lwir或者visible，但是要注意，将pipeline修改成单图的pipeline

modality = 'lwir'
load_RGBT = True

#load_from = '../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
# load_from = None

base_lr = 1e-2

# `img_scale` can either be a tuple (single-scale) or a list of tuple (multi-scale), There are 3 multiscale modes.
# Please see my_load_rgbt_pipeline.ResizeRGBT for more details.

bs = 8
img_scale = (640, 512)

lwir_mean = [44.712, 42.793, 44.712]
lwir_std = [28.165, 28.041, 28.165]

rgb_mean = [91.392, 84.984, 75.076]
rgb_std = [67.841, 64.853, 65.535]

model = dict(
    type='FasterRCNN_RGBT',
    share_weights=dict(backbone=True, neck=True, dense_head=False),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),

    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5
    ),

    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],# 高宽比
            strides=[4, 8, 16, 32, 64],
        ),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),

    # StandardRoIHead_RGBT
    roi_head=dict(
        type='StandardRoIHead',
        bbox_head=dict(
            num_classes=1,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)
        )),

    l2_norm_scale=None,
    )

data_root = '/UsrFile/Usr3/zx/KAIST-MBNet/coco_format'
data_type = 'CocoDataset'
classes = ('person',)

if not load_RGBT:
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', img_scale=img_scale, keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='Normalize',
            mean=lwir_mean,
            std=lwir_std,
            to_rgb=True),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ]

    test_pipeline = [
        dict(type='LoadFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=img_scale,
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(
                    type='Normalize',
                    mean=lwir_mean,
                    std=lwir_std,
                    to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ])
    ]
else:
    train_pipeline = [
        dict(type='LoadRGBTFromFile'),
        dict(type='LoadRGBTAnnotations', with_bbox=True),
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
        dict(type='CollectRGBT', keys=['lwir_img', 'rgb_img', 'gt_bboxes', 'gt_labels'])
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

optimizer = dict(type='SGD', lr=base_lr, momentum=0.9, weight_decay=0.0001)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=300,
    warmup_ratio=5e-4,
    step=[4, 6])
runner = dict(type='EpochBasedRunner', max_epochs=8)

data = dict(
    samples_per_gpu=bs,
    workers_per_gpu=8,
    train=dict(
        ann_file=f'{data_root}/{modality}_train.json',
        classes=classes,
        img_prefix='',
        pipeline=train_pipeline
    ),
    val=dict(
        ann_file=f'{data_root}/{modality}_test.json',
        classes=classes,
        img_prefix='',
        pipeline=test_pipeline
    ),
    test=dict(
        ann_file=f'{data_root}/{modality}_test.json',
        classes=classes,
        img_prefix='',
        pipeline=test_pipeline
    )
)
