load_RGBT = True
base_lr = 1e-2

# `img_scale` can either be a tuple (single-scale) or a list of tuple (multi-scale), There are 3 multiscale modes.
# Please see my_load_rgbt_pipeline.ResizeRGBT for more details.

bs = 3
img_scale = (640, 512)

lwir_mean = [44.712, 42.793, 44.712]
lwir_std = [28.165, 28.041, 28.165]     # [28.165, 28.041, 28.165]

rgb_mean = [91.392, 84.984, 75.076]
rgb_std = [67.841, 64.853, 65.535]  # [67.841, 64.853, 65.535]

classes = ['person', 'people', 'person?', 'cyclist', 'person?a']  # mmdet/datasets/coco.py 171行显示，需要将所有类别传入，否则将忽略其它类别的标签
data_root = '/home/ivlab/new_home/zx/cross-modality-det/datasets/KAIST-Sanitized/coco_format'

model = dict(
    type='FCOS_RGBT',
    share_weights=dict(backbone=True),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,  # 从in_channels[start_level]开始做FPN
        add_extra_convs='on_output',    # 在512维度上加额外的卷积层
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=len(classes),
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        center_sampling=True,
        conv_bias=True),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100)
)

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
        to_rgb=True),  # to_rgb：先从bgr转换成rgb，再归一化
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
        ann_file=f'{data_root}/lwir_train.json',
        classes=classes,
        img_prefix='',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=f'{data_root}/lwir_test.json',
        classes=classes,
        img_prefix='',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=f'{data_root}/lwir_test.json',
        classes=classes,
        img_prefix='',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox', classwise=True)

optimizer = dict(
    type='SGD',
    lr=base_lr,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(bias_lr_mult=2.0, bias_decay_mult=0.0))

optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
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
