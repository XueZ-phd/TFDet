import os
_base_ = [
    '../_base_/datasets/coco_detection.py', '../_base_/default_runtime.py'
]
modality = 'lwir'
load_RGBT = True

# server configuration
server = '2070'
# check server
assert server in ['2070', '3060', 'RTX8000']
if server == '3060':
    data_root = '/UsrFile/Usr3/zx/KAIST-Sanitized/coco_format'
elif server == '2070':
    data_root = '/home/ivlab/new_home/zx/cross-modality-det/datasets/KAIST-Sanitized/coco_format'
elif server == 'RTX8000':
    data_root = None
assert os.path.isdir(data_root)

img_scale = (640, 512)
bs = 4
lwir_mean = [44.712, 42.793, 44.712]
lwir_std = [28.165, 28.041, 28.165]

rgb_mean = [91.392, 84.984, 75.076]
rgb_std = [67.841, 64.853, 65.535]
classes = ['person', 'people', 'person?', 'cyclist']

model = dict(
    type='DETR_RGBT',
    share_weights=dict(backbone=True),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    bbox_head=dict(
        type='DETRHead',
        num_classes=len(classes),
        in_channels=2048,
        transformer=dict(
            type='Transformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1)
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DetrTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1),
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        loss_cls=dict(
            type='CrossEntropyLoss',
            bg_cls_weight=0.1,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=1.),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
    test_cfg=dict(max_per_img=100))

img_norm_cfg = {}

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

data = dict(
    samples_per_gpu=bs,
    workers_per_gpu=8,
    train=dict(
        ann_file=f'{data_root}/{modality}_train.json',
        classes=classes,
        img_prefix='',
        pipeline=train_pipeline),
    val=dict(
        ann_file=f'{data_root}/{modality}_test.json',
        classes=classes,
        img_prefix='',
        pipeline=test_pipeline),
    test=dict(
        ann_file=f'{data_root}/{modality}_test.json',
        classes=classes,
        img_prefix='',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[100])
runner = dict(type='EpochBasedRunner', max_epochs=150)
