_base_ = 'faster_rcnn_r50_fpn_sanitized-kaist.py'

bs = 1
data = dict(
    samples_per_gpu=bs,)

model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        #norm_cfg=dict(type='BN', requires_grad=True),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_32x4d')))