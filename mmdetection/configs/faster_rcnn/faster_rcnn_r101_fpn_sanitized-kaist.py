_base_ = 'faster_rcnn_r50_fpn_sanitized-kaist.py'

bs = 2
data = dict(
    samples_per_gpu=bs,)

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))