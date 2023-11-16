_base_ = './faster_rcnn_r50_fpn_sanitized-kaist.py'

data = dict(samples_per_gpu=3)

model = dict(
    type='FasterRCNN_RGBTwMask',

    backbone=dict(
        type='ResNet_wo_MaxPool',
        depth=50,
        frozen_stages=-1,
        out_indices=(1, 2, 3),
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet50')),

    neck=dict(
        in_channels=[512, 1024, 2048],
        num_outs=4),

    rpn_head=dict(
        anchor_generator=dict(strides=[4, 8, 16, 32]),
    ),

    roi_head=dict(
        bbox_roi_extractor=dict(featmap_strides=[4, 8, 16, 32])),

    test_cfg=dict(
        rcnn=dict(
            score_thr=0.01,  #default 0.1
            )),
    )