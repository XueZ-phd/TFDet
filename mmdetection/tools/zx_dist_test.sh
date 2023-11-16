#!/usr/bin/env bash

CONFIG='configs/faster_rcnn/faster_rcnn_vgg16_fpn_sanitized-kaist_v3.py'  # vgg16
ID='9'
CHECKPOINT="runs/FasterRCNN_vgg16_w_mask_SpaAttV2_ROIFocalLoss5_CIOU20_cosineSE_notDetach_negEntropy1/epoch_$ID.pth"
OUT_PATH="runs/FasterRCNN_vgg16_w_mask_SpaAttV2_ROIFocalLoss5_CIOU20_cosineSE_notDetach_negEntropy1/epoch_$ID.pkl"
GPUS=$1
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    --out $OUT_PATH \
    ${@:2}


