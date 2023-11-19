#!/usr/bin/env bash

CONFIG='configs/faster_rcnn/faster_rcnn_r50_fpn_dcnGWConvGloballCC_llvip_640x512.py'
ID='7'
CHECKPOINT="runs_llvip/FasterRCNN_r50wMask_ROIFocalLoss5_CIOU20_cosineSE_dcnGWConvGlobalCC_640x512/epoch_$ID.pth"
OUT_PATH="runs_llvip/FasterRCNN_r50wMask_ROIFocalLoss5_CIOU20_cosineSE_dcnGWConvGlobalCC_640x512/epoch_$ID.pkl"
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


