#!/usr/bin/env bash

set -x

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29531}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS --master_port=$PORT $(dirname "$0")/v_buffer.py $CONFIG --launcher pytorch ${@:3}
# Any arguments from the third one are captured by ${@:3}
