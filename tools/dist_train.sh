#!/usr/bin/env bash
# DETR3D传入config_path,gpus,port为默认
CONFIG=$1
GPUS=$2
PORT=${PORT:-28500}
# more details about distributed.launch:https://blog.csdn.net/magic_ll/article/details/122359490
# 这里的distributed为单机多卡训练模式，需要指定gpus,port,train.py,如果是多机多卡必须要指定节点个数与rank等参数
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
