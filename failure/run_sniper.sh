#!/bin/bash
# 配置
DATA_PATH="/home/joey1/Documents/joey/Data/ASCAD/ascadv2-extracted.h5"
CKP_DIR="./checkpoints_final"
RESULT_PATH="./results_final"

# 🌟 核心修改：锁定黄金学习率 1e-5
LEARNING_RATE=1e-5

# 清理旧目录
rm -rf ${CKP_DIR} ${RESULT_PATH}
mkdir -p ${CKP_DIR} ${RESULT_PATH}

echo "Starting FINAL Strategy: Constant 1e-5..."

# 我们使用普通的 train.py 即可，但要确保 warmup 为 0
# 如果你的 train.py 接收 warmup_steps 参数：
python train.py \
    --data_path "${DATA_PATH}" \
    --checkpoint_dir "${CKP_DIR}" \
    --result_path "${RESULT_PATH}" \
    --learning_rate ${LEARNING_RATE} \
    --input_length 15000 \
    --train_batch_size 64 \
    --train_steps 200000 \
    --warmup_steps 0 \
    --warm_start False