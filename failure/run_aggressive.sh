#!/bin/bash
# 配置
DATA_PATH="/home/joey1/Documents/joey/Data/ASCAD/ascadv2-extracted.h5"
CKP_DIR="./checkpoints_aggressive"
RESULT_PATH="./results_aggressive"

# 🌟 激进学习率: 3e-4 (比 1e-4 大3倍，比 1e-3 安全)
# 如果你一定要试 1e-3，就在这里改，但我强烈建议先试 3e-4
LEARNING_RATE=3e-4 

# 清理旧数据
rm -rf ${CKP_DIR} ${RESULT_PATH}
mkdir -p ${CKP_DIR} ${RESULT_PATH}

echo "Starting AGGRESSIVE Training..."
echo "Initial LR: ${LEARNING_RATE}"

# 启动 (注意文件名是 train_aggressive.py)
python train_aggressive.py \
    --data_path "${DATA_PATH}" \
    --checkpoint_dir "${CKP_DIR}" \
    --result_path "${RESULT_PATH}" \
    --learning_rate ${LEARNING_RATE} \
    --input_length 15000 \
    --train_batch_size 64 \
    --train_steps 400000