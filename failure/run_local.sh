#!/bin/bash

# --- 1. 配置区 (已适配旧服务器路径) ---
# 数据路径 (根据你刚才evaluate命令里的路径填写的)
DATA_PATH="/home/joey1/Documents/joey/Data/ASCAD/ascadv2-extracted.h5"
CKP_DIR="./checkpoints_ascadv2"
RESULT_PATH="./results_ascadv2"

# 关键训练参数 (配合Warmup使用的最佳配置)
LEARNING_RATE=1e-5
TRAIN_STEPS=400000

# --- 2. 准备工作 ---
# 激活环境 (确保后台运行时环境正确)
# 假设你的conda安装在默认位置，如果报错找不到conda，可以注释掉这行，
# 但前提是你运行nohup前已经在终端激活了 (estranet) 环境
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate estranet

echo "Cleaning previous results..."
rm -rf ${CKP_DIR} ${RESULT_PATH}
mkdir -p ${CKP_DIR} ${RESULT_PATH}

echo "Starting training on $(hostname)..."
echo "Data: ${DATA_PATH}"
echo "LR: ${LEARNING_RATE}"

# --- 3. 启动命令 ---
python train.py \
    --data_path "${DATA_PATH}" \
    --checkpoint_dir "${CKP_DIR}" \
    --result_path "${RESULT_PATH}" \
    --learning_rate ${LEARNING_RATE} \
    --input_length 15000 \
    --train_batch_size 64 \
    --train_steps ${TRAIN_STEPS} \
    --n_layer 2 \
    --d_model 128 \
    --n_head 8 \
    --warm_start False