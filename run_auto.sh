#!/bin/bash

PYTHON_EXEC="python"
echo "Using python: $(which python)"

# --- 配置区 ---
# 请确保数据路径正确
DATA_PATH="/home/joey1/Documents/joey/Data/ASCAD/ascadv2-extracted.h5"

# 输出文件夹 (干净的新目录)
CKP_DIR="./checkpoints_auto"
RESULT_PATH="./results_auto"

# 清理旧数据 (既然你已经备份了 failure，这里我们强制清理，保证从零开始)
echo "🧹 Cleaning up old auto results..."
rm -rf ${CKP_DIR} ${RESULT_PATH}
mkdir -p ${CKP_DIR} ${RESULT_PATH}

# --- 🌟 核心参数设置 🌟 ---

# 方案 A: 新服务器 (GPU-6000ada) 推荐配置
# 理由: 之前跑出了 Rank 3，6e-5 动能完美
# LEARNING_RATE=6e-5

# 方案 B: 旧服务器 (NiCE-DES) 推荐配置
# 理由: 之前跑出了 Rank 49，5e-5 是这里的极限
# 如果在旧服务器跑，请取消下面这行的注释，并注释掉上面那行
LEARNING_RATE=5e-5

echo "=========================================================="
echo "🚀 Starting AUTO-PILOT Training"
echo "   Initial LR: ${LEARNING_RATE}"
echo "   Strategy:   1 Epoch Warmup -> Hold -> Decay at Rank < 20"
echo "=========================================================="

$PYTHON_EXEC train_auto.py \
    --data_path "${DATA_PATH}" \
    --checkpoint_dir "${CKP_DIR}" \
    --result_path "${RESULT_PATH}" \
    --learning_rate ${LEARNING_RATE} \
    --input_length 15000 \
    --train_batch_size 64 \
    --train_steps 400000 \
    --d_model 128 \
    --n_layer 2 \
    --n_head 8