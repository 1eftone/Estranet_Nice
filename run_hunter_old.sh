#!/bin/bash

PYTHON_EXEC="python"
echo "Using python: $(which python)"

# --- 📁 路径配置 ---
DATA_PATH="/home/joey1/Documents/joey/Data/ASCAD/ascadv2-extracted.h5"

# --- 🔥 关键修改 1: 改名！保护之前的 Rank 6 模型 ---
# 我们把这次的新实验命名为 hunter_adam，避免误删 hunter_old 里的好模型
CKP_DIR="./checkpoints_hunter_adam"
RESULT_PATH="./results_hunter_adam"

# --- 🧹 自动清理区 ---
echo "🧹 Cleaning up ADAM results to start FRESH..."

# 只清理 _adam 的文件夹，不碰 _old
if [ -d "${CKP_DIR}" ]; then
    echo "   Removing existing ${CKP_DIR}..."
    rm -rf ${CKP_DIR}
fi

if [ -d "${RESULT_PATH}" ]; then
    echo "   Removing existing ${RESULT_PATH}..."
    rm -rf ${RESULT_PATH}
fi

mkdir -p ${CKP_DIR} ${RESULT_PATH}

# --- 🌟 核心参数修改 ---

# 🔥 修改 2: 学习率调整
# AdamW 需要比 SCOOP/SGD 更大的学习率。
# 1e-4 是 Transformer/ResNet 配合 Adam 的黄金标准。
LEARNING_RATE=1e-4

echo "=========================================================="
echo "🚀 Starting STRATEGY: AdamW + SCOOP Hybrid"
echo "   Target Script: train_hunter_old.py (Updated Code)"
echo "   Optimizer:     AdamW (Adaptive Step)"
echo "   Initial LR:    ${LEARNING_RATE}"
echo "   Clip Norm:     1.0 (Tighter constraint for Adam)"
echo "=========================================================="

# 启动训练
$PYTHON_EXEC train_hunter_old.py \
    --data_path "${DATA_PATH}" \
    --checkpoint_dir "${CKP_DIR}" \
    --result_path "${RESULT_PATH}" \
    --learning_rate ${LEARNING_RATE} \
    --input_length 15000 \
    --train_batch_size 64 \
    --train_steps 400000 \
    --d_model 128 \
    --n_layer 2 \
    --n_head 8 \
    --clip 1.0  # 🔥 修改 3: 梯度裁剪从 5.0 降为 1.0 (Adam 需要更稳的约束)

echo "✅ Training finished."