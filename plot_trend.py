import torch
import os
import matplotlib.pyplot as plt
import csv  # 使用自带的csv模块代替pandas
from model import EstraNet
# 引入你现有的评估函数
from evaluate import evaluate_in_memory 

# --- 配置区 ---
# 确保这个路径是对的
MODEL_PATH = "/home/joey1/Documents/joey/projects/Estranet_pytorch/checkpoints_hunter_adam/estranet_best_rank.pth"
DATA_PATH = "/home/joey1/Documents/joey/Data/ASCAD/ascadv2-extracted.h5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 测试节点：从 1000 到 10000，每隔 1000 测一次
TRACE_STEPS = list(range(500, 10500, 500)) 

def main():
    print(f"Loading Model from: {MODEL_PATH}")
    # 1. 初始化模型 (确保参数与训练时一致)
    model = EstraNet(d_model=128, n_head=8, n_layers=2).to(DEVICE)
    
    # 加载权重
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint
    
    model.eval()
    print("Model loaded successfully.")

    # 用于绘图的数据列表
    traces_list = []
    ranks_list = []
    loss_list = []

    print(f"\nStarting Trend Analysis on {DEVICE}...")
    print(f"{'Traces':<10} | {'Rank':<10} | {'Loss':<10}")
    print("-" * 35)

    # 2. 循环测试
    csv_rows = []
    for n_test in TRACE_STEPS:
        try:
            rank, loss = evaluate_in_memory(model, DATA_PATH, n_test=n_test, device=DEVICE)
            
            # 记录数据
            traces_list.append(n_test)
            ranks_list.append(rank)
            loss_list.append(loss)
            
            # 保存行数据以便写入CSV
            csv_rows.append([n_test, rank, loss])
            
            print(f"{n_test:<10} | {rank:<10} | {loss:.4f}")
        except Exception as e:
            print(f"Error at {n_test} traces: {e}")
            break

    # 3. 保存结果到 CSV (使用 Python 自带 csv 模块)
    csv_file = "rank_trend_epoch_1.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Traces', 'Rank', 'Loss']) # 表头
        writer.writerows(csv_rows)
    print(f"\nData saved to {csv_file}")

    # 4. 画图 (使用 Matplotlib)
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(traces_list, ranks_list, marker='o', linestyle='-', color='b', label='Key Rank')
        
        plt.title(f'Key Rank Trend (Epoch 1)\nLR=4e-5 (Warmup Phase)', fontsize=14)
        plt.xlabel('Number of Traces', fontsize=12)
        plt.ylabel('Key Rank (Log Scale)', fontsize=12)
        plt.yscale('log') # 对数坐标
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        
        img_file = "rank_trend_epoch_1.png"
        plt.savefig(img_file)
        print(f"Plot saved to {img_file}")
    except Exception as e:
        print(f"Plotting failed (maybe matplotlib missing?): {e}")
        print("But CSV data is saved safely.")

if __name__ == "__main__":
    main()