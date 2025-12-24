import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
from evaluate import evaluate_in_memory
from dataset import ASCADv2Dataset
from model import EstraNet
from scoop import SCOOP 

def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 SNIPER MODE ON: {device}")
    
    model = EstraNet(d_model=args.d_model, n_head=args.n_head, n_layers=args.n_layer).to(device)
    
    # 初始学习率 (比如 5e-5)
    optimizer = SCOOP(model.parameters(), lr=args.learning_rate, rho=0.96)
    
    if not os.path.exists(args.result_path): os.makedirs(args.result_path)
    if not os.path.exists(args.checkpoint_dir): os.makedirs(args.checkpoint_dir)

    dataset = ASCADv2Dataset(args.data_path, split='train', input_len=args.input_length)
    loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    criterion = nn.CrossEntropyLoss()
    
    # --- 狙击手配置 ---
    TRIGGER_RANK = 60       # 🎯 触发阈值：只要 Rank 小于 60
    DECAY_FACTOR = 0.1      # 📉 降速倍率：LR * 0.1
    has_triggered = False   # 标记是否已经降速过

    # 手动 Warmup (前 10%)
    steps_per_epoch = len(loader)
    warmup_steps = int(args.train_steps * 0.1)
    
    log_file = os.path.join(args.result_path, "train_log_sniper.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f: f.write("Epoch,TrainLoss,TestLoss,Rank,LR\n")

    hessian_freq = 10
    epochs = args.train_steps // len(loader) + 1
    global_step = 0
    TARGET_LR = args.learning_rate

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        total_loss = 0
        count = 0
        
        for i, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            update_h = (i % hessian_freq == 0)
            
            # --- 手动 LR 控制逻辑 ---
            if not has_triggered:
                # 1. 如果还没触发降速，执行 Warmup 或保持 LR
                if global_step < warmup_steps:
                    lr_scale = float(global_step) / float(max(1, warmup_steps))
                    current_lr = TARGET_LR * lr_scale
                    set_lr(optimizer, current_lr)
                else:
                    # Warmup 结束，保持最高速度
                    set_lr(optimizer, TARGET_LR)
            # 如果已经触发降速，LR 保持不变（已经在触发时降过了）

            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward(create_graph=update_h)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            if update_h: optimizer.hutchinson_hessian()
            optimizer.step()
            if update_h: optimizer.zero_grad()
            
            total_loss += loss.item()
            count += 1
            global_step += 1
            
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.6f}"})
            
            if global_step >= args.train_steps: break

        # --- 验证与狙击判断 ---
        rank, test_loss = evaluate_in_memory(model, args.data_path, n_test=10000, device=device)
        avg_train_loss = total_loss / count
        print(f"\nEpoch {epoch}: Train {avg_train_loss:.4f} | Test {test_loss:.4f} | Rank {rank} | LR {current_lr:.6f}")
        
        # 🔥 狙击逻辑 🔥
        if rank < TRIGGER_RANK and not has_triggered:
            print(f"🚀 SNIPER TRIGGERED! Rank {rank} < {TRIGGER_RANK}")
            
            # 立即降速 10 倍
            new_lr = current_lr * DECAY_FACTOR
            set_lr(optimizer, new_lr)
            has_triggered = True
            
            print(f"📉 LR dropped to {new_lr:.1e}. Keeping it low for convergence.")
            
            # 保存这个珍贵的模型
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"estranet_sniper_hit_rank{rank}.pth"))

        with open(log_file, "a") as f:
            f.write(f"{epoch},{avg_train_loss:.4f},{test_loss:.4f},{rank},{current_lr:.6f}\n")
            
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "estranet_latest.pth"))
        
        if global_step >= args.train_steps: break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 只需要基本参数
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--result_path", type=str, default="./results")
    parser.add_argument("--learning_rate", type=float, default=5e-5) # 默认 5e-5
    parser.add_argument("--input_length", type=int, default=15000)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--train_steps", type=int, default=400000)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--clip", type=float, default=5.0)
    
    args, unknown = parser.parse_known_args()
    train(args)