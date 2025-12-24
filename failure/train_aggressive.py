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

def set_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Starting AGGRESSIVE Training on {device}")
    
    model = EstraNet(d_model=args.d_model, n_head=args.n_head, n_layers=args.n_layer).to(device)
    
    # 初始设置
    # 注意：我们手动控制学习率，所以这里初始化给一个极小值，后面手动Warmup
    optimizer = SCOOP(model.parameters(), lr=1e-8, rho=0.96)
    
    if not os.path.exists(args.result_path): os.makedirs(args.result_path)
    if not os.path.exists(args.checkpoint_dir): os.makedirs(args.checkpoint_dir)

    dataset = ASCADv2Dataset(args.data_path, split='train', input_len=args.input_length)
    loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    criterion = nn.CrossEntropyLoss()
    
    # --- ⚙️ 激进策略配置 ---
    TARGET_LR = args.learning_rate  # 比如 3e-4
    WARMUP_EPOCHS = 10              # 超长热身：10个Epoch
    steps_per_epoch = len(loader)
    warmup_steps = steps_per_epoch * WARMUP_EPOCHS
    
    # 状态标记
    has_decayed = False             # 是否已经触发过“10倍减速”
    
    log_file = os.path.join(args.result_path, "train_log_aggressive.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f: f.write("Epoch,TrainLoss,TestLoss,Rank,LR\n")

    hessian_freq = 10
    epochs = args.train_steps // len(loader) + 1
    global_step = 0

    print(f"Plan: Warmup to {TARGET_LR} for {WARMUP_EPOCHS} epochs.")
    print(f"Trigger: If Rank < 100, LR will drop to {TARGET_LR / 10:.1e}")

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        total_loss = 0
        count = 0
        
        for i, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            update_h = (i % hessian_freq == 0)
            
            # --- 1. 手动 Warmup 逻辑 ---
            if global_step < warmup_steps:
                # 线性增加
                warmup_lr = TARGET_LR * (global_step / warmup_steps)
                set_learning_rate(optimizer, warmup_lr)
            # Warmup 结束后，保持 TARGET_LR，直到触发 decay
            
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

        # --- 验证阶段 ---
        # 强制使用 10000 条，确保 Rank 是真的
        rank, test_loss = evaluate_in_memory(model, args.data_path, n_test=10000, device=device)
        
        avg_train_loss = total_loss / count
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}: Train {avg_train_loss:.4f} | Test {test_loss:.4f} | Rank {rank} | LR {current_lr:.6f}")
        
        # --- ⚙️ 2. 智能刹车逻辑 ---
        # 如果 Rank 确实降到了 100 以下，并且还没有减速过，且 Warmup 已经结束
        if rank < 100 and not has_decayed and global_step > warmup_steps:
            print(f"🚀 SUCCESS! Rank {rank} < 100 detected.")
            print(f"📉 Triggering 10x LR Decay: {current_lr:.1e} -> {current_lr * 0.1:.1e}")
            
            # 永久减速
            TARGET_LR = TARGET_LR * 0.1 
            set_learning_rate(optimizer, TARGET_LR)
            has_decayed = True
            
            # 保存这个关键时刻的模型
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"estranet_breakthrough_rank{rank}.pth"))

        with open(log_file, "a") as f:
            f.write(f"{epoch},{avg_train_loss:.4f},{test_loss:.4f},{rank},{current_lr:.6f}\n")
            
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "estranet_latest.pth"))
        if rank < 100 or epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"estranet_epoch_{epoch}.pth"))
        
        if global_step >= args.train_steps: break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--result_path", type=str, default="./results")
    parser.add_argument("--learning_rate", type=float, default=3e-4) # 默认激进 LR
    parser.add_argument("--input_length", type=int, default=15000)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--train_steps", type=int, default=400000)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--clip", type=float, default=5.0)
    
    args, unknown = parser.parse_known_args()
    train(args)