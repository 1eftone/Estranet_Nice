import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import argparse
import os
import math
from evaluate import evaluate_in_memory
from dataset import ASCADv2Dataset
from model import EstraNet
from scoop import SCOOP 

# --- 调度器函数 ---
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step: int):
        # 1. Warmup 阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # 2. Decay 阶段
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 1.0 - progress)
    return LambdaLR(optimizer, lr_lambda)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. 初始化模型
    model = EstraNet(d_model=args.d_model, n_head=args.n_head, n_layers=args.n_layer).to(device)
    
    # 2. 加载旧权重 (如果有)
    if args.warm_start.lower() == 'true' and args.model_path:
        print(f"🎯 Loading Checkpoint from: {args.model_path}")
        if os.path.exists(args.model_path):
            checkpoint = torch.load(args.model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("✅ Checkpoint Loaded Successfully!")
        else:
            print(f"❌ Warning: Checkpoint {args.model_path} not found! Starting scratch.")
    else:
        print("🆕 Starting from scratch.")

    # 3. 优化器
    optimizer = SCOOP(model.parameters(), lr=args.learning_rate, rho=0.96)
    
    if not os.path.exists(args.result_path): os.makedirs(args.result_path)
    if not os.path.exists(args.checkpoint_dir): os.makedirs(args.checkpoint_dir)

    dataset = ASCADv2Dataset(args.data_path, split='train', input_len=args.input_length)
    loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    criterion = nn.CrossEntropyLoss()
    
    # --- 4. 关键：使用传入的 warmup_steps ---
    print(f"Total Steps: {args.train_steps}, Warmup Steps: {args.warmup_steps}")
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, args.train_steps)
    
    log_file = os.path.join(args.result_path, "train_log.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f: f.write("Epoch,TrainLoss,TestLoss,Rank,LR\n")

    hessian_freq = 10
    epochs = args.train_steps // len(loader) + 1
    global_step = 0

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        total_loss = 0
        count = 0
        
        for i, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            update_h = (i % hessian_freq == 0)
            
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward(create_graph=update_h)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            if update_h: optimizer.hutchinson_hessian()
            optimizer.step()
            scheduler.step()
            
            if update_h: optimizer.zero_grad()
            
            total_loss += loss.item()
            count += 1
            global_step += 1
            
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.6f}"})
            
            if global_step >= args.train_steps: break

        # 验证
        # 强制使用 10000 条，确保数据真实
        rank, test_loss = evaluate_in_memory(model, args.data_path, n_test=10000, device=device)
        avg_train_loss = total_loss / count
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}: Train {avg_train_loss:.4f} | Test {test_loss:.4f} | Rank {rank} | LR {current_lr:.6f}")
        
        with open(log_file, "a") as f:
            f.write(f"{epoch},{avg_train_loss:.4f},{test_loss:.4f},{rank},{current_lr:.6f}\n")
            
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "estranet_latest.pth"))
        # 如果 Rank 不错，或者每10轮存一次
        if rank < 100 or epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"estranet_epoch_{epoch}.pth"))
        
        if global_step >= args.train_steps: break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--result_path", type=str, default="./results")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--input_length", type=int, default=15000)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--train_steps", type=int, default=400000)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--warm_start", type=str, default='False')
    parser.add_argument("--model_path", type=str, default="")
    # 🔥 这里的 default=0 只是兜底，真正的数值由 .sh 文件传入
    parser.add_argument("--warmup_steps", type=int, default=0) 

    args, unknown = parser.parse_known_args()
    train(args)