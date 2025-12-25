import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import sys
from evaluate import evaluate_in_memory
from dataset import ASCADv2Dataset
from model import EstraNet
from scoop import SCOOP 

def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔥 OLD SERVER REVENGE: New SCOOP + Auto-Brake on {device}")
    
    model = EstraNet(d_model=args.d_model, n_head=args.n_head, n_layers=args.n_layer).to(device)
    
    # 使用新版 SCOOP (Adam Hybrid)
    optimizer = SCOOP(model.parameters(), lr=args.learning_rate, rho=0.96)
    set_lr(optimizer, 1e-8) 
    
    if not os.path.exists(args.result_path): os.makedirs(args.result_path)
    if not os.path.exists(args.checkpoint_dir): os.makedirs(args.checkpoint_dir)

    dataset = ASCADv2Dataset(args.data_path, split='train', input_len=args.input_length)
    loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    criterion = nn.CrossEntropyLoss()
    
    # --- 策略配置 ---
    TARGET_LR = args.learning_rate # 建议 3e-5
    
    best_rank = float('inf')
    best_test_loss = float('inf')
    
    # 🛑 Auto-Brake 配置
    loss_patience_counter = 0
    LOSS_PATIENCE_LIMIT = 1  # 旧服务器只要 Loss 反弹一次就立刻降速
    loss_decay_triggered = False

    TRIGGER_SNIPER_RANK = 20
    sniper_triggered = False
    
    TRIGGER_FREEZE_RANK = 5
    freeze_triggered = False
    
    steps_per_epoch = len(loader)
    warmup_steps = steps_per_epoch * 3 
    
    log_file = os.path.join(args.result_path, "train_log_revenge.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f: f.write("Epoch,TrainLoss,TestLoss,Rank,LR\n")

    hessian_freq = 10
    epochs = args.train_steps // steps_per_epoch + 1
    global_step = 0

    print(f"⚙️ Config: LR={TARGET_LR}. Strategy: Strict Auto-Brake (Patience=1).\n")

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        total_loss = 0
        count = 0
        
        for i, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            update_h = (i % hessian_freq == 0)
            
            # --- LR Warmup ---
            if not loss_decay_triggered and not sniper_triggered and not freeze_triggered:
                if global_step < warmup_steps:
                    lr_scale = float(global_step) / float(max(1, warmup_steps))
                    current_lr = TARGET_LR * lr_scale
                    set_lr(optimizer, current_lr)
                else:
                    current_lr = TARGET_LR
                    set_lr(optimizer, current_lr)
            
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
            
            curr_lr_display = optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{curr_lr_display:.6f}"})
            
            if global_step >= args.train_steps: break

        # --- 验证 ---
        rank, test_loss = evaluate_in_memory(model, args.data_path, n_test=10000, device=device)
        avg_train_loss = total_loss / count
        curr_lr_display = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch}: Train {avg_train_loss:.4f} | Test {test_loss:.4f} | Rank {rank} | LR {curr_lr_display:.6e}")
        
        with open(log_file, "a") as f:
            f.write(f"{epoch},{avg_train_loss:.4f},{test_loss:.4f},{rank},{curr_lr_display:.6e}\n")
            
        if rank < best_rank:
            print(f"⭐ New Best Rank! ({best_rank} -> {rank}).")
            best_rank = rank
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "estranet_best_rank.pth"))
        
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "estranet_latest.pth"))

        # --- 决策 ---
        if rank == 0:
            print("\n🎉🎉🎉 Rank 0 Achieved! 🎉🎉🎉")
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "estranet_RANK_0.pth"))
            sys.exit(0)

        # 优先级 1: Rank < 5
        if rank < TRIGGER_FREEZE_RANK and not freeze_triggered:
            print(f"\n❄️ DEEP FREEZE: Rank {rank} < {TRIGGER_FREEZE_RANK}")
            new_lr = curr_lr_display * 0.2 
            set_lr(optimizer, new_lr)
            freeze_triggered = True
            sniper_triggered = True
            loss_decay_triggered = True # 锁定所有状态
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"estranet_freeze_rank{rank}.pth"))

        # 优先级 2: Rank < 20
        elif rank < TRIGGER_SNIPER_RANK and not sniper_triggered:
            print(f"\n🚀 Sniper Triggered: Rank {rank} < {TRIGGER_SNIPER_RANK}")
            new_lr = curr_lr_display * 0.2
            set_lr(optimizer, new_lr)
            sniper_triggered = True
            loss_decay_triggered = True
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"estranet_decay_rank{rank}.pth"))
            
        # 优先级 3: Loss 刹车
        elif not loss_decay_triggered and epoch > 2:
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                loss_patience_counter = 0
            else:
                loss_patience_counter += 1
                print(f"⚠️ Warning: Test Loss Rising ({loss_patience_counter}/{LOSS_PATIENCE_LIMIT})")
                
                if loss_patience_counter >= LOSS_PATIENCE_LIMIT:
                    print(f"🛑 AUTO-BRAKE: Loss rising. Cutting LR.")
                    new_lr = curr_lr_display * 0.5 
                    print(f"📉 Braking: {curr_lr_display:.2e} -> {new_lr:.2e}")
                    set_lr(optimizer, new_lr)
                    loss_decay_triggered = True
                    best_test_loss = test_loss 
                    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"estranet_autobrake_epoch{epoch}.pth"))

        if global_step >= args.train_steps: break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 🌟 关键参数：3e-5 (旧服务器推荐起步速度)
    parser.add_argument("--learning_rate", type=float, default=3e-5) 
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_revenge")
    parser.add_argument("--result_path", type=str, default="./results_revenge")
    parser.add_argument("--input_length", type=int, default=15000)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--train_steps", type=int, default=400000)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--clip", type=float, default=5.0)
    
    args, unknown = parser.parse_known_args()
    train(args)