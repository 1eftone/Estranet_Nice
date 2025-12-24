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
    print(f"\n🔥 AUTO-PILOT (PRIORITY MODE) STARTED on {device}")
    
    model = EstraNet(d_model=args.d_model, n_head=args.n_head, n_layers=args.n_layer).to(device)
    
    # 初始 LR 设为 0 (等待 Warmup)
    optimizer = SCOOP(model.parameters(), lr=args.learning_rate, rho=0.96)
    set_lr(optimizer, 1e-8) # 防止除零
    
    if not os.path.exists(args.result_path): os.makedirs(args.result_path)
    if not os.path.exists(args.checkpoint_dir): os.makedirs(args.checkpoint_dir)

    dataset = ASCADv2Dataset(args.data_path, split='train', input_len=args.input_length)
    loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    criterion = nn.CrossEntropyLoss()
    
    # --- 🤖 策略参数 ---
    TARGET_LR = args.learning_rate  # Old Server: 1e-5
    
    best_rank = float('inf')
    
    # 优先级 1: 决赛圈
    TRIGGER_1_RANK = 20
    DECAY_1_FACTOR = 0.2  # 1/5
    triggered_1 = False
    
    # 优先级 0: 终极锁定 (Deep Freeze)
    TRIGGER_2_RANK = 5
    DECAY_2_FACTOR = 0.2  # 1/5
    triggered_2 = False
    
    # Warmup 设置
    steps_per_epoch = len(loader)
    warmup_steps = steps_per_epoch * 3 
    
    log_file = os.path.join(args.result_path, "train_log_priority.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f: f.write("Epoch,TrainLoss,TestLoss,Rank,LR\n")

    hessian_freq = 10
    epochs = args.train_steps // steps_per_epoch + 1
    global_step = 0

    print(f"⚙️ Config: Warmup=3 Epochs. Priority Check: Rank<{TRIGGER_2_RANK} >> Rank<{TRIGGER_1_RANK} >> Warmup.\n")

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        total_loss = 0
        count = 0
        
        for i, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            update_h = (i % hessian_freq == 0)
            
            # --- 🚦 绝对优先级逻辑 ---
            # 只有在【完全没有触发过任何降速】的情况下，才允许执行 Warmup
            if not triggered_1 and not triggered_2:
                if global_step < warmup_steps:
                    # Warmup 爬升阶段
                    lr_scale = float(global_step) / float(max(1, warmup_steps))
                    current_lr = TARGET_LR * lr_scale
                    set_lr(optimizer, current_lr)
                else:
                    # Warmup 结束，保持目标速度
                    current_lr = TARGET_LR
                    set_lr(optimizer, current_lr)
            else:
                # 一旦 triggered 为 True，LR 就被锁死在降速后的值，
                # 无论 global_step 是多少，绝对不再执行 Warmup 逻辑！
                pass

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

        # --- 📊 验证 ---
        rank, test_loss = evaluate_in_memory(model, args.data_path, n_test=10000, device=device)
        avg_train_loss = total_loss / count
        curr_lr_display = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch}: Train {avg_train_loss:.4f} | Test {test_loss:.4f} | Rank {rank} | LR {curr_lr_display:.6e}")
        
        with open(log_file, "a") as f:
            f.write(f"{epoch},{avg_train_loss:.4f},{test_loss:.4f},{rank},{curr_lr_display:.6e}\n")
            
        # 🔥 1. 保存最佳模型 (Safety Net)
        if rank < best_rank:
            print(f"⭐ Saving Best Model (Rank {rank})")
            best_rank = rank
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "estranet_best_rank.pth"))
        
        # 备份最新
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "estranet_latest.pth"))

        # --- 🤖 决策逻辑 (优先级最高) ---
        
        # 🏆 胜利
        if rank == 0:
            print("\n🎉🎉🎉 Rank 0 Achieved! STOP. 🎉🎉🎉")
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "estranet_RANK_0.pth"))
            sys.exit(0)

        # 🔥 优先级 0: 终极锁定 (Deep Freeze, Rank < 5)
        # 只要 Rank 掉进 5，不管之前有没有触发过狙击，立刻执行最高级冻结
        if rank < TRIGGER_2_RANK and not triggered_2:
            print(f"\n❄️ ULTIMATE LOCK: Rank {rank} < {TRIGGER_2_RANK}")
            print(f"   Interrupting everything to FREEZE the model.")
            
            # 关键：基于【当前瞬间】的 LR 进行降速
            # 如果是在 Warmup 期间触发，curr_lr_display 很小，这就对了！
            new_lr = curr_lr_display * DECAY_2_FACTOR 
            
            print(f"📉 Deep Freeze LR: {curr_lr_display:.2e} -> {new_lr:.2e}")
            set_lr(optimizer, new_lr)
            
            triggered_2 = True
            triggered_1 = True # 同时屏蔽掉阶段1的逻辑
            
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"estranet_freeze_rank{rank}.pth"))

        # ✨ 优先级 1: 狙击 (Sniper, Rank < 20)
        # 只有在没触发过更高级锁定的情况下执行
        elif rank < TRIGGER_1_RANK and not triggered_1:
            print(f"\n🚀 Sniper Triggered: Rank {rank} < {TRIGGER_1_RANK}")
            print(f"   Warmup/Training interrupted. Holding position.")
            
            new_lr = curr_lr_display * DECAY_1_FACTOR
            
            print(f"📉 Dropping LR: {curr_lr_display:.2e} -> {new_lr:.2e}")
            set_lr(optimizer, new_lr)
            
            triggered_1 = True
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"estranet_decay_rank{rank}.pth"))

        if global_step >= args.train_steps: break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_final")
    parser.add_argument("--result_path", type=str, default="./results_final")
    # 旧服务器默认 1e-5
    parser.add_argument("--learning_rate", type=float, default=1e-5) 
    parser.add_argument("--input_length", type=int, default=15000)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--train_steps", type=int, default=400000)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--clip", type=float, default=5.0)
    
    args, unknown = parser.parse_known_args()
    train(args)