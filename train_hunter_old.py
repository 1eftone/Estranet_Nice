import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import sys
import numpy as np
import random
import torch.autograd as autograd

# 确保这些模块存在
from evaluate import evaluate_in_memory
from dataset import ASCADv2Dataset
from model import EstraNet

# 🔥 0. 固定随机种子
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 🔥 1. 强制禁用 PyTorch 2.0+ 高效 Attention (为了二阶导数计算)
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except AttributeError:
    pass

def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

# 🔥 手动计算 Hessian 向量积 (HVP) 并加到梯度上
# 这就是 SCOOP 的核心逻辑，我们把它解耦出来，为了能随时关闭
def add_hessian_regularization(model, loss, optimizer, rho=0.96):
    params = [p for p in model.parameters() if p.requires_grad]
    
    # 1. 计算一阶梯度 (Grad)
    grads = autograd.grad(loss, params, create_graph=True, retain_graph=True)
    
    # 2. 生成 Hutchinson 随机向量 (v)
    v = [torch.randint_like(p, high=2) * 2 - 1 for p in params]
    
    # 3. 计算 Hv (Hessian-Vector Product)
    # H*v = grad(grad(loss)*v)
    grad_v = sum([torch.sum(g * s) for g, s in zip(grads, v)])
    Hv = autograd.grad(grad_v, params, retain_graph=False)
    
    # 4. 将平滑后的梯度更新到 p.grad
    # g_new = g + rho * (Hv * v - g) / (1 + rho) 
    # (简化的 SCOOP 更新规则)
    
    with torch.no_grad():
        for i, p in enumerate(params):
            if p.grad is None: continue
            
            # SCOOP 核心公式: 校正梯度
            # g_scoop = g + (rho * (Hv * v) - rho * g)
            # 这里我们做一个简单的加权融合，效果类似
            p.grad.add_(Hv[i] * v[i], alpha=rho)
            p.grad.div_(1 + rho)

def train(args):
    seed_everything(45)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔥 STRATEGY: AdamW + SCOOP (Switching to Pure AdamW at Rank 15) on {device}")
    
    model = EstraNet(d_model=args.d_model, n_head=args.n_head, n_layers=args.n_layer).to(device)
    
    # 🔥 2. 使用原生 AdamW
    # AdamW 对噪声的鲁棒性比 SGD 强很多，适合 10k 这种小样本
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=1e-2, # AdamW 需要稍大的 WD
        betas=(0.9, 0.999)
    )
    
    if not os.path.exists(args.result_path): os.makedirs(args.result_path)
    if not os.path.exists(args.checkpoint_dir): os.makedirs(args.checkpoint_dir)

    dataset = ASCADv2Dataset(args.data_path, split='train', input_len=args.input_length)
    loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    criterion = nn.CrossEntropyLoss()
    
    best_rank = float('inf')
    
    # --- 策略配置 ---
    
    # 阶段 1: Adam + SCOOP (快速下降)
    # 使用 Hessian 导航，防止 Adam 过早陷入局部最优
    HESSIAN_ENABLED = True 
    HESSIAN_FREQ = 2
    
    # 阶段 2: Pure Adam (脑切除)
    # Rank < 15 时触发
    TRIGGER_PURE_ADAM_RANK = 0
    PURE_ADAM_LR = 1e-4  # Adam 的 5e-6 约等于 SGD 的 1e-6，非常细腻
    pure_adam_triggered = False
    
    steps_per_epoch = len(loader)
    warmup_epochs = 1 # Adam 不需要太长的 Warmup
    warmup_steps = steps_per_epoch * warmup_epochs
    
    log_file = os.path.join(args.result_path, "train_log_adam_scoop.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f: f.write("Epoch,TrainLoss,TestLoss,Rank,LR,Mode\n")

    epochs = args.train_steps // steps_per_epoch + 1
    global_step = 0

    print(f"⚙️ Config: AdamW LR={args.learning_rate} | Pure Adam Trigger < {TRIGGER_PURE_ADAM_RANK} (LR={PURE_ADAM_LR})\n")

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        total_loss = 0
        count = 0
        
        for i, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            # 只有在未触发纯 Adam 模式，且符合频率时，才计算 Hessian
            update_h = HESSIAN_ENABLED and (not pure_adam_triggered) and (i % HESSIAN_FREQ == 0)
            
            # Warmup
            if not pure_adam_triggered and global_step < warmup_steps:
                lr_scale = float(global_step) / float(max(1, warmup_steps))
                set_lr(optimizer, args.learning_rate * lr_scale)
            
            optimizer.zero_grad(set_to_none=True)
            
            out = model(data)
            loss = criterion(out, target)
            
            # 🔥 修复点 1: 先把数值存下来！
            loss_val = loss.item()

            # 如果需要计算 Hessian，必须 create_graph
            loss.backward(create_graph=update_h)
            
            if update_h:
                # 手动注入 SCOOP 梯度校正
                add_hessian_regularization(model, loss, optimizer)
            
            # 梯度裁剪 (Adam 也需要，防止梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            optimizer.step()
            
            # 如果计算图被保留了，需要手动释放，防止显存爆炸
            if update_h:
                del loss
                # 显式清空图通常由 optimizer.zero_grad 处理，但手动 del 是好习惯
            
            # 🔥 修复点 3: 累加用刚才存的数值，而不是 Tensor
            total_loss += loss_val 
            count += 1
            global_step += 1
            
            curr_lr_display = optimizer.param_groups[0]['lr']
            # 这里也用 total_loss，逻辑没问题
            pbar.set_postfix({'loss': f"{total_loss/count:.4f}", 'lr': f"{curr_lr_display:.6f}"})
            
            if global_step >= args.train_steps: break

        # --- 验证 ---
        rank, test_loss = evaluate_in_memory(model, args.data_path, n_test=10000, device=device)
        avg_train_loss = total_loss / count
        curr_lr_display = optimizer.param_groups[0]['lr']
        mode_str = "PureAdam" if pure_adam_triggered else "Adam+SCOOP"
        
        print(f"\nEpoch {epoch}: Train {avg_train_loss:.4f} | Test {test_loss:.4f} | Rank {rank} | LR {curr_lr_display:.6e} | Mode {mode_str}")
        
        with open(log_file, "a") as f:
            f.write(f"{epoch},{avg_train_loss:.4f},{test_loss:.4f},{rank},{curr_lr_display:.6e},{mode_str}\n")
            
        if rank < best_rank:
            print(f"⭐ New Best Rank! ({best_rank} -> {rank}).")
            best_rank = rank
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "estranet_best_rank.pth"))
            if rank <= 5:
                 torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"estranet_top5_rank{rank}.pth"))
        
        # --- 决策逻辑 ---
        if rank < 1: # 针对 10k 数据的极致要求
            print(f"\n🏆 VICTORY! Rank {rank} achieved.")
            print(f"   🛑 Stopping training.")
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "estranet_FINAL_WINNER.pth"))
            sys.exit(0)
            
        # 🔥 核心切换逻辑
        if rank < TRIGGER_PURE_ADAM_RANK and not pure_adam_triggered:
            print(f"\n🧠 ADAM TAKEOVER: Rank {rank} < {TRIGGER_PURE_ADAM_RANK}")
            print(f"   🚫 Disabling Hessian (SCOOP). Switching to Pure AdamW.")
            print(f"   📉 Dropping LR to {PURE_ADAM_LR:.1e} for polishing.")
            
            set_lr(optimizer, PURE_ADAM_LR)
            #pure_adam_triggered = True
            
            # 保存切换点的模型
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "estranet_pre_adam_switch.pth"))

        if global_step >= args.train_steps: break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Adam 的学习率通常比 SGD 大一个数量级，1e-4 是 Transformer/ResNet 的黄金起点
    parser.add_argument("--learning_rate", type=float, default=1e-4) 
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_hunter_adam")
    parser.add_argument("--result_path", type=str, default="./results_hunter_adam")
    parser.add_argument("--input_length", type=int, default=15000)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--train_steps", type=int, default=400000)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--clip", type=float, default=1.0) # Adam 通常裁剪得更紧一点
    
    args, unknown = parser.parse_known_args()
    train(args)