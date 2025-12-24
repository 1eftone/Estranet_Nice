import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import sys
import numpy as np

from evaluate import evaluate_in_memory
from dataset import ASCADv2Dataset
from model import EstraNet
from scoop import SCOOP 

class RankScheduler:
    def __init__(self, optimizer, mode='min', patience=10, factor_down=0.5, factor_up=1.0, min_lr=1e-6, max_lr=1e-3, verbose=True):
        self.optimizer = optimizer
        self.patience = patience
        self.factor_down = factor_down
        self.factor_up = factor_up
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.verbose = verbose
        
        self.best_rank = float('inf')
        self.num_bad_epochs = 0
    
    def step(self, current_rank):
        if current_rank < self.best_rank:
            if self.verbose:
                print(f"\n[Scheduler] Rank improved from {self.best_rank} to {current_rank}.")
            self.best_rank = current_rank
            self.num_bad_epochs = 0
            self._adjust_lr(self.factor_down)
        else:
            self.num_bad_epochs += 1
            if self.verbose:
                print(f"\n[Scheduler] Rank did not improve for {self.num_bad_epochs}/{self.patience} epochs.")
            
            if self.num_bad_epochs >= self.patience:
                if self.verbose:
                    print(f"[Scheduler] Patience exhausted! LR adjustment triggered.")
                self._adjust_lr(self.factor_up)
                self.num_bad_epochs = 0

    def _adjust_lr(self, factor):
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * factor
            if new_lr > self.max_lr: new_lr = self.max_lr
            if new_lr < self.min_lr: new_lr = self.min_lr
            if new_lr != old_lr:
                param_group['lr'] = new_lr
                if self.verbose:
                    print(f"[Scheduler] Learning Rate adjusted: {old_lr:.6f} -> {new_lr:.6f}")

def finetune(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not args.resume or not os.path.isfile(args.resume):
        print("Error: Fine-tuning requires a valid checkpoint path via --resume")
        return

    # 【修改点 1】 更新日志文件名和表头
    log_file = "finetune_auto.log"
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            # TrainLoss vs TestLoss
            f.write("Epoch,TrainLoss,TestLoss,Rank,LR\n")

    print(f"Loading Dataset...")
    train_dataset = ASCADv2Dataset(args.data_path, split='train', input_len=15000)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    print(f"Initializing Model...")
    model = EstraNet(d_model=512, n_head=16).to(device)
    
    print(f"Loading Checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint)
    
    current_lr = args.lr
    print(f"Initializing SCOOP for Finetuning with LR: {current_lr}")
    optimizer = SCOOP(model.parameters(), lr=current_lr, rho=0.96)
    
    # 保持你确认过的最佳 Scheduler 配置
    scheduler = RankScheduler(optimizer, patience=10, factor_up=1.0, factor_down=0.5)
    
    criterion = nn.CrossEntropyLoss()
    hessian_update_freq = 10 
    
    print("Evaluating initial rank...")
    # 【修改点 2】 接收两个返回值
    initial_rank, initial_test_loss = evaluate_in_memory(model, args.data_path, n_test=2000, device=device)
    print(f"Initial Rank: {initial_rank} | Test Loss: {initial_test_loss:.4f}")
    best_rank = initial_rank
    scheduler.best_rank = initial_rank

    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Auto-Finetune Epoch {epoch+1}/{args.epochs}")
        total_loss = 0
        count = 0
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            update_hessian = (batch_idx % hessian_update_freq == 0)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward(create_graph=update_hessian)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if update_hessian:
                optimizer.hutchinson_hessian()
            
            optimizer.step()
            
            if update_hessian:
                optimizer.zero_grad() 
            
            loss_val = loss.item()
            total_loss += loss_val
            count += 1
            curr_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f"{loss_val:.4f}", 'lr': f"{curr_lr:.6f}"})
        
        avg_train_loss = total_loss / count if count > 0 else 0
        
        # --- Validate ---
        print(f"\nValidating Epoch {epoch+1}...")
        # 【修改点 3】 接收 Test Loss
        current_rank, avg_test_loss = evaluate_in_memory(model, args.data_path, n_test=2000, device=device)
        print(f"Epoch {epoch+1} -> TrainLoss: {avg_train_loss:.4f} | TestLoss: {avg_test_loss:.4f} | Rank: {current_rank} | LR: {curr_lr:.6f}")
        
        scheduler.step(current_rank)
        
        # --- Log ---
        # 【修改点 4】 写入新格式
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{avg_train_loss:.4f},{avg_test_loss:.4f},{current_rank},{curr_lr:.6f}\n")

        # --- Save ---
        torch.save(model.state_dict(), "estranet_finetuned_latest.pth")
        torch.save(model.state_dict(), f"estranet_finetuned_epoch_{epoch+1}.pth")
        
        if current_rank < best_rank:
            best_rank = current_rank
            torch.save(model.state_dict(), "estranet_finetuned_best.pth")
            print(f"New Best Finetuned Model! Rank: {best_rank}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/joey1/Documents/joey/Data/ASCAD/ascadv2-extracted.h5")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--start_epoch", type=int, required=True)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()
    finetune(args)