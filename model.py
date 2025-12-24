import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.checkpoint  # <--- 请务必添加这一行！

class LinearAttention(nn.Module):
    def __init__(self, d_model, n_head, d_kernel_map=128):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.d_kernel_map = d_kernel_map
        
        self.register_buffer(
            'projection_matrix', 
            torch.randn(self.d_head, d_kernel_map)
        )
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def fourier_kernel(self, x):
        projection = torch.matmul(x, self.projection_matrix)
        projection = projection / math.sqrt(self.d_head)
        return torch.cat([torch.cos(projection), torch.sin(projection)], dim=-1) / math.sqrt(self.d_kernel_map)

    def forward(self, x):
        B, L, _ = x.shape
        H = self.n_head
        
        q = self.q_proj(x).view(B, L, H, -1)
        k = self.k_proj(x).view(B, L, H, -1)
        v = self.v_proj(x).view(B, L, H, -1)
        
        q_prime = self.fourier_kernel(q)
        k_prime = self.fourier_kernel(k)
        
        kv = torch.einsum('blhm,blhd->bhmd', k_prime, v)
        out = torch.einsum('blhm,bhmd->blhd', q_prime, kv)
        
        out = out.reshape(B, L, self.d_model)
        return self.out_proj(out)

class EstraNetBlock(nn.Module):
    def __init__(self, d_model, n_head, d_inner, dropout=0.1):
        super().__init__()
        self.attn = LinearAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.linear1 = nn.Linear(d_model, d_inner)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_inner, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x2 = self.norm1(x)
        x2 = self.attn(x2)
        x = x + self.dropout1(x2)
        
        x2 = self.norm2(x)
        x2 = self.linear2(self.dropout(F.relu(self.linear1(x2))))
        x = x + self.dropout2(x2)
        return x

class EstraNet(nn.Module):
    def __init__(self, num_classes=256, d_model=512, n_head=16, n_layers=4, input_len=15000):
        super().__init__()
        
        # Stem: 15000 -> 150
        self.stem = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=500, stride=100, padding=200),
            nn.BatchNorm1d(d_model),
            nn.SELU()
        )
        
        # Positional Encoding
        reduced_len = math.ceil(input_len / 100) + 20 
        self.pos_embed = nn.Parameter(torch.zeros(1, reduced_len, d_model))
        
        # Transformer
        self.layers = nn.ModuleList([
            EstraNetBlock(d_model, n_head, d_inner=d_model*4)
            for _ in range(n_layers)
        ])
        
        self.head = nn.Linear(d_model, num_classes)

        # --- 【关键修改】 初始化权重 ---
        self._init_weights()

    def _init_weights(self):
        """
        显式初始化所有权重。
        特别是将 head 的权重设为极小值，防止初始 Loss 爆炸。
        """
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # Kaiming Init 适合 ReLU/SELU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 【最重要的一步】
        # 将分类头初始化为接近 0，确保初始 Logits 接近 0
        nn.init.normal_(self.head.weight, mean=0, std=0.001)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x):
        x = self.stem(x) 
        x = x.transpose(1, 2) 
        
        B, L, D = x.shape
        x = x + self.pos_embed[:, :L, :]
        
        for layer in self.layers:
            x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        
        x = x.mean(dim=1)
        logits = self.head(x)
        return logits