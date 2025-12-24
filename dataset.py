import torch
import h5py
import numpy as np
from torch.utils.data import Dataset

class ASCADv2Dataset(Dataset):
    def __init__(self, h5_path, split='train', input_len=15000):
        self.h5_path = h5_path
        self.split = split
        self.input_len = input_len
        
        # 1. 打开文件一次获取长度
        with h5py.File(self.h5_path, 'r') as f:
            if split == 'train':
                self.group_name = 'Profiling_traces'
            else:
                self.group_name = 'Attack_traces'
            self.n_samples = f[self.group_name]['traces'].shape[0]

        self.h5_file = None
        self.traces = None
        self.labels = None

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            self.traces = self.h5_file[self.group_name]['traces']
            self.labels = self.h5_file[self.group_name]['labels']

        # 3. 读取 Trace
        trace_raw = self.traces[idx, :self.input_len]
        
        # --- 新增：标准化 (Normalization) ---
        # 这步至关重要！将数据拉回 mean=0, std=1
        # 使用 float64 计算以防溢出，然后转回 float32
        trace_raw = trace_raw.astype(np.float64)
        mean = np.mean(trace_raw)
        std = np.std(trace_raw)
        # 防止除以0
        if std < 1e-10:
            std = 1.0
        trace_norm = (trace_raw - mean) / std
        # ----------------------------------

        # 转为 Tensor
        trace = torch.tensor(trace_norm, dtype=torch.float32).unsqueeze(0)
        
        # 4. 处理 Label (之前的核弹级修复保持不变)
        label_raw = self.labels[idx]
        label_arr = np.array(label_raw)
        if label_arr.dtype.names is not None:
             label_arr = label_arr[label_arr.dtype.names[0]]
        final_val = label_arr.flatten()[0]
        final_label = int(final_val)
        
        return trace, torch.tensor(final_label, dtype=torch.long)