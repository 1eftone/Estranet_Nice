import h5py
import numpy as np
import sys
import os

# ================= 配置区域 =================
# 将此处修改为您的 .h5 文件路径
FILE_PATH = "/home/joey1/Documents/joey/Data/ASCAD/ascadv2-extracted.h5"  
# ===========================================

def check_dataset(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件 {file_path}")
        return

    print(f"🔍 正在检查文件: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            # 1. 打印文件顶层结构，帮助定位
            print("\n📂 文件结构 (Keys):", list(f.keys()))
            
            # 常见的 ASCAD 组名
            groups_to_check = ['Profiling_traces', 'Attack_traces']
            
            for group_name in groups_to_check:
                if group_name not in f:
                    continue
                
                print(f"\n{'='*40}")
                print(f"🧐 正在分析组: [{group_name}]")
                print(f"{'='*40}")
                
                group = f[group_name]
                
                # --- A. 尝试获取 Label ---
                labels = None
                
                # 情况 1: Label 在 metadata 中 (标准 ASCAD 格式)
                if 'metadata' in group:
                    meta = group['metadata']
                    # metadata 通常是一个结构化数组，查看它的字段名
                    if meta.dtype.names and 'label' in meta.dtype.names:
                        print("✅ 在 metadata 中找到 'label' 字段")
                        labels = meta['label']
                    elif meta.dtype.names and 'prediction' in meta.dtype.names:
                        # 有些 extracted 版本把 label 叫 prediction
                         print("✅ 在 metadata 中找到 'prediction' 字段")
                         labels = meta['prediction']
                    # 如果是原始 ASCADv2，可能没有直接的 label，只有 plaintext 和 key
                    elif meta.dtype.names and 'plaintext' in meta.dtype.names and 'key' in meta.dtype.names:
                        print("⚠️ metadata 中未直接发现 'label'。正在尝试根据 Byte 0 动态计算...")
                        # 简单的 AES Sbox (用于计算 ASCADv2 Byte 0)
                        sbox = np.array([
                            0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
                            0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
                            0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
                            0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
                            0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
                            0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
                            0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
                            0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
                            0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
                            0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
                            0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
                            0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
                            0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
                            0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
                            0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
                            0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16 
                        ])
                        # Profiling Byte 0: Sbox(P[0] ^ K[0])
                        # 注意：这里假设你想看的是 Byte 0。如果是 Byte 2，请修改索引。
                        # 对于 Variable Key，我们看 ID (Share 1 ^ Share 2) 还是 Value？
                        # ASCADv2 通常攻击的是第一个 SBox 输出的 masked value。
                        # 这里我们简单计算 unmasked value 用于检查分布
                        pt = meta['plaintext'][:, 0] # Byte 0
                        key = meta['key'][:, 0]      # Byte 0
                        labels = sbox[pt ^ key]
                        print("⚡ 动态计算 Label 完成 (Target: SBox(P[0]^K[0]))")

                # 情况 2: Label 是独立的 dataset (如 Y_profiling)
                elif 'labels' in group:
                    print("✅ 找到独立的 'labels' 数据集")
                    labels = group['labels'][:]
                
                # --- B. 统计分布 ---
                if labels is not None:
                    print(f"📊 总样本数: {len(labels)}")
                    
                    # 检查 NaN
                    if np.issubdtype(labels.dtype, np.number):
                         if np.isnan(labels).any():
                             print("❌ 警告：Label 中发现 NaN 值！")
                    
                    # 统计频率
                    unique, counts = np.unique(labels, return_counts=True)
                    
                    print(f"🔢 Label 种类数量: {len(unique)} (应为 256)")
                    print(f"📉 最小样本数的 Label: {unique[np.argmin(counts)]} (Count: {np.min(counts)})")
                    print(f"📈 最大样本数的 Label: {unique[np.argmax(counts)]} (Count: {np.max(counts)})")
                    print(f"⚖️ 平均每类样本数: {np.mean(counts):.2f}")
                    
                    if len(unique) < 256:
                        print("⚠️ 警告: Label 种类少于 256，可能存在缺类！")
                    
                    # 简单的直方图展示
                    print("\n分布概览 (前10个 Label):")
                    for u, c in zip(unique[:10], counts[:10]):
                        print(f"  Label {u:3d}: {c} traces")
                    print("  ...")
                else:
                    print("❌ 未能找到或计算出 Label。请检查 dataset.py 中的读取逻辑。")

                # --- C. 检查 Traces 是否有 NaN (Null Traces) ---
                if 'traces' in group:
                    traces = group['traces']
                    print(f"\n🌊 正在检查 Traces 数据完整性 (Shape: {traces.shape})...")
                    # 为了速度，只检查前 1000 条和后 1000 条
                    check_subset = traces[:1000]
                    if np.isnan(check_subset).any():
                         print("❌ 严重警告：Traces (头部) 中发现 NaN！")
                    else:
                         print("✅ Traces (头部) 数据正常 (无 NaN)。")
                         
    except Exception as e:
        print(f"❌ 读取错误: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        FILE_PATH = sys.argv[1]
    
    check_dataset(FILE_PATH)