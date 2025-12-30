import h5py
import numpy as np

# 旧服务器上的路径
DATA_PATH = "/home/joey1/Documents/joey/Data/ASCAD/ascadv2-extracted.h5"

def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"📄 Dataset: {name} | Shape: {obj.shape} | Type: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"📂 Group: {name}")

print(f"🔍 Inspecting: {DATA_PATH}")

try:
    with h5py.File(DATA_PATH, "r") as f:
        # 遍历打印所有层级结构
        f.visititems(print_structure)
        
        # 重点检查 Attack_traces (通常用作 Test)
        if 'Attack_traces' in f:
            traces = f['Attack_traces']['traces']
            print(f"\n✅ Found 'Attack_traces' (Test Set): {traces.shape[0]} traces.")
        
        # 重点检查 Profiling_traces (通常用作 Train)
        if 'Profiling_traces' in f:
            traces = f['Profiling_traces']['traces']
            print(f"✅ Found 'Profiling_traces' (Train Set): {traces.shape[0]} traces.")

except Exception as e:
    print(f"❌ Error reading file: {e}")