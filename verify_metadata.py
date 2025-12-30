import h5py
import numpy as np
import sys

# AES S-Box 标准表
SBOX = np.array([
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

def check_metadata(file_path):
    print(f"📂 打开文件: {file_path}")
    try:
        with h5py.File(file_path, 'r') as f:
            group = f['Profiling_traces']
            if 'metadata' not in group:
                print("❌ 错误：Profiling_traces 下没有 metadata。")
                return

            meta = group['metadata']
            print(f"\n📋 Metadata 包含的字段 (dtype names):")
            print(meta.dtype.names)
            
            # 1. 提取基础信息
            try:
                plaintext = meta['plaintext'] # shape (N, 16)
                key = meta['key']             # shape (N, 16)
                print(f"✅ 成功加载 Plaintext 和 Key。样本数: {len(plaintext)}")
            except ValueError:
                print("❌ 严重错误：Metadata 中缺少 'plaintext' 或 'key' 字段！")
                return

            # 2. 我们自己计算 Label (Ground Truth)
            # ASCADv2 默认攻击目标：SBox(P[0] ^ K[0])
            print("\n🧮 正在计算我们定义的 Label: SBox[P[0] ^ K[0]] ...")
            pt_byte0 = plaintext[:, 0]
            key_byte0 = key[:, 0]
            calculated_label = SBOX[pt_byte0 ^ key_byte0]
            print(f"   -> 计算完成。前5个值: {calculated_label[:5]}")

            # 3. 暴力比对：Metadata 里有没有字段跟我们算的一样？
            print("\n🔍 开始全字段比对 (寻找是否存在预置的 SBox 输出)...")
            found_match = False
            
            for field_name in meta.dtype.names:
                # 跳过 plaintext 和 key，只看其他未知字段
                if field_name in ['plaintext', 'key']:
                    continue
                
                data = meta[field_name]
                
                # 如果是多字节字段 (例如 masks 是 16字节)，我们逐个字节比对
                if len(data.shape) > 1 and data.shape[1] > 1:
                    for i in range(data.shape[1]):
                        column_data = data[:, i]
                        # 比较
                        if np.array_equal(column_data, calculated_label):
                            print(f"🎯 发现匹配！字段 '{field_name}' 的第 [{i}] 个字节与计算的 Label 完全一致！")
                            found_match = True
                else:
                    # 单字节字段
                    if np.array_equal(data, calculated_label):
                        print(f"🎯 发现匹配！字段 '{field_name}' 与计算的 Label 完全一致！")
                        found_match = True

            if not found_match:
                print("\n⚠️  结论：Metadata 中没有包含显式的 'SBox输出' 字段。")
                print("✅ 确认：我们需要像代码里那样自己计算 (calculated_label) 是完全正确的。")
                
                # 额外检查 Masks
                if 'masks' in meta.dtype.names:
                    print("\n🎭 检查 Masks 字段...")
                    masks = meta['masks']
                    # 检查是否是 Masked SBox: SBox[P^K] ^ M
                    # 通常 ASCADv2 的第一个 Mask 是用来掩码输出的，或者是输入的 mask
                    # 我们可以简单算一下看看
                    masked_label = calculated_label ^ masks[:, 0] # 假设 r_out 是 masks[0]
                    print(f"   Masks[0] 前5个值: {masks[:, 0][:5]}")
                    print(f"   假设 Masked Label (SBox^M[0]) 前5个值: {masked_label[:5]}")
                    
            else:
                print("\n✅ 结论：Metadata 包含了 SBox 输出，我们的计算与之一致。")

    except Exception as e:
        print(f"❌ 发生异常: {e}")

if __name__ == "__main__":
    # 请修改为你的文件路径
    path = "/home/joey1/Documents/joey/Data/ASCAD/ascadv2-extracted.h5" 
    check_metadata(path)