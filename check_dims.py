import os
import numpy as np
import pandas as pd

# 配置你的基础路径
base_dir = "/root/disk2/runzhi/BLEEP/GSE240429_data/data/filtered_expression_matrices"

print("-" * 60)
print(f"{'Slide':<10} | {'Barcodes (tsv)':<15} | {'Harmony (.npy)':<15} | {'HVG (.npy)':<15}")
print("-" * 60)

for slide in ["1", "2", "3", "4"]:
    slide_dir = os.path.join(base_dir, slide)
    
    # 1. 检查 barcodes 数量
    barcode_file = os.path.join(slide_dir, "barcodes.tsv")
    if os.path.exists(barcode_file):
        # 很多 barcode 文件没有 header，所以直接读取所有行
        with open(barcode_file, 'r') as f:
            num_barcodes = sum(1 for line in f if line.strip())
    else:
        num_barcodes = "Missing"

    # 2. 检查 harmony_matrix.npy 的细胞数 (行数)
    harmony_file = os.path.join(slide_dir, "harmony_matrix.npy")
    if os.path.exists(harmony_file):
        try:
            harmony_mat = np.load(harmony_file)
            num_harmony = harmony_mat.shape[0]
        except Exception:
            num_harmony = "Load Error"
    else:
        num_harmony = "Missing"

    # 3. 检查 hvg_matrix.npy 的细胞数 (行数)
    hvg_file = os.path.join(slide_dir, "hvg_matrix.npy")
    if os.path.exists(hvg_file):
        try:
            hvg_mat = np.load(hvg_file)
            num_hvg = hvg_mat.shape[0]
        except Exception:
            num_hvg = "Load Error"
    else:
        num_hvg = "Missing"

    print(f"Slide {slide:<4} | {str(num_barcodes):<15} | {str(num_harmony):<15} | {str(num_hvg):<15}")

print("-" * 60)