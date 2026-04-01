import scanpy as sc
import scipy.io as sio
import numpy as np
import pandas as pd

def hvg_selection_and_processing(train_exp_paths, all_exp_paths, n_top_genes=1000):
    # 1. 仅使用【训练集】来挑选 HVG，防止数据泄露
    hvg_bools = []
    for d in train_exp_paths:
        adata = sio.mmread(d).toarray()
        adata = sc.AnnData(X=adata.T)
        
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        hvg_bools.append(adata.var['highly_variable'])
    
    # 取训练集 HVG 的并集
    hvg_union = hvg_bools[0]
    for i in range(1, len(hvg_bools)):
        hvg_union = hvg_union | hvg_bools[i]
    print("Number of HVGs from Training Set: ", hvg_union.sum())

    # 2. 对【所有】数据集应用这个 HVG 集合，并在单切片内部进行 Z-score 标准化
    processed_mtxs = []
    for d in all_exp_paths:
        adata = sio.mmread(d).toarray()
        adata = sc.AnnData(X=adata.T)

        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        
        # 只保留 HVG 基因
        adata = adata[:, hvg_union].copy()
        
        # 关键步骤：在单张切片内部进行 Z-score 标准化
        # 这样既消除了单张切片的全局测序深度漂移，又没有引入跨测试集的数据泄露
        sc.pp.scale(adata, max_value=10) 
        
        processed_mtxs.append(adata.X)
        
    return processed_mtxs

# 区分训练集和所有数据集
train_paths = [
    "GSE240429_data/data/filtered_expression_matrices/1/matrix.mtx",
    "GSE240429_data/data/filtered_expression_matrices/2/matrix.mtx",
]

all_paths = [
    "GSE240429_data/data/filtered_expression_matrices/1/matrix.mtx",
    "GSE240429_data/data/filtered_expression_matrices/2/matrix.mtx",
    "GSE240429_data/data/filtered_expression_matrices/3/matrix.mtx",
    "GSE240429_data/data/filtered_expression_matrices/4/matrix.mtx",
]

# 运行处理
final_mtxs = hvg_selection_and_processing(train_paths, all_paths)

# 直接保存，不再使用 Harmony
for i in range(len(final_mtxs)):
    # 保证保存形状为 (genes, spots)，与你之前 dataset.py 期望的一致
    np.save(
        "GSE240429_data/data/filtered_expression_matrices/" + str(i + 1) + "/harmony_matrix.npy", 
        final_mtxs[i].T 
    )
    print(f"Saved Slide {i+1} shape: {final_mtxs[i].T.shape}")