#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用 extract_embeddings_uni.py 导出的严格对齐文件做 retrieval 和评测。

输入目录结构示例：
embeddings_root/
  slide_1/
    img_embeddings.npy
    spot_embeddings.npy
    expr.npy
    barcodes.npy
    gene_names.npy
    meta.json
  slide_2/
  slide_3/
  slide_4/

用法示例：
python run_inference_uni.py \
  --embeddings-root /root/disk2/runzhi/BLEEP/result/embeddings/bleep_baseline_uni_clean \
  --query-slide 3 \
  --reference-slides 1,2,4 \
  --methods simple,weighted,average \
  --top-k 50 \
  --temp 0.07 \
  --block-size 1024
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_csv_list(x: str) -> List[str]:
    return [s.strip() for s in str(x).split(",") if s.strip()]


def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def softmax_np(x: np.ndarray, axis: int = -1, temp: float = 1.0) -> np.ndarray:
    x = x / max(float(temp), 1e-8)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def load_slide_package(root: Path, slide: str) -> Dict:
    slide_dir = root / f"slide_{slide}"
    if not slide_dir.exists():
        raise FileNotFoundError(f"Slide package not found: {slide_dir}")

    img = np.load(slide_dir / "img_embeddings.npy")
    spot = np.load(slide_dir / "spot_embeddings.npy")
    expr = np.load(slide_dir / "expr.npy")
    barcodes = np.load(slide_dir / "barcodes.npy", allow_pickle=True)
    gene_names = np.load(slide_dir / "gene_names.npy", allow_pickle=True)
    meta_path = slide_dir / "meta.json"

    if img.ndim != 2 or spot.ndim != 2 or expr.ndim != 2:
        raise ValueError(f"Slide {slide}: img/spot/expr must be 2D")

    n = img.shape[0]
    if not (spot.shape[0] == n and expr.shape[0] == n and len(barcodes) == n):
        raise ValueError(
            f"Slide {slide}: row mismatch "
            f"img={img.shape}, spot={spot.shape}, expr={expr.shape}, barcodes={len(barcodes)}"
        )

    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return {
        "img": img.astype(np.float32, copy=False),
        "spot": spot.astype(np.float32, copy=False),
        "expr": expr.astype(np.float32, copy=False),
        "barcodes": np.asarray(barcodes).astype(str),
        "gene_names": np.asarray(gene_names).astype(str),
        "meta": meta,
        "slide_dir": str(slide_dir),
    }


def find_matches_blockwise(
    spot_key: np.ndarray,
    image_query: np.ndarray,
    top_k: int = 50,
    block_size: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    block-wise cosine retrieval
    返回：
    - indices: [n_query, top_k]
    - topk_similarity: [n_query, top_k]
    """
    spot_key = l2_normalize(spot_key.astype(np.float32, copy=False), axis=1)
    image_query = l2_normalize(image_query.astype(np.float32, copy=False), axis=1)

    n_query = image_query.shape[0]
    n_ref = spot_key.shape[0]
    top_k = int(min(top_k, n_ref))

    all_indices = []
    all_scores = []

    for start in range(0, n_query, block_size):
        end = min(start + block_size, n_query)
        q = image_query[start:end]     # [b, d]
        sim = q @ spot_key.T           # [b, n_ref]

        idx = np.argpartition(sim, -top_k, axis=1)[:, -top_k:]
        part = np.take_along_axis(sim, idx, axis=1)

        order = np.argsort(part, axis=1)[:, ::-1]
        idx = np.take_along_axis(idx, order, axis=1)
        part = np.take_along_axis(part, order, axis=1)

        all_indices.append(idx.astype(np.int64, copy=False))
        all_scores.append(part.astype(np.float32, copy=False))

    indices = np.concatenate(all_indices, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    return indices, scores


def aggregate_predictions(
    method: str,
    indices: np.ndarray,
    topk_similarity: np.ndarray,
    spot_key: np.ndarray,
    expression_key: np.ndarray,
    temp: float = 0.07,
) -> Tuple[np.ndarray, np.ndarray]:
    method = str(method).lower()
    n_query = indices.shape[0]

    if method == "simple":
        pred_spot_emb = spot_key[indices[:, 0], :]
        pred_expr = expression_key[indices[:, 0], :]
        return pred_spot_emb.astype(np.float32, copy=False), pred_expr.astype(np.float32, copy=False)

    if method == "average":
        pred_spot_emb = np.stack([spot_key[idx].mean(axis=0) for idx in indices], axis=0)
        pred_expr = np.stack([expression_key[idx].mean(axis=0) for idx in indices], axis=0)
        return pred_spot_emb.astype(np.float32, copy=False), pred_expr.astype(np.float32, copy=False)

    if method == "weighted":
        weights = softmax_np(topk_similarity, axis=1, temp=temp)
        pred_spot_emb = np.zeros((n_query, spot_key.shape[1]), dtype=np.float32)
        pred_expr = np.zeros((n_query, expression_key.shape[1]), dtype=np.float32)

        for i in range(n_query):
            idx = indices[i]
            w = weights[i]
            pred_spot_emb[i, :] = np.average(spot_key[idx], axis=0, weights=w)
            pred_expr[i, :] = np.average(expression_key[idx], axis=0, weights=w)

        return pred_spot_emb, pred_expr

    raise ValueError(f"Unknown method: {method}")


def safe_corrcoef(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("safe_corrcoef expects 1D arrays")
    if np.std(x) < eps or np.std(y) < eps:
        return np.nan
    c = np.corrcoef(x, y)[0, 1]
    if np.isnan(c):
        return np.nan
    return float(c)


def compute_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    gene_names: np.ndarray,
    marker_gene_list: List[str],
) -> Dict:
    if pred.shape != true.shape:
        raise ValueError(f"pred.shape={pred.shape} != true.shape={true.shape}")

    spot_pcc_all = np.array(
        [safe_corrcoef(pred[i, :], true[i, :]) for i in range(pred.shape[0])],
        dtype=np.float32,
    )
    gene_pcc_all = np.array(
        [safe_corrcoef(pred[:, j], true[:, j]) for j in range(pred.shape[1])],
        dtype=np.float32,
    )

    spot_pcc = float(np.nanmean(spot_pcc_all))
    gene_pcc = float(np.nanmean(gene_pcc_all))

    top_hvg_n = int(min(1000, true.shape[1]))
    top_hvg_idx = np.argsort(np.var(true, axis=0))[-top_hvg_n:]
    top_hvg_pcc = float(np.nanmean(gene_pcc_all[top_hvg_idx]))

    marker_gene_pcc = np.nan
    marker_hit_genes = []
    marker_idx = []
    if gene_names is not None and len(gene_names) == true.shape[1]:
        gene_names_upper = np.asarray([g.upper() for g in gene_names], dtype=object)
        for g in marker_gene_list:
            hits = np.where(gene_names_upper == g.upper())[0]
            if len(hits) > 0:
                marker_idx.append(int(hits[0]))
                marker_hit_genes.append(g)
        if len(marker_idx) > 0:
            marker_gene_pcc = float(np.nanmean(gene_pcc_all[np.array(marker_idx, dtype=int)]))

    metrics = {
        "gene_wise_pcc": gene_pcc,
        "spot_wise_pcc": spot_pcc,
        "top_hvg_pcc": top_hvg_pcc,
        "marker_gene_pcc": marker_gene_pcc,
        "marker_hit_genes": marker_hit_genes,
        "n_query_spots": int(pred.shape[0]),
        "n_genes": int(pred.shape[1]),
        "top_hvg_n": int(top_hvg_n),
        "n_valid_gene_pcc": int(np.sum(~np.isnan(gene_pcc_all))),
        "n_valid_spot_pcc": int(np.sum(~np.isnan(spot_pcc_all))),
        "pred_mean_gene_std_across_spots": float(np.mean(np.std(pred, axis=0))),
        "true_mean_gene_std_across_spots": float(np.mean(np.std(true, axis=0))),
        "pred_mean_spot_std_across_genes": float(np.mean(np.std(pred, axis=1))),
        "true_mean_spot_std_across_genes": float(np.mean(np.std(true, axis=1))),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser("Run UNI retrieval inference from clean slide packages")
    parser.add_argument("--embeddings-root", type=str, required=True)
    parser.add_argument("--query-slide", type=str, required=True)
    parser.add_argument("--reference-slides", type=str, required=True)
    parser.add_argument("--methods", type=str, default="simple,weighted,average")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--temp", type=float, default=0.07)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument(
        "--marker-genes",
        type=str,
        default="HAL,CYP3A4,VWF,SOX9,KRT7,ANXA4,ACTA2,DCN",
    )
    args = parser.parse_args()

    root = Path(args.embeddings_root)
    query_slide = str(args.query_slide)
    reference_slides = parse_csv_list(args.reference_slides)
    methods = [m.lower() for m in parse_csv_list(args.methods)]
    marker_gene_list = parse_csv_list(args.marker_genes)

    all_slides = reference_slides + [query_slide]
    per_slide = {s: load_slide_package(root, s) for s in all_slides}

    # query / reference
    image_query = per_slide[query_slide]["img"]
    expression_gt = per_slide[query_slide]["expr"]
    query_barcodes = per_slide[query_slide]["barcodes"]
    gene_names = per_slide[query_slide]["gene_names"]

    spot_key = np.concatenate([per_slide[s]["spot"] for s in reference_slides], axis=0)
    expression_key = np.concatenate([per_slide[s]["expr"] for s in reference_slides], axis=0)
    ref_barcodes = np.concatenate([per_slide[s]["barcodes"] for s in reference_slides], axis=0)
    ref_slide_ids = np.concatenate(
        [np.asarray([s] * per_slide[s]["spot"].shape[0], dtype=object) for s in reference_slides],
        axis=0,
    )

    # retrieval
    indices, topk_similarity = find_matches_blockwise(
        spot_key=spot_key,
        image_query=image_query,
        top_k=args.top_k,
        block_size=args.block_size,
    )

    # 输出目录
    out_dir = root / f"inference_query{query_slide}_ref{'-'.join(reference_slides)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "matched_indices.npy", indices)
    np.save(out_dir / "topk_similarity.npy", topk_similarity)

    # top1 诊断
    top1 = indices[:, 0]
    n_unique_top1 = len(np.unique(top1))
    top1_reuse_ratio = len(top1) / max(n_unique_top1, 1)

    top1_ref_barcodes = ref_barcodes[top1]
    top1_ref_slides = ref_slide_ids[top1]

    np.save(out_dir / "query_barcodes.npy", query_barcodes)
    np.save(out_dir / "reference_barcodes.npy", ref_barcodes)
    np.save(out_dir / "top1_ref_barcodes.npy", top1_ref_barcodes)
    np.save(out_dir / "top1_ref_slides.npy", top1_ref_slides)

    all_metrics = {}
    for method in methods:
        pred_spot_emb, pred_expr = aggregate_predictions(
            method=method,
            indices=indices,
            topk_similarity=topk_similarity,
            spot_key=spot_key,
            expression_key=expression_key,
            temp=args.temp,
        )

        method_dir = out_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)

        np.save(method_dir / "pred_spot_embeddings.npy", pred_spot_emb)
        np.save(method_dir / "pred_expr.npy", pred_expr)
        np.save(method_dir / "true_expr.npy", expression_gt)

        metrics = compute_metrics(
            pred=pred_expr,
            true=expression_gt,
            gene_names=gene_names,
            marker_gene_list=marker_gene_list,
        )
        all_metrics[method] = metrics

        with open(method_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    report = {
        "query_slide": query_slide,
        "reference_slides": reference_slides,
        "methods": methods,
        "top_k": int(args.top_k),
        "temp": float(args.temp),
        "block_size": int(args.block_size),
        "shapes": {
            "image_query": list(image_query.shape),
            "spot_key": list(spot_key.shape),
            "expression_gt": list(expression_gt.shape),
            "expression_key": list(expression_key.shape),
        },
        "diagnostics": {
            "n_query": int(len(top1)),
            "n_unique_top1": int(n_unique_top1),
            "top1_reuse_ratio": float(top1_reuse_ratio),
            "image_query_mean_std_across_spots": float(np.mean(np.std(image_query, axis=0))),
            "spot_key_mean_std_across_spots": float(np.mean(np.std(spot_key, axis=0))),
        },
        "metrics": all_metrics,
    }

    with open(out_dir / "run_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("Inference summary")
    print("=" * 60)
    print(f"embeddings_root: {root}")
    print(f"query_slide: {query_slide}")
    print(f"reference_slides: {reference_slides}")
    print(f"methods: {methods}")
    print(f"top_k: {args.top_k}, temp: {args.temp}, block_size: {args.block_size}")
    print(f"output_dir: {out_dir}")
    print()
    print("Per-slide meta:")
    for s in all_slides:
        print(f"  slide {s}: {per_slide[s]['meta']}")
    print()
    print("Diagnostics:")
    print(f"  n_query: {len(top1)}")
    print(f"  n_unique_top1: {n_unique_top1}")
    print(f"  top1_reuse_ratio: {top1_reuse_ratio}")
    print(f"  image_query_mean_std_across_spots: {np.mean(np.std(image_query, axis=0))}")
    print(f"  spot_key_mean_std_across_spots: {np.mean(np.std(spot_key, axis=0))}")
    print()
    print("Metric summary:")
    for method, metrics in all_metrics.items():
        print(f"  [{method}]")
        for k, v in metrics.items():
            print(f"    {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()