# if __name__ == "__main__":
#     raise SystemExit(main())
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UNI / BLEEP-style retrieval inference script.

用法示例：
python inference_UNI.py run \
  --embeddings-dir /root/disk2/runzhi/BLEEP/result/embeddings/bleep_baseline_uni \
  --base-data-dir /root/disk2/runzhi/BLEEP/GSE240429_data \
  --query-slide 3 \
  --reference-slides 1,2,4 \
  --methods simple,weighted,average \
  --top-k 50 \
  --temp 0.07 \
  --block-size 1024
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================
# 固定数据集元信息（GSE240429）
# =========================
SLIDE_META = {
    "1": {
        "name": "C73_A1",
        "reduced_mtx": "data/filtered_expression_matrices/1/harmony_matrix.npy",
        "barcode": "data/filtered_expression_matrices/1/barcodes.tsv",
        "features": "data/filtered_expression_matrices/1/features.tsv",
    },
    "2": {
        "name": "C73_B1",
        "reduced_mtx": "data/filtered_expression_matrices/2/harmony_matrix.npy",
        "barcode": "data/filtered_expression_matrices/2/barcodes.tsv",
        "features": "data/filtered_expression_matrices/2/features.tsv",
    },
    "3": {
        "name": "C73_C1",
        "reduced_mtx": "data/filtered_expression_matrices/3/harmony_matrix.npy",
        "barcode": "data/filtered_expression_matrices/3/barcodes.tsv",
        "features": "data/filtered_expression_matrices/3/features.tsv",
    },
    "4": {
        "name": "C73_D1",
        "reduced_mtx": "data/filtered_expression_matrices/4/harmony_matrix.npy",
        "barcode": "data/filtered_expression_matrices/4/barcodes.tsv",
        "features": "data/filtered_expression_matrices/4/features.tsv",
    },
}


# =========================
# 基础工具函数
# =========================
def _parse_csv_list(x: str) -> List[str]:
    return [s.strip() for s in str(x).split(",") if s.strip()]


def _as_2d(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={arr.shape}")
    return arr


def load_npy_2d(path: Path, name: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    arr = np.load(path)
    return _as_2d(arr, name)


def ensure_rows_match(arr: np.ndarray, expected_rows: int, name: str) -> np.ndarray:
    """
    允许 shape = [expected_rows, d] 或 [d, expected_rows]。
    """
    arr = _as_2d(arr, name)
    if arr.shape[0] == expected_rows:
        return arr.astype(np.float32, copy=False)
    if arr.shape[1] == expected_rows:
        return arr.T.astype(np.float32, copy=False)
    raise ValueError(
        f"{name} cannot align to expected_rows={expected_rows}, shape={arr.shape}"
    )


def load_barcodes(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Barcode file not found: {path}")
    if path.suffix == ".npy":
        bc = np.load(path, allow_pickle=True)
        return np.asarray(bc).astype(str)
    df = pd.read_csv(path, header=None, sep="\t")
    return df.iloc[:, 0].astype(str).values


def align_rows_by_barcode(
    arr: np.ndarray,
    source_barcodes: np.ndarray,
    target_barcodes: np.ndarray,
    name: str,
) -> Tuple[np.ndarray, Dict]:
    """
    仅在 source/target barcode 数量完全匹配且一一对应时做重排。
    否则不在这里强行对齐。
    """
    arr = _as_2d(arr, name)
    source_barcodes = np.asarray(source_barcodes).astype(str)
    target_barcodes = np.asarray(target_barcodes).astype(str)

    if len(source_barcodes) != arr.shape[0]:
        raise ValueError(
            f"{name}: len(source_barcodes)={len(source_barcodes)} != arr.rows={arr.shape[0]}"
        )

    src_map = {b: i for i, b in enumerate(source_barcodes)}
    idx = []
    missing = []
    for b in target_barcodes:
        if b in src_map:
            idx.append(src_map[b])
        else:
            missing.append(b)

    if len(missing) > 0:
        raise ValueError(
            f"{name}: cannot align by barcode, missing {len(missing)} barcodes"
        )

    reordered = arr[np.asarray(idx, dtype=np.int64), :]
    report = {
        "source_n": int(len(source_barcodes)),
        "target_n": int(len(target_barcodes)),
        "missing_n": int(len(missing)),
    }
    return reordered, report


# =========================
# 关键修复：从 embeddings + expr 共同推断 n_spots
# =========================
def infer_n_spots_from_arrays(
    img_raw: np.ndarray,
    spot_raw: np.ndarray,
    expr_raw: np.ndarray,
    slide: str,
    barcode_count: Optional[int] = None,
) -> int:
    """
    从 img/spot embeddings 和 expression matrix 共同推断 n_spots。
    优先用三者共同维度；如果不唯一，再尝试用 barcode_count 消歧。
    """
    img_raw = _as_2d(img_raw, f"img_embeddings_{slide}")
    spot_raw = _as_2d(spot_raw, f"spot_embeddings_{slide}")
    expr_raw = _as_2d(expr_raw, f"expression_{slide}")

    common = set(img_raw.shape) & set(spot_raw.shape) & set(expr_raw.shape)

    if len(common) == 1:
        return int(next(iter(common)))

    if barcode_count is not None and barcode_count in common:
        return int(barcode_count)

    raise ValueError(
        f"Cannot infer unique n_spots for slide {slide}. "
        f"img={img_raw.shape}, spot={spot_raw.shape}, expr={expr_raw.shape}, "
        f"barcode_count={barcode_count}, common={sorted(common)}"
    )


def load_slide_triplet(
    embeddings_dir: Path,
    base_data_dir: Path,
    slide: str,
    slide_meta: dict,
    emb_barcodes_dir: Optional[Path] = None,
):
    """
    统一读取一个 slide 的：
    - img embeddings
    - spot embeddings
    - expression
    并自动推断 n_spots，再做 shape 对齐。
    barcodes.tsv 只做校验，不再作为主基准。
    """
    img_path = embeddings_dir / f"img_embeddings_{slide}.npy"
    spot_path = embeddings_dir / f"spot_embeddings_{slide}.npy"
    expr_path = base_data_dir / slide_meta["reduced_mtx"]

    img_raw = load_npy_2d(img_path, f"img_embeddings_{slide}")
    spot_raw = load_npy_2d(spot_path, f"spot_embeddings_{slide}")
    expr_raw = load_npy_2d(expr_path, f"expression({slide_meta['reduced_mtx']})")

    bc_path = base_data_dir / slide_meta["barcode"]
    barcodes_expr = None
    barcode_count = None
    if bc_path.exists():
        try:
            barcodes_expr = load_barcodes(bc_path)
            barcode_count = int(len(barcodes_expr))
        except Exception as e:
            print(f"[WARN] slide {slide}: failed to load barcodes from {bc_path}: {e}")
            barcodes_expr = None
            barcode_count = None

    n_spots = infer_n_spots_from_arrays(
        img_raw=img_raw,
        spot_raw=spot_raw,
        expr_raw=expr_raw,
        slide=slide,
        barcode_count=barcode_count,
    )

    img = ensure_rows_match(img_raw, expected_rows=n_spots, name=f"img_embeddings_{slide}")
    spot = ensure_rows_match(spot_raw, expected_rows=n_spots, name=f"spot_embeddings_{slide}")
    expr = ensure_rows_match(expr_raw, expected_rows=n_spots, name=f"expression({slide_meta['reduced_mtx']})")

    alignment_report = {
        "slide": slide,
        "n_spots_inferred": int(n_spots),
        "img_shape_raw": list(img_raw.shape),
        "spot_shape_raw": list(spot_raw.shape),
        "expr_shape_raw": list(expr_raw.shape),
        "img_shape_final": list(img.shape),
        "spot_shape_final": list(spot.shape),
        "expr_shape_final": list(expr.shape),
    }

    if barcodes_expr is not None:
        alignment_report["barcode_rows"] = int(len(barcodes_expr))
        if len(barcodes_expr) != n_spots:
            print(
                f"[WARN] slide {slide}: barcode rows={len(barcodes_expr)} != inferred n_spots={n_spots}. "
                f"Will skip barcode-based strict alignment for this slide."
            )
            barcodes_expr = None

    if emb_barcodes_dir is not None and barcodes_expr is not None:
        emb_barcodes = None
        cand_npy = emb_barcodes_dir / f"barcodes_{slide}.npy"
        cand_tsv = emb_barcodes_dir / f"barcodes_{slide}.tsv"

        if cand_npy.exists():
            emb_barcodes = load_barcodes(cand_npy)
        elif cand_tsv.exists():
            emb_barcodes = load_barcodes(cand_tsv)

        if emb_barcodes is not None:
            if len(emb_barcodes) != n_spots:
                print(
                    f"[WARN] slide {slide}: embedding barcode rows={len(emb_barcodes)} "
                    f"!= inferred n_spots={n_spots}. Skip barcode reordering."
                )
            else:
                img, rep_img = align_rows_by_barcode(
                    img, emb_barcodes, barcodes_expr, name=f"img_embeddings_{slide}"
                )
                spot, rep_spot = align_rows_by_barcode(
                    spot, emb_barcodes, barcodes_expr, name=f"spot_embeddings_{slide}"
                )
                alignment_report["img_barcode_align"] = rep_img
                alignment_report["spot_barcode_align"] = rep_spot

    return {
        "img": img,
        "spot": spot,
        "expr": expr,
        "barcodes": barcodes_expr,
        "alignment_report": alignment_report,
    }


# =========================
# 检索与聚合
# =========================
def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def softmax_np(x: np.ndarray, axis: int = -1, temp: float = 1.0) -> np.ndarray:
    x = x / max(float(temp), 1e-8)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


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
        q = image_query[start:end]                                  # [b, d]
        sim = q @ spot_key.T                                        # [b, n_ref]

        idx = np.argpartition(sim, -top_k, axis=1)[:, -top_k:]      # unsorted top-k
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
    """
    返回：
    - pred_spot_emb: [n_query, d]
    - pred_expr:     [n_query, g]
    """
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

    raise ValueError(f"Unknown retrieval method: {method}")


# =========================
# 指标
# =========================
def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    c = np.corrcoef(x, y)[0, 1]
    if np.isnan(c):
        return np.nan
    return float(c)


def compute_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    query_slide: str,
    base_data_dir: Path,
    marker_gene_list: Optional[List[str]] = None,
) -> Dict:
    pred = _as_2d(pred, "pred")
    true = _as_2d(true, "true")

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

    if marker_gene_list is None:
        marker_gene_list = ["HAL", "CYP3A4", "VWF", "SOX9", "KRT7", "ANXA4", "ACTA2", "DCN"]

    marker_gene_pcc = np.nan
    marker_hit_genes = []

    features_path = base_data_dir / SLIDE_META[query_slide]["features"]
    hvg_union_path = base_data_dir / "data/filtered_expression_matrices/hvg_union.npy"

    if features_path.exists():
        gene_names_all = pd.read_csv(features_path, header=None, sep="\t").iloc[:, 1].astype(str).values
        gene_names_eval = None

        if len(gene_names_all) == true.shape[1]:
            gene_names_eval = gene_names_all
        elif hvg_union_path.exists():
            hvg_union = np.load(hvg_union_path).astype(int)
            if len(hvg_union) == true.shape[1] and np.max(hvg_union) < len(gene_names_all):
                gene_names_eval = gene_names_all[hvg_union]

        if gene_names_eval is not None and len(gene_names_eval) == true.shape[1]:
            marker_idx = []
            for g in marker_gene_list:
                hits = np.where(gene_names_eval == g)[0]
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
        "pred_mean_gene_std_across_spots": float(np.mean(np.std(pred, axis=0))),
        "true_mean_gene_std_across_spots": float(np.mean(np.std(true, axis=0))),
        "pred_mean_spot_std_across_genes": float(np.mean(np.std(pred, axis=1))),
        "true_mean_spot_std_across_genes": float(np.mean(np.std(true, axis=1))),
    }
    return metrics


# =========================
# 主流程
# =========================
def run_pipeline(args):
    embeddings_dir = Path(args.embeddings_dir)
    base_data_dir = Path(args.base_data_dir)
    query_slide = str(args.query_slide)
    reference_slides = [str(s) for s in _parse_csv_list(args.reference_slides)]
    methods = [m.strip().lower() for m in _parse_csv_list(args.methods)]
    emb_barcodes_dir = Path(args.emb_barcodes_dir) if args.emb_barcodes_dir else None

    if query_slide not in SLIDE_META:
        raise ValueError(f"Unknown query slide: {query_slide}")
    for s in reference_slides:
        if s not in SLIDE_META:
            raise ValueError(f"Unknown reference slide: {s}")
    if query_slide in reference_slides:
        raise ValueError("query_slide should not be included in reference_slides")

    all_slides = reference_slides + [query_slide]

    # 1) 逐 slide 读取
    per_slide = {}
    alignment_reports = {}

    for s in all_slides:
        loaded = load_slide_triplet(
            embeddings_dir=embeddings_dir,
            base_data_dir=base_data_dir,
            slide=s,
            slide_meta=SLIDE_META[s],
            emb_barcodes_dir=emb_barcodes_dir,
        )

        per_slide[s] = {
            "img": loaded["img"],
            "spot": loaded["spot"],
            "expr": loaded["expr"],
            "barcodes": loaded["barcodes"],
        }
        alignment_reports[s] = loaded["alignment_report"]

    # 2) 组装 query / reference
    image_query = per_slide[query_slide]["img"]
    expression_gt = per_slide[query_slide]["expr"]

    spot_key = np.concatenate([per_slide[s]["spot"] for s in reference_slides], axis=0)
    expression_key = np.concatenate([per_slide[s]["expr"] for s in reference_slides], axis=0)

    assert image_query.shape[0] == expression_gt.shape[0], (
        f"Query size mismatch: image_query={image_query.shape}, expression_gt={expression_gt.shape}"
    )

    # 3) retrieval
    indices, topk_similarity = find_matches_blockwise(
        spot_key=spot_key,
        image_query=image_query,
        top_k=args.top_k,
        block_size=args.block_size,
    )

    # 4) 输出目录
    out_root = embeddings_dir / f"inference_query{query_slide}_ref{'-'.join(reference_slides)}"
    out_root.mkdir(parents=True, exist_ok=True)

    # 保存全局检索结果
    np.save(out_root / "matched_indices.npy", indices)
    np.save(out_root / "topk_similarity.npy", topk_similarity)

    # 5) 逐方法聚合 + 评测
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

        method_dir = out_root / method
        method_dir.mkdir(parents=True, exist_ok=True)

        np.save(method_dir / "matched_spot_embeddings_pred.npy", pred_spot_emb.T)
        np.save(method_dir / "matched_spot_expression_pred.npy", pred_expr.T)

        metrics = compute_metrics(
            pred=pred_expr,
            true=expression_gt,
            query_slide=query_slide,
            base_data_dir=base_data_dir,
        )
        all_metrics[method] = metrics

        with open(method_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 6) 保存报告
    report = {
        "query_slide": query_slide,
        "reference_slides": reference_slides,
        "methods": methods,
        "top_k": int(args.top_k),
        "temp": float(args.temp),
        "block_size": int(args.block_size),
        "alignment_reports": alignment_reports,
        "shapes": {
            "image_query": list(image_query.shape),
            "spot_key": list(spot_key.shape),
            "expression_gt": list(expression_gt.shape),
            "expression_key": list(expression_key.shape),
        },
        "metrics": all_metrics,
    }

    with open(out_root / "run_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 7) 打印摘要
    print("=" * 60)
    print("Inference summary")
    print("=" * 60)
    print(f"embeddings_dir: {embeddings_dir}")
    print(f"base_data_dir: {base_data_dir}")
    print(f"query_slide: {query_slide}")
    print(f"reference_slides: {reference_slides}")
    print(f"methods: {methods}")
    print(f"top_k: {args.top_k}, temp: {args.temp}, block_size: {args.block_size}")
    print(f"output_dir: {out_root}")
    print()

    print("Alignment reports:")
    for s in all_slides:
        print(f"  slide {s}: {alignment_reports[s]}")

    print()
    print("Metric summary:")
    for method, metrics in all_metrics.items():
        print(f"  [{method}]")
        for k, v in metrics.items():
            print(f"    {k}: {v}")
    print("=" * 60)

    matched_indices = np.load(out_root / "matched_indices.npy")
    top1 = matched_indices[:, 0]
    print("n_query:", len(top1))
    print("n_unique_top1:", len(np.unique(top1)))
    print("top1 reuse ratio:", len(top1) / max(len(np.unique(top1)), 1))

    print("image_query mean std across spots:", float(np.mean(np.std(image_query, axis=0))))
    print("spot_key mean std across spots:", float(np.mean(np.std(spot_key, axis=0))))


# =========================
# CLI
# =========================
def build_parser():
    parser = argparse.ArgumentParser("BLEEP / UNI retrieval inference")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run retrieval inference")
    run.add_argument("--embeddings-dir", type=str, required=True)
    run.add_argument("--base-data-dir", type=str, required=True)
    run.add_argument("--query-slide", type=str, required=True)
    run.add_argument("--reference-slides", type=str, required=True)
    run.add_argument("--methods", type=str, default="simple,weighted,average")
    run.add_argument("--top-k", type=int, default=50)
    run.add_argument("--temp", type=float, default=0.07)
    run.add_argument("--block-size", type=int, default=1024)
    run.add_argument(
        "--emb-barcodes-dir",
        type=str,
        default=None,
        help="optional directory containing barcodes_<slide>.npy/tsv for embeddings-side strict reordering",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        run_pipeline(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()