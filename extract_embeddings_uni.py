#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
逐 slide 导出 UNI checkpoint 的：
- img embeddings
- spot embeddings
- expr
- barcodes
- gene_names

输出目录结构示例：
out_dir/
  slide_1/
    img_embeddings.npy
    spot_embeddings.npy
    expr.npy
    barcodes.npy
    gene_names.npy
    meta.json
  slide_2/
    ...
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# 大图 WSI 保护关闭
Image.MAX_IMAGE_PIXELS = None

from dataset import CLIPDataset
from models import CLIPModel_UNI
from utils import load_model_checkpoint


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SLIDE_META = {
    "1": {
        "name": "C73_A1",
        "image": "images/GEX_C73_A1_Merged.tif",
        "spatial": "data/tissue_pos_matrices/tissue_positions_list_1.csv",
        "barcode": "data/filtered_expression_matrices/1/barcodes.tsv",
        "reduced_mtx": "data/filtered_expression_matrices/1/harmony_matrix.npy",
        "features": "data/filtered_expression_matrices/1/features.tsv",
    },
    "2": {
        "name": "C73_B1",
        "image": "images/GEX_C73_B1_Merged.tif",
        "spatial": "data/tissue_pos_matrices/tissue_positions_list_2.csv",
        "barcode": "data/filtered_expression_matrices/2/barcodes.tsv",
        "reduced_mtx": "data/filtered_expression_matrices/2/harmony_matrix.npy",
        "features": "data/filtered_expression_matrices/2/features.tsv",
    },
    "3": {
        "name": "C73_C1",
        "image": "images/GEX_C73_C1_Merged.tif",
        "spatial": "data/tissue_pos_matrices/tissue_positions_list_3.csv",
        "barcode": "data/filtered_expression_matrices/3/barcodes.tsv",
        "reduced_mtx": "data/filtered_expression_matrices/3/harmony_matrix.npy",
        "features": "data/filtered_expression_matrices/3/features.tsv",
    },
    "4": {
        "name": "C73_D1",
        "image": "images/GEX_C73_D1_Merged.tif",
        "spatial": "data/tissue_pos_matrices/tissue_positions_list_4.csv",
        "barcode": "data/filtered_expression_matrices/4/barcodes.tsv",
        "reduced_mtx": "data/filtered_expression_matrices/4/harmony_matrix.npy",
        "features": "data/filtered_expression_matrices/4/features.tsv",
    },
}


def args_to_dict(args_obj):
    if isinstance(args_obj, dict):
        return args_obj
    if hasattr(args_obj, "__dict__"):
        return dict(vars(args_obj))
    return {}


def build_uni_model_from_checkpoint(checkpoint_path: Path, default_uni_model_dir: str) -> Tuple[torch.nn.Module, Dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = {}
    if isinstance(ckpt, dict):
        ckpt_args = args_to_dict(ckpt.get("args", {}))

    ckpt_model = str(ckpt_args.get("model", "uni")).lower()
    if ckpt_model != "uni":
        raise ValueError(f"Checkpoint is not UNI: model={ckpt_model}, ckpt={checkpoint_path}")

    use_spot_encoder = bool(ckpt_args.get("use_spot_encoder", False))
    uni_model_dir = str(ckpt_args.get("uni_model_dir", default_uni_model_dir))

    model = CLIPModel_UNI(
        uni_model_dir=uni_model_dir,
        use_spot_encoder=use_spot_encoder,
    )

    load_info = load_model_checkpoint(
        model=model,
        checkpoint_path=str(checkpoint_path),
        map_location="cpu",
        strict=False,
        rename_map={"well": "spot"},
    )
    model = model.to(DEVICE)
    model.eval()

    meta = {
        "checkpoint_path": str(checkpoint_path),
        "ckpt_model_arg": ckpt_model,
        "use_spot_encoder": use_spot_encoder,
        "uni_model_dir": uni_model_dir,
        "missing_keys": len(load_info["missing_keys"]),
        "unexpected_keys": len(load_info["unexpected_keys"]),
    }
    return model, meta


def infer_gene_names(base_data_dir: Path, slide: str, expr_shape_1: int) -> np.ndarray:
    features_path = base_data_dir / SLIDE_META[slide]["features"]
    hvg_union_path = base_data_dir / "data/filtered_expression_matrices/hvg_union.npy"

    if not features_path.exists():
        return np.array([], dtype=object)

    gene_names_all = pd.read_csv(features_path, header=None, sep="\t").iloc[:, 1].astype(str).values

    if len(gene_names_all) == expr_shape_1:
        return gene_names_all

    if hvg_union_path.exists():
        hvg_union = np.load(hvg_union_path).astype(int)
        if len(hvg_union) == expr_shape_1 and np.max(hvg_union) < len(gene_names_all):
            return gene_names_all[hvg_union]

    return np.array([], dtype=object)


def build_dataset_for_slide(base_data_dir: Path, slide: str, patch_size: int) -> CLIPDataset:
    meta = SLIDE_META[slide]
    image_path = base_data_dir / meta["image"]
    spatial_path = base_data_dir / meta["spatial"]
    barcode_path = base_data_dir / meta["barcode"]
    reduced_mtx_path = base_data_dir / meta["reduced_mtx"]

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not spatial_path.exists():
        raise FileNotFoundError(f"Spatial positions not found: {spatial_path}")
    if not barcode_path.exists():
        raise FileNotFoundError(f"Barcode file not found: {barcode_path}")
    if not reduced_mtx_path.exists():
        raise FileNotFoundError(f"Reduced matrix not found: {reduced_mtx_path}")

    ds = CLIPDataset(
        image_path=str(image_path),
        spatial_pos_path=str(spatial_path),
        barcode_path=str(barcode_path),
        reduced_mtx_path=str(reduced_mtx_path),
        is_train=False,
        patch_size=patch_size,
        slide_id=slide,
    )
    return ds


def extract_one_slide(
    model: torch.nn.Module,
    dataset: CLIPDataset,
    batch_size: int,
    num_workers: int,
) -> Dict[str, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    img_embs = []
    spot_embs = []
    expr_rows = []
    barcodes = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extract slide {dataset.slide_id}"):
            images = batch["image"].to(DEVICE, non_blocking=True)
            expr = batch["reduced_expression"].to(DEVICE, non_blocking=True)

            img_feat = model.encode_image(images)
            img_proj = model.image_projection(img_feat)

            spot_feat = model.encode_spot(expr)
            spot_proj = model.spot_projection(spot_feat)

            img_embs.append(img_proj.detach().cpu().numpy().astype(np.float32))
            spot_embs.append(spot_proj.detach().cpu().numpy().astype(np.float32))
            expr_rows.append(batch["reduced_expression"].cpu().numpy().astype(np.float32))

            # DataLoader 对字符串通常会收成 list[str]
            batch_barcodes = list(batch["barcode"])
            barcodes.extend(batch_barcodes)

    img_embs = np.concatenate(img_embs, axis=0)
    spot_embs = np.concatenate(spot_embs, axis=0)
    expr_rows = np.concatenate(expr_rows, axis=0)
    barcodes = np.asarray(barcodes, dtype=object)

    if not (img_embs.shape[0] == spot_embs.shape[0] == expr_rows.shape[0] == len(barcodes)):
        raise RuntimeError(
            f"Row count mismatch after extraction: "
            f"img={img_embs.shape}, spot={spot_embs.shape}, expr={expr_rows.shape}, barcodes={len(barcodes)}"
        )

    return {
        "img_embeddings": img_embs,
        "spot_embeddings": spot_embs,
        "expr": expr_rows,
        "barcodes": barcodes,
    }


def main():
    parser = argparse.ArgumentParser("Extract UNI embeddings per slide")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--base-data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--slides", type=str, default="1,2,3,4")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--default-uni-model-dir", type=str, default="/root/disk2/runzhi/BLEEP/UNI_Offline_Model")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    base_data_dir = Path(args.base_data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slides = [s.strip() for s in args.slides.split(",") if s.strip()]

    model, model_meta = build_uni_model_from_checkpoint(checkpoint_path, args.default_uni_model_dir)
    print("Loaded UNI model:")
    print(json.dumps(model_meta, indent=2, ensure_ascii=False))

    global_report = {
        "checkpoint": str(checkpoint_path),
        "base_data_dir": str(base_data_dir),
        "out_dir": str(out_dir),
        "slides": slides,
        "device": DEVICE,
        "model_meta": model_meta,
        "slides_meta": {},
    }

    for slide in slides:
        if slide not in SLIDE_META:
            raise ValueError(f"Unknown slide: {slide}")

        ds = build_dataset_for_slide(base_data_dir, slide, patch_size=args.patch_size)
        out = extract_one_slide(
            model=model,
            dataset=ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        gene_names = infer_gene_names(base_data_dir, slide, out["expr"].shape[1])

        slide_dir = out_dir / f"slide_{slide}"
        slide_dir.mkdir(parents=True, exist_ok=True)

        np.save(slide_dir / "img_embeddings.npy", out["img_embeddings"])
        np.save(slide_dir / "spot_embeddings.npy", out["spot_embeddings"])
        np.save(slide_dir / "expr.npy", out["expr"])
        np.save(slide_dir / "barcodes.npy", out["barcodes"])
        np.save(slide_dir / "gene_names.npy", gene_names)

        slide_meta = {
            "slide": slide,
            "slide_name": SLIDE_META[slide]["name"],
            "n_spots": int(out["img_embeddings"].shape[0]),
            "img_dim": int(out["img_embeddings"].shape[1]),
            "spot_dim": int(out["spot_embeddings"].shape[1]),
            "n_genes": int(out["expr"].shape[1]),
            "img_mean_std_across_spots": float(np.mean(np.std(out["img_embeddings"], axis=0))),
            "spot_mean_std_across_spots": float(np.mean(np.std(out["spot_embeddings"], axis=0))),
            "expr_mean_std_across_spots": float(np.mean(np.std(out["expr"], axis=0))),
            "first_10_barcodes": out["barcodes"][:10].tolist(),
        }

        with open(slide_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(slide_meta, f, indent=2, ensure_ascii=False)

        global_report["slides_meta"][slide] = slide_meta
        print(f"[OK] slide {slide} saved to {slide_dir}")
        print(json.dumps(slide_meta, indent=2, ensure_ascii=False))

    with open(out_dir / "extract_report.json", "w", encoding="utf-8") as f:
        json.dump(global_report, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("Extraction completed.")
    print(f"Output dir: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()