import argparse
import os
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import config as CFG
from dataset import CLIPDataset
from models import (
    CLIPModel,
    CLIPModel_ViT,
    CLIPModel_ViT_L,
    CLIPModel_CLIP,
    CLIPModel_UNI,
    CLIPModel_resnet101,
    CLIPModel_resnet152,
)
from modules import AdaptiveRegionGenerator, DualHypergraphAligner, FixedGridRegionGenerator
from utils import AvgMeter, get_lr


parser = argparse.ArgumentParser(description="DDP training for BLEEP")
parser.add_argument("--exp_name", type=str, default="clip", help="experiment name")
parser.add_argument("--batch_size", type=int, default=256, help="batch size per GPU")
parser.add_argument("--max_epochs", type=int, default=4, help="number of epochs")
parser.add_argument("--num_workers", type=int, default=0, help="num_workers for dataloader")

parser.add_argument("--init_method", default="env://", type=str, help="ddp init_method")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--world_size", default=1, type=int, help="number of processes")
parser.add_argument("--distributed", action="store_true", help="force distributed training")

parser.add_argument("--model", type=str, default="resnet50", help="image encoder type")
parser.add_argument(
    "--uni_model_dir",
    type=str,
    default="/root/disk2/runzhi/BLEEP/UNI_Offline_Model",
    help="local offline UNI checkpoint directory (contains config.json and pytorch_model.bin)",
)
parser.add_argument(
    "--use_spot_encoder",
    action="store_true",
    default=CFG.use_spot_encoder,
    help="enable optional SpotEncoder tower (default: off for BLEEP-like baseline path)",
)
parser.add_argument(
    "--train_slides",
    type=str,
    default="1,2",
    help='comma-separated train slide ids, e.g. "1,2,4"',
)
parser.add_argument(
    "--val_slides",
    type=str,
    default="4",
    help='comma-separated val slide ids, e.g. "3"',
)
parser.add_argument(
    "--test_slides",
    type=str,
    default="3",
    help='comma-separated test slide ids for explicit split bookkeeping',
)

parser.add_argument(
    "--region_mode",
    type=str,
    choices=["none", "fixed", "adaptive"],
    default="none",
    help="region generator mode: none (baseline), fixed (k x k deterministic), adaptive (ARG)",
)
parser.add_argument("--fixed_grid_size", type=int, default=8, help="fixed-grid baseline size k")
parser.add_argument(
    "--use_arg",
    action="store_true",
    help=argparse.SUPPRESS,  # deprecated: kept for backward compatibility
)
parser.add_argument("--arg_grid_size", type=int, default=8, help="ARG grid size k")
parser.add_argument("--arg_topk", type=int, default=3, help="ARG top-k candidates")
parser.add_argument("--arg_heads", type=int, default=8, help="ARG ARSA attention heads")
parser.add_argument("--arg_dropout", type=float, default=0.1, help="ARG ARSA dropout")

parser.add_argument("--use_dual_hg", action="store_true", help="enable DualHypergraphAligner")
parser.add_argument("--dual_hg_out_dim", type=int, default=256, help="DualHG output dim")
parser.add_argument("--dual_hg_radius", type=float, default=150.0, help="ST recall radius D")
parser.add_argument("--dual_hg_k", type=int, default=5, help="hypergraph KNN neighbors")
parser.add_argument("--dual_hg_temp", type=float, default=0.07, help="InfoNCE temperature")
parser.add_argument("--dual_hg_weight", type=float, default=1.0, help="weight for L_align")
parser.add_argument(
    "--spatial_debug_interval",
    type=int,
    default=0,
    help="if > 0, print spatial branch debug every N train/val steps on rank 0",
)


SLIDE_META = {
    "1": {
        "slide_id": "C73_A1",
        "image": "images/GEX_C73_A1_Merged.tif",
        "spatial_pos": "data/tissue_pos_matrices/tissue_positions_list_1.csv",
        "reduced_mtx": "data/filtered_expression_matrices/1/harmony_matrix.npy",
        "barcode": "data/filtered_expression_matrices/1/barcodes.tsv",
    },
    "2": {
        "slide_id": "C73_B1",
        "image": "images/GEX_C73_B1_Merged.tif",
        "spatial_pos": "data/tissue_pos_matrices/tissue_positions_list_2.csv",
        "reduced_mtx": "data/filtered_expression_matrices/2/harmony_matrix.npy",
        "barcode": "data/filtered_expression_matrices/2/barcodes.tsv",
    },
    "3": {
        "slide_id": "C73_C1",
        "image": "images/GEX_C73_C1_Merged.tif",
        "spatial_pos": "data/tissue_pos_matrices/tissue_positions_list_3.csv",
        "reduced_mtx": "data/filtered_expression_matrices/3/harmony_matrix.npy",
        "barcode": "data/filtered_expression_matrices/3/barcodes.tsv",
    },
    "4": {
        "slide_id": "C73_D1",
        "image": "images/GEX_C73_D1_Merged.tif",
        "spatial_pos": "data/tissue_pos_matrices/tissue_positions_list_4.csv",
        "reduced_mtx": "data/filtered_expression_matrices/4/harmony_matrix.npy",
        "barcode": "data/filtered_expression_matrices/4/barcodes.tsv",
    },
}


def _is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_main_process() -> bool:
    return (not _is_dist_initialized()) or dist.get_rank() == 0


def _get_image_feature_dim(model_name: str) -> int:
    if model_name in ("clip", "vit"):
        return 768
    if model_name in ("vit_l", "uni"):
        return 1024
    return 2048


def _extract_spatial_coords(spatial_coords, device: torch.device) -> torch.Tensor:
    """将 DataLoader 的坐标字段统一转换为 (B, 2) float tensor。"""
    if torch.is_tensor(spatial_coords):
        coords = spatial_coords
    elif (
        isinstance(spatial_coords, (list, tuple))
        and len(spatial_coords) == 2
        and all(torch.is_tensor(x) for x in spatial_coords)
    ):
        coords = torch.stack(spatial_coords, dim=1)
    else:
        coords = torch.as_tensor(spatial_coords)

    if coords.ndim != 2:
        raise ValueError(f"spatial_coords must be 2D, got shape {tuple(coords.shape)}")
    if coords.shape[1] != 2:
        if coords.shape[0] == 2:
            coords = coords.t()
        else:
            raise ValueError(f"spatial_coords must have last dim=2, got shape {tuple(coords.shape)}")

    return coords.to(device=device, dtype=torch.float32)


def _infer_dist_env(args) -> Tuple[int, int, int, bool]:
    ngpus_per_node = max(torch.cuda.device_count(), 1)

    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    elif "SLURM_NTASKS" in os.environ:
        world_size = int(os.environ["SLURM_NTASKS"])
    else:
        world_size = int(args.world_size)

    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
    else:
        rank = 0

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_LOCALID" in os.environ:
        local_rank = int(os.environ["SLURM_LOCALID"])
    else:
        local_rank = rank % ngpus_per_node

    distributed = bool(args.distributed or world_size > 1)
    return rank, local_rank, world_size, distributed


def _init_distributed(args):
    rank, local_rank, world_size, distributed = _infer_dist_env(args)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if distributed:
        init_method = args.init_method
        # 兼容直接 python 启动的单机调试（缺省 env:// 变量时自动回退到本地 tcp）。
        if init_method == "env://" and (
            "MASTER_ADDR" not in os.environ or "MASTER_PORT" not in os.environ
        ):
            init_method = "tcp://127.0.0.1:29500"

        dist.init_process_group(
            backend=args.dist_backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )

    args.rank = rank
    args.local_rank = local_rank
    args.world_size = world_size
    args.distributed = distributed

    if _is_main_process():
        print(
            f"Distributed init done. rank={rank}, local_rank={local_rank}, "
            f"world_size={world_size}, distributed={distributed}",
            flush=True,
        )

    return device


def build_loaders(args):
    base_dir = os.path.join(os.path.dirname(__file__), "GSE240429_data")

    def _parse_slide_list(raw_value: str, arg_name: str) -> List[str]:
        ids = [x.strip() for x in str(raw_value).split(",") if x.strip()]
        if not ids:
            raise ValueError(f"`{arg_name}` is empty. Please provide at least one slide id.")
        invalid = [x for x in ids if x not in SLIDE_META]
        if invalid:
            raise ValueError(
                f"`{arg_name}` has invalid ids {invalid}. Valid ids are {sorted(SLIDE_META.keys())}."
            )
        return ids

    def _build_dataset_for_slide(slide_num: str, is_train: bool) -> CLIPDataset:
        slide_meta = SLIDE_META[slide_num]
        return CLIPDataset(
            image_path=os.path.join(base_dir, slide_meta["image"]),
            spatial_pos_path=os.path.join(base_dir, slide_meta["spatial_pos"]),
            reduced_mtx_path=os.path.join(base_dir, slide_meta["reduced_mtx"]),
            barcode_path=os.path.join(base_dir, slide_meta["barcode"]),
            slide_id=slide_meta["slide_id"],
            is_train=is_train,
        )

    train_slide_nums = _parse_slide_list(args.train_slides, "train_slides")
    val_slide_nums = _parse_slide_list(args.val_slides, "val_slides")
    test_slide_nums = _parse_slide_list(args.test_slides, "test_slides")

    train_sets = [_build_dataset_for_slide(slide_num, is_train=True) for slide_num in train_slide_nums]
    val_sets = [_build_dataset_for_slide(slide_num, is_train=False) for slide_num in val_slide_nums]

    train_dataset = train_sets[0] if len(train_sets) == 1 else ConcatDataset(train_sets)
    val_dataset = val_sets[0] if len(val_sets) == 1 else ConcatDataset(val_sets)

    if _is_main_process():
        train_slide_ids = [SLIDE_META[x]["slide_id"] for x in train_slide_nums]
        val_slide_ids = [SLIDE_META[x]["slide_id"] for x in val_slide_nums]
        test_slide_ids = [SLIDE_META[x]["slide_id"] for x in test_slide_nums]
        print(
            f"Slide split configured | train={train_slide_nums} ({train_slide_ids}) "
            f"| val={val_slide_nums} ({val_slide_ids}) | test={test_slide_nums} ({test_slide_ids})",
            flush=True,
        )
        print(f"Train samples={len(train_dataset)}, Val samples={len(val_dataset)}", flush=True)

    # 关键修复：训练/验证都使用 DistributedSampler
    # 训练 sampler 打乱，验证 sampler 不打乱；这样每张卡只验证自己子集，避免重复计算。
    if args.distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
            drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    return train_loader, val_loader


def _build_model(args, device: torch.device):
    if args.model == "clip":
        model = CLIPModel_CLIP(use_spot_encoder=args.use_spot_encoder).to(device)
        model_name = "CLIP"
    elif args.model == "vit":
        model = CLIPModel_ViT(use_spot_encoder=args.use_spot_encoder).to(device)
        model_name = "ViT"
    elif args.model == "vit_l":
        model = CLIPModel_ViT_L(use_spot_encoder=args.use_spot_encoder).to(device)
        model_name = "ViT-L"
    elif args.model == "uni":
        model = CLIPModel_UNI(
            uni_model_dir=args.uni_model_dir,
            use_spot_encoder=args.use_spot_encoder,
        ).to(device)
        model_name = "UNI(local)"
    elif args.model == "resnet101":
        model = CLIPModel_resnet101(use_spot_encoder=args.use_spot_encoder).to(device)
        model_name = "ResNet101"
    elif args.model == "resnet152":
        model = CLIPModel_resnet152(use_spot_encoder=args.use_spot_encoder).to(device)
        model_name = "ResNet152"
    else:
        model = CLIPModel(use_spot_encoder=args.use_spot_encoder).to(device)
        model_name = "ResNet50"

    if _is_main_process():
        if args.model == "uni":
            print(
                "Image encoder config | model=uni, "
                f"use_spot_encoder={bool(args.use_spot_encoder)}, "
                f"uni_model_dir={os.path.abspath(args.uni_model_dir)}, "
                f"image_feature_dim={_get_image_feature_dim(args.model)}",
                flush=True,
            )
        else:
            print(
                f"Image encoder is {model_name}, use_spot_encoder={bool(args.use_spot_encoder)}",
                flush=True,
            )

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    return model


def _all_gather_avg_loss(local_sum: float, local_count: int, device: torch.device) -> float:
    stats = torch.tensor([local_sum, float(local_count)], dtype=torch.float64, device=device)

    if _is_dist_initialized():
        gathered = [torch.zeros_like(stats) for _ in range(dist.get_world_size())]
        # 关键修复：验证集指标在所有 rank 间 all_gather 后再做全局平均。
        dist.all_gather(gathered, stats)
        stacked = torch.stack(gathered, dim=0)
        total_sum = stacked[:, 0].sum()
        total_count = stacked[:, 1].sum().clamp_min(1.0)
    else:
        total_sum = stats[0]
        total_count = stats[1].clamp_min(1.0)

    return (total_sum / total_count).item()


def _normalize_slide_ids(slide_ids, expected_batch: int) -> List[str]:
    if isinstance(slide_ids, (list, tuple)):
        ids = [str(x) for x in slide_ids]
    elif torch.is_tensor(slide_ids):
        if slide_ids.ndim != 1:
            raise ValueError(f"`slide_id` tensor must be 1D, got shape {tuple(slide_ids.shape)}")
        ids = [str(x.item()) for x in slide_ids]
    else:
        raise ValueError(f"Unsupported `slide_id` type: {type(slide_ids)}")

    if len(ids) != expected_batch:
        raise ValueError(
            f"`slide_id` batch size mismatch: got {len(ids)}, expected {expected_batch}."
        )
    return ids


def _ddp_safe_zero_align_from_module(
    module: Optional[nn.Module], ref_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Build a zero-valued align loss that still references all aligner params.
    This avoids DDP graph mismatch when local spatial fragments are invalid/skipped.
    """
    zero = ref_tensor.sum() * 0.0
    if module is None:
        return zero
    for p in module.parameters():
        zero = zero + p.sum() * 0.0
    return zero


def _run_spatial_branch(
    *,
    args,
    batch,
    image_features: torch.Tensor,
    reduced_expression: torch.Tensor,
    device: torch.device,
    region_generator: nn.Module,
    dual_hg_aligner: Optional[nn.Module],
):
    if "slide_id" not in batch:
        raise ValueError(
            "Spatial modules require `slide_id` in batch. "
            "Please update dataset/collate to return per-sample slide identity."
        )
    if region_generator is None:
        raise ValueError("`region_generator` is None while spatial branch is enabled.")
    if args.use_dual_hg and dual_hg_aligner is None:
        raise ValueError("`dual_hg_aligner` is None while `--use_dual_hg` is enabled.")

    coords = _extract_spatial_coords(batch["spatial_coords"], device=device)
    slide_ids = _normalize_slide_ids(batch["slide_id"], expected_batch=image_features.size(0))

    grouped_indices: Dict[str, List[int]] = {}
    for i, slide_id in enumerate(slide_ids):
        grouped_indices.setdefault(slide_id, []).append(i)

    region_count = 0
    region_count_by_slide: Dict[str, int] = {}
    align_terms: List[torch.Tensor] = []
    skipped_dual_hg = 0
    valid_dual_hg_slides = 0

    for slide_id, idx_list in grouped_indices.items():
        if any(slide_ids[i] != slide_id for i in idx_list):
            raise RuntimeError("Internal error: cross-slide index grouping failed.")

        idx_tensor = torch.as_tensor(idx_list, dtype=torch.long, device=device)
        image_features_i = image_features.index_select(0, idx_tensor)
        reduced_expression_i = reduced_expression.index_select(0, idx_tensor)
        coords_i = coords.index_select(0, idx_tensor)

        # 区域分配器在本轮用于构建 region indices，不参与梯度更新。
        with torch.no_grad():
            z_he_valid, region_indices = region_generator(image_features_i.detach(), coords_i)
        slide_region_count = int(z_he_valid.size(0))
        region_count_by_slide[str(slide_id)] = slide_region_count
        region_count += slide_region_count

        if args.use_dual_hg:
            zero_slide_align = _ddp_safe_zero_align_from_module(dual_hg_aligner, image_features_i)
            if image_features_i.size(0) < 2:
                # 样本太少的 slide 子集不参与 DualHG，避免构图退化。
                skipped_dual_hg += 1
                align_terms.append(zero_slide_align)
                continue

            slide_align = dual_hg_aligner(
                F=image_features_i,
                T=coords_i,
                region_indices=region_indices,
                E=reduced_expression_i,
                S=coords_i,
            )

            if torch.isfinite(slide_align).all():
                align_terms.append(slide_align)
                valid_dual_hg_slides += 1
            else:
                skipped_dual_hg += 1
                align_terms.append(zero_slide_align)

    align_loss = None
    if args.use_dual_hg:
        if align_terms:
            align_loss = torch.stack(align_terms).mean()
        else:
            align_loss = _ddp_safe_zero_align_from_module(dual_hg_aligner, image_features)

    return {
        "align_loss": align_loss,
        "region_count": region_count,
        "region_count_by_slide": region_count_by_slide,
        "num_slides": len(grouped_indices),
        "skipped_dual_hg": skipped_dual_hg,
        "valid_dual_hg_slides": valid_dual_hg_slides,
    }


def train_epoch(
    model,
    train_loader,
    optimizer,
    args,
    device: torch.device,
    region_generator: Optional[nn.Module] = None,
    dual_hg_aligner: Optional[nn.Module] = None,
):
    loss_meter = AvgMeter("train_loss")
    batch_regions_meter = AvgMeter("train_batch_regions")
    slide_regions_meter = AvgMeter("train_slide_regions")
    progress = tqdm(
        train_loader,
        total=len(train_loader),
        disable=not _is_main_process(),
        leave=False,
    )

    for step_idx, batch in enumerate(progress):
        image = batch["image"].to(device, non_blocking=True)
        reduced_expression = batch["reduced_expression"].to(device, non_blocking=True)
        model_batch = {"image": image, "reduced_expression": reduced_expression}

        need_features = args.region_mode != "none"
        forward_out = model(model_batch, return_features=need_features)

        if need_features:
            clip_loss = forward_out["loss"]
            image_features = forward_out["image_features"]
        else:
            clip_loss = forward_out
            image_features = None

        loss = clip_loss
        align_loss = None
        region_count = None
        skipped_dual_hg = 0

        if need_features:
            spatial_out = _run_spatial_branch(
                args=args,
                batch=batch,
                image_features=image_features,
                reduced_expression=reduced_expression,
                device=device,
                region_generator=region_generator,
                dual_hg_aligner=dual_hg_aligner,
            )
            region_count = spatial_out["region_count"]
            region_count_by_slide = spatial_out["region_count_by_slide"]
            skipped_dual_hg = spatial_out["skipped_dual_hg"]
            valid_dual_hg_slides = spatial_out["valid_dual_hg_slides"]
            batch_regions_meter.update(float(region_count), 1)
            for _, slide_region_count in region_count_by_slide.items():
                slide_regions_meter.update(float(slide_region_count), 1)

            if args.use_dual_hg:
                align_loss = spatial_out["align_loss"]
                loss = clip_loss + args.dual_hg_weight * align_loss

            if _is_main_process():
                debug_every = int(args.spatial_debug_interval)
                should_log_debug = (debug_every > 0 and step_idx % debug_every == 0) or (
                    args.use_dual_hg and skipped_dual_hg > 0
                )
                if should_log_debug:
                    print(
                        f"[SpatialDebug][train][step={step_idx}] "
                        f"region_mode={args.region_mode}, "
                        f"valid_spatial_slides={valid_dual_hg_slides}, "
                        f"skipped_dual_hg={skipped_dual_hg}, "
                        f"region_count_by_slide={region_count_by_slide}",
                        flush=True,
                    )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 关键修复：不再手写 all_reduce 梯度聚合。
        # DDP 会在 backward 过程中自动执行 Ring-AllReduce 同步。
        optimizer.step()

        bs = image.size(0)
        loss_meter.update(loss.item(), bs)

        if align_loss is None and region_count is None:
            progress.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
        elif align_loss is None:
            progress.set_postfix(
                train_loss=loss_meter.avg,
                lr=get_lr(optimizer),
                regions=region_count,
                mean_regions=slide_regions_meter.avg if slide_regions_meter.count > 0 else 0.0,
            )
        elif region_count is None:
            progress.set_postfix(
                train_loss=loss_meter.avg,
                lr=get_lr(optimizer),
                clip_loss=float(clip_loss.detach().item()),
                align_loss=float(align_loss.detach().item()),
            )
        else:
            progress.set_postfix(
                train_loss=loss_meter.avg,
                lr=get_lr(optimizer),
                regions=region_count,
                mean_regions=slide_regions_meter.avg if slide_regions_meter.count > 0 else 0.0,
                skipped_dual_hg=skipped_dual_hg,
                clip_loss=float(clip_loss.detach().item()),
                align_loss=float(align_loss.detach().item()),
            )

    return loss_meter, batch_regions_meter, slide_regions_meter


def val_epoch(
    model,
    val_loader,
    args,
    device: torch.device,
    region_generator: Optional[nn.Module] = None,
    dual_hg_aligner: Optional[nn.Module] = None,
):
    loss_meter = AvgMeter("val_loss")
    batch_regions_meter = AvgMeter("val_batch_regions")
    slide_regions_meter = AvgMeter("val_slide_regions")
    local_sum = 0.0
    local_count = 0

    progress = tqdm(
        val_loader,
        total=len(val_loader),
        disable=not _is_main_process(),
        leave=False,
    )

    for step_idx, batch in enumerate(progress):
        image = batch["image"].to(device, non_blocking=True)
        reduced_expression = batch["reduced_expression"].to(device, non_blocking=True)
        model_batch = {"image": image, "reduced_expression": reduced_expression}

        need_features = args.region_mode != "none"
        forward_out = model(model_batch, return_features=need_features)

        if need_features:
            clip_loss = forward_out["loss"]
            image_features = forward_out["image_features"]
        else:
            clip_loss = forward_out
            image_features = None

        loss = clip_loss
        align_loss = None
        region_count = None
        skipped_dual_hg = 0

        if need_features:
            spatial_out = _run_spatial_branch(
                args=args,
                batch=batch,
                image_features=image_features,
                reduced_expression=reduced_expression,
                device=device,
                region_generator=region_generator,
                dual_hg_aligner=dual_hg_aligner,
            )
            region_count = spatial_out["region_count"]
            region_count_by_slide = spatial_out["region_count_by_slide"]
            skipped_dual_hg = spatial_out["skipped_dual_hg"]
            valid_dual_hg_slides = spatial_out["valid_dual_hg_slides"]
            batch_regions_meter.update(float(region_count), 1)
            for _, slide_region_count in region_count_by_slide.items():
                slide_regions_meter.update(float(slide_region_count), 1)

            if args.use_dual_hg:
                align_loss = spatial_out["align_loss"]
                loss = clip_loss + args.dual_hg_weight * align_loss

            if _is_main_process():
                debug_every = int(args.spatial_debug_interval)
                should_log_debug = (debug_every > 0 and step_idx % debug_every == 0) or (
                    args.use_dual_hg and skipped_dual_hg > 0
                )
                if should_log_debug:
                    print(
                        f"[SpatialDebug][val][step={step_idx}] "
                        f"region_mode={args.region_mode}, "
                        f"valid_spatial_slides={valid_dual_hg_slides}, "
                        f"skipped_dual_hg={skipped_dual_hg}, "
                        f"region_count_by_slide={region_count_by_slide}",
                        flush=True,
                    )

        bs = image.size(0)
        loss_meter.update(loss.item(), bs)
        local_sum += float(loss.item()) * bs
        local_count += bs

        if align_loss is None and region_count is None:
            progress.set_postfix(val_loss=loss_meter.avg)
        elif align_loss is None:
            progress.set_postfix(
                val_loss=loss_meter.avg,
                regions=region_count,
                mean_regions=slide_regions_meter.avg if slide_regions_meter.count > 0 else 0.0,
            )
        elif region_count is None:
            progress.set_postfix(
                val_loss=loss_meter.avg,
                clip_loss=float(clip_loss.detach().item()),
                align_loss=float(align_loss.detach().item()),
            )
        else:
            progress.set_postfix(
                val_loss=loss_meter.avg,
                regions=region_count,
                mean_regions=slide_regions_meter.avg if slide_regions_meter.count > 0 else 0.0,
                skipped_dual_hg=skipped_dual_hg,
                clip_loss=float(clip_loss.detach().item()),
                align_loss=float(align_loss.detach().item()),
            )

    global_avg = _all_gather_avg_loss(local_sum=local_sum, local_count=local_count, device=device)
    return loss_meter, global_avg, batch_regions_meter, slide_regions_meter


def _unwrap_model(model):
    return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model


def cleanup():
    if _is_dist_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main():
    args = parser.parse_args()

    # Backward compatibility: map deprecated --use_arg to region_mode=adaptive.
    if args.use_arg:
        if args.region_mode == "none":
            args.region_mode = "adaptive"
            if _is_main_process():
                print("[Deprecated] `--use_arg` is mapped to `--region_mode adaptive`.", flush=True)
        elif args.region_mode != "adaptive":
            raise ValueError(
                "Conflicting args: `--use_arg` implies adaptive mode, "
                f"but got `--region_mode {args.region_mode}`."
            )

    if args.region_mode == "none" and args.use_dual_hg:
        raise ValueError("`--use_dual_hg` requires `--region_mode fixed` or `--region_mode adaptive`.")
    if args.region_mode not in ("none", "fixed", "adaptive"):
        raise ValueError(f"Unsupported region_mode: {args.region_mode}")

    device = _init_distributed(args)
    torch.backends.cudnn.benchmark = True

    if _is_main_process():
        if args.region_mode == "fixed":
            region_cfg = f"fixed(grid_size={args.fixed_grid_size})"
        elif args.region_mode == "adaptive":
            region_cfg = f"adaptive(arg_grid_size={args.arg_grid_size}, arg_topk={args.arg_topk})"
        else:
            region_cfg = "none"
        print(
            f"Experiment config | region_mode={region_cfg} | dual_hg={bool(args.use_dual_hg)}",
            flush=True,
        )

    model = _build_model(args, device)

    region_generator = None
    if args.region_mode == "fixed":
        fixed_feature_dim = _get_image_feature_dim(args.model)
        region_generator = FixedGridRegionGenerator(
            feature_dim=fixed_feature_dim,
            grid_size=args.fixed_grid_size,
        ).to(device)
        region_generator.requires_grad_(False)
        region_generator.eval()
        if _is_main_process():
            print(
                f"Region mode=fixed | feature_dim={fixed_feature_dim}, grid_size={args.fixed_grid_size}, "
                "aggregator=mean_pool",
                flush=True,
            )
    elif args.region_mode == "adaptive":
        arg_feature_dim = _get_image_feature_dim(args.model)
        region_generator = AdaptiveRegionGenerator(
            feature_dim=arg_feature_dim,
            grid_size=args.arg_grid_size,
            topk=args.arg_topk,
            arsa_num_heads=args.arg_heads,
            arsa_dropout=args.arg_dropout,
        ).to(device)
        region_generator.requires_grad_(False)
        region_generator.eval()
        if _is_main_process():
            print(
                f"Region mode=adaptive(ARG) | feature_dim={arg_feature_dim}, grid_size={args.arg_grid_size}, "
                f"topk={args.arg_topk}, heads={args.arg_heads}, dropout={args.arg_dropout}",
                flush=True,
            )
    else:
        if _is_main_process():
            print("Region mode=none | baseline BLEEP path", flush=True)

    dual_hg_aligner = None
    if args.use_dual_hg:
        he_dim = _get_image_feature_dim(args.model)
        dual_hg_aligner = DualHypergraphAligner(
            he_dim=he_dim,
            st_dim=CFG.spot_embedding,
            d_out=args.dual_hg_out_dim,
            st_radius=args.dual_hg_radius,
            k_hg=args.dual_hg_k,
            temperature=args.dual_hg_temp,
        ).to(device)

        # 关键修复：DualHG 作为可训练模块也交由 DDP 托管，同步梯度。
        if args.distributed:
            dual_hg_aligner = nn.parallel.DistributedDataParallel(
                dual_hg_aligner,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
            )

        if _is_main_process():
            print(
                f"DualHG enabled: he_dim={he_dim}, st_dim={CFG.spot_embedding}, "
                f"d_out={args.dual_hg_out_dim}, radius={args.dual_hg_radius}, "
                f"k={args.dual_hg_k}, temp={args.dual_hg_temp}, weight={args.dual_hg_weight}",
                flush=True,
            )
    elif _is_main_process():
        print("DualHG disabled", flush=True)

    train_loader, val_loader = build_loaders(args)

    optim_params = list(model.parameters())
    if dual_hg_aligner is not None:
        optim_params += list(dual_hg_aligner.parameters())

    optimizer = torch.optim.AdamW(
        optim_params,
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=CFG.patience,
        factor=CFG.factor,
    )

    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(args.max_epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        if dual_hg_aligner is not None:
            dual_hg_aligner.train()

        train_loss, train_batch_regions_meter, train_slide_regions_meter = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            args=args,
            device=device,
            region_generator=region_generator,
            dual_hg_aligner=dual_hg_aligner,
        )

        model.eval()
        if dual_hg_aligner is not None:
            dual_hg_aligner.eval()

        with torch.no_grad():
            val_loss_local, val_loss_global, val_batch_regions_meter, val_slide_regions_meter = val_epoch(
                model=model,
                val_loader=val_loader,
                args=args,
                device=device,
                region_generator=region_generator,
                dual_hg_aligner=dual_hg_aligner,
            )

        lr_scheduler.step(val_loss_global)

        if _is_main_process():
            if args.region_mode == "none":
                train_batch_regions_avg = "N/A"
                train_slide_regions_avg = "N/A"
                val_batch_regions_avg = "N/A"
                val_slide_regions_avg = "N/A"
            else:
                train_batch_regions_avg = f"{train_batch_regions_meter.avg:.3f}"
                train_slide_regions_avg = f"{train_slide_regions_meter.avg:.3f}"
                val_batch_regions_avg = f"{val_batch_regions_meter.avg:.3f}"
                val_slide_regions_avg = f"{val_slide_regions_meter.avg:.3f}"

            print(
                f"Epoch [{epoch + 1}/{args.max_epochs}] "
                f"train_loss(local)={train_loss.avg:.6f}, "
                f"val_loss(local)={val_loss_local.avg:.6f}, "
                f"val_loss(global)={val_loss_global:.6f}, "
                f"train_batch_regions_avg={train_batch_regions_avg}, "
                f"train_slide_regions_avg={train_slide_regions_avg}, "
                f"val_batch_regions_avg={val_batch_regions_avg}, "
                f"val_slide_regions_avg={val_slide_regions_avg}",
                flush=True,
            )

            if val_loss_global < best_val_loss:
                os.makedirs(str(args.exp_name), exist_ok=True)
                best_val_loss = val_loss_global
                best_epoch = epoch + 1

                model_state = _unwrap_model(model).state_dict()
                ckpt: Dict[str, object] = {
                    "epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "model": model_state,
                    "model_state_dict": model_state,
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                }
                if dual_hg_aligner is not None:
                    dual_hg_state = _unwrap_model(dual_hg_aligner).state_dict()
                    ckpt["dual_hg_aligner"] = dual_hg_state
                    ckpt["dual_hg_state_dict"] = dual_hg_state

                torch.save(ckpt, os.path.join(str(args.exp_name), "best.pt"))
                print(f"Saved Best Model at epoch {best_epoch}, val_loss={best_val_loss:.6f}", flush=True)

    if _is_main_process():
        print(f"Training completed. best_epoch={best_epoch}, best_val_loss={best_val_loss:.6f}", flush=True)

    cleanup()


if __name__ == "__main__":
    main()
