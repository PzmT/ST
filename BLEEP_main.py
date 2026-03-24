import argparse
import os
from typing import Dict, Optional, Tuple

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
    CLIPModel_resnet101,
    CLIPModel_resnet152,
)
from modules import AdaptiveRegionGenerator, DualHypergraphAligner
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

parser.add_argument("--use_arg", action="store_true", help="enable AdaptiveRegionGenerator")
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


def _is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_main_process() -> bool:
    return (not _is_dist_initialized()) or dist.get_rank() == 0


def _get_image_feature_dim(model_name: str) -> int:
    if model_name in ("clip", "vit"):
        return 768
    if model_name == "vit_l":
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
    """
    数据切分策略（关键修复）：
    - 训练集：dataset1 + dataset2（切片级拼接）
    - 验证集：dataset4（独立切片）

    这样可彻底避免同一物理切片内 spot 级 random_split 带来的空间泄漏。
    """
    base_dir = os.path.join(os.path.dirname(__file__), "GSE240429_data")

    dataset1_train = CLIPDataset(
        image_path=os.path.join(base_dir, "images/GEX_C73_A1_Merged.tif"),
        spatial_pos_path=os.path.join(base_dir, "data/tissue_pos_matrices/tissue_positions_list_1.csv"),
        reduced_mtx_path=os.path.join(base_dir, "data/filtered_expression_matrices/1/harmony_matrix.npy"),
        barcode_path=os.path.join(base_dir, "data/filtered_expression_matrices/1/barcodes.tsv"),
        is_train=True,
    )
    dataset2_train = CLIPDataset(
        image_path=os.path.join(base_dir, "images/GEX_C73_B1_Merged.tif"),
        spatial_pos_path=os.path.join(base_dir, "data/tissue_pos_matrices/tissue_positions_list_2.csv"),
        reduced_mtx_path=os.path.join(base_dir, "data/filtered_expression_matrices/2/harmony_matrix.npy"),
        barcode_path=os.path.join(base_dir, "data/filtered_expression_matrices/2/barcodes.tsv"),
        is_train=True,
    )
    dataset4_val = CLIPDataset(
        image_path=os.path.join(base_dir, "images/GEX_C73_D1_Merged.tif"),
        spatial_pos_path=os.path.join(base_dir, "data/tissue_pos_matrices/tissue_positions_list_4.csv"),
        reduced_mtx_path=os.path.join(base_dir, "data/filtered_expression_matrices/4/harmony_matrix.npy"),
        barcode_path=os.path.join(base_dir, "data/filtered_expression_matrices/4/barcodes.tsv"),
        is_train=False,
    )

    train_dataset = ConcatDataset([dataset1_train, dataset2_train])
    val_dataset = dataset4_val

    if _is_main_process():
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
        model = CLIPModel_CLIP().to(device)
        model_name = "CLIP"
    elif args.model == "vit":
        model = CLIPModel_ViT().to(device)
        model_name = "ViT"
    elif args.model == "vit_l":
        model = CLIPModel_ViT_L().to(device)
        model_name = "ViT-L"
    elif args.model == "resnet101":
        model = CLIPModel_resnet101().to(device)
        model_name = "ResNet101"
    elif args.model == "resnet152":
        model = CLIPModel_resnet152().to(device)
        model_name = "ResNet152"
    else:
        model = CLIPModel().to(device)
        model_name = "ResNet50"

    if _is_main_process():
        print(f"Image encoder is {model_name}", flush=True)

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


def train_epoch(
    model,
    train_loader,
    optimizer,
    args,
    device: torch.device,
    arg_generator: Optional[AdaptiveRegionGenerator] = None,
    dual_hg_aligner: Optional[nn.Module] = None,
):
    loss_meter = AvgMeter("train_loss")
    progress = tqdm(
        train_loader,
        total=len(train_loader),
        disable=not _is_main_process(),
        leave=False,
    )

    for batch in progress:
        image = batch["image"].to(device, non_blocking=True)
        reduced_expression = batch["reduced_expression"].to(device, non_blocking=True)
        model_batch = {"image": image, "reduced_expression": reduced_expression}

        need_features = args.use_arg or args.use_dual_hg
        forward_out = model(model_batch, return_features=need_features)

        if need_features:
            clip_loss = forward_out["loss"]
            image_features = forward_out["image_features"]
        else:
            clip_loss = forward_out
            image_features = None

        loss = clip_loss
        align_loss = None
        arg_region_count = None

        if need_features:
            coords = _extract_spatial_coords(batch["spatial_coords"], device=device)
            with torch.no_grad():
                z_he_valid, region_indices = arg_generator(image_features.detach(), coords)
                arg_region_count = int(z_he_valid.size(0))

            if args.use_dual_hg:
                align_loss = dual_hg_aligner(
                    F=image_features,
                    T=coords,
                    region_indices=region_indices,
                    E=reduced_expression,
                    S=coords,
                )
                loss = clip_loss + args.dual_hg_weight * align_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 关键修复：不再手写 all_reduce 梯度聚合。
        # DDP 会在 backward 过程中自动执行 Ring-AllReduce 同步。
        optimizer.step()

        bs = image.size(0)
        loss_meter.update(loss.item(), bs)

        if align_loss is None and arg_region_count is None:
            progress.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
        elif align_loss is None:
            progress.set_postfix(
                train_loss=loss_meter.avg,
                lr=get_lr(optimizer),
                arg_regions=arg_region_count,
            )
        elif arg_region_count is None:
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
                arg_regions=arg_region_count,
                clip_loss=float(clip_loss.detach().item()),
                align_loss=float(align_loss.detach().item()),
            )

    return loss_meter


def val_epoch(
    model,
    val_loader,
    args,
    device: torch.device,
    arg_generator: Optional[AdaptiveRegionGenerator] = None,
    dual_hg_aligner: Optional[nn.Module] = None,
):
    loss_meter = AvgMeter("val_loss")
    local_sum = 0.0
    local_count = 0

    progress = tqdm(
        val_loader,
        total=len(val_loader),
        disable=not _is_main_process(),
        leave=False,
    )

    for batch in progress:
        image = batch["image"].to(device, non_blocking=True)
        reduced_expression = batch["reduced_expression"].to(device, non_blocking=True)
        model_batch = {"image": image, "reduced_expression": reduced_expression}

        need_features = args.use_arg or args.use_dual_hg
        forward_out = model(model_batch, return_features=need_features)

        if need_features:
            clip_loss = forward_out["loss"]
            image_features = forward_out["image_features"]
        else:
            clip_loss = forward_out
            image_features = None

        loss = clip_loss
        align_loss = None
        arg_region_count = None

        if need_features:
            coords = _extract_spatial_coords(batch["spatial_coords"], device=device)
            z_he_valid, region_indices = arg_generator(image_features, coords)
            arg_region_count = int(z_he_valid.size(0))

            if args.use_dual_hg:
                align_loss = dual_hg_aligner(
                    F=image_features,
                    T=coords,
                    region_indices=region_indices,
                    E=reduced_expression,
                    S=coords,
                )
                loss = clip_loss + args.dual_hg_weight * align_loss

        bs = image.size(0)
        loss_meter.update(loss.item(), bs)
        local_sum += float(loss.item()) * bs
        local_count += bs

        if align_loss is None and arg_region_count is None:
            progress.set_postfix(val_loss=loss_meter.avg)
        elif align_loss is None:
            progress.set_postfix(val_loss=loss_meter.avg, arg_regions=arg_region_count)
        elif arg_region_count is None:
            progress.set_postfix(
                val_loss=loss_meter.avg,
                clip_loss=float(clip_loss.detach().item()),
                align_loss=float(align_loss.detach().item()),
            )
        else:
            progress.set_postfix(
                val_loss=loss_meter.avg,
                arg_regions=arg_region_count,
                clip_loss=float(clip_loss.detach().item()),
                align_loss=float(align_loss.detach().item()),
            )

    global_avg = _all_gather_avg_loss(local_sum=local_sum, local_count=local_count, device=device)
    return loss_meter, global_avg


def _unwrap_model(model):
    return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model


def cleanup():
    if _is_dist_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main():
    args = parser.parse_args()

    if args.use_dual_hg and not args.use_arg:
        raise ValueError("`--use_dual_hg` requires `--use_arg` because DualHG uses ARG region indices.")

    device = _init_distributed(args)
    torch.backends.cudnn.benchmark = True

    model = _build_model(args, device)

    arg_generator = None
    if args.use_arg:
        arg_feature_dim = _get_image_feature_dim(args.model)
        arg_generator = AdaptiveRegionGenerator(
            feature_dim=arg_feature_dim,
            grid_size=args.arg_grid_size,
            topk=args.arg_topk,
            arsa_num_heads=args.arg_heads,
            arsa_dropout=args.arg_dropout,
        ).to(device)
        arg_generator.eval()
        if _is_main_process():
            print(
                f"ARG enabled: feature_dim={arg_feature_dim}, grid_size={args.arg_grid_size}, "
                f"topk={args.arg_topk}, heads={args.arg_heads}, dropout={args.arg_dropout}",
                flush=True,
            )

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

        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            args=args,
            device=device,
            arg_generator=arg_generator,
            dual_hg_aligner=dual_hg_aligner,
        )

        model.eval()
        if dual_hg_aligner is not None:
            dual_hg_aligner.eval()

        with torch.no_grad():
            val_loss_local, val_loss_global = val_epoch(
                model=model,
                val_loader=val_loader,
                args=args,
                device=device,
                arg_generator=arg_generator,
                dual_hg_aligner=dual_hg_aligner,
            )

        lr_scheduler.step(val_loss_global)

        if _is_main_process():
            print(
                f"Epoch [{epoch + 1}/{args.max_epochs}] "
                f"train_loss(local)={train_loss.avg:.6f}, "
                f"val_loss(local)={val_loss_local.avg:.6f}, "
                f"val_loss(global)={val_loss_global:.6f}",
                flush=True,
            )

            if val_loss_global < best_val_loss:
                os.makedirs(str(args.exp_name), exist_ok=True)
                best_val_loss = val_loss_global
                best_epoch = epoch + 1

                ckpt: Dict[str, object] = {
                    "epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "model": _unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                }
                if dual_hg_aligner is not None:
                    ckpt["dual_hg_aligner"] = _unwrap_model(dual_hg_aligner).state_dict()

                torch.save(ckpt, os.path.join(str(args.exp_name), "best.pt"))
                print(f"Saved Best Model at epoch {best_epoch}, val_loss={best_val_loss:.6f}", flush=True)

    if _is_main_process():
        print(f"Training completed. best_epoch={best_epoch}, best_val_loss={best_val_loss:.6f}", flush=True)

    cleanup()


if __name__ == "__main__":
    main()
