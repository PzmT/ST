from collections import OrderedDict
from typing import Dict, Optional

import torch


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value
    return cleaned


def _rename_state_dict_keys(
    state_dict: Dict[str, torch.Tensor],
    rename_map: Optional[Dict[str, str]] = None,
) -> Dict[str, torch.Tensor]:
    if not rename_map:
        return state_dict

    renamed = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        for old, new in rename_map.items():
            new_key = new_key.replace(old, new)
        renamed[new_key] = value
    return renamed


def extract_model_state_dict(checkpoint_obj) -> Dict[str, torch.Tensor]:
    """
    兼容两类格式：
    1) 结构化 checkpoint: {"model": ...} 或 {"model_state_dict": ...}
    2) 纯模型 state_dict
    """
    if not isinstance(checkpoint_obj, dict):
        raise TypeError(f"Checkpoint must be a dict-like object, got {type(checkpoint_obj)}")

    if "model" in checkpoint_obj and isinstance(checkpoint_obj["model"], dict):
        state_dict = checkpoint_obj["model"]
    elif "model_state_dict" in checkpoint_obj and isinstance(checkpoint_obj["model_state_dict"], dict):
        state_dict = checkpoint_obj["model_state_dict"]
    else:
        state_dict = checkpoint_obj

    return _strip_module_prefix(state_dict)


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    map_location="cpu",
    strict: bool = True,
    rename_map: Optional[Dict[str, str]] = None,
):
    """
    统一模型加载入口，返回 missing/unexpected 便于推理端显式检查。
    """
    checkpoint_obj = torch.load(checkpoint_path, map_location=map_location)
    state_dict = extract_model_state_dict(checkpoint_obj)
    state_dict = _rename_state_dict_keys(state_dict, rename_map=rename_map)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "checkpoint_obj": checkpoint_obj,
    }
