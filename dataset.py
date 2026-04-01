import numpy as np
import pandas as pd
import torch
from PIL import Image
import random
import os
import re
# 新增这一行：解除 PIL 对超大病理图像的像素读取限制
Image.MAX_IMAGE_PIXELS = None

def _hflip_pil(image: Image.Image) -> Image.Image:
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def _vflip_pil(image: Image.Image) -> Image.Image:
    return image.transpose(Image.FLIP_TOP_BOTTOM)


def _rotate_pil(image: Image.Image, angle_degrees: int) -> Image.Image:
    # For multiples of 90 degrees, PIL keeps the original size.
    # Using bilinear interpolation is safe for general angles too.
    return image.rotate(angle_degrees, resample=Image.BILINEAR)


def _to_tensor_pil(image: Image.Image) -> torch.Tensor:
    # Convert uint8 RGB (H, W, C) -> float tensor (C, H, W) in [0, 1]
    arr = np.array(image, dtype=np.uint8)
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float().div(255.0)
    return t


def _normalize_tensor(t: torch.Tensor, mean, std) -> torch.Tensor:
    # mean/std are 3-element lists in RGB order.
    mean_t = torch.tensor(mean, dtype=t.dtype, device=t.device).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=t.dtype, device=t.device).view(3, 1, 1)
    return t.sub(mean_t).div(std_t)


def _infer_slide_id(image_path: str) -> str:
    """
    从图像文件名推断稳定的切片 ID（如 C73_A1）。
    若推断失败，回退到去扩展名文件名，保证可复现。
    """
    stem = os.path.splitext(os.path.basename(str(image_path)))[0]
    upper = stem.upper()

    match = re.search(r"C73_([ABCD]1)", upper)
    if match is not None:
        return f"C73_{match.group(1)}"

    token_map = {
        "A1": "C73_A1",
        "B1": "C73_B1",
        "C1": "C73_C1",
        "D1": "C73_D1",
    }
    for token, slide_id in token_map.items():
        if token in upper:
            return slide_id

    return stem


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_path,
        spatial_pos_path,
        barcode_path,
        reduced_mtx_path,
        is_train: bool = True,
        patch_size: int = 224,
        slide_id: str = None,
    ):
        # image_path 是整张切片级高分辨率图像
        try:
            # Keep everything in RGB to avoid cv2 dependency.
            self.whole_image = np.array(Image.open(image_path).convert("RGB"))
        except Exception as e:
            raise FileNotFoundError(f"Failed to load image: {image_path}") from e

        self.image_h, self.image_w = self.whole_image.shape[:2]
        self.patch_size = int(patch_size)
        self.half_patch = self.patch_size // 2

        self.spatial_pos_csv = pd.read_csv(spatial_pos_path, sep=",", header=None)
        self.barcode_tsv = pd.read_csv(barcode_path, sep="\t", header=None)

        # preprocess 输出默认为 (gene x cell)，这里转成 (cell x feature)
        self.reduced_matrix = np.load(reduced_mtx_path).T

        # 仅保留 tissue 内被检测到的 spot，避免条码和表达矩阵错位
        detected_col = self.spatial_pos_csv[1]
        detected_mask = (
            (detected_col == 1)
            | (detected_col == True)
            | (detected_col.astype(str).str.lower().isin(["true", "1", "t", "yes"]))
        )
        detected_barcodes = set(self.spatial_pos_csv.loc[detected_mask, 0].tolist())

        self.barcode_tsv = self.barcode_tsv[self.barcode_tsv[0].isin(detected_barcodes)]
        self.barcode_tsv = self.barcode_tsv.reset_index(drop=True)

        if self.reduced_matrix.shape[0] != len(self.barcode_tsv):
            raise ValueError(
                "Barcodes / reduced matrix cell-axis mismatch. "
                f"reduced_matrix cells={self.reduced_matrix.shape[0]} vs "
                f"filtered barcodes={len(self.barcode_tsv)}."
            )

        # 按 barcode 一次性对齐坐标，避免 __getitem__ 里反复 dataframe 检索
        spatial_df = self.spatial_pos_csv.set_index(0)
        ordered_spatial = spatial_df.loc[self.barcode_tsv[0].values]
        self.spatial_coords = ordered_spatial[[4, 5]].to_numpy(dtype=np.float32)

        self.barcodes = self.barcode_tsv[0].tolist()
        self.reduced_matrix = self.reduced_matrix.astype(np.float32, copy=False)
        self.is_train = is_train
        self.slide_id = str(slide_id) if slide_id is not None else _infer_slide_id(image_path)

        print("Finished loading all files")

    def transform(self, image: np.ndarray) -> torch.Tensor:
        image = Image.fromarray(image)

        if self.is_train:
            # 训练时开启随机翻转 + 90 度旋转增强
            if random.random() > 0.5:
                image = _hflip_pil(image)
            if random.random() > 0.5:
                image = _vflip_pil(image)

            angle = random.choice([180, 90, 0, -90])
            image = _rotate_pil(image, angle)

        image = _to_tensor_pil(image)
        image = _normalize_tensor(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return image

    def _crop_patch_with_padding(self, center_y: float, center_x: float) -> np.ndarray:
        """
        在 (center_y, center_x) 处裁剪 patch。
        若越界则使用 Zero-padding（NumPy 方式）补齐，确保最终尺寸固定为 224x224。
        """
        cy = int(round(float(center_y)))
        cx = int(round(float(center_x)))

        y1 = cy - self.half_patch
        y2 = y1 + self.patch_size
        x1 = cx - self.half_patch
        x2 = x1 + self.patch_size

        # 目标 patch 先创建全零，后续只把有效重叠区域拷贝进去
        patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=self.whole_image.dtype)

        src_y1 = max(0, y1)
        src_y2 = min(self.image_h, y2)
        src_x1 = max(0, x1)
        src_x2 = min(self.image_w, x2)

        if src_y1 < src_y2 and src_x1 < src_x2:
            dst_y1 = src_y1 - y1
            dst_x1 = src_x1 - x1
            dst_y2 = dst_y1 + (src_y2 - src_y1)
            dst_x2 = dst_x1 + (src_x2 - src_x1)

            patch[dst_y1:dst_y2, dst_x1:dst_x2] = self.whole_image[src_y1:src_y2, src_x1:src_x2]

        return patch

    def __getitem__(self, idx):
        item = {}

        barcode = self.barcodes[idx]
        v1, v2 = self.spatial_coords[idx]  # v1: y, v2: x

        patch_rgb = self._crop_patch_with_padding(v1, v2)
        image = self.transform(patch_rgb)

        # 防御式检查：无论边界如何，shape 必须稳定为 (3, 224, 224)
        expected_shape = (3, self.patch_size, self.patch_size)
        if tuple(image.shape) != expected_shape:
            raise RuntimeError(
                f"Patch tensor shape mismatch: got {tuple(image.shape)}, expected {expected_shape}"
            )

        item["image"] = image.float()
        item["reduced_expression"] = torch.tensor(self.reduced_matrix[idx, :], dtype=torch.float32)
        item["barcode"] = barcode
        item["spatial_coords"] = torch.tensor([v1, v2], dtype=torch.float32)
        item["slide_id"] = self.slide_id

        return item

    def __len__(self):
        return len(self.barcodes)
