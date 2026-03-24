import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import random


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_path,
        spatial_pos_path,
        barcode_path,
        reduced_mtx_path,
        is_train: bool = True,
        patch_size: int = 224,
    ):
        # image_path 是整张切片级高分辨率图像
        self.whole_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if self.whole_image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")

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

        print("Finished loading all files")

    def transform(self, image: np.ndarray) -> torch.Tensor:
        image = Image.fromarray(image)

        if self.is_train:
            # 训练时开启随机翻转 + 90 度旋转增强
            if random.random() > 0.5:
                image = TF.hflip(image)
            if random.random() > 0.5:
                image = TF.vflip(image)

            angle = random.choice([180, 90, 0, -90])
            image = TF.rotate(image, angle)

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

        patch_bgr = self._crop_patch_with_padding(v1, v2)
        # OpenCV 读入是 BGR，转成 RGB 再喂给 PIL/torchvision
        patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)

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

        return item

    def __len__(self):
        return len(self.barcodes)
