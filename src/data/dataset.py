"""PyTorch Dataset — 2D 切片级 / 2.5D triplet.

从 data/mini/npy/ 加载 Sagittal 切片.
数据层级:
  npy_root/StudyInstanceUID/SeriesInstanceUID/SOPInstanceUID.npy

模式:
- 2D:  每张切片独立作为样本,  [1, H, W] 灰度图
- 2.5D: 3 张相邻切片堆叠,   [3, H, W] → 原生 ImageNet 预训练输入
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.transforms import apply_transform


class KneeSliceDataset(Dataset):
    """Sagittal 切片数据集, 支持 2D 和 2.5D triplet 模式.

    2D 模式 (in_channels=1):
      每张切片一个样本, 输出 [1, H, W]

    2.5D 模式 (in_channels=3):
      同一 series 内按 SOPInstanceUID 排序后, 取连续 3 张切片堆叠
      输出 [3, H, W], 可直接用 ImageNet 预训练权重

    Args:
        metadata_df: StudyInstanceUID + SeriesInstanceUID + SOPInstanceUID
        labels_df: StudyInstanceUID + 12 个 label 列
        npy_root: npy 文件根目录
        image_size: 输出图像尺寸
        in_channels: 1 = 2D 单切片, 3 = 2.5D triplet
        slice_offset: triplet 的切片间距 (默认 1, 即相邻)
        is_train: 是否训练模式
        target_columns: 12 个标签列名
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        npy_root: str | Path = "data/mini/npy",
        image_size: int = 224,
        in_channels: int = 3,
        slice_offset: int = 1,
        is_train: bool = True,
        target_columns: list[str] | None = None,
        transform: "A.Compose | None" = None,
    ):
        self.npy_root = Path(npy_root)
        self.image_size = image_size
        self.in_channels = in_channels
        self.slice_offset = slice_offset
        self.is_train = is_train
        self.transform = transform

        self.target_columns = target_columns or [
            "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
            "Medial OA", "Lateral OA", "PF OA",
            "Effusion", "Synovitis", "Baker's",
            "Contusion", "Fracture",
        ]

        # 标签: StudyInstanceUID → label vector
        label_map = labels_df.set_index("StudyInstanceUID")[self.target_columns]

        # --- 构建样本索引 ---
        # 按 (study, series) 分组, 组内按 SOPInstanceUID 排序
        self.records = []
        skipped = 0

        groups = metadata_df.groupby(["StudyInstanceUID", "SeriesInstanceUID"])

        for (sid, series_uid), group in groups:
            if sid not in label_map.index:
                skipped += 1
                continue

            # 按文件名排序 (SOPInstanceUID 字典序对应切片顺序)
            group = group.sort_values("SOPInstanceUID")
            sop_uids = group["SOPInstanceUID"].tolist()
            labels = label_map.loc[sid].values.astype(np.float32)
            n_slices = len(sop_uids)

            if in_channels == 1:
                # 2D 模式: 每张切片一个样本
                for sop in sop_uids:
                    self.records.append({
                        "study_uid": sid,
                        "series_uid": series_uid,
                        "sop_uids": [sop],
                        "labels": labels,
                    })
            else:
                # 2.5D 模式: 取连续 3 张 (offset=1 即 [z-1, z, z+1])
                step = slice_offset
                for z in range(step, n_slices - step):
                    self.records.append({
                        "study_uid": sid,
                        "series_uid": series_uid,
                        "sop_uids": [sop_uids[z - step], sop_uids[z], sop_uids[z + step]],
                        "labels": labels,
                    })

        if skipped:
            print(f"警告: {skipped} 个 study 在 labels_df 中找不到, 已跳过")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        # 加载切片 → [in_channels, H, W]
        slices = []
        for sop in rec["sop_uids"]:
            npy_path = (
                self.npy_root / rec["study_uid"] / rec["series_uid"] / f"{sop}.npy"
            )
            img = np.load(npy_path).astype(np.float32)           # [H, W]

            if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
                import cv2
                img = cv2.resize(img, (self.image_size, self.image_size))

            slices.append(img)

        image = np.stack(slices, axis=0)                         # [C, H, W]

        # 应用数据增强 (训练时: 空间+像素变换; 验证时: 仅 resize)
        if self.transform is not None:
            image = apply_transform(image, self.transform)

        return {
            "image": torch.from_numpy(image.copy()),
            "labels": torch.from_numpy(rec["labels"]),
            "study_uid": rec["study_uid"],
        }
