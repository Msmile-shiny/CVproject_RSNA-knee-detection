"""PyTorch Dataset — 2.5D 三平面采样.

从预处理好的 .npy MRI volume (研究级) 中抽取:
- Axial 面 3 个连续切片
- Coronal 面 3 个连续切片
- Sagittal 面 3 个连续切片

每个面的 3 切片堆叠为 3 通道输入, 模拟 RGB 三通道给 ImageNet 预训练 backbone.

数据层级:
  npy_root/
    study_uid/
      series_uid/
        sop_uid.npy    (H, W) 单切片
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TriplaneDataset(Dataset):
    """2.5D 三平面 MRI 数据集.

    每个 sample 返回三个面的切片堆叠:
      axial:   [3, H, W]
      coronal: [3, H, W]
      sagittal:[3, H, W]

    Args:
        index_df: DataFrame, 至少包含:
            - study_uid: str
            - patient_id: str
            - 12 个 label 列 (0/1)
        npy_root: 预处理 npy 文件根目录
        image_size: 输出图像尺寸 (默认 256)
        slice_offset: 三切片偏移间距 (默认 ±1)
        is_train: 是否训练模式 (做数据增强)
    """

    def __init__(
        self,
        index_df: pd.DataFrame,
        npy_root: str | Path = "data/mini/npy",
        image_size: int = 256,
        slice_offset: int = 1,
        target_columns: list[str] | None = None,
        is_train: bool = True,
    ):
        self.df = index_df.reset_index(drop=True)
        self.npy_root = Path(npy_root)
        self.image_size = image_size
        self.slice_offset = slice_offset
        self.is_train = is_train

        if target_columns is None:
            # 自动检测 12 个目标列
            self.target_columns = [c for c in self.df.columns if c not in (
                "study_uid", "patient_id", "series_uid", "sop_uid",
                "filepath", "split",
            )]
        else:
            self.target_columns = list(target_columns)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        study_uid = row["study_uid"]

        # 加载三平面切片
        axial = self._load_plane(study_uid, plane="axial")
        coronal = self._load_plane(study_uid, plane="coronal")
        sagittal = self._load_plane(study_uid, plane="sagittal")

        # 标签
        labels = torch.tensor(
            row[self.target_columns].values.astype(np.float32),
            dtype=torch.float32,
        )

        return {
            "axial": axial,
            "coronal": coronal,
            "sagittal": sagittal,
            "labels": labels,
            "study_uid": study_uid,
            "patient_id": str(row.get("patient_id", study_uid)),
        }

    def _load_plane(self, study_uid: str, plane: str) -> torch.Tensor:
        """加载一个平面的 3 个相邻切片 → [3, H, W].

        TODO: 实际实现需要根据 volume 坐标选择对应平面的切片.
        当前为占位, 返回随机张量以便管线联调.
        """
        # TODO: 从 npy_root/study_uid/ 加载对应平面的切片
        volume = torch.randn(3, self.image_size, self.image_size)
        return volume
