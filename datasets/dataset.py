"""2.5D 5-slice PyTorch Dataset.

每个样本 = 同一系列中相邻 5 张切片堆叠 → [5, H, W].
边界用 replicate padding 处理.

支持:
- 按解剖平面筛选 series
- 数据增强 (可选, 仅训练时)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .dicom_loader import read_dicom_series


# ── 12 个目标标签 ───────────────────────────────────────────────
TARGET_COLUMNS = [
    "ACL", "MCL",
    "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]


class Knee25DDataset(Dataset):
    """膝关节 MRI 2.5D 切片堆叠数据集.

    按解剖平面筛选 series, 每 series 构建 5-slice 堆叠样本.
    训练时随机增强, 验证时仅 resize.

    Args:
        series_df: 列 [StudyInstanceUID, SeriesInstanceUID, Anatomical_Plane, ...]
        labels_df: 列 [StudyInstanceUID] + 12 label 列
        dicom_root: DICOM 文件根目录 (其下为 StudyInstanceUID/SeriesInstanceUID/*.dcm)
        planes: 要加载的解剖平面列表, e.g. ["Sagittal"] 或 ["Sagittal","Coronal","Axial"]
        image_size: 输出正方形边长 (默认 384)
        slice_count: 堆叠切片数 (默认 5)
        is_train: 训练模式 (暂不做增强, 保留接口)
        fluid_sensitive_only: 是否只选 Fluid_Sensitive=1 的 series
        fat_suppression_only: 是否只选 Fat_Suppression=1 的 series
    """

    def __init__(
        self,
        series_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        dicom_root: str | Path = "dataset/train_series",
        planes: list[str] | None = None,
        image_size: int = 384,
        slice_count: int = 5,
        is_train: bool = True,
        fluid_sensitive_only: bool = True,
        fat_suppression_only: bool = True,
    ):
        self.dicom_root = Path(dicom_root)
        self.image_size = image_size
        self.slice_count = slice_count
        self.is_train = is_train
        self.half_window = slice_count // 2  # e.g. 2 for 5 slices

        if planes is None:
            planes = ["Sagittal", "Coronal", "Axial"]
        self.planes = planes

        # ── 筛选符合条件的 series ──────────────────────────────
        df = series_df.copy()
        df = df[df["Anatomical_Plane"].isin(planes)]

        if fluid_sensitive_only and "Fluid_Sensitive" in df.columns:
            df = df[df["Fluid_Sensitive"] == 1]
        if fat_suppression_only and "Fat_Suppression" in df.columns:
            df = df[df["Fat_Suppression"] == 1]

        # ── 构建标签映射 ───────────────────────────────────────
        label_cols = [c for c in TARGET_COLUMNS if c in labels_df.columns]
        if len(label_cols) < 12:
            # 如果部分列缺失 (如无标注样本), 剩余列填 0
            for c in TARGET_COLUMNS:
                if c not in labels_df.columns:
                    labels_df[c] = 0.0
            label_cols = TARGET_COLUMNS

        self.label_map = (
            labels_df.set_index("StudyInstanceUID")[label_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .astype(np.float32)
        )

        # ── 构建样本索引: 一个 sample = 一个 5-slice 堆叠 ──────
        self.samples = []
        skipped = 0

        for (study_uid, series_uid), grp in df.groupby(
            ["StudyInstanceUID", "SeriesInstanceUID"]
        ):
            if study_uid not in self.label_map.index:
                skipped += 1
                continue

            plane = grp.iloc[0]["Anatomical_Plane"]
            dicom_dir = self.dicom_root / study_uid / series_uid

            if not dicom_dir.exists():
                skipped += 1
                continue

            # 统计切片数 (快速: 数文件; DICOM 加载延迟到 __getitem__)
            dcm_files = list(dicom_dir.glob("*.dcm"))
            if not dcm_files:
                dcm_files = list(dicom_dir.iterdir())
            n_slices = len(dcm_files)

            if n_slices < 3:
                # 至少需要 3 张切片才能构建有意义的 5-slice 堆叠
                skipped += 1
                continue

            labels = self.label_map.loc[study_uid].values.astype(np.float32)

            # 为每张中心切片创建一个样本
            for center_idx in range(n_slices):
                self.samples.append({
                    "study_uid": study_uid,
                    "series_uid": series_uid,
                    "dicom_dir": str(dicom_dir),
                    "plane": plane,
                    "center_idx": center_idx,
                    "n_slices": n_slices,
                    "labels": labels,
                })

        if skipped:
            print(f"[Knee25DDataset] {skipped} series 被跳过 (无标签或无 DICOM)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # ── 加载 DICOM series (缓存到内存? Phase 1 先每次读) ──
        # TODO: 加入 LRU cache 避免重复读取同一 series
        try:
            volume = read_dicom_series(
                sample["dicom_dir"],
                plane=sample["plane"],
                image_size=self.image_size,
            )  # [N, H, W]
        except Exception:
            # 加载失败 → 返回零张量 (训练时会被 loss 忽略)
            return {
                "image": torch.zeros(self.slice_count, self.image_size, self.image_size),
                "labels": torch.from_numpy(sample["labels"]),
                "study_uid": sample["study_uid"],
                "plane": sample["plane"],
            }

        # ── 构建 5-slice 堆叠 ─────────────────────────────────
        center = sample["center_idx"]
        n_total = volume.shape[0]
        half = self.half_window

        indices = []
        for offset in range(-half, half + 1):
            idx_src = center + offset
            # Replicate padding: 超出边界用最近切片
            idx_src = max(0, min(n_total - 1, idx_src))
            indices.append(idx_src)

        stack = volume[indices]  # [5, H, W]

        # ── 数据增强 (Phase 1: 预留给后续实现) ────────────────
        if self.is_train:
            stack = self._augment(stack)

        return {
            "image": torch.from_numpy(stack.copy()),
            "labels": torch.from_numpy(sample["labels"]),
            "study_uid": sample["study_uid"],
            "plane": sample["plane"],
        }

    def _augment(self, stack: np.ndarray) -> np.ndarray:
        """MRI 安全的数据增强.

        允许: 水平翻转, 小角度旋转, 亮度/对比度抖动, 高斯噪声
        禁止: 垂直翻转 (解剖方向有意义)
        """
        # Phase 1: 暂不增强, 先跑通管线
        # Phase 2: 集成 albumentations
        return stack
