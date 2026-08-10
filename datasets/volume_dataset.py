"""3D Volume Dataset — 将整组 DICOM 序列加载为 3D volume.

与 TriPlaneDataset 不同: 不生成逐切片样本, 而是返回整个 volume 的子段.

设计:
- 选取最佳 Sagittal 系列 (最高 Fluid_Sensitive + Fat_Suppression 分)
- 从 volume 中心取 volume_depth 个连续切片
- Resize 到 volume_size × volume_size
- 返回 [1, volume_depth, volume_size, volume_size]

缺失处理:
- 无 Sagittal 或切片不足 → 跳过该 study
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .dicom_loader import read_dicom_series

# ── TARGET_COLUMNS ────────────────────────────────────────────
TARGET_COLUMNS = [
    "ACL", "MCL",
    "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]


class VolumeDataset(Dataset):
    """3D MRI volume 数据集.

    每个 sample 返回一个固定尺寸的 3D sub-volume.

    Args:
        series_df: train_series.csv / test_series.csv
        labels_df: StudyInstanceUID-indexed 标签 DataFrame
        dicom_root: DICOM 根目录
        volume_depth: 输出 volume 深度 (切片数, 默认 32)
        volume_size: 输出空间尺寸 (默认 128)
        is_train: 训练/验证模式
        plane: 选取的解剖平面 (默认 Sagittal)
    """

    def __init__(
        self,
        series_df: "pd.DataFrame",
        labels_df: "pd.DataFrame",
        dicom_root: str | Path = "dataset/train_series",
        volume_depth: int = 32,
        volume_size: int = 128,
        is_train: bool = True,
        plane: str = "Sagittal",
    ):
        import pandas as pd

        self.dicom_root = Path(dicom_root)
        self.volume_depth = volume_depth
        self.volume_size = volume_size
        self.is_train = is_train
        self.plane = plane
        self.half_depth = volume_depth // 2

        # ── 构建标签映射 ───────────────────────────────────────
        if "StudyInstanceUID" in labels_df.columns:
            label_df_indexed = labels_df.set_index("StudyInstanceUID")
        else:
            label_df_indexed = labels_df.copy()

        label_cols = [c for c in TARGET_COLUMNS if c in label_df_indexed.columns]
        if len(label_cols) < 12:
            for c in TARGET_COLUMNS:
                if c not in label_df_indexed.columns:
                    label_df_indexed[c] = 0.0

        self.label_map = (
            label_df_indexed[TARGET_COLUMNS]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .astype(np.float32)
        )
        self.weight_map = pd.DataFrame(index=self.label_map.index)
        for target in TARGET_COLUMNS:
            column = f"weight_{target}"
            self.weight_map[target] = (
                pd.to_numeric(label_df_indexed[column], errors="coerce").fillna(0.0)
                if column in label_df_indexed.columns else 1.0
            )
        self.weight_map = self.weight_map.astype(np.float32)

        # ── 选取每个 study 的最佳 series ────────────────────────
        df = series_df.copy()
        df = df[df["Anatomical_Plane"] == plane]

        # 评分: Fluid_Sensitive + Fat_Suppression 越高越好
        if "Fluid_Sensitive" in df.columns:
            df["_score"] = df["Fluid_Sensitive"].fillna(0).astype(int)
        else:
            df["_score"] = 0
        if "Fat_Suppression" in df.columns:
            df["_score"] += df["Fat_Suppression"].fillna(0).astype(int)

        df = df.sort_values("_score", ascending=False)
        best_series = df.groupby("StudyInstanceUID").first().reset_index()

        # ── 构建样本列表 ───────────────────────────────────────
        self.samples: list[dict] = []
        skipped_no_label = 0
        skipped_no_dicom = 0

        for _, row in best_series.iterrows():
            sid = row["StudyInstanceUID"]
            if sid not in self.label_map.index:
                skipped_no_label += 1
                continue

            series_uid = row["SeriesInstanceUID"]
            dicom_dir = self.dicom_root / sid / series_uid

            if not dicom_dir.exists():
                skipped_no_dicom += 1
                continue

            # 快速统计切片数
            dcm_files = list(dicom_dir.glob("*.dcm"))
            if not dcm_files:
                dcm_files = list(dicom_dir.iterdir())
            n_slices = len(dcm_files)

            if n_slices < 3:
                skipped_no_dicom += 1
                continue

            # 取中心位置切片范围
            center = n_slices // 2
            start = max(0, center - self.half_depth)
            end = min(n_slices, center + self.half_depth + 1)

            labels = self.label_map.loc[sid].values.astype(np.float32)
            label_weights = self.weight_map.loc[sid].values.astype(np.float32)

            self.samples.append({
                "study_uid": sid,
                "series_uid": series_uid,
                "dicom_dir": str(dicom_dir),
                "n_slices": n_slices,
                "slice_start": start,
                "slice_end": end,
                "labels": labels,
                "label_weights": label_weights,
            })

        if skipped_no_label or skipped_no_dicom:
            print(
                f"[VolumeDataset] 跳过: {skipped_no_label} 无标签, "
                f"{skipped_no_dicom} 无 DICOM"
            )
        print(f"[VolumeDataset] {len(self.samples)} 个 3D volume 样本 "
              f"(plane={plane}, depth={volume_depth}, size={volume_size})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # ── 加载 volume ────────────────────────────────────────
        try:
            volume = read_dicom_series(
                sample["dicom_dir"],
                plane=self.plane,
                image_size=self.volume_size,
            )  # [N, volume_size, volume_size]
        except Exception:
            return {
                "volume": torch.zeros(1, self.volume_depth, self.volume_size, self.volume_size),
                "labels": torch.from_numpy(sample["labels"]),
                "label_weights": torch.from_numpy(sample["label_weights"]),
                "study_uid": sample["study_uid"],
            }

        n_total = volume.shape[0]
        start = sample["slice_start"]
        end = sample["slice_end"]

        # 取出目标范围, 不足用 replicate padding 补齐
        valid = volume[max(0, start):min(n_total, end)]
        need = self.volume_depth

        if valid.shape[0] < need:
            deficit = need - valid.shape[0]
            pad_before = deficit // 2
            pad_after = deficit - pad_before
            valid = np.concatenate([
                np.repeat(valid[:1], pad_before, axis=0),
                valid,
                np.repeat(valid[-1:], pad_after, axis=0),
            ], axis=0)
        elif valid.shape[0] > need:
            # 截断到精确长度
            trim_start = (valid.shape[0] - need) // 2
            valid = valid[trim_start:trim_start + need]

        # [D, H, W] → [1, D, H, W]
        volume_tensor = torch.from_numpy(valid.copy()).unsqueeze(0)

        return {
            "volume": volume_tensor,
            "labels": torch.from_numpy(sample["labels"]),
            "label_weights": torch.from_numpy(sample["label_weights"]),
            "study_uid": sample["study_uid"],
        }
