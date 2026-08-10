"""Tri-Plane 2.5D Dataset — 每个 sample 返回 Sagittal + Coronal + Axial 三平面 5-slice 堆叠.

设计:
- 以 Sagittal 为锚定平面, 每个 Sagittal slice 位置生成一个 sample
- 同时加载对应 study 的 Coronal 和 Axial 系列
- 缺失平面返回全零 tensor, 模型通过 missing_emb 处理
- LRU cache 缓存读过的 volume, 避免重复 DICOM I/O
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .dicom_loader import read_dicom_series
from .dataset import TARGET_COLUMNS


# ── 简单的 volume 缓存 (按路径) ──────────────────────────────────
_volume_cache: dict[str, np.ndarray] = {}
_MAX_CACHE_SIZE = 64


def _cached_read(series_dir: str, plane: str, image_size: int) -> np.ndarray:
    """读取 DICOM volume, 带 LRU 缓存."""
    key = f"{series_dir}@{plane}@{image_size}"
    if key not in _volume_cache:
        if len(_volume_cache) >= _MAX_CACHE_SIZE:
            # 删除最早的条目
            oldest = next(iter(_volume_cache))
            del _volume_cache[oldest]
        _volume_cache[key] = read_dicom_series(series_dir, plane=plane, image_size=image_size)
    return _volume_cache[key]


def clear_cache():
    """清空 volume 缓存 (释放内存)."""
    _volume_cache.clear()


class TriPlaneDataset(Dataset):
    """三平面 2.5D 膝关节 MRI 数据集.

    每个 sample:
        {
            "sag": [5, H, W],   # Sagittal 5-slice stack
            "cor": [5, H, W],   # Coronal 5-slice stack
            "ax":  [5, H, W],   # Axial 5-slice stack
            "labels": [12],     # 多标签
            "study_uid": str,
            "series_uids": {"sag": str, "cor": str, "ax": str},
        }

    缺失平面 → 全零 tensor + 日志 warning (首次).

    Args:
        series_df: train_series.csv
        labels_df: StudyInstanceUID-indexed labels (gold 或 pseudo)
        dicom_root: DICOM 根目录
        image_size: 输出图像尺寸
        slice_count: 堆叠切片数 (默认 5)
        is_train: 训练/验证模式
        planes: 要加载的平面 (默认全部 3 个)
    """

    PLANES = ["Sagittal", "Coronal", "Axial"]

    def __init__(
        self,
        series_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        dicom_root: str | Path = "dataset/train_series",
        image_size: int = 384,
        slice_count: int = 5,
        is_train: bool = True,
        planes: list[str] | None = None,
    ):
        self.dicom_root = Path(dicom_root)
        self.image_size = image_size
        self.slice_count = slice_count
        self.is_train = is_train
        self.half_window = slice_count // 2

        if planes is None:
            planes = self.PLANES
        self.planes = planes

        # ── 构建标签映射 ───────────────────────────────────────
        # Handle both index-already-set and column-has-StudyInstanceUID
        if "StudyInstanceUID" in labels_df.columns:
            label_df_indexed = labels_df.set_index("StudyInstanceUID")
        else:
            label_df_indexed = labels_df.copy()

        label_cols = [c for c in TARGET_COLUMNS if c in label_df_indexed.columns]
        if len(label_cols) < 12:
            for c in TARGET_COLUMNS:
                if c not in label_df_indexed.columns:
                    label_df_indexed[c] = 0.0
            label_cols = TARGET_COLUMNS

        self.label_map = (
            label_df_indexed[label_cols]
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

        # ── 按 Study × Plane 索引最佳 series ────────────────────
        # 每个 (study_uid, plane) 选取一个最佳 series
        df = series_df.copy()
        df = df[df["Anatomical_Plane"].isin(self.planes)]

        # 优先 Fluid_Sensitive + Fat_Suppression
        if "Fluid_Sensitive" in df.columns:
            df["_score"] = df["Fluid_Sensitive"].fillna(0).astype(int)
        else:
            df["_score"] = 0
        if "Fat_Suppression" in df.columns:
            df["_score"] += df["Fat_Suppression"].fillna(0).astype(int)

        # 每个 (study, plane) 取分数最高的 series
        df = df.sort_values("_score", ascending=False)
        best_series = df.groupby(
            ["StudyInstanceUID", "Anatomical_Plane"]
        ).first().reset_index()

        # Pivot: study × plane → series_uid + dicom_dir
        self.study_plane_map: dict[str, dict[str, dict]] = {}
        for _, row in best_series.iterrows():
            sid = row["StudyInstanceUID"]
            plane = row["Anatomical_Plane"]
            series_uid = row["SeriesInstanceUID"]
            dicom_dir = self.dicom_root / sid / series_uid

            if sid not in self.study_plane_map:
                self.study_plane_map[sid] = {}
            self.study_plane_map[sid][plane] = {
                "series_uid": series_uid,
                "dicom_dir": str(dicom_dir),
                "exists": dicom_dir.exists(),
                "n_slices": (
                    len(list(dicom_dir.glob("*.dcm")))
                    if dicom_dir.exists() else 0
                ),
            }

        # ── 构建样本索引 (以 Sagittal 为锚) ──────────────────────
        self.samples: list[dict] = []
        skipped_no_label = 0
        skipped_no_dicom = 0

        for sid, plane_info in self.study_plane_map.items():
            if sid not in self.label_map.index:
                skipped_no_label += 1
                continue

            # 必须有 Sagittal (锚定平面)
            if "Sagittal" not in plane_info or not plane_info["Sagittal"]["exists"]:
                skipped_no_dicom += 1
                continue

            sag_info = plane_info["Sagittal"]
            n_sag = sag_info["n_slices"]
            if n_sag < 3:
                skipped_no_dicom += 1
                continue

            labels = self.label_map.loc[sid].values.astype(np.float32)
            label_weights = self.weight_map.loc[sid].values.astype(np.float32)

            # 每个 Sagittal slice 位置生成一个 sample
            for center_idx in range(n_sag):
                sample = {
                    "study_uid": sid,
                    "labels": labels,
                    "label_weights": label_weights,
                }
                # 每个平面的信息
                for plane in self.planes:
                    if plane in plane_info and plane_info[plane]["exists"]:
                        pinfo = plane_info[plane]
                        p_n = pinfo["n_slices"]
                        # map anchor center_idx to this plane's slice position
                        mapped_center = int(center_idx / max(n_sag - 1, 1) * max(p_n - 1, 1))
                        sample[f"{plane}_dicom_dir"] = pinfo["dicom_dir"]
                        sample[f"{plane}_center_idx"] = mapped_center
                        sample[f"{plane}_n_slices"] = p_n
                        sample[f"{plane}_series_uid"] = pinfo["series_uid"]
                    else:
                        sample[f"{plane}_dicom_dir"] = ""
                        sample[f"{plane}_center_idx"] = 0
                        sample[f"{plane}_n_slices"] = 0
                        sample[f"{plane}_series_uid"] = ""

                self.samples.append(sample)

        if skipped_no_label or skipped_no_dicom:
            print(
                f"[TriPlaneDataset] 跳过: {skipped_no_label} 无标签, "
                f"{skipped_no_dicom} 无 DICOM"
            )
        print(f"[TriPlaneDataset] {len(self.samples)} 个三平面样本 (来自 {len(self.study_plane_map)} studies)")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_stack(self, dicom_dir: str, plane: str, center_idx: int, n_slices: int) -> torch.Tensor:
        """加载单平面 5-slice stack.

        Returns:
            [5, H, W] tensor, 或全零 (缺失时)
        """
        if not dicom_dir or n_slices < 1:
            return torch.zeros(self.slice_count, self.image_size, self.image_size)

        try:
            volume = _cached_read(dicom_dir, plane, self.image_size)
        except Exception:
            return torch.zeros(self.slice_count, self.image_size, self.image_size)

        n_total = volume.shape[0]
        half = self.half_window

        indices = []
        for offset in range(-half, half + 1):
            idx = center_idx + offset
            idx = max(0, min(n_total - 1, idx))
            indices.append(idx)

        stack = volume[indices]  # [5, H, W]
        return torch.from_numpy(stack.copy())

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        sag = self._load_stack(
            sample["Sagittal_dicom_dir"], "Sagittal",
            sample["Sagittal_center_idx"], sample["Sagittal_n_slices"],
        )
        cor = self._load_stack(
            sample["Coronal_dicom_dir"], "Coronal",
            sample["Coronal_center_idx"], sample["Coronal_n_slices"],
        )
        ax = self._load_stack(
            sample["Axial_dicom_dir"], "Axial",
            sample["Axial_center_idx"], sample["Axial_n_slices"],
        )

        return {
            "sag": sag,
            "cor": cor,
            "ax": ax,
            "labels": torch.from_numpy(sample["labels"]),
            "label_weights": torch.from_numpy(sample["label_weights"]),
            "study_uid": sample["study_uid"],
            "series_uids": {
                "sag": sample.get("Sagittal_series_uid", ""),
                "cor": sample.get("Coronal_series_uid", ""),
                "ax": sample.get("Axial_series_uid", ""),
            },
        }
