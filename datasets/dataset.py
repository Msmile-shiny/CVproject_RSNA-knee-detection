"""2.5D Knee MRI Dataset with Pseudo-Label Support.

Reads DICOM series, builds 5-slice stacks, and serves (image, label) pairs
for training. Supports Kaggle input paths and pseudo-label merging.

DICOM → sort by ImagePosition → percentile normalize → resize → 5-slice stack.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ── 12 target classes ─────────────────────────────────────────
TARGET_COLUMNS = [
    "ACL", "MCL",
    "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

# ── DICOM tag constants ───────────────────────────────────────
TAG_IMAGE_POSITION = (0x0020, 0x0032)

PLANE_SORT_AXIS = {"Sagittal": 0, "Coronal": 1, "Axial": 2}


# ═══════════════════════════════════════════════════════════════
# DICOM I/O helpers (self-contained, no external file dependency)
# ═══════════════════════════════════════════════════════════════


def _get_slice_position(ds, plane: str | None = None) -> float:
    """Extract spatial position from a DICOM slice."""
    try:
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp and len(ipp) >= 3:
            axis = PLANE_SORT_AXIS.get(plane, 2) if plane else 2
            return float(ipp[axis])
    except Exception:
        pass
    try:
        sl = getattr(ds, "SliceLocation", None)
        if sl is not None:
            return float(sl)
    except Exception:
        pass
    try:
        return float(getattr(ds, "InstanceNumber", 0))
    except Exception:
        return 0.0


def read_dicom_series(
    series_dir: str | Path,
    plane: str | None = None,
    image_size: int = 384,
    lower_pct: float = 0.5,
    upper_pct: float = 99.5,
) -> np.ndarray:
    """Read all DICOM slices from a series, sort, normalize, resize.

    Returns:
        [N_slices, H, W] float32 array normalized to [0, 1].
    """
    import pydicom

    series_dir = Path(series_dir)
    dcm_paths = sorted(series_dir.glob("*.dcm"))
    if not dcm_paths:
        dcm_paths = sorted(series_dir.glob("*"))

    slices_info = []
    for p in dcm_paths:
        try:
            ds = pydicom.dcmread(str(p), force=True)
            pos = _get_slice_position(ds, plane)
            img = ds.pixel_array.astype(np.float32)
            slices_info.append((pos, img))
        except Exception:
            continue

    if not slices_info:
        raise RuntimeError(f"No DICOM slices readable: {series_dir}")

    slices_info.sort(key=lambda x: x[0])
    images = np.stack([img for _, img in slices_info], axis=0)  # [N, H, W]

    # Percentile clip + min-max normalize
    v_low = np.percentile(images, lower_pct)
    v_high = np.percentile(images, upper_pct)
    images = np.clip(images, v_low, v_high)
    images = (images - v_low) / max(v_high - v_low, 1e-6)

    # Resize to target size
    try:
        import cv2
        resized = []
        for img in images:
            r = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            resized.append(r)
        images = np.stack(resized, axis=0)
    except ImportError:
        # Fallback: center crop to square, then resize with scipy
        h, w = images.shape[1], images.shape[2]
        sz = min(h, w)
        start_h = (h - sz) // 2
        start_w = (w - sz) // 2
        images = images[:, start_h:start_h + sz, start_w:start_w + sz]
        from scipy.ndimage import zoom
        scale = image_size / sz
        images = zoom(images, (1, scale, scale), order=1)

    return images.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════


class Knee25DPseudoDataset(Dataset):
    """2.5D knee MRI dataset with pseudo-label support.

    Builds 5-slice stack samples from Sagittal T2/PD fat-suppressed series.
    Labels can come from gold annotations, pseudo-labels, or a mix.

    Args:
        series_df: train_series.csv filtered to desired planes/sequences
        labels_df: StudyInstanceUID-indexed DataFrame with 12 label columns
        dicom_root: Path to DICOM directory (Kaggle: /kaggle/input/.../train_images)
        image_size: Output image size (square)
        slice_count: Number of slices per stack (default 5)
        is_train: If True, apply training augmentations (placeholder for now)
    """

    def __init__(
        self,
        series_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        dicom_root: str | Path = "/kaggle/input/rsna-2026-knee-abnormality-detection/train_images",
        image_size: int = 384,
        slice_count: int = 5,
        is_train: bool = True,
    ):
        self.dicom_root = Path(dicom_root)
        self.image_size = image_size
        self.slice_count = slice_count
        self.is_train = is_train
        self.half_window = slice_count // 2  # 2 for 5 slices

        # ── Filter series ──────────────────────────────────────
        # Only Sagittal, T2/PD, fat-suppressed (same as Ensemble baseline)
        df = series_df.copy()
        df = df[df["Anatomical_Plane"] == "Sagittal"]
        if "Fluid_Sensitive" in df.columns:
            df = df[df["Fluid_Sensitive"] == 1]
        if "Fat_Suppression" in df.columns:
            df = df[df["Fat_Suppression"] == 1]

        self.series_df = df

        # ── Build label lookup ─────────────────────────────────
        label_cols = [c for c in TARGET_COLUMNS if c in labels_df.columns]
        self.label_cols = label_cols if label_cols else TARGET_COLUMNS

        self.label_map = (
            labels_df[self.label_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .astype(np.float32)
        )

        # ── Build sample index ─────────────────────────────────
        self.samples: list[dict] = []
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

            dcm_files = list(dicom_dir.glob("*.dcm"))
            if not dcm_files:
                dcm_files = list(dicom_dir.glob("*"))
            n_slices = len(dcm_files)

            if n_slices < 3:
                skipped += 1
                continue

            labels = self.label_map.loc[study_uid].values.astype(np.float32)

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
            print(f"[Knee25DPseudoDataset] {skipped} series skipped (no labels or no DICOM)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        try:
            volume = read_dicom_series(
                sample["dicom_dir"],
                plane=sample["plane"],
                image_size=self.image_size,
            )
        except Exception:
            return {
                "image": torch.zeros(self.slice_count, self.image_size, self.image_size),
                "labels": torch.from_numpy(sample["labels"]),
                "study_uid": sample["study_uid"],
                "plane": sample["plane"],
            }

        # Build 5-slice stack with replicate padding
        center = sample["center_idx"]
        n_total = volume.shape[0]
        half = self.half_window

        indices = []
        for offset in range(-half, half + 1):
            idx_src = center + offset
            idx_src = max(0, min(n_total - 1, idx_src))
            indices.append(idx_src)

        stack = volume[indices]  # [5, H, W]

        # ── Optional augmentation placeholder ──────────────────
        if self.is_train:
            stack = self._augment(stack)

        return {
            "image": torch.from_numpy(stack.copy()),
            "labels": torch.from_numpy(sample["labels"]),
            "study_uid": sample["study_uid"],
            "plane": sample["plane"],
        }

    def _augment(self, stack: np.ndarray) -> np.ndarray:
        """MRI-safe augmentations.

        Allowed: horizontal flip, small rotation, brightness/contrast jitter
        Forbidden: vertical flip (anatomical orientation matters)
        """
        # Phase 1: return as-is, augmentations added later
        return stack
