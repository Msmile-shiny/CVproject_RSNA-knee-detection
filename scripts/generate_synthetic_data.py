"""生成合成 DICOM 数据用于本地管线验证.

创建:
  dataset/train_series/
    Study_XX/
      Series_XX_Sagittal/   →  ~16 个合成 .dcm 文件
      Series_XX_Coronal/
      Series_XX_Axial/

同时生成对应的 train.csv 和 train_series.csv.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset

# ── 配置 ─────────────────────────────────────────────────────
N_STUDIES = 20
SLICES_PER_SERIES = 16
IMAGE_SIZE = 512  # 原始 DICOM 尺寸
TARGET_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
PLANES = ["Sagittal", "Coronal", "Axial"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # scripts/ → project root
DICOM_ROOT = PROJECT_ROOT / "dataset" / "train_series"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

# ── 生成 DICOM ────────────────────────────────────────────────


def create_synthetic_dicom(
    filepath: Path,
    study_uid: str,
    series_uid: str,
    sop_uid: str,
    slice_idx: int,
    plane: str,
) -> None:
    """生成单张合成 DICOM 文件."""
    rng = np.random.RandomState(hash(sop_uid) % 2**31)

    # 创建合成像素数据 — 模拟 MRI 切片
    img = rng.randn(IMAGE_SIZE, IMAGE_SIZE).astype(np.float32) * 200 + 1000

    # 添加一些"解剖结构" (圆形/椭圆形区域)
    cx, cy = IMAGE_SIZE // 2 + int(rng.randn() * 30), IMAGE_SIZE // 2 + int(rng.randn() * 30)
    y, x = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
    for _ in range(3):
        rx, ry = 60 + rng.rand() * 80, 40 + rng.rand() * 60
        mask = ((x - cx) ** 2 / rx**2 + (y - cy) ** 2 / ry**2) < 1.0
        img[mask] += rng.rand() * 500
        cx += int(rng.randn() * 20)
        cy += int(rng.randn() * 20)

    img = img.astype(np.uint16)
    img = np.clip(img, 0, 4095)

    # 构建 DICOM 元数据
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    ds = FileDataset(str(filepath), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = pydicom.uid.CTImageStorage
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.PatientID = study_uid[:20]

    # ImagePositionPatient: 根据平面设置排序坐标
    if plane == "Sagittal":
        ds.ImagePositionPatient = [float(slice_idx * 3.0), 0.0, 0.0]
    elif plane == "Coronal":
        ds.ImagePositionPatient = [0.0, float(slice_idx * 3.0), 0.0]
    else:
        ds.ImagePositionPatient = [0.0, 0.0, float(slice_idx * 3.0)]

    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.SliceLocation = str(slice_idx)
    ds.InstanceNumber = slice_idx + 1

    ds.Rows = IMAGE_SIZE
    ds.Columns = IMAGE_SIZE
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = img.tobytes()

    filepath.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(filepath), write_like_original=False)


# ── 主流程 ────────────────────────────────────────────────────


def main():
    print(f"生成 {N_STUDIES} 个合成 Study × {len(PLANES)} 平面...")
    print(f"DICOM 输出: {DICOM_ROOT}")
    print(f"CSV 输出:   {METADATA_DIR}")

    rng = np.random.RandomState(2026)

    study_records = []
    series_records = []
    label_records = []

    for i in range(N_STUDIES):
        study_uid = f"Study_{i:03d}"
        print(f"  {study_uid}...", end=" ")

        # 生成标签 (~10% 正样本率模拟真实分布)
        labels = {}
        for col in TARGET_COLUMNS:
            labels[col] = int(rng.rand() < 0.10)
        # 确保至少有 ~25% 的 study 有异常
        if i < N_STUDIES * 0.25:
            chosen = list(rng.choice(TARGET_COLUMNS, size=rng.randint(1, 4)))
            for c in chosen:
                labels[c] = 1

        # 放射学报告 (占位)
        report = f"Synthetic report for {study_uid}. "
        abnormal = [c for c in TARGET_COLUMNS if labels[c] == 1]
        if abnormal:
            report += f"Findings: {', '.join(abnormal)}."

        label_records.append({"StudyInstanceUID": study_uid, "Report": report, **labels})

        for plane in PLANES:
            series_uid = f"Series_{i:03d}_{plane}"
            sop_base = f"1.2.840.{i:04d}.{PLANES.index(plane):d}"

            # 生成 DICOM 切片
            for slice_idx in range(SLICES_PER_SERIES):
                sop_uid = f"{sop_base}.{slice_idx:04d}"
                dcm_path = DICOM_ROOT / study_uid / series_uid / f"{sop_uid}.dcm"
                create_synthetic_dicom(
                    dcm_path, study_uid, series_uid, sop_uid, slice_idx, plane
                )

            series_records.append({
                "StudyInstanceUID": study_uid,
                "SeriesInstanceUID": series_uid,
                "Fluid_Sensitive": 1 if plane == "Sagittal" else (1 if rng.rand() > 0.3 else 0),
                "Fat_Suppression": 1 if plane in ("Sagittal", "Coronal") else (1 if rng.rand() > 0.3 else 0),
                "Anatomical_Plane": plane,
            })

        study_records.append(study_uid)
        print("done")

    # ── 写 CSV ────────────────────────────────────────────
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    labels_df = pd.DataFrame(label_records)
    labels_df.to_csv(METADATA_DIR / "train.csv", index=False)

    series_df = pd.DataFrame(series_records)
    series_df.to_csv(METADATA_DIR / "train_series.csv", index=False)

    # 同时生成空测试 CSV
    pd.DataFrame({"StudyInstanceUID": [f"Test_{j:03d}" for j in range(5)]}).to_csv(
        METADATA_DIR / "test.csv", index=False
    )
    pd.DataFrame({
        "StudyInstanceUID": [f"Test_{j:03d}" for j in range(5) for _ in range(3)],
        "SeriesInstanceUID": [f"Test_Series_{j:03d}_{p}" for j in range(5) for p in PLANES],
        "Fluid_Sensitive": [1] * 15,
        "Fat_Suppression": [1] * 15,
        "Anatomical_Plane": PLANES * 5,
    }).to_csv(METADATA_DIR / "test_series.csv", index=False)

    # sample submission
    sub = pd.DataFrame(
        {c: 0.5 for c in TARGET_COLUMNS},
        index=[f"Test_{j:03d}" for j in range(5)],
    )
    sub.index.name = "StudyInstanceUID"
    sub.to_csv(METADATA_DIR / "sample_submission.csv")

    # ── 统计 ────────────────────────────────────────────
    n_dcm = sum(1 for _ in DICOM_ROOT.rglob("*.dcm"))
    print(f"\n✅ 完成!")
    print(f"   Study 数:     {len(study_records)}")
    print(f"   Series 数:    {len(series_records)}")
    print(f"   DICOM 切片:   {n_dcm}")
    print(f"   正样本率:     {labels_df[TARGET_COLUMNS].mean().mean():.2%}")
    print(f"   数据大小:     ~{n_dcm * 0.5:.0f} MB")


if __name__ == "__main__":
    main()
