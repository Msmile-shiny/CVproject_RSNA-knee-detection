"""DICOM 文件读取与预处理.

职责:
- 读取单个 series 的全部 DICOM 切片
- 按 ImagePositionPatient 排序
- 像素值归一化 (percentile clip → min-max)
- 缩放到目标尺寸

MRI 安全约束:
- 不做垂直翻转 (解剖方向有意义)
- 不做过度像素变换 (保留组织对比度)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ── DICOM 标签常量 ──────────────────────────────────────────────
TAG_IMAGE_POSITION = (0x0020, 0x0032)   # ImagePositionPatient: [x, y, z]
TAG_ORIENTATION = (0x0020, 0x0037)       # ImageOrientationPatient: [6 floats]

# 解剖平面对应的排序轴 (基于标准 DICOM 坐标系)
PLANE_SORT_AXIS = {
    "Sagittal": 0,   # X 轴: lateral → medial
    "Coronal": 1,     # Y 轴: posterior → anterior
    "Axial": 2,       # Z 轴: inferior → superior
}


def _get_slice_position(ds: "pydicom.Dataset", plane: str | None = None) -> float:
    """从 DICOM header 提取切片空间位置.

    优先使用 ImagePositionPatient 的对应坐标轴,
    若无则回退到 InstanceNumber 或 SliceLocation.
    """
    # 方法 1: ImagePositionPatient
    try:
        ipp = ds.ImagePositionPatient
        if ipp and len(ipp) >= 3:
            axis = PLANE_SORT_AXIS.get(plane, 2) if plane else 2
            val = float(ipp[axis])
            if val is not None:
                return val
    except Exception:
        pass

    # 方法 2: SliceLocation
    try:
        sl = ds.SliceLocation
        if sl is not None:
            return float(sl)
    except Exception:
        pass

    # 方法 3: InstanceNumber
    try:
        return float(ds.InstanceNumber)
    except Exception:
        return 0.0


def read_dicom_series(
    series_dir: str | Path,
    plane: str | None = None,
    image_size: int = 384,
    lower_pct: float = 0.5,
    upper_pct: float = 99.5,
) -> np.ndarray:
    """读取一个 DICOM series 的全部切片, 排序后返回 [N, H, W] 数组.

    Args:
        series_dir: 包含 .dcm 文件的目录
        plane: 解剖平面 (用于确定排序轴)
        image_size: 输出正方形边长
        lower_pct: 像素值下百分位 (用于 clip)
        upper_pct: 像素值上百分位

    Returns:
        [N_slices, H, W] float32 数组, 已排序 & 归一化
    """
    if not HAS_PYDICOM:
        raise ImportError("pydicom 未安装. pip install pydicom")

    series_dir = Path(series_dir)
    dcm_paths = sorted(series_dir.glob("*.dcm"))
    if not dcm_paths:
        dcm_paths = sorted(series_dir.glob("*"))  # 无扩展名兜底
    if not dcm_paths:
        raise FileNotFoundError(f"{series_dir} 中未找到 DICOM 文件")

    # 读取所有切片 + 提取位置
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
        raise RuntimeError(f"无法读取任何 DICOM 切片: {series_dir}")

    # 按空间位置排序
    slices_info.sort(key=lambda x: x[0])

    # 合并 → [N, H, W]
    images = np.stack([img for _, img in slices_info], axis=0)

    # 像素归一化
    images = normalize_dicom(images, lower_pct=lower_pct, upper_pct=upper_pct)

    # 缩放到目标尺寸
    if HAS_CV2:
        resized = []
        for img in images:
            r = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            resized.append(r)
        images = np.stack(resized, axis=0)
    else:
        # 纯 numpy 回退: 简单 crop/pad + resize (粗糙但可用)
        h, w = images.shape[1], images.shape[2]
        if h != image_size or w != image_size:
            # 中心裁剪到正方形, 然后缩放
            sz = min(h, w)
            start_h = (h - sz) // 2
            start_w = (w - sz) // 2
            images = images[:, start_h:start_h+sz, start_w:start_w+sz]
            # 最近邻插值缩放
            from scipy.ndimage import zoom
            scale = image_size / sz
            images = zoom(images, (1, scale, scale), order=1)

    return images


def normalize_dicom(
    images: np.ndarray,
    lower_pct: float = 0.5,
    upper_pct: float = 99.5,
    eps: float = 1e-6,
) -> np.ndarray:
    """百分位裁剪 + min-max 归一化到 [0, 1].

    Args:
        images: [..., H, W] float32 数组
        lower_pct: 下百分位
        upper_pct: 上百分位
        eps: 防除零

    Returns:
        归一化后的 [..., H, W]
    """
    v_low = np.percentile(images, lower_pct)
    v_high = np.percentile(images, upper_pct)
    images = np.clip(images, v_low, v_high)
    images = (images - v_low) / max(v_high - v_low, eps)
    return images.astype(np.float32)
