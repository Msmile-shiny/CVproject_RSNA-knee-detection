# ============================================================
# v4: DICOM I/O — 空间排序 + 物理裁剪 + 侧性归一化 + 并行读取
# ============================================================

# ---- 空间切片排序 (Ref2: dominant_axis) ----
PLANE_AXIS = {"Sagittal": 0, "Coronal": 1, "Axial": 2}

def spatially_sorted_files(series_dir, plane=None):
    """按 ImagePositionPatient 在切片法线方向上的投影排序。
    文件名排序的 Spearman 相关系数仅 0.009——完全随机。
    """
    series_dir = Path(series_dir)
    files = sorted(f.name for f in series_dir.iterdir() if f.name.endswith('.dcm'))
    if not files:
        return []

    axis = PLANE_AXIS.get(plane, 2)
    rows = []
    for fname in files:
        try:
            ds = pydicom.dcmread(
                str(series_dir / fname), stop_before_pixels=True, force=True,
                specific_tags=['ImagePositionPatient', 'InstanceNumber'])
            ipp = getattr(ds, 'ImagePositionPatient', None)
            instance = getattr(ds, 'InstanceNumber', None)
            if ipp is not None and len(ipp) >= 3:
                candidate = np.array(ipp[:3], dtype=np.float64)
                pos = float(candidate[axis]) if np.isfinite(candidate).all() else None
            else:
                pos = None
            inst_val = float(instance) if instance is not None else None
        except Exception:
            pos, inst_val = None, None
        rows.append((fname, pos, inst_val))

    positioned = [r for r in rows if r[1] is not None]
    threshold = max(2, int(0.8 * len(rows)))

    if len(positioned) >= threshold:
        # 主排序：通过平面坐标
        rows.sort(key=lambda r: (
            r[1] if r[1] is not None else 0.0,
            r[2] if r[2] is not None else float('inf'),
        ))
    elif sum(r[2] is not None for r in rows) >= threshold:
        rows.sort(key=lambda r: (
            r[2] if r[2] is not None else float('inf'),
        ))
    # else: 保持文件名顺序

    return [r[0] for r in rows]


# ---- 侧性归一化 ----
def normalise_laterality(image, plane, laterality):
    """右膝映射为左膝：冠/轴面水平翻转，矢面反转切片顺序。"""
    if laterality != 'R':
        return image
    # image: [N_slices, H, W] numpy
    if plane in ('Coronal', 'Axial'):
        return np.flip(image, axis=-1).copy()  # 水平翻转
    else:
        return np.flip(image, axis=0).copy()    # 反转切片顺序


# ---- 物理裁剪 ----
def physical_crop(volume, px, crop_mm=160.0):
    """基于 PixelSpacing 裁剪到固定物理 FOV，消除不同扫描仪的空间尺度差异。"""
    if px is None or not np.isfinite(px) or px <= 0:
        return volume
    desired = int(round(crop_mm / px))
    h, w = volume.shape[1], volume.shape[2]
    if not (16 < desired < min(h, w)):
        return volume
    cy, cx = h // 2, w // 2
    half = desired // 2
    return volume[:, max(0, cy - half):cy + half, max(0, cx - half):cx + half]


# ---- 读取单 series 为 volume ----
def read_series_volume(series_dir, plane=None, laterality=None,
                       image_size=224, crop_mm=160.0):
    """读取 DICOM 序列 → 空间排序 → 物理裁剪 → 侧性归一化 → 归一化 → 缩放。"""
    sorted_files = spatially_sorted_files(series_dir, plane)
    if not sorted_files:
        return None, None

    series_dir = Path(series_dir)
    slices_info = []
    px = None

    for fname in sorted_files:
        try:
            ds = pydicom.dcmread(str(series_dir / fname), force=True)
            img = ds.pixel_array.astype(np.float32)

            # Rescale
            slope = float(getattr(ds, 'RescaleSlope', 1) or 1)
            intercept = float(getattr(ds, 'RescaleIntercept', 0) or 0)
            img = img * slope + intercept

            # PixelSpacing (取第一个有效值)
            if px is None:
                ps = getattr(ds, 'PixelSpacing', None)
                if ps is not None and len(ps) >= 1:
                    try:
                        px = float(ps[0])
                    except Exception:
                        pass

            slices_info.append(img)
        except Exception:
            slices_info.append(np.zeros((image_size, image_size), dtype=np.float32))

    if not slices_info:
        return None, None

    volume = np.stack(slices_info, axis=0)  # [N, H, W]

    # 物理裁剪
    volume = physical_crop(volume, px, crop_mm)

    # 侧性归一化
    volume = normalise_laterality(volume, plane, laterality)

    # 鲁棒归一化 (1st-99th percentile)
    v_low, v_high = np.percentile(volume, [1.0, 99.0])
    volume = np.clip(volume, v_low, v_high)
    denom = max(v_high - v_low, 1e-6)
    volume = (volume - v_low) / denom

    # 缩放到 target size
    resized = []
    for img in volume:
        r = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        resized.append(r)
    return np.stack(resized, axis=0).astype(np.float32), px


# ---- 缓存切片采样 ----
def sample_cache_slices(volume, n_cache=9, center_pct=(0.2, 0.8)):
    """从 volume 的 central 60% 区域均匀采样 n_cache 个切片。"""
    n_total = volume.shape[0]
    if n_total <= n_cache:
        indices = list(range(n_total))
        while len(indices) < n_cache:
            indices.append(indices[-1])
        return volume[np.array(indices)]

    low = int(center_pct[0] * (n_total - 1))
    high = int(center_pct[1] * (n_total - 1))
    if high <= low:
        low, high = 0, n_total - 1
    indices = np.unique(np.linspace(low, high, n_cache).astype(int))
    while len(indices) < n_cache:
        indices = np.append(indices, indices[-1])
    return volume[indices[:n_cache]]

print('DICOM I/O v4 ready.')
