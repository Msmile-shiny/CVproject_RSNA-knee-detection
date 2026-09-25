# ============================================================
# DICOM I/O — 空间排序 + 物理裁剪 + 侧性归一化
# ============================================================

PLANE_AXIS = {"Sagittal": 0, "Coronal": 1, "Axial": 2}


def _list_dcm_files(series_dir):
    """列出 DICOM 文件（不依赖 .dcm 扩展名）。"""
    sd = Path(series_dir)
    if not sd.is_dir():
        return []
    all_files = sorted(f.name for f in sd.iterdir() if f.is_file())
    dcm = [f for f in all_files if f.endswith('.dcm')]
    return dcm if dcm else [f for f in all_files if not f.startswith('.')]


def spatially_sorted_files(series_dir, plane=None):
    """按 ImagePositionPatient 空间排序。"""
    series_dir = Path(series_dir)
    files = _list_dcm_files(series_dir)
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
        rows.sort(key=lambda r: (
            r[1] if r[1] is not None else 0.0,
            r[2] if r[2] is not None else float('inf'),
        ))
    elif sum(r[2] is not None for r in rows) >= threshold:
        rows.sort(key=lambda r: (
            r[2] if r[2] is not None else float('inf'),
        ))

    return [r[0] for r in rows]


def normalise_laterality(image, plane, laterality):
    """右膝→左膝归一化。"""
    if laterality != 'R':
        return image
    if plane in ('Coronal', 'Axial'):
        return np.flip(image, axis=-1).copy()
    else:
        return np.flip(image, axis=0).copy()


def physical_crop(volume, px, crop_mm=160.0):
    """固定物理 FOV 裁剪。"""
    if px is None or not np.isfinite(px) or px <= 0:
        return volume
    desired = int(round(crop_mm / px))
    h, w = volume.shape[1], volume.shape[2]
    if not (16 < desired < min(h, w)):
        return volume
    cy, cx = h // 2, w // 2
    half = desired // 2
    return volume[:, max(0, cy - half):cy + half, max(0, cx - half):cx + half]


def read_series_volume(series_dir, plane=None, laterality=None,
                       image_size=224, crop_mm=160.0):
    """读取 DICOM → 排序 → 裁剪 → 侧性 → 归一化 → 缩放。"""
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
            slope = float(getattr(ds, 'RescaleSlope', 1) or 1)
            intercept = float(getattr(ds, 'RescaleIntercept', 0) or 0)
            img = img * slope + intercept
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

    volume = np.stack(slices_info, axis=0)
    volume = physical_crop(volume, px, crop_mm)
    volume = normalise_laterality(volume, plane, laterality)

    v_low, v_high = np.percentile(volume, [1.0, 99.0])
    volume = np.clip(volume, v_low, v_high)
    denom = max(v_high - v_low, 1e-6)
    volume = (volume - v_low) / denom

    resized = []
    for img in volume:
        r = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        resized.append(r)
    return np.stack(resized, axis=0).astype(np.float32), px


def sample_cache_slices(volume, n_cache=9, center_pct=(0.2, 0.8)):
    """从 volume central 60% 均匀采样 n_cache 切片。"""
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


print('DICOM I/O ready.')
