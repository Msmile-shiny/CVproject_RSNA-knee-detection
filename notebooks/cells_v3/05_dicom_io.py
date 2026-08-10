# ============================================================
# v3: DICOM I/O — spatial sorting, normalization, 9-slice sampling
# ============================================================

PLANE_SORT_AXIS = {'Sagittal': 0, 'Coronal': 1, 'Axial': 2}

_DICOM_SPECIFIC_TAGS = [
    (0x0020, 0x0032),  # ImagePositionPatient
    (0x0020, 0x0013),  # InstanceNumber
    (0x0028, 0x0002),  # SamplesPerPixel
    (0x0028, 0x0004),  # PhotometricInterpretation
    (0x0028, 0x0010),  # Rows
    (0x0028, 0x0011),  # Columns
    (0x0028, 0x0100),  # BitsAllocated
    (0x0028, 0x0101),  # BitsStored
    (0x0028, 0x0102),  # HighBit
    (0x0028, 0x0103),  # PixelRepresentation
    (0x0028, 0x0030),  # PixelSpacing
    (0x0028, 0x1052),  # RescaleIntercept
    (0x0028, 0x1053),  # RescaleSlope
    (0x7FE0, 0x0010),  # PixelData
]

def _get_slice_position(ds, plane=None):
    try:
        ipp = getattr(ds, 'ImagePositionPatient', None)
        if ipp and len(ipp) >= 3:
            axis = PLANE_SORT_AXIS.get(plane, 2) if plane else 2
            return float(ipp[axis])
    except: pass
    try:
        sl = getattr(ds, 'SliceLocation', None)
        if sl is not None: return float(sl)
    except: pass
    try: return float(getattr(ds, 'InstanceNumber', 0))
    except: return 0.0


def read_series_volume(series_dir, plane=None, image_size=224):
    """Read and normalize all slices in a DICOM series.

    Returns:
        volume: np.ndarray [N_slices, H, W] float32 in [0, 1]
        px: PixelSpacing or None
    """
    series_dir = Path(series_dir)
    dcm_paths = sorted(series_dir.glob('*.dcm'))
    if not dcm_paths:
        dcm_paths = sorted(series_dir.glob('*'))
    if not dcm_paths:
        raise RuntimeError(f'No DICOM files: {series_dir}')

    slices_info = []
    px = None
    for p in dcm_paths:
        try:
            ds = pydicom.dcmread(str(p), force=True, specific_tags=_DICOM_SPECIFIC_TAGS)
            pos = _get_slice_position(ds, plane)
            img = ds.pixel_array.astype(np.float32)
            # Rescale
            slope = float(getattr(ds, 'RescaleSlope', 1) or 1)
            intercept = float(getattr(ds, 'RescaleIntercept', 0) or 0)
            img = img * slope + intercept
            slices_info.append((pos, img))
            if px is None:
                try:
                    ps = getattr(ds, 'PixelSpacing', None)
                    if ps and len(ps) >= 2:
                        px = float(ps[0])
                except: pass
        except Exception:
            continue

    if not slices_info:
        raise RuntimeError(f'No readable DICOM: {series_dir}')

    slices_info.sort(key=lambda x: x[0])
    images = np.stack([img for _, img in slices_info], axis=0)

    # Robust percentile normalization
    v_low = np.percentile(images, 1.0)
    v_high = np.percentile(images, 99.0)
    images = np.clip(images, v_low, v_high)
    denom = max(v_high - v_low, 1e-6)
    images = (images - v_low) / denom

    # Resize
    resized = []
    for img in images:
        r = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        resized.append(r)
    return np.stack(resized, axis=0).astype(np.float32), px


def sample_cache_slices(volume, n_cache=9, center_pct=(0.2, 0.8)):
    """Sample N slices uniformly from the central portion of the stack.

    Matching reference: linspace(low_idx, high_idx, n_cache).
    """
    n_total = volume.shape[0]
    if n_total <= n_cache:
        # Repeat last slice if needed
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


def normalise_laterality(image, plane, laterality):
    """Mirror right knees to left-knee convention (reference code pattern)."""
    if laterality != 'R':
        return image
    # image: [H, W] or [N, H, W]
    if plane in ('Coronal', 'Axial'):
        return np.flip(image, axis=-1)
    return np.flip(image, axis=-2)  # Sagittal: flip H
