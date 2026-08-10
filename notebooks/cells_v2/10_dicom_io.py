# ============================================================
# DICOM I/O helpers (fast path: specific_tags to minimise I/O)
# ============================================================

PLANE_SORT_AXIS = {'Sagittal': 0, 'Coronal': 1, 'Axial': 2}

_DICOM_SPECIFIC_TAGS = [
    (0x0020, 0x0032), (0x0020, 0x1041), (0x0020, 0x0013),
    (0x0028, 0x0002), (0x0028, 0x0004), (0x0028, 0x0010),
    (0x0028, 0x0011), (0x0028, 0x0100), (0x0028, 0x0101),
    (0x0028, 0x0102), (0x0028, 0x0103), (0x7FE0, 0x0010),
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


def read_dicom_series(series_dir, plane=None, image_size=392, lower_pct=0.5, upper_pct=99.5):
    series_dir = Path(series_dir)
    dcm_paths = sorted(series_dir.glob('*.dcm'))
    if not dcm_paths: dcm_paths = sorted(series_dir.glob('*'))

    slices_info = []
    for p in dcm_paths:
        try:
            ds = pydicom.dcmread(str(p), force=True, specific_tags=_DICOM_SPECIFIC_TAGS)
            pos = _get_slice_position(ds, plane)
            img = ds.pixel_array.astype(np.float32)
            slices_info.append((pos, img))
        except: continue

    if not slices_info: raise RuntimeError(f'No DICOM readable: {series_dir}')
    slices_info.sort(key=lambda x: x[0])
    images = np.stack([img for _, img in slices_info], axis=0)

    v_low = np.percentile(images, lower_pct)
    v_high = np.percentile(images, upper_pct)
    images = np.clip(images, v_low, v_high)
    images = (images - v_low) / max(v_high - v_low, 1e-6)

    resized = []
    for img in images:
        r = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        resized.append(r)
    return np.stack(resized, axis=0).astype(np.float32)
