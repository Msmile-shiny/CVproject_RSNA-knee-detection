# ============================================================
# Slot Matching + DICOM Header Scan (test set)
# ============================================================

_SEP = re.compile(r'[_\-.]')
_FATSAT_RX = re.compile(
    r'\bfs\b|fatsat|fat sat|\bstir\b|\bspair\b|\bspir\b|\bwe\b|'
    r'water excit|\btirm\b|\bsting\b|\bfatsup\b'
)
_T1_RX = re.compile(r'\bt1\b|\bt1w\b')
_T2_RX = re.compile(r'\bt2\b|\bt2w\b')
_PD_RX = re.compile(r'\bpd\b|\bpdw\b|proton|\bdp\b|dens')


def _find_dicom_files(series_dir):
    """列出 DICOM 文件（不依赖 .dcm 扩展名，test 集可能无后缀）。"""
    sd = Path(series_dir)
    if not sd.is_dir():
        return []
    all_files = sorted([f for f in sd.iterdir() if f.is_file()])
    dcm = [f for f in all_files if f.suffix == '.dcm']
    return dcm if dcm else [f for f in all_files if not f.name.startswith('.')]


def _scan_test_dicoms(dicom_root):
    """扫描 test DICOM 目录，从 header 推断 plane / fluid / fatsat。"""
    rows = []
    root = Path(dicom_root)
    if not root.exists():
        return rows
    for study_dir in sorted(root.iterdir()):
        if not study_dir.is_dir():
            continue
        study_uid = study_dir.name
        for series_dir in sorted(study_dir.iterdir()):
            if not series_dir.is_dir():
                continue
            series_uid = series_dir.name
            dcm_files = _find_dicom_files(series_dir)
            if not dcm_files:
                continue
            try:
                ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True, force=True)

                # Anatomical Plane
                iop = getattr(ds, 'ImageOrientationPatient', None)
                plane = 'Axial'
                if iop is not None and len(iop) >= 6:
                    try:
                        row_cos = np.array([float(iop[0]), float(iop[1]), float(iop[2])])
                        col_cos = np.array([float(iop[3]), float(iop[4]), float(iop[5])])
                        normal = np.cross(row_cos, col_cos)
                        dominant = int(np.argmax(np.abs(normal)))
                        plane = {0: 'Sagittal', 1: 'Coronal', 2: 'Axial'}[dominant]
                    except Exception:
                        pass

                # Fat Suppression
                desc = str(getattr(ds, 'SeriesDescription', '')).lower()
                scan_opts = str(getattr(ds, 'ScanOptions', '')).upper()
                fs_kw = ['fs', 'fatsat', 'fat sat', 'stir', 'spair', 'spir', 'we',
                         'water excit', 'tirm', 'fatsup']
                has_fs = any(kw in desc for kw in fs_kw)
                has_fs = has_fs or any(kw in scan_opts for kw in ['FS', 'FATSAT', 'SPAIR', 'SPIR'])

                # Fluid Sensitive
                seq_name = str(getattr(ds, 'SequenceName', '')).lower()
                is_t1 = any(kw in desc or kw in seq_name for kw in ['t1', 't1w'])
                is_t2 = any(kw in desc or kw in seq_name for kw in ['t2', 't2w'])
                is_pd = any(kw in desc for kw in ['pd', 'pdw', 'proton', 'dp', 'dens'])
                has_fluid = (is_t2 or is_pd) and not is_t1

                rows.append({
                    'StudyInstanceUID': study_uid,
                    'SeriesInstanceUID': series_uid,
                    'Anatomical_Plane': plane,
                    'Fluid_Sensitive': 1 if has_fluid else 0,
                    'Fat_Suppression': 1 if has_fs else 0,
                })
            except Exception:
                continue
    return rows


# ---- Slot Matching ----
def match_slots_for_study(study_series_df):
    """为单个 study 的每个 slot 匹配最优 series。"""
    slots_found = {}
    for slot_name, plane, fluid, fatsat in SLOTS:
        candidates = study_series_df[
            (study_series_df['Anatomical_Plane'] == plane)
            & (study_series_df['Fluid_Sensitive'] == (1 if fluid else 0))
            & (study_series_df['Fat_Suppression'] == (1 if fatsat else 0))
        ]
        if len(candidates) == 0 and not fluid:
            candidates = study_series_df[
                (study_series_df['Anatomical_Plane'] == plane)
                & (study_series_df['Fluid_Sensitive'] == 0)
            ]
        if len(candidates) > 0:
            best = candidates.sort_values('n_slices', ascending=False).iloc[0]
            slots_found[slot_name] = {
                'series_uid': best['SeriesInstanceUID'],
                'dir': best['dir'],
                'n_slices': int(best['n_slices']),
                'plane': plane,
            }
        else:
            slots_found[slot_name] = None
    return slots_found


def build_study_slot_map(series_meta, dicom_root):
    """为所有 study 构建 slot→series 映射。"""
    df = series_meta.copy()
    df['StudyInstanceUID'] = df['StudyInstanceUID'].astype(str)
    df['SeriesInstanceUID'] = df['SeriesInstanceUID'].astype(str)

    dirs, n_slices_list = [], []
    for _, row in df.iterrows():
        d = str(dicom_root / row['StudyInstanceUID'] / row['SeriesInstanceUID'])
        dirs.append(d)
        if os.path.isdir(d):
            files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
            n_dcm = len([f for f in files if f.endswith('.dcm')])
            if n_dcm == 0:
                n_dcm = len([f for f in files if not f.startswith('.')])
            n_slices_list.append(n_dcm)
        else:
            n_slices_list.append(0)
    df['dir'] = dirs
    df['n_slices'] = n_slices_list

    slot_map = {}
    for study_uid, grp in df.groupby('StudyInstanceUID'):
        slot_map[study_uid] = match_slots_for_study(grp)

    if IS_MAIN:
        slot_counts = {}
        for slots in slot_map.values():
            for name, sid in slots.items():
                slot_counts[name] = slot_counts.get(name, 0) + (1 if sid is not None else 0)
        n_studies = len(slot_map)
        print(f'Slot map: {n_studies} studies')
        for name, count in slot_counts.items():
            print(f'  {name:<18s}: {count:5d}/{n_studies} ({count/max(n_studies,1)*100:.0f}%)')

    return slot_map


print('Slot matching ready.')
