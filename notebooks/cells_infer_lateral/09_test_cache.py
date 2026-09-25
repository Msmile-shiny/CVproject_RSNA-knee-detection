# ============================================================
# Test 集 slot 匹配 + 双分辨率缓存 (v5 288px/9片 与 rad 224px/7片)
# 与 v5 推理 cell Part 4 / v6a 训练 cell 17 逐字一致
# ============================================================

print('\n--- Test Set Slot Matching ---')

test_df = pd.read_csv(comp_input / 'test.csv')
test_df['StudyInstanceUID'] = test_df['StudyInstanceUID'].astype(str)
test_dicom_root = comp_input / 'test_series'


def _find_dicom_files(series_dir):
    """列出目录中的 DICOM 文件（不依赖扩展名，竞赛 test 集 DICOM 无 .dcm 后缀）。"""
    all_files = sorted([f for f in series_dir.iterdir() if f.is_file()])
    dcm = [f for f in all_files if f.suffix == '.dcm']
    return dcm if dcm else [f for f in all_files if not f.name.startswith('.')]


def _scan_test_dicoms(dicom_root):
    """扫描测试集 DICOM 目录，从 header 推断 plane / fluid / fatsat。"""
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

                desc = str(getattr(ds, 'SeriesDescription', '')).lower()
                seq_name = str(getattr(ds, 'SequenceName', '')).lower()
                scan_opts = str(getattr(ds, 'ScanOptions', '')).upper()

                fs_kw = ['fs', 'fatsat', 'fat sat', 'stir', 'spair', 'spir', 'we',
                         'water excit', 'tirm', 'fatsup']
                has_fs = any(kw in desc for kw in fs_kw)
                has_fs = has_fs or any(kw in scan_opts for kw in ['FS', 'FATSAT', 'SPAIR', 'SPIR'])

                t1_kw = ['t1', 't1w']
                is_t1 = any(kw in desc or kw in seq_name for kw in t1_kw)
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


test_slot_map = {}
test_series_path = comp_input / 'test_series.csv'

if test_series_path.exists():
    test_series = pd.read_csv(test_series_path)
    test_series['StudyInstanceUID'] = test_series['StudyInstanceUID'].astype(str)
    test_series['SeriesInstanceUID'] = test_series['SeriesInstanceUID'].astype(str)
    print(f'test_series.csv: {len(test_series)} series, '
          f'{test_series["StudyInstanceUID"].nunique()} studies')

    test_slot_map, _ = build_study_slot_map(test_series, test_dicom_root)
    csv_studies = len(test_slot_map)

    if csv_studies < max(10, len(test_df) * 0.5):
        print(f'CSV coverage ({csv_studies}/{len(test_df)}) insufficient, '
              f'scanning DICOM headers...')
        dicom_rows = _scan_test_dicoms(test_dicom_root)
        if dicom_rows:
            test_series = pd.DataFrame(dicom_rows)
            test_slot_map, _ = build_study_slot_map(test_series, test_dicom_root)
            print(f'DICOM scan: {len(test_series)} series, '
                  f'{test_series["StudyInstanceUID"].nunique()} studies → '
                  f'{len(test_slot_map)} studies matched')
        else:
            print(f'DICOM scan returned 0 rows, keeping CSV results ({csv_studies} studies)')
else:
    print('test_series.csv not found, scanning DICOM headers...')
    dicom_rows = _scan_test_dicoms(test_dicom_root)
    if dicom_rows:
        test_series = pd.DataFrame(dicom_rows)
        test_slot_map, _ = build_study_slot_map(test_series, test_dicom_root)
        print(f'DICOM scan: {len(test_series)} series, '
              f'{len(test_slot_map)} studies matched')

test_studies = sorted(test_slot_map.keys())
print(f'Test studies with slot match: {len(test_studies)}/{len(test_df)}')

# ---- 双分辨率测试缓存 ----
test_study_idx = {uid: i for i, uid in enumerate(test_studies)}


def _build_test_cache(image_size, cache_slices, crop_mm, tag):
    """为全部 test 研究构建缓存 [n_test, 6, cache_slices, H, W] uint8。"""
    if len(test_studies) == 0:
        return None, None
    n_test = len(test_studies)
    cache = np.zeros((n_test, N_SLOT, cache_slices, image_size, image_size),
                     dtype=np.uint8)
    mask = np.zeros((n_test, N_SLOT), dtype=np.float32)

    t0 = time.time()
    jobs = []
    for row_idx, study_uid in enumerate(test_studies):
        study_slots = test_slot_map[study_uid]
        for slot_idx, (slot_name, plane, fluid, fatsat) in enumerate(SLOTS):
            slot_info = study_slots.get(slot_name)
            if slot_info is not None:
                jobs.append((row_idx, slot_idx, slot_name, plane, slot_info, None,
                             image_size, cache_slices, crop_mm, (0.2, 0.8)))

    print(f'Decoding {len(jobs)} test slot-series @ {image_size}px/{cache_slices} slices...')
    completed, failed = 0, 0
    with ThreadPoolExecutor(max_workers=CFG_V5['pix_threads']) as pool:
        for row_idx, slot_idx, result in pool.map(_read_slot_job, jobs):
            completed += 1
            if result is not None:
                cache[row_idx, slot_idx] = result
                mask[row_idx, slot_idx] = 1.0
            else:
                failed += 1
            if completed % 1000 == 0:
                print(f'  [{completed}/{len(jobs)}] {time.time()-t0:.0f}s', flush=True)

    print(f'{tag} test cache: {n_test} studies, '
          f'{cache.nbytes / 1024**2:.1f} MB in {time.time()-t0:.0f}s ({failed} failed)')
    return cache, mask


TEST_CACHE_V5, TEST_MASK_V5 = _build_test_cache(
    CFG_V5['image_size'], CFG_V5['cache_slices'], CFG_V5['crop_mm'], 'v5-288')
TEST_CACHE_RAD, TEST_MASK_RAD = _build_test_cache(
    CFG_RAD['image_size'], CFG_RAD['cache_slices'], CFG_RAD['crop_mm'], 'rad-224')
gc.collect()
