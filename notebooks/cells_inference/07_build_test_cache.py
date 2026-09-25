# ============================================================
# Build Test Cache — 扫描 DICOM + 构建槽位映射 + 并行读取
# ============================================================

comp_input = Path(CFG['comp_input'])
test_dicom_root = comp_input / CFG['test_dicom_subdir']
output_dir = Path(CFG['output_dir'])

# ---- Load test.csv ----
test_df = pd.read_csv(comp_input / 'test.csv')
test_df['StudyInstanceUID'] = test_df['StudyInstanceUID'].astype(str)
print(f'Test studies (from test.csv): {len(test_df)}')

# ---- Build test series metadata ----
# 优先 test_series.csv，不足则扫描 DICOM headers
test_series_path = comp_input / 'test_series.csv'
test_series_loaded = False

# ★ 重写逻辑：CSV 结果不会被 DICOM scan 失败覆盖
test_slot_map = {}

if test_series_path.exists():
    test_series = pd.read_csv(test_series_path)
    test_series['StudyInstanceUID'] = test_series['StudyInstanceUID'].astype(str)
    test_series['SeriesInstanceUID'] = test_series['SeriesInstanceUID'].astype(str)
    print(f'test_series.csv: {len(test_series)} series, '
          f'{test_series["StudyInstanceUID"].nunique()} studies')

    test_slot_map = build_study_slot_map(test_series, test_dicom_root)
    csv_studies = len(test_slot_map)

    if csv_studies < max(10, len(test_df) * 0.5):
        print(f'CSV coverage ({csv_studies}/{len(test_df)}) insufficient, '
              f'scanning DICOM headers...')
        dicom_rows = _scan_test_dicoms(test_dicom_root)
        if dicom_rows:
            test_series = pd.DataFrame(dicom_rows)
            test_slot_map = build_study_slot_map(test_series, test_dicom_root)
            print(f'DICOM scan: {len(test_series)} series, '
                  f'{test_series["StudyInstanceUID"].nunique()} studies → '
                  f'{len(test_slot_map)} studies matched')
        else:
            # DICOM scan 失败 → 保留 CSV 结果（即使不完整）
            print(f'DICOM scan returned 0 rows, keeping CSV results ({csv_studies} studies)')
    # else: CSV 覆盖率够了，直接用
else:
    print('test_series.csv not found, scanning DICOM headers...')
    dicom_rows = _scan_test_dicoms(test_dicom_root)
    if dicom_rows:
        test_series = pd.DataFrame(dicom_rows)
        test_slot_map = build_study_slot_map(test_series, test_dicom_root)
        print(f'DICOM scan: {len(test_series)} series, '
              f'{len(test_slot_map)} studies matched')

test_studies = sorted(test_slot_map.keys())
print(f'Test studies with slot match: {len(test_studies)}/{len(test_df)}')

if len(test_studies) == 0:
    raise RuntimeError(
        'No test studies found with slot matching! '
        'Check that test DICOMs exist at: ' + str(test_dicom_root))

# ---- 快速侧性检测（每个 study 扫描一个 DICOM header）----
def _detect_laterality_fast(slot_map, dicom_root):
    laterality_map = {}
    root = Path(dicom_root)
    for study_uid, study_slots in slot_map.items():
        lat = None
        for slot_name, slot_info in study_slots.items():
            if slot_info is None:
                continue
            series_dir = Path(slot_info['dir']) if 'dir' in slot_info else None
            if series_dir is None or not series_dir.exists():
                continue
            dcm_files = _list_dcm_files(series_dir)
            if not dcm_files:
                continue
            try:
                ds = pydicom.dcmread(
                    str(series_dir / dcm_files[0]), stop_before_pixels=True, force=True,
                    specific_tags=['Laterality', 'ImageLaterality', 'ImagePositionPatient'])
                for tag_name in ['Laterality', 'ImageLaterality']:
                    val = getattr(ds, tag_name, None)
                    if val is not None:
                        val = str(val).strip().upper()
                        if val and val[0] in ('L', 'R'):
                            lat = val[0]
                            break
                if lat is not None:
                    break
                ipp = getattr(ds, 'ImagePositionPatient', None)
                if ipp is not None and len(ipp) >= 1:
                    try:
                        x = float(str(ipp[0]).split('\\')[0].split('|')[0])
                        if abs(x) >= 5.0:
                            lat = 'R' if x < 0 else 'L'
                            break
                    except Exception:
                        pass
            except Exception:
                continue
        laterality_map[study_uid] = lat
    return laterality_map


t_lat = time.time()
laterality_map = _detect_laterality_fast(test_slot_map, test_dicom_root)
n_lat = sum(1 for v in laterality_map.values() if v is not None)
if IS_MAIN:
    print(f'Laterality detected: {n_lat}/{len(laterality_map)} studies '
          f'({n_lat/max(len(laterality_map),1)*100:.1f}%), '
          f'({time.time()-t_lat:.1f}s)')

# ---- Pre-allocate cache ----
n_test = len(test_studies)
TEST_CACHE = np.zeros((n_test, N_SLOT, CFG['cache_slices'], CFG['image_size'], CFG['image_size']), dtype=np.uint8)
TEST_MASK = np.zeros((n_test, N_SLOT), dtype=np.float32)
test_study_idx = {}

for row_idx, study_uid in enumerate(test_studies):
    test_study_idx[study_uid] = row_idx

# ---- Parallel DICOM read ----
def _read_slot_job(args):
    row_idx, slot_idx, slot_name, plane, slot_info, laterality = args
    if slot_info is None:
        return row_idx, slot_idx, None
    series_dir = Path(slot_info['dir']) if 'dir' in slot_info else None
    if series_dir is None or not series_dir.exists():
        return row_idx, slot_idx, None
    try:
        volume, px = read_series_volume(
            str(series_dir), plane=plane, laterality=laterality,
            image_size=CFG['image_size'], crop_mm=CFG['crop_mm'])
        if volume is None or volume.shape[0] < 3:
            return row_idx, slot_idx, None
        sampled = sample_cache_slices(
            volume, n_cache=CFG['cache_slices'], center_pct=CFG['center_pct'])
        sampled_uint8 = (sampled * 255).clip(0, 255).round().astype(np.uint8)
        return row_idx, slot_idx, sampled_uint8
    except Exception:
        return row_idx, slot_idx, None


jobs = []
for row_idx, study_uid in enumerate(test_studies):
    study_slots = test_slot_map[study_uid]
    lat = laterality_map.get(study_uid)
    for slot_idx, (slot_name, plane, fluid, fatsat) in enumerate(SLOTS):
        slot_info = study_slots.get(slot_name)
        if slot_info is not None:
            jobs.append((row_idx, slot_idx, slot_name, plane, slot_info, lat))

print(f'Decoding {len(jobs)} test slot-series (parallel, {CFG["pix_threads"]} threads)...')

t_cache = time.time()
completed = 0
failed = 0
with ThreadPoolExecutor(max_workers=CFG['pix_threads']) as pool:
    for row_idx, slot_idx, result in pool.map(_read_slot_job, jobs):
        completed += 1
        if result is not None:
            TEST_CACHE[row_idx, slot_idx] = result
            TEST_MASK[row_idx, slot_idx] = 1.0
        else:
            failed += 1
        if completed % 2000 == 0:
            elapsed = time.time() - t_cache
            eta = (elapsed / completed) * (len(jobs) - completed) / 60
            print(f'  [{completed}/{len(jobs)}] {completed/len(jobs)*100:.0f}% | '
                  f'{elapsed:.0f}s | ~{eta:.0f}min left', flush=True)

cache_time = time.time() - t_cache
total_series = int(TEST_MASK.sum())
print(f'\nTest cache built: {n_test} studies, {total_series} series, '
      f'{TEST_CACHE.nbytes/1024**3:.1f} GB in {cache_time:.0f}s')
print(f'  Avg slots/study: {total_series/max(n_test,1):.1f} | Failed: {failed}')

gc.collect()
