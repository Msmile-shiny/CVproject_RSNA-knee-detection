# ============================================================
# v4: Build RAM Cache — 并行 DICOM 读取 + 空间排序 + 物理裁剪 + 侧性归一化
# ============================================================

dicom_root = Path(CFG['comp_input']) / CFG['dicom_subdir']
print(f'DICOM root: {dicom_root}')

# ---- Build slot mapping ----
slot_map, study_series_map = build_study_slot_map(series_meta, dicom_root)

# ---- ★ 侧性检测 ----
# 快速扫描：每个 study 只读一个 DICOM header 来获取 Laterality
all_needed_uids = set(train_labels.index) | set(val_labels.index)
print(f'Studies to cache: {len(all_needed_uids)}')

needed_slot_map = {uid: slot_map[uid] for uid in all_needed_uids if uid in slot_map}

# ★ 快速侧性检测：每个 study 扫描一个 DICOM header
def _detect_laterality_fast(needed_slot_map):
    """为每个 study 快速检测侧性（只读每个 study 第一个有效 series 的 header）。"""
    laterality_map = {}
    for study_uid, study_slots in needed_slot_map.items():
        lat = None
        for slot_name, slot_info in study_slots.items():
            if slot_info is None:
                continue
            series_dir = Path(slot_info['dir']) if 'dir' in slot_info else None
            if series_dir is None or not series_dir.exists():
                continue
            dcm_files = sorted([f for f in series_dir.iterdir() if f.name.endswith('.dcm')])
            if not dcm_files:
                continue
            try:
                ds = pydicom.dcmread(
                    str(dcm_files[0]), stop_before_pixels=True, force=True,
                    specific_tags=['Laterality', 'ImageLaterality', 'ImagePositionPatient'])
                # 优先 DICOM Laterality 标签
                for tag_name in ['Laterality', 'ImageLaterality']:
                    val = getattr(ds, tag_name, None)
                    if val is not None:
                        val = str(val).strip().upper()
                        if val and val[0] in ('L', 'R'):
                            lat = val[0]
                            break
                if lat is not None:
                    break
                # Fallback: ImagePositionPatient 几何推断 (LPS: +x = 患者左侧)
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
laterality_map = _detect_laterality_fast(needed_slot_map)
n_lat = sum(1 for v in laterality_map.values() if v is not None)
n_right = sum(1 for v in laterality_map.values() if v == 'R')
if IS_MAIN:
    print(f'Laterality detected: {n_lat}/{len(laterality_map)} studies '
          f'({n_lat/max(len(laterality_map),1)*100:.1f}%), '
          f'R={n_right}, L={n_lat-n_right}, '
          f'({time.time()-t_lat:.1f}s)')

# ---- Pre-allocate cache ----
n_cache_studies = len(needed_slot_map)
cache_shape = (n_cache_studies, N_SLOT, CFG['cache_slices'], CFG['image_size'], CFG['image_size'])

SLOT_CACHE = np.zeros(cache_shape, dtype=np.uint8)
SLOT_MASK = np.zeros((n_cache_studies, N_SLOT), dtype=np.float32)
study_index = {}

print(f'Cache: {cache_shape} = {SLOT_CACHE.nbytes / 1024**3:.2f} GB uint8')

# ---- ★ 并行 DICOM 读取 ----
def _read_slot_job(args):
    """单个 slot 的读取任务（用于 ThreadPoolExecutor）"""
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


# ---- Fill cache ----
t_cache = time.time()
sorted_uids = sorted(needed_slot_map)

for row_idx, study_uid in enumerate(sorted_uids):
    study_index[study_uid] = row_idx

# 收集所有读取任务
jobs = []
for row_idx, study_uid in enumerate(sorted_uids):
    study_slots = needed_slot_map[study_uid]
    lat = laterality_map.get(study_uid)  # ★ 侧性归一化
    for slot_idx, (slot_name, plane, fluid, fatsat) in enumerate(SLOTS):
        slot_info = study_slots.get(slot_name)
        if slot_info is not None:
            jobs.append((row_idx, slot_idx, slot_name, plane, slot_info, lat))

print(f'Decoding {len(jobs)} slot-series (parallel, {CFG["pix_threads"]} threads)...')

completed = 0
failed = 0
with ThreadPoolExecutor(max_workers=CFG['pix_threads']) as pool:
    for row_idx, slot_idx, result in pool.map(_read_slot_job, jobs):
        completed += 1
        if result is not None:
            SLOT_CACHE[row_idx, slot_idx] = result
            SLOT_MASK[row_idx, slot_idx] = 1.0
        else:
            failed += 1

        if completed % 2000 == 0:
            elapsed = time.time() - t_cache
            pct = completed / len(jobs) * 100
            eta = (elapsed / completed) * (len(jobs) - completed) / 60
            print(f'  [{completed}/{len(jobs)}] {pct:.0f}% | {elapsed:.0f}s | ~{eta:.0f}min left', flush=True)

cache_time = time.time() - t_cache
total_series = int(SLOT_MASK.sum())
print(f'\nCache built: {n_cache_studies} studies, {total_series} series, '
      f'{SLOT_CACHE.nbytes / 1024**3:.1f} GB in {cache_time:.0f}s')
print(f'  Avg slots/study: {total_series/max(n_cache_studies,1):.1f}')
print(f'  Failed reads: {failed}')
print(f'  ★ Physical crop: {CFG["crop_mm"]}mm | Laterality: {n_right}R/{n_lat-n_right}L | Threads: {CFG["pix_threads"]}')

# ★ 伪标签自蒸馏（如果启用）
if use_distill and IS_MAIN:
    print('\n★ Self-distillation: running v3 inference on unlabeled studies...')
    print('  (Skipped — v3 checkpoint not available or self-distill disabled)')
    # 完整实现需要：加载 v3 model → 对每个 unlabeled study 推理 → 更新 pseudo_labels

gc.collect()
