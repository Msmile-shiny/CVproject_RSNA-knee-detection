# ============================================================
# 构建 gold 研究缓存 — 两套分辨率共享 slot 匹配 + 侧性检测
#   v5:  288px / 9 片 (~2 分钟, ~250MB)
#   rad: 224px / 7 片 (~1.5 分钟, ~120MB)
# 训练期整个 train 缓存要 ~1h, 这里只解 58 个 gold → 很快
# ============================================================

dicom_root = Path(CFG_V5['comp_input']) / CFG_V5['dicom_subdir']
print(f'DICOM root: {dicom_root}')

# ---- slot 匹配 (仅 gold 研究) ----
slot_map, _ = build_study_slot_map(gold_series_meta, dicom_root)

needed_slot_map = {uid: slot_map[uid] for uid in gold_studies if uid in slot_map}
print(f'Gold studies with slot map: {len(needed_slot_map)}/{len(gold_studies)}')

# ---- 快速侧性检测 (训练期同款, 只读每个 study 第一个有效 series 的 header) ----
def _detect_laterality_fast(needed_slot_map):
    laterality_map = {}
    for study_uid, study_slots in needed_slot_map.items():
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
laterality_map = _detect_laterality_fast(needed_slot_map)
n_lat = sum(1 for v in laterality_map.values() if v is not None)
n_right = sum(1 for v in laterality_map.values() if v == 'R')
if IS_MAIN:
    print(f'Laterality detected: {n_lat}/{len(laterality_map)} studies '
          f'({n_lat/max(len(laterality_map),1)*100:.1f}%), '
          f'R={n_right}, L={n_lat-n_right}, ({time.time()-t_lat:.1f}s)')

gold_study_index = {uid: i for i, uid in enumerate(gold_studies)}

# ---- 并行 DICOM 读取 (参数化: 两套分辨率同款逻辑) ----
def _read_slot_job(args):
    """单个 slot 的读取任务（用于 ThreadPoolExecutor）"""
    (row_idx, slot_idx, slot_name, plane, slot_info, laterality,
     image_size, cache_slices, crop_mm, center_pct) = args
    if slot_info is None:
        return row_idx, slot_idx, None

    series_dir = Path(slot_info['dir']) if 'dir' in slot_info else None
    if series_dir is None or not series_dir.exists():
        return row_idx, slot_idx, None

    try:
        volume, px = read_series_volume(
            str(series_dir), plane=plane, laterality=laterality,
            image_size=image_size, crop_mm=crop_mm)
        if volume is None or volume.shape[0] < 3:
            return row_idx, slot_idx, None

        sampled = sample_cache_slices(
            volume, n_cache=cache_slices, center_pct=center_pct)
        sampled_uint8 = (sampled * 255).clip(0, 255).round().astype(np.uint8)
        return row_idx, slot_idx, sampled_uint8
    except Exception:
        return row_idx, slot_idx, None


def _build_gold_cache(image_size, cache_slices, crop_mm, tag):
    """为全部 gold 研究构建缓存 [n_gold, 6, cache_slices, H, W] uint8。"""
    cache_shape = (len(gold_studies), N_SLOT, cache_slices, image_size, image_size)
    cache = np.zeros(cache_shape, dtype=np.uint8)
    mask = np.zeros((len(gold_studies), N_SLOT), dtype=np.float32)

    jobs = []
    for row_idx, study_uid in enumerate(gold_studies):
        study_slots = needed_slot_map.get(study_uid, {})
        lat = laterality_map.get(study_uid)
        for slot_idx, (slot_name, plane, fluid, fatsat) in enumerate(SLOTS):
            slot_info = study_slots.get(slot_name)
            if slot_info is not None:
                jobs.append((row_idx, slot_idx, slot_name, plane, slot_info, lat,
                             image_size, cache_slices, crop_mm, (0.2, 0.8)))

    t0 = time.time()
    print(f'Decoding {len(jobs)} gold slot-series @ {image_size}px/{cache_slices} '
          f'slices ({CFG_V5["pix_threads"]} threads)...')
    completed, failed = 0, 0
    with ThreadPoolExecutor(max_workers=CFG_V5['pix_threads']) as pool:
        for row_idx, slot_idx, result in pool.map(_read_slot_job, jobs):
            completed += 1
            if result is not None:
                cache[row_idx, slot_idx] = result
                mask[row_idx, slot_idx] = 1.0
            else:
                failed += 1
            if completed % 100 == 0:
                print(f'  [{completed}/{len(jobs)}] {time.time()-t0:.0f}s', flush=True)

    print(f'{tag} gold cache: {int(mask.sum())} series, '
          f'{cache.nbytes / 1024**2:.1f} MB in {time.time()-t0:.0f}s ({failed} failed)')
    return cache, mask

GOLD_CACHE_V5, GOLD_MASK_V5 = _build_gold_cache(
    CFG_V5['image_size'], CFG_V5['cache_slices'], CFG_V5['crop_mm'], 'v5-288')
GOLD_CACHE_RAD, GOLD_MASK_RAD = _build_gold_cache(
    CFG_RAD['image_size'], CFG_RAD['cache_slices'], CFG_RAD['crop_mm'], 'rad-224')

print(f'\n★ Physical crop: {CFG_V5["crop_mm"]}mm | Laterality: '
      f'{n_right}R/{n_lat-n_right}L | Threads: {CFG_V5["pix_threads"]}')
gc.collect()
