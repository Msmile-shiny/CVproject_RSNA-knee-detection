# ============================================================
# v3: Build multi-slot RAM cache [N, 6, 9, 224, 224] uint8
# ============================================================

dicom_root = Path(CFG['comp_input']) / CFG['dicom_subdir']
print(f'DICOM root: {dicom_root}')

# ---- Build slot mapping ----
# This calls the function defined in cell 04 (slot_matching.py)
slot_map, study_series_map = build_study_slot_map(series_meta, dicom_root)

# ---- Collect all studies needed for caching ----
# Training studies + validation studies
all_needed_uids = set(train_labels.index) | set(val_labels.index)
print(f'Studies to cache: {len(all_needed_uids)}')

# Build slot map for needed studies
# Slot map gives us which series to read for each slot
needed_slot_map = {}
for uid in all_needed_uids:
    if uid in slot_map:
        needed_slot_map[uid] = slot_map[uid]

# ---- Pre-allocate cache ----
n_cache_studies = len(needed_slot_map)
cache_shape = (n_cache_studies, N_SLOT, CFG['cache_slices'], CFG['image_size'], CFG['image_size'])
SLOT_CACHE = np.zeros(cache_shape, dtype=np.uint8)
SLOT_MASK = np.zeros((n_cache_studies, N_SLOT), dtype=np.float32)
study_index = {}  # {study_uid: row_index}

print(f'Cache: {cache_shape} = {SLOT_CACHE.nbytes / 1024**3:.2f} GB uint8')

# ---- Fill cache ----
t_cache = time.time()
completed = 0
failed = 0

for row_idx, study_uid in enumerate(sorted(needed_slot_map)):
    study_index[study_uid] = row_idx
    study_slots = needed_slot_map[study_uid]

    for slot_idx, (slot_name, plane, fluid, fatsat) in enumerate(SLOTS):
        slot_info = study_slots.get(slot_name)
        if slot_info is None:
            continue  # slot stays zero (mask already 0)

        series_dir = Path(slot_info['dir']) if 'dir' in slot_info else None
        if series_dir is None or not series_dir.exists():
            continue

        try:
            volume, px = read_series_volume(
                str(series_dir), plane=plane, image_size=CFG['image_size'])
            if volume is None or volume.shape[0] < 3:
                continue

            # Sample 9 slices from central 60%
            sampled = sample_cache_slices(
                volume, n_cache=CFG['cache_slices'], center_pct=CFG['center_pct'])

            # Laterality normalization
            laterality = None  # simplified: skip laterality for now
            if laterality and plane:
                for s in range(sampled.shape[0]):
                    sampled[s] = normalise_laterality(sampled[s], plane, laterality)

            # Convert to uint8 [0, 255]
            sampled_uint8 = (sampled * 255).clip(0, 255).round().astype(np.uint8)
            SLOT_CACHE[row_idx, slot_idx] = sampled_uint8
            SLOT_MASK[row_idx, slot_idx] = 1.0

        except Exception:
            failed += 1
            continue

    completed += 1
    if completed % 200 == 0:
        elapsed = time.time() - t_cache
        eta = (elapsed / completed) * (n_cache_studies - completed) / 60
        print(f'  [{completed:4d}/{n_cache_studies}] {elapsed:.0f}s | ~{eta:.0f}min remaining')

cache_time = time.time() - t_cache
total_series = int(SLOT_MASK.sum())
print(f'\nCache built: {n_cache_studies} studies, {total_series} series, '
      f'{SLOT_CACHE.nbytes / 1024**3:.1f} GB in {cache_time:.0f}s')
print(f'  Avg slots/study: {total_series/n_cache_studies:.1f}')
print(f'  Failed reads: {failed}')
gc.collect()
