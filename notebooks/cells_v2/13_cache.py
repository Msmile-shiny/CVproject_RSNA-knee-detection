dicom_root = Path(CFG['comp_input']) / CFG['dicom_subdir']
print(f'DICOM root: {dicom_root}')
print(f'DICOM root exists: {dicom_root.exists()}')

def _collect_series_dirs(series_df, labels_df):
    df = series_df.copy()
    df = df[df['Anatomical_Plane'] == 'Sagittal']
    if 'Fluid_Sensitive' in df.columns: df = df[df['Fluid_Sensitive'] == 1]
    if 'Fat_Suppression' in df.columns: df = df[df['Fat_Suppression'] == 1]
    dirs = set()
    for (study_uid, series_uid), grp in df.groupby(['StudyInstanceUID', 'SeriesInstanceUID']):
        if study_uid not in labels_df.index:
            continue
        d = dicom_root / study_uid / series_uid
        if d.exists():
            dirs.add(str(d))
    return dirs

train_dirs = _collect_series_dirs(series_meta, train_labels)
val_dirs = _collect_series_dirs(series_meta, val_labels)
all_dirs = train_dirs | val_dirs

print(f'\nUnique series to cache: {len(all_dirs)}  (train: {len(train_dirs)}, val: {len(val_dirs)})')

VOLUME_CACHE = {}
t_cache = time.time()
failed_series = []
first_success = False

for i, d in enumerate(sorted(all_dirs)):
    try:
        vol = read_dicom_series(d, plane='Sagittal', image_size=CFG['image_size'])
        VOLUME_CACHE[d] = (vol * 255).clip(0, 255).astype(np.uint8)

        if not first_success:
            first_success = True
            elapsed = time.time() - t_cache
            print(f'  OK First series loaded ({vol.shape[0]} slices, {vol.shape[1]}x{vol.shape[2]}) '
                  f'in {elapsed:.0f}s')
    except Exception as e:
        failed_series.append((d, str(e)))
        continue

    if (i + 1) % 100 == 0:
        elapsed = time.time() - t_cache
        ram_gb = sum(v.nbytes for v in VOLUME_CACHE.values()) / 1024**3
        eta_min = (elapsed / (i + 1 - len(failed_series))) * (len(all_dirs) - (i + 1)) / 60
        print(f'  [{i+1:4d}/{len(all_dirs)}] {ram_gb:.1f} GB | {elapsed:.0f}s | ~{eta_min:.0f}min remaining')

cache_time = time.time() - t_cache
ram_gb = sum(v.nbytes for v in VOLUME_CACHE.values()) / 1024**3
print(f'\nVolume cache: {len(VOLUME_CACHE)} series, {ram_gb:.1f} GB in {cache_time:.0f}s')
if failed_series:
    print(f'  {len(failed_series)}/{len(all_dirs)} series failed ({(len(failed_series)/len(all_dirs)*100):.1f}%)')
gc.collect()
