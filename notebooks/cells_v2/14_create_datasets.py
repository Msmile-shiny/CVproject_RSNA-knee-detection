print(f'\n--- Creating datasets (v2: soft labels) ---')

train_ds = Knee25DSoftLabelDataset(series_meta, train_labels, dicom_root,
                                    image_size=CFG['image_size'], slice_count=CFG['slice_count'],
                                    is_train=True, center_stride=CFG['center_stride'],
                                    volume_cache=VOLUME_CACHE, use_soft_labels=True)
val_ds = Knee25DSoftLabelDataset(series_meta, val_labels, dicom_root,
                                  image_size=CFG['image_size'], slice_count=CFG['slice_count'],
                                  is_train=False, center_stride=1,
                                  volume_cache=VOLUME_CACHE, use_soft_labels=False)

if IS_MAIN:
    print(f'Train samples: {len(train_ds):,}  (stride={CFG["center_stride"]})')
    print(f'Val samples:   {len(val_ds):,}  (stride=1)')
    if len(train_ds) == 0:
        print('  WARNING: TRAIN DATASET IS EMPTY!')
    if len(val_ds) == 0:
        print('  WARNING: VAL DATASET IS EMPTY!')

    # Compare with v1: how many more studies do we use?
    # v1 used HIGH-only filtering; v2 keeps all
    train_studies = set(s['study_uid'] for s in train_ds.samples)
    val_studies = set(s['study_uid'] for s in val_ds.samples)
    print(f'  Unique train studies: {len(train_studies):,}')
    print(f'  Unique val studies:   {len(val_studies):,}')
