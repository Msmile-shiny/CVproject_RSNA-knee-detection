# ============================================================
# v4: DataLoaders
# ============================================================

train_ds = MultiViewDataset(
    study_uids=list(train_labels.index),
    slot_map=needed_slot_map,
    cache=SLOT_CACHE, mask_array=SLOT_MASK,
    labels_df=train_labels, study_index=study_index,
    is_train=True,
)

val_ds = MultiViewDataset(
    study_uids=list(val_labels.index),
    slot_map=needed_slot_map,
    cache=SLOT_CACHE, mask_array=SLOT_MASK,
    labels_df=val_labels, study_index=study_index,
    is_train=False,
)

train_loader = DataLoader(
    train_ds, batch_size=CFG['batch_size'], shuffle=True,
    num_workers=CFG['num_workers'], pin_memory=True, drop_last=True,
)

val_loader = DataLoader(
    val_ds, batch_size=CFG['batch_size'], shuffle=False,
    num_workers=CFG['num_workers'], pin_memory=True,
)

if IS_MAIN:
    print(f'Train: {len(train_ds)} studies → {len(train_loader)} batches × {CFG["batch_size"]}')
    print(f'Val:   {len(val_ds)} studies → {len(val_loader)} batches × {CFG["batch_size"]}')
