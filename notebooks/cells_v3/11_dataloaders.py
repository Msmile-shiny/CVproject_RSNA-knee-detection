# ============================================================
# v3: DataLoaders — train + val
# ============================================================

# Training: all studies return soft-label format (gold → prob=hard, weight=1, mask=1)
train_ds = MultiViewDataset(
    study_uids=train_labels.index.values,
    slot_map=slot_map,
    cache=SLOT_CACHE,
    mask_array=SLOT_MASK,
    labels_df=train_labels,
    study_index=study_index,
    is_train=True,
)

# Validation: gold studies return hard labels
val_ds = MultiViewDataset(
    study_uids=val_labels.index.values,
    slot_map=slot_map,
    cache=SLOT_CACHE,
    mask_array=SLOT_MASK,
    labels_df=val_labels,
    study_index=study_index,
    is_train=False,
)

# DataLoader config
loader_kw = dict(
    num_workers=CFG['num_workers'], pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True if CFG['num_workers'] > 0 else False,
)
train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True, **loader_kw)
val_loader = DataLoader(val_ds, batch_size=CFG['batch_size'], shuffle=False, **loader_kw)

if IS_MAIN:
    print(f'Train batches: {len(train_loader):,}  (batch={CFG["batch_size"]}, '
          f'grad_accum={CFG["grad_accum_steps"]})')
    print(f'Val batches:   {len(val_loader):,}')
    eff_batch = CFG['batch_size'] * max(N_GPUS, 1) * CFG['grad_accum_steps']
    print(f'Effective batch: {eff_batch}')
