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

# ---- 随机种子: DataLoader shuffle 顺序与 worker 的确定性 ----
#   每 seed 一个会话 → 不同训练顺序 (自集成成员独立性的主要来源之一)
_gen = torch.Generator()
_gen.manual_seed(CFG['seed'])

def seed_worker(worker_id):
    worker_seed = CFG['seed'] + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

train_loader = DataLoader(
    train_ds, batch_size=CFG['batch_size'], shuffle=True,
    num_workers=CFG['num_workers'], pin_memory=True, drop_last=True,
    generator=_gen, worker_init_fn=seed_worker,
)

val_loader = DataLoader(
    val_ds, batch_size=CFG['batch_size'], shuffle=False,
    num_workers=CFG['num_workers'], pin_memory=True,
)

if IS_MAIN:
    print(f'Train: {len(train_ds)} studies → {len(train_loader)} batches × {CFG["batch_size"]}')
    print(f'Val:   {len(val_ds)} studies → {len(val_loader)} batches × {CFG["batch_size"]}')
