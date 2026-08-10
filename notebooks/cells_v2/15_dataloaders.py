loader_kw = dict(num_workers=CFG['num_workers'], pin_memory=True, prefetch_factor=2,
                 persistent_workers=True if CFG['num_workers'] > 0 else False)
train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True, **loader_kw)
val_loader = DataLoader(val_ds, batch_size=CFG['batch_size'], shuffle=False, **loader_kw)

if IS_MAIN:
    print(f'Train batches: {len(train_loader):,}  (batch_size={CFG["batch_size"]})')
    print(f'Val batches:   {len(val_loader):,}')
