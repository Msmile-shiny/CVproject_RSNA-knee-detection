# ============================================================
# v4: Sanity Check — 管线完整性检查
# ============================================================

if IS_MAIN:
    print('=' * 50)
    print('SANITY CHECK')
    print('=' * 50)

    # 缓存形状
    print(f'Cache: {SLOT_CACHE.shape} | {SLOT_CACHE.dtype} | {SLOT_CACHE.nbytes/1024**3:.2f} GB')
    print(f'Mask:  {SLOT_MASK.shape} | slots/study: {SLOT_MASK.sum(axis=1).mean():.1f}')

    # 训练/验证集
    print(f'Train: {len(train_ds):,} studies | Val: {len(val_ds):,} studies')

    # 前向传播测试
    batch = next(iter(train_loader))
    slots = batch['slots'].to(DEVICE)
    mask = batch['mask'].to(DEVICE)
    with torch.no_grad():
        logits = model.module(slots, mask) if N_GPUS > 1 else model(slots, mask)
    print(f'Forward: {slots.shape} → {logits.shape} | '
          f'logits range [{logits.min().item():.3f}, {logits.max().item():.3f}]')
    print(f'  Mean sigmoid: {torch.sigmoid(logits).mean().item():.3f}')

    # GPU 内存
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1024**3
        print(f'GPU memory: {mem:.2f} GB allocated')

    print('Sanity check PASSED.')
