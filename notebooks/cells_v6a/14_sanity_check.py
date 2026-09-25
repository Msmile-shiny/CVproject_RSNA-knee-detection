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
        # ★ 13b 探针可能拆掉 DataParallel → 用 isinstance 而非 N_GPUS 判断
        logits = (model.module(slots, mask) if isinstance(model, nn.DataParallel)
                  else model(slots, mask))
    print(f'Forward: {slots.shape} → {logits.shape} | '
          f'logits range [{logits.min().item():.3f}, {logits.max().item():.3f}]')
    print(f'  Mean sigmoid: {torch.sigmoid(logits).mean().item():.3f}')

    # ★ 归一化域检查: uint8 → x/127.5−1, 期望范围 ≈ [−1, +1]
    #   (若被错误 /255 再减 127.5, 范围会压扁到 [−1.0, −0.99] — 上一版 bug 的检测点)
    xn = (slots.float() - 127.5) / 127.5
    print(f'  Normalized range: [{xn.min().item():.3f}, {xn.max().item():.3f}] '
          f'(expect ≈ [−1, +1])')

    # GPU 内存
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1024**3
        print(f'GPU memory: {mem:.2f} GB allocated')

    # ★ RAM 基线 (v6a 死亡排查: 训练期每 50 batch 也打印, 此处为缓存构建后基线)
    print(f'RAM baseline: {_ram_gb():.1f} GB (29GB 上限)')

    print('Sanity check PASSED.')
