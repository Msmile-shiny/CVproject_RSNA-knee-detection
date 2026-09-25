# ============================================================
# v6a: CUDA 泄漏探针 + 训练配置决策
# ============================================================
# 背景: 三跑 RAM 观测 → 训练段每 batch 泄漏 ~1.1MB 宿主内存 (每 epoch +0.85GB,
#       21ep 末 28.3GB 撞 29GB 上限)。本地 CPU 复现 0 泄漏 → 泄漏在 CUDA 侧,
#       且与 num_workers/pin_memory/cache 大小无关 (三种配置同速率)。
#       疑似: DataParallel per-forward replicate / AMP / cuDNN 之一。
#
# 本 cell: 用真实训练步做 5 个 200-batch 探针 (数据搬运 / DP+AMP / 单GPU+AMP /
#          单GPU+AMP+cudnn_off / 单GPU+fp32+cudnn_off), 测每批 RSS 增量,
#          然后按保守优先级选正式训练配置并重建 model/scaler/epochs。
# 耗时 ~3-5 分钟。CPU 环境自动跳过 (本地冒烟)。
#
# 注意: 探针在真实 optimizer/EMA 上做 ~800 步更新 (~1 epoch 训练量),
#       LR 仍为初始值, 等价于训练提前开始, 无副作用 (保留进度)。
# ============================================================

if not torch.cuda.is_available():
    print('CUDA 不可用 — 跳过 CUDA 泄漏探针 (本地 CPU 冒烟)。')
else:
    try:
        import psutil as _psutil
    except ImportError:
        _psutil = None

    def _rss_mb():
        return _psutil.Process().memory_info().rss / 1024 ** 2 if _psutil else -1.0

    def _stream_batches(loader):
        while True:
            for batch in loader:
                yield batch

    _stream = _stream_batches(train_loader)

    # ---- 探针步骤定义 (与 train_epoch 真实搬运/前向/反向路径一致) ----

    def _step_copy():
        b = next(_stream)
        s = b['slots'].to(DEVICE, non_blocking=True)
        m = b['mask'].to(DEVICE, non_blocking=True)
        p = b['prob_targets'].to(DEVICE, non_blocking=True)
        w = b['weights'].to(DEVICE, non_blocking=True)
        sm = b['soft_masks'].to(DEVICE, non_blocking=True)
        del s, m, p, w, sm

    def _step_train(m):
        b = next(_stream)
        slots = b['slots'].to(DEVICE, non_blocking=True)
        mask = b['mask'].to(DEVICE, non_blocking=True)
        pt = b['prob_targets'].to(DEVICE, non_blocking=True)
        w = b['weights'].to(DEVICE, non_blocking=True)
        sm = b['soft_masks'].to(DEVICE, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=_use_amp):
            loss = criterion(m(slots, mask), pt, w, sm)
        if _use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()
        ema.update()

    _use_amp = True

    def _run_probe(label, step_fn, n_batches=200, warmup=10):
        t0 = time.time()
        for _ in range(warmup):
            step_fn()
            _HB['t'] = time.time()   # 探针期间也喂心跳, 避免看门狗误报
        torch.cuda.synchronize()
        gc.collect()
        r0 = _rss_mb()
        for _ in range(n_batches):
            step_fn()
            _HB['t'] = time.time()
        torch.cuda.synchronize()
        gc.collect()
        r1 = _rss_mb()
        dt = time.time() - t0
        rate = (r1 - r0) / n_batches
        _PROBE[label] = rate
        print(f'[PROBE {label}] ΔRSS {(r1-r0):+.0f} MB / {n_batches} batches '
              f'= {rate:+.2f} MB/batch | {n_batches/dt:.0f} batches/s', flush=True)
        return rate

    _PROBE = {}

    print('--- CUDA 泄漏探针 (每探针 200 训练批, 参照泄漏 ~1.1 MB/batch) ---', flush=True)

    _run_probe('P0-copy', _step_copy)                    # 数据搬运路径 (排除 staging)
    _run_probe('P1-dp+amp', lambda: _step_train(model))   # 现状: DataParallel + AMP + 反向

    _m1 = model.module if isinstance(model, nn.DataParallel) else model
    _run_probe('P2-sg+amp', lambda: _step_train(_m1))     # 单 GPU + AMP

    torch.backends.cudnn.enabled = False
    _run_probe('P3-sg+amp-nocudnn', lambda: _step_train(_m1))  # 单 GPU + AMP + cuDNN off

    _use_amp = False
    _run_probe('P4-sg+fp32-nocudnn', lambda: _step_train(_m1))  # 单 GPU + fp32 + cuDNN off
    torch.backends.cudnn.enabled = True
    _use_amp = True

    # ---- 决策: 保守优先级链 (第一个平直的配置) ----
    FLAT = 0.20  # MB/batch 平直判据 (噪声 ~0.05, 泄漏 ~1.1)

    if _PROBE.get('P2-sg+amp', 99) < FLAT:
        choice = dict(dp=False, amp=True, cudnn=True, epochs=30,
                      label='单GPU+AMP+cudnn (首选: 机制最简)')
    elif _PROBE.get('P1-dp+amp', 99) < FLAT:
        choice = dict(dp=True, amp=True, cudnn=True, epochs=30,
                      label='DataParallel+AMP+cudnn (现状, v5 同款)')
    elif _PROBE.get('P3-sg+amp-nocudnn', 99) < FLAT:
        choice = dict(dp=False, amp=True, cudnn=False, epochs=30,
                      label='单GPU+AMP+cudnn_off')
    elif _PROBE.get('P4-sg+fp32-nocudnn', 99) < FLAT:
        choice = dict(dp=False, amp=False, cudnn=False, epochs=30,
                      label='单GPU+fp32+cudnn_off')
    else:
        # 全漏 → 驱动/库级泄漏: 用泄漏最慢的配置, epochs 缩到 RAM 预算内
        slow = min(_PROBE, key=_PROBE.get)
        cfg_map = {
            'P1-dp+amp':          dict(dp=True, amp=True, cudnn=True),
            'P2-sg+amp':          dict(dp=False, amp=True, cudnn=True),
            'P3-sg+amp-nocudnn':  dict(dp=False, amp=True, cudnn=False),
            'P4-sg+fp32-nocudnn': dict(dp=False, amp=False, cudnn=False),
        }
        choice = cfg_map[slow]
        budget_mb = (27.0 - _ram_gb()) * 1024
        epochs = int(budget_mb / (_PROBE[slow] * 734))   # 734 ≈ 724 train + 10 val 批
        choice['epochs'] = max(6, min(24, epochs))
        choice['label'] = (f'{slow} 配置 (全漏, 最慢) + epochs={choice["epochs"]} '
                           f'(RAM 预算)')

    print(f'[PROBE DECISION] {choice["label"]}', flush=True)

    # ---- 应用决策: 重建 model wrapper / scaler / epochs ----
    torch.backends.cudnn.enabled = choice['cudnn']
    if not choice['dp'] and isinstance(model, nn.DataParallel):
        model = model.module            # 去 DP wrapper → 单 GPU (释放 GPU1 副本)
        torch.cuda.empty_cache()
        print('  → DataParallel 已拆下, 单 GPU 训练 (GPU1 副本释放)')
    scaler = torch.amp.GradScaler('cuda') if choice['amp'] else None
    CFG['epochs'] = choice['epochs']

    if IS_MAIN:
        print(f'  → 正式训练配置: model={"DP" if choice["dp"] else "单GPU"}, '
              f'AMP={choice["amp"]}, cudnn={choice["cudnn"]}, epochs={choice["epochs"]}')
