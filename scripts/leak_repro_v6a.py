# ============================================================
# v6a 泄漏复现实验 (本地 CPU, 免费) — 用真实 cell 代码 + 假数据
#
# 背景: Kaggle 三跑 RAM 观测 → 每个训练 batch 泄漏 ~1.1MB 宿主内存
#       (epoch 末 11.1→28.3GB, 21ep 线性; 训练段 +0.8GB/ep, val+save ≈0)。
#       泄漏与 num_workers(2/0)、pin_memory(T/F)、cache 大小(9/7) 无关 → 在计算路径。
#
# 本脚本: exec 真实 cells_v6a 的 02/03/06/07/08/11/12/13 到同一命名空间
#         (仅 03_config 的 CFG 被覆盖为 CPU/小数据), 用假 cache 跑
#         train_epoch/validate_epoch, 逐阶段测 RSS 增长率, 二分定位泄漏源。
#
# 用法:  (export PATH=/c/Users/eason/miniconda3/envs/d2l/Library/bin:$PATH)
#        /c/Users/eason/miniconda3/envs/d2l/python.exe scripts/leak_repro_v6a.py [stage]
#  stage: dataloader | forward | train | full   (默认 full)
# ============================================================
import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch

CELLS = Path(__file__).resolve().parents[1] / 'notebooks' / 'cells_v6a'
NS = {'__name__': 'leak_repro'}


def exec_cell(name):
    code = (CELLS / name).read_text(encoding='utf-8')
    exec(compile(code, name, 'exec'), NS)
    gc.collect()


def rss_gb():
    return psutil.Process().memory_info().rss / 1024 ** 3


def build_fake_data(n_train, n_val):
    """假数据: 随机 uint8 cache + 软标签。形状与 Kaggle 完全一致 (224px, 7 slices)。"""
    n = n_train + n_val
    NS['SLOT_CACHE'] = np.random.default_rng(0).integers(
        0, 256, size=(n, 6, 7, 224, 224), dtype=np.uint8)
    NS['SLOT_MASK'] = np.ones((n, 6), dtype=np.float32)
    NS['study_index'] = {f's{i}': i for i in range(n)}
    NS['needed_slot_map'] = None

    tr_uids = [f's{i}' for i in range(n_train)]
    vl_uids = [f's{i}' for i in range(n_train, n)]
    tr = pd.DataFrame(index=tr_uids)
    for c in NS['TARGET_COLUMNS']:
        tr[c] = 0.0
    rng = np.random.default_rng(42)
    for c in NS['PROB_COLS']:
        tr[c] = rng.random(n_train)
    for c in NS['WEIGHT_COLS']:
        tr[c] = rng.random(n_train) * 0.6 + 0.35
    for c in NS['MASK_COLS']:
        tr[c] = 1.0
    tr['is_gold'] = False
    vl = pd.DataFrame(index=vl_uids)
    for c in NS['TARGET_COLUMNS']:
        vl[c] = rng.integers(0, 2, n_val)
        vl[f'mask_{c}'] = 1.0
    vl['is_gold'] = True
    NS['train_labels'] = tr
    NS['val_labels'] = vl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', nargs='?', default='full',
                    choices=['dataloader', 'forward', 'train', 'full'])
    args = ap.parse_args()

    # ---- 1. exec 真实 cells ----
    exec_cell('02_imports.py')
    exec_cell('03_config.py')
    CFG = NS['CFG']
    # CPU/小数据覆盖 (其余配置不动, 保持与 Kaggle 一致)
    CFG.update({
        'output_dir': str(Path(__file__).resolve().parents[1] / 'tmp_leak_repro'),
        'mixed_precision': False,      # CPU 无 AMP
        'num_workers': 0,
        'batch_size': 6,
        'epochs': 30,
        'max_train_minutes': 999,
    })
    exec_cell('06_model.py')           # SlotHead/RadResNetModel/tta_jitter/stack_views/diagnostic_pool
    exec_cell('07_loss.py')
    exec_cell('08_dataset.py')

    n_train, n_val = 60, 12
    build_fake_data(n_train, n_val)

    exec_cell('11_dataloaders.py')     # train_loader/val_loader
    exec_cell('12_train_val.py')       # train_epoch/validate_epoch/_ram_gb/watchdog
    exec_cell('13_build_model.py')     # model/optimizer/ema (CPU: 无 DataParallel, 随机初始化)

    model, optimizer, ema, criterion = NS['model'], NS['optimizer'], NS['ema'], NS['criterion']
    train_epoch, validate_epoch = NS['train_epoch'], NS['validate_epoch']
    train_loader, val_loader = NS['train_loader'], NS['val_loader']

    # 预热 (排除一次性初始化: MKLDNN 内核/线程池等)
    print(f'== stage={args.stage} | warmup 1 epoch (RSS {rss_gb():.2f}GB) ==')
    if args.stage in ('forward', 'train', 'full'):
        train_epoch(model, train_loader, optimizer, criterion, None, 0, ema=ema)
    else:
        for batch in train_loader:
            pass
    gc.collect()
    base = rss_gb()
    print(f'baseline RSS={base:.2f}GB')

    # ---- 2. 逐阶段测泄漏 ----
    n_ep = 6
    for ep in range(n_ep):
        t0 = rss_gb()
        if args.stage == 'dataloader':
            for batch in train_loader:
                pass
        elif args.stage == 'forward':
            for batch in train_loader:
                slots = batch['slots']
                mask = batch['mask']
                with torch.no_grad():
                    model(slots, mask)
        elif args.stage == 'train':
            train_epoch(model, train_loader, optimizer, criterion, None, ep, ema=ema)
        else:  # full: train + val + save
            train_epoch(model, train_loader, optimizer, criterion, None, ep, ema=ema)
            ema.apply_shadow()
            validate_epoch(model, val_loader, NS['nn'].BCEWithLogitsLoss())
            ema.restore()
        gc.collect()
        t1 = rss_gb()
        print(f'epoch {ep + 1}: RSS {t0:.3f} → {t1:.3f} GB (Δ{t1 - t0:+.3f})')

    print(f'\nTOTAL: {rss_gb():.3f}GB - base {base:.3f}GB = +{rss_gb() - base:.3f}GB '
          f'over {n_ep} epochs ({(rss_gb() - base) / n_ep:+.3f} GB/epoch)')


if __name__ == '__main__':
    main()
