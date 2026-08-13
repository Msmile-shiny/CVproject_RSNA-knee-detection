# -*- coding: utf-8 -*-
"""v5 本地冒烟测试 — 在 RTX GPU 上执行 cells_v5 的训练管线 (跳过 DICOM, 合成缓存).

验证:
  1. 09: v5 融合标签加载 (真实 v5_labels.csv, 4349 train rows)
  2. 13: 288px 模型构建 + pos_embed 518(37²)→288(20²) 插值路径
  3. WeightedSoftBCE 前向/反向 + train_epoch 一步训练
  4. validate_epoch (7 窗口 TTA + 诊断池化) 输出形状
"""
from __future__ import annotations

import io
import math
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / 'notebooks' / 'cells_v5'

# ---- 执行 notebook cell (与 build 脚本相同顺序, 跳过 DICOM 相关) ----
def exec_cell(ns, fname):
    src = (CELLS / fname).read_text(encoding='utf-8')
    compile(src, fname, 'exec')
    buf = io.StringIO()
    with redirect_stdout(buf):
        exec(src, ns)
    return buf.getvalue()


def main() -> int:
    ns = {'__name__': '__main__'}
    print('=== cell 01: setup ===')
    print(exec_cell(ns, '01_setup.py'))
    print('=== cell 02: imports ===')
    print(exec_cell(ns, '02_imports.py'))
    print('=== cell 03: config ===')
    print(exec_cell(ns, '03_config.py'))

    # ---- patch CFG for local smoke ----
    CFG = ns['CFG']
    CFG.update({
        'comp_input': str(ROOT / 'data' / 'metadata'),   # 本地 train.csv/train_series.csv
        'label_input': str(ROOT / 'data' / 'processed'), # 本地 v5_labels.csv
        'output_dir': tempfile.mkdtemp(prefix='v5_smoke_'),
        'batch_size': 2,
        'num_workers': 0,
        'mixed_precision': False,
        'epochs': 1,
        'max_train_minutes': 5,
        'dinov2_weights': str(ROOT / 'data' / 'processed' / '_smoke_dinov2_fake.pt'),
    })

    # ---- 伪造 518-native 权重 (pos_embed [1,1370,384]) 以触发插值路径 ----
    fake_path = Path(CFG['dinov2_weights'])
    if not fake_path.exists():
        import timm as _timm
        base = _timm.create_model(CFG['dinov2_variant'], pretrained=False,
                                  num_classes=0, img_size=CFG['image_size'])
        sd = {k: v.clone() for k, v in base.state_dict().items()}
        assert sd['pos_embed'].shape == (1, 401, 384), sd['pos_embed'].shape
        sd['pos_embed'] = torch.randn(1, 1370, 384)   # 518 原生: 37²+1 tokens
        torch.save(sd, fake_path)
        print(f'fake 518-native weights written: {fake_path}')

    print('=== cell 06: model ===')
    print(exec_cell(ns, '06_model.py'))
    print('=== cell 07: loss ===')
    print(exec_cell(ns, '07_loss.py'))
    print('=== cell 09: load data (real v5 labels) ===')
    print(exec_cell(ns, '09_load_data.py'))

    # ---- 断言 1: 标签 ----
    train_labels, val_labels = ns['train_labels'], ns['val_labels']
    assert len(train_labels) == 4349, f'train rows {len(train_labels)} != 4349'
    probs = train_labels[[f'prob_{c}' for c in ns['TARGET_COLUMNS']]].values
    weights = train_labels[[f'weight_{c}' for c in ns['TARGET_COLUMNS']]].values
    # float32 舍入误差: 允许 ±0.002
    assert probs.min() >= 0.019 and probs.max() <= 0.981, (probs.min(), probs.max())
    assert weights.min() >= 0.249 and weights.max() <= 1.001, (weights.min(), weights.max())
    assert len(val_labels) == 58, f'val rows {len(val_labels)} != 58'
    print(f'[PASS] labels: {len(train_labels)} train (prob [{probs.min():.2f},{probs.max():.2f}], '
          f'weight [{weights.min():.2f},{weights.max():.2f}]) | {len(val_labels)} gold val')

    # ---- 跳过 04/05/10 (DICOM), 注入合成缓存 ----
    n_cache = 4
    ns['needed_slot_map'] = {}
    ns['SLOT_CACHE'] = np.random.randint(0, 255,
        (n_cache, ns['N_SLOT'], CFG['cache_slices'], CFG['image_size'], CFG['image_size']),
        dtype=np.uint8)
    ns['SLOT_MASK'] = np.ones((n_cache, ns['N_SLOT']), dtype=np.float32)
    uids = list(train_labels.index[:3]) + list(val_labels.index[:1])
    ns['study_index'] = {u: i for i, u in enumerate(uids)}
    print(f'synthetic cache: {ns["SLOT_CACHE"].shape} for uids {uids}')

    print('=== cell 08: dataset ===')
    print(exec_cell(ns, '08_dataset.py'))
    print('=== cell 11: dataloaders ===')
    print(exec_cell(ns, '11_dataloaders.py'))
    print('=== cell 12: train/val ===')
    print(exec_cell(ns, '12_train_val.py'))
    print('=== cell 13: build model (pos_embed 37²→20²) ===')
    out13 = exec_cell(ns, '13_build_model.py')
    print(out13)

    # ---- 断言 2: pos_embed 插值 ----
    assert 'pos_embed interpolated: [37×37] → [20×20]' in out13, 'interpolation log missing'
    model = ns['model']
    assert model.dinov2.pos_embed.shape == (1, 401, 384), model.dinov2.pos_embed.shape
    print('[PASS] pos_embed interpolated 518(37×37)→288(20×20); model @288 OK')

    print('=== cell 14: sanity (GPU forward) ===')
    print(exec_cell(ns, '14_sanity_check.py'))

    # ---- 断言 3: 一步训练 + 损失有限 ----
    train_epoch, validate_epoch = ns['train_epoch'], ns['validate_epoch']
    model, optimizer, criterion = ns['model'], ns['optimizer'], ns['criterion']
    scaler, ema = ns['scaler'], ns['ema']
    train_loader, val_loader = ns['train_loader'], ns['val_loader']

    loss = train_epoch(model, train_loader, optimizer, criterion, scaler, 1, ema=ema)
    assert math.isfinite(loss) and loss < 0.8, f'train loss {loss}'
    print(f'[PASS] 1-step train loss={loss:.4f} (finite, < ln2~0.69 附近)')

    # ---- 断言 4: 验证 (TTA + 诊断池化) ----
    vm = validate_epoch(model, val_loader, None)
    assert vm['probs'].shape == (1, 12), vm['probs'].shape
    assert len(vm['uids']) == 1
    print(f'[PASS] validate: probs {vm["probs"].shape}, loss={vm["loss"]:.4f}')

    # ---- 断言 5: 权重保存/加载往返 (17 的加载路径) ----
    ckpt = tempfile.mkdtemp(prefix='v5_ckpt_')
    torch.save({'epoch': 1, 'model': model.state_dict(),
                'ema': ema.state_dict(), 'auc': 0.5, 'config': CFG,
                'targets': ns['TARGET_COLUMNS'], 'slots': ns['SLOTS']},
               Path(ckpt) / 'best_model.pt')
    import timm as _timm
    bb = _timm.create_model(CFG['dinov2_variant'], pretrained=False,
                            num_classes=0, img_size=CFG['image_size'])
    m2 = ns['MultiViewModel'](dinov2_model=bb, n_slots=ns['N_SLOT'],
                              cls_dim=CFG['cls_dim'], n_classes=CFG['num_classes'],
                              slot_hidden=CFG['slot_hidden'], dropout=0.0,
                              unfreeze_layers=CFG['unfreeze_layers'])
    st = torch.load(Path(ckpt) / 'best_model.pt', map_location='cpu', weights_only=False)['model']
    m2.load_state_dict(st, strict=True)
    print('[PASS] checkpoint save/load round-trip (strict)')

    print('\n=== SMOKE TEST PASSED ===')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
