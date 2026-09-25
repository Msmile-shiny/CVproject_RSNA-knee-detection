# -*- coding: utf-8 -*-
"""v6a 本地 CPU 冒烟测试 — 执行 cells_v6a 训练管线 (跳过 DICOM, 合成缓存).

★ 本机 GPU 冒烟永久禁止 (两次 HYPERVISOR_ERROR 蓝屏) → 强制 CPU
  (monkeypatch torch.cuda.device_count=0, DEVICE 自动落到 cpu)。

验证:
  1. cell 13: 真实 RadImageNet 权重 (radimagenet_resnet50_notop.pt) 加载 +
     backbone.* 键逆映射 + 严格加载 + 编码器全冻结断言
  2. RadResNetModel: 前向 [B,S,3,224,224] uint8 → [B,12];
     归一化 x/127.5−1 与手工计算逐位一致; train() 后 backbone 仍 eval
  3. EMA: shadow 只含 head 参数 (冻结 backbone 排除)
  4. 1 步训练 loss 有限; validate_epoch (7 窗口 TTA + jitter + 诊断池化, B=2)
  5. checkpoint 保存/加载往返 (17 的加载路径) + EMA swap 输出一致
  6. jitter/stack_views/diagnostic_pool 复用 v5 断言 (B>1 防串位)
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
CELLS = ROOT / 'notebooks' / 'cells_v6a'
RAD_PT = ROOT / 'datasets' / 'radimagenet_raw' / 'radimagenet_resnet50_notop.pt'


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

    # ★ 强制 CPU (本机 GPU 冒烟禁止 — 蓝屏史)
    torch.cuda.device_count = lambda: 0
    torch.cuda.is_available = lambda: False   # 13b 探针守卫: 永不触碰本机 GPU
    print('=== cell 03: config (CPU forced) ===')
    print(exec_cell(ns, '03_config.py'))

    CFG = ns['CFG']
    assert CFG['image_size'] == 224 and CFG['unfreeze_layers'] == 0
    assert ns['SEED_TAG'] == 'rad' and ns['CKPT_NAME'] == 'best_model_rad.pt'
    assert str(ns['DEVICE']) == 'cpu', f'DEVICE={ns["DEVICE"]} — 本机冒烟必须 CPU'
    CFG.update({
        'comp_input': str(ROOT / 'data' / 'metadata'),   # 本地 train.csv/train_series.csv
        'label_input': str(ROOT / 'data' / 'processed'), # 本地 v5_labels.csv
        'output_dir': tempfile.mkdtemp(prefix='v6a_smoke_'),
        'batch_size': 2,
        'num_workers': 0,
        'mixed_precision': False,
        'epochs': 1,
        'max_train_minutes': 5,
        'rad_weights': str(RAD_PT),                       # ★ 真实转换权重走 cell 13 路径
    })
    assert RAD_PT.exists(), f'RadImageNet .pt not found: {RAD_PT}'

    print('=== cell 06: model ===')
    print(exec_cell(ns, '06_model.py'))

    # ---- 断言 A: jitter + diagnostic_pool + stack_views (v5 同款) ----
    tta_jitter, diag_pool = ns['tta_jitter'], ns['diagnostic_pool']
    x = torch.randint(0, 255, (4, 6, 3, 224, 224), dtype=torch.uint8)
    xj = tta_jitter(x)
    assert xj.shape == x.shape and xj.dtype == x.dtype
    assert float((xj != x).float().mean()) > 0.9
    assert torch.equal(xj, tta_jitter(x)), 'tta_jitter 非确定性'
    print('[PASS] tta_jitter: 确定性增广')

    logits_v = torch.randn(2, 14, 12)
    out = diag_pool(logits_v, n_orig=7)
    j_syn = ns['TARGET_COLUMNS'].index('Synovitis')
    j_frac = ns['TARGET_COLUMNS'].index('Fracture')
    view_mean = (torch.sigmoid(logits_v[:, :7]) + torch.sigmoid(logits_v[:, 7:])) / 2
    assert torch.allclose(out[:, j_syn], torch.sigmoid(logits_v[:, :7, j_syn]).mean(1), atol=1e-5)
    assert torch.allclose(out[:, j_frac], view_mean[:, :, j_frac].max(1).values, atol=1e-5)
    print('[PASS] diagnostic_pool: 视图平均 + original_mean')

    stack_views = ns['stack_views']
    Bt, Wt, Ct = 3, 4, 2
    lg = torch.arange(2 * Bt * Wt * Ct, dtype=torch.float32).reshape(2 * Bt * Wt, Ct)
    v = stack_views(lg, Bt, Wt, Wt)
    for b in range(Bt):
        assert torch.equal(v[b, 0], lg[b * Wt]), '跨研究串位'
        assert torch.equal(v[b, Wt], lg[Bt * Wt + b * Wt])
    print('[PASS] stack_views: B-major 分组 (B>1)')

    print('=== cell 07: loss ===')
    print(exec_cell(ns, '07_loss.py'))
    print('=== cell 09: load data (real v5 labels) ===')
    print(exec_cell(ns, '09_load_data.py'))

    # ---- 断言 1: 标签 (与 v5 一致) ----
    train_labels, val_labels = ns['train_labels'], ns['val_labels']
    assert len(train_labels) == 4349 and len(val_labels) == 58
    print(f'[PASS] labels: {len(train_labels)} train | {len(val_labels)} gold val')

    # ---- 跳过 04/05/10 (DICOM), 注入合成缓存 (3 train + 2 val → val B=2) ----
    n_cache = 5
    ns['needed_slot_map'] = {}
    ns['SLOT_CACHE'] = np.random.randint(0, 255,
        (n_cache, ns['N_SLOT'], CFG['cache_slices'], 224, 224), dtype=np.uint8)
    ns['SLOT_MASK'] = np.ones((n_cache, ns['N_SLOT']), dtype=np.float32)
    uids = list(train_labels.index[:3]) + list(val_labels.index[:2])
    ns['study_index'] = {u: i for i, u in enumerate(uids)}
    print(f'synthetic cache: {ns["SLOT_CACHE"].shape} for uids {uids}')

    print('=== cell 08: dataset ===')
    print(exec_cell(ns, '08_dataset.py'))
    print('=== cell 11: dataloaders ===')
    print(exec_cell(ns, '11_dataloaders.py'))
    print('=== cell 12: train/val ===')
    print(exec_cell(ns, '12_train_val.py'))
    print('=== cell 13: build model (real RadImageNet weights) ===')
    out13 = exec_cell(ns, '13_build_model.py')
    print(out13)
    assert 'RadImageNet pretrained weights loaded' in out13, 'cell 13 未走真实权重路径'

    print('=== cell 13b: cuda probe (CPU skip) ===')
    out13b = exec_cell(ns, '13b_cuda_probe.py')
    print(out13b)
    assert '跳过' in out13b, 'cell 13b 在 CPU 冒烟下必须走跳过分支'

    model = ns['model']

    # ---- 断言 2: 冻结 + 权重真实性 ----
    backbone_grad = [n for n, p in model.named_parameters()
                     if n.startswith('backbone') and p.requires_grad]
    head_grad = [n for n, p in model.named_parameters()
                 if n.startswith('head') and p.requires_grad]
    assert len(backbone_grad) == 0, f'backbone 未冻结: {backbone_grad[:5]}'
    assert len(head_grad) > 0, 'head 无可训练参数'
    assert model.backbone.fc is not None and isinstance(model.backbone.fc, torch.nn.Identity)
    ref_sd = torch.load(RAD_PT, map_location='cpu', weights_only=True)
    assert torch.equal(model.backbone.conv1.weight.data, ref_sd['backbone.0.weight']), \
        'conv1 与转换 .pt 不一致'
    assert torch.equal(model.backbone.layer4[2].conv3.weight.data,
                       ref_sd['backbone.7.2.conv3.weight'])
    print(f'[PASS] 冻结: backbone 0 可训练 / head {len(head_grad)} 参数组; '
          f'权重 = 官方 h5 转换版 (conv1/layer4.2.conv3 逐位一致)')

    # ---- 断言 3: 归一化 x/127.5−1 + 前向形状 ----
    model.eval()   # dropout=0.2 在 train 模式随机 → 分解比对必须在 eval
    with torch.no_grad():
        ximg = torch.randint(0, 255, (1, 6, 3, 224, 224), dtype=torch.uint8)
        mask = torch.ones(1, 6)
        logits = model(ximg, mask)
        assert logits.shape == (1, 12), logits.shape
        xn = ximg.reshape(6, 3, 224, 224).float()
        xn = (xn - model.mean) / model.std   # ★ uint8 域: = x/127.5 − 1 (勿先 /255)
        feats = model.backbone(xn)
        assert feats.shape == (6, 2048), feats.shape
        manual = model.head(feats.reshape(1, 6, -1), mask)
        assert torch.allclose(logits, manual, atol=1e-6), '前向与手工分解不一致'
        # mean/std buffer = 127.5 (RadImageNet 训练归一化)
        assert model.mean.unique().item() == 127.5 and model.std.unique().item() == 127.5
        # ★ 归一化域断言: 全黑→−1, 全白→+1 (防 /255 混入压扁动态范围回归)
        lo = (torch.zeros(1, 3, 1, 1) - model.mean) / model.std
        hi = (torch.full((1, 3, 1, 1), 255.0) - model.mean) / model.std
        assert lo.unique().item() == -1.0 and hi.unique().item() == 1.0, \
            f'归一化域错误: [{lo.unique().item()}, {hi.unique().item()}] 应为 [−1, +1]'
    print(f'[PASS] 前向 [1,6,3,224,224]→[1,12]; GAP=2048d; '
          f'归一化 x/127.5−1 (uint8 域, 全黑→−1/全白→+1) 与手工计算一致')

    # ---- 断言 4: train() 后 backbone 仍 eval (BN running stats) ----
    model.train()
    assert not model.backbone.training, 'backbone 进入 train 模式 (BN 会用 batch stats)'
    assert model.head.training, 'head 未进入 train 模式'
    model.eval()
    print('[PASS] train() 只切 head; backbone 恒 eval (BN running stats)')

    # ---- 断言 5: EMA 只 shadow head ----
    ema = ns['ema']
    assert len(ema.shadow) == len(head_grad), \
        f'EMA shadow {len(ema.shadow)} != head params {len(head_grad)}'
    assert all(k.startswith('head.') for k in ema.shadow), 'shadow 含 backbone 键'
    print(f'[PASS] EMA: shadow {len(ema.shadow)} 个 head 参数 (backbone 排除)')

    print('=== cell 14: sanity (CPU forward) ===')
    print(exec_cell(ns, '14_sanity_check.py'))

    # ---- 断言 6: 一步训练 + 损失有限 ----
    train_epoch, validate_epoch = ns['train_epoch'], ns['validate_epoch']
    optimizer, criterion = ns['optimizer'], ns['criterion']
    scaler = ns['scaler']
    train_loader, val_loader = ns['train_loader'], ns['val_loader']

    loss = train_epoch(model, train_loader, optimizer, criterion, scaler, 1, ema=ema)
    assert math.isfinite(loss) and loss < 0.8, f'train loss {loss}'
    print(f'[PASS] 1-step train loss={loss:.4f} (finite)')

    # ---- 断言 7: 验证 (TTA + 诊断池化 + jitter, B=2) + 确定性 ----
    vm = validate_epoch(model, val_loader, None)
    assert vm['probs'].shape == (2, 12), vm['probs'].shape
    assert len(vm['uids']) == 2
    assert CFG['tta_jitter'] is True
    vm2 = validate_epoch(model, val_loader, None)
    assert np.allclose(vm['probs'], vm2['probs'], atol=1e-6), 'validate 非确定性'
    print(f'[PASS] validate: probs {vm["probs"].shape}, loss={vm["loss"]:.4f} '
          f'(jitter ON, 两次一致)')

    # ---- 断言 8: checkpoint 往返 (17 的加载路径) + EMA swap ----
    ckpt_dir = tempfile.mkdtemp(prefix='v6a_ckpt_')
    torch.save({'epoch': 1, 'model': model.state_dict(),
                'ema': ema.state_dict(), 'auc': 0.5, 'config': CFG,
                'targets': ns['TARGET_COLUMNS'], 'slots': ns['SLOTS']},
               Path(ckpt_dir) / ns['CKPT_NAME'])
    ckpt = torch.load(Path(ckpt_dir) / ns['CKPT_NAME'], map_location='cpu',
                      weights_only=False)
    import torchvision
    bb2 = torchvision.models.resnet50(weights=None)
    bb2.fc = torch.nn.Identity()
    m2 = ns['RadResNetModel'](backbone=bb2, n_slots=ns['N_SLOT'],
                              feature_dim=CFG['feature_dim'],
                              n_classes=CFG['num_classes'],
                              slot_hidden=CFG['slot_hidden'], dropout=0.0,
                              unfreeze_layers=CFG['unfreeze_layers'])
    st = dict(ckpt['model'])
    for name in st:                       # ★ 17 的 EMA swap 逻辑
        if name in ckpt['ema']['shadow']:
            st[name] = ckpt['ema']['shadow'][name]
    m2.load_state_dict(st, strict=True)
    m2.eval()
    with torch.no_grad():
        o1 = model(ximg, mask)
        o2 = m2(ximg, mask)
    assert torch.allclose(o1, o2, atol=1e-6), 'EMA 往返输出不一致'
    assert (ckpt.get('config') or {}).get('seed') == CFG['seed'], 'ckpt seed 交叉核对'
    print('[PASS] checkpoint 往返 (strict) + EMA swap 输出一致 + seed 交叉核对')

    print('\n=== V6A SMOKE TEST PASSED (CPU) ===')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
