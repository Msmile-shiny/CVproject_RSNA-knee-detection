# -*- coding: utf-8 -*-
"""lateral_swap 推理 notebook 本地 CPU 冒烟测试 — 执行 cells_infer_lateral
(跳过 DICOM 解码, 合成缓存; 模型推理用 StubModel, 融合算术用真实 gold 矩阵).

★ 本机 GPU 冒烟永久禁止 (两次 HYPERVISOR_ERROR 蓝屏) → 强制 CPU
  (monkeypatch torch.cuda.device_count=0, DEVICE 自动落到 cpu)。

验证:
  1. cell 02/05/06: 配置 + 双模型定义前向 (rad 归一化 x/127.5−1 域断言)
  2. cell 08: 真实 4+1 checkpoint 扫描 + 配置交叉核对 (v5 vs CFG_V5, rad vs CFG_RAD)
     + 真实 state_dict 键兼容 (strict 加载 missing/unexpected 为空)
  3. cell 09-11: 合成 test/gold 缓存 → 窗口化/TTA/jitter/诊断池化/0.5 补齐 全链路
  4. cell 12: 真实 per-seed + rad gold 矩阵 (训练会话产物) →
     BASE ≈ 0.8958±0.002, SWAP ≈ 0.9057±0.002, LM/LO 增量 ≈ +0.060/+0.059;
     合成 test → submission.csv / _base / _lateral_blend 三产物数值逐位校验
  5. notebook JSON 结构 + 全部 code cell 可编译
"""
from __future__ import annotations

import io
import json
import math
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / 'notebooks' / 'cells_infer_lateral'
NOTEBOOK = ROOT / 'notebooks' / 'kaggle_inference_lateral_swap.ipynb'
TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
    'Contusion', 'Fracture',
]
SWAP = ('Lateral Meniscus', 'Lateral OA')


def exec_cell(ns, fname):
    src = (CELLS / fname).read_text(encoding='utf-8')
    compile(src, fname, 'exec')
    buf = io.StringIO()
    with redirect_stdout(buf):
        exec(src, ns)
    return buf.getvalue()


class StubModel(torch.nn.Module):
    """占位推理模型: 返回 [N, 12] 零 logits (绕过 132MB 真模型的前向成本)。"""

    def forward(self, images, mask):
        return torch.zeros(images.shape[0], 12)


def load_gold_matrix(path):
    """训练会话 gold CSV → [58, 12] prob 矩阵 (按 gold_studies 顺序)。"""
    df = pd.read_csv(path)
    df['StudyInstanceUID'] = df['StudyInstanceUID'].astype(str)
    df = df.set_index('StudyInstanceUID').loc[[u for u in gold_uids_order]]
    return np.stack([df[f'prob_{c}'].to_numpy() for c in TARGETS], axis=1)


def main() -> int:
    ns = {'__name__': '__main__'}
    print('=== cell 01: imports ===')
    print(exec_cell(ns, '01_imports.py'))

    # ★ 强制 CPU (本机 GPU 冒烟禁止 — 蓝屏史)
    torch.cuda.device_count = lambda: 0
    torch.cuda.is_available = lambda: False

    print('=== cell 02: config (CPU forced) ===')
    print(exec_cell(ns, '02_config.py'))
    CFG_V5, CFG_RAD = ns['CFG_V5'], ns['CFG_RAD']
    assert str(ns['DEVICE']) == 'cpu', f'DEVICE={ns["DEVICE"]} — 本机冒烟必须 CPU'
    assert CFG_V5['image_size'] == 288 and CFG_V5['cache_slices'] == 9
    assert CFG_RAD['image_size'] == 224 and CFG_RAD['cache_slices'] == 7
    assert ns['N_WINDOWS_V5'] == 7 and ns['N_WINDOWS_RAD'] == 5
    print('[PASS] 双配置: v5 288px/9片/7窗口 | rad 224px/7片/5窗口')

    print('=== cell 03/04: slot matching + dicom io ===')
    print(exec_cell(ns, '03_slot_matching.py'))
    print(exec_cell(ns, '04_dicom_io.py'))

    print('=== cell 05: models ===')
    print(exec_cell(ns, '05_model.py'))

    # ---- 断言 A: rad 归一化域 + 前向 (v6a 冒烟同款) ----
    RadResNetModel, SlotHead = ns['RadResNetModel'], ns['SlotHead']
    import torchvision
    bb = torchvision.models.resnet50(weights=None)
    bb.fc = torch.nn.Identity()
    mrad = RadResNetModel(backbone=bb, n_slots=ns['N_SLOT'], feature_dim=2048,
                          n_classes=12, slot_hidden=256, dropout=0.0,
                          unfreeze_layers=0).eval()
    with torch.no_grad():
        ximg = torch.randint(0, 255, (1, 6, 3, 224, 224), dtype=torch.uint8)
        mask = torch.ones(1, 6)
        out = mrad(ximg, mask)
        assert out.shape == (1, 12), out.shape
        assert mrad.mean.unique().item() == 127.5 and mrad.std.unique().item() == 127.5
        lo = (torch.zeros(1, 3, 1, 1) - mrad.mean) / mrad.std
        hi = (torch.full((1, 3, 1, 1), 255.0) - mrad.mean) / mrad.std
        assert lo.unique().item() == -1.0 and hi.unique().item() == 1.0, \
            f'rad 归一化域错误: [{lo.unique().item()}, {hi.unique().item()}] 应为 [−1, +1]'
    print('[PASS] RadResNetModel 前向 [1,6,3,224,224]→[1,12]; 归一化 x/127.5−1 (全黑→−1/全白→+1)')

    # ---- 断言 B: v5 MultiViewModel 前向 (1 窗口 @288, CPU) ----
    MultiViewModel = ns['MultiViewModel']
    import timm
    bb_v5 = timm.create_model(CFG_V5['dinov2_variant'], pretrained=False,
                              num_classes=0, img_size=288)
    mv5 = MultiViewModel(dinov2_model=bb_v5, n_slots=ns['N_SLOT'], cls_dim=384,
                         n_classes=12, slot_hidden=256, dropout=0.0,
                         unfreeze_layers=6).eval()
    with torch.no_grad():
        out5 = mv5(torch.randint(0, 255, (1, 1, 3, 288, 288), dtype=torch.uint8),
                   torch.ones(1, 1))
        assert out5.shape == (1, 12), out5.shape
    print('[PASS] MultiViewModel 前向 [1,1,3,288,288]→[1,12] (timm vit-small@288, CPU)')

    # ---- 断言 C: jitter + stack_views + diagnostic_pool (v6a 冒烟同款) ----
    tta_jitter, diag_pool, stack_views = ns['tta_jitter'], ns['diagnostic_pool'], ns['stack_views']
    x = torch.randint(0, 255, (4, 6, 3, 32, 32), dtype=torch.uint8)
    xj = tta_jitter(x)
    assert xj.shape == x.shape and xj.dtype == x.dtype
    assert float((xj != x).float().mean()) > 0.9
    assert torch.equal(xj, tta_jitter(x)), 'tta_jitter 非确定性'
    logits_v = torch.randn(2, 14, 12)
    out = diag_pool(logits_v, n_orig=7)
    j_syn = ns['TARGET_COLUMNS'].index('Synovitis')
    j_frac = ns['TARGET_COLUMNS'].index('Fracture')
    view_mean = (torch.sigmoid(logits_v[:, :7]) + torch.sigmoid(logits_v[:, 7:])) / 2
    assert torch.allclose(out[:, j_syn], torch.sigmoid(logits_v[:, :7, j_syn]).mean(1), atol=1e-5)
    assert torch.allclose(out[:, j_frac], view_mean[:, :, j_frac].max(1).values, atol=1e-5)
    Bt, Wt, Ct = 3, 4, 2
    lg = torch.arange(2 * Bt * Wt * Ct, dtype=torch.float32).reshape(2 * Bt * Wt, Ct)
    v = stack_views(lg, Bt, Wt, Wt)
    for b in range(Bt):
        assert torch.equal(v[b, 0], lg[b * Wt]), '跨研究串位'
        assert torch.equal(v[b, Wt], lg[Bt * Wt + b * Wt])
    print('[PASS] tta_jitter 确定性 / diagnostic_pool / stack_views B-major (B>1)')

    # ============================================================
    # 合成竞赛元数据 (真实 58 gold 标签 + 合成 5 test)
    # ============================================================
    global gold_uids_order
    gold_dfs = {}
    for seed_tag, path in [('s42', ROOT / 'results' / 'v5s1' / 'gold_validation_predictions_s42.csv'),
                           ('s142', ROOT / 'results' / 'v5s2' / 'gold_validation_predictions_s142.csv'),
                           ('s242', ROOT / 'results' / 'v5s3' / 'gold_validation_predictions_s242.csv'),
                           ('rad', ROOT / 'results' / 'v6a' / 'gold_validation_predictions_rad.csv')]:
        assert path.exists(), f'gold CSV 缺失: {path}'
        df = pd.read_csv(path)
        df['StudyInstanceUID'] = df['StudyInstanceUID'].astype(str)
        gold_dfs[seed_tag] = df.set_index('StudyInstanceUID')
    gold_uids_order = sorted(gold_dfs['s42'].index)
    assert len(gold_uids_order) == 58, f'gold {len(gold_uids_order)} != 58'
    for tag in ('s142', 's242', 'rad'):
        assert sorted(gold_dfs[tag].index) == gold_uids_order, f'{tag} UID 集不一致'

    meta_dir = Path(tempfile.mkdtemp(prefix='lateral_meta_'))
    train_csv = pd.DataFrame({'StudyInstanceUID': gold_uids_order})
    for c in TARGETS:
        train_csv[c] = [int(gold_dfs['s42'].loc[u, f'true_{c}']) for u in gold_uids_order]
    train_csv.to_csv(meta_dir / 'train.csv', index=False)

    planes = ['Sagittal', 'Coronal', 'Axial', 'Sagittal', 'Coronal', 'Sagittal']
    fluids = [1, 1, 1, 1, 0, 0]
    fatsats = [1, 1, 1, 0, 0, 0]
    train_series_rows = []
    for u in gold_uids_order:
        for i in range(6):
            train_series_rows.append({
                'StudyInstanceUID': u, 'SeriesInstanceUID': f'SER_{u[:12]}_{i}',
                'Anatomical_Plane': planes[i], 'Fluid_Sensitive': fluids[i],
                'Fat_Suppression': fatsats[i]})
    pd.DataFrame(train_series_rows).to_csv(meta_dir / 'train_series.csv', index=False)

    test_uids = [f'1.0.TEST.{i:04d}' for i in range(5)]
    pd.DataFrame({'StudyInstanceUID': test_uids}).to_csv(meta_dir / 'test.csv', index=False)
    test_series_rows = []
    for u in test_uids:
        for i in range(6):
            test_series_rows.append({
                'StudyInstanceUID': u, 'SeriesInstanceUID': f'SER_{u}_{i}',
                'Anatomical_Plane': planes[i], 'Fluid_Sensitive': fluids[i],
                'Fat_Suppression': fatsats[i]})
    pd.DataFrame(test_series_rows).to_csv(meta_dir / 'test_series.csv', index=False)
    print(f'synthetic metadata at {meta_dir} (train.csv 58 真实 gold 标签 + 5 合成 test)')

    out_dir = Path(tempfile.mkdtemp(prefix='lateral_out_'))
    CFG_V5.update({'comp_input': str(meta_dir),
                   'ckpt_input': str(ROOT / 'results'),   # 真实 checkpoint 扫描
                   'output_dir': str(out_dir)})
    CFG_RAD.update({'comp_input': str(meta_dir), 'output_dir': str(out_dir)})

    print('=== cell 06: load gold ===')
    print(exec_cell(ns, '06_load_gold.py'))
    assert len(ns['gold_studies']) == 58
    assert sorted(ns['gold_studies']) == gold_uids_order
    print('[PASS] gold_studies = 58 真实 UID, 与训练会话 gold CSV 一致')

    # ---- 跳过 07 (DICOM 解码), 注入合成缓存 (32px 加速) ----
    ns['_read_slot_job'] = lambda args: (args[0], args[1], None)
    ns['gold_study_index'] = {u: i for i, u in enumerate(ns['gold_studies'])}
    ns['GOLD_CACHE_V5'] = np.random.randint(0, 255, (58, 6, 9, 32, 32), dtype=np.uint8)
    ns['GOLD_MASK_V5'] = np.ones((58, 6), dtype=np.float32)
    ns['GOLD_CACHE_RAD'] = np.random.randint(0, 255, (58, 6, 7, 32, 32), dtype=np.uint8)
    ns['GOLD_MASK_RAD'] = np.ones((58, 6), dtype=np.float32)
    print('synthetic gold caches injected (58x6x{9,7}x32x32)')

    print('=== cell 08: checkpoint 发现 + 交叉核对 + 模型构建 (真实 .pt) ===')
    out08 = exec_cell(ns, '08_checkpoints.py')
    print(out08)
    for k in ('s42', 's142', 's242', 'rad'):
        assert f'{k:>5s}' in out08, f'交叉核对输出缺 {k}'
    ck = ns['ckpt_meta']
    assert set(ns['seeds']) == {42, 142, 242}, f'seeds={ns["seeds"]}'
    assert ns['has_rad'] is True
    print('[PASS] 4 成员 checkpoint 扫描 + 配置交叉核对通过')

    # ---- 真实 state_dict 键兼容 (strict 加载 missing/unexpected 必须为空) ----
    def _strip_ema(st):
        if next(iter(st)).startswith('module.'):
            st = {k.replace('module.', '', 1): v for k, v in st.items()}
        return st

    st_rad = dict(ck['rad']['model'])
    if ck['rad'].get('ema') and ck['rad']['ema'].get('shadow'):
        for name in st_rad:
            if name in ck['rad']['ema']['shadow']:
                st_rad[name] = ck['rad']['ema']['shadow'][name]
    missing, unexpected = ns['infer_model_rad'].load_state_dict(_strip_ema(st_rad), strict=False)
    assert not missing and not unexpected, f'rad 键不兼容: missing={missing[:3]} unexpected={unexpected[:3]}'
    st_s42 = dict(ck[42]['model'])
    if ck[42].get('ema') and ck[42]['ema'].get('shadow'):
        for name in st_s42:
            if name in ck[42]['ema']['shadow']:
                st_s42[name] = ck[42]['ema']['shadow'][name]
    missing, unexpected = ns['infer_model_v5'].load_state_dict(_strip_ema(st_s42), strict=False)
    assert not missing and not unexpected, f'v5 键不兼容: missing={missing[:3]} unexpected={unexpected[:3]}'
    print('[PASS] 真实 checkpoint state_dict 键与推理模型 strict 兼容 (rad + s42)')

    # ---- 换 StubModel (绕过真实前向成本) ----
    ns['infer_model_v5'] = StubModel().eval()
    ns['infer_model_rad'] = StubModel().eval()

    print('=== cell 09: test slot 匹配 + 双缓存 (解码 stub) ===')
    print(exec_cell(ns, '09_test_cache.py'))
    assert ns['test_studies'] == test_uids, f'test_studies={ns["test_studies"]}'
    # 覆盖为 32px 合成缓存 (含 1 个全空槽研究 → 0.5 补齐路径)
    ns['TEST_CACHE_V5'] = np.random.randint(0, 255, (5, 6, 9, 32, 32), dtype=np.uint8)
    ns['TEST_MASK_V5'] = np.ones((5, 6), dtype=np.float32)
    ns['TEST_MASK_V5'][4] = 0.0
    ns['TEST_CACHE_RAD'] = np.random.randint(0, 255, (5, 6, 7, 32, 32), dtype=np.uint8)
    ns['TEST_MASK_RAD'] = np.ones((5, 6), dtype=np.float32)
    print('[PASS] test_studies 5 研究; 合成 test 缓存注入 (study#4 全空槽)')

    print('=== cell 10: v5 成员推理 (StubModel) ===')
    print(exec_cell(ns, '10_infer_v5.py'))
    assert set(ns['member_gold_probs'].keys()) >= {42, 142, 242}
    assert ns['member_test_probs'][42].shape == (5, 12)
    assert np.allclose(ns['member_test_probs'][42][4], 0.5), '全空槽研究未补 0.5'
    assert np.allclose(ns['member_test_probs'][42][:4], 0.5), 'stub 期望 0.5'
    print('[PASS] v5 成员循环: gold 58 行 + test 5 行 (0.5 补齐)')

    print('=== cell 11: rad 成员推理 (StubModel) ===')
    print(exec_cell(ns, '11_infer_rad.py'))
    assert 'rad' in ns['member_gold_probs']
    assert 'GOLD_CACHE_V5' not in ns, 'gold 缓存未释放'
    print('[PASS] rad 成员循环 + gold 缓存释放')

    # ---- 覆盖为真实 gold 矩阵 (训练会话产物), 验证融合算术 ----
    compute_gold_aucs = ns['compute_gold_aucs']
    real = {k: load_gold_matrix(ROOT / 'results' / 'v5s1' / 'gold_validation_predictions_s42.csv') for k in (42,)}
    real[142] = load_gold_matrix(ROOT / 'results' / 'v5s2' / 'gold_validation_predictions_s142.csv')
    real[242] = load_gold_matrix(ROOT / 'results' / 'v5s3' / 'gold_validation_predictions_s242.csv')
    real['rad'] = load_gold_matrix(ROOT / 'results' / 'v6a' / 'gold_validation_predictions_rad.csv')
    ns['member_gold_probs'] = real
    ns['member_aucs'] = {k: compute_gold_aucs(p) for k, p in real.items()}
    ns['has_spec'] = False   # spec 仅诊断; 冒烟未注入其真实矩阵, 关闭其打印分支

    # 随机 test 矩阵 (可区分 swap/base/blend, 检验提交产物逐位正确)
    rng = np.random.RandomState(0)
    ns['member_test_probs'] = {42: rng.rand(5, 12), 142: rng.rand(5, 12),
                               242: rng.rand(5, 12), 'rad': rng.rand(5, 12)}

    print('=== cell 12: lateral swap 融合 + 提交 (真实 gold 矩阵) ===')
    out12 = exec_cell(ns, '12_fusion_swap.py')
    print(out12)

    base_macro = ns['base_macro']
    swap_macro = ns['swap_macro']
    assert abs(base_macro - 0.8958) < 0.002, f'BASE gold {base_macro:.4f} != 0.8958±0.002'
    assert abs(swap_macro - 0.9057) < 0.002, f'SWAP gold {swap_macro:.4f} != 0.9057±0.002'
    j_lm, j_lo = TARGETS.index('Lateral Meniscus'), TARGETS.index('Lateral OA')
    d_lm = ns['swap_aucs'][TARGETS[j_lm]] - ns['base_aucs'][TARGETS[j_lm]]
    d_lo = ns['swap_aucs'][TARGETS[j_lo]] - ns['base_aucs'][TARGETS[j_lo]]
    assert abs(d_lm - 0.060) < 0.01 and abs(d_lo - 0.059) < 0.01, \
        f'LM/LO 增量 ({d_lm:+.3f}/{d_lo:+.3f}) != +0.060/+0.059'
    print(f'[PASS] 融合算术: BASE {base_macro:.4f} / SWAP {swap_macro:.4f} '
          f'(LM +{d_lm:.3f}, LO +{d_lo:.3f})')

    # ---- 提交产物逐位校验 ----
    sub_swap = pd.read_csv(out_dir / 'submission.csv')
    sub_base = pd.read_csv(out_dir / 'submission_base.csv')
    sub_blend = pd.read_csv(out_dir / 'submission_lateral_blend.csv')
    assert list(sub_swap.columns) == ['StudyInstanceUID'] + TARGETS
    assert len(sub_swap) == 5
    base_test = ns['base_test']
    rad_rank = rankdata(ns['member_test_probs']['rad'], axis=0, method='average') / 5
    for j, c in enumerate(TARGETS):
        col = sub_swap[c].to_numpy()
        if c in SWAP:
            assert np.allclose(col, rad_rank[:, j], atol=1e-6), f'{c} 非 rad_rank'
        else:
            assert np.allclose(col, base_test[:, j], atol=1e-6), f'{c} 非 base'
        assert np.allclose(sub_base[c].to_numpy(), base_test[:, j], atol=1e-6)
        if c in SWAP:
            assert np.allclose(sub_blend[c].to_numpy(),
                               (base_test[:, j] + rad_rank[:, j]) / 2, atol=1e-6)
        else:
            assert np.allclose(sub_blend[c].to_numpy(), base_test[:, j], atol=1e-6)
    print('[PASS] submission.csv (swap) / _base / _lateral_blend 逐位正确')

    # ---- notebook JSON 结构 + code cell 编译 ----
    nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    assert len(nb['cells']) == 25 and len(code_cells) == 12, \
        f'cells={len(nb["cells"])} code={len(code_cells)}'
    for c in code_cells:
        compile(''.join(c['source']), 'cell', 'exec')
    print(f'[PASS] notebook 结构: {len(nb["cells"])} cells ({len(code_cells)} code) 全部可编译')

    print('\n=== LATERAL SWAP INFERENCE SMOKE TEST PASSED (CPU) ===')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
