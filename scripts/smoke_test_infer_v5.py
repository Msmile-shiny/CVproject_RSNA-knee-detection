# -*- coding: utf-8 -*-
"""⚠️⚠️ 勿在本机运行 — 2026-08-15 曾两次触发笔记本蓝屏 (HYPERVISOR_ERROR) ⚠️⚠️

本脚本按 Kaggle 服务器规格设计 (cell 08 单批 672 张 288² 图过 DINOv2),
在 RTX 5060 Laptop (8GB) 上会导致驱动挂起 → 蓝屏。本机 GPU 冒烟已永久禁止,
验证走: (a) 静态检查 (build_infer_v5.py + 编译核对) 已通过;
(b) Kaggle 上直接跑推理 notebook, gold 读数对照训练会话 (0.8933/0.8899/0.8937)。
本文件仅存档, 不要执行。

推理 notebook 本地冒烟测试 — cells_infer_v5 全流程 (合成 DICOM + 真实 checkpoint).

验证:
  1. find_checkpoints: 从 results/ 找到 3 个真实成员 (s42/s142/s242)
  2. 配置交叉核对: checkpoint config vs 推理 CFG (数据侧参数逐项一致)
  3. gold 缓存构建 + 每成员 7 窗口 TTA+jitter 推理 (B=8 批次, 覆盖 B>1)
  4. test 缓存 + 推理 (B=3, 覆盖 B>1)
  5. rank-mean 融合: fused_gold == 手工重算; submission.csv 3 行且与手工一致
  6. 产物: 每成员 gold CSV (58 行 true_/prob_) + fused CSV + submission

说明: 本地无真实 DICOM → 用假 read_series_volume 注入随机 volume,
解码/裁剪/缩放路径走通, 但预测值无意义 (AUC ~0.5 正常)。
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / 'notebooks' / 'cells_infer_v5'
META = ROOT / 'data' / 'metadata'


def exec_cell(ns, fname):
    src = (CELLS / fname).read_text(encoding='utf-8')
    compile(src, fname, 'exec')
    buf = io.StringIO()
    with redirect_stdout(buf):
        exec(src, ns)
    return buf.getvalue()


def make_dummy_dicoms(comp_input):
    """为 gold/test 的每个 series 建假 DICOM 目录 (空文件, 只让 slot 匹配通过)。"""
    import pandas as pd

    tm = pd.read_csv(META / 'train.csv')
    tm['StudyInstanceUID'] = tm['StudyInstanceUID'].astype(str)
    cols = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
            'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
            'Contusion', 'Fracture']
    gold = set(tm[tm[cols].notna().all(axis=1)]['StudyInstanceUID'].unique())

    sm = pd.read_csv(META / 'train_series.csv')
    sm['StudyInstanceUID'] = sm['StudyInstanceUID'].astype(str)
    sm['SeriesInstanceUID'] = sm['SeriesInstanceUID'].astype(str)
    gold_rows = sm[sm['StudyInstanceUID'].isin(gold)]

    ts = pd.read_csv(META / 'test_series.csv')
    ts['StudyInstanceUID'] = ts['StudyInstanceUID'].astype(str)
    ts['SeriesInstanceUID'] = ts['SeriesInstanceUID'].astype(str)

    n_dirs = 0
    for subdir, rows in [('train_series', gold_rows), ('test_series', ts)]:
        base = Path(comp_input) / subdir
        for uid, sid in zip(rows['StudyInstanceUID'], rows['SeriesInstanceUID']):
            d = base / uid / sid
            d.mkdir(parents=True, exist_ok=True)
            for k in range(5):
                (d / f'f{k:03d}').touch()   # 无扩展名 (竞赛同款)
            n_dirs += 1
    return n_dirs


def main() -> int:
    ns = {'__name__': '__main__'}

    # ---- 临时 comp_input: 真实元数据 CSV + 假 DICOM 目录 ----
    comp_input = Path(tempfile.mkdtemp(prefix='infer_smoke_comp_'))
    for f in ['train.csv', 'train_series.csv', 'test.csv', 'test_series.csv']:
        (comp_input / f).write_bytes((META / f).read_bytes())
    n_dirs = make_dummy_dicoms(comp_input)
    print(f'dummy DICOM dirs: {n_dirs}')

    print('=== cell 01: imports ===')
    print(exec_cell(ns, '01_imports.py'))
    print('=== cell 02: config ===')
    print(exec_cell(ns, '02_config.py'))

    # ---- patch CFG for local smoke ----
    CFG = ns['CFG']
    CFG.update({
        'comp_input': str(comp_input),
        'ckpt_input': str(ROOT / 'results'),   # rglob 找到 3 个真实 checkpoint
        'output_dir': tempfile.mkdtemp(prefix='infer_smoke_out_'),
        'pix_threads': 2,
    })

    print('=== cell 03: slot matching ===')
    print(exec_cell(ns, '03_slot_matching.py'))
    print('=== cell 04: dicom io ===')
    print(exec_cell(ns, '04_dicom_io.py'))

    # ---- 注入假 read_series_volume: 本地无真实 DICOM, 返回随机 volume ----
    # 但复用真实 physical_crop + 侧性归一化 + resize 路径 (与 cell 04 同款后处理)
    def fake_read_series_volume(series_dir, plane=None, laterality=None,
                                image_size=224, crop_mm=160.0):
        vol = np.random.rand(15, 300, 300).astype(np.float32)  # [0,1] 15 切片
        vol = ns['physical_crop'](vol, 0.5, crop_mm)
        vol = ns['normalise_laterality'](vol, plane, laterality)
        resized = [ns['cv2'].resize(img, (image_size, image_size),
                                    interpolation=ns['cv2'].INTER_LINEAR)
                   for img in vol]
        return np.stack(resized, axis=0).astype(np.float32), 0.5
    ns['read_series_volume'] = fake_read_series_volume
    print('fake read_series_volume injected (real crop/resize path)')

    print('=== cell 05: model ===')
    print(exec_cell(ns, '05_model.py'))
    print('=== cell 06: load gold ===')
    print(exec_cell(ns, '06_load_gold.py'))

    # ---- 断言 1: gold 加载 ----
    assert len(ns['gold_studies']) == 58, f'gold {len(ns["gold_studies"])} != 58'
    assert len(ns['gold_labels']) == 58
    print(f'[PASS] gold: {len(ns["gold_studies"])} studies')

    print('=== cell 07: gold cache ===')
    out7 = exec_cell(ns, '07_cache_gold.py')
    print(out7)
    # ---- 断言 2: 缓存填充 ----
    assert ns['GOLD_CACHE'].shape == (58, 6, 9, 288, 288), ns['GOLD_CACHE'].shape
    assert ns['GOLD_MASK'].sum() > 250, f'gold mask 填充不足: {ns["GOLD_MASK"].sum()}'
    print(f'[PASS] gold cache: {ns["GOLD_CACHE"].shape}, '
          f'{int(ns["GOLD_MASK"].sum())} series decoded')

    print('=== cell 08: infer + fusion ===')
    out8 = exec_cell(ns, '08_infer_fusion.py')
    print(out8)

    # ---- 断言 3: 成员发现 + 配置核对通过 ----
    checkpoints = ns['checkpoints']
    seeds = [s for s, _ in checkpoints]
    assert seeds == [42, 142, 242], f'seeds {seeds} != [42, 142, 242]'
    for seed, ck in ns['ckpt_meta'].items():
        assert ck.get('ema') and ck['ema'].get('shadow'), f's{seed} EMA missing'
    print(f'[PASS] 3 真实 checkpoint 定位 + 配置交叉核对通过 (EMA 齐)')

    # ---- 断言 4: 成员推理形状 ----
    mgp = ns['member_gold_probs']
    assert sorted(mgp) == seeds and mgp[42].shape == (58, 12), mgp[42].shape
    for s in seeds:
        m = ns['member_aucs'][s][1]
        assert 0.3 < m < 0.7, f's{s} AUC {m} 异常 (随机缓存应 ~0.5)'
    mtp = ns['member_test_probs']
    assert sorted(mtp) == seeds and mtp[42].shape == (3, 12), mtp[42].shape
    print(f'[PASS] 成员推理: gold {mgp[42].shape} (B=8 批次) / test {mtp[42].shape} (B=3)')

    # ---- 断言 5: rank-mean 融合 == 手工重算 ----
    from scipy.stats import rankdata
    manual_gold = np.mean(
        [rankdata(mgp[s], axis=0, method='average') / 58 for s in seeds], axis=0)
    assert np.allclose(manual_gold, ns['fused_gold'], atol=1e-9), 'gold 融合与手工不一致'
    manual_test = np.mean(
        [rankdata(mtp[s], axis=0, method='average') / 3 for s in seeds], axis=0)
    import pandas as pd
    sub = pd.read_csv(Path(CFG['output_dir']) / 'submission.csv')
    assert list(sub.columns) == ['StudyInstanceUID'] + ns['TARGET_COLUMNS']
    assert len(sub) == 3
    assert np.allclose(manual_test, sub[ns['TARGET_COLUMNS']].values, atol=1e-9), \
        'submission 与手工融合不一致'
    print(f'[PASS] rank-mean 融合 == 手工重算 (gold {ns["fused_gold"].shape} / submission 3 行)')

    # ---- 断言 6: 产物文件 ----
    out_dir = Path(CFG['output_dir'])
    for s in seeds:
        p = out_dir / f'gold_validation_predictions_s{s}.csv'
        assert p.exists(), p
        g = pd.read_csv(p)
        assert len(g) == 58 and f'true_ACL' in g.columns and f'prob_ACL' in g.columns
    gf = pd.read_csv(out_dir / 'gold_validation_predictions_fused.csv')
    assert len(gf) == 58
    assert out_dir.joinpath('gold_validation_auc_fused.csv').exists()
    print('[PASS] 产物: 3 成员 gold CSV + fused CSV + submission.csv')

    print('\n=== INFER SMOKE TEST PASSED ===')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
