"""v47 fork α 扫描: 第 21 成员 (v5 3-seed rank-mean) 混入 v47 核心 (58 gold, CPU only).

问题: fork 提交②的 OUR_ALPHA 取多少?
  final = (1-α)·theirs + α·ours_rank_mean
  本地唯一评测面 = 58 gold (LB 为非公开 hidden 集, 不可用)。

成员:
  theirs  v47 核心 20 成员 _combine 输出 (gold 池): fork_v47_gold_v3/gold_members/
          gold_combined_weighted_rank.csv — 与测试端 submission.csv 同一代码路径。
          注意: 此为「核心 20 成员」代理, 不含 E10/E11/M2 后续段 (gold emission 未覆盖)。
  ours    v5 3-seed rank-mean: 每 seed rankdata(prob, axis=0, average)/n, 三 seed 平均
          (与 ours_blend_cell 公式逐字一致)。

输出:
  α ∈ {0, 0.05, ..., 0.30} + ours alone (α=1.0) 的全局 macro AUC;
  α=0 (纯 theirs) vs ours alone 的逐类 AUC 差 (仅供理解, 裁决只用全局)。

裁决准则 (58-gold 全局限定):
  best α 相对 α=0 提升 < +0.003 → 维持 α=0 (纯复刻, 提交①);
  提升可观 (≥+0.003) 且曲线单调/平滑 → 取该 α 提交②。

用法 (d2l env, CPU):
  PYTHONIOENCODING=utf-8 python scripts/alpha_scan_v47_v5.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
           'Contusion', 'Fracture']
SEEDS = [(42, RESULTS / 'v5s1' / 'gold_validation_predictions_s42.csv'),
         (142, RESULTS / 'v5s2' / 'gold_validation_predictions_s142.csv'),
         (242, RESULTS / 'v5s3' / 'gold_validation_predictions_s242.csv')]
THEIRS = (RESULTS / 'fork_v47_gold_v3' / 'gold_members'
          / 'gold_combined_weighted_rank.csv')
ALPHAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def macro_auc(y_true, y_prob):
    aucs = [roc_auc_score(y_true[:, j], y_prob[:, j]) for j in range(y_true.shape[1])]
    return np.mean(aucs), aucs


def main():
    th_df = pd.read_csv(THEIRS, dtype={'StudyInstanceUID': str})
    th = th_df[TARGETS].to_numpy(np.float64)
    n = len(th_df)
    assert n == 58, f'expected 58 gold studies, got {n}'

    # 58-gold 真值: 三个 seed 文件内嵌 true_* 列 (三者应一致)
    y_true = None
    raw = {}
    for seed, path in SEEDS:
        df = pd.read_csv(path, dtype={'StudyInstanceUID': str})
        assert (df['StudyInstanceUID'] == th_df['StudyInstanceUID']).all(), \
            f'seed {seed} UID misaligned'
        true_cols = [c for c in df.columns if c.startswith('true_')]
        y = df[true_cols].to_numpy(np.float64)
        if y_true is None:
            y_true = y
        else:
            assert np.array_equal(y_true, y), f'seed {seed} labels diverge'
        prob_cols = [c for c in df.columns if c.startswith('prob_')]
        raw[seed] = df[prob_cols].to_numpy(np.float64)

    # ours: 与 blend cell 逐字一致的 rank mean
    ours = np.mean([rankdata(raw[s], axis=0, method='average') / n for s, _ in SEEDS],
                   axis=0)

    # theirs (纯复刻) 基线
    auc0, auc0_cls = macro_auc(y_true, th)
    auc_ours, auc_ours_cls = macro_auc(y_true, ours)
    print(f'{n} gold studies | theirs (core 20 _combine): macro {auc0:.4f} | '
          f'ours (3-seed rank-mean): macro {auc_ours:.4f}')
    print()

    # α 扫描
    print(f'{"alpha":>6} | {"macro AUC":>10} | {"vs alpha=0":>10} | vs ours alone')
    print('-' * 58)
    best = (0.0, auc0)
    for a in ALPHAS:
        blend = (1.0 - a) * th + a * ours
        auc, _ = macro_auc(y_true, blend)
        if auc > best[1] + 1e-6:
            best = (a, auc)
        print(f'{a:>6.2f} | {auc:>10.4f} | {auc - auc0:>+10.4f} | {auc - auc_ours:>+9.4f}')
    print('-' * 58)
    print(f'best alpha = {best[0]:.2f} (macro {best[1]:.4f}, vs alpha=0: '
          f'{best[1] - auc0:+.4f})')
    print()

    # 逐类对照 (纯理解用; 裁决只用全局)
    print('per-class AUC 对照 (theirs | ours | diff):')
    for j, t in enumerate(TARGETS):
        print(f'  {t:>18}: {auc0_cls[j]:.4f} | {auc_ours_cls[j]:.4f} | '
              f'{auc_ours_cls[j] - auc0_cls[j]:+.4f}')

    verdict = ('≥+0.003: α 融合有全局杠杆 (提交②)' if best[1] - auc0 >= 0.003
               else '<+0.003: 无全局杠杆 → 维持 α=0 纯复刻 (提交①)')
    print()
    print(f'VERDICT: {verdict} (58-gold 全局裁决, 不含 E10/E11/M2 段)')


if __name__ == '__main__':
    main()
