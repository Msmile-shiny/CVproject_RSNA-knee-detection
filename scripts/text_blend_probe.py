# -*- coding: utf-8 -*-
"""离线探针 (纯 CPU, 不碰 GPU/torch): 文本教师分数 rank 混合能涨多少 gold?

背景: v5_fusion_report.csv 显示文本教师在不同 target 上强弱分化严重
(ACL 0.953 / MCL 0.964 / Baker's 0.940 vs Synovitis 0.709 / Effusion 0.777)。
0.91 参考 notebook 的做法是 per-finding 的 rank 加权混合专家分数。
本探针在 58 gold 上回答: 图像三成员融合 rank × 文本 rank, 宏 AUC 怎么变。

⚠️ 口径警告: 文本教师是用 train.csv 标签训的, 58 gold 的文本分数存在标签泄漏,
gold 增益乐观; 最终以 LB 为准。本探针只做"方向有没有肉"的判断。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = r'd:\Learn\Programming\Python\project\CVproject_RSNA-knee-detection'
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
           'Contusion', 'Fracture']


def macro_auc(prob: pd.DataFrame, truth: pd.DataFrame) -> float:
    aucs = []
    for c in TARGETS:
        y = truth[c].values
        if len(np.unique(y)) < 2:
            continue
        aucs.append(roc_auc_score(y, prob[c].values))
    return float(np.mean(aucs))


def rank_cols(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        rankdata(df.values, axis=0, method='average') / len(df),
        index=df.index, columns=df.columns)


def main() -> None:
    fused = pd.read_csv(f'{ROOT}/results/fusion_v5s1s2s3/gold_predictions_fused.csv',
                        dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    text = pd.read_csv(f'{ROOT}/data/processed/report_labels_v2.csv',
                       dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[TARGETS]
    train = pd.read_csv(f'{ROOT}/data/metadata/train.csv',
                        dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    gold = train[TARGETS].dropna()

    common = gold.index.intersection(fused.index).intersection(text.index)
    n = len(common)
    y = gold.loc[common]
    r_img = rank_cols(fused.loc[common])
    r_txt = rank_cols(text.loc[common])

    base = macro_auc(r_img, y)
    print(f'gold studies: {n}')
    print(f'基线 (三成员融合 rank): macro AUC = {base:.4f}  (对照: 0.8959)')

    # ---- 逐 target 现状: 融合图像 vs 文本 ----
    print('\n逐 target AUC (融合图像 | 文本教师 | 差):')
    auc_img, auc_txt = {}, {}
    for c in TARGETS:
        if len(np.unique(y[c])) < 2:
            continue
        a_i = roc_auc_score(y[c], r_img[c])
        a_t = roc_auc_score(y[c], r_txt[c])
        auc_img[c], auc_txt[c] = a_i, a_t
        flag = ' <<文本更强' if a_t > a_i + 0.02 else (' <<图像更强' if a_i > a_t + 0.02 else '')
        print(f'  {c:18s} {a_i:.3f} | {a_t:.3f} | {a_t - a_i:+.3f}{flag}')

    # ---- 探针 A: 全局混合 (所有 target 同一权重) ----
    print('\n探针 A — 全局混合 (1-w)·图像 + w·文本:')
    for w in [0.1, 0.2, 0.3, 0.5]:
        a = macro_auc((1 - w) * r_img + w * r_txt, y)
        print(f'  w={w:.1f} → {a:.4f} ({a - base:+.4f})')

    # ---- 探针 B: 仅混合"文本更强"的 target (数据驱动规则) ----
    strong = [c for c in TARGETS if auc_txt.get(c, 0) > auc_img.get(c, 0) + 0.02]
    weak = [c for c in TARGETS if c not in strong]
    print(f'\n探针 B — 仅混合文本更强 target {strong}:')
    for w in [0.25, 0.5, 0.75]:
        blend = r_img.copy()
        for c in strong:
            blend[c] = (1 - w) * r_img[c] + w * r_txt[c]
        a = macro_auc(blend, y)
        print(f'  w={w:.2f} → {a:.4f} ({a - base:+.4f})  弱 target 保持纯图像: {weak}')

    # ---- 探针 C: 逐 target 贪心 (每个 target 独立选 w∈{0,.25,.5,.75,1}) ----
    print('\n探针 C — 逐 target 贪心:')
    best_w = {}
    cur = r_img.copy()
    for c in TARGETS:
        gains = []
        for w in [0, 0.25, 0.5, 0.75, 1.0]:
            trial = cur.copy()
            trial[c] = (1 - w) * r_img[c] + w * r_txt[c]
            gains.append((macro_auc(trial, y), w))
        a, w = max(gains, key=lambda t: t[0])
        best_w[c] = w
        cur[c] = (1 - w) * r_img[c] + w * r_txt[c]
    print(f'  {best_w}')
    print(f'  宏 AUC → {macro_auc(cur, y):.4f} ({macro_auc(cur, y) - base:+.4f})')
    print('  ⚠️ 贪心在 58 研究上必然过拟合, 只参考"哪些 target 值得给文本权重"')

    print('\n结论口径: 文本分数在 gold 上泄漏 (教师见过这些标签), 增益需 LB 验证。')


if __name__ == '__main__':
    main()
