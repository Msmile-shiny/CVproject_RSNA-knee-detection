# -*- coding: utf-8 -*-
"""构建 v5 融合训练标签：gold × 文本提取器 × 公开集成 OOF (teacher-student 数据侧).

原理 (v5 规划第 ①③ 项):
  - 公开 20 成员集成 (pilkwang) 的 oof.npz 是对全部 4407 个训练研究的 OOF 预测
    (4 seeds × 5 folds, 每研究恰好被每 seed 的 1 个 fold 留出 → 预测为无泄漏 OOF),
    作为 image-side teacher 软标签。
  - 9 语言提取器 (report_extractor_v2) 的 score 作为 text-side 软标签, __conf 为
    提及置信度 (静默 → 低权重, 从不断言阴性)。
  - 每个 finding 在 58 个 gold 研究上拟合 2 特征逻辑回归:
        logit(p_fused) = a + b_text·logit(score) + b_oof·logit(oof)
    系数裁剪到 [0, 1.5], 得到 per-finding 的 teacher 融合权重。
  - 样本权重: w = (0.35 + 0.65·conf), 文本有提及时再乘 text/oof 一致性
    (0.6 + 0.4·(1-|text-oof|)), 下限 0.25。
  - gold 行: prob=硬标签, weight=1.0 (v5 拆分中 gold 仍全部进验证集, 此列供后续
    per-seed holdout 使用)。

输出:
  data/processed/v5_labels.csv          (UID + prob_* + weight_* + mask_*)
  data/processed/v5_fusion_report.csv   (per-target 系数 + 诊断 AUC)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
           'Contusion', 'Fracture']
PROB_COLS = [f'prob_{c}' for c in TARGETS]
WEIGHT_COLS = [f'weight_{c}' for c in TARGETS]
MASK_COLS = [f'mask_{c}' for c in TARGETS]


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))


def fit_fusion(y, x_text_logit, x_oof_logit):
    """在 gold 行上拟合 logit(p) = a + b1·logit(text) + b2·logit(oof), 系数裁剪 [0,1.5]."""
    X = np.stack([x_text_logit, x_oof_logit], axis=1)
    if len(np.unique(y)) < 2:
        return 0.0, 1.0, 1.0  # gold 全正/全负: 退化, 平均融合
    try:
        clf = LogisticRegression(C=0.5, max_iter=1000, solver='lbfgs')
        clf.fit(X, y)
        b1, b2 = float(clf.coef_[0, 0]), float(clf.coef_[0, 1])
        a = float(clf.intercept_[0])
    except Exception:
        return 0.0, 1.0, 1.0
    b1, b2 = float(np.clip(b1, 0.0, 1.5)), float(np.clip(b2, 0.0, 1.5))
    if b1 + b2 < 1e-6:  # 拟合失败/信号为零 → 平均融合
        b1 = b2 = 1.0
        a = 0.0
    return a, b1, b2


def main() -> int:
    # ---- 载入三路信号 ----
    train = pd.read_csv(ROOT / 'data/metadata/train.csv')
    train['StudyInstanceUID'] = train['StudyInstanceUID'].astype(str)
    train = train.set_index('StudyInstanceUID')

    text = pd.read_csv(ROOT / 'data/processed/report_labels_v2.csv', index_col=0)
    text.index = text.index.astype(str)

    z = np.load(ROOT / 'kaggle_dataset/archive/oof.npz', allow_pickle=True)
    oof_ids = z['ids'].astype(str)
    oof_pred = pd.DataFrame(z['pred'], index=oof_ids, columns=TARGETS)
    oof_gold = pd.Series(z['gold_mask'].astype(bool), index=oof_ids)

    # ---- gold 行交叉验证 (train.csv 全 12 列非空 vs oof.npz gold_mask) ----
    gold_all = train[TARGETS].notna().all(axis=1)
    agree = (gold_all == oof_gold.reindex(train.index).fillna(False)).mean()
    print(f'gold 行核对: train.csv={int(gold_all.sum())}, oof.npz={int(oof_gold.sum())}, '
          f'一致率={agree:.3f}')

    # ---- 提取器移植验证: 我们的 score 预测作者的 y_derived ----
    y_derived = pd.DataFrame(z['y_derived'], index=oof_ids, columns=TARGETS)
    port_rows = []
    for t_ in TARGETS:
        y = y_derived[t_].values
        p = text[t_].reindex(y_derived.index).values
        a = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float('nan')
        port_rows.append((t_, round(a, 3)))
    print('提取器移植验证 (our score AUC vs author y_derived):')
    print(pd.DataFrame(port_rows, columns=['target', 'auc']).to_string(index=False))

    # ---- per-finding: gold 诊断 + 拟合融合系数 ----
    rows, coefs = [], {}
    for t_ in TARGETS:
        gidx = train.index[gold_all]
        y = train.loc[gidx, t_].astype(int).values
        p_text = text.loc[gidx, t_].values
        p_oof = oof_pred.loc[gidx, t_].values

        auc_text = roc_auc_score(y, p_text) if len(np.unique(y)) > 1 else float('nan')
        auc_oof = roc_auc_score(y, p_oof) if len(np.unique(y)) > 1 else float('nan')

        a, b1, b2 = fit_fusion(y, logit(p_text), logit(p_oof))
        p_fused = 1 / (1 + np.exp(-(a + b1 * logit(p_text) + b2 * logit(p_oof))))
        auc_fused = roc_auc_score(y, p_fused) if len(np.unique(y)) > 1 else float('nan')

        # 兜底: LR 拟合若显著劣于两个单路信号 (小样本噪声), 改用 AUC 加权定比融合
        best_single = max(auc_text, auc_oof)
        if not np.isnan(auc_fused) and auc_fused < best_single - 0.03:
            w_t = max(0.05, auc_text - 0.5)
            w_o = max(0.05, auc_oof - 0.5)
            b1, b2 = 2 * w_t / (w_t + w_o), 2 * w_o / (w_t + w_o)
            a = 0.0
            p_fused = 1 / (1 + np.exp(-(b1 * logit(p_text) + b2 * logit(p_oof))))
            auc_fused = roc_auc_score(y, p_fused) if len(np.unique(y)) > 1 else float('nan')

        coefs[t_] = (a, b1, b2)
        rows.append((t_, int(y.sum()), round(auc_text, 3), round(auc_oof, 3),
                     round(auc_fused, 3), round(b1, 2), round(b2, 2)))
    report = pd.DataFrame(rows, columns=['target', 'n_pos', 'auc_text', 'auc_oof',
                                         'auc_fused_gold', 'b_text', 'b_oof'])
    print('\nGold 58 上诊断 (auc_fused 为 in-sample, 仅参考):')
    print(report.to_string(index=False))

    # ---- 记忆检查: gold 行的 oof 预测是否异常极端 (相对非 gold 行) ----
    g, ng = oof_pred.loc[gold_all].values, oof_pred.loc[~gold_all].values
    print(f'\n记忆检查: oof pred 均值 gold={g.mean():.3f} non-gold={ng.mean():.3f}; '
          f'|p-0.5| 均值 gold={np.abs(g-0.5).mean():.3f} non-gold={np.abs(ng-0.5).mean():.3f}')

    # ---- 生成全量融合标签 ----
    out = pd.DataFrame(index=train.index)
    for t_ in TARGETS:
        a, b1, b2 = coefs[t_]
        p_text = text[t_].values
        p_oof = oof_pred[t_].reindex(train.index).values
        conf = text[t_ + '__conf'].values

        fused = 1 / (1 + np.exp(-(a + b1 * logit(p_text) + b2 * logit(p_oof))))
        fused = np.clip(fused, 0.02, 0.98)

        w = 0.35 + 0.65 * conf
        mentioned = conf > 0.1
        w[mentioned] *= 0.6 + 0.4 * (1 - np.abs(p_text[mentioned] - p_oof[mentioned]))
        w = np.clip(w, 0.25, 1.0)

        out[f'prob_{t_}'] = fused.astype(np.float32)
        out[f'weight_{t_}'] = w.astype(np.float32)
        out[f'mask_{t_}'] = np.ones(len(train), dtype=np.float32)

    # gold 行: 硬标签 + 权重 1 (v5 中仍走验证集, 此列备用)
    for t_ in TARGETS:
        out.loc[gold_all, f'prob_{t_}'] = train.loc[gold_all, t_].astype(np.float32)
        out.loc[gold_all, f'weight_{t_}'] = 1.0

    # ---- 分布汇总 ----
    summ = []
    for t_ in TARGETS:
        pr = out[f'prob_{t_}']
        corr = np.corrcoef(text[t_].values, oof_pred[t_].reindex(train.index).values)[0, 1]
        summ.append((t_, round(float(pr.mean()), 3), round(float(pr[pr > 0.5].mean()), 3)
                     if (pr > 0.5).any() else 0.0,
                     round(float(out[f'weight_{t_}'].mean()), 3),
                     round(float((text[t_ + '__npos'] == 0).mean()), 3), round(corr, 3)))
    print('\n融合标签分布 (non-gold 行):')
    print(pd.DataFrame(summ, columns=['target', 'mean_prob', 'mean_pos_prob',
                                      'mean_weight', 'text_silence', 'r_text_oof'])
          .to_string(index=False))

    # ---- 保存 ----
    out.index.name = 'StudyInstanceUID'
    out.to_csv(ROOT / 'data/processed/v5_labels.csv')
    report.to_csv(ROOT / 'data/processed/v5_fusion_report.csv', index=False)
    print(f'\nsaved {ROOT / "data/processed/v5_labels.csv"} ({len(out)} rows)')
    print(f'saved {ROOT / "data/processed/v5_fusion_report.csv"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
