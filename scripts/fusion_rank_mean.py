# -*- coding: utf-8 -*-
"""三成员 seed 集成 rank-mean 融合（58 gold 评估 + 测试预测合并）。

用法:
    python scripts/fusion_rank_mean.py --members v5s1 v5s2 v5s3 --out-dir results/fusion_v5s1s2s3

每个成员目录需要:
    gold_validation_predictions_s{seed}.csv  格式: StudyInstanceUID, true_<t>, prob_<t> x12
    submission.csv                          格式: StudyInstanceUID + 12 预测列

输出:
    <out>/gold_fusion_eval.txt              逐类 AUC + 宏 AUC（rank-mean vs prob-mean vs 各成员）
    <out>/gold_predictions_fused.csv        融合后 58 gold 预测
    <out>/submission_fused.csv              融合后测试预测（UID 顺序与成员一致）
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
           'Contusion', 'Fracture']


def load_member(member_dir):
    """读一个成员目录，返回 (gold_prob_df, gold_true_df, test_df)。"""
    gold_path = None
    for f in sorted(os.listdir(member_dir)):
        if f.startswith('gold_validation_predictions_') and f.endswith('.csv'):
            gold_path = os.path.join(member_dir, f)
            break
    if gold_path is None:
        raise FileNotFoundError(f'{member_dir}: 找不到 gold_validation_predictions_*.csv')
    gold = pd.read_csv(gold_path)
    test = pd.read_csv(os.path.join(member_dir, 'submission.csv'))
    uid_col = gold.columns[0]
    if uid_col != 'StudyInstanceUID':
        gold = gold.rename(columns={uid_col: 'StudyInstanceUID'})
        test = test.rename(columns={test.columns[0]: 'StudyInstanceUID'})
    assert gold['StudyInstanceUID'].is_unique, f'{member_dir}: gold UID 有重复'
    true_cols = [f'true_{t}' for t in TARGETS]
    prob_cols = [f'prob_{t}' for t in TARGETS]
    missing = [c for c in true_cols + prob_cols if c not in gold.columns]
    assert not missing, f'{member_dir}: 缺列 {missing[:5]}'
    gold_true = gold[['StudyInstanceUID'] + true_cols].rename(
        columns={f'true_{t}': t for t in TARGETS})
    gold_prob = gold[['StudyInstanceUID'] + prob_cols].rename(
        columns={f'prob_{t}': t for t in TARGETS})
    assert all(c in test.columns for c in TARGETS), f'{member_dir}: submission 缺目标列'
    return gold_prob, gold_true, test[['StudyInstanceUID'] + TARGETS]


def macro_auc(prob_df, true_df):
    aucs = {}
    for t in TARGETS:
        m = prob_df[['StudyInstanceUID', t]].merge(
            true_df[['StudyInstanceUID', t]], on='StudyInstanceUID', suffixes=('_p', '_t'))
        y = m[f'{t}_t'].values
        p = m[f'{t}_p'].values
        if len(np.unique(y)) < 2:
            aucs[t] = np.nan
        else:
            aucs[t] = roc_auc_score(y, p)
    valid = [a for a in aucs.values() if not np.isnan(a)]
    return aucs, float(np.mean(valid))


def rank_mean(dfs, targets):
    acc = None
    for df in dfs:
        x = rankdata(df[targets].values, axis=0, method='average') / len(df)
        acc = x if acc is None else acc + x
    return acc / len(dfs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--members', nargs='+', required=True,
                    help='成员目录（相对 results/ 或绝对路径）')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    results_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    members = [m if os.path.isabs(m) else os.path.join(results_root, 'results', m)
               for m in args.members]
    out_dir = os.path.join(results_root, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    golds_prob, golds_true, tests, names = [], [], [], []
    for mdir in members:
        gp, gt, t = load_member(mdir)
        names.append(os.path.basename(mdir.rstrip('/\\')))
        if golds_prob:
            assert list(gp['StudyInstanceUID']) == list(golds_prob[0]['StudyInstanceUID']), \
                f'{names[-1]} gold UID 顺序与其他成员不一致'
            assert list(gt['StudyInstanceUID']) == list(golds_true[0]['StudyInstanceUID'])
            assert (gt[TARGETS].values == golds_true[0][TARGETS].values).all(), \
                f'{names[-1]} 真值列与其他成员不一致'
        golds_prob.append(gp)
        golds_true.append(gt)
        tests.append(t)

    lines = []
    def log(s=''):
        lines.append(s)
        print(s)

    log('=' * 70)
    log(f'成员: {", ".join(names)} | gold 研究数: {len(golds_prob[0])}')
    log('=' * 70)

    # ---- 各成员局部 AUC 复算（与 Kaggle 报告交叉验证） ----
    member_macro = {}
    for name, gp, gt in zip(names, golds_prob, golds_true):
        aucs, m = macro_auc(gp, gt)
        member_macro[name] = m
        worst = min(aucs, key=aucs.get)
        log(f'{name}: 宏 AUC {m:.4f}（最弱 {worst} {aucs[worst]:.4f}）')

    # ---- 成员间相关 ----
    log('\n--- 成员间 Pearson 相关（逐类平均） ---')
    for i in range(len(golds_prob)):
        for j in range(i + 1, len(golds_prob)):
            corrs = [np.corrcoef(golds_prob[i][t], golds_prob[j][t])[0, 1] for t in TARGETS]
            log(f'{names[i]} vs {names[j]}: mean r = {np.mean(corrs):.3f} '
                f'(min {np.min(corrs):.3f} @ {TARGETS[int(np.argmin(corrs))]}, '
                f'max {np.max(corrs):.3f} @ {TARGETS[int(np.argmax(corrs))]})')

    # ---- 融合 ----
    fused = {}
    rk = rank_mean(golds_prob, TARGETS)
    fused['rank'] = pd.DataFrame(rk, columns=TARGETS).assign(
        StudyInstanceUID=golds_prob[0]['StudyInstanceUID'])
    pb = sum(gp[TARGETS].values for gp in golds_prob) / len(golds_prob)
    fused['prob'] = pd.DataFrame(pb, columns=TARGETS).assign(
        StudyInstanceUID=golds_prob[0]['StudyInstanceUID'])

    log('\n--- 融合评估（58 gold） ---')
    for kind, fd in fused.items():
        aucs, m = macro_auc(fd, golds_true[0])
        log(f'{kind}-mean 融合: 宏 AUC {m:.4f}')
    fused['rank'].to_csv(os.path.join(out_dir, 'gold_predictions_fused.csv'), index=False)

    # ---- 与最佳单成员逐类对比 ----
    log('\n--- rank 融合 vs 最佳单成员（逐类差，58 研究上均为噪声量级） ---')
    best_name = max(member_macro, key=member_macro.get)
    bi = names.index(best_name)
    aucs_f, m_f = macro_auc(fused['rank'], golds_true[0])
    aucs_b, _ = macro_auc(golds_prob[bi], golds_true[0])
    log(f'基准 = {best_name} ({member_macro[best_name]:.4f})')
    for t in sorted(TARGETS, key=lambda t: aucs_f[t] - aucs_b[t]):
        log(f'  {t:20s} {aucs_f[t] - aucs_b[t]:+.4f}')
    log(f'  宏 Δ = {m_f - member_macro[best_name]:+.4f}')

    # ---- 测试预测融合 ----
    log('\n--- 测试集融合 ---')
    for i, t in enumerate(tests):
        assert list(t['StudyInstanceUID']) == list(tests[0]['StudyInstanceUID']), \
            f'{names[i]} test UID 顺序不一致'
        assert len(t) == len(tests[0]), f'{names[i]} test 研究数不一致'
    t_rk = rank_mean(tests, TARGETS)
    t_fused = pd.DataFrame(t_rk, columns=TARGETS).assign(
        StudyInstanceUID=tests[0]['StudyInstanceUID'])
    t_fused.to_csv(os.path.join(out_dir, 'submission_fused.csv'), index=False)
    log(f'融合测试预测 {len(t_fused)} 研究 → {os.path.join(out_dir, "submission_fused.csv")}')
    log('  均值: ' + ', '.join(f'{t} {t_fused[t].mean():.3f}' for t in TARGETS))

    with open(os.path.join(out_dir, 'gold_fusion_eval.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\n评估报告已写入 {os.path.join(out_dir, "gold_fusion_eval.txt")}')


if __name__ == '__main__':
    main()
