# ============================================================
# ★ LATERAL SWAP 融合 + 提交 (零训练, 纯算术)
#
# base = 3 seed rank-mean (逐类 rankdata 平均, 与 0.91 方案同构)
# rad_rank = rad 成员逐类 test 池内 rank
# swap  = base 全 12 类 + Lateral Meniscus / Lateral OA 换成 rad_rank
# blend = base 全 12 类 + 这两类 = 0.5·base + 0.5·rad_rank (保守版)
#
# 依据 (fusion_scan_v6.py V6 oracle, 58 gold):
#   Lateral Meniscus: base 0.7528 -> rad 0.8124 (+0.060)
#   Lateral OA:      base 0.8114 -> rad 0.8704 (+0.059)
#   → swap gold 宏期望 ≈ 0.9057 (base 0.8958 + (0.0596+0.0590)/12)
# 产物:
#   submission.csv                 = swap   (提交这个)
#   submission_base.csv            = base   (回退用, 同会话产物可同台对比)
#   submission_lateral_blend.csv   = blend  (保守中间档)
# ============================================================

print('\n--- Lateral swap fusion ---')

SWAP_TARGETS = ('Lateral Meniscus', 'Lateral OA')
SWAP_IDX = [TARGET_COLUMNS.index(c) for c in SWAP_TARGETS]


def rank_of(p):
    return rankdata(p, axis=0, method='average') / len(p)


def rank_mean(probs_list):
    """逐类 rankdata 平均 (rank 空间融合)。"""
    acc = np.zeros_like(probs_list[0], dtype=np.float64)
    for p in probs_list:
        acc += rank_of(p)
    return acc / len(probs_list)


def lateral_fuse(base, rad_probs):
    """返回 (swap, blend): 两类换成 rad_rank / 两类半混。"""
    rad_rank = rank_of(rad_probs)
    swap = base.copy()
    blend = base.copy()
    for j in SWAP_IDX:
        swap[:, j] = rad_rank[:, j]
        blend[:, j] = (base[:, j] + rad_rank[:, j]) / 2
    return swap, blend


seed_keys = sorted(k for k in member_gold_probs if isinstance(k, int))

# ---- gold 侧裁决 ----
base_gold = (rank_mean([member_gold_probs[s] for s in seed_keys])
             if len(seed_keys) > 1
             else rank_of(member_gold_probs[seed_keys[0]]))
swap_gold, blend_gold = lateral_fuse(base_gold, member_gold_probs['rad'])

base_aucs, base_macro = compute_gold_aucs(base_gold)
swap_aucs, swap_macro = compute_gold_aucs(swap_gold)
blend_aucs, blend_macro = compute_gold_aucs(blend_gold)

print(f'\n{"variant":<12s} {"Macro AUC (58 gold)":>20s}')
for s in seed_keys:
    print(f'{member_label(s):<12s} {member_aucs[s][1]:20.4f}')
if has_spec:
    print(f'{"spec":<12s} {member_aucs["spec"][1]:20.4f}  (仅诊断, 不融合)')
print(f'{"rad":<12s} {member_aucs["rad"][1]:20.4f}')
print(f'{"BASE":<12s} {base_macro:20.4f}  [对照: 本地融合实测 0.8958/0.8959]')
print(f'{"SWAP":<12s} {swap_macro:20.4f}  [期望 ≈0.9057]')
print(f'{"BLEND":<12s} {blend_macro:20.4f}')
print(f'\nLateral 两类 gold 对照 (base → swap):')
for c in SWAP_TARGETS:
    print(f'  {c:<18s} {base_aucs[c]:.4f} → {swap_aucs[c]:.4f} '
          f'({swap_aucs[c] - base_aucs[c]:+.4f})  '
          f'[融合扫描 oracle: +0.060 / +0.059]')

# 保存融合 gold 产物 (本地 fusion-scan 基础设施同格式)
def gold_rows_of(probs):
    rows = []
    for i, uid in enumerate(gold_studies):
        row = {'StudyInstanceUID': uid}
        for j, c in enumerate(TARGET_COLUMNS):
            row[f'true_{c}'] = int(gold_labels_arr[i, j])
            row[f'prob_{c}'] = float(probs[i, j])
        rows.append(row)
    return rows

pd.DataFrame(gold_rows_of(base_gold)).to_csv(
    output_dir / 'gold_validation_predictions_fused.csv', index=False)
pd.DataFrame(gold_rows_of(swap_gold)).to_csv(
    output_dir / 'gold_validation_predictions_lateral_swap.csv', index=False)
pd.DataFrame(gold_rows_of(blend_gold)).to_csv(
    output_dir / 'gold_validation_predictions_lateral_blend.csv', index=False)
auc_rows_f = [{'class': c, 'auc': base_aucs[c], 'n_pos': int(gold_labels_arr[:, i].sum())}
              for i, c in enumerate(TARGET_COLUMNS)]
pd.DataFrame(auc_rows_f + [{'class': 'macro_avg', 'auc': base_macro, 'n_pos': 0}]
             ).to_csv(output_dir / 'gold_validation_auc_fused.csv', index=False)

# ---- test 侧提交 ----
def make_submission(probs, fill_all):
    """probs [n_test, 12] → DataFrame; 缺行以 0.5 补齐。"""
    submission_rows = []
    if probs is not None:
        for row_idx, study_uid in enumerate(test_studies):
            row = {'StudyInstanceUID': study_uid}
            for j, c in enumerate(TARGET_COLUMNS):
                row[c] = float(probs[row_idx, j])
            submission_rows.append(row)
    else:
        print('No test DICOMs found — filling all studies with 0.5')
    submission_df = pd.DataFrame(submission_rows)
    full = test_df[['StudyInstanceUID']].merge(
        submission_df, on='StudyInstanceUID', how='left')
    for c in TARGET_COLUMNS:
        full[c] = full[c].fillna(0.5)
    return full

if member_test_probs.get('rad') is not None and all(
        member_test_probs.get(s) is not None for s in seed_keys):
    base_test = (rank_mean([member_test_probs[s] for s in seed_keys])
                 if len(seed_keys) > 1
                 else rank_of(member_test_probs[seed_keys[0]]))
    swap_test, blend_test = lateral_fuse(base_test, member_test_probs['rad'])

    make_submission(swap_test, test_df).to_csv(
        output_dir / 'submission.csv', index=False)
    make_submission(base_test, test_df).to_csv(
        output_dir / 'submission_base.csv', index=False)
    make_submission(blend_test, test_df).to_csv(
        output_dir / 'submission_lateral_blend.csv', index=False)

    swap_sub = make_submission(swap_test, test_df)
    print(f'\nSubmission saved: {output_dir / "submission.csv"}  (★ LATERAL SWAP — 提交这个)')
    print(f'  {output_dir / "submission_base.csv"}  (回退)')
    print(f'  {output_dir / "submission_lateral_blend.csv"}  (保守)')
    print(f'  Studies: {len(swap_sub)} (expected: {len(test_df)})')
    for c in TARGET_COLUMNS:
        vals = swap_sub[c].values
        flag = ' <== rad_rank' if c in SWAP_TARGETS else ''
        print(f'  {c:<20s}: mean={vals.mean():.4f}, std={vals.std():.4f}, '
              f'>0.5={np.mean(vals > 0.5):.1%}{flag}')
else:
    print('\nWARNING: 成员 test 推理不完整 (test DICOM 缺失?), 未生成 submission。')

print(f'\nDone!')
print(f'  Members: {len(seed_keys)} seeds'
      + (' + spec (archived, not fused)' if has_spec else '')
      + ' + rad')
print(f'  Gold: BASE {base_macro:.4f} | SWAP {swap_macro:.4f} | BLEND {blend_macro:.4f}')
print(f'  裁决提示: LB 在非公开 test 集评分; 先投 swap, 掉分就回退 base。')
