# ============================================================
# v5spec: Load metadata + 词表软标签 (报告规则提取器) + Gold 全部→验证
# ============================================================

comp_input = Path(CFG['comp_input'])
label_input = Path(CFG['label_input'])

# ---- Load competition metadata ----
train_meta = pd.read_csv(comp_input / 'train.csv')
train_meta['StudyInstanceUID'] = train_meta['StudyInstanceUID'].astype(str)
series_meta = pd.read_csv(comp_input / 'train_series.csv')
series_meta['StudyInstanceUID'] = series_meta['StudyInstanceUID'].astype(str)
series_meta['SeriesInstanceUID'] = series_meta['SeriesInstanceUID'].astype(str)

# ---- Split gold vs unlabeled ----
label_cols_present = [c for c in TARGET_COLUMNS if c in train_meta.columns]
has_all_labels = train_meta[label_cols_present].notna().all(axis=1)
gold_df = train_meta[has_all_labels].copy()
unlabeled_df = train_meta[~has_all_labels].copy()

# ★ v5: 全部 gold → 验证 (与 v4 相同, 保证与 v4 0.833 可比)
gold_studies = sorted(gold_df['StudyInstanceUID'].unique())
unlabeled_studies = sorted(unlabeled_df['StudyInstanceUID'].unique())

val_gold_uids = set(gold_studies)       # ★ 全部 gold → val
train_gold_uids = set()                 # ★ 训练不使用 gold

if IS_MAIN:
    print(f'Gold studies (all 12 labeled): {len(gold_studies)}')
    print(f'Unlabeled studies: {len(unlabeled_studies)}')
    print(f'★ v5 split: Train={len(train_gold_uids)} gold + all fused, Val={len(val_gold_uids)} gold')

# ---- Gold labels (val only) ----
gold_labels = gold_df[['StudyInstanceUID'] + label_cols_present].copy()
gold_labels = gold_labels.set_index('StudyInstanceUID')
for c in TARGET_COLUMNS:
    if c not in gold_labels.columns:
        gold_labels[c] = np.nan
gold_labels = gold_labels.apply(pd.to_numeric, errors='coerce')
n_pos_per_class = (gold_labels > 0).sum(axis=0)
if IS_MAIN:
    print(f'Gold positives per class: min={int(n_pos_per_class.min())}, '
          f'max={int(n_pos_per_class.max())}, mean={n_pos_per_class.mean():.1f}')

# ---- ★ spec 词表软标签 (本地 scripts/report_extractor_v2.py 生成) ----
# 每 target 三个字段 (规则提取器, 无训练/无标签泄漏):
#   score: 0.04~0.97 有序分级 (0.5+ = 阳性证据, 0.04-0.2 = 断言阴性, 0.28 = 报告沉默)
#   conf:  证据置信度 (阳性 0.55+0.15·n_pos, 阴性 0.45+0.12·n_neg, 沉默 0.05)
#   npos/nneg: 正/负提及数 (诊断用)
# 标签协议: prob = clip(score, 0.01, 0.99); weight = max(conf, 0.5)
#   沉默行保底权重 0.5——"报告未提及"对排序仍是有信息的弱负证据,
#   但权重低于明确证据 (0.55-1.0), 与 v5 融合标签的权重思路一致
spec_label_file = label_input / 'report_labels_v2.csv'
if not spec_label_file.exists():
    raise FileNotFoundError(
        f'spec labels not found: {spec_label_file} — '
        f'upload data/processed/report_labels_v2.csv as a new version of the '
        f'v5 labels Kaggle Dataset and mount it; or set CFG["label_input"]')

spec_df = pd.read_csv(spec_label_file)
spec_df['StudyInstanceUID'] = spec_df['StudyInstanceUID'].astype(str)

spec_labels = spec_df[['StudyInstanceUID']].copy()
for c in TARGET_COLUMNS:
    spec_labels[f'prob_{c}'] = spec_df[c].clip(0.01, 0.99)  # 词表 score → 软标签
for c in TARGET_COLUMNS:
    spec_labels[f'weight_{c}'] = spec_df[f'{c}__conf'].clip(lower=0.5)
for c in MASK_COLS:
    spec_labels[c] = 1.0
spec_labels = spec_labels.set_index('StudyInstanceUID')
for c in PROB_COLS + WEIGHT_COLS + MASK_COLS:
    spec_labels[c] = pd.to_numeric(spec_labels[c], errors='coerce').fillna(
        0.5 if 'prob' in c else 0.1).astype(np.float32)

if IS_MAIN:
    n_missing = len(set(unlabeled_studies) - set(spec_labels.index))
    print(f'spec labels: {len(spec_labels):,} rows; missing for unlabeled: {n_missing}')

# ---- Train labels (词表软标签 for all unlabeled studies; gold 不进训练) ----
all_train_rows = []
for uid in unlabeled_studies:
    if uid not in spec_labels.index:
        continue
    row = {'StudyInstanceUID': uid}
    for c in TARGET_COLUMNS:
        row[c] = 0.0
    for c in PROB_COLS:
        row[c] = float(spec_labels.loc[uid, c])
    for c in WEIGHT_COLS:
        row[c] = float(spec_labels.loc[uid, c])
    for c in MASK_COLS:
        row[c] = float(spec_labels.loc[uid, c])
    row['is_gold'] = False
    all_train_rows.append(row)

train_labels = pd.DataFrame(all_train_rows).set_index('StudyInstanceUID')

# ---- Val labels (gold only) ----
val_rows = []
for uid in val_gold_uids:
    if uid not in gold_labels.index:
        continue
    row = {'StudyInstanceUID': uid}
    for c in TARGET_COLUMNS:
        raw = gold_labels.loc[uid, c]
        is_labeled = not pd.isna(raw)
        row[c] = float(raw) if is_labeled else 0.0
        row[f'mask_{c}'] = 1.0 if is_labeled else 0.0
    row['is_gold'] = True
    val_rows.append(row)
val_labels = pd.DataFrame(val_rows).set_index('StudyInstanceUID')

if IS_MAIN:
    n_train = len(train_labels)
    print(f'\nTrain: {n_train:,} studies (词表软标签)')
    print(f'Val:   {len(val_labels):,} studies (all gold-labeled)')
    val_labeled = val_labels[MASK_COLS].sum(axis=0) if len(val_labels) > 0 else pd.Series(0, index=MASK_COLS)
    print(f'  Labeled per class: min={int(val_labeled.min())}, mean={val_labeled.mean():.1f}')

    # 标签分布 (正类率 / 平均权重) — 对照 scripts/text_blend_probe.py 的口径
    dist = pd.DataFrame({
        'pos_rate': [(train_labels[f'prob_{c}'] > 0.5).mean() for c in TARGET_COLUMNS],
        'mean_prob': [train_labels[f'prob_{c}'].mean() for c in TARGET_COLUMNS],
        'mean_weight': [train_labels[f'weight_{c}'].mean() for c in TARGET_COLUMNS],
    }, index=TARGET_COLUMNS).round(3)
    print(dist.to_string())

    # ★ 词表自身在 gold 上的 AUC = 专家蒸馏上限参考
    #   探针 (本地): ACL 0.953 / MCL 0.964 / LM 0.846 / LOA 0.839 (词表强类)
    #   Synovitis 0.709 / Effusion 0.777 (词表弱类, 靠训练时低权重自然弱化)
    try:
        from sklearn.metrics import roc_auc_score as _auc
        rows = []
        for c in TARGET_COLUMNS:
            y = gold_labels[c].dropna()
            if len(y) > 0 and len(set(y)) > 1:
                p = spec_labels[f'prob_{c}'].reindex(y.index)
                rows.append((c, round(_auc(y, p), 3), int(y.sum())))
        print('\n  Lexicon gold AUC (蒸馏上限, n=pos):')
        print(pd.DataFrame(rows, columns=['target', 'lexicon_auc', 'n_pos']).to_string(index=False))
    except Exception as _e:
        print(f'  (lexicon AUC 对照跳过: {_e})')
