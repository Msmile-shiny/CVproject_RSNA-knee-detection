# ============================================================
# v5: Load metadata + v5 融合软标签 (teacher-student) + Gold 全部→验证
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

# ---- ★ v5 融合软标签 (本地 scripts/build_v5_labels.py 生成) ----
# prob_*:  融合概率 = per-finding 逻辑回归 (gold 上拟合) 混合
#          提取器 score (text teacher) × 公开 20 成员集成 OOF (image teacher)
# weight_*: 置信度权重 = (0.35+0.65·conf) × text/oof 一致性, gold 行 = 1.0
# mask_*:  1.0 (软标签全参与, 权重即置信度)
v5_label_file = label_input / 'v5_labels.csv'
if not v5_label_file.exists():
    raise FileNotFoundError(
        f'v5 labels not found: {v5_label_file} — '
        f'upload data/processed/v5_labels.csv as a Kaggle Dataset '
        f'(root dir) and mount it; or set CFG["label_input"]')

fused_df = pd.read_csv(v5_label_file)
fused_df['StudyInstanceUID'] = fused_df['StudyInstanceUID'].astype(str)

fused_labels = fused_df[['StudyInstanceUID']].copy()
for c in PROB_COLS:
    fused_labels[c] = fused_df[c]
for c in WEIGHT_COLS:
    fused_labels[c] = fused_df[c]
for c in MASK_COLS:
    fused_labels[c] = fused_df[c]
fused_labels = fused_labels.set_index('StudyInstanceUID')
for c in PROB_COLS + WEIGHT_COLS + MASK_COLS:
    fused_labels[c] = pd.to_numeric(fused_labels[c], errors='coerce').fillna(
        0.5 if 'prob' in c else 0.1).astype(np.float32)

if IS_MAIN:
    n_missing = len(set(unlabeled_studies) - set(fused_labels.index))
    print(f'v5 labels: {len(fused_labels):,} rows; missing for unlabeled: {n_missing}')

# ---- Train labels (fused soft labels for all unlabeled studies) ----
all_train_rows = []
for uid in unlabeled_studies:
    if uid not in fused_labels.index:
        continue
    row = {'StudyInstanceUID': uid}
    for c in TARGET_COLUMNS:
        row[c] = 0.0
    for c in PROB_COLS:
        row[c] = float(fused_labels.loc[uid, c])
    for c in WEIGHT_COLS:
        row[c] = float(fused_labels.loc[uid, c])
    for c in MASK_COLS:
        row[c] = float(fused_labels.loc[uid, c])
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
    print(f'\nTrain: {n_train:,} studies (v5 fused soft labels)')
    print(f'Val:   {len(val_labels):,} studies (all gold-labeled)')
    val_labeled = val_labels[MASK_COLS].sum(axis=0) if len(val_labels) > 0 else pd.Series(0, index=MASK_COLS)
    print(f'  Labeled per class: min={int(val_labeled.min())}, mean={val_labeled.mean():.1f}')

    # 标签分布 (正类率 / 平均权重) — 与本地融合报告对照
    dist = pd.DataFrame({
        'pos_rate': [(train_labels[f'prob_{c}'] > 0.5).mean() for c in TARGET_COLUMNS],
        'mean_prob': [train_labels[f'prob_{c}'].mean() for c in TARGET_COLUMNS],
        'mean_weight': [train_labels[f'weight_{c}'].mean() for c in TARGET_COLUMNS],
    }, index=TARGET_COLUMNS).round(3)
    print(dist.to_string())
