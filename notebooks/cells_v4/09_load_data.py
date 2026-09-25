# ============================================================
# v4: Load metadata + 伪标签自蒸馏 + Gold 全部→验证
# ============================================================

comp_input = Path(CFG['comp_input'])
pseudo_input = Path(CFG['pseudo_input'])

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

# ★ v4: 全部 gold → 验证
gold_studies = sorted(gold_df['StudyInstanceUID'].unique())
unlabeled_studies = sorted(unlabeled_df['StudyInstanceUID'].unique())

val_gold_uids = set(gold_studies)       # ★ 全部 gold → val
train_gold_uids = set()                 # ★ 训练不使用 gold

if IS_MAIN:
    print(f'Gold studies (all 12 labeled): {len(gold_studies)}')
    print(f'Unlabeled studies: {len(unlabeled_studies)}')
    print(f'★ v4 split: Train={len(train_gold_uids)} gold + all pseudo, Val={len(val_gold_uids)} gold')

# ---- Gold labels ----
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

# ---- ★ 伪标签自蒸馏 ----
# 如果开启 self-distill，用 v3 checkpoint 给全量无标注数据重新打标签
# 否则使用 v2 的旧伪标签
v3_ckpt = Path('/kaggle/input/rsna-knee-v3-checkpoint/best_model.pt')
use_distill = CFG.get('use_self_distill', False) and v3_ckpt.exists()

if use_distill and IS_MAIN:
    print('\n★ Self-distillation: re-labeling unlabeled studies with v3 checkpoint...')
    # 延迟加载 v3 model（在后续 cell 中实现）
    print('  (will be done in cache/validation cell)')

# Load calibrated pseudo-labels
calibrated_df = pd.read_csv(pseudo_input / 'pseudo_labels_calibrated.csv')
calibrated_df['StudyInstanceUID'] = calibrated_df['StudyInstanceUID'].astype(str)

pseudo_labels = calibrated_df[['StudyInstanceUID']].copy()
for c in PROB_COLS:
    pseudo_labels[c] = calibrated_df[c]
for c in WEIGHT_COLS:
    pseudo_labels[c] = calibrated_df[c]
for c in MASK_COLS:
    pseudo_labels[c] = calibrated_df[c]
pseudo_labels = pseudo_labels.set_index('StudyInstanceUID')
for c in PROB_COLS + WEIGHT_COLS + MASK_COLS:
    pseudo_labels[c] = pd.to_numeric(pseudo_labels[c], errors='coerce').fillna(
        0.5 if 'prob' in c else 0.1).astype(np.float32)

# ---- Train labels (pseudo only) ----
all_train_rows = []
for uid in unlabeled_studies:
    if uid not in pseudo_labels.index:
        continue
    row = {'StudyInstanceUID': uid}
    for c in TARGET_COLUMNS:
        row[c] = 0.0
    for c in PROB_COLS:
        row[c] = float(pseudo_labels.loc[uid, c])
    for c in WEIGHT_COLS:
        row[c] = float(pseudo_labels.loc[uid, c])
    for c in MASK_COLS:
        row[c] = float(pseudo_labels.loc[uid, c])
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
    n_pseudo = len(train_labels)
    print(f'\nTrain: {n_pseudo:,} studies (all pseudo-labeled)')
    print(f'Val:   {len(val_labels):,} studies (all gold-labeled)')
    val_labeled = val_labels[MASK_COLS].sum(axis=0) if len(val_labels) > 0 else pd.Series(0, index=MASK_COLS)
    print(f'  Labeled per class: min={int(val_labeled.min())}, mean={val_labeled.mean():.1f}')
