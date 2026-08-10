# ============================================================
# v3: Load metadata + pseudo labels
# ============================================================

comp_input = Path(CFG['comp_input'])
pseudo_input = Path(CFG['pseudo_input'])

# Load competition metadata
train_meta = pd.read_csv(comp_input / 'train.csv')
train_meta['StudyInstanceUID'] = train_meta['StudyInstanceUID'].astype(str)
series_meta = pd.read_csv(comp_input / 'train_series.csv')
series_meta['StudyInstanceUID'] = series_meta['StudyInstanceUID'].astype(str)
series_meta['SeriesInstanceUID'] = series_meta['SeriesInstanceUID'].astype(str)

# Split gold vs unlabeled
# Use .any(axis=1) instead of .all(axis=1): a study is "gold" if it has
# at least one human-annotated label. Unlabeled targets are masked out
# (mask=0) in training loss and validation AUC.
label_cols_present = [c for c in TARGET_COLUMNS if c in train_meta.columns]
has_any_label = train_meta[label_cols_present].notna().any(axis=1)
gold_df = train_meta[has_any_label].copy()
unlabeled_df = train_meta[~has_any_label].copy()

# Split gold into train/val (80/20)
gold_studies = gold_df['StudyInstanceUID'].values
np.random.seed(42)
np.random.shuffle(gold_studies)
n_val = int(len(gold_studies) * 0.2)
val_gold_uids = set(gold_studies[:n_val])
train_gold_uids = set(gold_studies[n_val:])

if IS_MAIN:
    n_labeled_per_study = gold_df[label_cols_present].notna().sum(axis=1)
    print(f'Gold studies (any label): {len(gold_df)} '
          f'(avg {n_labeled_per_study.mean():.1f} labels/study)')
    print(f'  Train gold: {len(train_gold_uids)} | Val gold: {len(val_gold_uids)}')
    print(f'Unlabeled studies: {len(unlabeled_df)}')

# Load calibrated pseudo-labels
calibrated_df = pd.read_csv(pseudo_input / 'pseudo_labels_calibrated.csv')
calibrated_df['StudyInstanceUID'] = calibrated_df['StudyInstanceUID'].astype(str)

# Build soft-label DataFrame (for pseudo-labeled studies, used in training)
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

# Build hard-label DataFrame (for gold studies, used in training + validation)
# Keep NaN as NaN — we use them to build per-target masks.
# "Labeled negative" (0.0) and "unlabeled" (NaN) are different things.
gold_labels = gold_df[['StudyInstanceUID'] + label_cols_present].copy()
gold_labels = gold_labels.set_index('StudyInstanceUID')
for c in TARGET_COLUMNS:
    if c not in gold_labels.columns:
        gold_labels[c] = np.nan
gold_labels = gold_labels.apply(pd.to_numeric, errors='coerce')

# Training set:
#   - Gold studies in train split → hard labels
#   - All unlabeled studies → soft labels
# Combine into one labels DataFrame with both hard and soft columns
# We'll distinguish in the dataset via a flag

# For training: use train_gold_uids (hard) + unlabeled (soft)
train_study_uids = list(train_gold_uids) + list(unlabeled_df['StudyInstanceUID'].unique())

# For validation: use val_gold_uids (hard labels only)
val_study_uids = list(val_gold_uids)

# Training labels: gold rows get hard labels + dummy soft cols; pseudo rows get soft labels
# Build unified labels_df
all_train_rows = []
for uid in train_gold_uids:
    if uid not in gold_labels.index:
        continue
    row = {'StudyInstanceUID': uid}
    # Gold labels → "soft" format with per-target mask
    #   labeled target:   prob=hard_label, weight=1.0, mask=1.0
    #   unlabeled target: prob=0.5, weight=0.0, mask=0.0 (skipped in loss)
    for c in TARGET_COLUMNS:
        raw = gold_labels.loc[uid, c]
        is_labeled = not pd.isna(raw)
        hard_val = float(raw) if is_labeled else 0.0
        row[c] = hard_val
        row[f'prob_{c}'] = max(0.01, min(0.99, hard_val)) if is_labeled else 0.5
        row[f'weight_{c}'] = 1.0 if is_labeled else 0.0
        row[f'mask_{c}'] = 1.0 if is_labeled else 0.0
    row['is_gold'] = True
    all_train_rows.append(row)

for uid in unlabeled_df['StudyInstanceUID'].unique():
    if uid not in pseudo_labels.index:
        continue
    row = {'StudyInstanceUID': uid}
    for c in TARGET_COLUMNS:
        row[c] = 0.0  # dummy, not used
    for c in PROB_COLS:
        row[c] = float(pseudo_labels.loc[uid, c])
    for c in WEIGHT_COLS:
        row[c] = float(pseudo_labels.loc[uid, c])
    for c in MASK_COLS:
        row[c] = float(pseudo_labels.loc[uid, c])
    row['is_gold'] = False
    all_train_rows.append(row)

train_labels = pd.DataFrame(all_train_rows).set_index('StudyInstanceUID')
# Build val_labels with mask columns (same format as train, so dataset can
# return val_masks for partial-label validation).
val_labels_raw = gold_labels[gold_labels.index.isin(val_gold_uids)].copy()
val_rows = []
for uid in val_labels_raw.index:
    row = {'StudyInstanceUID': uid}
    for c in TARGET_COLUMNS:
        raw = val_labels_raw.loc[uid, c]
        is_labeled = not pd.isna(raw)
        row[c] = float(raw) if is_labeled else 0.0
        row[f'prob_{c}'] = 0.0   # unused in val, placeholder
        row[f'weight_{c}'] = 0.0  # unused in val, placeholder
        row[f'mask_{c}'] = 1.0 if is_labeled else 0.0
    row['is_gold'] = True
    val_rows.append(row)
val_labels = pd.DataFrame(val_rows).set_index('StudyInstanceUID')

# Training subset limit (for faster iteration)
if CFG['train_studies_limit'] and CFG['train_studies_limit'] < len(train_labels):
    subset_uids = np.random.choice(
        train_labels.index.values, CFG['train_studies_limit'], replace=False)
    train_labels = train_labels.loc[subset_uids]
    if IS_MAIN:
        print(f'Training subset: {len(train_labels)} studies (limit={CFG["train_studies_limit"]})')

if IS_MAIN:
    n_gold_train = train_labels['is_gold'].sum() if 'is_gold' in train_labels.columns else 0
    train_gold_masks = train_labels[train_labels['is_gold']][MASK_COLS].values if n_gold_train > 0 else np.zeros((0, 12))
    train_labeled_pct = train_gold_masks.mean() * 100 if n_gold_train > 0 else 0
    print(f'\nTrain studies: {len(train_labels):,}  '
          f'(gold={n_gold_train}, pseudo={len(train_labels) - n_gold_train})')
    if n_gold_train > 0:
        print(f'  Gold train label coverage: {train_labeled_pct:.0f}% of targets labeled (partial labels)')

    val_masks = val_labels[MASK_COLS].values
    val_labeled_per_class = val_masks.sum(axis=0)
    print(f'Val studies:   {len(val_labels):,}')
    print(f'  Labeled per class: min={int(val_labeled_per_class.min())}, '
          f'max={int(val_labeled_per_class.max())}, '
          f'mean={val_labeled_per_class.mean():.1f}')
