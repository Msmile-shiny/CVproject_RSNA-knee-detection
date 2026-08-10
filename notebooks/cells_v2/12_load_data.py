comp_input = Path(CFG['comp_input'])
pseudo_input = Path(CFG['pseudo_input'])

# Load competition metadata
train_meta = pd.read_csv(comp_input / 'train.csv')
series_meta = pd.read_csv(comp_input / 'train_series.csv')

# Split gold vs unlabeled
label_cols_present = [c for c in TARGET_COLUMNS if c in train_meta.columns]
has_gold = train_meta[label_cols_present].notna().all(axis=1)
gold_df = train_meta[has_gold].copy()

if IS_MAIN:
    print(f'Gold studies: {len(gold_df)}')
    print(f'Total studies: {len(train_meta)}')

# -- v2: Load CALIBRATED pseudo-labels ---------------------------
# pseudo_labels_calibrated.csv columns:
#   StudyInstanceUID | pred_* (12) | conf_* (12) | prob_* (12) | weight_* (12) | mask_* (12)
calibrated_df = pd.read_csv(pseudo_input / 'pseudo_labels_calibrated.csv')

if IS_MAIN:
    print(f'Calibrated pseudo-labels: {len(calibrated_df):,} studies')
    print(f'  Columns ({len(calibrated_df.columns)}): {list(calibrated_df.columns)[:5]}...')

    # -- Quick stats on calibration -----------------------------
    print(f'\n  Calibration summary per class:')
    print(f'  {"Class":<20s} {"Mean Prob":>9s} {"Mean Wgt":>9s} {"%Masked":>8s}')
    print(f'  {"-"*20} {"-"*9} {"-"*9} {"-"*8}')
    for c in TARGET_COLUMNS:
        p_mean = calibrated_df[f'prob_{c}'].mean()
        w_mean = calibrated_df[f'weight_{c}'].mean()
        m_pct = calibrated_df[f'mask_{c}'].mean() * 100
        print(f'  {c:<20s} {p_mean:9.4f} {w_mean:9.4f} {m_pct:7.1f}%')

# Build soft-label DataFrame (for training)
train_labels = calibrated_df[['StudyInstanceUID']].copy()
for c in PROB_COLS:
    train_labels[c] = calibrated_df[c]
for c in WEIGHT_COLS:
    train_labels[c] = calibrated_df[c]
for c in MASK_COLS:
    train_labels[c] = calibrated_df[c]
train_labels = train_labels.set_index('StudyInstanceUID')

# Ensure float32
for c in PROB_COLS + WEIGHT_COLS + MASK_COLS:
    train_labels[c] = pd.to_numeric(train_labels[c], errors='coerce').fillna(0.5 if 'prob' in c else 0.1).astype(np.float32)

# Build hard-label DataFrame (for validation)
val_labels = gold_df[['StudyInstanceUID'] + label_cols_present].copy()
val_labels = val_labels.set_index('StudyInstanceUID')
for c in TARGET_COLUMNS:
    if c not in val_labels.columns:
        val_labels[c] = 0.0
val_labels = val_labels.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)

if IS_MAIN:
    print(f'\nTrain studies (soft labels): {len(train_labels):,}')
    print(f'Val studies (gold labels):   {len(val_labels):,}')
    print(f'\n  v2: No row-level confidence filtering -- using per-class weights instead')
    print(f'  v1 used HIGH-only filtering which discarded ~50% of studies')
    print(f'  v2 keeps ALL studies and uses weight_*/mask_* to handle uncertainty')
