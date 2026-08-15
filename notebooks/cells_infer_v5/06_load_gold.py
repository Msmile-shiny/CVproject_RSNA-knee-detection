# ============================================================
# 加载 gold (58 全标注研究) — 仅用于验证, 推理 notebook 无需 v5 标签数据集
# ============================================================

comp_input = Path(CFG['comp_input'])

# ---- 竞赛元数据 ----
train_meta = pd.read_csv(comp_input / 'train.csv')
train_meta['StudyInstanceUID'] = train_meta['StudyInstanceUID'].astype(str)

# ---- 58 gold (12 类全标注) ----
label_cols_present = [c for c in TARGET_COLUMNS if c in train_meta.columns]
has_all_labels = train_meta[label_cols_present].notna().all(axis=1)
gold_df = train_meta[has_all_labels].copy()
gold_studies = sorted(gold_df['StudyInstanceUID'].unique())

gold_labels = gold_df[['StudyInstanceUID'] + label_cols_present].copy()
gold_labels = gold_labels.set_index('StudyInstanceUID')
for c in TARGET_COLUMNS:
    if c not in gold_labels.columns:
        gold_labels[c] = np.nan
gold_labels = gold_labels.apply(pd.to_numeric, errors='coerce')

n_pos_per_class = (gold_labels > 0).sum(axis=0)

# ---- train_series 元数据 (只保留 gold 研究, 供 slot 匹配) ----
series_meta = pd.read_csv(comp_input / 'train_series.csv')
series_meta['StudyInstanceUID'] = series_meta['StudyInstanceUID'].astype(str)
series_meta['SeriesInstanceUID'] = series_meta['SeriesInstanceUID'].astype(str)
gold_series_meta = series_meta[
    series_meta['StudyInstanceUID'].isin(set(gold_studies))].copy()

if IS_MAIN:
    print(f'Gold studies (all 12 labeled): {len(gold_studies)} (expect 58)')
    print(f'Gold positives per class: min={int(n_pos_per_class.min())}, '
          f'max={int(n_pos_per_class.max())}, mean={n_pos_per_class.mean():.1f}')
    print(f'Gold series meta rows: {len(gold_series_meta)}')
