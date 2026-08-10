# ============================================================
# v4: Test Set Inference + Submission.csv + Gold Validation
# ============================================================
#
# 1. 在 gold 验证集上做完整 TTA + 诊断池化，计算真实 AUC
# 2. 在 test 集上推理，生成 submission.csv
# ============================================================

print('=' * 60)
print('GOLD VALIDATION + TEST INFERENCE')
print('=' * 60)

# ---- 加载最佳 checkpoint ----
best_ckpt_path = output_dir / 'checkpoints' / 'best_model.pt'
if not best_ckpt_path.exists():
    raise FileNotFoundError(
        f'Best model checkpoint not found: {best_ckpt_path}\n'
        'Training must produce best_model.pt (best validation AUC). '
        'Check that at least one epoch completed and saved a best model.')

print(f'Loading checkpoint: {best_ckpt_path}')
ckpt = torch.load(best_ckpt_path, map_location='cpu', weights_only=False)

# ---- Build inference model ----
infer_backbone = timm.create_model(
    CFG['dinov2_variant'], pretrained=False, num_classes=0, img_size=CFG['image_size'])

infer_model = MultiViewModel(
    dinov2_model=infer_backbone, n_slots=N_SLOT, cls_dim=CFG['cls_dim'],
    n_classes=CFG['num_classes'], slot_hidden=CFG['slot_hidden'],
    dropout=0.0, unfreeze_layers=CFG['unfreeze_layers'],
).to(DEVICE)

# 加载权重（处理 DataParallel 前缀 + EMA）
state_dict = ckpt['model']
first_key = next(iter(state_dict))
if first_key.startswith('module.'):
    state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

# ★ 优先使用 EMA 权重
if ckpt.get('ema') and ckpt['ema'].get('shadow'):
    for name in state_dict:
        ema_key = name
        if ema_key in ckpt['ema']['shadow']:
            state_dict[name] = ckpt['ema']['shadow'][ema_key]
    print('  Using EMA weights for inference')

infer_model.load_state_dict(state_dict, strict=False)
infer_model.eval()
print(f'  Model loaded: epoch={ckpt.get("epoch")}, AUC={ckpt.get("auc", 0):.4f}')

# ============================================================
# Part A: Gold Validation (7-window TTA + 诊断池化)
# ============================================================

print('\n--- Gold Validation ---')
gold_cache = SLOT_CACHE  # reuse training cache
gold_mask_arr = SLOT_MASK
gold_val_uids = sorted(val_gold_uids)

# Filter to cached gold studies
cached_gold = [u for u in gold_val_uids if u in study_index]
print(f'Gold studies in cache: {len(cached_gold)}/{len(gold_val_uids)}')

# Build gold validation dataset with all 7 windows
N_WINDOWS = CFG['cache_slices'] - CFG['group_size'] + 1  # 7

gold_rows = []
for uid in cached_gold:
    ri = study_index[uid]
    slots_all = torch.from_numpy(gold_cache[ri].copy())  # [6, 9, H, W]
    m = torch.from_numpy(gold_mask_arr[ri].copy())        # [6]
    windows = torch.stack([slots_all[:, w:w+CFG['group_size']] for w in range(N_WINDOWS)], dim=0)
    gold_rows.append((windows, m, uid))

# 分批推理
gold_labels_map = gold_labels
gold_probs_list, gold_uids_list = [], []

@torch.no_grad()
def infer_gold_batch(windows_batch, mask_batch, model):
    """TTA + 诊断池化"""
    B = windows_batch.shape[0]
    W = N_WINDOWS
    flat = windows_batch.reshape(B * W, *windows_batch.shape[2:]).to(DEVICE)
    flat_mask = mask_batch.unsqueeze(1).expand(B, W, -1).reshape(B * W, -1).to(DEVICE)
    logits = model(flat, flat_mask)  # [B*W, 12]
    logits_w = logits.reshape(B, W, -1)  # [B, W, 12]
    return diagnostic_pool(logits_w.cpu())  # [B, 12]

for start in range(0, len(gold_rows), 8):
    batch = gold_rows[start:start+8]
    windows_batch = torch.stack([r[0] for r in batch])
    mask_batch = torch.stack([r[1] for r in batch])
    uids = [r[2] for r in batch]

    probs = infer_gold_batch(windows_batch, mask_batch, infer_model)
    gold_probs_list.append(probs)
    gold_uids_list.extend(uids)

gold_probs_all = torch.cat(gold_probs_list).numpy()

# Build labels
gold_labels_arr = np.zeros((len(gold_uids_list), N_CLASSES))
for i, uid in enumerate(gold_uids_list):
    for j, c in enumerate(TARGET_COLUMNS):
        raw = gold_labels_map.loc[uid, c] if uid in gold_labels_map.index else np.nan
        gold_labels_arr[i, j] = float(raw) if not pd.isna(raw) else 0.0

# AUC
gold_aucs = {}
for i, c in enumerate(TARGET_COLUMNS):
    yt, yp = gold_labels_arr[:, i], gold_probs_all[:, i]
    # Only labeled studies (all should be labeled for gold)
    valid = yt >= 0
    yt, yp = yt[valid], yp[valid]
    n_pos = int(yt.sum())
    if n_pos > 0 and n_pos < len(yt):
        try: gold_aucs[c] = float(roc_auc_score(yt, yp))
        except Exception: gold_aucs[c] = float('nan')
    else: gold_aucs[c] = float('nan')

valid_aucs = [v for v in gold_aucs.values() if not math.isnan(v)]
gold_macro = float(np.mean(valid_aucs)) if valid_aucs else float('nan')

print(f'\nGold Validation ({len(gold_uids_list)} studies, {N_WINDOWS}-window TTA + diag pool):')
print(f'  {"Class":<20s} {"AUC":>7s} {"Pos":>5s}')
for i, c in enumerate(TARGET_COLUMNS):
    a = gold_aucs[c]
    auc_s = f'{a:.4f}' if not math.isnan(a) else '  N/A  '
    print(f'  {c:<20s} {auc_s:>7s} {int(gold_labels_arr[:, i].sum()):5d}')
print(f'  {"Macro AUC":<20s} {gold_macro:7.4f}')

# ============================================================
# Part B: Test Set Inference → submission.csv
# ============================================================

print('\n--- Test Set Inference ---')

# Load test metadata
test_df = pd.read_csv(comp_input / 'test.csv')
test_df['StudyInstanceUID'] = test_df['StudyInstanceUID'].astype(str)
test_series = pd.read_csv(comp_input / 'test_series.csv')
test_series['StudyInstanceUID'] = test_series['StudyInstanceUID'].astype(str)
test_series['SeriesInstanceUID'] = test_series['SeriesInstanceUID'].astype(str)

# Build slot map for test studies
test_dicom_root = comp_input / 'test_series'
test_slot_map, _ = build_study_slot_map(test_series, test_dicom_root)
test_studies = sorted(test_slot_map.keys())
print(f'Test studies with DICOM: {len(test_studies)}/{len(test_df)}')

# Build test cache
n_test = len(test_studies)
TEST_CACHE = np.zeros((n_test, N_SLOT, CFG['cache_slices'], CFG['image_size'], CFG['image_size']), dtype=np.uint8)
TEST_MASK = np.zeros((n_test, N_SLOT), dtype=np.float32)
test_study_idx = {}

t0 = time.time()
jobs = []
for row_idx, study_uid in enumerate(test_studies):
    test_study_idx[study_uid] = row_idx
    study_slots = test_slot_map[study_uid]
    for slot_idx, (slot_name, plane, fluid, fatsat) in enumerate(SLOTS):
        slot_info = study_slots.get(slot_name)
        if slot_info is not None:
            jobs.append((row_idx, slot_idx, slot_name, plane, slot_info, None))

print(f'Decoding {len(jobs)} test slot-series...')
completed, failed = 0, 0
with ThreadPoolExecutor(max_workers=CFG['pix_threads']) as pool:
    for row_idx, slot_idx, result in pool.map(_read_slot_job, jobs):
        completed += 1
        if result is not None:
            TEST_CACHE[row_idx, slot_idx] = result
            TEST_MASK[row_idx, slot_idx] = 1.0
        else:
            failed += 1
        if completed % 1000 == 0:
            print(f'  [{completed}/{len(jobs)}] {time.time()-t0:.0f}s', flush=True)

print(f'Test cache: {n_test} studies in {time.time()-t0:.0f}s ({failed} failed)')

# TTA inference on test set
print('Running TTA inference on test set...')
test_probs = np.zeros((n_test, N_CLASSES), dtype=np.float32)

@torch.no_grad()
def infer_test_batch(indices, model):
    windows_list, masks_list = [], []
    for idx in indices:
        windows_list.append(torch.stack([
            torch.from_numpy(TEST_CACHE[idx, :, w:w+CFG['group_size']].copy())
            for w in range(N_WINDOWS)
        ], dim=0))
        masks_list.append(torch.from_numpy(TEST_MASK[idx].copy()))

    windows_batch = torch.stack(windows_list)  # [B, 7, 6, 3, H, W]
    mask_batch = torch.stack(masks_list)        # [B, 6]

    B, W = windows_batch.shape[0], N_WINDOWS
    flat = windows_batch.reshape(B * W, *windows_batch.shape[2:]).to(DEVICE)
    flat_mask = mask_batch.unsqueeze(1).expand(B, W, -1).reshape(B * W, -1).to(DEVICE)
    logits = model(flat, flat_mask)
    return diagnostic_pool(logits.reshape(B, W, -1).cpu())

t1 = time.time()
for start in range(0, n_test, 8):
    idx = list(range(start, min(start + 8, n_test)))
    test_probs[idx] = infer_test_batch(idx, infer_model).numpy()
    if start % 200 == 0:
        print(f'  [{start}/{n_test}] {time.time()-t1:.0f}s', flush=True)

print(f'Test inference done in {time.time()-t1:.0f}s')

# ---- Build submission.csv ----
submission_rows = []
for row_idx, study_uid in enumerate(test_studies):
    row = {'StudyInstanceUID': study_uid}
    for j, c in enumerate(TARGET_COLUMNS):
        row[c] = float(test_probs[row_idx, j])
    submission_rows.append(row)

submission_df = pd.DataFrame(submission_rows)

# Ensure all test studies are present (fill missing with 0.5)
full_submission = test_df[['StudyInstanceUID']].merge(
    submission_df, on='StudyInstanceUID', how='left')
for c in TARGET_COLUMNS:
    full_submission[c] = full_submission[c].fillna(0.5)

submission_path = output_dir / 'submission.csv'
full_submission.to_csv(submission_path, index=False)
print(f'\nSubmission saved: {submission_path}')
print(f'  Studies: {len(full_submission)} (expected: {len(test_df)})')
print(f'  Mean prob: {full_submission[TARGET_COLUMNS].values.mean():.4f}')

# Quick stats
for c in TARGET_COLUMNS:
    vals = full_submission[c].values
    print(f'  {c:<20s}: mean={vals.mean():.4f}, std={vals.std():.4f}, '
          f'>0.5={np.mean(vals>0.5):.1%}')

# ---- Save gold validation results ----
gold_rows_out = []
for i, uid in enumerate(gold_uids_list):
    row = {'StudyInstanceUID': uid}
    for j, c in enumerate(TARGET_COLUMNS):
        row[f'true_{c}'] = int(gold_labels_arr[i, j])
        row[f'prob_{c}'] = float(gold_probs_all[i, j])
    gold_rows_out.append(row)
pd.DataFrame(gold_rows_out).to_csv(output_dir / 'gold_validation_predictions.csv', index=False)

auc_rows = [{'class': c, 'auc': gold_aucs[c], 'n_pos': int(gold_labels_arr[:, i].sum())}
            for i, c in enumerate(TARGET_COLUMNS)]
pd.DataFrame(auc_rows + [{'class': 'macro_avg', 'auc': gold_macro, 'n_pos': 0}]
            ).to_csv(output_dir / 'gold_validation_auc.csv', index=False)

print(f'\nDone!')
print(f'  Gold AUC: {gold_macro:.4f}')
print(f'  Submission: {submission_path}')
print(f'  Ready to submit to Kaggle!')
