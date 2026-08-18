# ============================================================
# v5: Test Set Inference + Submission.csv + Gold Validation
# ============================================================
#
# 1. 在 gold 验证集上做完整 TTA + 诊断池化，计算真实 AUC
# 2. 在 test 集上推理，生成 submission.csv
# ============================================================

print('=' * 60)
print('GOLD VALIDATION + TEST INFERENCE')
print('=' * 60)

# ---- 加载最佳 checkpoint ----
best_ckpt_path = output_dir / 'checkpoints' / CKPT_NAME
if not best_ckpt_path.exists():
    raise FileNotFoundError(
        f'Best model checkpoint not found: {best_ckpt_path}\n'
        f'Training must produce {CKPT_NAME} (best validation AUC). '
        'Check that at least one epoch completed and saved a best model.')

print(f'Loading checkpoint: {best_ckpt_path}')
ckpt = torch.load(best_ckpt_path, map_location='cpu', weights_only=False)

# ★ seed 交叉验证: 防止下载归档时混用不同会话的产物
ckpt_seed = (ckpt.get('config') or {}).get('seed', '?')
print(f'  Checkpoint seed: s{ckpt_seed} | 本会话 seed: s{CFG["seed"]}')

# ---- Build inference model ----
# (checkpoint 已含全部权重 — 不需要挂载 RadImageNet 权重数据集)
infer_backbone = torchvision.models.resnet50(weights=None)
infer_backbone.fc = nn.Identity()

infer_model = RadResNetModel(
    backbone=infer_backbone, n_slots=N_SLOT, feature_dim=CFG['feature_dim'],
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
    """TTA + 诊断池化（jitter 视图平均 → per-target 窗口池化）"""
    B = windows_batch.shape[0]
    W = N_WINDOWS
    flat = windows_batch.reshape(B * W, *windows_batch.shape[2:]).to(DEVICE)  # B-major
    flat_mask = mask_batch.unsqueeze(1).expand(B, W, -1).reshape(B * W, -1).to(DEVICE)
    if CFG.get('tta_jitter', False):
        flat = torch.cat([flat, tta_jitter(flat)], dim=0)   # [2*B*W, ...] 原始块在前
        flat_mask = flat_mask.repeat(2, 1)
        n_orig = W
    else:
        n_orig = None
    logits = model(flat, flat_mask)  # [V*B*W, 12]
    logits_v = stack_views(logits, B, W, n_orig)  # [B, V*W, 12]
    return diagnostic_pool(logits_v.cpu(), n_orig=n_orig)  # [B, 12]

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

print(f'\nGold Validation ({len(gold_uids_list)} studies, '
      f'{N_WINDOWS}-window TTA{"+jitter" if CFG.get("tta_jitter", False) else ""} + diag pool):')
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

# ---- Build test series metadata ----
# 优先使用 test_series.csv；如果不足，扫描 DICOM 目录构建元数据
test_dicom_root = comp_input / 'test_series'

def _find_dicom_files(series_dir):
    """列出目录中的 DICOM 文件（不依赖扩展名，竞赛 test 集 DICOM 无 .dcm 后缀）。"""
    all_files = sorted([f for f in series_dir.iterdir() if f.is_file()])
    # 优先 .dcm 后缀；若无，取所有文件（跳过隐藏文件）
    dcm = [f for f in all_files if f.suffix == '.dcm']
    return dcm if dcm else [f for f in all_files if not f.name.startswith('.')]


def _scan_test_dicoms(dicom_root):
    """扫描测试集 DICOM 目录，从 header 推断 plane / fluid / fatsat。"""
    rows = []
    root = Path(dicom_root)
    if not root.exists():
        return rows
    for study_dir in sorted(root.iterdir()):
        if not study_dir.is_dir():
            continue
        study_uid = study_dir.name
        for series_dir in sorted(study_dir.iterdir()):
            if not series_dir.is_dir():
                continue
            series_uid = series_dir.name
            dcm_files = _find_dicom_files(series_dir)
            if not dcm_files:
                continue
            try:
                ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True, force=True)

                # ★ Anatomical Plane (from ImageOrientationPatient)
                iop = getattr(ds, 'ImageOrientationPatient', None)
                plane = 'Axial'  # default
                if iop is not None and len(iop) >= 6:
                    try:
                        row_cos = np.array([float(iop[0]), float(iop[1]), float(iop[2])])
                        col_cos = np.array([float(iop[3]), float(iop[4]), float(iop[5])])
                        normal = np.cross(row_cos, col_cos)
                        dominant = int(np.argmax(np.abs(normal)))
                        plane = {0: 'Sagittal', 1: 'Coronal', 2: 'Axial'}[dominant]
                    except Exception:
                        pass

                # ★ Fat Suppression (from ScanOptions / SeriesDescription)
                desc = str(getattr(ds, 'SeriesDescription', '')).lower()
                seq_name = str(getattr(ds, 'SequenceName', '')).lower()
                scan_opts = str(getattr(ds, 'ScanOptions', '')).upper()

                fs_kw = ['fs', 'fatsat', 'fat sat', 'stir', 'spair', 'spir', 'we',
                         'water excit', 'tirm', 'fatsup']
                has_fs = any(kw in desc for kw in fs_kw)
                has_fs = has_fs or any(kw in scan_opts for kw in ['FS', 'FATSAT', 'SPAIR', 'SPIR'])

                # ★ Fluid Sensitive (T2 / PD weighted)
                t1_kw = ['t1', 't1w']
                is_t1 = any(kw in desc or kw in seq_name for kw in t1_kw)
                is_t2 = any(kw in desc or kw in seq_name for kw in ['t2', 't2w'])
                is_pd = any(kw in desc for kw in ['pd', 'pdw', 'proton', 'dp', 'dens'])
                has_fluid = (is_t2 or is_pd) and not is_t1

                rows.append({
                    'StudyInstanceUID': study_uid,
                    'SeriesInstanceUID': series_uid,
                    'Anatomical_Plane': plane,
                    'Fluid_Sensitive': 1 if has_fluid else 0,
                    'Fat_Suppression': 1 if has_fs else 0,
                })
            except Exception:
                continue
    return rows


# ★ 重写逻辑：CSV 结果不会被 DICOM scan 失败覆盖
test_slot_map = {}
test_series_path = comp_input / 'test_series.csv'

if test_series_path.exists():
    test_series = pd.read_csv(test_series_path)
    test_series['StudyInstanceUID'] = test_series['StudyInstanceUID'].astype(str)
    test_series['SeriesInstanceUID'] = test_series['SeriesInstanceUID'].astype(str)
    print(f'test_series.csv: {len(test_series)} series, '
          f'{test_series["StudyInstanceUID"].nunique()} studies')

    # 先走 CSV 路径
    test_slot_map, _ = build_study_slot_map(test_series, test_dicom_root)
    csv_studies = len(test_slot_map)

    # ★ 如果 CSV 覆盖不足（< 50% test studies），扫描 DICOM 补充
    if csv_studies < max(10, len(test_df) * 0.5):
        print(f'CSV coverage ({csv_studies}/{len(test_df)}) insufficient, '
              f'scanning DICOM headers...')
        dicom_rows = _scan_test_dicoms(test_dicom_root)
        if dicom_rows:
            test_series = pd.DataFrame(dicom_rows)
            test_slot_map, _ = build_study_slot_map(test_series, test_dicom_root)
            print(f'DICOM scan: {len(test_series)} series, '
                  f'{test_series["StudyInstanceUID"].nunique()} studies → '
                  f'{len(test_slot_map)} studies matched')
        else:
            print(f'DICOM scan returned 0 rows, keeping CSV results ({csv_studies} studies)')
    # else: CSV 覆盖率够了，直接用
else:
    print('test_series.csv not found, scanning DICOM headers...')
    dicom_rows = _scan_test_dicoms(test_dicom_root)
    if dicom_rows:
        test_series = pd.DataFrame(dicom_rows)
        test_slot_map, _ = build_study_slot_map(test_series, test_dicom_root)
        print(f'DICOM scan: {len(test_series)} series, '
              f'{len(test_slot_map)} studies matched')

test_studies = sorted(test_slot_map.keys())
print(f'Test studies with slot match: {len(test_studies)}/{len(test_df)}')

# ★ 释放训练缓存，为测试缓存腾出内存
del SLOT_CACHE, SLOT_MASK
gc.collect()
print(f'Freed train cache for test set (GPU: {torch.cuda.memory_allocated()/1024**3:.1f} GB)')

# Build test cache + inference (if DICOMs available)
if len(test_studies) > 0:
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
        windows_list, masks_list, empty_mask = [], [], []
        for idx in indices:
            windows_list.append(torch.stack([
                torch.from_numpy(TEST_CACHE[idx, :, w:w+CFG['group_size']].copy())
                for w in range(N_WINDOWS)
            ], dim=0))
            masks_list.append(torch.from_numpy(TEST_MASK[idx].copy()))
            empty_mask.append(TEST_MASK[idx].sum() == 0)  # Track studies with no slots

        windows_batch = torch.stack(windows_list)
        mask_batch = torch.stack(masks_list)

        B, W = windows_batch.shape[0], N_WINDOWS
        flat = windows_batch.reshape(B * W, *windows_batch.shape[2:]).to(DEVICE)  # B-major
        flat_mask = mask_batch.unsqueeze(1).expand(B, W, -1).reshape(B * W, -1).to(DEVICE)
        if CFG.get('tta_jitter', False):
            flat = torch.cat([flat, tta_jitter(flat)], dim=0)   # [2*B*W, ...] 原始块在前
            flat_mask = flat_mask.repeat(2, 1)
            n_orig = W
        else:
            n_orig = None
        logits = model(flat, flat_mask)
        probs = diagnostic_pool(
            stack_views(logits, B, W, n_orig).cpu(), n_orig=n_orig)  # [B, C]

        # Studies with no slots → fill 0.5
        for i, is_empty in enumerate(empty_mask):
            if is_empty:
                probs[i] = 0.5
        return probs

    t1 = time.time()
    for start in range(0, n_test, 8):
        idx = list(range(start, min(start + 8, n_test)))
        test_probs[idx] = infer_test_batch(idx, infer_model).numpy()
        if start % 200 == 0:
            print(f'  [{start}/{n_test}] {time.time()-t1:.0f}s', flush=True)

    print(f'Test inference done in {time.time()-t1:.0f}s')

    # Build submission rows from inference results
    submission_rows = []
    for row_idx, study_uid in enumerate(test_studies):
        row = {'StudyInstanceUID': study_uid}
        for j, c in enumerate(TARGET_COLUMNS):
            row[c] = float(test_probs[row_idx, j])
        submission_rows.append(row)
else:
    print('No test DICOMs found — filling all studies with 0.5')
    submission_rows = []

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
pd.DataFrame(gold_rows_out).to_csv(
    output_dir / f'gold_validation_predictions_{SEED_TAG}.csv', index=False)

auc_rows = [{'class': c, 'auc': gold_aucs[c], 'n_pos': int(gold_labels_arr[:, i].sum())}
            for i, c in enumerate(TARGET_COLUMNS)]
pd.DataFrame(auc_rows + [{'class': 'macro_avg', 'auc': gold_macro, 'n_pos': 0}]
            ).to_csv(output_dir / f'gold_validation_auc_{SEED_TAG}.csv', index=False)

print(f'\nDone!')
print(f'  Gold AUC: {gold_macro:.4f}')
print(f'  Submission: {submission_path}')
print(f'  Ready to submit to Kaggle!')
