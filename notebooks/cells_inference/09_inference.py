# ============================================================
# Test Inference — 7-Window TTA + 诊断池化 -> submission.csv
# ============================================================

N_WINDOWS = CFG['cache_slices'] - CFG['group_size'] + 1  # 7
print(f'Running TTA inference: {n_test} studies, {N_WINDOWS}-window TTA + diag pool')

test_probs = np.zeros((n_test, N_CLASSES), dtype=np.float32)

@torch.no_grad()
def infer_test_batch(indices, model):
    windows_list, masks_list, empty_mask = [], [], []
    for idx in indices:
        windows_list.append(torch.stack([
            torch.from_numpy(TEST_CACHE[idx, :, w:w + CFG['group_size']].copy())
            for w in range(N_WINDOWS)
        ], dim=0))
        masks_list.append(torch.from_numpy(TEST_MASK[idx].copy()))
        empty_mask.append(TEST_MASK[idx].sum() == 0)

    windows_batch = torch.stack(windows_list)
    mask_batch = torch.stack(masks_list)

    B, W = windows_batch.shape[0], N_WINDOWS
    flat = windows_batch.reshape(B * W, *windows_batch.shape[2:]).to(DEVICE)
    flat_mask = mask_batch.unsqueeze(1).expand(B, W, -1).reshape(B * W, -1).to(DEVICE)
    logits = model(flat, flat_mask)
    probs = diagnostic_pool(logits.reshape(B, W, -1).cpu())

    for i, is_empty in enumerate(empty_mask):
        if is_empty:
            probs[i] = 0.5
    return probs


t_infer = time.time()
batch_size = CFG['batch_size']
for start in range(0, n_test, batch_size):
    idx = list(range(start, min(start + batch_size, n_test)))
    test_probs[idx] = infer_test_batch(idx, model).numpy()
    if start % 200 == 0:
        print(f'  [{start}/{n_test}] {time.time()-t_infer:.0f}s', flush=True)

print(f'Test inference done in {time.time()-t_infer:.0f}s')

# ---- Build submission.csv ----
submission_rows = []
for row_idx, study_uid in enumerate(test_studies):
    row = {'StudyInstanceUID': study_uid}
    for j, c in enumerate(TARGET_COLUMNS):
        row[c] = float(test_probs[row_idx, j])
    submission_rows.append(row)

submission_df = pd.DataFrame(submission_rows)

# 确保所有 test.csv 中的 study 都在 submission 中
full_submission = test_df[['StudyInstanceUID']].merge(
    submission_df, on='StudyInstanceUID', how='left')
for c in TARGET_COLUMNS:
    full_submission[c] = full_submission[c].fillna(0.5)

submission_path = output_dir / 'submission.csv'
full_submission.to_csv(submission_path, index=False)
print(f'\nSubmission saved: {submission_path}')
print(f'  Studies: {len(full_submission)} (expected: {len(test_df)})')
print(f'  Mean prob: {full_submission[TARGET_COLUMNS].values.mean():.4f}')
print()

# Per-class stats
for c in TARGET_COLUMNS:
    vals = full_submission[c].values
    print(f'  {c:<20s}: mean={vals.mean():.4f}, std={vals.std():.4f}, '
          f'>0.5={np.mean(vals>0.5):.1%}')

print(f'\nDone! Submit {submission_path} to Kaggle.')
