# ============================================================
# v5 seed 成员推理 — gold (58 研究) + test
#   7 窗口 TTA + jitter + 诊断池化 (与训练 cell 17 / 推理 notebook 逐字一致)
# 通用批量推理函数 (cache/mask/窗口数参数化) 供 rad 成员复用
# ============================================================

print('\n--- V5 member inference (288px, 7-window TTA) ---')


def make_window_rows(cache, mask, uids, study_index, group_size, n_windows):
    """把缓存切成 [per-study 窗口列表]: [(windows [W,6,3,H,W], mask [6], uid)]。

    uids: 有序研究列表 (决定 probs 行序); study_index: uid -> 缓存行号。
    """
    rows = []
    for uid in uids:
        idx = study_index[uid]
        slots_all = torch.from_numpy(cache[idx].copy())
        m = torch.from_numpy(mask[idx].copy())
        windows = torch.stack(
            [slots_all[:, w:w + group_size] for w in range(n_windows)], dim=0)
        rows.append((windows, m, uid))
    return rows


@torch.no_grad()
def infer_batch(windows_batch, mask_batch, model, n_windows, tta_jitter_on):
    """TTA + 诊断池化（jitter 视图平均 → per-target 窗口池化）"""
    B = windows_batch.shape[0]
    W = n_windows
    flat = windows_batch.reshape(B * W, *windows_batch.shape[2:]).to(DEVICE)  # B-major
    flat_mask = mask_batch.unsqueeze(1).expand(B, W, -1).reshape(B * W, -1).to(DEVICE)
    if tta_jitter_on:
        flat = torch.cat([flat, tta_jitter(flat)], dim=0)   # [2*B*W, ...] 原始块在前
        flat_mask = flat_mask.repeat(2, 1)
        n_orig = W
    else:
        n_orig = None
    logits = model(flat, flat_mask)  # [V*B*W, 12]
    logits_v = stack_views(logits, B, W, n_orig)  # [B, V*W, 12]
    return diagnostic_pool(logits_v.cpu(), n_orig=n_orig)  # [B, 12]


def run_cached_inference(rows, model, n_windows, tta_jitter_on, batch_size=8):
    """分批推理 rows → [n, 12] probs。"""
    probs_list = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        windows_batch = torch.stack([r[0] for r in batch])
        mask_batch = torch.stack([r[1] for r in batch])
        probs_list.append(infer_batch(
            windows_batch, mask_batch, model, n_windows, tta_jitter_on))
    return torch.cat(probs_list).numpy()


def run_test_inference(cache, mask, model, group_size, n_windows,
                       tta_jitter_on, label):
    """全 test 推理 → [n_test, 12] probs (或 None)。"""
    if cache is None:
        return None
    n_test = cache.shape[0]
    test_probs = np.zeros((n_test, N_CLASSES), dtype=np.float32)
    t1 = time.time()
    for start in range(0, n_test, 8):
        idx = list(range(start, min(start + 8, n_test)))
        windows_list, masks_list, empty_mask = [], [], []
        for i in idx:
            windows_list.append(torch.stack([
                torch.from_numpy(cache[i, :, w:w + group_size].copy())
                for w in range(n_windows)
            ], dim=0))
            masks_list.append(torch.from_numpy(mask[i].copy()))
            empty_mask.append(mask[i].sum() == 0)

        probs = infer_batch(
            torch.stack(windows_list), torch.stack(masks_list),
            model, n_windows, tta_jitter_on)
        # Studies with no slots → fill 0.5
        for i, is_empty in enumerate(empty_mask):
            if is_empty:
                probs[i] = 0.5
        test_probs[idx] = probs.numpy()
        if start % 200 == 0:
            print(f'  [{start}/{n_test}] {time.time()-t1:.0f}s', flush=True)
    print(f'  {label} test inference: {n_test} studies in {time.time()-t1:.0f}s')
    return test_probs


# ---- gold 真值标签 ----
gold_labels_arr = np.zeros((len(gold_studies), N_CLASSES))
for i, uid in enumerate(gold_studies):
    for j, c in enumerate(TARGET_COLUMNS):
        raw = gold_labels.loc[uid, c] if uid in gold_labels.index else np.nan
        gold_labels_arr[i, j] = float(raw) if not pd.isna(raw) else 0.0


def compute_gold_aucs(probs):
    aucs = {}
    for i, c in enumerate(TARGET_COLUMNS):
        yt, yp = gold_labels_arr[:, i], probs[:, i]
        n_pos = int(yt.sum())
        if n_pos > 0 and n_pos < len(yt):
            try:
                aucs[c] = float(roc_auc_score(yt, yp))
            except Exception:
                aucs[c] = float('nan')
        else:
            aucs[c] = float('nan')
    valid = [v for v in aucs.values() if not math.isnan(v)]
    macro = float(np.mean(valid)) if valid else float('nan')
    return aucs, macro


# ---- gold 窗口行 (v5: 7 窗口) ----
gold_rows_v5 = make_window_rows(
    GOLD_CACHE_V5, GOLD_MASK_V5, gold_studies, gold_study_index,
    CFG_V5['group_size'], N_WINDOWS_V5)

# ---- 成员循环 ----
member_gold_probs = {}   # key -> [58, 12]
member_test_probs = {}   # key -> [n_test, 12] (或 None)
member_aucs = {}         # key -> (per-class dict, macro)

v5_keys = seeds + (['spec'] if has_spec else [])
for key in v5_keys:
    ck = ckpt_meta[key]
    tag = member_label(key)
    print(f'\n[{tag}] Loading {Path(checkpoints[key]).name} ...')
    ema_note = load_member_weights(infer_model_v5, ck)
    print(f'  epoch={ck.get("epoch")}, AUC={ck.get("auc", 0):.4f}, weights={ema_note}')

    t0 = time.time()
    gold_probs = run_cached_inference(
        gold_rows_v5, infer_model_v5, N_WINDOWS_V5,
        CFG_V5['tta_jitter'])
    aucs, macro = compute_gold_aucs(gold_probs)
    member_gold_probs[key] = gold_probs
    member_aucs[key] = (aucs, macro)
    print(f'  Gold (58 studies, {N_WINDOWS_V5}-window TTA+jitter + diag pool): '
          f'Macro AUC {macro:.4f} ({time.time()-t0:.0f}s)')

    # 保存成员 gold 预测 (与训练产物同格式, 可逐位对比)
    gold_rows_out = []
    for i, uid in enumerate(gold_studies):
        row = {'StudyInstanceUID': uid}
        for j, c in enumerate(TARGET_COLUMNS):
            row[f'true_{c}'] = int(gold_labels_arr[i, j])
            row[f'prob_{c}'] = float(gold_probs[i, j])
        gold_rows_out.append(row)
    pd.DataFrame(gold_rows_out).to_csv(
        output_dir / f'gold_validation_predictions_{tag}.csv', index=False)
    auc_rows = [{'class': c, 'auc': aucs[c], 'n_pos': int(gold_labels_arr[:, i].sum())}
                for i, c in enumerate(TARGET_COLUMNS)]
    pd.DataFrame(auc_rows + [{'class': 'macro_avg', 'auc': macro, 'n_pos': 0}]
                 ).to_csv(output_dir / f'gold_validation_auc_{tag}.csv', index=False)

    if TEST_CACHE_V5 is not None:
        test_probs = run_test_inference(
            TEST_CACHE_V5, TEST_MASK_V5, infer_model_v5,
            CFG_V5['group_size'], N_WINDOWS_V5, CFG_V5['tta_jitter'], tag)
        member_test_probs[key] = test_probs
