# ============================================================
# rad 成员推理 (v6a RadImageNet R50 @224px, 5 窗口 TTA + jitter)
#   推理后释放双 gold 缓存, 为评分侧可能的大 test 集腾内存
# ============================================================

print('\n--- RAD member inference (224px, 5-window TTA) ---')

ck = ckpt_meta['rad']
print(f'[rad] Loading {Path(checkpoints["rad"]).name} ...')
ema_note = load_member_weights(infer_model_rad, ck)
print(f'  epoch={ck.get("epoch")}, AUC={ck.get("auc", 0):.4f}, weights={ema_note}')

# ---- gold (58 研究, 5 窗口 @224) ----
gold_rows_rad = make_window_rows(
    GOLD_CACHE_RAD, GOLD_MASK_RAD, gold_studies, gold_study_index,
    CFG_RAD['group_size'], N_WINDOWS_RAD)

t0 = time.time()
rad_gold_probs = run_cached_inference(
    gold_rows_rad, infer_model_rad, N_WINDOWS_RAD, CFG_RAD['tta_jitter'])
rad_aucs, rad_macro = compute_gold_aucs(rad_gold_probs)
member_gold_probs['rad'] = rad_gold_probs
member_aucs['rad'] = (rad_aucs, rad_macro)
print(f'  Gold (58 studies, {N_WINDOWS_RAD}-window TTA+jitter + diag pool): '
      f'Macro AUC {rad_macro:.4f} ({time.time()-t0:.0f}s)'
      f'  [对照: v6a 训练会话 rad gold = 0.8137]')

gold_rows_out = []
for i, uid in enumerate(gold_studies):
    row = {'StudyInstanceUID': uid}
    for j, c in enumerate(TARGET_COLUMNS):
        row[f'true_{c}'] = int(gold_labels_arr[i, j])
        row[f'prob_{c}'] = float(rad_gold_probs[i, j])
    gold_rows_out.append(row)
pd.DataFrame(gold_rows_out).to_csv(
    output_dir / 'gold_validation_predictions_rad.csv', index=False)
auc_rows = [{'class': c, 'auc': rad_aucs[c], 'n_pos': int(gold_labels_arr[:, i].sum())}
            for i, c in enumerate(TARGET_COLUMNS)]
pd.DataFrame(auc_rows + [{'class': 'macro_avg', 'auc': rad_macro, 'n_pos': 0}]
             ).to_csv(output_dir / 'gold_validation_auc_rad.csv', index=False)

# ---- test ----
if TEST_CACHE_RAD is not None:
    rad_test_probs = run_test_inference(
        TEST_CACHE_RAD, TEST_MASK_RAD, infer_model_rad,
        CFG_RAD['group_size'], N_WINDOWS_RAD, CFG_RAD['tta_jitter'], 'rad')
    member_test_probs['rad'] = rad_test_probs
else:
    member_test_probs['rad'] = None

# ---- 释放 gold 缓存 (两套) + 窗口行, 腾内存 ----
del GOLD_CACHE_V5, GOLD_MASK_V5, GOLD_CACHE_RAD, GOLD_MASK_RAD
del gold_rows_v5, gold_rows_rad
gc.collect()
if DEVICE.type == 'cuda':
    with torch.cuda.device(DEVICE):
        torch.cuda.empty_cache()
print('Gold caches freed.')
