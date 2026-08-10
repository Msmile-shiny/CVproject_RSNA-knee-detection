# ============================================================
# Per-Class Threshold Analysis
# ============================================================
if IS_MAIN:
    preds_path = Path(CFG['output_dir']) / 'validation_predictions_best.csv'
    if preds_path.exists():
        preds = pd.read_csv(preds_path)

        print('=' * 70)
        print('FINAL THRESHOLD ANALYSIS (Best Epoch)')
        print('=' * 70)

        summary = []
        for c in TARGET_COLUMNS:
            y_true = preds[f'true_{c}'].values
            y_prob = preds[f'pred_{c}'].values
            n_pos = int(y_true.sum())
            if n_pos == 0: continue

            best_thr, best_f1 = 0.5, 0.0
            for thr in np.arange(0.01, 0.99, 0.01):
                y_pred = (y_prob >= thr).astype(int)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_thr = f1, thr

            f1_default = f1_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0)
            pred_mean = y_prob.mean()
            pct_05 = (y_prob > 0.5).mean() * 100

            need = 'YES' if best_thr < 0.35 else ('maybe' if best_thr < 0.45 else 'no')
            summary.append({
                'class': c, 'best_threshold': best_thr,
                'f1_at_0.5': f1_default, 'f1_at_best': best_f1,
                'pred_mean': pred_mean, 'pct_above_0.5': pct_05,
                'need_tuning': need,
            })

        summary_df = pd.DataFrame(summary)
        print(summary_df.to_string(index=False))

        n_need = (summary_df['need_tuning'] == 'YES').sum()
        n_maybe = (summary_df['need_tuning'] == 'maybe').sum()

        print(f'\n{"="*70}')
        print(f'VERDICT')
        print(f'{"="*70}')

        if n_need >= 6:
            print(f'{n_need}/12 classes NEED threshold tuning, {n_maybe} maybe.')
            print(f'RECOMMENDATION: Apply per-class optimal thresholds.')
        elif n_need >= 2 or n_maybe >= 3:
            print(f'{n_need} need tuning, {n_maybe} maybe.')
            print(f'RECOMMENDATION: Optional per-class thresholds for lagging classes.')
        else:
            print(f'Only {n_need} class(es) need tuning.')
            print(f'RECOMMENDATION: Multi-view fixed calibration. Use default threshold=0.5.')

        summary_df.to_csv(Path(CFG['output_dir']) / 'threshold_analysis.csv', index=False)
        print(f'\nAnalysis saved: threshold_analysis.csv')
    else:
        print('No validation predictions found. Run training first.')
