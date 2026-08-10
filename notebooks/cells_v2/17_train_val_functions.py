# ============================================================
# Training & Validation Functions -- v2: soft label support
# ============================================================

def train_epoch(model, loader, optimizer, criterion, scaler, epoch):
    """v2: uses soft prob targets + weights + masks from dataset.

    Each batch provides:
      - prob_targets: [B, 12] calibrated probabilities
      - weights:      [B, 12] per-class training weights
      - masks:        [B, 12] binary mask (0=skip this class for this sample)

    The loss re-weights uncertain pseudo-labels so the model focuses on
    reliable signals rather than memorizing LLM mistakes.

    Supports gradient accumulation (CFG['grad_accum_steps']) for larger
    effective batch sizes without extra VRAM.
    """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    use_amp = scaler is not None
    grad_accum = CFG.get('grad_accum_steps', 1)

    for bi, batch in enumerate(loader):
        images = batch['image'].to(DEVICE, non_blocking=True)
        prob_targets = batch['prob_targets'].to(DEVICE, non_blocking=True)
        weights = batch['weights'].to(DEVICE, non_blocking=True)
        masks = batch['masks'].to(DEVICE, non_blocking=True)

        # channels_last conversion for CNN speedup
        if CFG.get('channels_last', False):
            images = images.to(memory_format=torch.channels_last)

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, prob_targets, weights, masks)
            loss = loss / grad_accum  # normalize for gradient accumulation

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Step only after accumulating enough gradients
        if (bi + 1) % grad_accum == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum  # report un-normalized loss

        if IS_MAIN and bi % 20 == 0:
            # Show active mask ratio for monitoring
            active_pct = masks.sum().item() / masks.numel() * 100
            print(f'  Epoch {epoch:3d} [{bi:4d}/{len(loader):4d}] loss={loss.item()*grad_accum:.4f} '
                  f'| active_mask={active_pct:.0f}%', flush=True)

    return total_loss / len(loader)


@torch.no_grad()
def validate_epoch(model, loader, criterion, val_batch_size=None):
    """Study-level validation on gold labels.

    Uses standard BCE loss (hard labels) for comparability with v1.
    Aggregates slice-level predictions to study-level via mean pooling.

    val_batch_size: if set, chunks each loader batch into smaller sub-batches
    to reduce peak GPU memory during validation forward pass.
    """
    model.eval()

    study_probs = defaultdict(list)
    study_targets_dict = {}
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        images = batch['image'].to(DEVICE, non_blocking=True)
        labels = batch['labels'].to(DEVICE, non_blocking=True)
        uids = batch['study_uid']

        # Chunked forward pass to limit peak VRAM during validation
        if val_batch_size and images.size(0) > val_batch_size:
            all_logits = []
            for start in range(0, images.size(0), val_batch_size):
                chunk = images[start:start + val_batch_size]
                all_logits.append(model(chunk))
                # Free intermediates immediately
                if start + val_batch_size < images.size(0):
                    torch.cuda.empty_cache()
            logits = torch.cat(all_logits, dim=0)
        else:
            logits = model(images)

        total_loss += F.binary_cross_entropy_with_logits(logits, labels).item()
        n_batches += 1

        probs = torch.sigmoid(logits).cpu().numpy()
        for i, uid in enumerate(uids):
            study_probs[uid].append(probs[i])
            if uid not in study_targets_dict:
                study_targets_dict[uid] = labels[i].cpu().numpy()

    study_uids = list(study_targets_dict.keys())
    n_studies = len(study_uids)
    study_preds = np.zeros((n_studies, 12), dtype=np.float32)
    study_targets = np.zeros((n_studies, 12), dtype=np.float32)

    for i, uid in enumerate(study_uids):
        study_preds[i] = np.mean(study_probs[uid], axis=0)
        study_targets[i] = study_targets_dict[uid]

    per_class = {}
    aucs = []

    for i, c in enumerate(TARGET_COLUMNS):
        y_true = study_targets[:, i]
        y_prob = study_preds[:, i]
        n_pos = int(y_true.sum())

        metrics = {'auc': float('nan'), 'accuracy': float('nan'),
                   'precision': float('nan'), 'recall': float('nan'),
                   'f1': float('nan'), 'n_pos': n_pos, 'n_total': n_studies}

        if n_pos == 0 or n_pos == n_studies:
            y_pred_binary = (y_prob >= 0.5).astype(int)
            metrics['accuracy'] = float((y_true == y_pred_binary).mean())
            per_class[c] = metrics
            continue

        try:
            a = roc_auc_score(y_true, y_prob)
            metrics['auc'] = float(a)
            aucs.append(a)
        except Exception:
            pass

        y_pred_binary = (y_prob >= 0.5).astype(int)
        tp = int(((y_pred_binary == 1) & (y_true == 1)).sum())
        fp = int(((y_pred_binary == 1) & (y_true == 0)).sum())
        fn = int(((y_pred_binary == 0) & (y_true == 1)).sum())
        tn = int(((y_pred_binary == 0) & (y_true == 0)).sum())

        metrics['accuracy'] = float((tp + tn) / n_studies)
        metrics['precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        metrics['recall'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        metrics['f1'] = float(2 * metrics['precision'] * metrics['recall'] /
                              (metrics['precision'] + metrics['recall'])) if (metrics['precision'] + metrics['recall']) > 0 else 0.0
        per_class[c] = metrics

    return {
        'loss': total_loss / max(n_batches, 1),
        'macro_auc': float(np.mean(aucs)) if aucs else 0.0,
        'per_class': per_class,
        'study_preds': study_preds,
        'study_targets': study_targets,
        'study_uids': study_uids,
    }


# -- Per-class threshold analysis (v2 NEW) -----------------------
def analyze_thresholds(val_metrics):
    """Determine whether per-class threshold tuning is needed.

    For each class, sweeps threshold [0.05, 0.95] and finds the optimal
    threshold maximizing F1 score on validation gold labels.

    Interpretation:
      - best_threshold ~= 0.5 for most classes
        -> soft labels fixed the calibration problem -> tuning NOT needed
      - best_threshold << 0.5 for many classes
        -> model is still under-confident -> tuning IS needed
    """
    print(f'\n  {"="*70}')
    print(f'  Per-Class Threshold Analysis')
    print(f'  {"="*70}')
    print(f'  {"Class":<20s} {"Best Thr":>8s} {"F1@0.5":>7s} {"F1@best":>7s} '
          f'{"Pred Mean":>9s} {"%>0.5":>6s} {"Need?":>6s}')
    print(f'  {"-"*20} {"-"*8} {"-"*7} {"-"*7} {"-"*9} {"-"*6} {"-"*6}')

    need_tuning = []
    threshold_table = {}

    for i, c in enumerate(TARGET_COLUMNS):
        y_true = val_metrics['study_targets'][:, i]
        y_prob = val_metrics['study_preds'][:, i]

        if y_true.sum() == 0:
            continue

        best_thr, best_f1 = 0.5, 0.0
        for thr in np.arange(0.05, 0.95, 0.01):
            y_pred = (y_prob >= thr).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr

        f1_default = f1_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0)
        pct_above = (y_prob > 0.5).mean() * 100
        pred_mean = y_prob.mean()

        needs = 'YES' if best_thr < 0.35 else ('maybe' if best_thr < 0.45 else 'no')
        if best_thr < 0.40:
            need_tuning.append((c, best_thr, f1_default, best_f1))

        threshold_table[c] = {
            'best_threshold': float(best_thr),
            'f1_at_0.5': float(f1_default),
            'f1_at_best': float(best_f1),
            'pred_mean': float(pred_mean),
            'pct_above_0.5': float(pct_above),
        }

        print(f'  {c:<20s} {best_thr:8.2f} {f1_default:7.3f} {best_f1:7.3f} '
              f'{pred_mean:9.4f} {pct_above:5.1f}% {needs:>6s}')

    n_need = len(need_tuning)
    print(f'\n  {n_need}/12 classes need threshold < 0.40')
    if n_need >= 6:
        print(f'  -> Per-class threshold tuning is NECESSARY')
    elif n_need >= 2:
        print(f'  -> Per-class threshold tuning RECOMMENDED for a few classes')
    else:
        print(f'  -> Soft labels fixed calibration! Use default threshold=0.5')

    return threshold_table


def save_validation_report(val_metrics, output_dir, epoch=None, is_best=False, thresholds=None):
    out = Path(output_dir)
    rows = []
    for c in TARGET_COLUMNS:
        m = val_metrics['per_class'][c]
        row = {'class': c, 'auc': m['auc'], 'accuracy': m['accuracy'],
               'precision': m['precision'], 'recall': m['recall'],
               'f1': m['f1'], 'n_pos': m['n_pos'], 'n_total': m['n_total']}
        if thresholds and c in thresholds:
            row['best_threshold'] = thresholds[c]['best_threshold']
            row['f1_at_best'] = thresholds[c]['f1_at_best']
        rows.append(row)

    report_df = pd.DataFrame(rows)
    report_df['macro_auc'] = val_metrics['macro_auc']
    report_df['val_loss'] = val_metrics['loss']

    tag = f'_epoch{epoch}' if epoch else ''
    if is_best: tag = '_best'
    report_path = out / f'validation_report{tag}.csv'
    report_df.to_csv(report_path, index=False)
    print(f'  Report: {report_path}')

    study_rows = []
    for i, uid in enumerate(val_metrics['study_uids']):
        row = {'StudyInstanceUID': uid}
        for j, c in enumerate(TARGET_COLUMNS):
            row[f'true_{c}'] = int(val_metrics['study_targets'][i, j])
            row[f'pred_{c}'] = float(val_metrics['study_preds'][i, j])
        study_rows.append(row)
    preds_df = pd.DataFrame(study_rows)
    preds_path = out / f'validation_predictions{tag}.csv'
    preds_df.to_csv(preds_path, index=False)
    print(f'  Predictions: {preds_path} ({len(study_rows)} studies)')

    return report_df


def print_validation_summary(val_metrics):
    print(f'\n  {"Class":<20s} {"AUC":>7s} {"Acc":>7s} {"Prec":>7s} {"Rec":>7s} {"F1":>7s} {"Pos":>5s}')
    print(f'  {"-"*20} {"-"*7} {"-"*7} {"-"*7} {"-"*7} {"-"*7} {"-"*5}')
    for c in TARGET_COLUMNS:
        m = val_metrics['per_class'][c]
        auc_str = f'{m["auc"]:.3f}' if not math.isnan(m['auc']) else '  N/A  '
        print(f'  {c:<20s} {auc_str:>7s} {m["accuracy"]:7.3f} {m["precision"]:7.3f} '
              f'{m["recall"]:7.3f} {m["f1"]:7.3f} {m["n_pos"]:5d}')
    print(f'  {"-"*20} {"-"*7} {"-"*7} {"-"*7} {"-"*7} {"-"*7} {"-"*5}')
    print(f'  {"Macro AUC":<20s} {val_metrics["macro_auc"]:7.3f}')
    print()
