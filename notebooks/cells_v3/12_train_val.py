# ============================================================
# v3: Training & Validation Functions
# ============================================================

def train_epoch(model, loader, optimizer, criterion, scaler, epoch):
    """v3: Unified training — all samples use soft-label format.

    Gold studies: prob = hard label (0/1), weight = 1.0, mask = 1.0
    Pseudo studies: prob = calibrated, weight = reliability, mask = thresholded
    Both use the same WeightedSoftBCELoss.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()
    use_amp = scaler is not None
    grad_accum = CFG.get('grad_accum_steps', 1)

    for bi, batch in enumerate(loader):
        slots = batch['slots'].to(DEVICE, non_blocking=True)
        mask = batch['mask'].to(DEVICE, non_blocking=True)
        prob_targets = batch['prob_targets'].to(DEVICE, non_blocking=True)
        weights = batch['weights'].to(DEVICE, non_blocking=True)
        soft_masks = batch['soft_masks'].to(DEVICE, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(slots, mask)
            loss = criterion(logits, prob_targets, weights, soft_masks)
            loss = loss / grad_accum

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

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

        total_loss += loss.item() * grad_accum
        n_batches += 1

        if IS_MAIN and bi % 20 == 0:
            slots_present = mask.sum(dim=1).mean().item()
            print(f'  Epoch {epoch:3d} [{bi:4d}/{len(loader):4d}] '
                  f'loss={loss.item()*grad_accum:.4f} | slots={slots_present:.1f}/6',
                  flush=True)

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate_epoch(model, loader, criterion_hard):
    """Study-level validation on gold labels with partial-label support.

    Each gold study may have only a subset of the 12 targets labeled.
    Loss and per-class AUC are computed only on labeled targets.
    """
    model.eval()

    all_logits = []
    all_labels = []
    all_masks = []
    all_uids = []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        slots = batch['slots'].to(DEVICE, non_blocking=True)
        mask = batch['mask'].to(DEVICE, non_blocking=True)
        labels = batch['labels'].to(DEVICE, non_blocking=True)
        val_masks = batch['val_masks'].to(DEVICE, non_blocking=True)
        uids = batch['study_uid']

        logits = model(slots, mask)

        # Masked BCE: only compute loss on labeled targets
        active = val_masks > 0.5
        if active.any():
            loss_val = F.binary_cross_entropy_with_logits(
                logits[active], labels[active], reduction='mean')
            total_loss += loss_val.item()
        n_batches += 1

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_masks.append(val_masks.cpu())
        all_uids.extend(uids)

    # Concatenate
    logits_all = torch.cat(all_logits, dim=0).numpy()  # [N_val, 12]
    labels_all = torch.cat(all_labels, dim=0).numpy()
    masks_all = torch.cat(all_masks, dim=0).numpy()
    probs_all = 1.0 / (1.0 + np.exp(-logits_all))  # sigmoid

    # Per-class metrics — computed on LABELED studies only per class
    per_class = {}
    aucs = []
    n_studies = len(all_uids)

    for i, c in enumerate(TARGET_COLUMNS):
        # Filter to studies where THIS target is labeled
        labeled_idx = masks_all[:, i] > 0.5
        n_labeled = int(labeled_idx.sum())

        metrics = {
            'auc': float('nan'), 'accuracy': float('nan'),
            'precision': float('nan'), 'recall': float('nan'),
            'f1': float('nan'), 'n_pos': 0, 'n_total': n_labeled,
        }

        # Skip classes with too few labeled studies for meaningful metrics
        if n_labeled <= 1:
            per_class[c] = metrics
            continue

        y_true = labels_all[:, i][labeled_idx]
        y_prob = probs_all[:, i][labeled_idx]
        n_pos = int(y_true.sum())
        n_neg = n_labeled - n_pos
        metrics['n_pos'] = n_pos

        if n_pos == 0 or n_neg == 0:
            # All same label → AUC undefined, but accuracy still meaningful
            y_pred_binary = (y_prob >= 0.5).astype(int)
            metrics['accuracy'] = float((y_true == y_pred_binary).mean())
            per_class[c] = metrics
            # Still track if we have enough for AUC
            if n_pos > 0 and n_neg > 0:
                try:
                    a = roc_auc_score(y_true, y_prob)
                    metrics['auc'] = float(a)
                except Exception:
                    pass
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
        tn = n_labeled - tp - fp - fn

        metrics['accuracy'] = float((tp + tn) / n_labeled)
        metrics['precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        metrics['recall'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        metrics['f1'] = float(
            2 * metrics['precision'] * metrics['recall']
            / (metrics['precision'] + metrics['recall'])
        ) if (metrics['precision'] + metrics['recall']) > 0 else 0.0
        per_class[c] = metrics

    return {
        'loss': total_loss / max(n_batches, 1),
        'macro_auc': float(np.mean(aucs)) if aucs else 0.0,
        'per_class': per_class,
        'probs': probs_all,
        'labels': labels_all,
        'uids': all_uids,
    }


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


def save_validation_report(val_metrics, output_dir, epoch=None, is_best=False):
    out = Path(output_dir)
    rows = []
    for c in TARGET_COLUMNS:
        m = val_metrics['per_class'][c]
        rows.append({
            'class': c, 'auc': m['auc'], 'accuracy': m['accuracy'],
            'precision': m['precision'], 'recall': m['recall'],
            'f1': m['f1'], 'n_pos': m['n_pos'], 'n_total': m['n_total'],
        })

    report_df = pd.DataFrame(rows)
    report_df['macro_auc'] = val_metrics['macro_auc']
    report_df['val_loss'] = val_metrics['loss']

    tag = '_best' if is_best else f'_epoch{epoch}'
    report_path = out / f'validation_report{tag}.csv'
    report_df.to_csv(report_path, index=False)

    # Save per-study predictions
    study_rows = []
    for i, uid in enumerate(val_metrics['uids']):
        row = {'StudyInstanceUID': uid}
        for j, c in enumerate(TARGET_COLUMNS):
            row[f'true_{c}'] = int(val_metrics['labels'][i, j])
            row[f'pred_{c}'] = float(val_metrics['probs'][i, j])
        study_rows.append(row)
    preds_df = pd.DataFrame(study_rows)
    preds_path = out / f'validation_predictions{tag}.csv'
    preds_df.to_csv(preds_path, index=False)

    if IS_MAIN:
        print(f'  Report: {report_path}')
        print(f'  Predictions: {preds_path} ({len(study_rows)} studies)')

    return report_df
