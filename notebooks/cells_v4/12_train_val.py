# ============================================================
# v4: Training & Validation — Focal Loss + 诊断池化 + EMA
# ============================================================

# ---- EMA ----
class EMAModel:
    """Exponential Moving Average of model weights."""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self):
        """Replace model params with EMA params (for validation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original model params."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self):
        return {'decay': self.decay, 'shadow': self.shadow}

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']


# ---- Training ----
def train_epoch(model, loader, optimizer, criterion, scaler, epoch, ema=None):
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

            if ema is not None:
                ema.update()

        total_loss += loss.item() * grad_accum
        n_batches += 1

        if IS_MAIN and bi % 50 == 0:
            slots_present = mask.sum(dim=1).mean().item()
            print(f'  Epoch {epoch:3d} [{bi:4d}/{len(loader):4d}] '
                  f'loss={loss.item()*grad_accum:.4f} | slots={slots_present:.1f}/6',
                  flush=True)

    return total_loss / max(n_batches, 1)


# ---- Validation (with 7-window TTA + diagnostic pooling) ----
@torch.no_grad()
def validate_epoch(model, loader, criterion_hard):
    model.eval()

    n_windows = CFG['cache_slices'] - CFG['group_size'] + 1  # 7

    all_probs, all_labels, all_masks, all_uids = [], [], [], []
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        slots_full = batch['slots'].to(DEVICE, non_blocking=True)  # [B, 6, 9, H, W]
        mask = batch['mask'].to(DEVICE, non_blocking=True)          # [B, 6]
        labels = batch['labels'].to(DEVICE, non_blocking=True)      # [B, 12]
        val_masks = batch['val_masks'].to(DEVICE, non_blocking=True)  # [B, 12]
        uids = batch['study_uid']

        B = slots_full.shape[0]

        # ★ 损失只在中间窗口上算（省算力）
        mid_start = (CFG['cache_slices'] - CFG['group_size']) // 2  # 3
        slots_mid = slots_full[:, :, mid_start:mid_start + CFG['group_size']]  # [B, 6, 3, H, W]
        logits_mid = model(slots_mid, mask)
        active = val_masks > 0.5
        if active.any():
            loss_val = F.binary_cross_entropy_with_logits(
                logits_mid[active], labels[active], reduction='mean')
            total_loss += loss_val.item()
        n_batches += 1

        # ★ 7窗口 TTA + 诊断池化：单次批量前向传播
        window_list = [slots_full[:, :, w:w + CFG['group_size']] for w in range(n_windows)]
        slots_flat = torch.cat(window_list, dim=0)     # [B*7, 6, 3, H, W]
        mask_flat = mask.repeat(n_windows, 1)           # [B*7, 6]
        logits_flat = model(slots_flat, mask_flat)      # [B*7, C]
        logits_windows = logits_flat.reshape(B, n_windows, -1)  # [B, 7, C]
        probs_tta = diagnostic_pool(logits_windows)     # [B, C]

        all_probs.append(probs_tta.cpu())
        all_labels.append(labels.cpu())
        all_masks.append(val_masks.cpu())
        all_uids.extend(uids)

    probs_all = torch.cat(all_probs, dim=0).numpy()
    labels_all = torch.cat(all_labels, dim=0).numpy()
    masks_all = torch.cat(all_masks, dim=0).numpy()

    # Per-class AUC on labeled studies only
    aucs = []
    per_class = {}
    for i, c in enumerate(TARGET_COLUMNS):
        labeled_idx = masks_all[:, i] > 0.5
        n_labeled = int(labeled_idx.sum())
        metrics = {'auc': float('nan'), 'n_pos': 0, 'n_total': n_labeled}

        if n_labeled > 1:
            y_true = labels_all[:, i][labeled_idx]
            y_prob = probs_all[:, i][labeled_idx]
            n_pos = int(y_true.sum())
            metrics['n_pos'] = n_pos
            if n_pos > 0 and n_pos < n_labeled:
                try:
                    metrics['auc'] = float(roc_auc_score(y_true, y_prob))
                    aucs.append(metrics['auc'])
                except Exception:
                    pass
        per_class[c] = metrics

    return {
        'loss': total_loss / max(n_batches, 1),
        'macro_auc': float(np.mean(aucs)) if aucs else 0.0,
        'per_class': per_class,
        'probs': probs_all, 'labels': labels_all,
        'uids': all_uids,
    }


def print_validation_summary(val_metrics):
    print(f'\n  {"Class":<20s} {"AUC":>7s} {"Pos":>5s}')
    print(f'  {"-"*20} {"-"*7} {"-"*5}')
    for c in TARGET_COLUMNS:
        m = val_metrics['per_class'][c]
        auc_str = f'{m["auc"]:.3f}' if not math.isnan(m['auc']) else '  N/A  '
        print(f'  {c:<20s} {auc_str:>7s} {m["n_pos"]:5d}')
    print(f'  {"-"*20} {"-"*7} {"-"*5}')
    print(f'  {"Macro AUC":<20s} {val_metrics["macro_auc"]:7.3f}')
    print()


def save_validation_report(val_metrics, output_dir, epoch=None, is_best=False):
    out = Path(output_dir)
    rows = []
    for c in TARGET_COLUMNS:
        m = val_metrics['per_class'][c]
        rows.append({'class': c, 'auc': m['auc'], 'n_pos': m['n_pos'],
                     'n_total': m['n_total']})
    report_df = pd.DataFrame(rows)
    report_df['macro_auc'] = val_metrics['macro_auc']
    report_df['val_loss'] = val_metrics['loss']

    tag = '_best' if is_best else f'_epoch{epoch}'
    report_df.to_csv(out / f'validation_report{tag}.csv', index=False)
    return report_df

print('Training functions v4 ready (EMA + FocalLoss).')
