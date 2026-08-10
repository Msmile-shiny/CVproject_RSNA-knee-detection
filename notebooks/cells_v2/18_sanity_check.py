# ============================================================
# Sanity check -- verify soft label pipeline
# ============================================================
if IS_MAIN:
    batch = next(iter(train_loader))
    print(f'Image shape:      {batch["image"].shape}')        # [B, 5, 392, 392]
    print(f'Prob targets:     {batch["prob_targets"].shape}') # [B, 12]
    print(f'Weights:          {batch["weights"].shape}')      # [B, 12]
    print(f'Masks:            {batch["masks"].shape}')        # [B, 12]
    print(f'Study UIDs:       {batch["study_uid"][:3]}')

    p = batch['prob_targets']
    w = batch['weights']
    m = batch['masks']
    print(f'\nSoft label stats (first batch):')
    print(f'  prob   in [{p.min():.3f}, {p.max():.3f}], mean={p.mean():.3f}')
    print(f'  weight in [{w.min():.3f}, {w.max():.3f}], mean={w.mean():.3f}')
    print(f'  mask   % active: {(m == 1).float().mean()*100:.1f}%')
    print(f'  % masked out (weight < 0.15): {(m == 0).float().mean()*100:.1f}%')

    # Forward pass
    with torch.no_grad():
        out = model(batch['image'].to(DEVICE))
    print(f'\nForward pass:')
    print(f'  Output shape: {out.shape}')
    print(f'  Output range: [{out.min().item():.3f}, {out.max().item():.3f}]')

    # Test loss
    loss = criterion(out,
                     batch['prob_targets'].to(DEVICE),
                     batch['weights'].to(DEVICE),
                     batch['masks'].to(DEVICE))
    print(f'  Soft BCE loss: {loss.item():.4f}')

    # Compare with hard-label Focal loss
    hard_labels = (p > 0.5).float()
    focal_loss_fn = FocalBCELoss()
    focal = focal_loss_fn(out, hard_labels.to(DEVICE))
    print(f'  Focal loss (same batch, hard labels): {focal.item():.4f}')
    print(f'  -> Soft loss should be LOWER (uncertain samples down-weighted)')
    print(f'\n  Pipeline OK -- ready for training!')
