# ============================================================
# v3: Sanity check — verify multi-view pipeline
# ============================================================
if IS_MAIN:
    batch = next(iter(train_loader))

    print(f'Slots shape:      {batch["slots"].shape}')       # [B, 6, 3, 224, 224]
    print(f'Mask shape:       {batch["mask"].shape}')        # [B, 6]
    print(f'Prob targets:     {batch["prob_targets"].shape}') # [B, 12]
    print(f'Weights:          {batch["weights"].shape}')      # [B, 12]
    print(f'Soft masks:       {batch["soft_masks"].shape}')   # [B, 12]
    print(f'Study UIDs:       {batch["study_uid"][:3]}')

    m = batch['mask']
    slots_present = m.sum(dim=1)
    print(f'\nSlots present per study: min={slots_present.min().item():.0f} '
          f'max={slots_present.max().item():.0f} mean={slots_present.float().mean().item():.1f}')

    p = batch['prob_targets']
    w = batch['weights']
    sm = batch['soft_masks']
    print(f'Soft label stats:')
    print(f'  prob   in [{p.min():.3f}, {p.max():.3f}], mean={p.mean():.3f}')
    print(f'  weight in [{w.min():.3f}, {w.max():.3f}], mean={w.mean():.3f}')
    print(f'  mask   % active: {(sm == 1).float().mean()*100:.1f}%')

    # Forward pass
    with torch.no_grad():
        out = model(batch['slots'].to(DEVICE), batch['mask'].to(DEVICE))
    print(f'\nForward pass:')
    print(f'  Output shape: {out.shape}')  # [B, 12]
    print(f'  Output range: [{out.min().item():.3f}, {out.max().item():.3f}]')

    # Test loss
    loss = criterion(
        out,
        batch['prob_targets'].to(DEVICE),
        batch['weights'].to(DEVICE),
        batch['soft_masks'].to(DEVICE),
    )
    print(f'  Soft BCE loss: {loss.item():.4f}')

    # Verify model output is reasonable
    probs_test = torch.sigmoid(out).cpu().numpy()
    print(f'  Pred prob range: [{probs_test.min():.3f}, {probs_test.max():.3f}]')
    print(f'  Pred prob mean: {probs_test.mean():.3f}')
    print(f'\n  Pipeline OK — ready for training!')
