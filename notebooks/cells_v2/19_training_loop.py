if IS_MAIN:
    print(f'\n{"="*60}')
    print(f'v2 Training -- Soft Labels + Unfreeze {CFG["unfreeze_layers"]} DINOv2 layers')
    print(f'Batch={CFG["batch_size"]} (x{N_GPUS} GPUs) | Train samples={len(train_ds):,}')
    print(f'Train studies (all): {len(set(s["study_uid"] for s in train_ds.samples)):,}')
    print(f'Val studies:         {len(val_ds):,}')
    print(f'{"="*60}\n')

best_auc = 0.0
best_epoch = 0
patience = 0
ckpt_dir = Path(CFG['output_dir']) / 'checkpoints'
ckpt_dir.mkdir(parents=True, exist_ok=True)
t_start = time.time()

history = []

for epoch in range(1, CFG['epochs'] + 1):
    t0 = time.time()

    train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, epoch)
    val_metrics = validate_epoch(model, val_loader, criterion)
    scheduler.step()

    if IS_MAIN:
        epoch_time = time.time() - t0
        elapsed = time.time() - t_start
        vram = torch.cuda.max_memory_allocated(DEVICE) / 1024**3
        torch.cuda.reset_peak_memory_stats(DEVICE)
        lr_now = optimizer.param_groups[0]['lr']

        print(f'\n-- Epoch {epoch:3d}/{CFG["epochs"]} --')
        print(f'  Train Loss: {train_loss:.4f}  |  Val Loss: {val_metrics["loss"]:.4f}')
        print(f'  Val Macro AUC: {val_metrics["macro_auc"]:.4f}  |  LR: {lr_now:.2e}')
        print(f'  Time: {epoch_time:.0f}s epoch | {elapsed/60:.0f}min total | VRAM: {vram:.1f}GB')

        print_validation_summary(val_metrics)

        # v2: Per-class threshold check every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            threshold_table = analyze_thresholds(val_metrics)

        history.append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_loss': val_metrics['loss'], 'macro_auc': val_metrics['macro_auc'],
        })

        # -- Checkpoint on improvement --
        current_auc = val_metrics['macro_auc']

        if current_auc > best_auc + 0.0005:
            best_auc = current_auc
            best_epoch = epoch
            patience = 0
            state = model.module.state_dict() if N_GPUS > 1 else model.state_dict()
            ckpt_path = ckpt_dir / 'best_model.pt'
            torch.save({'epoch': epoch, 'model': state, 'auc': best_auc, 'config': CFG}, ckpt_path)
            print(f'  >> Best model saved (AUC={best_auc:.4f})')

            # Full threshold analysis on best model
            threshold_table = analyze_thresholds(val_metrics)
            save_validation_report(val_metrics, CFG['output_dir'], epoch=epoch, is_best=True,
                                   thresholds=threshold_table)
        else:
            patience += 1
            if patience >= CFG['early_stop_patience']:
                print(f'\n  Early stopping triggered at epoch {epoch}')
                break

# -- Final Report --
if IS_MAIN:
    total_time = time.time() - t_start
    print(f'\n{"="*60}')
    print(f'v2 Training Complete')
    print(f'  Soft Labels + Unfreeze {CFG["unfreeze_layers"]} DINOv2 layers')
    print(f'  Best Val Macro AUC: {best_auc:.4f} (epoch {best_epoch})')
    print(f'  Total Time:         {total_time/3600:.1f} hours')
    print(f'{"="*60}')

    history_df = pd.DataFrame(history)
    history_df.to_csv(Path(CFG['output_dir']) / 'training_history.csv', index=False)
    print(f'\nOutput files in {CFG["output_dir"]}/:')
    print(f'  training_history.csv')
    print(f'  checkpoints/best_model.pt')
    print(f'  validation_report_best.csv')
    print(f'  validation_predictions_best.csv')
    print(f'{"="*60}')
