# ============================================================
# v5: Training Loop — EMA + Early Stopping + 墙钟保护 + 最佳模型保存
# ============================================================

output_dir = Path(CFG['output_dir'])
(output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)

best_auc = 0.0
best_epoch = 0
patience_counter = 0
history = []

criterion_val = nn.BCEWithLogitsLoss(reduction='mean')

print(f'Training: {len(train_ds)} studies, {CFG["epochs"]} epochs')
print(f'  Effective batch = {CFG["batch_size"]} × {N_GPUS} GPU × {CFG["grad_accum_steps"]} accum = '
      f'{CFG["batch_size"] * max(N_GPUS, 1) * CFG["grad_accum_steps"]}')
print(f'  WeightedSoftBCE (confidence-weighted) | EMA({CFG["ema_decay"]})')
print(f'  ★ {CFG["image_size"]}px / Physical crop: {CFG["crop_mm"]}mm | Laterality norm | Spatial ordering')
print(f'  ★ Wall-clock budget: {CFG["max_train_minutes"]}min (Kaggle 9h 会话上限)')
print(f'  ★ Seed: {CFG["seed"]} | RadImageNet R50 frozen → checkpoint {CKPT_NAME} (v6 异架构成员)')
print()

t_start = time.time()

for epoch in range(1, CFG['epochs'] + 1):
    t_epoch = time.time()

    # Train
    train_loss = train_epoch(
        model, train_loader, optimizer, criterion, scaler, epoch, ema=ema)
    scheduler.step()

    # Validate (with EMA weights)
    if ema is not None:
        ema.apply_shadow()
    val_metrics = validate_epoch(model, val_loader, criterion_val)
    if ema is not None:
        ema.restore()

    val_loss = val_metrics['loss']
    val_auc = val_metrics['macro_auc']

    elapsed = time.time() - t_epoch
    eta_total = (time.time() - t_start) / epoch * (CFG['epochs'] - epoch) / 60

    history.append({
        'epoch': epoch, 'train_loss': train_loss,
        'val_loss': val_loss, 'macro_auc': val_auc,
    })

    if IS_MAIN:
        print(f'--- Epoch {epoch:3d} | '
              f'train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | '
              f'val_auc={val_auc:.4f} | {elapsed:.0f}s | ~{eta_total:.0f}min left'
              f' | RAM={_ram_gb():.1f}GB ---')
        print_validation_summary(val_metrics)

    # Save best
    is_best = val_auc > best_auc
    if is_best:
        best_auc = val_auc
        best_epoch = epoch
        patience_counter = 0

        # 获取实际模型（去掉 DataParallel wrapper; 13b 探针可能已拆掉 DP）
        save_model = model.module if isinstance(model, nn.DataParallel) else model
        # ★ 内存防御: 显式转 CPU 再保存 (避免 GPU state_dict clone + pickle 中间峰值)
        ckpt = {
            'epoch': epoch,
            'model': {k: v.cpu() for k, v in save_model.state_dict().items()},
            'ema': None,
            'auc': best_auc,
            'config': CFG,
            'targets': TARGET_COLUMNS,
            'slots': SLOTS,
        }
        if ema is not None:
            ema_sd = ema.state_dict()
            ema_sd['shadow'] = {k: v.cpu() for k, v in ema_sd['shadow'].items()}
            ckpt['ema'] = ema_sd
        torch.save(ckpt, output_dir / 'checkpoints' / CKPT_NAME)

        save_validation_report(val_metrics, output_dir, epoch=epoch, is_best=True)
        print(f'  ★ Best model saved (epoch={epoch}, AUC={best_auc:.4f})')
    else:
        patience_counter += 1

    # 定期保存
    if epoch % 10 == 0:
        save_model = model.module if isinstance(model, nn.DataParallel) else model
        per_ckpt = {
            'epoch': epoch,
            'model': {k: v.cpu() for k, v in save_model.state_dict().items()},
            'ema': None,
            'auc': val_auc,
            'config': CFG, 'targets': TARGET_COLUMNS, 'slots': SLOTS,
        }
        if ema is not None:
            per_ema = ema.state_dict()
            per_ema['shadow'] = {k: v.cpu() for k, v in per_ema['shadow'].items()}
            per_ckpt['ema'] = per_ema
        torch.save(per_ckpt, output_dir / 'checkpoints' / f'model_epoch{epoch}.pt')

    # Early stopping
    if patience_counter >= CFG['early_stop_patience']:
        print(f'Early stopping at epoch {epoch} (patience={CFG["early_stop_patience"]})')
        break

    # ★ 内存防御: 长会话 (Kaggle 9h) 下每 epoch 回收 Python 层碎片
    gc.collect()

    # ★ 墙钟保护: 训练超过预算即优雅停止 (best checkpoint 已在上面保存)
    elapsed_min = (time.time() - t_start) / 60
    if elapsed_min > CFG['max_train_minutes']:
        print(f'Wall-clock budget reached ({elapsed_min:.0f}min > '
              f'{CFG["max_train_minutes"]}min) — stopping after epoch {epoch}')
        break

    # ★ RAM 保护: 接近 29GB 上限时优雅停止 (三跑撞顶死亡教训;
    #   best 已保存, 停止后继续执行 cell 17 推理 → 本跑仍产出 gold CSV + submission)
    if _ram_gb() > 27.0:
        print(f'RAM guard: {_ram_gb():.1f}GB > 27GB — stopping after epoch {epoch} '
              f'(best checkpoint saved, 继续 cell 17 推理)')
        break

# ---- Save training history ----
hist_df = pd.DataFrame(history)
hist_df.to_csv(output_dir / 'training_history.csv', index=False)

total_time = time.time() - t_start
print(f'\n{"="*60}')
print(f'Training complete: {total_time/60:.0f}min | Best AUC={best_auc:.4f} @ epoch {best_epoch}')
print(f'Best model: {output_dir / "checkpoints" / CKPT_NAME}')
print(f'{"="*60}')
