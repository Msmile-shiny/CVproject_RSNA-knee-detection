# ============================================================
# v3: Build model, optimizer, scheduler
# ============================================================

if IS_MAIN: print('Loading DINOv2 backbone...')

dinov2_backbone = timm.create_model(
    CFG['dinov2_variant'], pretrained=True, num_classes=0,
    img_size=CFG['image_size'],
)

model = MultiViewModel(
    dinov2_model=dinov2_backbone,
    n_slots=N_SLOT,
    cls_dim=CFG['cls_dim'],
    n_classes=CFG['num_classes'],
    slot_hidden=CFG['slot_hidden'],
    dropout=CFG['dropout'],
    unfreeze_layers=CFG['unfreeze_layers'],
).to(DEVICE)

if N_GPUS > 1:
    model = nn.DataParallel(model)
    print(f'[Model] DataParallel across {N_GPUS} GPUs')

# Separate LR for backbone vs head
backbone_params = []
head_params = []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if 'dinov2' in name:
        backbone_params.append(p)
    else:
        head_params.append(p)

optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': CFG['backbone_lr']},
    {'params': head_params, 'lr': CFG['lr']},
], weight_decay=CFG['weight_decay'])

if IS_MAIN:
    n_backbone = sum(p.numel() for p in backbone_params)
    n_head = sum(p.numel() for p in head_params)
    print(f'Optimizer: backbone {n_backbone/1e6:.1f}M params @ lr={CFG["backbone_lr"]}')
    print(f'           head     {n_head/1e6:.1f}M params @ lr={CFG["lr"]}')

criterion = WeightedSoftBCELoss()
criterion_val = HardBCELoss()

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=CFG['lr_t0'], T_mult=CFG['lr_t_mult'], eta_min=CFG['lr_eta_min'])

scaler = torch.amp.GradScaler('cuda') if CFG['mixed_precision'] else None
