# ============================================================
# Configuration -- v2: Soft Labels + Partial Unfreeze
# ============================================================

TARGET_COLUMNS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA',
    'Effusion', 'Synovitis', "Baker's",
    'Contusion', 'Fracture',
]

# Soft-label column names (from calibrated CSV)
PROB_COLS   = [f'prob_{c}' for c in TARGET_COLUMNS]
WEIGHT_COLS = [f'weight_{c}' for c in TARGET_COLUMNS]
MASK_COLS   = [f'mask_{c}' for c in TARGET_COLUMNS]

CFG = {
    # --- Paths (Kaggle) ---
    'comp_input':   '/kaggle/input/competitions/rsna-knee-abnormality-detection',
    # v2: upload pseudo_labels_calibrated.csv as a Kaggle Dataset named "rsna-knee-soft-labels"
    'pseudo_input': '/kaggle/input/datasets/easoncyy/rsna-knee-soft-labels',
    'dicom_subdir': 'train_series',
    'output_dir':   '/kaggle/working',

    # --- Data ---
    'image_size': 392,          # RESTORED: 280 loses detail for small lesions
    'slice_count': 5,
    'center_stride': 3,

    # --- Model ---
    'dinov2_variant': 'vit_small_patch14_dinov2.lvd142m',
    'cls_dim': 384,
    'spa_channels': 64,
    'num_classes': 12,

    # --- v2: Unfreeze strategy ---
    # Start conservative (2), verify no overfitting, then increase to 4→6.
    # Soft labels slow down memorization but don't prevent it.
    # More layers = more capacity to memorize the ~22% wrong pseudo-labels.
    'unfreeze_layers': 2,

    # --- Training ---
    'batch_size': 8,            # reduced for 392 image_size + VRAM safety
    'grad_accum_steps': 2,      # effective batch = 8 * 2 GPUs * 2 accum = 32
    'grad_accum_steps': 1,      # set >1 for larger effective batch without more VRAM
    'epochs': 50,
    'lr': 2e-4,
    'backbone_lr': 1e-5,        # v2: lower LR for unfrozen backbone layers
    'weight_decay': 1e-4,
    'lr_t0': 15,
    'lr_t_mult': 2,
    'lr_eta_min': 1e-6,
    'dropout': 0.2,
    'head_dropout': 0.3,
    'grad_clip': 1.0,
    'early_stop_patience': 10,
    'mixed_precision': True,
    'num_workers': 2,           # fewer workers = less CPU RAM pressure
    'channels_last': False,     # DISABLED: not worth risk with DINOv2 ViT
    'use_torch_compile': False,  # DISABLED: conflicts with DataParallel (attr lookup fails)
}

# Device setup
N_GPUS = torch.cuda.device_count()
DEVICE = torch.device('cuda' if N_GPUS > 0 else 'cpu')
IS_MAIN = True

if IS_MAIN:
    print(f'GPUs: {N_GPUS} | Device: {DEVICE}')
    print(f'--- v2: Soft Labels + Unfreeze {CFG["unfreeze_layers"]} layers ---')
    for k, v in CFG.items():
        print(f'  {k}: {v}')
