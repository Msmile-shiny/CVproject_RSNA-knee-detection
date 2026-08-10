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
    'image_size': 392,
    'slice_count': 5,
    'center_stride': 3,

    # --- Model ---
    'dinov2_variant': 'vit_small_patch14_dinov2.lvd142m',
    'cls_dim': 384,
    'spa_channels': 64,
    'num_classes': 12,

    # --- v2: Unfreeze strategy ---
    'unfreeze_layers': 6,       # Number of last DINOv2 layers to unfreeze (0=all frozen)

    # --- Training ---
    'batch_size': 16,           # reduced: unfrozen backbone uses more VRAM
    'epochs': 50,
    'lr': 2e-4,
    'backbone_lr': 1e-5,        # v2: lower LR for unfrozen backbone layers
    'weight_decay': 1e-4,
    'lr_t0': 15,                # increased: more steps between restarts
    'lr_t_mult': 2,
    'lr_eta_min': 1e-6,
    'dropout': 0.2,             # increased slight regularization
    'head_dropout': 0.3,
    'grad_clip': 1.0,
    'early_stop_patience': 10,
    'mixed_precision': True,
    'num_workers': 4,
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
