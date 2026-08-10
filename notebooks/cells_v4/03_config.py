# ============================================================
# v4: Configuration — 全量数据 + 物理裁剪 + 侧性归一化 + 诊断池化 + EMA + Focal Loss
# ============================================================

TARGET_COLUMNS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA',
    'Effusion', 'Synovitis', "Baker's",
    'Contusion', 'Fracture',
]
N_CLASSES = len(TARGET_COLUMNS)

# Soft-label columns
PROB_COLS   = [f'prob_{c}' for c in TARGET_COLUMNS]
WEIGHT_COLS = [f'weight_{c}' for c in TARGET_COLUMNS]
MASK_COLS   = [f'mask_{c}' for c in TARGET_COLUMNS]

# ---- 6 Clinical Slots ----
SLOTS = [
    ("SAG_FLUID_FS",   "Sagittal", True,  True),
    ("COR_FLUID_FS",   "Coronal",  True,  True),
    ("AX_FLUID_FS",    "Axial",    True,  True),
    ("SAG_FLUID_NOFS", "Sagittal", True,  False),
    ("COR_T1",         "Coronal",  False, False),
    ("SAG_T1",         "Sagittal", False, False),
]
N_SLOT = len(SLOTS)

# ---- Anatomical Priors ----
SLOT_PRIORS = {
    "ACL": (0, 3, 5), "MCL": (1, 4),
    "Medial Meniscus": (0, 1, 3, 4), "Lateral Meniscus": (0, 1, 3, 4),
    "Medial OA": (1, 4, 5), "Lateral OA": (1, 4, 5),
    "PF OA": (0, 2, 5), "Effusion": (0, 2), "Synovitis": (0, 2),
    "Baker's": (0,), "Contusion": (0, 1, 2), "Fracture": (0, 1, 2, 4, 5),
}

# ---- Diagnostic-specific TTA pooling ----
# 局部病灶用 max（保留最强信号），ACL/MCL 用 top2，弥漫性病变用 mean
DIAG_POOL = {
    "Fracture": "max", "Contusion": "max",
    "Medial Meniscus": "max", "Lateral Meniscus": "max",
    "Baker's": "max",
    "ACL": "top2", "MCL": "top2",
    # 其余（OA, Effusion, Synovitis）默认 mean
}

CFG = {
    # --- Paths ---
    'comp_input':   '/kaggle/input/competitions/rsna-knee-abnormality-detection',
    'pseudo_input': '/kaggle/input/datasets/easoncyy/rsna-knee-soft-labels',
    'dicom_subdir': 'train_series',
    'output_dir':   '/kaggle/working',

    # --- Data ---
    'image_size': 224,
    'crop_mm': 160.0,            # ★ 物理裁剪：固定 160mm FOV
    'cache_slices': 9,
    'group_size': 3,             # 3 adjacent slices → RGB channels
    'center_pct': (0.2, 0.8),

    # --- Model ---
    'dinov2_variant': 'vit_small_patch14_dinov2.lvd142m',
    'cls_dim': 384,
    'feature_dim': 1152,
    'slot_hidden': 256,
    'num_classes': 12,
    'unfreeze_layers': 6,
    'dropout': 0.2,

    # --- Training ---
    'batch_size': 6,
    'grad_accum_steps': 2,
    'epochs': 40,
    'lr': 2e-4,
    'backbone_lr': 1e-5,
    'weight_decay': 1e-4,
    'lr_t0': 15,
    'lr_t_mult': 2,
    'lr_eta_min': 1e-6,
    'grad_clip': 1.0,
    'early_stop_patience': 15,
    'mixed_precision': True,
    'num_workers': 2,

    # --- ★ v4 新增 ---
    'ema_decay': 0.999,          # ★ EMA 权重平均
    'focal_alpha': 0.25,         # ★ Focal Loss alpha (对正类的关注度)
    'focal_gamma': 2.0,          # ★ Focal Loss gamma (对难例的关注度)
    'use_self_distill': True,    # ★ 伪标签自蒸馏
    'diag_pool_train': True,     # ★ 训练时也做诊断池化
    'hdr_threads': 8,            # ★ DICOM 并行读取线程数
    'pix_threads': 4,            # ★ 像素解码并行线程数
}

# Device
N_GPUS = torch.cuda.device_count()
DEVICE = torch.device('cuda' if N_GPUS > 0 else 'cpu')

if IS_MAIN:
    print(f'GPUs: {N_GPUS} | Device: {DEVICE}')
    print(f'--- v4: Full Data + Physical Crop + Laterality + Diag Pool + EMA + Focal ---')
    for k, v in CFG.items():
        print(f'  {k}: {v}')
