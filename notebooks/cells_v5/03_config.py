# ============================================================
# v5: Configuration — 288px/130mm 奈奎斯特分辨率 + v5 融合软标签 (teacher-student)
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
    # ★ v5 融合标签数据集 (本地 scripts/build_v5_labels.py 生成 v5_labels.csv 后上传)
    'label_input':  '/kaggle/input/datasets/easoncyy/rsna-knee-v5-labels',
    'dicom_subdir': 'train_series',
    'output_dir':   '/kaggle/working',

    # --- Data ---
    'image_size': 288,             # ★ v5: 288px@130mm = 0.451mm/px 满足奈奎斯特
                                   #   (v4: 224px@160mm = 0.714mm/px 不满足)
                                   #   RAM 缓存 ~19.7GB + 运行时 ~4GB → 需要 T4x2 (~29GB 系统内存);
                                   #   P100 (16GB RAM) 请改用 256 + cache_slices 7 (12.1GB 缓存, 0.508mm/px)
    'crop_mm': 130.0,              # ★ v5: 130mm FOV (膝关节 ~130mm, 无浪费像素)
    'cache_slices': 9,
    'group_size': 3,               # 3 adjacent slices → RGB channels
    'center_pct': (0.2, 0.8),

    # --- Model ---
    'dinov2_variant': 'vit_small_patch14_dinov2.lvd142m',
    'dinov2_weights': '/kaggle/input/rsna-dinov2-weights/dinov2_vits14.pth',  # ★ 竞赛禁网，权重打包为 Dataset
    'cls_dim': 384,
    'feature_dim': 1152,
    'slot_hidden': 256,
    'num_classes': 12,
    'unfreeze_layers': 6,
    'dropout': 0.2,

    # --- Training ---
    'batch_size': 6,
    'grad_accum_steps': 2,
    'epochs': 30,                  # ★ v5: 40→30 (288px 计算量 1.65×, 靠标签质量补偿)
    'lr': 2e-4,
    'backbone_lr': 1e-5,
    'weight_decay': 1e-4,
    'lr_t0': 15,
    'lr_t_mult': 2,
    'lr_eta_min': 1e-6,
    'grad_clip': 1.0,
    'early_stop_patience': 12,
    'mixed_precision': True,
    'num_workers': 2,

    # --- ★ v5 ---
    'ema_decay': 0.999,            # EMA 权重平均 (v4 沿用)
    'max_train_minutes': 420,      # ★ 训练墙钟保护 (Kaggle 9h 会话上限内留出推理时间)
    'diag_pool_train': True,       # 训练时也做诊断池化
    'hdr_threads': 8,              # DICOM 并行读取线程数
    'pix_threads': 4,              # 像素解码并行线程数
}

# Device
N_GPUS = torch.cuda.device_count()
DEVICE = torch.device('cuda' if N_GPUS > 0 else 'cpu')

if IS_MAIN:
    print(f'GPUs: {N_GPUS} | Device: {DEVICE}')
    print(f'--- v5: 288px/130mm + Fused Soft Labels (text×OOF teacher) + Confidence-weighted BCE ---')
    for k, v in CFG.items():
        print(f'  {k}: {v}')
