# ============================================================
# v6a: Configuration — RadImageNet R50 冻结编码器 @224 + v5 融合软标签
#   (v6 = 异架构集成; v6a = 第一成员: 架构+领域+分辨率三重多样性)
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
# 与 0.91 notebook 的 TTA_TARGET_POOL 逐项一致
DIAG_POOL = {
    "Fracture": "max", "Contusion": "max",
    "Medial Meniscus": "max", "Lateral Meniscus": "max",
    "Baker's": "max",
    "ACL": "top2", "MCL": "top2",
    # ★ 0.91 同款: 仅用无 jitter 原始视图平均
    #   (jitter TTA 开启时生效; 关闭时所有视图皆原始, 等价于 mean)
    "Synovitis": "original_mean",
    # 其余（OA, Effusion）默认 mean
}

# ---- Jitter TTA 增广 (0.91 notebook augment() 移植) ----
AUG_ROT_DEG = 8.0          # 旋转 ±8°
AUG_SCALE = 0.08           # 缩放 +[0, 8%]
AUG_SHIFT = 0.05           # 平移 ±5%
AUG_INTENSITY = 0.1        # 强度 ±10%
AUG_SEED = 42              # 增广视图固定种子（确定性, 跨验证/测试/提交可复现）

CFG = {
    # --- Paths ---
    'comp_input':   '/kaggle/input/competitions/rsna-knee-abnormality-detection',
    # ★ v5 融合标签数据集 (本地 scripts/build_v5_labels.py 生成 v5_labels.csv 后上传)
    'label_input':  '/kaggle/input/datasets/easoncyy/rsna-knee-v5-labels',
    'dicom_subdir': 'train_series',
    'output_dir':   '/kaggle/working',

    # --- Data ---
    'image_size': 224,             # ★ v6a: R50 原生分辨率 224px@130mm = 0.580mm/px
                                   #   (低于奈奎斯特 0.451 — 有意为之: 分辨率多样性成员;
                                   #   FOV 与 v5 相同, 窗口内容一致, 只是更粗采样)
                                   #   RAM 缓存 ~19.7GB×(224/288)² ≈ 11.9GB + 运行时 ~4GB
                                   #   → T4x2 稳妥; 单 T4 (13GB) 边缘
    'crop_mm': 130.0,              # FOV 与 v5 一致 (膝关节 ~130mm, 无浪费像素)
    'cache_slices': 7,             # ★ 内存防御: 9→7 (11.1→8.6GB RAM, 两次会话 ~112min 处死亡);
                                   #   TTA 窗口 7→5 (影响 ~0.005 可接受), cache 构建 71→~55min
    'group_size': 3,               # 3 adjacent slices → RGB channels
    'center_pct': (0.2, 0.8),

    # --- Model ---
    # ★ RadImageNet ResNet50 官方 notop 权重 (本地 scripts/convert_radimagenet_r50.py
    #   从官方 h5 转换 + bias 吸收进 BN, 上传为 Kaggle Dataset 后挂载; 默认 slug:
    #   rsna-radimagenet-r50/radimagenet_resnet50_notop.pt — 与上传目录名不一致时改这里)
    'rad_weights': '/kaggle/input/rsna-radimagenet-r50/radimagenet_resnet50_notop.pt',
    'feature_dim': 2048,           # ResNet50 GAP 特征 (layer4 → avgpool)
    'slot_hidden': 256,
    'num_classes': 12,
    'unfreeze_layers': 0,          # ★ v6a: 编码器全冻结 (只训 SlotHead; 多样性成员)
    'dropout': 0.2,

    # --- Training ---
    'batch_size': 6,
    'grad_accum_steps': 2,
    'epochs': 30,                  # 与 v5 同 (标签质量相同; R50 每 epoch 计算量远小于 ViT-s)
    'seed': 42,                    # 数据顺序/头初始化种子 (架构不同 → 与 v5s1 同 seed 不影响成员独立性)
    'lr': 2e-4,
    'backbone_lr': 1e-5,           # v6a 全冻结 → 无 backbone 参数组, 此值未使用 (13 处理空组)
    'weight_decay': 1e-4,
    'lr_t0': 15,
    'lr_t_mult': 2,
    'lr_eta_min': 1e-6,
    'grad_clip': 1.0,
    'early_stop_patience': 12,
    'mixed_precision': True,
    'num_workers': 0,              # ★ 内存防御: 0 = 主进程加载 (无 fork/COW/pinned 池, 内存模型最简)

    # --- ★ v5 ---
    'ema_decay': 0.999,            # EMA 权重平均 (v4 沿用)
    'max_train_minutes': 420,      # ★ 训练墙钟保护 (Kaggle 9h 会话上限内留出推理时间)
    'diag_pool_train': True,       # 训练时也做诊断池化
    'tta_jitter': True,            # ★ jitter TTA (0.91 移植): 每窗口额外 1 个确定性增广视图,
                                   #   视图平均后再窗口池化; 验证/推理成本 ×2 (训练不变)
    'hdr_threads': 8,              # DICOM 并行读取线程数
    'pix_threads': 4,              # 像素解码并行线程数
}

# ---- 全局随机种子 — seed 自集成的成员独立性来源 ----
#   换 CFG['seed'] = 新成员: 训练顺序/头初始化/优化路径全部不同 → 半独立
import random
random.seed(CFG['seed'])
np.random.seed(CFG['seed'])
torch.manual_seed(CFG['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG['seed'])

SEED_TAG = 'rad'                          # ★ v6a 成员标签 (v6 融合端按文件名识别成员)
CKPT_NAME = f'best_model_{SEED_TAG}.pt'   # 15 保存 / 17 加载共用

# Device
N_GPUS = torch.cuda.device_count()
DEVICE = torch.device('cuda' if N_GPUS > 0 else 'cpu')

if IS_MAIN:
    print(f'GPUs: {N_GPUS} | Device: {DEVICE}')
    print(f'--- v6a: RadImageNet R50 frozen @224px/130mm + v5 Fused Soft Labels ---')
    for k, v in CFG.items():
        print(f'  {k}: {v}')
