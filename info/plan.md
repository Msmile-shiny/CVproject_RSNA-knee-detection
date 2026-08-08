# RSNA 2026 膝关节异常检测 — 图像模型训练计划书

> **竞赛**: RSNA 2026 Knee Abnormality Detection (Kaggle)
> **硬件**: RTX 5060 8GB VRAM (本地) → 云端扩展
> **截止**: 2026-10-22（约 11 周）
> **创建日期**: 2026-08-06

---

## 目录

1. [竞赛分析：评估指标与损失函数](#1-竞赛分析评估指标与损失函数)
2. [Mini 数据集构建方案](#2-mini-数据集构建方案)
3. [阶段一：2D 切片级模型](#3-阶段一2d-切片级模型)
4. [阶段二：2.5D 三平面融合模型](#4-阶段二25d-三平面融合模型)
5. [阶段三：轻量 3D 模型](#5-阶段三轻量-3d-模型)
6. [总体时间线与里程碑](#6-总体时间线与里程碑)

---

## 1. 竞赛分析：评估指标与损失函数

### 1.1 任务类型判断

根据 Kaggle 页面截图中的描述：

> *"Submissions are evaluated by the **average area under the ROC curve** between the predicted confidence scores and the observed targets **across the twelve targets**."*

> *"The final score is, in other words, the **macro-averaged AUC ROC**."*

| 维度 | 分析结论 |
|------|----------|
| **任务性质** | **多标签分类**（Multilabel Classification） |
| **标签数量** | **12 个二分类目标** |
| **输出形式** | 对每个目标输出一个 0~1 的置信度分数（confidence score） |
| **评估指标** | **Macro-averaged AUC ROC**（12 个目标的 AUC 取算术平均） |
| **数据模态** | MRI 图像 (DICOM) + 多语言放射学报告 |

### 1.2 AUC ROC 的本质

AUC ROC 衡量的是：随机抽取一个正样本和一个负样本，模型给正样本的分数高于负样本的概率。

因此：
- AUC 关注的是"相对排序"而非"绝对概率"
- 校准度有帮助但不是核心
- 类别不均衡对 AUC 的影响天然小于对 Accuracy/F1 的影响
- macro averaging 意味着罕见类的排序质量同样重要

### 1.3 损失函数推荐策略：渐进式升级

**Step 1 — BCEWithLogitsLoss + Label Smoothing (第一周)**
- pos_weight: 根据 Mini 数据集的 12 个类别频率动态计算
- label_smoothing: 0.05

**Step 2 — Asymmetric Loss (第二周起，主力)**
- γ_neg=4, γ_pos=1
- clip=0.05
- 参考: "Asymmetric Loss For Multi-Label Classification" (ICCV 2021)

**Step 3 — SmoothAUC Loss (第三周后)**
- 直接优化 AUC 的可微近似

### 1.4 验证策略

- Patient-level split (铁律)
- StratifiedGroupKFold (K=5)
- 监控: Val Loss, Per-class AUC (12), Macro AUC, AUC std

---

## 2. Mini 数据集构建方案

### 2.1 核心原则

| 原则 | 原因 |
|------|------|
| Patient-level 抽样 | 同一患者的多个序列/切片高度相关 |
| 12 类全覆盖 | Macro AUC = 12 类的算术平均 |
| 站点覆盖 | 16 个站点, 扫描协议/机器型号不同 |
| 序列覆盖 | 不同序列反映不同组织对比度 |

### 2.2 目标规模

| 场景 | Exam 数 | 说明 |
|------|:-------:|------|
| 最小可训练 | 150-200 | 仅适合 2D 快速验证 |
| **推荐目标** | **300-400** | 平衡训练效果和迭代速度 |
| 上限 | 500 | 如果全量分布极度不均衡 |

---

## 3. 阶段一：2D 切片级模型

### 3.1 目标

- 将 3D MRI 拆解为 2D 切片，建立多标签分类基线
- 验证 DICOM → Tensor → 12-dim 置信度分数的全流程管线
- 为后续阶段提供预训练权重

### 3.2 Backbone 推荐

**ResNet50 + EfficientNetV2-S 双模型集成**

| 维度 | ResNet50 | EfficientNetV2-S |
|------|:--------:|:----------------:|
| 架构范式 | 标准卷积 | Fused-MBConv |
| 输入尺寸 | 256×256 | 384×384 |
| 8GB 下 Batch | ~32-64 | ~8-12 |

### 3.3 阶段一训练配置

| 超参数 | ResNet50 | EfficientNetV2-S |
|--------|----------|-------------------|
| 输入尺寸 | 256×256 | 384×384 |
| Batch Size | 32 | 12 |
| 优化器 | AdamW | AdamW |
| 学习率 | 1e-4 | 1e-4 |
| Epochs | 50 | 50 |
| 混合精度 | AMP (fp16) | AMP (fp16) |
| Label Smoothing | 0.05 | 0.05 |

### 3.4 集成策略

1. ResNet50 独立训练 → 保存最佳权重
2. EfficientNetV2-S 独立训练 → 保存最佳权重
3. 冻结两个 backbone, 只训练融合层 + Head (5 epochs)
4. 全组件解冻, 用 1/10 学习率联合微调 (10 epochs)

---

## 4. 阶段二：2.5D 三平面融合模型

- 三平面 (Axial/Coronal/Sagittal) 各独立 backbone → Cross-Attention 融合
- ResNet50 (×3, 独立权重)
- Batch Size: 8-12
- 学习率: 5e-5

---

## 5. 阶段三：轻量 3D 模型

- ResNet3D-18, 输入 128×128×32
- Batch=1, 梯度累积 4 步
- AMP + Gradient Checkpointing 必须
- 目的: 验证 3D 管线可跑通，为上云全量训练做准备

---

## 6. 总体时间线与里程碑

```
Week 1  ─ EDA + Mini 集构建
Week 2  ─ ResNet50 baseline
Week 3  ─ EfficientNetV2-S + 集成
Week 4  ─ Grad-CAM + 阶段一完成
Week 5-6 ─ 阶段二 (2.5D 三平面)
Week 7-8 ─ 阶段三 (3D 本地跑通)
Week 9-10 ─ 云端全量训练
Week 11 ─ 最终提交
```

---

## 附录 A: 项目目录结构

```
CVproject_RSNA-knee-detection/
├── plan.md
├── data/
│   ├── metadata/            # 原始 CSV（只读）
│   └── processed/           # 预处理 npy（从 Kaggle 下载解压）
├── notebooks/
│   ├── 01_eda_label_distribution.ipynb
│   ├── 02_stratified_sampling.ipynb
│   └── kaggle_preprocess.ipynb
├── outputs/                  # 模型权重 & 训练日志
├── src/
│   ├── data/
│   │   ├── dicom_loader.py
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── resnet.py
│   │   ├── efficientnet.py
│   │   ├── ensemble.py
│   │   ├── triplane.py
│   │   └── resnet3d.py
│   ├── losses/
│   │   ├── bce_smoothing.py
│   │   ├── focal_loss.py
│   │   ├── asymmetric_loss.py
│   │   └── smooth_auc.py
│   ├── metrics.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── configs/
└── requirements.txt
```

## 附录 B: 环境依赖

```
torch>=2.1.0
torchvision>=0.16.0
monai>=1.3.0
pydicom>=2.4.0
numpy>=1.24.0
pandas>=2.0.0
albumentations>=1.3.0
timm>=0.9.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
opencv-python>=4.8.0
```

---

> **核心约束速查**：
> - 评估指标：**12 类 macro-averaged AUC ROC**
> - 硬件上限：**RTX 5060 8GB VRAM**
> - 推荐主力损失：**ASL + SmoothAUC 联合**
> - 推荐 backbone：**ResNet50 + EfficientNetV2-S 集成**
> - 数据分离铁律：**Patient-level split**
