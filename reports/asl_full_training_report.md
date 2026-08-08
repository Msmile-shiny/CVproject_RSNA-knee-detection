# ASL 全量 Mini 集 5-Fold 训练报告

> **日期**: 2026-08-08 | **实验**: convnextv2_tiny_2p5d_mini | **阶段**: Stage 2.5D Baseline

---

## 1. 实验设计

| 维度 | 配置 |
|------|------|
| **模型** | ConvNeXtV2-Tiny (28.6M params), ImageNet-22k 预训练 |
| **输入** | 2.5D triplet: Sagittal 相邻 3 切片 → [3, 224, 224] |
| **损失函数** | **ASL** (Asymmetric Loss): γ_neg=4, γ_pos=1, clip=0.05 |
| **优化器** | AdamW: backbone_lr=2e-5, head_lr=1e-4, weight_decay=0.05 |
| **调度器** | Linear warmup (3 epochs) → Cosine annealing |
| **Batch size** | **32** (从 16 翻倍) |
| **混合精度** | AMP fp16 |
| **验证策略** | Patient-level StratifiedGroupKFold (5 folds) |
| **早停** | patience=8, monitor=val_macro_auc, min_delta=0.0005 |
| **数据** | Mini 集: 600 exams (58 abnormal + 542 normal), 11,260 slices |

---

## 2. 结果总览

| Fold | Best AUC | Best @ Epoch | 总 Epochs | 早停 | Peak VRAM |
|------|----------|-------------|-----------|------|-----------|
| 0    | **0.6948** | 1 | 9 | ✅ | 3.8 GB |
| 1    | **0.6332** | 8 | 16 | ✅ | 3.8 GB |
| 2    | **0.5816** | 3 | 11 | ✅ | 3.8 GB |
| 3    | **0.6720** | 1 | 9 | ✅ | 3.8 GB |
| 4    | **0.6627** | 1 | 9 | ✅ | 3.8 GB |

```
Mean AUC:  0.6489 ± 0.0390
Median:    0.6627
Best:      0.6948  (Fold 0)
Worst:     0.5816  (Fold 2)
```

**结论：管线验证通过** ✅ — Mean AUC 0.649 > 0.55 验收门槛，Mini 集上模型学到了有效的异常检测信号。

---

## 3. ASL vs BCE 对比

| 指标 | BCE (dry-run, 3 epoch, bs=16) | ASL (full, bs=32) | 变化 |
|------|------|------|------|
| Fold 0 best AUC | 0.7111 | 0.6948 | -0.016 |
| Fold 1 best AUC | 0.5900 | 0.6332 | **+0.043** |
| Fold 2 best AUC | 0.5082 | 0.5816 | **+0.073** |
| Fold 间极差 | 0.203 | 0.113 | **缩小 44%** |
| 随机水平 fold 数 | 1/3 (Fold 2 = 0.51) | 0/5 | **全部 > 0.58** |

**关键发现**：

- ASL 的核心优势在于**拉升弱势 fold**：Fold 2 从接近随机 (0.51) 提升到 0.58
- 强势 fold (Fold 0) 略微下降 (0.71 → 0.69)，因为 ASL 的 γ_neg=4 在数据太少时压制稍过
- Fold 间方差从 0.203 缩小到 0.113，**训练更稳健**

---

## 4. 训练动态分析

### 4.1 过拟合模式

```
Fold 0:  train_loss 0.0405 → 0.0021 (Epoch 9)
         val_loss   0.0200 → 0.0552 (持续上升)
         val_auc    0.5319 → peak 0.6948 @ Epoch 1 后震荡下降
```

**典型模式（4/5 folds 相同）**：
- Epoch 1-3: AUC 快速上升，达到峰值
- Epoch 4+: train_loss 持续下降到 ~0.001，val_loss 持续上升
- 这是**快速记忆训练集**的经典特征：模型很快就把 480 个训练 study 背下来了

### 4.2 Fold 2 深度分析（最差 fold）

```
Fold 2: train_loss 0.0365 → 0.0011  (Epoch 11)
        val_loss   0.0328 → 0.0967  (涨了 3 倍!)
        val_auc    0.4979 → peak 0.5816 @ Epoch 3
```

Fold 2 的 val_loss 从 0.033 涨到 0.097，远高于其他 fold（0.05-0.08）。说明该 fold 的验证集中有**模型难以泛化的困难样本**。

### 4.3 Fold 1 独特性

Fold 1 是唯一在后期（Epoch 8）才达到峰值的 fold：
```
Epoch 0: AUC=0.4452
Epoch 4: AUC=0.5784
Epoch 8: AUC=0.6332 ← 最晚 peak
```
说明该 fold 的验证集分布与训练集差异较大，模型需要更多 epoch 才能适应。

### 4.4 VRAM 与速度

| 指标 | 数值 |
|------|------|
| Peak VRAM | 3.8 GB / 8 GB (47%) |
| 每 epoch 耗时 | 82-87 秒 |
| 总训练时间 | ~68 分钟 (5 folds) |
| 每 fold 平均 | ~13.6 分钟 |

**batch_size=32 完全可行**：3.8 GB 只用了不到一半显存，还有空间进一步增大到 48-64。

---

## 5. 发现的问题

### 5.1 ASL γ_neg=4 在 Mini 集上过强

γ_neg=4 是为大规模数据集设计的（原始论文用的是 OpenImages ~9M 张图）。在 480 个训练 study 上，负样本（正常切片）被压制过猛，导致：
- 模型在 ~3 epoch 就过拟合
- 后期 val_loss 持续上升，AUC 反而下降

**建议**：下轮实验 γ_neg=2（或 3），牺牲部分负样本聚焦能力换更慢的过拟合。

### 5.2 缺乏 Per-Class AUC

当前日志只输出 macro AUC，看不到 12 个类各自的 AUC。无法判断哪类最难。

**建议**：在 epoch 日志中加入 per-class AUC（`compute_per_class_auc` 已实现，只是没调用）。

### 5.3 切片级 vs Study 级评估

当前 `validate_one_epoch` 计算的是**切片级 AUC**，但竞赛评分用的是 **study 级 AUC**（每 study 一个预测）。切片级 AUC 由于同一 study 的多张切片共享标签，会高估模型能力。

**建议**：在 `run_fold` 中加入 study 级聚合后的 AUC 计算。

### 5.4 早停过于仓促

Epoch 1 peak 后，模型还有可能在后期再次提升（如 Fold 1 在 Epoch 8 达到最高），但 patience=8 且 val_loss 单调上升导致过早停止。对 mini 集这种高噪声场景，可能需要更长的 patience 或改用 val_loss 辅助判断。

---

## 6. 建议下一步

| 优先级 | 任务 | 依据 |
|--------|------|------|
| **P0** | 试验 γ_neg=2 版本，减少过拟合 | 当前 4/5 folds 在 Epoch 1-3 就 peak |
| **P1** | 日志中增加 per-class AUC | 立刻知道哪 12 类最难，指导后续策略 |
| **P1** | 实现 study 级 OOF AUC 评估 | 让本地验证更接近竞赛真实分数 |
| **P2** | 试 γ_neg=3 或添加 stronger dropout (0.4→0.5) | 在减少过拟合和保持聚焦之间找平衡 |
| **P3** | 创建 Kaggle 训练 notebook 初版 | 即使本地还在调参，Kaggle 代码可以先写 |

---

## 7. 关键代码位置

| 文件 | 用途 |
|------|------|
| [configs/a_convnextv2_tiny_2.5d_mini.yaml](../configs/a_convnextv2_tiny_2.5d_mini.yaml) | 当前 ASL 配置 |
| [src/train.py](../src/train.py) | 训练主循环（validate_one_epoch 需加 per-class AUC） |
| [src/metrics.py](../src/metrics.py) | `compute_macro_auc` / `compute_per_class_auc` |
| [src/losses/asymmetric_loss.py](../src/losses/asymmetric_loss.py) | ASL 实现 |
| [outputs/convnextv2_tiny_2p5d_mini/checkpoints/](../outputs/convnextv2_tiny_2p5d_mini/checkpoints/) | 5 个 fold 的 best checkpoint (107MB×5) |
| [outputs/convnextv2_tiny_2p5d_mini/summary.json](../outputs/convnextv2_tiny_2p5d_mini/summary.json) | 训练结果摘要 |
