# 2026-08-08 工作复盘

---

## 一、做了什么

### 管线验证阶段

1. **阅读 `03_pipeline_verification.ipynb`**，确认 8 项检查项覆盖完整链路
2. **分析模型选型**：结论是不换 backbone（ConvNeXtV2-Tiny 28M 够用），但损失函数从 BCE 升级到 ASL
3. **制定全量数据策略**：本地 mini 集验证管线 → Kaggle notebook 全量训练+推理+提交

### 损失函数升级：BCE → ASL

4. **切换 ASL loss**：修改 [configs/a_convnextv2_tiny_2.5d_mini.yaml](../configs/a_convnextv2_tiny_2.5d_mini.yaml)，`bce_with_logits` → `asl`，γ_neg=4, γ_pos=1, clip=0.05
5. **batch_size 16→32**：利用剩余显存（当前只用 3.8/8GB）
6. **完整 5-fold 30-epoch 训练**：

```
Baseline (ASL γ_neg=4):
  Fold AUCs: 0.6948 | 0.6332 | 0.5816 | 0.6720 | 0.6627
  Mean AUC:  0.6489 ± 0.039
  总耗时: ~68 分钟
```

7. **生成报告**：[reports/asl_full_training_report.md](../reports/asl_full_training_report.md)

### 代码改进

8. **Per-class AUC 日志**：在 `validate_one_epoch` 和 `run_fold` 中加入 12 类逐类 AUC 输出
9. **修复 FutureWarning**：`torch.cuda.amp` → `torch.amp` 新 API，终端不再刷警告

### 反过拟合实验：Exp A

10. **创建实验 A**：[configs/b_convnextv2_tiny_2.5d_mini_expA.yaml](../configs/b_convnextv2_tiny_2.5d_mini_expA.yaml)
    - γ_neg: 4→2, dropout: 0.3→0.5, drop_path: 0.1→0.2, warmup: 3→1
11. **Exp A 结果**：

```
Exp A (ASL γ_neg=2):
  Fold AUCs: 0.6650 | 0.6259 | 0.5501 | 0.6358 | 0.6636
  Mean AUC:  0.6281 ± 0.042   ← 比 baseline 低 0.021
```

12. **生成对比报告**：[reports/expA_vs_baseline_report.md](../reports/expA_vs_baseline_report.md)

---

## 二、关键发现

### 1. ASL 比 BCE 更好

| BCE (dry-run) | ASL (full) |
|---------------|------------|
| 最差 fold 0.51 | 最差 fold 0.58 |
| Fold 间方差 0.20 | Fold 间方差 0.11 |

ASL 把底线从 0.51 拉到 0.58，训练更稳健。

### 2. γ_neg=4 > γ_neg=2（在 mini 集上）

Exp A 证明了：**在小数据集+极度不均衡场景下，高 γ_neg 是正确策略。** 模型必须快速聚焦到稀少的异常 case，不能把容量浪费在 434 个正常 case 上。dropout=0.5 的正则化力度过大，反而削弱了信号。

### 3. MCL 是致命短板

Per-class AUC 排名：
```
Effusion:0.71  LOA:0.69  Synovitis:0.69  ...  Baker's:0.53  MCL:0.44
```

**MCL（内侧副韧带）AUC=0.44，几乎随机。** 根本原因是 Sagittal 面切不到 MCL 的有效特征——它是物理限制，不是算法问题。

### 4. 当前最优配置

```
模型: ConvNeXtV2-Tiny (28M), ImageNet-22k 预训练
输入: 2.5D Sagittal triplet [3, 224, 224]
损失: ASL γ_neg=4, γ_pos=1, clip=0.05
优化: AdamW, backbone_lr=2e-5, head_lr=1e-4
调度: warmup=3, cosine annealing
批量: 32, AMP fp16
正则: dropout=0.3, drop_path=0.1
显存: 3.8/8GB
```

---

## 三、文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `configs/a_convnextv2_tiny_2.5d_mini.yaml` | 修改 | loss→ASL, batch=32 |
| `configs/b_convnextv2_tiny_2.5d_mini_expA.yaml` | 新建 | Exp A 配置 |
| `src/train.py` | 修改 | 修复 AMP 警告 + per-class AUC 日志 |
| `src/losses/__init__.py` | 已有 | ASL/BCE/Focal/SmoothAUC 工厂 |
| `src/losses/asymmetric_loss.py` | 已有 | ASL 实现 |
| `reports/asl_full_training_report.md` | 新建 | Baseline 训练报告 |
| `reports/expA_vs_baseline_report.md` | 新建 | Exp A 对比报告 |
| `outputs/convnextv2_tiny_2p5d_mini/` | 生成 | 5-fold checkpoints (107MB×5) |
| `outputs/convnextv2_tiny_2p5d_mini_expA/` | 已清理 | Exp A 结果已删除 |

---

## 四、尚待解决

| 问题 | 优先级 | 方向 |
|------|--------|------|
| MCL AUC=0.44 | **最高** | 需要 Coronal 序列数据 |
| Baker's AUC=0.53 | 高 | 位置变异大，需要更广视野 |
| Mini 集过拟合 | 中 | 全量数据天然缓解 |
| 切片级→Study 级评估 | 中 | `aggregate_to_study` 已实现，未接入训练 |
| Kaggle Notebook | 中 | 全量训练+推理+提交 |
