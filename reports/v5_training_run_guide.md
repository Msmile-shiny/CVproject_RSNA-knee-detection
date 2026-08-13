# v5 训练 — Kaggle 运行指南

> 目标：在 v4（gold 0.833 / LB 0.835）基础上，用 v5 三件套重训单模型：
> ① 融合软标签（文本提取器 × 公开 20 成员集成 OOF teacher）② 288px/130mm 奈奎斯特分辨率 ③ 置信度加权软 BCE。
> 预期 gold AUC 0.85-0.87，LB 0.85+。之后（v5 规划 ④⑤）再做 3-5 seed 自集成 + OOF rank 融合冲击 0.90。

---

## 一、本地已完成的准备（已全部验证）

| 产物 | 位置 | 状态 |
|------|------|------|
| v5 融合标签 | `data/processed/v5_labels.csv`（4407 行 × 25 列） | ✅ 已生成 |
| 融合诊断报告 | `data/processed/v5_fusion_report.csv` | ✅ 已生成 |
| v5 训练 notebook | `notebooks/kaggle_train_v5_multiview.ipynb` | ✅ 已构建 |
| 本地 GPU 冒烟测试 | `scripts/smoke_test_v5.py` | ✅ 全过（RTX 5060 实跑：标签 4349/58、pos_embed 37²→20²、WeightedSoftBCE、1-step loss 0.374、TTA 验证、checkpoint 往返） |

**融合标签关键数字**（58 gold 上 in-sample AUC，text vs OOF vs fused）：

| target | text | OOF | fused | 主导 teacher |
|--------|------|-----|-------|--------------|
| ACL | 0.953 | 0.869 | 0.951 | text |
| MCL | 0.964 | 0.884 | 0.966 | text |
| Medial Meniscus | 0.894 | 0.850 | **0.959** | 互补 |
| Lateral Meniscus | 0.846 | 0.642 | 0.845 | text |
| Medial OA | 0.895 | 0.950 | **0.967** | OOF |
| Effusion | 0.777 | 0.943 | 0.934 | OOF |
| Synovitis | 0.709 | 0.742 | 0.769 | 双弱（已知弱类） |
| Fracture | 0.833 | 0.894 | **0.910** | 互补 |

- OOF teacher 来源：pilkwang 公开权重包 `oof.npz`（4 seeds × 5 folds，每研究被每 seed 恰好 1 个 fold 留出 → 全 4407 研究无泄漏 OOF）
- 记忆检查通过：gold 行的 OOF 预测置信度不高于非 gold 行（无 in-fold 记忆痕迹）
- 提取器移植验证通过：我们 score 预测作者 y_derived 的 AUC 0.85-0.96

## 二、Kaggle 操作步骤

### 1. 上传 v5 标签数据集

- 新建 Dataset（如 `rsna-knee-v5-labels`），把 `data/processed/v5_labels.csv` 放**根目录**
- 2MB 小文件，网页上传即可

### 2. 上传 notebook

- 上传 `notebooks/kaggle_train_v5_multiview.ipynb`
- 挂载：竞赛数据（自动）、**v5 标签数据集**、`rsna-dinov2-weights`（timm 权重，与 v4 相同）
- **加速器必须选 T4x2**：288px 缓存 = 19.7GB + 运行时 ~4GB ≈ 24GB；T4x2 GPU 会话系统内存 ~29GB（超过 ~30GB 会触发内核重启，我们留了余量）；P100 会话只有 16GB 系统内存，装不下
  - 若只能用 P100：把 cell 3 里 `CFG['image_size']` 改成 `256` 且 `cache_slices` 改成 `7`（缓存 12.1GB；0.508mm/px，仍优于 v4；TTA 窗口 7→5）
- Run All，训练约 5-7 小时（30 epochs × 288px，含 420min 墙钟保护 + early stop 12）

### 3. 输出检查

| 产物 | 说明 |
|------|------|
| `validation_report_best.csv` | 每类 gold AUC —— **对比 v4：gold 0.833** |
| `checkpoints/best_model.pt` | ★ 下载留档（后续 seed 自集成 / warm-start 用） |
| `training_history.csv` | loss/AUC 曲线 |
| `gold_validation_auc.csv` + `gold_validation_predictions.csv` | gold 逐研究预测 |
| `submission.csv` | 提交，预期 LB 0.85+ |

**训练日志核对点**（本地冒烟已全过；以下核对点用于确认 Kaggle 环境无差异）：
- cell 9 输出：`Train: 4,349 studies (v5 fused soft labels)`，`missing for unlabeled: 0`
- cell 9 的 pos_rate 表应与本地融合报告一致（ACL ~0.17, MM ~0.37, Synovitis 0.179）
- cell 13 打印 `pos_embed interpolated: [37×37] → [20×20]` 和 `Loss: WeightedSoftBCE` ← **若缺这两行立即停止**
- cell 14 打印 `Forward: torch.Size([6, 6, 3, 288, 288]) → torch.Size([6, 12])` 与 `Sanity check PASSED`
- 第一个 epoch 的 train_loss 应落在 0.4-0.7（软标签 BCE），v4 硬标签是 ~0.02 量级，别被这个差异吓到

## 三、v5 与 v4 的差异一览

| 维度 | v4 | v5 |
|------|-----|-----|
| 标签 | FocalBCE 硬伪标签 | 融合软标签 + 置信度权重（gold 上 per-finding 逻辑回归） |
| 损失 | FocalLoss(0.25, 2) | WeightedSoftBCE |
| 分辨率 | 224px@160mm (0.714mm/px) | **288px@130mm (0.451mm/px)** 满足奈奎斯特 |
| epochs | 40 | 30 + 墙钟 420min 保护 |
| 验证 | 58 gold | 58 gold（与 v4 可比） |

## 四、风险与备选

| 风险 | 概率 | 预案 |
|------|------|------|
| T4x2 排队/配额不足 | 中 | P100 + image_size=256（指南已内置开关） |
| 30 epochs 在 9h 内跑不完 | 低 | 墙钟保护会提前停止；best_model.pt 已保存 |
| 融合标签在 LB 上不涨 | 低 | gold 验证即可判断；若 fused < text-only，下一版退化为纯文本标签 |
| Synovitis 依旧弱 | 高（已知） | v5 规划 ⑥：Synovitis specialist（RTA-HMIL 蓝本 +0.084） |
| 内存不足（缓存 19.7GB） | 低（T4x2 30GB RAM） | image_size=256；或 cache_slices 9→7 |

## 五、后续（v5 规划 ④-⑧）

1. ④ 3-5 seeds 自集成：同一 notebook 换 `seed`（03_config 加 `torch.manual_seed`），每 seed 一个 9h 会话，各留 5% gold 做 OOF
2. ⑤ OOF 目标级 rank 融合（0.899 的 +0.003 手法）
3. ⑥ Synovitis / 弱类 specialists（0.899 的 +0.084 蓝本）
4. ⑦ EfficientNet-B3 多样性分支（0.903 冠军路线）
5. ⑧ 可选：public 成员 fine-tune 实验
6. 第二代自蒸馏：v5 各 seed OOF → v6 teacher
