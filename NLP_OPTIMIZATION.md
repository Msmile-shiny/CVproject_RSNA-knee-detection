# NLP 伪标签优化（二阶段校准）

## 为什么不能把“准确率 0.9”直接当比赛目标

比赛指标是 12 个目标的 macro ROC-AUC，不是逐标签 accuracy。当前 58 条 gold
报告上的 NLP hard-label accuracy 约为 0.84，但类别不平衡会让 accuracy 偏乐观；
Macro F1 只有约 0.78。尤其是 Synovitis、Lateral OA、Contusion 和 Fracture，
伪标签噪声会直接限制 MRI 模型上限。

## 新增校准层

运行：

```bash
python scripts/calibrate_pseudo_labels.py
```

脚本用 `pseudo_labels_valid.csv` 中的 58 条 gold 样本估计：

```text
P(gold=1 | LLM pred, confidence, class)
```

为避免小样本置信度分组过拟合，使用分层 beta-binomial shrinkage。输出：

- `prob_<class>`：训练用软标签；
- `weight_<class>`：逐类别可靠性权重；
- `mask_<class>`：是否参与该类别 loss；
- `nlp_calibration_report.csv`：逐类别校准诊断。

不要再要求一条 study 的 12 类全部为 HIGH 才保留。应逐类别使用 mask/weight，
否则一个不确定的 Synovitis 会连带丢弃另外 11 个可靠标签。

## 当前本地校准诊断

在现有 58 条 gold 上：

- Hard-label Macro F1：约 0.778；
- Hard-label mean Brier：约 0.159；
- 校准后 mean Brier（同样本诊断值）：约 0.119；
- 校准概率 Macro AUC（同样本诊断值）：约 0.847。

这些是小样本、同样本诊断值，不是 Kaggle 分数，也不能视作独立 CV。下一步应做
bootstrap 区间或增加人工复核集，尤其人工复核以下类别：Synovitis、Lateral OA、
Effusion、Contusion、Fracture。

## 推荐训练方式

对每个类别分别计算 masked soft BCE：

```python
raw = torch.nn.functional.binary_cross_entropy_with_logits(
    logits, soft_targets, reduction="none"
)
loss = (raw * target_weights * target_mask).sum() / target_mask.sum().clamp_min(1)
```

Gold 标签始终使用 target=0/1、weight=1、mask=1。伪标签建议先让 soft-label loss
占总 loss 的 0.3–0.5，再逐步增加；不要把校准后的伪标签等同于 gold。

## 进一步提升 NLP 标签质量

1. 对 58 条 gold 做分层 bootstrap，观察每类 precision/recall 的不确定区间。
2. 对弱类别进行主动学习：优先人工复核低权重且 MRI 模型高不确定的报告。
3. 每条报告保存 LLM 原始结构化概率或多次投票比例，而不是只保存 0/1。
4. 使用两个提示词/模型交叉验证；规则、LLM 和第二模型一致时才给高权重。
5. OA 必须明确 compartment；骨髓水肿不能自动等同 Contusion；Effusion 与
   Synovitis 必须分开处理。
