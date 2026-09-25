# DINOv2 Refiner v2 — Soft Labels + Partial Unfreeze

### 相比 v1 的核心变更

| 维度 | v1 (昨天) | v2 (本次) |
|------|----------|----------|
| **标签** | 硬标签 (0/1) + HIGH 行级过滤 | 软标签 (prob_*/weight_*/mask_*) 逐类校准 |
| **损失函数** | FocalBCELoss (γ=2, α=0.25) | WeightedSoftBCELoss (逐类权重) |
| **Backbone** | DINOv2 完全冻结 | 最后 N 层解冻 (默认 6) |
| **置信度过滤** | HIGH-only, 丢弃整行 | 无行级过滤, 用 weight 逐类降权 |

### 为什么这些变更能解决过拟合？

1. **软标签**: LLM 说 ACL=1 + HIGH 置信度 → prob=0.80 (不是 1.0)。模型学到"这个可能是撕裂，但不用 100% 相信"
2. **逐类权重**: Effusion LLM 不可靠 → weight≈0.48, 模型知道"积液判断参考就好, 别全信"
3. **解冻 backbone**: DINOv2 最后 6 层从 ImageNet 语义适配到 MRI 病理特征，1.1M → 4.6M 可训练参数, 噪声抵抗能力大幅提升
