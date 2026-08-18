# ============================================================
# v5: Loss Functions — WeightedSoftBCELoss (置信度加权软 BCE) + FocalBCELoss (v4 遗留)
# ============================================================

class FocalBCELoss(nn.Module):
    """Focal Loss for binary classification — v4 遗留, v5 不再使用。

    v4 用它处理「硬伪标签 + 类别不平衡」; v5 改用融合软标签 + 置信度权重后,
    软 BCE 直接携带置信度, focal 的难例加权与权重机制重叠, 反而放大噪声。
    FL = -alpha * (1 - pt)^gamma * log(pt)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets, weight=None, mask=None):
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.where(targets > 0.5, probs, 1.0 - probs)
        focal_weight = (1.0 - pt) ** self.gamma
        alpha_weight = torch.where(targets > 0.5, self.alpha, 1.0 - self.alpha)
        loss = alpha_weight * focal_weight * bce
        if weight is not None:
            loss = loss * weight
        if mask is not None:
            loss = loss * mask
        if self.reduction == 'mean':
            denom = mask.sum() if mask is not None else loss.numel()
            return loss.sum() / max(denom, 1.0)
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class WeightedSoftBCELoss(nn.Module):
    """★ v5 主损失：置信度加权的软标签 BCE (teacher-student 训练目标)。

    软标签 = per-finding 融合概率 (文本提取器 × 公开集成 OOF, 见 v5_labels.csv):
      - 携带两个 teacher 的置信度与 inter-class 结构 (0.73 与 0.80 的区分度
        优于两个硬标签 1/1)
      - 权重 = 标签置信度: 文本提及 (高 conf) 与 text/oof 一致时 → 1.0,
        静默 → ~0.35 (弱拉取, 从不断言阴性)
      - gold 行 (若有) weight=1.0, mask 控制参与
    loss = mean(bce(logits, prob) * weight * mask) / sum(mask)
    """
    def forward(self, logits, prob_targets, weights, mask):
        bce = F.binary_cross_entropy_with_logits(
            logits, prob_targets, reduction='none')
        loss = bce * weights * mask
        denom = mask.sum() + 1e-8
        return loss.sum() / denom


if IS_MAIN:
    print('Loss functions v5 ready (WeightedSoftBCE primary; FocalBCE legacy).')
