# ============================================================
# v4: Loss Functions — FocalLoss + WeightedSoftBCE
# ============================================================

class FocalBCELoss(nn.Module):
    """Focal Loss for binary classification — 处理类别不平衡。

    FL = -alpha * (1 - pt)^gamma * log(pt)
    其中 pt = p if y=1 else 1-p
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets, weight=None, mask=None):
        """
        Args:
            logits: [B, C] raw logits
            targets: [B, C] hard labels (0/1)
            weight: [B, C] sample weights (optional)
            mask: [B, C] 1=participate in loss, 0=ignore
        """
        probs = torch.sigmoid(logits)
        # BCE: -[y*log(p) + (1-y)*log(1-p)]
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Focal weight: (1 - pt)^gamma
        pt = torch.where(targets > 0.5, probs, 1.0 - probs)
        focal_weight = (1.0 - pt) ** self.gamma

        # Alpha balancing
        alpha_weight = torch.where(
            targets > 0.5, self.alpha, 1.0 - self.alpha)

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
    """统一软标签损失：gold 和 pseudo 共用同一个 loss。

    Gold studies: prob=hard_label, weight=1.0, mask=1.0
    Pseudo studies: prob=calibrated, weight=reliability, mask=thresholded
    """
    def forward(self, logits, prob_targets, weights, mask):
        bce = F.binary_cross_entropy_with_logits(
            logits, prob_targets, reduction='none')
        loss = bce * weights * mask
        denom = mask.sum() + 1e-8
        return loss.sum() / denom


if IS_MAIN:
    print('Loss functions v4 ready (FocalBCE + WeightedSoftBCE).')
