# ============================================================
# v3: Weighted Soft BCE Loss (reused from v2, unchanged)
# ============================================================

class WeightedSoftBCELoss(nn.Module):
    """BCE loss with soft probability targets and per-class reliability weights.

    Shape:
        logits:       [B, C]  model output logits
        prob_targets: [B, C]  calibrated probabilities (0.01 ~ 0.99)
        weights:      [B, C]  per-class training weights (0.05 ~ 1.0)
        masks:        [B, C]  binary mask (0=ignore this class for this sample)
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logits, prob_targets, weights, masks):
        targets = prob_targets.clamp(self.eps, 1.0 - self.eps)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        weighted = bce * weights * masks
        denom = masks.sum().clamp(min=1)
        return weighted.sum() / denom


class HardBCELoss(nn.Module):
    """Standard BCE for gold-labeled studies."""
    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets)
