# ============================================================
# 3f. v2: Weighted Soft BCE Loss
# ============================================================
#
# Replaces FocalBCELoss. Key differences:
#   1. Targets are soft probabilities (prob in [0.01, 0.99]), not hard 0/1
#   2. Each class has a training weight based on LLM calibration reliability
#   3. Each class has a mask -- allows ignoring uncertain samples per-class
#
# Loss per element:
#   L = -weight * mask * [prob * log(sigmoid(logit)) + (1-prob) * log(1 - sigmoid(logit))]
#
# Why this fixes overfitting:
#   - Focal: forces model to fit hard labels, penalizing "uncertain" predictions
#   - Soft:  lets model be uncertain where LLM is unreliable (Effusion, Synovitis)
#   - Weight: down-weights unreliable classes rather than discarding the whole row
#   - Mask:   ignores samples where even the calibration is unsure (weight < 0.15)

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
        # Clamp targets away from 0/1 for BCE numerical stability
        targets = prob_targets.clamp(self.eps, 1.0 - self.eps)

        # Standard BCE per element
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Apply per-class weight and mask
        weighted = bce * weights * masks

        # Normalize by total active mask elements
        denom = masks.sum().clamp(min=1)
        return weighted.sum() / denom


# For comparison / ablation: keep FocalBCELoss available
class FocalBCELoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1.0 - p_t) ** self.gamma
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (alpha_weight * focal_weight * bce).mean()
