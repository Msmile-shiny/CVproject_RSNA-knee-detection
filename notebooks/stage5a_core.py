"""Stage 5A reusable tensor operations; no data loading or network side effects."""
import numpy as np
import torch
from torch import nn


def window_centers(n, limit=32):
    if n < 3:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.linspace(1, n - 2, min(limit, n - 2)).round().astype(np.int64))


class DenseMIL(nn.Module):
    def __init__(self, dim=1152, hidden=128, classes=12, slots=6, pooling='attention'):
        super().__init__()
        self.pooling = pooling
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot = nn.Embedding(slots, hidden)
        self.scale = nn.Embedding(2, hidden)
        self.position = nn.Linear(1, hidden, bias=False)
        self.a = nn.Linear(hidden, hidden)
        self.b = nn.Linear(hidden, hidden)
        self.query = nn.Linear(hidden, classes, bias=False)
        self.out = nn.Parameter(torch.randn(classes, hidden) * .02)
        self.bias = nn.Parameter(torch.zeros(classes))
        self.drop = nn.Dropout(.2)

    def forward(self, x, mask, slot, pos, scale, return_attention=False):
        h = self.proj(x) + self.slot(slot) + self.position(pos.unsqueeze(-1)) + self.scale(scale)
        score = self.query(torch.tanh(self.a(h)) * torch.sigmoid(self.b(h))).transpose(1, 2)
        if self.pooling == 'mean':
            score = torch.zeros_like(score)
        valid = mask[:, None, :].bool()
        attention = score.masked_fill(~valid, -1e4).softmax(-1) * valid
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
        context = torch.einsum('bct,bth->bch', attention, h)
        logits = (self.drop(context) * self.out).sum(-1) + self.bias
        return (logits, attention) if return_attention else logits


def selected_tokens(attention, mask, per_class=2):
    """At most 24 coarse tokens, chosen from image attention only, never labels."""
    valid = np.flatnonzero(mask)
    if not len(valid):
        return np.empty(0, dtype=np.int64)
    chosen = []
    for row in attention:
        chosen.extend(valid[np.argsort(-row[valid], kind='stable')[:per_class]])
    return np.unique(chosen)


def weighted_bce(logits, targets, weights, masks):
    weight = weights * masks
    return (nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none') * weight).sum() / weight.sum().clamp_min(1e-8)
