"""Trainable multi-sequence reference. No network calls or task state at import."""
import hashlib
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet34


def stable_fold(group, folds=5):
    return int(hashlib.sha256(('stage6-v1:' + str(group)).encode()).hexdigest()[:16], 16) % folds


def centers(length, count):
    if length < 3:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.linspace(1, length - 2, min(count, length - 2)).round().astype(np.int64))


class KneeResNet(nn.Module):
    def __init__(self, weights=None, slots=6, classes=12, pooling='mean'):
        super().__init__()
        self.encoder = resnet34(weights=None)
        if weights is not None:
            self.encoder.load_state_dict(weights, strict=True)
        self.encoder.fc = nn.Identity()
        self.pooling = pooling
        self.slot_embedding = nn.Embedding(slots, 512)
        self.attention = nn.Linear(512, classes)
        self.head = nn.Parameter(torch.randn(classes, 512) * .01)
        self.bias = nn.Parameter(torch.zeros(classes))
        self.dropout = nn.Dropout(.2)

    def train(self, mode=True):
        super().train(mode)
        # Small study batches: retain pretrained running statistics, train affine params.
        for module in self.encoder.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, images, valid, slots):
        b, n, c, h, w = images.shape
        valid = valid.bool()
        if not valid.any(1).all():
            raise ValueError('A study has no valid image windows')
        indices = valid.flatten().nonzero().flatten()
        encoded = self.encoder(images.reshape(-1, c, h, w)[indices])
        features = encoded.new_zeros(b * n, 512).index_copy(0, indices, encoded).reshape(b, n, 512)
        features = features + self.slot_embedding(slots)
        scores = self.attention(features).transpose(1, 2)
        if self.pooling == 'mean':
            scores = torch.zeros_like(scores)
        elif self.pooling != 'attention':
            raise ValueError(self.pooling)
        attention = scores.masked_fill(~valid[:, None], -torch.inf).softmax(-1)
        pooled = torch.einsum('bcn,bnd->bcd', attention, features)
        return (self.dropout(pooled) * self.head).sum(-1) + self.bias


def loss_fn(logits, target, weight, mask):
    effective = weight * mask
    if effective.sum() <= 0:
        raise ValueError('Batch has no supervised labels')
    return (nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none') * effective).sum() / effective.sum()
