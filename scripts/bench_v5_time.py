# -*- coding: utf-8 -*-
"""v5 单步训练计时 — 本地 RTX 5060 实测, 外推 Kaggle 2xT4.

方法: 用 cells_v5 的真实模型结构 (timm DINOv2-small @288, unfreeze 6 blocks,
SlotHead) + 合成批次 [6 studies × 6 slots × 3 slices × H²], AMP on,
forward+backward+optimizer.step, 与 Kaggle train_epoch 相同的每步计算量。
同时测 224px (v4 配置) 作对照比例。
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
           'Contusion', 'Fracture']
SLOT_PRIORS = {
    "ACL": (0, 3, 5), "MCL": (1, 4),
    "Medial Meniscus": (0, 1, 3, 4), "Lateral Meniscus": (0, 1, 3, 4),
    "Medial OA": (1, 4, 5), "Lateral OA": (1, 4, 5),
    "PF OA": (0, 2, 5), "Effusion": (0, 2), "Synovitis": (0, 2),
    "Baker's": (0,), "Contusion": (0, 1, 2), "Fracture": (0, 1, 2, 4, 5),
}


class SlotHead(nn.Module):
    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden
        prior = torch.zeros(n_out, n_slot)
        for t, idx in SLOT_PRIORS.items():
            prior[TARGETS.index(t), list(idx)] = 0.55
        self.register_buffer("slot_prior", prior)

    def forward(self, x, mask):
        h = self.proj(x) + self.slot_emb
        att = (torch.einsum("bsh,oh->bos", h, self.query) / (self.hidden ** 0.5)
               + self.slot_prior.unsqueeze(0))
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias


class BenchModel(nn.Module):
    """与 cells_v5/06_model.py 的 MultiViewModel 相同的计算结构。"""
    def __init__(self, bb, unfreeze=6):
        super().__init__()
        self.dinov2 = bb
        n_blocks = len(self.dinov2.blocks)
        if unfreeze > 0:
            for p in self.dinov2.parameters():
                p.requires_grad = False
            for block in self.dinov2.blocks[n_blocks - unfreeze:]:
                for p in block.parameters():
                    p.requires_grad = True
        self.head = SlotHead(1152, 6, 12)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images, mask):
        B, S = images.shape[:2]
        x = images.reshape(B * S, 3, images.shape[-2], images.shape[-1])
        x = x.float().div_(255.0)
        x = (x - self.mean) / self.std
        f = self.dinov2.forward_features(x)
        cls, patches = f[:, 0], f[:, 1:]
        mean_p = patches.mean(1)
        k = max(1, patches.shape[1] // 8)
        focal = patches.topk(k, dim=1).values.mean(1)
        feats = torch.cat([cls, mean_p, focal], 1).reshape(B, S, -1)
        return self.head(feats, mask)


def bench(image_size, steps=8, warmup=2, batch=6):
    bb = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False,
                           num_classes=0, img_size=image_size)
    model = BenchModel(bb).cuda().train()
    model.dinov2.eval()

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    crit = lambda logits, y: F.binary_cross_entropy_with_logits(logits, y)

    slots = torch.randint(0, 255, (batch, 6, 3, image_size, image_size),
                          dtype=torch.uint8, device='cuda')
    mask = torch.ones(batch, 6, device='cuda')
    y = torch.rand(batch, 12, device='cuda')

    ts = []
    for i in range(steps):
        t0 = time.perf_counter()
        with torch.amp.autocast('cuda', enabled=True):
            logits = model(slots, mask)
            loss = crit(logits, y)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)

    ts = ts[warmup:]
    ms = 1000 * np.mean(ts)
    print(f'  {image_size}px: {ms:.0f} ms/step (batch {batch}×6 slots×3 slices, AMP)')
    return ms


if __name__ == '__main__':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    torch.manual_seed(2026)
    t224 = bench(224)
    t288 = bench(288)
    print(f'288/224 计算量实测比: {t288 / t224:.2f} (理论 {(288/224)**2:.2f})')

    # 外推 Kaggle 2xT4 (DataParallel, 每 GPU batch 3):
    # T4 FP16 tensor ≈ 65 TFLOPS; 5060 笔记本 ≈ 100-110 → 单卡比 ~0.6
    # 2xT4 DP 效率 ~0.9/卡 → 合计 ~1.1× 5060。取保守: T4x2 = 0.9× 5060
    for label, scale in (('optimistic (1.1×)', 1.1), ('conservative (0.9×)', 0.9)):
        per_epoch_min = 725 / (6 / 6) * t288 * scale / 1000 / 60 * (6 / 6)
        print(f'  T4x2 {label}: {t288*scale:.0f} ms/step → {per_epoch_min:.1f} min/epoch '
              f'→ 30 epochs = {per_epoch_min*30/60:.1f} h')
