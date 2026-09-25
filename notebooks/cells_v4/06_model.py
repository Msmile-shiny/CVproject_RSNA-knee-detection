# ============================================================
# v4: SlotHead + MultiViewModel — 支持诊断池化
# ============================================================

class SlotHead(nn.Module):
    """Per-diagnosis attention over MRI slots with anatomical priors."""

    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden

        prior = torch.zeros(n_out, n_slot)
        for target_name, slot_indices in SLOT_PRIORS.items():
            if target_name in TARGET_COLUMNS:
                prior[TARGET_COLUMNS.index(target_name), list(slot_indices)] = 0.55
        self.register_buffer("slot_prior", prior)

    def forward(self, x, mask):
        h = self.proj(x) + self.slot_emb                              # [B, S, H]
        attention = (
            torch.einsum("bsh,oh->bos", h, self.query)                # [B, n_out, S]
            / math.sqrt(self.hidden)
            + self.slot_prior.unsqueeze(0)
        )
        attention = attention.masked_fill(
            mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        context = self.drop(torch.einsum("bos,bsh->boh", attention, h))
        return (context * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias


class MultiViewModel(nn.Module):
    """DINOv2 + SlotHead for multi-view knee MRI."""

    def __init__(self, dinov2_model, n_slots=6, cls_dim=384,
                 n_classes=12, slot_hidden=256, dropout=0.2,
                 unfreeze_layers=6):
        super().__init__()
        self.n_slots = n_slots
        self.cls_dim = cls_dim
        self.feature_dim = cls_dim * 3
        self.unfreeze_layers = unfreeze_layers

        self.dinov2 = dinov2_model
        n_blocks = len(self.dinov2.blocks)
        if unfreeze_layers > 0:
            for p in self.dinov2.parameters():
                p.requires_grad = False
            unfreeze_start = max(0, n_blocks - unfreeze_layers)
            for block in self.dinov2.blocks[unfreeze_start:]:
                for p in block.parameters():
                    p.requires_grad = True
            if hasattr(self.dinov2, 'norm'):
                for p in self.dinov2.norm.parameters():
                    p.requires_grad = True

        self.head = SlotHead(
            dim=self.feature_dim, n_slot=n_slots, n_out=n_classes,
            hidden=slot_hidden, p=dropout)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _extract_features(self, x_3ch):
        if self.unfreeze_layers > 0:
            features = self.dinov2.forward_features(x_3ch)
        else:
            with torch.no_grad():
                features = self.dinov2.forward_features(x_3ch)
        cls = features[:, 0, :]
        patches = features[:, 1:, :]
        mean_p = patches.mean(dim=1)
        k = max(1, patches.shape[1] // 8)
        focal = patches.topk(k, dim=1).values.mean(dim=1)
        return torch.cat([cls, mean_p, focal], dim=1)

    def forward(self, images, mask):
        """images: [B, S, 3, H, W] uint8 or [B*W, S, 3, H, W] for TTA"""
        B, S = images.shape[:2]
        x = images.reshape(B * S, 3, images.shape[-2], images.shape[-1])
        x = x.float().div_(255.0)
        x = (x - self.mean) / self.std
        features = self._extract_features(x)
        features = features.reshape(B, S, -1)
        return self.head(features, mask)

    def train(self, mode=True):
        super().train(mode)
        self.dinov2.eval()
        return self


# ---- ★ 诊断特异性 TTA 池化 ----
DIAG_POOL_IDX = {}
for target_name, mode in DIAG_POOL.items():
    if target_name in TARGET_COLUMNS:
        DIAG_POOL_IDX[TARGET_COLUMNS.index(target_name)] = mode


def diagnostic_pool(logits_windows, pool_idx=None):
    """对 [B, W, C] logits 应用诊断特异性池化。

    - max:  局部病灶保留最强信号窗口
    - top2: ACL/MCL 取前2强窗口平均
    - mean: 弥漫性病变取全窗口平均（默认）
    """
    if pool_idx is None:
        pool_idx = DIAG_POOL_IDX

    B, W, C = logits_windows.shape
    probs = torch.sigmoid(logits_windows)               # [B, W, C]
    result = probs.mean(dim=1)                          # [B, C] — 默认 mean

    for j, mode in pool_idx.items():
        x = probs[:, :, j]                             # [B, W]
        if mode == 'max':
            result[:, j] = x.max(dim=1).values
        elif mode == 'top2':
            result[:, j] = x.topk(min(2, W), dim=1).values.mean(dim=1)

    return result  # [B, C]


if IS_MAIN:
    n_total = sum(p.numel() for p in SlotHead(1152, 6, 12).parameters())
    print(f'SlotHead params: {n_total/1e6:.3f}M')
    print(f'Diag pool targets: {list(DIAG_POOL_IDX.keys())}')
    print('Model v4 ready.')
