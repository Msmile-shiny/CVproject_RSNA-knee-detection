# ============================================================
# v5 SlotHead + MultiViewModel (DINOv2) + RadResNetModel (RadImageNet R50)
#   SlotHead/tta_jitter/stack_views/diagnostic_pool 为两管线共用 (逐字一致)
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
    """DINOv2 + SlotHead for multi-view knee MRI (v5 成员, 288px)。"""

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


class RadResNetModel(nn.Module):
    """RadImageNet ResNet50 (torchvision, fc→Identity) + SlotHead (rad 成员, 224px)。

    编码器全冻结 (unfreeze_layers=0), BN 保持 eval (running stats) —
    吸收进 running_mean 的 conv bias 依赖此模式。
    """

    def __init__(self, backbone, n_slots=6, feature_dim=2048,
                 n_classes=12, slot_hidden=256, dropout=0.2,
                 unfreeze_layers=0):
        super().__init__()
        self.n_slots = n_slots
        self.feature_dim = feature_dim
        self.unfreeze_layers = unfreeze_layers

        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        if unfreeze_layers > 0:
            for block in self.backbone.layer4[-unfreeze_layers:]:
                for p in block.parameters():
                    p.requires_grad = True

        self.head = SlotHead(
            dim=self.feature_dim, n_slot=n_slots, n_out=n_classes,
            hidden=slot_hidden, p=dropout)

        # ★ RadImageNet 训练归一化 = x/127.5 − 1 (uint8 域, 勿先 /255)
        self.register_buffer("mean", torch.tensor([127.5, 127.5, 127.5]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([127.5, 127.5, 127.5]).view(1, 3, 1, 1))

    def _extract_features(self, x_3ch):
        if self.unfreeze_layers > 0:
            return self.backbone(x_3ch)                        # [N, 2048] (fc=Identity)
        with torch.no_grad():
            return self.backbone(x_3ch)

    def forward(self, images, mask):
        """images: [B, S, 3, H, W] uint8 or [B*W, S, 3, H, W] for TTA"""
        B, S = images.shape[:2]
        x = images.reshape(B * S, 3, images.shape[-2], images.shape[-1])
        x = x.float()
        x = (x - self.mean) / self.std   # ★ uint8 域: = x/127.5 − 1 (勿先 /255)
        features = self._extract_features(x)
        features = features.reshape(B, S, -1)
        return self.head(features, mask)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()   # 冻结编码器: BN 始终用 running stats
        return self


# ---- ★ Jitter TTA 增广视图 (0.91 notebook augment() 移植, 两管线共用) ----
def tta_jitter(imgs, seed=AUG_SEED):
    """每窗口生成一个确定性增广视图。

    几何（旋转 ±AUG_ROT_DEG° / 缩放 +[0, AUG_SCALE] / 平移 ±AUG_SHIFT）+
    强度 ±AUG_INTENSITY，border 填充（0.91 同款）。
    固定种子 → 同一批输入每次生成相同增广，验证/测试/提交全程可复现。
    输入 [..., 3, H, W] uint8 → 输出同形状同 dtype。
    """
    lead = imgs.shape[:-3]
    x = imgs.reshape(-1, *imgs.shape[-3:]).float()
    n, dev = (x.shape[0], x.device)
    gen = torch.Generator(device=dev).manual_seed(int(seed) % (2 ** 63 - 1))

    rot = (torch.rand(n, device=dev, generator=gen) - 0.5) * 2 * (AUG_ROT_DEG * np.pi / 180)
    sc = 1.0 + torch.rand(n, device=dev, generator=gen) * AUG_SCALE
    tx = (torch.rand(n, device=dev, generator=gen) - 0.5) * 2 * AUG_SHIFT
    ty = (torch.rand(n, device=dev, generator=gen) - 0.5) * 2 * AUG_SHIFT
    cos, sin = (torch.cos(rot) / sc, torch.sin(rot) / sc)

    theta = torch.zeros(n, 2, 3, device=dev, dtype=torch.float32)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = (cos, -sin, tx)
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = (sin, cos, ty)

    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)

    scale = 1.0 + (torch.rand(n, 1, 1, 1, device=dev, generator=gen) - 0.5) * 2 * AUG_INTENSITY
    x = (x * scale).clamp(0, 255)
    return x.reshape(*lead, *x.shape[-3:]).to(imgs.dtype)


def stack_views(logits_flat, B, W, n_orig):
    """TTA 视图分组: [V*B*W, C]（B-major：每研究 W 行连续，原始块在前）→ [B, V*W, C]。

    V=2（jitter 开启）时输出每研究 [前 W 行原始视图, 后 W 行增广视图]；
    n_orig=None（无 jitter）时即 [B, W, C]。
    ★ 不可用 reshape(B, -1, C) 直接切——行序是研究大循环，会跨研究串位。
    """
    if n_orig is None:
        return logits_flat.reshape(B, W, -1)
    return logits_flat.view(2, B, W, -1).permute(1, 0, 2, 3).reshape(B, 2 * W, -1)


# ---- ★ 诊断特异性 TTA 池化 ----
DIAG_POOL_IDX = {}
for target_name, mode in DIAG_POOL.items():
    if target_name in TARGET_COLUMNS:
        DIAG_POOL_IDX[TARGET_COLUMNS.index(target_name)] = mode


def diagnostic_pool(logits_views, pool_idx=None, n_orig=None):
    """对 [B, V, C] logits 应用诊断特异性池化。

    - max:           局部病灶保留最强信号窗口
    - top2:          ACL/MCL 取前2强窗口平均
    - mean:          弥漫性病变取全窗口平均（默认）
    - original_mean: 仅无 jitter 原始视图平均（Synovitis, 0.91 同款）

    jitter TTA 模式（n_orig 给定）: 前 n_orig 个视图为原始视图、其余为增广视图；
    先按窗口做视图平均（0.91 的 win_probs），再做 per-target 窗口池化。
    n_orig=None 时全部视图视为原始视图（original_mean ≡ mean，与旧版行为一致）。
    """
    if pool_idx is None:
        pool_idx = DIAG_POOL_IDX

    B, V, C = logits_views.shape
    if n_orig is not None:
        orig_probs = torch.sigmoid(logits_views[:, :n_orig])             # [B, W, C]
        probs = (orig_probs + torch.sigmoid(logits_views[:, n_orig:])) / 2  # 视图平均
    else:
        probs = torch.sigmoid(logits_views)
        orig_probs = probs

    result = probs.mean(dim=1)                          # [B, C] — 默认 mean

    for j, mode in pool_idx.items():
        x = probs[:, :, j]                             # [B, W]
        if mode == 'max':
            result[:, j] = x.max(dim=1).values
        elif mode == 'top2':
            result[:, j] = x.topk(min(2, x.shape[1]), dim=1).values.mean(dim=1)
        elif mode == 'original_mean':
            result[:, j] = orig_probs[:, :, j].mean(dim=1)

    return result  # [B, C]


if IS_MAIN:
    n_v5 = sum(p.numel() for p in SlotHead(1152, 6, 12).parameters())
    n_rad = sum(p.numel() for p in SlotHead(2048, 6, 12).parameters())
    print(f'SlotHead params: v5 dim=1152 {n_v5/1e6:.3f}M | rad dim=2048 {n_rad/1e6:.3f}M')
    print(f'Diag pool targets: {list(DIAG_POOL_IDX.keys())}')
    print(f'Jitter TTA: {"ON" if CFG_V5.get("tta_jitter", False) else "OFF"} '
          f'(rot ±{AUG_ROT_DEG:.0f}°, scale +{AUG_SCALE:.0%}, '
          f'shift ±{AUG_SHIFT:.0%}, intensity ±{AUG_INTENSITY:.0%})')
    print('Models ready (v5 MultiViewModel + rad RadResNetModel).')
