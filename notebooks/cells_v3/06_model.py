# ============================================================
# v3: SlotHead + MultiViewModel — matching reference architecture
# ============================================================

class SlotHead(nn.Module):
    """Per-diagnosis attention over MRI slots with anatomical priors.

    Matching reference code (knee-rsna.ipynb):
      - Projects slot features: LayerNorm → Linear → GELU
      - Learned slot embedding + per-diagnosis query vectors
      - Anatomical prior biases attention (soft, additive, 0.55)
      - Mask fills missing slots with -1e4 before softmax
    """

    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
        )
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden

        # Anatomical prior: which slots each diagnosis prefers
        prior = torch.zeros(n_out, n_slot)
        for target_name, slot_indices in SLOT_PRIORS.items():
            if target_name in TARGET_COLUMNS:
                prior[TARGET_COLUMNS.index(target_name), list(slot_indices)] = 0.55
        self.register_buffer("slot_prior", prior)

    def forward(self, x, mask):
        """x: [B, S, D]  slot features
           mask: [B, S]  1=present, 0=missing
        Returns: [B, n_out] logits
        """
        h = self.proj(x) + self.slot_emb                              # [B, S, H]
        attention = (
            torch.einsum("bsh,oh->bos", h, self.query)                # [B, n_out, S]
            / math.sqrt(self.hidden)
            + self.slot_prior.unsqueeze(0)                             # add anatomical bias
        )
        attention = attention.masked_fill(
            mask.unsqueeze(1) < 0.5, -1e4
        ).softmax(-1)                                                  # [B, n_out, S]
        context = self.drop(torch.einsum("bos,bsh->boh", attention, h))  # [B, n_out, H]
        return (context * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias


class MultiViewModel(nn.Module):
    """DINOv2 + SlotHead for multi-view knee MRI.

    Data flow:
      Input:  [B, 6, 3, 224, 224]  (batch, slots, RGB-channels, H, W)
      → DINOv2 per-slot: [B*6, 3, 224, 224] → [B*6, 257, 384]
      → CLS[0] + mean(patches) + focal_topk(patches) → [B*6, 1152]
      → Reshape: [B, 6, 1152]
      → SlotHead: [B, 6, 1152] × mask → [B, 12]

    Matching reference code:
      - CLS token + mean patch + focal top-k (12.5% of patches)
      - ImageNet normalization
      - Partial DINOv2 unfreeze
    """

    def __init__(self, dinov2_model, n_slots=6, cls_dim=384,
                 n_classes=12, slot_hidden=256, dropout=0.2,
                 unfreeze_layers=6):
        super().__init__()
        self.n_slots = n_slots
        self.cls_dim = cls_dim
        self.feature_dim = cls_dim * 3  # CLS + mean + focal
        self.unfreeze_layers = unfreeze_layers

        self.dinov2 = dinov2_model

        # Partial unfreeze (matching reference)
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
            trainable_dino = sum(p.numel() for p in self.dinov2.parameters() if p.requires_grad)
            total_dino = sum(p.numel() for p in self.dinov2.parameters())
            if IS_MAIN:
                print(f'[DINOv2] Blocks {unfreeze_start}-{n_blocks-1} UNFROZEN '
                      f'({trainable_dino/1e6:.1f}M / {total_dino/1e6:.1f}M params)')

        # SlotHead
        self.head = SlotHead(
            dim=self.feature_dim, n_slot=n_slots, n_out=n_classes,
            hidden=slot_hidden, p=dropout,
        )

        # ImageNet normalization (matching reference)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Summary
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if IS_MAIN:
            print(f'[MultiViewModel] Total: {total/1e6:.1f}M | '
                  f'Trainable: {trainable/1e6:.1f}M ({trainable/total*100:.0f}%)')

    def _extract_features(self, x_3ch):
        """Extract CLS + mean + focal_topk from DINOv2 features.

        x_3ch: [B_total, 3, H, W] float32, already normalized
        Returns: [B_total, feature_dim]
        """
        if self.unfreeze_layers > 0:
            features = self.dinov2.forward_features(x_3ch)  # [B, N+1, D]
        else:
            with torch.no_grad():
                features = self.dinov2.forward_features(x_3ch)

        cls = features[:, 0, :]                              # [B, D]
        patches = features[:, 1:, :]                         # [B, N, D]
        mean_p = patches.mean(dim=1)                         # [B, D]
        k = max(1, patches.shape[1] // 8)                    # top 12.5%
        focal = patches.topk(k, dim=1).values.mean(dim=1)    # [B, D]
        return torch.cat([cls, mean_p, focal], dim=1)        # [B, 3*D]

    def forward(self, images, mask):
        """images: [B, S, 3, H, W] uint8 in [0, 255]
           mask:   [B, S] float32, 1=present, 0=missing
        Returns: [B, n_classes] logits
        """
        B, S = images.shape[:2]

        # Flatten slots → batch dimension
        x = images.reshape(B * S, 3, images.shape[-2], images.shape[-1])
        x = x.float().div_(255.0)
        x = (x - self.mean) / self.std

        # DINOv2 feature extraction
        features = self._extract_features(x)                 # [B*S, feature_dim]
        features = features.reshape(B, S, -1)                # [B, S, feature_dim]

        # SlotHead with mask
        return self.head(features, mask)                     # [B, n_classes]

    def train(self, mode=True):
        super().train(mode)
        # DINOv2 stays in eval mode (deterministic DropPath/norm)
        # but parameters still receive gradients when unfrozen
        self.dinov2.eval()
        return self
