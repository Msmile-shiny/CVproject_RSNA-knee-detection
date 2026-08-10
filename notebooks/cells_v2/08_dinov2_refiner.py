# ============================================================
# 3e. DINOv2Refiner -- v2: supports partial backbone unfreeze
# ============================================================

class DINOv2Refiner(nn.Module):
    """DINOv2 (partially frozen) + SPA + CrossModalFusion + SliceTransformer + Head.

    v2 changes from v1:
      - unfreeze_layers > 0: unfreezes the last N DINOv2 transformer blocks
      - backbone_lr in optimizer group for lower LR on unfrozen DINOv2 layers
      - trainable parameter count jumps from ~1M to ~4.6M (with unfreeze_layers=6)
    """

    def __init__(self, dinov2_model, spa_channels=64, cls_dim=384,
                 num_slices=5, num_classes=12, num_heads=4,
                 st_layers=2, dropout=0.1,
                 unfreeze_layers=6):
        super().__init__()
        self.num_slices = num_slices
        self.cls_dim = cls_dim
        self.unfreeze_layers = unfreeze_layers

        self.dinov2 = dinov2_model

        # -- v2: Partial unfreeze ----------------------------------
        # DINOv2 ViT-S has 12 blocks: dinov2.blocks[0]..blocks[11]
        # Unfreeze the LAST N blocks + final norm layer
        n_blocks = len(self.dinov2.blocks)

        if unfreeze_layers == 0:
            # v1 behavior: fully frozen
            for p in self.dinov2.parameters():
                p.requires_grad = False
            self.dinov2.eval()
            if IS_MAIN:
                print('[DINOv2] Fully FROZEN (v1 compatibility mode)')
        else:
            # Freeze ALL first, then selectively unfreeze last N blocks
            for p in self.dinov2.parameters():
                p.requires_grad = False

            unfreeze_start = max(0, n_blocks - unfreeze_layers)
            for block in self.dinov2.blocks[unfreeze_start:]:
                for p in block.parameters():
                    p.requires_grad = True

            # Also unfreeze final norm
            if hasattr(self.dinov2, 'norm'):
                for p in self.dinov2.norm.parameters():
                    p.requires_grad = True

            # Keep in eval mode for deterministic DropPath/BatchNorm behavior
            # Parameters still receive gradients -- this is intentional
            self.dinov2.eval()

            trainable_dino = sum(p.numel() for p in self.dinov2.parameters() if p.requires_grad)
            total_dino = sum(p.numel() for p in self.dinov2.parameters())
            if IS_MAIN:
                print(f'[DINOv2] Blocks {unfreeze_start}-{n_blocks-1} UNFROZEN '
                      f'({trainable_dino/1e6:.1f}M / {total_dino/1e6:.1f}M params, '
                      f'{trainable_dino/total_dino*100:.0f}%)')

        # -- Trainable components (unchanged) ----------------------
        self.spa = SPAModule(in_channels=num_slices, base_ch=spa_channels)
        self.fusion = CrossModalFusion(cls_dim=cls_dim, cnn_dim=spa_channels*4,
                                       num_heads=num_heads, dropout=dropout)
        self.slice_transformer = SliceTransformer(dim=cls_dim, num_heads=num_heads,
                                                   num_layers=st_layers, dropout=dropout)
        self.head = ClassificationHead(in_features=cls_dim, hidden=512,
                                       num_classes=num_classes, dropout=CFG['head_dropout'])

        # -- Parameter summary -------------------------------------
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if IS_MAIN:
            print(f'[DINOv2Refiner] Total: {total/1e6:.1f}M | Trainable: {trainable/1e6:.1f}M '
                  f'({trainable/total*100:.0f}%) | Frozen: {(total-trainable)/1e6:.1f}M')

    def _extract_cls_batched(self, x_3ch):
        """Extract [CLS] tokens from batched 3-channel inputs.

        When unfreeze_layers > 0, runs with gradients (no torch.no_grad).
        When unfreeze_layers == 0, uses torch.no_grad for efficiency.
        """
        if self.unfreeze_layers > 0:
            features = self.dinov2.forward_features(x_3ch)
        else:
            with torch.no_grad():
                features = self.dinov2.forward_features(x_3ch)
        return features[:, 0, :]  # [B_total, cls_dim]

    def forward(self, x):
        B = x.shape[0]

        # -- SPA on 5-channel input ---------------------------------
        spa_features = self.spa(x)

        # -- Batched DINOv2: [B, 5, H, W] -> [B*5, 3, H, W] --------
        x_5bhw = x.permute(1, 0, 2, 3).contiguous()
        x_flat = x_5bhw.view(B * self.num_slices, 1, x.shape[-2], x.shape[-1])
        x_flat_3ch = x_flat.expand(-1, 3, -1, -1)
        all_cls = self._extract_cls_batched(x_flat_3ch)

        all_cls = all_cls.view(self.num_slices, B, self.cls_dim)
        all_cls = all_cls.transpose(0, 1).contiguous()

        # -- Cross-modal fusion per slice ---------------------------
        fused = []
        for i in range(self.num_slices):
            enhanced = self.fusion(all_cls[:, i, :], spa_features['s8'])
            fused.append(enhanced)

        slice_tokens = torch.stack(fused, dim=1)
        study_feature = self.slice_transformer(slice_tokens)
        return self.head(study_feature)

    def train(self, mode=True):
        super().train(mode)
        # DINOv2 stays in eval mode (deterministic DropPath/BatchNorm)
        # but parameters still receive gradients when unfrozen
        self.dinov2.eval()
        return self
