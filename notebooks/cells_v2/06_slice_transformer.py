# ============================================================
# 3c. Slice Transformer -- unchanged from v1
# ============================================================

class SliceTransformer(nn.Module):
    def __init__(self, dim=384, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, 5, dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim*4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, slice_features):
        tokens = slice_features + self.pos_embed
        tokens = self.transformer(tokens)
        return self.norm(tokens.mean(dim=1))
