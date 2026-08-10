# ============================================================
# 3b. Cross-Modal Fusion -- unchanged from v1
# ============================================================

class CrossModalFusion(nn.Module):
    def __init__(self, cls_dim=384, cnn_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = cls_dim // num_heads
        self.cnn_proj = nn.Linear(cnn_dim, cls_dim)
        self.q_proj = nn.Linear(cls_dim, cls_dim)
        self.k_proj = nn.Linear(cls_dim, cls_dim)
        self.v_proj = nn.Linear(cls_dim, cls_dim)
        self.out_proj = nn.Linear(cls_dim, cls_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(cls_dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, cls_token, cnn_features):
        B, C, H, W = cnn_features.shape
        D = cls_token.shape[-1]
        cnn_seq = cnn_features.flatten(2).transpose(1, 2)
        cnn_seq = self.cnn_proj(cnn_seq)
        q = self.q_proj(cls_token).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(cnn_seq).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(cnn_seq).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = self.dropout(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, D)
        out = self.out_proj(out)
        gate = self.gate.tanh()
        return self.norm(cls_token + gate * out)
