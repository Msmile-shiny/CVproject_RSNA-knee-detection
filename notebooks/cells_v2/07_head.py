# ============================================================
# 3d. Classification Head -- unchanged from v1
# ============================================================

class ClassificationHead(nn.Module):
    def __init__(self, in_features=384, hidden=512, num_classes=12, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
    def forward(self, x): return self.head(x)
