# ============================================================
# 3a. CNN Spatial Pattern Adapter (SPA) -- unchanged from v1
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
    def forward(self, x): return self.act(self.bn(self.conv(x)))


class SPAModule(nn.Module):
    """Multi-scale spatial feature extractor."""
    def __init__(self, in_channels=5, base_ch=64):
        super().__init__()
        self.stem = ConvBlock(in_channels, base_ch)
        self.stage1 = nn.Sequential(ConvBlock(base_ch, base_ch), ConvBlock(base_ch, base_ch, 2))
        self.stage2 = nn.Sequential(ConvBlock(base_ch, base_ch*2), ConvBlock(base_ch*2, base_ch*2, 2))
        self.stage3 = nn.Sequential(ConvBlock(base_ch*2, base_ch*4), ConvBlock(base_ch*4, base_ch*4, 2))

    def forward(self, x):
        x = self.stem(x)
        s2 = self.stage1(x)
        s4 = self.stage2(s2)
        s8 = self.stage3(s4)
        return {'s2': s2, 's4': s4, 's8': s8}
