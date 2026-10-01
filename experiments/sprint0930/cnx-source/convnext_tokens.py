#!/usr/bin/env python3
"""ConvNeXt as a token producer, so xcodex can read it unchanged.

WHY A THIRD ARCHITECTURE AT ALL. Ensembling paid only when the members were
genuinely different: G1 (triplet, 4 passes, native 3-ch stem) + a5_bs32_rank
(16-channel, 1 pass) gained +0.016 gold over the better member, while an
ensemble of the three 16-channel a5 variants LOST 0.004 gold because they are
near-duplicates. ConvNeXt is about as far from a ViT as this pipeline can go --
hierarchical, convolutional, local receptive fields -- so it is the strongest
available source of decorrelation.

WHY STAGE 2 AND NOT THE FINAL STAGE. At 336 px ConvNeXt's stride-32 output is
10x10 = 100 tokens, i.e. 33.6 mm of knee per token, which cannot localise a ~1 mm
meniscal tear. The stride-16 stage gives 21x21 = 441 tokens of 512 -- the SAME
441 tokens DINOv3 ViT-S produces at this resolution, so the readout, the label
queries and the slot handling all carry over with no change.

CLS BY GLOBAL AVERAGE, AND WHY THE "BETTER" IDEA WAS WORSE. ConvNeXt has no
class token. timm does expose a canonical global vector -- forward_head(
pre_logits=True) is norm -> global average pool -> 1024-d -- but it is not a
token and it is not the patch dimension.

The adapter therefore prepends the spatial MEAN of the stage-2 tokens. This is
exactly what ConvNeXt's own classifier pools, costs no new parameters, and is
meaningful from step 0.

A deeper-stage CLS was tried and REVERTED. Taking stage 3, pooling it and
projecting 1024 -> 512 does give a summary that is not a restatement of the
patch mean -- but that projection is randomly initialised AND, living under
enc.*, it lands in the BACKBONE parameter group at --bb-lr 1.1e-4, surrounded by
pretrained weights. It would barely train in 16 epochs, and the low
cosine-to-patch-mean that made it look attractive was mostly measuring random
projection noise.

The cost of the patch mean is real and should be reported: xcodex's baseline
branch becomes a plain average of the same tokens its delta branch attends over,
so the two are less complementary than in the ViT arms, where CLS is separately
learned. They are still not identical -- the baseline aggregates per-slot CLS
across SLOTS by mean and max, while the delta attends over every patch of every
slot -- but the CLS carries no learned selection.

NO cond=token. The slot token has to be inserted into a patch SEQUENCE that the
attention blocks then mix; ConvNeXt has no such sequence. Use cond=post, which
adds the slot embedding to every token after encoding and is already supported
by Net. This is a real difference from the ViT arms and must be reported as one:
the conditioning is weaker, not merely implemented differently.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ViTMap(nn.Module):
    """A ViT's patch tokens as a 2-D map, so spatial_mil can read them.

    THE CONTROL THIS EXISTS FOR. ConvNeXt+spatial_mil beat the ViT+xcodex
    reference on fold 0, but that arm changed THREE things at once: the backbone
    (convnet vs transformer), the readout (per-label spatial MIL vs xcodex), and
    the conditioning (cond=post vs cond=token). Running the SAME spatial_mil
    readout on the ViT isolates the backbone: if ViT+spatial_mil also beats
    ViT+xcodex, the readout was the win and ConvNeXt is incidental.

    DINOv3 ViT-S/16 at 336 px emits [CLS, reg x4, patch x441]; the 441 patches
    are a 21x21 grid at 6.19 mm per position -- finer than ConvNeXt's stage-3
    13.0 mm, so the two are not resolution-matched and that stays a confound.
    """

    def __init__(self, name: str, in_chans: int, img_size: int,
                 pretrained: bool = True):
        super().__init__()
        import timm
        self.vit = timm.create_model(name, pretrained=pretrained, num_classes=0,
                                     in_chans=in_chans, img_size=img_size)
        self.num_features = self.vit.num_features
        self.prefix = getattr(self.vit, "num_prefix_tokens", 1)

    def fresh_param_names(self):
        return []

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        f = self.vit.forward_features(x)[:, self.prefix:]     # drop CLS+registers
        n, p, d = f.shape
        g = int(round(p ** 0.5))
        if g * g != p:
            raise RuntimeError(f"{p} patch tokens is not a square grid")
        return f.transpose(1, 2).reshape(n, d, g, g)          # (N, D, g, g)


class ConvNeXtMap(nn.Module):
    """The LAST ConvNeXt feature map, unmodified. (N, 1024, 10, 10) at 336 px.

    THE POINT: no adapter, no decoder, no CLS, no new pretrained-scale weights.
    forward_features returns exactly what ConvNeXt computes, and the per-label
    spatial MIL readout consumes the map directly.

    WHY THE COARSEST MAP IS NOT THE OBVIOUS MISTAKE IT LOOKS LIKE. 10x10 = 100
    positions of 13.0 mm each sounds far too blunt for a 1 mm meniscal tear, and
    an earlier version of this file argued exactly that. But resolution cuts both
    ways once the pooling is the thing being learned:

        map        positions   mean-pool dilution   lesion as % of one position
        stage 3          100          100x                    0.6%
        stage 2          441          441x                    2.6%
        FPN 1<-2       1,764        1,764x                   10.4%

    A fine grid gives MAX pooling a cleaner peak; a coarse grid gives MEAN
    pooling less to divide by. Neither dominates, and SpatialMILReadout learns
    where each label sits between them.

    What the last map does dominate on is everything else. It is the
    representation ConvNeXt's own classifier was trained on, so it needs no
    freshly-initialised decoder -- and every decoder parameter is both untrained
    and, under the trainer's default rule, stuck at the pretrained learning rate.
    A one-level FPN added 787k random weights, 3.6% of the backbone, appearing
    from nowhere on 4,349 weakly-labelled studies. Here the only new parameters
    are the 12-way 1x1 head, and upsampling never invents detail that stride-16
    features did not contain anyway.

    A 1 mm lesion is 2.6 px at the INPUT (336 px over a 130 mm crop), so it is
    sub-position at EVERY stage including stride-4. No choice of map recovers it;
    the choice only decides how much background each position mixes in.
    """

    def __init__(self, name: str, in_chans: int, pretrained: bool = True,
                 stage: int = 3):
        super().__init__()
        import timm
        self.body = timm.create_model(name, pretrained=pretrained, num_classes=0,
                                      in_chans=in_chans, features_only=True,
                                      out_indices=(stage,))
        self.num_features = int(self.body.feature_info.channels()[-1])

    def fresh_param_names(self):
        return []                      # nothing here is randomly initialised

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)[-1]        # (N, C, H, W)


class ConvNeXtFPN(nn.Module):
    """Stage-2 semantics carried down onto the stage-1 grid: 42x42 at 3.10 mm.

    THE RESOLUTION ARGUMENT, MEASURED. At 336 px over a 130 mm crop the input is
    0.387 mm/px, so a 1 mm meniscal tear is 2.6 px BEFORE the encoder sees it and
    is sub-position at every stage (0.16 positions at stride 16, 0.32 at stride
    8). Nothing recovers what the input did not resolve -- but the FRACTION of a
    position occupied by the lesion still matters for max-like pooling:

        stride 16, 6.19 mm/pos   a 1 mm tear is ~1/38 of the area -> 97% background
        stride  8, 3.10 mm/pos   ~1/10 of the area                -> 90% background

    So the finer grid gives the per-label LSE a cleaner peak to find. This is the
    one thing ConvNeXt has that a ViT structurally does not: DINOv3 ViT-S/16
    samples at 6.19 mm and has no other resolution available.

    Stage 1 alone is the wrong answer -- it is only about six blocks deep, so it
    is spatially fine and semantically thin. This upsamples stage 2 and adds it
    to a projected stage 1, the standard top-down FPN merge, giving deep features
    on the fine grid for one 1x1 conv and a bilinear resize.
    """

    def __init__(self, name: str, in_chans: int, pretrained: bool = True,
                 out_dim: int = 256):
        super().__init__()
        import timm
        self.body = timm.create_model(name, pretrained=pretrained, num_classes=0,
                                      in_chans=in_chans, features_only=True,
                                      out_indices=(1, 2))
        c1, c2 = self.body.feature_info.channels()
        self.lat1 = nn.Conv2d(c1, out_dim, 1)
        self.lat2 = nn.Conv2d(c2, out_dim, 1)
        self.smooth = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.num_features = out_dim

    def fresh_param_names(self):
        """Names of RANDOMLY INITIALISED params, so the trainer can give them the
        head LR. Everything under body.* is pretrained; the decoder is not.

        Without this the decoder trains at --bb-lr 1.1e-4, the rate chosen for
        pretrained weights -- the same mistake a stage-3 CLS projection made
        earlier in this file's history. A fresh 787k-parameter decoder at that
        rate would barely move in 16 epochs.
        """
        return [n for n, _ in self.named_parameters() if not n.startswith("body.")]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """-> (N, out_dim, H1, W1): the fine grid carrying deep semantics."""
        f1, f2 = self.body(x)
        up = nn.functional.interpolate(self.lat2(f2), size=f1.shape[-2:],
                                       mode="bilinear", align_corners=False)
        return self.smooth(self.lat1(f1) + up)


class ConvNeXtTokens(nn.Module):
    """(N, C, H, W) -> (N, 1 + H'W', D): global-mean CLS then the stage map.

    Exposes `num_features` so Net/Readout size themselves from it, and no
    `num_prefix_tokens`, so Net._encode's prefix slice defaults to 1 and keeps
    [CLS, patches] exactly.
    """

    def __init__(self, name: str, in_chans: int, stage: int = 2,
                 pretrained: bool = True):
        super().__init__()
        import timm
        self.body = timm.create_model(name, pretrained=pretrained,
                                      num_classes=0, in_chans=in_chans,
                                      features_only=True, out_indices=(stage,))
        self.num_features = int(self.body.feature_info.channels()[-1])
        self.stage = stage
        # ConvNeXt's own head normalises before pooling; without this the token
        # scale entering xcodex depends on depth and drifts from the ViT arms.
        self.norm = nn.LayerNorm(self.num_features)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        f = self.body(x)[-1]                      # (N, D, H, W), stride 16
        t = self.norm(f.flatten(2).transpose(1, 2))           # (N, HW, D)
        cls = t.mean(1, keepdim=True)             # ConvNeXt pools by averaging
        return torch.cat([cls, t], 1)             # (N, 1+HW, D)

    def forward_head(self, x: torch.Tensor, pre_logits: bool = True):
        """Non-token pools ask for a single vector; give them the CLS."""
        return x[:, 0]

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


if __name__ == "__main__":
    m = ConvNeXtTokens("convnext_base", in_chans=16, pretrained=False).eval()
    x = torch.rand(2, 16, 336, 336)
    with torch.no_grad():
        f = m.forward_features(x)
    print(f"convnext_base  in_chans=16 @336 -> {tuple(f.shape)}  "
          f"num_features {m.num_features}")
    print(f"  441 patch tokens: {f.shape[1] - 1 == 441}")
    print(f"  CLS is the patch mean: "
          f"{torch.allclose(f[:, 0], f[:, 1:].mean(1), atol=1e-5)}")
    print(f"  no new randomly-initialised parameters: "
          f"{not any('cls_proj' in n for n, _ in m.named_parameters())}")
    n = sum(p.numel() for p in m.parameters())
    print(f"  parameters {n/1e6:.1f}M  (DINOv3 ViT-S is 22.1M)")
