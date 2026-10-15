#!/usr/bin/env python3
"""RSNA Knee — model: 2.5D backbone + per-finding attention pooling.

The single biggest structural fix over the failed attempt (flat CNN, all slices equal):
each of the 12 findings gets its OWN attention head over the slice stack, so the model
learns which slices matter for which finding. Backbone is license-clean (timm, Apache-2.0/MIT).

Reference numbers (public 0.924 single model): coatnet_rmlp_2_rw_384, 2.5D 3-slice RGB,
per-finding attention. We keep the backbone swappable; default to a modest tier
(small beats large here — confirmed by RSNA winners).
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F

LABELS=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
        "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
NUM_FINDINGS=len(LABELS)

class PerFindingAttention(nn.Module):
    """Given per-slice features [B, S, D], produce [B, NUM_FINDINGS, D] via a
    separate attention distribution per finding (unshared weights — sharing hurt in practice)."""
    def __init__(self, dim, n_find=NUM_FINDINGS, gated=False, pos_dim=0):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_find, dim)*0.02)  # one query per finding
        self.proj = nn.Linear(dim, dim)
        # Positional encoding of each window (in-block position + plane id), added to the KEYS only, so the
        # attention can be position-aware while the pooled features stay pure content. Zero-initialised and
        # created only when pos_dim > 0: default construction and forward are bit-identical to before.
        self.pos_proj = None
        if pos_dim > 0:
            self.pos_proj = nn.Linear(pos_dim, dim)
            nn.init.zeros_(self.pos_proj.weight); nn.init.zeros_(self.pos_proj.bias)
        # GATED variant (Ilse et al., ICML 2018, arxiv.org/abs/1802.04712 -- "gated attention" in
        # ABMIL). Our default IS ABMIL's NON-gated form, so this is the one-module upgrade the MIL
        # literature measures as competitive-to-better. The gate multiplies the key path by
        # sigmoid(U(feats)) so the attention can suppress a slice on CONTENT, not just on similarity
        # to the finding query. DEFAULT OFF: self.gate is None -> forward() is bit-identical to before.
        self.gate = nn.Linear(dim, dim) if gated else None
    def forward(self, feats, bias=None, pos=None):  # feats: [B,S,D]; bias: [n_find,S] or None; pos: [B,S,P] or None
        k = self.proj(feats)             # [B,S,D]
        if self.gate is not None:
            k = k * torch.sigmoid(self.gate(feats))   # gated ABMIL: elementwise content gate
        if self.pos_proj is not None and pos is not None:
            k = k + self.pos_proj(pos)   # positional term on the key path only (zero-init -> no-op at epoch 0)
        # attention logits per finding: [B, n_find, S]
        att = torch.einsum('fd,bsd->bfs', self.q, k) / (feats.shape[-1]**0.5)
        if bias is not None:
            att = att + bias             # additive prior BEFORE softmax (0 everywhere = no change)
        att = att.softmax(-1)
        pooled = torch.einsum('bfs,bsd->bfd', att, feats)  # [B,n_find,D]
        return pooled, att

WIN5_MAP = (0, 0, 1, 2, 2)   # 5-slice window channel -> RGB channel whose pretrained filter it inherits
WIN5_SCALE = (0.5, 0.5, 1.0, 0.5, 0.5)   # ... and its share of that filter (each source channel sums to 1)
POS_DIM = 16                 # pos_enc: 8 sinusoidal dims of in-block position + 8 of plane id (see KneeModel._pos_enc)

def inflate_stem_3to5(backbone):
    """Inflate the backbone's first conv from 3 to 5 input channels IN PLACE (win_width=5).
    new_w[:, c] = old_w[:, WIN5_MAP[c]] * WIN5_SCALE[c]: the centre channel keeps the centre (G) filter at full
    weight, each outer pair shares the nearest edge (R / B) filter at half weight, so the channel sum is unchanged
    and a 5-window of IDENTICAL slices gives EXACTLY the pre-activation of the 3-window (a uniform 3/5 would not:
    2*wR + wG + 2*wB != wR + wG + wB). Only the stem weight changes shape; every other state_dict key is untouched."""
    conv = None
    for m in backbone.modules():
        if isinstance(m, nn.Conv2d):
            conv = m
            break
    assert conv is not None and conv.in_channels == 3, f"cannot inflate stem: first conv is {conv}"
    with torch.no_grad():
        w = conv.weight
        scale = torch.tensor(WIN5_SCALE, dtype=w.dtype, device=w.device).view(1, 5, 1, 1)
        conv.weight = nn.Parameter((w[:, list(WIN5_MAP)] * scale).contiguous())
    conv.in_channels = 5
    return conv

class KneeModel(nn.Module):
    def __init__(self, backbone="coatnet_rmlp_2_rw_384", stack=64, pretrained=True, grad_ckpt=True,
                 img_size=None, drop_path_rate=0.0, pool="lstm_attn", init_weights="",
                 chan_mode="adjacent", plane_emb=False, plane_sizes=(32, 20, 12), plane_prior=0.0,
                 shared_head=False, win_shuffle=0.0, gated_attn=False,
                 freeze_backbone=False, lora_r=0, lora_alpha=16.0, win_width=3, pos_enc=False):
        """pool: how per-window features become a study prediction.
           lstm_attn (default, unchanged): bi-LSTM over windows + per-finding attention + 12 linear heads.
           attn:     per-finding attention directly on backbone features (Raptor / CoAtNet consensus, no LSTM).
           meanmax:  mean + max over windows -> LayerNorm/MLP head (forum simple-CNN consensus, D0 ResNet34).
           win_width: 3 (default, shipped) or 5 adjacent slices per window ("adjacent" chan_mode only); 5 inflates
                      the backbone stem from 3 to 5 input channels so identical slices give identical pre-activations.
           pos_enc:   sinusoidal in-block position + plane id of each window, added to the attention KEYS (zero-init)."""
        super().__init__()
        assert pool in ("lstm_attn", "attn", "meanmax"), pool
        self.pool = pool
        assert chan_mode in ("adjacent", "repeat", "native"), chan_mode
        self.chan_mode = chan_mode      # "adjacent" = shipped 2.5D triplet; "repeat" = centre slice x3
        assert win_width in (3, 5), win_width
        assert win_width == 3 or chan_mode == "adjacent", "win_width=5 is only defined for chan_mode=adjacent"
        self.win_width = int(win_width)
        import timm
        kw = dict(num_classes=0, in_chans=3, global_pool='avg', drop_path_rate=drop_path_rate)
        if img_size and ("coatnet" in backbone or "maxvit" in backbone or "maxxvit" in backbone):
            kw["img_size"] = img_size          # rel-pos MaxxViT family is built for a fixed input size
        elif img_size and backbone.startswith(("vit_", "eva", "deit")):   # prefix: 'nextvit_' must NOT match
            # plain ViTs have fixed position embeddings: resample them to our input size, and pad when the
            # size is not a multiple of the patch (384 / 14 for DINOv2) instead of cropping
            kw.update(img_size=img_size, dynamic_img_size=True, dynamic_img_pad=True)
            # global_pool='avg' makes timm ViTs build an fc_norm instead of the pretrained final norm, so the
            # checkpoint no longer loads; use the CLS token, as the ViT was pretrained.
            kw["global_pool"] = "token"
        elif img_size and "swin" in backbone:
            kw["img_size"] = img_size          # Swin window grid is built for a fixed input size
            kw["window_size"] = 12             # 384/32 = 12: windows must tile the final 12x12 stage grid
        # 2.5D: feed 3-slice RGB windows. stack=64 slices -> 62 windows (or pad to 64/3).
        self.backbone = timm.create_model(backbone, pretrained=pretrained, **kw)
        if init_weights:
            # replace the (ImageNet) backbone weights with a converted checkpoint, e.g. RadImageNet.
            # strict=True: a wrong file must fail here, not train silently from a half-initialised backbone.
            sd = torch.load(init_weights, map_location="cpu")
            self.backbone.load_state_dict(sd, strict=True)
            print(f"[model] backbone initialised from {init_weights}", flush=True)
        if self.win_width == 5:
            inflate_stem_3to5(self.backbone)
        if freeze_backbone:
            # AnyMC3D's recipe: a LARGE backbone is only safe at our data scale when it is frozen and adapted
            # with a small number of params. lora_r=0 freezes it outright (head-only training).
            if lora_r > 0:
                from lora import apply_lora
                n_wrapped, n_train = apply_lora(self.backbone, r=lora_r, alpha=lora_alpha)
                print(f"[model] backbone FROZEN + LoRA r={lora_r} a={lora_alpha} on {n_wrapped} layers "
                      f"({n_train/1e6:.2f}M trainable)", flush=True)
            else:
                for p in self.backbone.parameters():
                    p.requires_grad = False
                print("[model] backbone FROZEN (no LoRA): head-only training", flush=True)
        if grad_ckpt and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)   # 64 slices/study -> activations dominate VRAM
        cfg = self.backbone.pretrained_cfg
        mean = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1)
        std  = torch.tensor(cfg.get("std",  (0.229, 0.224, 0.225))).view(1, 3, 1, 1)
        if self.win_width == 5:
            # each inflated channel keeps the mean/std of the RGB channel whose filter it copied, so a 5-window of
            # identical slices is normalised exactly as the 3-window was (see inflate_stem_3to5)
            mean = mean[:, list(WIN5_MAP)].clone()
            std  = std[:, list(WIN5_MAP)].clone()
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)
        # with num_classes=0 timm returns pre-logits features, whose width is head_hidden_size
        # (differs from num_features on backbones that have a pre-logits MLP)
        self.dim = getattr(self.backbone, "head_hidden_size", None) or self.backbone.num_features
        if pool == "meanmax":
            self.mlp = nn.Sequential(nn.LayerNorm(2 * self.dim), nn.Dropout(0.2), nn.Linear(2 * self.dim, 512),
                                     nn.GELU(), nn.Dropout(0.2), nn.Linear(512, NUM_FINDINGS))
        else:
            if pool == "lstm_attn":       # created first, exactly as before, so default state_dict keys are unchanged
                self.slice_lstm = nn.LSTM(self.dim, self.dim//2, batch_first=True, bidirectional=True)
            self.attn = PerFindingAttention(self.dim, gated=gated_attn, pos_dim=POS_DIM if pos_enc else 0)
            self.head = nn.ModuleList([nn.Linear(self.dim, 1) for _ in range(NUM_FINDINGS)])
            # ML-Decoder-style SHARED projection head: one Linear(dim, 1) applied to every finding's
            # pooled vector, instead of 12 unshared heads. ML-Decoder (Ridnik et al.) found the per-class
            # projection redundant, and CheXFusion measured +0.008 mAP swapping it in. The motivation here
            # is regularisation: 12 unshared heads on 12 noisy soft labels is where we would overfit.
            # Created only when asked, so the default state_dict keys are unchanged.
            self.shared_head = nn.Linear(self.dim, 1) if shared_head else None
        self.stack = stack
        # Plane (segment) embedding: our pooling sees 62 window features with no marker of WHICH PLANE each
        # came from, so it can only tell a coronal window from an axial one by image content. CheXFusion
        # (CXR-LT 2023 1st) measured segment embeddings + window shuffling at 0.366 -> 0.368 -> 0.372 mAP.
        # Zero-initialised and gated by plane_emb=False, so the default forward is bit-identical.
        self.plane_sizes = tuple(plane_sizes)
        self.plane_emb = None
        if plane_emb:
            self.plane_emb = nn.Parameter(torch.zeros(3, self.dim))   # sagittal / coronal / axial
        # Fixed finding->plane attention prior (not learned). 0.0 = off, and a strength of 0 is an
        # exact no-op because the bias tensor is then all zeros.
        self.plane_prior_strength = float(plane_prior)
        self.win_shuffle = float(win_shuffle)        # 0 = off (default): the forward path is untouched
        self.pos_enc = bool(pos_enc)                 # off (default): attn gets pos=None, pos_proj is never created
    def _pos_enc(self, idx, n_win, device):
        """[1, n_win, POS_DIM] fixed (not learned) positional code per scored window: 8 sinusoidal dims of the
        window's normalised position inside its plane block (0..1) + 8 sinusoidal dims of its plane id.
        Indexed by `idx` exactly like _plane_ids, so the train-windows subset gets the full path's codes."""
        sizes = self.plane_sizes
        rel = torch.empty(sum(sizes)); pl = torch.empty(sum(sizes))
        o = 0
        for p, n in enumerate(sizes):
            rel[o:o + n] = torch.arange(n, dtype=torch.float32) / max(n - 1, 1)
            pl[o:o + n] = float(p)
            o += n
        if idx is not None:
            sel = torch.as_tensor(list(idx), dtype=torch.long)
            rel, pl = rel[sel], pl[sel]
        assert rel.numel() == n_win, f"plane_sizes {sizes} does not cover {n_win} windows"
        half = POS_DIM // 4                                            # 4 freqs x (sin,cos) x (position, plane)
        octave = 2.0 ** torch.arange(half, dtype=torch.float32)[None]  # 1, 2, 4, 8
        ang_r = rel[:, None] * math.pi * octave                        # position: half-periods of 1, 1/2, 1/4, 1/8 block
        ang_p = pl[:, None] * (math.pi / 3.0) * octave                 # plane id 0/1/2 -> 3 distinct codes
        pos = torch.cat([ang_r.sin(), ang_r.cos(), ang_p.sin(), ang_p.cos()], -1)   # [n_win, POS_DIM]
        return pos[None].to(device)
    def _plane_ids(self, idx, n_win, device):
        """Plane id (0 sag, 1 cor, 2 ax) for each of the n_win windows actually scored.
        self.plane_sizes counts WINDOWS per plane, so it is the same for adjacent and native modes."""
        import torch
        sizes = self.plane_sizes
        full = torch.empty(sum(sizes), dtype=torch.long)
        o = 0
        for p, n in enumerate(sizes):
            full[o:o + n] = p
            o += n
        sel = full if idx is None else full[torch.as_tensor(list(idx), dtype=torch.long)]
        assert sel.numel() == n_win, f"plane_sizes {sizes} does not cover {n_win} windows"
        return sel.to(device)

    def forward(self, x, centers=None):  # x: [B, stack, H, W]
        """centers: optional sorted 1-D sequence of window-centre slice indices (training-time random
        subset, shared by the batch). None = every slice, which is what eval and Kaggle inference use.
        Sorted order keeps the bi-LSTM's anatomical sequence valid."""
        B,S,H,W = x.shape
        C = getattr(self, "win_width", 3)                        # slices per window (3 shipped, 5 opt-in)
        # build 3-slice RGB windows (i-1,i,i+1); pad ends by repeat
        xp = torch.cat([x[:,:1], x, x[:,-1:]], 1)               # [B,S+2,H,W]
        idx = range(S) if centers is None else [int(c) for c in centers]
        if getattr(self, "chan_mode", "adjacent") == "native":
            # cache holds [B, 3*K, H, W]: K consecutive native (j-1,j,j+1) triplets. No padding/windowing.
            assert S % 3 == 0, f"native chan_mode needs a multiple of 3 channels, got {S}"
            wins = x.reshape(B, S // 3, 3, H, W)
            if centers is not None:
                wins = wins[:, [int(c) for c in centers]]
        elif getattr(self, "chan_mode", "adjacent") == "repeat":
            # RadImageNet (and any single-channel medical pretraining) was trained on ONE grayscale image
            # replicated across RGB; feeding it 3 DIFFERENT slices is out of distribution for its first
            # conv. "repeat" gives the centre slice three times, keeping the window count identical.
            wins = torch.stack([xp[:,i+1:i+2].expand(-1,3,-1,-1) for i in idx],1)   # [B,K,3,H,W]
        elif C == 5:
            # 5-slice windows (i-2..i+2): pad by repeating each end slice twice; window i is xp5[:, i:i+5]
            xp5 = torch.cat([x[:,:1], xp, x[:,-1:]], 1)         # [B,S+4,H,W]
            wins = torch.stack([xp5[:,i:i+5] for i in idx],1)      # [B,K,5,H,W]
        else:
            wins = torch.stack([xp[:,i:i+3] for i in idx],1)        # [B,K,3,H,W]
        p = getattr(self, "chan_shuffle", 0.0)                   # train-only aug; absent/0 = off
        if self.training and p > 0:
            wins = channel_shuffle(wins, p)
        S = wins.shape[1]
        wins = wins.reshape(B*S,wins.shape[2],H,W)
        wins = (wins - self.mean) / self.std                     # ImageNet normalisation the backbone expects
        feats = self.backbone(wins).reshape(B,S,self.dim)        # [B,S,D]
        if self.plane_emb is not None:
            # one embedding per plane, added to every window feature from that plane's block.
            # In native mode the window count is S//3, and `idx` enumerated channels, so rebuild it.
            widx = (None if centers is None else [int(c) for c in centers])
            pid = self._plane_ids(widx, feats.shape[1], feats.device)
            feats = feats + self.plane_emb[pid]
        if self.pool == "meanmax":
            z = torch.cat([feats.mean(1), feats.amax(1)], -1)    # [B,2D]
            return self.mlp(z), None
        if self.pool == "lstm_attn":
            if self.training and self.win_shuffle > 0 and torch.rand(()) < self.win_shuffle:
                # Shuffle the window ORDER before the bi-LSTM (CheXFusion's "window shuffling"): it stops the
                # LSTM leaning on slice order, which is an artefact of our fixed layout rather than anatomy.
                # Train-time only, and a no-op for attn/meanmax pooling, which are order-invariant already.
                feats = feats[:, torch.randperm(feats.shape[1], device=feats.device)]
            feats,_ = self.slice_lstm(feats)                     # [B,S,D] (bi-LSTM over slices)
        bias = None
        if self.plane_prior_strength > 0:
            pid = self._plane_ids(None if centers is None else [int(c) for c in centers],
                                  feats.shape[1], feats.device)
            allow = torch.zeros(NUM_FINDINGS, 3, device=feats.device)
            for fi, lab in enumerate(LABELS):
                for pl in PLANE_PRIOR[lab]:
                    allow[fi, pl] = 1.0
            bias = allow[:, pid] * self.plane_prior_strength      # [12, n_win]
        if getattr(self, "pos_enc", False):
            pos = self._pos_enc(None if centers is None else [int(c) for c in centers],
                                feats.shape[1], feats.device).to(feats.dtype).expand(B, -1, -1)   # [B,n_win,16]
            pooled, att = self.attn(feats, bias, pos)             # [B,12,D]; pos enters the KEYS only
        else:
            pooled, att = self.attn(feats, bias)                  # [B,12,D]
        if getattr(self, "shared_head", None) is not None:
            logits = self.shared_head(pooled).squeeze(-1)          # [B,12], one shared projection
        else:
            logits = torch.cat([self.head[i](pooled[:,i]) for i in range(NUM_FINDINGS)],1)  # [B,12]
        return logits, att

def channel_shuffle(wins, p):
    """Training aug: with probability p per window, randomly permute its 3 slice channels (i-1,i,i+1).
    wins: [B,K,3,H,W]. Pixel values are unchanged, only their channel order (reported to help 2.5D
    stacks in RSNA-2024 lumbar top solutions; unverified source - judged by our own 2-seed A/B)."""
    B, K = wins.shape[:2]
    perm = torch.argsort(torch.rand(B, K, wins.shape[2], device=wins.device), dim=-1)
    shuffled = torch.gather(wins, 2, perm[..., None, None].expand(-1, -1, -1, *wins.shape[3:]))
    sel = (torch.rand(B, K, device=wins.device) < p)[..., None, None, None]
    return torch.where(sel, shuffled, wins)


class AUCMLoss(nn.Module):
    """AUC-margin (min-max) loss, one per finding (Yuan et al., ICCV 2021, arxiv.org/abs/2012.03173;
    formula as in LibAUC's AUCMLoss), extended to SOFT labels by replacing 1{y=1} with y and 1{y=0} with 1-y.
    a, b are minimised with the model; alpha is MAXIMISED: a backward hook negates its gradient so the same
    (descent) optimiser performs ascent, and clamp_alpha() keeps it >= 0 after each step.
    priors: per-finding positive rate p (mean training soft label)."""
    def __init__(self, priors, margin=1.0):
        super().__init__()
        self.register_buffer("p", torch.as_tensor(priors, dtype=torch.float32))
        self.margin = float(margin)
        n = self.p.numel()
        self.a = nn.Parameter(torch.zeros(n))
        self.b = nn.Parameter(torch.zeros(n))
        self.alpha = nn.Parameter(torch.zeros(n))
        self.alpha.register_hook(lambda g: -g)

    def forward(self, logits, y):
        s = torch.sigmoid(logits.float())                     # [B,12] scores in (0,1)
        y = y.float(); p = self.p
        pos, neg = y, 1.0 - y
        loss = ((1 - p) * ((s - self.a) ** 2 * pos).mean(0)
                + p * ((s - self.b) ** 2 * neg).mean(0)
                + 2 * self.alpha * (p * (1 - p) * self.margin + (p * s * neg - (1 - p) * s * pos).mean(0))
                - p * (1 - p) * self.alpha ** 2)
        return loss.mean()

    @torch.no_grad()
    def clamp_alpha(self):
        self.alpha.clamp_(min=0.0)


# Finding -> plane prior (2026-10-06). Which planes physically show each finding, from the public 0.944
# kernel's SLOT_PRIOR_TABLE (strength 0.55) and standard MSK MRI practice: collateral ligaments are read
# on coronal, the patellofemoral joint on axial, effusion/synovitis on fluid-sensitive sagittal+axial,
# menisci on sagittal+coronal. A POSITIVE bias on the allowed planes, added to the attention logits
# before softmax, so the model starts biased the right way but can still look anywhere.
# Order must match LABELS. Planes: 0 sagittal, 1 coronal, 2 axial.
PLANE_PRIOR = {
    "ACL":              (0,),
    "MCL":              (1,),
    "Medial Meniscus":  (0, 1),
    "Lateral Meniscus": (0, 1),
    "Medial OA":        (0, 1),
    "Lateral OA":       (0, 1),
    "PF OA":            (0, 2),
    "Effusion":         (0, 2),
    "Synovitis":        (0, 2),
    "Baker's":          (0,),
    "Contusion":        (0, 1, 2),
    "Fracture":         (0, 1, 2),
}


def soft_bce(logits, soft_targets, aux_weight=0.0):
    """Soft-label BCE. soft_targets in [0,1]. Optional per-finding aux not used by default."""
    return F.binary_cross_entropy_with_logits(logits, soft_targets)


def asl_wbce(logits, soft_targets, prevalence, gamma_neg=4.0, gamma_pos=0.0, clip=0.05):
    """Asymmetric Loss + prevalence-weighted BCE, SUMMED. Soft-label generalisation.

    CheXFusion (CXR-LT 2023 1st place) ablated these single-variable: BCE -> wBCE -> ASL -> ASL+wBCE,
    with the SUM best on common AND rare labels at once, because each half fails in the opposite
    direction (wBCE underfits the common labels, ASL alone underweights them).
    - ASL: focus on hard NEGATIVES (gamma_neg) and shift easy negatives by `clip` so they get exactly
      zero gradient. The two branches are weighted by y and (1-y), which extends to y in [0,1] directly.
    - wBCE: per-finding weights w = y*exp(1-rho) + (1-y)*exp(rho), rho = that finding's positive rate.
    prevalence: [n_find] positive rate per finding, from the training labels.
    """
    p = torch.sigmoid(logits)
    y = soft_targets
    pm = (p - clip).clamp(min=1e-8)                      # probability margin on the negative branch
    pos = y * ((1 - p).clamp(min=1e-8) ** gamma_pos) * torch.log(p.clamp(min=1e-8))
    neg = (1 - y) * (pm ** gamma_neg) * torch.log((1 - pm).clamp(min=1e-8))
    asl = -(pos + neg)
    rho = prevalence.clamp(1e-4, 1 - 1e-4).to(logits.device)
    w = y * torch.exp(1 - rho) + (1 - y) * torch.exp(rho)
    wbce = w * F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return asl.mean() + wbce.mean()

if __name__=="__main__":
    # shape self-test on CPU with a tiny fake backbone substitute (no timm/pretrained needed)
    print("model.py defines KneeModel with per-finding attention over", NUM_FINDINGS, "findings")
    print("stack->windows->backbone->biLSTM->per-finding attn->12 heads")
