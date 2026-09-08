"""Auxiliary within-batch ranking; no extra image passes or random sampling."""
def trusted_pair_rank_loss(logits, states, soft_masks, margin=0.1):
    logits = logits.float()
    losses = []
    counts = []
    for j in range(logits.shape[1]):
        active = soft_masks[:, j] > 0
        pos = logits[(states[:, j] == 1) & active, j]
        neg = logits[(states[:, j] == -1) & active, j]
        counts.append(pos.numel() * neg.numel())
        if pos.numel() and neg.numel():
            losses.append(F.softplus(neg[None, :] - pos[:, None] + margin).mean())
    loss = torch.stack(losses).mean() if losses else logits.sum() * 0.0
    return loss, counts
