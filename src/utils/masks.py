import torch

def create_masks(src, tgt, src_pad_idx, tgt_pad_idx):
    device = src.device

    # src_mask: (batch, 1, 1, src_seq_len) — True = keep
    src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)

    # tgt padding mask: (batch, 1, tgt_len, 1) ← dim 3 not dim 2
    tgt_pad_mask = (tgt != tgt_pad_idx).unsqueeze(1).unsqueeze(3)

    # Causal mask: (1, 1, tgt_len, tgt_len) — True = keep (lower triangle)
    size = tgt.size(1)
    causal_mask = torch.tril(
        torch.ones((size, size), device=device)
    ).bool().unsqueeze(0).unsqueeze(0)

    # Combine: (batch, 1, tgt_len, tgt_len)
    tgt_mask = tgt_pad_mask & causal_mask

    return src_mask, tgt_mask