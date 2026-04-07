"""
REVE-style spatio-temporal contiguous block masking.
"""
import torch


def block_mask(
    B: int, C: int, P: int,
    ratio:   float = 0.75,
    block_t: int   = 4,
    block_c: int   = 2,
    device:  torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    REVE-style spatio-temporal contiguous block masking.

    Randomly places (block_c × block_t) blocks of True in a (C, P) grid
    until approximately the target mask ratio is reached.

    Args:
        B: batch size
        C: number of channels
        P: number of time patches
        ratio: fraction of patches to mask
        block_t: temporal block span (patches)
        block_c: channel block span
        device: torch device

    Returns: (B, C, P) bool tensor — True = masked (must be reconstructed)
    """
    block_size = block_c * block_t
    # Sample enough blocks to reach ~ratio coverage (2.5x accounts for overlap)
    n_blocks = max(int(C * P * ratio / block_size * 2.5), 20)

    # Sample all block origins at once — fully vectorized, no Python loops over B
    c0 = torch.randint(0, C, (B, n_blocks), device=device)  # (B, n_blocks)
    t0 = torch.randint(0, P, (B, n_blocks), device=device)  # (B, n_blocks)

    mask = torch.zeros(B, C, P, dtype=torch.bool, device=device)
    b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, n_blocks)

    # Only block_c * block_t iterations (typically 8) — not B * 5000
    for dc in range(block_c):
        for dt in range(block_t):
            ci = (c0 + dc).clamp(max=C - 1)
            ti = (t0 + dt).clamp(max=P - 1)
            mask[b_idx, ci, ti] = True

    return mask
