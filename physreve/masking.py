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

    # Sample origins so a full block always fits — no boundary clamping needed.
    # Clamping caused silent cell double-writes at edges, shrinking blocks and
    # making the true mask ratio lower than the target near boundaries.
    c_max = max(1, C - block_c + 1)
    p_max = max(1, P - block_t + 1)
    c0 = torch.randint(0, c_max, (B, n_blocks), device=device)  # (B, n_blocks)
    t0 = torch.randint(0, p_max, (B, n_blocks), device=device)  # (B, n_blocks)

    mask = torch.zeros(B, C, P, dtype=torch.bool, device=device)
    b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, n_blocks)

    # Only block_c * block_t iterations (typically 8) — not B * 5000
    for dc in range(block_c):
        for dt in range(block_t):
            mask[b_idx, c0 + dc, t0 + dt] = True

    return mask
