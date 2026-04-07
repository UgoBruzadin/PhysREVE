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
    until the target mask ratio is reached.

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
    mask   = torch.zeros(B, C, P, dtype=torch.bool, device=device)
    target = int(C * P * ratio)

    for b in range(B):
        n_masked = 0
        for _ in range(5000):
            if n_masked >= target:
                break
            c0 = torch.randint(0, C, ()).item()
            t0 = torch.randint(0, P, ()).item()
            c1 = min(c0 + block_c, C)
            t1 = min(t0 + block_t, P)
            mask[b, c0:c1, t0:t1] = True
            n_masked = mask[b].sum().item()

    return mask
