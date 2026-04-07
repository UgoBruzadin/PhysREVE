"""
PhysREVE loss functions:
  mae_loss, physics_loss, snr_alignment_loss, asymmetry_loss,
  pretrain_losses, finetune_losses
"""
import torch
import torch.nn.functional as F

from .config import PhysREVEConfig


def mae_loss(
    recon:  torch.Tensor,    # (B, C, P, patch_size)
    target: torch.Tensor,    # (B, C, T)  raw EEG
    mask:   torch.Tensor,    # (B, C, P) bool
) -> torch.Tensor:
    """
    REVE-style MAE loss: MSE on masked patches only.
    Targets are z-scored within each patch for stability.
    """
    B, C, P, ps = recon.shape
    t_patches = target[:, :, :P * ps].reshape(B, C, P, ps)
    t_mu  = t_patches.mean(dim=-1, keepdim=True)
    t_std = t_patches.std( dim=-1, keepdim=True).clamp(min=1e-6)
    t_n   = (t_patches - t_mu) / t_std

    if mask.sum() == 0:
        return recon.new_zeros([])
    return F.mse_loss(recon[mask], t_n[mask])


def physics_loss(
    L:        torch.Tensor,   # (n_ch, n_src)  column-normalised leadfield
    src_acts: torch.Tensor,   # (B, n_src, P)
    sensor_p: torch.Tensor,   # (B, n_ch, P)
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Forward model consistency: L @ source_est ≈ sensor.

    Enforces topographic shape consistency, not absolute amplitude.
    Gradient flows through src_acts only (L is fixed physics).
    """
    recon = torch.einsum('cs,bsp->bcp', L, src_acts)

    r_n = recon    / (recon.norm(   dim=1, keepdim=True) + eps)
    y_n = sensor_p / (sensor_p.norm(dim=1, keepdim=True) + eps)

    return F.mse_loss(r_n, y_n)


def snr_alignment_loss(
    patch_enc:    torch.Tensor,   # (B, C*P, d)
    x_raw:        torch.Tensor,   # (B, C, T)
    patch_size:   int,
    sfreq:        float = 200.0,
    neural_bands: list  = None,   # list of (lo, hi) Hz tuples
    snr_quantile: float = 0.5,
    temperature:  float = 0.1,
    eps:          float = 1e-8,
) -> torch.Tensor:
    """
    EEGPT-inspired SNR alignment loss.

    High-SNR patches (predominantly neural rather than artifact) are pulled
    together in representation space using InfoNCE.

    neural_bands defaults to [(4,8),(8,13),(13,30)] (theta/alpha/beta).
    For seizure detection you may want to include higher gamma bands, e.g.
    [(1,4),(4,8),(8,13),(13,30),(30,80)].
    """
    if neural_bands is None:
        neural_bands = [(4, 8), (8, 13), (13, 30)]

    B, C, T = x_raw.shape
    P = T // patch_size

    xp = x_raw[:, :, :P * patch_size].reshape(B, C, P, patch_size)
    with torch.no_grad():
        spec   = torch.fft.rfft(xp, dim=-1).abs().pow(2)
        freqs  = torch.fft.rfftfreq(patch_size, d=1.0 / sfreq).to(x_raw.device)

        neural_mask = torch.zeros(len(freqs), dtype=torch.bool, device=x_raw.device)
        for lo, hi in neural_bands:
            neural_mask |= (freqs >= lo) & (freqs <= hi)

        neural_pwr = spec[..., neural_mask].mean(dim=(1, 3))
        total_pwr  = spec.mean(dim=(1, 3)).clamp(min=eps)
        snr        = neural_pwr / total_pwr

        thresh  = snr.median(dim=1, keepdim=True).values
        hi_mask = snr >= thresh

    z   = patch_enc.reshape(B, C, P, -1).mean(dim=1)
    z_n = F.normalize(z, dim=-1)

    total_loss = z.new_zeros([])
    n_terms = 0

    for b in range(B):
        hi_idx = hi_mask[b].nonzero(as_tuple=True)[0]
        K = len(hi_idx)
        if K < 2:
            continue
        z_hi = z_n[b, hi_idx]
        sim  = z_hi @ z_hi.T / temperature
        for i in range(K - 1):
            numerator   = sim[i, i + 1].exp()
            denominator = sim[i].exp().sum() - sim[i, i].exp() + eps
            total_loss  = total_loss - torch.log(numerator / denominator + eps)
            n_terms    += 1

    return total_loss / max(n_terms, 1)


def asymmetry_loss(
    src_acts: torch.Tensor,   # (B, n_src, P)
    labels:   torch.Tensor,   # (B,)  class indices
    lh_idx:   torch.Tensor,   # LongTensor — left-ROI source indices
    rh_idx:   torch.Tensor,   # LongTensor — right-ROI source indices
    left_class:  int = 0,     # label index for left lateralised class
    right_class: int = 1,     # label index for right lateralised class
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Hemispheric ERD asymmetry constraint.

    Originally designed for left/right motor imagery (motor cortex ERD).
    Can be adapted for any task with known hemispheric lateralisation:
      - Seizure detection: e.g. ictal vs interictal with known focus laterality
      - Language: left vs right hemisphere dominance

    The contralateral hemisphere desynchronises (ERD):
      left_class  → right-ROI power < left-ROI power
      right_class → left-ROI power < right-ROI power

    Only trials belonging to left_class or right_class contribute to the loss.
    """
    lh_p = src_acts[:, lh_idx, :].pow(2).mean(dim=(1, 2))
    rh_p = src_acts[:, rh_idx, :].pow(2).mean(dim=(1, 2))

    raw  = (rh_p - lh_p) / (lh_p + rh_p + eps)
    asym = torch.tanh(3.0 * raw)

    mi_mask = (labels == left_class) | (labels == right_class)
    if mi_mask.sum() == 0:
        return src_acts.new_zeros([])

    sign = torch.where(
        labels[mi_mask] == right_class,
        asym.new_ones(mi_mask.sum()),
        -asym.new_ones(mi_mask.sum())
    )
    return -torch.mean(sign * asym[mi_mask])


def pretrain_losses(
    recon:     torch.Tensor,
    src_acts:  torch.Tensor,
    sen_p:     torch.Tensor,
    patch_enc: torch.Tensor,
    x_raw:     torch.Tensor,
    mask:      torch.Tensor,
    L:         torch.Tensor,
    cfg:       PhysREVEConfig,
    sfreq:     float = 200.0,
    neural_bands: list = None,
):
    """
    Pretraining total loss: L_MAE + λ_phys * L_phys + λ_snr * L_snr
    Returns (total_loss, {component_name: float}).
    """
    lmae  = mae_loss(recon, x_raw, mask)
    lphys = physics_loss(L, src_acts, sen_p)
    lsnr  = snr_alignment_loss(
        patch_enc, x_raw, cfg.patch_size,
        sfreq=sfreq, neural_bands=neural_bands
    )

    total = lmae + cfg.lambda_phys * lphys + cfg.lambda_snr * lsnr
    return total, {
        'mae':   lmae.item(),
        'phys':  lphys.item(),
        'snr':   lsnr.item(),
        'total': total.item()
    }


def finetune_losses(
    logits:   torch.Tensor,
    src_acts: torch.Tensor,
    sen_p:    torch.Tensor,
    labels:   torch.Tensor,
    L:        torch.Tensor,
    cfg:      PhysREVEConfig,
    lh_idx:   torch.Tensor = None,
    rh_idx:   torch.Tensor = None,
    epoch:    int = 1,
    left_class:  int = 0,
    right_class: int = 1,
    label_smoothing: float = 0.1,
    lambda_phys:     float = None,  # overrides cfg.lambda_phys when set.
                                    # Use 0.0 for small datasets where the
                                    # physics gradient competes with CE.
):
    """
    Fine-tuning total loss: L_CE + λ_phys * L_phys + λ_asym * L_asym

    L_asym is zeroed for the first 10 epochs so CE can stabilise.
    Pass lh_idx=None and rh_idx=None to disable asymmetry loss entirely
    (e.g. for seizure detection without hemispheric lateralisation).

    Returns (total_loss, {component_name: float}).
    """
    phys_w = cfg.lambda_phys if lambda_phys is None else lambda_phys

    lce   = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
    lphys = physics_loss(L, src_acts, sen_p)

    if lh_idx is not None and rh_idx is not None:
        lasym = asymmetry_loss(src_acts, labels, lh_idx, rh_idx,
                               left_class=left_class, right_class=right_class)
        asym_w = cfg.lambda_asym if epoch >= 10 else 0.0
    else:
        lasym  = logits.new_zeros([])
        asym_w = 0.0

    total = lce + phys_w * lphys + asym_w * lasym
    return total, {
        'ce':    lce.item(),
        'phys':  lphys.item(),
        'asym':  lasym.item(),
        'total': total.item()
    }
