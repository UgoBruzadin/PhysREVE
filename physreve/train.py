"""
PhysREVE training routines:
  run_pretraining, run_mae_pretraining, run_finetuning, run_baseline_finetune
"""
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import PhysREVEConfig
from .masking import block_mask
from .model import (
    PhysREVEPretrainModel,
    PhysREVEFineTuneModel,
    LeadfieldAttentionBias,
    count_params,
)
from .losses import pretrain_losses, finetune_losses, mae_loss


def _elec_xyz_batch(elec_xyz: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Expand (n_ch, 3) electrode positions to (B, n_ch, 3)."""
    return elec_xyz.unsqueeze(0).expand(batch_size, -1, -1)


def run_pretraining(
    cfg:       PhysREVEConfig,
    loader:    DataLoader,
    L_row:     torch.Tensor,   # (n_ch, n_src) row-normalised leadfield
    L_col:     torch.Tensor,   # (n_ch, n_src) col-normalised leadfield
    elec_xyz:  torch.Tensor,   # (n_ch, 3) electrode positions in metres
    device:    torch.device,
    n_epochs:  int   = 30,
    lr:        float = 3e-4,
    wd:        float = 1e-4,
    warmup_epochs: int = 5,
    sfreq:     float = 200.0,
    neural_bands: list = None,
    save_path: str   = None,
):
    """
    Pretrain PhysREVE on unlabeled EEG.

    Args:
        cfg:        PhysREVEConfig
        loader:     DataLoader of unlabeled EEG trials (B, C, T)
        L_row:      row-normalised leadfield for attention bias
        L_col:      column-normalised leadfield for physics loss
        elec_xyz:   (n_ch, 3) electrode positions on device
        device:     torch.device
        n_epochs:   number of training epochs
        lr:         initial learning rate
        wd:         weight decay
        warmup_epochs: linear warmup duration
        sfreq:      sampling frequency (Hz) — for SNR loss bandpower
        neural_bands: list of (lo, hi) Hz tuples for SNR loss
        save_path:  if given, saves pretrained state_dict here

    Returns:
        pretrained_model: PhysREVEPretrainModel (on device)
        history: dict of lists with keys 'mae', 'phys', 'snr', 'total'
    """
    n_ch = elec_xyz.shape[0]
    # Infer n_patches from first batch
    first_batch = next(iter(loader))
    T_win = first_batch.shape[-1]
    n_patches = T_win // cfg.patch_size

    lf_bias = LeadfieldAttentionBias(L_row, cfg).to(device)
    model   = PhysREVEPretrainModel(cfg, lf_bias).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / max(n_epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    hist  = {k: [] for k in ('mae', 'phys', 'snr', 'total')}

    print(f'Pretraining PhysREVE — {count_params(model):,} parameters')
    print(f'Epochs: {n_epochs}  |  Batches/epoch: {len(loader)}')
    print(f'Losses: MAE + {cfg.lambda_phys}·L_phys + {cfg.lambda_snr}·L_snr')

    for epoch in range(1, n_epochs + 1):
        model.train()
        ep_parts = {k: 0.0 for k in hist}
        n = 0

        for Xb in loader:
            Xb  = Xb.to(device)
            bs  = Xb.shape[0]
            xyz = _elec_xyz_batch(elec_xyz, bs)

            msk = block_mask(
                bs, n_ch, n_patches,
                ratio=cfg.mask_ratio,
                block_t=cfg.block_t,
                block_c=cfg.block_c,
                device=device
            )

            recon, src_acts, sen_p, patch_enc = model(Xb, xyz, msk)
            loss, parts = pretrain_losses(
                recon, src_acts, sen_p, patch_enc, Xb, msk,
                L_col, cfg, sfreq=sfreq, neural_bands=neural_bands
            )

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            for k, v in parts.items():
                ep_parts[k] += v * bs
            n += bs

        sched.step()
        for k in hist:
            hist[k].append(ep_parts[k] / n)

        if epoch % 5 == 0 or epoch == 1:
            print(f'  Ep {epoch:3d}/{n_epochs}  '
                  f'L_mae={hist["mae"][-1]:.4f}  '
                  f'L_phys={hist["phys"][-1]:.4f}  '
                  f'L_snr={hist["snr"][-1]:.4f}  '
                  f'L_total={hist["total"][-1]:.4f}  '
                  f'lr={sched.get_last_lr()[0]:.2e}')

    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print(f'Pretrained weights saved: {save_path}')

    return model, hist


def run_finetuning(
    pretrained:    PhysREVEPretrainModel,
    cfg:           PhysREVEConfig,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    L_col:         torch.Tensor,   # (n_ch, n_src)
    elec_xyz:      torch.Tensor,   # (n_ch, 3)
    device:        torch.device,
    n_classes:     int   = 4,
    n_epochs:      int   = 50,
    lr_head:       float = 1e-3,
    lr_enc:        float = 1e-4,
    wd:            float = 1e-4,
    freeze_enc_epochs: int = 5,
    lh_idx:        torch.Tensor = None,
    rh_idx:        torch.Tensor = None,
    left_class:    int = 0,
    right_class:   int = 1,
    label_smoothing: float = 0.1,
    lambda_phys_ft: float = 0.0,   # physics loss weight DURING fine-tuning.
                                    # Default 0: with small labelled datasets
                                    # (~200 trials), L_phys gradient competes
                                    # with CE and hurts convergence.
                                    # Set to cfg.lambda_phys to use full physics.
    save_path:     str = None,
):
    """
    Fine-tune PhysREVE on labeled EEG.

    Uses differential learning rates:
    - Encoder (pretrained): low LR — preserve learned representations
    - Classifier head (random init): high LR — learn task quickly

    Args:
        pretrained:   PhysREVEPretrainModel from run_pretraining (or random init)
        cfg:          PhysREVEConfig
        train_loader: labeled DataLoader yielding (X, y)
        val_loader:   labeled DataLoader for validation
        L_col:        column-normalised leadfield for physics loss
        elec_xyz:     (n_ch, 3) electrode positions
        device:       torch.device
        n_classes:    number of output classes (4 for MI, 2 for seizure, etc.)
        n_epochs:     training epochs
        lr_head:      learning rate for the new classification head
        lr_enc:       learning rate for the pretrained encoder
        wd:           weight decay
        freeze_enc_epochs: freeze encoder for this many epochs to warm up head
        lh_idx:       left-ROI source indices for asymmetry loss (None to disable)
        rh_idx:       right-ROI source indices for asymmetry loss (None to disable)
        left_class:   label index for left-lateralised class (asymmetry loss)
        right_class:  label index for right-lateralised class (asymmetry loss)
        label_smoothing: label smoothing for cross-entropy
        save_path:    if given, saves best model state_dict here

    Returns:
        model: PhysREVEFineTuneModel (best val checkpoint, on device)
        history: dict of lists
    """
    model = PhysREVEFineTuneModel(cfg, pretrained, n_classes=n_classes).to(device)

    enc_params  = list(model.encoder.parameters()) + list(model.src_decoder.parameters())
    head_params = list(model.classifier.parameters())
    opt = torch.optim.AdamW([
        {'params': enc_params,  'lr': lr_enc},
        {'params': head_params, 'lr': lr_head},
    ], weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr_enc / 10)

    for p in enc_params:
        p.requires_grad = False

    hist = {k: [] for k in ('train_acc', 'val_acc', 'ce', 'phys', 'asym')}
    best_val = 0.0
    best_state = None

    print(f'Fine-tuning PhysREVE ({n_classes} classes)')
    print(f'LR: encoder={lr_enc}  head={lr_head}  freeze_warmup={freeze_enc_epochs} epochs')
    asym_str = f'{cfg.lambda_asym}·L_asym' if lh_idx is not None else 'disabled'
    phys_str = f'{lambda_phys_ft}·L_phys' if lambda_phys_ft > 0 else 'L_phys disabled (small dataset)'
    print(f'Losses: CE + {phys_str} + {asym_str}')

    for epoch in range(1, n_epochs + 1):
        if epoch == freeze_enc_epochs + 1:
            for p in enc_params:
                p.requires_grad = True
            print(f'  [Epoch {epoch}] Encoder unfrozen')

        model.train()
        tr_parts = {k: 0.0 for k in ('ce', 'phys', 'asym')}
        tr_acc = 0.0
        n = 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            xyz = _elec_xyz_batch(elec_xyz, len(Xb))

            logits, src_acts, sen_p = model(Xb, xyz)
            loss, parts = finetune_losses(
                logits, src_acts, sen_p, yb, L_col, cfg,
                lh_idx=lh_idx, rh_idx=rh_idx, epoch=epoch,
                left_class=left_class, right_class=right_class,
                label_smoothing=label_smoothing,
                lambda_phys=lambda_phys_ft
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = len(yb)
            for k in ('ce', 'phys', 'asym'):
                tr_parts[k] += parts[k] * bs
            tr_acc += (logits.argmax(1) == yb).float().sum().item()
            n += bs

        sched.step()
        tr_acc /= n

        model.eval()
        va_acc = 0.0
        nv = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits, _, _ = model(Xb, _elec_xyz_batch(elec_xyz, len(Xb)))
                va_acc += (logits.argmax(1) == yb).float().sum().item()
                nv += len(yb)
        va_acc /= nv

        for k in ('ce', 'phys', 'asym'):
            hist[k].append(tr_parts[k] / n)
        hist['train_acc'].append(tr_acc)
        hist['val_acc'].append(va_acc)

        if va_acc > best_val:
            best_val  = va_acc
            best_state = deepcopy(model.state_dict())

        if epoch % 10 == 0 or epoch == 1:
            print(f'  Ep {epoch:3d}/{n_epochs}  '
                  f'train={tr_acc:.3f}  val={va_acc:.3f}  '
                  f'ce={tr_parts["ce"]/n:.3f}  '
                  f'phys={tr_parts["phys"]/n:.4f}  '
                  f'asym={tr_parts["asym"]/n:.4f}')

    model.load_state_dict(best_state)
    print(f'  Best val acc: {best_val:.3f}')

    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print(f'Fine-tuned model saved: {save_path}')

    return model, hist


def run_mae_pretraining(
    cfg:       PhysREVEConfig,
    loader:    DataLoader,
    L_row:     torch.Tensor,
    elec_xyz:  torch.Tensor,
    device:    torch.device,
    n_epochs:  int   = 30,
    lr:        float = 3e-4,
    wd:        float = 1e-4,
    warmup_epochs: int = 5,
    save_path: str   = None,
):
    """
    Base-REVE pretraining: MAE loss only, no physics components.

    Identical architecture to PhysREVE but:
    - Leadfield attention bias alpha frozen at 0 (no physics prior)
    - No source decoder physics loss (lambda_phys = 0)
    - No SNR alignment loss (lambda_snr = 0)

    Use for ablation: isolates the contribution of physics components
    relative to pure MAE pretraining.

    Returns:
        pretrained_model: PhysREVEPretrainModel (on device)
        history: dict with keys 'mae', 'total'
    """
    n_ch = elec_xyz.shape[0]
    first_batch = next(iter(loader))
    T_win = first_batch.shape[-1]
    n_patches = T_win // cfg.patch_size

    lf_bias = LeadfieldAttentionBias(L_row, cfg).to(device)
    # Freeze the leadfield bias scale at 0 — no physics prior
    lf_bias.alpha.data.fill_(0.0)
    lf_bias.alpha.requires_grad = False

    model = PhysREVEPretrainModel(cfg, lf_bias).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / max(n_epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    hist  = {k: [] for k in ('mae', 'total')}

    print(f'Base-REVE pretraining (MAE only) — {count_params(model):,} parameters')
    print(f'Epochs: {n_epochs}  |  Batches/epoch: {len(loader)}')
    print('Losses: MAE only (no L_phys, no L_snr, leadfield bias frozen at 0)')

    for epoch in range(1, n_epochs + 1):
        model.train()
        ep_mae = 0.0
        n = 0

        for Xb in loader:
            Xb  = Xb.to(device)
            bs  = Xb.shape[0]
            xyz = _elec_xyz_batch(elec_xyz, bs)

            msk = block_mask(
                bs, n_ch, n_patches,
                ratio=cfg.mask_ratio,
                block_t=cfg.block_t,
                block_c=cfg.block_c,
                device=device
            )

            recon, src_acts, sen_p, patch_enc = model(Xb, xyz, msk)

            loss = mae_loss(recon, Xb, msk)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_mae += loss.item() * bs
            n += bs

        sched.step()
        hist['mae'].append(ep_mae / n)
        hist['total'].append(ep_mae / n)

        if epoch % 5 == 0 or epoch == 1:
            print(f'  Ep {epoch:3d}/{n_epochs}  '
                  f'L_mae={hist["mae"][-1]:.4f}  '
                  f'lr={sched.get_last_lr()[0]:.2e}')

    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print(f'Base-REVE weights saved: {save_path}')

    return model, hist


def run_baseline_finetune(
    cfg:          PhysREVEConfig,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    L_row:        torch.Tensor,
    L_col:        torch.Tensor,
    elec_xyz:     torch.Tensor,
    device:       torch.device,
    n_classes:    int   = 4,
    n_epochs:     int   = 50,
    lr:           float = 3e-4,
    **kwargs,
):
    """
    Baseline: same architecture, random init, no physics pretraining.

    Wraps run_finetuning with a freshly initialised PhysREVEPretrainModel.
    Key differences from run_finetuning:
    - freeze_enc_epochs=0: with random init, freezing the encoder for any
      warmup period means the head trains on random features → bad init.
      The freeze warmup is only beneficial when the encoder is pretrained.
    - Same LR for encoder and head (no differential LR needed — no pretrained
      weights to protect).
    """
    lf_bias = LeadfieldAttentionBias(L_row, cfg).to(device)
    scratch  = PhysREVEPretrainModel(cfg, lf_bias).to(device)
    # Explicitly override freeze_enc_epochs=0 unless caller supplies it.
    kwargs.setdefault('freeze_enc_epochs', 0)
    return run_finetuning(
        scratch, cfg, train_loader, val_loader,
        L_col=L_col, elec_xyz=elec_xyz, device=device,
        n_classes=n_classes,
        n_epochs=n_epochs, lr_head=lr, lr_enc=lr,
        **kwargs
    )
