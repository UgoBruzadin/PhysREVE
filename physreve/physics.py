"""
Physics components: leadfield computation and physics-grounded attention bias.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import mne
from mne.datasets import fetch_fsaverage

from .config import PhysREVEConfig


def build_leadfield(
    ch_names:   list,
    sfreq:      float = 200.0,
    grid_pos:   float = 10.0,    # mm — source grid spacing
    mindist:    float = 5.0,     # mm — min source-to-surface distance
    verbose:    bool  = False,
    return_raw: bool  = False,   # if True, also return unnormalised L
):
    """
    Compute the EEG leadfield matrix for a given electrode configuration.

    Uses MNE's 3-layer sphere conductor model — no individual MRI needed.
    The sphere is automatically fitted to the electrode positions.

    Returns
    -------
    L_col  : np.ndarray  (n_ch, n_src)  column-normalised  — for physics loss
    L_row  : np.ndarray  (n_ch, n_src)  row-normalised     — for attention bias
    src_pos: np.ndarray  (n_src, 3)     active source positions (metres)
    info   : mne.Info
    L      : np.ndarray  (n_ch, n_src)  unnormalised (only when return_raw=True)
    """
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage, match_case=False, on_missing='warn')

    sphere = mne.make_sphere_model('auto', 'auto', info, verbose=False)

    src = mne.setup_volume_source_space(
        pos=grid_pos, sphere=sphere, mindist=mindist, verbose=False
    )
    n_sources = src[0]['nuse']
    if verbose:
        print(f'  Source space: {n_sources} active dipoles')

    fs_dir = Path(fetch_fsaverage(verbose=False)).parent
    trans  = str(fs_dir / 'fsaverage' / 'bem' / 'fsaverage-trans.fif')
    fwd    = mne.make_forward_solution(
        info, trans=trans, src=src, bem=sphere,
        meg=False, eeg=True, verbose=False
    )
    fwd_f  = mne.convert_forward_solution(fwd, force_fixed=True, verbose=False)
    L      = fwd_f['sol']['data'].astype(np.float32)   # (n_ch, n_src)

    if verbose:
        print(f'  Leadfield shape: {L.shape}')

    col_n  = np.linalg.norm(L, axis=0, keepdims=True).clip(min=1e-8)
    L_col  = L / col_n

    row_n  = np.linalg.norm(L, axis=1, keepdims=True).clip(min=1e-8)
    L_row  = L / row_n

    active_mask = src[0]['inuse'].astype(bool)
    src_pos = src[0]['rr'][active_mask]  # (n_src, 3)

    if return_raw:
        return L_col, L_row, src_pos, info, L
    return L_col, L_row, src_pos, info


def motor_roi_indices(info, src_pos, ch_names, radius=0.030, depth=0.78):
    """
    Find source indices near C3 (left motor) and C4 (right motor).

    Useful for the fine-tuning hemispheric asymmetry loss in motor imagery.
    For seizure detection this is not needed unless you have a specific ROI.

    depth : fraction of electrode-to-centre distance → approximate cortical depth
    radius: search radius in metres (30mm default)

    Returns lh_idx, rh_idx as numpy arrays of source indices.
    """
    def elec_pos(name):
        idx = ch_names.index(name)
        return np.array(info['chs'][idx]['loc'][:3])

    c3 = elec_pos('C3') * depth
    c4 = elec_pos('C4') * depth

    d_lh = np.linalg.norm(src_pos - c3, axis=1)
    d_rh = np.linalg.norm(src_pos - c4, axis=1)

    lh = np.where(d_lh < radius)[0]
    rh = np.where(d_rh < radius)[0]

    # Fallback: take nearest 10 sources
    if len(lh) < 3:
        lh = np.argsort(d_lh)[:10]
    if len(rh) < 3:
        rh = np.argsort(d_rh)[:10]

    return lh, rh


def roi_indices_near_electrode(info, src_pos, ch_names, elec_name,
                               radius=0.030, depth=0.78, fallback_k=10):
    """
    Generic version of motor_roi_indices: find sources near any electrode.

    Useful for building task-specific ROIs (e.g., temporal lobe for seizures).

    Returns a numpy array of source indices near the specified electrode.
    """
    idx = ch_names.index(elec_name)
    pos = np.array(info['chs'][idx]['loc'][:3]) * depth
    dists = np.linalg.norm(src_pos - pos, axis=1)
    roi = np.where(dists < radius)[0]
    if len(roi) < 3:
        roi = np.argsort(dists)[:fallback_k]
    return roi


def compute_leadfield_bias(L_row: torch.Tensor) -> torch.Tensor:
    """
    Compute electrode-electrode physics similarity from the leadfield.

    L_row: (n_ch, n_src)  row-normalised leadfield
    Returns: (n_ch, n_ch)  cosine similarity ∈ [-1, 1]
    """
    return L_row @ L_row.T


class LeadfieldAttentionBias(nn.Module):
    """
    Adds a fixed, physics-grounded bias to spatial attention scores.

    Holds the precomputed electrode-electrode similarity matrix.
    Only the learnable scale α is trained; the leadfield itself is frozen.
    """
    def __init__(self, L_row: torch.Tensor, cfg: PhysREVEConfig):
        super().__init__()
        bias = compute_leadfield_bias(L_row)  # (n_ch, n_ch)
        self.register_buffer('bias', bias)
        self.alpha = nn.Parameter(torch.tensor(cfg.leadfield_bias_scale))

    def forward(
        self,
        attn_logits: torch.Tensor,   # (B*H, n_tok, n_tok)
        ch_tok_mask: torch.Tensor,   # (n_tok, n_tok) bool
        ch_indices:  torch.Tensor,   # (n_tok,) int — -1 for CLS
    ) -> torch.Tensor:
        n_tok = attn_logits.shape[-1]
        bias_mat = torch.zeros(n_tok, n_tok, device=attn_logits.device)
        valid = ch_tok_mask
        if valid.any():
            i_idx = ch_indices.clamp(min=0)
            full_bias = self.bias[i_idx][:, i_idx]
            bias_mat[valid] = full_bias[valid]
        return attn_logits + self.alpha * bias_mat.unsqueeze(0)
