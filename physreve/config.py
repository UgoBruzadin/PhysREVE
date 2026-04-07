from dataclasses import dataclass


@dataclass
class PhysREVEConfig:
    """
    Central configuration for PhysREVE.

    Design choices and their rationale:
    ─────────────────────────────────────────────────────────────────────
    d_model=256     REVE-base size. Enough expressivity, fits on a single GPU.
    n_heads=8       4 for spatial attention, 4 for temporal attention.
    patch_size=50   samples per patch.
    n_sources=1284  ico-3 source space, both hemispheres (642 × 2).
                    Reduced to ~300 if using volume sphere model.
    lambda_phys     Physics consistency loss weight. Start at 0.15,
                    anneal to 0 over last 20% of training if needed.
    lambda_snr      SNR alignment loss weight. Prevents learning artifacts.
                    Borrowed from EEGPT.
    d_pos_*         4D positional encoding: 64 dims per coordinate.
                    Must sum to d_model.
    """
    # ── Transformer ─────────────────────────────────────────────────────
    d_model      : int   = 256
    n_heads      : int   = 8
    n_layers     : int   = 12
    d_ff         : int   = 1024
    dropout      : float = 0.2

    # ── Tokenisation ────────────────────────────────────────────────────
    patch_size   : int   = 50       # samples per patch
    max_channels : int   = 256
    max_patches  : int   = 512
    mask_ratio   : float = 0.75
    block_t      : int   = 4        # contiguous block masking: temporal span
    block_c      : int   = 2        # contiguous block masking: channel span

    # ── Source space ────────────────────────────────────────────────────
    n_sources    : int   = 1284     # updated after build_leadfield

    # ── Physics loss weights ─────────────────────────────────────────────
    lambda_phys  : float = 0.15
    lambda_snr   : float = 0.05
    lambda_asym  : float = 0.05     # fine-tuning only

    # ── Physics architecture ─────────────────────────────────────────────
    leadfield_bias_scale : float = 0.5

    # ── 4D positional encoding  (must sum to d_model) ──────────────────
    d_pos_x      : int   = 64
    d_pos_y      : int   = 64
    d_pos_z      : int   = 64
    d_pos_t      : int   = 64

    # ── Experiment metadata ──────────────────────────────────────────────
    exp_name     : str   = ''

    def __post_init__(self):
        total = self.d_pos_x + self.d_pos_y + self.d_pos_z + self.d_pos_t
        assert total == self.d_model, (
            f'4D pos enc dims sum to {total}, expected {self.d_model}'
        )
