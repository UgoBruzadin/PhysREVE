"""
PhysREVE model architecture:
  FourDPositionalEncoding, PhysREVEPatchEmbed, PhysREVETransformerLayer,
  PhysREVEEncoder, PhysREVEPretrainModel, PhysREVEFineTuneModel
"""
import torch
import torch.nn as nn

from .config import PhysREVEConfig
from .physics import LeadfieldAttentionBias


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class FourDPositionalEncoding(nn.Module):
    """
    Continuous 4D positional encoding for EEG patches.

    Each patch token (electrode e, time window t) receives a positional encoding
    built from the electrode's 3D scalp coordinates plus the temporal patch index.

    Uses independent sinusoidal encodings per axis — geometry is explicit and
    interpretable: axis frequencies are orthogonal.

    Args:
        cfg: PhysREVEConfig
        coord_noise_std: σ for coordinate perturbation during training (metres)
    """
    def __init__(self, cfg: PhysREVEConfig, coord_noise_std: float = 0.003):
        super().__init__()
        self.cfg = cfg
        self.noise_std = coord_noise_std

        def freq_base(dim):
            return 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

        self.register_buffer('freq_x', freq_base(cfg.d_pos_x))
        self.register_buffer('freq_y', freq_base(cfg.d_pos_y))
        self.register_buffer('freq_z', freq_base(cfg.d_pos_z))
        self.register_buffer('freq_t', freq_base(cfg.d_pos_t))

    def _encode_axis(self, v: torch.Tensor, freqs: torch.Tensor, dim: int):
        angles = v.unsqueeze(-1) * freqs
        enc = torch.zeros(*v.shape, dim, device=v.device)
        enc[..., 0::2] = torch.sin(angles)
        enc[..., 1::2] = torch.cos(angles)
        return enc

    def forward(
        self,
        elec_xyz: torch.Tensor,   # (B, n_ch, 3) electrode positions in metres
        t_idx:    torch.Tensor,   # (B, n_patches) temporal patch indices
    ) -> torch.Tensor:
        """Returns: (B, n_ch, n_patches, d_model)"""
        B, C, _ = elec_xyz.shape
        P = t_idx.shape[1]

        if self.training and self.noise_std > 0:
            elec_xyz = elec_xyz + torch.randn_like(elec_xyz) * self.noise_std

        ex = self._encode_axis(elec_xyz[..., 0], self.freq_x, self.cfg.d_pos_x)
        ey = self._encode_axis(elec_xyz[..., 1], self.freq_y, self.cfg.d_pos_y)
        ez = self._encode_axis(elec_xyz[..., 2], self.freq_z, self.cfg.d_pos_z)
        spatial = torch.cat([ex, ey, ez], dim=-1)           # (B, C, d_spatial)

        et = self._encode_axis(t_idx.float(), self.freq_t, self.cfg.d_pos_t)  # (B,P,dt)

        spatial_bcast  = spatial.unsqueeze(2).expand(B, C, P, -1)
        temporal_bcast = et.unsqueeze(1).expand(B, C, P, -1)

        return torch.cat([spatial_bcast, temporal_bcast], dim=-1)  # (B, C, P, d_model)


class PhysREVEPatchEmbed(nn.Module):
    """
    Tokenise raw EEG into (channel, time-patch) tokens.

    Pipeline:
      raw EEG (B, C, T)
        → depthwise temporal conv → patch features  (B, C, P, d_model)
        → add 4D positional encoding
        → flatten to token sequence  (B, C*P, d_model)
    """
    def __init__(self, cfg: PhysREVEConfig):
        super().__init__()
        self.patch_size = cfg.patch_size
        self.d = cfg.d_model

        self.spatial = nn.Conv1d(1, cfg.d_model, kernel_size=1, bias=False)
        self.temporal = nn.Sequential(
            nn.Conv1d(cfg.d_model, cfg.d_model,
                      kernel_size=cfg.patch_size,
                      stride=cfg.patch_size, bias=False),
            nn.GELU()
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.pos_enc = FourDPositionalEncoding(cfg)

    def forward(self, x: torch.Tensor, elec_xyz: torch.Tensor):
        """
        x:        (B, C, T)   raw EEG, z-scored
        elec_xyz: (B, C, 3)   electrode positions in metres

        Returns:
          tokens:        (B, C*P, d)
          sensor_patches:(B, C, P)    raw signal averaged per patch
          ch_indices:    (C*P,)       channel index per token
        """
        B, C, T = x.shape
        P = T // self.patch_size

        xp = x[:, :, :P * self.patch_size]
        xp = xp.reshape(B, C, P, self.patch_size).mean(-1)  # (B, C, P)

        x_ch = x.reshape(B * C, 1, T)
        h    = self.spatial(x_ch)
        h    = self.temporal(h)
        h    = h.transpose(1, 2)
        h    = self.norm(h)
        h    = h.reshape(B, C, P, self.d)

        t_idx = torch.arange(P, device=x.device).unsqueeze(0).expand(B, -1)
        pe    = self.pos_enc(elec_xyz, t_idx)
        h     = h + pe

        tokens = h.reshape(B, C * P, self.d)
        ch_indices = torch.arange(C, device=x.device).unsqueeze(1)\
                         .expand(C, P).reshape(-1)

        return tokens, xp, ch_indices


class PhysREVETransformerLayer(nn.Module):
    """
    Transformer encoder layer with leadfield attention bias.
    """
    def __init__(self, cfg: PhysREVEConfig, lf_bias: LeadfieldAttentionBias):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads,
            dropout=cfg.dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout)
        )
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.lf_bias = lf_bias

    def forward(
        self,
        x:          torch.Tensor,   # (B, n_tok, d)
        ch_indices: torch.Tensor,   # (n_tok,) int
        attn_mask:  torch.Tensor = None,
    ):
        x_n = self.norm1(x)

        # Build physics-grounded additive bias and inject into attention logits
        # before softmax — the correct REVE integration (ALiBi-style).
        lf_bias = self.lf_bias.compute_bias(ch_indices)    # (n_tok, n_tok)
        if self.lf_bias.alpha.item() != 0.0:
            effective_mask = lf_bias if attn_mask is None else lf_bias + attn_mask
        else:
            effective_mask = attn_mask

        attn_out, _ = self.attn(x_n, x_n, x_n,
                                attn_mask=effective_mask,
                                need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class PhysREVEEncoder(nn.Module):
    """
    Full PhysREVE encoder: patch embedding → Transformer → CLS embedding.

    Used for both pretraining and fine-tuning.
    """
    def __init__(self, cfg: PhysREVEConfig, lf_bias: LeadfieldAttentionBias):
        super().__init__()
        self.patch_embed = PhysREVEPatchEmbed(cfg)
        self.cls_token   = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.layers      = nn.ModuleList([
            PhysREVETransformerLayer(cfg, lf_bias) for _ in range(cfg.n_layers)
        ])
        self.final_norm  = nn.LayerNorm(cfg.d_model)
        self.d = cfg.d_model

    def forward(self, x: torch.Tensor, elec_xyz: torch.Tensor):
        """
        x:        (B, C, T)
        elec_xyz: (B, C, 3)

        Returns:
          cls_out:       (B, d)
          patch_out:     (B, C*P, d)
          sensor_patches:(B, C, P)
          ch_indices:    (C*P+1,)   — includes CLS at index 0 (value -1)
        """
        B = x.shape[0]
        tokens, sen_p, ch_idx = self.patch_embed(x, elec_xyz)

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        ch_idx_cls = torch.cat([torch.tensor([-1], device=x.device), ch_idx])

        for layer in self.layers:
            tokens = layer(tokens, ch_idx_cls)

        tokens = self.final_norm(tokens)
        return tokens[:, 0], tokens[:, 1:], sen_p, ch_idx_cls[1:]


class PhysREVEPretrainModel(nn.Module):
    """
    PhysREVE pretraining model.

    = PhysREVEEncoder
    + MAE decoder  (reconstruct masked sensor patches)
    + Source decoder (estimate source activations → physics loss)

    The source decoder is retained after pretraining for the asymmetry loss
    during fine-tuning, then discarded for inference.
    """
    def __init__(self, cfg: PhysREVEConfig, lf_bias: LeadfieldAttentionBias):
        super().__init__()
        d = cfg.d_model

        self.encoder = PhysREVEEncoder(cfg, lf_bias)

        self.mask_token  = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        dec_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=4, dim_feedforward=d * 2,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.mae_decoder = nn.TransformerEncoder(dec_layer, num_layers=2)
        self.mae_head    = nn.Linear(d, cfg.patch_size)

        self.src_decoder = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, cfg.n_sources)
        )

    def forward(
        self,
        x:        torch.Tensor,   # (B, C, T)
        elec_xyz: torch.Tensor,   # (B, C, 3)
        mask:     torch.Tensor,   # (B, C, P) bool — True = masked
    ):
        """
        Returns:
          mae_recon:     (B, C, P, patch_size)
          src_acts:      (B, n_src, P)
          sensor_patches:(B, C, P)
          patch_enc:     (B, C*P, d)
        """
        B, C, T = x.shape
        ps = self.encoder.patch_embed.patch_size
        P = T // ps

        # Zero masked patches — vectorized: expand (B,C,P) → (B,C,T) via repeat_interleave
        mask_t = mask.repeat_interleave(ps, dim=2)          # (B, C, P*ps)
        x_masked = x.clone()
        x_masked[:, :, :P * ps] = x[:, :, :P * ps].masked_fill(mask_t, 0.0)

        cls_out, patch_enc, sen_p, ch_idx = self.encoder(x_masked, elec_xyz)

        # Replace masked token positions with mask token — vectorized
        mask_flat = mask.reshape(B, C * P)                  # (B, C*P) bool
        patch_enc_dec = torch.where(
            mask_flat.unsqueeze(-1),
            self.mask_token.expand(B, C * P, -1),
            patch_enc,
        )

        dec_out = self.mae_decoder(patch_enc_dec)
        mae_recon_flat = self.mae_head(dec_out)
        mae_recon = mae_recon_flat.reshape(B, C, P, -1)

        src_all  = self.src_decoder(patch_enc)           # (B, C*P, n_src)
        src_all  = src_all.reshape(B, C, P, -1)
        src_avg  = src_all.mean(dim=1)                   # (B, P, n_src)
        src_acts = src_avg.transpose(1, 2)               # (B, n_src, P)

        return mae_recon, src_acts, sen_p, patch_enc


class PhysREVEFineTuneModel(nn.Module):
    """
    PhysREVE fine-tuning model.

    = Pretrained PhysREVEEncoder
    + Source decoder (for asymmetry loss during fine-tuning)
    + Classification head

    n_classes: number of output classes. Set to 4 for BCI IV 2a motor imagery,
               2 for binary seizure detection, etc.
    """
    def __init__(
        self,
        cfg:            PhysREVEConfig,
        pretrain_model: PhysREVEPretrainModel,
        n_classes:      int = 4,
    ):
        super().__init__()
        self.encoder     = pretrain_model.encoder
        self.src_decoder = pretrain_model.src_decoder
        self.classifier  = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, n_classes)
        )
        self._init_head()

    def _init_head(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, elec_xyz: torch.Tensor):
        """
        Returns:
          logits:   (B, n_classes)
          src_acts: (B, n_src, P)
          sen_p:    (B, C, P)
        """
        cls_out, patch_enc, sen_p, ch_idx = self.encoder(x, elec_xyz)
        logits = self.classifier(cls_out)

        B, C, T = x.shape
        P = T // self.encoder.patch_embed.patch_size
        src_all  = self.src_decoder(patch_enc)
        src_all  = src_all.reshape(B, C, P, -1).mean(1)
        src_acts = src_all.transpose(1, 2)

        return logits, src_acts, sen_p
