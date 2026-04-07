# PhysREVE — Change Log

All modifications to the notebook and supporting files are documented here in reverse-chronological order. Each entry records *what* changed, *where*, and *why*.

---

## [0.2.0] — 2026-04-07

### Fine-Tuning Improvements (3 changes to `physreve_notebook.ipynb`)

These changes were motivated by the root-cause analysis in [`physreve_analysis.md`](physreve_analysis.md), specifically sections 2.2 (asymmetry collapse) and 2.6 (encoder under-adaptation).

---

#### 1. Raised `lambda_asym`: 0.08 → 0.15
**Location:** `PhysREVEConfig` dataclass (Cell 5)

```python
# Before
lambda_asym  : float = 0.08     # hemispheric ERD asymmetry

# After
lambda_asym  : float = 0.15     # hemispheric ERD asymmetry (raised: stronger neuroscience prior)
```

**Why:** The asymmetry loss was saturating to −1.0 by epoch 20 and contributing no useful gradient. Increasing its weight forces the encoder to reorganise representations around the contralateral ERD pattern — the dominant, well-established neural signature of motor imagery. Combined with the freeze warmup below, the head is stable before the encoder receives this stronger push.

---

#### 2. Raised encoder LR: 3e-5 → 1e-4
**Location:** `run_finetuning()` signature (Cell 24)

```python
# Before
lr_enc:    float = 3e-5,    # lower LR for pretrained encoder

# After
lr_enc:    float = 1e-4,    # lower LR for pretrained encoder (raised to close gap with head)
```

**Why:** The original 33× gap between `lr_head=1e-3` and `lr_enc=3e-5` near-froze the encoder. The head was forced to classify *around* fixed pretrained representations rather than letting the physics losses reshape them. At 1e-4 (a 10× differential) the encoder can adapt while still being guided by the head's gradients.

---

#### 3. Added 5-epoch encoder freeze warmup
**Location:** `run_finetuning()` signature and training loop (Cell 24)

```python
# New parameter
freeze_enc_epochs: int = 5,     # keep encoder frozen for first N epochs

# Freeze on entry (added after optimizer setup)
for p in enc_params:
    p.requires_grad = False

# Unfreeze inside training loop at epoch freeze_enc_epochs + 1
if epoch == freeze_enc_epochs + 1:
    for p in enc_params:
        p.requires_grad = True
    print(f'  [Epoch {epoch}] Encoder unfrozen — physics alignment begins')
```

**Why:** With a higher encoder LR (change #2), the encoder could corrupt pretrained weights before the classification head has stable gradients to guide it. Freezing the encoder for the first 5 epochs lets the head orient to the task structure first, then full fine-tuning begins from a better-initialised head. This is a standard "warmup then unfreeze" pattern for transfer learning.

---

### Documentation Added

| File | Description |
|---|---|
| `README.md` | Project overview, hypothesis, architecture diagrams, results, roadmap |
| `CHANGELOG.md` | This file |

---

## [0.1.0] — 2026-04-07 (initial state at analysis time)

### Baseline Configuration

State of the notebook when the root-cause analysis was conducted. Recorded here as a reference point for all future diffs.

| Parameter | Value | Location |
|---|---|---|
| `patch_size` | 50 | `PhysREVEConfig` |
| `lambda_phys` | 0.15 | `PhysREVEConfig` |
| `lambda_snr` | 0.05 | `PhysREVEConfig` |
| `lambda_asym` | 0.08 | `PhysREVEConfig` |
| `lr_head` | 1e-3 | `run_finetuning()` |
| `lr_enc` | 3e-5 | `run_finetuning()` |
| `n_epochs` (pretrain) | 30 | `run_pretraining()` |
| `n_epochs` (finetune) | 50 | `run_finetuning()` |
| `mask_ratio` | 0.75 | `PhysREVEConfig` |
| `dropout` | 0.1 | `PhysREVEConfig` |
| Encoder freeze warmup | None | — |
| Gradient clipping | 1.0 | `run_finetuning()` loop |

### Observed Results at Baseline

| Model | Test Accuracy |
|---|---|
| Chance | 25.0% |
| PhysREVE (pretrained) | 29.9% |
| Baseline (random init) | 39.1% |

**Ablation (all from pretrained encoder):**

| Configuration | Val Accuracy |
|---|---|
| CE only | 33.3% |
| L_phys only | 31.0% |
| L_asym only | 32.2% |
| Full PhysREVE | 34.5% |

### Known Issues at This Version

| ID | Severity | Description | Planned Fix |
|---|---|---|---|
| BUG-01 | Critical | `L_snr = 0.0` every epoch — InfoNCE loop exits with K≤1 patches | Reduce `patch_size` further if needed |
| BUG-02 | High | `L_asym` saturates to −1.0 by epoch 20 | Add `tanh` clamping; see analysis §2.2 |
| BUG-03 | High | Pretraining on same 403 trials used for fine-tuning | Cross-subject pretraining (subjects 2–9) |
| BUG-04 | Medium | Encoder under-adapts due to 33× LR gap | Fixed in v0.2.0 (change #2 above) |

---

## Planned (not yet implemented)

Ordered by expected impact — see [`physreve_analysis.md §3`](physreve_analysis.md) for full code snippets.

| Priority | Change | Expected Impact |
|---|---|---|
| P1 | Fix asymmetry loss — add `tanh` clamping, lower initial weight, enable after epoch 10 | Stops degenerate collapse (BUG-02) |
| P2 | Cross-subject pretraining — subjects 2–9 pretrain, subject 1 fine-tune | +5–15% accuracy (BUG-03) |
| P3 | Linear probe diagnostic phase before full fine-tuning | Validates pretrained features |
| P4 | Anneal `lambda_phys` to 0 over last 30% of fine-tuning | +2–5% stability |
| P5 | Increase `dropout` 0.1→0.2 + `label_smoothing=0.1` | Reduces overfitting |
| P6 | Increase pretrain to 100 epochs with 10-epoch warmup | +2–5% with cross-subject data |
