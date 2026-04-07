# PhysREVE — Change Log

All modifications to the notebook and supporting files are documented here in reverse-chronological order. Each entry records *what* changed, *where*, and *why*.

---

## [0.4.0] — 2026-04-07  *(EXP_004)*

### P2 + P5: Cross-subject pretraining and overfitting reduction

**Motivation:** Even after P1, the model still overfits — train hits 100% while val peaks ~44%. Two root causes remain:
1. **BUG-03**: Pretraining used subject 1 data, the same 403 trials used for fine-tuning. The encoder already "saw" these trials unlabeled, so fine-tuning was nearly just repeating seen data.
2. **Overfitting**: 24M parameters on ~280 labeled training trials is severely under-constrained. Dropout at 0.1 is too low; the model memorises.

---

#### 1. Cross-subject pretraining — subjects 2–9 pretrain, subject 1 fine-tune
**Location:** Data loading cell (Cell 22)

```python
# Before: single-subject load, pretrain and fine-tune on same pool
X_all, y_str_all, _ = paradigm.get_data(dataset, subjects=[1], ...)
pretrain_ds = UnlabeledEEGDataset(X_all[idx_tr])   # ← same 282 trials
finetune_ds = LabeledEEGDataset(X_all[idx_tr], ...) # ← same 282 trials

# After: held-out subjects for pretraining
X_pretrain_parts = []
for subj in range(2, 10):
    X_s, _, _ = paradigm.get_data(dataset, subjects=[subj], ...)
    X_pretrain_parts.append(X_s)
X_pretrain = np.concatenate(X_pretrain_parts, axis=0)  # ~3224 trials, 8 subjects

X_s1, y_str_s1, _ = paradigm.get_data(dataset, subjects=[1], ...)
pretrain_ds = UnlabeledEEGDataset(X_pretrain)   # subjects 2-9, never fine-tuned
finetune_ds = LabeledEEGDataset(X_s1[idx_tr], ...) # subject 1 only
```

**Why:** The encoder now learns physics-consistent EEG representations from 8 subjects it will never be fine-tuned on. When subject 1 fine-tuning begins, the encoder is a genuinely general EEG prior rather than a subject-1-specific one. Expected gain: +5–15% accuracy. This also means the pretrain pool is ~8× larger (~3224 vs ~282 trials), giving the physics losses much more signal.

---

#### 2. Dropout increased: 0.1 → 0.2
**Location:** `PhysREVEConfig` (Cell 5)

```python
# Before
dropout      : float = 0.1

# After
dropout      : float = 0.2      # increased: reduce overfitting (24M params, ~280 train trials)
```

**Why:** At 24M parameters trained on 280 examples, the model has ~85k parameters per training sample. Dropout at 0.1 drops only 10% of activations — too weak a regulariser. 0.2 is the standard starting point for this parameter-to-data ratio in transformer fine-tuning.

---

#### 3. Label smoothing: 0.0 → 0.1
**Location:** `finetune_losses()` cross-entropy call (Cell 18)

```python
# Before
lce = F.cross_entropy(logits, labels)

# After
lce = F.cross_entropy(logits, labels, label_smoothing=0.1)
```

**Why:** Without label smoothing the model is trained to produce logit differences → ∞ for the correct class. On this dataset (4 classes, 25% chance) the soft targets become `[0.033, 0.033, 0.033, 0.9]` instead of `[0, 0, 0, 1]`. This directly penalises overconfident predictions and is one of the most reliable small-model overfitting remedies.

---

### Expected combined effect of v0.4.0

| Change | Primary effect |
|---|---|
| Cross-subject pretraining | Better pretrained features; breaks data leakage |
| Dropout 0.2 | Smaller train–val accuracy gap |
| Label smoothing 0.1 | Less overconfident logits; better calibration |

Going into this run: train hits 100%, val ~44% (EXP_003 projection). Target: val 48–55%, reduced train–val gap.

---

## [0.3.0] — 2026-04-07

### P1 Fix: Asymmetry Loss Collapse (3 changes to `physreve_notebook.ipynb`)

**Motivation:** In v0.2.0 the asymmetry loss was still saturating to −1.0 by epoch 20 (BUG-02). The raw normalised asymmetry `(rh−lh)/(rh+lh)` is bounded in (−1, 1) and hard-saturates there — once the model pushes source activations to an extreme ratio, gradients effectively vanish and the loss provides no further learning signal. This was confirmed by the per-class results: Left Hand (F1=0.13) was drastically worse than Right Hand (F1=0.42), consistent with the loss collapsing in the direction that favours right-hand trials.

**Baseline going into this version (v0.2.0 results):**

| Model | Test Accuracy |
|---|---|
| Baseline (random init) | 28.7% |
| PhysREVE (physics pretrained) | 37.9% (+9.2pp over baseline) |

---

#### 1. `asymmetry_loss` — `tanh` clamping on normalised asymmetry
**Location:** `asymmetry_loss()` in Cell 17

```python
# Before
asym = (rh_p - lh_p) / (lh_p + rh_p + eps)              # (B,)

# After
raw  = (rh_p - lh_p) / (lh_p + rh_p + eps)              # (B,) in (-1,1)
asym = torch.tanh(3.0 * raw)                             # still in (-1,1), smoother
```

**Why:** `tanh(3x)` maps (−1,1) → (−1,1) but with a smooth, non-zero gradient everywhere. When `raw` approaches ±1 (saturation), `tanh` pulls the gradient back from ≈0 to a small but non-negligible value. The scale factor 3 means the output reaches ~0.995 at `raw=±1` — preserving full expressiveness — while the gradient at that point is `3 × sech²(3) ≈ 0.13` instead of 0. This is the minimal change that breaks the collapse without altering the loss semantics.

---

#### 2. `finetune_losses` — delayed asymmetry activation (epoch ≥ 10)
**Location:** `finetune_losses()` in Cell 17

```python
# Before (signature)
def finetune_losses(logits, src_acts, sen_p, labels, L, cfg, lh_idx, rh_idx):
    ...
    total = lce + cfg.lambda_phys * lphys + cfg.lambda_asym * lasym

# After (signature + body)
def finetune_losses(logits, src_acts, sen_p, labels, L, cfg,
                    lh_idx, rh_idx, epoch: int = 1):
    ...
    # Delayed asymmetry: ramp in after epoch 10
    asym_w = cfg.lambda_asym if epoch >= 10 else 0.0
    total = lce + cfg.lambda_phys * lphys + asym_w * lasym
```

**Why:** In the first ~10 epochs the classification head is orienting to the task; the CE loss is high and gradients are noisy. If `L_asym` fires immediately it competes with a head that hasn't learned to use source activations yet, causing the collapse to happen before the model has any useful representations. Delaying to epoch 10 (after the encoder has unfrozen at epoch 6 and had 4 epochs of joint training) ensures the asymmetry loss lands on an already-meaningful representation.

---

#### 3. `run_finetuning` — pass `epoch` to `finetune_losses`
**Location:** training loop in Cell 24

```python
# Before
loss, parts = finetune_losses(
    logits, src_acts, sen_p, yb, L_col, cfg, LH_IDX, RH_IDX
)

# After
loss, parts = finetune_losses(
    logits, src_acts, sen_p, yb, L_col, cfg, LH_IDX, RH_IDX,
    epoch=epoch
)
```

**Why:** Mechanical plumbing for the delayed activation above.

---

#### 4. `lambda_asym` reduced: 0.15 → 0.05
**Location:** `PhysREVEConfig` in Cell 5

```python
# Before
lambda_asym  : float = 0.15     # hemispheric ERD asymmetry (raised: stronger neuroscience prior)

# After
lambda_asym  : float = 0.05     # hemispheric ERD asymmetry (reduced: tanh + delay handle stability)
```

**Why:** In v0.2.0 we raised it to 0.15 to compensate for the saturating loss providing weak signal. With tanh clamping that signal is now healthy — a strong weight would over-constrain the encoder. 0.05 keeps it as a regulariser rather than a dominant objective.

---

### Expected Effect of P1

The left-hand class should recover the most (was F1=0.13 because `L_asym` was collapsing in the direction that "solved" right-hand at left-hand's expense). Right-hand should remain stable or improve slightly. Overall accuracy target: 40–45%.

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
| ~~P1~~ | ~~Fix asymmetry loss — add `tanh` clamping, lower initial weight, enable after epoch 10~~ | **Done in v0.3.0** |
| ~~P2~~ | ~~Cross-subject pretraining — subjects 2–9 pretrain, subject 1 fine-tune~~ | **Done in v0.4.0** |
| P3 | Linear probe diagnostic phase before full fine-tuning | Validates pretrained features |
| P4 | Anneal `lambda_phys` to 0 over last 30% of fine-tuning | +2–5% stability |
| ~~P5~~ | ~~Increase `dropout` 0.1→0.2 + `label_smoothing=0.1`~~ | **Done in v0.4.0** |
| P6 | Increase pretrain to 100 epochs with 10-epoch warmup | +2–5% with cross-subject data |
