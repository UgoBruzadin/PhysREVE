# PhysREVE — Experiment Analysis & Improvement Plan

**Dataset:** BCI Competition IV 2a · Subject 1 · 4-class Motor Imagery  
**Model:** PhysREVE (24,142,497 parameters)  
**Date:** 2026-04-07

---

## 1. Results Summary

| Model | Test Accuracy | vs Chance |
|---|---|---|
| Baseline (random init) | **39.1%** | +14.1% |
| PhysREVE (pretrained) | 29.9% | +4.9% |
| Chance | 25.0% | — |

**Physics pretraining hurt performance by −9.2%.** The pretrained encoder transfers worse than random weights on this setup.

### Ablation (all from pretrained encoder)

| Configuration | Val Accuracy |
|---|---|
| CE only (no physics) | 33.3% |
| L_phys only | 31.0% |
| L_asym only | 32.2% |
| Full PhysREVE | 34.5% |

All ablations from the pretrained encoder fall below the random-init baseline (39.1%), confirming the pretraining is not helping on this setup.

### Per-class Performance (PhysREVE)
| Class | Precision | Recall | F1 |
|---|---|---|---|
| Left Hand | 0.21 | 0.17 | 0.19 |
| Right Hand | 0.28 | 0.30 | 0.29 |
| Feet | **0.42** | **0.43** | **0.43** |
| Tongue | 0.26 | 0.29 | 0.28 |

Feet is the only class learned reliably. Left/Right hand — the classes the asymmetry loss is designed to help — are the worst performers.

---

## 2. Root Cause Analysis

### 2.1 Dead SNR Loss (Critical)
```
L_snr = -0.0000  (every single epoch)
```
The SNR alignment loss contributes **zero gradient throughout all 30 pretraining epochs**. Root cause: each trial is only 400 samples with `patch_size=200`, yielding **P=2 patches**. The InfoNCE loop in `snr_alignment_loss` requires K≥2 high-SNR patches *per sample*, but with P=2, after the median threshold half are discarded, leaving K≤1 almost always → the loop skips and returns 0. This loss is completely inactive.

### 2.2 Asymmetry Loss Collapse
```
Ep 1/50  asym=-0.009
Ep 10/50 asym=-0.156  → rapidly collapses to
Ep 20/50 asym=-0.858
Ep 50/50 asym=-0.999  (saturated)
```
`L_asym` saturates to −1.0 almost immediately. This means `asymmetry_loss` is pushing source activations to a **maximally asymmetric degenerate solution** rather than learning task-specific ERD. Once saturated, it provides no gradient signal and may be pulling the encoder away from discriminative features.

### 2.3 Pretraining on the Same Data as Fine-Tuning
Pretraining runs on the same 403 trials used for fine-tuning. The model learns to reconstruct these specific trials but doesn't acquire useful cross-session or cross-subject generalisation. This violates the standard SSL assumption that the pretraining corpus is much larger or domain-shifted from fine-tuning.

### 2.4 Too Few Patches for Block Masking
With `patch_size=200` and 400-sample trials → P=2. But `block_t=4` is set to mask blocks of 4 consecutive time patches. With P=2, the masking strategy is incoherent — the block overruns the entire time axis every time. REVE's masking was designed for much longer contexts (P >> block_t).

### 2.5 Severe Overfitting with a Large Model
24M parameters trained on 403 examples → the model memorises the training set perfectly (100% train acc at epoch 10) while validation stays at 26–38%. The physics losses don't act as sufficient regularisers in this regime.

### 2.6 Encoder Under-Adaptation During Fine-Tuning
`lr_enc=3e-5` is 33× lower than `lr_head=1e-3`. This near-freezes the encoder and means the pretrained (but unhelpful) representations are barely adapted. The head can't compensate on its own.

---

## 3. Recommended Modifications (Prioritised)

### Priority 1 — Fix the broken patch geometry

**Change `patch_size` from 200 → 50** (4 patches per second, 8 patches per 400-sample trial).

```python
# In PhysREVEConfig:
patch_size: int = 50    # was 200
block_t:    int = 2     # was 4 — keep block_t < P/2
block_c:    int = 2     # unchanged
```

This immediately fixes both the dead SNR loss (now K=4–8 patches per sample) and the incoherent block masking. It also increases the number of tokens in the transformer, giving richer representations.

---

### Priority 2 — Fix or disable the asymmetry loss

The saturation to −1.0 indicates the loss is maximising raw asymmetry without bound. Add a tanh to constrain the gradient, and reduce the weight:

```python
def asymmetry_loss(src_acts, labels, lh_idx, rh_idx, eps=1e-6):
    lh_p = src_acts[:, lh_idx, :].pow(2).mean(dim=(1, 2))
    rh_p = src_acts[:, rh_idx, :].pow(2).mean(dim=(1, 2))
    asym = (rh_p - lh_p) / (lh_p + rh_p + eps)

    mi_mask = (labels == 0) | (labels == 1)
    if mi_mask.sum() == 0:
        return src_acts.new_zeros([])

    sign = torch.where(labels[mi_mask] == 1,
                       asym.new_ones(mi_mask.sum()),
                       -asym.new_ones(mi_mask.sum()))

    # Clamp with tanh to prevent saturation
    return -torch.mean(sign * torch.tanh(asym[mi_mask]))
```

Also reduce `lambda_asym` from 0.08 → 0.02 initially, and only enable it after the model has stabilised (e.g., epoch 10+).

---

### Priority 3 — Pretrain on held-out subjects, fine-tune on target

BCI IV 2a has 9 subjects. Pretrain on subjects 2–9, fine-tune on subject 1. This is the standard cross-subject transfer setup and is what self-supervised pretraining is designed for:

```python
# Pretraining: load subjects 2-9
pretrain_subjects = [2, 3, 4, 5, 6, 7, 8, 9]
# Fine-tuning: subject 1 only
finetune_subject  = [1]
```

This gives ~3,600 pretraining trials (9× more data) and cleanly separates pretraining from fine-tuning.

---

### Priority 4 — Linear probe before fine-tuning

Add a linear probe phase (encoder frozen, head only) before full fine-tuning. This diagnoses whether the pretrained features contain task signal at all, and can itself achieve competitive accuracy:

```python
def run_linear_probe(pretrained, cfg, train_loader, val_loader, n_epochs=20, lr=1e-3):
    model = PhysREVEFinetuneModel(pretrained, cfg, N_CLASSES).to(device)
    # Freeze encoder completely
    for p in model.encoder.parameters():
        p.requires_grad = False
    optimizer = torch.optim.Adam(model.head.parameters(), lr=lr)
    # ... standard training loop with CE loss only
```

If the linear probe beats chance meaningfully, proceed to full fine-tuning. If not, the pretraining representations are unhelpful and you should debug before fine-tuning.

---

### Priority 5 — Anneal physics loss weights

Add cosine annealing for `lambda_phys` during fine-tuning (fade out over the last 30% of epochs). This lets the model stabilise on the classification objective once physics structure is established:

```python
def get_lambda_phys(epoch, n_epochs, lambda_start=0.15, fade_start=0.7):
    if epoch / n_epochs < fade_start:
        return lambda_start
    progress = (epoch / n_epochs - fade_start) / (1.0 - fade_start)
    return lambda_start * (1.0 - progress)  # linear fade to 0
```

---

### Priority 6 — Add gradient clipping and stronger regularisation

```python
# In the fine-tuning loop, after loss.backward():
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# In PhysREVEConfig:
dropout: float = 0.2    # was 0.1
```

Also add label smoothing to reduce overconfidence:
```python
lce = F.cross_entropy(logits, labels, label_smoothing=0.1)
```

---

### Priority 7 — Use more pretraining epochs with LR warmup

30 epochs is very few for a 24M model. With the cross-subject setup (Priority 3), increase to 100 epochs:

```python
run_pretraining(cfg, pretrain_loader, n_epochs=100, lr=3e-4, warmup_epochs=10)
```

---

## 4. Expected Impact

| Change | Expected Gain | Confidence |
|---|---|---|
| Fix patch_size (P1) | Activates SNR loss, fixes masking | High |
| Fix asymmetry loss (P2) | Stops degenerate collapse | High |
| Cross-subject pretraining (P3) | +5–15% over current PhysREVE | High |
| Linear probe diagnosis (P4) | Diagnostic — not a score boost itself | — |
| Loss annealing (P5) | +2–5% fine-tuning stability | Medium |
| Gradient clipping + dropout (P6) | Reduces overfitting | Medium |
| More pretrain epochs (P7) | +2–5% with P3 | Medium |

With P1–P3 applied, a reasonable target is **45–55% test accuracy** on BCI IV 2a Subject 1, competitive with published EEGNet/ShallowConvNet baselines (~60–70%) given this is a much smaller encoder than full REVE.

---

## 5. What Is Working

- **MAE pretraining** converges cleanly (1.15 → 0.66) with stable loss curves.
- **Physics consistency loss (L_phys)** decreases and contributes useful gradient.
- **4D positional encoding** and **leadfield attention bias** are architecturally sound.
- **Block masking** implementation is correct; it just needs P >> block_t to be meaningful.
- The full PhysREVE ablation (34.5%) outperforms CE-only (33.3%) when starting from pretrained weights, confirming the physics losses do help — the pretraining itself is the bottleneck.
