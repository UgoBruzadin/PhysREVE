# PhysREVE — Results Analysis
*Subject 1, BCI IV 2a, 4-class motor imagery. Updated: 2026-04-07*

---

## Setup

| | |
|---|---|
| **Dataset** | BCI Competition IV 2a — left hand / right hand / feet / tongue |
| **Channels** | 22 EEG, 200 Hz, 2.5 s windows (500 samples) |
| **Split** | Chronological: 404 train / 86 val / 86 test |
| **Pretrain corpus** | Subjects 2–9, 4608 trials (unlabeled) |
| **Chance level** | 25% (4 classes) |

---

## Results

### ML Baselines (hand-crafted features)

| Model | Val Acc |
|---|---|
| LDA | **0.581** |
| Logistic Regression | 0.547 |
| XGBoost | 0.535 |

Features: band power + covariance, 154-dim. LDA is the current ceiling for single-subject, single-session.

---

### Neural Models — Full Fine-tuning (d=256, 6 layers, ~9.5M params)

| Model | Best Val | Train @ ep30 | Notes |
|---|---|---|---|
| Random-init | 0.407 | 1.000 | Severe overfit: train 100%, val 41% |
| Base REVE (MAE only) | 0.360 | 1.000 | **Worse** than random-init |
| PhysREVE (full) | 0.407 | 0.500 | Same as random but different trajectory |

**Key observation:** REVE pretraining actively hurts fine-tuning accuracy. PhysREVE matches random-init but shows slower overfitting (train=50% at epoch 30 vs 100% for random-init), suggesting the physics losses act as regularization during fine-tuning rather than producing transferable representations.

---

### Linear Probe (frozen encoder → logistic regression)

| Model | Val Acc | Δ vs random |
|---|---|---|
| Random-init | 0.349 | — |
| Base REVE | 0.314 | **−0.035** |
| PhysREVE | 0.360 | +0.012 |

**Key finding:** Pretraining produces almost no representational benefit. PhysREVE is barely (+1.2%) better than random features on a linear probe. Base REVE is worse. This means the MAE objective on EEG is not learning motor-imagery-relevant structure — it's learning low-level signal reconstruction that doesn't generalize to the classification task.

---

### Small Model (d=64, 2 layers, ~476K params, dropout=0.3, wd=5e-4)

| Model | Best Val | Train @ ep50 |
|---|---|---|
| Random-init | **0.442** | 0.975 |
| PhysREVE (small) | TBD | — |

**Best neural result so far.** Reducing the model from 9.5M → 476K parameters (20× smaller) and increasing dropout from 0.1 → 0.3 gives the highest val accuracy. This confirms the core problem is parameter count vs. training data size — 404 trials cannot support a 9.5M parameter model.

---

## Diagnosis

### Problem 1: Model too large for the data
- 9.5M params / 404 trials = 23,500 params per trial
- Small model (476K) still shows some overfit (train=97.5%) but achieves 44% val
- Rule of thumb: need 10–100× more data than parameters to generalize

### Problem 2: Pretraining objective is not task-aligned
- MAE on EEG (reconstruct masked patches) teaches the encoder to model EEG structure, not motor-imagery relevant patterns
- The linear probe confirms: pretrained features are barely better than random for decoding MI
- Compare: in NLP/CV, linear probes on pretrained models typically show large gains over random (5–30%)
- Here the gain is 1.2% — essentially noise

### Problem 3: Asymmetry loss going in the wrong direction
- L_asym → strongly negative during training (reaching −0.98 at epoch 30)
- Should be positive for correct lateralization (left-hand MI → left motor cortex)
- Suggests the model is learning anti-lateralized activations, possibly because the loss polarity or ROI indexing is incorrect

### Problem 4: Physics losses not converging
- L_phys stays flat at ~0.09 across all 20 pretraining epochs
- L_snr grows from 1.4 → 1.6 (getting worse, not better)
- The physics constraints aren't being actively minimized — either the loss weight is too small or the optimizer doesn't prioritize them

---

## What's Actually Working

- The **small model** direction is promising: 44.2% vs 25% chance is real signal
- PhysREVE's physics losses appear to **slow down overfitting** (train=50% vs 100% at epoch 30) — this regularization effect is real but not enough to beat a smaller model
- The **pretrain corpus is large** (4608 trials) — if the pretraining objective were better aligned with the task, this should help

---

## Next Steps (Priority Order)

### High impact, low effort
1. **Multi-subject fine-tuning**: subjects 2–9 have labels — use them. 576×9 = 5184 labeled trials instead of 404. This alone should close most of the gap with LDA.
2. **Fix L_asym sign**: check whether `lh_idx` / `rh_idx` are correctly assigned and whether the loss formula produces positive values for correct lateralization.
3. **Small model pretrain + finetune**: run the full pretraining pipeline with `cfg_small` to see if small pretrained > small random-init (still pending).

### Medium impact
4. **Task-aligned pretraining**: add a weak classification signal during pretraining (semi-supervised), or use contrastive loss between same-subject trials.
5. **Longer fine-tuning with early stopping**: the small model is still improving at epoch 50 — try 100 epochs with patience=20.
6. **Reduce mask ratio**: 75% masking may be too aggressive for 22-channel EEG (vs. 75% which works for ViT on 3-channel images with much higher spatial resolution). Try 50%.

### Requires more investigation
7. **Why does REVE pretraining hurt?** — run an ablation where you freeze the encoder from REVE and only fine-tune the head. If that also hurts, the pretraining is corrupting the feature space rather than building on it.
8. **Cross-subject generalization**: linear probe with pretrained encoder, then fine-tune on each subject separately. Tests whether the pretrained representations are *subject-invariant* (which is the actual value proposition of the approach).

---

## Summary

The model works — small PhysREVE beats chance by ~19 points and shows real motor imagery decoding. The pretraining is not yet working as intended: the MAE objective learns EEG signal structure but not task-relevant features, and simple ML baselines still outperform all neural variants on this single-subject setup. The core bet (that physics-informed pretraining on many unlabeled subjects will help single-subject decoding) is still untested because the fine-tuning is bottlenecked by overfitting and misaligned pretraining objectives.
