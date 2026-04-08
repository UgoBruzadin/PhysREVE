# PhysREVE — Physics-Informed EEG Foundation Models

**PhysREVE** investigates whether injecting biophysical constraints derived from the EEG forward model improves cross-subject generalisation of pretrained EEG foundation models — without retraining the backbone.

The central tool is the **leadfield matrix** `L ∈ ℝ^(C×N)`, which maps N cortical dipole sources to C scalp electrodes via Maxwell's equations. From `L` we derive a physics-grounded electrode similarity matrix `B_sim = L_row · L_row^T` that encodes which electrodes share neural generators — a relationship determined by skull geometry, not by any individual subject's EEG statistics.

---

## Core finding (EXP_003)

**+14.8% cross-subject improvement** from injecting `B_sim` as a channel-mixing residual into a frozen pretrained REVE backbone.

```
F_out = F  +  α · (B_sim ⊗ F)

where  F      = REVE output features (B, C, P, d) — frozen, no gradient
       B_sim  = leadfield electrode similarity (C, C) — fixed physics
       α      = one learnable scalar, initialised at 0
```

REVE's 30M parameters never change. The only new parameter is `α`.

| Condition | Physics | Mean Bal. Acc. (9 subjects LOSO) | vs Baseline |
|---|---|:---:|:---:|
| A — Frozen REVE baseline | — | 0.338 ± 0.045 | — |
| B — + Leadfield B_sim | α · B_sim | **0.388 ± 0.079** | **+14.8%** |
| C — + B_sim + jitter | α · B_sim + aug | 0.375 ± 0.054 | +10.9% |

*BCI Competition IV 2a, 4-class motor imagery, leave-one-subject-out, 9 subjects. Chance = 0.25.*

B beats A on **7/9 subjects**. C beats A on **8/9 subjects**.

---

## Why it works

EEG statistics vary substantially across subjects (skull thickness, head geometry, electrode placement). The forward physics does not — `B_sim[C3, C1]` is high regardless of who is in the chair, because C3 and C1 always sit over overlapping motor cortex sources.

The channel mixing pre-amplifies within-cluster signal before pooling:

```
F_out[C3] = F[C3]  +  α × ( 0.90·F[C1]  +  0.72·F[Cz]  +  (−0.18)·F[C4]  + ... )
                              ─── share left motor sources ───   ← right hemisphere, suppressed
```

When the adapter then pools across channels, the left/right motor contrast survives — not because we taught the model any neuroscience, but because physics told us which electrodes were at the same location.

---

## The EEG forward model

```
v = L · a

  v ∈ ℝ^(C×T)    electrode voltages  (22 channels × time)
  L ∈ ℝ^(C×N)    leadfield matrix    (maps N≈1500 dipoles → C electrodes)
  a ∈ ℝ^(N×T)    source activations  (the neural signal of interest)
```

`L[c, s]` = voltage at electrode `c` when source `s` fires with unit amplitude, everything else silent. Computed analytically from a sphere head model — no individual MRI required.

**Row-normalising** L gives each electrode's source-sensitivity as a unit vector. The dot product of two such vectors is their cosine similarity through source space:

```
B_sim = L_row @ L_row.T     (C × C),  values ∈ [−1, 1]

B_sim[i,j] high   →  electrodes i and j see overlapping sources
B_sim[i,j] near 0 →  electrodes i and j see mostly different sources
```

**Validated on 109-subject PhysioNet EEGBCI dataset (physreve_hypotheses.ipynb):**
- Physics prior predicts empirical EEG cross-channel correlations better than scalp-geometry RBF in **all 5 frequency bands**
- Largest divergence in alpha/beta — the motor-relevant bands

---

## Repository structure

```
PhysREVE/
├── physreve/                     Core library
│   ├── model.py                  PhysREVEEncoder, PhysREVEPretrainModel, PhysREVEFineTuneModel
│   ├── physics.py                build_leadfield(), LeadfieldAttentionBias, B_sim computation
│   ├── losses.py                 mae_loss, physics_loss, snr_alignment_loss, asymmetry_loss
│   ├── train.py                  run_pretraining, run_finetuning
│   ├── data.py                   Dataset utilities
│   ├── masking.py                Block masking for MAE pretraining
│   ├── evaluate.py               Evaluation utilities
│   ├── baselines.py              LDA, LogReg, XGBoost baselines
│   └── datasets/                 BCI IV 2a, CHB-MIT loaders
│
├── experiments/
│   ├── reve_spectral_adversarial.ipynb   EXP_003: leadfield injection into frozen REVE (main result)
│   ├── physreve_paper_outline.ipynb      Architecture diagrams and figure drafts
│   ├── baselines_notebook.ipynb          Classical ML baselines
│   ├── leadfield_visualization.ipynb     Forward model visualization
│   ├── presentation.ipynb               Slide deck (RISE-compatible)
│   ├── EXP_001/results.json             v0.1 single-subject baseline
│   ├── EXP_002/results.json             v0.2 improved fine-tuning schedule
│   └── EXP_003/results.json             v0.3 leadfield injection LOSO
│
├── physreve_hypotheses.ipynb     Hypothesis testing on 109-subject PhysioNet dataset
├── quick_wins_notebook.ipynb     Single-subject PhysREVE vs baselines
├── quick_wins_seizure_notebook.ipynb  Seizure detection (CHB-MIT)
├── RESULTS_ANALYSIS.md           Detailed result breakdowns
└── CHANGELOG.md                  Version history
```

---

## What was tried and why it failed (EXP_001/002)

Before the injection approach, we trained PhysREVE from scratch with a full physics architecture (4D positional encoding, leadfield attention bias inside the Transformer, source decoder, physics + SNR losses).

**Cross-subject LOSO result: 0.250 — chance level.**

Root causes:
1. **Data starvation.** ~4,600 training examples for 9.5M parameters (ratio ~500 params/example).
2. **No pretraining.** REVE's representational quality comes from pretraining on massive corpora. Random-init PhysREVE starts from nothing.
3. **Confounded signal.** Can't distinguish physics hurting vs. lack of domain knowledge.

**Lesson:** Physics as a training constraint cannot substitute for pretraining on large data. Physics as an injection into a pretrained frozen backbone is the right framing — backbone quality is fixed, the only variable is whether physics structure helps.

Single-subject results (quick_wins_notebook.ipynb) showed PhysREVE at 40.7% val accuracy — equal to random init, below LDA at 58.1% — confirming the data bottleneck.

---

## Hypothesis testing results (physreve_hypotheses.ipynb)

109-subject PhysioNet EEGBCI dataset, 64 channels, 160 Hz.

| Hypothesis | Finding |
|---|---|
| **H1** Physics prior better predicts empirical EEG correlations than scalp RBF | ✅ Physics wins 5/5 frequency bands. Largest gap in alpha/beta. |
| **H2** Electrode jitter σ=5mm is safe augmentation | ✅ B_sim correlation r > 0.95 up to 5mm. Matches real cap placement error. |
| **H3** SVD jitter σ_rel=10% covers conductivity uncertainty | ✅ Covers true BEM conductivity variation range without collapsing B_sim. |
| **H4** Per-subject skull conductivity estimable from EEG | ⚠️ Ground-truth fitting is degenerate (collapses to 2 discrete values). Open problem. |
| **H5** Physics-weighted features improve within-subject classification | ❌ Raw band-power (0.615) beats leadfield-weighted (0.579). Physics doesn't help within-subject — confirms the benefit is cross-subject only. |

---

## Outlook

### Immediate
- Log `α` per fold — verify that `α` magnitude predicts performance gain
- Per-channel `α` (22×22 matrix) — different electrode pairs, different weights
- Combine spectral features + leadfield mixing — two orthogonal physics signals

### Near-term
- **True attention bias** — inject `B_sim` inside REVE's QK dot products pre-softmax (architecturally cleaner; requires patching REVE internals)
- **Physics auxiliary loss** — source decoder head on REVE features, trained with `L_phys = ||L·ŝ − y||²`; no backbone modification
- **Individual BEM** — subject-specific `B_sim` from digitised electrode positions; should close the Subject 6 anatomy gap

### Scale
- **LaBraM backbone** — larger pretrained model; does the physics gain persist?
- **Seizure detection (CHB-MIT)** — same injection, different task, cross-patient generalisation
- **Individual leadfields from EEG** — estimate per-subject `B_sim` deviation from alpha spatial spread (H4 open problem)

---

## Running the main experiment

```python
# In experiments/reve_spectral_adversarial.ipynb
# Requires: HuggingFace login for brain-bzh/reve-base

# Three conditions, LOSO across 9 subjects:
CONDITIONS = {
    'A_baseline':      # Frozen REVE → MLP
    'B_leadfield':     # Frozen REVE + α·B_sim channel mixing
    'C_leadfield_aug': # B + electrode jitter 5mm + SVD noise 10%
}
```

```bash
pip install -r requirements.txt
# Then open experiments/reve_spectral_adversarial.ipynb in Jupyter
```

---

## References

- **REVE**: *Scalable EEG Foundation Model*, brain-bzh/reve-base (HuggingFace)
- **BCI Competition IV 2a**: Brunner et al. — 4-class motor imagery, 9 subjects, 22 ch @ 250 Hz
- **PhysioNet EEGBCI**: Goldberger et al. — 109 subjects, 64 ch, 160 Hz, imagined movement
- **MNE-Python**: Forward model computation — sphere model, standard_1020 montage
- **EEGPT**: SNR alignment loss design
