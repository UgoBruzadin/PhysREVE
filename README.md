# PhysREVE — Physics-Informed EEG Foundation Model

**PhysREVE** extends the [REVE](https://arxiv.org/abs/2502.xxxxx) EEG foundation model with biophysical constraints derived from the electromagnetic forward model of the brain. The central hypothesis is that grounding self-supervised EEG pretraining in known neuroscience — source localisation physics and hemispheric motor dynamics — produces representations that generalise better to downstream BCI tasks than sensor-space learning alone.

---

## Hypothesis

> **Every EEG foundation model trained purely on sensor signals ignores the biophysical equation that generated those signals. Encoding this equation as an inductive bias during pretraining should yield representations that are more neurally faithful, more data-efficient, and more transferable to motor imagery classification.**

The biophysical forward model of EEG is:

```
y  =  L · s  +  ε

where:
  y  ∈ ℝ^(C×T)    sensor-space EEG        (22 channels × time)
  L  ∈ ℝ^(C×N)    leadfield matrix         (maps N=1,284 brain sources → C electrodes)
  s  ∈ ℝ^(N×T)    cortical source currents  (the neural signal of interest)
  ε               sensor noise & artifacts
```

Standard EEG models learn `f(y)` directly. PhysREVE learns `f(y)` while simultaneously enforcing `L · ŝ ≈ y`, anchoring the learned representations to the brain's actual geometry.

---

## Architecture

```
                         ┌──────────────────────────────────────────────┐
                         │              PhysREVE Model                  │
                         │                                              │
  EEG Input              │   ┌──────────────────┐                      │
  y ∈ ℝ^(C×T)  ─────────┼──▶│  Patch Tokeniser │  patch_size=50 smpl  │
                         │   └────────┬─────────┘                      │
                         │            │                                  │
                         │   ┌────────▼──────────────┐                 │
                         │   │  4D Positional Encoder │                 │
                         │   │  (x, y, z, t) coords  │                 │
                         │   └────────┬───────────────┘                 │
                         │            │                                  │
  Leadfield L            │   ┌────────▼───────────────────────┐        │
  (MNE forward) ─────────┼──▶│  Spatio-Temporal Transformer   │        │
                         │   │  + Leadfield Attention Bias    │        │
                         │   │  12 layers · 8 heads · d=256   │        │
                         │   └────────┬───────────────────────┘        │
                         │            │                                  │
                         │      ┌─────┴──────┐                         │
                         │      ▼            ▼                          │
                         │  ┌───────┐  ┌──────────────┐               │
                         │  │  MAE  │  │ Source Decoder│               │
                         │  │Decoder│  │  ŝ ∈ ℝ^(N×T) │               │
                         │  └───┬───┘  └──────┬───────┘               │
                         │      │             │                         │
                         │   L_mae         L·ŝ ≈ y  →  L_phys         │
                         │                   │                          │
                         │              InfoNCE(SNR) → L_snr            │
                         └──────────────────────────────────────────────┘
```

### Physics Components

| Component | Where | What it enforces |
|---|---|---|
| **4D Positional Encoding** (x,y,z,t) | Patch embedding | Geometric electrode identity |
| **Leadfield Attention Bias** | Spatial attention heads | Which channels share brain sources |
| **Source Decoder + L_phys** | Pretraining loss | `L·ŝ ≈ y` — reconstructed sources must project back to observed sensors |
| **SNR Alignment L_snr** | Pretraining loss | High-SNR patches are similar in source space (from EEGPT) |
| **Hemispheric Asymmetry L_asym** | Fine-tuning loss | Contralateral ERD during motor imagery |

### Why the Leadfield Bias Works

The figure below shows the full chain: cortical sources → skull → scalp electrodes (Panel 1), the raw leadfield matrix L (Panel 2), and the resulting attention bias B = L_row · L_row^T that PhysREVE adds to attention logits (Panel 3). REVE has no access to any of this structure.

![PhysREVE Leadfield Visualization](physreve_leadfield_vis.png)

> Generate this figure by running `experiments/leadfield_visualization.ipynb`.

---

## Training Pipeline

### Phase 1 — Self-Supervised Pretraining (unlabeled EEG)

```
  Unlabeled EEG trials
         │
         ▼
  ┌─────────────────────┐
  │   Block Masking     │   mask 75% of patches in contiguous blocks
  │   (spatial + time)  │
  └──────────┬──────────┘
             │
             ▼
  ┌──────────────────────────────────────────────────────┐
  │              Pretraining Objective                   │
  │                                                      │
  │   L_total = L_mae                                    │
  │           + 0.15 · L_phys   (source consistency)    │
  │           + 0.05 · L_snr    (SNR-patch alignment)   │
  └──────────────────────────────────────────────────────┘
```

**L_phys** penalises the MSE between the sensor-space EEG `y` and the forward projection of the decoded sources `L·ŝ`. This forces the model to learn representations that are consistent with brain geometry.

**L_snr** is an InfoNCE contrastive loss: patches with high signal-to-noise ratio (estimated by spectral power in the 8–30 Hz band) should be similar in source space and dissimilar to low-SNR patches.

---

### Phase 2 — Fine-tuning on Labeled BCI Data

```
  Pretrained Encoder  ─────────────────────────────────┐
                                                        │
  Labeled EEG trials                                    ▼
  (motor imagery)  ─────▶  ┌─────────────────────────────────────────┐
                            │           Fine-tuning Objective         │
                            │                                         │
                            │  L_total = L_CE                         │
                            │          + 0.15 · L_phys               │
                            │          + 0.15 · L_asym               │
                            └─────────────────────────────────────────┘
                                              │
                                              ▼
                                     4-class prediction
                                  (Left/Right Hand, Feet, Tongue)
```

#### Encoder Warmup Schedule

```
  Epochs 1–5   │  Encoder FROZEN  │  Head learns task structure
  ─────────────┼──────────────────┼─────────────────────────────
  Epochs 6–50  │  Encoder ACTIVE  │  Physics losses reshape representations
               │  lr_enc = 1e-4   │
               │  lr_head = 1e-3  │  (10× differential)
```

This prevents the higher encoder LR from corrupting pretrained weights before the classification head has stable gradients.

---

## The Hemispheric ERD Asymmetry Loss

Motor imagery produces **Event-Related Desynchronisation (ERD)**: a suppression of 8–30 Hz power in the contralateral motor cortex. This is the clearest, most consistent neurophysiological signature of motor imagination.

```
  Left Hand Imagery:                Right Hand Imagery:

  Left hemisphere   Right hemisphere   Left hemisphere   Right hemisphere
  ┌──────────┐      ┌──────────┐       ┌──────────┐      ┌──────────┐
  │          │      │   ERD    │       │   ERD    │      │          │
  │  normal  │      │ (↓ power)│       │ (↓ power)│      │  normal  │
  │          │      │    ██    │       │    ██    │      │          │
  └──────────┘      └──────────┘       └──────────┘      └──────────┘
         motor cortex regions (C3/C4 electrodes)
```

**L_asym** pushes source activations in the right hemisphere to be weaker than the left during left-hand imagery (and vice versa), explicitly teaching the model the well-established neuroscience of motor control. This is the physics the model cannot learn from sensor signals alone.

---

## Results

### Motor Imagery — BCI IV 2a, Subject 1, 4-class (chronological split)

| Model | Val Acc | Notes |
|---|---|---|
| Chance | 25.0% | — |
| LDA (band power) | **58.1%** | Best overall |
| Logistic Regression | 54.7% | |
| XGBoost | 53.5% | |
| Random-init (d=256, 6L) | 40.7% | Train=100% — severe overfit |
| Base REVE (MAE pretrained) | 36.0% | Worse than random-init |
| PhysREVE (d=256, 6L) | 40.7% | Same as random; physics slows overfit |
| **Small random-init (d=64, 2L)** | **44.2%** | Best neural result |

**Linear probe (frozen encoder → logistic regression):**

| Encoder | Val Acc | Δ vs random |
|---|---|---|
| Random-init | 34.9% | — |
| Base REVE | 31.4% | −3.5% |
| PhysREVE | 36.0% | +1.2% |

Setup: 404 train / 86 val trials, subjects 2–9 as unlabeled pretraining corpus (4,608 trials).  
Full analysis: [RESULTS_ANALYSIS.md](RESULTS_ANALYSIS.md) · Notebooks: [quick_wins_notebook.ipynb](quick_wins_notebook.ipynb)

### Seizure Detection — CHB-MIT Patient 01 (binary, AUC-primary)

| Model | Acc | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| LDA | — | — | — | — |
| LogReg | — | — | — | — |
| Random-init (d=128, 6L) | — | — | — | — |
| Base REVE | — | — | — | — |
| PhysREVE (d=128, 6L) | — | — | — | — |
| Small random-init (d=64, 2L) | — | — | — | — |

*Results pending — run [quick_wins_seizure_notebook.ipynb](quick_wins_seizure_notebook.ipynb) on Colab.*  
Note: accuracy is misleading for imbalanced seizure data — AUC and sensitivity are the primary metrics.

---

## Conclusions

### What works

- **Physics as regularisation:** PhysREVE's physics losses (L_phys, L_snr) slow down overfitting during fine-tuning — train accuracy at epoch 30 is 50% for PhysREVE vs 100% for random-init with identical val accuracy. The constraint is active even when the pretraining representations aren't better.
- **Small models for small data:** Reducing the architecture from 9.5M → 476K parameters (d=256→64, 6→2 layers) and increasing dropout from 0.1→0.3 gives the best neural result (44.2% vs 40.7%). Architecture size matters more than pretraining strategy when labeled data is scarce.
- **ML baselines are a real ceiling:** LDA at 58% on band-power features beats every neural model for single-subject MI. This is expected and well-documented in the EEG literature for small-N settings. The baseline is not wrong — the neural models need more data.

### What doesn't work yet

- **MAE pretraining objective is misaligned:** The linear probe shows Base REVE representations are 3.5% *worse* than random for MI decoding. MAE learns class-neutral EEG autocorrelation structure (spectral, spatial) that has to be partially unlearned during fine-tuning. PhysREVE is only +1.2% better than random on a frozen probe.
- **L_phys and L_snr are not converging:** Both losses plateau or worsen within 5 pretraining epochs. L_phys stays flat (~0.09), L_snr increases (1.4→1.6). The physics constraints are not being meaningfully optimised.
- **L_asym inverts:** The asymmetry loss reaches −0.98 during fine-tuning — the model learns anti-lateralised activations. This is likely a sign error or index assignment bug in `lh_idx`/`rh_idx`.

### Priority fixes

| Priority | Fix | Expected impact |
|---|---|---|
| 1 | **Multi-subject labeled fine-tuning** (subjects 2–9 have labels → 5,184 trials) | Removes the data bottleneck entirely |
| 2 | **Fix L_asym sign** — check `lh_idx`/`rh_idx` and loss polarity | Enables the hemispheric ERD signal to train correctly |
| 3 | **Reduce mask ratio** (0.75→0.50) — too aggressive for 22-channel EEG | Lets L_mae converge further; may activate L_snr |
| 4 | **Increase L_phys weight** (0.15→0.5) | Forces physics constraint to compete with L_mae |
| 5 | **Task-aligned pretraining** — contrastive loss or pseudo-labels from LDA | Directly aligns pretraining with the classification objective |

---

## Roadmap

Near-term (expected to push MI accuracy to 50–60%, competitive with EEGNet):

1. **Multi-subject fine-tuning** — use labeled data from all 9 BCI IV 2a subjects
2. **Fix L_asym** — correct sign/index so hemispheric ERD trains in the right direction
3. **Task-aligned pretraining** — contrastive or pseudo-label loss to replace pure MAE
4. **Cross-dataset** — pretrain on MOABB benchmark (20+ datasets), evaluate zero-shot

Seizure-specific:

1. **Multi-patient pretraining** — CHB-MIT patients 2–20 as pretraining corpus
2. **Threshold optimisation** — sweep decision threshold for target sensitivity ≥90%
3. **Temporal context** — sliding-window majority vote across consecutive windows

---

## References

- **REVE**: *Scalable EEG Foundation Model* (NeurIPS 2025)
- **EEGPT**: SNR alignment loss design
- **BCI Competition IV 2a**: Brunner et al., 4-class motor imagery dataset, 9 subjects, 22 channels @ 250 Hz
- Forward model computed via [MNE-Python](https://mne.tools) using the `fsaverage` ico-3 source space (1,284 dipoles)
