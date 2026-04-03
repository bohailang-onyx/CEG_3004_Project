# CEG3004 Environmental Sound Classification
## Team
- Group ID: Pr_14
- Members: Chai Yong Huat and Yeo Hong Yi

---

##Project Title

A Robust Audio Classification Pipeline for Environmental Sound Classification (ESC-50) Under Clean, Noisy, and Band-Limited Conditions

---

## Project Overview

This project classifies 2,000 audio clips (5 seconds each, mono) across 50 environmental sound classes using a DSP feature extraction pipeline combined with a machine learning ensemble classifier. The system is specifically designed to remain robust against additive noise and band-limiting distortions, which are the two main distortion types present in the submission set.

The submission set contains 1,200 clips — 400 clean, 400 noisy, and 400 band-limited — all generated from the same underlying sound events.

---

## Repository Structure

```
.
├── CEG3004_Project_Colab.ipynb   # Main Colab notebook (run this to reproduce)
├── src/                          # Modular source code
│   ├── __init__.py
│   ├── audio_processing.py       # Audio loading, preprocessing, augmentation
│   ├── feature_extraction.py     # DSP feature extraction (639-dim vector)
│   ├── model.py                  # Model building, training, evaluation
│   └── predict.py                # Submission prediction generation
├── figures/                      # Visualizations generated from notebook
│   ├── fig1_signal_comparison.png
│   ├── fig2_prediction_distribution.png
│   ├── fig3_instability.png
│   ├── fig4_confusion_matrix.png
│   └── fig5_feature_importance.png
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

---

## DSP Pipeline

### 1. Preprocessing (`src/audio_processing.py`)

| Step | Technique | Purpose |
|------|-----------|---------|
| 1 | Pre-emphasis filter (coeff=0.97) | Boosts high frequencies to compensate for spectral tilt and high-frequency loss under band-limiting |
| 2 | Silence trimming (20 dB threshold) | Removes non-informative leading/trailing silence |
| 3 | Fixed-length padding/truncation (5s) | Ensures uniform input length for feature extraction |
| 4 | RMS normalization (target=0.1) | Consistent energy levels across clips regardless of recording volume |

### 2. Feature Extraction (`src/feature_extraction.py`)

The pipeline extracts a **639-dimensional feature vector** per clip:

| Feature Group | Dimensions | Description |
|---------------|-----------|-------------|
| MFCC (narrow) | 180 | 30 MFCCs + delta + delta-delta (n_fft=1024) |
| MFCC (wide) | 180 | Same at coarser resolution (n_fft=2048) |
| Log-Mel + CMVN | 192 | 64-band log-mel with cepstral mean/variance normalization |
| Spectral Shape | 25 | Centroid, bandwidth, rolloff, flatness, ZCR with percentile pooling |
| Spectral Contrast | 14 | Peak-to-valley ratio across 7 frequency bands |
| Chroma | 24 | Pitch class profile (12 classes × 2 stats) |
| Tonnetz | 12 | Tonal centroid features derived from harmonic component |
| RMS Energy | 6 | Energy statistics with percentiles |
| Temporal | 6 | Autocorrelation periodicity + envelope dynamics |

**Why these features were chosen:**

The feature set was designed around two core robustness requirements: surviving additive noise and surviving band-limiting.

**MFCCs** are the standard baseline for environmental sound classification because they compactly represent the spectral envelope in a perceptually meaningful way. We use **dual-resolution MFCCs** (n_fft=1024 and n_fft=2048) deliberately: the narrow window captures fine-grained spectral detail that discriminates similar-sounding classes under clean conditions, while the wide window captures coarser spectral shape that remains stable even when noise or band-limiting degrades fine structure. If one resolution fails under distortion, the other compensates.

**CMVN (Cepstral Mean and Variance Normalization)** on the log-mel features subtracts the per-feature mean and divides by the standard deviation across time frames. This normalizes out slowly varying channel effects and additive noise floor shifts — a noise addition that raises the energy uniformly across the spectrum is largely cancelled by CMVN, making these features significantly more stable under the noisy submission condition.

**Percentile pooling** (25th, 50th, 75th percentiles across time frames) is used instead of plain mean/std for spectral shape features. Mean and standard deviation are sensitive to outlier frames caused by noise bursts. Percentile-based statistics are more robust because a noise spike only shifts the sample at its percentile rank rather than distorting the whole statistic.

**Spectral contrast** measures the difference between spectral peaks and valleys within each frequency band. This ratio is more stable under noise than absolute spectral energy — additive noise raises both peaks and valleys roughly equally, preserving the contrast ratio. Under band-limiting, the lower-frequency bands remain unaffected and retain their discriminative contrast.

**Pre-emphasis** (y[n] = y[n] − 0.97·y[n−1]) amplifies high-frequency components before feature extraction. This partially compensates for the energy loss caused by the band-limiting LPF, ensuring that features derived from the upper spectral region are not entirely lost.

### 3. Data Augmentation

Training data is augmented **7× (1 original + 6 augmented)** to simulate submission conditions:

| Augmentation | Parameters | Simulates |
|-------------|-----------|-----------|
| White noise | SNR 5–20 dB | Noisy submission condition |
| Butterworth LPF | Cutoff 1–4 kHz | Band-limited submission condition |
| Random combination | Mixed | Compound distortions |
| Time stretch | Rate 0.85–1.15 | Duration variability |
| Pitch shift | ±2 semitones | Pitch variability |
| Time mask + noise | 5–15% masked | Temporal robustness |

Augmenting with the exact same distortion types present in the submission set directly exposes the model to those conditions during training, rather than relying solely on feature-level robustness.

### 4. Classifier (`src/model.py`)

**Why a hard-voting ensemble of tree-based classifiers?**

For a 50-class classification problem with a 639-dimensional hand-crafted feature vector and ~14,000 training samples (2,000 × 7 augmentations), tree-based ensemble methods are a well-suited choice for the following reasons:

- **No gradient vanishing / exploding**: Tree ensembles handle mixed-scale features natively without requiring careful architecture tuning, unlike neural networks.
- **Robustness to irrelevant features**: Random Forest and Extra Trees use random feature subsampling at each split, which reduces the influence of noisy or less discriminative dimensions in the 639-dim vector.
- **Complementary error patterns**: The three classifiers use fundamentally different learning strategies — sequential boosting (HGB), bootstrap aggregation (RF), and random splits (Extra Trees). Their prediction errors are less correlated, so hard voting corrects mistakes that any single model would make.
- **Practical training time**: Gradient boosted trees with 1500 iterations and two forests of 500 trees each train in under 10 minutes on Colab, making experimentation feasible within the project timeline.
- **No overfitting risk from augmentation leakage**: Tree models with regularization (l2, min_samples_leaf) generalize well even when augmented and original clips share the same sound identity.

A neural network (e.g., CNN on mel spectrograms) could potentially achieve higher accuracy on clean data, but would require significantly more training data and careful regularization to be competitive under the noisy and band-limited conditions given the dataset size.


A **hard-voting ensemble** of three diverse tree-based classifiers:

| Classifier | Parameters |
|-----------|-----------|
| HistGradientBoostingClassifier | 1500 iterations, lr=0.03, max_depth=8, l2=0.01 |
| RandomForestClassifier | 500 trees |
| ExtraTreesClassifier | 500 trees |

Features are normalized with `StandardScaler` before classification. Hard voting combines three independent predictions, reducing variance compared to any single classifier. The three classifiers are intentionally diverse — boosting, bagging, and random splits — so their errors are less correlated and voting corrects individual mistakes.

---

## How to Reproduce

### Option 1: Google Colab (Recommended)

1. Open `CEG3004_Project_Colab.ipynb` in Google Colab
2. Ensure `GROUP_ID = "Pr_14"` is set in the first code cell
3. Mount Google Drive or upload the dataset when prompted
4. Run all cells sequentially (Runtime → Run all)
5. The model (`.joblib`) and predictions (`.csv`) are automatically downloaded

### Option 2: Local Environment

```bash
# 1. Clone the repository
git clone https://github.com/bohailang-onyx/CEG_3004_Project.git
cd CEG_3004_Project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the dataset in the project root:
#   data/train/labels.csv
#   data/train/audio/*.wav
#   data/submission/metadata.csv
#   data/submission/audio/*.wav

# 4. Run the notebook
jupyter notebook CEG3004_Project_Colab.ipynb
```

---

## Dependencies

- Python 3.8+
- `numpy`, `scipy`, `pandas`
- `scikit-learn` — StandardScaler, Pipeline, ensemble classifiers
- `librosa` — audio loading and DSP feature extraction
- `soundfile`, `tqdm`, `joblib`

```bash
pip install -r requirements.txt
```

---

## Results

### Validation Performance

The model was evaluated on a held-out 20% stratified validation split of the training set (clean audio only):

| Metric | Score |
|--------|-------|
| Validation Accuracy | ~72% |
| Weighted F1 Score | ~0.71 |

> The validation set uses only clean training audio. The submission set introduces noisy and band-limited conditions not seen during validation, so submission performance is expected to differ. The 7× augmentation strategy is designed to close this gap by exposing the model to matching distortion types during training.

### Submission Predictions

The model generates predictions for **1,200 submission clips** across all three conditions:

| Condition | Clips | Weighting |
|-----------|-------|-----------|
| Clean | 400 | 50% |
| Noisy | 400 | 25% |
| Band-limited | 400 | 25% |

Final performance score = **50% Clean + 25% Noisy + 25% Band-limited**

---

## Visualizations

### Figure 1 — Signal Comparison: Clean vs Noisy vs Band-limited

Waveform and log-mel spectrogram for all three conditions on the same clip. The noisy condition raises the noise floor uniformly across all mel bands. Band-limiting visibly removes energy in the upper frequency bands above ~3 kHz.

![Signal Comparison](figures/fig1_signal_comparison.png)

---

### Figure 2 — Prediction Distribution Across Conditions (Top 20 Classes)

Shows how many times each class was predicted under each condition. Large bar height differences for the same class reveal where the model's decisions shift most under distortion.

![Prediction Distribution](figures/fig2_prediction_distribution.png)

---

### Figure 3 — Per-Class Prediction Instability Under Distortion

Instability score = |clean count − noisy count| + |clean count − band-limited count|. A high score means the model frequently changes its prediction for that class depending on the distortion applied.

![Instability](figures/fig3_instability.png)

---

### Figure 4 — Confusion Matrix (15 Most Confused Classes, Validation Set)

Subset of the full 50×50 confusion matrix showing the 15 classes with the lowest per-class validation accuracy. Off-diagonal entries reveal which specific class pairs are most frequently confused.

![Confusion Matrix](figures/fig4_confusion_matrix.png)

---

### Figure 5 — Feature Group Importance (Random Forest)

Relative contribution of each DSP feature group to the Random Forest classifier's decisions. Confirms which parts of the 639-dimensional feature vector carry the most discriminative signal for ESC-50.

![Feature Importance](figures/fig5_feature_importance.png)

---

## Error Analysis

### Instability Under Distortion

Analysis of the 1,200 submission predictions reveals clear patterns in which classes are most affected by each distortion type.

**`rain` — most noise-sensitive class.**
Predictions spike from 12 (clean) to 55 (noisy), a 4.6× increase and the largest single instability in the submission set. The cause is structural: white noise has a similar broadband, temporally uniform texture to rain. Even with CMVN normalization, at low SNR the model conflates the two. This is a fundamental limitation of feature-based classifiers for this class pair — a deeper temporal model would be required to distinguish them reliably at very low SNR.

**`airplane`, `dog`, `clapping` — most band-limit sensitive.**
These classes gain false prominence under band-limiting. When the Butterworth LPF removes high-frequency content from competing classes, low-frequency dominant sounds (engine rumble, dog barks, low-frequency claps) occupy a larger share of the remaining spectral space. `airplane` predictions increase from 5 (clean) to 17 (band-limited), and `dog` from 5 to 21.

**`clock_tick` — transient masking under noise.**
Predictions collapse from 15 (clean) to 1 (noisy). The click transients that define `clock_tick` are sharp, high-energy, short-duration events. Additive noise masks these transients, making the signal resemble a continuous texture class such as `crackling_fire` or `rain`. Percentile pooling partially mitigates this but cannot fully recover heavily masked transients at low SNR.

**`frog` and `coughing` — low-energy class collapse.**
Both drop to 0 noisy predictions. These classes have relatively subtle, low-energy spectral signatures that are easily overwhelmed when additive noise raises the noise floor.

### Most Stable Classes

Classes with strong pitch-periodic structure (`rooster`, `siren`, `crying_baby`) and dominant low-to-mid frequency energy (`chainsaw`, `engine`) remain consistently predicted across all conditions. Their fundamental frequencies and harmonic structure persist even under noise and band-limiting, making them naturally robust.

### Design Decisions Motivated by This Analysis

| Observation | Design Response |
|-------------|----------------|
| `rain`/noise confusion | Augment training with SNR 5–20 dB white noise to expose model to the confusion boundary |
| Band-limit hurts high-freq classes | Pre-emphasis + dual-resolution MFCC retains coarse spectral structure when fine detail is lost |
| Transient masking (`clock_tick`) | Percentile pooling preserves frame-level peak statistics better than mean/std |
| Low-energy class collapse (`frog`, `coughing`) | RMS normalization + silence trimming ensures consistent energy baseline |

---

## Grading Checklist

- [x] All source code in `src/`
- [x] Clear README with DSP rationale
- [x] Reproducible instructions (Colab + local)
- [x] Data augmentation matching submission distortions
- [x] Ensemble classifier with StandardScaler pipeline
- [x] Visualizations in `figures/`
- [x] Error analysis grounded in submission predictions
- [x] Predictions CSV: `Pr_14_predictions.csv`
- [x] Model file: `Pr_14_model.joblib`
