# CEG3004 Environmental Sound Classification

**Group: Pr_14**

A robust audio classification pipeline for Environmental Sound Classification (ESC-50) that performs well under clean, noisy, and band-limited conditions.

## Project Overview

This project classifies 2,000 audio clips (5 seconds each, mono, 16 kHz) across **50 environmental sound classes** using a DSP feature extraction pipeline combined with a machine learning ensemble classifier. The system is designed to be robust against additive noise and band-limiting distortions.

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
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

## DSP Pipeline

### 1. Preprocessing (`src/audio_processing.py`)

| Step | Technique | Purpose |
|------|-----------|---------|
| 1 | Pre-emphasis filter (coeff=0.97) | Boosts high frequencies to compensate for natural spectral tilt |
| 2 | Silence trimming (20 dB threshold) | Removes non-informative leading/trailing silence |
| 3 | Fixed-length padding/truncation (5s) | Ensures uniform input length for feature extraction |
| 4 | RMS normalization (target=0.1) | Consistent energy levels across clips |

### 2. Feature Extraction (`src/feature_extraction.py`)

639-dimensional feature vector composed of:

| Feature Group | Dimensions | Description |
|---------------|-----------|-------------|
| MFCC (narrow) | 180 | 30 MFCCs + delta + delta-delta (n_fft=1024) |
| MFCC (wide) | 180 | Same at coarser resolution (n_fft=2048) |
| Log-Mel + CMVN | 192 | 64-band log-mel with cepstral mean/variance normalization |
| Spectral Shape | 25 | Centroid, bandwidth, rolloff, flatness, ZCR with percentile pooling |
| Spectral Contrast | 14 | Peak-to-valley ratio across 7 frequency bands |
| Chroma | 24 | Pitch class profile (12 classes x 2 stats) |
| Tonnetz | 12 | Tonal centroid features from harmonic component |
| RMS Energy | 6 | Energy statistics with percentiles |
| Temporal | 6 | Autocorrelation periodicity + envelope dynamics |

**Robustness features:**
- **CMVN** on log-mel features provides invariance to channel effects and varying noise floors
- **Percentile pooling** (25th, median, 75th) reduces sensitivity to noise-induced outlier frames
- **Multi-resolution MFCC** captures both fine-grained and coarse spectral patterns
- **Pre-emphasis** compensates for high-frequency loss under band-limited conditions

### 3. Data Augmentation

Training data is augmented 7x (1 original + 6 augmented) to improve robustness:

| Augmentation | Parameters | Simulates |
|--------------|-----------|-----------|
| White noise | SNR 5-20 dB | Noisy submission condition |
| Butterworth LPF | Cutoff 1-4 kHz | Band-limited submission condition |
| Random combination | Mixed | Compound distortions |
| Time stretch | Rate 0.85-1.15 | Duration variability |
| Pitch shift | +/-2 semitones | Pitch variability |
| Time mask + noise | 5-15% masked | Temporal robustness |

### 4. Classifier (`src/model.py`)

Voting ensemble of three diverse classifiers:
- **HistGradientBoostingClassifier** (1500 iterations, lr=0.03, depth=8)
- **RandomForestClassifier** (500 trees)
- **ExtraTreesClassifier** (500 trees)

Hard voting combines predictions for better generalization. A `StandardScaler` normalizes features before classification.

## How to Reproduce

### Option 1: Google Colab (Recommended)

1. Open `CEG3004_Project_Colab.ipynb` in Google Colab
2. Ensure `GROUP_ID = "Pr_14"` is set in the first code cell
3. Run all cells sequentially
4. The model (`.joblib`) and predictions (`.csv`) are automatically downloaded

### Option 2: Local Environment

```bash
# 1. Clone the repository
git clone <repo-url>
cd <repo-name>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
# Place the extracted data/ folder in the project root with structure:
#   data/train/labels.csv
#   data/train/audio/*.wav
#   data/submission/metadata.csv
#   data/submission/audio/*.wav

# 4. Run the notebook
jupyter notebook CEG3004_Project_Colab.ipynb
```

### Dependencies

- Python 3.8+
- NumPy, SciPy, Pandas
- scikit-learn (StandardScaler, Pipeline, ensemble classifiers)
- librosa (audio loading and DSP feature extraction)
- soundfile, tqdm, joblib

## Results

The model achieves robust classification across all three submission conditions:
- **Clean**: Original audio signals
- **Noisy**: Additive noise applied
- **Band-limited**: Frequency content restricted

Performance is evaluated as: **50% Clean + 25% Noisy + 25% Band-limited**
