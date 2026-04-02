# CEG3004 DSP Mini-Project: Environmental Sound Classification

## Team
- Group ID: Pr_14
- Members: Chai Yong Huat and Yeo Hong Yi

## Project Overview
This project builds a robust environmental sound classification pipeline using DSP-based feature extraction and machine learning. The model is trained on labeled audio clips and evaluated on clean, noisy, and band-limited submission data.

## Objectives
- Extract meaningful DSP features from 5-second audio clips
- Train a classifier for 50 sound classes
- Improve robustness under noisy and band-limited conditions

## Dataset
- Based on ESC-50
- 50 classes
- 5-second mono audio clips
- Submission set contains:
  - 400 clean clips
  - 400 noisy clips
  - 400 band-limited clips

## Methodology
### Preprocessing
- Mono loading at 16 kHz
- Pre-emphasis filtering
- Silence trimming
- Fixed-length padding/truncation to 5 seconds
- RMS normalization

### Feature Extraction
- MFCC statistics
- Delta and delta-delta MFCC
- Log-mel spectrogram statistics
- Spectral centroid / bandwidth / rolloff / flatness / ZCR
- Spectral contrast
- Chroma
- Tonnetz
- RMS energy
- Temporal features
- Wide-window MFCC variant

### Data Augmentation
- Additive white noise
- Band-limiting
- Random gain
- Time stretching
- Pitch shifting
- Time masking
- Combined augmentation

### Model
- StandardScaler
- VotingClassifier
  - HistGradientBoostingClassifier
  - RandomForestClassifier
  - ExtraTreesClassifier

## Experiments
Summarized in `experiments/experiment_log.md`.

## Repository Structure
[show folder tree]

## Reproducibility
1. Install dependencies:
   `pip install -r requirements.txt`
2. Open and run `notebooks/CEG3004_Project_Colab.ipynb`
3. Download dataset from provided Google Drive link
4. Run all cells
5. Outputs generated:
   - `Pr_14_model.joblib`
   - `Pr_14_predictions.csv`

## Submission Files
- Model file
- Prediction CSV
- GitHub link
