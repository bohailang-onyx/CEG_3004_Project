# CEG3004 Project – Environmental Sound Classification

## 1. Project Overview

This project implements a robust Environmental Sound Classification pipeline for the CEG3004 project. The objective is to classify environmental audio clips into 50 sound classes using digital signal processing (DSP) feature extraction and machine learning. The system is designed to perform not only on clean audio, but also under noisy and band-limited conditions.

The project is based on a dataset derived from ESC-50, where each audio clip is 5 seconds long and belongs to one of 50 sound categories. The final system is evaluated on three conditions:
- Clean audio
- Noisy audio
- Band-limited audio

This project focuses on:
- meaningful DSP feature engineering
- robustness against distortions
- structured model experimentation
- reproducible implementation

---

## 2. Objective

The main goal of this project is to build an environmental sound classifier that performs well under realistic signal distortions. Rather than relying only on model complexity, this work emphasizes strong DSP fundamentals, meaningful feature design, and systematic experimentation.

---

## 3. Dataset Description

The dataset contains environmental sound clips across 50 classes. Each clip is:
- mono channel
- 5 seconds long
- labeled in the training set
- unlabeled in the submission set

The submission set includes 3 versions of each clip:
- clean
- noisy
- band-limited

These versions correspond to the same underlying sound event and are intended to evaluate model robustness.

---

## 4. Repository Structure

```text
CEG3004-Environmental-Sound-Classification/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── notebooks/
│   └── CEG3004_Project_Colab.ipynb
├── src/
│   ├── features.py
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── experiments/
│   ├── experiment_log.md
│   ├── results_table.csv
│   └── observations.md
├── figures/
│   ├── waveform_example.png
│   ├── spectrogram_example.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── submit/
│   ├── Pr_14_predictions.csv
│   ├── Pr_14_model.joblib
│   └── github_link.txt
└── docs/
    └── methodology.md
