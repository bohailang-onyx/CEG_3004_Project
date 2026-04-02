# CEG3004 Project – Environmental Sound Classification

## 1. Project Overview

This project implements a robust Environmental Sound Classification pipeline for the CEG3004 DSP mini-project. The objective is to classify environmental audio clips into 50 sound classes using digital signal processing (DSP) feature engineering and machine learning.

The system is designed not only for clean audio, but also to remain robust under noisy and band-limited conditions. This is important because the submission set includes all three versions of each sound event.

The final implementation focuses on:
- strong DSP fundamentals
- robustness-oriented preprocessing and augmentation
- meaningful feature engineering
- ensemble-based machine learning
- reproducible workflow

---

## 2. Project Objective

The aim of this project is to design a robust audio classification pipeline for Environmental Sound Classification that performs well under clean and distorted conditions.

The project emphasizes:
- meaningful DSP feature extraction
- robustness to distortions
- systematic experimentation
- clean engineering and documentation practices

---

## 3. Dataset Summary

The dataset is derived from ESC-50 and contains environmental sound clips across 50 classes.

Each clip is:
- mono audio
- 5 seconds long

The project uses:
- a labeled training set in `data/train`
- an unlabeled submission set in `data/submission`

The submission set contains three versions of each sound clip:
- clean
- noisy
- band-limited

---

## 4. Repository Structure

```text
CEG3004_Project/
├── README.md
├── .gitignore
├── requirements.txt
├── docs/
│   └── methodology.md
├── experiments/
│   └── experiment_log.md
├── notebooks/
│   └── CEG3004_Project_Colab.ipynb
└── submit/
    ├── Pr_14_predictions.csv
    ├── github_link.txt
    └── README.md
