"""
Model Training Module
=====================

Implements the classification pipeline using a Voting Ensemble of three diverse
tree-based classifiers:

1. HistGradientBoostingClassifier - Sequential boosting with histogram binning
   for fast training on large feature sets. Strong on tabular data.
2. RandomForestClassifier - Bagging ensemble that reduces variance through
   bootstrap aggregation and random feature selection.
3. ExtraTreesClassifier - Extremely Randomized Trees that add extra randomness
   in split selection, further reducing variance and improving robustness.

The ensemble uses hard voting (majority vote) to combine predictions, which
provides better generalization than any single model.

A StandardScaler is applied first to normalize features to zero mean and unit
variance, which is important because features span different scales (e.g.,
MFCCs vs. spectral centroid in Hz).
"""

import numpy as np
import joblib
from sklearn.ensemble import (
    VotingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score


def build_model():
    """Build the classification pipeline.

    Returns
    -------
    model : sklearn.pipeline.Pipeline
        StandardScaler + VotingClassifier ensemble.
    """
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', VotingClassifier(estimators=[
            ('hgb', HistGradientBoostingClassifier(
                max_iter=1500,
                max_depth=8,
                learning_rate=0.03,
                max_leaf_nodes=63,
                min_samples_leaf=2,
                l2_regularization=0.01,
                max_bins=255,
                random_state=42
            )),
            ('rf', RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )),
            ('et', ExtraTreesClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )),
        ], voting='hard'))
    ])
    return model


def train_and_evaluate(X, y, idx_to_label, test_size=0.2, random_state=42):
    """Train model and print evaluation metrics.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        Label indices.
    idx_to_label : dict
        Mapping from index to class name.
    test_size : float
        Fraction of data for validation.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    model : sklearn.pipeline.Pipeline
        Trained model.
    macro_f1 : float
        Macro F1-score on validation set.
    """
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = build_model()
    classes = sorted(idx_to_label.keys())
    target_names = [idx_to_label[i] for i in classes]

    print(f"Training on {X_tr.shape[0]} samples with {X_tr.shape[1]} features...")
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_va)

    print(classification_report(y_va, y_pred, target_names=target_names))
    macro_f1 = f1_score(y_va, y_pred, average='macro')
    print(f'Macro-F1: {macro_f1:.4f}')

    return model, macro_f1


def save_model(model, path):
    """Save trained model to disk."""
    joblib.dump(model, path)
    print(f'Model saved to {path}')


def load_model(path):
    """Load trained model from disk."""
    return joblib.load(path)