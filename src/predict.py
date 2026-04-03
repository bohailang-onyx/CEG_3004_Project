"""
Prediction / Submission Generation Module
==========================================

Generates predictions for the submission set (clean, noisy, band-limited clips)
using the trained model and feature extraction pipeline.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from .feature_extraction import extract_features


def generate_predictions(model, idx_to_label, submission_dir, output_csv):
    """Generate predictions for all submission clips.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Trained classification model.
    idx_to_label : dict
        Mapping from class index to label string.
    submission_dir : str
        Path to submission directory containing metadata.csv and audio/.
    output_csv : str
        Output path for predictions CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['clip_id', 'predicted_label'].
    """
    sub_meta = pd.read_csv(os.path.join(submission_dir, 'metadata.csv'))
    sub_meta['clip_id'] = sub_meta['clip_id'].astype(str)
    audio_sub_dir = os.path.join(submission_dir, 'audio')

    print(f'Generating predictions for {len(sub_meta)} submission clips...')

    pred_rows = []
    for _, r in tqdm(sub_meta.iterrows(), total=len(sub_meta)):
        clip_id = r['clip_id']
        wav_path = os.path.join(audio_sub_dir, f'{clip_id}.wav')
        feat = extract_features(wav_path)
        pred_idx = int(model.predict(feat.reshape(1, -1))[0])
        pred_label = idx_to_label[pred_idx]
        pred_rows.append((clip_id, pred_label))

    out = pd.DataFrame(pred_rows, columns=['clip_id', 'predicted_label'])
    out.to_csv(output_csv, index=False)
    print(f'Predictions saved to {output_csv}')

    return out