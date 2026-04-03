"""
DSP Feature Extraction Module
==============================

Extracts a comprehensive 639-dimensional feature vector per audio clip by combining
multiple DSP-based feature families. Each feature group captures different aspects
of the audio signal:

Feature Groups:
- MFCCs (180-dim): Mel-frequency cepstral coefficients with delta and delta-delta,
  capturing spectral envelope shape. Computed at two resolutions (n_fft=1024 and 2048)
  for multi-scale analysis.
- Log-Mel Spectrogram (192-dim): 64-band log-mel features with CMVN (Cepstral Mean
  and Variance Normalization) for channel-invariant representation.
- Spectral Shape (25-dim): Centroid, bandwidth, rolloff, flatness, and ZCR with
  percentile pooling (25th, median, 75th) for robust temporal summarization.
- Spectral Contrast (14-dim): Peak-to-valley ratio across frequency bands,
  effective for distinguishing harmonic vs. noisy sound textures.
- Chroma (24-dim): Pitch class profile capturing tonal content.
- Tonnetz (12-dim): Tonal centroid features from harmonic component.
- RMS Energy (6-dim): Energy statistics including percentiles for dynamics.
- Temporal (6-dim): Autocorrelation peak (periodicity detector) and amplitude
  envelope dynamics (attack/decay characteristics).

Robustness Strategy:
- CMVN on log-mel features provides invariance to channel effects and noise floors.
- Percentile pooling (instead of only mean/std) reduces sensitivity to outlier frames
  caused by noise bursts or spectral artifacts.
- Multi-resolution MFCC captures both fine-grained and coarse spectral patterns.
- Pre-emphasis in preprocessing compensates for high-frequency energy loss in
  band-limited conditions.
"""

import numpy as np
import librosa

from .audio_processing import load_audio, preprocess_audio


def features_mfcc_stats(y, sr, n_mfcc=30, n_fft=1024, hop=256):
    """Extract MFCC statistics: mean and std of MFCCs, delta, and delta-delta.

    MFCCs approximate the human auditory system's frequency resolution using
    the mel scale. Delta and delta-delta coefficients capture temporal dynamics
    of the spectral envelope.

    Returns: 180-dim vector (30 coefficients x 3 derivatives x 2 statistics).
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    def stats(M):
        return np.concatenate([M.mean(axis=1), M.std(axis=1)], axis=0)

    return np.concatenate([stats(mfcc), stats(d1), stats(d2)], axis=0).astype(np.float32)


def features_log_mel_stats(y, sr, n_mels=64, n_fft=1024, hop=256):
    """Extract log-mel spectrogram statistics with CMVN.

    CMVN (Cepstral Mean and Variance Normalization) subtracts the per-band mean
    and divides by per-band std, removing channel-specific bias. This is critical
    for robustness to noise and band-limiting, as these distortions shift the
    spectral floor differently across frequency bands.

    Returns: 192-dim vector (64 mel bands x 3 statistics).
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # CMVN: normalize each frequency band independently
    log_mel = (log_mel - log_mel.mean(axis=1, keepdims=True)) / (log_mel.std(axis=1, keepdims=True) + 1e-8)
    mean = log_mel.mean(axis=1)
    std = log_mel.std(axis=1)
    median = np.median(log_mel, axis=1)
    return np.concatenate([mean, std, median], axis=0).astype(np.float32)


def features_spectral(y, sr, n_fft=1024, hop=256):
    """Extract spectral shape descriptors with percentile pooling.

    - Spectral Centroid: "brightness" of the sound (frequency center of mass)
    - Spectral Bandwidth: spread of the spectrum around the centroid
    - Spectral Rolloff: frequency below which 85% of energy is concentrated
    - Spectral Flatness: tonality measure (0=tonal, 1=noise-like)
    - Zero-Crossing Rate: simple time-domain feature correlated with noisiness

    Percentile pooling (25th, median, 75th) is more robust than mean/std alone,
    as it reduces the influence of outlier frames from noise bursts.

    Returns: 25-dim vector (5 features x 5 statistics).
    """
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)

    feats = []
    for feat in [centroid, bandwidth, rolloff, flatness, zcr]:
        feats.extend([feat.mean(), feat.std(), np.median(feat),
                      np.percentile(feat, 25), np.percentile(feat, 75)])
    return np.array(feats, dtype=np.float32)


def features_spectral_contrast(y, sr, n_fft=1024, hop=256):
    """Extract spectral contrast features.

    Measures the peak-to-valley ratio in each sub-band of the spectrum.
    Harmonic sounds (e.g., music, bird calls) have high contrast, while
    noisy sounds (e.g., rain, static) have low contrast. This distinction
    is particularly useful for environmental sound classification.

    Returns: 14-dim vector (7 bands x 2 statistics).
    """
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_bands=6)
    mean = contrast.mean(axis=1)
    std = contrast.std(axis=1)
    return np.concatenate([mean, std], axis=0).astype(np.float32)


def features_chroma(y, sr, n_fft=1024, hop=256):
    """Extract chroma (pitch class profile) features.

    Projects the spectrum onto 12 pitch classes (C, C#, ..., B).
    Useful for sounds with tonal content (e.g., instruments, sirens).

    Returns: 24-dim vector (12 pitch classes x 2 statistics).
    """
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    mean = chroma.mean(axis=1)
    std = chroma.std(axis=1)
    return np.concatenate([mean, std], axis=0).astype(np.float32)


def features_tonnetz(y, sr):
    """Extract tonnetz (tonal centroid) features.

    Computed from the harmonic component of the signal, capturing
    tonal relationships (fifths, minor thirds, major thirds).

    Returns: 12-dim vector (6 dimensions x 2 statistics).
    """
    harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
    mean = tonnetz.mean(axis=1)
    std = tonnetz.std(axis=1)
    return np.concatenate([mean, std], axis=0).astype(np.float32)


def features_rms_energy(y, n_fft=1024, hop=256):
    """Extract RMS energy statistics with percentiles.

    Captures the overall loudness profile and dynamic range of the clip.
    The 10th and 90th percentiles provide a robust measure of the
    energy floor and ceiling.

    Returns: 6-dim vector.
    """
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)
    return np.array([rms.mean(), rms.std(), np.median(rms), rms.max(),
                     np.percentile(rms, 10), np.percentile(rms, 90)], dtype=np.float32)


def features_temporal(y, sr):
    """Extract temporal features: autocorrelation and envelope dynamics.

    - Autocorrelation peak: detects periodicity (useful for distinguishing
      periodic sounds like engine hum from impulsive sounds like clapping).
    - Envelope dynamics: attack/decay characteristics via amplitude envelope
      difference statistics.

    Returns: 6-dim vector.
    """
    seg = y[:sr]
    autocorr = np.correlate(seg, seg, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr = autocorr / (autocorr[0] + 1e-8)
    peak_idx = np.argmax(autocorr[50:500]) + 50 if len(autocorr) > 500 else 0
    peak_val = autocorr[peak_idx] if peak_idx > 0 else 0.0

    envelope = np.abs(y)
    frame_len = sr // 20
    if len(envelope) > frame_len:
        n_frames = len(envelope) // frame_len
        envelope_frames = envelope[:n_frames * frame_len].reshape(n_frames, frame_len).mean(axis=1)
        env_diff = np.diff(envelope_frames)
        env_feats = [envelope_frames.mean(), envelope_frames.std(), env_diff.mean(), env_diff.std()]
    else:
        env_feats = [0.0, 0.0, 0.0, 0.0]

    return np.array([peak_val, float(peak_idx) / sr] + env_feats, dtype=np.float32)


def extract_features_from_array(y, sr):
    """Extract full 639-dimensional feature vector from preprocessed audio.

    Concatenates all feature groups into a single vector for classification.

    Parameters
    ----------
    y : np.ndarray
        Preprocessed audio signal (5 seconds at target sr).
    sr : int
        Sample rate.

    Returns
    -------
    feat : np.ndarray
        1D feature vector (639 dimensions).
    """
    mfcc_feat = features_mfcc_stats(y, sr)                                      # 180-dim
    mel_feat = features_log_mel_stats(y, sr)                                     # 192-dim
    spec_feat = features_spectral(y, sr)                                         # 25-dim
    contrast_feat = features_spectral_contrast(y, sr)                            # 14-dim
    chroma_feat = features_chroma(y, sr)                                         # 24-dim
    tonnetz_feat = features_tonnetz(y, sr)                                       # 12-dim
    rms_feat = features_rms_energy(y)                                            # 6-dim
    temporal_feat = features_temporal(y, sr)                                      # 6-dim
    mfcc_feat_wide = features_mfcc_stats(y, sr, n_mfcc=30, n_fft=2048, hop=512) # 180-dim

    feat = np.concatenate([
        mfcc_feat,        # 180
        mel_feat,         # 192
        spec_feat,        # 25
        contrast_feat,    # 14
        chroma_feat,      # 24
        tonnetz_feat,     # 12
        rms_feat,         # 6
        temporal_feat,    # 6
        mfcc_feat_wide,   # 180
    ], axis=0)

    return feat


def extract_features(path, sr=16000):
    """Extract features from a WAV file path.

    Loads audio, applies preprocessing, then extracts the full feature vector.

    Parameters
    ----------
    path : str
        Path to WAV file.
    sr : int
        Target sample rate.

    Returns
    -------
    np.ndarray
        1D feature vector.
    """
    y, sr = load_audio(path, sr=sr)
    y = preprocess_audio(y, sr)
    return extract_features_from_array(y, sr)