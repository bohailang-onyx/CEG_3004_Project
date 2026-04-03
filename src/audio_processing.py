"""
Audio Loading, Preprocessing, and Augmentation Module
=====================================================

This module handles the DSP preprocessing pipeline:
1. Audio loading and resampling to 16 kHz mono
2. Preprocessing: pre-emphasis, silence trimming, fixed-length padding, RMS normalization
3. Data augmentation: noise injection, band-limiting, gain, time-stretch, pitch-shift, time-masking

DSP Concepts:
- Pre-emphasis filter (y[n] = x[n] - 0.97*x[n-1]) boosts high frequencies to compensate
  for the natural spectral tilt of speech/environmental sounds, improving MFCC extraction.
- RMS normalization ensures consistent energy levels across clips, which is critical for
  robustness when the model encounters noisy or band-limited test signals.
- Butterworth low-pass filtering simulates band-limited channel conditions by restricting
  frequency content, matching the submission set's distortion profile.
"""

import numpy as np
import librosa
from scipy.signal import butter, sosfilt


def load_audio(path, sr=16000):
    """Load mono audio and resample to target sample rate.

    Parameters
    ----------
    path : str
        Path to the WAV file.
    sr : int
        Target sample rate (default 16000 Hz).

    Returns
    -------
    y : np.ndarray
        Audio time series (float32).
    sr : int
        Sample rate.
    """
    y, sr_out = librosa.load(path, sr=sr, mono=True)
    y = np.nan_to_num(y).astype(np.float32)
    return y, sr_out


def preprocess_audio(y, sr):
    """Apply DSP preprocessing pipeline to raw audio.

    Pipeline steps:
    1. Pre-emphasis filter (coeff=0.97) to flatten spectral tilt
    2. Silence trimming (threshold=20 dB) to remove dead segments
    3. Fixed-length padding/truncation to exactly 5 seconds
    4. RMS normalization to target level 0.1

    Parameters
    ----------
    y : np.ndarray
        Raw audio signal.
    sr : int
        Sample rate.

    Returns
    -------
    y : np.ndarray
        Preprocessed audio (float32), length = 5 * sr.
    """
    # 1. Pre-emphasis filter to boost high frequencies
    pre_emphasis_coeff = 0.97
    y = np.append(y[0], y[1:] - pre_emphasis_coeff * y[:-1])

    # 2. Trim leading/trailing silence
    y, _ = librosa.effects.trim(y, top_db=20)

    # 3. Fixed-length padding/truncation to exactly 5 seconds
    target_len = 5 * sr
    if len(y) > target_len:
        y = y[:target_len]
    elif len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')

    # 4. RMS normalization
    rms = np.sqrt(np.mean(y ** 2))
    if rms > 1e-6:
        y = y / rms * 0.1

    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# Data Augmentation Functions
# These simulate the noisy and band-limited conditions in the submission set.
# ---------------------------------------------------------------------------

def augment_add_noise(y, snr_db=None):
    """Add white Gaussian noise at a random SNR.

    Simulates additive noise distortion present in the submission set.
    SNR range [5, 20] dB covers mild to moderate noise conditions.

    Parameters
    ----------
    y : np.ndarray
        Input audio signal.
    snr_db : float or None
        Signal-to-noise ratio in dB. If None, randomly sampled from U(5, 20).

    Returns
    -------
    np.ndarray
        Noisy audio signal.
    """
    if snr_db is None:
        snr_db = np.random.uniform(5, 20)
    sig_power = np.mean(y ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    return (y + noise).astype(np.float32)


def augment_bandlimit(y, sr, cutoff=None):
    """Apply low-pass Butterworth filter to simulate band-limiting.

    Uses a 5th-order Butterworth filter (maximally flat passband) to restrict
    frequency content, matching the band-limited condition in the submission set.

    Parameters
    ----------
    y : np.ndarray
        Input audio signal.
    sr : int
        Sample rate.
    cutoff : float or None
        Cutoff frequency in Hz. If None, randomly sampled from U(1000, 4000).

    Returns
    -------
    np.ndarray
        Band-limited audio signal.
    """
    if cutoff is None:
        cutoff = np.random.uniform(1000, 4000)
    nyq = sr / 2.0
    norm_cutoff = cutoff / nyq
    if norm_cutoff >= 1.0:
        return y
    sos = butter(5, norm_cutoff, btype='low', output='sos')
    return sosfilt(sos, y).astype(np.float32)


def augment_gain(y, gain_db=None):
    """Apply random gain change between -6 and +6 dB."""
    if gain_db is None:
        gain_db = np.random.uniform(-6, 6)
    gain = 10 ** (gain_db / 20)
    return (y * gain).astype(np.float32)


def augment_time_stretch(y, rate=None):
    """Time stretch without changing pitch."""
    if rate is None:
        rate = np.random.uniform(0.85, 1.15)
    y_stretched = librosa.effects.time_stretch(y, rate=rate)
    return y_stretched.astype(np.float32)


def augment_pitch_shift(y, sr, n_steps=None):
    """Pitch shift by random semitones."""
    if n_steps is None:
        n_steps = np.random.uniform(-2, 2)
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    return y_shifted.astype(np.float32)


def augment_time_mask(y, max_mask_fraction=0.15):
    """Zero out a random contiguous segment (SpecAugment-style on waveform).

    This regularization technique forces the model to not rely on any single
    temporal region, improving generalization.
    """
    mask_len = int(len(y) * np.random.uniform(0.05, max_mask_fraction))
    start = np.random.randint(0, max(1, len(y) - mask_len))
    y_masked = y.copy()
    y_masked[start:start + mask_len] = 0.0
    return y_masked


def augment_combined(y, sr):
    """Apply a random combination of augmentations.

    Each augmentation is applied with independent probability, creating diverse
    training conditions that improve robustness to unseen distortions.
    """
    if np.random.random() < 0.5:
        y = augment_add_noise(y)
    if np.random.random() < 0.5:
        y = augment_bandlimit(y, sr)
    if np.random.random() < 0.5:
        y = augment_gain(y)
    if np.random.random() < 0.3:
        y = augment_time_stretch(y)
    if np.random.random() < 0.3:
        y = augment_pitch_shift(y, sr)
    return y