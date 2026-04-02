5.1 Preprocessing

The raw audio signals were first preprocessed before feature extraction. The preprocessing steps include:

loading audio into a consistent format
resampling where needed
converting to mono if required
normalizing waveform amplitude
ensuring fixed clip duration

These steps help ensure that the extracted features are consistent across all training and test samples.

5.2 DSP Feature Extraction

Several DSP-based audio features were extracted to represent the acoustic properties of each sound clip.

MFCC (Mel-Frequency Cepstral Coefficients)

MFCCs were used as the core feature representation because they capture the timbral characteristics of sound in a compact and effective way. They are commonly used in audio classification tasks and are robust for many environmental sound categories.

Delta and Delta-Delta MFCC

To capture short-term temporal changes in the signal, delta and delta-delta MFCCs were included. These represent the first and second order changes of MFCC features and help model dynamic patterns over time.

Spectral Centroid

The spectral centroid measures the “center of mass” of the frequency spectrum. It helps distinguish sounds with brighter frequency content from darker or lower-frequency sounds.

Spectral Bandwidth

This feature measures how spread out the spectral energy is. It is useful for identifying sounds that have narrow or broad frequency distributions.

Spectral Rolloff

Spectral rolloff indicates the frequency below which a fixed percentage of the spectral energy is contained. This helps describe the high-frequency distribution of the sound.

Zero Crossing Rate

Zero crossing rate measures how often the waveform changes sign. It is useful for distinguishing noisy or percussive sounds from more tonal sounds.

Chroma / Mel Spectral Features (if used)

Additional spectral representations such as chroma or mel-spectrogram statistics were explored to improve robustness and capture richer harmonic or frequency-based information.

5.3 Feature Aggregation

Since each audio clip is a time-varying signal, frame-level features were aggregated into clip-level statistics such as:

mean
standard deviation
minimum
maximum

This converts variable-length frame information into a fixed-length feature vector suitable for machine learning.

5.4 Robustness Strategy

To improve performance under noisy and band-limited conditions, the following robustness strategies were considered:

using multiple complementary DSP features
normalization of features
augmentation with noise
feature selection / combination experiments
model comparison under different feature sets

This was important because the test set explicitly evaluates clean, noisy, and band-limited signals.
