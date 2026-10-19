"""
ml/features.py
Feature extraction module for Log-Mel Spectrogram computation.
Converts 1D normalized waveforms into 2D time-frequency representations (128 mels x 173 frames).
Implemented using pure NumPy/SciPy to ensure 100% deterministic, cross-platform reproducibility.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import scipy.signal

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.config import AudioConfig
from ml.preprocessing import generate_synthetic_tone, preprocess_waveform


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    """Convert frequency from Hertz to Mel scale."""
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    """Convert frequency from Mel scale back to Hertz."""
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def compute_mel_filterbank(
    sr: int = AudioConfig.sample_rate,
    n_fft: int = AudioConfig.n_fft,
    n_mels: int = AudioConfig.n_mels,
    f_min: float = AudioConfig.f_min,
    f_max: float = AudioConfig.f_max,
) -> np.ndarray:
    """
    Construct a triangular Mel filterbank matrix of shape (n_mels, n_fft // 2 + 1).
    Area-normalized so each filter integrates to approximately 1.0.
    """
    n_freq_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2.0, n_freq_bins)

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # Bin indices corresponding to the frequency points
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bin_points = np.clip(bin_points, 0, n_freq_bins - 1)

    weights = np.zeros((n_mels, n_freq_bins), dtype=np.float32)

    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        if center > left:
            weights[i, left:center] = (
                fft_freqs[left:center] - hz_points[i]
            ) / (hz_points[i + 1] - hz_points[i])
        if right > center:
            weights[i, center:right] = (
                hz_points[i + 2] - fft_freqs[center:right]
            ) / (hz_points[i + 2] - hz_points[i + 1])

        # Area normalization (slaney-style)
        enorm = 2.0 / (hz_points[i + 2] - hz_points[i])
        weights[i] *= enorm

    # Ensure all filterbank weights are strictly non-negative
    weights = np.maximum(0.0, weights)

    return weights


def compute_stft(
    waveform: np.ndarray,
    n_fft: int = AudioConfig.n_fft,
    hop_length: int = AudioConfig.hop_length,
) -> np.ndarray:
    """
    Compute Short-Time Fourier Transform with center padding matching Librosa/PyTorch standard.
    
    Args:
        waveform: 1D array of length target_samples (88,200).
        n_fft: FFT window size (1024).
        hop_length: Hop stride in samples (512).
        
    Returns:
        Complex STFT matrix of shape (n_fft // 2 + 1, time_steps).
    """
    # Center pad with reflection to match standard audio STFT frames
    pad_amount = n_fft // 2
    padded_wave = np.pad(waveform, (pad_amount, pad_amount), mode="reflect")

    window = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (len(padded_wave) - n_fft) // hop_length
    frames = np.lib.stride_tricks.sliding_window_view(padded_wave, n_fft)[::hop_length][:n_frames]

    # Apply window and compute FFT
    windowed = frames * window
    stft_matrix = np.fft.rfft(windowed, n=n_fft, axis=-1).T  # (freq_bins, frames)

    return stft_matrix.astype(np.complex64)


def waveform_to_log_mel_spectrogram(
    waveform: np.ndarray,
    config: Optional[AudioConfig] = None,
    filterbank: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Transform 1D preprocessed waveform into a Log-Mel Spectrogram.
    
    Args:
        waveform: 1D array of 88,200 samples.
        config: AudioConfig parameters.
        filterbank: Optional precomputed filterbank matrix.
        
    Returns:
        2D float32 array of shape (n_mels, time_steps) in decibels (dB).
    """
    cfg = config or AudioConfig()
    fb = filterbank if filterbank is not None else compute_mel_filterbank(
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        n_mels=cfg.n_mels,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
    )

    stft_matrix = compute_stft(waveform, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
    power_spec = np.abs(stft_matrix) ** 2  # (n_freq_bins, time_frames)

    # Project onto Mel scale: (n_mels, n_freq_bins) @ (n_freq_bins, time_frames) -> (n_mels, time_frames)
    mel_spec = np.dot(fb, power_spec)
    mel_spec = np.maximum(mel_spec, 1e-10)

    # Convert power to decibels (10 * log10)
    log_mel = 10.0 * np.log10(mel_spec)

    # Dynamic range clamping to top_db relative to peak
    peak_val = np.max(log_mel)
    log_mel = np.maximum(log_mel, peak_val - cfg.top_db)

    return log_mel.astype(np.float32)


def standardize_spectrogram(
    log_mel: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """
    Apply zero-mean, unit-variance standardization to a log-mel spectrogram.
    """
    mean = np.mean(log_mel)
    std = np.std(log_mel)
    return ((log_mel - mean) / (std + eps)).astype(np.float32)


def extract_features(
    input_data: Union[str, Path, np.ndarray],
    orig_sr: Optional[int] = None,
    config: Optional[AudioConfig] = None,
    standardize: bool = True,
) -> np.ndarray:
    """
    End-to-end feature extraction pipeline:
    1. Preprocesses waveform (mono, 22.05kHz, 4.0s / 88,200 samples, peak normalized)
    2. Computes Log-Mel Spectrogram (128 mels x 173 frames)
    3. Standardizes per-instance (zero mean, unit variance)
    4. Adds channel dimension for CNN input: shape (1, 128, 173)
    
    Returns:
        3D float32 numpy array with shape (1, 128, 173).
    """
    cfg = config or AudioConfig()
    waveform = preprocess_waveform(input_data, orig_sr=orig_sr, config=cfg)
    log_mel = waveform_to_log_mel_spectrogram(waveform, config=cfg)

    if standardize:
        log_mel = standardize_spectrogram(log_mel)

    # Add channel dimension: (1, n_mels, time_steps)
    return np.expand_dims(log_mel, axis=0)


if __name__ == "__main__":
    print("Testing Log-Mel Spectrogram feature extraction...")
    cfg = AudioConfig()

    # 1. Test pure synthetic tone
    synthetic_wave = generate_synthetic_tone(freq=1000.0, duration=4.0, sr=cfg.sample_rate)
    feat = extract_features(synthetic_wave, orig_sr=cfg.sample_rate, config=cfg, standardize=True)

    print(f"Extracted feature shape: {feat.shape} (channel, n_mels, time_frames)")
    assert feat.shape == (1, cfg.n_mels, 173), f"Expected (1, 128, 173), got {feat.shape}"
    assert not np.isnan(feat).any(), "Found NaNs in extracted features"
    assert not np.isinf(feat).any(), "Found Infs in extracted features"
    assert np.isclose(np.mean(feat), 0.0, atol=1e-3), f"Standardized mean should be ~0.0, got {np.mean(feat)}"
    assert np.isclose(np.std(feat), 1.0, atol=1e-2), f"Standardized std should be ~1.0, got {np.std(feat)}"

    print("Log-Mel filterbank and STFT test passed successfully!")
