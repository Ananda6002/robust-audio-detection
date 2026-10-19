"""
tests/test_data_pipeline.py
Automated unit and regression test suite for Phase 2 data pipeline components.
Tests metadata validation, fold segregation, leakage prevention, audio preprocessing,
and log-mel spectrogram extraction.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.config import AudioConfig, TrainingConfig
from ml.dataset import UrbanSound8KDataset, create_dataloaders, verify_no_leakage
from ml.features import (
    compute_mel_filterbank,
    compute_stft,
    extract_features,
    standardize_spectrogram,
    waveform_to_log_mel_spectrogram,
)
from ml.metadata import (
    create_synthetic_metadata_for_testing,
    get_class_distribution,
    get_fold_split,
    load_metadata,
)
from ml.preprocessing import (
    generate_synthetic_tone,
    normalize_audio,
    pad_or_crop,
    preprocess_waveform,
    resample_audio,
    to_mono,
)


# ==========================================
# 1. Metadata & Leakage Prevention Tests
# ==========================================

def test_metadata_synthetic_schema():
    """Verify synthetic metadata adheres to required UrbanSound8K columns and ranges."""
    df = create_synthetic_metadata_for_testing(num_samples=30)
    assert len(df) == 30
    assert "slice_file_name" in df.columns
    assert "fold" in df.columns
    assert "classID" in df.columns
    assert "fsID" in df.columns
    assert df["fold"].between(1, 10).all()
    assert df["classID"].between(0, 9).all()


def test_fold_split_zero_leakage():
    """Verify that get_fold_split produces 100% disjoint recording sources and slices."""
    df = create_synthetic_metadata_for_testing(num_samples=50)
    train_df, val_df, test_df = get_fold_split(df, train_folds=(1, 2, 3, 4, 5, 6, 7, 8), val_fold=9, test_fold=10)

    # Check fold IDs
    assert set(train_df["fold"].unique()).issubset(set(range(1, 9)))
    assert set(val_df["fold"].unique()) == {9}
    assert set(test_df["fold"].unique()) == {10}

    # Verify zero slice overlap
    train_slices = set(train_df["slice_file_name"])
    val_slices = set(val_df["slice_file_name"])
    test_slices = set(test_df["slice_file_name"])
    assert len(train_slices.intersection(val_slices)) == 0
    assert len(train_slices.intersection(test_slices)) == 0
    assert len(val_slices.intersection(test_slices)) == 0

    # Verify zero recording source (fsID) overlap
    train_fs = set(train_df["fsID"])
    val_fs = set(val_df["fsID"])
    test_fs = set(test_df["fsID"])
    assert len(train_fs.intersection(val_fs)) == 0
    assert len(train_fs.intersection(test_fs)) == 0
    assert len(val_fs.intersection(test_fs)) == 0


def test_get_class_distribution():
    """Verify class distribution returns counts for all 10 classes."""
    df = create_synthetic_metadata_for_testing(num_samples=20)
    dist = get_class_distribution(df)
    assert len(dist) == 10
    assert sum(dist.values()) == 20


# ==========================================
# 2. Audio Preprocessing Unit Tests
# ==========================================

def test_to_mono_conversions():
    """Verify stereo-to-mono downmixing across channel orientations."""
    # 1D already mono
    mono_in = np.ones(100, dtype=np.float32)
    assert np.array_equal(to_mono(mono_in), mono_in)

    # 2D (samples, 2)
    stereo_cols = np.ones((100, 2), dtype=np.float32)
    stereo_cols[:, 1] = 3.0
    mono_out = to_mono(stereo_cols)
    assert mono_out.shape == (100,)
    assert np.allclose(mono_out, 2.0)

    # 2D (2, samples)
    stereo_rows = np.ones((2, 100), dtype=np.float32)
    stereo_rows[1, :] = 5.0
    mono_rows_out = to_mono(stereo_rows)
    assert mono_rows_out.shape == (100,)
    assert np.allclose(mono_rows_out, 3.0)


def test_pad_or_crop_durations():
    """Verify fixed-duration alignment to exact target sample count (88,200)."""
    target = AudioConfig.target_samples  # 88,200

    # Shorter clip (40,000 samples) -> symmetrically zero-padded
    short_clip = np.ones(40000, dtype=np.float32)
    padded = pad_or_crop(short_clip, target_samples=target, mode="center")
    assert padded.shape == (target,)
    assert padded[0] == 0.0
    assert padded[-1] == 0.0

    # Longer clip (120,000 samples) -> center-cropped
    long_clip = np.arange(120000, dtype=np.float32)
    cropped = pad_or_crop(long_clip, target_samples=target, mode="center")
    assert cropped.shape == (target,)
    expected_start = (120000 - target) // 2
    assert cropped[0] == expected_start


def test_resample_audio():
    """Verify polyphase resampling converts audio sampling rate accurately."""
    sr_orig = 44100
    sr_target = 22050
    duration = 2.0
    tone_44k = generate_synthetic_tone(freq=440.0, duration=duration, sr=sr_orig)
    assert len(tone_44k) == int(sr_orig * duration)

    resampled = resample_audio(tone_44k, orig_sr=sr_orig, target_sr=sr_target)
    assert len(resampled) == int(sr_target * duration)


def test_normalize_audio_peak():
    """Verify peak normalization constrains amplitude to target threshold."""
    wave = np.array([-2.5, 0.5, 3.0, -1.0], dtype=np.float32)
    normalized = normalize_audio(wave, target_peak=0.95)
    assert np.isclose(np.max(np.abs(normalized)), 0.95)


def test_preprocess_waveform_composite():
    """Verify end-to-end preprocessing produces guaranteed (88200,) float32 array."""
    cfg = AudioConfig()
    tone = generate_synthetic_tone(freq=500.0, duration=1.8, sr=cfg.sample_rate)
    processed = preprocess_waveform(tone, orig_sr=cfg.sample_rate, config=cfg)
    assert processed.shape == (cfg.target_samples,)
    assert processed.dtype == np.float32
    assert np.max(np.abs(processed)) <= 0.95 + 1e-5


# ==========================================
# 3. Feature Extraction Unit Tests
# ==========================================

def test_mel_filterbank_dimensions():
    """Verify triangular Mel filterbank matrix shape and non-negativity."""
    cfg = AudioConfig()
    fb = compute_mel_filterbank(
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        n_mels=cfg.n_mels,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
    )
    expected_freq_bins = cfg.n_fft // 2 + 1  # 513
    assert fb.shape == (cfg.n_mels, expected_freq_bins)
    assert (fb >= 0.0).all(), "Filterbank weights must be non-negative"


def test_extract_features_shape_and_stability():
    """Verify log-mel spectrogram produces shape (1, 128, 173) with zero NaNs/Infs."""
    cfg = AudioConfig()
    tone = generate_synthetic_tone(freq=600.0, duration=4.0, sr=cfg.sample_rate)
    feat = extract_features(tone, orig_sr=cfg.sample_rate, config=cfg, standardize=True)

    assert feat.shape == (1, cfg.n_mels, 173)
    assert not np.isnan(feat).any()
    assert not np.isinf(feat).any()
    # Check standardization properties
    assert np.isclose(np.mean(feat), 0.0, atol=1e-3)
    assert np.isclose(np.std(feat), 1.0, atol=1e-2)


# ==========================================
# 4. Dataset & DataLoader Contracts
# ==========================================

def test_dataset_and_dataloader_integration():
    """Verify PyTorch DataLoader yields properly formatted batches and verifies no leakage."""
    synthetic_df = create_synthetic_metadata_for_testing(num_samples=30)
    train_loader, val_loader, test_loader = create_dataloaders(
        metadata_df=synthetic_df,
        batch_size=4,
        num_workers=0,
        allow_synthetic_fallback=True,
    )

    batch_x, batch_y, batch_meta = next(iter(train_loader))
    assert batch_x.shape == (4, 1, 128, 173)
    assert batch_y.shape == (4,)
    assert batch_x.dtype == torch.float32
    assert batch_y.dtype == torch.int64
    assert len(batch_meta["slice_file_name"]) == 4

    # Run programmatic leakage check
    leakage = verify_no_leakage(train_loader, val_loader, test_loader)
    assert leakage["is_leak_free"] is True
    assert leakage["fs_leakage_detected"] is False
    assert leakage["slice_leakage_detected"] is False
