"""
ml/preprocessing.py
Audio preprocessing module for UrbanSound8K and inference recordings.
Performs mono conversion, resampling to 22,050 Hz, fixed-duration padding/cropping
to 4.0 seconds (88,200 samples), and peak amplitude normalization.
"""

import sys
from math import gcd
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import scipy.signal
import scipy.io.wavfile

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.config import AudioConfig

# Optional backends
try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except ImportError:
    sf = None
    _HAS_SOUNDFILE = False

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    librosa = None
    _HAS_LIBROSA = False


def load_audio(
    file_path: Union[str, Path], target_sr: int = AudioConfig.sample_rate
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file into a float32 numpy array with amplitude in [-1.0, 1.0].
    Attempts soundfile -> librosa -> scipy.io.wavfile in sequence.
    
    Args:
        file_path: Path to the audio file.
        target_sr: Desired sampling rate (used by librosa if loaded via librosa).
        
    Returns:
        Tuple of (waveform: np.ndarray, sample_rate: int)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {path}")

    # 1. Try soundfile
    if _HAS_SOUNDFILE:
        try:
            data, sr = sf.read(str(path), dtype="float32")
            return data, sr
        except Exception:
            pass

    # 2. Try librosa
    if _HAS_LIBROSA:
        try:
            data, sr = librosa.load(str(path), sr=None, mono=False)
            return data.astype(np.float32), sr
        except Exception:
            pass

    # 3. Fallback to scipy.io.wavfile
    try:
        sr, data = scipy.io.wavfile.read(str(path))
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128.0) / 128.0
        else:
            data = data.astype(np.float32)
        return data, sr
    except Exception as exc:
        raise RuntimeError(f"Failed to load audio from {path}: {exc}") from exc


def to_mono(waveform: np.ndarray) -> np.ndarray:
    """
    Convert a multi-channel waveform array to a 1D mono waveform.
    Handles shapes (samples, channels) and (channels, samples).
    """
    if waveform.ndim == 1:
        return waveform

    if waveform.ndim == 2:
        # If shape is (channels, samples) with channels <= 8 and samples > 8
        if waveform.shape[0] < waveform.shape[1] and waveform.shape[0] <= 8:
            return np.mean(waveform, axis=0)
        # If shape is (samples, channels)
        return np.mean(waveform, axis=1)

    raise ValueError(f"Unexpected waveform shape: {waveform.shape}")


def resample_audio(
    waveform: np.ndarray,
    orig_sr: int,
    target_sr: int = AudioConfig.sample_rate,
) -> np.ndarray:
    """
    Resample waveform to target_sr using polyphase filtering.
    """
    if orig_sr == target_sr:
        return waveform

    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(f"Sample rates must be positive. Got orig={orig_sr}, target={target_sr}")

    # Calculate coprime rational factor
    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g

    resampled = scipy.signal.resample_poly(waveform, up, down).astype(np.float32)
    return resampled


def pad_or_crop(
    waveform: np.ndarray,
    target_samples: int = AudioConfig.target_samples,
    mode: str = "center",
) -> np.ndarray:
    """
    Adjust audio length to exactly target_samples.
    - Shorter recordings are padded symmetrically with zeros.
    - Longer recordings are cropped to the center window.
    
    Args:
        waveform: 1D audio array.
        target_samples: Expected length in samples (default: 88,200 = 4.0s @ 22,050 Hz).
        mode: Alignment mode ("center" or "start").
        
    Returns:
        1D array of exact length target_samples.
    """
    n_samples = len(waveform)

    if n_samples == target_samples:
        return waveform.astype(np.float32)

    if n_samples < target_samples:
        diff = target_samples - n_samples
        if mode == "center":
            left = diff // 2
            right = diff - left
            return np.pad(waveform, (left, right), mode="constant").astype(np.float32)
        else:
            return np.pad(waveform, (0, diff), mode="constant").astype(np.float32)

    # If longer, crop
    if mode == "center":
        start = (n_samples - target_samples) // 2
        return waveform[start : start + target_samples].astype(np.float32)
    else:
        return waveform[:target_samples].astype(np.float32)


def normalize_audio(
    waveform: np.ndarray,
    target_peak: float = 0.95,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Peak-normalize waveform amplitude to target_peak.
    Prevents digital clipping and maintains consistent dynamic range.
    """
    peak = np.max(np.abs(waveform))
    if peak > eps:
        return (waveform * (target_peak / peak)).astype(np.float32)
    return waveform.astype(np.float32)


def preprocess_waveform(
    input_data: Union[str, Path, np.ndarray],
    orig_sr: Optional[int] = None,
    config: Optional[AudioConfig] = None,
) -> np.ndarray:
    """
    Complete end-to-end audio preprocessing pipeline:
    1. Load audio (if file path is provided)
    2. Convert to mono
    3. Resample to target sample rate (22,050 Hz)
    4. Pad or crop to exact fixed duration (4.0s = 88,200 samples)
    5. Peak normalize amplitude to 0.95
    
    Returns:
        Preprocessed 1D float32 numpy array with shape (88200,).
    """
    cfg = config or AudioConfig()

    if isinstance(input_data, (str, Path)):
        waveform, sr = load_audio(input_data)
    else:
        waveform = input_data
        sr = orig_sr or cfg.sample_rate

    mono = to_mono(waveform)
    resampled = resample_audio(mono, orig_sr=sr, target_sr=cfg.sample_rate)
    padded = pad_or_crop(resampled, target_samples=cfg.target_samples, mode="center")
    normalized = normalize_audio(padded, target_peak=0.95)

    return normalized


def generate_synthetic_tone(
    freq: float = 440.0,
    duration: float = 2.5,
    sr: int = AudioConfig.sample_rate,
) -> np.ndarray:
    """
    Generate a pure synthetic sine tone for testing pipelines without external audio files.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * freq * t)


if __name__ == "__main__":
    print("Testing audio preprocessing pipeline...")
    cfg = AudioConfig()

    # 1. Test short audio (2.5s) padding
    short_tone = generate_synthetic_tone(freq=440.0, duration=2.5, sr=cfg.sample_rate)
    proc_short = preprocess_waveform(short_tone, orig_sr=cfg.sample_rate, config=cfg)
    assert proc_short.shape == (cfg.target_samples,), f"Expected shape {(cfg.target_samples,)}, got {proc_short.shape}"
    assert np.max(np.abs(proc_short)) <= 0.95 + 1e-5, "Normalization exceeded target peak"
    print(f"Short audio test passed: input 2.5s -> padded to shape {proc_short.shape}")

    # 2. Test long audio (6.0s) cropping
    long_tone = generate_synthetic_tone(freq=880.0, duration=6.0, sr=cfg.sample_rate)
    proc_long = preprocess_waveform(long_tone, orig_sr=cfg.sample_rate, config=cfg)
    assert proc_long.shape == (cfg.target_samples,), f"Expected shape {(cfg.target_samples,)}, got {proc_long.shape}"
    print(f"Long audio test passed: input 6.0s -> cropped to shape {proc_long.shape}")

    # 3. Test stereo downmixing
    stereo_tone = np.column_stack([short_tone, short_tone * 0.8])
    proc_stereo = preprocess_waveform(stereo_tone, orig_sr=cfg.sample_rate, config=cfg)
    assert proc_stereo.shape == (cfg.target_samples,), f"Expected shape {(cfg.target_samples,)}, got {proc_stereo.shape}"
    print(f"Stereo audio test passed: input shape {stereo_tone.shape} -> mono shape {proc_stereo.shape}")

    # 4. Test resampling from 44,100 Hz to 22,050 Hz
    tone_44k = generate_synthetic_tone(freq=440.0, duration=3.0, sr=44100)
    proc_resampled = preprocess_waveform(tone_44k, orig_sr=44100, config=cfg)
    assert proc_resampled.shape == (cfg.target_samples,), f"Expected shape {(cfg.target_samples,)}, got {proc_resampled.shape}"
    print(f"Resampling test passed: 44.1kHz -> 22.05kHz, shape {proc_resampled.shape}")

    print("All preprocessing tests passed successfully!")
