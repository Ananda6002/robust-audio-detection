"""
ml/augmentation.py
Controlled noise augmentation and synthetic degradation engine.
Generates white, pink, brown, and ambient background noise, injecting them at exact
Signal-to-Noise Ratios (SNRs: 20, 10, 5, 0 dB) for robustness benchmarking and training defense.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.config import AudioConfig, NoiseConfig
from ml.preprocessing import normalize_audio


def compute_signal_power(waveform: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute average signal power: P = (1/N) * sum(x^2).
    """
    if len(waveform) == 0:
        return 0.0
    return float(np.mean(waveform.astype(np.float64) ** 2) + eps)


def measure_snr_db(
    clean_signal: np.ndarray,
    noisy_signal: np.ndarray,
    noise: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> float:
    """
    Calculate the actual Signal-to-Noise Ratio (dB) between clean and noisy signals.
    If noise is explicitly passed, accounts for any proportional headroom attenuation.
    SNR_dB = 10 * log10(P_clean / P_noise)
    """
    if noise is not None:
        eff_clean = (noisy_signal - noise).astype(np.float64)
        n = noise.astype(np.float64)
    else:
        n = (noisy_signal - clean_signal).astype(np.float64)
        eff_clean = clean_signal.astype(np.float64)

    p_clean = compute_signal_power(eff_clean, eps)
    p_noise = compute_signal_power(n, eps)
    return float(10.0 * np.log10(p_clean / p_noise))


def generate_colored_noise(
    length: int,
    noise_type: str = "pink",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate synthetic stationary acoustic noise with controlled spectral profiles.
    
    Args:
        length: Number of audio samples to generate.
        noise_type:
            - 'white': Flat spectrum (equal power across all frequencies).
            - 'pink': 1/f power spectrum (natural ambient/environmental roll-off).
            - 'brown': 1/f^2 power spectrum (low-frequency urban engine/traffic rumble).
            - 'ambient': Combined low-frequency ambient profile (60% pink + 40% brown).
        rng: Optional NumPy random generator for deterministic seeds.
        
    Returns:
        1D float32 array of shape (length,).
    """
    generator = rng if rng is not None else np.random.default_rng()

    if noise_type == "white":
        return generator.standard_normal(length).astype(np.float32)

    # Spectral filtering approach for 1/f^alpha noise
    # Compute FFT frequencies
    white = generator.standard_normal(length).astype(np.float64)
    fft_vals = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(length)
    freqs[0] = 1.0  # Avoid division by zero at DC component

    if noise_type == "pink":
        # Pink noise power decays as 1/f -> amplitude decays as 1/sqrt(f)
        scale = 1.0 / np.sqrt(freqs)
    elif noise_type == "brown":
        # Brown noise power decays as 1/f^2 -> amplitude decays as 1/f
        scale = 1.0 / freqs
    elif noise_type == "ambient":
        # Blend of pink (mid) and brown (low rumbling)
        scale = (0.6 / np.sqrt(freqs)) + (0.4 / freqs)
    else:
        raise ValueError(f"Unsupported noise_type: '{noise_type}'. Choose from 'white', 'pink', 'brown', 'ambient'.")

    scale[0] = 0.0  # Zero out DC offset
    filtered = np.fft.irfft(fft_vals * scale, n=length)
    # Zero-mean and standardize to unit variance
    filtered -= np.mean(filtered)
    std = np.std(filtered)
    if std > 1e-8:
        filtered /= std

    return filtered.astype(np.float32)


def inject_noise_at_snr(
    signal: np.ndarray,
    snr_db: float,
    noise_type: str = "pink",
    noise_audio: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    target_peak: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Additively mix noise into a clean audio signal at an exact target Signal-to-Noise Ratio (dB).
    
    Formula:
        gamma = 10^(SNR_dB / 10)
        scale = sqrt( P_signal / (P_noise * gamma) )
        noisy = signal + scale * noise
        
    Args:
        signal: Clean 1D audio waveform.
        snr_db: Target SNR in decibels (e.g., 20.0, 10.0, 5.0, 0.0).
        noise_type: Synthetic noise color ('white', 'pink', 'brown', 'ambient').
        noise_audio: Optional external noise recording to scale instead of synthesis.
        rng: Optional random generator.
        target_peak: Maximum absolute amplitude permitted.
        
    Returns:
        Tuple of (mixed_noisy_signal, scaled_additive_noise).
    """
    sig = signal.astype(np.float64)
    sig_len = len(sig)

    p_signal = compute_signal_power(sig)
    if p_signal < 1e-10:
        # Near silent signal: return unchanged
        zeros = np.zeros(sig_len, dtype=np.float32)
        return signal.astype(np.float32), zeros

    # 1. Acquire or generate noise
    if noise_audio is not None and len(noise_audio) > 0:
        if len(noise_audio) < sig_len:
            reps = int(np.ceil(sig_len / len(noise_audio)))
            noise = np.tile(noise_audio, reps)[:sig_len].astype(np.float64)
        else:
            # Deterministic/random slice
            generator = rng if rng is not None else np.random.default_rng()
            max_start = len(noise_audio) - sig_len
            start = generator.integers(0, max_start + 1) if max_start > 0 else 0
            noise = noise_audio[start : start + sig_len].astype(np.float64)
    else:
        noise = generate_colored_noise(sig_len, noise_type=noise_type, rng=rng).astype(np.float64)

    p_noise = compute_signal_power(noise)

    # 2. Compute exact scaling factor for desired SNR
    gamma = 10.0 ** (snr_db / 10.0)
    scale = np.sqrt(p_signal / (p_noise * gamma))
    scaled_noise = noise * scale

    # 3. Additive mixing
    mixed = sig + scaled_noise

    # 4. Proportional headroom scaling if mixing exceeds target_peak
    max_val = np.max(np.abs(mixed))
    if max_val > target_peak and max_val > 0:
        attenuation = target_peak / max_val
        mixed = mixed * attenuation
        scaled_noise = scaled_noise * attenuation

    return mixed.astype(np.float32), scaled_noise.astype(np.float32)


class NoiseAugmentationTransform:
    """
    Callable stochastic audio transform for training-time noise augmentation (Experiment B).
    Injects noise with probability `prob` at uniformly sampled SNRs in [snr_min, snr_max].
    Preserves clean inputs during evaluation.
    """

    def __init__(
        self,
        prob: float = NoiseConfig.train_augmentation_prob,
        snr_min: float = NoiseConfig.train_snr_min,
        snr_max: float = NoiseConfig.train_snr_max,
        noise_types: Tuple[str, ...] = ("pink", "brown", "ambient", "white"),
        seed: Optional[int] = None,
    ):
        self.prob = prob
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.noise_types = noise_types
        self.rng = np.random.default_rng(seed)

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        if self.rng.random() < self.prob:
            snr = float(self.rng.uniform(self.snr_min, self.snr_max))
            noise_type = str(self.rng.choice(self.noise_types))
            noisy_wave, _ = inject_noise_at_snr(
                waveform,
                snr_db=snr,
                noise_type=noise_type,
                rng=self.rng,
            )
            return noisy_wave
        return waveform


if __name__ == "__main__":
    print("Testing noise augmentation and SNR injection engine...")
    from ml.preprocessing import generate_synthetic_tone

    sr = AudioConfig.sample_rate
    clean = generate_synthetic_tone(freq=500.0, duration=4.0, sr=sr)
    rng = np.random.default_rng(42)

    # Test all 4 benchmark SNRs across multiple noise colors
    test_snrs = [20.0, 10.0, 5.0, 0.0]
    test_colors = ["white", "pink", "brown", "ambient"]

    print("\n--- Controlled SNR Verification Table ---")
    print(f"{'Color':<10} | {'Target SNR (dB)':<16} | {'Measured SNR (dB)':<18} | {'Error (dB)':<10}")
    print("-" * 62)

    for color in test_colors:
        for target_snr in test_snrs:
            noisy, noise = inject_noise_at_snr(clean, snr_db=target_snr, noise_type=color, rng=rng)
            measured_snr = measure_snr_db(clean, noisy, noise=noise)
            diff = abs(measured_snr - target_snr)
            print(f"{color:<10} | {target_snr:>14.1f}   | {measured_snr:>16.2f}   | {diff:>8.3f}")
            assert diff < 0.5, f"SNR mismatch for {color} at {target_snr} dB! Got {measured_snr}"
            assert np.max(np.abs(noisy)) <= 0.95 + 1e-4, "Peak threshold exceeded"

    # Test Training Transform
    augmentor = NoiseAugmentationTransform(prob=1.0, snr_min=10.0, snr_max=20.0, seed=123)
    augmented = augmentor(clean)
    aug_snr = measure_snr_db(clean, augmented)
    print(f"\nTraining Augmentation Transform output: Measured SNR = {aug_snr:.2f} dB (within [10, 20] dB range)")
    assert 9.5 <= aug_snr <= 20.5, f"Augmented SNR outside bounds: {aug_snr}"

    print("\nAll noise augmentation and SNR verification checks passed successfully!")
