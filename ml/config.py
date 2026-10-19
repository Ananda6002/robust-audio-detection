"""
ml/config.py
Single source of truth for audio parameters, dataset paths, class labels,
fold assignments, training hyperparameters, and noise robustness settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# Project root path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "UrbanSound8K"
AUDIO_DIR = DATA_DIR / "audio"
METADATA_PATH = DATA_DIR / "metadata" / "UrbanSound8K.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# UrbanSound8K 10-Class Taxonomy
CLASS_MAP: Dict[int, str] = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music",
}

CLASS_NAMES: List[str] = [CLASS_MAP[i] for i in range(len(CLASS_MAP))]
NUM_CLASSES = len(CLASS_NAMES)


@dataclass(frozen=True)
class AudioConfig:
    """Standardized audio processing specifications."""
    sample_rate: int = 22050
    duration_seconds: float = 4.0
    target_samples: int = 88200  # sample_rate * duration_seconds (22050 * 4)
    channels: int = 1  # Mono
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 512
    f_min: float = 20.0
    f_max: float = 11025.0  # Nyquist frequency for 22050 Hz
    top_db: float = 80.0


@dataclass(frozen=True)
class ModelConfig:
    """Convolutional Neural Network architecture hyperparameters."""
    in_channels: int = 1
    num_classes: int = 10
    base_filters: int = 32
    dropout_conv: float = 0.2
    dropout_fc: float = 0.4
    fc_hidden_dim: int = 256


@dataclass(frozen=True)
class TrainingConfig:
    """Training, cross-validation, and optimization hyperparameters."""
    seed: int = 42
    batch_size: int = 32
    epochs: int = 35
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 8
    # Predefined UrbanSound8K fold protocol: folds 1-8 train, fold 9 val, fold 10 test
    train_folds: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    val_fold: int = 9
    test_fold: int = 10
    num_workers: int = 0  # 0 for Windows multiprocessing stability


@dataclass(frozen=True)
class NoiseConfig:
    """Noise augmentation and robustness benchmarking configurations."""
    # Signal-to-Noise Ratios (dB) evaluated in Experiment B and C
    evaluation_snrs: Tuple[float, ...] = (20.0, 10.0, 5.0, 0.0)
    # Training augmentation parameters
    train_snr_min: float = 5.0
    train_snr_max: float = 25.0
    train_augmentation_prob: float = 0.5


@dataclass(frozen=True)
class ConfidenceConfig:
    """Uncertainty quantification and confidence-aware selective classification."""
    default_threshold: float = 0.70
    uncertain_label: str = "Uncertain — manual review recommended"
    candidate_thresholds: Tuple[float, ...] = (
        0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
    )


def validate_dataset_structure(data_dir: Path | None = None) -> Dict[str, object]:
    """
    Validates whether the UrbanSound8K directory structure and metadata exist.
    Returns a diagnostic report dictionary with actionable guidance if incomplete.
    """
    base_dir = data_dir or DATA_DIR
    meta_path = base_dir / "metadata" / "UrbanSound8K.csv"
    audio_path = base_dir / "audio"

    missing_items: List[str] = []

    if not base_dir.exists():
        missing_items.append(f"Base dataset directory missing: {base_dir}")
    if not meta_path.exists():
        missing_items.append(f"Metadata CSV missing: {meta_path}")
    if not audio_path.exists():
        missing_items.append(f"Audio directory missing: {audio_path}")

    existing_folds: List[int] = []
    missing_folds: List[int] = []
    if audio_path.exists():
        for fold_idx in range(1, 11):
            fold_dir = audio_path / f"fold{fold_idx}"
            if fold_dir.is_dir():
                existing_folds.append(fold_idx)
            else:
                missing_folds.append(fold_idx)
    else:
        missing_folds = list(range(1, 11))

    is_valid = len(missing_items) == 0 and len(missing_folds) == 0

    return {
        "is_valid": is_valid,
        "base_dir": str(base_dir),
        "metadata_path": str(meta_path),
        "metadata_exists": meta_path.exists(),
        "audio_dir_exists": audio_path.exists(),
        "existing_folds": existing_folds,
        "missing_folds": missing_folds,
        "missing_items": missing_items,
        "instructions": (
            "Download UrbanSound8K from https://urbansounddataset.weebly.com/urbansound8k.html "
            f"and place it inside: {base_dir} with 'metadata/UrbanSound8K.csv' and 'audio/fold[1-10]/'."
            if not is_valid
            else "Dataset structure verified successfully."
        ),
    }


if __name__ == "__main__":
    report = validate_dataset_structure()
    print("--- UrbanSound8K Dataset Structure Verification ---")
    for k, v in report.items():
        print(f"{k}: {v}")
