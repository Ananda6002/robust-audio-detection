"""
ml/dataset.py
PyTorch Dataset and DataLoader abstractions for UrbanSound8K.
Guarantees strict fold segregation, automated leakage verification,
and deterministic fallback for end-to-end smoke testing.
"""

import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.config import AUDIO_DIR, AudioConfig, TrainingConfig
from ml.features import extract_features
from ml.metadata import (
    create_synthetic_metadata_for_testing,
    get_fold_split,
    load_metadata,
    resolve_audio_path,
)
from ml.preprocessing import generate_synthetic_tone, preprocess_waveform


class UrbanSound8KDataset(Dataset):
    """
    PyTorch Dataset for UrbanSound8K audio excerpts.
    Extracts log-mel spectrograms on-the-fly from raw audio or generates
    deterministic test tones if raw audio is not present in local testing.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        audio_dir: Optional[Path] = None,
        waveform_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        feature_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        config: Optional[AudioConfig] = None,
        allow_synthetic_fallback: bool = True,
    ):
        """
        Args:
            metadata: Filtered DataFrame for the assigned fold(s).
            audio_dir: Base directory containing fold1..fold10 subdirectories.
            waveform_transform: Optional callable applied to 1D waveform (e.g. noise augmentation).
            feature_transform: Optional callable applied to extracted tensor.
            config: Audio processing specifications.
            allow_synthetic_fallback: If True, generates a synthetic tone when a file is absent
                                      to enable complete testing before full 6GB download.
        """
        self.metadata = metadata.reset_index(drop=True)
        self.audio_dir = Path(audio_dir) if audio_dir else AUDIO_DIR
        self.waveform_transform = waveform_transform
        self.feature_transform = feature_transform
        self.config = config or AudioConfig()
        self.allow_synthetic_fallback = allow_synthetic_fallback

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        row = self.metadata.iloc[idx]
        file_path = resolve_audio_path(row, audio_dir=self.audio_dir)

        if file_path.exists():
            # Load and preprocess real audio
            waveform = preprocess_waveform(file_path, config=self.config)
        elif self.allow_synthetic_fallback:
            # Deterministic tone based on classID and fsID for testing
            freq = 200.0 + (int(row["classID"]) + 1) * 150.0
            duration = float(min(4.0, max(1.0, row.get("end", 4.0) - row.get("start", 0.0))))
            raw_tone = generate_synthetic_tone(freq=freq, duration=duration, sr=self.config.sample_rate)
            waveform = preprocess_waveform(raw_tone, orig_sr=self.config.sample_rate, config=self.config)
        else:
            raise FileNotFoundError(f"Audio recording not found: {file_path}")

        # Apply waveform-level transform (e.g., additive noise in Experiment B)
        if self.waveform_transform is not None:
            waveform = self.waveform_transform(waveform)

        # Extract standardized Log-Mel Spectrogram of shape (1, 128, 173)
        features_np = extract_features(waveform, orig_sr=self.config.sample_rate, config=self.config)
        features = torch.from_numpy(features_np).float()

        if self.feature_transform is not None:
            features = self.feature_transform(features)

        label = torch.tensor(int(row["classID"]), dtype=torch.long)

        meta_info = {
            "slice_file_name": str(row["slice_file_name"]),
            "fsID": int(row["fsID"]),
            "fold": int(row["fold"]),
            "classID": int(row["classID"]),
            "class": str(row["class"]),
        }

        return features, label, meta_info


def create_dataloaders(
    metadata_df: Optional[pd.DataFrame] = None,
    audio_dir: Optional[Path] = None,
    batch_size: int = TrainingConfig.batch_size,
    train_waveform_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    train_folds: Optional[Tuple[int, ...]] = None,
    val_fold: Optional[int] = None,
    test_fold: Optional[int] = None,
    num_workers: int = TrainingConfig.num_workers,
    allow_synthetic_fallback: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Construct train, validation, and test DataLoaders strictly partitioned by fold.
    """
    if metadata_df is None:
        metadata_df = load_metadata()

    train_df, val_df, test_df = get_fold_split(
        metadata_df,
        train_folds=train_folds,
        val_fold=val_fold,
        test_fold=test_fold,
    )

    train_dataset = UrbanSound8KDataset(
        metadata=train_df,
        audio_dir=audio_dir,
        waveform_transform=train_waveform_transform,
        allow_synthetic_fallback=allow_synthetic_fallback,
    )
    val_dataset = UrbanSound8KDataset(
        metadata=val_df,
        audio_dir=audio_dir,
        waveform_transform=None,  # Validation is always evaluated in original clean state
        allow_synthetic_fallback=allow_synthetic_fallback,
    )
    test_dataset = UrbanSound8KDataset(
        metadata=test_df,
        audio_dir=audio_dir,
        waveform_transform=None,
        allow_synthetic_fallback=allow_synthetic_fallback,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader


def verify_no_leakage(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> Dict[str, bool]:
    """
    Programmatic verification check ensuring zero recording ID (fsID) or slice overlap
    across train, val, and test splits.
    """
    def _extract_ids(loader: DataLoader) -> Tuple[Set[int], Set[str]]:
        dataset = loader.dataset
        assert isinstance(dataset, UrbanSound8KDataset)
        fs_ids = set(dataset.metadata["fsID"].astype(int))
        slice_names = set(dataset.metadata["slice_file_name"].astype(str))
        return fs_ids, slice_names

    train_fs, train_slices = _extract_ids(train_loader)
    val_fs, val_slices = _extract_ids(val_loader)
    test_fs, test_slices = _extract_ids(test_loader)

    fs_leakage = bool(
        train_fs.intersection(val_fs)
        or train_fs.intersection(test_fs)
        or val_fs.intersection(test_fs)
    )

    slice_leakage = bool(
        train_slices.intersection(val_slices)
        or train_slices.intersection(test_slices)
        or val_slices.intersection(test_slices)
    )

    return {
        "is_leak_free": (not fs_leakage) and (not slice_leakage),
        "fs_leakage_detected": fs_leakage,
        "slice_leakage_detected": slice_leakage,
        "train_samples": len(train_slices),
        "val_samples": len(val_slices),
        "test_samples": len(test_slices),
    }


if __name__ == "__main__":
    print("Testing PyTorch UrbanSound8KDataset and DataLoaders...")
    synthetic_df = create_synthetic_metadata_for_testing(num_samples=40)

    train_l, val_l, test_l = create_dataloaders(
        metadata_df=synthetic_df,
        batch_size=4,
        num_workers=0,
        allow_synthetic_fallback=True,
    )

    # 1. Verify batch shape and forward compatibility
    first_features, first_labels, first_meta = next(iter(train_l))
    print(f"Batch features shape: {first_features.shape}")
    print(f"Batch labels shape: {first_labels.shape}")
    print(f"First sample filename: {first_meta['slice_file_name'][0]}")

    assert first_features.shape == (4, 1, 128, 173), f"Expected (4, 1, 128, 173), got {first_features.shape}"
    assert first_labels.shape == (4,), f"Expected (4,), got {first_labels.shape}"
    assert first_features.dtype == torch.float32, "Features must be torch.float32"
    assert first_labels.dtype == torch.int64, "Labels must be torch.long (int64)"

    # 2. Verify fold segregation & zero leakage
    leakage_report = verify_no_leakage(train_l, val_l, test_l)
    print("Leakage verification report:")
    for k, v in leakage_report.items():
        print(f"  {k}: {v}")

    assert leakage_report["is_leak_free"], "Leakage detected between splits!"
    print("Dataset and DataLoader tests completed successfully!")
