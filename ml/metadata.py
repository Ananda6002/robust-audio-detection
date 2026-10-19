"""
ml/metadata.py
Robust loader, validator, and partitioner for UrbanSound8K metadata.
Guarantees strict fold-based isolation and prevents data leakage.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path for direct execution
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from ml.config import (
    AUDIO_DIR,
    CLASS_MAP,
    CLASS_NAMES,
    METADATA_PATH,
    TrainingConfig,
)

REQUIRED_COLUMNS: List[str] = [
    "slice_file_name",
    "fsID",
    "start",
    "end",
    "salience",
    "fold",
    "classID",
    "class",
]


def load_metadata(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load and validate the official UrbanSound8K metadata CSV.
    
    Args:
        csv_path: Optional path to UrbanSound8K.csv. Defaults to METADATA_PATH.
        
    Returns:
        pd.DataFrame containing validated metadata.
        
    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If required columns are missing or data types are invalid.
    """
    path = Path(csv_path) if csv_path else METADATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"UrbanSound8K metadata file not found at: {path}. "
            "Please download the dataset and place it in the data/ directory."
        )

    df = pd.read_csv(path)

    # 1. Column presence check
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Metadata CSV is missing required columns: {missing}. "
            f"Expected columns: {REQUIRED_COLUMNS}"
        )

    # 2. Type casting and sanitization
    df["fold"] = df["fold"].astype(int)
    df["classID"] = df["classID"].astype(int)
    df["fsID"] = df["fsID"].astype(int)

    # 3. Value range assertions
    invalid_folds = df[~df["fold"].between(1, 10)]
    if not invalid_folds.empty:
        raise ValueError(
            f"Found {len(invalid_folds)} rows with fold not in 1..10."
        )

    invalid_classes = df[~df["classID"].between(0, 9)]
    if not invalid_classes.empty:
        raise ValueError(
            f"Found {len(invalid_classes)} rows with classID not in 0..9."
        )

    return df


def resolve_audio_path(
    row: pd.Series, audio_dir: Optional[Path] = None
) -> Path:
    """
    Construct the absolute file path for an audio slice according to fold hierarchy.
    """
    base = Path(audio_dir) if audio_dir else AUDIO_DIR
    fold_dir = base / f"fold{int(row['fold'])}"
    return fold_dir / str(row["slice_file_name"])


def get_fold_split(
    df: pd.DataFrame,
    train_folds: Optional[Tuple[int, ...]] = None,
    val_fold: Optional[int] = None,
    test_fold: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partition the dataset into train, validation, and test splits strictly by fold.
    Never randomly shuffles or mixes slices across predefined folds.
    
    Args:
        df: Validated metadata DataFrame.
        train_folds: Folds to include in training (default: 1..8).
        val_fold: Fold to use for validation / early stopping (default: 9).
        test_fold: Fold reserved strictly for held-out evaluation (default: 10).
        
    Returns:
        (train_df, val_df, test_df)
    """
    t_folds = train_folds if train_folds is not None else TrainingConfig.train_folds
    v_fold = val_fold if val_fold is not None else TrainingConfig.val_fold
    te_fold = test_fold if test_fold is not None else TrainingConfig.test_fold

    # Ensure mutually exclusive folds
    overlap = set(t_folds).intersection({v_fold, te_fold})
    if overlap or (v_fold == te_fold):
        raise ValueError(
            f"Fold overlap detected! Train: {t_folds}, Val: {v_fold}, Test: {te_fold}"
        )

    train_df = df[df["fold"].isin(t_folds)].copy().reset_index(drop=True)
    val_df = df[df["fold"] == v_fold].copy().reset_index(drop=True)
    test_df = df[df["fold"] == te_fold].copy().reset_index(drop=True)

    # Cross-fold leakage verification (fsID check: original recording sources must not cross folds)
    train_fs = set(train_df["fsID"].unique())
    val_fs = set(val_df["fsID"].unique())
    test_fs = set(test_df["fsID"].unique())

    if train_fs.intersection(val_fs):
        raise AssertionError("Data leakage! fsID shared between train and validation folds.")
    if train_fs.intersection(test_fs):
        raise AssertionError("Data leakage! fsID shared between train and test folds.")
    if val_fs.intersection(test_fs):
        raise AssertionError("Data leakage! fsID shared between validation and test folds.")

    return train_df, val_df, test_df


def get_class_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """Return sample counts per class name for a given metadata split."""
    counts = df["classID"].value_counts().to_dict()
    return {CLASS_MAP[c_id]: int(counts.get(c_id, 0)) for c_id in range(len(CLASS_MAP))}


def create_synthetic_metadata_for_testing(
    num_samples: int = 50, seed: int = 42
) -> pd.DataFrame:
    """
    Generate deterministic synthetic UrbanSound8K metadata for unit tests and smoke tests.
    Respects schema and guarantees valid fold assignments.
    """
    rng = np.random.default_rng(seed)
    records = []

    for i in range(num_samples):
        # Distribute across 10 folds
        fold = (i % 10) + 1
        class_id = i % 10
        fs_id = 1000 + fold * 100 + (i // 10)
        filename = f"{fs_id}-{class_id}-0-{i}.wav"

        records.append({
            "slice_file_name": filename,
            "fsID": fs_id,
            "start": float(rng.uniform(0.0, 5.0)),
            "end": float(rng.uniform(5.1, 9.0)),
            "salience": int(rng.choice([1, 2])),
            "fold": fold,
            "classID": class_id,
            "class": CLASS_MAP[class_id],
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("Generating synthetic metadata for verification...")
    synth_df = create_synthetic_metadata_for_testing(num_samples=30)
    train, val, test = get_fold_split(synth_df)
    print(f"Total synthetic records: {len(synth_df)}")
    print(f"Train split (folds 1-8): {len(train)} records")
    print(f"Validation split (fold 9): {len(val)} records")
    print(f"Test split (fold 10): {len(test)} records")
    print("Class distribution in synthetic set:")
    for cls_name, count in get_class_distribution(synth_df).items():
        print(f"  {cls_name}: {count}")
    print("Metadata module verification completed successfully.")
