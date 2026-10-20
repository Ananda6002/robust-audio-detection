"""
ml/metrics.py
Scientific evaluation metrics, confusion matrix computation, JSON persistence,
and publication-quality visualization routines for environmental sound classification.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Use headless backend for Matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.config import CLASS_MAP, CLASS_NAMES, RESULTS_DIR


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively convert NumPy scalars, arrays, and non-serializable objects into standard Python types.
    """
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj


def compute_classification_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    y_probs: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute full classification suite:
    - Overall Accuracy
    - Macro Precision, Recall, F1
    - Per-class Precision, Recall, F1, and Support
    - Raw and Normalized Confusion Matrices
    - Top-3 Accuracy (if probabilities provided)
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    names = class_names or CLASS_NAMES
    num_classes = len(names)

    acc = float(accuracy_score(y_t, y_p))
    macro_prec = float(precision_score(y_t, y_p, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_t, y_p, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))

    # Per-class metrics
    per_class_prec = precision_score(y_t, y_p, average=None, labels=list(range(num_classes)), zero_division=0)
    per_class_rec = recall_score(y_t, y_p, average=None, labels=list(range(num_classes)), zero_division=0)
    per_class_f1 = f1_score(y_t, y_p, average=None, labels=list(range(num_classes)), zero_division=0)

    per_class_metrics = {}
    for idx, c_name in enumerate(names):
        support = int((y_t == idx).sum())
        per_class_metrics[c_name] = {
            "class_id": idx,
            "precision": float(per_class_prec[idx]),
            "recall": float(per_class_rec[idx]),
            "f1": float(per_class_f1[idx]),
            "support": support,
        }

    # Confusion Matrix
    cm_raw = confusion_matrix(y_t, y_p, labels=list(range(num_classes)))
    row_sums = cm_raw.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(row_sums > 0, cm_raw / row_sums, 0.0)

    metrics = {
        "accuracy": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "total_samples": len(y_t),
        "per_class": per_class_metrics,
        "confusion_matrix_raw": cm_raw.tolist(),
        "confusion_matrix_norm": cm_norm.tolist(),
        "class_names": names,
    }

    # Optional top-3 accuracy
    if y_probs is not None and y_probs.ndim == 2:
        top3_preds = np.argsort(y_probs, axis=-1)[:, -3:]
        top3_correct = np.any(top3_preds == y_t[:, None], axis=-1)
        metrics["top3_accuracy"] = float(top3_correct.mean())

    return metrics


def save_metrics_json(metrics: Dict[str, Any], filepath: Union[str, Path]) -> Path:
    """Safely serialize evaluation metrics to JSON."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = sanitize_for_json(metrics)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=2)
    return path


def load_metrics_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load evaluation metrics JSON file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: Union[str, Path],
    title_suffix: str = "",
) -> Path:
    """
    Generate training and validation loss and accuracy curves.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Loss curve
    ax1.plot(epochs, history.get("train_loss", []), label="Train Loss", color="#2563eb", linewidth=2)
    ax1.plot(epochs, history.get("val_loss", []), label="Val Loss", color="#dc2626", linewidth=2, linestyle="--")
    ax1.set_title(f"Loss Progression {title_suffix}".strip(), fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True)

    # Accuracy curve
    ax2.plot(epochs, history.get("train_acc", []), label="Train Acc", color="#16a34a", linewidth=2)
    ax2.plot(epochs, history.get("val_acc", []), label="Val Acc", color="#ea580c", linewidth=2, linestyle="--")
    ax2.set_title(f"Accuracy Progression {title_suffix}".strip(), fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_matrix(
    cm: Union[np.ndarray, List[List[float]]],
    class_names: List[str],
    output_path: Union[str, Path],
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> Path:
    """
    Render and save a labeled confusion matrix heatmap.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mat = np.array(cm, dtype=float)
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            mat = np.where(row_sums > 0, mat / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    cax = ax.imshow(mat, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=9)

    ax.set_ylabel("True Ground-Truth Class", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")

    # Annotate cells
    thresh = mat.max() / 2.0 if mat.max() > 0 else 0.5
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            text = f"{val:.2f}" if normalize else f"{int(val)}"
            color = "white" if val > thresh else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_snr_robustness_curve(
    snr_levels: List[float],
    model_f1_scores: Dict[str, List[float]],
    output_path: Union[str, Path],
) -> Path:
    """
    Plot Macro-F1 across signal-to-noise ratios (dB) comparing multiple models.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    colors = ["#2563eb", "#16a34a", "#9333ea", "#ea580c"]
    markers = ["o", "s", "^", "D"]

    for idx, (model_name, f1_vals) in enumerate(model_f1_scores.items()):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        ax.plot(
            snr_levels,
            f1_vals,
            label=model_name,
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=7,
        )

    ax.set_title("Noise Robustness Comparison Across SNR Levels", fontsize=13, fontweight="bold")
    ax.set_xlabel("Signal-to-Noise Ratio (dB) [Higher = Less Noise]", fontsize=11)
    ax.set_ylabel("Macro F1-Score", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, loc="lower right")

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("Testing metrics and visualization module...")
    rng = np.random.default_rng(42)

    # 1. Synthetic predictions test
    y_true_mock = rng.integers(0, 10, size=100)
    y_pred_mock = y_true_mock.copy()
    # Introduce 20% random error
    corrupt_idx = rng.choice(100, size=20, replace=False)
    y_pred_mock[corrupt_idx] = rng.integers(0, 10, size=20)

    y_probs_mock = np.zeros((100, 10), dtype=float)
    for i, p in enumerate(y_pred_mock):
        y_probs_mock[i, p] = 0.8
        y_probs_mock[i, :] += 0.02

    res = compute_classification_metrics(y_true_mock, y_pred_mock, y_probs=y_probs_mock)
    print(f"Accuracy: {res['accuracy']:.4f}")
    print(f"Macro F1: {res['macro_f1']:.4f}")
    assert 0.0 <= res["accuracy"] <= 1.0
    assert 0.0 <= res["macro_f1"] <= 1.0
    assert len(res["per_class"]) == 10

    # 2. Test JSON persistence
    test_json = RESULTS_DIR / "smoke_test_metrics.json"
    save_metrics_json(res, test_json)
    loaded_res = load_metrics_json(test_json)
    assert np.isclose(loaded_res["accuracy"], res["accuracy"])
    if test_json.exists():
        test_json.unlink()

    # 3. Test Plotting
    smoke_history = {
        "train_loss": [1.5, 1.2, 0.9, 0.6],
        "val_loss": [1.6, 1.3, 1.0, 0.8],
        "train_acc": [0.4, 0.6, 0.75, 0.85],
        "val_acc": [0.35, 0.55, 0.7, 0.78],
    }
    curve_path = RESULTS_DIR / "smoke_training_curves.png"
    cm_path = RESULTS_DIR / "smoke_confusion_matrix.png"
    snr_path = RESULTS_DIR / "smoke_snr_curve.png"

    plot_training_curves(smoke_history, curve_path)
    plot_confusion_matrix(res["confusion_matrix_norm"], CLASS_NAMES, cm_path)
    plot_snr_robustness_curve(
        snr_levels=[20.0, 10.0, 5.0, 0.0],
        model_f1_scores={"Baseline": [0.82, 0.71, 0.58, 0.44], "Noise-Augmented": [0.80, 0.75, 0.69, 0.60]},
        output_path=snr_path,
    )

    assert curve_path.exists() and curve_path.stat().st_size > 0
    assert cm_path.exists() and cm_path.stat().st_size > 0
    assert snr_path.exists() and snr_path.stat().st_size > 0

    # Cleanup smoke images
    curve_path.unlink()
    cm_path.unlink()
    snr_path.unlink()

    print("All metrics and visualization tests passed successfully!")
