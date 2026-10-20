"""
experiments/evaluate_selective.py
Experiment C: Confidence-Aware Prediction and Selective Classification.
Calibrates model logits with Temperature Scaling, evaluates confidence thresholding,
and quantifies the trade-off between Prediction Coverage, Rejection Rate, and Selective Risk.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader

# Use headless backend for Matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.calibration import (
    TemperatureScaler,
    calibrate_model_from_loader,
    compute_ece,
    plot_reliability_diagram,
)
from ml.config import (
    CLASS_NAMES,
    ConfidenceConfig,
    MODELS_DIR,
    RESULTS_DIR,
    ModelConfig,
    TrainingConfig,
    validate_dataset_structure,
)
from ml.dataset import create_dataloaders
from ml.metadata import (
    create_synthetic_metadata_for_testing,
    load_metadata,
)
from ml.metrics import (
    compute_classification_metrics,
    save_metrics_json,
)
from ml.models import AcousticCNN
from ml.training_utils import get_device, load_checkpoint, set_seed


def plot_risk_coverage_curve(
    thresholds: List[float],
    coverages: List[float],
    risks: List[float],
    accuracies: List[float],
    output_path: Union[str, Path],
    selected_threshold: float = 0.70,
) -> Path:
    """
    Plot the trade-off between Coverage Rate (%) and Selective Risk (Error Rate %).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8.5, 5.5), dpi=150)

    # Coverage curve (Left axis)
    color_cov = "#2563eb"
    ax1.set_xlabel("Confidence Rejection Threshold (tau)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Coverage Rate (%) [Accepted Samples]", color=color_cov, fontsize=11, fontweight="bold")
    l1 = ax1.plot(
        thresholds,
        [c * 100 for c in coverages],
        "o-",
        color=color_cov,
        linewidth=2.2,
        label="Coverage Rate (%)",
    )
    ax1.tick_params(axis="y", labelcolor=color_cov)
    ax1.set_ylim(-5, 105)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Selective Accuracy / Risk curve (Right axis)
    ax2 = ax1.twinx()
    color_acc = "#16a34a"
    ax2.set_ylabel("Selective Accuracy (%) [On Accepted]", color=color_acc, fontsize=11, fontweight="bold")
    l2 = ax2.plot(
        thresholds,
        [a * 100 for a in accuracies],
        "s--",
        color=color_acc,
        linewidth=2.2,
        label="Selective Accuracy (%)",
    )
    ax2.tick_params(axis="y", labelcolor=color_acc)
    ax2.set_ylim(-5, 105)

    # Highlight selected threshold
    if selected_threshold in thresholds:
        ax1.axvline(selected_threshold, color="#dc2626", linestyle=":", linewidth=1.8, label=f"Default tau={selected_threshold}")

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower left", frameon=True)

    plt.title("Selective Prediction: Coverage vs. Selective Accuracy Trade-off", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def run_selective_evaluation(
    checkpoint_path: Path,
    val_fold: int = TrainingConfig.val_fold,
    test_fold: int = TrainingConfig.test_fold,
    thresholds: Tuple[float, ...] = ConfidenceConfig.candidate_thresholds,
    default_threshold: float = ConfidenceConfig.default_threshold,
    device_name: Optional[str] = None,
    smoke_test: bool = False,
    output_json: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Execute Experiment C (Confidence Calibration & Selective Prediction Evaluation).
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")

    set_seed(TrainingConfig.seed)
    device = get_device(device_name)

    print("============================================================")
    print("  ACOUSTIGUARD AI — EXPERIMENT C: CONFIDENCE-AWARE PREDICTION")
    print("============================================================")
    print(f"Model Checkpoint:  {ckpt_path.name}")
    print(f"Validation Fold:   Fold {val_fold} (used for temperature fitting)")
    print(f"Test Fold:         Fold {test_fold} (held-out evaluation)")
    print(f"Default Threshold: tau = {default_threshold}")

    # 1. Load Architecture and Weights
    model = AcousticCNN(ModelConfig()).to(device)
    load_checkpoint(ckpt_path, model, device=device)
    model.eval()

    # 2. Dataset Loading
    dataset_status = validate_dataset_structure()
    is_real_data = dataset_status["is_valid"] and not smoke_test

    if is_real_data:
        full_df = load_metadata()
    else:
        full_df = create_synthetic_metadata_for_testing(num_samples=100, seed=555)

    train_folds_dummy = tuple(f for f in range(1, 11) if f not in (val_fold, test_fold))

    _, val_loader, test_loader = create_dataloaders(
        metadata_df=full_df,
        batch_size=16,
        train_folds=train_folds_dummy,
        val_fold=val_fold,
        test_fold=test_fold,
        num_workers=0,
        allow_synthetic_fallback=True,
    )

    # 3. Fit Temperature Scaler on Validation Set
    print("\nFitting Temperature Scaling calibration on Validation Fold...")
    scaler, cal_stats = calibrate_model_from_loader(model, val_loader, device=device)
    optimal_t = cal_stats["optimal_temperature"]
    print(f"Optimal Temperature (T): {optimal_t:.4f}")
    print(f"Validation ECE: {cal_stats['uncalibrated_ece']*100:.2f}% -> {cal_stats['calibrated_ece']*100:.2f}% "
          f"(Diff ECE = {cal_stats['ece_reduction']*100:+.2f}%)")

    scaler_save_path = MODELS_DIR / "temperature_scaler.pt"
    scaler.save(scaler_save_path)
    print(f"Saved calibrated temperature scaler to: {scaler_save_path}")

    # 4. Evaluate Held-out Test Set
    print("\nExtracting test predictions and calibrated probabilities...")
    test_logits_list = []
    test_targets_list = []

    with torch.no_grad():
        for features, targets, _ in test_loader:
            features = features.to(device)
            logits = model(features)
            test_logits_list.append(logits.cpu())
            test_targets_list.append(targets.cpu())

    test_logits = torch.cat(test_logits_list, dim=0)
    test_targets = torch.cat(test_targets_list, dim=0).numpy()

    # Probabilities
    uncal_probs = torch.softmax(test_logits, dim=-1).numpy()
    cal_probs = scaler.predict_proba(test_logits).numpy()

    # Calibration error on test set
    test_uncal_ece = compute_ece(test_targets, uncal_probs)
    test_cal_ece = compute_ece(test_targets, cal_probs)

    # Reliability diagram
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    diagram_path = RESULTS_DIR / "reliability_diagram.png"
    plot_reliability_diagram(test_uncal_ece, test_cal_ece, diagram_path)

    # 5. Selective Prediction Threshold Sweep
    confidences = np.max(cal_probs, axis=1)
    predictions = np.argmax(cal_probs, axis=1)
    total_test_samples = len(test_targets)

    threshold_results: List[Dict[str, Any]] = []
    cov_list: List[float] = []
    risk_list: List[float] = []
    acc_list: List[float] = []

    for tau in thresholds:
        accepted_mask = confidences >= tau
        num_accepted = int(np.sum(accepted_mask))
        coverage = num_accepted / total_test_samples
        rejection_rate = 1.0 - coverage

        if num_accepted > 0:
            accepted_correct = np.sum(predictions[accepted_mask] == test_targets[accepted_mask])
            sel_accuracy = float(accepted_correct / num_accepted)
            sel_risk = 1.0 - sel_accuracy
        else:
            sel_accuracy = 1.0
            sel_risk = 0.0

        cov_list.append(coverage)
        risk_list.append(sel_risk)
        acc_list.append(sel_accuracy)

        threshold_results.append({
            "threshold": float(tau),
            "num_accepted": num_accepted,
            "num_rejected": total_test_samples - num_accepted,
            "coverage_rate": float(coverage),
            "rejection_rate": float(rejection_rate),
            "selective_accuracy": float(sel_accuracy),
            "selective_risk": float(sel_risk),
        })

    # Plot Coverage vs Risk Curve
    curve_path = RESULTS_DIR / "risk_coverage_curve.png"
    plot_risk_coverage_curve(
        thresholds=list(thresholds),
        coverages=cov_list,
        risks=risk_list,
        accuracies=acc_list,
        output_path=curve_path,
        selected_threshold=default_threshold,
    )

    # 6. Default Threshold Operational Metrics
    def_accepted_mask = confidences >= default_threshold
    def_accepted_count = int(np.sum(def_accepted_mask))
    def_coverage = def_accepted_count / total_test_samples
    def_rejected_count = total_test_samples - def_accepted_count

    report = {
        "experiment_name": "Experiment C: Confidence-Aware Prediction",
        "description": "Temperature Scaling calibration and selective abstention on low confidence.",
        "checkpoint": str(ckpt_path),
        "test_fold": test_fold,
        "optimal_temperature": float(optimal_t),
        "test_uncalibrated_ece": float(test_uncal_ece["ece"]),
        "test_calibrated_ece": float(test_cal_ece["ece"]),
        "default_threshold": float(default_threshold),
        "default_metrics": {
            "total_samples": total_test_samples,
            "accepted_samples": def_accepted_count,
            "rejected_samples": def_rejected_count,
            "coverage_rate": float(def_coverage),
            "rejection_rate": float(1.0 - def_coverage),
            "abstention_label": ConfidenceConfig.uncertain_label,
        },
        "threshold_sweep": threshold_results,
        "plots": {
            "reliability_diagram": str(diagram_path),
            "risk_coverage_curve": str(curve_path),
        },
        "timestamp": time.time(),
    }

    out_file = Path(output_json) if output_json else RESULTS_DIR / "selective_prediction_report.json"
    save_metrics_json(report, out_file)

    # 7. Print Trade-off Table
    print("\n==========================================================================")
    print("  SELECTIVE CLASSIFICATION TRADE-OFF TABLE (EXPERIMENT C)")
    print("==========================================================================")
    print(f"{'Tau':<8} | {'Accepted':<10} | {'Coverage':<10} | {'Rejection':<10} | {'Selective Acc':<14} | {'Risk':<8}")
    print("-" * 72)
    for row in threshold_results:
        mark = " <-- DEFAULT" if abs(row["threshold"] - default_threshold) < 1e-4 else ""
        print(f"{row['threshold']:<8.2f} | {row['num_accepted']:<10} | "
              f"{row['coverage_rate']*100:>8.1f}% | {row['rejection_rate']*100:>8.1f}% | "
              f"{row['selective_accuracy']*100:>12.2f}% | {row['selective_risk']*100:>6.2f}%{mark}")
    print("==========================================================================")
    print(f"Reliability Diagram: {diagram_path}")
    print(f"Risk-Coverage Curve: {curve_path}")
    print(f"Results JSON:        {out_file}")
    print("==========================================================================\n")

    return report


def main():
    parser = argparse.ArgumentParser(description="AcoustiGuard AI — Experiment C: Selective Prediction")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(MODELS_DIR / "augmented_cnn_best.pt"),
        help="Path to trained model checkpoint",
    )
    parser.add_argument("--val-fold", type=int, default=9, help="Validation fold used for calibration")
    parser.add_argument("--test-fold", type=int, default=10, help="Held-out test fold")
    parser.add_argument("--threshold", type=float, default=0.70, help="Default rejection threshold")
    parser.add_argument("--smoke-test", action="store_true", help="Force synthetic test data")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu' or 'cuda')")
    parser.add_argument("--output-json", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    # Fallback to baseline checkpoint if augmented checkpoint not found
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        ckpt = MODELS_DIR / "baseline_cnn_best.pt"

    run_selective_evaluation(
        checkpoint_path=ckpt,
        val_fold=args.val_fold,
        test_fold=args.test_fold,
        default_threshold=args.threshold,
        smoke_test=args.smoke_test,
        device_name=args.device,
        output_json=Path(args.output_json) if args.output_json else None,
    )


if __name__ == "__main__":
    main()
