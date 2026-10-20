"""
experiments/evaluate_noise.py
Noise Robustness Evaluation Engine for Environmental Sound Classification.
Evaluates any model checkpoint across clean audio and controlled Signal-to-Noise Ratios
(20 dB, 10 dB, 5 dB, 0 dB) using additive colored noise, quantifying performance degradation.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.augmentation import inject_noise_at_snr
from ml.config import (
    CLASS_NAMES,
    MODELS_DIR,
    RESULTS_DIR,
    ModelConfig,
    NoiseConfig,
    TrainingConfig,
    validate_dataset_structure,
)
from ml.dataset import UrbanSound8KDataset
from ml.metadata import (
    create_synthetic_metadata_for_testing,
    load_metadata,
)
from ml.metrics import (
    compute_classification_metrics,
    plot_snr_robustness_curve,
    save_metrics_json,
)
from ml.models import AcousticCNN
from ml.training_utils import get_device, load_checkpoint, set_seed


def evaluate_noise_robustness(
    checkpoint_path: Path,
    test_fold: int = TrainingConfig.test_fold,
    noise_type: str = "ambient",
    snr_levels: Tuple[float, ...] = NoiseConfig.evaluation_snrs,
    batch_size: int = TrainingConfig.batch_size,
    device_name: Optional[str] = None,
    smoke_test: bool = False,
    output_json: Optional[Path] = None,
    plot_curves: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate model across clean and controlled SNR degradation conditions.
    
    Conditions:
    - Clean (No added noise)
    - 20 dB SNR (Light ambient noise)
    - 10 dB SNR (Moderate background noise)
    - 5 dB SNR (Heavy acoustic interference)
    - 0 dB SNR (Severe degradation / equal signal and noise power)
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")

    set_seed(TrainingConfig.seed)
    device = get_device(device_name)

    print("============================================================")
    print("  ACOUSTIGUARD AI — NOISE ROBUSTNESS BENCHMARKING")
    print("============================================================")
    print(f"Model Checkpoint:  {ckpt_path.name}")
    print(f"Device:            {device}")
    print(f"Test Fold:         Fold {test_fold}")
    print(f"Noise Profile:     {noise_type}")
    print(f"Evaluated SNRs:    Clean, {', '.join(f'{s} dB' for s in snr_levels)}")

    # 1. Load Architecture and Weights
    model = AcousticCNN(ModelConfig()).to(device)
    ckpt_data = load_checkpoint(ckpt_path, model, device=device)
    model.eval()

    # 2. Prepare Held-out Test Metadata
    dataset_status = validate_dataset_structure()
    is_real_data = dataset_status["is_valid"] and not smoke_test

    if is_real_data:
        full_df = load_metadata()
        test_df = full_df[full_df["fold"] == test_fold].copy().reset_index(drop=True)
    else:
        full_df = create_synthetic_metadata_for_testing(num_samples=50, seed=100)
        test_df = full_df[full_df["fold"] == test_fold].copy().reset_index(drop=True)

    test_samples_count = len(test_df)
    print(f"Test set size: {test_samples_count} audio excerpts strictly from Fold {test_fold}")

    # Build evaluation conditions list: ("Clean", None), (20dB, 20.0), (10dB, 10.0), ...
    conditions: List[Tuple[str, Optional[float]]] = [("Clean", None)]
    for snr in snr_levels:
        conditions.append((f"{int(snr) if snr.is_integer() else snr} dB", float(snr)))

    condition_metrics: Dict[str, Any] = {}
    f1_by_snr: List[float] = []
    acc_by_snr: List[float] = []
    condition_labels: List[str] = []

    # 3. Benchmark across conditions
    print("\nExecuting evaluation across degradation conditions...")
    for cond_name, snr_val in conditions:
        t0 = time.perf_counter()

        # Define deterministic noise transform for current SNR condition
        if snr_val is None:
            transform = None
        else:
            # Deterministic noise injection using seed per index
            def make_noise_fn(target_snr: float, n_color: str):
                def _inject(waveform: np.ndarray) -> np.ndarray:
                    noisy, _ = inject_noise_at_snr(
                        waveform,
                        snr_db=target_snr,
                        noise_type=n_color,
                    )
                    return noisy
                return _inject

            transform = make_noise_fn(snr_val, noise_type)

        test_dataset = UrbanSound8KDataset(
            metadata=test_df,
            waveform_transform=transform,
            allow_synthetic_fallback=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for features, targets, _ in test_loader:
                features = features.to(device)
                logits = model(features)
                probs = torch.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(targets.numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())

        elapsed = time.perf_counter() - t0
        y_true = np.array(all_targets, dtype=int)
        y_pred = np.array(all_preds, dtype=int)
        y_probs = np.array(all_probs, dtype=float)

        c_metrics = compute_classification_metrics(y_true, y_pred, y_probs=y_probs, class_names=CLASS_NAMES)
        c_metrics["elapsed_time_sec"] = elapsed
        c_metrics["snr_db"] = snr_val

        condition_metrics[cond_name] = c_metrics
        f1_by_snr.append(c_metrics["macro_f1"])
        acc_by_snr.append(c_metrics["accuracy"])
        condition_labels.append(cond_name)

        print(f"  Condition: {cond_name:<8} | Acc: {c_metrics['accuracy']*100:5.2f}% | "
              f"Macro-F1: {c_metrics['macro_f1']*100:5.2f}% | Time: {elapsed:.2f}s")

    # 4. Generate Robustness Summary Table
    results_payload = {
        "benchmark_name": f"Noise Robustness Benchmark: {ckpt_path.name}",
        "checkpoint": str(ckpt_path),
        "test_fold": test_fold,
        "noise_type": noise_type,
        "is_smoke_test": not is_real_data,
        "samples_evaluated": test_samples_count,
        "conditions": condition_metrics,
        "summary": {
            "condition_labels": condition_labels,
            "macro_f1_scores": f1_by_snr,
            "accuracies": acc_by_snr,
        },
        "timestamp": time.time(),
    }

    # 5. Persist JSON & Robustness Plots
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = (
        Path(output_json)
        if output_json
        else RESULTS_DIR / f"noise_robustness_{ckpt_path.stem}.json"
    )
    save_metrics_json(results_payload, out_file)

    if plot_curves:
        # Plot numeric SNRs (replace Clean with 30 dB for plotting layout)
        numeric_snrs = [30.0] + list(snr_levels)
        curve_file = RESULTS_DIR / f"snr_curve_{ckpt_path.stem}.png"
        plot_snr_robustness_curve(
            snr_levels=numeric_snrs,
            model_f1_scores={f"{ckpt_path.stem} ({noise_type})": f1_by_snr},
            output_path=curve_file,
        )
        results_payload["robustness_curve_plot"] = str(curve_file)

    print("\n------------------------------------------------------------")
    print("  NOISE ROBUSTNESS BENCHMARK SUMMARY TABLE")
    print("------------------------------------------------------------")
    print(f"{'Condition':<10} | {'Accuracy':<10} | {'Macro Precision':<16} | {'Macro Recall':<14} | {'Macro F1':<10}")
    print("-" * 68)
    for c_name in condition_labels:
        cm = condition_metrics[c_name]
        print(f"{c_name:<10} | {cm['accuracy']*100:>8.2f}% | {cm['macro_precision']*100:>14.2f}% | "
              f"{cm['macro_recall']*100:>12.2f}% | {cm['macro_f1']*100:>8.2f}%")
    print("------------------------------------------------------------")
    print(f"Results JSON: {out_file}")
    if plot_curves:
        print(f"Curves Plot:  {curve_file}")
    print("------------------------------------------------------------\n")

    return results_payload


def main():
    parser = argparse.ArgumentParser(description="AcoustiGuard AI — Noise Robustness Evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(MODELS_DIR / "baseline_cnn_best.pt"),
        help="Path to trained checkpoint (.pt)",
    )
    parser.add_argument("--test-fold", type=int, default=10, help="Held-out test fold (1-10)")
    parser.add_argument("--noise-type", type=str, default="ambient", choices=["ambient", "pink", "brown", "white"])
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu' or 'cuda')")
    parser.add_argument("--output-json", type=str, default=None, help="Output JSON path")
    parser.add_argument("--smoke-test", action="store_true", help="Force synthetic test data")
    args = parser.parse_args()

    evaluate_noise_robustness(
        checkpoint_path=Path(args.checkpoint),
        test_fold=args.test_fold,
        noise_type=args.noise_type,
        batch_size=args.batch_size,
        device_name=args.device,
        output_json=Path(args.output_json) if args.output_json else None,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
