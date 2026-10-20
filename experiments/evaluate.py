"""
experiments/evaluate.py
Standalone model evaluation script for environmental sound classification.
Evaluates any saved checkpoint (.pt) on a specified held-out test fold,
computes scientific classification metrics, benchmarks latency, and saves structured results.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.config import (
    CLASS_NAMES,
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
    plot_confusion_matrix,
    save_metrics_json,
)
from ml.models import (
    AcousticCNN,
    benchmark_inference_latency,
    count_parameters,
)
from ml.training_utils import get_device, load_checkpoint, set_seed


def evaluate_model(
    checkpoint_path: Path,
    test_fold: int = TrainingConfig.test_fold,
    batch_size: int = TrainingConfig.batch_size,
    device_name: Optional[str] = None,
    output_json: Optional[Path] = None,
    smoke_test: bool = False,
    plot_cm: bool = True,
) -> Dict[str, Any]:
    """
    Load a checkpoint and perform comprehensive evaluation on the specified test fold.
    
    Args:
        checkpoint_path: Path to .pt model weights.
        test_fold: Predefined UrbanSound8K fold to evaluate on (default: 10).
        batch_size: Evaluation batch size.
        device_name: Optional hardware device ('cpu' or 'cuda').
        output_json: Optional destination path for metrics JSON.
        smoke_test: Force synthetic test data if true.
        plot_cm: If true, generates and saves confusion matrix plot.
        
    Returns:
        Structured evaluation metrics dictionary.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {ckpt_path}")

    set_seed(TrainingConfig.seed)
    device = get_device(device_name)

    print("============================================================")
    print("  ACOUSTIGUARD AI — MODEL EVALUATION PIPELINE")
    print("============================================================")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"Device:     {device}")
    print(f"Test Fold:  Fold {test_fold}")

    # 1. Load Model Architecture & Checkpoint
    model = AcousticCNN(ModelConfig()).to(device)
    ckpt_data = load_checkpoint(ckpt_path, model, device=device)
    model.eval()
    param_info = count_parameters(model)
    print(f"Loaded weights from Epoch {ckpt_data.get('epoch', 'N/A')}")
    print(f"Parameters: {param_info['total_params']:,} ({param_info['trainable_params']:,} trainable)")

    # 2. Prepare Evaluation DataLoader
    dataset_status = validate_dataset_structure()
    is_real_data = dataset_status["is_valid"] and not smoke_test

    if is_real_data:
        print(f"Loading real UrbanSound8K test partition for Fold {test_fold}...")
        meta_df = load_metadata()
    else:
        print("Using synthetic test partition for evaluation...")
        meta_df = create_synthetic_metadata_for_testing(num_samples=50, seed=100)

    val_fold_candidate = 9 if test_fold != 9 else 8
    train_folds_dummy = tuple(f for f in range(1, 11) if f not in (val_fold_candidate, test_fold))

    _, _, test_loader = create_dataloaders(
        metadata_df=meta_df,
        batch_size=batch_size,
        train_folds=train_folds_dummy,
        val_fold=val_fold_candidate,
        test_fold=test_fold,
        num_workers=0,
        allow_synthetic_fallback=True,
    )

    print(f"Test Dataset size: {len(test_loader.dataset)} samples across {len(test_loader)} batches")

    # 3. Inference Loop
    all_preds = []
    all_targets = []
    all_probs = []

    t_start = time.perf_counter()
    with torch.no_grad():
        for features, targets, _ in test_loader:
            features = features.to(device)
            logits = model(features)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
    eval_elapsed = time.perf_counter() - t_start

    y_true = np.array(all_targets, dtype=int)
    y_pred = np.array(all_preds, dtype=int)
    y_probs = np.array(all_probs, dtype=float)

    # 4. Compute Metrics
    metrics = compute_classification_metrics(y_true, y_pred, y_probs=y_probs, class_names=CLASS_NAMES)
    latency_stats = benchmark_inference_latency(model, device="cpu")

    report = {
        "evaluation_name": f"Evaluation of {ckpt_path.name} on Fold {test_fold}",
        "checkpoint": str(ckpt_path),
        "test_fold": test_fold,
        "is_smoke_test": not is_real_data,
        "samples_evaluated": len(y_true),
        "total_eval_time_sec": eval_elapsed,
        "metrics": metrics,
        "latency_stats": latency_stats,
        "timestamp": time.time(),
    }

    # 5. Persist JSON & Confusion Matrix
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = (
        Path(output_json)
        if output_json
        else RESULTS_DIR / f"eval_{ckpt_path.stem}_fold{test_fold}.json"
    )
    save_metrics_json(report, out_json)

    if plot_cm:
        cm_path = RESULTS_DIR / f"cm_{ckpt_path.stem}_fold{test_fold}.png"
        plot_confusion_matrix(
            metrics["confusion_matrix_norm"],
            CLASS_NAMES,
            cm_path,
            title=f"Confusion Matrix — {ckpt_path.name} (Fold {test_fold})",
        )
        report["confusion_matrix_plot"] = str(cm_path)

    print("\n------------------------------------------------------------")
    print(f"  EVALUATION RESULTS — FOLD {test_fold}")
    print("------------------------------------------------------------")
    print(f"Overall Accuracy:        {metrics['accuracy']*100:.2f}%")
    print(f"Macro Precision:         {metrics['macro_precision']*100:.2f}%")
    print(f"Macro Recall:            {metrics['macro_recall']*100:.2f}%")
    print(f"Macro F1-Score:          {metrics['macro_f1']*100:.2f}%")
    if "top3_accuracy" in metrics:
        print(f"Top-3 Accuracy:          {metrics['top3_accuracy']*100:.2f}%")
    print(f"Mean CPU Latency:        {latency_stats['mean_latency_ms']:.2f} ms")
    print(f"Metrics Saved:           {out_json}")
    if plot_cm:
        print(f"Plot Saved:              {cm_path}")
    print("------------------------------------------------------------\n")

    return report


def main():
    parser = argparse.ArgumentParser(description="AcoustiGuard AI — Standalone Model Evaluator")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(MODELS_DIR / "baseline_cnn_best.pt"),
        help="Path to .pt checkpoint file",
    )
    parser.add_argument("--test-fold", type=int, default=10, help="UrbanSound8K fold to test (1-10)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda')")
    parser.add_argument("--output-json", type=str, default=None, help="Custom output JSON path")
    parser.add_argument("--smoke-test", action="store_true", help="Force synthetic test data")
    args = parser.parse_args()

    evaluate_model(
        checkpoint_path=Path(args.checkpoint),
        test_fold=args.test_fold,
        batch_size=args.batch_size,
        device_name=args.device,
        output_json=Path(args.output_json) if args.output_json else None,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
