"""
experiments/train_baseline.py
Experiment A: Baseline CNN Training and Evaluation on Clean UrbanSound8K Audio.
Strictly respects predefined folds (train: 1-8, val: 9, test: 10), saves best checkpoint,
evaluates held-out test data, and records verifiable metrics and plots to results/.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.config import (
    CLASS_NAMES,
    MODELS_DIR,
    RESULTS_DIR,
    AudioConfig,
    ModelConfig,
    TrainingConfig,
    validate_dataset_structure,
)
from ml.dataset import create_dataloaders, verify_no_leakage
from ml.metadata import (
    create_synthetic_metadata_for_testing,
    load_metadata,
)
from ml.metrics import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    save_metrics_json,
)
from ml.models import (
    AcousticCNN,
    benchmark_inference_latency,
    count_parameters,
    estimate_model_size_mb,
)
from ml.training_utils import (
    EarlyStopping,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    train_one_epoch,
    validate_one_epoch,
)


def run_baseline_experiment(
    epochs: int = TrainingConfig.epochs,
    batch_size: int = TrainingConfig.batch_size,
    learning_rate: float = TrainingConfig.learning_rate,
    patience: int = TrainingConfig.early_stopping_patience,
    test_fold: int = TrainingConfig.test_fold,
    val_fold: int = TrainingConfig.val_fold,
    smoke_test: bool = False,
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute Experiment A (Baseline).
    
    1. Loads dataset and verifies zero data leakage across folds.
    2. Initializes AcousticCNN model and Adam optimizer.
    3. Trains across epochs with EarlyStopping on validation loss.
    4. Restores best model weights.
    5. Evaluates on held-out test fold.
    6. Saves metrics, training curves, and confusion matrix to results/.
    """
    start_time = time.time()
    set_seed(TrainingConfig.seed)
    device = get_device(device_name)
    print(f"============================================================")
    print(f"  ACOUSTIGUARD AI — EXPERIMENT A: BASELINE MODEL TRAINING")
    print(f"============================================================")
    print(f"Target Device: {device}")
    print(f"Evaluation Protocol: Train Folds = {[f for f in range(1, 11) if f not in (val_fold, test_fold)]}, "
          f"Val Fold = {val_fold}, Held-out Test Fold = {test_fold}")

    # 1. Dataset loading and smoke-test check
    dataset_status = validate_dataset_structure()
    is_real_data = dataset_status["is_valid"] and not smoke_test

    if is_real_data:
        print("Using official UrbanSound8K dataset from data/UrbanSound8K...")
        metadata_df = load_metadata()
    else:
        print("NOTICE: Real dataset not detected or --smoke-test enabled.")
        print("Generating deterministic synthetic metadata and tones for smoke training...")
        metadata_df = create_synthetic_metadata_for_testing(num_samples=100, seed=42)

    # Calculate train folds
    train_folds = tuple(f for f in range(1, 11) if f not in (val_fold, test_fold))

    train_loader, val_loader, test_loader = create_dataloaders(
        metadata_df=metadata_df,
        batch_size=batch_size,
        train_folds=train_folds,
        val_fold=val_fold,
        test_fold=test_fold,
        num_workers=0,
        allow_synthetic_fallback=True,
    )

    # 2. Leakage verification
    leakage = verify_no_leakage(train_loader, val_loader, test_loader)
    print(f"Data Leakage Check: is_leak_free={leakage['is_leak_free']} "
          f"(Train samples={leakage['train_samples']}, Val={leakage['val_samples']}, Test={leakage['test_samples']})")
    if not leakage["is_leak_free"]:
        raise AssertionError("Data leakage detected between dataset splits!")

    # 3. Model setup
    model = AcousticCNN(ModelConfig()).to(device)
    param_info = count_parameters(model)
    model_size_mb = estimate_model_size_mb(model)
    print(f"AcousticCNN initialized: {param_info['total_params']:,} parameters ({model_size_mb:.2f} MB)")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=TrainingConfig.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopping(patience=patience, mode="min")

    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_checkpoint_path = MODELS_DIR / "baseline_cnn_best.pt"

    # 4. Training Loop
    print(f"\nStarting training for up to {epochs} epochs (EarlyStopping patience={patience})...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_targets = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        improved = stopper(val_loss, model)
        flag = " [BEST - Saved]" if improved else ""

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.1f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:5.1f}% | "
              f"Time: {elapsed:4.1f}s{flag}")

        if improved:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"val_loss": val_loss, "val_acc": val_acc},
                filepath=best_checkpoint_path,
            )

        if stopper.early_stop:
            print(f"\nEarly stopping triggered after {epoch} epochs (patience={patience}).")
            break

    # 5. Restore Best Model and Evaluate on Held-out Test Set
    print("\nLoading best checkpoint weights for held-out test evaluation...")
    if best_checkpoint_path.exists():
        load_checkpoint(best_checkpoint_path, model, device=device)
    else:
        stopper.restore_best_weights(model)

    model.eval()
    test_preds_list = []
    test_targets_list = []
    test_probs_list = []
    test_total_loss = 0.0
    test_samples = 0

    with torch.no_grad():
        for features, targets, _ in test_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            loss = criterion(logits, targets)

            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            b_size = targets.size(0)
            test_total_loss += loss.item() * b_size
            test_samples += b_size

            test_preds_list.extend(preds.cpu().numpy().tolist())
            test_targets_list.extend(targets.cpu().numpy().tolist())
            test_probs_list.extend(probs.cpu().numpy().tolist())

    test_loss = test_total_loss / max(1, test_samples)
    y_true = np.array(test_targets_list, dtype=int)
    y_pred = np.array(test_preds_list, dtype=int)
    y_probs = np.array(test_probs_list, dtype=float)

    # 6. Compute Scientific Metrics
    test_metrics = compute_classification_metrics(y_true, y_pred, y_probs=y_probs, class_names=CLASS_NAMES)
    latency_stats = benchmark_inference_latency(model, device="cpu")

    experiment_report = {
        "experiment_name": "Experiment A: Baseline CNN",
        "description": "Standard compact CNN trained strictly on clean audio excerpts.",
        "dataset": "UrbanSound8K",
        "is_smoke_test": not is_real_data,
        "train_folds": list(train_folds),
        "val_fold": val_fold,
        "test_fold": test_fold,
        "total_epochs_trained": len(history["train_loss"]),
        "best_epoch": stopper.counter,
        "final_test_loss": float(test_loss),
        "metrics": test_metrics,
        "history": history,
        "model_parameters": param_info,
        "model_size_mb": float(model_size_mb),
        "cpu_latency_stats": latency_stats,
        "elapsed_training_seconds": float(time.time() - start_time),
    }

    # 7. Persist Artifacts to results/
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_file = RESULTS_DIR / "baseline_metrics.json"
    curves_file = RESULTS_DIR / "baseline_training_curves.png"
    cm_file = RESULTS_DIR / "baseline_confusion_matrix.png"

    save_metrics_json(experiment_report, metrics_file)
    plot_training_curves(history, curves_file, title_suffix="(Baseline CNN)")
    plot_confusion_matrix(test_metrics["confusion_matrix_norm"], CLASS_NAMES, cm_file, title="Baseline CNN — Test Confusion Matrix")

    print("\n------------------------------------------------------------")
    print("  EXPERIMENT A (BASELINE) EVALUATION SUMMARY")
    print("------------------------------------------------------------")
    print(f"Test Accuracy:         {test_metrics['accuracy']*100:.2f}%")
    print(f"Test Macro Precision:  {test_metrics['macro_precision']*100:.2f}%")
    print(f"Test Macro Recall:     {test_metrics['macro_recall']*100:.2f}%")
    print(f"Test Macro F1-Score:   {test_metrics['macro_f1']*100:.2f}%")
    print(f"Mean CPU Latency:      {latency_stats['mean_latency_ms']:.2f} ms")
    print(f"Saved Checkpoint:      {best_checkpoint_path}")
    print(f"Saved Metrics JSON:    {metrics_file}")
    print(f"Saved Curves Plot:     {curves_file}")
    print(f"Saved Confusion Mtx:   {cm_file}")
    print("------------------------------------------------------------\n")

    return experiment_report


def main():
    parser = argparse.ArgumentParser(description="AcoustiGuard AI — Train Baseline CNN (Experiment A)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--test-fold", type=int, default=10, help="Predefined test fold (1-10)")
    parser.add_argument("--val-fold", type=int, default=9, help="Predefined validation fold (1-10)")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast smoke test with synthetic audio")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda')")
    args = parser.parse_args()

    run_baseline_experiment(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        test_fold=args.test_fold,
        val_fold=args.val_fold,
        smoke_test=args.smoke_test,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
