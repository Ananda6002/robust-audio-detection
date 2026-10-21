"""
backend/app/routes/analytics.py
Model information, experiment tracking, and real scientific metrics endpoints.
Reads and serves actual evaluation artifacts from disk (never hardcoding fabricated results).
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.config import (
    CLASS_MAP,
    CLASS_NAMES,
    ConfidenceConfig,
    MODELS_DIR,
    NoiseConfig,
    RESULTS_DIR,
    TrainingConfig,
    validate_dataset_structure,
)
from ml.metrics import load_metrics_json
from ml.models import AcousticCNN, count_parameters, estimate_model_size_mb
from ml.training_utils import get_device

router = APIRouter()


# ==========================================
# 1. Model Info Endpoint
# ==========================================

@router.get(
    "/model-info",
    summary="Model Architecture & Dataset Metadata",
    description="Retrieve deep learning architecture specifications, parameter counts, dataset details, and known limitations.",
)
async def get_model_info() -> Dict[str, Any]:
    """
    Return comprehensive technical specifications for the deployed system.
    """
    dev = get_device()
    sample_model = AcousticCNN()
    param_info = count_parameters(sample_model)
    model_size_mb = estimate_model_size_mb(sample_model)

    # Check for calibrated temperature
    temp_path = MODELS_DIR / "temperature_scaler.pt"
    calibrated_temp = None
    if temp_path.exists():
        try:
            scaler_data = torch.load(temp_path, map_location="cpu")
            calibrated_temp = float(scaler_data.get("temperature", 1.0))
        except Exception:
            pass

    dataset_check = validate_dataset_structure()

    return {
        "model_name": "AcoustiGuard AI — AcousticCNN",
        "architecture_type": "4-Stage 2D Deep Convolutional Neural Network with Adaptive Spatial Pooling",
        "parameters": {
            "total_parameters": param_info["total_params"],
            "trainable_parameters": param_info["trainable_params"],
            "model_size_mb": round(model_size_mb, 2),
            "base_filter_dimension": 32,
            "hidden_fc_dimension": 256,
        },
        "input_representation": {
            "feature_type": "Log-Mel Spectrogram",
            "tensor_shape": [1, 128, 173],
            "sampling_rate_hz": 22050,
            "duration_seconds": 4.0,
            "target_samples": 88200,
            "fft_window_size": 1024,
            "hop_length": 512,
            "mel_frequency_bins": 128,
            "frequency_range_hz": [20.0, 11025.0],
            "dynamic_range_clamp_db": 80.0,
        },
        "taxonomy": {
            "num_classes": len(CLASS_NAMES),
            "classes": CLASS_NAMES,
            "class_map": CLASS_MAP,
        },
        "training_protocol": {
            "dataset_name": "UrbanSound8K",
            "total_dataset_excerpts": 8732,
            "fold_strategy": "10-Fold Predefined Stratification (Zero Cross-Fold Slice Leakage)",
            "train_folds": list(TrainingConfig.train_folds),
            "validation_fold": TrainingConfig.val_fold,
            "test_fold": TrainingConfig.test_fold,
            "optimizer": "Adam (Initial LR=1e-3, Weight Decay=1e-4)",
            "lr_scheduler": "ReduceLROnPlateau (factor=0.5, patience=3)",
            "early_stopping_patience": TrainingConfig.early_stopping_patience,
        },
        "confidence_calibration": {
            "calibration_method": "Post-Hoc Temperature Scaling (Guo et al., 2017)",
            "learned_temperature": calibrated_temp,
            "default_abstention_threshold": ConfidenceConfig.default_threshold,
            "uncertainty_label": ConfidenceConfig.uncertain_label,
        },
        "hardware_environment": {
            "active_device": str(dev),
            "cuda_available": torch.cuda.is_available(),
        },
        "dataset_status": {
            "is_real_dataset_present": dataset_check["is_valid"],
            "existing_folds": dataset_check["existing_folds"],
        },
        "known_limitations": [
            "Fixed 4.0-second input requirement: shorter clips are zero-padded and longer clips are center-cropped.",
            "Synthetic additive noise may differ from complex multi-source reverberant acoustic environments.",
            "Classification performance under extreme noise (<0 dB) requires human-in-the-loop review.",
            "Designed primarily for urban outdoor acoustics matching the UrbanSound8K taxonomy.",
        ],
    }


# ==========================================
# 2. Experiments Registry Endpoint
# ==========================================

@router.get(
    "/experiments",
    summary="Research Experiments Overview",
    description="Retrieve research configurations, methodology, and execution statuses for Experiments A, B, and C.",
)
async def get_experiments_summary() -> List[Dict[str, Any]]:
    """
    Return high-level experimental metadata and completion statuses.
    """
    baseline_done = (RESULTS_DIR / "baseline_metrics.json").exists()
    augmented_done = (RESULTS_DIR / "augmented_metrics.json").exists()
    comparison_done = (RESULTS_DIR / "experiment_comparison.json").exists()
    selective_done = (RESULTS_DIR / "selective_prediction_report.json").exists()

    return [
        {
            "id": "exp_a",
            "name": "Experiment A: Baseline CNN",
            "scientific_objective": (
                "Establish baseline classification benchmark on clean urban audio recordings "
                "under the official 10-fold cross-validation protocol."
            ),
            "training_condition": "Original clean audio recordings only",
            "evaluation_conditions": ["Clean Audio", "20 dB SNR", "10 dB SNR", "5 dB SNR", "0 dB SNR"],
            "checkpoint": "models/baseline_cnn_best.pt",
            "is_completed": baseline_done,
        },
        {
            "id": "exp_b",
            "name": "Experiment B: Noise Augmentation Defense",
            "scientific_objective": (
                "Investigate whether training-time additive noise augmentation improves classification "
                "robustness under variable signal-to-noise ratios (20 dB, 10 dB, 5 dB, 0 dB)."
            ),
            "training_condition": "Clean audio + dynamic additive ambient noise (p=0.5, SNR 5-25 dB)",
            "evaluation_conditions": ["Clean Audio", "20 dB SNR", "10 dB SNR", "5 dB SNR", "0 dB SNR"],
            "checkpoint": "models/augmented_cnn_best.pt",
            "is_completed": augmented_done and comparison_done,
        },
        {
            "id": "exp_c",
            "name": "Experiment C: Confidence-Aware Selective Prediction",
            "scientific_objective": (
                "Calibrate model confidence using Temperature Scaling on validation data and introduce "
                "an abstention threshold mechanism to detect unreliable and noise-corrupted predictions."
            ),
            "training_condition": "Post-hoc temperature scaling fitted on validation fold logits",
            "evaluation_conditions": ["Reliability Diagrams", "Expected Calibration Error (ECE)", "Coverage vs. Risk Trade-off"],
            "checkpoint": "models/temperature_scaler.pt",
            "is_completed": selective_done,
        },
    ]


# ==========================================
# 3. Real Metrics & Plots Endpoint
# ==========================================

@router.get(
    "/metrics",
    summary="Actual Research Metrics & Plot URLs",
    description="Loads actual JSON metric reports and static plot assets produced by the experimental runs.",
)
async def get_all_metrics() -> Dict[str, Any]:
    """
    Loads and aggregates all real experiment outputs from results/ directory.
    """
    def _safe_load(filename: str) -> Optional[Dict[str, Any]]:
        path = RESULTS_DIR / filename
        if path.exists():
            try:
                return load_metrics_json(path)
            except Exception:
                return None
        return None

    baseline_metrics = _safe_load("baseline_metrics.json")
    augmented_metrics = _safe_load("augmented_metrics.json")
    comparison_data = _safe_load("experiment_comparison.json")
    selective_report = _safe_load("selective_prediction_report.json")
    base_noise_bench = _safe_load("noise_robustness_baseline_cnn_best.json")
    aug_noise_bench = _safe_load("noise_robustness_augmented_cnn_best.json")

    # Static plot asset URLs
    plot_assets = {
        "baseline_training_curves": "/static/results/baseline_training_curves.png" if (RESULTS_DIR / "baseline_training_curves.png").exists() else None,
        "baseline_confusion_matrix": "/static/results/baseline_confusion_matrix.png" if (RESULTS_DIR / "baseline_confusion_matrix.png").exists() else None,
        "augmented_training_curves": "/static/results/augmented_training_curves.png" if (RESULTS_DIR / "augmented_training_curves.png").exists() else None,
        "augmented_confusion_matrix": "/static/results/augmented_confusion_matrix.png" if (RESULTS_DIR / "augmented_confusion_matrix.png").exists() else None,
        "comparison_snr_curve": "/static/results/comparison_snr_curve.png" if (RESULTS_DIR / "comparison_snr_curve.png").exists() else None,
        "snr_curve_baseline": "/static/results/snr_curve_baseline_cnn_best.png" if (RESULTS_DIR / "snr_curve_baseline_cnn_best.png").exists() else None,
        "reliability_diagram": "/static/results/reliability_diagram.png" if (RESULTS_DIR / "reliability_diagram.png").exists() else None,
        "risk_coverage_curve": "/static/results/risk_coverage_curve.png" if (RESULTS_DIR / "risk_coverage_curve.png").exists() else None,
    }

    return {
        "status": "success",
        "has_results": any(
            x is not None for x in [baseline_metrics, augmented_metrics, comparison_data, selective_report]
        ),
        "data": {
            "baseline": baseline_metrics,
            "augmented": augmented_metrics,
            "comparison": comparison_data,
            "selective_prediction": selective_report,
            "noise_benchmarks": {
                "baseline": base_noise_bench,
                "augmented": aug_noise_bench,
            },
        },
        "plots": plot_assets,
    }


if __name__ == "__main__":
    from fastapi.testclient import TestClient
    from backend.app.main import app

    print("Testing /api/model-info, /api/experiments, and /api/metrics endpoints...")
    client = TestClient(app)

    # 1. Test /api/model-info
    r_info = client.get("/api/model-info")
    print(f"Model-info Status Code: {r_info.status_code}")
    assert r_info.status_code == 200
    info_json = r_info.json()
    assert "parameters" in info_json
    assert info_json["taxonomy"]["num_classes"] == 10

    # 2. Test /api/experiments
    r_exp = client.get("/api/experiments")
    print(f"Experiments Status Code: {r_exp.status_code}")
    assert r_exp.status_code == 200
    exp_json = r_exp.json()
    assert len(exp_json) == 3

    # 3. Test /api/metrics
    r_met = client.get("/api/metrics")
    print(f"Metrics Status Code:    {r_met.status_code}")
    assert r_met.status_code == 200
    met_json = r_met.json()
    assert "data" in met_json
    assert "plots" in met_json
    print(f"Has real results loaded: {met_json['has_results']}")
    print("Analytics and metrics endpoints verified successfully!")
