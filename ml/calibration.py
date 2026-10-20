"""
ml/calibration.py
Confidence calibration module for deep acoustic classifiers.
Implements Temperature Scaling (Guo et al., 2017), Expected Calibration Error (ECE),
Maximum Calibration Error (MCE), and reliability diagram visualization.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Use headless backend for Matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.config import MODELS_DIR, RESULTS_DIR


def compute_ece(
    y_true: Union[List[int], np.ndarray],
    y_probs: Union[List[List[float]], np.ndarray],
    num_bins: int = 10,
) -> Dict[str, Any]:
    """
    Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    
    Partitions predictions into M confidence intervals:
        ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
        MCE = max_{m=1}^M |acc(B_m) - conf(B_m)|
        
    Args:
        y_true: Ground truth integer class labels of shape (N,).
        y_probs: Predicted class probabilities of shape (N, num_classes).
        num_bins: Number of confidence bins (standard is 10 or 15).
        
    Returns:
        Dictionary containing ECE, MCE, and per-bin diagnostics.
    """
    targets = np.asarray(y_true, dtype=int)
    probs = np.asarray(y_probs, dtype=float)

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets).astype(float)

    n_samples = len(targets)
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)

    ece = 0.0
    mce = 0.0

    bin_accs: List[float] = []
    bin_confs: List[float] = []
    bin_counts: List[int] = []
    bin_edges: List[Tuple[float, float]] = []

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Samples within confidence bin [lower, upper)
        if i == num_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        count = int(np.sum(in_bin))
        bin_counts.append(count)
        bin_edges.append((float(bin_lower), float(bin_upper)))

        if count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            gap = abs(bin_acc - bin_conf)

            ece += (count / n_samples) * gap
            mce = max(mce, gap)

            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2.0)

    return {
        "ece": float(ece),
        "mce": float(mce),
        "num_bins": num_bins,
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
        "bin_edges": bin_edges,
        "total_samples": n_samples,
    }


class TemperatureScaler(nn.Module):
    """
    Post-hoc confidence calibration via learned scalar temperature T > 0.
    Calibrated probabilities: p_i = softmax(z_i / T)
    Preserves exact argmax predictions while correcting overconfident probabilities.
    """

    def __init__(self, initial_temp: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * float(initial_temp))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature parameter."""
        temp = torch.clamp(self.temperature, min=0.05, max=10.0)
        return logits / temp

    @torch.no_grad()
    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute calibrated probabilities."""
        scaled = self.forward(logits)
        return torch.softmax(scaled, dim=-1)

    def fit(
        self,
        logits: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
        max_iter: int = 50,
        lr: float = 0.02,
    ) -> float:
        """
        Fit temperature parameter on validation logits by minimizing Cross-Entropy Loss (NLL).
        
        Args:
            logits: Unscaled validation logits of shape (N, C).
            labels: Ground truth validation labels of shape (N,).
            
        Returns:
            Optimal temperature scalar.
        """
        logits_t = torch.as_tensor(logits, dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.long)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits_t), labels_t)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        # Constrain to valid positive range
        with torch.no_grad():
            self.temperature.clamp_(min=0.05, max=10.0)

        return float(self.temperature.item())

    def save(self, filepath: Union[str, Path]) -> Path:
        """Persist calibrated temperature scalar to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"temperature": float(self.temperature.item())}, path)
        return path

    def load(self, filepath: Union[str, Path]) -> float:
        """Load calibrated temperature scalar from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Temperature scaler checkpoint not found: {path}")
        data = torch.load(path, map_location="cpu")
        val = float(data.get("temperature", 1.0))
        with torch.no_grad():
            self.temperature.copy_(torch.tensor([val]))
        return val


def calibrate_model_from_loader(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[TemperatureScaler, Dict[str, Any]]:
    """
    Extract validation set logits and fit a TemperatureScaler.
    
    Returns:
        (fitted_scaler, calibration_diagnostics)
    """
    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for features, targets, _ in val_loader:
            features = features.to(device)
            logits = model(features)
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    val_logits = torch.cat(all_logits, dim=0)
    val_targets = torch.cat(all_targets, dim=0)

    uncal_probs = torch.softmax(val_logits, dim=-1).numpy()
    uncal_ece = compute_ece(val_targets.numpy(), uncal_probs)

    scaler = TemperatureScaler(initial_temp=1.5)
    opt_t = scaler.fit(val_logits, val_targets)

    cal_probs = scaler.predict_proba(val_logits).numpy()
    cal_ece = compute_ece(val_targets.numpy(), cal_probs)

    diagnostics = {
        "optimal_temperature": opt_t,
        "uncalibrated_ece": uncal_ece["ece"],
        "calibrated_ece": cal_ece["ece"],
        "ece_reduction": float(uncal_ece["ece"] - cal_ece["ece"]),
        "uncalibrated_mce": uncal_ece["mce"],
        "calibrated_mce": cal_ece["mce"],
        "uncalibrated_diagnostics": uncal_ece,
        "calibrated_diagnostics": cal_ece,
    }

    return scaler, diagnostics


def plot_reliability_diagram(
    uncal_ece_data: Dict[str, Any],
    cal_ece_data: Optional[Dict[str, Any]],
    output_path: Union[str, Path],
    title: str = "Reliability Diagram (Confidence Calibration)",
) -> Path:
    """
    Plot confidence vs. accuracy reliability diagrams comparing uncalibrated vs. calibrated probabilities.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    # Ideal calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=1.5, alpha=0.7)

    # Uncalibrated curve
    confs_uncal = uncal_ece_data["bin_confidences"]
    accs_uncal = uncal_ece_data["bin_accuracies"]
    ece_uncal = uncal_ece_data["ece"]
    ax.plot(
        confs_uncal,
        accs_uncal,
        "s-",
        color="#dc2626",
        label=f"Uncalibrated (ECE = {ece_uncal*100:.2f}%)",
        linewidth=2,
        markersize=6,
    )

    # Calibrated curve (if available)
    if cal_ece_data is not None:
        confs_cal = cal_ece_data["bin_confidences"]
        accs_cal = cal_ece_data["bin_accuracies"]
        ece_cal = cal_ece_data["ece"]
        ax.plot(
            confs_cal,
            accs_cal,
            "o-",
            color="#16a34a",
            label=f"Temperature Scaled (ECE = {ece_cal*100:.2f}%)",
            linewidth=2,
            markersize=6,
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Predicted Confidence", fontsize=11)
    ax.set_ylabel("Empirical Accuracy", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, loc="upper left")

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("Testing confidence calibration module...")
    rng = np.random.default_rng(42)

    # 1. Synthesize overconfident predictions (common in deep neural nets)
    num_samples = 300
    num_classes = 10
    synthetic_targets = rng.integers(0, num_classes, size=num_samples)

    # Generate logits with artificially high magnitude (causing overconfidence)
    synthetic_logits = rng.normal(0, 1.0, size=(num_samples, num_classes))
    for i, t in enumerate(synthetic_targets):
        if rng.random() > 0.4:  # 60% accurate, but logits will predict with ~90% confidence
            synthetic_logits[i, t] += 4.5
        else:
            wrong_class = (t + 1) % num_classes
            synthetic_logits[i, wrong_class] += 4.5

    raw_probs = torch.softmax(torch.from_numpy(synthetic_logits).float(), dim=-1).numpy()
    uncal_report = compute_ece(synthetic_targets, raw_probs)
    print(f"Uncalibrated Expected Calibration Error (ECE): {uncal_report['ece']*100:.2f}%")
    print(f"Uncalibrated Maximum Calibration Error (MCE):    {uncal_report['mce']*100:.2f}%")

    # 2. Fit Temperature Scaler
    scaler = TemperatureScaler(initial_temp=1.0)
    opt_temp = scaler.fit(synthetic_logits, synthetic_targets)
    print(f"Optimized Temperature (T): {opt_temp:.3f}")
    assert opt_temp > 1.0, f"Expected temperature > 1.0 to soften overconfident logits, got {opt_temp}"

    cal_probs = scaler.predict_proba(torch.from_numpy(synthetic_logits).float()).numpy()
    cal_report = compute_ece(synthetic_targets, cal_probs)
    print(f"Calibrated Expected Calibration Error (ECE):   {cal_report['ece']*100:.2f}%")
    print(f"Calibrated Maximum Calibration Error (MCE):      {cal_report['mce']*100:.2f}%")
    assert cal_report["ece"] < uncal_report["ece"], "Temperature scaling failed to reduce ECE!"

    # 3. Test Scaler Persistence
    test_scaler_path = MODELS_DIR / "smoke_temperature_scaler.pt"
    scaler.save(test_scaler_path)
    new_scaler = TemperatureScaler(initial_temp=1.0)
    loaded_t = new_scaler.load(test_scaler_path)
    assert np.isclose(loaded_t, opt_temp, atol=1e-4)
    if test_scaler_path.exists():
        test_scaler_path.unlink()

    # 4. Test Reliability Diagram Plotting
    diag_path = RESULTS_DIR / "smoke_reliability_diagram.png"
    plot_reliability_diagram(uncal_report, cal_report, diag_path)
    assert diag_path.exists() and diag_path.stat().st_size > 0
    diag_path.unlink()

    print("All calibration and reliability checks passed successfully!")
