"""
ml/training_utils.py
Training utilities, reproducibility managers, checkpoint persistence,
and standardized optimization loops for baseline and experimental models.
"""

import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.config import MODELS_DIR, TrainingConfig


def set_seed(seed: int = TrainingConfig.seed) -> None:
    """
    Enforce deterministic execution across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preferred: Optional[str] = None) -> torch.device:
    """
    Detect and return the compute device (CUDA GPU or CPU).
    """
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class EarlyStopping:
    """
    Monitors a validation metric and flags when training should terminate.
    Maintains a copy of the best performing weights.
    """

    def __init__(
        self,
        patience: int = 8,
        min_delta: float = 1e-4,
        mode: str = "min",
    ):
        """
        Args:
            patience: Number of epochs to wait without improvement before stopping.
            min_delta: Minimum change required to qualify as an improvement.
            mode: "min" for metrics like loss; "max" for metrics like accuracy/F1.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None

    def __call__(self, current_metric: float, model: nn.Module) -> bool:
        """
        Update tracker with current epoch metric.
        Returns True if score improved, False otherwise.
        """
        score = -current_metric if self.mode == "min" else current_metric

        if self.best_score is None:
            self.best_score = score
            self.best_state_dict = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            return True

        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.best_state_dict = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            self.counter = 0
            return True

    def restore_best_weights(self, model: nn.Module) -> None:
        """Restore the best cached state dictionary into the model."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[optim.Optimizer],
    epoch: int,
    metrics: Dict[str, Any],
    filepath: Path,
) -> Path:
    """
    Save model checkpoint with optimizer state and metadata.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "metrics": metrics,
        "timestamp": time.time(),
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load weights and optimizer state from checkpoint into existing model.
    """
    target_device = device or get_device()
    checkpoint = torch.load(filepath, map_location=target_device)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Execute one training epoch over the dataloader.
    
    Returns:
        (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for features, targets, _ in dataloader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total_samples += batch_size

    epoch_loss = total_loss / max(1, total_samples)
    epoch_acc = correct / max(1, total_samples)
    return float(epoch_loss), float(epoch_acc)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Execute one evaluation pass over the dataloader.
    
    Returns:
        (val_loss, val_accuracy, all_predictions, all_targets)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    total_samples = 0

    for features, targets, _ in dataloader:
        features = features.to(device)
        targets = targets.to(device)

        logits = model(features)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(targets.cpu().numpy().tolist())
        total_samples += batch_size

    val_loss = total_loss / max(1, total_samples)
    preds_arr = np.array(all_preds, dtype=np.int64)
    targets_arr = np.array(all_targets, dtype=np.int64)
    val_acc = float((preds_arr == targets_arr).mean()) if len(targets_arr) > 0 else 0.0

    return float(val_loss), float(val_acc), preds_arr, targets_arr


if __name__ == "__main__":
    print("Testing training utilities...")
    set_seed(42)
    dev = get_device()
    print(f"Active Device: {dev}")

    # 1. Test Model and Optimizer initialization
    from ml.models import AcousticCNN
    model = AcousticCNN().to(dev)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 2. Checkpoint Save and Load Test
    test_ckpt_path = MODELS_DIR / "test_smoke_checkpoint.pt"
    saved_path = save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=1,
        metrics={"loss": 0.45, "accuracy": 0.88},
        filepath=test_ckpt_path,
    )
    assert saved_path.exists(), "Checkpoint file was not written"

    new_model = AcousticCNN().to(dev)
    loaded = load_checkpoint(test_ckpt_path, new_model, device=dev)
    assert loaded["epoch"] == 1
    assert loaded["metrics"]["accuracy"] == 0.88

    # Assert weights identical
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2), "Loaded weights do not match original"

    # Cleanup smoke checkpoint
    if test_ckpt_path.exists():
        test_ckpt_path.unlink()

    # 3. EarlyStopping Logic Test
    stopper = EarlyStopping(patience=2, mode="min")
    assert stopper(1.0, model) is True  # epoch 1: best=1.0
    assert stopper(0.8, model) is True  # epoch 2: best=0.8
    assert stopper(0.85, model) is False  # epoch 3: counter=1
    assert stopper(0.84, model) is False  # epoch 4: counter=2 -> early_stop=True
    assert stopper.early_stop is True

    print("Training utilities verification completed successfully!")
