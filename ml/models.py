"""
ml/models.py
Compact, noise-resilient Convolutional Neural Network (AcousticCNN)
for environmental sound classification on Log-Mel Spectrogram representations.
Optimized for rapid CPU and GPU inference, low parameter footprint, and high generalization.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.config import ModelConfig


class ConvBlock(nn.Module):
    """
    Standard acoustic convolutional building block:
    Conv2d (3x3) -> BatchNorm2d -> ReLU -> MaxPool2d (2x2) -> Dropout2d
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.2,
        pool_size: Tuple[int, int] = (2, 2),
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,  # BatchNorm handles bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.drop = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.drop(x)
        return x


class AcousticCNN(nn.Module):
    """
    Compact 4-stage 2D Convolutional Neural Network for environmental audio classification.
    Processes log-mel spectrogram tensors of shape (B, 1, 128, 173).
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        cfg = config or ModelConfig()
        self.config = cfg

        b = cfg.base_filters  # 32
        d_conv = cfg.dropout_conv  # 0.2
        d_fc = cfg.dropout_fc  # 0.4
        hidden = cfg.fc_hidden_dim  # 256
        n_classes = cfg.num_classes  # 10

        # Stage 1: (B, 1, 128, 173) -> (B, 32, 64, 86)
        self.block1 = ConvBlock(cfg.in_channels, b, dropout=d_conv, pool_size=(2, 2))
        # Stage 2: (B, 32, 64, 86) -> (B, 64, 32, 43)
        self.block2 = ConvBlock(b, b * 2, dropout=d_conv, pool_size=(2, 2))
        # Stage 3: (B, 64, 32, 43) -> (B, 128, 16, 21)
        self.block3 = ConvBlock(b * 2, b * 4, dropout=d_conv, pool_size=(2, 2))
        # Stage 4: (B, 128, 16, 21) -> (B, 128, 8, 10)
        self.block4 = ConvBlock(b * 4, b * 4, dropout=d_conv, pool_size=(2, 2))

        # Spatial aggregation to fixed feature representation
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        flattened_dim = (b * 4) * 4 * 4  # 128 * 16 = 2048

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(flattened_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=d_fc),
            nn.Linear(hidden, n_classes),
        )

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract post-pooling feature embedding before classification."""
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.adaptive_pool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing uncalibrated class logits.
        
        Args:
            x: Input tensor of shape (B, 1, 128, 173) or (B, 128, 173).
            
        Returns:
            Logits tensor of shape (B, 10).
        """
        features = self.extract_embedding(x)
        logits = self.classifier(features)
        return logits

    @torch.no_grad()
    def predict_proba(
        self, x: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute softmax probabilities with optional temperature scaling.
        
        Args:
            x: Spectrogram input tensor.
            temperature: Scalar T > 0. T=1.0 is standard softmax;
                         calibrated T adjusts probability dispersion.
        """
        self.eval()
        logits = self.forward(x)
        scaled_logits = logits / max(1e-4, float(temperature))
        return F.softmax(scaled_logits, dim=-1)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Calculate parameter breakdown of the neural network."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "non_trainable_params": total - trainable,
    }


def estimate_model_size_mb(model: nn.Module) -> float:
    """Estimate model memory footprint in megabytes (float32 parameters and buffers)."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024.0 * 1024.0)


def benchmark_inference_latency(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 1, 128, 173),
    num_warmup: int = 5,
    num_runs: int = 25,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Benchmark forward pass inference latency in milliseconds (ms).
    """
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(*input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    latencies_ms = []
    with torch.no_grad():
        for _ in range(num_runs):
            t_start = time.perf_counter()
            _ = model(dummy_input)
            t_end = time.perf_counter()
            latencies_ms.append((t_end - t_start) * 1000.0)

    return {
        "mean_latency_ms": float(torch.tensor(latencies_ms).mean().item()),
        "p50_latency_ms": float(torch.tensor(latencies_ms).median().item()),
        "p95_latency_ms": float(torch.tensor(latencies_ms).quantile(0.95).item()),
        "min_latency_ms": float(min(latencies_ms)),
        "max_latency_ms": float(max(latencies_ms)),
        "device": device,
    }


if __name__ == "__main__":
    print("Testing AcousticCNN architecture...")
    model = AcousticCNN()
    params = count_parameters(model)
    size_mb = estimate_model_size_mb(model)

    print(f"Total Parameters: {params['total_params']:,}")
    print(f"Trainable Parameters: {params['trainable_params']:,}")
    print(f"Model Memory Footprint: {size_mb:.2f} MB")

    # 1. Forward Pass Check
    dummy_x = torch.randn(4, 1, 128, 173)
    logits = model(dummy_x)
    print(f"Forward output logits shape: {logits.shape}")
    assert logits.shape == (4, 10), f"Expected (4, 10), got {logits.shape}"
    assert not torch.isnan(logits).any(), "Found NaNs in logits"

    # 2. Probability & Temperature Scaling Check
    probs = model.predict_proba(dummy_x, temperature=1.5)
    print(f"Probability output shape: {probs.shape}")
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5), "Probabilities must sum to 1.0"

    # 3. Backward Pass & Gradient Flow
    criterion = nn.CrossEntropyLoss()
    target = torch.tensor([0, 3, 7, 9], dtype=torch.long)
    loss = criterion(logits, target)
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0, "No gradients computed"
    print(f"Gradient check passed: computed {len(grad_norms)} parameter gradients. Loss: {loss.item():.4f}")

    # 4. Latency Benchmark
    latency_report = benchmark_inference_latency(model, device="cpu")
    print(f"CPU Inference Latency: {latency_report['mean_latency_ms']:.2f} ms (P95: {latency_report['p95_latency_ms']:.2f} ms)")

    print("AcousticCNN model verification completed successfully!")
