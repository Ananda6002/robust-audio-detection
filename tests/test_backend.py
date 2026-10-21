"""
tests/test_backend.py
Comprehensive integration test suite for the FastAPI backend REST API.
Validates all endpoints (/api/health, /api/model-info, /api/experiments,
/api/metrics, /api/predict, /api/robustness-test, and /static/results/*).
"""

import io
import sys
from pathlib import Path
import numpy as np
import pytest
import scipy.io.wavfile
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.app.main import app
from ml.config import CLASS_NAMES, RESULTS_DIR


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


@pytest.fixture(scope="module")
def test_wav_bytes():
    """Generate in-memory valid 16-bit PCM WAV audio for tests."""
    sr = 22050
    duration = 2.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (np.sin(2 * np.pi * 500.0 * t) * 32767).astype(np.int16)

    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, sr, audio)
    return buf.getvalue()


# ==========================================
# 1. System Health & Metadata Tests
# ==========================================

def test_health_endpoint(client):
    """Verify /api/health returns online status and active device."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "device" in data
    assert "models_available" in data
    assert "ready_for_inference" in data


def test_model_info_endpoint(client):
    """Verify /api/model-info returns 10-class taxonomy and architecture parameters."""
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["taxonomy"]["num_classes"] == 10
    assert data["taxonomy"]["classes"] == CLASS_NAMES
    assert data["parameters"]["total_parameters"] > 100_000
    assert data["input_representation"]["tensor_shape"] == [1, 128, 173]
    assert len(data["known_limitations"]) > 0


def test_experiments_endpoint(client):
    """Verify /api/experiments returns all 3 research experiments."""
    response = client.get("/api/experiments")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 3
    exp_ids = [exp["id"] for exp in data]
    assert "exp_a" in exp_ids
    assert "exp_b" in exp_ids
    assert "exp_c" in exp_ids


def test_metrics_endpoint(client):
    """Verify /api/metrics aggregates real research results and plot URLs."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "data" in data
    assert "plots" in data
    assert "comparison" in data["data"]


# ==========================================
# 2. Audio Prediction Endpoint Tests
# ==========================================

def test_predict_valid_audio(client, test_wav_bytes):
    """Verify /api/predict processes audio and returns 10-class probability distribution."""
    response = client.post(
        "/api/predict",
        files={"file": ("recording.wav", test_wav_bytes, "audio/wav")},
        data={"model_type": "augmented", "threshold": "0.70", "use_calibration": "true"},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["predicted_class_name"] in CLASS_NAMES
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["uncertainty_status"] in ("confident", "uncertain")
    assert isinstance(data["is_uncertain"], bool)

    # Verify probability distribution
    probs = data["probabilities"]
    assert len(probs) == 10
    prob_sum = sum(p["probability"] for p in probs)
    assert np.isclose(prob_sum, 1.0, atol=1e-3)

    # Verify visualization previews
    assert len(data["waveform_preview"]) > 0
    assert len(data["spectrogram_preview"]) == 32
    assert len(data["spectrogram_preview"][0]) == 32


def test_predict_invalid_extension(client):
    """Verify /api/predict rejects non-audio file extensions with 400 Bad Request."""
    dummy_text = b"This is not an audio file"
    response = client.post(
        "/api/predict",
        files={"file": ("document.txt", dummy_text, "text/plain")},
        data={"model_type": "augmented", "threshold": "0.70"},
    )
    assert response.status_code == 400
    assert "Unsupported audio format" in response.json()["detail"]


def test_predict_empty_file(client):
    """Verify /api/predict rejects empty audio files with 400 Bad Request."""
    response = client.post(
        "/api/predict",
        files={"file": ("empty.wav", b"", "audio/wav")},
        data={"model_type": "augmented", "threshold": "0.70"},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


# ==========================================
# 3. Robustness Lab & Static Asset Tests
# ==========================================

def test_robustness_test_endpoint(client, test_wav_bytes):
    """Verify /api/robustness-test evaluates clean vs noisy side-by-side."""
    response = client.post(
        "/api/robustness-test",
        files={"file": ("input.wav", test_wav_bytes, "audio/wav")},
        data={"snr_db": "10.0", "noise_type": "ambient", "threshold": "0.70"},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["snr_db"] == 10.0
    assert data["noise_type"] == "ambient"
    assert "clean_prediction" in data
    assert "noisy_prediction" in data
    assert isinstance(data["confidence_drop"], float)
    assert isinstance(data["prediction_changed"], bool)
    assert len(data["summary_verdict"]) > 0


def test_static_results_serving(client):
    """Verify static file mount serves existing plot assets with correct content-type."""
    # Find any existing image in RESULTS_DIR
    png_files = list(RESULTS_DIR.glob("*.png"))
    if len(png_files) > 0:
        sample_png = png_files[0]
        url = f"/static/results/{sample_png.name}"
        response = client.get(url)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("image/png")
