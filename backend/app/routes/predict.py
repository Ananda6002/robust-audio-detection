"""
backend/app/routes/predict.py
Audio prediction and robustness testing endpoints.
Implements secure audio ingestion, exact training preprocessing, model caching,
temperature-calibrated inference, and confidence-aware selective classification.
"""

import io
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import scipy.io.wavfile
import torch
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.augmentation import inject_noise_at_snr
from ml.calibration import TemperatureScaler
from ml.config import (
    CLASS_MAP,
    CLASS_NAMES,
    ConfidenceConfig,
    MODELS_DIR,
    ModelConfig,
)
from ml.features import extract_features
from ml.models import AcousticCNN
from ml.preprocessing import preprocess_waveform
from ml.training_utils import get_device, load_checkpoint

router = APIRouter()

# Max audio upload size: 15 MB
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac"}

# Global model cache to avoid reloading weights per inference request
_MODEL_CACHE: Dict[str, AcousticCNN] = {}
_SCALER_CACHE: Optional[TemperatureScaler] = None


# ==========================================
# Pydantic Response Schemas
# ==========================================

class ClassProbability(BaseModel):
    class_id: int
    class_name: str
    probability: float


class PredictionResponse(BaseModel):
    predicted_class_id: int
    predicted_class_name: str
    confidence: float
    is_uncertain: bool
    uncertainty_status: str  # "confident" or "uncertain"
    status_message: str
    threshold_applied: float
    probabilities: List[ClassProbability]
    model_used: str
    calibrated: bool
    temperature: float
    inference_latency_ms: float
    waveform_preview: List[float]  # Subsampled waveform for UI plotting
    spectrogram_preview: List[List[float]]  # Downsampled spectrogram grid for UI heatmap


class RobustnessTestResponse(BaseModel):
    snr_db: float
    noise_type: str
    confidence_drop: float
    prediction_changed: bool
    summary_verdict: str
    clean_prediction: PredictionResponse
    noisy_prediction: PredictionResponse


# ==========================================
# Model Loading Helpers
# ==========================================

def get_or_load_model(model_type: str = "augmented") -> Tuple[AcousticCNN, str]:
    """
    Retrieve cached AcousticCNN model or load weights from disk on demand.
    """
    device = get_device()
    cache_key = f"{model_type}_{device}"

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key], model_type

    # Resolve preferred and fallback checkpoint paths
    if model_type == "augmented":
        pref_path = MODELS_DIR / "augmented_cnn_best.pt"
        fall_path = MODELS_DIR / "baseline_cnn_best.pt"
    else:
        pref_path = MODELS_DIR / "baseline_cnn_best.pt"
        fall_path = MODELS_DIR / "augmented_cnn_best.pt"

    target_path = pref_path if pref_path.exists() else fall_path
    if not target_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Model checkpoint not found on server ({pref_path.name} or {fall_path.name}). "
                "Please train the model first using the training scripts."
            ),
        )

    model = AcousticCNN(ModelConfig()).to(device)
    load_checkpoint(target_path, model, device=device)
    model.eval()

    _MODEL_CACHE[cache_key] = model
    actual_type = "augmented" if "augmented" in target_path.name else "baseline"
    return model, actual_type


def get_or_load_scaler() -> Optional[TemperatureScaler]:
    """Retrieve or load cached TemperatureScaler."""
    global _SCALER_CACHE
    if _SCALER_CACHE is not None:
        return _SCALER_CACHE

    scaler_path = MODELS_DIR / "temperature_scaler.pt"
    if scaler_path.exists():
        scaler = TemperatureScaler()
        scaler.load(scaler_path)
        _SCALER_CACHE = scaler
        return scaler
    return None


def run_single_inference(
    waveform: np.ndarray,
    model_type: str = "augmented",
    threshold: float = ConfidenceConfig.default_threshold,
    use_calibration: bool = True,
) -> PredictionResponse:
    """
    Execute end-to-end inference over a normalized 88,200-sample waveform array.
    """
    t_start = time.perf_counter()
    device = get_device()
    model, actual_model_type = get_or_load_model(model_type)

    # 1. Feature Extraction: shape (1, 128, 173)
    features_np = extract_features(waveform)
    features_tensor = torch.from_numpy(features_np).unsqueeze(0).to(device)  # (1, 1, 128, 173)

    # 2. Forward Pass
    with torch.no_grad():
        logits = model(features_tensor)

        temperature = 1.0
        scaler = get_or_load_scaler() if use_calibration else None

        if scaler is not None and use_calibration:
            probs_tensor = scaler.predict_proba(logits.cpu())
            temperature = float(scaler.temperature.item())
            is_calibrated = True
        else:
            probs_tensor = torch.softmax(logits.cpu(), dim=-1)
            is_calibrated = False

    latency_ms = (time.perf_counter() - t_start) * 1000.0

    probs_arr = probs_tensor.squeeze(0).numpy()
    pred_class_id = int(np.argmax(probs_arr))
    pred_class_name = CLASS_MAP[pred_class_id]
    confidence = float(probs_arr[pred_class_id])

    # 3. Confidence-Aware Selective Check
    is_uncertain = bool(confidence < threshold)
    if is_uncertain:
        uncertainty_status = "uncertain"
        status_message = ConfidenceConfig.uncertain_label
    else:
        uncertainty_status = "confident"
        status_message = f"Reliable detection: {pred_class_name} ({confidence*100:.1f}%)"

    probabilities_list = [
        ClassProbability(
            class_id=i,
            class_name=CLASS_NAMES[i],
            probability=float(probs_arr[i]),
        )
        for i in range(len(CLASS_NAMES))
    ]
    # Sort descending by probability
    probabilities_list.sort(key=lambda x: x.probability, reverse=True)

    # 4. Downsampled Previews for UI Rendering
    # Waveform preview: 120 points
    step_wave = max(1, len(waveform) // 120)
    waveform_subsampled = [float(val) for val in waveform[::step_wave][:120]]

    # Spectrogram preview: 32 frequency bins x 32 time frames
    spec_2d = features_np.squeeze(0)  # (128, 173)
    row_step = max(1, spec_2d.shape[0] // 32)
    col_step = max(1, spec_2d.shape[1] // 32)
    spec_preview = spec_2d[::row_step, ::col_step][:32, :32].tolist()

    return PredictionResponse(
        predicted_class_id=pred_class_id,
        predicted_class_name=pred_class_name,
        confidence=confidence,
        is_uncertain=is_uncertain,
        uncertainty_status=uncertainty_status,
        status_message=status_message,
        threshold_applied=threshold,
        probabilities=probabilities_list,
        model_used=f"AcousticCNN ({actual_model_type})",
        calibrated=is_calibrated,
        temperature=temperature,
        inference_latency_ms=float(latency_ms),
        waveform_preview=waveform_subsampled,
        spectrogram_preview=spec_preview,
    )


# ==========================================
# Audio Upload Validation & Processing
# ==========================================

async def validate_and_preprocess_upload(file: UploadFile) -> np.ndarray:
    """Read, validate, and preprocess uploaded audio file."""
    # Check extension
    filename = file.filename or "recording.wav"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format '{ext}'. Allowed formats: {sorted(list(ALLOWED_EXTENSIONS))}",
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty (0 bytes).",
        )
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size limit of {MAX_UPLOAD_BYTES // (1024*1024)} MB.",
        )

    # Save to temp file and preprocess
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        waveform = preprocess_waveform(tmp_path)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to decode audio file: {str(exc)}",
        ) from exc
    finally:
        if tmp_path.exists():
            os.remove(tmp_path)

    return waveform


# ==========================================
# REST Endpoints
# ==========================================

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify Audio Recording",
    description="Upload a WAV/MP3 recording to receive sound class probabilities, confidence, and uncertainty status.",
)
async def predict_audio(
    file: UploadFile = File(..., description="Audio recording file (WAV, MP3, FLAC, OGG)"),
    model_type: str = Form("augmented", description="'augmented' or 'baseline'"),
    threshold: float = Form(0.70, description="Confidence abstention threshold (0.5 - 0.95)"),
    use_calibration: bool = Form(True, description="Enable temperature scaling calibration"),
) -> PredictionResponse:
    waveform = await validate_and_preprocess_upload(file)
    clamped_thresh = float(np.clip(threshold, 0.1, 0.99))
    return run_single_inference(
        waveform=waveform,
        model_type=model_type,
        threshold=clamped_thresh,
        use_calibration=use_calibration,
    )


@router.post(
    "/robustness-test",
    response_model=RobustnessTestResponse,
    summary="Interactive Robustness Demonstration",
    description="Inject controlled synthetic background noise at a designated SNR and evaluate clean vs noisy side-by-side.",
)
async def robustness_test(
    file: UploadFile = File(..., description="Clean audio recording"),
    snr_db: float = Form(10.0, description="Target Signal-to-Noise Ratio (dB)"),
    noise_type: str = Form("ambient", description="Noise profile ('ambient', 'pink', 'brown', 'white')"),
    model_type: str = Form("augmented", description="'augmented' or 'baseline'"),
    threshold: float = Form(0.70, description="Uncertainty threshold"),
) -> RobustnessTestResponse:
    clean_waveform = await validate_and_preprocess_upload(file)

    # Inject exact synthetic noise
    noisy_waveform, _ = inject_noise_at_snr(
        clean_waveform,
        snr_db=snr_db,
        noise_type=noise_type,
    )

    clean_res = run_single_inference(clean_waveform, model_type=model_type, threshold=threshold)
    noisy_res = run_single_inference(noisy_waveform, model_type=model_type, threshold=threshold)

    conf_drop = float(clean_res.confidence - noisy_res.confidence)
    pred_changed = bool(clean_res.predicted_class_id != noisy_res.predicted_class_id)

    if pred_changed:
        verdict = f"Prediction changed from {clean_res.predicted_class_name} to {noisy_res.predicted_class_name} under {snr_db} dB {noise_type} noise."
    elif noisy_res.is_uncertain:
        verdict = f"Class preserved ({clean_res.predicted_class_name}), but confidence dropped below threshold ({noisy_res.confidence*100:.1f}% < {threshold*100:.0f}%). Marked uncertain."
    else:
        verdict = f"Classification robust to {snr_db} dB {noise_type} noise. Maintained confident prediction."

    return RobustnessTestResponse(
        snr_db=float(snr_db),
        noise_type=noise_type,
        confidence_drop=conf_drop,
        prediction_changed=pred_changed,
        summary_verdict=verdict,
        clean_prediction=clean_res,
        noisy_prediction=noisy_res,
    )


if __name__ == "__main__":
    from fastapi.testclient import TestClient
    from backend.app.main import app

    print("Testing /api/predict and /api/robustness-test endpoints...")
    client = TestClient(app)

    # 1. Synthesize a clean 16-bit WAV file in memory
    sr = 22050
    t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
    synthetic_audio = (np.sin(2 * np.pi * 500.0 * t) * 32767).astype(np.int16)

    wav_io = io.BytesIO()
    scipy.io.wavfile.write(wav_io, sr, synthetic_audio)
    wav_bytes = wav_io.getvalue()

    # 2. Test /api/predict
    response = client.post(
        "/api/predict",
        files={"file": ("test_tone.wav", wav_bytes, "audio/wav")},
        data={"model_type": "augmented", "threshold": "0.70", "use_calibration": "true"},
    )
    print(f"Predict Status Code: {response.status_code}")
    assert response.status_code == 200
    pred_data = response.json()
    print(f"Predicted Class: {pred_data['predicted_class_name']}")
    print(f"Confidence:      {pred_data['confidence']*100:.2f}%")
    print(f"Status:          {pred_data['status_message']}")
    print(f"Latency:         {pred_data['inference_latency_ms']:.2f} ms")
    assert len(pred_data["probabilities"]) == 10
    assert len(pred_data["waveform_preview"]) > 0

    # 3. Test /api/robustness-test
    rob_resp = client.post(
        "/api/robustness-test",
        files={"file": ("test_tone.wav", wav_bytes, "audio/wav")},
        data={"snr_db": "10.0", "noise_type": "ambient", "threshold": "0.70"},
    )
    print(f"Robustness Status Code: {rob_resp.status_code}")
    assert rob_resp.status_code == 200
    rob_data = rob_resp.json()
    print(f"Verdict:         {rob_data['summary_verdict']}")
    print(f"Confidence Drop: {rob_data['confidence_drop']*100:.2f}%")

    print("Prediction and robustness endpoints verified successfully!")
