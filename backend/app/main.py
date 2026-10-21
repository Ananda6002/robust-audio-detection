"""
backend/app/main.py
Root FastAPI application server for AcoustiGuard AI.
Configures CORS middleware, static mounts for experimental plots, global error handlers,
and system health check endpoint (GET /api/health).
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.config import MODELS_DIR, RESULTS_DIR
from ml.training_utils import get_device

# Initialize FastAPI App
app = FastAPI(
    title="AcoustiGuard AI API",
    version="1.0.0",
    description=(
        "Scientific REST API for Noise-Robust Environmental Sound Classification, "
        "Confidence Calibration, and Selective Uncertainty Handling on UrbanSound8K."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# 1. CORS Middleware Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "*",  # Local development flexibility
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Static Results Mount (Serving Plots directly to Frontend)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

# 3. Dynamic Router Inclusions
try:
    from backend.app.routes.predict import router as predict_router
    app.include_router(predict_router, prefix="/api", tags=["Inference & Prediction"])
except ImportError:
    pass

try:
    from backend.app.routes.analytics import router as analytics_router
    app.include_router(analytics_router, prefix="/api", tags=["Analytics & Model Info"])
except ImportError:
    pass


# 4. Global Exception Handlers
@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": "Resource Not Found", "detail": str(exc)},
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Invalid Input Data", "detail": str(exc)},
    )


# 5. Core Health Endpoint
@app.get("/api/health", tags=["Health & Status"])
async def health_check() -> Dict[str, Any]:
    """
    Returns service availability, active hardware acceleration, and checkpoint statuses.
    """
    dev = get_device()
    models_status = {
        "baseline_cnn": (MODELS_DIR / "baseline_cnn_best.pt").exists(),
        "augmented_cnn": (MODELS_DIR / "augmented_cnn_best.pt").exists(),
        "temperature_scaler": (MODELS_DIR / "temperature_scaler.pt").exists(),
    }
    results_status = {
        "baseline_metrics": (RESULTS_DIR / "baseline_metrics.json").exists(),
        "augmented_metrics": (RESULTS_DIR / "augmented_metrics.json").exists(),
        "experiment_comparison": (RESULTS_DIR / "experiment_comparison.json").exists(),
        "selective_prediction": (RESULTS_DIR / "selective_prediction_report.json").exists(),
    }

    return {
        "status": "online",
        "service": "AcoustiGuard AI API",
        "timestamp": time.time(),
        "device": str(dev),
        "models_available": models_status,
        "results_available": results_status,
        "ready_for_inference": models_status["baseline_cnn"] or models_status["augmented_cnn"],
    }


if __name__ == "__main__":
    from fastapi.testclient import TestClient

    print("Testing FastAPI health endpoint with TestClient...")
    client = TestClient(app)
    response = client.get("/api/health")
    print(f"Status Code: {response.status_code}")
    print("Payload:")
    for k, v in response.json().items():
        print(f"  {k}: {v}")

    assert response.status_code == 200
    assert response.json()["status"] == "online"
    print("FastAPI application initialized and verified successfully!")
