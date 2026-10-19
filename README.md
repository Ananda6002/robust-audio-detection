# AcoustiGuard AI: Noise-Robust Environmental Sound Classification

A research-oriented deep learning system for robust environmental sound classification under variable signal-to-noise ratios (SNR), featuring confidence-aware prediction, temperature-scaled calibration, and selective classification.

---

## 1. Research Problem & Objectives

### Research Question
> *"Can noise augmentation improve the robustness of environmental sound classification under varying signal-to-noise ratios, and can confidence-aware prediction reliably detect and reject degraded or out-of-distribution audio inputs?"*

### Objectives
1. **Baseline Sound Classification:** Train a compact Convolutional Neural Network (CNN) on log-mel spectrogram representations of urban environmental audio using the official UrbanSound8K fold protocol.
2. **Noise Robustness Evaluation:** Quantify classification degradation across controlled Signal-to-Noise Ratios (20 dB, 10 dB, 5 dB, 0 dB) using additive ambient background noise.
3. **Noise Augmentation Defense:** Train noise-augmented models with training-only additive noise (preserving strict test-set isolation) and measure generalization gains across SNRs.
4. **Confidence Calibration & Selective Classification:** Apply temperature scaling calibration on validation data and introduce an uncertainty threshold mechanism to abstain from low-confidence predictions ("Uncertain — manual review recommended").
5. **Interactive Scientific Application:** Deploy the trained pipeline with a high-performance FastAPI backend and a React/Vite/Tailwind CSS web dashboard ("AcoustiGuard AI") with real-time waveform/spectrogram rendering and robustness testing labs.

---

## 2. Dataset: UrbanSound8K

The primary benchmark used is **UrbanSound8K**:
* **Source:** [UrbanSound8K Official Dataset](https://urbansounddataset.weebly.com/urbansound8k.html)
* **Scale:** 8,732 audio excerpts (<= 4.0s duration) across 10 urban sound classes:
  0. `air_conditioner`
  1. `car_horn`
  2. `children_playing`
  3. `dog_bark`
  4. `drilling`
  5. `engine_idling`
  6. `gun_shot`
  7. `jackhammer`
  8. `siren`
  9. `street_music`
* **Predefined Folds:** 10 predefined folds (`fold1` to `fold10`). 
* **Leakage Prevention Rule:** Audio slices from the same original recording are placed in the same fold. **Never randomly mix slices across folds.** All cross-validation and train/validation/test splits strictly adhere to the predefined fold structure.

### Dataset Directory Setup
Download and extract UrbanSound8K into the `data/` directory:
```
robust_audio_detection/
├── data/
│   └── UrbanSound8K/
│       ├── audio/
│       │   ├── fold1/
│       │   ├── fold2/
│       │   └── ... fold10/
│       └── metadata/
│           └── UrbanSound8K.csv
```
> **Note:** The dataset and model checkpoints (`.pt`, `.pth`) are excluded from Git via `.gitignore`.

---

## 3. Repository Architecture

```
robust_audio_detection/
├── backend/               # FastAPI application & REST endpoints
│   ├── app/
│   │   ├── main.py        # API server entrypoint
│   │   ├── routes/        # /api/predict, /api/robustness-test, /api/metrics
│   │   └── schemas/       # Pydantic request/response models
├── frontend/              # AcoustiGuard AI web application (React, Vite, TS, Tailwind)
│   ├── src/
│   │   ├── components/    # Audio upload, spectrogram visualizer, robustness lab
│   │   └── pages/         # Dashboard, Lab, Experiment Analytics, Model Info
├── ml/                    # Core deep learning pipeline
│   ├── config.py          # Centralized audio, model, and training hyperparameters
│   ├── dataset.py         # PyTorch Dataset, metadata loader, fold partitioner
│   ├── preprocessing.py   # Audio normalization, resampling, fixed-duration padding/cropping
│   ├── features.py        # Log-mel spectrogram extraction
│   ├── models.py          # Compact CNN architecture (AcousticCNN)
│   └── augmentation.py    # Training-time noise injection and SNR scaling
├── experiments/           # Experiment runners & calibration
│   ├── train_baseline.py  # Exp A: Baseline training on clean audio
│   ├── train_augmented.py # Exp B: Noise-augmented training
│   ├── evaluate_noise.py  # Benchmark clean vs noisy across SNRs (20, 10, 5, 0 dB)
│   └── calibrate.py       # Exp C: Temperature scaling & selective prediction
├── research/              # Literature review, citations, research report draft
├── docs/                  # Architecture specs, evaluation protocols, API reference
├── tests/                 # Unit and integration test suites (Pytest)
├── results/               # Real evaluation metrics JSONs, confusion matrices, loss curves
├── requirements.txt       # Python dependencies
└── README.md
```

---

## 4. Experimental Framework

| Experiment | Description | Training Condition | Evaluation Conditions |
|---|---|---|---|
| **Experiment A (Baseline)** | Standard Compact CNN | Clean audio only | Clean audio + Noisy (20, 10, 5, 0 dB) |
| **Experiment B (Augmented)** | Noise-Augmented CNN | Clean + Additive Noise (Random SNR 5-25 dB) | Clean audio + Noisy (20, 10, 5, 0 dB) |
| **Experiment C (Confidence-Aware)** | Calibrated Prediction + Abstention | Temperature-scaled logits on validation fold | Selective coverage vs. risk at $\tau \in [0.5, 0.95]$ |

### Evaluation Metrics
- **Classification Performance:** Overall Accuracy, Macro Precision, Macro Recall, Macro F1-score, Per-Class F1.
- **Robustness:** Macro F1 vs SNR curves.
- **Operational Metrics:** Model Parameter Count, Model Size (MB), Inference Latency (CPU vs GPU in ms).
- **Uncertainty Metrics:** Expected Calibration Error (ECE), Coverage Rate (% answered), Selective Risk (Error rate on answered queries).

---

## 5. Quickstart & Installation

### Local Environment Setup
```powershell
# 1. Clone repository and navigate to root
cd robust_audio_detection

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install ML and backend dependencies
pip install -r requirements.txt
```

### Running Tests
```powershell
pytest tests/ -v
```

### Starting Backend & Frontend
```powershell
# Backend (Port 8000)
uvicorn backend.app.main:app --reload --port 8000

# Frontend (Port 5173)
cd frontend
npm install
npm run dev
```

---

## 6. Citation & Attribution
UrbanSound8K dataset citation:
> J. Salamon, C. Jacoby and J. P. Bello, *"A Dataset and Taxonomy for Urban Sound Research"*, 22nd ACM International Conference on Multimedia (ACM-MM'14), Orlando, FL, USA, Nov. 2014.
