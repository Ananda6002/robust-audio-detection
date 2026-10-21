# AcoustiGuard AI

**AcoustiGuard AI** is an AI-powered environmental sound classification application designed for reliable audio recognition in noisy, real-world environments. Built on PyTorch, FastAPI, and React, the system classifies 10 urban sound categories from the UrbanSound8K benchmark, evaluates resilience against severe acoustic corruptions (down to 0 dB SNR), and employs temperature-calibrated confidence thresholding to flag ambiguous inputs as *"Uncertain — manual review recommended"*.

---

## Key Capabilities

- **10-Class Environmental Sound Classification**: Accurately classifies urban audio recordings across 10 categories: *air conditioner, car horn, children playing, dog bark, drilling, engine idling, gun shot, jackhammer, siren, and street music*.
- **Noise Robustness Defense**: Features both a standard Baseline model and a Noise-Augmented model trained with synthetic additive noise (white, pink, brown, and ambient noise across 5–25 dB SNRs) to preserve accuracy under real-world acoustic degradation.
- **Controlled Noise Stress-Testing Lab**: Evaluates model performance across user-specified Signal-to-Noise Ratios (20 dB, 10 dB, 5 dB, 0 dB) and noise colorations with interactive side-by-side confidence drops and verdict analysis.
- **Confidence-Aware Selective Classification**: Utilizes post-hoc temperature scaling to calibrate softmax probabilities, rejecting low-confidence predictions below a tunable threshold $\tau \in [0.50, 0.95]$ to minimize false alarms in mission-critical monitoring.
- **Interactive Full-Stack Interface**: High-performance FastAPI backend serving cached PyTorch models paired with a responsive React + Vite + Tailwind CSS dashboard featuring audio playback, waveform visualization, and live model comparison metrics.

---

## System Architecture

```
robust_audio_detection/
├── ml/                                 # Audio processing & deep learning pipeline
│   ├── config.py                       # Audio specs (22,050 Hz, 4s, 128 mels), fold logic, SNRs
│   ├── metadata.py                     # UrbanSound8K CSV validator & 10-fold split manager
│   ├── preprocessing.py                # Resampling, mono downmix, zero-padding / centering
│   ├── features.py                     # Mel filterbank & log-mel spectrogram extraction
│   ├── dataset.py                      # PyTorch Dataset & DataLoader factory with leakage check
│   ├── models.py                       # AcousticCNN architecture (767,722 parameters, 2.93 MB)
│   ├── training_utils.py               # Hardware selector, deterministic seeding, EarlyStopping
│   ├── metrics.py                      # Accuracy, F1, per-class metrics, confusion matrices
│   ├── augmentation.py                 # Additive noise mixing (white/pink/brown/ambient at 20-0 dB)
│   └── calibration.py                  # Temperature scaling (L-BFGS on validation NLL), ECE/MCE
│
├── experiments/                        # Model training & evaluation scripts
│   ├── train_baseline.py               # Train Baseline AcousticCNN on clean audio
│   ├── train_augmented.py              # Train Noise-Augmented AcousticCNN
│   ├── evaluate.py                     # Standalone model evaluator
│   ├── evaluate_noise.py               # Multi-SNR robustness benchmark
│   └── evaluate_selective.py           # Confidence calibration & selective prediction analysis
│
├── models/                             # Saved model checkpoints & scalers
│   ├── baseline_cnn_best.pt            # Baseline model weights
│   ├── augmented_cnn_best.pt           # Noise-augmented model weights
│   └── temperature_scaler.pt           # Learned temperature parameter ($T$)
│
├── results/                            # Metrics JSONs and performance visualizations
│   ├── baseline_metrics.json           # Baseline model test metrics
│   ├── augmented_metrics.json          # Augmented model test metrics
│   ├── experiment_comparison.json      # Comparison table across SNRs
│   ├── selective_prediction_report.json# Coverage vs. risk data
│   └── *.png                           # Confusion matrices, SNR degradation curves, loss plots
│
├── backend/                            # FastAPI REST API
│   └── app/
│       ├── main.py                     # Server entrypoint, CORS, static mounts, /api/health
│       └── routes/
│           ├── predict.py              # POST /api/predict & POST /api/robustness-test
│           └── analytics.py            # GET /api/model-info, /api/metrics, /api/experiments
│
├── frontend/                           # React / Vite / TypeScript dashboard
│   ├── src/
│   │   ├── App.tsx                     # Main interactive application UI
│   │   ├── App.test.tsx                # Vitest automated test suite
│   │   ├── main.tsx                    # React DOM bootstrap
│   │   └── index.css                   # Global dark styling & glassmorphism
│   ├── tailwind.config.js              # Theme color palette
│   └── vite.config.ts                  # Vite build & API proxy configuration
│
└── tests/                              # Automated test suites
    ├── test_data_pipeline.py           # 11 unit tests for audio, dataset, and zero-leakage
    └── test_backend.py                 # 9 integration tests for FastAPI REST endpoints
```

---

## Technology Stack

| Layer | Technologies |
|---|---|
| **Deep Learning & Audio** | PyTorch, TorchAudio, NumPy, SciPy, Librosa, SoundFile |
| **Backend REST API** | FastAPI, Uvicorn, Python-Multipart, Pydantic, HTTPX |
| **Frontend Dashboard** | React 18, Vite, TypeScript, Tailwind CSS, Lucide React, Recharts |
| **Testing & Quality** | Pytest (Python), Vitest (React/TypeScript) |

---

## Installation & Setup

### Prerequisites
- Python 3.10+ (Tested on Python 3.11 & 3.14)
- Node.js 18+ and npm 9+
- Git

### 1. Backend & ML Setup

```powershell
# Navigate to the project root
cd robust_audio_detection

# Create and activate a Python virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1    # On Windows PowerShell
# source venv/bin/activate      # On Linux / macOS

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Frontend Setup

```powershell
# Navigate to the frontend directory
cd frontend

# Install npm dependencies
npm install

# Return to root
cd ..
```

---

## Dataset Configuration

The system uses the **UrbanSound8K** dataset (8,732 audio excerpts $\le 4.0$ seconds across 10 classes and 10 predefined folds).

### Standard UrbanSound8K Setup
1. Download [UrbanSound8K.tar.gz](https://urbansounddataset.weebly.com/urbansound8k.html).
2. Extract into the `data/` directory so the folder structure matches:
   ```
   robust_audio_detection/data/UrbanSound8K/
   ├── audio/
   │   ├── fold1/
   │   ├── fold2/
   │   └── ... fold10/
   └── metadata/
       └── UrbanSound8K.csv
   ```
> **Zero-Leakage Guarantee**: The data partitioner enforces strict fold isolation (folds 1–8 for training, fold 9 for validation, fold 10 for held-out testing). Slices from the same recording share the same `fsID` and are never mixed across splits.

### Automated Smoke-Test Verification Dataset
To test and verify the entire training and inference pipeline without downloading the full 6 GB dataset, run the built-in synthetic metadata and audio generator:

```powershell
python -m ml.metadata --smoke-test
```

---

## Model Training & Evaluation Workflow

All training and evaluation scripts can be executed via Python modules from the project root:

```powershell
# 1. Train the Baseline CNN (Clean audio only, saves to models/baseline_cnn_best.pt)
python -m experiments.train_baseline

# 2. Train the Noise-Augmented CNN (Additive noise injection, saves to models/augmented_cnn_best.pt)
python -m experiments.train_augmented

# 3. Benchmark Noise Robustness across SNRs (Clean, 20 dB, 10 dB, 5 dB, 0 dB)
python -m experiments.evaluate_noise

# 4. Calibrate Probabilities & Evaluate Selective Classification
python -m experiments.evaluate_selective
```

All trained checkpoints are saved to `models/`, and metrics, loss curves, confusion matrices, and SNR curves are exported to `results/`.

---

## Running the Application

### 1. Start the FastAPI Backend Server
```powershell
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
The backend API is now running at `http://127.0.0.1:8000`. Interactive Swagger API docs are accessible at `http://127.0.0.1:8000/docs`.

### 2. Start the React Frontend Development Server
In a separate terminal:
```powershell
cd frontend
npm run dev
```
Open your browser and navigate to `http://localhost:5173`.

---

## Application Features & Navigation

- **Audio Analysis Tab**: Upload any WAV, MP3, FLAC, or OGG recording (up to 4.0 seconds). Select between the Baseline and Noise-Augmented models, adjust the confidence rejection threshold, and toggle temperature calibration. View spectrogram representations, predicted class, top probability distributions, and uncertainty flags.
- **Robustness Lab Tab**: Stress-test any audio sample by injecting controlled levels of synthetic noise (white, pink, brown, ambient) at target SNRs (20 dB, 10 dB, 5 dB, 0 dB). Directly observe confidence drops and whether the classification changes.
- **Performance Analytics Tab**: Interactive visualization of model benchmarks, accuracy degradation curves across SNRs, risk-coverage trade-offs, and confusion matrices loaded directly from disk.
- **Model & Taxonomy Tab**: Details on the `AcousticCNN` layer architecture, parameter count, input shape, 10-class taxonomy, and operational limitations.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | System health, device (CPU/CUDA), and available model weights |
| `POST` | `/api/predict` | Run audio classification with threshold rejection & calibration |
| `POST` | `/api/robustness-test` | Inject noise at target SNR and compare clean vs. noisy predictions |
| `GET` | `/api/model-info` | Architecture specs, parameter counts, and taxonomy |
| `GET` | `/api/metrics` | Model comparison tables and performance statistics |
| `GET` | `/static/results/{filename}` | Serve generated confusion matrices and performance plots |

---

## Running Automated Tests

### Python Backend & Pipeline Tests (Pytest)
```powershell
pytest tests/ -v
```
- `tests/test_data_pipeline.py`: Validates audio loading, resampling, padding, log-mel spectrogram features, dataset split isolation, and zero-leakage assertions (11 tests).
- `tests/test_backend.py`: Validates FastAPI `/api/health`, `/api/model-info`, `/api/metrics`, `/api/predict`, and `/api/robustness-test` endpoints (9 tests).

### Frontend UI & Logic Tests (Vitest)
```powershell
cd frontend
npm test
```
- `frontend/src/App.test.tsx`: Validates component rendering, navigation tabs, parameter controls, selective classification rejection logic, and taxonomy integrity (6 tests).

### Frontend Production Build
```powershell
cd frontend
npm run build
```
Validates TypeScript compilation (`tsc`) and Vite bundling.
