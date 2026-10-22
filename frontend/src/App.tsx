import React, { useState, useEffect, useRef } from "react";
import {
  ShieldCheck,
  Activity,
  Cpu,
  FileAudio,
  FlaskConical,
  BarChart3,
  Info,
  Volume2,
  AlertTriangle,
  CheckCircle2,
  RefreshCw,
  Sliders,
  Sparkles,
  Upload,
  Play,
  Pause,
  ExternalLink,
  ChevronRight,
  TrendingDown,
  Layers,
  HelpCircle,
} from "lucide-react";

// ==========================================
// Types & API Interfaces
// ==========================================

interface ClassProbability {
  class_id: number;
  class_name: string;
  probability: number;
}

interface PredictionData {
  predicted_class_id: number;
  predicted_class_name: string;
  confidence: number;
  is_uncertain: boolean;
  uncertainty_status: "confident" | "uncertain";
  status_message: string;
  threshold_applied: number;
  probabilities: ClassProbability[];
  model_used: string;
  calibrated: boolean;
  temperature: number;
  inference_latency_ms: number;
  waveform_preview: number[];
  spectrogram_preview: number[][];
}

interface RobustnessData {
  snr_db: number;
  noise_type: string;
  confidence_drop: number;
  prediction_changed: boolean;
  summary_verdict: string;
  clean_prediction: PredictionData;
  noisy_prediction: PredictionData;
}

interface HealthData {
  status: string;
  service: string;
  device: string;
  models_available: {
    baseline_cnn: boolean;
    augmented_cnn: boolean;
    temperature_scaler: boolean;
  };
  results_available: {
    baseline_metrics: boolean;
    augmented_metrics: boolean;
    experiment_comparison: boolean;
    selective_prediction: boolean;
  };
  ready_for_inference: boolean;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<"analyze" | "robustness" | "research" | "model_info">("analyze");
  const [health, setHealth] = useState<HealthData | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean>(false);
  const [loadingHealth, setLoadingHealth] = useState<boolean>(true);

  // Audio Analysis State
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [modelType, setModelType] = useState<"augmented" | "baseline">("augmented");
  const [threshold, setThreshold] = useState<number>(0.7);
  const [useCalibration, setUseCalibration] = useState<boolean>(true);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  // Robustness Lab State
  const [robFile, setRobFile] = useState<File | null>(null);
  const [robAudioUrl, setRobAudioUrl] = useState<string | null>(null);
  const [targetSnr, setTargetSnr] = useState<number>(10.0);
  const [noiseType, setNoiseType] = useState<string>("ambient");
  const [isTestingRobustness, setIsTestingRobustness] = useState<boolean>(false);
  const [robustnessResult, setRobustnessResult] = useState<RobustnessData | null>(null);
  const [robError, setRobError] = useState<string | null>(null);

  // Research Metrics State
  const [metricsData, setMetricsData] = useState<any>(null);
  const [loadingMetrics, setLoadingMetrics] = useState<boolean>(false);

  // Model Info State
  const [modelInfo, setModelInfo] = useState<any>(null);
  const [loadingModelInfo, setLoadingModelInfo] = useState<boolean>(false);

  // Audio Player State
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);

  // 1. Fetch Backend Health Telemetry
  const checkHealth = async () => {
    setLoadingHealth(true);
    try {
      const res = await fetch("/api/health");
      if (res.ok) {
        const data: HealthData = await res.json();
        setHealth(data);
        setBackendOnline(true);
      } else {
        setBackendOnline(false);
      }
    } catch {
      setBackendOnline(false);
    } finally {
      setLoadingHealth(false);
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // 2. Fetch Research Metrics on Tab Switch
  useEffect(() => {
    if (activeTab === "research" && !metricsData) {
      setLoadingMetrics(true);
      fetch("/api/metrics")
        .then((r) => r.json())
        .then((d) => setMetricsData(d))
        .catch((e) => console.error("Error loading metrics:", e))
        .finally(() => setLoadingMetrics(false));
    }
    if (activeTab === "model_info" && !modelInfo) {
      setLoadingModelInfo(true);
      fetch("/api/model-info")
        .then((r) => r.json())
        .then((d) => setModelInfo(d))
        .catch((e) => console.error("Error loading model info:", e))
        .finally(() => setLoadingModelInfo(false));
    }
  }, [activeTab]);

  // 3. Audio File Selection & Playback Handler
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPrediction(null);
      setAnalysisError(null);
      const url = URL.createObjectURL(file);
      setAudioUrl(url);
      setIsPlaying(false);
    }
  };

  const handleRobFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setRobFile(file);
      setRobustnessResult(null);
      setRobError(null);
      setRobAudioUrl(URL.createObjectURL(file));
    }
  };

  // Helper to generate a standardized WAV Blob
  const buildWavBlob = (samples: Float32Array, sr: number = 22050): Blob => {
    const numSamples = samples.length;
    const buffer = new ArrayBuffer(44 + numSamples * 2);
    const view = new DataView(buffer);
    const writeStr = (offset: number, s: string) => {
      for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
    };
    writeStr(0, "RIFF");
    view.setUint32(4, 36 + numSamples * 2, true);
    writeStr(8, "WAVE");
    writeStr(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // Mono
    view.setUint32(24, sr, true);
    view.setUint32(28, sr * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeStr(36, "data");
    view.setUint32(40, numSamples * 2, true);

    for (let i = 0; i < numSamples; i++) {
      const s = Math.max(-1.0, Math.min(1.0, samples[i]));
      view.setInt16(44 + i * 2, s < 0 ? s * 32768 : s * 32767, true);
    }
    return new Blob([buffer], { type: "audio/wav" });
  };

  // Generate synthetic in-browser acoustic samples for zero-friction testing
  const generateSamplePreset = (
    preset: "siren" | "car_horn" | "engine_idling" | "jackhammer",
    targetTab: "analyze" | "robustness" = "analyze"
  ) => {
    const sr = 22050;
    const duration = 3.5;
    const numSamples = Math.floor(sr * duration);
    const samples = new Float32Array(numSamples);

    if (preset === "siren") {
      for (let i = 0; i < numSamples; i++) {
        const t = i / sr;
        const freq = 650 + 350 * Math.sin(2 * Math.PI * 1.2 * t);
        samples[i] = 0.6 * Math.sin(2 * Math.PI * freq * t);
      }
    } else if (preset === "car_horn") {
      for (let i = 0; i < numSamples; i++) {
        const t = i / sr;
        const burst = (t % 1.2 < 0.8) ? 1.0 : 0.05;
        samples[i] = 0.35 * burst * (Math.sin(2 * Math.PI * 440 * t) + 0.8 * Math.sin(2 * Math.PI * 554 * t));
      }
    } else if (preset === "engine_idling") {
      for (let i = 0; i < numSamples; i++) {
        const t = i / sr;
        const rumble = 0.5 * Math.sin(2 * Math.PI * 62 * t) + 0.3 * Math.sin(2 * Math.PI * 124 * t);
        const noise = (Math.random() * 2 - 1) * 0.15;
        samples[i] = 0.7 * (rumble + noise);
      }
    } else if (preset === "jackhammer") {
      for (let i = 0; i < numSamples; i++) {
        const t = i / sr;
        const phase = (t * 18) % 1.0;
        const strikeEnv = Math.exp(-phase * 15.0);
        const metallic = Math.sin(2 * Math.PI * 380 * t) + 0.5 * Math.sin(2 * Math.PI * 820 * t);
        const impactNoise = (Math.random() * 2 - 1) * 0.5;
        samples[i] = 0.8 * strikeEnv * (metallic + impactNoise);
      }
    }

    const filename = `synthetic_${preset}_sample.wav`;
    const blob = buildWavBlob(samples, sr);
    const file = new File([blob], filename, { type: "audio/wav" });
    const url = URL.createObjectURL(blob);

    if (targetTab === "robustness") {
      setRobFile(file);
      setRobAudioUrl(url);
      setRobustnessResult(null);
      setRobError(null);
    } else {
      setSelectedFile(file);
      setAudioUrl(url);
      setPrediction(null);
      setAnalysisError(null);
      setIsPlaying(false);
    }
  };

  // 4. Submit Audio Prediction Request
  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setIsAnalyzing(true);
    setAnalysisError(null);

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("model_type", modelType);
    formData.append("threshold", threshold.toString());
    formData.append("use_calibration", useCalibration ? "true" : "false");

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server returned HTTP ${res.status}`);
      }
      const data: PredictionData = await res.json();
      setPrediction(data);
    } catch (err: any) {
      setAnalysisError(err.message || "Failed to analyze audio recording.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  // 5. Submit Robustness Lab Test
  const handleRobustnessTest = async () => {
    const fileToUse = robFile || selectedFile;
    if (!fileToUse) return;

    setIsTestingRobustness(true);
    setRobError(null);

    const formData = new FormData();
    formData.append("file", fileToUse);
    formData.append("snr_db", targetSnr.toString());
    formData.append("noise_type", noiseType);
    formData.append("model_type", modelType);
    formData.append("threshold", threshold.toString());

    try {
      const res = await fetch("/api/robustness-test", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server returned HTTP ${res.status}`);
      }
      const data: RobustnessData = await res.json();
      setRobustnessResult(data);
    } catch (err: any) {
      setRobError(err.message || "Robustness test execution failed.");
    } finally {
      setIsTestingRobustness(false);
    }
  };

  const toggleAudio = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play();
      setIsPlaying(true);
    }
  };

  return (
    <div className="min-h-screen flex flex-col font-sans">
      {/* Top Navigation & Telemetry Bar */}
      <header className="sticky top-0 z-50 glass-panel border-b border-slate-800/80 px-6 py-3.5 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          {/* Logo & App Branding */}
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-brand-600 to-sky-500 flex items-center justify-center shadow-lg shadow-brand-500/20 border border-brand-400/30">
              <ShieldCheck className="w-6 h-6 text-white" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-lg tracking-tight text-white">AcoustiGuard AI</span>
                <span className="px-2 py-0.5 text-xs font-mono font-medium rounded-full bg-brand-500/10 text-brand-400 border border-brand-500/20">
                  UrbanSound8K
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">
                Noise-Robust Environmental Audio Classification &amp; Confidence-Aware Inference
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex items-center space-x-1 bg-slate-900/80 p-1 rounded-xl border border-slate-800">
            <button
              onClick={() => setActiveTab("analyze")}
              className={`flex items-center space-x-1.5 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === "analyze"
                  ? "bg-brand-600 text-white shadow-md shadow-brand-600/30 font-semibold"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
              }`}
            >
              <FileAudio className="w-3.5 h-3.5" />
              <span>Audio Analysis</span>
            </button>

            <button
              onClick={() => setActiveTab("robustness")}
              className={`flex items-center space-x-1.5 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === "robustness"
                  ? "bg-brand-600 text-white shadow-md shadow-brand-600/30 font-semibold"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
              }`}
            >
              <FlaskConical className="w-3.5 h-3.5" />
              <span>Robustness Lab</span>
            </button>

            <button
              onClick={() => setActiveTab("research")}
              className={`flex items-center space-x-1.5 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === "research"
                  ? "bg-brand-600 text-white shadow-md shadow-brand-600/30 font-semibold"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
              }`}
            >
              <BarChart3 className="w-3.5 h-3.5" />
              <span>Performance Analytics</span>
            </button>

            <button
              onClick={() => setActiveTab("model_info")}
              className={`flex items-center space-x-1.5 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === "model_info"
                  ? "bg-brand-600 text-white shadow-md shadow-brand-600/30 font-semibold"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
              }`}
            >
              <Info className="w-3.5 h-3.5" />
              <span>Model & Taxonomy</span>
            </button>
          </nav>

          {/* Backend Status Telemetry Indicator */}
          <div className="flex items-center space-x-2">
            <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-slate-900 border border-slate-800 text-xs font-mono">
              <span
                className={`w-2 h-2 rounded-full ${
                  backendOnline ? "bg-emerald-500 animate-pulse" : "bg-rose-500"
                }`}
              />
              <span className="text-slate-300">{backendOnline ? "API Online" : "Offline"}</span>
              {health?.device && (
                <span className="text-slate-500 border-l border-slate-800 pl-1.5 uppercase">
                  {health.device}
                </span>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main View Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 md:p-8 space-y-6">
        {/* Offline Helper Banner */}
        {!backendOnline && !loadingHealth && (
          <div className="p-4 rounded-2xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs flex items-center justify-between shadow-lg">
            <div className="flex items-center space-x-2.5">
              <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0" />
              <span>
                <strong>FastAPI Backend is currently offline.</strong> Start both backend and frontend by running{" "}
                <code className="bg-slate-900 px-2 py-0.5 rounded text-amber-200 font-mono border border-slate-800">
                  python run.py
                </code>{" "}
                in your terminal.
              </span>
            </div>
            <button
              onClick={checkHealth}
              className="px-3 py-1.5 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 text-amber-200 font-medium transition text-xs shrink-0"
            >
              Retry Connection
            </button>
          </div>
        )}

        {/* ==================================================== */}
        {/* TAB 1: AUDIO ANALYSIS & PREDICTION                   */}
        {/* ==================================================== */}
        {activeTab === "analyze" && (
          <div className="space-y-6">
            {/* Header intro */}
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
              <div>
                <h1 className="text-2xl font-bold tracking-tight text-white flex items-center space-x-2">
                  <span>Environmental Sound Classifier</span>
                  <span className="text-xs font-normal text-indigo-400 bg-indigo-500/10 px-2.5 py-0.5 rounded-md border border-indigo-500/20">
                    Confidence-Aware
                  </span>
                </h1>
                <p className="text-sm text-slate-400 mt-1">
                  Upload an urban environmental audio recording to analyze its spectrogram, predict sound category, and detect uncertainty.
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs text-slate-400 font-medium flex items-center space-x-1">
                  <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                  <span>Try Sample:</span>
                </span>
                <button
                  type="button"
                  onClick={() => generateSamplePreset("siren", "analyze")}
                  className="px-2.5 py-1.5 rounded-lg text-xs font-medium bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-700/80 hover:border-brand-500/50 transition flex items-center space-x-1"
                >
                  <span>🚨 Siren</span>
                </button>
                <button
                  type="button"
                  onClick={() => generateSamplePreset("car_horn", "analyze")}
                  className="px-2.5 py-1.5 rounded-lg text-xs font-medium bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-700/80 hover:border-brand-500/50 transition flex items-center space-x-1"
                >
                  <span>🚗 Car Horn</span>
                </button>
                <button
                  type="button"
                  onClick={() => generateSamplePreset("engine_idling", "analyze")}
                  className="px-2.5 py-1.5 rounded-lg text-xs font-medium bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-700/80 hover:border-brand-500/50 transition flex items-center space-x-1"
                >
                  <span>🚚 Engine</span>
                </button>
                <button
                  type="button"
                  onClick={() => generateSamplePreset("jackhammer", "analyze")}
                  className="px-2.5 py-1.5 rounded-lg text-xs font-medium bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-700/80 hover:border-brand-500/50 transition flex items-center space-x-1"
                >
                  <span>🔨 Jackhammer</span>
                </button>
              </div>
            </div>

            {/* Grid: Audio Input & Controls + Results */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Left Column: Upload, Player, Parameters (5 cols) */}
              <div className="lg:col-span-5 space-y-6">
                {/* Upload Card */}
                <div className="glass-panel p-5 rounded-2xl space-y-4">
                  <div className="flex items-center justify-between">
                    <h2 className="text-sm font-semibold text-white flex items-center space-x-2">
                      <FileAudio className="w-4 h-4 text-brand-400" />
                      <span>Audio Input</span>
                    </h2>
                    {selectedFile && (
                      <span className="text-xs text-slate-400 truncate max-w-[160px] font-mono">
                        {selectedFile.name}
                      </span>
                    )}
                  </div>

                  <label
                    htmlFor="audio-upload"
                    className="border-2 border-dashed border-slate-700/80 hover:border-brand-500/50 rounded-xl p-6 flex flex-col items-center justify-center text-center cursor-pointer transition-colors bg-slate-900/40 hover:bg-slate-900/70 group"
                  >
                    <Upload className="w-8 h-8 text-slate-500 group-hover:text-brand-400 transition mb-2" />
                    <span className="text-sm font-medium text-slate-300 group-hover:text-white">
                      Click to choose audio or drop file
                    </span>
                    <span className="text-xs text-slate-500 mt-1">
                      WAV, MP3, FLAC, OGG (Up to 4.0s @ 22,050 Hz)
                    </span>
                    <input
                      id="audio-upload"
                      type="file"
                      accept=".wav,.mp3,.flac,.ogg,audio/*"
                      onChange={handleFileChange}
                      className="hidden"
                    />
                  </label>

                  {/* Audio Player if file loaded */}
                  {audioUrl && (
                    <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <button
                          onClick={toggleAudio}
                          className="w-8 h-8 rounded-full bg-brand-600 hover:bg-brand-500 text-white flex items-center justify-center transition shadow-md"
                        >
                          {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
                        </button>
                        <div className="text-xs">
                          <p className="font-medium text-slate-200 truncate max-w-[180px]">
                            {selectedFile?.name || "Uploaded Audio"}
                          </p>
                          <span className="text-slate-500 font-mono">Standardized 4.0s excerpt</span>
                        </div>
                      </div>
                      <Volume2 className="w-4 h-4 text-slate-400" />
                      <audio
                        ref={audioRef}
                        src={audioUrl}
                        onEnded={() => setIsPlaying(false)}
                        className="hidden"
                      />
                    </div>
                  )}

                  {/* Inference Controls */}
                  <div className="space-y-4 pt-2 border-t border-slate-800/80">
                    <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center space-x-1.5">
                      <Sliders className="w-3.5 h-3.5" />
                      <span>Inference Parameters</span>
                    </h3>

                    {/* Model Selector */}
                    <div>
                      <label className="text-xs text-slate-300 block mb-1 font-medium">Model Variant</label>
                      <div className="grid grid-cols-2 gap-2">
                        <button
                          type="button"
                          onClick={() => setModelType("augmented")}
                          className={`px-3 py-2 rounded-xl text-xs font-medium border text-left transition ${
                            modelType === "augmented"
                              ? "bg-brand-500/10 border-brand-500/50 text-brand-300 shadow-sm"
                              : "bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-300"
                          }`}
                        >
                          <div className="font-semibold text-white">Noise-Augmented</div>
                          <div className="text-[11px] text-slate-400">Trained with noise defense</div>
                        </button>

                        <button
                          type="button"
                          onClick={() => setModelType("baseline")}
                          className={`px-3 py-2 rounded-xl text-xs font-medium border text-left transition ${
                            modelType === "baseline"
                              ? "bg-brand-500/10 border-brand-500/50 text-brand-300 shadow-sm"
                              : "bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-300"
                          }`}
                        >
                          <div className="font-semibold text-white">Baseline CNN</div>
                          <div className="text-[11px] text-slate-400">Clean audio training only</div>
                        </button>
                      </div>
                    </div>

                    {/* Threshold Slider */}
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-300 font-medium">Uncertainty Rejection Threshold (&tau;)</span>
                        <span className="font-mono text-brand-400 font-bold">{(threshold * 100).toFixed(0)}%</span>
                      </div>
                      <input
                        type="range"
                        min="0.5"
                        max="0.95"
                        step="0.05"
                        value={threshold}
                        onChange={(e) => setThreshold(parseFloat(e.target.value))}
                        className="w-full accent-brand-500 bg-slate-800 rounded-lg h-2 cursor-pointer"
                      />
                      <div className="flex justify-between text-[10px] text-slate-500 mt-1 font-mono">
                        <span>50% (Permissive)</span>
                        <span>70% (Default)</span>
                        <span>95% (Conservative)</span>
                      </div>
                    </div>

                    {/* Temperature Scaling Toggle */}
                    <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/60 border border-slate-800 text-xs">
                      <div>
                        <div className="font-medium text-slate-200">Temperature Scaling</div>
                        <div className="text-[11px] text-slate-500">Post-hoc confidence calibration</div>
                      </div>
                      <button
                        type="button"
                        onClick={() => setUseCalibration(!useCalibration)}
                        className={`w-10 h-5 rounded-full transition-colors relative ${
                          useCalibration ? "bg-brand-600" : "bg-slate-700"
                        }`}
                      >
                        <span
                          className={`w-3.5 h-3.5 rounded-full bg-white absolute top-0.75 transition-transform ${
                            useCalibration ? "left-5.5" : "left-1"
                          }`}
                        />
                      </button>
                    </div>

                    {/* Run Classification Button */}
                    <button
                      disabled={!selectedFile || isAnalyzing}
                      onClick={handleAnalyze}
                      className="w-full py-3 rounded-xl font-semibold text-sm bg-gradient-to-r from-brand-600 to-indigo-600 hover:from-brand-500 hover:to-indigo-500 text-white shadow-lg shadow-brand-600/25 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2 transition"
                    >
                      {isAnalyzing ? (
                        <>
                          <RefreshCw className="w-4 h-4 animate-spin" />
                          <span>Classifying Audio...</span>
                        </>
                      ) : (
                        <>
                          <Activity className="w-4 h-4" />
                          <span>Classify Environmental Sound</span>
                        </>
                      )}
                    </button>

                    {analysisError && (
                      <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-start space-x-2">
                        <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                        <span>{analysisError}</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Right Column: Prediction Results, Spectrogram & Probabilities (7 cols) */}
              <div className="lg:col-span-7 space-y-6">
                {prediction ? (
                  <>
                    {/* Primary Prediction Verdict Card */}
                    <div
                      className={`glass-panel p-6 rounded-2xl border transition-all ${
                        prediction.is_uncertain
                          ? "border-amber-500/40 bg-gradient-to-br from-slate-900/90 via-slate-900/80 to-amber-950/20"
                          : "border-emerald-500/40 bg-gradient-to-br from-slate-900/90 via-slate-900/80 to-emerald-950/20"
                      }`}
                    >
                      <div className="flex items-start justify-between">
                        <div>
                          <div className="flex items-center space-x-2 mb-1">
                            <span className="text-xs font-mono uppercase tracking-wider text-slate-400">
                              Classification Output
                            </span>
                            <span className="text-slate-600">•</span>
                            <span className="text-xs font-mono text-slate-400">
                              Latency: {prediction.inference_latency_ms.toFixed(1)} ms
                            </span>
                          </div>
                          <h2 className="text-2xl font-extrabold text-white capitalize tracking-tight">
                            {prediction.predicted_class_name.replace(/_/g, " ")}
                          </h2>
                        </div>

                        {/* Uncertainty Badge */}
                        <div
                          className={`px-3 py-1.5 rounded-full text-xs font-semibold flex items-center space-x-1.5 border shadow-sm ${
                            prediction.is_uncertain
                              ? "bg-amber-500/10 border-amber-500/30 text-amber-400"
                              : "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
                          }`}
                        >
                          {prediction.is_uncertain ? (
                            <>
                              <AlertTriangle className="w-4 h-4" />
                              <span>Uncertain Prediction</span>
                            </>
                          ) : (
                            <>
                              <CheckCircle2 className="w-4 h-4" />
                              <span>Reliable Prediction</span>
                            </>
                          )}
                        </div>
                      </div>

                      {/* Confidence Meter Bar */}
                      <div className="mt-5 space-y-1.5">
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-400">
                            Model Confidence {prediction.calibrated ? "(Temperature Scaled)" : "(Raw)"}
                          </span>
                          <span className="font-mono font-bold text-white">
                            {(prediction.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
                          <div
                            className={`h-full transition-all duration-500 rounded-full ${
                              prediction.is_uncertain
                                ? "bg-amber-500"
                                : "bg-gradient-to-r from-brand-500 to-emerald-500"
                            }`}
                            style={{ width: `${Math.min(100, prediction.confidence * 100)}%` }}
                          />
                        </div>
                        <p className="text-xs text-slate-400 mt-2 italic">
                          "{prediction.status_message}"
                        </p>
                      </div>
                    </div>

                    {/* Acoustic Feature Visualizer (Waveform & Spectrogram Preview) */}
                    <div className="glass-panel p-5 rounded-2xl space-y-4">
                      <div className="flex items-center justify-between">
                        <h3 className="text-sm font-semibold text-white flex items-center space-x-2">
                          <Activity className="w-4 h-4 text-sky-400" />
                          <span>Extracted Acoustic Features</span>
                        </h3>
                        <span className="text-xs text-slate-500 font-mono">
                          128 Mels &bull; 88,200 Samples
                        </span>
                      </div>

                      {/* Waveform Micro-Visualizer */}
                      <div>
                        <div className="text-[11px] font-mono text-slate-400 mb-1">
                          Standardized Audio Waveform (Envelope)
                        </div>
                        <div className="h-16 w-full bg-slate-950/80 rounded-xl p-2 border border-slate-800 flex items-center justify-between gap-0.5 overflow-hidden">
                          {prediction.waveform_preview.map((val, idx) => {
                            const barHeight = Math.max(4, Math.abs(val) * 55);
                            return (
                              <div
                                key={idx}
                                className="w-full bg-indigo-500/70 hover:bg-sky-400 rounded-full transition-all"
                                style={{ height: `${barHeight}px` }}
                              />
                            );
                          })}
                        </div>
                      </div>

                      {/* Log-Mel Spectrogram Heatmap Preview */}
                      <div>
                        <div className="text-[11px] font-mono text-slate-400 mb-1">
                          Log-Mel Spectrogram Heatmap (Time &times; Frequency Bins)
                        </div>
                        <div className="h-24 w-full bg-slate-950/90 rounded-xl p-1.5 border border-slate-800 grid grid-rows-8 grid-flow-col gap-0.5 overflow-hidden">
                          {prediction.spectrogram_preview.slice(0, 8).map((row, rIdx) =>
                            row.slice(0, 32).map((cellVal, cIdx) => {
                              // Normalize cell value to color opacity (0.1 to 1.0)
                              const normIntensity = Math.min(1, Math.max(0.1, (cellVal + 2) / 4));
                              return (
                                <div
                                  key={`${rIdx}-${cIdx}`}
                                  className="w-full h-full rounded-[1px] spectrogram-cell"
                                  style={{
                                    backgroundColor: `rgba(99, 102, 241, ${normIntensity})`,
                                  }}
                                  title={`Mel: ${rIdx}, Frame: ${cIdx}, Val: ${cellVal.toFixed(2)}`}
                                />
                              );
                            })
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Probability Distribution Breakdown */}
                    <div className="glass-panel p-5 rounded-2xl space-y-3">
                      <h3 className="text-sm font-semibold text-white flex items-center justify-between">
                        <span>Class Probability Distribution (10 Classes)</span>
                        <span className="text-xs font-mono text-slate-400">Softmax Logits</span>
                      </h3>

                      <div className="space-y-2 pt-1">
                        {prediction.probabilities.map((item, idx) => {
                          const isTop = idx === 0;
                          return (
                            <div key={item.class_id} className="space-y-1">
                              <div className="flex justify-between text-xs">
                                <span
                                  className={`capitalize truncate ${
                                    isTop ? "font-bold text-white" : "text-slate-400"
                                  }`}
                                >
                                  {item.class_name.replace(/_/g, " ")}
                                </span>
                                <span className={`font-mono ${isTop ? "text-brand-400 font-bold" : "text-slate-500"}`}>
                                  {(item.probability * 100).toFixed(1)}%
                                </span>
                              </div>
                              <div className="w-full bg-slate-900 rounded-full h-1.5 overflow-hidden">
                                <div
                                  className={`h-full rounded-full transition-all duration-300 ${
                                    isTop ? "bg-brand-500" : "bg-slate-700"
                                  }`}
                                  style={{ width: `${Math.max(2, item.probability * 100)}%` }}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </>
                ) : (
                  /* Empty state when no audio analyzed yet */
                  <div className="glass-panel rounded-2xl p-12 text-center flex flex-col items-center justify-center space-y-4 text-slate-400 h-full min-h-[420px]">
                    <div className="w-16 h-16 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-center text-slate-600">
                      <FileAudio className="w-8 h-8" />
                    </div>
                    <div>
                      <h3 className="text-base font-semibold text-slate-200">No Audio Analyzed Yet</h3>
                      <p className="text-xs text-slate-500 max-w-sm mt-1">
                        Select an audio file or click "Generate Test Siren" above to classify environmental sounds and visualize acoustic features.
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ==================================================== */}
        {/* TAB 2: ROBUSTNESS LAB (CONTROLLED EXPERIMENT DEMO)   */}
        {/* ==================================================== */}
        {activeTab === "robustness" && (
          <div className="space-y-6">
            <div>
              <h1 className="text-2xl font-bold tracking-tight text-white flex items-center space-x-2">
                <span>Robustness Testing Laboratory</span>
                <span className="text-xs font-normal text-sky-400 bg-sky-500/10 px-2.5 py-0.5 rounded-md border border-sky-500/20">
                  Controlled SNR Injection
                </span>
              </h1>
              <p className="text-sm text-slate-400 mt-1">
                Evaluate audio recordings under controlled synthetic background noise (20, 10, 5, 0 dB SNR) to observe how acoustic interference impacts model reliability.
              </p>
            </div>

            {/* Test Configuration Controls */}
            <div className="glass-panel p-6 rounded-2xl space-y-4">
              <h2 className="text-sm font-semibold text-white flex items-center space-x-2">
                <Sliders className="w-4 h-4 text-sky-400" />
                <span>Controlled Degradation Parameters</span>
              </h2>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Audio Source */}
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <label className="text-xs text-slate-300 font-medium">Audio Source</label>
                    <button
                      type="button"
                      onClick={() => generateSamplePreset("siren", "robustness")}
                      className="text-[11px] text-sky-400 hover:text-sky-300 flex items-center space-x-1 transition font-medium"
                    >
                      <Sparkles className="w-3 h-3 text-amber-400" />
                      <span>Use Test Siren</span>
                    </button>
                  </div>
                  <label className="border border-slate-700/80 hover:border-sky-500/50 rounded-xl p-3 flex items-center space-x-3 cursor-pointer bg-slate-900/60 transition">
                    <Upload className="w-5 h-5 text-sky-400" />
                    <span className="text-xs text-slate-300 truncate">
                      {robFile?.name || selectedFile?.name || "Choose clean audio recording..."}
                    </span>
                    <input
                      type="file"
                      accept=".wav,.mp3,.flac,.ogg,audio/*"
                      onChange={handleRobFileChange}
                      className="hidden"
                    />
                  </label>
                </div>

                {/* Target SNR Selector */}
                <div>
                  <div className="flex justify-between text-xs mb-1.5">
                    <span className="text-slate-300 font-medium">Target SNR (Signal-to-Noise Ratio)</span>
                    <span className="font-mono text-sky-400 font-bold">{targetSnr} dB</span>
                  </div>
                  <div className="grid grid-cols-4 gap-1.5">
                    {[20.0, 10.0, 5.0, 0.0].map((val) => (
                      <button
                        key={val}
                        type="button"
                        onClick={() => setTargetSnr(val)}
                        className={`py-2 rounded-xl text-xs font-mono font-medium border transition ${
                          targetSnr === val
                            ? "bg-sky-500/15 border-sky-500 text-sky-300 font-bold"
                            : "bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-300"
                        }`}
                      >
                        {val} dB
                      </button>
                    ))}
                  </div>
                </div>

                {/* Noise Profile */}
                <div>
                  <label className="text-xs text-slate-300 font-medium block mb-1.5">Noise Profile</label>
                  <select
                    value={noiseType}
                    onChange={(e) => setNoiseType(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-sky-500"
                  >
                    <option value="ambient">Ambient (Urban background blend)</option>
                    <option value="pink">Pink Noise (1/f natural acoustic decay)</option>
                    <option value="brown">Brown Noise (1/f^2 traffic/engine rumble)</option>
                    <option value="white">White Noise (Broadband uniform noise)</option>
                  </select>
                </div>
              </div>

              <div className="pt-2 flex justify-end">
                <button
                  disabled={isTestingRobustness || (!robFile && !selectedFile)}
                  onClick={handleRobustnessTest}
                  className="px-6 py-2.5 rounded-xl font-semibold text-xs bg-sky-600 hover:bg-sky-500 text-white shadow-lg shadow-sky-600/20 disabled:opacity-50 transition flex items-center space-x-2"
                >
                  {isTestingRobustness ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      <span>Injecting Noise & Testing...</span>
                    </>
                  ) : (
                    <>
                      <FlaskConical className="w-3.5 h-3.5" />
                      <span>Run Robustness Comparison</span>
                    </>
                  )}
                </button>
              </div>

              {robError && (
                <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-center space-x-2">
                  <AlertTriangle className="w-4 h-4 shrink-0" />
                  <span>{robError}</span>
                </div>
              )}
            </div>

            {/* Side-by-Side Robustness Results */}
            {robustnessResult && (
              <div className="space-y-6">
                {/* Summary Banner */}
                <div className="glass-panel p-5 rounded-2xl border border-sky-500/30 bg-gradient-to-r from-slate-900 to-sky-950/30 flex items-center justify-between">
                  <div>
                    <span className="text-xs font-mono uppercase text-sky-400 tracking-wider">
                      Experiment Demonstration Verdict
                    </span>
                    <h3 className="text-lg font-bold text-white mt-0.5">
                      {robustnessResult.summary_verdict}
                    </h3>
                  </div>

                  <div className="text-right font-mono">
                    <span className="text-xs text-slate-400">Confidence Shift</span>
                    <p
                      className={`text-lg font-bold ${
                        robustnessResult.confidence_drop > 0 ? "text-amber-400" : "text-emerald-400"
                      }`}
                    >
                      {robustnessResult.confidence_drop > 0 ? "-" : "+"}
                      {Math.abs(robustnessResult.confidence_drop * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>

                {/* Side-by-Side Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Clean Condition Card */}
                  <div className="glass-panel p-6 rounded-2xl border-slate-800 space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-mono font-semibold text-emerald-400 uppercase tracking-wider">
                        Condition: Original Clean Recording
                      </span>
                      <span className="text-xs font-mono text-slate-500">&infin; dB SNR</span>
                    </div>

                    <div>
                      <h4 className="text-2xl font-bold text-white capitalize">
                        {robustnessResult.clean_prediction.predicted_class_name.replace(/_/g, " ")}
                      </h4>
                      <p className="text-xs text-slate-400 mt-0.5">
                        Confidence: {(robustnessResult.clean_prediction.confidence * 100).toFixed(1)}%
                      </p>
                    </div>

                    {/* Waveform preview */}
                    <div className="h-14 w-full bg-slate-950 rounded-xl p-2 border border-slate-800 flex items-center justify-between gap-0.5 overflow-hidden">
                      {robustnessResult.clean_prediction.waveform_preview.map((val, idx) => (
                        <div
                          key={idx}
                          className="w-full bg-emerald-500/60 rounded-full"
                          style={{ height: `${Math.max(4, Math.abs(val) * 45)}px` }}
                        />
                      ))}
                    </div>

                    <div className="text-xs text-slate-400">
                      Status:{" "}
                      <span className="text-emerald-400 font-medium">
                        {robustnessResult.clean_prediction.status_message}
                      </span>
                    </div>
                  </div>

                  {/* Noisy Condition Card */}
                  <div className="glass-panel p-6 rounded-2xl border-slate-800 space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-mono font-semibold text-amber-400 uppercase tracking-wider">
                        Condition: Degraded with {robustnessResult.noise_type} noise
                      </span>
                      <span className="text-xs font-mono text-amber-400 font-bold">
                        {robustnessResult.snr_db} dB SNR
                      </span>
                    </div>

                    <div>
                      <h4 className="text-2xl font-bold text-white capitalize">
                        {robustnessResult.noisy_prediction.predicted_class_name.replace(/_/g, " ")}
                      </h4>
                      <p className="text-xs text-slate-400 mt-0.5">
                        Confidence: {(robustnessResult.noisy_prediction.confidence * 100).toFixed(1)}%
                      </p>
                    </div>

                    {/* Noisy Waveform preview */}
                    <div className="h-14 w-full bg-slate-950 rounded-xl p-2 border border-slate-800 flex items-center justify-between gap-0.5 overflow-hidden">
                      {robustnessResult.noisy_prediction.waveform_preview.map((val, idx) => (
                        <div
                          key={idx}
                          className="w-full bg-amber-500/60 rounded-full"
                          style={{ height: `${Math.max(4, Math.abs(val) * 45)}px` }}
                        />
                      ))}
                    </div>

                    <div className="text-xs text-slate-400">
                      Status:{" "}
                      <span
                        className={`font-medium ${
                          robustnessResult.noisy_prediction.is_uncertain
                            ? "text-amber-400"
                            : "text-emerald-400"
                        }`}
                      >
                        {robustnessResult.noisy_prediction.status_message}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ==================================================== */}
        {/* TAB 3: PERFORMANCE ANALYTICS & BENCHMARKS           */}
        {/* ==================================================== */}
        {activeTab === "research" && (
          <div className="space-y-6">
            <div>
              <h1 className="text-2xl font-bold tracking-tight text-white flex items-center space-x-2">
                <span>Performance &amp; Evaluation Benchmarks</span>
                <span className="text-xs font-normal text-brand-400 bg-brand-500/10 px-2.5 py-0.5 rounded-md border border-brand-500/20">
                  Model Telemetry
                </span>
              </h1>
              <p className="text-sm text-slate-400 mt-1">
                Evaluation metrics and benchmarks loaded directly from model evaluation runs (Baseline Model, Noise-Augmented Model, and Selective Classification).
              </p>
            </div>

            {loadingMetrics ? (
              <div className="glass-panel p-12 text-center rounded-2xl flex flex-col items-center justify-center space-y-3">
                <RefreshCw className="w-8 h-8 text-brand-400 animate-spin" />
                <p className="text-sm text-slate-300 font-medium">Loading evaluation results from disk...</p>
              </div>
            ) : metricsData ? (
              <div className="space-y-6">
                {/* 1. Model Comparison Table */}
                {metricsData.data?.comparison?.comparison_table && (
                  <div className="glass-panel p-6 rounded-2xl space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <h2 className="text-base font-bold text-white">
                          Model Comparison: Baseline vs. Noise-Augmented
                        </h2>
                        <p className="text-xs text-slate-400 mt-0.5">
                          Evaluated on held-out test data under variable Signal-to-Noise Ratios (Clean &rarr; 0 dB).
                        </p>
                      </div>
                      <span className="text-xs font-mono text-brand-400 bg-brand-500/10 px-2.5 py-1 rounded-full border border-brand-500/20">
                        10-Fold Segregation
                      </span>
                    </div>

                    <div className="overflow-x-auto">
                      <table className="w-full text-xs text-left border-collapse">
                        <thead>
                          <tr className="border-b border-slate-800 text-slate-400 font-mono">
                            <th className="py-2.5 px-3">Condition (SNR)</th>
                            <th className="py-2.5 px-3">Baseline F1 (Exp A)</th>
                            <th className="py-2.5 px-3">Augmented F1 (Exp B)</th>
                            <th className="py-2.5 px-3 text-brand-400">&Delta; Macro F1</th>
                            <th className="py-2.5 px-3">Baseline Acc</th>
                            <th className="py-2.5 px-3">Augmented Acc</th>
                            <th className="py-2.5 px-3 text-sky-400">&Delta; Accuracy</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/60 font-mono">
                          {metricsData.data.comparison.comparison_table.map((row: any, i: number) => {
                            const dF1 = row.delta_macro_f1 * 100;
                            const dAcc = row.delta_accuracy * 100;
                            return (
                              <tr key={i} className="hover:bg-slate-900/40">
                                <td className="py-2.5 px-3 font-semibold text-white">{row.condition}</td>
                                <td className="py-2.5 px-3 text-slate-300">{(row.baseline_macro_f1 * 100).toFixed(2)}%</td>
                                <td className="py-2.5 px-3 text-slate-200 font-medium">{(row.augmented_macro_f1 * 100).toFixed(2)}%</td>
                                <td className={`py-2.5 px-3 font-bold ${dF1 >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                                  {dF1 >= 0 ? "+" : ""}{dF1.toFixed(2)}%
                                </td>
                                <td className="py-2.5 px-3 text-slate-300">{(row.baseline_accuracy * 100).toFixed(2)}%</td>
                                <td className="py-2.5 px-3 text-slate-200 font-medium">{(row.augmented_accuracy * 100).toFixed(2)}%</td>
                                <td className={`py-2.5 px-3 font-bold ${dAcc >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                                  {dAcc >= 0 ? "+" : ""}{dAcc.toFixed(2)}%
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* 2. Publication Plots Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Robustness SNR Degradation Curve */}
                  {metricsData.plots?.comparison_snr_curve && (
                    <div className="glass-panel p-5 rounded-2xl space-y-3">
                      <div className="flex justify-between items-center">
                        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 font-mono">
                          Macro-F1 vs. Signal-to-Noise Ratio (dB)
                        </h3>
                        <a
                          href={metricsData.plots.comparison_snr_curve}
                          target="_blank"
                          rel="noreferrer"
                          className="text-xs text-brand-400 hover:text-brand-300 flex items-center space-x-1"
                        >
                          <span>Full Res</span>
                          <ExternalLink className="w-3 h-3" />
                        </a>
                      </div>
                      <img
                        src={metricsData.plots.comparison_snr_curve}
                        alt="SNR Robustness Curve"
                        className="rounded-xl border border-slate-800 w-full"
                      />
                    </div>
                  )}

                  {/* Selective Prediction Risk-Coverage Curve */}
                  {metricsData.plots?.risk_coverage_curve && (
                    <div className="glass-panel p-5 rounded-2xl space-y-3">
                      <div className="flex justify-between items-center">
                        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 font-mono">
                          Coverage vs. Selective Accuracy (Exp C)
                        </h3>
                        <a
                          href={metricsData.plots.risk_coverage_curve}
                          target="_blank"
                          rel="noreferrer"
                          className="text-xs text-brand-400 hover:text-brand-300 flex items-center space-x-1"
                        >
                          <span>Full Res</span>
                          <ExternalLink className="w-3 h-3" />
                        </a>
                      </div>
                      <img
                        src={metricsData.plots.risk_coverage_curve}
                        alt="Risk Coverage Curve"
                        className="rounded-xl border border-slate-800 w-full"
                      />
                    </div>
                  )}

                  {/* Reliability Diagram */}
                  {metricsData.plots?.reliability_diagram && (
                    <div className="glass-panel p-5 rounded-2xl space-y-3">
                      <div className="flex justify-between items-center">
                        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 font-mono">
                          Reliability Diagram (Temperature Scaling)
                        </h3>
                        <a
                          href={metricsData.plots.reliability_diagram}
                          target="_blank"
                          rel="noreferrer"
                          className="text-xs text-brand-400 hover:text-brand-300 flex items-center space-x-1"
                        >
                          <span>Full Res</span>
                          <ExternalLink className="w-3 h-3" />
                        </a>
                      </div>
                      <img
                        src={metricsData.plots.reliability_diagram}
                        alt="Reliability Diagram"
                        className="rounded-xl border border-slate-800 w-full"
                      />
                    </div>
                  )}

                  {/* Confusion Matrix */}
                  {metricsData.plots?.baseline_confusion_matrix && (
                    <div className="glass-panel p-5 rounded-2xl space-y-3">
                      <div className="flex justify-between items-center">
                        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 font-mono">
                          Test Confusion Matrix (10 Urban Sound Classes)
                        </h3>
                        <a
                          href={metricsData.plots.baseline_confusion_matrix}
                          target="_blank"
                          rel="noreferrer"
                          className="text-xs text-brand-400 hover:text-brand-300 flex items-center space-x-1"
                        >
                          <span>Full Res</span>
                          <ExternalLink className="w-3 h-3" />
                        </a>
                      </div>
                      <img
                        src={metricsData.plots.baseline_confusion_matrix}
                        alt="Confusion Matrix"
                        className="rounded-xl border border-slate-800 w-full"
                      />
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="glass-panel p-12 text-center rounded-2xl text-slate-400">
                <Info className="w-8 h-8 mx-auto text-slate-600 mb-2" />
                <p>No experiment metrics found. Please train models to view scientific results.</p>
              </div>
            )}
          </div>
        )}

        {/* ==================================================== */}
        {/* TAB 4: MODEL SPECIFICATIONS & TAXONOMY               */}
        {/* ==================================================== */}
        {activeTab === "model_info" && (
          <div className="space-y-6">
            <div>
              <h1 className="text-2xl font-bold tracking-tight text-white flex items-center space-x-2">
                <span>Model Architecture & Taxonomy</span>
                <span className="text-xs font-normal text-emerald-400 bg-emerald-500/10 px-2.5 py-0.5 rounded-md border border-emerald-500/20">
                  Technical Specifications
                </span>
              </h1>
              <p className="text-sm text-slate-400 mt-1">
                Deep neural network specifications, 10-class taxonomy, cross-validation protocol, and scientific limitations.
              </p>
            </div>

            {loadingModelInfo ? (
              <div className="glass-panel p-12 text-center rounded-2xl flex flex-col items-center justify-center space-y-3">
                <RefreshCw className="w-8 h-8 text-brand-400 animate-spin" />
                <p className="text-sm text-slate-300">Fetching technical specifications...</p>
              </div>
            ) : modelInfo ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Architecture Specifications */}
                <div className="glass-panel p-6 rounded-2xl space-y-4">
                  <h2 className="text-sm font-semibold text-white flex items-center space-x-2">
                    <Layers className="w-4 h-4 text-brand-400" />
                    <span>Neural Network Architecture</span>
                  </h2>

                  <div className="space-y-2.5 text-xs">
                    <div className="flex justify-between py-1.5 border-b border-slate-800">
                      <span className="text-slate-400">Model Name:</span>
                      <span className="font-semibold text-white">{modelInfo.model_name}</span>
                    </div>
                    <div className="flex justify-between py-1.5 border-b border-slate-800">
                      <span className="text-slate-400">Total Parameters:</span>
                      <span className="font-mono text-brand-400 font-bold">
                        {modelInfo.parameters.total_parameters.toLocaleString()}
                      </span>
                    </div>
                    <div className="flex justify-between py-1.5 border-b border-slate-800">
                      <span className="text-slate-400">Checkpoint Size:</span>
                      <span className="font-mono text-slate-200">{modelInfo.parameters.model_size_mb} MB</span>
                    </div>
                    <div className="flex justify-between py-1.5 border-b border-slate-800">
                      <span className="text-slate-400">Input Representation:</span>
                      <span className="font-mono text-slate-200">
                        Log-Mel Spectrogram {JSON.stringify(modelInfo.input_representation.tensor_shape)}
                      </span>
                    </div>
                    <div className="flex justify-between py-1.5 border-b border-slate-800">
                      <span className="text-slate-400">Sample Rate:</span>
                      <span className="font-mono text-slate-200">
                        {modelInfo.input_representation.sampling_rate_hz} Hz (Fixed 4.0s)
                      </span>
                    </div>
                    <div className="flex justify-between py-1.5 border-b border-slate-800">
                      <span className="text-slate-400">Calibration Method:</span>
                      <span className="text-emerald-400 font-medium">
                        {modelInfo.confidence_calibration.calibration_method}
                      </span>
                    </div>
                  </div>
                </div>

                {/* UrbanSound8K Taxonomy */}
                <div className="glass-panel p-6 rounded-2xl space-y-4">
                  <h2 className="text-sm font-semibold text-white flex items-center space-x-2">
                    <Volume2 className="w-4 h-4 text-sky-400" />
                    <span>UrbanSound8K Taxonomy (10 Classes)</span>
                  </h2>

                  <div className="grid grid-cols-2 gap-2">
                    {modelInfo.taxonomy.classes.map((cls: string, idx: number) => (
                      <div
                        key={idx}
                        className="p-2.5 rounded-xl bg-slate-900 border border-slate-800 text-xs flex items-center justify-between"
                      >
                        <span className="font-mono text-slate-500">#{idx}</span>
                        <span className="font-medium text-slate-200 capitalize">
                          {cls.replace(/_/g, " ")}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Technical Limitations */}
                <div className="glass-panel p-6 rounded-2xl space-y-3 md:col-span-2">
                  <h2 className="text-sm font-semibold text-white flex items-center space-x-2">
                    <HelpCircle className="w-4 h-4 text-amber-400" />
                    <span>Technical &amp; Operational Limitations</span>
                  </h2>

                  <ul className="space-y-2 text-xs text-slate-400">
                    {modelInfo.known_limitations.map((item: string, i: number) => (
                      <li key={i} className="flex items-start space-x-2">
                        <span className="text-brand-400 font-bold">&bull;</span>
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            ) : null}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 py-6 text-center text-xs text-slate-500 font-mono">
        AcoustiGuard AI &bull; Environmental Audio Classification Application &bull; UrbanSound8K
      </footer>
    </div>
  );
}
