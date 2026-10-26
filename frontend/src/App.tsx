import React, { useState, useEffect, useRef } from "react";
import {
  Activity,
  Sliders,
  Sparkles,
  Upload,
  Play,
  Pause,
  AlertTriangle,
  RefreshCw,
  SlidersHorizontal,
  Volume2,
  Cpu,
  Layers,
  HelpCircle,
  TrendingDown,
  Maximize2,
  CheckCircle2,
  ChevronRight,
  Terminal,
  Zap,
  Radio,
  FileAudio,
  Trash2,
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

  // Metrics & Model Info
  const [metricsData, setMetricsData] = useState<any | null>(null);
  const [loadingMetrics, setLoadingMetrics] = useState<boolean>(false);
  const [modelInfo, setModelInfo] = useState<any | null>(null);
  const [loadingModelInfo, setLoadingModelInfo] = useState<boolean>(false);

  // Audio Player State
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const audioSectionRef = useRef<HTMLDivElement | null>(null);

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

  // 2. Fetch Metrics on Tab Switch
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

  const handleClearFile = () => {
    setSelectedFile(null);
    setAudioUrl(null);
    setPrediction(null);
    setAnalysisError(null);
    setIsPlaying(false);
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
        const burst = t % 1.2 < 0.8 ? 1.0 : 0.05;
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

  const scrollToAudioSection = () => {
    audioSectionRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="min-h-screen flex flex-col font-sans bg-[#101820] text-[#F4F0E8] antialiased selection:bg-[#C7F464] selection:text-[#101820]">
      {/* Top Console Navigation & Telemetry Bar */}
      <header className="sticky top-0 z-50 bg-[#101820]/95 border-b border-[#202A33] px-4 md:px-8 py-3.5 backdrop-blur-md">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          {/* Audio Intelligence Brand Mark */}
          <div className="flex items-center space-x-3.5">
            <div className="w-9 h-9 bg-[#202A33] border border-[#2c3b47] flex items-center justify-center relative shadow-sm">
              <div className="flex items-end space-x-0.5 h-4">
                <span className="w-1 bg-[#00F0FF] h-2.5" title="Cyan (High-Freq)" />
                <span className="w-1 bg-[#C7F464] h-4" title="Lime (Mid-High)" />
                <span className="w-1 bg-[#BD00FF] h-3" title="Violet (Resonance)" />
                <span className="w-1 bg-[#FF6B35] h-3.5" title="Orange (Transient)" />
                <span className="w-1 bg-[#F59E0B] h-1.5" title="Amber (Sub-Bass)" />
              </div>
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-display font-bold text-lg tracking-tight text-[#F4F0E8]">
                  AcoustiGuard AI
                </span>
                <span className="px-2 py-0.5 text-[10px] font-mono uppercase font-semibold bg-[#202A33] text-[#00F0FF] border border-[#00F0FF]/30">
                  UrbanSound8K
                </span>
              </div>
              <p className="text-[11px] text-[#64748B] hidden sm:block font-mono tracking-tight">
                Noise-Robust Environmental Audio Classification
              </p>
            </div>
          </div>

          {/* Rectangular Navigation Tabs with Colored Accents */}
          <nav className="flex items-center space-x-1 bg-[#141e28] p-1 border border-[#202A33]">
            <button
              onClick={() => setActiveTab("analyze")}
              className={`px-3.5 py-1.5 text-xs font-mono font-medium transition-all focus:outline-none focus:ring-1 focus:ring-[#C7F464] ${
                activeTab === "analyze"
                  ? "bg-[#C7F464] text-[#101820] font-bold shadow-sharp-lime"
                  : "text-[#ded7c8] hover:text-[#C7F464] hover:bg-[#202A33]"
              }`}
            >
              [ 01 // Audio Analysis ]
            </button>

            <button
              onClick={() => setActiveTab("robustness")}
              className={`px-3.5 py-1.5 text-xs font-mono font-medium transition-all focus:outline-none focus:ring-1 focus:ring-[#00F0FF] ${
                activeTab === "robustness"
                  ? "bg-[#00F0FF] text-[#101820] font-bold shadow-sharp-cyan"
                  : "text-[#ded7c8] hover:text-[#00F0FF] hover:bg-[#202A33]"
              }`}
            >
              [ 02 // Robustness Lab ]
            </button>

            <button
              onClick={() => setActiveTab("research")}
              className={`px-3.5 py-1.5 text-xs font-mono font-medium transition-all focus:outline-none focus:ring-1 focus:ring-[#BD00FF] ${
                activeTab === "research"
                  ? "bg-[#BD00FF] text-white font-bold shadow-sharp-violet"
                  : "text-[#ded7c8] hover:text-[#D946EF] hover:bg-[#202A33]"
              }`}
            >
              [ 03 // Performance Analytics ]
            </button>

            <button
              onClick={() => setActiveTab("model_info")}
              className={`px-3.5 py-1.5 text-xs font-mono font-medium transition-all focus:outline-none focus:ring-1 focus:ring-[#F59E0B] ${
                activeTab === "model_info"
                  ? "bg-[#F59E0B] text-[#101820] font-bold shadow-sharp-amber"
                  : "text-[#ded7c8] hover:text-[#F59E0B] hover:bg-[#202A33]"
              }`}
            >
              [ 04 // Model &amp; Taxonomy ]
            </button>
          </nav>

          {/* Hardware & Telemetry LED Badge */}
          <div className="flex items-center space-x-2">
            <div className="flex items-center space-x-2 px-3 py-1 bg-[#141e28] border border-[#202A33] text-[11px] font-mono">
              <span
                className={`w-2 h-2 ${
                  backendOnline ? "bg-[#10B981] animate-pulse glow-emerald" : "bg-[#FF6B35]"
                }`}
              />
              <span className={`font-bold ${backendOnline ? "text-[#10B981]" : "text-[#FF6B35]"}`}>
                {backendOnline ? "ENGINE ONLINE" : "OFFLINE"}
              </span>
              {health?.device && (
                <span className="text-[#00F0FF] border-l border-[#202A33] pl-2 uppercase font-bold">
                  {health.device}
                </span>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-8 space-y-8">
        {/* Offline Helper Alert */}
        {!backendOnline && !loadingHealth && (
          <div className="p-4 bg-[#202A33] border-l-4 border-[#FF6B35] text-xs font-mono flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <AlertTriangle className="w-5 h-5 text-[#FF6B35] shrink-0" />
              <span>
                <strong className="text-[#F4F0E8]">FastAPI Backend is currently offline.</strong> Start both backend and frontend servers with{" "}
                <code className="bg-[#101820] px-2 py-0.5 text-[#C7F464] border border-[#2c3b47]">
                  python run.py
                </code>.
              </span>
            </div>
            <button
              onClick={checkHealth}
              className="px-3 py-1.5 bg-[#FF6B35] hover:bg-[#f35a22] text-[#101820] font-mono font-bold text-xs uppercase transition shrink-0"
            >
              Retry Connection
            </button>
          </div>
        )}

        {/* ==================================================== */}
        {/* TAB 1: AUDIO ANALYSIS & PREDICTION                   */}
        {/* ==================================================== */}
        {activeTab === "analyze" && (
          <div className="space-y-8">
            {/* HERO SECTION: Asymmetric Two-Column Editorial Header */}
            <div className="border border-[#202A33] bg-[#141e28] p-6 md:p-10 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-96 h-96 bg-gradient-to-br from-[#C7F464]/5 to-transparent pointer-events-none" />

              <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center relative z-10">
                {/* Left Column: Bold Typography & Action */}
                <div className="lg:col-span-7 space-y-5">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="px-2.5 py-1 text-[10px] font-mono uppercase font-bold bg-[#00F0FF]/10 text-[#00F0FF] border border-[#00F0FF]/40 shadow-sm">
                      AUDIO INTELLIGENCE
                    </span>
                    <span className="px-2.5 py-1 text-[10px] font-mono uppercase font-bold bg-[#BD00FF]/10 text-[#D946EF] border border-[#BD00FF]/40 shadow-sm">
                      10 SOUND CLASSES
                    </span>
                    <span className="px-2.5 py-1 text-[10px] font-mono uppercase font-bold bg-[#FF6B35]/10 text-[#FF6B35] border border-[#FF6B35]/40 shadow-sm">
                      NOISE ROBUSTNESS
                    </span>
                    <span className="px-2.5 py-1 text-[10px] font-mono uppercase font-bold bg-[#F59E0B]/10 text-[#F59E0B] border border-[#F59E0B]/40 shadow-sm">
                      CONFIDENCE-AWARE
                    </span>
                  </div>

                  <h1 className="text-4xl md:text-5xl lg:text-6xl font-display font-bold tracking-tight text-[#F4F0E8] leading-[1.05]">
                    Every Sound <br />
                    <span className="bg-gradient-to-r from-[#C7F464] via-[#00F0FF] to-[#BD00FF] bg-clip-text text-transparent underline decoration-[#00F0FF] decoration-2 underline-offset-8">
                      Has a Story.
                    </span>
                  </h1>

                  <p className="text-sm md:text-base text-[#ded7c8] max-w-xl leading-relaxed">
                    AcoustiGuard AI decodes urban environmental acoustic signatures with deep convolutional neural networks.
                    Features multi-SNR noise tolerance, confidence calibration, and selective classification to isolate ambiguous signals.
                  </p>

                  <div className="flex flex-wrap items-center gap-4 pt-2">
                    <button
                      onClick={scrollToAudioSection}
                      className="px-6 py-3 bg-[#C7F464] hover:bg-[#b6e94e] text-[#101820] font-display font-bold text-sm tracking-wide transition shadow-sharp-lime flex items-center space-x-2 focus:outline-none focus:ring-2 focus:ring-[#C7F464]"
                    >
                      <Zap className="w-4 h-4 fill-current" />
                      <span>Analyze Audio</span>
                    </button>

                    <div className="flex items-center space-x-2 text-xs font-mono text-[#ded7c8] border border-[#202A33] px-3.5 py-2.5 bg-[#101820]">
                      <Terminal className="w-3.5 h-3.5 text-[#00F0FF]" />
                      <span>22.05 kHz &bull; 128 Mels &bull; 88,200 Samples</span>
                    </div>
                  </div>
                </div>

                {/* Right Column: Decorative Multi-Band Soundwave & Spectrum Visualizer */}
                <div className="lg:col-span-5 border border-[#202A33] bg-[#101820] p-6 space-y-4 accent-edge-cyan">
                  <div className="flex items-center justify-between text-xs font-mono text-[#64748B] border-b border-[#202A33] pb-2">
                    <span className="text-[#ded7c8]">LIVE SPECTRUM VISUALIZER</span>
                    <span className="text-[#00F0FF] font-bold">CH 01 // MULTI-BAND</span>
                  </div>

                  {/* Multi-Chromatic Frequency Spectrum Bars */}
                  <div className="h-36 flex items-end justify-between gap-1 px-1 pt-4">
                    {[
                      35, 60, 85, 95, 65, 100, 80, 45, 90, 95, 70, 45, 80, 90, 100, 85, 60, 95, 75, 50, 85, 70,
                      55, 80, 95, 65, 40, 90, 75, 50, 65, 85,
                    ].map((heightPct, idx) => {
                      // Rich Audio Hardware Color Sequencing: Cyan -> Lime -> Violet -> Orange -> Amber
                      const barColor =
                        idx % 5 === 0
                          ? "bg-[#00F0FF] shadow-[0_0_8px_rgba(0,240,255,0.4)]"
                          : idx % 5 === 1
                          ? "bg-[#C7F464] shadow-[0_0_8px_rgba(199,244,100,0.4)]"
                          : idx % 5 === 2
                          ? "bg-[#BD00FF] shadow-[0_0_8px_rgba(189,0,255,0.4)]"
                          : idx % 5 === 3
                          ? "bg-[#FF6B35] shadow-[0_0_8px_rgba(255,107,53,0.4)]"
                          : "bg-[#F59E0B] shadow-[0_0_8px_rgba(245,158,11,0.4)]";

                      return (
                        <div
                          key={idx}
                          className={`w-full transition-all duration-300 ${barColor}`}
                          style={{ height: `${heightPct}%` }}
                        />
                      );
                    })}
                  </div>

                  <div className="grid grid-cols-3 gap-2 pt-2 border-t border-[#202A33] text-[10px] font-mono">
                    <div className="text-center p-2 bg-[#141e28] border border-[#202A33]">
                      <div className="text-[#64748B]">LATENCY</div>
                      <div className="text-[#00F0FF] font-bold mt-0.5">~13.4 ms</div>
                    </div>
                    <div className="text-center p-2 bg-[#141e28] border border-[#202A33]">
                      <div className="text-[#64748B]">PARAMETERS</div>
                      <div className="text-[#D946EF] font-bold mt-0.5">767.7 K</div>
                    </div>
                    <div className="text-center p-2 bg-[#141e28] border border-[#202A33]">
                      <div className="text-[#64748B]">CHECKPOINT</div>
                      <div className="text-[#C7F464] font-bold mt-0.5">2.93 MB</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* SECTION 2: AUDIO WORKSTATION & PREDICTION */}
            <div ref={audioSectionRef} className="space-y-6">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-[#202A33] pb-4">
                <div>
                  <h2 className="text-xl md:text-2xl font-display font-bold text-[#F4F0E8] flex items-center space-x-2">
                    <span>Environmental Sound Classifier</span>
                    <span className="text-[10px] font-mono px-2 py-0.5 bg-[#202A33] text-[#00F0FF] border border-[#00F0FF]/30">
                      Confidence-Aware
                    </span>
                  </h2>
                  <p className="text-xs text-[#ded7c8] mt-1 font-mono">
                    Upload an audio recording or select an acoustic preset to extract log-mel features and classify.
                  </p>
                </div>

                {/* Instant Preset Buttons with Signature Category Color Themes */}
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-xs font-mono text-[#64748B] mr-0.5">PRESETS:</span>
                  <button
                    type="button"
                    onClick={() => generateSamplePreset("siren", "analyze")}
                    className="px-3 py-1.5 text-xs font-mono font-bold bg-[#FF6B35]/10 hover:bg-[#FF6B35]/25 text-[#FF6B35] border border-[#FF6B35]/50 hover:border-[#FF6B35] shadow-sm transition"
                  >
                    🚨 SIREN
                  </button>
                  <button
                    type="button"
                    onClick={() => generateSamplePreset("car_horn", "analyze")}
                    className="px-3 py-1.5 text-xs font-mono font-bold bg-[#F59E0B]/10 hover:bg-[#F59E0B]/25 text-[#F59E0B] border border-[#F59E0B]/50 hover:border-[#F59E0B] shadow-sm transition"
                  >
                    🚗 HORN
                  </button>
                  <button
                    type="button"
                    onClick={() => generateSamplePreset("engine_idling", "analyze")}
                    className="px-3 py-1.5 text-xs font-mono font-bold bg-[#00F0FF]/10 hover:bg-[#00F0FF]/25 text-[#00F0FF] border border-[#00F0FF]/50 hover:border-[#00F0FF] shadow-sm transition"
                  >
                    🚚 ENGINE
                  </button>
                  <button
                    type="button"
                    onClick={() => generateSamplePreset("jackhammer", "analyze")}
                    className="px-3 py-1.5 text-xs font-mono font-bold bg-[#BD00FF]/10 hover:bg-[#BD00FF]/25 text-[#D946EF] border border-[#BD00FF]/50 hover:border-[#BD00FF] shadow-sm transition"
                  >
                    🔨 JACKHAMMER
                  </button>
                </div>
              </div>

              {/* Workstation Grid: Input Panel (5 cols) + Results (7 cols) */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                {/* Left Column: Upload Workstation & Parameters */}
                <div className="lg:col-span-5 space-y-6">
                  {/* Upload Card */}
                  <div className="bg-[#141e28] border border-[#202A33] p-6 space-y-5 accent-edge-lime">
                    <div className="flex items-center justify-between border-b border-[#202A33] pb-3">
                      <h3 className="text-xs font-mono uppercase tracking-wider text-[#F4F0E8] flex items-center space-x-2">
                        <span className="text-[#C7F464] font-bold">01 //</span>
                        <span>Audio Input</span>
                      </h3>
                      {selectedFile && (
                        <div className="flex items-center space-x-2">
                          <span className="text-xs font-mono text-[#C7F464] truncate max-w-[140px]">
                            {selectedFile.name}
                          </span>
                          <button
                            onClick={handleClearFile}
                            title="Clear loaded audio"
                            className="p-1 hover:text-[#FF6B35] text-[#64748B] transition"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      )}
                    </div>

                    {/* Square Dropzone */}
                    <label
                      htmlFor="audio-upload"
                      className="border-2 border-dashed border-[#202A33] hover:border-[#C7F464] p-8 flex flex-col items-center justify-center text-center cursor-pointer transition-colors bg-[#101820] hover:bg-[#15202b] group relative"
                    >
                      <div className="w-12 h-12 bg-[#202A33] group-hover:bg-[#C7F464] text-[#C7F464] group-hover:text-[#101820] flex items-center justify-center mb-3 transition">
                        <Upload className="w-6 h-6" />
                      </div>
                      <span className="text-sm font-display font-bold text-[#F4F0E8] group-hover:text-[#C7F464]">
                        Click to choose audio or drop file
                      </span>
                      <span className="text-xs font-mono text-[#64748B] mt-1.5">
                        WAV, MP3, FLAC, OGG &bull; Up to 4.0s @ 22.05 kHz
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
                      <div className="p-3 bg-[#101820] border border-[#202A33] flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <button
                            onClick={toggleAudio}
                            className="w-9 h-9 bg-[#C7F464] hover:bg-[#b6e94e] text-[#101820] flex items-center justify-center font-bold transition shadow-sm"
                          >
                            {isPlaying ? <Pause className="w-4 h-4 fill-current" /> : <Play className="w-4 h-4 fill-current ml-0.5" />}
                          </button>
                          <div className="text-xs">
                            <p className="font-mono text-[#F4F0E8] truncate max-w-[200px]">
                              {selectedFile?.name || "Uploaded Audio"}
                            </p>
                            <span className="text-[10px] text-[#64748B] font-mono">22,050 Hz Mono &bull; Standardized</span>
                          </div>
                        </div>
                        <Volume2 className="w-4 h-4 text-[#64748B]" />
                        <audio
                          ref={audioRef}
                          src={audioUrl}
                          onEnded={() => setIsPlaying(false)}
                          className="hidden"
                        />
                      </div>
                    )}

                    {/* Inference Parameters */}
                    <div className="space-y-4 pt-3 border-t border-[#202A33]">
                      <h4 className="text-xs font-mono uppercase tracking-wider text-[#64748B] flex items-center space-x-1.5">
                        <SlidersHorizontal className="w-3.5 h-3.5" />
                        <span>Inference Parameters</span>
                      </h4>

                      {/* Model Selector */}
                      <div>
                        <label className="text-xs font-mono text-[#ded7c8] block mb-1.5">MODEL VARIANT</label>
                        <div className="grid grid-cols-2 gap-2">
                          <button
                            type="button"
                            onClick={() => setModelType("augmented")}
                            className={`p-3 text-left border transition ${
                              modelType === "augmented"
                                ? "bg-[#101820] border-[#C7F464] shadow-sharp-lime"
                                : "bg-[#101820] border-[#202A33] hover:border-[#64748B]"
                            }`}
                          >
                            <div className={`font-display font-bold text-xs ${modelType === "augmented" ? "text-[#C7F464]" : "text-[#F4F0E8]"}`}>
                              Noise-Augmented
                            </div>
                            <div className="text-[10px] font-mono text-[#64748B] mt-0.5">Trained with noise defense</div>
                          </button>

                          <button
                            type="button"
                            onClick={() => setModelType("baseline")}
                            className={`p-3 text-left border transition ${
                              modelType === "baseline"
                                ? "bg-[#101820] border-[#C7F464] shadow-sharp-lime"
                                : "bg-[#101820] border-[#202A33] hover:border-[#64748B]"
                            }`}
                          >
                            <div className={`font-display font-bold text-xs ${modelType === "baseline" ? "text-[#C7F464]" : "text-[#F4F0E8]"}`}>
                              Baseline CNN
                            </div>
                            <div className="text-[10px] font-mono text-[#64748B] mt-0.5">Clean audio training only</div>
                          </button>
                        </div>
                      </div>

                      {/* Confidence Threshold Slider */}
                      <div>
                        <div className="flex justify-between text-xs font-mono mb-1.5">
                          <span className="text-[#ded7c8]">CONFIDENCE THRESHOLD (&tau;)</span>
                          <span className="text-[#C7F464] font-bold">{(threshold * 100).toFixed(0)}%</span>
                        </div>
                        <input
                          type="range"
                          min="0.50"
                          max="0.95"
                          step="0.05"
                          value={threshold}
                          onChange={(e) => setThreshold(parseFloat(e.target.value))}
                          className="w-full h-2 cursor-pointer"
                        />
                        <div className="flex justify-between text-[10px] text-[#64748B] mt-1 font-mono">
                          <span>50% (Permissive)</span>
                          <span>70% (Default)</span>
                          <span>95% (Strict)</span>
                        </div>
                      </div>

                      {/* Temperature Scaling Toggle */}
                      <div className="flex items-center justify-between p-3 bg-[#101820] border border-[#202A33] text-xs">
                        <div>
                          <div className="font-mono text-[#F4F0E8] font-bold">Temperature Calibration</div>
                          <div className="text-[10px] font-mono text-[#64748B]">Post-hoc NLL calibration on validation fold</div>
                        </div>
                        <button
                          type="button"
                          onClick={() => setUseCalibration(!useCalibration)}
                          className={`px-3 py-1 text-xs font-mono font-bold transition border ${
                            useCalibration
                              ? "bg-[#C7F464] text-[#101820] border-[#C7F464]"
                              : "bg-[#202A33] text-[#64748B] border-[#202A33]"
                          }`}
                        >
                          {useCalibration ? "ENABLED" : "BYPASS"}
                        </button>
                      </div>

                      {/* Action Button: Rectangular Sharp Electric Lime */}
                      <button
                        disabled={!selectedFile || isAnalyzing}
                        onClick={handleAnalyze}
                        className="w-full py-3.5 bg-[#C7F464] hover:bg-[#b6e94e] text-[#101820] font-display font-bold text-sm tracking-wide shadow-sharp-lime disabled:opacity-40 disabled:cursor-not-allowed transition flex items-center justify-center space-x-2"
                      >
                        {isAnalyzing ? (
                          <>
                            <RefreshCw className="w-4 h-4 animate-spin" />
                            <span>Processing Audio Features...</span>
                          </>
                        ) : (
                          <>
                            <Activity className="w-4 h-4" />
                            <span>Classify Environmental Sound</span>
                          </>
                        )}
                      </button>

                      {analysisError && (
                        <div className="p-3 bg-[#202A33] border-l-4 border-[#FF6B35] text-[#FF6B35] text-xs font-mono">
                          {analysisError}
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Right Column: Prediction Results & Spectrogram */}
                <div className="lg:col-span-7 space-y-6">
                  {prediction ? (
                    <div className="space-y-6">
                      {/* Prediction Hero Card */}
                      <div className="bg-[#141e28] border border-[#202A33] p-6 relative overflow-hidden accent-edge-lime">
                        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-[#202A33] pb-4">
                          <div>
                            <span className="text-[10px] font-mono uppercase tracking-wider text-[#64748B]">
                              IDENTIFIED CLASS [TARGET]
                            </span>
                            <h2 className="text-3xl md:text-4xl font-display font-bold text-[#F4F0E8] capitalize tracking-tight mt-1">
                              {prediction.predicted_class_name.replace(/_/g, " ")}
                            </h2>
                          </div>

                          <div className="flex items-center space-x-3">
                            <div className="text-right">
                              <span className="text-[10px] font-mono text-[#64748B]">CONFIDENCE</span>
                              <div className="text-2xl font-mono font-bold text-[#C7F464]">
                                {(prediction.confidence * 100).toFixed(1)}%
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* Geometric Confidence Meter */}
                        <div className="mt-4 space-y-1.5">
                          <div className="flex justify-between text-[11px] font-mono text-[#64748B]">
                            <span>Softmax Probability</span>
                            <span>Threshold: {(prediction.threshold_applied * 100).toFixed(0)}%</span>
                          </div>
                          <div className="h-3 w-full bg-[#101820] border border-[#202A33] p-0.5">
                            <div
                              className="h-full bg-[#C7F464] transition-all duration-500"
                              style={{ width: `${Math.min(100, prediction.confidence * 100)}%` }}
                            />
                          </div>
                        </div>

                        {/* Uncertainty Alert */}
                        {prediction.is_uncertain ? (
                          <div className="mt-4 p-3 bg-[#202A33] border-l-4 border-[#FF6B35] text-[#FF6B35] text-xs font-mono flex items-center justify-between">
                            <div className="flex items-center space-x-2">
                              <AlertTriangle className="w-4 h-4 shrink-0" />
                              <span>{prediction.status_message}</span>
                            </div>
                            <span className="font-bold uppercase tracking-wider text-[10px] bg-[#101820] px-2 py-0.5 border border-[#FF6B35]">
                              LOW CONFIDENCE
                            </span>
                          </div>
                        ) : (
                          <div className="mt-4 p-3 bg-[#202A33] border-l-4 border-[#C7F464] text-[#C7F464] text-xs font-mono flex items-center justify-between">
                            <div className="flex items-center space-x-2">
                              <CheckCircle2 className="w-4 h-4 shrink-0" />
                              <span>{prediction.status_message}</span>
                            </div>
                            <span className="font-bold uppercase tracking-wider text-[10px] bg-[#101820] px-2 py-0.5 border border-[#C7F464]">
                              CONFIDENT
                            </span>
                          </div>
                        )}

                        {/* Telemetry Micro-Grid */}
                        <div className="grid grid-cols-3 gap-2 mt-4 pt-4 border-t border-[#202A33] text-[10px] font-mono">
                          <div>
                            <span className="text-[#64748B]">INFERENCE TIME</span>
                            <div className="text-[#F4F0E8] font-bold mt-0.5">{prediction.inference_latency_ms.toFixed(1)} ms</div>
                          </div>
                          <div>
                            <span className="text-[#64748B]">TEMPERATURE</span>
                            <div className="text-[#F4F0E8] font-bold mt-0.5">T = {prediction.temperature.toFixed(3)}</div>
                          </div>
                          <div>
                            <span className="text-[#64748B]">ARCHITECTURE</span>
                            <div className="text-[#C7F464] font-bold mt-0.5">{prediction.model_used}</div>
                          </div>
                        </div>
                      </div>

                      {/* Extracted Features: Waveform & Spectrogram */}
                      <div className="bg-[#141e28] border border-[#202A33] p-6 space-y-4">
                        <div className="flex items-center justify-between border-b border-[#202A33] pb-2">
                          <h3 className="text-xs font-mono uppercase tracking-wider text-[#F4F0E8] flex items-center space-x-2">
                            <span className="text-[#C7F464] font-bold">02 //</span>
                            <span>Extracted Acoustic Features</span>
                          </h3>
                          <span className="text-[10px] font-mono text-[#64748B]">
                            128 Mels &bull; 88,200 Samples
                          </span>
                        </div>

                        {/* Waveform Micro-Visualizer */}
                        <div>
                          <div className="text-[10px] font-mono text-[#64748B] mb-1">
                            NORMALIZED WAVEFORM ENVELOPE (TIME DOMAIN)
                          </div>
                          <div className="h-16 w-full bg-[#101820] p-1 border border-[#202A33] flex items-center justify-between gap-0.5 overflow-hidden">
                            {prediction.waveform_preview.map((val, idx) => {
                              const barHeight = Math.max(4, Math.abs(val) * 55);
                              return (
                                <div
                                  key={idx}
                                  className="w-full bg-[#C7F464] hover:bg-white transition-all"
                                  style={{ height: `${barHeight}px` }}
                                />
                              );
                            })}
                          </div>
                        </div>

                        {/* Spectrogram Grid */}
                        <div>
                          <div className="text-[10px] font-mono text-[#64748B] mb-1">
                            LOG-MEL SPECTROGRAM HEATMAP (128 MEL BINS &times; TIME FRAMES)
                          </div>
                          <div className="h-28 w-full bg-[#101820] p-1 border border-[#202A33] grid grid-rows-8 grid-flow-col gap-0.5 overflow-hidden">
                            {prediction.spectrogram_preview.slice(0, 8).map((row, rIdx) =>
                              row.slice(0, 32).map((cellVal, cIdx) => {
                                const normIntensity = Math.min(1, Math.max(0.08, (cellVal + 2) / 4));
                                return (
                                  <div
                                    key={`${rIdx}-${cIdx}`}
                                    className="w-full h-full spectrogram-cell"
                                    style={{
                                      backgroundColor: `rgba(199, 244, 100, ${normIntensity})`,
                                    }}
                                    title={`Mel: ${rIdx}, Frame: ${cIdx}, Val: ${cellVal.toFixed(2)}`}
                                  />
                                );
                              })
                            )}
                          </div>
                        </div>
                      </div>

                      {/* 10-Class Probability Distribution List */}
                      <div className="bg-[#141e28] border border-[#202A33] p-6 space-y-3">
                        <div className="flex items-center justify-between border-b border-[#202A33] pb-2">
                          <h3 className="text-xs font-mono uppercase tracking-wider text-[#F4F0E8] flex items-center space-x-2">
                            <span className="text-[#C7F464] font-bold">03 //</span>
                            <span>Ranked Class Probabilities</span>
                          </h3>
                          <span className="text-[10px] font-mono text-[#64748B]">10 UrbanSound8K Classes</span>
                        </div>

                        <div className="space-y-2 pt-1">
                          {prediction.probabilities.map((item, idx) => {
                            // Multi-spectral ranking colors
                            const rankColor =
                              idx === 0
                                ? { text: "text-[#C7F464] font-bold", bar: "bg-[#C7F464] shadow-[0_0_6px_rgba(199,244,100,0.4)]" }
                                : idx === 1
                                ? { text: "text-[#00F0FF] font-semibold", bar: "bg-[#00F0FF] shadow-[0_0_6px_rgba(0,240,255,0.4)]" }
                                : idx === 2
                                ? { text: "text-[#D946EF]", bar: "bg-[#BD00FF]" }
                                : idx === 3
                                ? { text: "text-[#F59E0B]", bar: "bg-[#F59E0B]" }
                                : { text: "text-[#ded7c8]", bar: "bg-[#2c3b47]" };

                            return (
                              <div key={item.class_id} className="space-y-1">
                                <div className="flex justify-between text-xs font-mono">
                                  <span className={rankColor.text}>
                                    {String(idx + 1).padStart(2, "0")}. {item.class_name.replace(/_/g, " ")}
                                  </span>
                                  <span className={idx === 0 ? "font-bold text-[#C7F464]" : "text-[#64748B]"}>
                                    {(item.probability * 100).toFixed(1)}%
                                  </span>
                                </div>
                                <div className="w-full bg-[#101820] h-1.5 border border-[#202A33] overflow-hidden">
                                  <div
                                    className={`h-full transition-all duration-300 ${rankColor.bar}`}
                                    style={{ width: `${item.probability * 100}%` }}
                                  />
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    </div>
                  ) : (
                    /* Empty State Placeholder */
                    <div className="bg-[#141e28] border border-[#202A33] p-12 text-center flex flex-col items-center justify-center space-y-4">
                      <div className="w-14 h-14 bg-[#101820] border border-[#202A33] flex items-center justify-center text-[#64748B]">
                        <Volume2 className="w-7 h-7" />
                      </div>
                      <div>
                        <h3 className="text-lg font-display font-bold text-[#F4F0E8]">No Audio Ingested</h3>
                        <p className="text-xs font-mono text-[#64748B] max-w-sm mt-1">
                          Drop an audio recording in the upload area or select a preset above to begin feature extraction and classification.
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ==================================================== */}
        {/* TAB 2: ROBUSTNESS LAB (CONTROLLED NOISE INJECTION)    */}
        {/* ==================================================== */}
        {activeTab === "robustness" && (
          <div className="space-y-8">
            <div className="border border-[#202A33] bg-[#141e28] p-6 md:p-8 accent-edge-lime">
              <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                <div>
                  <div className="flex items-center space-x-2">
                    <span className="text-[10px] font-mono px-2 py-0.5 bg-[#202A33] text-[#C7F464]">
                      LABORATORY
                    </span>
                    <span className="text-[10px] font-mono text-[#64748B]">
                      CONTROLLED SNR DEGRADATION
                    </span>
                  </div>
                  <h1 className="text-3xl font-display font-bold text-[#F4F0E8] mt-1">
                    Noise Robustness Testing Laboratory
                  </h1>
                  <p className="text-xs font-mono text-[#ded7c8] mt-1 max-w-2xl">
                    Stress-test environmental recordings under controlled synthetic noise (20, 10, 5, 0 dB SNR).
                    Observe exact confidence shifts and verify whether predictions withstand acoustic interference.
                  </p>
                </div>
              </div>
            </div>

            {/* Test Configuration Controls */}
            <div className="bg-[#141e28] border border-[#202A33] p-6 space-y-6">
              <h3 className="text-xs font-mono uppercase tracking-wider text-[#64748B] flex items-center space-x-2 border-b border-[#202A33] pb-2">
                <Sliders className="w-4 h-4 text-[#C7F464]" />
                <span>Controlled Degradation Parameters</span>
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Audio Source */}
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <label className="text-xs font-mono text-[#ded7c8]">AUDIO SOURCE</label>
                    <button
                      type="button"
                      onClick={() => generateSamplePreset("siren", "robustness")}
                      className="text-[10px] font-mono text-[#C7F464] hover:underline flex items-center space-x-1"
                    >
                      <Sparkles className="w-3 h-3 text-[#C7F464]" />
                      <span>Use Test Siren</span>
                    </button>
                  </div>
                  <label className="border border-[#202A33] hover:border-[#C7F464] p-3 flex items-center space-x-3 cursor-pointer bg-[#101820] transition">
                    <Upload className="w-4 h-4 text-[#C7F464]" />
                    <span className="text-xs font-mono text-[#F4F0E8] truncate">
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
                  <div className="flex justify-between text-xs font-mono mb-1.5">
                    <span className="text-[#ded7c8]">TARGET SNR LEVEL</span>
                    <span className="text-[#C7F464] font-bold">{targetSnr} dB</span>
                  </div>
                  <div className="grid grid-cols-4 gap-1.5">
                    {[20.0, 10.0, 5.0, 0.0].map((val) => (
                      <button
                        key={val}
                        type="button"
                        onClick={() => setTargetSnr(val)}
                        className={`py-2 text-xs font-mono font-bold border transition ${
                          targetSnr === val
                            ? "bg-[#C7F464] text-[#101820] border-[#C7F464] shadow-sharp-lime"
                            : "bg-[#101820] border-[#202A33] text-[#ded7c8] hover:border-[#64748B]"
                        }`}
                      >
                        {val} dB
                      </button>
                    ))}
                  </div>
                </div>

                {/* Noise Profile */}
                <div>
                  <label className="text-xs font-mono text-[#ded7c8] block mb-1.5">NOISE PROFILE</label>
                  <select
                    value={noiseType}
                    onChange={(e) => setNoiseType(e.target.value)}
                    className="w-full bg-[#101820] border border-[#202A33] px-3 py-2 text-xs font-mono text-[#F4F0E8] focus:outline-none focus:border-[#C7F464]"
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
                  className="px-6 py-3 font-display font-bold text-xs bg-[#C7F464] hover:bg-[#b6e94e] text-[#101820] shadow-sharp-lime disabled:opacity-40 transition flex items-center space-x-2"
                >
                  {isTestingRobustness ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      <span>Injecting Noise & Testing...</span>
                    </>
                  ) : (
                    <>
                      <Zap className="w-3.5 h-3.5 fill-current" />
                      <span>Run Robustness Comparison</span>
                    </>
                  )}
                </button>
              </div>

              {robError && (
                <div className="p-3 bg-[#202A33] border-l-4 border-[#FF6B35] text-[#FF6B35] text-xs font-mono">
                  {robError}
                </div>
              )}
            </div>

            {/* Side-by-Side Robustness Results */}
            {robustnessResult && (
              <div className="space-y-6">
                {/* Summary Banner */}
                <div className="bg-[#141e28] border border-[#202A33] p-6 flex flex-col md:flex-row md:items-center md:justify-between gap-4 accent-edge-lime">
                  <div>
                    <span className="text-[10px] font-mono uppercase tracking-wider text-[#C7F464]">
                      STRESS-TEST VERDICT
                    </span>
                    <h3 className="text-xl font-display font-bold text-[#F4F0E8] mt-1">
                      {robustnessResult.summary_verdict}
                    </h3>
                  </div>

                  <div className="text-right font-mono">
                    <span className="text-xs text-[#64748B]">CONFIDENCE SHIFT</span>
                    <p
                      className={`text-xl font-bold ${
                        robustnessResult.confidence_drop > 0 ? "text-[#FF6B35]" : "text-[#C7F464]"
                      }`}
                    >
                      {robustnessResult.confidence_drop > 0
                        ? `-${(robustnessResult.confidence_drop * 100).toFixed(1)}%`
                        : `+${(-robustnessResult.confidence_drop * 100).toFixed(1)}%`}
                    </p>
                  </div>
                </div>

                {/* Side-by-Side Clean vs Noisy Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Clean Prediction */}
                  <div className="bg-[#141e28] border border-[#202A33] p-6 space-y-4">
                    <div className="flex items-center justify-between border-b border-[#202A33] pb-2 font-mono text-xs">
                      <span className="text-[#C7F464] font-bold">CLEAN AUDIO (BASE)</span>
                      <span className="text-[#64748B]">INF: {robustnessResult.clean_prediction.inference_latency_ms.toFixed(1)} ms</span>
                    </div>

                    <div>
                      <span className="text-[10px] font-mono text-[#64748B]">PREDICTED CLASS</span>
                      <h4 className="text-2xl font-display font-bold text-[#F4F0E8] capitalize">
                        {robustnessResult.clean_prediction.predicted_class_name.replace(/_/g, " ")}
                      </h4>
                    </div>

                    <div className="space-y-1">
                      <div className="flex justify-between text-xs font-mono">
                        <span className="text-[#64748B]">Confidence</span>
                        <span className="text-[#C7F464] font-bold">
                          {(robustnessResult.clean_prediction.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-[#101820] h-2 border border-[#202A33]">
                        <div
                          className="h-full bg-[#C7F464]"
                          style={{ width: `${robustnessResult.clean_prediction.confidence * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Noisy Prediction */}
                  <div className="bg-[#141e28] border border-[#202A33] p-6 space-y-4">
                    <div className="flex items-center justify-between border-b border-[#202A33] pb-2 font-mono text-xs">
                      <span className="text-[#FF6B35] font-bold">
                        CORRUPTED ({robustnessResult.snr_db} dB {robustnessResult.noise_type.toUpperCase()})
                      </span>
                      <span className="text-[#64748B]">INF: {robustnessResult.noisy_prediction.inference_latency_ms.toFixed(1)} ms</span>
                    </div>

                    <div>
                      <span className="text-[10px] font-mono text-[#64748B]">PREDICTED CLASS</span>
                      <h4 className="text-2xl font-display font-bold text-[#F4F0E8] capitalize">
                        {robustnessResult.noisy_prediction.predicted_class_name.replace(/_/g, " ")}
                      </h4>
                    </div>

                    <div className="space-y-1">
                      <div className="flex justify-between text-xs font-mono">
                        <span className="text-[#64748B]">Confidence</span>
                        <span className="text-[#FF6B35] font-bold">
                          {(robustnessResult.noisy_prediction.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-[#101820] h-2 border border-[#202A33]">
                        <div
                          className="h-full bg-[#FF6B35]"
                          style={{ width: `${robustnessResult.noisy_prediction.confidence * 100}%` }}
                        />
                      </div>
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
          <div className="space-y-8">
            <div className="border border-[#202A33] bg-[#141e28] p-6 md:p-8 accent-edge-lime">
              <span className="text-[10px] font-mono px-2 py-0.5 bg-[#202A33] text-[#C7F464]">
                METRICS TELEMETRY
              </span>
              <h1 className="text-3xl font-display font-bold text-[#F4F0E8] mt-1">
                Performance &amp; Evaluation Benchmarks
              </h1>
              <p className="text-xs font-mono text-[#ded7c8] mt-1 max-w-2xl">
                Evaluation results loaded directly from disk artifacts. Compare the clean-trained Baseline against the Noise-Augmented model across controlled SNR degradation levels.
              </p>
            </div>

            {loadingMetrics ? (
              <div className="bg-[#141e28] border border-[#202A33] p-12 text-center flex flex-col items-center justify-center space-y-3">
                <RefreshCw className="w-6 h-6 text-[#C7F464] animate-spin" />
                <p className="text-xs font-mono text-[#ded7c8]">Loading evaluation metrics from disk...</p>
              </div>
            ) : metricsData ? (
              <div className="space-y-8">
                {/* 1. Comparison Table */}
                {metricsData.data?.comparison?.comparison_table && (
                  <div className="bg-[#141e28] border border-[#202A33] p-6 space-y-4">
                    <div className="flex items-center justify-between border-b border-[#202A33] pb-3">
                      <div>
                        <h3 className="text-sm font-display font-bold text-[#F4F0E8]">
                          Model Comparison: Baseline vs. Noise-Augmented
                        </h3>
                        <p className="text-xs font-mono text-[#64748B] mt-0.5">
                          Evaluated on held-out test data (Fold 10) under variable Signal-to-Noise Ratios (Clean &rarr; 0 dB).
                        </p>
                      </div>
                      <span className="text-[10px] font-mono text-[#C7F464] bg-[#202A33] px-2 py-1">
                        10-FOLD SPLIT
                      </span>
                    </div>

                    <div className="overflow-x-auto">
                      <table className="w-full text-xs font-mono text-left">
                        <thead>
                          <tr className="border-b border-[#202A33] text-[#64748B]">
                            <th className="py-2.5 px-3">SNR CONDITION</th>
                            <th className="py-2.5 px-3">BASELINE ACCURACY</th>
                            <th className="py-2.5 px-3">AUGMENTED ACCURACY</th>
                            <th className="py-2.5 px-3">ACCURACY DELTA</th>
                            <th className="py-2.5 px-3">F1 RETENTION</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-[#202A33]">
                          {metricsData.data.comparison.comparison_table.map((row: any) => (
                            <tr key={row.condition} className="hover:bg-[#101820]">
                              <td className="py-2.5 px-3 font-bold text-[#F4F0E8]">{row.condition}</td>
                              <td className="py-2.5 px-3 text-[#ded7c8]">{(row.baseline_accuracy * 100).toFixed(1)}%</td>
                              <td className="py-2.5 px-3 text-[#C7F464] font-bold">{(row.augmented_accuracy * 100).toFixed(1)}%</td>
                              <td className="py-2.5 px-3">
                                <span
                                  className={`px-1.5 py-0.5 ${
                                    row.accuracy_diff >= 0
                                      ? "text-[#C7F464] bg-[#C7F464]/10"
                                      : "text-[#FF6B35] bg-[#FF6B35]/10"
                                  }`}
                                >
                                  {row.accuracy_diff >= 0 ? `+${(row.accuracy_diff * 100).toFixed(1)}%` : `${(row.accuracy_diff * 100).toFixed(1)}%`}
                                </span>
                              </td>
                              <td className="py-2.5 px-3 text-[#64748B]">
                                {(row.augmented_macro_f1 * 100).toFixed(1)}% Macro F1
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* 2. Visual Metric Artifacts Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Robustness Degradation Curve */}
                  <div className="bg-[#141e28] border border-[#202A33] p-6 space-y-3">
                    <div className="flex items-center justify-between border-b border-[#202A33] pb-2 font-mono text-xs">
                      <span className="text-[#C7F464] font-bold">ROBUSTNESS DEGRADATION CURVE</span>
                      <span className="text-[#64748B]">results/comparison_snr_curve.png</span>
                    </div>
                    <img
                      src="/static/results/comparison_snr_curve.png"
                      alt="Robustness SNR Degradation Curve"
                      className="w-full h-auto border border-[#202A33] bg-[#101820]"
                      onError={(e) => {
                        (e.target as HTMLElement).style.display = "none";
                      }}
                    />
                  </div>

                  {/* Calibration Reliability Diagram */}
                  <div className="bg-[#141e28] border border-[#202A33] p-6 space-y-3">
                    <div className="flex items-center justify-between border-b border-[#202A33] pb-2 font-mono text-xs">
                      <span className="text-[#C7F464] font-bold">CALIBRATION DIAGNOSTIC</span>
                      <span className="text-[#64748B]">results/reliability_diagram.png</span>
                    </div>
                    <img
                      src="/static/results/reliability_diagram.png"
                      alt="Reliability Diagram"
                      className="w-full h-auto border border-[#202A33] bg-[#101820]"
                      onError={(e) => {
                        (e.target as HTMLElement).style.display = "none";
                      }}
                    />
                  </div>
                </div>
              </div>
            ) : null}
          </div>
        )}

        {/* ==================================================== */}
        {/* TAB 4: MODEL SPECIFICATIONS & TAXONOMY               */}
        {/* ==================================================== */}
        {activeTab === "model_info" && (
          <div className="space-y-8">
            <div className="border border-[#202A33] bg-[#141e28] p-6 md:p-8 accent-edge-lime">
              <span className="text-[10px] font-mono px-2 py-0.5 bg-[#202A33] text-[#C7F464]">
                SPECIFICATIONS
              </span>
              <h1 className="text-3xl font-display font-bold text-[#F4F0E8] mt-1">
                AcousticCNN Architecture &amp; Taxonomy
              </h1>
              <p className="text-xs font-mono text-[#ded7c8] mt-1 max-w-2xl">
                Technical layer specifications, parameter budget, and the 10 urban sound classes defined in the UrbanSound8K standard taxonomy.
              </p>
            </div>

            {loadingModelInfo ? (
              <div className="bg-[#141e28] border border-[#202A33] p-12 text-center flex flex-col items-center justify-center space-y-3">
                <RefreshCw className="w-6 h-6 text-[#C7F464] animate-spin" />
                <p className="text-xs font-mono text-[#ded7c8]">Loading architecture specs...</p>
              </div>
            ) : modelInfo ? (
              <div className="space-y-6">
                {/* Spec Metric Cards with Distinct Hardware Accent Edges */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-[#141e28] border border-[#202A33] p-4 font-mono accent-edge-violet">
                    <span className="text-[10px] text-[#D946EF] font-bold">TOTAL PARAMETERS</span>
                    <div className="text-xl font-bold text-[#F4F0E8] mt-1">{modelInfo.parameter_count.toLocaleString()}</div>
                    <div className="text-[10px] text-[#D946EF] mt-0.5">Compact CNN</div>
                  </div>
                  <div className="bg-[#141e28] border border-[#202A33] p-4 font-mono accent-edge-cyan">
                    <span className="text-[10px] text-[#00F0FF] font-bold">MODEL FOOTPRINT</span>
                    <div className="text-xl font-bold text-[#F4F0E8] mt-1">{modelInfo.estimated_size_mb} MB</div>
                    <div className="text-[10px] text-[#00F0FF] mt-0.5">Edge deployable</div>
                  </div>
                  <div className="bg-[#141e28] border border-[#202A33] p-4 font-mono accent-edge-amber">
                    <span className="text-[10px] text-[#F59E0B] font-bold">INPUT TENSOR</span>
                    <div className="text-xl font-bold text-[#F4F0E8] mt-1">1 &times; 128 &times; 173</div>
                    <div className="text-[10px] text-[#F59E0B] mt-0.5">Log-Mel STFT</div>
                  </div>
                  <div className="bg-[#141e28] border border-[#202A33] p-4 font-mono accent-edge-lime">
                    <span className="text-[10px] text-[#C7F464] font-bold">CLASSES</span>
                    <div className="text-xl font-bold text-[#C7F464] mt-1">{modelInfo.num_classes} Categories</div>
                    <div className="text-[10px] text-[#C7F464] mt-0.5">UrbanSound8K</div>
                  </div>
                </div>

                {/* 10-Class Taxonomy Matrix */}
                <div className="bg-[#141e28] border border-[#202A33] p-6 space-y-4">
                  <h3 className="text-xs font-mono uppercase tracking-wider text-[#64748B] border-b border-[#202A33] pb-2">
                    Official UrbanSound8K 10-Class Taxonomy
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                    {modelInfo.taxonomy.classes.map((cls: string, idx: number) => (
                      <div
                        key={cls}
                        className="p-3 bg-[#101820] border border-[#202A33] font-mono text-xs flex items-center justify-between hover:border-[#C7F464] transition"
                      >
                        <span className="text-[#64748B]">#{idx}</span>
                        <span className="font-bold text-[#F4F0E8] capitalize truncate ml-2">
                          {cls.replace(/_/g, " ")}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Technical Limitations */}
                <div className="bg-[#141e28] border border-[#202A33] p-6 space-y-3">
                  <h3 className="text-xs font-mono uppercase tracking-wider text-[#FF6B35] flex items-center space-x-2 border-b border-[#202A33] pb-2">
                    <HelpCircle className="w-4 h-4" />
                    <span>Technical &amp; Operational Limitations</span>
                  </h3>
                  <ul className="space-y-2 text-xs font-mono text-[#ded7c8]">
                    {modelInfo.known_limitations.map((item: string, i: number) => (
                      <li key={i} className="flex items-start space-x-2">
                        <span className="text-[#C7F464] font-bold">&bull;</span>
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

      {/* Industrial Hardware Footer */}
      <footer className="border-t border-[#202A33] py-6 px-4 md:px-8 bg-[#101820] text-center text-xs font-mono text-[#64748B]">
        AcoustiGuard AI &bull; Environmental Audio Classification Application &bull; UrbanSound8K &bull; All Systems Operational
      </footer>
    </div>
  );
}
