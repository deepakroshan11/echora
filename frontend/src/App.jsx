import { useState, useEffect, useRef, useCallback } from "react";
import { fetchLanguages, uploadVideo, getJobStatus, getDownloadUrl } from "./api";
import CameraRecorder from "./components/CameraRecorder";
import {
  Video,
  UploadCloud,
  Globe,
  Mic,
  Wand2,
  Film,
  CheckCircle2,
  AlertCircle,
  Download,
  RotateCcw,
  Trash2,
  ShieldCheck,
  Cpu,
  Sparkles,
  FileVideo,
  Layers,
  ArrowRight,
  Camera,
} from "lucide-react";

// Pipeline steps with professional icons and zero emojis
const PIPELINE_STEPS = [
  { id: "extract",    icon: Film,   label: "Audio Track Separation",          detail: "Demuxing high-fidelity vocal track" },
  { id: "transcribe", icon: Mic,    label: "Speech Recognition (Whisper)",    detail: "Generating synchronized transcript" },
  { id: "translate",  icon: Globe,  label: "Neural Translation Engine",       detail: "Adapting idioms & conversational grammar" },
  { id: "clone",      icon: Wand2,  label: "Voice Cloning & Synthesis",       detail: "Synthesizing audio with speaker timbre" },
  { id: "remux",      icon: Video,  label: "Final Audio/Video Remuxing",      detail: "Rebuilding container with synced speech" },
];

function usePipelineAnimation(active) {
  const [step, setStep] = useState(0);
  useEffect(() => {
    if (!active) {
      setStep(0);
      return;
    }
    const interval = setInterval(() => {
      setStep((s) => (s < PIPELINE_STEPS.length - 1 ? s + 1 : s));
    }, 14000);
    return () => clearInterval(interval);
  }, [active]);
  return step;
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function App() {
  // Languages
  const [languages, setLanguages] = useState([]);
  const [langsError, setLangsError] = useState(null);

  // Form state
  const [inputMode, setInputMode] = useState("upload"); // 'upload' | 'camera'
  const [videoFile, setVideoFile] = useState(null);
  const [isRecordedClip, setIsRecordedClip] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [targetLang, setTargetLang] = useState("");
  const [dragOver, setDragOver] = useState(false);

  // Job state
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState("idle"); // idle | uploading | processing | complete | error
  const [errorMsg, setErrorMsg] = useState(null);
  const [slowWarning, setSlowWarning] = useState(false);

  const fileInputRef = useRef(null);
  const pollIntervalRef = useRef(null);
  const slowTimerRef = useRef(null);
  const activeStep = usePipelineAnimation(jobStatus === "processing");

  // Fetch languages on mount
  useEffect(() => {
    fetchLanguages()
      .then((langs) => {
        setLanguages(langs);
        if (langs.length > 0) setTargetLang(langs[0].code);
      })
      .catch((err) => {
        setLangsError(err.message);
        const fallback = [
          { code: "en", name: "English" },
          { code: "ta", name: "Tamil" },
          { code: "hi", name: "Hindi" },
          { code: "te", name: "Telugu" },
          { code: "kn", name: "Kannada" },
          { code: "ml", name: "Malayalam" },
        ];
        setLanguages(fallback);
        setTargetLang("en");
      });
  }, []);

  // Manage preview URL cleanup
  useEffect(() => {
    if (!videoFile) {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        setPreviewUrl(null);
      }
      return;
    }
    const url = URL.createObjectURL(videoFile);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [videoFile]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      clearInterval(pollIntervalRef.current);
      clearTimeout(slowTimerRef.current);
    };
  }, []);

  // File handling
  const handleFileSelect = useCallback((file) => {
    if (!file) return;
    if (!file.type.startsWith("video/")) {
      alert("Please upload a valid video file (MP4, MOV, WebM).");
      return;
    }
    setIsRecordedClip(false);
    setVideoFile(file);
  }, []);

  const handleRecordingComplete = useCallback((file) => {
    setIsRecordedClip(true);
    setVideoFile(file);
    setInputMode("upload");
  }, []);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files?.[0];
      handleFileSelect(file);
    },
    [handleFileSelect]
  );

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };
  const handleDragLeave = () => setDragOver(false);

  // Poll job status
  const startPolling = useCallback((id) => {
    slowTimerRef.current = setTimeout(() => setSlowWarning(true), 15000);

    pollIntervalRef.current = setInterval(async () => {
      try {
        const data = await getJobStatus(id);
        if (data.status === "complete") {
          clearInterval(pollIntervalRef.current);
          clearTimeout(slowTimerRef.current);
          setJobStatus("complete");
          setSlowWarning(false);
        } else if (data.status === "error") {
          clearInterval(pollIntervalRef.current);
          clearTimeout(slowTimerRef.current);
          setJobStatus("error");
          setErrorMsg(data.error_message || "An unexpected error occurred during synthesis.");
          setSlowWarning(false);
        }
      } catch (err) {
        console.warn("Polling warning:", err.message);
      }
    }, 2500);
  }, []);

  // Direct submit with full granted execution
  const handleSubmit = useCallback(async () => {
    if (!videoFile || !targetLang) return;

    setJobStatus("uploading");
    setErrorMsg(null);
    setSlowWarning(false);

    try {
      const id = await uploadVideo(videoFile, targetLang);
      setJobId(id);
      setJobStatus("processing");
      startPolling(id);
    } catch (err) {
      setJobStatus("error");
      setErrorMsg(err.message);
    }
  }, [videoFile, targetLang, startPolling]);

  // Reset form
  const handleReset = useCallback(() => {
    clearInterval(pollIntervalRef.current);
    clearTimeout(slowTimerRef.current);
    setVideoFile(null);
    setIsRecordedClip(false);
    setJobId(null);
    setJobStatus("idle");
    setErrorMsg(null);
    setSlowWarning(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, []);

  const isIdle = jobStatus === "idle";
  const isUploading = jobStatus === "uploading";
  const isProcessing = jobStatus === "processing";
  const isComplete = jobStatus === "complete";
  const isError = jobStatus === "error";
  const isBusy = isUploading || isProcessing;
  const canSubmit = !!videoFile && !!targetLang && isIdle;

  const currentLangObj = languages.find((l) => l.code === targetLang);

  return (
    <div className="app-container">
      {/* ── Navigation / App Bar ── */}
      <nav className="navbar" aria-label="Top navigation">
        <div className="brand">
          <div className="brand-icon-wrapper">
            <Mic size={20} strokeWidth={2.2} />
          </div>
          <div className="brand-text">
            <span className="brand-title">Echora</span>
            <span className="brand-tagline">Autonomous Voice Dubbing</span>
          </div>
        </div>

        <div className="nav-badges">
          <div className="badge badge-navy">
            <span className="badge-pulse-dot" />
            <span>Pipeline Online</span>
          </div>
          <div className="badge badge-permission" title="Full execution granted without prompts">
            <ShieldCheck size={14} strokeWidth={2.2} />
            <span>Full Grant Active</span>
          </div>
        </div>
      </nav>

      {/* ── Hero Section ── */}
      <section className="hero-section">
        <div className="hero-pill">
          <Sparkles size={14} color="#1d4ed8" strokeWidth={2.2} />
          <span>Speaker-Preserved Multilingual Synthesis</span>
        </div>
        <h1 className="hero-title">High-Fidelity AI Video Dubbing</h1>
        <p className="hero-subtitle">
          Translate short video content seamlessly across languages while retaining
          the speaker's original vocal tone, cadence, and prosody.
        </p>
      </section>

      {/* ── Main Workspace Card ── */}
      <main className="workspace-card" role="main">
        {/* Step Indicator Header */}
        <div className="stepper-header" aria-label="Pipeline sequence">
          <div className={`step-indicator ${isIdle ? "active" : "completed"}`}>
            <div className="step-number">1</div>
            <div className="step-content">
              <span className="step-title">Source Media</span>
              <span className="step-desc">Upload video file</span>
            </div>
          </div>
          <div className={`step-indicator ${isIdle && videoFile ? "active" : isBusy || isComplete ? "completed" : ""}`}>
            <div className="step-number">2</div>
            <div className="step-content">
              <span className="step-title">Target Language</span>
              <span className="step-desc">Select translation</span>
            </div>
          </div>
          <div className={`step-indicator ${isBusy ? "active" : isComplete ? "completed" : ""}`}>
            <div className="step-number">3</div>
            <div className="step-content">
              <span className="step-title">Synthesis</span>
              <span className="step-desc">Voice clone &amp; remux</span>
            </div>
          </div>
        </div>

        {/* ── Idle State: Upload & Configure ── */}
        {isIdle && (
          <div>
            {/* Step 1: Video File Selection */}
            <div className="form-section-title">
              <FileVideo size={18} className="form-section-icon" strokeWidth={2.2} />
              <span>Step 1: Provide Video Source</span>
            </div>

            {!videoFile ? (
              <div>
                {/* Switcher: Upload File vs Live Camera */}
                <div className="source-mode-switcher">
                  <button
                    type="button"
                    className={`source-mode-btn ${inputMode === "upload" ? "active" : ""}`}
                    onClick={() => setInputMode("upload")}
                    aria-label="Upload Video File"
                  >
                    <UploadCloud size={16} />
                    <span>Upload Video File</span>
                  </button>
                  <button
                    type="button"
                    className={`source-mode-btn ${inputMode === "camera" ? "active" : ""}`}
                    onClick={() => setInputMode("camera")}
                    aria-label="Open Live Camera"
                  >
                    <Camera size={16} />
                    <span>Open Live Camera</span>
                    <span className="source-mode-badge">Live</span>
                  </button>
                </div>

                {inputMode === "camera" ? (
                  <CameraRecorder
                    onRecordingComplete={handleRecordingComplete}
                    onCancel={() => setInputMode("upload")}
                  />
                ) : (
                  <div
                    id="dropzone"
                    className={`dropzone ${dragOver ? "drag-over" : ""}`}
                    onClick={() => fileInputRef.current?.click()}
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    role="button"
                    tabIndex={0}
                    aria-label="Upload video file"
                    onKeyDown={(e) => e.key === "Enter" && fileInputRef.current?.click()}
                  >
                    <div className="dropzone-icon-circle">
                      <UploadCloud size={28} strokeWidth={2} />
                    </div>
                    <div>
                      <p className="dropzone-heading">Drop your video here, or browse device</p>
                      <p className="dropzone-subtext">Click anywhere in this area to select your clip</p>
                    </div>
                    <span className="dropzone-format-badge">MP4, MOV, WebM · Up to 60s Recommended</span>
                    <input
                      ref={fileInputRef}
                      type="file"
                      id="video-file-input"
                      accept="video/*"
                      style={{ display: "none" }}
                      onChange={(e) => handleFileSelect(e.target.files?.[0])}
                    />
                  </div>
                )}
              </div>
            ) : (
              <div>
                <div className="file-preview-card">
                  <div className="file-preview-left">
                    <div className="file-preview-thumbnail">
                      {isRecordedClip ? (
                        <Camera size={22} strokeWidth={2.2} />
                      ) : (
                        <Film size={22} strokeWidth={2.2} />
                      )}
                    </div>
                    <div className="file-meta-content">
                      <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
                        <span className="file-meta-name" title={videoFile.name}>{videoFile.name}</span>
                        {isRecordedClip && (
                          <span className="live-badge-card">
                            <span className="badge-pulse-dot" style={{ width: 6, height: 6 }} />
                            Live Camera
                          </span>
                        )}
                      </div>
                      <div className="file-meta-sub">
                        <span>{formatBytes(videoFile.size)}</span>
                        <span>•</span>
                        <span className="file-meta-pill">{videoFile.type || "video/mp4"}</span>
                      </div>
                    </div>
                  </div>
                  <div className="file-preview-actions">
                    <button
                      type="button"
                      className="btn-remove-file"
                      onClick={() => {
                        setVideoFile(null);
                        setIsRecordedClip(false);
                      }}
                      aria-label="Remove video file"
                    >
                      <Trash2 size={14} strokeWidth={2} />
                      <span>Remove</span>
                    </button>
                  </div>
                </div>

                {previewUrl && (
                  <div className="video-preview-wrapper">
                    <video
                      src={previewUrl}
                      controls
                      playsInline
                      className="video-preview-element"
                    />
                  </div>
                )}
              </div>
            )}

            <div className="section-divider" />

            {/* Step 2: Language Selection */}
            <div className="form-section-title">
              <Globe size={18} className="form-section-icon" strokeWidth={2.2} />
              <span>Step 2: Target Language</span>
            </div>

            <div className="language-grid" role="radiogroup" aria-label="Select target language">
              {languages.map((lang) => {
                const isSelected = targetLang === lang.code;
                return (
                  <div
                    key={lang.code}
                    className={`language-card ${isSelected ? "selected" : ""}`}
                    onClick={() => setTargetLang(lang.code)}
                    role="radio"
                    aria-checked={isSelected}
                    tabIndex={0}
                    onKeyDown={(e) => e.key === "Enter" && setTargetLang(lang.code)}
                  >
                    <span className="language-card-name">{lang.name}</span>
                    <span className="language-card-code">{lang.code.toUpperCase()}</span>
                  </div>
                );
              })}
            </div>

            {langsError && (
              <p style={{ fontSize: "0.78rem", color: "var(--warning)", marginTop: "8px", fontWeight: 500 }}>
                Backend connection notice: defaulted to local language table.
              </p>
            )}

            <div className="section-divider" />

            {/* Direct Permission & Submit Section */}
            <div className="permission-grant-bar">
              <div className="permission-grant-left">
                <ShieldCheck size={16} color="#1d4ed8" strokeWidth={2.4} />
                <span>Execution Grant: Full Direct Access Enabled</span>
              </div>
              <div className="permission-grant-right">
                Zero Confirmation Friction
              </div>
            </div>

            <button
              id="dub-btn"
              type="button"
              className="btn-submit-action"
              onClick={handleSubmit}
              disabled={!canSubmit}
              aria-label="Execute video dubbing pipeline"
            >
              {canSubmit ? (
                <>
                  <Sparkles size={18} strokeWidth={2.2} />
                  <span>Execute Dubbing Pipeline ({currentLangObj?.name || "Selected"})</span>
                  <ArrowRight size={18} strokeWidth={2.2} />
                </>
              ) : (
                <span>Upload a video to enable dubbing</span>
              )}
            </button>
          </div>
        )}

        {/* ── Uploading State ── */}
        {isUploading && (
          <div className="processing-container" aria-live="polite">
            <div className="spinner-navy" role="status" aria-label="Uploading media" />
            <h2 className="processing-title">Transferring Source Media</h2>
            <p className="processing-subtitle">
              Securely uploading your video to the local inference engine...
            </p>
          </div>
        )}

        {/* ── Processing State ── */}
        {isProcessing && (
          <div className="processing-container" aria-live="polite">
            <div className="spinner-navy" role="status" aria-label="Synthesizing media" />
            <h2 className="processing-title">Synthesis in Progress</h2>
            <p className="processing-subtitle">
              Running deep neural pipeline: transcription, translation, and speaker voice cloning.
            </p>

            <div className="pipeline-stepper" aria-label="Pipeline progression">
              {PIPELINE_STEPS.map((s, i) => {
                const IconComponent = s.icon;
                const isStepCompleted = i < activeStep;
                const isStepActive = i === activeStep;
                const stepClass = isStepCompleted ? "completed" : isStepActive ? "active" : "pending";

                return (
                  <div key={s.id} className={`pipeline-stage-item ${stepClass}`}>
                    <div className="pipeline-stage-left">
                      <div className="pipeline-stage-icon-circle">
                        {isStepCompleted ? (
                          <CheckCircle2 size={16} strokeWidth={2.4} />
                        ) : (
                          <IconComponent size={16} strokeWidth={2.2} />
                        )}
                      </div>
                      <div style={{ textAlign: "left" }}>
                        <div className="pipeline-stage-label">{s.label}</div>
                        <div style={{ fontSize: "0.74rem", color: "var(--navy-500)" }}>{s.detail}</div>
                      </div>
                    </div>
                    <span className="pipeline-stage-status">
                      {isStepCompleted ? "Completed" : isStepActive ? "Active" : "Queued"}
                    </span>
                  </div>
                );
              })}
            </div>

            {slowWarning && (
              <div className="notice-box" role="status">
                <Cpu size={18} color="#0f172a" strokeWidth={2} style={{ flexShrink: 0 }} />
                <span>
                  First-run initialisation: Neural models are loading into system memory. Subsequent executions will run significantly faster.
                </span>
              </div>
            )}
          </div>
        )}

        {/* ── Completed State ── */}
        {isComplete && (
          <div className="completed-container" aria-live="polite">
            <div className="completed-badge-icon">
              <CheckCircle2 size={28} strokeWidth={2.2} />
            </div>
            <h2 className="completed-title">Dubbed Video Synthesized</h2>
            <p className="completed-subtitle">
              Your video has been rendered with the cloned vocal signature and synchronized audio.
            </p>

            <div className="output-player-wrapper">
              <video
                src={getDownloadUrl(jobId)}
                controls
                autoPlay
                playsInline
                className="output-player-element"
              />
            </div>

            <div className="completed-actions">
              <a
                id="download-link"
                className="btn-download-primary"
                href={getDownloadUrl(jobId)}
                download={`echora_${jobId}.mp4`}
                aria-label="Download synthesized video file"
              >
                <Download size={18} strokeWidth={2.2} />
                <span>Download Dubbed Video</span>
              </a>

              <button
                type="button"
                className="btn-reset-secondary"
                onClick={handleReset}
                aria-label="Process another video"
              >
                <RotateCcw size={16} strokeWidth={2} />
                <span>Process Another</span>
              </button>
            </div>
          </div>
        )}

        {/* ── Error State ── */}
        {isError && (
          <div className="error-container" aria-live="assertive">
            <div className="error-badge-icon">
              <AlertCircle size={28} strokeWidth={2.2} />
            </div>
            <h2 className="error-title">Synthesis Interrupted</h2>
            <p className="processing-subtitle">
              The neural pipeline encountered an unexpected issue while processing the media stream.
            </p>

            {errorMsg && (
              <div className="error-message-box">
                {errorMsg}
              </div>
            )}

            <button
              type="button"
              className="btn-submit-action"
              style={{ maxWidth: "280px" }}
              onClick={handleReset}
              aria-label="Reset and try again"
            >
              <RotateCcw size={16} strokeWidth={2} />
              <span>Retry Pipeline</span>
            </button>
          </div>
        )}
      </main>

      {/* ── Architectural Features Strip ── */}
      <div className="features-strip" aria-label="System architecture">
        {[
          { label: "Whisper Neural ASR", icon: Mic },
          { label: "Automated Translation", icon: Globe },
          { label: "XTTS v2 / Fish Speech", icon: Wand2 },
          { label: "Zero API Fees", icon: ShieldCheck },
          { label: "Hardware Accelerated FFmpeg", icon: Film },
        ].map(({ label, icon: Icon }) => (
          <div className="feature-pill-badge" key={label}>
            <Icon size={14} className="feature-pill-icon" strokeWidth={2} />
            <span>{label}</span>
          </div>
        ))}
      </div>

      {/* ── Global Footer ── */}
      <footer className="global-footer">
        <p className="footer-copy">Echora Neural Dubbing Engine · Full Local Sovereignty</p>
        <p className="footer-tech">Engineered with FastAPI · PyTorch · Vite &amp; React · Zero External Telemetry</p>
      </footer>
    </div>
  );
}
