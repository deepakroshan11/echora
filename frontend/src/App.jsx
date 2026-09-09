import { useState, useEffect, useRef, useCallback } from "react";
import { fetchLanguages, uploadVideo, getJobStatus, getDownloadUrl } from "./api";

// Pipeline step labels shown during processing
const PIPELINE_STEPS = [
  { id: "extract",    icon: "🎵", label: "Extracting audio from video" },
  { id: "transcribe", icon: "📝", label: "Transcribing Tamil speech (Whisper)" },
  { id: "translate",  icon: "🌐", label: "Translating to target language" },
  { id: "clone",      icon: "🎙️", label: "Cloning voice & synthesizing speech" },
  { id: "remux",      icon: "🎬", label: "Remixing audio into video" },
];

// Rotate through pipeline steps while processing to give visual feedback
function usePipelineAnimation(active) {
  const [step, setStep] = useState(0);
  useEffect(() => {
    if (!active) { setStep(0); return; }
    const interval = setInterval(() => {
      setStep((s) => (s < PIPELINE_STEPS.length - 1 ? s + 1 : s));
    }, 12000); // advance every ~12 seconds (rough pipeline timing)
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
  const [videoFile, setVideoFile] = useState(null);
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
        // Fallback hardcoded list if backend unreachable
        const fallback = [
          { code: "en", name: "English" },
          { code: "hi", name: "Hindi" },
          { code: "te", name: "Telugu" },
          { code: "kn", name: "Kannada" },
          { code: "ml", name: "Malayalam" },
        ];
        setLanguages(fallback);
        setTargetLang("en");
      });
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      clearInterval(pollIntervalRef.current);
      clearTimeout(slowTimerRef.current);
    };
  }, []);

  // ── File handling ──────────────────────────────────────────────────────────
  const handleFileSelect = useCallback((file) => {
    if (!file) return;
    if (!file.type.startsWith("video/")) {
      alert("Please select a video file.");
      return;
    }
    setVideoFile(file);
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

  const handleDragOver = (e) => { e.preventDefault(); setDragOver(true); };
  const handleDragLeave = () => setDragOver(false);

  // ── Poll job status ────────────────────────────────────────────────────────
  const startPolling = useCallback((id) => {
    // Show a cold-start warning after 15 seconds of waiting
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
          setErrorMsg(data.error_message || "An unknown error occurred.");
          setSlowWarning(false);
        }
        // else: still queued/processing — keep polling
      } catch (err) {
        // Network hiccup — don't crash, just keep polling
        console.warn("Poll error:", err.message);
      }
    }, 2500);
  }, []);

  // ── Submit ─────────────────────────────────────────────────────────────────
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

  // ── Reset ──────────────────────────────────────────────────────────────────
  const handleReset = useCallback(() => {
    clearInterval(pollIntervalRef.current);
    clearTimeout(slowTimerRef.current);
    setVideoFile(null);
    setJobId(null);
    setJobStatus("idle");
    setErrorMsg(null);
    setSlowWarning(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, []);

  // ── Derived state ──────────────────────────────────────────────────────────
  const isIdle = jobStatus === "idle";
  const isUploading = jobStatus === "uploading";
  const isProcessing = jobStatus === "processing";
  const isComplete = jobStatus === "complete";
  const isError = jobStatus === "error";
  const isBusy = isUploading || isProcessing;
  const canSubmit = !!videoFile && !!targetLang && isIdle;

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header__logo">
          <div className="header__icon">🎙️</div>
          <h1 className="header__title">Echora</h1>
        </div>
        <div className="header__badge">AI Voice Cloning &amp; Dubbing</div>
        <p className="header__tagline">
          Upload a short video and instantly dub it into your language
          — using the speaker's own cloned voice.
        </p>
      </header>

      {/* ── Main Card ── */}
      <main className="main-card" role="main">

        {/* ── Upload Section (shown when idle) ── */}
        {isIdle && (
          <>
            <p className="section-label">1 · Upload your video</p>

            {/* Dropzone */}
            <div
              id="dropzone"
              className={`dropzone ${dragOver ? "drag-over" : ""} ${videoFile ? "has-file" : ""}`}
              onClick={() => !videoFile && fileInputRef.current?.click()}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              role="button"
              tabIndex={0}
              aria-label="Upload video"
              onKeyDown={(e) => e.key === "Enter" && fileInputRef.current?.click()}
            >
              {!videoFile ? (
                <>
                  <span className="dropzone__icon">📂</span>
                  <p className="dropzone__title">Drop your Tamil video here</p>
                  <p className="dropzone__subtitle">or click to browse</p>
                  <p className="dropzone__hint">MP4, MOV, WEBM · 30–60 seconds recommended</p>
                </>
              ) : (
                <>
                  <span className="dropzone__icon">✅</span>
                  <p className="dropzone__title">Video ready</p>
                </>
              )}
              <input
                ref={fileInputRef}
                type="file"
                id="video-file-input"
                accept="video/*"
                style={{ display: "none" }}
                onChange={(e) => handleFileSelect(e.target.files?.[0])}
              />
            </div>

            {/* File info pill */}
            {videoFile && (
              <div className="file-info">
                <span className="file-info__icon">🎬</span>
                <span className="file-info__name">{videoFile.name}</span>
                <span className="file-info__size">{formatBytes(videoFile.size)}</span>
                <button
                  className="file-info__remove"
                  onClick={(e) => { e.stopPropagation(); setVideoFile(null); }}
                  aria-label="Remove selected file"
                  title="Remove"
                >✕</button>
              </div>
            )}

            <div className="divider" />

            {/* Language selector */}
            <p className="section-label">2 · Choose target language</p>
            <div className="select-wrapper">
              <select
                id="language-select"
                value={targetLang}
                onChange={(e) => setTargetLang(e.target.value)}
                aria-label="Target language"
              >
                {languages.map((lang) => (
                  <option key={lang.code} value={lang.code}>
                    {lang.name}
                  </option>
                ))}
              </select>
            </div>
            {langsError && (
              <p style={{ fontSize: "0.75rem", color: "var(--clr-warning)", marginTop: "6px" }}>
                ⚠ Could not reach backend — using default languages. ({langsError})
              </p>
            )}

            <div className="divider" />

            {/* Submit */}
            <button
              id="dub-btn"
              className="btn-primary"
              onClick={handleSubmit}
              disabled={!canSubmit}
              aria-label="Start dubbing"
            >
              {canSubmit ? "🚀  Dub Now" : "Select a video to continue"}
            </button>
          </>
        )}

        {/* ── Uploading state ── */}
        {isUploading && (
          <div className="status-processing" aria-live="polite">
            <div className="spinner" role="status" aria-label="Uploading" />
            <p className="status-processing__title">Uploading video…</p>
            <p className="status-processing__subtitle">
              Please wait while your video is being sent to the server.
            </p>
          </div>
        )}

        {/* ── Processing state ── */}
        {isProcessing && (
          <div className="status-processing" aria-live="polite">
            <div className="spinner" role="status" aria-label="Processing" />
            <p className="status-processing__title">Dubbing in progress…</p>
            <p className="status-processing__subtitle">
              AI pipeline is running — this takes 45–90 seconds on CPU.
            </p>

            {/* Animated pipeline steps */}
            <div className="pipeline-steps" aria-label="Pipeline progress">
              {PIPELINE_STEPS.map((s, i) => (
                <div
                  key={s.id}
                  className={`pipeline-step ${
                    i < activeStep ? "done" : i === activeStep ? "active" : ""
                  }`}
                >
                  <span className="pipeline-step__icon">
                    {i < activeStep ? "✅" : i === activeStep ? "⚙️" : s.icon}
                  </span>
                  {s.label}
                </div>
              ))}
            </div>

            {/* Cold-start warning */}
            {slowWarning && (
              <div className="cold-start-notice" role="status">
                <span className="cold-start-notice__icon">☕</span>
                <span>
                  The server may be waking up from sleep — first requests can take
                  an extra 30–90 seconds. Hang tight!
                </span>
              </div>
            )}
          </div>
        )}

        {/* ── Complete state ── */}
        {isComplete && (
          <div className="status-complete" aria-live="polite">
            <span className="status-complete__icon">🎉</span>
            <p className="status-complete__title">Dubbed video ready!</p>
            <p className="status-complete__subtitle">
              Your video has been dubbed in the selected language using the original speaker's voice.
            </p>
            <a
              id="download-link"
              className="btn-download"
              href={getDownloadUrl(jobId)}
              download={`echora_${jobId}.mp4`}
              aria-label="Download dubbed video"
            >
              ⬇️  Download Video
            </a>
            <br />
            <button className="btn-reset" onClick={handleReset} aria-label="Dub another video">
              ↩ Dub another video
            </button>
          </div>
        )}

        {/* ── Error state ── */}
        {isError && (
          <div className="status-error" aria-live="assertive">
            <span className="status-error__icon">❌</span>
            <p className="status-error__title">Something went wrong</p>
            {errorMsg && (
              <pre className="status-error__message">{errorMsg}</pre>
            )}
            <button className="btn-reset" onClick={handleReset} aria-label="Try again">
              ↩ Try again
            </button>
          </div>
        )}
      </main>

      {/* ── Feature Pills ── */}
      <div className="features" aria-label="Features">
        {[
          "Whisper ASR",
          "LibreTranslate",
          "Fish Speech / XTTS v2",
          "Voice Cloning",
          "FFmpeg",
          "$0/month",
        ].map((f) => (
          <span className="feature-pill" key={f}>
            <span className="feature-pill__dot" />
            {f}
          </span>
        ))}
      </div>

      {/* ── Footer ── */}
      <footer className="footer">
        <p>Echora v1 · Voice Cloning &amp; Multilingual Dubbing · Zero paid APIs</p>
        <p>Powered by Whisper · LibreTranslate · Fish Speech · FFmpeg · FastAPI</p>
      </footer>
    </div>
  );
}
