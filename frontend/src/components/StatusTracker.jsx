const STATUS_STEPS = [
  { key: "queued", label: "Queued", icon: "⏳" },
  { key: "processing", label: "Processing", icon: "⚙️" },
  { key: "complete", label: "Complete", icon: "✅" },
];

const PIPELINE_STEPS = [
  "Extracting audio from video",
  "Transcribing Tamil speech with Whisper",
  "Translating text with LibreTranslate",
  "Cloning voice & synthesizing speech",
  "Remuxing final dubbed video",
];

export default function StatusTracker({ status, jobId, elapsedSeconds, isServerWaking }) {
  const stepIndex = status === "queued" ? 0 : status === "processing" ? 1 : 2;

  // Estimate current pipeline step from elapsed time (rough heuristics for 45s clip on CPU)
  const pipelineStep = Math.min(
    Math.floor(elapsedSeconds / 18),
    PIPELINE_STEPS.length - 1
  );

  return (
    <div className="status-tracker" id="status-tracker">
      {/* Server waking banner */}
      {isServerWaking && (
        <div className="waking-banner" role="alert">
          <span className="waking-icon">🌅</span>
          <span>
            <strong>Waking up the server</strong> — Hugging Face Spaces sleeps after inactivity.
            This first request may take 30–90 seconds. Hang tight!
          </span>
        </div>
      )}

      {/* Status chips */}
      <div className="status-steps">
        {STATUS_STEPS.map((step, i) => (
          <div
            key={step.key}
            className={`status-chip ${
              i < stepIndex ? "done" : i === stepIndex ? "active" : "pending"
            }`}
          >
            <span className="chip-icon">{step.icon}</span>
            <span>{step.label}</span>
          </div>
        ))}
      </div>

      {/* Progress bar */}
      <div className="progress-bar-wrap">
        <div
          className={`progress-bar ${status === "processing" ? "animating" : ""}`}
          style={{
            width:
              status === "queued"
                ? "5%"
                : status === "processing"
                ? `${10 + (pipelineStep / (PIPELINE_STEPS.length - 1)) * 80}%`
                : "100%",
          }}
        />
      </div>

      {/* Current pipeline step label */}
      {status === "processing" && (
        <div className="pipeline-label" id="pipeline-step-label">
          <div className="pulse-dot" />
          <span>{PIPELINE_STEPS[pipelineStep]}</span>
        </div>
      )}

      {/* Elapsed timer */}
      <div className="elapsed-timer">
        <span className="elapsed-icon">⏱</span>
        <span>Elapsed: {elapsedSeconds}s</span>
        {status === "processing" && (
          <span className="eta">  ·  Expected: 60–120s on CPU</span>
        )}
      </div>

      {/* Job ID */}
      <div className="job-id-label">
        Job ID: <code id="job-id-display">{jobId}</code>
      </div>
    </div>
  );
}
