import { getDownloadUrl } from "../api";

export default function DownloadSection({ jobId, targetLangName, elapsedSeconds, onReset, error }) {
  const downloadUrl = getDownloadUrl(jobId);

  if (error) {
    return (
      <div className="result-section error-state" id="error-section">
        <div className="result-icon error-icon">💔</div>
        <h2 className="result-title">Processing Failed</h2>
        <p className="error-message">{error}</p>
        <button className="retry-btn" id="retry-button" onClick={onReset}>
          ↩ Try Again
        </button>
      </div>
    );
  }

  return (
    <div className="result-section success-state" id="download-section">
      {/* Celebration header */}
      <div className="result-icon success-icon">🎉</div>
      <h2 className="result-title">Your Dubbed Video is Ready!</h2>
      <p className="result-sub">
        Dubbed to <strong>{targetLangName}</strong> · Processed in <strong>{elapsedSeconds}s</strong>
      </p>

      {/* Download button */}
      <a
        href={downloadUrl}
        download={`echora_${jobId.slice(0, 8)}.mp4`}
        className="download-btn"
        id="download-video-button"
        aria-label={`Download dubbed video in ${targetLangName}`}
      >
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="7 10 12 15 17 10" />
          <line x1="12" y1="15" x2="12" y2="3" />
        </svg>
        Download MP4
      </a>

      {/* Known limitations note */}
      <div className="limitations-note">
        <strong>ℹ️ About the dubbed audio:</strong> Translation quality depends on LibreTranslate
        (open-source, self-hosted). Voice clone quality improves with a clean, noise-free reference clip.
        Lip-sync is not applied in v1.
      </div>

      {/* Dub another */}
      <button className="dub-another-btn" id="dub-another-button" onClick={onReset}>
        + Dub Another Video
      </button>
    </div>
  );
}
