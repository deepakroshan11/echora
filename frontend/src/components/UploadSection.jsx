import { useState, useRef } from "react";

const LANGUAGES = []; // Populated by parent from API

export default function UploadSection({ languages, onSubmit, isLoading }) {
  const [file, setFile] = useState(null);
  const [targetLang, setTargetLang] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selected = e.target.files?.[0];
    if (selected) setFile(selected);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    const dropped = e.dataTransfer.files?.[0];
    if (dropped && dropped.type.startsWith("video/")) setFile(dropped);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragActive(true);
  };

  const handleDragLeave = () => setDragActive(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!file || !targetLang) return;
    onSubmit(file, targetLang);
  };

  const formatSize = (bytes) => {
    if (bytes < 1e6) return `${(bytes / 1e3).toFixed(1)} KB`;
    return `${(bytes / 1e6).toFixed(1)} MB`;
  };

  const canSubmit = file && targetLang && !isLoading;

  return (
    <form className="upload-form" onSubmit={handleSubmit}>
      {/* Drop zone */}
      <div
        className={`drop-zone ${dragActive ? "drag-active" : ""} ${file ? "has-file" : ""}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => fileInputRef.current?.click()}
        role="button"
        tabIndex={0}
        id="video-drop-zone"
        aria-label="Drop video file here or click to browse"
        onKeyDown={(e) => e.key === "Enter" && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          onChange={handleFileChange}
          style={{ display: "none" }}
          id="video-file-input"
        />
        {file ? (
          <div className="file-preview">
            <div className="file-icon">🎬</div>
            <div className="file-info">
              <span className="file-name">{file.name}</span>
              <span className="file-size">{formatSize(file.size)}</span>
            </div>
            <button
              type="button"
              className="remove-file"
              onClick={(e) => {
                e.stopPropagation();
                setFile(null);
                if (fileInputRef.current) fileInputRef.current.value = "";
              }}
              aria-label="Remove file"
            >
              ✕
            </button>
          </div>
        ) : (
          <div className="drop-prompt">
            <div className="drop-icon">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>
            <p className="drop-main">Drop your Tamil video here</p>
            <p className="drop-sub">or click to browse · MP4, MOV, MKV, WebM · up to 500 MB</p>
          </div>
        )}
      </div>

      {/* Language selector */}
      <div className="lang-select-wrap">
        <label htmlFor="target-language-select" className="lang-label">
          Target Language
        </label>
        <div className="select-wrapper">
          <select
            id="target-language-select"
            value={targetLang}
            onChange={(e) => setTargetLang(e.target.value)}
            className="lang-select"
            required
          >
            <option value="">— Choose a language —</option>
            {languages.map((lang) => (
              <option key={lang.code} value={lang.code}>
                {lang.name}
              </option>
            ))}
          </select>
          <div className="select-arrow">▾</div>
        </div>
      </div>

      {/* Submit */}
      <button
        type="submit"
        id="dub-now-button"
        className={`dub-btn ${canSubmit ? "active" : "disabled"}`}
        disabled={!canSubmit}
      >
        {isLoading ? (
          <span className="btn-loading">
            <span className="spinner" />
            Uploading…
          </span>
        ) : (
          <span className="btn-text">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polygon points="5 3 19 12 5 21 5 3" />
            </svg>
            Dub Now
          </span>
        )}
      </button>
    </form>
  );
}
