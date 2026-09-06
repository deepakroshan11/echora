import { useState, useEffect, useRef, useCallback } from "react";
import { fetchLanguages, uploadVideo, pollStatus } from "./api";
import UploadSection from "./components/UploadSection";
import StatusTracker from "./components/StatusTracker";
import DownloadSection from "./components/DownloadSection";
import "./index.css";

// App phases
const PHASE = {
  UPLOAD: "upload",
  PROCESSING: "processing",
  DONE: "done",
  ERROR: "error",
};

const POLL_INTERVAL_MS = 2500;
const SLOW_RESPONSE_THRESHOLD_MS = 6000;

export default function App() {
  const [phase, setPhase] = useState(PHASE.UPLOAD);
  const [languages, setLanguages] = useState([]);
  const [langMap, setLangMap] = useState({});
  const [isUploading, setIsUploading] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [targetLang, setTargetLang] = useState("");
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [isServerWaking, setIsServerWaking] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const pollRef = useRef(null);
  const timerRef = useRef(null);
  const startTimeRef = useRef(null);

  // ── Load languages on mount ──────────────────────────────────
  useEffect(() => {
    fetchLanguages()
      .then((langs) => {
        setLanguages(langs);
        const map = {};
        langs.forEach((l) => (map[l.code] = l.name));
        setLangMap(map);
      })
      .catch(() => {
        // Fallback static list if backend isn't up yet
        const fallback = [
          { code: "en", name: "English" },
          { code: "hi", name: "Hindi" },
          { code: "te", name: "Telugu" },
          { code: "kn", name: "Kannada" },
          { code: "ml", name: "Malayalam" },
        ];
        setLanguages(fallback);
        const map = {};
        fallback.forEach((l) => (map[l.code] = l.name));
        setLangMap(map);
      });
  }, []);

  // ── Elapsed timer ────────────────────────────────────────────
  const startTimer = useCallback(() => {
    startTimeRef.current = Date.now();
    timerRef.current = setInterval(() => {
      const elapsed = Math.floor((Date.now() - startTimeRef.current) / 1000);
      setElapsedSeconds(elapsed);
      if (elapsed > SLOW_RESPONSE_THRESHOLD_MS / 1000) {
        setIsServerWaking(true);
      }
    }, 1000);
  }, []);

  const stopTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  // ── Polling ──────────────────────────────────────────────────
  const startPolling = useCallback((id) => {
    pollRef.current = setInterval(async () => {
      try {
        const data = await pollStatus(id);
        setJobStatus(data.status);
        setIsServerWaking(false);

        if (data.status === "complete") {
          clearInterval(pollRef.current);
          stopTimer();
          setPhase(PHASE.DONE);
        } else if (data.status === "error") {
          clearInterval(pollRef.current);
          stopTimer();
          setErrorMessage(data.error_message || "An unknown error occurred.");
          setPhase(PHASE.ERROR);
        }
      } catch {
        // Network hiccup — keep polling
      }
    }, POLL_INTERVAL_MS);
  }, [stopTimer]);

  // ── Upload handler ───────────────────────────────────────────
  const handleUpload = async (file, lang) => {
    setIsUploading(true);
    setTargetLang(lang);
    setElapsedSeconds(0);
    setIsServerWaking(false);

    try {
      const result = await uploadVideo(file, lang);
      setJobId(result.job_id);
      setJobStatus("queued");
      setPhase(PHASE.PROCESSING);
      startTimer();
      startPolling(result.job_id);
    } catch (err) {
      setErrorMessage(err.message);
      setPhase(PHASE.ERROR);
    } finally {
      setIsUploading(false);
    }
  };

  // ── Reset ────────────────────────────────────────────────────
  const handleReset = () => {
    clearInterval(pollRef.current);
    stopTimer();
    setPhase(PHASE.UPLOAD);
    setJobId(null);
    setJobStatus(null);
    setTargetLang("");
    setElapsedSeconds(0);
    setIsServerWaking(false);
    setErrorMessage("");
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      clearInterval(pollRef.current);
      stopTimer();
    };
  }, [stopTimer]);

  return (
    <div className="app">
      {/* Background orbs */}
      <div className="bg-orb orb-1" />
      <div className="bg-orb orb-2" />
      <div className="bg-orb orb-3" />

      {/* Header */}
      <header className="app-header">
        <div className="logo">
          <span className="logo-icon">🎙️</span>
          <span className="logo-text">ShortsDub</span>
        </div>
        <p className="logo-tagline">AI Video Dubbing · Tamil → Any Language · Free & Open Source</p>
      </header>

      {/* Main card */}
      <main className="main-card">
        {(phase === PHASE.UPLOAD) && (
          <section className="card-section fade-in" aria-label="Upload video">
            <div className="card-header">
              <h1 className="card-title">Dub Your Tamil Short</h1>
              <p className="card-subtitle">
                Upload a 30–60 sec Tamil video. We'll transcribe, translate, and clone the speaker's
                voice — all for free, no API keys required.
              </p>
            </div>
            <UploadSection
              languages={languages}
              onSubmit={handleUpload}
              isLoading={isUploading}
            />
          </section>
        )}

        {(phase === PHASE.PROCESSING) && (
          <section className="card-section fade-in" aria-label="Processing status">
            <div className="card-header">
              <h1 className="card-title">Dubbing in Progress</h1>
              <p className="card-subtitle">
                Your video is being transcribed, translated to{" "}
                <strong>{langMap[targetLang] || targetLang}</strong>, and voiced using the
                original speaker's cloned voice.
              </p>
            </div>
            <StatusTracker
              status={jobStatus || "queued"}
              jobId={jobId}
              elapsedSeconds={elapsedSeconds}
              isServerWaking={isServerWaking}
            />
          </section>
        )}

        {(phase === PHASE.DONE || phase === PHASE.ERROR) && (
          <section className="card-section fade-in" aria-label="Result">
            <DownloadSection
              jobId={jobId}
              targetLangName={langMap[targetLang] || targetLang}
              elapsedSeconds={elapsedSeconds}
              onReset={handleReset}
              error={phase === PHASE.ERROR ? errorMessage : null}
            />
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>
          Powered by{" "}
          <a href="https://github.com/openai/whisper" target="_blank" rel="noopener noreferrer">Whisper</a>
          {" · "}
          <a href="https://libretranslate.com" target="_blank" rel="noopener noreferrer">LibreTranslate</a>
          {" · "}
          <a href="https://github.com/coqui-ai/TTS" target="_blank" rel="noopener noreferrer">Coqui XTTS v2</a>
          {" · "}
          <a href="https://ffmpeg.org" target="_blank" rel="noopener noreferrer">FFmpeg</a>
        </p>
        <p className="footer-cost">💸 $0/month · No API keys · Fully open source</p>
      </footer>
    </div>
  );
}
