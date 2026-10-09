import { useState, useEffect, useRef, useCallback } from "react";
import {
  Camera,
  Video,
  Square,
  RefreshCw,
  Check,
  X,
  FlipHorizontal,
  RotateCcw,
  AlertCircle,
  Volume2,
  Mic,
} from "lucide-react";

/**
 * Helper to pick the best supported MediaRecorder MIME type on this device/browser.
 */
function getBestSupportedMimeType() {
  if (typeof MediaRecorder === "undefined") return "";
  const candidates = [
    "video/webm;codecs=vp9,opus",
    "video/webm;codecs=vp8,opus",
    "video/webm;codecs=h264,opus",
    "video/webm",
    "video/mp4;codecs=avc1,mp4a.40.2",
    "video/mp4",
  ];
  for (const t of candidates) {
    if (MediaRecorder.isTypeSupported(t)) return t;
  }
  return "";
}

/**
 * Format seconds into mm:ss
 */
function formatTime(totalSeconds) {
  const m = Math.floor(totalSeconds / 60);
  const s = totalSeconds % 60;
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

export default function CameraRecorder({ onRecordingComplete, onCancel }) {
  // Device & Stream state
  const [stream, setStream] = useState(null);
  const [permissionError, setPermissionError] = useState(null);
  const [isInitializing, setIsInitializing] = useState(true);
  const [facingMode, setFacingMode] = useState("user"); // 'user' (front) or 'environment' (back)
  const [hasMultipleCameras, setHasMultipleCameras] = useState(false);
  const [isMirrored, setIsMirrored] = useState(true);

  // Recording state
  const [recordingState, setRecordingState] = useState("preview"); // 'preview' | 'recording' | 'review'
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [recordedPreviewUrl, setRecordedPreviewUrl] = useState(null);
  const [audioLevel, setAudioLevel] = useState(0);

  // Refs
  const liveVideoRef = useRef(null);
  const reviewVideoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerIntervalRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const animFrameRef = useRef(null);

  // Check if device has multiple cameras (front/rear)
  useEffect(() => {
    if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
      navigator.mediaDevices.enumerateDevices().then((devices) => {
        const videoInputs = devices.filter((d) => d.kind === "videoinput");
        setHasMultipleCameras(videoInputs.length > 1);
      }).catch(() => {});
    }
  }, []);

  // Stop current active stream tracks
  const stopStreamTracks = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      audioContextRef.current.close().catch(() => {});
    }
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
    }
  }, [stream]);

  // Audio level meter monitor
  const startAudioMeter = useCallback((mediaStream) => {
    try {
      const audioTrack = mediaStream.getAudioTracks()[0];
      if (!audioTrack) return;

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;

      const audioCtx = new AudioCtx();
      audioContextRef.current = audioCtx;
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      analyserRef.current = analyser;

      const source = audioCtx.createMediaStreamSource(mediaStream);
      source.connect(analyser);

      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      const updateMeter = () => {
        if (!analyserRef.current) return;
        analyserRef.current.getByteFrequencyData(dataArray);
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
          sum += dataArray[i];
        }
        const avg = sum / dataArray.length;
        setAudioLevel(Math.min(100, Math.round((avg / 128) * 100)));
        animFrameRef.current = requestAnimationFrame(updateMeter);
      };
      updateMeter();
    } catch (err) {
      console.warn("Audio meter setup skipped:", err);
    }
  }, []);

  // Initialize or re-initialize camera stream
  const startCamera = useCallback(async (facing) => {
    setIsInitializing(true);
    setPermissionError(null);

    // Stop previous tracks if any
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setPermissionError("Camera access is not supported by your browser or environment. Try using HTTPS or Chrome.");
      setIsInitializing(false);
      return;
    }

    try {
      const constraints = {
        video: {
          facingMode: facing,
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      };

      const newStream = await navigator.mediaDevices.getUserMedia(constraints);
      setStream(newStream);

      if (liveVideoRef.current) {
        liveVideoRef.current.srcObject = newStream;
      }

      startAudioMeter(newStream);
      setIsInitializing(false);
    } catch (err) {
      console.error("Camera access failed:", err);
      let msg = "Could not access camera or microphone. Please check your browser permissions.";
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        msg = "Camera permission was denied. Please allow camera and microphone access in your browser settings.";
      } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
        msg = "No camera or microphone found on this device.";
      } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
        msg = "Camera is already in use by another application.";
      }
      setPermissionError(msg);
      setIsInitializing(false);
    }
  }, [stream, startAudioMeter]);

  // Initial camera startup
  useEffect(() => {
    startCamera(facingMode);
    return () => {
      stopStreamTracks();
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Cleanup blob URL on review unmount or discard
  useEffect(() => {
    return () => {
      if (recordedPreviewUrl) {
        URL.revokeObjectURL(recordedPreviewUrl);
      }
    };
  }, [recordedPreviewUrl]);

  // Flip camera (front vs back)
  const handleSwitchCamera = () => {
    const nextFacing = facingMode === "user" ? "environment" : "user";
    setFacingMode(nextFacing);
    setIsMirrored(nextFacing === "user");
    startCamera(nextFacing);
  };

  // Start recording
  const handleStartRecording = () => {
    if (!stream) return;

    chunksRef.current = [];
    const mimeType = getBestSupportedMimeType();
    const options = mimeType ? { mimeType } : undefined;

    try {
      const recorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      recorder.onstop = () => {
        const recordedMime = recorder.mimeType || mimeType || "video/webm";
        const blob = new Blob(chunksRef.current, { type: recordedMime });
        setRecordedBlob(blob);
        const url = URL.createObjectURL(blob);
        setRecordedPreviewUrl(url);
        setRecordingState("review");
      };

      recorder.start(500); // 500ms timeslices for smooth buffering
      setRecordingState("recording");
      setRecordingSeconds(0);

      timerIntervalRef.current = setInterval(() => {
        setRecordingSeconds((prev) => {
          // Auto-stop at 120 seconds (safety ceiling for short video dubbing)
          if (prev >= 120) {
            handleStopRecording();
            return prev;
          }
          return prev + 1;
        });
      }, 1000);
    } catch (err) {
      console.error("Failed to start MediaRecorder:", err);
      alert("Recording could not be started: " + err.message);
    }
  };

  // Stop recording
  const handleStopRecording = () => {
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }

    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state !== "inactive"
    ) {
      mediaRecorderRef.current.stop();
    }
  };

  // Retake / discard recording
  const handleRetake = () => {
    if (recordedPreviewUrl) {
      URL.revokeObjectURL(recordedPreviewUrl);
      setRecordedPreviewUrl(null);
    }
    setRecordedBlob(null);
    setRecordingSeconds(0);
    setRecordingState("preview");

    // Rebind stream to preview video
    if (liveVideoRef.current && stream) {
      liveVideoRef.current.srcObject = stream;
    }
  };

  // Confirm and use this recorded video
  const handleConfirmRecording = () => {
    if (!recordedBlob) return;

    const mime = recordedBlob.type || "video/webm";
    const ext = mime.includes("mp4") ? "mp4" : "webm";
    const filename = `echora_live_recording_${Date.now()}.${ext}`;

    const file = new File([recordedBlob], filename, {
      type: mime,
      lastModified: Date.now(),
    });

    // Stop camera before leaving
    stopStreamTracks();
    onRecordingComplete(file);
  };

  // Cancel and exit camera
  const handleCancel = () => {
    stopStreamTracks();
    if (onCancel) onCancel();
  };

  return (
    <div className="camera-recorder-wrapper" aria-label="Live Camera Video Recorder">
      {/* ── Top Bar: Status, Camera Switcher, Close ── */}
      <div className="camera-recorder-header">
        <div className="camera-recorder-status">
          <div className={`camera-status-dot ${recordingState === "recording" ? "pulsing-red" : "online-green"}`} />
          <span className="camera-status-text">
            {recordingState === "recording"
              ? `Recording live (${formatTime(recordingSeconds)})`
              : recordingState === "review"
              ? "Preview Recorded Video"
              : "Live Camera Active"}
          </span>
        </div>

        <div className="camera-header-actions">
          {recordingState === "preview" && hasMultipleCameras && (
            <button
              type="button"
              className="camera-ctrl-btn"
              onClick={handleSwitchCamera}
              title="Flip camera (Front / Rear)"
              aria-label="Flip camera"
            >
              <RotateCcw size={15} />
              <span>Flip Camera</span>
            </button>
          )}

          {recordingState === "preview" && (
            <button
              type="button"
              className="camera-ctrl-btn"
              onClick={() => setIsMirrored((m) => !m)}
              title="Toggle mirror mode"
              aria-label="Toggle mirror preview"
            >
              <FlipHorizontal size={15} />
              <span>{isMirrored ? "Mirrored" : "Normal"}</span>
            </button>
          )}

          <button
            type="button"
            className="camera-close-btn"
            onClick={handleCancel}
            title="Cancel and return to upload"
            aria-label="Close camera"
          >
            <X size={18} />
          </button>
        </div>
      </div>

      {/* ── Error state ── */}
      {permissionError && (
        <div className="camera-error-banner">
          <AlertCircle size={20} className="camera-error-icon" />
          <div className="camera-error-content">
            <strong>Camera Access Unavailable</strong>
            <p>{permissionError}</p>
          </div>
          <button
            type="button"
            className="camera-btn-retry"
            onClick={() => startCamera(facingMode)}
          >
            <RefreshCw size={14} />
            <span>Retry</span>
          </button>
        </div>
      )}

      {/* ── Viewfinder View (Live Preview or Recorded Review) ── */}
      <div className="camera-viewfinder-container">
        {/* Live video preview */}
        <video
          ref={liveVideoRef}
          autoPlay
          playsInline
          muted
          style={{
            display: recordingState === "review" ? "none" : "block",
            transform: isMirrored ? "scaleX(-1)" : "none",
          }}
          className="camera-video-feed"
        />

        {/* Recorded clip preview */}
        {recordingState === "review" && recordedPreviewUrl && (
          <video
            ref={reviewVideoRef}
            src={recordedPreviewUrl}
            controls
            playsInline
            autoPlay
            className="camera-video-feed camera-video-review"
          />
        )}

        {/* Viewfinder Overlays when live */}
        {recordingState !== "review" && (
          <>
            {/* Viewfinder Corner Crosshairs */}
            <div className="viewfinder-corner top-left" />
            <div className="viewfinder-corner top-right" />
            <div className="viewfinder-corner bottom-left" />
            <div className="viewfinder-corner bottom-right" />

            {/* Recording Timer Badge */}
            {recordingState === "recording" && (
              <div className="recording-timer-overlay">
                <span className="rec-dot" />
                <span className="rec-digits">{formatTime(recordingSeconds)}</span>
                <span className="rec-max-notice">/ 02:00 max</span>
              </div>
            )}

            {/* Live Audio Level Meter */}
            <div className="audio-level-indicator" title="Live Microphone Level">
              <Mic size={14} className={audioLevel > 5 ? "mic-active" : "mic-idle"} />
              <div className="audio-meter-bar-track">
                <div
                  className="audio-meter-bar-fill"
                  style={{ width: `${Math.min(100, audioLevel * 1.5)}%` }}
                />
              </div>
            </div>
          </>
        )}
      </div>

      {/* ── Bottom Controls ── */}
      <div className="camera-recorder-controls">
        {recordingState === "preview" && (
          <div className="controls-preview-mode">
            <button
              type="button"
              className="btn-shutter-record"
              onClick={handleStartRecording}
              disabled={isInitializing || !!permissionError}
              aria-label="Start recording video"
            >
              <div className="shutter-inner-circle" />
            </button>
            <span className="shutter-hint">Tap to start live recording</span>
          </div>
        )}

        {recordingState === "recording" && (
          <div className="controls-recording-mode">
            <button
              type="button"
              className="btn-shutter-stop"
              onClick={handleStopRecording}
              aria-label="Stop recording"
            >
              <Square size={20} fill="#ffffff" strokeWidth={0} />
            </button>
            <span className="shutter-hint text-recording">
              Recording in progress • Tap to stop
            </span>
          </div>
        )}

        {recordingState === "review" && (
          <div className="controls-review-mode">
            <button
              type="button"
              className="btn-camera-retake"
              onClick={handleRetake}
              aria-label="Retake recording"
            >
              <RefreshCw size={16} />
              <span>Retake</span>
            </button>

            <button
              type="button"
              className="btn-camera-confirm"
              onClick={handleConfirmRecording}
              aria-label="Use this recording for dubbing"
            >
              <Check size={18} strokeWidth={2.5} />
              <span>Use Recording ({formatTime(recordingSeconds)})</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
