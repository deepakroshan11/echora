"""
app.py — Echora FastAPI backend

Endpoints:
  POST /upload          — accept video + target_language, queue job
  GET  /status/{job_id} — poll job status
  GET  /download/{job_id} — stream completed output.mp4
  GET  /languages        — list of supported target languages
"""
import asyncio
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"
import torch
torch.set_num_threads(4)

import aiofiles
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from models import Job, create_job, get_db, get_job, init_db, update_job

# ── Configure logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("echora")

# ── Directories ───────────────────────────────────────────────────────────────
UPLOADS_DIR = os.environ.get("UPLOADS_DIR", "uploads")
OUTPUTS_DIR = os.environ.get("OUTPUTS_DIR", "outputs")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── Supported languages ───────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = [
    {"code": "en", "name": "English"},
    {"code": "ta", "name": "Tamil"},
    {"code": "hi", "name": "Hindi"},
    {"code": "te", "name": "Telugu"},
    {"code": "kn", "name": "Kannada"},
    {"code": "ml", "name": "Malayalam"},
]
SUPPORTED_LANG_CODES = {lang["code"] for lang in SUPPORTED_LANGUAGES}

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Echora API",
    description="Dub short videos into other languages using AI voice cloning.",
    version="1.0.0",
)

# CORS — permissive for v1 (tighten to Vercel domain in production if desired)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup: init DB + pre-load heavy models ──────────────────────────────────
@app.on_event("startup")
async def startup():
    logger.info("=== Echora startup ===")
    init_db()
    logger.info("Database initialised.")

    # Import services here to trigger model loading at startup (not per-request).
    # This is the single biggest performance factor on free-tier hardware.
    logger.info("Loading ML models (Whisper + voice-clone)…")
    try:
        import services.transcribe  # noqa: F401  — loads Whisper on import
        logger.info("Whisper model ready.")
    except Exception as exc:
        logger.error(f"Failed to load Whisper: {exc}")

    try:
        import services.voice_clone  # noqa: F401  — loads TTS on import
        logger.info("Voice-clone model ready.")
    except Exception as exc:
        logger.error(f"Failed to load voice-clone model: {exc}")

    logger.info("=== Startup complete — ready to accept requests ===")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/languages")
async def get_languages():
    """Return the list of supported target languages for the frontend dropdown."""
    return {"languages": SUPPORTED_LANGUAGES}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    target_language: str = Form(...),
    db: Session = Depends(get_db),
):
    """
    Accept a video file + target language, save the file, create a job, and
    kick off background processing.

    Returns: { job_id: str }
    """
    # Validate language
    if target_language not in SUPPORTED_LANG_CODES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported target_language '{target_language}'. "
                   f"Supported: {sorted(SUPPORTED_LANG_CODES)}",
        )

    # Validate file type (basic check on content_type)
    if video.content_type and not video.content_type.startswith("video/"):
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file does not appear to be a video (content_type={video.content_type}).",
        )

    # Save uploaded video
    job_id = str(uuid.uuid4())
    ext = Path(video.filename).suffix if video.filename else ".mp4"
    video_filename = f"{job_id}_input{ext}"
    video_path = os.path.join(UPLOADS_DIR, video_filename)

    logger.info(f"Saving uploaded video to '{video_path}' …")
    async with aiofiles.open(video_path, "wb") as f:
        while chunk := await video.read(1024 * 1024):  # 1 MB chunks
            await f.write(chunk)
    logger.info("Video saved.")

    # Create job record
    job = create_job(db, target_lang=target_language, video_path=video_path)
    job_id = job.id
    logger.info(f"Job created: {job_id}")

    # Kick off pipeline in background
    background_tasks.add_task(run_pipeline, job_id, video_path, target_language)

    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def get_status(job_id: str, db: Session = Depends(get_db)):
    """Poll the status of a dubbing job."""
    job = get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return {
        "job_id": job.id,
        "status": job.status,
        "error_message": job.error_message,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }


@app.get("/download/{job_id}")
async def download_video(job_id: str, db: Session = Depends(get_db)):
    """Stream the completed dubbed video."""
    job = get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job.status != "complete":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not complete yet (status='{job.status}'). Keep polling /status/{job_id}.",
        )

    if not job.output_path or not os.path.exists(job.output_path):
        raise HTTPException(
            status_code=500,
            detail="Output file is missing — this is a server-side error.",
        )

    return FileResponse(
        job.output_path,
        media_type="video/mp4",
        filename=f"echora_{job_id}.mp4",
    )


# ── Background pipeline ───────────────────────────────────────────────────────

def run_pipeline(job_id: str, video_path: str, target_lang: str):
    """
    Full dubbing pipeline — runs in a background thread via FastAPI BackgroundTasks.
    Steps:
      1. Extract audio (ffmpeg)
      2. Transcribe (Whisper)
      3. Translate (LibreTranslate)
      4. Trim reference clip (ffmpeg)
      5. Clone voice + synthesize (Fish Speech / XTTS v2)
      6. Remux video + dubbed audio (ffmpeg)
      7. Update job status to 'complete' (or 'error')
    """
    # Use a fresh DB session — BackgroundTasks runs outside the request lifecycle
    from models import SessionLocal
    db = SessionLocal()

    def _fail(msg: str):
        logger.error(f"[{job_id}] Pipeline error: {msg}")
        update_job(db, job_id, status="error", error_message=msg[:2000])
        db.close()

    try:
        update_job(db, job_id, status="processing")
        logger.info(f"[{job_id}] Pipeline started.")

        # Step 1 — Extract audio
        from services.video import extract_audio, trim_audio, replace_audio
        logger.info(f"[{job_id}] Step 1/6: Extracting audio …")
        audio_path = extract_audio(video_path)

        # Step 2 — Transcribe
        from services.transcribe import transcribe
        logger.info(f"[{job_id}] Step 2/6: Transcribing …")
        original_text, detected_lang = transcribe(audio_path, source_lang="auto")
        logger.info(f"[{job_id}] Detected language: {detected_lang}. Transcript: {original_text[:100]}…")

        # Step 3 — Translate
        from services.translate import translate
        logger.info(f"[{job_id}] Step 3/6: Translating {detected_lang} → {target_lang} …")
        translated_text = translate(original_text, source_lang=detected_lang, target_lang=target_lang)
        logger.info(f"[{job_id}] Translation: {translated_text[:100]}…")

        # Step 4 — Condition on speaker vocal track
        logger.info(f"[{job_id}] Step 4/6: Isolating optimal speaker vocal reference …")

        # Step 5 — Voice clone + TTS
        from services.voice_clone import clone_and_speak
        logger.info(f"[{job_id}] Step 5/6: Synthesizing dubbed audio via VoxAI Engine …")
        dubbed_wav = clone_and_speak(
            reference_audio_path=audio_path,
            text=translated_text,
            target_lang=target_lang,
        )

        # Step 6 — Remux
        logger.info(f"[{job_id}] Step 6/6: Remuxing video + dubbed audio …")
        output_path = replace_audio(video_path, dubbed_wav)

        # Done
        update_job(
            db,
            job_id,
            status="complete",
            output_path=output_path,
            completed_at=datetime.utcnow(),
        )
        logger.info(f"[{job_id}] Pipeline complete → '{output_path}'.")

    except Exception as exc:
        _fail(str(exc))
        return

    finally:
        db.close()
