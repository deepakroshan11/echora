"""
ShortsDub FastAPI Backend

Endpoints:
  POST /upload           — accept video + target_language, start async job
  GET  /status/{job_id}  — poll job status
  GET  /download/{job_id} — stream completed output video
  GET  /languages        — list of supported target languages
  GET  /health           — health check

Heavy models (Whisper + voice clone) are loaded ONCE at startup.
A concurrency semaphore ensures only one job runs at a time (prevents OOM on free-tier).
"""

import asyncio
import logging
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import aiofiles
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from models import create_job, get_job, init_db, update_job
from services import transcribe as transcribe_svc
from services import translate as translate_svc
from services import video as video_svc
from services import voice_clone as voice_svc

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
UPLOADS_DIR = os.getenv("UPLOADS_DIR", "uploads")
JOBS_DIR = os.getenv("JOBS_DIR", "jobs")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "500"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Single-job concurrency guard — prevents OOM from concurrent ML inference
_job_semaphore = asyncio.Semaphore(1)

# ─────────────────────────────────────────────────────────────
# Supported languages
# ─────────────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "ta": "Tamil",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "zh": "Chinese (Simplified)",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ar": "Arabic",
    "ru": "Russian",
    "tr": "Turkish",
}


# ─────────────────────────────────────────────────────────────
# Lifespan — load models at startup
# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy ML models and init DB before serving any requests."""
    logger.info("=== ShortsDub startup: initializing resources ===")
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(JOBS_DIR, exist_ok=True)

    # Initialize SQLite (creates tables if not exist)
    init_db()
    logger.info("Database initialized.")

    # Load ML models in a thread pool (blocking I/O, avoid blocking event loop)
    loop = asyncio.get_event_loop()
    logger.info("Loading Whisper model (this happens at first import of transcribe module)...")
    # transcribe module loads Whisper at import time — just force the import
    await loop.run_in_executor(None, lambda: transcribe_svc.transcribe.__module__)

    logger.info("Loading voice cloning model...")
    await loop.run_in_executor(None, voice_svc.load_model)

    logger.info("=== ShortsDub startup complete — ready to serve ===")
    yield
    logger.info("=== ShortsDub shutting down ===")


# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="ShortsDub API",
    description="Dub Tamil short videos into other languages using open-source AI.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins (frontend may be on Vercel, localhost, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────
async def run_pipeline(job_id: str, video_path: str, target_lang: str):
    """
    Full dubbing pipeline — runs as a background task.
    Guarded by a semaphore so only one job runs at a time (OOM prevention).
    """
    async with _job_semaphore:
        logger.info(f"[{job_id}] Pipeline starting: {video_path!r} → {target_lang}")
        update_job(job_id, status="processing")

        try:
            loop = asyncio.get_event_loop()

            # Step 1: Extract audio from video
            logger.info(f"[{job_id}] Step 1/5: Extracting audio...")
            audio_path = await loop.run_in_executor(
                None, video_svc.extract_audio, video_path, job_id
            )

            # Step 2: Transcribe audio (Tamil → text)
            logger.info(f"[{job_id}] Step 2/5: Transcribing audio (Tamil)...")
            original_text = await loop.run_in_executor(
                None, transcribe_svc.transcribe, audio_path, "ta"
            )
            logger.info(f"[{job_id}] Transcription: {original_text[:100]!r}...")

            # Step 3: Translate text
            logger.info(f"[{job_id}] Step 3/5: Translating ta → {target_lang}...")
            translated_text = await loop.run_in_executor(
                None, translate_svc.translate, original_text, "ta", target_lang
            )
            logger.info(f"[{job_id}] Translation: {translated_text[:100]!r}...")

            # Step 4: Trim voice reference clip (first 8 seconds of audio)
            logger.info(f"[{job_id}] Step 4/5: Cloning voice & synthesizing speech...")
            voice_ref_path = await loop.run_in_executor(
                None, video_svc.trim_audio, audio_path, job_id, 8.0
            )

            # Step 5: Voice clone + TTS synthesis
            dubbed_audio_path = await loop.run_in_executor(
                None, voice_svc.clone_and_speak,
                voice_ref_path, translated_text, target_lang, job_id
            )

            # Step 6: Remux video with dubbed audio
            logger.info(f"[{job_id}] Step 5/5: Remuxing video with dubbed audio...")
            output_path = await loop.run_in_executor(
                None, video_svc.replace_audio, video_path, dubbed_audio_path, job_id
            )

            update_job(
                job_id,
                status="complete",
                output_path=output_path,
                completed_at=datetime.utcnow(),
            )
            logger.info(f"[{job_id}] ✅ Pipeline complete: {output_path!r}")

        except Exception as exc:
            logger.error(f"[{job_id}] ❌ Pipeline error: {exc}", exc_info=True)
            update_job(
                job_id,
                status="error",
                error_message=str(exc),
                completed_at=datetime.utcnow(),
            )


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "shortsdub-backend"}


@app.get("/languages")
async def get_languages():
    """Return supported target language codes and display names."""
    return {
        "languages": [
            {"code": code, "name": name}
            for code, name in SUPPORTED_LANGUAGES.items()
        ]
    }


@app.post("/upload")
async def upload(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    target_language: str = Form(...),
):
    """
    Accept a video file upload and start an async dubbing job.
    Returns a job_id immediately; use GET /status/{job_id} to poll.
    """
    if target_language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{target_language}'. "
                   f"Supported: {list(SUPPORTED_LANGUAGES.keys())}",
        )

    # Validate file type loosely (accept anything video/*)
    content_type = video.content_type or ""
    if not content_type.startswith("video/") and not video.filename.lower().endswith(
        (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid video file (mp4, mov, avi, mkv, webm, m4v).",
        )

    # Save uploaded file
    job_id = str(uuid.uuid4())
    upload_dir = os.path.join(UPLOADS_DIR, job_id)
    os.makedirs(upload_dir, exist_ok=True)

    suffix = Path(video.filename).suffix or ".mp4"
    video_path = os.path.join(upload_dir, f"input{suffix}")

    async with aiofiles.open(video_path, "wb") as f:
        content = await video.read()
        if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB} MB.",
            )
        await f.write(content)

    logger.info(f"[{job_id}] Video saved to {video_path!r} ({len(content)/1e6:.1f} MB)")

    # Create job record in DB
    job = create_job(target_lang=target_language, video_path=video_path)

    # Kick off pipeline as background task
    background_tasks.add_task(run_pipeline, job_id, video_path, target_language)

    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Upload successful. Processing has started.",
    }


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Poll job status. Status ∈ queued|processing|complete|error."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return {
        "job_id": job.id,
        "status": job.status,
        "source_lang": job.source_lang,
        "target_lang": job.target_lang,
        "error_message": job.error_message,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }


@app.get("/download/{job_id}")
async def download(job_id: str):
    """Stream the completed dubbed video. Returns 404 if job is not done."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.status != "complete":
        raise HTTPException(
            status_code=409,
            detail=f"Job is not complete yet (status: {job.status}).",
        )
    if not job.output_path or not os.path.exists(job.output_path):
        raise HTTPException(
            status_code=404, detail="Output file not found. The job may have failed."
        )
    return FileResponse(
        path=job.output_path,
        media_type="video/mp4",
        filename=f"shortsdub_{job_id[:8]}.mp4",
    )
