"""
services/transcribe.py

Whisper-based speech-to-text.
Model is loaded ONCE at module import time so it's ready for all requests.
Default source language is Tamil ("ta").
"""
import logging
import whisper

logger = logging.getLogger(__name__)

# ── Load model at import time (not per-request) ──────────────────────────────
# "base" is the best speed/accuracy tradeoff for CPU-only free-tier hosts.
# Fall back to "tiny" by setting env var WHISPER_MODEL=tiny
import os
_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")
logger.info(f"Loading Whisper model '{_MODEL_SIZE}'…")
_model = whisper.load_model(_MODEL_SIZE)
logger.info("Whisper model loaded.")


def transcribe(audio_path: str, source_lang: str = "auto") -> tuple[str, str]:
    """
    Transcribe audio_path and return (transcript, detected_lang).
    Uses automatic language identification if source_lang is 'auto' or None.
    """
    logger.info(f"Transcribing '{audio_path}' (source_lang={source_lang}) …")
    
    lang = None if source_lang in ("auto", None, "") else source_lang
    try:
        kwargs = {
            "fp16": False,
            "verbose": False,
            "task": "transcribe",
            "temperature": 0.0,
            "condition_on_previous_text": True,
        }
        if lang:
            kwargs["language"] = lang
            if lang == "ta":
                kwargs["initial_prompt"] = "வணக்கம், இந்த வீடியோவில் நாம் பேசும் முக்கியமான விஷயங்கள்."

        result = _model.transcribe(audio_path, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"Whisper transcription failed: {exc}") from exc

    text = result.get("text", "").strip()
    detected_lang = result.get("language", "ta")
    if not text:
        raise RuntimeError("Whisper returned an empty transcript — check audio quality.")

    logger.info(f"Transcription complete (lang={detected_lang}, {len(text)} chars): {text[:100]}…")
    return text, detected_lang

