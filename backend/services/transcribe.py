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


def transcribe(audio_path: str, source_lang: str = "ta") -> str:
    """
    Transcribe audio_path and return the full transcript as a plain string.
    Uses initial prompt conditioning for Tamil to preserve grammar and punctuation.
    """
    logger.info(f"Transcribing '{audio_path}' (lang={source_lang}) …")
    
    # Prompt priming: guides Whisper on vocabulary, script, and natural sentence punctuation
    initial_prompt = (
        "வணக்கம், இந்த வீடியோவில் நாம் பேசும் முக்கியமான விஷயங்கள். தெளிவாக, இயல்பாக கேளுங்கள்."
        if source_lang == "ta" else None
    )

    try:
        kwargs = {
            "language": source_lang,
            "fp16": False,
            "verbose": False,
            "task": "transcribe",
            "temperature": 0.0,  # greedy decoding gives most accurate words, least hallucination
            "condition_on_previous_text": True,
        }
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt

        result = _model.transcribe(audio_path, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"Whisper transcription failed: {exc}") from exc

    text = result.get("text", "").strip()
    if not text:
        raise RuntimeError("Whisper returned an empty transcript — check audio quality.")

    logger.info(f"Transcription complete ({len(text)} chars): {text[:100]}…")
    return text

