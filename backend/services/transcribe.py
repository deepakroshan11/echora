"""
Transcription service using OpenAI Whisper (MIT license, runs locally).

Model is loaded ONCE at module import time to avoid per-request reload overhead.
Uses 'base' model for speed on CPU-only free-tier hardware.
"""

import logging
import whisper

logger = logging.getLogger(__name__)

# Load model at import time — critical for performance on free-tier hardware
logger.info("Loading Whisper 'base' model... (this may take a moment on first run)")
_model = whisper.load_model("base")
logger.info("Whisper model loaded successfully.")


def transcribe(audio_path: str, source_lang: str = "ta") -> str:
    """
    Transcribe audio file to text.

    Args:
        audio_path: Path to WAV/MP3/etc audio file.
        source_lang: BCP-47 language code of source audio (default: 'ta' for Tamil).

    Returns:
        Transcribed text string.

    Raises:
        RuntimeError: If transcription fails or produces empty text.
    """
    logger.info(f"Transcribing {audio_path!r} (lang={source_lang})")
    try:
        result = _model.transcribe(
            audio_path,
            language=source_lang,
            task="transcribe",
            fp16=False,  # CPU-safe — fp16 requires CUDA
        )
        text = result.get("text", "").strip()
        if not text:
            raise RuntimeError("Whisper returned empty transcription.")
        logger.info(f"Transcription complete: {len(text)} chars")
        return text
    except Exception as exc:
        logger.error(f"Transcription failed: {exc}")
        raise RuntimeError(f"Transcription failed: {exc}") from exc
