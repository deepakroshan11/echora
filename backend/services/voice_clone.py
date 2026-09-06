"""
Voice cloning + TTS service.

Primary engine: Fish Speech (Apache 2.0) — attempted first.
Fallback engine: Coqui XTTS v2 (CPML, non-commercial) — used if Fish Speech unavailable.

The model is loaded ONCE at module import time for performance.
"""

import logging
import os
import subprocess
import sys
import uuid

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Language code mapping
# LibreTranslate codes → XTTS v2 / Fish Speech language names
# ─────────────────────────────────────────────────────────────
LANG_CODE_MAP = {
    "en": "en",
    "hi": "hi",
    "te": "te",
    "kn": "kn",
    "ml": "ml",
    "ta": "ta",
    "fr": "fr",
    "de": "de",
    "es": "es",
    "zh": "zh-cn",
    "ja": "ja",
    "ko": "ko",
    "pt": "pt",
    "ar": "ar",
    "ru": "ru",
    "tr": "tr",
}

JOBS_DIR = os.getenv("JOBS_DIR", "jobs")
TTS_BACKEND = None   # Will be set to "fish_speech" or "coqui_xtts"
_tts_model = None    # Coqui TTS object (if using XTTS v2 fallback)


def _try_load_fish_speech() -> bool:
    """
    Attempt to import and configure Fish Speech.
    Returns True if successful, False if not available.
    """
    try:
        # Fish Speech CLI-based usage (most reliable install method)
        result = subprocess.run(
            [sys.executable, "-c", "import fish_speech; print('ok')"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and "ok" in result.stdout:
            logger.info("Fish Speech is available.")
            return True
    except Exception:
        pass
    logger.info("Fish Speech not available — will use Coqui XTTS v2 fallback.")
    return False


def _load_coqui_xtts():
    """Load Coqui XTTS v2 model. Called once at startup."""
    global _tts_model
    logger.info("Loading Coqui XTTS v2 model... (first load downloads ~2 GB — subsequent runs use cache)")
    try:
        from TTS.api import TTS
        # XTTS v2 — multilingual voice cloning model
        _tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
        logger.info("Coqui XTTS v2 loaded successfully.")
    except Exception as exc:
        logger.error(f"Failed to load Coqui XTTS v2: {exc}")
        raise RuntimeError(f"Voice cloning model failed to load: {exc}") from exc


def load_model():
    """
    Load the voice cloning model at application startup.
    Tries Fish Speech first, falls back to Coqui XTTS v2.
    """
    global TTS_BACKEND
    if _try_load_fish_speech():
        TTS_BACKEND = "fish_speech"
    else:
        TTS_BACKEND = "coqui_xtts"
        _load_coqui_xtts()
    logger.info(f"Voice cloning backend: {TTS_BACKEND}")


def _synthesize_fish_speech(reference_wav: str, text: str, language: str, out_path: str):
    """
    Synthesize speech using Fish Speech CLI.
    Fish Speech exposes a command-line interface for inference.
    """
    # Fish Speech CLI: fish_speech.infer --text "..." --reference_audio ref.wav --output out.wav
    cmd = [
        sys.executable, "-m", "fish_speech.infer",
        "--text", text,
        "--reference_audio", reference_wav,
        "--language", language,
        "--output", out_path,
    ]
    logger.info(f"Fish Speech synthesis: {' '.join(cmd[:6])}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"Fish Speech synthesis failed:\n{result.stderr[-1000:]}")


def _synthesize_coqui_xtts(reference_wav: str, text: str, language: str, out_path: str):
    """
    Synthesize speech using Coqui XTTS v2.
    Clones voice from reference_wav, generates text in the target language.
    """
    if _tts_model is None:
        raise RuntimeError("XTTS model not loaded. Call load_model() first.")
    xtts_lang = LANG_CODE_MAP.get(language, language)
    logger.info(f"XTTS v2 synthesis: lang={xtts_lang}, text_len={len(text)}, ref={reference_wav!r}")
    _tts_model.tts_to_file(
        text=text,
        speaker_wav=reference_wav,
        language=xtts_lang,
        file_path=out_path,
    )


def clone_and_speak(
    reference_audio_path: str,
    text: str,
    target_lang: str,
    job_id: str,
) -> str:
    """
    Clone the voice from reference_audio_path and synthesize text in target_lang.

    The reference clip should already be trimmed to 6–10 seconds by the caller
    (services/video.py::trim_audio).

    Args:
        reference_audio_path: Path to trimmed voice reference WAV (6–10s, mono 22050Hz).
        text: Translated text to synthesize.
        target_lang: BCP-47 target language code (e.g. 'en', 'hi', 'te').
        job_id: Job ID for output path organization.

    Returns:
        Path to synthesized dubbed audio WAV file.

    Raises:
        RuntimeError: If synthesis fails.
    """
    if not text or not text.strip():
        raise ValueError("Cannot synthesize empty text.")

    out_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dubbed_audio.wav")

    language = LANG_CODE_MAP.get(target_lang, target_lang)
    logger.info(f"Synthesizing voice clone: lang={language}, chars={len(text)}, backend={TTS_BACKEND}")

    try:
        if TTS_BACKEND == "fish_speech":
            _synthesize_fish_speech(reference_audio_path, text, language, out_path)
        else:
            _synthesize_coqui_xtts(reference_audio_path, text, language, out_path)
        logger.info(f"Dubbed audio written to {out_path!r}")
        return out_path
    except Exception as exc:
        logger.error(f"Voice synthesis failed: {exc}")
        raise RuntimeError(f"Voice cloning/synthesis failed: {exc}") from exc
