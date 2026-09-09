"""
services/voice_clone.py

Voice cloning + TTS service.

Strategy (in order):
  1. Try Fish Speech (Apache-2.0) — preferred
  2. Fall back to Coqui XTTS v2 (TTS package, CPML non-commercial license)

The model is loaded ONCE at module import time.
Exposes: clone_and_speak(reference_audio_path, text, target_lang) -> wav_path
"""
import os
import logging
import tempfile

logger = logging.getLogger(__name__)

OUTPUTS_DIR = os.environ.get("OUTPUTS_DIR", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── Language code mapping ─────────────────────────────────────────────────────
# LibreTranslate / Whisper use BCP-47 codes; TTS models may use different names.
LANG_CODE_TO_NAME = {
    "en": "en",
    "hi": "hi",
    "te": "te",
    "kn": "kn",
    "ml": "ml",
    "ta": "ta",
}

# ── Model Loading ─────────────────────────────────────────────────────────────
_ENGINE = None   # "fish" | "xtts"
_tts_model = None


def _try_load_fish_speech():
    """Attempt to load Fish Speech. Returns True if successful."""
    try:
        # Fish Speech doesn't have a stable pip package name yet;
        # it's typically used via its CLI or by cloning the repo.
        # We attempt to import the inference module if the user has it installed.
        import fish_speech  # noqa: F401
        logger.info("Fish Speech found — using as voice clone engine.")
        return True
    except ImportError:
        logger.warning("Fish Speech not found — falling back to Coqui XTTS v2.")
        return False


def _load_xtts():
    """Load Coqui XTTS v2 model. Returns the TTS instance."""
    import torch
    import torchaudio
    import soundfile as sf

    # PyTorch 2.6 default weights_only=True breaks Coqui TTS checkpoint deserialisation.
    _orig_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load

    # Torchaudio 2.11 torchcodec patch using soundfile for Windows audio decoding
    def _patched_ta_load(uri, **kwargs):
        data, sr = sf.read(uri, dtype='float32')
        tensor = torch.from_numpy(data)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        else:
            tensor = tensor.T
        return tensor, sr
    torchaudio.load = _patched_ta_load

    from TTS.api import TTS as CoquiTTS
    logger.info("Loading Coqui XTTS v2 model (this may take a few minutes on first run)…")
    model = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2")
    logger.info("Coqui XTTS v2 loaded.")
    return model




# Load at import time
try:
    if _try_load_fish_speech():
        _ENGINE = "fish"
        import fish_speech as _fish_speech_module  # type: ignore
    else:
        _ENGINE = "xtts"
        _tts_model = _load_xtts()
except Exception as exc:
    logger.error(f"Failed to load primary voice-clone engine ({exc}). Will fall back to gTTS when needed.")
    _ENGINE = "gtts"

logger.info(f"Voice clone engine: {_ENGINE}")


# ── Public API ────────────────────────────────────────────────────────────────

def _synthesize_gtts(text: str, target_lang: str, output_wav: str) -> str:
    """Fallback TTS using gTTS + ffmpeg conversion to WAV."""
    import subprocess
    from gtts import gTTS
    logger.info(f"gTTS fallback synthesis: lang={target_lang}, {len(text)} chars")
    
    mp3_tmp = output_wav + ".mp3"
    try:
        tts = gTTS(text=text, lang=target_lang)
        tts.save(mp3_tmp)
        # Convert MP3 to WAV using ffmpeg
        cmd = ["ffmpeg", "-y", "-i", mp3_tmp, "-ar", "22050", "-ac", "1", output_wav]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {res.stderr}")
        logger.info(f"gTTS output saved to '{output_wav}'.")
        return output_wav
    finally:
        if os.path.exists(mp3_tmp):
            try:
                os.remove(mp3_tmp)
            except Exception:
                pass


def clone_and_speak(
    reference_audio_path: str,
    text: str,
    target_lang: str,
    output_wav: str | None = None,
) -> str:
    """
    Generate speech using the voice cloned from reference_audio_path.

    Args:
        reference_audio_path: Path to a short WAV clip (6-10 sec) of the speaker.
        text:                 Text to synthesize in the target language.
        target_lang:          BCP-47 language code (e.g. "en", "hi").
        output_wav:           Output path for the generated WAV.
                              If None, a temp file in OUTPUTS_DIR is created.

    Returns:
        Path to the generated WAV file.
    """
    if output_wav is None:
        fd, output_wav = tempfile.mkstemp(suffix="_dubbed.wav", dir=OUTPUTS_DIR)
        os.close(fd)

    try:
        if _ENGINE == "fish":
            return _synthesize_fish(reference_audio_path, text, target_lang, output_wav)
        elif _ENGINE == "xtts" and _tts_model is not None:
            return _synthesize_xtts(reference_audio_path, text, target_lang, output_wav)
        else:
            return _synthesize_gtts(text, target_lang, output_wav)
    except Exception as exc:
        logger.warning(f"Voice clone synthesis failed ({exc}). Falling back to gTTS…")
        return _synthesize_gtts(text, target_lang, output_wav)


def _synthesize_xtts(
    reference_audio_path: str,
    text: str,
    target_lang: str,
    output_wav: str,
) -> str:
    """Synthesize using Coqui XTTS v2 with tuned hyperparameters for natural voice cloning."""
    lang = LANG_CODE_TO_NAME.get(target_lang, "en")
    logger.info(f"XTTS synthesis: lang={lang}, ref='{reference_audio_path}', {len(text)} chars")

    try:
        # Temperature 0.65 prevents pitch instability / robotic artifacts
        # Repetition penalty 2.5 avoids looping or stuttering on syllables
        _tts_model.tts_to_file(
            text=text,
            speaker_wav=reference_audio_path,
            language=lang,
            file_path=output_wav,
            temperature=0.65,
            repetition_penalty=2.5,
            speed=1.0,
        )
    except TypeError:
        # Fallback if tts_to_file signature varies across TTS releases
        _tts_model.tts_to_file(
            text=text,
            speaker_wav=reference_audio_path,
            language=lang,
            file_path=output_wav,
        )
    except Exception as exc:
        raise RuntimeError(f"XTTS synthesis failed: {exc}") from exc

    logger.info(f"XTTS output saved to '{output_wav}'.")
    return output_wav



def _synthesize_fish(
    reference_audio_path: str,
    text: str,
    target_lang: str,
    output_wav: str,
) -> str:
    """
    Synthesize using Fish Speech.
    """
    import subprocess
    import sys

    fish_script = os.environ.get("FISH_SPEECH_INFER_SCRIPT", "fish_speech/inference.py")
    cmd = [
        sys.executable, fish_script,
        "--reference-audio", reference_audio_path,
        "--text", text,
        "--lang", target_lang,
        "--output", output_wav,
    ]
    logger.info(f"Fish Speech command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Fish Speech inference failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    logger.info(f"Fish Speech output saved to '{output_wav}'.")
    return output_wav

