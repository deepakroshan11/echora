"""
Translation service — thin wrapper around a self-hosted LibreTranslate instance.

LibreTranslate must be running separately (local Docker or a second HF Space).
Set TRANSLATE_URL environment variable to point at it.
Default: http://localhost:5000/translate
"""

import logging
import os
import requests

logger = logging.getLogger(__name__)

TRANSLATE_URL = os.getenv("TRANSLATE_URL", "http://localhost:5000/translate")
LIBRETRANSLATE_API_KEY = os.getenv("LIBRETRANSLATE_API_KEY", "")  # Empty = no key needed on self-hosted


def translate(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate text using LibreTranslate.

    Args:
        text: Source text to translate.
        source_lang: BCP-47 source language code (e.g. 'ta').
        target_lang: BCP-47 target language code (e.g. 'en').

    Returns:
        Translated text string.

    Raises:
        RuntimeError: On HTTP errors, timeouts, or empty responses.
    """
    if not text or not text.strip():
        raise ValueError("Cannot translate empty text.")

    payload = {
        "q": text,
        "source": source_lang,
        "target": target_lang,
        "format": "text",
    }
    if LIBRETRANSLATE_API_KEY:
        payload["api_key"] = LIBRETRANSLATE_API_KEY

    logger.info(f"Translating {len(text)} chars: {source_lang} → {target_lang} via {TRANSLATE_URL}")
    try:
        response = requests.post(
            TRANSLATE_URL,
            json=payload,
            timeout=60,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        data = response.json()
        translated = data.get("translatedText", "").strip()
        if not translated:
            raise RuntimeError("LibreTranslate returned empty translation.")
        logger.info(f"Translation complete: {len(translated)} chars")
        return translated
    except requests.exceptions.Timeout:
        raise RuntimeError("LibreTranslate request timed out (60s). Is the service running?")
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to LibreTranslate at {TRANSLATE_URL}. "
            "Check TRANSLATE_URL env var and ensure the service is running."
        )
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(f"LibreTranslate HTTP error {exc.response.status_code}: {exc.response.text}") from exc
    except Exception as exc:
        raise RuntimeError(f"Translation failed: {exc}") from exc
