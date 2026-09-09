"""
services/translate.py

Wrapper around LibreTranslate + Google Translate with contextual pre/post-processing
for natural, idiomatic Tamil → English translation and prosodic TTS formatting.
"""
import os
import re
import html
import logging
import requests
import urllib.parse

logger = logging.getLogger(__name__)

TRANSLATE_URL = os.environ.get("TRANSLATE_URL", "http://localhost:5000/translate")
REQUEST_TIMEOUT = int(os.environ.get("TRANSLATE_TIMEOUT", "30"))

# Spoken/Colloquial Tamil mapping dictionary for common short video expressions
TAMIL_COLLOQUIAL_MAP = {
    "வணக்கம்": "Hello",
    "என்னடா": "Hey man,",
    "பாஸ்": "Boss",
    "ப்ரோ": "Bro",
    "செம": "Awesome",
    "கலக்கல்": "Amazing",
    "வரேன்": "I'm coming",
    "போறேன்": "I'm going",
    "என்ன ஆச்சு": "What happened",
    "பரவாயில்லை": "No problem",
}


def _preprocess_tamil_text(text: str) -> str:
    """Clean and normalize spoken Tamil transcription artifacts before translation."""
    cleaned = text.strip()
    # Replace multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned


def _postprocess_translated_text(translated: str, target_lang: str) -> str:
    """
    Post-process translation output to ensure natural grammar,
    proper capitalization, and prosodic punctuation cues for TTS models.
    """
    text = translated.strip()
    if not text:
        return text

    # Capitalize first letter
    text = text[0].upper() + text[1:]

    # Fix common literal translation artifacts in English
    if target_lang == "en":
        text = re.sub(r'\bwhat happened to\b', 'what happened', text, flags=re.IGNORECASE)
        text = re.sub(r'\bvery nice video\b', 'great video', text, flags=re.IGNORECASE)
        # Ensure sentence ending punctuation for natural TTS prosody
        if not text.endswith(('.', '!', '?', ';')):
            text += '.'

    return text


def _fallback_translate(text: str, source_lang: str, target_lang: str) -> str:
    """Fallback translation using Google Web API (no API key required)."""
    logger.info(f"Using fallback translator for {source_lang} → {target_lang} …")
    url = f"https://translate.google.com/m?sl={source_lang}&tl={target_lang}&q={urllib.parse.quote(text)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        match = re.search(r'class="result-container">(.*?)</div>', resp.text, re.DOTALL)
        if match:
            translated = html.unescape(match.group(1).strip())
            if translated:
                logger.info(f"Fallback translation complete ({len(translated)} chars).")
                return translated
    except Exception as exc:
        logger.warning(f"Google Web fallback failed: {exc}")

    # Second fallback: googletrans
    try:
        from googletrans import Translator
        translator = Translator()
        res = translator.translate(text, src=source_lang, dest=target_lang)
        if res and res.text:
            logger.info(f"googletrans translation complete ({len(res.text)} chars).")
            return res.text
    except Exception as exc:
        logger.warning(f"googletrans fallback failed: {exc}")

    raise RuntimeError(f"All translation attempts failed for {source_lang} → {target_lang}.")


def translate(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate *text* from source_lang to target_lang via LibreTranslate with automatic fallback
    and prosodic/grammatical post-processing.
    """
    if not text or not text.strip():
        raise ValueError("translate() received empty text — nothing to translate.")

    clean_input = _preprocess_tamil_text(text)
    
    payload = {
        "q": clean_input,
        "source": source_lang,
        "target": target_lang,
        "format": "text",
    }

    logger.info(f"Translating {len(clean_input)} chars: {source_lang} → {target_lang} …")
    translated = None
    try:
        resp = requests.post(TRANSLATE_URL, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        raw_translated = data.get("translatedText", "").strip()
        if raw_translated:
            logger.info(f"LibreTranslate complete ({len(raw_translated)} chars).")
            translated = raw_translated
    except Exception as exc:
        logger.warning(
            f"LibreTranslate unavailable or language unsupported ({exc}). Falling back to Google Translate…"
        )

    if not translated:
        translated = _fallback_translate(clean_input, source_lang, target_lang)

    final_output = _postprocess_translated_text(translated, target_lang)
    logger.info(f"Final post-processed translation: '{final_output}'")
    return final_output


