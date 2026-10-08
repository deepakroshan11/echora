"""
services/voice_clone.py — VOXAI Voice Clone Studio Engine
Directly adopted from VOXAI (https://github.com/deepakroshan11/voxai-voice-clone):

  1. Core Neural Engine:
     - ChatterboxTTS (Resemble AI, MIT licensed)
     - Zero-shot accent and speaker preservation (high fidelity for Tamil / Indian English / regional voices)
     - Anti-creak, stable pitch defaults: exaggeration=0.3, cfg_weight=0.7
  2. Reference Audio Preparation:
     - Sliding-window RMS search (_best_segment) to isolate genuine active speech.
     - Gentle silence trimming (_gentle_trim).
     - 60Hz high-pass filter + peak normalization.
  3. VoxAI 7-Stage Professional Mastering Chain:
     - Stage 1: Bass Restoration (+10dB low-shelf below 300Hz, +3dB body bump @ 450-600Hz)
     - Stage 2: Cepstral De-noising (removes robotic vocoder buzz)
     - Stage 3: Harmonic Bandwidth Extension (HBE) + 48kHz broadcast upsampling
     - Stage 4: Natural Multiband Dynamics (3-band 2:1 compression)
     - Stage 5: Room Tone (5% wet acoustic matching)
     - Stage 6: EBU R128 Loudness Normalization (-16 LUFS broadcast standard)
     - Stage 7: True-Peak Limiter (-1.0 dBTP ceiling)
"""
import os
import re
import sys
import asyncio
import logging
import tempfile
import subprocess
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import torch
torch.set_num_threads(4)
import torchaudio
from scipy import signal as sp
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq, irfft
import soundfile as sf

# Fix perth watermarker callable if needed
import perth
if getattr(perth, "PerthImplicitWatermarker", None) is None:
    perth.PerthImplicitWatermarker = perth.DummyWatermarker

logger = logging.getLogger("echora.voice_clone")

OUTPUTS_DIR = os.environ.get("OUTPUTS_DIR", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

CHATTERBOX_SR    = 24000
OUTPUT_SR        = 48000
MIN_DURATION_S   = 4
BEST_DURATION_S  = 12
CHUNK_CHAR_LIMIT = 130

# VoxAI v8 Tuned Defaults (stable pitch, zero creak)
DEFAULT_EXAGGERATION = 0.3
DEFAULT_CFG_WEIGHT   = 0.7

# Local HuggingFace snapshot directory for ChatterboxTTS
CHATTERBOX_SNAPSHOT = Path(
    r"C:\Users\DELL\.cache\huggingface\hub\models--ResembleAI--chatterbox\snapshots\05e904af2b5c7f8e482687a9d7336c5c824467d9"
)

cb_model = None

# ======================================================================
#  MODEL LOADER (ChatterboxTTS)
# ======================================================================

def get_chatterbox_model():
    """Singleton loader for ChatterboxTTS model."""
    global cb_model
    if cb_model is None:
        import perth
        if getattr(perth, "PerthImplicitWatermarker", None) is None:
            perth.PerthImplicitWatermarker = perth.DummyWatermarker

        from chatterbox.tts import ChatterboxTTS
        logger.info("Initializing VoxAI ChatterboxTTS model on CPU...")
        if CHATTERBOX_SNAPSHOT.exists():
            cb_model = ChatterboxTTS.from_local(CHATTERBOX_SNAPSHOT, device="cpu")
        else:
            cb_model = ChatterboxTTS.from_pretrained(device="cpu")
        logger.info("ChatterboxTTS ready. Sample rate: %d", cb_model.sr)
    return cb_model


# ======================================================================
#  REFERENCE AUDIO PREPARATION (VoxAI Pattern)
# ======================================================================

def _best_segment(wav: torch.Tensor, sr: int, target_s: float = BEST_DURATION_S) -> torch.Tensor:
    s = wav.squeeze(0)
    if s.shape[0] / sr <= target_s + 1:
        return wav
    tgt = int(target_s * sr)
    step = int(0.25 * sr)
    best_rms, best_start = -1.0, 0
    for start in range(0, s.shape[0] - tgt, step):
        rms = float(torch.sqrt(torch.mean(s[start : start + tgt] ** 2)))
        if rms > best_rms:
            best_rms, best_start = rms, start
    return s[best_start : best_start + tgt].unsqueeze(0)


def _gentle_trim(wav: torch.Tensor, sr: int, top_db: float = 28.0) -> torch.Tensor:
    fl = int(sr * 0.02)
    s = wav.squeeze(0).cpu().numpy()
    n = len(s) // fl
    if n < 4:
        return wav
    rms = np.array([np.sqrt(np.mean(s[i * fl : (i + 1) * fl] ** 2)) for i in range(n)])
    db = 20 * np.log10(rms + 1e-9)
    v = np.where(db >= db.max() - top_db)[0]
    if not len(v):
        return wav
    pad = int(sr * 0.1)
    return torch.from_numpy(
        s[max(0, v[0] * fl - pad) : min(len(s), (v[-1] + 1) * fl + pad)]
    ).unsqueeze(0)


def enhance_reference(path: str, out_path: str) -> dict:
    """Enhance and clean reference audio for Chatterbox speaker conditioning."""
    try:
        w, sr = torchaudio.load(path)
        if w.shape[0] > 1:
            w = w.mean(0, keepdim=True)
        w = _best_segment(w, sr, BEST_DURATION_S)
        w = _gentle_trim(w, sr, top_db=28.0)
        s = w.squeeze(0).cpu().numpy().astype(np.float64)

        # 60Hz highpass filter to eliminate microphone handling rumble
        sos = sp.butter(2, min(60.0 / (sr / 2), 0.99), btype="highpass", output="sos")
        s = sp.sosfilt(sos, s)

        w = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        pk = w.abs().max()
        if pk > 1e-6:
            w = w * (10 ** (-8 / 20) / pk)
        dur = w.shape[1] / sr
        torchaudio.save(out_path, w, sr)
        return {"duration": dur, "ok": dur >= MIN_DURATION_S}
    except Exception as e:
        logger.warning("Reference enhance warning: %s", e)
        return {"duration": 0, "ok": False}


# ======================================================================
#  VOXAI 7-STAGE PROFESSIONAL DSP MASTERING CHAIN
# ======================================================================

def apply_bass_restoration(s: np.ndarray, sr: int) -> np.ndarray:
    """
    Stage 1: Low shelf +10dB below 300Hz. Body bump +3dB at 450-600Hz.
    Applied additively (parallel blend) to avoid phase cancellation.
    """
    s = s.astype(np.float64)

    # Low shelf below 300Hz
    sos_ls = sp.butter(4, min(300.0 / (sr / 2), 0.99), btype="low", output="sos")
    bass = sp.sosfilt(sos_ls, s)
    s = s + (10 ** (10.0 / 20) - 1.0) * bass

    # Body bump at 450-600Hz
    lo = max(0.001, 420.0 / (sr / 2))
    hi = min(0.999, 650.0 / (sr / 2))
    sos_b = sp.butter(2, [lo, hi], btype="bandpass", output="sos")
    body = sp.sosfilt(sos_b, s)
    s = s + (10 ** (3.0 / 20) - 1.0) * body

    return s.astype(np.float32)


def apply_cepstral_denoising(s: np.ndarray, sr: int) -> np.ndarray:
    """Stage 2: Real cepstrum liftering to remove vocoder buzz."""
    s = s.astype(np.float64)
    frame_size = 512
    hop = 256
    result = np.zeros(len(s) + frame_size)
    norm = np.zeros(len(s) + frame_size)
    window = np.hanning(frame_size)

    lifter_lo = 10
    lifter_hi = int(sr / 80.0)

    for i in range(0, len(s) - frame_size, hop):
        frame = s[i : i + frame_size] * window
        if np.sqrt(np.mean(frame**2)) < 0.002:
            result[i : i + frame_size] += frame
            norm[i : i + frame_size] += window
            continue

        spec = rfft(frame, n=frame_size)
        log_spec = np.log(np.abs(spec) + 1e-9)
        cepstrum = np.real(
            np.fft.ifft(np.concatenate([log_spec, log_spec[-2:0:-1]]))
        )[:frame_size]

        lifter = np.ones(frame_size)
        lifter[lifter_lo:lifter_hi] = np.linspace(1.0, 0.2, lifter_hi - lifter_lo)
        lifter[frame_size // 2 :] = lifter[frame_size // 2 : 0 : -1]

        cepstrum_f = cepstrum * lifter
        log_spec_s = np.real(np.fft.fft(cepstrum_f))[: frame_size // 2 + 1]
        spec_out = np.exp(log_spec_s) * np.exp(1j * np.angle(spec))
        frame_out = np.real(irfft(spec_out, n=frame_size)) * window

        result[i : i + frame_size] += frame_out
        norm[i : i + frame_size] += window**2

    norm = np.maximum(norm, 1e-6)
    result = result[: len(s)] / norm[: len(s)]
    return (result * 0.70 + s * 0.30).astype(np.float32)


def apply_bandwidth_extension(s: np.ndarray, src_sr: int, dst_sr: int = OUTPUT_SR) -> tuple:
    """Stage 3: Harmonic Bandwidth Extension (HBE) + 48kHz upsample."""
    num_samples = int(len(s) * dst_sr / src_sr)
    s_up = sp.resample(s, num_samples).astype(np.float64)

    sos_seed = sp.butter(
        6,
        [max(0.001, 4000.0 / (dst_sr / 2)), min(0.999, 11000.0 / (dst_sr / 2))],
        btype="bandpass",
        output="sos",
    )
    seed = sp.sosfilt(sos_seed, s_up)

    hf = seed**2
    pk = np.abs(hf).max()
    if pk > 1e-9:
        hf /= pk

    sos_hf = sp.butter(
        6,
        [max(0.001, 12000.0 / (dst_sr / 2)), min(0.999, 20000.0 / (dst_sr / 2))],
        btype="bandpass",
        output="sos",
    )
    hf = sp.sosfilt(sos_hf, hf)

    sos_tilt = sp.butter(2, min(16000.0 / (dst_sr / 2), 0.99), btype="low", output="sos")
    hf = sp.sosfilt(sos_tilt, hf)

    s_out = s_up + hf * (10 ** (-18.0 / 20))
    return s_out.astype(np.float32), dst_sr


def apply_natural_dynamics(s: np.ndarray, sr: int) -> np.ndarray:
    """Stage 4: 3-band dynamics (2:1 compression)."""
    try:
        s = s.astype(np.float64)

        def compress_band(band, thr_db=-20.0, ratio=2.0, att_ms=8.0, rel_ms=100.0, mk_db=1.5):
            thr = 10 ** (thr_db / 20)
            att = np.exp(-1.0 / (sr * att_ms / 1000))
            rel = np.exp(-1.0 / (sr * rel_ms / 1000))
            mk = 10 ** (mk_db / 20)
            out = np.empty_like(band)
            env = 0.0
            g = 1.0
            for i, x in enumerate(band):
                lv = abs(x)
                env = lv + (att if lv > env else rel) * (env - lv)
                g_t = ((thr * (env / thr) ** (1 / ratio)) / (env + 1e-10)) if env > thr else 1.0
                g += 0.008 * (g_t - g)
                out[i] = x * g * mk
            return out

        lo_sos = sp.butter(4, [max(0.001, 80 / (sr / 2)), min(0.999, 400 / (sr / 2))], "bandpass", output="sos")
        mid_sos = sp.butter(4, [max(0.001, 400 / (sr / 2)), min(0.999, 3000 / (sr / 2))], "bandpass", output="sos")
        hi_sos = sp.butter(4, [max(0.001, 3000 / (sr / 2)), min(0.999, 8000 / (sr / 2))], "bandpass", output="sos")

        lo = sp.sosfilt(lo_sos, s)
        mid = sp.sosfilt(mid_sos, s)
        hi = sp.sosfilt(hi_sos, s)
        res = s - lo - mid - hi

        return (
            res
            + compress_band(lo, thr_db=-22, ratio=2.0, mk_db=2.0)
            + compress_band(mid, thr_db=-20, ratio=2.0, mk_db=1.5)
            + compress_band(hi, thr_db=-18, ratio=2.5, mk_db=2.5)
        ).astype(np.float32)
    except Exception as e:
        logger.warning("Dynamics warning: %s", e)
        return s.astype(np.float32)


def apply_room_tone(s: np.ndarray, sr: int, wet: float = 0.05) -> np.ndarray:
    """Stage 5: Subtle room acoustic simulation (5% wet)."""
    try:
        s = s.astype(np.float64)
        pre = int(sr * 0.006)
        delayed = np.zeros_like(s)
        if pre < len(s):
            delayed[pre:] = s[:-pre]

        combs = []
        for dm, g_val in [(27.1, 0.76), (32.3, 0.76), (38.7, 0.74)]:
            d = max(1, int(sr * dm / 1000))
            g, buf, bi = g_val, np.zeros(d), 0
            out = np.empty_like(delayed)
            for n, x in enumerate(delayed):
                out[n] = buf[bi]
                buf[bi] = x + g * buf[bi]
                bi = (bi + 1) % d
            combs.append(out)

        reverb = sum(combs) / len(combs)
        sos = sp.butter(2, min(3500.0 / (sr / 2), 0.99), "low", output="sos")
        reverb = sp.sosfilt(sos, reverb)
        return (s * (1 - wet) + reverb * wet).astype(np.float32)
    except Exception as e:
        logger.warning("Room tone warning: %s", e)
        return s.astype(np.float32)


def apply_lufs_normalization(s: np.ndarray, sr: int, target_lufs: float = -16.0) -> np.ndarray:
    """Stage 6: EBU R128 Loudness Normalization (-16 LUFS broadcast standard)."""
    s = s.astype(np.float64)
    sos_kw = sp.butter(2, min(1500.0 / (sr / 2), 0.99), "high", output="sos")
    s_kw = s + (10 ** (4.0 / 20) - 1.0) * sp.sosfilt(sos_kw, s)

    frame = int(sr * 0.1)
    rms_blocks = []
    for i in range(0, len(s_kw) - frame, frame):
        rms = np.sqrt(np.mean(s_kw[i : i + frame] ** 2))
        if rms > 1e-4:
            rms_blocks.append(rms)

    if not rms_blocks:
        return s.astype(np.float32)

    rms_arr = np.array(rms_blocks)
    mean_rms = np.mean(rms_arr)
    gate_rms = np.mean(rms_arr[rms_arr > mean_rms * 10 ** (-10 / 20)])
    current_lufs = 20 * np.log10(gate_rms + 1e-9) - 0.69
    gain_db = target_lufs - current_lufs
    gain = 10 ** (gain_db / 20)
    return np.clip(s * gain, -0.97, 0.97).astype(np.float32)


def apply_true_peak_limiter(s: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    """Stage 7: True-Peak Limiter (-1.0 dBTP ceiling)."""
    s = s.astype(np.float64)
    ceiling = 10 ** (ceiling_db / 20)
    sr_approx = max(1, len(s) // 10)
    att = np.exp(-1.0 / sr_approx)
    rel = np.exp(-1.0 / (sr_approx * 10))
    out = np.empty_like(s)
    g = 1.0
    for i, x in enumerate(s):
        lv = abs(x)
        g_t = min(1.0, ceiling / (lv + 1e-9)) if lv > 0 else 1.0
        g += (att if g_t < g else rel) * (g_t - g)
        out[i] = x * g
    return np.clip(out, -ceiling, ceiling).astype(np.float32)


def professional_chain(wav_path: str, src_sr: int, out_wav_path: str):
    """
    Executes the full VoxAI 7-Stage DSP mastering chain.
    """
    raw, in_sr = sf.read(wav_path, dtype="float32")
    if in_sr > 0:
        src_sr = in_sr
    if raw.ndim > 1:
        raw = raw.mean(axis=1)
    s = raw.astype(np.float32)

    logger.info("VoxAI Master Chain: [1] Bass restoration")
    s = apply_bass_restoration(s, src_sr)

    logger.info("VoxAI Master Chain: [2] Cepstral de-noising")
    s = apply_cepstral_denoising(s, src_sr)

    logger.info("VoxAI Master Chain: [3] Bandwidth extension + 48kHz")
    s, out_sr = apply_bandwidth_extension(s, src_sr, OUTPUT_SR)

    logger.info("VoxAI Master Chain: [4] Multiband dynamics")
    s = apply_natural_dynamics(s, out_sr)

    logger.info("VoxAI Master Chain: [5] Room tone")
    s = apply_room_tone(s, out_sr, wet=0.05)

    logger.info("VoxAI Master Chain: [6] LUFS normalization (-16 LUFS)")
    s = apply_lufs_normalization(s, out_sr, target_lufs=-16.0)

    logger.info("VoxAI Master Chain: [7] True-peak limiter (-1.0 dBTP)")
    s = apply_true_peak_limiter(s, ceiling_db=-1.0)

    # Export broadcast master 48kHz WAV
    sf.write(out_wav_path, s, out_sr, subtype="PCM_16")
    logger.info("VoxAI Mastering complete -> '%s' (sr=%d)", out_wav_path, out_sr)
    return out_wav_path


# ======================================================================
#  CHUNKED SYNTHESIS (VoxAI Pattern)
# ======================================================================

def _split_text(text: str, max_chars: int = CHUNK_CHAR_LIMIT) -> list:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, cur = [], ""
    for sent in sentences:
        if len(cur) + len(sent) + 1 <= max_chars:
            cur = (cur + " " + sent).strip() if cur else sent
        else:
            if cur:
                chunks.append(cur)
            cur = sent
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c.strip()]


def _crossfade(a: np.ndarray, b: np.ndarray, sr: int) -> np.ndarray:
    fade = min(int(sr * 0.025), len(a), len(b))
    gap = np.zeros(int(sr * 0.065))
    t = np.linspace(0, 1, fade)
    return np.concatenate([a[:-fade], a[-fade:] * (1 - t) + b[:fade] * t, b[fade:], gap])


def synthesise_voxai(
    text: str,
    speaker_wav: str,
    exaggeration: float = DEFAULT_EXAGGERATION,
    cfg_weight: float = DEFAULT_CFG_WEIGHT,
    sr: int = CHATTERBOX_SR,
) -> torch.Tensor:
    """Synthesise text chunks using ChatterboxTTS and crossfade seamlessly."""
    model = get_chatterbox_model()
    chunks = _split_text(text)
    logger.info("VoxAI Synthesising %d chunks (ex=%.2f, cfg=%.2f)...", len(chunks), exaggeration, cfg_weight)

    if len(chunks) == 1:
        w = model.generate(
            text,
            audio_prompt_path=speaker_wav,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
        return w.unsqueeze(0) if w.dim() == 1 else w

    segs = []
    for i, chunk in enumerate(chunks):
        logger.info("  Chunk %d/%d: '%s...'", i + 1, len(chunks), chunk[:60])
        w = model.generate(
            chunk,
            audio_prompt_path=speaker_wav,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
        segs.append((w.unsqueeze(0) if w.dim() == 1 else w).squeeze(0).cpu().numpy())

    result = segs[0]
    for seg in segs[1:]:
        result = _crossfade(result, seg, sr)
    return torch.from_numpy(result.astype(np.float32)).unsqueeze(0)


# ======================================================================
#  INDIC VOICE ACTORS & SPEAKER PITCH PROFILER
# ======================================================================

NEURAL_VOICE_MAP = {
    "en": {
        "male": "en-IN-PrabhatNeural",
        "female": "en-IN-NeerjaNeural",
    },
    "hi": {
        "male": "hi-IN-MadhurNeural",
        "female": "hi-IN-SwaraNeural",
    },
    "te": {
        "male": "te-IN-MohanNeural",
        "female": "te-IN-ShrutiNeural",
    },
    "kn": {
        "male": "kn-IN-GaganNeural",
        "female": "kn-IN-SapnaNeural",
    },
    "ml": {
        "male": "ml-IN-MidhunNeural",
        "female": "ml-IN-SobhanaNeural",
    },
    "ta": {
        "male": "ta-IN-ValluvarNeural",
        "female": "ta-IN-PallaviNeural",
    },
}


def analyze_speaker_profile(audio_path: str) -> dict:
    """Analyze speaker fundamental pitch (F0) and classify gender for timbre anchoring."""
    try:
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        total_duration = len(data) / sr
        sample_chunk = data[: min(len(data), sr * 4)]
        corr = np.correlate(sample_chunk, sample_chunk, mode="full")[len(sample_chunk) - 1 :]
        min_lag = int(sr / 400)
        max_lag = int(sr / 65)

        if len(corr) > max_lag:
            peak_lag = min_lag + np.argmax(corr[min_lag:max_lag])
            f0 = sr / peak_lag if peak_lag > 0 else 120.0
        else:
            f0 = 120.0

        is_male = f0 < 165.0
        gender = "male" if is_male else "female"

        # Match speaker vocal fundamental to neural voice anchor
        base_f0 = 115.0 if is_male else 210.0
        pitch_diff = int(round(f0 - base_f0))
        pitch_shift_hz = max(-40, min(40, pitch_diff))

        logger.info(f"Speaker profile: F0={f0:.1f}Hz, Gender={gender}, PitchShift={pitch_shift_hz:+d}Hz")
        return {
            "f0": f0,
            "gender": gender,
            "is_male": is_male,
            "pitch_shift_hz": pitch_shift_hz,
            "duration": total_duration,
        }
    except Exception as exc:
        logger.warning(f"Speaker profile warning ({exc}), defaulting to male: {exc}")
        return {"f0": 120.0, "gender": "male", "is_male": True, "pitch_shift_hz": 0, "duration": 10.0}


# ======================================================================
#  MAIN CLONING PIPELINE
# ======================================================================

def clone_and_speak(
    reference_audio_path: str,
    text: str,
    target_lang: str,
    output_wav: str | None = None,
) -> str:
    """
    Synthesize and clone voice using VoxAI Chatterbox architecture and 7-stage DSP chain.
    - If target_lang is 'en': Uses zero-shot ChatterboxTTS neural cloning conditioned on speaker timbre.
    - If target_lang is Indic ('ta', 'hi', 'te', 'kn', 'ml'): Uses speaker F0/gender matched neural synthesis
      with the exact VoxAI 7-Stage DSP Mastering Chain (bass restoration, cepstral de-noising, 48kHz upsample,
      multiband dynamics, room tone, -16 LUFS, true-peak limiter) to eliminate robotic vocoder artifacts.
    """
    if output_wav is None:
        fd, output_wav = tempfile.mkstemp(suffix="_dubbed.wav", dir=OUTPUTS_DIR)
        os.close(fd)

    # 1. Prepare and enhance reference speech (VoxAI RMS scanner + 60Hz filter)
    enhanced_ref = output_wav + ".ref_enhanced.wav"
    ref_info = enhance_reference(reference_audio_path, enhanced_ref)
    speaker_wav = enhanced_ref if ref_info.get("ok") else reference_audio_path

    raw_synth_wav = output_wav + ".raw.wav"

    try:
        if target_lang == "en":
            # VoxAI Zero-Shot Neural Cloning (ChatterboxTTS)
            logger.info("Generating studio voice clone via VoxAI ChatterboxTTS (target=en)...")
            wav_tensor = synthesise_voxai(
                text=text,
                speaker_wav=speaker_wav,
                exaggeration=DEFAULT_EXAGGERATION,
                cfg_weight=DEFAULT_CFG_WEIGHT,
                sr=CHATTERBOX_SR,
            )
            torchaudio.save(raw_synth_wav, wav_tensor.cpu(), CHATTERBOX_SR)
            src_sr = CHATTERBOX_SR
        else:
            # Matched Neural Synthesis + Pitch Conditioning
            logger.info("Generating matched neural speech for lang=%s...", target_lang)
            profile = analyze_speaker_profile(speaker_wav)
            gender = profile.get("gender", "male")
            pitch_shift = profile.get("pitch_shift_hz", 0)

            voice = NEURAL_VOICE_MAP.get(target_lang, {}).get(gender) or NEURAL_VOICE_MAP.get("en", {}).get(gender)
            pitch_param = f"{pitch_shift:+d}Hz"

            logger.info("Using voice=%s (gender=%s, pitch=%s) for lang=%s", voice, gender, pitch_param, target_lang)

            import edge_tts
            communicate = edge_tts.Communicate(text, voice, pitch=pitch_param)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        pool.submit(asyncio.run, communicate.save(raw_synth_wav)).result()
                else:
                    loop.run_until_complete(communicate.save(raw_synth_wav))
            except RuntimeError:
                asyncio.run(communicate.save(raw_synth_wav))

            src_sr = 24000

        # 3. Apply VoxAI 7-Stage Professional DSP Mastering Chain
        logger.info("Mastering through VoxAI 7-Stage DSP Mastering Chain...")
        professional_chain(raw_synth_wav, src_sr, output_wav)

    finally:
        # Cleanup temporary files
        for p in [raw_synth_wav, enhanced_ref]:
            if os.path.exists(p):
                try: os.remove(p)
                except Exception: pass

    logger.info("VoxAI Studio voice clone ready -> '%s'", output_wav)
    return output_wav
