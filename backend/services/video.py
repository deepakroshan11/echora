"""
FFmpeg helper services for audio/video manipulation.

Requires ffmpeg to be installed in PATH (or /usr/bin/ffmpeg in Docker).
All functions run synchronously — wrap in asyncio.to_thread() if needed.
"""

import logging
import os
import subprocess
import tempfile

logger = logging.getLogger(__name__)

JOBS_DIR = os.getenv("JOBS_DIR", "jobs")


def _jobs_dir(job_id: str) -> str:
    """Return (and create) the directory for a specific job's files."""
    path = os.path.join(JOBS_DIR, job_id)
    os.makedirs(path, exist_ok=True)
    return path


def _run_ffmpeg(args: list[str], description: str):
    """Run an ffmpeg command, raising RuntimeError on failure."""
    cmd = ["ffmpeg", "-y"] + args  # -y = overwrite without prompting
    logger.info(f"FFmpeg [{description}]: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"FFmpeg [{description}] failed:\n{result.stderr}")
        raise RuntimeError(
            f"FFmpeg failed during '{description}':\n{result.stderr[-1000:]}"
        )
    return result


def get_duration(file_path: str) -> float:
    """Return duration of a media file in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def extract_audio(video_path: str, job_id: str) -> str:
    """
    Extract audio track from video as a 16kHz mono WAV file.
    16kHz is optimal for Whisper transcription and XTTS voice cloning.

    Returns:
        Path to extracted audio WAV file.
    """
    out_path = os.path.join(_jobs_dir(job_id), "audio.wav")
    _run_ffmpeg(
        [
            "-i", video_path,
            "-vn",                  # No video
            "-acodec", "pcm_s16le", # 16-bit PCM WAV
            "-ar", "16000",         # 16kHz sample rate
            "-ac", "1",             # Mono
            out_path,
        ],
        description="extract_audio",
    )
    logger.info(f"Audio extracted to {out_path!r}")
    return out_path


def trim_audio(audio_path: str, job_id: str, duration_sec: float = 8.0) -> str:
    """
    Trim an audio file to the first `duration_sec` seconds.
    Used to prepare the voice reference clip for cloning (6–10 sec is optimal).

    Returns:
        Path to trimmed audio WAV file.
    """
    out_path = os.path.join(_jobs_dir(job_id), "voice_ref.wav")
    _run_ffmpeg(
        [
            "-i", audio_path,
            "-t", str(duration_sec),  # Duration limit
            "-acodec", "pcm_s16le",
            "-ar", "22050",           # XTTS v2 expects 22050 Hz reference
            "-ac", "1",
            out_path,
        ],
        description="trim_reference",
    )
    logger.info(f"Voice reference trimmed to {out_path!r}")
    return out_path


def pad_or_trim_audio(audio_path: str, target_duration: float, job_id: str) -> str:
    """
    Ensure dubbed audio matches video duration exactly.
    - Shorter than target: pad with silence at the end.
    - Longer than target: trim to target duration.
    This avoids AV sync issues during muxing.

    Returns:
        Path to duration-adjusted WAV file.
    """
    actual = get_duration(audio_path)
    out_path = os.path.join(_jobs_dir(job_id), "dubbed_adjusted.wav")

    if abs(actual - target_duration) < 0.1:
        # Close enough — no adjustment needed
        return audio_path

    if actual < target_duration:
        # Pad with silence
        pad_sec = target_duration - actual
        logger.info(f"Padding dubbed audio by {pad_sec:.2f}s silence")
        _run_ffmpeg(
            [
                "-i", audio_path,
                "-af", f"apad=pad_dur={pad_sec}",
                "-t", str(target_duration),
                out_path,
            ],
            description="pad_audio",
        )
    else:
        # Trim
        logger.info(f"Trimming dubbed audio from {actual:.2f}s to {target_duration:.2f}s")
        _run_ffmpeg(
            ["-i", audio_path, "-t", str(target_duration), out_path],
            description="trim_audio",
        )

    return out_path


def replace_audio(video_path: str, new_audio_path: str, job_id: str) -> str:
    """
    Remux original video's video stream with new dubbed audio.
    Video stream is copied (no re-encode). Audio is encoded to AAC.

    Dubbed audio is duration-adjusted to match video before muxing.

    Returns:
        Path to output MP4 file.
    """
    out_path = os.path.join(_jobs_dir(job_id), "output.mp4")

    # Ensure dubbed audio matches video duration exactly
    video_duration = get_duration(video_path)
    adjusted_audio = pad_or_trim_audio(new_audio_path, video_duration, job_id)

    _run_ffmpeg(
        [
            "-i", video_path,        # Input 0: original video
            "-i", adjusted_audio,    # Input 1: dubbed audio
            "-map", "0:v:0",         # Take video from input 0
            "-map", "1:a:0",         # Take audio from input 1
            "-c:v", "copy",          # Copy video stream (fast, no quality loss)
            "-c:a", "aac",           # Encode audio as AAC
            "-b:a", "192k",          # Audio bitrate
            "-shortest",             # End when shortest stream ends
            out_path,
        ],
        description="replace_audio",
    )
    logger.info(f"Final video written to {out_path!r}")
    return out_path
