"""
services/video.py

FFmpeg helpers for audio extraction and audio-replacement in video files.

Requirements:
  - ffmpeg must be installed and on PATH (on Debian/Ubuntu: apt-get install -y ffmpeg)
  - pip install ffmpeg-python
"""
import os
import logging
import subprocess
import ffmpeg

logger = logging.getLogger(__name__)

OUTPUTS_DIR = os.environ.get("OUTPUTS_DIR", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def extract_audio(video_path: str, output_wav: str | None = None) -> str:
    """
    Extract the audio track from video_path and save it as a 16 kHz mono WAV.
    Whisper works best with 16 kHz mono audio.

    Args:
        video_path:  Path to the source video file.
        output_wav:  Destination path for the extracted WAV.
                     If None, saved alongside the video with a .wav extension.

    Returns:
        Path to the extracted WAV file.

    Raises:
        RuntimeError: If ffmpeg exits with a non-zero status.
    """
    if output_wav is None:
        base, _ = os.path.splitext(video_path)
        output_wav = f"{base}_audio.wav"

    logger.info(f"Extracting audio: '{video_path}' → '{output_wav}'")
    try:
        (
            ffmpeg
            .input(video_path)
            .output(
                output_wav,
                vn=None,          # no video
                ar=16000,         # sample rate: 16 kHz (Whisper optimum)
                ac=1,             # mono
                acodec="pcm_s16le",
            )
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as exc:
        raise RuntimeError(
            f"ffmpeg audio extraction failed: {exc.stderr.decode() if exc.stderr else str(exc)}"
        ) from exc

    logger.info(f"Audio extracted to '{output_wav}'.")
    return output_wav


def trim_audio(audio_path: str, duration_sec: float = 8.0, output_path: str | None = None) -> str:
    """
    Extract a high-SNR, normalized voice reference clip for zero-shot voice cloning.
    Skips the first second to avoid intro stingers/silence, filters rumble, and normalizes level.
    """
    if output_path is None:
        base, ext = os.path.splitext(audio_path)
        output_path = f"{base}_ref{ext}"

    logger.info(f"Extracting clean voice reference clip ({duration_sec}s) …")
    try:
        (
            ffmpeg
            .input(audio_path, ss=1.0, t=duration_sec)
            .filter('highpass', f=80)
            .filter('lowpass', f=8000)
            .filter('loudnorm')
            .output(output_path, acodec="pcm_s16le", ar=22050, ac=1)
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as exc:
        logger.warning(f"Filtered trim failed, falling back to basic trim: {exc}")
        (
            ffmpeg
            .input(audio_path, ss=0, t=duration_sec)
            .output(output_path, acodec="pcm_s16le", ar=22050, ac=1)
            .overwrite_output()
            .run(quiet=True)
        )

    logger.info(f"Voice reference clip saved to '{output_path}'.")
    return output_path



def get_duration(media_path: str) -> float:
    """Return the duration of a media file in seconds using ffprobe."""
    try:
        probe = ffmpeg.probe(media_path)
        return float(probe["format"]["duration"])
    except Exception as exc:
        logger.warning(f"Could not probe duration for '{media_path}': {exc}")
        return 0.0


def replace_audio(video_path: str, new_audio_path: str, output_path: str | None = None) -> str:
    """
    Replace the audio track of video_path with new_audio_path.
    - Video stream is COPIED (no re-encode) for speed.
    - Audio stream is re-encoded to AAC for broad compatibility.
    - If the new audio is shorter than the video, the video is trimmed to match.
    - If the new audio is longer, it is trimmed to the original video duration.

    Args:
        video_path:      Source video file (original).
        new_audio_path:  WAV file with the dubbed audio.
        output_path:     Destination for the output .mp4.
                         If None, saved in OUTPUTS_DIR.

    Returns:
        Path to the remuxed .mp4 file.

    Raises:
        RuntimeError: If ffmpeg exits with a non-zero status.
    """
    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(OUTPUTS_DIR, f"{base}_dubbed.mp4")

    video_dur = get_duration(video_path)
    audio_dur = get_duration(new_audio_path)

    # Use the shorter of the two durations so output is always playable
    use_dur = min(video_dur, audio_dur) if video_dur > 0 and audio_dur > 0 else None

    logger.info(
        f"Remuxing video (dur={video_dur:.1f}s) + dubbed audio (dur={audio_dur:.1f}s) "
        f"→ '{output_path}' (use_dur={use_dur:.1f}s if applicable)"
        if use_dur else f"→ '{output_path}'"
    )

    try:
        video_in = ffmpeg.input(video_path)
        audio_in = ffmpeg.input(new_audio_path)

        kwargs = {
            "vcodec": "copy",
            "acodec": "aac",
            "strict": "experimental",
        }
        if use_dur:
            kwargs["t"] = use_dur

        (
            ffmpeg
            .output(
                video_in.video,
                audio_in.audio,
                output_path,
                **kwargs,
            )
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as exc:
        raise RuntimeError(
            f"ffmpeg remux failed: {exc.stderr.decode() if exc.stderr else str(exc)}"
        ) from exc

    logger.info(f"Remux complete: '{output_path}'.")
    return output_path
