"""Audio preprocessing utilities for voice samples — normalize, trim, split."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a subprocess, raising on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Command failed: %s\nstderr: %s", " ".join(cmd), result.stderr)
        raise RuntimeError(f"ffmpeg error: {result.stderr.strip()}")
    return result


def normalize_audio(input_path: Path, output_path: Path, target_sr: int = 24000) -> None:
    """Convert audio to mono WAV at target sample rate with normalized volume."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y", "-i", str(input_path),
        "-ac", "1",
        "-ar", str(target_sr),
        "-filter:a", "loudnorm",
        str(output_path),
    ])
    logger.info("Normalized %s → %s (%d Hz mono)", input_path.name, output_path.name, target_sr)


def trim_silence(input_path: Path, output_path: Path, threshold_db: float = -40.0) -> None:
    """Remove leading and trailing silence from audio."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y", "-i", str(input_path),
        "-af", f"silenceremove=start_periods=1:start_silence=0.1:start_threshold={threshold_db}dB,"
               f"areverse,silenceremove=start_periods=1:start_silence=0.1:start_threshold={threshold_db}dB,areverse",
        str(output_path),
    ])


def get_duration(path: Path) -> float:
    """Get audio duration in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def split_audio(input_path: Path, output_dir: Path, segment_seconds: int = 10) -> list[Path]:
    """Split audio into segments of given duration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "segment_%03d.wav"
    _run([
        "ffmpeg", "-y", "-i", str(input_path),
        "-f", "segment", "-segment_time", str(segment_seconds),
        "-c", "copy",
        str(pattern),
    ])
    segments = sorted(output_dir.glob("segment_*.wav"))
    logger.info("Split %s into %d segments", input_path.name, len(segments))
    return segments
