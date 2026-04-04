"""Audiobook mastering pipeline — spoken-word quality via ffmpeg.

Single-pass approach for reliability:
  1. loudnorm to -19 LUFS (Audible ACX standard)
  2. Gentle 2:1 compression
  3. Presence boost at 4.5kHz (+2dB)

Output is validated — if peak is near zero the original is kept as fallback.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def _ffmpeg_env() -> dict:
    env = os.environ.copy()
    for p in ("/opt/homebrew/bin", "/usr/local/bin"):
        if p not in env.get("PATH", ""):
            env["PATH"] = p + ":" + env.get("PATH", "")
    return env


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, env=_ffmpeg_env())
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-1000:]}")


def _peak(path: str) -> float:
    try:
        audio, _ = sf.read(path)
        return float(np.max(np.abs(audio)))
    except Exception:
        return 0.0


class MasteringPipeline:
    """Spoken-word audio mastering for audiobook output.

    Args:
        target_lufs: Loudness target in LUFS (default -19, ACX standard).
        true_peak: True-peak ceiling in dBTP (default -1.5).
        lra: Loudness range (default 11 — more natural for spoken word).
        comp_threshold: Compressor threshold dBFS (default -20).
        comp_ratio: Compression ratio (default 2).
        presence_freq: Presence boost center frequency Hz (default 4500).
        presence_gain: Presence boost dB (default 2).
        sample_rate: Output sample rate Hz (default 24000).
    """

    def __init__(
        self,
        target_lufs: float = -19.0,
        true_peak: float = -1.5,
        lra: float = 11.0,
        comp_threshold: float = -20.0,
        comp_ratio: int = 2,
        presence_freq: int = 4500,
        presence_gain: float = 2.0,
        sample_rate: int = 24000,
    ):
        self.target_lufs = target_lufs
        self.true_peak = true_peak
        self.lra = lra
        self.comp_threshold = comp_threshold
        self.comp_ratio = comp_ratio
        self.presence_freq = presence_freq
        self.presence_gain = presence_gain
        self.sample_rate = sample_rate

    def _filter_chain(self) -> str:
        return ",".join([
            # Loudness normalization — single pass, no measurement step
            f"loudnorm=I={self.target_lufs}:TP={self.true_peak}:LRA={self.lra}",
            # Gentle compression
            (
                f"acompressor="
                f"threshold={self.comp_threshold}dB:"
                f"ratio={self.comp_ratio}:"
                f"attack=20:"
                f"release=250"
            ),
            # Presence boost for clarity
            (
                f"equalizer="
                f"f={self.presence_freq}:"
                f"t=q:"
                f"w=1.5:"
                f"g={self.presence_gain}"
            ),
        ])

    def master(self, input_path: str | Path, output_path: str | Path) -> None:
        """Apply mastering chain with silent-output guard.

        If the output is near-silent (peak < 0.01), the original file is
        copied to output_path with a warning instead of saving dead audio.
        """
        input_path = str(input_path)
        output_path = str(output_path)

        tmp = tempfile.mktemp(suffix="_mastered.wav", prefix="master_")
        try:
            _run([
                "ffmpeg", "-y",
                "-i", input_path,
                "-af", self._filter_chain(),
                "-ar", str(self.sample_rate),
                "-ac", "1",
                tmp,
            ])

            peak = _peak(tmp)
            if peak < 0.01:
                logger.warning(
                    "Mastering produced near-silent output (peak=%.4f) — "
                    "using unprocessed audio instead", peak
                )
                shutil.copy2(input_path, output_path)
            else:
                shutil.move(tmp, output_path)
                logger.info(
                    "Mastering complete → %.1f LUFS target, peak=%.3f",
                    self.target_lufs, peak,
                )
        except Exception as e:
            logger.warning("Mastering failed (%s) — using unprocessed audio", e)
            shutil.copy2(input_path, output_path)
            try:
                Path(tmp).unlink()
            except OSError:
                pass
