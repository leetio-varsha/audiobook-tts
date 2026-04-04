"""Audiobook mastering pipeline — spoken-word quality via ffmpeg.

Targets Audible/Storytel premium spoken-word quality:
  - Loudness: -19 LUFS mono (Audible ACX standard)
  - Gentle 2:1 compression (preserves dynamics, evens peaks)
  - Light de-esser (dip at ~7kHz, -2dB)
  - Presence boost (4.5kHz, +2dB) for clarity and intimacy
  - High-pass at 80Hz (remove room rumble)
  - True-peak ceiling: -1.5 dBTP

Two-pass loudnorm for accurate LUFS targeting.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _ffmpeg_env() -> dict:
    env = os.environ.copy()
    for p in ("/opt/homebrew/bin", "/usr/local/bin"):
        if p not in env.get("PATH", ""):
            env["PATH"] = p + ":" + env.get("PATH", "")
    return env


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True, env=_ffmpeg_env())
    if check and result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-1200:]}")
    return result


class MasteringPipeline:
    """Spoken-word audio mastering for audiobook output.

    Args:
        target_lufs: Loudness target in LUFS (default -19, ACX standard).
        true_peak: True-peak ceiling in dBTP (default -1.5).
        lra: Loudness range target (default 7 LU — good for spoken word).
        hp_freq: High-pass filter frequency in Hz (default 80).
        presence_freq: Presence boost frequency in Hz (default 4500).
        presence_gain: Presence boost in dB (default 2.0).
        deess_freq: De-esser notch frequency in Hz (default 7000).
        deess_gain: De-esser cut in dB (negative; default -2.5).
        comp_threshold: Compressor threshold in dBFS (default -20).
        comp_ratio: Compression ratio (default 2.0).
        sample_rate: Output sample rate in Hz (default 24000).
    """

    def __init__(
        self,
        target_lufs: float = -19.0,
        true_peak: float = -1.5,
        lra: float = 7.0,
        hp_freq: int = 80,
        presence_freq: int = 4500,
        presence_gain: float = 2.0,
        deess_freq: int = 7000,
        deess_gain: float = -2.5,
        comp_threshold: float = -20.0,
        comp_ratio: float = 2.0,
        sample_rate: int = 24000,
    ):
        self.target_lufs = target_lufs
        self.true_peak = true_peak
        self.lra = lra
        self.hp_freq = hp_freq
        self.presence_freq = presence_freq
        self.presence_gain = presence_gain
        self.deess_freq = deess_freq
        self.deess_gain = deess_gain
        self.comp_threshold = comp_threshold
        self.comp_ratio = comp_ratio
        self.sample_rate = sample_rate

    def _pre_filter_chain(self) -> str:
        """Build ffmpeg filter chain for all processing EXCEPT loudnorm."""
        return ",".join([
            # High-pass: remove low-frequency rumble
            f"highpass=f={self.hp_freq}:poles=2",

            # Gentle compression: even out dynamics without squashing
            (
                f"acompressor="
                f"threshold={self.comp_threshold}dB:"
                f"ratio={self.comp_ratio}:"
                f"attack=5:"
                f"release=100:"
                f"makeup=1"
            ),

            # De-esser: gentle notch around 7kHz to tame sibilance
            (
                f"equalizer="
                f"f={self.deess_freq}:"
                f"t=o:"
                f"w=2000:"
                f"g={self.deess_gain}"
            ),

            # Presence boost: add clarity and intelligibility around 4.5kHz
            (
                f"equalizer="
                f"f={self.presence_freq}:"
                f"t=o:"
                f"w=1500:"
                f"g={self.presence_gain}"
            ),
        ])

    def master(self, input_path: str | Path, output_path: str | Path) -> None:
        """Apply full mastering chain.

        Uses two-pass loudnorm for accurate LUFS targeting.
        Falls back to single-pass if measurement fails.
        """
        input_path = str(input_path)
        output_path = str(output_path)

        try:
            self._two_pass_master(input_path, output_path)
        except Exception as e:
            logger.warning("Two-pass loudnorm failed (%s), falling back to single-pass", e)
            self._single_pass_master(input_path, output_path)

    def _two_pass_master(self, input_path: str, output_path: str) -> None:
        """Accurate two-pass loudnorm mastering."""
        pre_chain = self._pre_filter_chain()

        # Pass 1: measure loudness stats
        measure_filter = (
            f"{pre_chain},"
            f"loudnorm=I={self.target_lufs}:TP={self.true_peak}:LRA={self.lra}"
            f":print_format=json"
        )
        result = _run([
            "ffmpeg", "-y", "-i", input_path,
            "-af", measure_filter,
            "-f", "null", "-",
        ], check=False)

        # Parse loudnorm stats from stderr
        stats = self._parse_loudnorm_stats(result.stderr)

        if not stats:
            raise ValueError("Could not parse loudnorm measurement stats")

        # Pass 2: apply with measured values for accurate normalization
        apply_filter = (
            f"{pre_chain},"
            f"loudnorm=I={self.target_lufs}:TP={self.true_peak}:LRA={self.lra}"
            f":measured_I={stats['input_i']}"
            f":measured_TP={stats['input_tp']}"
            f":measured_LRA={stats['input_lra']}"
            f":measured_thresh={stats['input_thresh']}"
            f":offset={stats['target_offset']}"
            f":linear=true"
        )
        _run([
            "ffmpeg", "-y", "-i", input_path,
            "-af", apply_filter,
            "-ar", str(self.sample_rate),
            "-ac", "1",
            output_path,
        ])
        logger.info(
            "Mastering complete: %.1f LUFS → %.1f LUFS, TP=%.1f dBTP",
            float(stats.get("input_i", 0)),
            self.target_lufs,
            self.true_peak,
        )

    def _single_pass_master(self, input_path: str, output_path: str) -> None:
        """Single-pass mastering (less accurate LUFS targeting)."""
        pre_chain = self._pre_filter_chain()
        full_chain = (
            f"{pre_chain},"
            f"loudnorm=I={self.target_lufs}:TP={self.true_peak}:LRA={self.lra}"
            f":print_format=none"
        )
        _run([
            "ffmpeg", "-y", "-i", input_path,
            "-af", full_chain,
            "-ar", str(self.sample_rate),
            "-ac", "1",
            output_path,
        ])
        logger.info("Single-pass mastering complete → %s", output_path)

    def _parse_loudnorm_stats(self, stderr: str) -> dict | None:
        """Extract loudnorm JSON from ffmpeg stderr.

        Returns None if stats are missing or contain unusable values (-inf).
        """
        try:
            start = stderr.rfind("{")
            end = stderr.rfind("}") + 1
            if start < 0 or end <= start:
                return None
            stats = json.loads(stderr[start:end])
            # Reject if any critical measurement is -inf (happens with near-silent audio)
            for key in ("input_i", "input_tp", "input_lra", "input_thresh"):
                val = str(stats.get(key, "")).lower()
                if "inf" in val or "nan" in val:
                    return None
            return stats
        except (json.JSONDecodeError, ValueError):
            return None
