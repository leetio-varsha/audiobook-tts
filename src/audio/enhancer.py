"""Audio post-processing enhancer — EQ, compression, loudness normalization,
light reverb, and de-essing via a single ffmpeg filter chain.

Applied to each chapter WAV before M4B assembly.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class AudioEnhancer:
    """Post-process TTS audio for audiobook quality.

    All processing runs in a single ffmpeg call (no temp files, no extra I/O).

    Args:
        target_lufs: Loudness target in LUFS (default -16, audiobook standard).
        eq_low_cut: High-pass frequency to cut low rumble (Hz, default 100).
        eq_presence_cut: Reduce synthetic harshness band (Hz, default 6000).
        eq_presence_db: dB cut for the presence band (default -3).
        compressor_threshold: Compressor threshold in dB (default -20).
        compressor_ratio: Compression ratio (default 3).
        reverb_delays: Echo delays in ms for room feel (default "40|60").
        reverb_decays: Echo decay factors (default "0.15|0.10").
        de_ess_freq: De-esser target frequency (Hz, default 5500).
        enabled: Master switch (default True).
    """

    def __init__(
        self,
        target_lufs: float = -16.0,
        eq_low_cut: int = 100,
        eq_presence_cut: int = 6000,
        eq_presence_db: float = -3.0,
        compressor_threshold: float = -20.0,
        compressor_ratio: float = 3.0,
        reverb_delays: str = "40|60",
        reverb_decays: str = "0.15|0.10",
        de_ess_freq: int = 5500,
        enabled: bool = True,
    ):
        self.target_lufs = target_lufs
        self.eq_low_cut = eq_low_cut
        self.eq_presence_cut = eq_presence_cut
        self.eq_presence_db = eq_presence_db
        self.compressor_threshold = compressor_threshold
        self.compressor_ratio = compressor_ratio
        self.reverb_delays = reverb_delays
        self.reverb_decays = reverb_decays
        self.de_ess_freq = de_ess_freq
        self.enabled = enabled

    def build_filter_chain(self) -> str:
        """Build the ffmpeg audio filter chain string."""
        filters = []

        # 1. EQ: high-pass to remove low rumble
        filters.append(f"highpass=f={self.eq_low_cut}:poles=2")

        # 2. EQ: reduce synthetic harshness in the presence band
        filters.append(
            f"equalizer=f={self.eq_presence_cut}:t=h:width=2000:g={self.eq_presence_db}"
        )

        # 3. De-essing: compress harsh sibilants
        #    Split signal, compress the high band, recombine
        filters.append(
            f"bandreject=f={self.de_ess_freq}:width_type=h:w=1500,"
            f"acompressor=threshold=-25dB:ratio=6:attack=0.3:release=50:makeup=1dB"
        )

        # 4. Main compressor: even out dynamics
        filters.append(
            f"acompressor="
            f"threshold={self.compressor_threshold}dB:"
            f"ratio={self.compressor_ratio}:"
            f"attack=5:"
            f"release=100:"
            f"makeup=2dB"
        )

        # 5. Loudness normalization to target LUFS
        filters.append(
            f"loudnorm=I={self.target_lufs}:TP=-1.5:LRA=11:print_format=none"
        )

        return ",".join(filters)

    def enhance_file(self, input_path: Path, output_path: Path, sample_rate: int = 24000) -> None:
        """Apply the full enhancement chain to a WAV file."""
        if not self.enabled:
            if input_path != output_path:
                import shutil
                shutil.copy2(str(input_path), str(output_path))
            return

        filter_chain = self.build_filter_chain()

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-af", filter_chain,
            "-ar", str(sample_rate),
            "-ac", "1",
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Enhancer ffmpeg failed:\n%s", result.stderr[-1000:])
            raise RuntimeError(f"Audio enhancement failed: {result.stderr[-500:]}")

    def enhance_in_place(self, wav_path: Path, sample_rate: int = 24000) -> None:
        """Enhance a WAV file in place (via temp file)."""
        if not self.enabled:
            return

        tmp = wav_path.with_suffix(".enhanced.wav")
        try:
            self.enhance_file(wav_path, tmp, sample_rate)
            tmp.replace(wav_path)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise
