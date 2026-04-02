"""Base TTS engine interface — all engines implement this contract.

The base class handles the shared pipeline:
  normalize → accentuate → split for length → synthesize → speed → pauses → crossfade → volume norm

Subclasses only implement _synthesize_chunk() — raw text-to-audio for a single chunk.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from src.tts.accentuator import RussianAccentuator
from src.tts.normalizer import RussianTextNormalizer

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000


class BaseTTSEngine(ABC):
    """Abstract base for all TTS engines.

    Subclasses must implement:
        _synthesize_chunk(text) -> np.ndarray  (float32, mono, at SAMPLE_RATE)
        engine_name -> str
    """

    # Punctuation-based pause durations (samples at 24kHz)
    _PAUSE_SAMPLES = {
        ",": int(SAMPLE_RATE * 0.08),
        ";": int(SAMPLE_RATE * 0.12),
        ":": int(SAMPLE_RATE * 0.10),
        "—": int(SAMPLE_RATE * 0.15),
        "–": int(SAMPLE_RATE * 0.15),
        "\n\n": int(SAMPLE_RATE * 0.40),
    }
    _CROSSFADE_SAMPLES = int(SAMPLE_RATE * 0.03)

    def __init__(
        self,
        language: str = "ru",
        speech_speed: float = 0.92,
        use_accent: bool = True,
        max_chunk_chars: int = 900,
        **kwargs,
    ):
        self.language = language
        self.speech_speed = speech_speed
        self.max_chunk_chars = max_chunk_chars

        # Shared preprocessors
        self._normalizer = RussianTextNormalizer()
        self._accentuator = RussianAccentuator()
        if use_accent and language == "ru":
            self._accentuator.load()

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Human-readable engine name for UI/logs."""
        ...

    @abstractmethod
    def _synthesize_chunk(self, text: str) -> np.ndarray:
        """Synthesize a single text chunk to audio.

        Args:
            text: Preprocessed text (already normalized + accented).

        Returns:
            Float32 numpy array, mono, at SAMPLE_RATE (24kHz).
        """
        ...

    def synthesize(self, text: str, output_path: str | Path) -> None:
        """Full synthesis pipeline: normalize → accent → split → synth → post-process → save."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Step 0: text normalization
        if self.language == "ru":
            text = self._normalizer.normalize(text)

        # Step 1: ruaccent preprocessing
        text = self._accentuator.process(text)

        # Step 2: split into sub-chunks if too long
        sub_chunks = self._split_for_engine(text)

        # Step 3: synthesize + post-process each sub-chunk
        audio_parts = []
        for chunk in sub_chunks:
            if not chunk.strip():
                continue
            audio = self._synthesize_chunk(chunk)

            # Speed adjustment
            if self.speech_speed != 1.0:
                audio = self._apply_speed(audio, self.speech_speed)

            # Punctuation pause
            pause = self._get_trailing_pause(chunk)
            if pause > 0:
                audio = np.concatenate([audio, np.zeros(pause, dtype=np.float32)])

            audio_parts.append(audio)

        if not audio_parts:
            audio_parts = [np.zeros(SAMPLE_RATE, dtype=np.float32)]

        # Step 4: crossfade join
        combined = self._crossfade_join(audio_parts)

        # Step 5: normalize volume
        combined = self._normalize_volume(combined)

        # Step 6: save
        sf.write(str(output_path), combined, SAMPLE_RATE)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _split_for_engine(self, text: str) -> list[str]:
        """Split text into sub-chunks respecting engine's character limit."""
        if len(text) <= self.max_chunk_chars:
            return [text]

        # First pass: split on sentence boundaries
        chunks = []
        current = ""
        for sent in text.replace(". ", ".\n").replace("! ", "!\n").replace("? ", "?\n").replace(", ", ",\n").split("\n"):
            if len(current) + len(sent) + 1 > self.max_chunk_chars and current:
                chunks.append(current.strip())
                current = sent
            else:
                current = current + " " + sent if current else sent
        if current.strip():
            chunks.append(current.strip())

        # Second pass: hard-split any chunks still over the limit
        result = []
        for chunk in chunks:
            while len(chunk) > self.max_chunk_chars:
                # Cut at last space before limit
                cut = chunk[:self.max_chunk_chars].rfind(" ")
                if cut < self.max_chunk_chars // 2:
                    cut = self.max_chunk_chars
                result.append(chunk[:cut].strip())
                chunk = chunk[cut:].strip()
            if chunk:
                result.append(chunk)

        return result or [text[:self.max_chunk_chars]]

    @staticmethod
    def _apply_speed(audio: np.ndarray, speed: float) -> np.ndarray:
        """Change playback speed WITHOUT changing pitch (time-stretch via ffmpeg atempo)."""
        if abs(speed - 1.0) < 0.01:
            return audio
        import subprocess, tempfile
        tmp_in = tempfile.mktemp(suffix=".wav")
        tmp_out = tempfile.mktemp(suffix=".wav")
        try:
            sf.write(tmp_in, audio, SAMPLE_RATE)
            # atempo range is [0.5, 100.0]; chain for values outside
            atempo_val = max(0.5, min(speed, 100.0))
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_in, "-af", f"atempo={atempo_val}",
                 "-ar", str(SAMPLE_RATE), tmp_out],
                capture_output=True, text=True,
            )
            result, _ = sf.read(tmp_out)
            return result.astype(np.float32)
        except Exception:
            # Fallback to simple resampling
            target_len = int(len(audio) / speed)
            indices = np.linspace(0, len(audio) - 1, target_len)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
        finally:
            import os
            for f in (tmp_in, tmp_out):
                try:
                    os.unlink(f)
                except OSError:
                    pass

    def _get_trailing_pause(self, text: str) -> int:
        text = text.rstrip()
        if text.endswith("\n\n"):
            return self._PAUSE_SAMPLES["\n\n"]
        for punct in ("—", "–", ";", ":", ","):
            if text.endswith(punct):
                return self._PAUSE_SAMPLES[punct]
        return 0

    def _crossfade_join(self, parts: list[np.ndarray]) -> np.ndarray:
        if len(parts) <= 1:
            return parts[0] if parts else np.zeros(0, dtype=np.float32)
        fade = self._CROSSFADE_SAMPLES
        result = parts[0]
        for part in parts[1:]:
            if len(result) < fade or len(part) < fade:
                result = np.concatenate([result, part])
                continue
            fade_out = np.linspace(1.0, 0.0, fade, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            result[-fade:] = result[-fade:] * fade_out + part[:fade] * fade_in
            result = np.concatenate([result, part[fade:]])
        return result

    @staticmethod
    def _normalize_volume(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
        peak = np.max(np.abs(audio))
        if peak < 1e-6:
            return audio
        target_peak = 10 ** (target_db / 20.0)
        return (audio * (target_peak / peak)).astype(np.float32)

    def set_language(self, language: str) -> None:
        self.language = language
