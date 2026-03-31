"""XTTS v2 TTS engine with MPS (Apple Metal) acceleration.

Uses Coqui TTS with the multilingual XTTS v2 model. Supports optional
voice cloning from a WAV speaker sample.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class TTSEngine:
    """Wrapper around Coqui TTS XTTS v2 model.

    Attributes:
        device: PyTorch device string — "mps" on Apple Silicon, else "cpu".
        tts: The loaded TTS model instance.
        voice_sample: Path to a WAV file for voice cloning (optional).
        language: BCP-47 language code, default "ru" (Russian).
    """

    MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

    def __init__(
        self,
        voice_sample: Optional[str | Path] = None,
        language: str = "ru",
    ):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info("TTS device: %s", self.device)

        # Import here so the module is importable without TTS installed
        # (useful for testing parsers without the heavy TTS dependency)
        from TTS.api import TTS  # type: ignore[import-untyped]

        self.tts = TTS(self.MODEL_NAME).to(self.device)
        self.voice_sample = str(voice_sample) if voice_sample else None
        self.language = language

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(self, text: str, output_path: str | Path) -> None:
        """Synthesize *text* to a WAV file at *output_path*.

        If a voice sample was provided at construction time, voice cloning
        is used. Otherwise the model's default speaker is used.

        Args:
            text: The text to synthesize (a single chunk, ~200 words).
            output_path: Destination path for the output WAV file.
        """
        output_path = str(output_path)

        if self.voice_sample:
            self.tts.tts_to_file(
                text=text,
                speaker_wav=self.voice_sample,
                language=self.language,
                file_path=output_path,
            )
        else:
            # Use first available speaker when no voice sample is provided
            speaker = None
            if hasattr(self.tts, "speakers") and self.tts.speakers:
                speaker = self.tts.speakers[0]

            self.tts.tts_to_file(
                text=text,
                speaker=speaker,
                language=self.language,
                file_path=output_path,
            )

    def set_voice(self, voice_sample: str | Path) -> None:
        """Update the voice sample used for cloning."""
        self.voice_sample = str(voice_sample)

    def set_language(self, language: str) -> None:
        """Update the synthesis language."""
        self.language = language
