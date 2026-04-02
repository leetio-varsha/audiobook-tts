"""Chatterbox TTS engine — multilingual voice cloning with MPS support.

Uses resemble-ai/chatterbox Multilingual model (550M params).
Russian supported via language_id="ru" + russian_text_stresser.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.tts.base import BaseTTSEngine, SAMPLE_RATE

logger = logging.getLogger(__name__)


class ChatterboxEngine(BaseTTSEngine):
    """Chatterbox Multilingual TTS with voice cloning on MPS.

    Args:
        ref_audio: Path to reference audio for voice cloning (5-15s recommended).
        language: Language code (default "ru").
        speech_speed: Speech speed multiplier.
        exaggeration: Expressiveness (0-1). Higher = more dramatic.
        cfg_weight: Text adherence (0-1). Higher = follows text more closely.
        temperature: Randomness (0-1). Lower = more stable pronunciation.
    """

    MAX_CHUNK_CHARS = 300

    def __init__(
        self,
        ref_audio: Optional[str | Path] = None,
        language: str = "ru",
        speech_speed: float = 1.0,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.7,
        temperature: float = 0.65,
        **kwargs,
    ):
        super().__init__(language=language, speech_speed=speech_speed, max_chunk_chars=self.MAX_CHUNK_CHARS, **kwargs)

        self.ref_audio = str(ref_audio) if ref_audio and Path(ref_audio).exists() else None
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.temperature = temperature

        # Language mapping
        self._lang_map = {
            "ru": "ru", "en": "en", "de": "de", "fr": "fr",
            "es": "es", "it": "it", "pt": "pt", "ja": "ja",
            "ko": "ko", "zh": "zh", "pl": "pl", "nl": "nl",
        }

        # Use MPS on Apple Silicon, fallback to CPU
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        logger.info("Loading Chatterbox Multilingual on %s…", device)
        from chatterbox import ChatterboxMultilingualTTS
        self._model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        self._sr = self._model.sr
        logger.info("Chatterbox loaded (SR=%d, device=%s)", self._sr, device)

    @property
    def engine_name(self) -> str:
        return "Chatterbox"

    @property
    def supports_voice_cloning(self) -> bool:
        return True

    def _synthesize_chunk(self, text: str) -> np.ndarray:
        """Synthesize a single chunk."""
        # Strip stress marks from our pipeline — Chatterbox has its own
        text = text.replace("+", "").replace("\u0301", "")

        lang_id = self._lang_map.get(self.language, "ru")

        try:
            wav = self._model.generate(
                text,
                language_id=lang_id,
                audio_prompt_path=self.ref_audio,
                exaggeration=self.exaggeration,
                cfg_weight=self.cfg_weight,
                temperature=self.temperature,
            )
        except Exception as e:
            logger.error("Chatterbox synthesis failed: %s", e)
            return np.zeros(SAMPLE_RATE, dtype=np.float32)

        # Convert to numpy
        if isinstance(wav, torch.Tensor):
            wav = wav.squeeze().cpu().numpy()

        # Resample if needed (Chatterbox outputs at 24kHz = our SAMPLE_RATE)
        if self._sr != SAMPLE_RATE:
            from scipy.signal import resample
            target_len = int(len(wav) * SAMPLE_RATE / self._sr)
            wav = resample(wav, target_len)

        return wav.astype(np.float32)

    def set_voice(self, ref_audio: str | Path, ref_text: str = "") -> None:
        """Set reference audio for voice cloning."""
        self.ref_audio = str(ref_audio)
        logger.info("Chatterbox voice reference set: %s", self.ref_audio)
