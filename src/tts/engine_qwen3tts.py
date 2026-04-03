"""Qwen3-TTS engine via mlx-audio — multilingual TTS with Russian support.

Uses Alibaba's Qwen3-TTS through mlx-audio for native Apple Silicon acceleration.
Three model variants:
  - base-0.6b   — lightweight, voice cloning, 10 languages incl. Russian
  - custom-1.7b — predefined speakers with emotion control
  - design-1.7b — voice design from text description
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.tts.base import BaseTTSEngine, SAMPLE_RATE

logger = logging.getLogger(__name__)

# Available MLX model variants
_MODEL_VARIANTS = {
    "base-0.6b": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
    "base-0.6b-q4": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
    "custom-1.7b": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
    "design-1.7b": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
}

DEFAULT_VARIANT = "base-0.6b"


class Qwen3TTSEngine(BaseTTSEngine):
    """Qwen3-TTS via mlx-audio with native Metal acceleration.

    Args:
        ref_audio: Path to reference audio for voice cloning (5-8s WAV recommended).
        ref_text: Optional transcript of the reference audio.
        language: Language code (default "ru").
        speech_speed: Speech speed multiplier.
        model_variant: One of the available variants:
            "base-0.6b"    — fast, voice cloning, Russian supported (default)
            "base-0.6b-q4" — quantized base, even less memory
            "custom-1.7b"  — predefined speakers + emotion control (en/zh only)
            "design-1.7b"  — voice from text description (en/zh only)
        speaker: Speaker name for custom-1.7b variant.
        instruct: Emotion/style instruction for custom-1.7b or voice description
            for design-1.7b.
    """

    MAX_CHUNK_CHARS = 500  # Qwen3-TTS handles longer chunks than Chatterbox

    def __init__(
        self,
        ref_audio: Optional[str | Path] = None,
        ref_text: str = "",
        language: str = "ru",
        speech_speed: float = 0.92,
        model_variant: str = DEFAULT_VARIANT,
        speaker: str = "",
        instruct: str = "",
        **kwargs,
    ):
        super().__init__(
            language=language,
            speech_speed=speech_speed,
            max_chunk_chars=self.MAX_CHUNK_CHARS,
            **kwargs,
        )
        # Qwen3-TTS doesn't understand '+' stress marks — only restore ё
        self.accent_mode = "yo_only"

        self.ref_audio = str(ref_audio) if ref_audio and Path(ref_audio).exists() else None
        self.ref_text = ref_text
        self.model_variant = model_variant
        self.speaker = speaker
        self.instruct = instruct

        model_id = _MODEL_VARIANTS.get(model_variant, _MODEL_VARIANTS[DEFAULT_VARIANT])
        logger.info("Loading Qwen3-TTS (%s) from %s…", model_variant, model_id)

        from mlx_audio.tts.utils import load_model
        self._model = load_model(model_id)

        logger.info("Qwen3-TTS loaded (variant=%s)", model_variant)

    @property
    def engine_name(self) -> str:
        return f"Qwen3-TTS-{self.model_variant}"

    @property
    def supports_voice_cloning(self) -> bool:
        return self.model_variant.startswith("base")

    def _synthesize_chunk(self, text: str) -> np.ndarray:
        """Synthesize a single chunk via mlx-audio Qwen3-TTS."""
        # accent_mode="yo_only" ensures text arrives with ё restored
        # but without '+' stress marks that confuse the tokenizer.
        try:
            if self.model_variant.startswith("custom"):
                results = list(self._model.generate_custom_voice(
                    text=text,
                    speaker=self.speaker or "Serena",
                    language=self._qwen_language(),
                    instruct=self.instruct or "",
                ))
            elif self.model_variant.startswith("design"):
                results = list(self._model.generate_voice_design(
                    text=text,
                    language=self._qwen_language(),
                    instruct=self.instruct or "A calm adult male voice with clear pronunciation.",
                ))
            else:
                # Base model — voice cloning via ref_audio
                gen_kwargs = {
                    "text": text,
                    "lang_code": self._qwen_language().lower(),  # e.g. "russian", not "auto"
                }
                if self.ref_audio:
                    gen_kwargs["ref_audio"] = self.ref_audio
                if self.ref_text:
                    gen_kwargs["ref_text"] = self.ref_text
                results = list(self._model.generate(**gen_kwargs))

            if not results:
                logger.warning("Qwen3-TTS returned empty results for chunk")
                return np.zeros(SAMPLE_RATE, dtype=np.float32)

            # mx.array → numpy
            audio = np.array(results[0].audio, dtype=np.float32)

            if audio.ndim > 1:
                audio = audio.squeeze()

            return audio

        except Exception as e:
            logger.error("Qwen3-TTS synthesis failed: %s", e)
            return np.zeros(SAMPLE_RATE, dtype=np.float32)

    def _qwen_language(self) -> str:
        """Map language code to Qwen3-TTS language name."""
        lang_map = {
            "ru": "Russian",
            "en": "English",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "de": "German",
            "fr": "French",
            "pt": "Portuguese",
            "es": "Spanish",
            "it": "Italian",
        }
        return lang_map.get(self.language, "Russian")

    def set_voice(self, ref_audio: str | Path, ref_text: str = "") -> None:
        """Set reference audio for voice cloning (base variants only)."""
        self.ref_audio = str(ref_audio)
        self.ref_text = ref_text
        logger.info("Qwen3-TTS voice reference set: %s", self.ref_audio)
