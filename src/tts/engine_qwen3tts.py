"""Qwen3-TTS engine via mlx-audio — multilingual TTS with Russian support.

Uses Alibaba's Qwen3-TTS through mlx-audio for native Apple Silicon acceleration.
Three model variants:
  - base-0.6b   — lightweight, voice cloning, 10 languages incl. Russian
  - custom-1.7b — predefined speakers with emotion control
  - design-1.7b — voice design from text description

Russian stress pipeline (src.text.stress):
  Overrides → OpenRussian dict (397k) → pymorphy3 lemma → silero-stress
  Stress marks are converted to Unicode combining acute (U+0301) which
  Qwen3-TTS, being an LLM, can interpret as standard Russian diacritics.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

from src.tts.base import BaseTTSEngine, SAMPLE_RATE

logger = logging.getLogger(__name__)

_VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"


def _plus_to_acute(text: str) -> str:
    """Convert '+' stress marks to Unicode combining acute accent (U+0301).

    з+амок → за́мок  (acute placed after the stressed vowel)

    Qwen3-TTS is LLM-based and understands standard Russian diacritics
    used in dictionaries/textbooks, but not the '+' TTS convention.
    """
    result = []
    i = 0
    while i < len(text):
        if text[i] == "+" and i + 1 < len(text) and text[i + 1] in _VOWELS:
            result.append(text[i + 1])   # stressed vowel
            result.append("\u0301")       # combining acute accent
            i += 2
        else:
            result.append(text[i])
            i += 1
    return "".join(result)

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
        speech_speed: float = 0.9,
        model_variant: str = DEFAULT_VARIANT,
        speaker: str = "",
        instruct: str = "",
        **kwargs,
    ):
        # Pass speech_speed=1.0 to base — we handle speed natively via model param.
        super().__init__(
            language=language,
            speech_speed=1.0,
            max_chunk_chars=self.MAX_CHUNK_CHARS,
            **kwargs,
        )
        # Native speed passed directly to model.generate(speed=...)
        self._native_speed = speech_speed

        # Base class only restores ё (no '+' marks — those confuse Qwen3's tokenizer).
        # Full stress pipeline runs in _synthesize_chunk and converts to Unicode acute.
        self.accent_mode = "yo_only"

        raw_ref = str(ref_audio) if ref_audio and Path(ref_audio).exists() else None
        self.ref_audio = self._normalize_ref_audio(raw_ref) if raw_ref else None
        self.ref_text = ref_text
        self.model_variant = model_variant
        self.speaker = speaker
        self.instruct = instruct

        model_id = _MODEL_VARIANTS.get(model_variant, _MODEL_VARIANTS[DEFAULT_VARIANT])
        logger.info("Loading Qwen3-TTS (%s) from %s…", model_variant, model_id)

        from mlx_audio.tts.utils import load_model
        self._model = load_model(model_id)

        logger.info("Qwen3-TTS loaded (variant=%s, speed=%.2f)", model_variant, self._native_speed)

    @property
    def engine_name(self) -> str:
        return f"Qwen3-TTS-{self.model_variant}"

    @property
    def supports_voice_cloning(self) -> bool:
        return self.model_variant.startswith("base")

    @staticmethod
    def _normalize_ref_audio(src: str, target_sr: int = 24000, target_sec: float = 10.0) -> str:
        """Resample reference to 24kHz and trim to 10s from the middle.

        Qwen3-TTS allocates memory proportional to the reference duration —
        passing a multi-hour file causes OOM (71GB+ allocation attempt).
        """
        import tempfile
        import soundfile as sf_local
        import numpy as np_local

        audio, sr = sf_local.read(src)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != target_sr:
            import torchaudio as ta
            import torch as th
            t = th.from_numpy(audio.astype(np.float32)).unsqueeze(0)
            audio = ta.transforms.Resample(sr, target_sr)(t).squeeze(0).numpy()
            sr = target_sr

        max_samples = int(target_sec * sr)
        if len(audio) > max_samples:
            start = (len(audio) - max_samples) // 2
            audio = audio[start: start + max_samples]

        tmp = tempfile.mktemp(suffix="_qwen_ref.wav", prefix="q3_")
        sf_local.write(tmp, audio.astype(np.float32), sr)
        logger.info("Qwen3 ref audio: normalized to %.1fs at %dHz → %s", target_sec, sr, tmp)
        return tmp

    def _synthesize_chunk(self, text: str) -> np.ndarray:
        """Synthesize a single chunk via mlx-audio Qwen3-TTS.

        Qwen3-TTS is an LLM trained on clean text — explicit stress marks
        (Unicode acute or '+') degrade quality. Pass plain text; the model
        handles prosody internally.
        Text arrives with ё restored by the base class (accent_mode="yo_only").
        """
        qwen_text = text

        try:
            if self.model_variant.startswith("custom"):
                results = list(self._model.generate_custom_voice(
                    text=qwen_text,
                    speaker=self.speaker or "Serena",
                    language=self._qwen_language(),
                    instruct=self.instruct or "",
                    speed=self._native_speed,
                ))
            elif self.model_variant.startswith("design"):
                results = list(self._model.generate_voice_design(
                    text=qwen_text,
                    language=self._qwen_language(),
                    instruct=self.instruct or "A calm adult male voice with clear pronunciation.",
                    speed=self._native_speed,
                ))
            else:
                # Base model — voice cloning via ref_audio
                gen_kwargs: dict = {
                    "text": qwen_text,
                    "lang_code": self._qwen_language().lower(),
                    "speed": self._native_speed,
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
