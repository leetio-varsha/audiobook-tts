"""Chatterbox TTS engine via mlx-audio — optimized for Apple Silicon.

Uses MLX framework for native Metal acceleration instead of PyTorch MPS.
Two model variants:
  - turbo (134M, 1-step diffusion, fast, English-centric)
  - full  (550M, multi-step, multilingual incl. Russian)
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from src.tts.base import BaseTTSEngine, SAMPLE_RATE

logger = logging.getLogger(__name__)

# Available MLX model variants
_MODEL_VARIANTS = {
    "multi-q4": "YUGOROU/Chatterbox-Multilingual-MLX-4bit",
    "full-fp16": "mlx-community/chatterbox-fp16",
    "full-q4": "mlx-community/chatterbox-4bit",
    "full-q8": "mlx-community/chatterbox-8bit",
    "turbo": "mlx-community/chatterbox-turbo-fp16",
}

DEFAULT_VARIANT = "multi-q4"  # Multilingual model — best for Russian
_DEFAULT_REF_TEXT = "Привет. Это пример спокойного голоса для озвучки аудиокниги."


class MLXChatterboxEngine(BaseTTSEngine):
    """Chatterbox via mlx-audio with native Metal acceleration.

    Args:
        ref_audio: Path to reference audio for voice cloning (5-15s recommended).
        language: Language code (default "ru").
        speech_speed: Speech speed multiplier.
        model_variant: One of "turbo", "turbo-q4", "full-q4".
            turbo/turbo-q4 — fastest, 1-step diffusion, English-centric.
            full-q4 — multilingual (Russian supported), quantized for memory.
    """

    MAX_CHUNK_CHARS = 180

    def __init__(
        self,
        ref_audio: Optional[str | Path] = None,
        language: str = "ru",
        speech_speed: float = 0.85,
        model_variant: str = DEFAULT_VARIANT,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.7,
        temperature: float = 0.65,
        **kwargs,
    ):
        super().__init__(
            language=language,
            speech_speed=speech_speed,
            max_chunk_chars=self.MAX_CHUNK_CHARS,
            **kwargs,
        )

        if ref_audio and Path(ref_audio).exists():
            self.ref_audio = self._trim_ref_audio(str(Path(ref_audio)))
            self._using_default_ref = False
        else:
            logger.info("No reference audio provided, generating default voice prompt...")
            self.ref_audio = self._generate_default_ref()
            self._using_default_ref = True
        self.model_variant = model_variant
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.temperature = temperature

        model_id = _MODEL_VARIANTS.get(model_variant, _MODEL_VARIANTS[DEFAULT_VARIANT])
        logger.info("Loading MLX Chatterbox (%s) from %s…", model_variant, model_id)

        from mlx_audio.tts.utils import load_model
        self._model = load_model(model_id)

        # Fix: YUGOROU multilingual model lacks "multilingual": true in config,
        # so mlx-audio doesn't load the MTLTokenizer. Load it manually.
        if self._model.mtl_tokenizer is None and "multilingual" in model_id.lower():
            self._patch_multilingual_tokenizer(model_id)

        logger.info("MLX Chatterbox loaded (variant=%s, multilingual=%s)",
                     model_variant, self._model.mtl_tokenizer is not None)

    @property
    def engine_name(self) -> str:
        return f"MLX-Chatterbox-{self.model_variant}"

    @property
    def supports_voice_cloning(self) -> bool:
        return True

    _LANG_MAP = {"ru": "ru", "en": "en", "de": "de", "fr": "fr", "es": "es",
                  "it": "it", "pt": "pt", "ja": "ja", "ko": "ko", "zh": "zh",
                  "pl": "pl", "nl": "nl"}

    def _synthesize_chunk(self, text: str) -> np.ndarray:
        """Synthesize a single chunk via mlx-audio."""
        # Strip stress marks — Chatterbox has its own prosody
        text = text.replace("+", "").replace("\u0301", "")

        lang = self._LANG_MAP.get(self.language, "en")
        attempts = [
            {"max_new_tokens": 220, "temperature": self.temperature, "cfg_weight": self.cfg_weight},
            {"max_new_tokens": 160, "temperature": min(self.temperature, 0.45), "cfg_weight": max(self.cfg_weight, 0.9)},
            {"max_new_tokens": 120, "temperature": 0.35, "cfg_weight": 1.0},
        ]

        for idx, params in enumerate(attempts, 1):
            try:
                gen_kwargs = {
                    "text": text,
                    "lang_code": lang,
                    "exaggeration": self.exaggeration,
                    "cfg_weight": params["cfg_weight"],
                    "temperature": params["temperature"],
                    "max_new_tokens": params["max_new_tokens"],
                }

                if self.ref_audio:
                    gen_kwargs["ref_audio"] = self.ref_audio

                results = list(self._model.generate(**gen_kwargs))
                if not results:
                    logger.warning("MLX Chatterbox returned empty results for chunk")
                    continue

                audio = np.array(results[0].audio, dtype=np.float32)
                if audio.ndim > 1:
                    audio = audio.squeeze()
                audio = self._trim_generated_audio(audio)

                if self._is_bad_generation(audio, text):
                    logger.warning(
                        "MLX Chatterbox bad generation on attempt %d: dur=%.2fs words=%d, retrying",
                        idx, len(audio) / SAMPLE_RATE, len(text.split()),
                    )
                    continue

                return audio

            except Exception as e:
                logger.warning("MLX Chatterbox attempt %d failed: %s", idx, e)

        logger.error("MLX Chatterbox synthesis failed after retries")
        return np.zeros(SAMPLE_RATE, dtype=np.float32)

    @staticmethod
    def _is_bad_generation(audio: np.ndarray, text: str) -> bool:
        """Detect near-silent or runaway outputs.

        The broken MLX Chatterbox generations observed so far are typically:
        - capped at ~22s/44s with almost all samples near zero
        - much longer than expected for the input text
        """
        if len(audio) < int(SAMPLE_RATE * 0.3):
            return True

        duration = len(audio) / SAMPLE_RATE
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        non_silent_ratio = float(np.mean(np.abs(audio) > 1e-4))
        words = max(1, len(text.split()))
        expected_max_duration = min(18.0, 1.2 + words * 0.42)

        if rms < 0.02 or non_silent_ratio < 0.08:
            return True
        if duration > expected_max_duration * 1.8:
            return True
        return False

    @staticmethod
    def _trim_generated_audio(audio: np.ndarray) -> np.ndarray:
        """Trim leading/trailing silence from generated chunks.

        MLX Chatterbox often leaves long quiet tails around otherwise usable
        speech. Keep a small margin so chunk joins still sound natural.
        """
        if len(audio) == 0:
            return audio

        mask = np.abs(audio) > 0.003
        if not np.any(mask):
            return audio

        idx = np.flatnonzero(mask)
        pad = int(SAMPLE_RATE * 0.08)
        start = max(0, int(idx[0]) - pad)
        end = min(len(audio), int(idx[-1]) + pad + 1)
        trimmed = audio[start:end]
        return trimmed if len(trimmed) else audio

    def _patch_multilingual_tokenizer(self, model_id: str) -> None:
        """Load MTLTokenizer for multilingual models missing the config flag.

        YUGOROU/Chatterbox-Multilingual-MLX-4bit has version="multilingual-1.0"
        but lacks "multilingual": true in config.json, so mlx-audio skips loading
        the MTLTokenizer. We load it manually and attach to the model.
        """
        try:
            from huggingface_hub import snapshot_download
            from mlx_audio.tts.models.chatterbox.tokenizer import MTLTokenizer

            snap_dir = Path(snapshot_download(model_id))
            tokenizer_path = snap_dir / "tokenizer.json"
            if tokenizer_path.exists():
                self._model.mtl_tokenizer = MTLTokenizer(tokenizer_path)
                logger.info("Patched MTLTokenizer from %s", tokenizer_path)
            else:
                logger.warning("tokenizer.json not found in %s", snap_dir)
        except Exception as e:
            logger.warning("Failed to patch multilingual tokenizer: %s", e)

    def set_voice(self, ref_audio: str | Path, ref_text: str = "") -> None:
        """Set reference audio for voice cloning."""
        self.ref_audio = self._trim_ref_audio(str(ref_audio))
        self._using_default_ref = False
        logger.info("MLX Chatterbox voice reference set: %s", self.ref_audio)

    @staticmethod
    def _trim_ref_audio(path: str, max_seconds: float = 8.0) -> str:
        """Trim long reference audio from the middle.

        Keeps the conditioning clip compact and speech-focused.
        """
        data, sr = sf.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Trim outer silence first so the remaining clip is mostly speech.
        mask = np.abs(data) > 0.003
        if np.any(mask):
            idx = np.flatnonzero(mask)
            pad = int(sr * 0.12)
            start = max(0, int(idx[0]) - pad)
            end = min(len(data), int(idx[-1]) + pad + 1)
            data = data[start:end]

        duration = len(data) / sr
        if duration <= max_seconds:
            out = tempfile.mktemp(suffix=".wav", prefix="chatterbox_ref_")
            sf.write(out, data, sr)
            return out

        mid = len(data) // 2
        half = int(sr * max_seconds / 2)
        trimmed = data[mid - half : mid + half]
        out = tempfile.mktemp(suffix=".wav", prefix="chatterbox_ref_")
        sf.write(out, trimmed, sr)
        logger.info("Trimmed MLX Chatterbox ref audio: %.1fs -> %.1fs", duration, max_seconds)
        return out

    @staticmethod
    def _generate_default_ref() -> str:
        """Generate a fallback reference audio via Silero.

        MLX Chatterbox degrades badly without a conditioning prompt on some
        model variants; a short synthetic prompt is better than passing none.
        """
        from src.tts.engine_silero import SileroEngine

        engine = SileroEngine(speaker="xenia", speech_speed=1.0)
        ref_path = Path(tempfile.mktemp(suffix=".wav", prefix="mlx_chatterbox_ref_"))
        engine.synthesize(_DEFAULT_REF_TEXT, ref_path)
        return str(ref_path)
