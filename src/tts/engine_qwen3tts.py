"""Qwen3-TTS engine — cinematic audiobook pipeline for Russian.

Uses Alibaba's Qwen3-TTS via mlx-audio (native Apple Silicon acceleration).

Key design: DIRECT model calls only — no base-class text preprocessing that
degrades Qwen3's LLM-based prosody. The engine manages its own pipeline:

  semantic_chunk → classify_emotion → generate_direct → pause_insert → master

Three model variants:
  base-0.6b   — voice cloning, 10 languages incl. Russian (default)
  custom-1.7b — predefined speakers with emotion control
  design-1.7b — voice design from text description
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from src.audio.mastering import MasteringPipeline
from src.text.emotion_classifier import EmotionClassifier, EmotionResult
from src.text.semantic_chunker import SemanticChunker
from src.tts.base import BaseTTSEngine, SAMPLE_RATE

logger = logging.getLogger(__name__)

_MODEL_VARIANTS = {
    "base-0.6b":     "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
    "base-0.6b-q4":  "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
    "custom-1.7b":   "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
    "design-1.7b":   "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
}

DEFAULT_VARIANT = "base-0.6b"


class Qwen3TTSEngine(BaseTTSEngine):
    """Qwen3-TTS with cinematic audiobook pipeline.

    Bypasses BaseTTSEngine's text preprocessors — Qwen3 is an LLM that
    works best on clean, natural text. Prosody, pacing, and structure are
    controlled via semantic chunking, emotion classification, and native
    model speed parameter.

    Args:
        ref_audio: Path to 5-10s WAV reference for voice cloning (base models).
        ref_text: Optional transcript of the reference clip.
        language: Language code (default "ru").
        speech_speed: Base speed multiplier (default 0.85).
        model_variant: Model to use (default "base-0.6b").
        speaker: Speaker name for custom-1.7b.
        instruct: Style instruction for custom-1.7b / design-1.7b.
        mastering: Enable mastering pass (default True).
    """

    MAX_CHUNK_CHARS = 450

    def __init__(
        self,
        ref_audio: Optional[str | Path] = None,
        ref_text: str = "",
        language: str = "ru",
        speech_speed: float = 0.85,
        model_variant: str = DEFAULT_VARIANT,
        speaker: str = "",
        instruct: str = "",
        mastering: bool = True,
        **kwargs,
    ):
        # Disable base-class accent/normalizer — Qwen3 handles prosody itself
        super().__init__(
            language=language,
            speech_speed=1.0,        # disable ffmpeg atempo in base
            use_accent=False,        # no stress mark injection
            max_chunk_chars=self.MAX_CHUNK_CHARS,
            **kwargs,
        )

        self._native_speed = speech_speed
        self.model_variant = model_variant
        self.speaker = speaker
        self.instruct = instruct

        # Pipeline components
        self._chunker = SemanticChunker(max_chars=self.MAX_CHUNK_CHARS)
        self._classifier = EmotionClassifier(base_speed=speech_speed)
        self._mastering = MasteringPipeline() if mastering else None

        # Prepare reference audio (trim to 10s — prevents OOM on long files)
        if ref_audio and Path(ref_audio).exists():
            self.ref_audio = self._normalize_ref_audio(str(ref_audio))
        else:
            self.ref_audio = None
        self.ref_text = ref_text

        model_id = _MODEL_VARIANTS.get(model_variant, _MODEL_VARIANTS[DEFAULT_VARIANT])
        logger.info("Loading Qwen3-TTS (%s)…", model_variant)
        from mlx_audio.tts.utils import load_model
        self._model = load_model(model_id)
        logger.info("Qwen3-TTS ready (variant=%s, speed=%.2f)", model_variant, speech_speed)

    # ------------------------------------------------------------------
    # Public: engine identity
    # ------------------------------------------------------------------

    @property
    def engine_name(self) -> str:
        return f"Qwen3-TTS-{self.model_variant}"

    @property
    def supports_voice_cloning(self) -> bool:
        return self.model_variant.startswith("base")

    # ------------------------------------------------------------------
    # Main synthesis pipeline (overrides base class)
    # ------------------------------------------------------------------

    def synthesize(self, text: str, output_path: str | Path) -> None:
        """Cinematic audiobook pipeline.

        semantic_chunk → emotion_classify → generate_direct
        → pause_insert → crossfade → master → save
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not text.strip():
            sf.write(str(output_path), np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
            return

        # Step 1: semantic chunking
        chunks = self._chunker.chunk(text)
        if not chunks:
            sf.write(str(output_path), np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
            return

        logger.info("Synthesizing %d semantic chunks…", len(chunks))

        # Step 2: generate audio for each chunk
        audio_parts: list[np.ndarray] = []
        pause_parts: list[int] = []
        prev_emotion: Optional[EmotionResult] = None

        for chunk in chunks:
            emotion = self._classifier.classify(
                chunk.text,
                prev_emotion=prev_emotion,
                split_type=chunk.split_type,
            )
            logger.debug(
                "Chunk %d [%s/%s]: %.40s…  → %r",
                chunk.index + 1,
                chunk.split_type,
                emotion.category,
                chunk.text,
                emotion,
            )

            audio = self._generate_chunk(chunk.text, emotion)
            audio_parts.append(audio)
            pause_parts.append(emotion.pause_after_ms)
            prev_emotion = emotion

        # Step 3: join with semantic pauses (no crossfade — preserves natural starts)
        combined = self._join_with_pauses(audio_parts, pause_parts)

        # Step 4: mastering pass
        if self._mastering is not None:
            tmp = tempfile.mktemp(suffix="_premaster.wav", prefix="qwen3_")
            try:
                sf.write(tmp, combined, SAMPLE_RATE)
                self._mastering.master(tmp, str(output_path))
            finally:
                try:
                    Path(tmp).unlink()
                except OSError:
                    pass
        else:
            sf.write(str(output_path), combined, SAMPLE_RATE)

    # ------------------------------------------------------------------
    # Chunk-level generation
    # ------------------------------------------------------------------

    def _generate_chunk(self, text: str, emotion: EmotionResult) -> np.ndarray:
        """Generate audio for one chunk via direct model call."""
        try:
            if self.model_variant.startswith("custom"):
                results = list(self._model.generate_custom_voice(
                    text=text,
                    speaker=self.speaker or "Serena",
                    language=self._qwen_language(),
                    instruct=self.instruct or emotion.style_hint,
                ))
            elif self.model_variant.startswith("design"):
                results = list(self._model.generate_voice_design(
                    text=text,
                    language=self._qwen_language(),
                    instruct=self.instruct or emotion.style_hint,
                ))
            else:
                # Base model: voice cloning + native speed control
                gen_kwargs: dict = {
                    "text": text,
                    "lang_code": self._qwen_language().lower(),
                    "speed": emotion.speed,
                }
                if self.ref_audio:
                    gen_kwargs["ref_audio"] = self.ref_audio
                if self.ref_text:
                    gen_kwargs["ref_text"] = self.ref_text
                results = list(self._model.generate(**gen_kwargs))

            if not results:
                logger.warning("Qwen3-TTS returned empty result for: %.40s", text)
                return np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)

            audio = np.array(results[0].audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.squeeze()
            return audio

        except Exception as e:
            logger.error("Qwen3-TTS generation failed: %s", e)
            return np.zeros(SAMPLE_RATE, dtype=np.float32)

    # ------------------------------------------------------------------
    # Audio assembly
    # ------------------------------------------------------------------

    def _join_with_pauses(
        self, parts: list[np.ndarray], pause_ms: list[int]
    ) -> np.ndarray:
        """Concatenate audio parts with silence gaps between them.

        No crossfade — audiobook prose needs distinct breath groups.
        """
        if not parts:
            return np.zeros(0, dtype=np.float32)
        if len(parts) == 1:
            return parts[0]

        segments: list[np.ndarray] = []
        for i, audio in enumerate(parts):
            segments.append(audio)
            # Add pause after each chunk except the last
            if i < len(parts) - 1 and i < len(pause_ms):
                ms = pause_ms[i]
                if ms > 0:
                    silence = np.zeros(int(SAMPLE_RATE * ms / 1000), dtype=np.float32)
                    segments.append(silence)

        return np.concatenate(segments)

    # ------------------------------------------------------------------
    # Reference audio normalization (OOM prevention)
    # ------------------------------------------------------------------

    def _normalize_ref_audio(self, src: str) -> str:
        """Trim reference audio to 10s from the middle.

        Qwen3-TTS is an LLM — a 41k-second reference causes OOM.
        No resampling: Qwen3 handles sample rate internally.
        """
        audio, sr = sf.read(src)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        target_samples = int(10.0 * sr)
        if len(audio) > target_samples:
            start = (len(audio) - target_samples) // 2
            audio = audio[start: start + target_samples]
            logger.info("Trimmed reference to 10s from middle (source: %s)", src)

        tmp = tempfile.mktemp(suffix="_ref_qwen3.wav", prefix="qwen3_")
        sf.write(tmp, audio.astype(np.float32), sr)
        return tmp

    # ------------------------------------------------------------------
    # Language mapping
    # ------------------------------------------------------------------

    def _qwen_language(self) -> str:
        return {
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
        }.get(self.language, "Russian")

    # ------------------------------------------------------------------
    # Public voice setter
    # ------------------------------------------------------------------

    def set_voice(self, ref_audio: str | Path, ref_text: str = "") -> None:
        """Update reference audio for voice cloning (base models only)."""
        self.ref_audio = self._normalize_ref_audio(str(ref_audio))
        self.ref_text = ref_text
        logger.info("Voice reference updated: %s", self.ref_audio)

    # ------------------------------------------------------------------
    # BaseTTSEngine abstract method (not used by our pipeline,
    # but required by the interface for compatibility with other code)
    # ------------------------------------------------------------------

    def _synthesize_chunk(self, text: str) -> np.ndarray:
        """Single-chunk synthesis (called only when using base class pipeline)."""
        from src.text.emotion_classifier import EmotionResult
        emotion = EmotionResult(
            category="narration",
            speed=self._native_speed,
            pause_after_ms=0,
            style_hint="calm narration",
        )
        return self._generate_chunk(text, emotion)
