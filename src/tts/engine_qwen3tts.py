"""Qwen3-TTS engine — cinematic audiobook pipeline for Russian.

Uses Alibaba's Qwen3-TTS via mlx-audio (native Apple Silicon / Metal acceleration).

Key design: DIRECT model calls only — no base-class text preprocessing that
degrades Qwen3's LLM-based prosody. The engine manages its own pipeline:

  semantic_chunk → classify_emotion → [warmup →] generate → pause_insert → master

Three model variants:
  base-0.6b   — voice cloning, 10 languages incl. Russian (default)
  custom-1.7b — predefined speakers with emotion control
  design-1.7b — voice design from text description

Batch optimization (base models with voice cloning):
  - Speaker embedding (ECAPA-TDNN) is extracted ONCE at init and cached.
    Without caching: ~30-50ms × N chunks wasted on redundant mel+encoder passes.
  - _prepare_generation_inputs is patched to auto-inject ref_audio, enabling
    batch_generate() to work with voice cloning (upstream only supports speaker names).
  - Chunks are grouped into batches for batched forward passes through the
    transformer — better Metal GPU utilization than sequential generate() calls.
  - Batch size is configurable (default 4). Smaller batches waste less on padding
    when chunk lengths vary; larger batches improve GPU throughput.

Performance notes (Apple M4 Max, 64GB, bf16 model):
  - First chunk without warmup is ~0.8-1.5s slower due to Metal kernel compilation.
    warmup=True (default) runs a 2-word dummy pass after load to amortize this.
  - Ref audio preloaded as mx.array — eliminates per-chunk disk reads.
  - All chunk audio is kept as numpy arrays in memory — no per-chunk disk I/O.
  - One ffmpeg call at the very end for mastering. Skip with fast_mode=True.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional
import tempfile

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
        warmup: Run a short dummy inference after load to warm Metal kernels
            (default True). Adds ~2-3s to startup, saves ~1s on first chunk.
        fast_mode: Skip mastering — useful during development/testing.
        batch_size: Number of chunks per batch_generate() call (default 8).
            Set to 1 to disable batching and use sequential generation.
            Benchmarked on M4 Max: bs=1 RTF=0.46x, bs=4 RTF=0.29x, bs=8 RTF=0.16x.
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
        warmup: bool = True,
        fast_mode: bool = False,
        batch_size: int = 8,
        **kwargs,
    ):
        # Disable base-class accent/normalizer — Qwen3 handles prosody itself
        super().__init__(
            language=language,
            speech_speed=1.0,       # disable ffmpeg atempo in base
            use_accent=False,       # no stress mark injection
            max_chunk_chars=self.MAX_CHUNK_CHARS,
            **kwargs,
        )

        self._native_speed = speech_speed
        self.model_variant = model_variant
        self.speaker = speaker
        self.instruct = instruct
        self.fast_mode = fast_mode
        self.batch_size = max(1, batch_size)

        # Cached language string — computed once, not per-chunk
        self._lang = self._resolve_language(language)

        # Pipeline components
        self._chunker = SemanticChunker(max_chars=self.MAX_CHUNK_CHARS)
        self._classifier = EmotionClassifier(base_speed=speech_speed)
        self._mastering = (MasteringPipeline() if (mastering and not fast_mode) else None)

        # Prepare reference audio (trim to 10s, preload as mx.array — prevents
        # OOM on long files AND eliminates per-chunk disk reads)
        if ref_audio and Path(ref_audio).exists():
            self.ref_audio = self._load_ref_as_array(str(ref_audio))
        else:
            self.ref_audio = None
        self.ref_text = ref_text

        # Load model
        t0 = time.perf_counter()
        model_id = _MODEL_VARIANTS.get(model_variant, _MODEL_VARIANTS[DEFAULT_VARIANT])
        logger.info("Loading Qwen3-TTS (%s)…", model_variant)
        from mlx_audio.tts.utils import load_model
        self._model = load_model(model_id)
        logger.info("Qwen3-TTS loaded in %.1fs", time.perf_counter() - t0)

        # Cache speaker embedding + patch model for batch voice cloning
        self._speaker_embed = None
        if self.ref_audio is not None and model_variant.startswith("base"):
            self._cache_speaker_embedding()

        # Warmup — compile Metal kernels before first real chunk
        if warmup:
            self._run_warmup()

        logger.info("Qwen3-TTS ready (variant=%s, speed=%.2f, batch=%d, fast_mode=%s)",
                    model_variant, speech_speed, self.batch_size, fast_mode)

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def _run_warmup(self) -> None:
        """Run a short dummy inference to compile Metal kernels.

        Without this, the first real chunk is ~0.8-1.5s slower than
        subsequent ones due to on-demand Metal kernel compilation.
        """
        t0 = time.perf_counter()
        try:
            dummy = list(self._model.generate(
                text="Раз.",
                lang_code=self._lang.lower(),
                speed=self._native_speed,
                **({"ref_audio": self.ref_audio} if self.ref_audio is not None else {}),
            ))
            # Discard warmup output — we only care about Metal being warm
            _ = dummy
            logger.info("Warmup complete in %.2fs", time.perf_counter() - t0)
        except Exception as e:
            logger.warning("Warmup failed (non-fatal): %s", e)

    # ------------------------------------------------------------------
    # Speaker embedding cache + model patching for batch voice cloning
    # ------------------------------------------------------------------

    def _cache_speaker_embedding(self) -> None:
        """Pre-compute speaker embedding from ref_audio and patch model.

        Two patches applied:
        1. extract_speaker_embedding → returns cached mx.array (skips ECAPA-TDNN)
        2. _prepare_generation_inputs → auto-injects ref_audio when not provided,
           so batch_generate() (which doesn't pass ref_audio) gets voice cloning.
        """
        import mlx.core as mx

        t0 = time.perf_counter()
        self._speaker_embed = self._model.extract_speaker_embedding(self.ref_audio)
        mx.eval(self._speaker_embed)
        logger.info("Speaker embedding cached in %.3fs (shape=%s)",
                    time.perf_counter() - t0, self._speaker_embed.shape)

        # Patch 1: skip ECAPA-TDNN on subsequent calls
        cached_embed = self._speaker_embed

        def _cached_extract(audio, sr=24000):
            return cached_embed

        self._model.extract_speaker_embedding = _cached_extract

        # Patch 2: inject ref_audio into _prepare_generation_inputs
        # so batch_generate → _prepare_batch_inputs → _prepare_generation_inputs
        # picks up voice cloning even though batch_generate doesn't pass ref_audio
        ref_audio = self.ref_audio
        original_prepare = self._model._prepare_generation_inputs

        def _patched_prepare(text, language="auto", speaker=None,
                             ref_audio=None, ref_text=None, instruct=None):
            if ref_audio is None and ref_audio_for_cloning is not None:
                ref_audio = ref_audio_for_cloning
            return original_prepare(
                text, language=language, speaker=speaker,
                ref_audio=ref_audio, ref_text=ref_text, instruct=instruct,
            )

        ref_audio_for_cloning = ref_audio
        self._model._prepare_generation_inputs = _patched_prepare

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
        """Cinematic audiobook pipeline with per-stage timing.

        semantic_chunk → emotion_classify → generate (batch or sequential)
        → pause_insert → master → save
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        t_total = time.perf_counter()

        if not text.strip():
            sf.write(str(output_path), np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
            return

        # Stage 1: semantic chunking
        t0 = time.perf_counter()
        chunks = self._chunker.chunk(text)
        t_chunk = time.perf_counter() - t0

        if not chunks:
            sf.write(str(output_path), np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
            return

        # Stage 1.5: classify emotions for all chunks (needed for pauses)
        emotions: list[EmotionResult] = []
        prev_emotion: Optional[EmotionResult] = None
        for chunk in chunks:
            emotion = self._classifier.classify(
                chunk.text,
                prev_emotion=prev_emotion,
                split_type=chunk.split_type,
            )
            emotions.append(emotion)
            prev_emotion = emotion

        # Stage 2: generate audio — batch or sequential
        use_batch = (
            self.batch_size > 1
            and self._speaker_embed is not None
            and self.model_variant.startswith("base")
            and len(chunks) > 1
        )

        t0 = time.perf_counter()
        if use_batch:
            logger.info("Synthesizing %d chunks in batches of %d…",
                        len(chunks), self.batch_size)
            audio_parts, chunk_times = self._generate_batch(chunks, emotions)
        else:
            logger.info("Synthesizing %d chunks sequentially…", len(chunks))
            audio_parts, chunk_times = self._generate_sequential(chunks, emotions)
        t_gen = time.perf_counter() - t0

        pause_parts = [e.pause_after_ms for e in emotions]

        # Stage 3: join with semantic pauses (in memory — no disk I/O)
        t0 = time.perf_counter()
        combined = self._join_with_pauses(audio_parts, pause_parts)
        t_join = time.perf_counter() - t0

        audio_dur = len(combined) / SAMPLE_RATE

        # Stage 4: mastering (one ffmpeg call at the very end)
        t0 = time.perf_counter()
        if self._mastering is not None:
            with tempfile.NamedTemporaryFile(
                suffix="_premaster.wav", prefix="qwen3_", delete=False
            ) as f:
                tmp_path = f.name
            try:
                sf.write(tmp_path, combined, SAMPLE_RATE)
                self._mastering.master(tmp_path, str(output_path))
            finally:
                try:
                    Path(tmp_path).unlink()
                except OSError:
                    pass
        else:
            sf.write(str(output_path), combined, SAMPLE_RATE)
        t_master = time.perf_counter() - t0

        # Timing summary
        t_wall = time.perf_counter() - t_total
        rtf = t_wall / audio_dur if audio_dur > 0 else 0
        mode = "batch" if use_batch else "seq"
        logger.info(
            "Synthesis complete (%s): %.1fs audio | wall=%.1fs RTF=%.2fx | "
            "chunk=%.2fs gen=%.1fs join=%.3fs master=%.2fs",
            mode, audio_dur, t_wall, rtf, t_chunk, t_gen, t_join, t_master,
        )
        if logger.isEnabledFor(logging.DEBUG):
            for i, (ct, ch) in enumerate(zip(chunk_times, chunks)):
                dur = len(audio_parts[i]) / SAMPLE_RATE
                logger.debug(
                    "  [%d] %.2fs → %.1fs audio (RTF %.2fx): %.45s",
                    i + 1, ct, dur, ct / max(dur, 0.01), ch.text,
                )

    # ------------------------------------------------------------------
    # Sequential generation (fallback for non-base models or batch_size=1)
    # ------------------------------------------------------------------

    def _generate_sequential(self, chunks, emotions) -> tuple[list[np.ndarray], list[float]]:
        """Generate audio for each chunk one at a time."""
        audio_parts: list[np.ndarray] = []
        chunk_times: list[float] = []

        for chunk, emotion in zip(chunks, emotions):
            logger.debug(
                "Chunk %d [%s/%s] speed=%.2f: %.50s",
                chunk.index + 1, emotion.category, chunk.split_type,
                emotion.speed, chunk.text,
            )
            tc = time.perf_counter()
            audio = self._generate_chunk(chunk.text, emotion)
            chunk_times.append(time.perf_counter() - tc)
            audio_parts.append(audio)

        return audio_parts, chunk_times

    def _generate_chunk(self, text: str, emotion: EmotionResult) -> np.ndarray:
        """Generate audio for one chunk via direct model call."""
        try:
            if self.model_variant.startswith("custom"):
                results = list(self._model.generate_custom_voice(
                    text=text,
                    speaker=self.speaker or "Serena",
                    language=self._lang,
                    instruct=self.instruct or emotion.style_hint,
                ))
            elif self.model_variant.startswith("design"):
                results = list(self._model.generate_voice_design(
                    text=text,
                    language=self._lang,
                    instruct=self.instruct or emotion.style_hint,
                ))
            else:
                # Base model: voice cloning + native speed control
                gen_kwargs: dict = {
                    "text": text,
                    "lang_code": self._lang.lower(),
                    "speed": emotion.speed,
                }
                if self.ref_audio is not None:
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
    # Batch generation (base models with voice cloning)
    # ------------------------------------------------------------------

    def _generate_batch(self, chunks, emotions) -> tuple[list[np.ndarray], list[float]]:
        """Generate audio for chunks in batches via batch_generate().

        Groups chunks into batches of self.batch_size, sorted by text length
        within each batch to minimize padding waste. Results are reassembled
        in original chunk order.
        """
        n = len(chunks)
        audio_parts: list[Optional[np.ndarray]] = [None] * n
        chunk_times: list[float] = [0.0] * n

        # Sort indices by text length for efficient batching (similar lengths
        # → less padding → less wasted computation)
        sorted_indices = sorted(range(n), key=lambda i: len(chunks[i].text))

        # Process in batches
        for batch_start in range(0, n, self.batch_size):
            batch_indices = sorted_indices[batch_start:batch_start + self.batch_size]
            batch_texts = [chunks[i].text for i in batch_indices]

            logger.debug(
                "Batch [%d-%d/%d]: %d texts, chars=%s",
                batch_start + 1, min(batch_start + self.batch_size, n), n,
                len(batch_texts),
                [len(t) for t in batch_texts],
            )

            tc = time.perf_counter()
            try:
                results = list(self._model.batch_generate(
                    texts=batch_texts,
                    lang_code=self._lang.lower(),
                    temperature=0.9,
                ))
                batch_time = time.perf_counter() - tc

                # Map results back to original indices by sequence_idx
                result_map: dict[int, np.ndarray] = {}
                for r in results:
                    audio = np.array(r.audio, dtype=np.float32)
                    if audio.ndim > 1:
                        audio = audio.squeeze()
                    result_map[r.sequence_idx] = audio

                per_chunk_time = batch_time / len(batch_indices)
                for pos, orig_idx in enumerate(batch_indices):
                    if pos in result_map:
                        audio_parts[orig_idx] = result_map[pos]
                    else:
                        logger.warning("Batch missing result for chunk %d", orig_idx + 1)
                        audio_parts[orig_idx] = np.zeros(
                            int(SAMPLE_RATE * 0.5), dtype=np.float32
                        )
                    chunk_times[orig_idx] = per_chunk_time

            except Exception as e:
                logger.warning("Batch generation failed, falling back to sequential: %s", e)
                batch_time = time.perf_counter() - tc
                # Fall back to sequential for this batch
                for orig_idx in batch_indices:
                    tc2 = time.perf_counter()
                    audio_parts[orig_idx] = self._generate_chunk(
                        chunks[orig_idx].text, emotions[orig_idx],
                    )
                    chunk_times[orig_idx] = time.perf_counter() - tc2

        # Replace any None entries with silence (shouldn't happen, but safety)
        for i in range(n):
            if audio_parts[i] is None:
                audio_parts[i] = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)

        return audio_parts, chunk_times

    # ------------------------------------------------------------------
    # Audio assembly
    # ------------------------------------------------------------------

    def _join_with_pauses(
        self, parts: list[np.ndarray], pause_ms: list[int]
    ) -> np.ndarray:
        """Concatenate audio parts with silence gaps — fully in memory."""
        if not parts:
            return np.zeros(0, dtype=np.float32)
        if len(parts) == 1:
            return parts[0]

        segments: list[np.ndarray] = []
        for i, audio in enumerate(parts):
            segments.append(audio)
            if i < len(parts) - 1 and i < len(pause_ms):
                ms = pause_ms[i]
                if ms > 0:
                    silence = np.zeros(int(SAMPLE_RATE * ms / 1000), dtype=np.float32)
                    segments.append(silence)

        return np.concatenate(segments)

    # ------------------------------------------------------------------
    # Reference audio normalization (OOM prevention)
    # ------------------------------------------------------------------

    def _load_ref_as_array(self, src: str):
        """Load reference audio as mx.array (trim to 10s, mono, 24kHz).

        Passing mx.array to model.generate() skips the per-chunk disk read that
        mlx-audio performs when given a file path — saves ~1.3s per chunk.
        OOM prevention: Qwen3 is an LLM; a 41k-second reference → 71GB malloc.
        """
        import mlx.core as mx
        from mlx_audio.utils import load_audio as mlx_load_audio

        audio, sr = sf.read(src)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        target_samples = int(10.0 * sr)
        if len(audio) > target_samples:
            start = (len(audio) - target_samples) // 2
            audio = audio[start: start + target_samples]
            logger.info("Trimmed reference to 10s from middle (source: %s)", src)

        # Write trimmed audio to a temp file so mlx_load_audio can resample
        # to 24kHz (required by the speaker encoder) — one-time cost at init
        with tempfile.NamedTemporaryFile(
            suffix="_ref_qwen3.wav", prefix="qwen3_", delete=False
        ) as f:
            tmp_path = f.name
        try:
            sf.write(tmp_path, audio.astype(np.float32), sr)
            ref_array = mlx_load_audio(tmp_path, sample_rate=24000)
        finally:
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass

        logger.info(
            "Ref audio preloaded as mx.array: %d samples @ 24kHz (source: %s)",
            ref_array.shape[0], src,
        )
        return ref_array

    # ------------------------------------------------------------------
    # Language mapping (resolved once at init)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_language(code: str) -> str:
        return {
            "ru": "Russian", "en": "English", "zh": "Chinese",
            "ja": "Japanese", "ko": "Korean", "de": "German",
            "fr": "French", "pt": "Portuguese", "es": "Spanish",
            "it": "Italian",
        }.get(code, "Russian")

    def _qwen_language(self) -> str:
        """Compatibility shim — use self._lang directly in new code."""
        return self._lang

    # ------------------------------------------------------------------
    # Public voice setter
    # ------------------------------------------------------------------

    def set_voice(self, ref_audio: str | Path, ref_text: str = "") -> None:
        """Update reference audio for voice cloning (base models only)."""
        self.ref_audio = self._load_ref_as_array(str(ref_audio))
        self.ref_text = ref_text
        # Re-cache speaker embedding for new voice
        if self.model_variant.startswith("base"):
            self._cache_speaker_embedding()
        logger.info("Voice reference updated from: %s", ref_audio)

    # ------------------------------------------------------------------
    # BaseTTSEngine abstract method — not used by our pipeline but
    # required by the interface for compatibility with other code
    # ------------------------------------------------------------------

    def _synthesize_chunk(self, text: str) -> np.ndarray:
        emotion = EmotionResult(
            category="narration",
            speed=self._native_speed,
            pause_after_ms=0,
            style_hint="calm narration",
        )
        return self._generate_chunk(text, emotion)
