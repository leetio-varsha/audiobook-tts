"""F5-TTS engine via f5-tts-mlx — native Apple Silicon acceleration.

Drop-in replacement for F5TTSEngine but runs on Metal via MLX instead of
CPU-only PyTorch. Same architecture, same quality, much faster on M-series.

Requires: pip install f5-tts-mlx
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from src.tts.base import BaseTTSEngine, SAMPLE_RATE

logger = logging.getLogger(__name__)

# MLX model variants
_MODEL_VARIANTS = {
    "russian": "ink-splatters/f5-tts-russian-mlx",
    "base": "lucasnewman/f5-tts-mlx",
    "base-q4": "alandao/f5-tts-mlx-4bit",
    "base-q8": "alandao/f5-tts-mlx-8bit",
}

DEFAULT_VARIANT = "russian"
_DEFAULT_REF_TEXT = "Привет, это пример моего голоса для аудиокниги."


class MLXF5TTSEngine(BaseTTSEngine):
    """F5-TTS via f5-tts-mlx with native Metal acceleration.

    Same architecture as F5TTSEngine but uses MLX instead of PyTorch.
    Runs on Metal GPU natively — no MPS instability, no CPU fallback.

    Args:
        ref_audio: Path to reference audio for voice cloning (5-10s WAV).
        ref_text: Transcript of the reference audio.
        language: Language code (default "ru").
        speech_speed: Speech speed multiplier.
        model_variant: One of "russian", "base", "base-q4", "base-q8".
        steps: ODE solver iterations (default 8, lower = faster).
        cfg_strength: Classifier-free guidance strength (default 2.0).
    """

    MAX_CHUNK_CHARS = 200

    def __init__(
        self,
        ref_audio: Optional[str | Path] = None,
        ref_text: str = "",
        language: str = "ru",
        speech_speed: float = 1.0,
        model_variant: str = DEFAULT_VARIANT,
        steps: int = 8,
        cfg_strength: float = 2.0,
        **kwargs,
    ):
        super().__init__(
            language=language,
            speech_speed=speech_speed,
            max_chunk_chars=self.MAX_CHUNK_CHARS,
            **kwargs,
        )

        self.ref_text = ref_text or _DEFAULT_REF_TEXT
        self.model_variant = model_variant
        self.steps = steps
        self.cfg_strength = cfg_strength

        model_id = _MODEL_VARIANTS.get(model_variant, _MODEL_VARIANTS[DEFAULT_VARIANT])
        self._model_name = model_id

        logger.info("Loading MLX F5-TTS (%s) from %s…", model_variant, model_id)

        # Validate f5-tts-mlx is installed
        try:
            import f5_tts_mlx  # noqa: F401
        except ImportError:
            raise ImportError(
                "f5-tts-mlx not installed. Run: pip install f5-tts-mlx"
            )

        # Fix repos with old naming convention (model.safetensors → model_v1.safetensors)
        self._ensure_model_compat(model_id)

        # Prepare reference audio — trim to 6-8s for optimal F5-TTS context
        if ref_audio and Path(ref_audio).exists():
            self.ref_audio = self._trim_ref_audio(str(Path(ref_audio)))
        else:
            logger.info("No reference audio provided, generating default with Silero...")
            self.ref_audio = self._generate_default_ref()
            self.ref_text = _DEFAULT_REF_TEXT

        # Compute dynamic max_chars from ref audio duration
        self._compute_max_chars()

        logger.info(
            "MLX F5-TTS ready (variant=%s, steps=%d, max_chars=%d)",
            model_variant, self.steps, self.max_chunk_chars,
        )

    def _compute_max_chars(self) -> None:
        """Compute max_chars based on reference audio duration."""
        try:
            data, sr = sf.read(self.ref_audio)
            ref_duration = len(data) / sr
            ref_text_len = len(self.ref_text.encode("utf-8"))
            # F5-TTS context window is ~22s total (ref + gen)
            available_duration = 22 - ref_duration
            if available_duration > 0 and ref_duration > 0:
                chars_per_sec = ref_text_len / ref_duration
                self.max_chunk_chars = max(80, int(chars_per_sec * available_duration * 0.85))
            else:
                self.max_chunk_chars = 100
            logger.info(
                "Dynamic max_chars=%d (ref: %.1fs, available: %.1fs)",
                self.max_chunk_chars, ref_duration, available_duration,
            )
        except Exception as e:
            logger.warning("Could not compute dynamic max_chars: %s", e)
            self.max_chunk_chars = 150

    @property
    def engine_name(self) -> str:
        return f"MLX-F5-TTS-{self.model_variant}"

    @property
    def supports_voice_cloning(self) -> bool:
        return True

    def _synthesize_chunk(self, text: str) -> np.ndarray:
        """Synthesize a single chunk via f5-tts-mlx.

        Uses generate() with output_path since it doesn't return audio directly.
        """
        import tempfile

        # Strip stress marks — F5-TTS handles prosody internally
        text = text.replace("+", "").replace("\u0301", "")

        for attempt in range(3):
            try:
                from f5_tts_mlx.generate import generate

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name

                # Estimate target duration from text length + ref audio ratio
                # to prevent runaway generation beyond F5-TTS context window.
                ref_data, ref_sr = sf.read(self.ref_audio)
                ref_dur = len(ref_data) / ref_sr
                ref_chars = len(self.ref_text.encode("utf-8"))
                if ref_chars > 0 and ref_dur > 0:
                    chars_per_sec = ref_chars / ref_dur
                    target_dur = min(len(text.encode("utf-8")) / chars_per_sec, 22 - ref_dur)
                    target_dur = max(target_dur, 2.0)  # at least 2s
                else:
                    target_dur = 12.0

                generate(
                    generation_text=text,
                    duration=target_dur,
                    model_name=self._model_name,
                    ref_audio_path=self.ref_audio,
                    ref_audio_text=self.ref_text,
                    steps=self.steps,
                    cfg_strength=self.cfg_strength,
                    sway_sampling_coef=-1.0,
                    speed=1.0,  # speed handled by BaseTTSEngine
                    output_path=tmp_path,
                )

                # Read generated audio from temp file
                audio, sr = sf.read(tmp_path)
                Path(tmp_path).unlink(missing_ok=True)

                audio = np.array(audio, dtype=np.float32)
                if audio.ndim > 1:
                    audio = audio.squeeze()

                if len(audio) < SAMPLE_RATE * 0.1:
                    logger.warning("MLX F5-TTS output too short (%d samples), retrying", len(audio))
                    continue

                return audio

            except Exception as e:
                err_msg = str(e).lower()
                logger.warning("MLX F5-TTS attempt %d failed: %s", attempt + 1, e)
                Path(tmp_path).unlink(missing_ok=True)
                if "size" in err_msg or "dimension" in err_msg or "shape" in err_msg:
                    # Reduce text for retry
                    text = text[:len(text) * 2 // 3]
                    for sep in (". ", "! ", "? ", ", "):
                        idx = text.rfind(sep)
                        if idx > len(text) // 2:
                            text = text[:idx + 1]
                            break
                    if not text.strip():
                        return np.zeros(SAMPLE_RATE, dtype=np.float32)
                else:
                    logger.error("MLX F5-TTS synthesis failed: %s", e)
                    return np.zeros(SAMPLE_RATE, dtype=np.float32)

        logger.error("MLX F5-TTS failed after 3 attempts, skipping chunk")
        return np.zeros(SAMPLE_RATE, dtype=np.float32)

    def set_voice(self, ref_audio: str | Path, ref_text: str = "") -> None:
        """Set reference audio for voice cloning."""
        self.ref_audio = str(ref_audio)
        self.ref_text = ref_text
        self._compute_max_chars()
        logger.info("MLX F5-TTS voice reference set: %s", self.ref_audio)

    @staticmethod
    def _trim_ref_audio(path: str, max_seconds: float = 7.0) -> str:
        """Trim reference audio to max_seconds from the middle.

        F5-TTS has a ~22s context window shared between ref + generated audio.
        Long refs (>8s) leave too little room for generation, causing empty output.
        Trim from the middle where speech is most likely clean.
        """
        data, sr = sf.read(path)
        duration = len(data) / sr
        if duration <= max_seconds:
            return path
        # Take from the middle
        mid = len(data) // 2
        half = int(sr * max_seconds / 2)
        trimmed = data[mid - half : mid + half]
        import tempfile
        out = tempfile.mktemp(suffix=".wav", prefix="f5ref_trim_")
        sf.write(out, trimmed, sr)
        logger.info("Trimmed ref audio: %.1fs → %.1fs", duration, max_seconds)
        return out

    @staticmethod
    def _ensure_model_compat(model_id: str) -> None:
        """Ensure model files match f5-tts-mlx expected naming.

        Some repos (e.g. ink-splatters/f5-tts-russian-mlx) use old naming
        (model.safetensors) while f5-tts-mlx v0.2.6+ expects model_v1.safetensors.
        Downloads missing files and creates symlinks as needed.
        """
        from huggingface_hub import snapshot_download, hf_hub_download

        # Download all available files first
        snap_dir = Path(snapshot_download(model_id))

        model_v1 = snap_dir / "model_v1.safetensors"
        model_old = snap_dir / "model.safetensors"

        if model_v1.exists():
            return  # Already compatible

        if not model_old.exists():
            logger.warning("No model weights found in %s", snap_dir)
            return

        # Create symlink: model_v1.safetensors → model.safetensors
        logger.info("Creating compat symlink: model_v1.safetensors → model.safetensors")
        model_v1.symlink_to(model_old.name)

        # Also handle quantized variants
        for bits in ("4b", "8b"):
            old_q = snap_dir / f"model_{bits}.safetensors"
            new_q = snap_dir / f"model_v1_{bits}.safetensors"
            if old_q.exists() and not new_q.exists():
                logger.info("Creating compat symlink: model_v1_%s.safetensors", bits)
                new_q.symlink_to(old_q.name)

        # duration_v2.safetensors is needed by f5-tts-mlx but missing from some repos.
        # Copy from the base repo if not present.
        duration_v2 = snap_dir / "duration_v2.safetensors"
        if not duration_v2.exists():
            logger.info("Downloading duration_v2.safetensors from base repo…")
            try:
                dur_path = hf_hub_download("lucasnewman/f5-tts-mlx", "duration_v2.safetensors")
                import shutil
                shutil.copy2(dur_path, str(duration_v2))
            except Exception as e:
                logger.warning("Could not fetch duration_v2.safetensors: %s", e)

    @staticmethod
    def _generate_default_ref() -> str:
        """Generate a default reference audio using Silero (fallback)."""
        import tempfile
        from src.tts.engine_silero import SileroEngine
        engine = SileroEngine(speaker="xenia", speech_speed=1.0)
        ref_path = Path(tempfile.mktemp(suffix=".wav", prefix="f5ref_mlx_"))
        engine.synthesize(_DEFAULT_REF_TEXT, ref_path)
        return str(ref_path)
