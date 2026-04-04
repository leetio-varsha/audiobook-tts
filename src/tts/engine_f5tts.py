"""F5-TTS engine — high-quality Russian TTS with zero-shot voice cloning.

Uses Misha24-10/F5-TTS_RUSSIAN fine-tuned on 5000+ hours of Russian audio
including audiobook data (Sova RuAudiobooks).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torchaudio

from src.tts.base import BaseTTSEngine, SAMPLE_RATE

logger = logging.getLogger(__name__)

_HF_CKPT = "hf://Misha24-10/F5-TTS_RUSSIAN/F5TTS_v1_Base_v2/model_last_inference.safetensors"
_HF_VOCAB = "hf://Misha24-10/F5-TTS_RUSSIAN/F5TTS_v1_Base/vocab.txt"

# hotstone228 — trained on 100k hours, potentially better stress handling
_HF_CKPT_HOTSTONE = "hf://hotstone228/F5-TTS-Russian/model_last.safetensors"
_HF_VOCAB_HOTSTONE = "hf://hotstone228/F5-TTS-Russian/vocab.txt"

_DEFAULT_REF_TEXT = "Привет, это пример моего голоса для аудиокниги."
_mlx_whisper_model = None


def _force_torchaudio_soundfile_backend() -> None:
    """Force torchaudio I/O through the soundfile backend.

    torchaudio 2.6 may auto-select the ffmpeg/torchcodec path, which breaks in
    this environment after MLX-related dependency changes. F5-TTS only needs
    simple local WAV I/O, so soundfile is the safer backend here.
    """
    if getattr(torchaudio, "_audiobook_tts_soundfile_patched", False):
        return

    orig_load = torchaudio.load
    orig_info = torchaudio.info
    orig_save = torchaudio.save

    def _load(*args, **kwargs):
        kwargs.setdefault("backend", "soundfile")
        return orig_load(*args, **kwargs)

    def _info(*args, **kwargs):
        kwargs.setdefault("backend", "soundfile")
        return orig_info(*args, **kwargs)

    def _save(*args, **kwargs):
        kwargs.setdefault("backend", "soundfile")
        return orig_save(*args, **kwargs)

    torchaudio.load = _load
    torchaudio.info = _info
    torchaudio.save = _save
    torchaudio._audiobook_tts_soundfile_patched = True


class F5TTSEngine(BaseTTSEngine):
    """F5-TTS Russian engine with voice cloning from reference audio.

    Uses CPU for reliable inference (MPS causes tensor mismatches).
    """

    MAX_CHUNK_CHARS = 150

    def __init__(
        self,
        ref_audio: Optional[str | Path] = None,
        ref_text: str = "",
        language: str = "ru",
        speech_speed: float = 0.85,
        nfe_steps: int = 32,
        cfg_strength: float = 2.0,
        ckpt_file: str = _HF_CKPT,
        vocab_file: str = _HF_VOCAB,
        **kwargs,
    ):
        super().__init__(language=language, speech_speed=speech_speed, max_chunk_chars=self.MAX_CHUNK_CHARS, **kwargs)

        self.ref_text = ref_text
        self.nfe_steps = nfe_steps
        self.cfg_strength = cfg_strength

        import os, torch
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        # Ensure ffmpeg is on PATH for pydub (used by F5-TTS preprocess)
        for p in ("/opt/homebrew/bin", "/usr/local/bin"):
            if p not in os.environ.get("PATH", ""):
                os.environ["PATH"] = p + ":" + os.environ.get("PATH", "")
        torch.set_num_threads(os.cpu_count() or 12)
        _force_torchaudio_soundfile_backend()

        logger.info("Loading F5-TTS Russian model (nfe_steps=%d)…", self.nfe_steps)

        # Resolve hf:// paths to local cache
        if ckpt_file.startswith("hf://"):
            from huggingface_hub import hf_hub_download
            parts = ckpt_file.replace("hf://", "").split("/", 2)
            ckpt_file = hf_hub_download(f"{parts[0]}/{parts[1]}", parts[2])
        if vocab_file.startswith("hf://"):
            from huggingface_hub import hf_hub_download
            parts = vocab_file.replace("hf://", "").split("/", 2)
            vocab_file = hf_hub_download(f"{parts[0]}/{parts[1]}", parts[2])

        import torch
        from f5_tts.api import F5TTS

        self._device = "cpu"

        self._f5 = F5TTS(
            model="F5TTS_v1_Base",
            ckpt_file=ckpt_file,
            vocab_file=vocab_file,
            device="cpu",
        )
        logger.info("F5-TTS loaded on CPU (NFE=%d)", self.nfe_steps)

        # Prepare reference audio
        if ref_audio and Path(ref_audio).exists():
            self.ref_audio = str(Path(ref_audio))
        else:
            logger.info("No reference audio provided, generating default with Silero...")
            self.ref_audio = self._generate_default_ref()
            self.ref_text = _DEFAULT_REF_TEXT

        # Let F5-TTS handle its own preprocessing (trim, silence, transcription).
        # This is the original working path — F5-TTS knows best how to prepare
        # its own ref audio/text pair.
        self._preprocess_ref()
        logger.info("Using ref audio: %s (max_chars=%d)", self.ref_audio, self.max_chunk_chars)

    def _normalize_ref_audio(self, src: str) -> str:
        """Resample reference audio to 24kHz and trim to 8-12s.

        F5-TTS is sensitive to sample rate — reference audio at anything other
        than 24kHz causes pitch shift in the synthesized output (chipmunk effect).
        Also trim long files to 8-12s of clean speech from the middle to avoid
        silence/noise at the start.
        """
        import torchaudio as ta

        audio, sr = sf.read(src)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        target_sr = 24000
        target_dur_sec = 10.0  # ideal reference duration for F5-TTS

        # Resample if needed
        if sr != target_sr:
            logger.info("Resampling reference audio %dHz → %dHz", sr, target_sr)
            import torch
            t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
            resampler = ta.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            audio = resampler(t).squeeze(0).numpy()
            sr = target_sr

        # Trim to target_dur_sec from the middle of the file (avoids intros/outros)
        max_samples = int(target_dur_sec * sr)
        if len(audio) > max_samples:
            start = (len(audio) - max_samples) // 2
            audio = audio[start: start + max_samples]
            logger.info("Trimmed reference to %.1fs from middle", target_dur_sec)

        # Save normalized reference to temp file
        tmp = tempfile.mktemp(suffix="_ref24k.wav", prefix="f5_")
        sf.write(tmp, audio.astype(np.float32), sr)
        return tmp

    def _preprocess_ref(self) -> None:
        """Preprocess reference audio and compute dynamic max_chars.

        1. Resample to 24kHz + trim to 10s (our step — avoids SR mismatch)
        2. F5-TTS's own preprocessing: silence trim + transcription
        """
        from f5_tts.infer.utils_infer import preprocess_ref_audio_text

        # Step 1: normalize SR and duration before F5-TTS sees it
        normalized_ref = self._normalize_ref_audio(self.ref_audio)

        # Step 2: let F5-TTS handle silence trimming and auto-transcription
        self._ref_file, self.ref_text = preprocess_ref_audio_text(
            normalized_ref, self.ref_text, show_info=logger.info,
        )
        # Compute max_chars based on ref audio duration
        audio, sr = sf.read(self._ref_file)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        ref_duration = len(audio) / sr
        ref_text_len = len(self.ref_text.encode("utf-8"))
        available_duration = 22 - ref_duration
        if available_duration > 0 and ref_duration > 0:
            chars_per_sec = ref_text_len / ref_duration
            computed = int(chars_per_sec * available_duration * 0.85)
            self.max_chunk_chars = max(80, min(computed, 250))
        else:
            self.max_chunk_chars = 100
        logger.info("Dynamic max_chars=%d (ref: %.1fs, text: %d chars, available: %.1fs)",
                     self.max_chunk_chars, ref_duration, ref_text_len, available_duration)

    @property
    def engine_name(self) -> str:
        return "F5-TTS Russian"

    @property
    def supports_voice_cloning(self) -> bool:
        return True

    @staticmethod
    def _prepare_text_for_f5(text: str) -> str:
        """Prepare text for F5-TTS Russian.

        Keeps '+' stress marks — Misha24-10 vocab includes '+' and the model
        was trained with stress notation. Removing '+' causes wrong stress.
        Only strips combining Unicode acute (U+0301) to avoid double-marking
        if both formats are present simultaneously.
        """
        return text.replace("\u0301", "")

    def _synthesize_chunk(self, text: str) -> np.ndarray:
        """Synthesize text using F5-TTS.

        Uses self._f5.infer() — the original F5-TTS API path.
        """
        attempt_text = self._prepare_text_for_f5(text)

        for attempt in range(3):
            try:
                wav, sr, _ = self._f5.infer(
                    ref_file=self._ref_file,
                    ref_text=self.ref_text,
                    gen_text=attempt_text,
                    nfe_step=self.nfe_steps,
                    cfg_strength=self.cfg_strength,
                    sway_sampling_coef=-1.0,
                )
                if not isinstance(wav, np.ndarray):
                    wav = wav.cpu().numpy()
                if len(wav) < sr * 0.1:
                    logger.warning("F5-TTS output too short (%d samples), retrying", len(wav))
                    continue
                break
            except Exception as e:
                err_msg = str(e).lower()
                logger.warning("F5-TTS attempt %d failed: %s", attempt + 1, e)
                if "size" in err_msg or "dimension" in err_msg or "mismatch" in err_msg:
                    attempt_text = attempt_text[:len(attempt_text) * 2 // 3]
                    for sep in (". ", "! ", "? ", ", "):
                        idx = attempt_text.rfind(sep)
                        if idx > len(attempt_text) // 2:
                            attempt_text = attempt_text[:idx + 1]
                            break
                    logger.warning("Reducing to %d chars", len(attempt_text))
                    if not attempt_text.strip():
                        return np.zeros(SAMPLE_RATE, dtype=np.float32)
                else:
                    logger.error("F5-TTS synthesis failed: %s", e)
                    return np.zeros(SAMPLE_RATE, dtype=np.float32)
        else:
            logger.error("F5-TTS failed after 3 attempts, skipping chunk")
            return np.zeros(SAMPLE_RATE, dtype=np.float32)

        # Resample to 24kHz if needed
        if sr != SAMPLE_RATE:
            from scipy.signal import resample
            target_len = int(len(wav) * SAMPLE_RATE / sr)
            wav = resample(wav, target_len)

        return wav.astype(np.float32)

    def set_voice(self, ref_audio: str | Path, ref_text: str = "") -> None:
        """Set reference audio for voice cloning."""
        self.ref_audio = str(ref_audio)
        self.ref_text = ref_text
        self._preprocess_ref()

    @staticmethod
    def _generate_default_ref() -> str:
        """Generate a default reference audio using Silero (fallback)."""
        from src.tts.engine_silero import SileroEngine
        engine = SileroEngine(speaker="xenia", speech_speed=1.0)
        ref_path = Path(tempfile.mktemp(suffix=".wav", prefix="f5ref_"))
        engine.synthesize(_DEFAULT_REF_TEXT, ref_path)
        return str(ref_path)
