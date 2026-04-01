"""TTS engine — Silero v4 (active) / XTTS v2 (commented out).

Silero is purpose-built for Russian and handles stress/accent placement
natively via put_accent=True + put_yo=True. It is faster and sounds more
natural for Russian than the generic XTTS v2 multilingual model.

Additional preprocessing via RussianAccentuator (ruaccent) can be enabled
to further improve accent placement for ambiguous omographs.

MPS (Apple Metal) acceleration is used automatically on Apple Silicon.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torchaudio

from src.tts.accentuator import RussianAccentuator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Silero speaker options for Russian (v4_ru model):
#   aidar    — male,   neutral
#   baya     — female, neutral
#   kseniya  — female, expressive
#   xenia    — female, clear / audiobook-style   ← default
#   eugene   — male,   expressive
# ---------------------------------------------------------------------------
DEFAULT_SPEAKER = "xenia"
SAMPLE_RATE = 24000


class TTSEngine:
    """Silero TTS v4 engine with MPS acceleration and ruaccent preprocessing.

    Args:
        voice_sample: Kept for API compatibility. In Silero mode you can
            also pass a speaker name (aidar/baya/kseniya/xenia/eugene)
            here instead of a WAV path.
        language: Language code. Silero v4_ru is Russian-only (``"ru"``).
        speaker: Silero built-in speaker name (default ``"xenia"``).
        use_accent: Enable ruaccent preprocessing for better stress marks.
            Requires ``pip install ruaccent``. Default True.
        accent_model_size: ruaccent model size: ``"small"`` / ``"medium"`` / ``"big"``.
    """

    _SILERO_REPO = "snakers4/silero-models"
    _SILERO_MODEL = "silero_tts"
    _SILERO_LANGUAGE = "ru"
    _SILERO_SPEAKER_MODEL = "v4_ru"

    _KNOWN_SPEAKERS = {"aidar", "baya", "kseniya", "xenia", "eugene"}

    def __init__(
        self,
        voice_sample: Optional[str | Path] = None,
        language: str = "ru",
        speaker: str = DEFAULT_SPEAKER,
        use_accent: bool = True,
        accent_model_size: str = "turbo2",
    ):
        # MPS on Apple Silicon; CPU fallback
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        logger.info("TTS device: %s", self.device)

        self.language = language
        self.speaker = speaker
        self.voice_sample = str(voice_sample) if voice_sample else None

        # --- Silero model ---------------------------------------------------
        logger.info("Loading Silero TTS model (v4_ru)…")
        self._model, _ = torch.hub.load(
            repo_or_dir=self._SILERO_REPO,
            model=self._SILERO_MODEL,
            language=self._SILERO_LANGUAGE,
            speaker=self._SILERO_SPEAKER_MODEL,
            trust_repo=True,
        )
        self._model.to(self.device)
        self._model.eval()
        logger.info("Silero loaded on %s, speaker=%s", self.device, self.speaker)

        # --- ruaccent -------------------------------------------------------
        self._accentuator = RussianAccentuator(model_size=accent_model_size)
        if use_accent:
            self._accentuator.load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(self, text: str, output_path: str | Path) -> None:
        """Synthesize *text* to a WAV file at *output_path*.

        Pipeline:
          1. Apply ruaccent stress marks (if available).
          2. Run Silero inference with put_accent=True, put_yo=True.
          3. Save as 24 kHz mono WAV via torchaudio.

        Args:
            text: A single text chunk (~200 words).
            output_path: Destination WAV file path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Step 1: ruaccent preprocessing
        text = self._accentuator.process(text)

        # Step 2: Silero inference
        with torch.no_grad():
            audio = self._model.apply_tts(
                text=text,
                speaker=self.speaker,
                sample_rate=SAMPLE_RATE,
                put_accent=True,   # Silero's built-in accent model
                put_yo=True,       # restore ё (prevents е/ё confusion)
            )

        # Step 3: save — audio is a 1-D CPU float32 tensor
        torchaudio.save(
            str(output_path),
            audio.unsqueeze(0).cpu(),
            SAMPLE_RATE,
        )

    def set_voice(self, voice_sample: str | Path) -> None:
        """Set speaker by name, or pass a WAV path (logs a warning for WAV).

        Silero does not support voice cloning. Pass a speaker name instead:
        aidar / baya / kseniya / xenia / eugene.
        """
        name = str(voice_sample)
        if name in self._KNOWN_SPEAKERS:
            self.speaker = name
            logger.info("Silero speaker changed to: %s", self.speaker)
        else:
            self.voice_sample = name
            logger.warning(
                "Voice cloning is not supported in Silero mode. "
                "To change the voice, pass a speaker name: %s",
                ", ".join(sorted(self._KNOWN_SPEAKERS)),
            )

    def set_language(self, language: str) -> None:
        """Update language code (Silero v4_ru supports Russian only)."""
        self.language = language
        if language != "ru":
            logger.warning(
                "Silero v4_ru is Russian-only. "
                "For other languages consider re-enabling XTTS v2."
            )


# =============================================================================
# XTTS v2 ENGINE — commented out, kept for reference / future re-enable
# =============================================================================
#
# To switch back to XTTS v2:
#   1. Rename this class to TTSEngine (replace Silero class above)
#   2. pip install TTS (Coqui)
#
# class XTTSEngine:
#     """Wrapper around Coqui TTS XTTS v2 model.
#
#     Attributes:
#         device: PyTorch device string — "mps" on Apple Silicon, else "cpu".
#         tts: The loaded TTS model instance.
#         voice_sample: Path to a WAV file for voice cloning (optional).
#         language: BCP-47 language code, default "ru" (Russian).
#     """
#
#     MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
#
#     def __init__(
#         self,
#         voice_sample: Optional[str | Path] = None,
#         language: str = "ru",
#     ):
#         import os
#         os.environ.setdefault("COQUI_TOS_AGREED", "1")
#
#         # Patch torch.load for Coqui TTS checkpoint compatibility
#         _orig = torch.load
#         torch.load = lambda *a, **kw: _orig(*a, **{**{"weights_only": False}, **kw})
#
#         self.device = "mps" if torch.backends.mps.is_available() else "cpu"
#         logger.info("TTS device: %s", self.device)
#
#         from TTS.api import TTS  # type: ignore[import-untyped]
#         self.tts = TTS(self.MODEL_NAME).to(self.device)
#         self.voice_sample = str(voice_sample) if voice_sample else None
#         self.language = language
#
#     def synthesize(self, text: str, output_path: str | Path) -> None:
#         output_path = str(output_path)
#         if self.voice_sample:
#             self.tts.tts_to_file(
#                 text=text,
#                 speaker_wav=self.voice_sample,
#                 language=self.language,
#                 file_path=output_path,
#             )
#         else:
#             sm = self.tts.synthesizer.tts_model.speaker_manager
#             speaker = list(sm.name_to_id)[0]
#             self.tts.tts_to_file(
#                 text=text,
#                 speaker=speaker,
#                 language=self.language,
#                 file_path=output_path,
#             )
#
#     def set_voice(self, voice_sample: str | Path) -> None:
#         self.voice_sample = str(voice_sample)
#
#     def set_language(self, language: str) -> None:
#         self.language = language
