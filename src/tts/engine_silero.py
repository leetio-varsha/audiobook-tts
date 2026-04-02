"""Silero TTS v5 engine — fast, high-quality Russian synthesis with built-in speakers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

from src.tts.base import BaseTTSEngine, SAMPLE_RATE

logger = logging.getLogger(__name__)

DEFAULT_SPEAKER = "xenia"


class SileroEngine(BaseTTSEngine):
    """Silero TTS v5 engine with MPS acceleration.

    Speakers: aidar (M), baya (F), kseniya (F), xenia (F, audiobook).
    """

    KNOWN_SPEAKERS = {"aidar", "baya", "kseniya", "xenia", "eugene"}

    def __init__(
        self,
        speaker: str = DEFAULT_SPEAKER,
        language: str = "ru",
        speech_speed: float = 0.92,
        **kwargs,
    ):
        super().__init__(language=language, speech_speed=speech_speed, max_chunk_chars=900, **kwargs)

        self.speaker = speaker
        # Silero v5 packaged model only supports CPU (no MPS)
        self.device = torch.device("cpu")

        logger.info("Loading Silero TTS v5_ru…")
        try:
            from silero import silero_tts
            self._model, _ = silero_tts(language="ru", speaker="v5_ru")
        except ImportError:
            logger.info("silero pip package not found, using torch.hub...")
            saved_mods = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "src" or k.startswith("src.")}
            cwd = str(Path.cwd())
            saved_paths = [p for p in sys.path if p in ("", ".", cwd)]
            for p in saved_paths:
                sys.path.remove(p)
            try:
                self._model, _ = torch.hub.load(
                    repo_or_dir="snakers4/silero-models",
                    model="silero_tts",
                    language="ru",
                    speaker="v5_ru",
                    trust_repo=True,
                )
            finally:
                for k in list(sys.modules):
                    if k == "src" or k.startswith("src."):
                        sys.modules.pop(k, None)
                sys.modules.update(saved_mods)
                for p in saved_paths:
                    if p not in sys.path:
                        sys.path.insert(0, p)

        logger.info("Silero v5 loaded (CPU), speaker=%s", self.speaker)

    @property
    def engine_name(self) -> str:
        return "Silero v5"

    def _synthesize_chunk(self, text: str) -> np.ndarray:
        with torch.no_grad():
            audio = self._model.apply_tts(
                text=text,
                speaker=self.speaker,
                sample_rate=SAMPLE_RATE,
                put_accent=True,
                put_yo=True,
            )
        return audio.cpu().numpy()

    def set_voice(self, speaker: str) -> None:
        if speaker in self.KNOWN_SPEAKERS:
            self.speaker = speaker
            logger.info("Silero speaker changed to: %s", self.speaker)
        else:
            logger.warning("Unknown speaker '%s'. Available: %s", speaker, ", ".join(sorted(self.KNOWN_SPEAKERS)))
