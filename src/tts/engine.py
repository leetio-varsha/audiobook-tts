"""TTS engine factory — backward-compatible entry point.

Usage:
    from src.tts.engine import create_engine, TTSEngine

    # New way: choose engine by name
    engine = create_engine("silero", speaker="xenia", speech_speed=0.92)
    engine = create_engine("f5tts", ref_audio="voice.wav")

    # Old way: TTSEngine is an alias for SileroEngine (backward compat)
    engine = TTSEngine(speaker="xenia")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.tts.base import BaseTTSEngine

logger = logging.getLogger(__name__)

# Engine registry
ENGINES = {
    "silero": {
        "name": "Silero v5 (Russian, fast)",
        "class": "src.tts.engine_silero.SileroEngine",
        "languages": ["ru"],
        "voice_cloning": False,
    },
    "f5tts": {
        "name": "F5-TTS Russian (voice cloning)",
        "class": "src.tts.engine_f5tts.F5TTSEngine",
        "languages": ["ru"],
        "voice_cloning": True,
    },
    "chatterbox": {
        "name": "Chatterbox (voice cloning, MPS)",
        "class": "src.tts.engine_chatterbox.ChatterboxEngine",
        "languages": ["ru", "en", "de", "fr", "es", "it", "pt", "ja", "ko", "zh", "pl", "nl"],
        "voice_cloning": True,
    },
}


def create_engine(engine_type: str = "silero", **kwargs) -> BaseTTSEngine:
    """Create a TTS engine by type name.

    Args:
        engine_type: Engine key from ENGINES registry ("silero", "f5tts").
        **kwargs: Engine-specific arguments.

    Returns:
        Configured BaseTTSEngine instance.
    """
    if engine_type not in ENGINES:
        raise ValueError(f"Unknown engine '{engine_type}'. Available: {list(ENGINES.keys())}")

    info = ENGINES[engine_type]
    module_path, class_name = info["class"].rsplit(".", 1)

    import importlib
    module = importlib.import_module(module_path)
    engine_cls = getattr(module, class_name)

    logger.info("Creating TTS engine: %s (%s)", engine_type, info["name"])
    return engine_cls(**kwargs)


def list_engines() -> dict[str, dict]:
    """Return the engine registry for UI display."""
    return ENGINES


# Backward compatibility: TTSEngine = SileroEngine
def TTSEngine(
    voice_sample: Optional[str | Path] = None,
    language: str = "ru",
    speaker: str = "xenia",
    use_accent: bool = True,
    accent_model_size: str = "turbo2",
    speech_speed: float = 0.92,
    **kwargs,
) -> BaseTTSEngine:
    """Backward-compatible factory — creates a SileroEngine."""
    return create_engine(
        "silero",
        speaker=speaker,
        language=language,
        speech_speed=speech_speed,
        use_accent=use_accent,
        accent_model_size=accent_model_size,
        **kwargs,
    )
