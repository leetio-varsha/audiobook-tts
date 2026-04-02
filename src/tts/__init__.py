"""TTS module: multi-engine support, text chunker, and Russian preprocessing."""

from src.tts.base import BaseTTSEngine, SAMPLE_RATE
from src.tts.engine import TTSEngine, create_engine, list_engines
from src.tts.chunker import TextChunker
from src.tts.accentuator import RussianAccentuator
from src.tts.normalizer import RussianTextNormalizer

__all__ = [
    "BaseTTSEngine",
    "TTSEngine",
    "create_engine",
    "list_engines",
    "TextChunker",
    "RussianAccentuator",
    "RussianTextNormalizer",
    "SAMPLE_RATE",
]
