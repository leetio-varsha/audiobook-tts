"""Russian text preprocessing: stress mark placement via silero-stress.

Adds stress marks ('+' before stressed vowel) and restores ё.
Used before TTS synthesis to improve pronunciation.

Falls back to ruaccent if silero-stress is not available.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class RussianAccentuator:
    """Wrapper around silero-stress (primary) or ruaccent (fallback).

    Usage::

        acc = RussianAccentuator()
        acc.load()
        text = acc.process("Замок был огромным")
        # → "За+мок был огро+мным"
    """

    def __init__(self, **kwargs):
        self._accentor = None
        self._backend = None
        self._available = False

    def load(self) -> bool:
        """Load stress model. Tries silero-stress first, then ruaccent."""
        # Try silero-stress (faster, 4M word dictionary)
        try:
            from silero_stress import load_accentor
            self._accentor = load_accentor(lang="ru")
            self._backend = "silero-stress"
            self._available = True
            logger.info("silero-stress loaded")
            return True
        except ImportError:
            logger.info("silero-stress not installed, trying ruaccent...")
        except Exception as exc:
            logger.warning("silero-stress failed: %s, trying ruaccent...", exc)

        # Fallback to ruaccent
        try:
            from ruaccent import RUAccent
            model = RUAccent()
            model.load(omograph_model_size="turbo2", use_dictionary=True)
            self._accentor = model
            self._backend = "ruaccent"
            self._available = True
            logger.info("ruaccent loaded (fallback)")
            return True
        except ImportError:
            logger.warning("No stress model available. Install: pip install silero-stress")
            return False
        except Exception as exc:
            logger.warning("ruaccent failed: %s", exc)
            return False

    def process(self, text: str) -> str:
        """Add stress marks to Russian text."""
        if not self._available or self._accentor is None:
            return text
        try:
            if self._backend == "silero-stress":
                return self._accentor(text)
            else:
                return self._accentor.process_all(text)
        except Exception as exc:
            logger.warning("Stress processing failed: %s", exc)
            return text

    @property
    def available(self) -> bool:
        return self._available
