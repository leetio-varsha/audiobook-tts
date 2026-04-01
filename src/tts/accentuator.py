"""Russian text preprocessing: stress mark placement via ruaccent.

Optionally used before TTS synthesis to improve pronunciation of
Russian text. Adds stress marks to stressed vowels, which helps the
TTS model produce correct accentuation.

Falls back gracefully if ruaccent is not installed.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class RussianAccentuator:
    """Wrapper around ruaccent for Russian stress mark placement.

    Usage::

        acc = RussianAccentuator()
        acc.load()
        text = acc.process("Замок был огромным")
        # → "За+мок был огро+мным"

    Args:
        model_size: Omograph disambiguation model size.
            ``"tiny2"``    — fastest, basic accuracy.
            ``"turbo2"``   — best speed/quality balance (default).
            ``"turbo3.1"`` — highest accuracy, slower.
    """

    def __init__(self, model_size: str = "turbo2"):
        self.model_size = model_size
        self._model = None
        self._available = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """Load the ruaccent model. Returns True on success."""
        try:
            from ruaccent import RUAccent  # type: ignore[import-untyped]

            self._model = RUAccent()
            self._model.load(
                omograph_model_size=self.model_size,
                use_dictionary=True,
            )
            self._available = True
            logger.info("ruaccent loaded (model_size=%s)", self.model_size)
            return True
        except ImportError:
            logger.warning(
                "ruaccent not installed — stress marks disabled. "
                "Run: pip install ruaccent"
            )
            return False
        except Exception as exc:
            logger.warning("ruaccent failed to load: %s", exc)
            return False

    def process(self, text: str) -> str:
        """Add stress marks to Russian text.

        Returns the original text unchanged when ruaccent is unavailable.
        """
        if not self._available or self._model is None:
            return text
        try:
            return self._model.process_all(text)
        except Exception as exc:
            logger.warning("ruaccent processing failed: %s", exc)
            return text

    @property
    def available(self) -> bool:
        """True if ruaccent loaded successfully."""
        return self._available
