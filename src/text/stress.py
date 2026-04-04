"""Multi-layer Russian stress pipeline.

Layer priority (highest to lowest):
  1. Custom overrides (data/stress_overrides.json) — manual corrections
  2. OpenRussian dict (data/stress_dict.json) — 397k inflected forms
  3. OpenRussian via pymorphy3 lemma — catches inflected forms not in dict
  4. silero-stress (4M word dictionary + context-aware BiLSTM) — fallback

Output: text with '+' placed before each stressed vowel.
Compatible with F5-TTS Russian (Misha24-10 vocab includes '+').

Usage::
    pipeline = RussianStressPipeline()
    pipeline.load()
    text = pipeline.process("Замок на двери был сломан.")
    # → "З+амок на дв+ери б+ыл сл+оман."
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
_WORD_RE = re.compile(r"[а-яёА-ЯЁ]+(?:[-'][а-яёА-ЯЁ]+)*", re.UNICODE)

# Default paths relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DICT_PATH = _PROJECT_ROOT / "data" / "stress_dict.json"
_OVERRIDES_PATH = _PROJECT_ROOT / "data" / "stress_overrides.json"


def _remove_plus(word: str) -> str:
    return word.replace("+", "")


def _count_vowels(word: str) -> int:
    return sum(1 for c in word if c in VOWELS)


def _match_case(original: str, stressed_lower: str) -> str:
    """Apply capitalisation of original word to the stressed (lower) form."""
    if not original:
        return stressed_lower
    result = []
    src_i = 0  # index into original (without '+')
    for ch in stressed_lower:
        if ch == "+":
            result.append("+")
            continue
        if src_i < len(original) and original[src_i].isupper():
            result.append(ch.upper())
        else:
            result.append(ch)
        src_i += 1
    return "".join(result)


def _already_stressed(word: str) -> bool:
    return "+" in word


def _stressed_vowel_index(stressed_word: str) -> int:
    """Return 1-based index of the stressed vowel among all vowels in word.

    Example: 'насто+ящий' → 3  (а=1, о=2, я=3)
    """
    vowel_n = 0
    for i, ch in enumerate(stressed_word):
        if ch == "+" and i + 1 < len(stressed_word) and stressed_word[i + 1].lower() in VOWELS:
            return vowel_n + 1
        if ch.lower() in VOWELS:
            vowel_n += 1
    return -1


def _apply_stress_at_vowel_index(word: str, idx: int) -> str:
    """Insert '+' before the n-th vowel (1-based) in word.

    Example: ('настоящая', 3) → 'насто+ящая'
    """
    if idx < 1:
        return word
    vowel_n = 0
    for i, ch in enumerate(word):
        if ch.lower() in VOWELS:
            vowel_n += 1
            if vowel_n == idx:
                return word[:i] + "+" + word[i:]
    return word


class RussianStressPipeline:
    """Multi-layer stress pipeline: overrides → dict → pymorphy3+dict → silero.

    Args:
        dict_path: Path to OpenRussian stress_dict.json.
        overrides_path: Path to manual stress_overrides.json.
    """

    def __init__(
        self,
        dict_path: str | Path = _DICT_PATH,
        overrides_path: str | Path = _OVERRIDES_PATH,
    ):
        self._dict_path = Path(dict_path)
        self._overrides_path = Path(overrides_path)
        self._dict: dict[str, str] = {}
        self._overrides: dict[str, str] = {}
        self._morph = None   # pymorphy3.MorphAnalyzer
        self._silero = None  # silero_stress accentor
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> "RussianStressPipeline":
        """Load all layers. Returns self for chaining."""
        self._load_overrides()
        self._load_dict()
        self._load_morph()
        self._load_silero()
        self._loaded = True
        logger.info(
            "RussianStressPipeline loaded: %d overrides, %d dict entries, "
            "pymorphy3=%s, silero=%s",
            len(self._overrides),
            len(self._dict),
            self._morph is not None,
            self._silero is not None,
        )
        return self

    def _load_overrides(self) -> None:
        if not self._overrides_path.exists():
            logger.warning("stress_overrides.json not found: %s", self._overrides_path)
            return
        with open(self._overrides_path, encoding="utf-8") as f:
            raw = json.load(f)
        self._overrides = {
            k.lower(): v
            for k, v in raw.items()
            if not k.startswith("_")  # skip comment keys
        }
        logger.debug("Loaded %d stress overrides", len(self._overrides))

    def _load_dict(self) -> None:
        if not self._dict_path.exists():
            logger.warning("stress_dict.json not found: %s", self._dict_path)
            return
        with open(self._dict_path, encoding="utf-8") as f:
            self._dict = json.load(f)
        logger.debug("Loaded %d OpenRussian dict entries", len(self._dict))

    def _load_morph(self) -> None:
        try:
            import pymorphy3
            self._morph = pymorphy3.MorphAnalyzer()
            logger.debug("pymorphy3 loaded")
        except ImportError:
            logger.warning("pymorphy3 not installed — lemma lookup disabled")
        except Exception as e:
            logger.warning("pymorphy3 load failed: %s", e)

    def _load_silero(self) -> None:
        try:
            from silero_stress import load_accentor
            self._silero = load_accentor(lang="ru")
            logger.debug("silero_stress loaded")
        except ImportError:
            logger.warning("silero_stress not installed — context-aware stress disabled")
        except Exception as e:
            logger.warning("silero_stress load failed: %s", e)

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def process(self, text: str) -> str:
        """Add '+' stress marks to Russian text.

        Processes word by word through the layer hierarchy. Words found in
        higher-priority layers are not re-processed by lower ones.
        """
        if not self._loaded:
            self.load()

        if not text.strip():
            return text

        # Collect unknown words for silero batch processing
        positions: list[tuple[int, int, str]] = []  # (start, end, original)
        resolved: dict[int, str] = {}  # start → stressed word

        for m in _WORD_RE.finditer(text):
            start, end = m.start(), m.end()
            word = m.group()
            positions.append((start, end, word))

            stressed = self._lookup(word)
            if stressed is not None:
                resolved[start] = _match_case(word, stressed)

        # Batch the unresolved words through silero_stress
        unresolved = [(s, e, w) for s, e, w in positions if s not in resolved]
        if unresolved and self._silero is not None:
            # Build a proxy sentence with just the unknown words so silero
            # still has some context from adjacent tokens.
            silero_resolved = self._silero_batch(text, unresolved)
            resolved.update(silero_resolved)

        # Reconstruct text with stress marks
        result = []
        prev = 0
        for start, end, word in positions:
            result.append(text[prev:start])
            result.append(resolved.get(start, word))  # fallback: original
            prev = end
        result.append(text[prev:])
        return "".join(result)

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def _lookup(self, word: str) -> Optional[str]:
        """Return stressed form (lowercase with +) or None if not found."""
        key = word.lower()

        # Skip single-vowel words — always stressed, no mark needed
        if _count_vowels(key) <= 1:
            return None  # let through unchanged (silero can handle it)

        # Layer 1: custom overrides
        if key in self._overrides:
            return self._overrides[key]

        # Layer 2: OpenRussian dict (exact form)
        if key in self._dict:
            return self._dict[key]

        # Layer 3: pymorphy3 normal form → dict, then transfer stress back to
        # original inflected form by vowel position
        if self._morph and self._dict:
            try:
                parses = self._morph.parse(key)
                if parses:
                    lemma = parses[0].normal_form.lower()
                    if lemma != key and lemma in self._dict:
                        stressed_lemma = self._dict[lemma]
                        idx = _stressed_vowel_index(stressed_lemma)
                        if idx > 0:
                            transferred = _apply_stress_at_vowel_index(key, idx)
                            if "+" in transferred:
                                return transferred
            except Exception:
                pass

        return None  # not found — will go to silero

    def _silero_batch(
        self, full_text: str, unresolved: list[tuple[int, int, str]]
    ) -> dict[int, str]:
        """Process the whole text with silero_stress and extract stressed forms
        for the positions in `unresolved`."""
        try:
            silero_out = self._silero(full_text)
        except Exception as e:
            logger.warning("silero_stress failed on full text: %s", e)
            return {}

        # Align silero output to original positions.
        # silero_stress preserves non-Russian chars so the position of
        # Cyrillic word N in the output matches word N in the input.
        result = {}
        silero_words = _WORD_RE.findall(silero_out)
        all_words = [(s, e, w) for s, e, w in
                     [(m.start(), m.end(), m.group())
                      for m in _WORD_RE.finditer(full_text)]]

        # Build index: original word position → silero stressed form
        silero_idx = 0
        for orig_start, _, orig_word in all_words:
            if silero_idx >= len(silero_words):
                break
            syl = silero_words[silero_idx]
            # Strip '+' from silero output for comparison with original
            if _remove_plus(syl).lower() == orig_word.lower():
                # Only store if silero actually added a stress mark
                if "+" in syl:
                    # Only use for positions we didn't already resolve
                    for u_start, _, u_word in unresolved:
                        if u_start == orig_start:
                            result[orig_start] = _match_case(orig_word, syl.lower())
                            break
                silero_idx += 1
            else:
                # Mismatch — skip silero token
                silero_idx += 1

        return result

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def lookup_word(self, word: str) -> Optional[str]:
        """Public single-word lookup (for debugging/testing)."""
        if not self._loaded:
            self.load()
        return self._lookup(word)

    @property
    def loaded(self) -> bool:
        return self._loaded
