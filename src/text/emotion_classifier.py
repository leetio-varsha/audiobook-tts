"""Emotion/tone classifier for audiobook TTS chunks.

Classifies each text chunk into a prosody category using lightweight heuristics
(punctuation density, dialogue markers, emotional keywords, sentence length).
Returns synthesis parameters: speed and pause durations.

Maintains continuity: neighboring chunks do not abruptly reset tone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# -----------------------------------------------------------------------
# Prosody parameters per category
# -----------------------------------------------------------------------

@dataclass
class EmotionResult:
    """Synthesis parameters derived from emotion classification."""
    category: str
    speed: float            # model native speed parameter
    pause_after_ms: int     # silence to insert AFTER this chunk
    style_hint: str         # human-readable, used in logs / design-1.7b instruct

    def __repr__(self) -> str:
        return f"<{self.category} speed={self.speed} pause={self.pause_after_ms}ms>"


# Base parameters for each category
_CATEGORY_PARAMS: dict[str, dict] = {
    # Calm story narration — the default
    "narration": {
        "speed": 0.85,
        "pause_after_ms": 350,
        "style": "calm cinematic narration, warm and measured pace",
    },
    # Character speech — slightly livelier
    "dialogue": {
        "speed": 0.90,
        "pause_after_ms": 220,
        "style": "natural expressive dialogue, clear articulation",
    },
    # Internal monologue, doubt, memory — slower
    "reflective": {
        "speed": 0.78,
        "pause_after_ms": 450,
        "style": "slow thoughtful narration, introspective and intimate",
    },
    # Building tension, action, urgency — pace picks up
    "tension": {
        "speed": 0.93,
        "pause_after_ms": 380,
        "style": "tense deliberate narration, suspenseful rhythm",
    },
    # Emotional peak — most energy, then pause for landing
    "climax": {
        "speed": 0.95,
        "pause_after_ms": 520,
        "style": "dramatic intense narration, emotionally heightened",
    },
    # Soft, close voice — whispered or intimate confession
    "whisper": {
        "speed": 0.80,
        "pause_after_ms": 380,
        "style": "intimate whispered narration, restrained and soft",
    },
    # Connective sentence, scene shift
    "transition": {
        "speed": 0.85,
        "pause_after_ms": 300,
        "style": "smooth neutral transition, steady gentle pace",
    },
    # Description-heavy, informative — measured and clear
    "exposition": {
        "speed": 0.82,
        "pause_after_ms": 360,
        "style": "clear measured exposition, grounded and informative",
    },
}

# -----------------------------------------------------------------------
# Keyword sets for heuristic classification
# -----------------------------------------------------------------------

_WHISPER_WORDS = re.compile(
    r'\b(?:шёпотом|шептал[аи]?|прошептал[аи]?|тихо|еле слышно|вполголоса|прошептал)\b',
    re.IGNORECASE | re.UNICODE,
)

_CLIMAX_WORDS = re.compile(
    r'\b(?:крикнул[аи]?|закричал[аи]?|воскликнул[аи]?|вскочил[аи]?|бросил[аи]?(?:ся)?|'
    r'ударил[аи]?|упал[аи]?|выстрел|взрыв|пронзил[аи]?|задрожал[аи]?)\b',
    re.IGNORECASE | re.UNICODE,
)

_REFLECTIVE_WORDS = re.compile(
    r'\b(?:подумал[аи]?|вспомнил[аи]?|казалось|словно|как будто|наверное|может быть|'
    r'вдруг осознал|вдруг понял|почему-то|странно|всё-таки|впрочем)\b',
    re.IGNORECASE | re.UNICODE,
)

_TENSION_WORDS = re.compile(
    r'\b(?:медленно|осторожно|напряжённо|неотвратим|что-то не так|почуял|насторожил|'
    r'замер|застыл|сердце|дыхание|тишина|молчание стало|темнота)\b',
    re.IGNORECASE | re.UNICODE,
)

_TRANSITION_WORDS = re.compile(
    r'^(?:тем временем|между тем|однако|впрочем|тогда|наконец|после этого|'
    r'через некоторое время|на следующий день|утром|вечером|ночью)',
    re.IGNORECASE | re.UNICODE,
)

# Dialogue: starts with em-dash or Russian open quote
_DIALOGUE_START = re.compile(r'^\s*[—–]\s|^\s*«', re.UNICODE)
# Internal thought: text in «» or italics markers, or "он подумал:"
_INTERNAL_THOUGHT = re.compile(r'[«"]\s*[А-Яа-яёЁ]', re.UNICODE)


class EmotionClassifier:
    """Classify text chunks and return synthesis parameters.

    Args:
        base_speed: Base speed multiplier (default 0.85). All category speeds
            are relative to this — allows global tempo adjustment.
        continuity_weight: How strongly previous emotion bleeds into current
            speed (0.0 = no bleed, 0.3 = 30% pull towards previous speed).
    """

    def __init__(
        self,
        base_speed: float = 0.85,
        continuity_weight: float = 0.25,
    ):
        self.base_speed = base_speed
        self.continuity_weight = continuity_weight
        # Ratio: each category's speed relative to 0.85 base
        self._speed_ratios = {
            k: v["speed"] / 0.85
            for k, v in _CATEGORY_PARAMS.items()
        }

    def classify(
        self,
        text: str,
        prev_emotion: Optional[EmotionResult] = None,
        split_type: str = "sentence",
    ) -> EmotionResult:
        """Classify text and return EmotionResult.

        Args:
            text: Chunk text to classify.
            prev_emotion: Result from previous chunk (for continuity).
            split_type: "paragraph", "scene", "sentence", "dialogue" — affects pause.
        """
        category = self._classify_category(text)
        params = _CATEGORY_PARAMS[category]

        # Scale speed relative to user's base_speed
        speed = self.base_speed * self._speed_ratios[category]

        # Continuity: blend with previous speed to avoid jarring jumps
        if prev_emotion is not None and self.continuity_weight > 0:
            # Only blend for neighbouring non-scene chunks
            if split_type not in ("scene",):
                w = self.continuity_weight
                speed = prev_emotion.speed * w + speed * (1.0 - w)

        # Pause: base from category, boosted by split type
        pause_ms = params["pause_after_ms"]
        if split_type == "paragraph":
            pause_ms = max(pause_ms, 550)
        elif split_type == "scene":
            pause_ms = 1200
        elif split_type == "dialogue" and category == "dialogue":
            pause_ms = min(pause_ms, 200)  # tighter dialogue exchange

        # Clamp speed to reasonable range
        speed = max(0.70, min(speed, 1.10))

        return EmotionResult(
            category=category,
            speed=round(speed, 3),
            pause_after_ms=pause_ms,
            style_hint=params["style"],
        )

    # ------------------------------------------------------------------
    # Heuristic classification
    # ------------------------------------------------------------------

    def _classify_category(self, text: str) -> str:
        text_stripped = text.strip()

        # ---- Dialogue detection ----
        if _DIALOGUE_START.match(text_stripped):
            # Sub-classify dialogue tone
            if _WHISPER_WORDS.search(text_stripped):
                return "whisper"
            if _CLIMAX_WORDS.search(text_stripped):
                return "climax"
            return "dialogue"

        # ---- Whisper ----
        if _WHISPER_WORDS.search(text_stripped):
            return "whisper"

        # ---- Climax ----
        excl_count = text_stripped.count("!")
        if excl_count >= 2 or _CLIMAX_WORDS.search(text_stripped):
            return "climax"

        # ---- Reflective (internal monologue, memories) ----
        ellipsis_count = text_stripped.count("…") + text_stripped.count("...")
        has_internal_thought = bool(_INTERNAL_THOUGHT.search(text_stripped))
        if ellipsis_count >= 2 or _REFLECTIVE_WORDS.search(text_stripped) or has_internal_thought:
            return "reflective"

        # ---- Tension ----
        if _TENSION_WORDS.search(text_stripped):
            return "tension"
        # High question density + short sentence → tense interrogation
        q_count = text_stripped.count("?")
        if q_count >= 2 and len(text_stripped) < 150:
            return "tension"

        # ---- Transition ----
        if _TRANSITION_WORDS.match(text_stripped) and len(text_stripped) < 120:
            return "transition"

        # ---- Exposition (long, descriptive, past tense) ----
        # Heuristic: very long sentences with no punctuation breaks
        words = text_stripped.split()
        avg_sentence_len = len(words) / max(1, (
            text_stripped.count(".") + text_stripped.count("!") + text_stripped.count("?") + 1
        ))
        if avg_sentence_len > 20 and len(text_stripped) > 200:
            return "exposition"

        # ---- Default: narration ----
        return "narration"
