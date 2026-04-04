"""Semantic text chunker for audiobook TTS.

Splits text at natural thought boundaries — paragraph breaks, sentence ends,
dialogue exchange completions. Avoids cutting inside dialogue attributions,
emotional reveals, or mid-sentence clauses.

Preferred unit: one complete thought (paragraph, dialogue exchange, narration beat).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Chars per chunk: Qwen3-TTS base is comfortable up to ~450 chars
MAX_CHUNK_CHARS = 450
PREFERRED_CHUNK_CHARS = 280

# Sentence-ending punctuation followed by whitespace and a sentence-starting char
_SENT_SPLIT = re.compile(
    r'(?<=[.!?…])\s+(?=[А-ЯЁ«—"\u201c])',
    re.UNICODE,
)

# Abbreviation patterns — do not split after these (e.g. "т.д.", "и др.", "А. Пушкин")
_ABBREV_END = re.compile(r'\b(?:[А-ЯЁа-яёA-Za-z]|т\.д|т\.п|и\s+др|напр|т\.е)\.\s*$', re.UNICODE)

# Dialogue: line starts with em-dash or is enclosed in Russian quotation marks
_DIALOGUE_START = re.compile(r'^\s*[—–]\s', re.UNICODE)
_DIALOGUE_QUOTE = re.compile(r'«[^»]+»', re.UNICODE)

# Scene break markers (*** — —— etc.)
_SCENE_BREAK = re.compile(r'^\s*[\*\-—–_=~]{2,}\s*$', re.UNICODE)


@dataclass
class SemanticChunk:
    """A chunk of text with prosody context."""
    text: str
    # How the text was split BEFORE this chunk (affects pre-chunk pause)
    split_type: str = "sentence"   # "paragraph", "scene", "sentence", "dialogue"
    is_dialogue: bool = False
    index: int = 0


class SemanticChunker:
    """Split text into semantically coherent chunks for audiobook TTS.

    Args:
        max_chars: Hard limit on chunk character count (default 450).
        preferred_chars: Preferred maximum; longer chunks are split at
            sentence boundaries first (default 280).
    """

    def __init__(
        self,
        max_chars: int = MAX_CHUNK_CHARS,
        preferred_chars: int = PREFERRED_CHUNK_CHARS,
    ):
        self.max_chars = max_chars
        self.preferred_chars = preferred_chars

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, text: str) -> list[SemanticChunk]:
        """Split text into semantic chunks.

        Returns:
            Ordered list of SemanticChunk objects.
        """
        if not text or not text.strip():
            return []

        text = text.replace("\r\n", "\n").replace("\r", "\n")

        paragraphs = self._split_paragraphs(text)
        raw: list[SemanticChunk] = []

        for para_text, para_type in paragraphs:
            if para_type == "scene":
                # Scene breaks carry no speakable text; mark the NEXT chunk
                if raw:
                    raw[-1] = SemanticChunk(
                        text=raw[-1].text,
                        split_type="scene",
                        is_dialogue=raw[-1].is_dialogue,
                        index=raw[-1].index,
                    )
                continue

            chunks = self._chunk_paragraph(para_text)
            first = True
            for c in chunks:
                c.split_type = "paragraph" if first else "sentence"
                first = False
                raw.append(c)

        # Merge orphan chunks (very short text) with next chunk
        merged = self._merge_orphans(raw)

        # Re-index
        for i, c in enumerate(merged):
            c.index = i

        return merged

    # ------------------------------------------------------------------
    # Paragraph splitting
    # ------------------------------------------------------------------

    def _split_paragraphs(self, text: str) -> list[tuple[str, str]]:
        """Split text into (para_text, type) pairs.

        type is "normal" or "scene" (for *** scene breaks).
        """
        parts = re.split(r'(\n\s*\n+)', text.strip())
        result: list[tuple[str, str]] = []

        for part in parts:
            stripped = part.strip()
            if not stripped:
                continue
            if _SCENE_BREAK.match(stripped):
                result.append(("", "scene"))
            elif "\n" not in stripped and len(stripped) < 5:
                # Just whitespace / trivial separator
                continue
            else:
                result.append((stripped, "normal"))

        return result

    # ------------------------------------------------------------------
    # Per-paragraph chunking
    # ------------------------------------------------------------------

    def _chunk_paragraph(self, text: str) -> list[SemanticChunk]:
        """Chunk a single paragraph respecting dialogue integrity."""
        text = text.strip()
        if not text:
            return []

        is_dialogue = bool(_DIALOGUE_START.match(text) or _DIALOGUE_QUOTE.search(text))

        # Short enough → single chunk
        if len(text) <= self.preferred_chars:
            return [SemanticChunk(text=text, is_dialogue=is_dialogue)]

        # Long dialogue: keep as one chunk unless WAY over limit
        if is_dialogue and len(text) <= self.max_chars:
            return [SemanticChunk(text=text, is_dialogue=is_dialogue)]

        # Split at sentence boundaries
        sentences = self._split_sentences(text)
        if len(sentences) == 1:
            # Un-splittable single sentence — return as is (may be long)
            return [SemanticChunk(text=text, is_dialogue=is_dialogue)]

        return self._pack_into_chunks(sentences, is_dialogue)

    def _pack_into_chunks(
        self, sentences: list[str], is_dialogue: bool
    ) -> list[SemanticChunk]:
        """Group sentences into chunks respecting max_chars."""
        chunks: list[SemanticChunk] = []
        current: list[str] = []

        for sent in sentences:
            candidate = " ".join(current + [sent]) if current else sent

            if len(candidate) > self.preferred_chars and current:
                # Flush and start new chunk
                chunks.append(SemanticChunk(
                    text=" ".join(current),
                    is_dialogue=is_dialogue,
                ))
                current = [sent]
            else:
                current.append(sent)

            # Hard limit: flush immediately even mid-accumulation
            if current and len(" ".join(current)) > self.max_chars:
                chunks.append(SemanticChunk(
                    text=" ".join(current),
                    is_dialogue=is_dialogue,
                ))
                current = []

        if current:
            chunks.append(SemanticChunk(
                text=" ".join(current),
                is_dialogue=is_dialogue,
            ))

        return chunks

    # ------------------------------------------------------------------
    # Sentence splitting
    # ------------------------------------------------------------------

    def _split_sentences(self, text: str) -> list[str]:
        """Split paragraph into sentences, preserving dialogue attribution."""
        # Use regex to find candidate split points
        parts = _SENT_SPLIT.split(text)
        if len(parts) <= 1:
            return [text]

        # Post-process: re-join if split was at abbreviation
        result: list[str] = []
        pending = parts[0]

        for part in parts[1:]:
            if _ABBREV_END.search(pending):
                # Abbreviation — re-join
                pending = pending + " " + part
            else:
                result.append(pending.strip())
                pending = part

        if pending.strip():
            result.append(pending.strip())

        return [s for s in result if s]

    # ------------------------------------------------------------------
    # Orphan merging
    # ------------------------------------------------------------------

    def _merge_orphans(self, chunks: list[SemanticChunk]) -> list[SemanticChunk]:
        """Merge very short chunks (< 40 chars) with their neighbor."""
        MIN_CHARS = 40
        if len(chunks) <= 1:
            return chunks

        merged: list[SemanticChunk] = []
        i = 0
        while i < len(chunks):
            c = chunks[i]
            # Merge forward: current chunk is too short AND next chunk exists
            if len(c.text) < MIN_CHARS and i + 1 < len(chunks):
                next_c = chunks[i + 1]
                combined = c.text + " " + next_c.text
                if len(combined) <= self.max_chars:
                    merged.append(SemanticChunk(
                        text=combined.strip(),
                        split_type=c.split_type,
                        is_dialogue=c.is_dialogue or next_c.is_dialogue,
                    ))
                    i += 2
                    continue
            merged.append(c)
            i += 1

        return merged
