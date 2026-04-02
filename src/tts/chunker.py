"""Smart text chunker for audiobook TTS.

Preserves paragraph structure and dialogue markers for natural prosody.
Splits text into chunks that respect:
  - Paragraph boundaries (double newline → long pause)
  - Sentence boundaries (. ! ? → medium pause)
  - Dialogue markers (— dash quotes → speaker change pause)
  - Clause boundaries (, ; : → short pause)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class TextChunk:
    """A chunk of text with prosody hints."""
    text: str
    # Pause to insert AFTER this chunk (in milliseconds)
    pause_after_ms: int = 0
    # Whether this chunk starts a new paragraph
    is_paragraph_start: bool = False
    # Whether this chunk contains dialogue
    is_dialogue: bool = False


class TextChunker:
    """Split text into chunks at natural boundaries.

    Preserves paragraph structure and marks dialogue for prosody.

    Args:
        max_words: Target maximum words per chunk (default 200).
    """

    # Paragraph break: two+ newlines or explicit scene break markers
    _PARA_SPLIT = re.compile(r"\n\s*\n+|\n?\s*\*\s*\*\s*\*\s*\n?")

    # Scene break markers (just decorators, no speakable text)
    _SCENE_BREAK_RE = re.compile(r"^\s*[\*\-—–_=~]{2,}\s*$")

    # Sentence-ending punctuation followed by space/newline
    _SENTENCE_SPLIT = re.compile(
        r"(?<=[.!?…])\s+(?=[A-ZА-ЯЁ\"\'«\u201c\(—])"
        r"|(?<=[.!?…])\n"
    )

    # Dialogue detection: starts with em-dash or is in «quotes»
    _DIALOGUE_RE = re.compile(r"^\s*[—–]\s|«[^»]+»")

    def __init__(self, max_words: int = 200):
        self.max_words = max_words

    def split(self, text: str) -> list[str]:
        """Split text into chunks (simple string list, backward compatible).

        Returns:
            List of non-empty text chunks.
        """
        return [c.text for c in self.split_rich(text)]

    def split_rich(self, text: str) -> list[TextChunk]:
        """Split text into chunks with prosody metadata.

        Returns:
            List of TextChunk objects with pause/dialogue hints.
        """
        if not text or not text.strip():
            return []

        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Step 1: split into paragraphs
        paragraphs = self._PARA_SPLIT.split(text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # Step 2: split paragraphs into sentences, accumulate into chunks
        all_chunks: list[TextChunk] = []

        for para_idx, paragraph in enumerate(paragraphs):
            # Skip scene break markers (*** --- etc.)
            if self._SCENE_BREAK_RE.match(paragraph):
                continue
            sentences = self._split_sentences(paragraph)
            if not sentences:
                continue
            is_dialogue = bool(self._DIALOGUE_RE.search(paragraph))

            # Accumulate sentences into chunks up to max_words
            current: list[str] = []
            current_words = 0

            for sent in sentences:
                word_count = len(sent.split())

                if current_words + word_count > self.max_words and current:
                    # Flush current chunk
                    chunk_text = " ".join(current)
                    all_chunks.append(TextChunk(
                        text=chunk_text,
                        pause_after_ms=150,  # sentence break within paragraph
                        is_dialogue=is_dialogue,
                    ))
                    current = [sent]
                    current_words = word_count
                else:
                    current.append(sent)
                    current_words += word_count

            # Flush remaining
            if current:
                chunk_text = " ".join(current)
                # Last chunk of paragraph gets longer pause
                is_last_para = (para_idx == len(paragraphs) - 1)
                all_chunks.append(TextChunk(
                    text=chunk_text,
                    pause_after_ms=0 if is_last_para else 400,  # paragraph break
                    is_paragraph_start=(len(all_chunks) == 0 or
                                        (all_chunks and all_chunks[-1].pause_after_ms >= 400)),
                    is_dialogue=is_dialogue,
                ))

        return all_chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split a paragraph into sentences."""
        parts = self._SENTENCE_SPLIT.split(text)
        return [p.strip() for p in parts if p.strip()]
