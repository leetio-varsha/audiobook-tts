"""Sentence-boundary text chunker.

Splits text into chunks of approximately `max_words` words without
splitting mid-sentence. Uses simple punctuation-based tokenization
(no NLTK dependency required).
"""

from __future__ import annotations

import re


class TextChunker:
    """Split text into chunks at sentence boundaries.

    Args:
        max_words: Target maximum words per chunk (default 200).
            Chunks may exceed this if a single sentence is very long.
    """

    # Sentence-ending punctuation patterns
    _SENTENCE_SPLIT_RE = re.compile(
        r"(?<=[.!?])\s+(?=[A-Z\u0410-\u042f\u0401\"\'\u201c\u00ab\(])"  # uppercase after punctuation
        r"|(?<=[.!?])\n"                          # newline after punctuation
        r"|\n{2,}",                               # blank lines (paragraph breaks)
    )

    def __init__(self, max_words: int = 200):
        self.max_words = max_words

    def split(self, text: str) -> list[str]:
        """Split *text* into chunks, each up to ~max_words words.

        Returns:
            List of non-empty text chunks.
        """
        if not text or not text.strip():
            return []

        sentences = self._tokenize_sentences(text)
        return self._accumulate_chunks(sentences)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tokenize_sentences(self, text: str) -> list[str]:
        """Split text into individual sentences."""
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Split on sentence boundaries
        parts = self._SENTENCE_SPLIT_RE.split(text)

        sentences: list[str] = []
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)

        return sentences

    def _accumulate_chunks(self, sentences: list[str]) -> list[str]:
        """Accumulate sentences into chunks of up to max_words words."""
        chunks: list[str] = []
        current_sentences: list[str] = []
        current_word_count = 0

        for sentence in sentences:
            word_count = len(sentence.split())

            if current_word_count + word_count > self.max_words and current_sentences:
                # Flush current chunk
                chunks.append(" ".join(current_sentences))
                current_sentences = [sentence]
                current_word_count = word_count
            else:
                current_sentences.append(sentence)
                current_word_count += word_count

        # Flush remaining sentences
        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return [c for c in chunks if c.strip()]
