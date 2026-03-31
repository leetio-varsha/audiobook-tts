"""audiobook-tts: Convert ebooks to audiobooks using local XTTS v2."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chapter:
    """A single chapter of a book."""
    title: str
    text: str


@dataclass
class Book:
    """Parsed book representation."""
    title: str
    author: str
    cover_bytes: Optional[bytes]
    chapters: list[Chapter] = field(default_factory=list)
