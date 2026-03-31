"""Book parsers for FB2, EPUB, and PDF formats."""

from pathlib import Path

from src import Book
from src.parser.fb2 import FB2Parser
from src.parser.epub import EPUBParser
from src.parser.pdf import PDFParser


def parse_book(path: str | Path) -> Book:
    """Auto-detect format and parse the book at the given path."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".fb2":
        return FB2Parser().parse(path)
    elif suffix == ".epub":
        return EPUBParser().parse(path)
    elif suffix == ".pdf":
        return PDFParser().parse(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix!r}. Supported: .fb2, .epub, .pdf")


__all__ = ["parse_book", "FB2Parser", "EPUBParser", "PDFParser"]
