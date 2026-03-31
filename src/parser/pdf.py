"""PDF parser using pdfminer.six.

Extracts text page by page, then groups pages into pseudo-chapters
(default: every 20 pages = one chapter).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

from pdfminer.high_level import extract_text_to_fp, extract_pages
from pdfminer.layout import LAParams, LTTextContainer, LTChar, LTPage
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter

from src import Book, Chapter


class PDFParser:
    """Parse a PDF file and return a Book dataclass."""

    def __init__(self, pages_per_chapter: int = 20):
        self.pages_per_chapter = pages_per_chapter

    def parse(self, path: str | Path) -> Book:
        path = Path(path)

        pages = self._extract_pages(path)

        title = path.stem.replace("_", " ").replace("-", " ").title()
        author = "Unknown Author"
        cover_bytes = None  # PDF cover extraction is non-trivial; skip

        chapters = self._group_into_chapters(pages)

        return Book(
            title=title,
            author=author,
            cover_bytes=cover_bytes,
            chapters=chapters,
        )

    # ------------------------------------------------------------------
    # Page extraction
    # ------------------------------------------------------------------

    def _extract_pages(self, path: Path) -> list[str]:
        """Extract text from each page, return list of page strings."""
        page_texts: list[str] = []

        with open(path, "rb") as f:
            resource_manager = PDFResourceManager()
            laparams = LAParams(
                line_margin=0.5,
                char_margin=2.0,
                boxes_flow=0.5,
            )

            for page in PDFPage.get_pages(f, check_extractable=True):
                output_buf = io.StringIO()
                converter = TextConverter(
                    resource_manager,
                    output_buf,
                    laparams=laparams,
                )
                interpreter = PDFPageInterpreter(resource_manager, converter)
                interpreter.process_page(page)
                converter.close()

                page_text = output_buf.getvalue()
                page_texts.append(page_text.strip())

        return page_texts

    # ------------------------------------------------------------------
    # Chapter grouping
    # ------------------------------------------------------------------

    def _group_into_chapters(self, pages: list[str]) -> list[Chapter]:
        """Group pages into pseudo-chapters of self.pages_per_chapter pages each."""
        if not pages:
            return [Chapter(title="Book", text="")]

        chapters: list[Chapter] = []
        total = len(pages)
        n = self.pages_per_chapter

        chapter_num = 1
        for start in range(0, total, n):
            end = min(start + n, total)
            chunk_pages = pages[start:end]
            text = "\n\n".join(p for p in chunk_pages if p)

            if not text.strip():
                continue

            page_start = start + 1
            page_end = end
            title = f"Part {chapter_num} (pages {page_start}–{page_end})"
            chapters.append(Chapter(title=title, text=text))
            chapter_num += 1

        if not chapters:
            all_text = "\n\n".join(p for p in pages if p)
            chapters.append(Chapter(title="Book", text=all_text))

        return chapters
