"""EPUB parser using ebooklib + BeautifulSoup4.

Reads EPUB spine items in order, extracts HTML content with BS4,
and groups them into chapters using the EPUB TOC/nav.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

from src import Book, Chapter


class EPUBParser:
    """Parse an EPUB file and return a Book dataclass."""

    def parse(self, path: str | Path) -> Book:
        path = Path(path)
        book_epub = epub.read_epub(str(path), options={"ignore_ncx": False})

        title = self._extract_title(book_epub)
        author = self._extract_author(book_epub)
        cover_bytes = self._extract_cover(book_epub)
        chapters = self._extract_chapters(book_epub)

        return Book(
            title=title,
            author=author,
            cover_bytes=cover_bytes,
            chapters=chapters,
        )

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def _extract_title(self, book_epub: epub.EpubBook) -> str:
        title = book_epub.get_metadata("DC", "title")
        if title:
            return str(title[0][0])
        return "Unknown Title"

    def _extract_author(self, book_epub: epub.EpubBook) -> str:
        creators = book_epub.get_metadata("DC", "creator")
        if creators:
            return str(creators[0][0])
        return "Unknown Author"

    def _extract_cover(self, book_epub: epub.EpubBook) -> Optional[bytes]:
        """Try several heuristics to find cover image bytes."""
        # 1. Look for item with id 'cover-image' or 'cover'
        for item_id in ("cover-image", "cover", "Cover", "CoverImage"):
            item = book_epub.get_item_with_id(item_id)
            if item and item.media_type.startswith("image/"):
                return item.get_content()

        # 2. Look for item with 'cover' in its href
        for item in book_epub.get_items_of_type(ebooklib.ITEM_IMAGE):
            if "cover" in item.get_name().lower():
                return item.get_content()

        # 3. Check metadata for cover item reference
        cover_meta = book_epub.get_metadata("OPF", "cover")
        if cover_meta:
            cover_id = cover_meta[0][1].get("content", "")
            item = book_epub.get_item_with_id(cover_id)
            if item:
                return item.get_content()

        return None

    # ------------------------------------------------------------------
    # Chapter extraction
    # ------------------------------------------------------------------

    def _extract_chapters(self, book_epub: epub.EpubBook) -> list[Chapter]:
        """Extract chapters from the EPUB spine."""
        # Build a map from href -> chapter title using the TOC
        toc_map = self._build_toc_map(book_epub)

        chapters: list[Chapter] = []
        chapter_index = 1

        for item in book_epub.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            # Only process items in the spine
            spine_ids = {s[0] for s in book_epub.spine}
            if item.get_id() not in spine_ids:
                continue

            text = self._html_to_text(item.get_content())
            if not text.strip():
                continue

            # Try to get a title from TOC map
            href = item.get_name()
            title = toc_map.get(href) or toc_map.get(href.split("#")[0])

            if not title:
                # Try to extract H1/H2 from the HTML
                title = self._extract_heading(item.get_content())

            if not title:
                title = f"Chapter {chapter_index}"

            chapters.append(Chapter(title=title, text=text))
            chapter_index += 1

        if not chapters:
            # Fallback: dump everything into one chapter
            all_text_parts = []
            for item in book_epub.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                t = self._html_to_text(item.get_content())
                if t.strip():
                    all_text_parts.append(t)
            if all_text_parts:
                chapters.append(Chapter(title="Book", text="\n\n".join(all_text_parts)))

        return chapters

    def _build_toc_map(self, book_epub: epub.EpubBook) -> dict[str, str]:
        """Build a map from document href -> chapter title from the TOC."""
        toc_map: dict[str, str] = {}
        self._walk_toc(book_epub.toc, toc_map)
        return toc_map

    def _walk_toc(self, toc_items, toc_map: dict[str, str]) -> None:
        """Recursively walk TOC entries."""
        for item in toc_items:
            if isinstance(item, epub.Link):
                href = item.href.split("#")[0]  # strip anchor
                toc_map[href] = item.title
            elif isinstance(item, tuple) and len(item) == 2:
                section, children = item
                if isinstance(section, epub.Section) and hasattr(section, "href"):
                    href = section.href.split("#")[0]
                    toc_map[href] = section.title
                self._walk_toc(children, toc_map)
            elif isinstance(item, (list, tuple)):
                self._walk_toc(item, toc_map)

    def _html_to_text(self, html_bytes: bytes) -> str:
        """Convert HTML content to plain text."""
        soup = BeautifulSoup(html_bytes, "html.parser")

        # Remove script/style elements
        for tag in soup(["script", "style", "head"]):
            tag.decompose()

        # Add newlines after block elements
        for tag in soup.find_all(["p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li"]):
            tag.append("\n")

        text = soup.get_text(separator=" ")

        # Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        return text

    def _extract_heading(self, html_bytes: bytes) -> Optional[str]:
        """Extract first H1 or H2 text from HTML."""
        soup = BeautifulSoup(html_bytes, "html.parser")
        for tag_name in ["h1", "h2", "h3"]:
            heading = soup.find(tag_name)
            if heading:
                text = heading.get_text(strip=True)
                if text:
                    return text
        return None
