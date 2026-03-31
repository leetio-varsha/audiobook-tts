"""FB2 parser using lxml.

FB2 (FictionBook 2) is an XML-based ebook format. Structure:
  FictionBook
    description
      title-info
        book-title
        author (first-name, last-name)
        coverpage
          image (l:href="#cover-image-id")
    binary (id="cover-image-id", content-type="image/jpeg")
    body
      section (title = chapter title)
        title > p
        p (body text)
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

from lxml import etree

from src import Book, Chapter


# FictionBook 2.0 namespace
FB2_NS = "http://www.gribuser.ru/xml/fictionbook/2.0"
XLINK_NS = "http://www.w3.org/1999/xlink"

_FB = f"{{{FB2_NS}}}"
_XLINK = f"{{{XLINK_NS}}}"


def _tag(local: str) -> str:
    return f"{_FB}{local}"


class FB2Parser:
    """Parse an FB2 file and return a Book dataclass."""

    def parse(self, path: str | Path) -> Book:
        path = Path(path)
        tree = etree.parse(str(path))
        root = tree.getroot()

        title = self._extract_title(root)
        author = self._extract_author(root)
        cover_bytes = self._extract_cover(root)
        chapters = self._extract_chapters(root)

        return Book(
            title=title,
            author=author,
            cover_bytes=cover_bytes,
            chapters=chapters,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ns(self, element: etree._Element, local: str) -> list[etree._Element]:
        """Find child elements by local name, ignoring namespace."""
        return element.findall(_tag(local))

    def _find(self, element: etree._Element, local: str) -> Optional[etree._Element]:
        return element.find(_tag(local))

    def _text_content(self, element: Optional[etree._Element]) -> str:
        """Return all text content of an element and its descendants."""
        if element is None:
            return ""
        return "".join(element.itertext()).strip()

    def _extract_title(self, root: etree._Element) -> str:
        try:
            desc = root.find(_tag("description"))
            title_info = desc.find(_tag("title-info"))
            book_title = title_info.find(_tag("book-title"))
            return self._text_content(book_title) or "Unknown Title"
        except (AttributeError, TypeError):
            return "Unknown Title"

    def _extract_author(self, root: etree._Element) -> str:
        try:
            desc = root.find(_tag("description"))
            title_info = desc.find(_tag("title-info"))
            author_el = title_info.find(_tag("author"))
            if author_el is None:
                return "Unknown Author"
            first = self._text_content(author_el.find(_tag("first-name")))
            last = self._text_content(author_el.find(_tag("last-name")))
            middle = self._text_content(author_el.find(_tag("middle-name")))
            parts = [p for p in [first, middle, last] if p]
            return " ".join(parts) if parts else "Unknown Author"
        except (AttributeError, TypeError):
            return "Unknown Author"

    def _extract_cover(self, root: etree._Element) -> Optional[bytes]:
        """Extract cover image bytes from the <binary> element."""
        try:
            desc = root.find(_tag("description"))
            title_info = desc.find(_tag("title-info"))
            coverpage = title_info.find(_tag("coverpage"))
            if coverpage is None:
                return None

            # <image l:href="#cover-image"/>
            image_el = coverpage.find(_tag("image"))
            if image_el is None:
                return None

            # href could be in xlink namespace or plain
            href = (
                image_el.get(f"{_XLINK}href")
                or image_el.get("href")
                or ""
            )
            if not href.startswith("#"):
                return None
            cover_id = href[1:]  # strip leading '#'

            # Find matching <binary id="...">
            for binary in root.findall(_tag("binary")):
                if binary.get("id") == cover_id:
                    b64_data = (binary.text or "").strip()
                    return base64.b64decode(b64_data)
        except Exception:
            pass
        return None

    def _extract_chapters(self, root: etree._Element) -> list[Chapter]:
        """Extract chapters from <body><section> elements."""
        chapters: list[Chapter] = []

        for body in root.findall(_tag("body")):
            for section in body.findall(_tag("section")):
                chapter = self._parse_section(section)
                if chapter:
                    chapters.append(chapter)

        # If no titled sections found, treat entire body as one chapter
        if not chapters:
            full_text = self._extract_body_text(root)
            if full_text:
                chapters.append(Chapter(title="Book", text=full_text))

        return chapters

    def _parse_section(self, section: etree._Element) -> Optional[Chapter]:
        """Parse a single <section> into a Chapter."""
        # Title from <title><p>...</p></title>
        title_el = section.find(_tag("title"))
        title_text = self._text_content(title_el) if title_el is not None else ""
        if not title_text:
            title_text = "Chapter"

        # Body text: all <p> elements (possibly nested sections)
        paragraphs: list[str] = []
        self._collect_paragraphs(section, paragraphs, skip_title=True)
        text = "\n\n".join(p for p in paragraphs if p)

        if not text.strip():
            return None

        return Chapter(title=title_text, text=text)

    def _collect_paragraphs(
        self,
        element: etree._Element,
        result: list[str],
        skip_title: bool = False,
    ) -> None:
        """Recursively collect paragraph text from an element."""
        for child in element:
            local = etree.QName(child.tag).localname if child.tag else ""
            if skip_title and local == "title":
                continue
            elif local == "p":
                text = self._text_content(child)
                if text:
                    result.append(text)
            elif local == "section":
                # Nested section: recurse
                self._collect_paragraphs(child, result, skip_title=False)
            elif local in ("epigraph", "annotation", "poem", "stanza", "v", "subtitle"):
                text = self._text_content(child)
                if text:
                    result.append(text)
            elif local == "empty-line":
                pass  # skip
            else:
                # Generic element — collect its text too
                text = self._text_content(child)
                if text:
                    result.append(text)

    def _extract_body_text(self, root: etree._Element) -> str:
        paragraphs: list[str] = []
        for body in root.findall(_tag("body")):
            self._collect_paragraphs(body, paragraphs)
        return "\n\n".join(paragraphs)
