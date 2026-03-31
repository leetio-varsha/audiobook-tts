#!/usr/bin/env python3
"""CLI for converting ebooks to M4B audiobooks using XTTS v2 (local, MPS-accelerated).

Usage:
    python convert.py --input book.fb2 --output audiobook.m4b
    python convert.py --input book.epub --output audiobook.m4b --voice voices/sample.wav
    python convert.py --input book.pdf --output audiobook.m4b --language en
    python convert.py --input book.fb2 --output out.m4b --voice voices/sample.wav --language ru
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("convert")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ebook (FB2/EPUB/PDF) to M4B audiobook using XTTS v2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        metavar="FILE",
        help="Path to the input ebook file (.fb2, .epub, .pdf)",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        metavar="FILE",
        help="Path to the output M4B file",
    )
    parser.add_argument(
        "--voice", "-v",
        default=None,
        metavar="WAV",
        help="Path to a WAV voice sample for cloning (optional)",
    )
    parser.add_argument(
        "--language", "-l",
        default="ru",
        metavar="LANG",
        help='TTS language code (default: "ru"). Examples: ru, en, de, fr, es',
    )
    parser.add_argument(
        "--cache-dir",
        default="./cache",
        metavar="DIR",
        help="Directory for WAV chunk cache (default: ./cache)",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=200,
        metavar="N",
        help="Maximum words per TTS chunk (default: 200)",
    )
    parser.add_argument(
        "--pages-per-chapter",
        type=int,
        default=20,
        metavar="N",
        help="Pages per pseudo-chapter for PDF input (default: 20)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def compute_book_hash(path: Path) -> str:
    """Compute MD5 hash of book file content for cache keying."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def get_cache_dir(base: Path, book_hash: str) -> Path:
    cache = base / book_hash
    cache.mkdir(parents=True, exist_ok=True)
    return cache


# ---------------------------------------------------------------------------
# Core conversion pipeline
# ---------------------------------------------------------------------------

def convert(
    input_path: Path,
    output_path: Path,
    voice_sample: Path | None,
    language: str,
    cache_base: Path,
    max_words: int,
    pages_per_chapter: int,
) -> None:
    # --- Parse book ---
    logger.info("Parsing book: %s", input_path)
    from src.parser import parse_book
    from src.parser.pdf import PDFParser

    if input_path.suffix.lower() == ".pdf":
        parser = PDFParser(pages_per_chapter=pages_per_chapter)
        book = parser.parse(input_path)
    else:
        book = parse_book(input_path)

    logger.info("Title: %s", book.title)
    logger.info("Author: %s", book.author)
    logger.info("Chapters: %d", len(book.chapters))

    if not book.chapters:
        logger.error("No chapters found in book. Aborting.")
        sys.exit(1)

    # --- Set up cache ---
    book_hash = compute_book_hash(input_path)
    cache_dir = get_cache_dir(cache_base, book_hash)
    logger.info("Cache directory: %s", cache_dir)

    # --- Load TTS engine ---
    logger.info("Loading XTTS v2 model (first run downloads ~2 GB)...")
    from src.tts.engine import TTSEngine
    from src.tts.chunker import TextChunker

    engine = TTSEngine(voice_sample=voice_sample, language=language)
    chunker = TextChunker(max_words=max_words)

    # --- Synthesize chunks ---
    # chapter_wav_lists[i] = list of WAV paths for chapter i
    chapter_wav_lists: list[list[Path]] = []
    chapter_titles: list[str] = []

    # Count total chunks across all chapters for progress bar
    all_chunks: list[tuple[int, int, str]] = []  # (chapter_idx, chunk_idx, text)
    for ch_idx, chapter in enumerate(book.chapters):
        chunks = chunker.split(chapter.text)
        for ck_idx, chunk_text in enumerate(chunks):
            all_chunks.append((ch_idx, ck_idx, chunk_text))

    logger.info("Total TTS chunks: %d", len(all_chunks))

    # Initialize chapter_wav_lists
    for chapter in book.chapters:
        chapter_wav_lists.append([])
        chapter_titles.append(chapter.title)

    # Global chunk index across all chapters
    global_idx = 0

    with tqdm(total=len(all_chunks), desc="Synthesizing", unit="chunk") as pbar:
        for ch_idx, ck_idx, chunk_text in all_chunks:
            wav_path = cache_dir / f"ch{ch_idx:04d}_ck{ck_idx:04d}.wav"

            if wav_path.exists() and wav_path.stat().st_size > 0:
                logger.debug("Cache hit: %s", wav_path.name)
            else:
                logger.debug(
                    "Synthesizing ch%d chunk%d (%d words)...",
                    ch_idx, ck_idx, len(chunk_text.split()),
                )
                engine.synthesize(chunk_text, wav_path)

            chapter_wav_lists[ch_idx].append(wav_path)
            global_idx += 1
            pbar.update(1)

    # --- Build M4B ---
    logger.info("Building M4B: %s", output_path)
    from src.audio.builder import AudioBuilder

    builder = AudioBuilder(
        output_path=output_path,
        title=book.title,
        author=book.author,
        cover_bytes=book.cover_bytes,
    )

    # Filter out empty chapters
    filtered_wavs = [wl for wl in chapter_wav_lists if wl]
    filtered_titles = [t for t, wl in zip(chapter_titles, chapter_wav_lists) if wl]

    builder.build(filtered_wavs, filtered_titles)
    logger.info("Done! Output: %s", output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    voice_sample: Path | None = None
    if args.voice:
        voice_sample = Path(args.voice)
        if not voice_sample.exists():
            logger.error("Voice sample not found: %s", voice_sample)
            sys.exit(1)

    cache_base = Path(args.cache_dir)

    convert(
        input_path=input_path,
        output_path=output_path,
        voice_sample=voice_sample,
        language=args.language,
        cache_base=cache_base,
        max_words=args.max_words,
        pages_per_chapter=args.pages_per_chapter,
    )


if __name__ == "__main__":
    main()
