#!/usr/bin/env python3
"""CLI for converting ebooks to M4B audiobooks using Silero TTS v4 + optional RVC.

Usage:
    python convert.py --input book.fb2 --output audiobook.m4b
    python convert.py --input book.epub --output audiobook.m4b --speaker aidar
    python convert.py --input book.pdf --output audiobook.m4b --language en
    python convert.py --input book.fb2 --output out.m4b --rvc-model MyVoice
    python convert.py --input book.fb2 --output out.m4b --rvc-model path/to/model.pth --pitch-shift -2
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
        description="Convert ebook (FB2/EPUB/PDF) to M4B audiobook using Silero TTS v4.",
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
        "--speaker", "-s",
        default="xenia",
        metavar="NAME",
        help="Silero speaker: xenia, baya, kseniya, aidar, eugene (default: xenia)",
    )
    parser.add_argument(
        "--language", "-l",
        default="ru",
        metavar="LANG",
        help='TTS language code (default: "ru")',
    )
    parser.add_argument(
        "--rvc-model",
        default=None,
        metavar="NAME_OR_PATH",
        help="RVC voice: a profile name from ./voices/ or a path to a .pth file",
    )
    parser.add_argument(
        "--rvc-index",
        default=None,
        metavar="PATH",
        help="Path to RVC .index file (optional, used with --rvc-model path)",
    )
    parser.add_argument(
        "--pitch-shift",
        type=int,
        default=0,
        metavar="N",
        help="RVC pitch shift in semitones (default: 0)",
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
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def compute_cache_key(book_path: Path, engine_type: str, speaker: str, rvc_model_path: Path | None, pitch_shift: int = 0) -> str:
    book_hash = compute_book_hash(book_path)
    if rvc_model_path:
        from src.rvc.converter import RVCConverter
        model_hash = RVCConverter.compute_model_hash(rvc_model_path)
        return f"{book_hash}_{engine_type}_{speaker}_{model_hash}_p{pitch_shift}"
    return f"{book_hash}_{engine_type}_{speaker}"


# ---------------------------------------------------------------------------
# Core conversion pipeline
# ---------------------------------------------------------------------------

def convert(
    input_path: Path,
    output_path: Path,
    speaker: str,
    language: str,
    rvc_model_path: Path | None,
    rvc_index_path: Path | None,
    pitch_shift: int,
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
    cache_key = compute_cache_key(input_path, speaker, rvc_model_path, pitch_shift)
    cache_dir = cache_base / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Cache directory: %s", cache_dir)

    # --- Load TTS engine ---
    logger.info("Loading Silero TTS (speaker: %s)...", speaker)
    from src.tts.engine import TTSEngine
    from src.tts.chunker import TextChunker

    engine = TTSEngine(language=language, speaker=speaker)
    chunker = TextChunker(max_words=max_words)

    # --- Load RVC converter (optional) ---
    rvc_converter = None
    if rvc_model_path:
        logger.info("Loading RVC model: %s (pitch: %+d)", rvc_model_path, pitch_shift)
        from src.rvc.converter import RVCConverter
        rvc_converter = RVCConverter(
            model_path=rvc_model_path,
            index_path=rvc_index_path,
            pitch_shift=pitch_shift,
        )

    # --- Synthesize chunks ---
    chapter_wav_lists: list[list[Path]] = []
    chapter_titles: list[str] = []

    all_chunks: list[tuple[int, int, str]] = []
    for ch_idx, chapter in enumerate(book.chapters):
        chunks = chunker.split(chapter.text)
        for ck_idx, chunk_text in enumerate(chunks):
            all_chunks.append((ch_idx, ck_idx, chunk_text))

    logger.info("Total TTS chunks: %d", len(all_chunks))

    for chapter in book.chapters:
        chapter_wav_lists.append([])
        chapter_titles.append(chapter.title)

    desc = "Synthesizing + RVC" if rvc_converter else "Synthesizing"
    with tqdm(total=len(all_chunks), desc=desc, unit="chunk") as pbar:
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
                if rvc_converter:
                    rvc_converter.convert_in_place(wav_path)

            chapter_wav_lists[ch_idx].append(wav_path)
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

    filtered_wavs = [wl for wl in chapter_wav_lists if wl]
    filtered_titles = [t for t, wl in zip(chapter_titles, chapter_wav_lists) if wl]

    builder.build(filtered_wavs, filtered_titles)

    if rvc_converter:
        rvc_converter.cleanup()

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

    # Resolve RVC model path
    rvc_model_path = None
    rvc_index_path = None
    if args.rvc_model:
        candidate = Path(args.rvc_model)
        if candidate.exists() and candidate.suffix == ".pth":
            rvc_model_path = candidate
            if args.rvc_index:
                rvc_index_path = Path(args.rvc_index)
        else:
            # Treat as a voice profile name
            from src.rvc.voice_manager import VoiceManager
            vm = VoiceManager()
            model_pth, model_idx = vm.get_model_paths(args.rvc_model)
            if model_pth:
                rvc_model_path = model_pth
                rvc_index_path = model_idx
            else:
                logger.error("RVC model not found: %s (not a file path or voice profile)", args.rvc_model)
                sys.exit(1)

    cache_base = Path(args.cache_dir).resolve()

    convert(
        input_path=input_path,
        output_path=output_path,
        speaker=args.speaker,
        language=args.language,
        rvc_model_path=rvc_model_path,
        rvc_index_path=rvc_index_path,
        pitch_shift=args.pitch_shift,
        cache_base=cache_base,
        max_words=args.max_words,
        pages_per_chapter=args.pages_per_chapter,
    )


if __name__ == "__main__":
    main()
