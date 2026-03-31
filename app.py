#!/usr/bin/env python3
"""Gradio web UI for audiobook-tts.

Run:
    python app.py

Then open http://localhost:7860 in your browser.

Features:
- Drag & drop book upload (FB2, EPUB, PDF)
- Optional voice sample upload for cloning
- Language selector
- Progress display during conversion
- Download link for the completed M4B
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Generator, Optional

import gradio as gr

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_DIR = Path("./cache")
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_FORMATS = [".fb2", ".epub", ".pdf"]

LANGUAGES = {
    "Russian (ru)": "ru",
    "English (en)": "en",
    "German (de)": "de",
    "French (fr)": "fr",
    "Spanish (es)": "es",
    "Italian (it)": "it",
    "Portuguese (pt)": "pt",
    "Polish (pl)": "pl",
    "Turkish (tr)": "tr",
    "Chinese (zh-cn)": "zh-cn",
    "Japanese (ja)": "ja",
    "Korean (ko)": "ko",
    "Arabic (ar)": "ar",
    "Czech (cs)": "cs",
    "Dutch (nl)": "nl",
    "Hungarian (hu)": "hu",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Core conversion function (yields progress messages)
# ---------------------------------------------------------------------------

def run_conversion(
    book_file: str | None,
    voice_file: str | None,
    output_name: str,
    language_label: str,
    max_words: int,
) -> Generator[tuple[str, str | None], None, None]:
    """Run the full conversion pipeline, yielding (log_message, output_path).

    This is a generator so Gradio can stream progress updates.
    """

    # Validate inputs
    if not book_file:
        yield "Error: No book file uploaded.", None
        return

    book_path = Path(book_file)
    if book_path.suffix.lower() not in SUPPORTED_FORMATS:
        yield f"Error: Unsupported format '{book_path.suffix}'. Use FB2, EPUB, or PDF.", None
        return

    if not output_name.strip():
        output_name = book_path.stem

    if not output_name.endswith(".m4b"):
        output_name = output_name.strip() + ".m4b"

    output_path = OUTPUT_DIR / output_name
    language = LANGUAGES.get(language_label, "ru")
    voice_path = Path(voice_file) if voice_file else None

    # --- Parse book ---
    yield "Parsing book...", None
    try:
        from src.parser import parse_book
        from src.parser.pdf import PDFParser

        if book_path.suffix.lower() == ".pdf":
            book = PDFParser().parse(book_path)
        else:
            book = parse_book(book_path)
    except Exception as e:
        yield f"Error parsing book: {e}", None
        return

    yield (
        f"Parsed: '{book.title}' by {book.author} — {len(book.chapters)} chapters.",
        None,
    )

    if not book.chapters:
        yield "Error: No chapters found in book.", None
        return

    # --- Load TTS engine ---
    yield "Loading XTTS v2 model (first run may download ~2 GB)...", None
    try:
        from src.tts.engine import TTSEngine
        from src.tts.chunker import TextChunker

        engine = TTSEngine(voice_sample=voice_path, language=language)
        chunker = TextChunker(max_words=max_words)
    except Exception as e:
        yield f"Error loading TTS engine: {e}", None
        return

    yield "TTS engine loaded. Starting synthesis...", None

    # --- Cache setup ---
    book_hash = compute_hash(book_path)
    cache_dir = CACHE_DIR / book_hash
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Build flat list of all chunks
    all_chunks: list[tuple[int, int, str]] = []
    for ch_idx, chapter in enumerate(book.chapters):
        for ck_idx, text in enumerate(chunker.split(chapter.text)):
            all_chunks.append((ch_idx, ck_idx, text))

    total = len(all_chunks)
    chapter_wav_lists: list[list[Path]] = [[] for _ in book.chapters]
    chapter_titles = [ch.title for ch in book.chapters]

    yield f"Total chunks to synthesize: {total}", None

    # --- Synthesize ---
    done = 0
    for ch_idx, ck_idx, chunk_text in all_chunks:
        wav_path = cache_dir / f"ch{ch_idx:04d}_ck{ck_idx:04d}.wav"

        if wav_path.exists() and wav_path.stat().st_size > 0:
            status = "(cached)"
        else:
            try:
                engine.synthesize(chunk_text, wav_path)
                status = ""
            except Exception as e:
                yield f"Error synthesizing chunk {done + 1}/{total}: {e}", None
                return

        chapter_wav_lists[ch_idx].append(wav_path)
        done += 1

        if done % 5 == 0 or done == total:
            pct = int(done / total * 100)
            yield f"Synthesizing... {done}/{total} chunks ({pct}%) {status}", None

    # --- Build M4B ---
    yield "Building M4B audiobook...", None
    try:
        from src.audio.builder import AudioBuilder

        builder = AudioBuilder(
            output_path=output_path,
            title=book.title,
            author=book.author,
            cover_bytes=book.cover_bytes,
        )

        filtered_wavs = [wl for wl in chapter_wav_lists if wl]
        filtered_titles = [
            t for t, wl in zip(chapter_titles, chapter_wav_lists) if wl
        ]
        builder.build(filtered_wavs, filtered_titles)
    except Exception as e:
        yield f"Error building M4B: {e}", None
        return

    yield f"Done! Audiobook saved: {output_path}", str(output_path)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="audiobook-tts",
        theme=gr.themes.Default(
            primary_hue="indigo",
            neutral_hue="slate",
        ),
    ) as demo:
        gr.Markdown(
            """
            # audiobook-tts
            Convert FB2 / EPUB / PDF ebooks to M4B audiobooks using **XTTS v2** — fully local, no cloud needed.
            Runs on Apple Silicon via **Metal (MPS)** acceleration.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                book_input = gr.File(
                    label="Book file (FB2, EPUB, PDF)",
                    file_types=[".fb2", ".epub", ".pdf"],
                    type="filepath",
                )
                voice_input = gr.File(
                    label="Voice sample for cloning — optional (WAV or MP3, 5–30s)",
                    file_types=[".wav", ".mp3"],
                    type="filepath",
                )
                language_dropdown = gr.Dropdown(
                    label="Language",
                    choices=list(LANGUAGES.keys()),
                    value="Russian (ru)",
                )
                output_name_input = gr.Textbox(
                    label="Output filename",
                    placeholder="audiobook.m4b",
                    value="",
                )
                max_words_slider = gr.Slider(
                    label="Max words per TTS chunk",
                    minimum=50,
                    maximum=400,
                    step=10,
                    value=200,
                    info="Smaller = more chunks, less memory. Larger = fewer chunks, better prosody.",
                )
                convert_btn = gr.Button("Convert to M4B", variant="primary", size="lg")

            with gr.Column(scale=1):
                progress_log = gr.Textbox(
                    label="Progress log",
                    lines=20,
                    max_lines=40,
                    interactive=False,
                    placeholder="Progress will appear here...",
                )
                download_output = gr.File(
                    label="Download audiobook",
                    interactive=False,
                    visible=False,
                )

        # State for accumulated log lines
        log_state = gr.State([])

        def on_convert(book_file, voice_file, output_name, language_label, max_words, log_lines):
            log_lines = []
            final_file = None

            for message, file_path in run_conversion(
                book_file=book_file,
                voice_file=voice_file,
                output_name=output_name,
                language_label=language_label,
                max_words=int(max_words),
            ):
                log_lines.append(message)
                log_text = "\n".join(log_lines)
                if file_path:
                    final_file = file_path
                yield log_text, gr.update(visible=bool(final_file), value=final_file), log_lines

        convert_btn.click(
            fn=on_convert,
            inputs=[
                book_input,
                voice_input,
                output_name_input,
                language_dropdown,
                max_words_slider,
                log_state,
            ],
            outputs=[progress_log, download_output, log_state],
        )

        gr.Markdown(
            """
            ---
            **Tips:**
            - First run downloads the XTTS v2 model (~2 GB) to `~/.local/share/tts/`
            - Voice cloning works best with a clean, quiet 5–30 second WAV sample
            - Chunks are cached in `./cache/` — interrupted conversions resume automatically
            - Output M4B files are saved to `./output/`
            - Requires `ffmpeg` installed: `brew install ffmpeg`
            """
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
