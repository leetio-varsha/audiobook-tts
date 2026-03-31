# audiobook-tts

Personal tool for converting ebooks to audiobooks using a local TTS model. No cloud services, no API costs — everything runs locally on Mac.

## Hardware

- Chip: Apple M4 Max
- RAM: 64 GB Unified Memory
- Acceleration: Metal (MPS) via PyTorch

## Stack

- Language: Python 3.11+
- TTS engine: Coqui TTS (XTTS v2)
- Acceleration: PyTorch + MPS (Apple Metal)
- FB2 parsing: lxml
- EPUB parsing: ebooklib + BeautifulSoup4
- PDF parsing: pdfminer.six
- Audio concatenation: ffmpeg
- Web interface: Gradio (localhost:7860)
- Dependencies: pip + venv

## Project Structure

```
audiobook-tts/
├── CLAUDE.md
├── .gitignore
├── requirements.txt
├── convert.py          # CLI entry point
├── app.py              # Gradio web UI
├── src/
│   ├── __init__.py
│   ├── parser/
│   │   ├── __init__.py
│   │   ├── fb2.py      # FB2 parser (lxml)
│   │   ├── epub.py     # EPUB parser (ebooklib + BS4)
│   │   └── pdf.py      # PDF parser (pdfminer.six)
│   ├── tts/
│   │   ├── __init__.py
│   │   ├── engine.py   # XTTS v2 engine with MPS
│   │   └── chunker.py  # Sentence-boundary text chunker
│   └── audio/
│       ├── __init__.py
│       └── builder.py  # M4B builder via ffmpeg
├── cache/              # WAV chunk cache (gitignored)
├── output/             # Final M4B files (gitignored)
└── voices/             # Voice sample WAVs (gitignored)
```

## Usage

### CLI

```bash
python convert.py --input book.fb2 --output audiobook.m4b
python convert.py --input book.epub --output audiobook.m4b --voice voices/sample.wav
python convert.py --input book.pdf --output audiobook.m4b --voice voices/sample.wav --language en
```

### Web UI

```bash
python app.py
# Open http://localhost:7860
```

## Key Design Decisions

### MPS Acceleration
TTS inference uses `torch.device("mps")` on Apple Silicon for Metal GPU acceleration. Falls back to CPU if MPS is unavailable.

### WAV Chunk Cache
Each chunk is cached to `./cache/{book_hash}/{chunk_index:04d}.wav`. On resume, already-cached chunks are skipped. The book hash is an MD5 of the book content.

### Text Chunking
Text is split at sentence boundaries (`. `, `! `, `? `, `.\n`) and accumulated until ~200 words. Sentences are never split mid-way.

### M4B Output
Final output is AAC 128 kbps, 24000 Hz, with chapter markers embedded via ffmpeg metadata format. Cover art and title/author metadata are embedded when available.

### Supported Formats
- **FB2**: FictionBook XML — chapters from `<section>` elements with `<title>`
- **EPUB**: Chapters from spine + TOC navigation
- **PDF**: Pseudo-chapters every 20 pages

## Dependencies

Install:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### requirements.txt content
- TTS (Coqui XTTS v2)
- torch (with MPS support — use the standard PyTorch wheel on macOS)
- lxml
- ebooklib
- beautifulsoup4
- pdfminer.six
- gradio
- tqdm

## Notes

- First run downloads the XTTS v2 model (~2 GB) to `~/.local/share/tts/`
- Voice cloning requires a clean 5–30 second WAV sample
- Language defaults to Russian (`ru`) — change `self.language` in `src/tts/engine.py`
- ffmpeg must be installed: `brew install ffmpeg`
