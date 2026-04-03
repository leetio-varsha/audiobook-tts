# audiobook-tts

Personal tool for converting ebooks to audiobooks using local TTS models. No cloud services, no API costs — everything runs locally on Mac.

## Hardware

- Chip: Apple M4 Max
- RAM: 64 GB Unified Memory
- Acceleration: Metal (MPS) via PyTorch

## Stack

- Language: Python 3.11+
- TTS engines: Silero v4 (Russian, fast) / F5-TTS (Russian, voice cloning) / Chatterbox (multilingual) / MLX-Chatterbox (Apple Silicon native) / Qwen3-TTS (MLX, multilingual)
- Voice conversion: RVC (optional, Silero only)
- Text preprocessing: ruaccent (stress marks) + custom normalizer (numbers, dates, abbreviations)
- Audio post-processing: ffmpeg (EQ, compression, loudness normalization)
- Acceleration: PyTorch + MPS (Apple Metal)
- Parsers: lxml (FB2), ebooklib + BS4 (EPUB), pdfminer.six (PDF)
- Audio: ffmpeg + soundfile
- Web interface: Gradio (localhost:7860)
- Dependencies: pip + venv

## Project Structure

```
audiobook-tts/
├── CLAUDE.md
├── .gitignore
├── requirements.txt
├── ui_settings.json     # Persisted UI settings
├── convert.py           # CLI entry point
├── app.py               # Gradio web UI
├── src/
│   ├── __init__.py
│   ├── parser/
│   │   ├── fb2.py       # FB2 parser (lxml)
│   │   ├── epub.py      # EPUB parser (ebooklib + BS4)
│   │   └── pdf.py       # PDF parser (pdfminer.six)
│   ├── tts/
│   │   ├── base.py      # BaseTTSEngine — abstract interface with shared pipeline
│   │   ├── engine.py    # Engine factory: create_engine("silero"|"f5tts")
│   │   ├── engine_silero.py  # Silero v4 — fast, 5 Russian speakers
│   │   ├── engine_f5tts.py   # F5-TTS — voice cloning, high quality
│   │   ├── engine_mlx_f5tts.py   # F5-TTS via f5-tts-mlx — Apple Silicon native
│   │   ├── engine_chatterbox.py  # Chatterbox — multilingual, PyTorch MPS
│   │   ├── engine_mlx_chatterbox.py # Chatterbox via mlx-audio — Apple Silicon native
│   │   ├── engine_qwen3tts.py    # Qwen3-TTS via mlx-audio — multilingual, voice cloning
│   │   ├── chunker.py   # Sentence-boundary text chunker
│   │   ├── normalizer.py # Russian text normalizer (numbers, dates, etc.)
│   │   └── accentuator.py # ruaccent stress mark preprocessor
│   ├── rvc/
│   │   ├── converter.py  # RVC inference wrapper (rvc-python)
│   │   ├── voice_manager.py # Voice import, YouTube download, model registry
│   │   ├── audio_prep.py # Audio preprocessing (normalize, trim, split)
│   │   └── trainer.py    # RVC training wrapper (calls RVC-WebUI scripts)
│   └── audio/
│       ├── builder.py    # M4B builder via ffmpeg
│       └── enhancer.py   # Audio post-processing (EQ, compression, loudnorm)
├── tools/
│   └── rvc-webui/       # Cloned RVC-WebUI for training (gitignored)
├── voices/              # Voice profiles with RVC models (gitignored)
├── cache/               # WAV chunk cache (gitignored)
└── output/              # Final M4B files (gitignored)
```

## TTS Engine Architecture

```
BaseTTSEngine (base.py) — shared pipeline:
  normalize → accentuate → split → synthesize → speed → pauses → crossfade → volume norm
  │
  ├── SileroEngine — fast, 5 Russian speakers, no voice cloning
  │     Speakers: xenia (default), baya, kseniya, aidar, eugene
  │
  ├── F5TTSEngine — Misha24-10/F5-TTS_RUSSIAN, voice cloning from reference audio
  │     Model: 1.3GB, auto-downloaded from HuggingFace
  │     Reference audio: 5-8s WAV, auto-trimmed from middle
  │
  ├── MLXF5TTSEngine — F5-TTS via f5-tts-mlx, native Apple Silicon
  │     Variants: russian (default), base, base-q4, base-q8
  │     Same quality as F5TTSEngine but runs on Metal GPU, not CPU
  │
  ├── ChatterboxEngine — resemble-ai/chatterbox, multilingual, PyTorch MPS
  │     550M params, voice cloning, 12 languages incl. Russian
  │
  ├── MLXChatterboxEngine — Chatterbox via mlx-audio, native Apple Silicon
  │     Variants: turbo (134M, fast), full-q4 (550M quantized, Russian)
  │     Uses MLX Metal acceleration instead of PyTorch MPS
  │
  └── Qwen3TTSEngine — Qwen3-TTS via mlx-audio, native Apple Silicon
        Variants: base-0.6b (voice cloning, 10 langs), custom-1.7b, design-1.7b
        Russian supported in base variant. Uses MLX Metal acceleration
```

## Usage

### CLI

```bash
python convert.py --input book.fb2 --output audiobook.m4b
python convert.py --input book.fb2 --output audiobook.m4b --speaker aidar
python convert.py --input book.fb2 --output out.m4b --rvc-model Buldakov --pitch-shift -2
```

### Web UI

```bash
python app.py
# Open http://localhost:7860
```

## Key Design Decisions

### Multi-Engine Architecture
`BaseTTSEngine` interface with shared preprocessing pipeline. Engines implement only `_synthesize_chunk()`. All features (normalizer, pauses, crossfade, volume norm, enhancer) work with any engine.

### Text Normalization Pipeline
1. **Normalizer** (normalizer.py): numbers → words, dates, abbreviations, symbols, Roman numerals
2. **Accentuator** (accentuator.py): ruaccent stress marks for proper Russian pronunciation
3. **Chunker** (chunker.py): sentence-boundary splitting, hard-enforced character limits per engine

### Audio Post-Processing
Per-chapter ffmpeg filter chain: highpass(100Hz) → EQ(-3dB@6kHz) → de-esser → compressor → loudnorm(-16 LUFS)

### WAV Chunk Cache
Cache key: `{book_hash}_{engine}_{speaker}_{rvc_hash}_{pitch}`. Changing any parameter creates a fresh cache.

### RVC Voice Conversion
Optional layer on top of Silero. Not needed with F5-TTS (has built-in voice cloning). faiss-cpu causes segfaults on macOS — index disabled, inference only.

### Known Issues
- RVC MPS inference can hang on long sessions — falls back to CPU
- rvc-python installed with --no-deps due to omegaconf/fairseq conflicts
- fairseq patched for weights_only=True (PyTorch 2.6+ compatibility)

## Notes

- First run downloads models (~1.3GB F5-TTS, ~38MB Silero) from HuggingFace
- F5-TTS voice cloning: upload 5-12s WAV reference or extract from YouTube URL
- ffmpeg required: `brew install ffmpeg`
- Settings persist across sessions in `ui_settings.json`
