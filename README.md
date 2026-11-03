# audiobook-tts

Turn an ebook (FB2, EPUB, PDF) into an M4B audiobook with local text-to-speech on Apple Silicon. No cloud services, no API costs: parsing, synthesis, voice conversion and mastering all run on the Mac.

Built for Russian-language books first (stress marks, ё normalization, homograph resolution), with multilingual engines available.

## What it does

- Parses FB2, EPUB and PDF into chapters.
- Normalizes text for speech: numbers, dates, abbreviations, Roman numerals, stress marks via `silero-stress` / `ruaccent`, ё restoration, homograph resolution.
- Splits into sentence-boundary chunks and synthesizes with one of five engines.
- Optional RVC voice conversion on top of Silero (train or import a voice, including from a YouTube URL).
- Masters each chapter with ffmpeg (highpass, EQ, de-esser, compressor, loudnorm to -16 LUFS) and builds a chaptered M4B.
- Caches synthesized WAV chunks per book/engine/voice, so re-runs only synthesize what changed.

## Engines

| Engine | Backend | Notes |
|---|---|---|
| Silero v4 | PyTorch | Fast, five Russian speakers, no cloning. Default for the CLI. |
| F5-TTS | PyTorch MPS or `f5-tts-mlx` | Voice cloning from a 5–12 s reference clip, Russian fine-tune. |
| Chatterbox | PyTorch MPS or `mlx-audio` | Multilingual, voice cloning. |
| Qwen3-TTS | `mlx-audio` | Multilingual voice cloning, best quality on Apple Silicon. |

All engines share one pipeline (`src/tts/base.py`): normalize → accentuate → split → synthesize → speed → pauses → crossfade → volume normalization. An engine only implements `_synthesize_chunk()`.

## Requirements

- macOS on Apple Silicon (tested on M4 Max, 64 GB). Metal acceleration via PyTorch MPS and MLX.
- Python 3.11+
- ffmpeg: `brew install ffmpeg`

Models download on first run from Hugging Face (Silero ~38 MB, F5-TTS ~1.3 GB, Qwen3-TTS 0.6B and up).

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Use

Web UI (all engines, voice management, settings persist in `ui_settings.json`):

```bash
python app.py
# open http://localhost:7860
```

CLI (Silero, optional RVC):

```bash
python convert.py --input book.fb2 --output audiobook.m4b
python convert.py --input book.fb2 --output audiobook.m4b --speaker aidar
python convert.py --input book.epub --output out.m4b --rvc-model my-voice --pitch-shift -2
python convert.py --help
```

## Layout

```
convert.py          CLI entry point
app.py              Gradio web UI
src/parser/         FB2, EPUB, PDF parsers
src/text/           ё normalizer, homograph resolver, semantic chunker, stress
src/tts/            engines, chunker, normalizer, accentuator
src/rvc/            RVC inference, training wrapper, voice manager
src/audio/          M4B builder, enhancer, mastering
tools/              benchmarks and experiments
data/               stress dictionary and overrides
```

`voices/`, `cache/` and `output/` are created locally and ignored by git.

## Known issues

- RVC inference on MPS can hang on long sessions; it falls back to CPU.
- `rvc-python` is installed with `--no-deps` because of `omegaconf` / `fairseq` conflicts; `fairseq` is patched for PyTorch 2.6+ (`weights_only=True`).
- Loudness normalization is disabled for the Qwen3 cinematic pipeline until the loudnorm zeroing bug is fixed.

## License

MIT
