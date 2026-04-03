#!/usr/bin/env python3
"""Gradio web UI for audiobook-tts — convert ebooks to M4B audiobooks locally."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Generator

import gradio as gr

from src.rvc.voice_manager import VoiceManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("app")

CACHE_DIR = Path("./cache").resolve()
OUTPUT_DIR = Path("./output").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_FORMATS = [".fb2", ".epub", ".pdf"]

def _subprocess_env() -> dict:
    """Return env dict with venv and /opt/homebrew/bin on PATH for yt-dlp, ffmpeg, etc."""
    import os as _os
    env = _os.environ.copy()
    venv_bin = str(Path(__file__).parent / ".venv" / "bin")
    for p in (venv_bin, "/opt/homebrew/bin", "/usr/local/bin"):
        if p not in env.get("PATH", ""):
            env["PATH"] = p + ":" + env.get("PATH", "")
    return env

SILERO_SPEAKERS = {
    "Xenia (жен., аудиокнижный)": "xenia",
    "Baya (жен., нейтральный)": "baya",
    "Kseniya (жен., экспрессивный)": "kseniya",
    "Aidar (муж., нейтральный)": "aidar",
    "Eugene (муж., экспрессивный)": "eugene",
}

TTS_ENGINES = {
    "Silero v5 — быстрый, 5 голосов": "silero",
    "F5-TTS — voice cloning": "f5tts",
    "Chatterbox — voice cloning, MPS": "chatterbox",
    "Qwen3-TTS — MLX, multilingual": "qwen3tts",
}

MLX_CHATTERBOX_VARIANTS = {
    "Multilingual Q4 (дефолт, русский)": "multi-q4",
    "Full FP16 (без квантизации)": "full-fp16",
    "Full Q8 (баланс)": "full-q8",
    "Full Q4 (меньше RAM)": "full-q4",
    "Turbo FP16 (быстрый, English)": "turbo",
}


QWEN3_VARIANTS = {
    "Base 0.6B — voice cloning (дефолт)": "base-0.6b",
    "Base 0.6B Q4 — меньше RAM": "base-0.6b-q4",
    "Custom 1.7B — predefined speakers": "custom-1.7b",
    "Design 1.7B — voice from description": "design-1.7b",
}

LANGUAGES = {"Russian": "ru", "English": "en", "German": "de", "French": "fr", "Spanish": "es", "Italian": "it", "Portuguese": "pt", "Polish": "pl", "Turkish": "tr"}

RVC_NONE = "Без RVC"
SETTINGS_PATH = Path("./ui_settings.json")
voice_manager = VoiceManager()

# Engine cache — avoid reloading heavy models on every preview/convert
_engine_cache: dict[str, object] = {}  # key → engine instance
_engine_cache_key: str = ""

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_PATH.read_text()) if SETTINGS_PATH.exists() else {}
    except (json.JSONDecodeError, OSError):
        return {}

def save_settings(**kw) -> None:
    s = load_settings()
    s.update(kw)
    SETTINGS_PATH.write_text(json.dumps(s, indent=2, ensure_ascii=False))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()

def compute_cache_key(
    book_path: Path, engine_type: str, speaker: str,
    rvc_model_path: Path | None = None, pitch_shift: int = 0,
    nfe_steps: int = 0, model_variant: str = "", ref_audio_file: str | None = None,
    ref_text: str = "", cfg_strength: float = 0.0, speech_speed: float = 1.0,
) -> str:
    bh = compute_hash(book_path)
    nfe_tag = f"_nfe{nfe_steps}" if nfe_steps and engine_type == "f5tts" else ""
    variant_tag = f"_v{model_variant}" if model_variant else ""
    ref_tag = ""
    extra_tag = ""
    if engine_type == "mlx-chatterbox":
        if ref_audio_file and Path(ref_audio_file).exists():
            ref_tag = f"_ref{compute_hash(Path(ref_audio_file))[:8]}"
        else:
            ref_tag = "_refdefault"
    elif engine_type == "f5tts":
        if ref_audio_file and Path(ref_audio_file).exists():
            ref_tag = f"_ref{compute_hash(Path(ref_audio_file))[:8]}"
        else:
            ref_tag = "_refdefault"
        if ref_text and ref_text.strip():
            ref_text_hash = hashlib.md5(ref_text.strip().encode("utf-8")).hexdigest()[:8]
            extra_tag += f"_rt{ref_text_hash}"
        extra_tag += f"_cfg{cfg_strength:.2f}_spd{speech_speed:.2f}"
    if rvc_model_path:
        from src.rvc.converter import RVCConverter
        mh = RVCConverter.compute_model_hash(rvc_model_path)
        return f"{bh}_{engine_type}_{speaker}_{mh}_p{pitch_shift}{nfe_tag}{variant_tag}{ref_tag}{extra_tag}"
    return f"{bh}_{engine_type}_{speaker}{nfe_tag}{variant_tag}{ref_tag}{extra_tag}"

def get_cached_engine(engine_type: str, **kwargs):
    """Return a cached engine or create a new one if params changed."""
    global _engine_cache, _engine_cache_key
    # Build a cache key from engine type + relevant params
    import json as _json
    # Sort kwargs for stable key; skip non-hashable or large values
    key_parts = {"engine": engine_type}
    for k, v in sorted(kwargs.items()):
        if k in ("ref_audio",):
            key_parts[k] = str(v) if v else ""
        elif isinstance(v, (str, int, float, bool)):
            key_parts[k] = v
    cache_key = _json.dumps(key_parts, sort_keys=True)

    if cache_key == _engine_cache_key and "engine" in _engine_cache:
        logger.info("Reusing cached %s engine", engine_type)
        return _engine_cache["engine"]

    from src.tts.engine import create_engine
    engine = create_engine(engine_type, **kwargs)
    _engine_cache["engine"] = engine
    _engine_cache_key = cache_key
    return engine


def get_rvc_choices() -> list[str]:
    return [RVC_NONE] + voice_manager.list_voice_names()

# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def run_conversion(
    book_file, output_name, engine_label, language_label, speaker_label,
    ref_audio_file, ref_text, rvc_voice_name, pitch_shift, speech_speed,
    max_words, nfe_steps, cfg_strength, model_variant_label, test_chunks=0,
    enhance=True,
) -> Generator[tuple[str, str | None], None, None]:

    if not book_file:
        yield "Upload a book file first.", None
        return

    book_path = Path(book_file)
    if book_path.suffix.lower() not in SUPPORTED_FORMATS:
        yield f"Unsupported format: {book_path.suffix}", None
        return

    if not output_name.strip():
        output_name = book_path.stem
    if not output_name.endswith(".m4b"):
        output_name = output_name.strip() + ".m4b"

    output_path = OUTPUT_DIR / output_name
    language = LANGUAGES.get(language_label, "ru")
    speaker = SILERO_SPEAKERS.get(speaker_label, "xenia")
    engine_type = TTS_ENGINES.get(engine_label, "silero")

    # Parse
    yield "Parsing book...", None
    try:
        from src.parser import parse_book
        from src.parser.pdf import PDFParser
        book = PDFParser().parse(book_path) if book_path.suffix.lower() == ".pdf" else parse_book(book_path)
    except Exception as e:
        yield f"Parse error: {e}", None
        return

    if not book.chapters:
        yield "No chapters found.", None
        return

    yield f"'{book.title}' by {book.author} — {len(book.chapters)} chapters", None

    # Resolve model variant
    model_variant = ""
    if engine_type == "mlx-chatterbox":
        model_variant = MLX_CHATTERBOX_VARIANTS.get(model_variant_label, "multi-q4")
    elif engine_type == "qwen3tts":
        model_variant = QWEN3_VARIANTS.get(model_variant_label, "base-0.6b")

    # Engine
    yield f"Loading {engine_type}...", None
    try:
        from src.tts.engine import create_engine
        from src.tts.chunker import TextChunker

        engine_kwargs = dict(language=language, speech_speed=float(speech_speed))
        if engine_type == "silero":
            engine_kwargs["speaker"] = speaker
        elif engine_type == "f5tts":
            if ref_audio_file:
                engine_kwargs["ref_audio"] = ref_audio_file
            if ref_text and ref_text.strip():
                engine_kwargs["ref_text"] = ref_text.strip()
            engine_kwargs["nfe_steps"] = int(nfe_steps)
            engine_kwargs["cfg_strength"] = float(cfg_strength)
        elif engine_type == "chatterbox":
            if ref_audio_file:
                engine_kwargs["ref_audio"] = ref_audio_file
        elif engine_type == "mlx-chatterbox":
            if ref_audio_file:
                engine_kwargs["ref_audio"] = ref_audio_file
            engine_kwargs["model_variant"] = model_variant
        elif engine_type == "qwen3tts":
            if ref_audio_file:
                engine_kwargs["ref_audio"] = ref_audio_file
            if ref_text and ref_text.strip():
                engine_kwargs["ref_text"] = ref_text.strip()
            engine_kwargs["model_variant"] = model_variant

        engine = get_cached_engine(engine_type, **engine_kwargs)
        chunker = TextChunker(max_words=int(max_words))
    except Exception as e:
        yield f"Engine error: {e}", None
        return

    yield f"{engine.engine_name} ready", None
    if engine_type == "mlx-chatterbox" and not ref_audio_file:
        yield "MLX Chatterbox: no voice reference uploaded, using generated fallback prompt", None

    # RVC (Silero only)
    rvc_converter = None
    rvc_model_path = None
    if engine_type == "silero" and rvc_voice_name and rvc_voice_name != RVC_NONE:
        model_pth, model_idx = voice_manager.get_model_paths(rvc_voice_name)
        if model_pth:
            yield f"Loading RVC: {rvc_voice_name}...", None
            try:
                from src.rvc.converter import RVCConverter
                rvc_converter = RVCConverter(model_path=model_pth, index_path=model_idx, pitch_shift=int(pitch_shift))
                rvc_model_path = model_pth
            except Exception as e:
                yield f"RVC failed: {e}, continuing without", None

    # Cache
    cache_key = compute_cache_key(
        book_path, engine_type, speaker, rvc_model_path, int(pitch_shift),
        int(nfe_steps) if engine_type == "f5tts" else 0,
        model_variant, ref_audio_file, ref_text, float(cfg_strength), float(speech_speed),
    )
    cache_dir = CACHE_DIR / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    chunk_pauses = []  # pause_after_ms for each chunk
    for ch_idx, chapter in enumerate(book.chapters):
        rich_chunks = chunker.split_rich(chapter.text)
        for ck_idx, rc in enumerate(rich_chunks):
            all_chunks.append((ch_idx, ck_idx, rc.text))
            # Add extra pause at dialogue↔narration transitions
            pause = rc.pause_after_ms
            if ck_idx > 0 and rich_chunks[ck_idx - 1].is_dialogue != rc.is_dialogue:
                pause = max(pause, 500)  # speaker change pause
            chunk_pauses.append(pause)

    # Truncate for testing
    test_limit = int(test_chunks) if test_chunks else 0
    if test_limit > 0:
        all_chunks = all_chunks[:test_limit]
        chunk_pauses = chunk_pauses[:test_limit]

    total = len(all_chunks)
    chapter_wav_lists = [[] for _ in book.chapters]
    chapter_pause_lists = [[] for _ in book.chapters]
    chapter_titles = [ch.title for ch in book.chapters]

    test_tag = f" (test: {test_limit} chunks)" if test_limit > 0 else ""
    yield f"Synthesizing {total} chunks{test_tag}...", None
    start_time = time.time()

    # Synthesize
    cached_count = 0
    synth_times = []
    prev_ch_idx = -1
    for done, (ch_idx, ck_idx, chunk_text) in enumerate(all_chunks, 1):
        wav_path = cache_dir / f"ch{ch_idx:04d}_ck{ck_idx:04d}.wav"

        if wav_path.exists() and wav_path.stat().st_size > 1000:
            cached_count += 1
            status = "⚡ cache"
            chunk_time = 0
        else:
            try:
                chunk_start = time.time()
                engine.synthesize(chunk_text, wav_path)
                if rvc_converter:
                    rvc_converter.convert_in_place(wav_path)
                chunk_time = time.time() - chunk_start
                synth_times.append(chunk_time)
                status = f"✓ {chunk_time:.0f}s"
            except Exception as e:
                yield f"❌ Chunk {done}/{total}: {e}", None
                return

        chapter_wav_lists[ch_idx].append(wav_path)
        chapter_pause_lists[ch_idx].append(chunk_pauses[done - 1] if done - 1 < len(chunk_pauses) else 150)

        elapsed = time.time() - start_time
        # ETA based on average synth time (skip cached chunks)
        remaining = total - done
        if synth_times:
            avg_time = sum(synth_times) / len(synth_times)
            # Estimate how many remaining will need synthesis
            cache_ratio = cached_count / done
            remaining_synth = int(remaining * (1 - cache_ratio))
            eta = remaining_synth * avg_time
        else:
            eta = 0
        eta_min = int(eta // 60)
        eta_sec = int(eta % 60)
        elapsed_min = int(elapsed // 60)
        elapsed_sec = int(elapsed % 60)

        progress_pct = done * 100 // total
        # Show chapter title only when it changes
        ch_line = ""
        if ch_idx != prev_ch_idx:
            ch_title = chapter_titles[ch_idx] if ch_idx < len(chapter_titles) else f"Ch {ch_idx}"
            ch_line = f"📖 {ch_title}\n"
            prev_ch_idx = ch_idx

        if synth_times:
            avg_time = sum(synth_times) / len(synth_times)
            stats = f"{elapsed_min}m{elapsed_sec:02d}s | ETA {eta_min}m{eta_sec:02d}s | ~{avg_time:.0f}s/чанк"
        else:
            stats = f"{elapsed_min}m{elapsed_sec:02d}s"
        yield f"{ch_line}[{done}/{total}] {progress_pct}% | {status} | {stats}", None

    # Build M4B
    yield "Building M4B...", None
    try:
        from src.audio.builder import AudioBuilder
        builder = AudioBuilder(
            output_path=output_path, title=book.title, author=book.author,
            cover_bytes=book.cover_bytes, enhance=enhance,
        )
        filtered_wavs = [wl for wl in chapter_wav_lists if wl]
        filtered_titles = [t for t, wl in zip(chapter_titles, chapter_wav_lists) if wl]
        filtered_pauses = [pl for pl, wl in zip(chapter_pause_lists, chapter_wav_lists) if wl]
        builder.build(filtered_wavs, filtered_titles, filtered_pauses)
    except Exception as e:
        yield f"M4B error: {e}", None
        return

    if rvc_converter:
        rvc_converter.cleanup()

    total_time = time.time() - start_time
    yield f"Done! {output_path.name} ({total_time / 60:.1f} min)", str(output_path)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

CSS = """
.main-title { text-align: center; margin-bottom: 0.5em; }
.main-title h1 { font-size: 2em; margin: 0; }
.main-title p { color: #888; margin: 0.2em 0 0; font-size: 0.95em; }
.convert-btn { min-height: 48px !important; font-size: 1.1em !important; }
"""

def build_ui() -> gr.Blocks:
    s = load_settings()
    _saved_engine = s.get("engine", "")
    _default_engine_label = _saved_engine if _saved_engine in TTS_ENGINES else "Qwen3-TTS — MLX, multilingual"
    _engine_type = TTS_ENGINES.get(_default_engine_label, "qwen3tts")
    _is_silero = _engine_type == "silero"
    _has_voice_clone = _engine_type in ("f5tts", "chatterbox", "qwen3tts")
    _has_steps = _engine_type == "f5tts"
    _has_variant = _engine_type == "qwen3tts"

    with gr.Blocks(title="audiobook-tts", css=CSS) as demo:
        gr.HTML(
            '<div class="main-title">'
            '<h1>audiobook-tts</h1>'
            '<p>FB2 / EPUB / PDF to M4B audiobook — fully local, no cloud</p>'
            '</div>'
        )

        with gr.Tabs():
            # ==================== CONVERT ====================
            with gr.TabItem("Convert"):
                with gr.Row():
                    with gr.Column(scale=1):
                        book_input = gr.File(label="Book (FB2, EPUB, PDF)", file_types=[".fb2", ".epub", ".pdf"], type="filepath")

                        with gr.Group():
                            engine_dropdown = gr.Dropdown(label="TTS Engine", choices=list(TTS_ENGINES.keys()), value=_default_engine_label)

                            # Silero voice
                            speaker_dropdown = gr.Dropdown(label="Speaker", choices=list(SILERO_SPEAKERS.keys()), value=s.get("speaker", "Xenia (жен., аудиокнижный)"), visible=_is_silero)

                            # Model variant (MLX engines)
                            # Pick correct variant choices for the saved engine
                            _variant_map = {
                                "qwen3tts": QWEN3_VARIANTS,
                            }
                            _init_variants = _variant_map.get(_engine_type, QWEN3_VARIANTS)
                            _saved_variant = s.get("model_variant", "")
                            _init_variant_value = _saved_variant if _saved_variant in _init_variants else list(_init_variants.keys())[0]

                            model_variant_dropdown = gr.Dropdown(
                                label="Model variant",
                                choices=list(_init_variants.keys()),
                                value=_init_variant_value,
                                visible=_has_variant,
                            )

                            # Voice reference (voice cloning engines)
                            ref_audio_input = gr.File(label="Voice reference (WAV, 5-12s)", file_types=[".wav", ".mp3"], type="filepath", visible=_has_voice_clone)
                            ref_text_input = gr.Textbox(label="Что говорят в референсе (авто если пусто)", placeholder="Точный текст из аудио-клипа…", visible=_has_voice_clone)
                            ref_yt_url = gr.Textbox(label="or YouTube URL for voice", placeholder="https://youtube.com/watch?v=...", visible=_has_voice_clone)
                            ref_yt_btn = gr.Button("Extract voice from YouTube", size="sm", visible=_has_voice_clone)
                            nfe_steps_slider = gr.Slider(label="Качество (NFE steps)", minimum=4, maximum=64, step=4, value=s.get("nfe_steps", 4), info="4 — быстро, 8 — баланс, 16+ — высокое качество", visible=_has_steps)
                            cfg_strength_slider = gr.Slider(label="Точность к тексту (CFG)", minimum=0.5, maximum=4.0, step=0.1, value=s.get("cfg_strength", 2.0), info="2.0 — норма. Выше — точнее, ниже — свободнее", visible=_has_steps)

                        with gr.Group():
                            speed_slider = gr.Slider(label="Speed", minimum=0.7, maximum=1.3, step=0.02, value=s.get("speech_speed", 0.92))
                            max_words_slider = gr.Slider(label="Words per chunk", minimum=50, maximum=400, step=10, value=s.get("max_words", 200), info="Larger = faster, fewer chunks")
                            test_chunks_slider = gr.Slider(label="Test chunks (0 = all)", minimum=0, maximum=50, step=1, value=0, info="Stop after N chunks for quick testing")
                            enhance_checkbox = gr.Checkbox(label="Audio enhancement (EQ, compression, loudnorm -16 LUFS)", value=s.get("enhance", True))
                            language_dropdown = gr.Dropdown(label="Language", choices=list(LANGUAGES.keys()), value=s.get("language", "Russian"))

                        # RVC (Silero only)
                        with gr.Group(visible=_is_silero) as rvc_group:
                            gr.Markdown("**RVC Voice Conversion**")
                            rvc_dropdown = gr.Dropdown(label="RVC Voice", choices=get_rvc_choices(), value=s.get("rvc_voice", RVC_NONE))
                            rvc_refresh_btn = gr.Button("Refresh", size="sm")
                            pitch_slider = gr.Slider(label="Pitch (semitones)", minimum=-12, maximum=12, step=1, value=s.get("pitch_shift", 0))

                        output_name_input = gr.Textbox(label="Output filename", placeholder="auto from book title")
                        with gr.Row():
                            convert_btn = gr.Button("Convert to M4B", variant="primary", size="lg", elem_classes=["convert-btn"])
                            clear_cache_btn = gr.Button("🗑 Сбросить кеш", variant="stop", size="sm")

                    with gr.Column(scale=1):
                        progress_log = gr.Textbox(label="Progress", lines=20, max_lines=40, interactive=False, placeholder="Upload a book and click Convert...")
                        download_output = gr.File(label="Download", interactive=False, visible=False)
                        with gr.Row():
                            preview_btn = gr.Button("Preview voice", variant="secondary", size="sm")
                            preview_status = gr.Textbox(show_label=False, interactive=False, scale=3, placeholder="Click to preview...")
                        preview_audio = gr.Audio(label="Preview", visible=False)

                # --- Handlers ---
                def on_engine_change(engine_label):
                    et = TTS_ENGINES.get(engine_label, "silero")
                    is_silero = et == "silero"
                    has_voice_clone = et in ("f5tts", "chatterbox", "qwen3tts")
                    has_ref_text = et in ("f5tts", "qwen3tts")
                    has_steps = et == "f5tts"

                    # Model variant dropdown — update choices based on engine
                    if et == "qwen3tts":
                        variant_update = gr.update(visible=True, choices=list(QWEN3_VARIANTS.keys()), value=list(QWEN3_VARIANTS.keys())[0])
                    else:
                        variant_update = gr.update(visible=False)

                    return (
                        gr.update(visible=is_silero),        # speaker
                        variant_update,                       # model_variant
                        gr.update(visible=has_voice_clone),  # ref_audio
                        gr.update(visible=has_ref_text),     # ref_text
                        gr.update(visible=has_voice_clone),  # ref_yt_url
                        gr.update(visible=has_voice_clone),  # ref_yt_btn
                        gr.update(visible=has_steps),        # nfe_steps
                        gr.update(visible=has_steps),        # cfg_strength
                        gr.update(visible=is_silero),        # rvc_group
                    )

                engine_dropdown.change(
                    fn=on_engine_change, inputs=[engine_dropdown],
                    outputs=[speaker_dropdown, model_variant_dropdown, ref_audio_input, ref_text_input, ref_yt_url, ref_yt_btn, nfe_steps_slider, cfg_strength_slider, rvc_group],
                )

                def on_extract_yt_voice(url):
                    """Download YouTube audio and extract a voice segment.

                    F5-TTS internally trims to 6-12s of speech using smart
                    silence detection, so we give it ~15s from near the start
                    (where speech typically begins after any intro).
                    """
                    if not url or not url.strip():
                        return gr.update(), ""
                    try:
                        import re as _re
                        raw_url = url.strip()
                        # Extract &t= timestamp (seconds) before cleaning URL
                        yt_start = 0
                        t_match = _re.search(r"[?&]t=(\d+)", raw_url)
                        if t_match:
                            yt_start = int(t_match.group(1))
                        # Clean URL for yt-dlp (strip query params)
                        clean_url = raw_url.split("&")[0].split("?")[0]
                        # Restore just the video ID param if needed
                        v_match = _re.search(r"[?&]v=([^&]+)", raw_url)
                        if v_match and "watch" in clean_url:
                            clean_url += "?v=" + v_match.group(1)

                        env = _subprocess_env()

                        tmp_dir = Path(tempfile.mkdtemp(prefix="yt_voice_"))
                        raw_path = tmp_dir / "raw.%(ext)s"

                        # Only download a 30s section (enough for 15s clip + buffer)
                        # --download-sections cuts server-side when possible
                        clip_len = 15
                        start = max(0, yt_start if yt_start > 0 else 5)
                        section_end = start + clip_len + 15  # extra buffer
                        section_spec = f"*{start}-{section_end}"

                        result = subprocess.run(
                            ["yt-dlp",
                             "--download-sections", section_spec,
                             "-x", "--audio-format", "wav", "--audio-quality", "0",
                             "-f", "ba",  # best audio only — skip video
                             "--cookies-from-browser", "chrome",
                             "-o", str(raw_path), "--no-playlist", clean_url],
                            capture_output=True, text=True, env=env,
                            timeout=120,  # 2 min max
                        )
                        if result.returncode != 0:
                            return gr.update(), f"yt-dlp error: {result.stderr[-300:] if result.stderr else 'unknown'}"
                        downloaded = list(tmp_dir.glob("raw.*"))
                        if not downloaded:
                            return gr.update(), "Error: no audio downloaded"

                        # Trim to exact clip and convert to 24kHz mono WAV
                        ref_path = tmp_dir / "voice_ref.wav"
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", str(downloaded[0]),
                             "-ss", "0", "-t", str(clip_len),
                             "-ac", "1", "-ar", "24000",
                             str(ref_path)],
                            capture_output=True, text=True, env=env,
                        )
                        return gr.update(value=str(ref_path)), f"Extracted {clip_len:.0f}s clip (start={start:.0f}s)"
                    except Exception as e:
                        return gr.update(), f"Error: {e}"

                ref_yt_btn.click(fn=on_extract_yt_voice, inputs=[ref_yt_url], outputs=[ref_audio_input, preview_status])

                def on_preview(engine_label, speaker_label, model_variant_label, ref_audio_file, ref_text, nfe_steps, cfg_strength, speech_speed):
                    engine_type = TTS_ENGINES.get(engine_label, "silero")
                    yield "Loading model...", gr.update()
                    try:
                        from src.tts.engine import create_engine
                        kw = dict(speech_speed=float(speech_speed))
                        if engine_type == "silero":
                            kw["speaker"] = SILERO_SPEAKERS.get(speaker_label, "xenia")
                        elif engine_type == "f5tts":
                            if ref_audio_file:
                                kw["ref_audio"] = ref_audio_file
                            if ref_text and ref_text.strip():
                                kw["ref_text"] = ref_text.strip()
                            kw["nfe_steps"] = int(nfe_steps)
                            kw["cfg_strength"] = float(cfg_strength)
                        elif engine_type == "chatterbox":
                            if ref_audio_file:
                                kw["ref_audio"] = ref_audio_file
                        elif engine_type == "mlx-chatterbox":
                            if ref_audio_file:
                                kw["ref_audio"] = ref_audio_file
                            kw["model_variant"] = MLX_CHATTERBOX_VARIANTS.get(model_variant_label, "multi-q4")
                        elif engine_type == "qwen3tts":
                            if ref_audio_file:
                                kw["ref_audio"] = ref_audio_file
                            if ref_text and ref_text.strip():
                                kw["ref_text"] = ref_text.strip()
                            kw["model_variant"] = QWEN3_VARIANTS.get(model_variant_label, "base-0.6b")
                        engine = get_cached_engine(engine_type, **kw)
                        yield "Synthesizing...", gr.update()
                        wav = Path(tempfile.mktemp(suffix=".wav"))
                        engine.synthesize("Солнце клонилось к закату, бросая длинные тени на старую мостовую. Тишина стояла такая, что слышно было, как ветер шелестит в кронах деревьев.", wav)
                        if wav.exists() and wav.stat().st_size > 100:
                            yield "Done!", gr.update(visible=True, value=str(wav))
                        else:
                            yield "Error: empty audio generated", gr.update()
                    except Exception as e:
                        logger.error("Preview failed: %s", e)
                        yield f"Error: {e}", gr.update()

                preview_btn.click(fn=on_preview, inputs=[engine_dropdown, speaker_dropdown, model_variant_dropdown, ref_audio_input, ref_text_input, nfe_steps_slider, cfg_strength_slider, speed_slider], outputs=[preview_status, preview_audio])
                rvc_refresh_btn.click(fn=lambda: gr.update(choices=get_rvc_choices()), outputs=[rvc_dropdown])

                def on_clear_cache(book_file):
                    import shutil
                    if not book_file:
                        return "Сначала загрузите книгу"
                    book_path = Path(book_file)
                    book_hash = compute_hash(book_path)
                    removed = 0
                    for d in CACHE_DIR.iterdir():
                        if d.is_dir() and d.name.startswith(book_hash):
                            shutil.rmtree(d)
                            removed += 1
                    if removed:
                        return f"Удалено {removed} кеш-директорий для {book_path.name}"
                    return f"Кеш для {book_path.name} не найден"

                clear_cache_btn.click(fn=on_clear_cache, inputs=[book_input], outputs=[progress_log])

                log_state = gr.State([])

                def on_convert(book_file, output_name, engine_label, language_label, speaker_label, model_variant_label, ref_audio_file, ref_text, rvc_voice_name, pitch_shift, speech_speed, max_words, test_chunks, enhance, nfe_steps, cfg_strength, log_lines):
                    save_settings(engine=engine_label, speaker=speaker_label, model_variant=model_variant_label, rvc_voice=rvc_voice_name, pitch_shift=int(pitch_shift), speech_speed=float(speech_speed), language=language_label, max_words=int(max_words), nfe_steps=int(nfe_steps), cfg_strength=float(cfg_strength), enhance=bool(enhance))
                    log_lines = []
                    final_file = None
                    for msg, fpath in run_conversion(book_file, output_name, engine_label, language_label, speaker_label, ref_audio_file, ref_text, rvc_voice_name, pitch_shift, speech_speed, max_words, nfe_steps, cfg_strength, model_variant_label, test_chunks, enhance):
                        log_lines.append(msg)
                        if fpath:
                            final_file = fpath
                        yield "\n".join(log_lines), gr.update(visible=bool(final_file), value=final_file), log_lines

                convert_btn.click(
                    fn=on_convert,
                    inputs=[book_input, output_name_input, engine_dropdown, language_dropdown, speaker_dropdown, model_variant_dropdown, ref_audio_input, ref_text_input, rvc_dropdown, pitch_slider, speed_slider, max_words_slider, test_chunks_slider, enhance_checkbox, nfe_steps_slider, cfg_strength_slider, log_state],
                    outputs=[progress_log, download_output, log_state],
                )

            # ==================== VOICE LAB ====================
            with gr.TabItem("Voice Lab"):
                gr.Markdown("### Voice Lab\nImport voices and RVC models for voice conversion.")

                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Tabs():
                            with gr.TabItem("YouTube"):
                                yt_voice_name = gr.Textbox(label="Voice name", placeholder="e.g. Buldakov")
                                yt_url_input = gr.Textbox(label="YouTube URL", placeholder="https://youtube.com/watch?v=...")
                                yt_btn = gr.Button("Download", variant="primary")
                                yt_status = gr.Textbox(label="Status", interactive=False, lines=2)

                            with gr.TabItem("Upload Audio"):
                                upload_voice_name = gr.Textbox(label="Voice name", placeholder="e.g. MyVoice")
                                audio_upload = gr.File(label="Audio (WAV/MP3)", file_types=[".wav", ".mp3"], file_count="multiple", type="filepath")
                                upload_btn = gr.Button("Import", variant="primary")
                                upload_status = gr.Textbox(label="Status", interactive=False, lines=2)

                            with gr.TabItem("Import RVC Model"):
                                model_voice_name = gr.Textbox(label="Voice name", placeholder="e.g. MyVoice")
                                rvc_model_upload = gr.File(label="Model (.pth)", file_types=[".pth"], type="filepath")
                                rvc_index_upload = gr.File(label="Index (.index) — optional", file_types=[".index"], type="filepath")
                                import_model_btn = gr.Button("Import", variant="primary")
                                import_status = gr.Textbox(label="Status", interactive=False, lines=2)

                    with gr.Column(scale=1):
                        voices_list = gr.Dataframe(headers=["Name", "Source", "Model", "Updated"], datatype=["str", "str", "bool", "str"], interactive=False, label="Voices")
                        refresh_list_btn = gr.Button("Refresh", size="sm")

                        with gr.Group():
                            gr.Markdown("**Train RVC Model**")
                            train_voice_name = gr.Textbox(label="Voice name", placeholder="e.g. Buldakov")
                            with gr.Row():
                                train_epochs = gr.Slider(label="Epochs", minimum=20, maximum=500, step=10, value=100)
                                train_batch = gr.Slider(label="Batch", minimum=1, maximum=16, step=1, value=8)
                            train_btn = gr.Button("Train", variant="primary")
                            train_status = gr.Textbox(label="Log", interactive=False, lines=6)

                        with gr.Group():
                            delete_name_input = gr.Textbox(label="Delete voice", placeholder="name...")
                            delete_btn = gr.Button("Delete", variant="stop", size="sm")
                            delete_status = gr.Textbox(interactive=False, lines=1, show_label=False)

                # --- Voice Lab handlers ---
                def refresh_voices_table():
                    voices = voice_manager.list_voices()
                    return [[v.get("name", "?"), v.get("source", "?"), v.get("has_model", False), v.get("updated", "?")[:19]] for v in voices] if voices else []

                upload_btn.click(fn=lambda n, f: f"Imported {len(f)} file(s) for '{n}'." if n and f and not voice_manager.import_audio_files(n.strip(), [Path(x) for x in f]) is None else "Error: name and files required", inputs=[upload_voice_name, audio_upload], outputs=[upload_status])

                def on_yt_download(name, url):
                    if not name or not url: return "Enter name and URL."
                    try:
                        voice_manager.import_from_youtube(name.strip(), url.strip())
                        return f"Downloaded for '{name.strip()}'."
                    except Exception as e: return f"Error: {e}"

                yt_btn.click(fn=on_yt_download, inputs=[yt_voice_name, yt_url_input], outputs=[yt_status])

                def on_import_model(name, model_file, index_file):
                    if not name or not model_file: return "Enter name and upload .pth."
                    try:
                        voice_manager.import_rvc_model(name.strip(), Path(model_file), Path(index_file) if index_file else None)
                        return f"Model imported for '{name.strip()}'!"
                    except Exception as e: return f"Error: {e}"

                import_model_btn.click(fn=on_import_model, inputs=[model_voice_name, rvc_model_upload, rvc_index_upload], outputs=[import_status])
                refresh_list_btn.click(fn=refresh_voices_table, outputs=[voices_list])

                def on_train(name, epochs, batch_size):
                    if not name: return "Enter voice name."
                    samples_dir = Path("./voices") / name.strip() / "samples"
                    if not samples_dir.exists() or not list(samples_dir.glob("*")):
                        return f"No samples for '{name}'. Import audio first."
                    try:
                        from src.rvc.trainer import train_voice
                        lines = []
                        model_path, index_path = train_voice(name=name.strip(), dataset_dir=samples_dir, total_epochs=int(epochs), batch_size=int(batch_size), on_progress=lambda s, d: lines.append(f"[{s}] {d}"))
                        voice_manager.import_rvc_model(name.strip(), model_path, index_path)
                        lines.append(f"Done! '{name.strip()}' ready.")
                        return "\n".join(lines)
                    except Exception as e: return f"Error: {e}"

                train_btn.click(fn=on_train, inputs=[train_voice_name, train_epochs, train_batch], outputs=[train_status])
                delete_btn.click(fn=lambda n: f"Deleted '{n}'." if n and voice_manager.delete_voice(n.strip()) else "Not found.", inputs=[delete_name_input], outputs=[delete_status])
                demo.load(fn=refresh_voices_table, outputs=[voices_list])

            # ==================== ENGINE LOG ====================
            with gr.TabItem("Engine Log"):
                gr.Markdown("""### Протестированные движки

Лог движков которые мы пробовали, что работает, что нет и почему.

---

#### ✅ Работающие движки

| Движок | Тип | Качество RU | Скорость | Примечания |
|--------|-----|-------------|----------|------------|
| **Silero v5** | PyTorch | ★★★★ | ★★★★★ | Быстрый, 5 голосов, понимает `+` ударения. Нет voice cloning. |
| **F5-TTS** | PyTorch MPS | ★★★★ | ★★★ | Voice cloning, хорошее русское произношение. ~6с/чанк на M4 Max. |
| **Chatterbox** | PyTorch MPS | ★★☆ | ★★ | Интонация местами неплохая, но timbre/дикция так себе. Медленнее, есть шум на стыках чанков. Пока под вопросом. |
| **Qwen3-TTS** | MLX Metal | ★★★ | ★★★ | Voice cloning, 10 языков. Не понимает ударения — ё восстанавливаем через silero-stress. |

---

#### ❌ Убранные движки

**MLX F5-TTS** (`f5-tts-mlx`, `ink-splatters/f5-tts-russian-mlx`)
- **Статус:** Удалён из UI
- **Проблема:** Генерирует мусор/билиберду на русском тексте
- **Причины:**
  1. Community-порт `f5-tts-mlx` — не официальная реализация, слабо оптимизирован
  2. Русская модель `ink-splatters/f5-tts-russian-mlx` — низкое качество конвертации весов
  3. `generate()` не возвращает аудио — приходится писать в temp-файл и читать обратно
  4. Нет контроля длительности — модель генерирует 30-70с мусора вместо 10-15с речи
  5. Контекстное окно ~22с (ref + gen) — длинный ref audio не оставляет места для генерации
  6. Первый чанк ~60с cold start (JIT компиляция MLX графов)
  7. При одинаковых steps (8) в 2 раза медленнее PyTorch F5-TTS
- **Вывод:** PyTorch F5-TTS с MPS значительно лучше по качеству и стабильнее
- **Файл:** `src/tts/engine_mlx_f5tts.py` (оставлен в репо на случай улучшений)

**MLX Chatterbox** (`mlx-audio`, `mlx-community/chatterbox-4bit`)
- **Статус:** Удалён из UI
- **Проблема:** Для русского непригоден. Голос далёк от референса, картавый/искажённый тембр, длинные паузы и частые near-silence чанки.
- **Причины:**
  1. Единственный явно multilingual MLX-вариант (`YUGOROU/...Multilingual...`) даёт плохой voice match на русском
  2. Регулярно генерирует runaway/near-silence чанки на 8-22-44 секунд
  3. Даже с `ref_audio`, trimming, retry и отбраковкой мусорных кусков результат остаётся слабым
  4. Квантизация дополнительно портит timbre и артикуляцию
  5. Длинные внутренние паузы и шум/артефакты делают итоговую книгу непригодной
- **Вывод:** MLX Chatterbox закрыт как тупиковый эксперимент для RU voice cloning
- **Файл:** `src/tts/engine_mlx_chatterbox.py` (оставлен в репо)

---

#### 🔬 Заметки по ударениям

- **Silero/F5-TTS:** Понимают `+` перед ударной гласной. `silero-stress` расставляет автоматически.
- **Qwen3-TTS:** Обучена **без** стресс-меток. `+` и U+0301 ломают токенизатор. Единственный рабочий хак — восстановление `ё` (всегда ударная). Используем `accent_mode="yo_only"`.
- **Chatterbox/MLX Chatterbox:** Свои внутренние модели произношения, но русский остаётся слабым местом. У MLX-варианта качество для RU признано непригодным.
""")

    return demo

if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
