#!/usr/bin/env python3
"""Deep sub-stage profiling of Qwen3-TTS generate() internals.

Measures each stage separately to find where the 31s/chunk generation time goes:
  1. Disk read (load_audio)
  2. Speaker embedding (ECAPA-TDNN mel + inference)
  3. Input prep (tokenize + embed)
  4. LLM decode loop (talker token generation)
  5. Vocoder decode (speech_tokenizer.decode)
  6. mx.eval sync

Strategy: monkey-patch model internals to inject timers.
"""

import logging
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("profile")

REF_AUDIO = "voices/Buldakov/samples/youtube.wav"

# Short literary chunks representing real audiobook text
CHUNKS = [
    "Осенний дождь барабанил по окнам кабинета, когда Николай Петрович наконец отложил перо.",
    "— Всё не то, — пробормотал он себе под нос. — Совсем не то.",
    "Дверь скрипнула. В комнату вошла Варвара — молодая женщина с усталыми глазами.",
    "— Опять не спишь? — тихо спросила она.",
    "За окном ветер усилился, и дождь хлестнул по стеклу с новой силой.",
]

SPEED = 0.85


def trim_ref_audio(src: str, max_sec: float = 10.0) -> mx.array:
    """Load and trim reference audio to max_sec, return as mx.array."""
    import soundfile as sf
    audio, sr = sf.read(src)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    target = int(max_sec * sr)
    if len(audio) > target:
        start = (len(audio) - target) // 2
        audio = audio[start: start + target]
    # Convert to mx.array directly — skips disk read in generate()
    return mx.array(audio.astype(np.float32))


def main():
    import soundfile as sf
    from mlx_audio.tts.utils import load_model
    from mlx_audio.utils import load_audio

    print("\n" + "="*65)
    print("Qwen3-TTS Deep Generation Profile")
    print("="*65)

    # ----------------------------------------------------------------
    # Load model
    # ----------------------------------------------------------------
    t0 = time.perf_counter()
    model = load_model("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16")
    t_load = time.perf_counter() - t0
    print(f"\nModel loaded: {t_load:.1f}s")

    # ----------------------------------------------------------------
    # Stage A: Disk read benchmark
    # ----------------------------------------------------------------
    print("\n--- Stage A: Disk read (load_audio) ---")
    times_disk = []
    for _ in range(3):
        t = time.perf_counter()
        audio_from_disk = load_audio(REF_AUDIO, sample_rate=24000)
        mx.eval(audio_from_disk)
        times_disk.append(time.perf_counter() - t)
    avg_disk = sum(times_disk) / len(times_disk)
    print(f"  load_audio: {avg_disk*1000:.1f}ms avg (n=3)")

    # ----------------------------------------------------------------
    # Stage B: Speaker embedding benchmark
    # ----------------------------------------------------------------
    print("\n--- Stage B: Speaker embedding (ECAPA-TDNN) ---")
    ref_mx = trim_ref_audio(REF_AUDIO)
    times_spk = []
    for _ in range(3):
        t = time.perf_counter()
        spk_embed = model.extract_speaker_embedding(ref_mx, sr=24000)
        mx.eval(spk_embed)
        times_spk.append(time.perf_counter() - t)
    avg_spk = sum(times_spk) / len(times_spk)
    print(f"  extract_speaker_embedding: {avg_spk*1000:.1f}ms avg (n=3)")
    print(f"  embedding shape: {spk_embed.shape}")

    # ----------------------------------------------------------------
    # Stage C: Input preparation benchmark (no speaker encoder)
    # ----------------------------------------------------------------
    print("\n--- Stage C: _prepare_generation_inputs (text only) ---")
    sample_text = CHUNKS[0]
    times_prep = []
    for _ in range(3):
        t = time.perf_counter()
        embeds, trailing, pad = model._prepare_generation_inputs(
            sample_text,
            language="russian",
            ref_audio=None,   # no speaker encoder
        )
        mx.eval(embeds, trailing, pad)
        times_prep.append(time.perf_counter() - t)
    avg_prep = sum(times_prep) / len(times_prep)
    print(f"  input prep (text only): {avg_prep*1000:.1f}ms avg (n=3)")

    # ----------------------------------------------------------------
    # Stage D: Monkey-patch speech_tokenizer.decode to time vocoder
    # ----------------------------------------------------------------
    print("\n--- Setting up vocoder timer ---")
    _orig_decode = model.speech_tokenizer.decode
    vocoder_times = []

    def _timed_decode(codes):
        t = time.perf_counter()
        result = _orig_decode(codes)
        mx.eval(result[0])
        vocoder_times.append(time.perf_counter() - t)
        return result

    model.speech_tokenizer.decode = _timed_decode
    print("  speech_tokenizer.decode patched")

    # ----------------------------------------------------------------
    # Stage E: Full generate() per chunk — timed
    # ----------------------------------------------------------------
    print("\n--- Stage E: Full generate() per chunk ---")
    print("  (Using pre-loaded mx.array for ref_audio to separate disk read)\n")

    chunk_results = []
    for i, chunk_text in enumerate(CHUNKS):
        t_total = time.perf_counter()

        # Time: disk read (using path, as engine does)
        t = time.perf_counter()
        _ = load_audio(REF_AUDIO, sample_rate=24000)
        mx.eval(_)
        t_disk_this = time.perf_counter() - t

        # Time: full generate with pre-loaded audio (no disk read overhead)
        vocoder_times.clear()
        t_gen = time.perf_counter()
        results = list(model.generate(
            text=chunk_text,
            lang_code="russian",
            speed=SPEED,
            ref_audio=ref_mx,   # mx.array — no disk read
        ))
        t_gen_elapsed = time.perf_counter() - t_gen

        if results:
            audio = np.array(results[0].audio, dtype=np.float32)
            audio_dur = len(audio) / 24000
        else:
            audio_dur = 0.0

        t_vocoder = vocoder_times[0] if vocoder_times else 0.0
        # Speaker embedding is ~avg_spk, already measured
        # LLM loop = gen_elapsed - speaker_embed - input_prep - vocoder
        t_llm = t_gen_elapsed - avg_spk - avg_prep - t_vocoder

        rtf = t_gen_elapsed / audio_dur if audio_dur > 0 else 0

        chunk_results.append({
            "text": chunk_text[:50],
            "audio_dur": audio_dur,
            "t_total": t_gen_elapsed,
            "t_disk": avg_disk,
            "t_spk": avg_spk,
            "t_prep": avg_prep,
            "t_llm": t_llm,
            "t_vocoder": t_vocoder,
            "rtf": rtf,
        })

        print(f"  [{i+1}] {audio_dur:.1f}s audio | gen={t_gen_elapsed:.2f}s | "
              f"RTF={rtf:.2f}x | vocoder={t_vocoder*1000:.0f}ms | {chunk_text[:40]!r}")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    avg_total   = sum(r["t_total"]   for r in chunk_results) / len(chunk_results)
    avg_spk_r   = sum(r["t_spk"]    for r in chunk_results) / len(chunk_results)
    avg_prep_r  = sum(r["t_prep"]   for r in chunk_results) / len(chunk_results)
    avg_llm     = sum(r["t_llm"]    for r in chunk_results) / len(chunk_results)
    avg_voc     = sum(r["t_vocoder"]for r in chunk_results) / len(chunk_results)
    avg_disk_r  = avg_disk

    def pct(t): return 100.0 * t / avg_total if avg_total > 0 else 0

    print(f"\n{'='*65}")
    print("Sub-stage breakdown (avg across chunks):")
    print(f"{'='*65}")
    print(f"  disk_read (load_audio):     {avg_disk_r*1000:6.1f}ms  ({pct(avg_disk_r):5.1f}%)")
    print(f"  speaker_embed (ECAPA-TDNN): {avg_spk_r*1000:6.1f}ms  ({pct(avg_spk_r):5.1f}%)")
    print(f"  input_prep (tokenize+embed):{avg_prep_r*1000:6.1f}ms  ({pct(avg_prep_r):5.1f}%)")
    print(f"  llm_decode (talker loop):   {avg_llm:6.2f}s   ({pct(avg_llm):5.1f}%)")
    print(f"  vocoder (speech_tokenizer): {avg_voc*1000:6.1f}ms  ({pct(avg_voc):5.1f}%)")
    print(f"  ─────────────────────────────────────────────")
    print(f"  TOTAL generate():           {avg_total:6.2f}s   (100%)")
    print(f"{'='*65}")

    print("\nTop optimization opportunities:")
    items = [
        ("disk_read",    avg_disk_r, "preload ref_audio as mx.array once"),
        ("speaker_embed",avg_spk_r,  "cache speaker embedding (compute once at load)"),
        ("llm_decode",   avg_llm,    "batch generation (2-3 chunks at once)"),
        ("vocoder",      avg_voc,    "streaming decode (overlap with generation)"),
    ]
    items.sort(key=lambda x: -x[1])
    for rank, (name, t, suggestion) in enumerate(items[:3], 1):
        print(f"  #{rank} {name} ({t*1000:.0f}ms avg): {suggestion}")


if __name__ == "__main__":
    main()
