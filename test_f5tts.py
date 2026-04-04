#!/usr/bin/env python3
"""F5-TTS quality test — stress marks fixed, speed 0.85, optionally hotstone228.

Usage:
    python test_f5tts.py              # Misha24-10 (default)
    python test_f5tts.py --hotstone   # hotstone228/F5-TTS-Russian (100k hours)
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_f5tts")

TEST_CHUNKS = [
    "Сергей Бурунов открыл глаза и уставился в потолок. Утро пришло незаметно, как всегда бывает в ноябре, когда темнота растворяется медленно и нехотя.",
    "За окном шумел город. Машины, голоса, далёкий гудок поезда — всё это складывалось в привычную симфонию московского утра.",
    "Он встал, подошёл к окну. На улице шёл мелкий дождь. Прохожие спешили, наклонив головы и прячась под зонтами.",
    "На кухне уже кипел чайник. Аня, как обычно, проснулась раньше и теперь возилась с завтраком, напевая что-то тихое и неразборчивое.",
    "— Доброе утро, — сказала она, не оборачиваясь. — Ты опять плохо спал? Я слышала, как ты ворочался.",
    "Бурунов налил себе чаю и сел за стол. Перед ним лежала газета, но он не стал её читать. Мысли были заняты другим.",
    "Сегодня предстояло важное совещание. Директор завода давно не собирал весь состав, и это само по себе было плохим знаком.",
    "Он допил чай, поднялся и начал собираться. День обещал быть долгим.",
]

REF_AUDIO = "voices/Buldakov/samples/youtube.wav"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def run_test(use_hotstone: bool = False):
    from src.tts.engine_f5tts import (
        F5TTSEngine,
        _HF_CKPT, _HF_VOCAB,
        _HF_CKPT_HOTSTONE, _HF_VOCAB_HOTSTONE,
    )

    if use_hotstone:
        ckpt, vocab = _HF_CKPT_HOTSTONE, _HF_VOCAB_HOTSTONE
        label = "hotstone228"
    else:
        ckpt, vocab = _HF_CKPT, _HF_VOCAB
        label = "misha24"

    output_wav = OUTPUT_DIR / f"test_f5tts_fixed_{label}.wav"

    info = sf.info(REF_AUDIO)
    logger.info("=== F5-TTS Test: %s ===", label)
    logger.info("Ref audio: %s (sr=%dHz, dur=%.1fs)", REF_AUDIO, info.samplerate, info.duration)
    logger.info("NFE=32, speed=0.85, stress marks: ENABLED")

    t_load = time.time()
    engine = F5TTSEngine(
        ref_audio=REF_AUDIO,
        ref_text="",
        nfe_steps=32,
        cfg_strength=2.0,
        speech_speed=0.85,
        ckpt_file=ckpt,
        vocab_file=vocab,
    )
    logger.info("Engine loaded in %.1fs", time.time() - t_load)

    chunk_times = []
    audio_parts = []
    total_audio_sec = 0.0

    for i, text in enumerate(TEST_CHUNKS):
        logger.info("--- Chunk %d/%d: %s...", i + 1, len(TEST_CHUNKS), text[:60])
        tmp = OUTPUT_DIR / f"_tmp_chunk_{i:02d}.wav"
        t0 = time.time()
        engine.synthesize(text, tmp)
        elapsed = time.time() - t0

        audio, sr = sf.read(str(tmp))
        dur = len(audio) / sr
        chunk_times.append(elapsed)
        total_audio_sec += dur
        audio_parts.append(audio)
        tmp.unlink()

        logger.info("  %.1fs audio in %.1fs (RTF=%.2f)", dur, elapsed, elapsed / max(dur, 0.1))

    combined = np.concatenate(audio_parts).astype(np.float32)
    sf.write(str(output_wav), combined, 24000)

    total_time = sum(chunk_times)
    avg_rtf = total_time / total_audio_sec if total_audio_sec > 0 else 0

    print("\n" + "=" * 60)
    print(f"RESULTS — {label}")
    print("=" * 60)
    print(f"Checkpoint  : {ckpt}")
    print(f"Stress marks: ENABLED (+ kept)")
    print(f"Speed       : 0.85x (atempo)")
    print(f"Output      : {output_wav.resolve()}")
    print(f"Audio       : {total_audio_sec:.1f}s")
    print()
    for i, (t, text) in enumerate(zip(chunk_times, TEST_CHUNKS)):
        print(f"  [{i+1}] {t:.1f}s — {text[:55]}...")
    print()
    print(f"Avg time/chunk : {sum(chunk_times)/len(chunk_times):.1f}s")
    print(f"Total gen time : {total_time:.1f}s")
    print(f"Avg RTF        : {avg_rtf:.2f}x")
    print("=" * 60)
    print(f"\nListen: open '{output_wav.resolve()}'")
    return output_wav


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotstone", action="store_true", help="Use hotstone228 checkpoint")
    parser.add_argument("--both", action="store_true", help="Run both checkpoints")
    args = parser.parse_args()

    if args.both:
        run_test(use_hotstone=False)
        run_test(use_hotstone=True)
    elif args.hotstone:
        run_test(use_hotstone=True)
    else:
        run_test(use_hotstone=False)
