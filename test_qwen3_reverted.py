#!/usr/bin/env python3
"""Qwen3-TTS revert test — matches original test_qwen3tts_8chunks.wav parameters.

Uses reverted engine (minimal changes: _normalize_ref_audio + native speed).
Same 8 chunks, same voice, same speed=0.85.
"""

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
logger = logging.getLogger("test_qwen3_reverted")

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
OUTPUT_WAV = OUTPUT_DIR / "test_qwen3tts_reverted.wav"


def main():
    from src.tts.engine_qwen3tts import Qwen3TTSEngine

    logger.info("=== Qwen3-TTS Reverted Engine Test ===")
    logger.info("Ref audio: %s", REF_AUDIO)
    logger.info("Speed: 0.85 (native), model: base-0.6b")

    t_load = time.time()
    engine = Qwen3TTSEngine(
        ref_audio=REF_AUDIO,
        speech_speed=0.85,
        model_variant="base-0.6b",
    )
    logger.info("Engine loaded in %.1fs", time.time() - t_load)

    chunk_times = []
    audio_parts = []
    total_audio_sec = 0.0

    for i, text in enumerate(TEST_CHUNKS):
        logger.info("--- Chunk %d/%d: %s...", i + 1, len(TEST_CHUNKS), text[:60])
        tmp = OUTPUT_DIR / f"_tmp_qwen3_chunk_{i:02d}.wav"
        t0 = time.time()
        engine.synthesize(text, str(tmp))
        elapsed = time.time() - t0

        audio, sr = sf.read(str(tmp))
        dur = len(audio) / sr
        chunk_times.append(elapsed)
        total_audio_sec += dur
        audio_parts.append(audio)
        tmp.unlink()

        logger.info("  %.1fs audio in %.1fs (RTF=%.2f)", dur, elapsed, elapsed / max(dur, 0.1))

    combined = np.concatenate(audio_parts).astype(np.float32)
    # Qwen3-TTS outputs at 24kHz
    sf.write(str(OUTPUT_WAV), combined, 24000)

    total_time = sum(chunk_times)
    avg_rtf = total_time / total_audio_sec if total_audio_sec > 0 else 0

    print("\n" + "=" * 60)
    print("RESULTS — Qwen3-TTS reverted engine")
    print("=" * 60)
    print(f"Output     : {OUTPUT_WAV.resolve()}")
    print(f"Audio      : {total_audio_sec:.1f}s")
    print()
    for i, (t, text) in enumerate(zip(chunk_times, TEST_CHUNKS)):
        print(f"  [{i+1}] {t:.1f}s — {text[:55]}...")
    print()
    print(f"Avg time/chunk : {sum(chunk_times)/len(chunk_times):.1f}s")
    print(f"Total gen time : {total_time:.1f}s")
    print(f"Avg RTF        : {avg_rtf:.2f}x")
    print("=" * 60)
    print(f"\nListen: open '{OUTPUT_WAV.resolve()}'")


if __name__ == "__main__":
    main()
