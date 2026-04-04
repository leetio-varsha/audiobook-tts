#!/usr/bin/env python3
"""Qwen3-TTS direct model test — NO engine wrapper.

Calls model.generate() directly, same as the original test that produced
test_qwen3tts_8chunks.wav. No text normalizer, no accentuator, no volume
norm, no ffmpeg. Native speed=0.85 param.
"""

import logging
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_qwen3_direct")

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
OUTPUT_WAV = OUTPUT_DIR / "test_qwen3tts_original_method.wav"


def prepare_ref(src: str, duration_sec: float = 10.0) -> str:
    """Trim reference to 10s from middle, save to temp file."""
    audio, sr = sf.read(src)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    target = int(duration_sec * sr)
    if len(audio) > target:
        start = (len(audio) - target) // 2
        audio = audio[start: start + target]
    tmp = tempfile.mktemp(suffix="_ref.wav", prefix="qwen3_direct_")
    sf.write(tmp, audio.astype(np.float32), sr)
    logger.info("Ref: %.1fs @ %dHz → %s", len(audio) / sr, sr, tmp)
    return tmp


def main():
    from mlx_audio.tts.utils import load_model

    logger.info("=== Qwen3-TTS DIRECT test (no engine wrapper) ===")

    t0 = time.time()
    model = load_model("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16")
    logger.info("Model loaded in %.1fs", time.time() - t0)

    ref_audio = prepare_ref(REF_AUDIO)

    chunk_times = []
    audio_parts = []
    total_audio_sec = 0.0

    for i, text in enumerate(TEST_CHUNKS):
        logger.info("--- Chunk %d/%d: %s...", i + 1, len(TEST_CHUNKS), text[:60])
        t0 = time.time()

        results = list(model.generate(
            text=text,
            lang_code="russian",
            ref_audio=ref_audio,
            speed=0.85,
        ))

        elapsed = time.time() - t0

        if not results:
            logger.warning("Empty result for chunk %d", i + 1)
            continue

        audio = np.array(results[0].audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()

        dur = len(audio) / 24000  # Qwen3-TTS outputs 24kHz
        chunk_times.append(elapsed)
        total_audio_sec += dur
        audio_parts.append(audio)

        logger.info("  %.1fs audio in %.1fs (RTF=%.2f)", dur, elapsed, elapsed / max(dur, 0.1))

    combined = np.concatenate(audio_parts).astype(np.float32)
    sf.write(str(OUTPUT_WAV), combined, 24000)

    total_time = sum(chunk_times)
    avg_rtf = total_time / total_audio_sec if total_audio_sec > 0 else 0

    print("\n" + "=" * 60)
    print("RESULTS — Qwen3-TTS DIRECT (no engine)")
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
