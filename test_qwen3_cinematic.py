#!/usr/bin/env python3
"""Qwen3-TTS cinematic pipeline test.

Tests the full new pipeline: semantic chunking + emotion classification
+ direct model calls + semantic pauses + mastering.

Literary Russian text with dialogue, narration, reflection, and tension.
Output: output/test_qwen3tts_cinematic.wav
"""

import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_cinematic")

# -----------------------------------------------------------------------
# Test text: original Russian literary prose with varied emotional texture
# -----------------------------------------------------------------------

TEST_TEXT = """
Осенний дождь барабанил по окнам кабинета, когда Николай Петрович наконец отложил перо. Бумаги на столе были исписаны убористым почерком — страница за страницей, — но ни одна строчка не казалась ему достаточно верной. Слова ложились не так, как он слышал их в голове.

Он встал, подошёл к окну. Двор был пуст. Одинокий фонарь раскачивался под порывами ветра, отбрасывая на мокрую брусчатку длинные дрожащие тени.

— Всё не то, — пробормотал он себе под нос. — Совсем не то.

Дверь скрипнула. В комнату вошла Варвара — молодая женщина с усталыми глазами и охапкой книг под мышкой. Она остановилась у порога, окинула взглядом разбросанные бумаги и его сутулую спину.

— Опять не спишь? — тихо спросила она.

— Работаю.

— В три часа ночи? — Она поставила книги на край стола и обернулась. — Николай, ты так долго не протянешь.

Он промолчал. За окном ветер усилился, и дождь хлестнул по стеклу с новой силой, точно подтверждая её слова.

«Может, она права», — подумал он. Но рука уже снова тянулась к перу.
"""

REF_AUDIO = "voices/Buldakov/samples/youtube.wav"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_WAV = OUTPUT_DIR / "test_qwen3tts_cinematic.wav"


def show_chunks(text: str) -> None:
    """Preview how the semantic chunker splits the text."""
    from src.text.semantic_chunker import SemanticChunker
    from src.text.emotion_classifier import EmotionClassifier

    chunker = SemanticChunker()
    classifier = EmotionClassifier()

    chunks = chunker.chunk(text)
    print(f"\n{'='*60}")
    print(f"Semantic chunks ({len(chunks)} total):")
    print("="*60)

    prev = None
    for c in chunks:
        emotion = classifier.classify(c.text, prev, c.split_type)
        print(f"\n[{c.index+1}] split={c.split_type} | {emotion.category}"
              f" | speed={emotion.speed:.2f} | pause={emotion.pause_after_ms}ms")
        print(f"  {c.text[:100]}{'...' if len(c.text) > 100 else ''}")
        prev = emotion
    print("="*60 + "\n")


def main():
    import soundfile as sf
    import numpy as np

    logger.info("=== Qwen3-TTS Cinematic Pipeline Test ===")

    # Preview chunking plan
    show_chunks(TEST_TEXT)

    # Load engine
    t_load = time.time()
    from src.tts.engine_qwen3tts import Qwen3TTSEngine

    engine = Qwen3TTSEngine(
        ref_audio=REF_AUDIO,
        speech_speed=0.85,
        model_variant="base-0.6b",
        mastering=True,
    )
    logger.info("Engine loaded in %.1fs", time.time() - t_load)

    # Synthesize
    t_synth = time.time()
    logger.info("Synthesizing…")
    engine.synthesize(TEST_TEXT, str(OUTPUT_WAV))
    elapsed = time.time() - t_synth

    # Report
    info = sf.info(str(OUTPUT_WAV))
    dur = info.duration
    rtf = elapsed / dur if dur > 0 else 0

    print(f"\n{'='*60}")
    print("RESULTS — Cinematic Pipeline")
    print("="*60)
    print(f"Output     : {OUTPUT_WAV.resolve()}")
    print(f"Duration   : {dur:.1f}s")
    print(f"Gen time   : {elapsed:.1f}s")
    print(f"RTF        : {rtf:.2f}x")
    print("="*60)
    print(f"\nListen: open '{OUTPUT_WAV.resolve()}'")


if __name__ == "__main__":
    main()
