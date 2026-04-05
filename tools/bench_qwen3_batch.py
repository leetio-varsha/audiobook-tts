#!/usr/bin/env python3
"""Benchmark Qwen3-TTS: sequential vs batch generation.

Usage:
    python tools/bench_qwen3_batch.py --ref voices/Buldakov/samples/youtube.wav
    python tools/bench_qwen3_batch.py --ref voices/Buldakov/samples/youtube.wav --batch-sizes 1,2,4,8
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bench")

# Sample Russian text — ~15 semantic chunks from Lovecraft
SAMPLE_TEXT = """
Когда Роберт Блейк нашёл в тёмной колокольне церкви на Федерал-Хилл тот странный, неизвестной породы камень, он не знал, что это приведёт его к гибели. Камень лежал в металлической шкатулке необычной формы, покрытый неведомыми письменами.

Блейк взял камень и долго всматривался в его глубину. Ему казалось, что внутри мерцают какие-то огни, складываясь в образы далёких миров и невообразимых существ.

— Что это за место? — прошептал он, оглядываясь по сторонам. Пыль лежала здесь столетиями. Никто не поднимался в эту колокольню с тех пор, как церковь закрыли в 1877 году.

Он нашёл также дневник последнего священника. Записи становились всё более бессвязными к концу. «Оно приходит в темноте», — гласила последняя строка. «Свет его отпугивает, но оно ждёт. Всегда ждёт.»

Блейк унёс камень к себе домой. По ночам он видел странные сны: бескрайние пустоши под чужими звёздами, циклопические города из чёрного камня, существа, чьи очертания невозможно было удержать в памяти.

С каждым днём он становился бледнее. Соседи замечали, что он бормочет на неизвестном языке. А когда разразилась гроза и во всём квартале погас свет, Роберт Блейк закричал.
""".strip()


def run_bench(ref_audio: str, batch_sizes: list[int], fast_mode: bool = True):
    """Run benchmark for each batch size."""
    from src.tts.engine_qwen3tts import Qwen3TTSEngine

    results = []

    for bs in batch_sizes:
        label = f"batch_size={bs}" if bs > 1 else "sequential"
        logger.info("=" * 60)
        logger.info("Benchmark: %s", label)
        logger.info("=" * 60)

        output_path = Path(f"/tmp/qwen3_bench_bs{bs}.wav")

        engine = Qwen3TTSEngine(
            ref_audio=ref_audio,
            language="ru",
            speech_speed=0.85,
            fast_mode=fast_mode,
            batch_size=bs,
            warmup=True,
        )

        # Run synthesis and measure
        t0 = time.perf_counter()
        engine.synthesize(SAMPLE_TEXT, output_path)
        wall = time.perf_counter() - t0

        # Get audio duration
        import soundfile as sf
        audio, sr = sf.read(str(output_path))
        audio_dur = len(audio) / sr
        rtf = wall / audio_dur if audio_dur > 0 else 0

        results.append({
            "batch_size": bs,
            "wall_s": wall,
            "audio_s": audio_dur,
            "rtf": rtf,
        })

        logger.info("Result: %.1fs wall → %.1fs audio (RTF=%.3fx)", wall, audio_dur, rtf)

        # Free model memory between runs
        del engine
        import gc
        gc.collect()

    # Summary table
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Mode':<20} {'Wall (s)':<12} {'Audio (s)':<12} {'RTF':<10} {'vs seq':<10}")
    print("-" * 60)

    baseline_wall = results[0]["wall_s"] if results else 1.0
    for r in results:
        label = f"batch={r['batch_size']}" if r["batch_size"] > 1 else "sequential"
        speedup = baseline_wall / r["wall_s"] if r["wall_s"] > 0 else 0
        speedup_str = f"{speedup:.2f}x" if r["batch_size"] > 1 else "baseline"
        print(f"{label:<20} {r['wall_s']:<12.1f} {r['audio_s']:<12.1f} {r['rtf']:<10.3f} {speedup_str:<10}")

    print()
    for r in results:
        print(f"  Output: /tmp/qwen3_bench_bs{r['batch_size']}.wav")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-TTS batch generation")
    parser.add_argument("--ref", required=True, help="Path to reference audio WAV")
    parser.add_argument("--batch-sizes", default="1,4", help="Comma-separated batch sizes to test (default: 1,4)")
    parser.add_argument("--with-mastering", action="store_true", help="Enable mastering (default: skip)")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    run_bench(args.ref, batch_sizes, fast_mode=not args.with_mastering)
