"""M4B audiobook builder using ffmpeg subprocess calls.

Pipeline per chapter:
  1. Concatenate WAV chunks for the chapter via ffmpeg concat demuxer
  2. Build chapter AAC files

Final pipeline:
  3. Concatenate all chapter AAC files
  4. Embed chapter markers via ffmpeg metadata file (;FFMETADATA1 format)
  5. Embed cover art if available
  6. Set title/artist metadata tags
  7. Output .m4b at AAC 128 kbps, 24000 Hz sample rate
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class AudioBuilder:
    """Build an M4B audiobook from per-chunk WAV files.

    Args:
        output_path: Destination .m4b file path.
        title: Book title (embedded as metadata).
        author: Book author (embedded as metadata).
        cover_bytes: Optional JPEG/PNG cover image bytes.
        sample_rate: WAV sample rate (default 24000 Hz — XTTS v2 output).
        bitrate: AAC bitrate (default "128k").
    """

    def __init__(
        self,
        output_path: str | Path,
        title: str = "",
        author: str = "",
        cover_bytes: Optional[bytes] = None,
        sample_rate: int = 24000,
        bitrate: str = "128k",
    ):
        self.output_path = Path(output_path)
        self.title = title
        self.author = author
        self.cover_bytes = cover_bytes
        self.sample_rate = sample_rate
        self.bitrate = bitrate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        chapter_wav_lists: list[list[Path]],
        chapter_titles: list[str],
    ) -> Path:
        """Build the M4B file from per-chapter lists of WAV chunk paths.

        Args:
            chapter_wav_lists: One list of WAV paths per chapter, in order.
            chapter_titles: Chapter title strings (same length as chapter_wav_lists).

        Returns:
            Path to the completed .m4b file.
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="audiobook_build_") as tmpdir:
            tmp = Path(tmpdir)

            # Step 1: Concatenate WAV chunks per chapter → chapter_N.wav
            chapter_wavs: list[Path] = []
            chapter_durations_ms: list[int] = []

            for i, wav_list in enumerate(chapter_wav_lists):
                if not wav_list:
                    continue
                chapter_wav = tmp / f"chapter_{i:04d}.wav"
                self._concat_wavs(wav_list, chapter_wav)
                duration_ms = self._get_duration_ms(chapter_wav)
                chapter_wavs.append(chapter_wav)
                chapter_durations_ms.append(duration_ms)

            if not chapter_wavs:
                raise RuntimeError("No audio content to build M4B from.")

            # Step 2: Concatenate all chapter WAVs → full_audio.wav
            full_wav = tmp / "full_audio.wav"
            self._concat_wavs(chapter_wavs, full_wav)

            # Step 3: Build ffmpeg metadata file with chapter markers
            titles_used = chapter_titles[: len(chapter_wavs)]
            metadata_path = tmp / "metadata.txt"
            self._write_metadata(metadata_path, titles_used, chapter_durations_ms)

            # Step 4: Write cover art if available
            cover_path: Optional[Path] = None
            if self.cover_bytes:
                cover_path = tmp / "cover.jpg"
                cover_path.write_bytes(self.cover_bytes)

            # Step 5: Encode to M4B
            self._encode_m4b(full_wav, metadata_path, cover_path)

        logger.info("M4B written to: %s", self.output_path)
        return self.output_path

    # ------------------------------------------------------------------
    # ffmpeg helpers
    # ------------------------------------------------------------------

    def _concat_wavs(self, wav_paths: list[Path], output: Path) -> None:
        """Concatenate multiple WAV files into one using ffmpeg concat demuxer."""
        # Write a concat list file
        list_file = output.with_suffix(".txt")
        with open(list_file, "w", encoding="utf-8") as f:
            for p in wav_paths:
                # Escape single quotes in paths
                escaped = str(p).replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(output),
        ]
        self._run(cmd, description=f"concat WAVs → {output.name}")

    def _get_duration_ms(self, wav_path: Path) -> int:
        """Return the duration of a WAV file in milliseconds."""
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(wav_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        duration_sec = float(result.stdout.strip())
        return int(duration_sec * 1000)

    def _write_metadata(
        self,
        path: Path,
        chapter_titles: list[str],
        chapter_durations_ms: list[int],
    ) -> None:
        """Write an ffmpeg chapter metadata file (;FFMETADATA1 format)."""
        lines: list[str] = [
            ";FFMETADATA1",
            f"title={self._escape_meta(self.title)}",
            f"artist={self._escape_meta(self.author)}",
            f"album={self._escape_meta(self.title)}",
            "",
        ]

        current_ms = 0
        for title, duration_ms in zip(chapter_titles, chapter_durations_ms):
            start_ms = current_ms
            end_ms = current_ms + duration_ms
            lines += [
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={start_ms}",
                f"END={end_ms}",
                f"title={self._escape_meta(title)}",
                "",
            ]
            current_ms = end_ms

        path.write_text("\n".join(lines), encoding="utf-8")

    def _encode_m4b(
        self,
        input_wav: Path,
        metadata_path: Path,
        cover_path: Optional[Path],
    ) -> None:
        """Encode WAV + metadata (+ optional cover) to M4B."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_wav),
            "-i", str(metadata_path),
            "-map_metadata", "1",
            "-map_chapters", "1",
        ]

        if cover_path and cover_path.exists():
            cmd += [
                "-i", str(cover_path),
                "-map", "0:a",
                "-map", "2:v",
                "-disposition:v", "attached_pic",
            ]
        else:
            cmd += ["-map", "0:a"]

        cmd += [
            "-c:a", "aac",
            "-b:a", self.bitrate,
            "-ar", str(self.sample_rate),
            "-movflags", "+faststart",
            str(self.output_path),
        ]

        self._run(cmd, description=f"encode → {self.output_path.name}")

    def _run(self, cmd: list[str], description: str = "") -> None:
        """Run an ffmpeg command, logging errors on failure."""
        logger.debug("Running: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error("ffmpeg stderr:\n%s", result.stderr)
            raise RuntimeError(
                f"ffmpeg failed ({description}): exit code {result.returncode}\n"
                f"{result.stderr[-2000:]}"
            )

    @staticmethod
    def _escape_meta(value: str) -> str:
        """Escape special characters for ffmpeg metadata values."""
        # ffmpeg metadata: escape '=', ';', '#', '\', and newlines
        value = value.replace("\\", "\\\\")
        value = value.replace("=", "\\=")
        value = value.replace(";", "\\;")
        value = value.replace("#", "\\#")
        value = value.replace("\n", "\\\n")
        return value
