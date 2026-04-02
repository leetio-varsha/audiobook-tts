"""Voice profile management — import audio, download from YouTube, manage RVC models."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.rvc.audio_prep import normalize_audio

logger = logging.getLogger(__name__)

VOICES_DIR = Path("./voices").resolve()
VOICES_DIR.mkdir(parents=True, exist_ok=True)


class VoiceManager:
    """Manage voice profiles stored under ``voices/{name}/``."""

    def __init__(self, voices_dir: Path = VOICES_DIR):
        self.voices_dir = voices_dir
        self.voices_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def import_audio_files(self, name: str, files: list[Path]) -> Path:
        """Copy and normalize WAV/MP3 files into a voice profile's samples dir."""
        profile_dir = self._ensure_profile(name)
        samples_dir = profile_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        for i, f in enumerate(files):
            dest = samples_dir / f"sample_{i:03d}.wav"
            normalize_audio(Path(f), dest)

        self._update_meta(name, source="upload")
        logger.info("Imported %d audio files for voice '%s'", len(files), name)
        return profile_dir

    def import_from_youtube(self, name: str, url: str) -> Path:
        """Download audio from a YouTube URL via yt-dlp."""
        profile_dir = self._ensure_profile(name)
        samples_dir = profile_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        raw_path = samples_dir / "youtube_raw.%(ext)s"
        out_path = samples_dir / "youtube.wav"

        cmd = [
            "yt-dlp",
            "-x", "--audio-format", "wav",
            "--audio-quality", "0",
            "-o", str(raw_path),
            "--no-playlist",
            url,
        ]
        logger.info("Downloading audio from YouTube: %s", url)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {result.stderr.strip()}")

        # Find the downloaded file (yt-dlp may produce youtube_raw.wav)
        downloaded = list(samples_dir.glob("youtube_raw.*"))
        if not downloaded:
            raise RuntimeError("yt-dlp produced no output file")

        # Normalize to standard format
        normalize_audio(downloaded[0], out_path)

        # Clean up raw file if different from output
        for f in downloaded:
            if f != out_path and f.exists():
                f.unlink()

        self._update_meta(name, source="youtube")
        logger.info("YouTube audio imported for voice '%s'", name)
        return profile_dir

    def import_rvc_model(self, name: str, model_path: Path, index_path: Path | None = None) -> Path:
        """Import pre-trained RVC .pth and optional .index files."""
        profile_dir = self._ensure_profile(name)

        shutil.copy2(str(model_path), str(profile_dir / "model.pth"))
        if index_path:
            shutil.copy2(str(index_path), str(profile_dir / "model.index"))

        self._update_meta(name, source="rvc_import", has_model=True)
        logger.info("RVC model imported for voice '%s'", name)
        return profile_dir

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def list_voices(self) -> list[dict]:
        """Return all voice profiles with metadata."""
        voices = []
        for meta_path in sorted(self.voices_dir.glob("*/meta.json")):
            try:
                meta = json.loads(meta_path.read_text())
                meta["path"] = str(meta_path.parent)
                voices.append(meta)
            except (json.JSONDecodeError, OSError):
                continue
        return voices

    def list_voice_names(self) -> list[str]:
        """Return names of voices that have an RVC model ready."""
        return [v["name"] for v in self.list_voices() if v.get("has_model")]

    def get_voice(self, name: str) -> dict | None:
        """Get voice profile metadata by name."""
        meta_path = self.voices_dir / name / "meta.json"
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        meta["path"] = str(meta_path.parent)
        return meta

    def get_model_paths(self, name: str) -> tuple[Path | None, Path | None]:
        """Return (model.pth, model.index) paths for a voice, or None if missing."""
        profile_dir = self.voices_dir / name
        model = profile_dir / "model.pth"
        index = profile_dir / "model.index"
        return (
            model if model.exists() else None,
            index if index.exists() else None,
        )

    def has_model(self, name: str) -> bool:
        """Check if voice has an RVC model file."""
        return (self.voices_dir / name / "model.pth").exists()

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_voice(self, name: str) -> bool:
        """Remove a voice profile entirely."""
        profile_dir = self.voices_dir / name
        if profile_dir.exists():
            shutil.rmtree(profile_dir)
            logger.info("Deleted voice '%s'", name)
            return True
        return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_profile(self, name: str) -> Path:
        profile_dir = self.voices_dir / name
        profile_dir.mkdir(parents=True, exist_ok=True)
        return profile_dir

    def _update_meta(self, name: str, source: str | None = None, has_model: bool | None = None) -> None:
        meta_path = self.voices_dir / name / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        else:
            meta = {"name": name, "created": datetime.now(timezone.utc).isoformat()}

        if source:
            meta["source"] = source
        if has_model is not None:
            meta["has_model"] = has_model
        elif self.has_model(name):
            meta["has_model"] = True

        meta["updated"] = datetime.now(timezone.utc).isoformat()
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
