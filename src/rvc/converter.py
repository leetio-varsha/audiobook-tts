"""RVC voice conversion — wraps rvc-python for inference."""

from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class RVCConverter:
    """Apply RVC voice conversion to WAV files.

    Args:
        model_path: Path to RVC ``.pth`` model file.
        index_path: Optional path to FAISS ``.index`` file.
        device: PyTorch device string (``"mps"``, ``"cuda"``, ``"cpu"``).
        pitch_shift: Semitone shift (positive = higher, negative = lower).
        f0_method: Pitch extraction method (``"rmvpe"`` recommended for quality).
    """

    def __init__(
        self,
        model_path: Path,
        index_path: Path | None = None,
        device: str | None = None,
        pitch_shift: int = 0,
        f0_method: str = "rmvpe",
    ):
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.model_path = Path(model_path)
        self.index_path = Path(index_path) if index_path else None
        self.device = device
        self.pitch_shift = pitch_shift
        self.f0_method = f0_method
        self._rvc = None

        logger.info(
            "RVCConverter: model=%s, index=%s, device=%s, pitch=%+d, f0=%s",
            self.model_path.name,
            self.index_path.name if self.index_path else "none",
            self.device,
            self.pitch_shift,
            self.f0_method,
        )

    def _ensure_loaded(self) -> None:
        """Lazy-load the RVC model on first use."""
        if self._rvc is not None:
            return

        try:
            from rvc_python.infer import RVCInference

            self._rvc = RVCInference(device=self.device)
            # Skip index file — faiss causes segfaults on macOS
            self._rvc.load_model(str(self.model_path), version="v2", index_path="")
            self._rvc.set_params(index_rate=0)
            logger.info("RVC model loaded on %s", self.device)
        except Exception:
            if self.device != "cpu":
                logger.warning("RVC failed on %s, falling back to CPU", self.device)
                from rvc_python.infer import RVCInference

                self._rvc = RVCInference(device="cpu")
                self._rvc.load_model(str(self.model_path), version="v2", index_path="")
                self._rvc.set_params(index_rate=0)
                self.device = "cpu"
                logger.info("RVC model loaded on CPU (fallback)")
            else:
                raise

    def convert(self, input_wav: Path, output_wav: Path) -> None:
        """Convert voice in *input_wav*, write result to *output_wav*."""
        self._ensure_loaded()
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        self._rvc.set_params(
            f0up_key=self.pitch_shift,
            f0method=self.f0_method,
        )
        self._rvc.infer_file(
            input_path=str(input_wav),
            output_path=str(output_wav),
        )

    def convert_in_place(self, wav_path: Path) -> None:
        """Convert voice in *wav_path*, overwriting the original file.

        If conversion fails, the original file is preserved and a warning is logged.
        """
        tmp = Path(tempfile.mktemp(suffix=".wav", dir=wav_path.parent))
        try:
            self.convert(wav_path, tmp)
            shutil.move(str(tmp), str(wav_path))
        except Exception as e:
            logger.warning("RVC conversion failed for %s, keeping original: %s", wav_path.name, e)
            if tmp.exists():
                tmp.unlink()

    @staticmethod
    def compute_model_hash(model_path: Path) -> str:
        """MD5 hash of the model file (first 8 chars) for cache key generation."""
        h = hashlib.md5()
        with open(model_path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                h.update(block)
        return h.hexdigest()[:8]

    def cleanup(self) -> None:
        """Release model from memory."""
        self._rvc = None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        logger.info("RVC model unloaded")
