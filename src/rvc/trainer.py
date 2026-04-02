"""RVC model training wrapper — calls RVC-WebUI scripts via subprocess."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Paths
TOOLS_DIR = Path(__file__).resolve().parent.parent.parent / "tools" / "rvc-webui"
ASSETS_DIR = TOOLS_DIR / "assets"
LOGS_DIR = TOOLS_DIR / "logs"

HUBERT_PATH = ASSETS_DIR / "hubert" / "hubert_base.pt"
RMVPE_PATH = ASSETS_DIR / "rmvpe" / "rmvpe.pt"
PRETRAINED_G = ASSETS_DIR / "pretrained_v2" / "f0G40k.pth"
PRETRAINED_D = ASSETS_DIR / "pretrained_v2" / "f0D40k.pth"


def _check_assets():
    """Verify pretrained models are downloaded."""
    missing = []
    for p in [HUBERT_PATH, RMVPE_PATH, PRETRAINED_G, PRETRAINED_D]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError(f"Missing pretrained models: {missing}")


def _run(cmd: list[str], cwd: str | None = None, env: dict | None = None) -> str:
    """Run subprocess, log output, raise on failure."""
    full_env = os.environ.copy()
    full_env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    full_env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    full_env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    if env:
        full_env.update(env)

    logger.info("Running: %s", " ".join(cmd[:5]) + ("..." if len(cmd) > 5 else ""))
    result = subprocess.run(
        cmd,
        cwd=cwd or str(TOOLS_DIR),
        capture_output=True,
        text=True,
        env=full_env,
    )
    if result.stdout:
        for line in result.stdout.strip().split("\n")[-10:]:
            logger.info("  %s", line)
    if result.returncode != 0:
        logger.error("Command failed:\n%s", result.stderr[-2000:] if result.stderr else "no stderr")
        raise RuntimeError(f"Training step failed: {result.stderr[-500:]}")
    return result.stdout


def train_voice(
    name: str,
    dataset_dir: Path,
    total_epochs: int = 100,
    batch_size: int = 8,
    sample_rate: str = "40k",
    version: str = "v2",
    save_every_epoch: int = 25,
    on_progress: callable = None,
) -> tuple[Path, Path | None]:
    """Train an RVC voice model from audio samples.

    Args:
        name: Voice profile name (used as experiment dir name).
        dataset_dir: Directory containing WAV/MP3 audio files.
        total_epochs: Total training epochs (default 100).
        batch_size: Training batch size (default 8 for 64GB RAM).
        sample_rate: Sample rate config — "40k" or "48k".
        version: RVC version — "v1" or "v2".
        save_every_epoch: Save checkpoint every N epochs.
        on_progress: Callback(step_name, detail) for progress updates.

    Returns:
        Tuple of (model_pth_path, index_path_or_none).
    """
    _check_assets()
    python = sys.executable
    cwd = str(TOOLS_DIR)

    exp_dir = LOGS_DIR / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    def progress(step: str, detail: str = ""):
        logger.info("[%s] %s %s", name, step, detail)
        if on_progress:
            on_progress(step, detail)

    # ---------------------------------------------------------------
    # Step 1: Preprocess — slice audio into training segments
    # ---------------------------------------------------------------
    progress("Preprocessing", "Slicing audio into segments...")

    sr_map = {"32k": 32000, "40k": 40000, "48k": 48000}
    sr_int = sr_map.get(sample_rate, 40000)

    _run([
        python, "infer/modules/train/preprocess.py",
        str(dataset_dir),   # inp_root
        str(sr_int),         # sr
        "2",                 # n_p (processes)
        str(exp_dir),        # exp_dir
        "False",             # noparallel
        "3.7",               # per (segment length)
    ], cwd=cwd)

    progress("Preprocessing", "Done.")

    # ---------------------------------------------------------------
    # Step 2: Extract F0 (pitch) using RMVPE
    # ---------------------------------------------------------------
    progress("F0 Extraction", "Extracting pitch contours with RMVPE...")

    _run([
        python, "infer/modules/train/extract/extract_f0_rmvpe.py",
        "1",                 # n_part
        "0",                 # i_part
        "0",                 # i_gpu (ignored on MPS)
        str(exp_dir),        # exp_dir
        "True",              # is_half
    ], cwd=cwd)

    progress("F0 Extraction", "Done.")

    # ---------------------------------------------------------------
    # Step 3: Extract HuBERT features
    # ---------------------------------------------------------------
    progress("Feature Extraction", "Extracting HuBERT embeddings...")

    device = "mps" if _has_mps() else "cpu"
    _run([
        python, "infer/modules/train/extract_feature_print.py",
        device,              # device
        "1",                 # n_part
        "0",                 # i_part
        str(exp_dir),        # exp_dir
        version,             # version (v1/v2)
        "False",             # is_half (False for MPS)
    ], cwd=cwd)

    progress("Feature Extraction", "Done.")

    # ---------------------------------------------------------------
    # Step 4: Prepare filelist
    # ---------------------------------------------------------------
    progress("Preparing", "Building training filelist...")
    _prepare_filelist(exp_dir, sr_int, version)
    progress("Preparing", "Done.")

    # ---------------------------------------------------------------
    # Step 5: Copy config
    # ---------------------------------------------------------------
    config_src = TOOLS_DIR / "configs" / version / f"{sample_rate}.json"
    if not config_src.exists():
        # Fall back to closest available
        configs = list((TOOLS_DIR / "configs" / version).glob("*.json"))
        config_src = configs[0] if configs else None
    if config_src:
        shutil.copy2(str(config_src), str(exp_dir / "config.json"))

    # ---------------------------------------------------------------
    # Step 6: Train model
    # ---------------------------------------------------------------
    progress("Training", f"Training for {total_epochs} epochs (batch_size={batch_size})...")

    _run([
        python, "infer/modules/train/train.py",
        "-se", str(save_every_epoch),
        "-te", str(total_epochs),
        "-pg", str(PRETRAINED_G),
        "-pd", str(PRETRAINED_D),
        "-sr", sample_rate,
        "-bs", str(batch_size),
        "-g", "0",
        "-e", str(exp_dir),
        "-v", version,
        "-l", "1",           # save latest
        "-c", "0",           # don't cache in GPU
        "-sw", "1",          # save weights
        "-f0", "1",          # use f0
    ], cwd=cwd)

    progress("Training", "Done.")

    # ---------------------------------------------------------------
    # Step 7: Build FAISS index
    # ---------------------------------------------------------------
    progress("Indexing", "Building FAISS search index...")
    index_path = _build_index(exp_dir, version)
    progress("Indexing", "Done.")

    # ---------------------------------------------------------------
    # Step 8: Find best model weights
    # ---------------------------------------------------------------
    model_path = _find_model(exp_dir, name)

    progress("Complete", f"Model: {model_path}")
    return model_path, index_path


def _has_mps() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


def _prepare_filelist(exp_dir: Path, sr: int, version: str):
    """Generate the training filelist (filelist.txt) from preprocessed data."""
    gt_wavs = exp_dir / "0_gt_wavs"
    features_dir = exp_dir / f"3_feature{'768' if version == 'v2' else '256'}"
    f0_dir = exp_dir / "2a_f0"
    f0nsf_dir = exp_dir / "2b-f0nsf"

    lines = []
    if gt_wavs.exists():
        for wav in sorted(gt_wavs.glob("*.wav")):
            stem = wav.stem
            feat = features_dir / f"{stem}.npy"
            f0 = f0_dir / f"{stem}.wav.npy"
            f0nsf = f0nsf_dir / f"{stem}.wav.npy"
            if feat.exists() and f0.exists() and f0nsf.exists():
                lines.append(f"{wav}|{feat}|{f0}|{f0nsf}|0")

    filelist_path = exp_dir / "filelist.txt"
    filelist_path.write_text("\n".join(lines))
    logger.info("Training filelist: %d entries", len(lines))

    # Also write to the config
    config_path = exp_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        config["data"]["training_files"] = str(filelist_path)
        config_path.write_text(json.dumps(config, indent=2))


def _build_index(exp_dir: Path, version: str) -> Path | None:
    """Build FAISS index from extracted features."""
    import faiss
    import numpy as np

    feature_dim = 768 if version == "v2" else 256
    features_dir = exp_dir / f"3_feature{feature_dim}"

    if not features_dir.exists():
        return None

    npys = sorted(features_dir.glob("*.npy"))
    if not npys:
        return None

    # Collect all features
    feats = []
    for npy in npys:
        feat = np.load(str(npy))
        feats.append(feat)

    big_npy = np.concatenate(feats, axis=0).astype("float32")

    # Build Flat index (simple, robust, no OMP issues)
    index = faiss.IndexFlatL2(feature_dim)
    index.add(big_npy)

    index_path = exp_dir / f"added_index.index"
    faiss.write_index(index, str(index_path))
    logger.info("FAISS index built: %d vectors", big_npy.shape[0])
    return index_path


def _find_model(exp_dir: Path, name: str) -> Path:
    """Find the final model .pth file in the weights directory."""
    # RVC saves weights to logs/{name}/weights/ or to the exp_dir
    weights_dir = exp_dir / "weights"

    # Look for the latest .pth that contains the experiment name
    candidates = []
    for d in [weights_dir, exp_dir, TOOLS_DIR / "assets" / "weights"]:
        if d.exists():
            candidates.extend(d.glob("*.pth"))

    if not candidates:
        raise FileNotFoundError(f"No model .pth found in {exp_dir}")

    # Prefer files with the experiment name, sort by modification time
    named = [c for c in candidates if name.lower() in c.name.lower()]
    best = sorted(named or candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return best[0]
