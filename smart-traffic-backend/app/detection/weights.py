"""Resolution of model checkpoints.

The purpose-trained ambulance weights are *not* redistributable through this
repository, so the service must behave well in three situations: the weights
are present, they are absent (public fallback), and there is no network at all.
"""

from __future__ import annotations

import logging
import shutil
import urllib.error
import urllib.request
from pathlib import Path

from app.errors import DetectorUnavailableError

__all__ = ["PUBLIC_MODEL_BASE_URL", "ensure_custom_weights", "ensure_public_checkpoint"]

logger = logging.getLogger(__name__)

#: Upstream release assets for the public Ultralytics checkpoints.
PUBLIC_MODEL_BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0"

_DOWNLOAD_TIMEOUT_S = 120


def ensure_custom_weights(path: Path) -> Path:
    """Return ``path`` or raise a message that tells the operator what to do."""
    if path.is_file():
        return path
    raise DetectorUnavailableError(
        f"Ambulance weights not found at {path}.\n"
        "These weights are trained per-deployment and are deliberately NOT "
        "committed to this repository.\n"
        "Options:\n"
        "  1. Train or obtain a single-class YOLOv8 ambulance checkpoint and "
        f"place it at {path} (or point ITCS_WEIGHTS_PATH at it).\n"
        "  2. Run with ITCS_DETECTOR=coco to use the auto-downloaded public "
        "YOLOv8n checkpoint. It detects generic COCO vehicle classes only and "
        "cannot identify an ambulance.\n"
        "  3. Leave ITCS_DETECTOR=auto, which picks option 2 for you."
    )


def ensure_public_checkpoint(name: str, cache_dir: Path) -> Path:
    """Return a local path to the public checkpoint ``name``, downloading once.

    The file lands in ``cache_dir`` so a Docker image can pre-fetch it at build
    time and a container can then run with no network access at all.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / name
    if target.is_file() and target.stat().st_size > 0:
        logger.info("Using cached public checkpoint %s.", target)
        return target

    url = f"{PUBLIC_MODEL_BASE_URL}/{name}"
    partial = target.with_suffix(target.suffix + ".part")
    logger.info("Downloading public checkpoint %s -> %s", url, target)
    try:
        with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_S) as response:
            if getattr(response, "status", 200) != 200:
                raise DetectorUnavailableError(
                    f"Download of {url} returned HTTP {response.status}."
                )
            with partial.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        partial.unlink(missing_ok=True)
        raise DetectorUnavailableError(
            f"Could not download the public checkpoint {name} from {url}: {exc}. "
            "Pre-seed it into ITCS_MODEL_CACHE_DIR (the Docker image does this at "
            "build time) or run with ITCS_DETECTOR=yolo and your own weights."
        ) from exc

    partial.replace(target)
    logger.info("Downloaded %s (%d bytes).", target, target.stat().st_size)
    return target
