"""Process configuration, parsed once from the environment.

Every tunable lives here as a field on :class:`Settings` so that the rest of the
code never reads ``os.environ`` directly. ``Settings`` is a frozen dataclass and
is built with :meth:`Settings.from_env`, which makes it trivial to construct a
bespoke configuration in a test without touching the real environment.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from app.errors import ConfigurationError

__all__ = ["BACKEND_DIR", "DEFAULT_WEIGHTS_PATH", "Settings"]

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS_PATH = BACKEND_DIR / "weights" / "best.pt"

#: Environments in which the fake ``stub`` detector may be used at all.
_STUB_ALLOWED_ENVIRONMENTS = frozenset({"development", "test"})


def _get(env: Mapping[str, str], name: str, default: str) -> str:
    raw = env.get(name)
    return default if raw is None or not raw.strip() else raw.strip()


def _get_int(env: Mapping[str, str], name: str, default: int, *, minimum: int = 1) -> int:
    raw = _get(env, name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}.") from exc
    if value < minimum:
        raise ConfigurationError(f"{name} must be >= {minimum}, got {value}.")
    return value


def _get_float(
    env: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = _get(env, name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number, got {raw!r}.") from exc
    if minimum is not None and value < minimum:
        raise ConfigurationError(f"{name} must be >= {minimum}, got {value}.")
    if maximum is not None and value > maximum:
        raise ConfigurationError(f"{name} must be <= {maximum}, got {value}.")
    return value


def _get_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _get(env, name, "true" if default else "false").lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be a boolean, got {raw!r}.")


def _get_list(env: Mapping[str, str], name: str, default: str) -> list[str]:
    return [item.strip() for item in _get(env, name, default).split(",") if item.strip()]


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable snapshot of every environment-driven knob."""

    # --- Environment ------------------------------------------------------
    #: Defaults to ``production`` deliberately: the fake detector is refused
    #: unless an operator has *also* opted out of production explicitly.
    environment: str = "production"

    # --- Detector selection ----------------------------------------------
    detector: str = "auto"
    weights_path: Path = DEFAULT_WEIGHTS_PATH
    #: Public checkpoint used by the ``coco`` fallback detector.
    public_model: str = "yolov8n.pt"
    model_cache_dir: Path = BACKEND_DIR / "weights"
    #: COCO labels treated as *candidate* emergency vehicles by the fallback.
    coco_emergency_classes: list[str] = field(default_factory=lambda: ["truck", "bus"])

    # --- Detection tuning -------------------------------------------------
    confidence_threshold: float = 0.75
    min_frames_with_ambulance: int = 1
    max_frames: int = 100
    frame_stride: int = 1
    batch_size: int = 8
    image_size: int = 640
    device: str = "auto"

    # --- Request handling -------------------------------------------------
    max_concurrent_inferences: int = 2
    inference_queue_slots: int = 8
    max_upload_bytes: int = 64 * 1024 * 1024
    allowed_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173"])

    # --- Response payload -------------------------------------------------
    #: A raw 4K frame base64-encodes to ~1.7 MB of JSON. These bound it.
    return_frame: bool = True
    max_frame_edge: int = 1280
    jpeg_quality: int = 85

    # --- Signal priority policy ------------------------------------------
    priority_min_confidence: float = 0.80
    priority_corroborating_sightings: int = 2
    priority_corroboration_window_s: float = 10.0
    priority_min_hold_s: float = 8.0
    priority_max_hold_s: float = 45.0
    priority_clear_after_s: float = 6.0
    priority_cooldown_s: float = 20.0

    # --- Logging ----------------------------------------------------------
    log_level: str = "INFO"

    @property
    def stub_allowed(self) -> bool:
        return self.environment in _STUB_ALLOWED_ENVIRONMENTS

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env

        weights = _get(env, "ITCS_WEIGHTS_PATH", str(DEFAULT_WEIGHTS_PATH))
        cache = _get(env, "ITCS_MODEL_CACHE_DIR", str(BACKEND_DIR / "weights"))

        settings = cls(
            environment=_get(env, "ITCS_ENV", "production").lower(),
            detector=_get(env, "ITCS_DETECTOR", "auto").lower(),
            weights_path=Path(weights).expanduser().resolve(),
            public_model=_get(env, "ITCS_PUBLIC_MODEL", "yolov8n.pt"),
            model_cache_dir=Path(cache).expanduser().resolve(),
            coco_emergency_classes=_get_list(
                env, "ITCS_COCO_EMERGENCY_CLASSES", "truck,bus"
            ),
            confidence_threshold=_get_float(
                env, "ITCS_CONFIDENCE_THRESHOLD", 0.75, minimum=0.0, maximum=1.0
            ),
            min_frames_with_ambulance=_get_int(env, "ITCS_MIN_FRAMES_WITH_AMBULANCE", 1),
            max_frames=_get_int(env, "ITCS_MAX_FRAMES", 100),
            frame_stride=_get_int(env, "ITCS_FRAME_STRIDE", 1),
            batch_size=_get_int(env, "ITCS_BATCH_SIZE", 8),
            image_size=_get_int(env, "ITCS_IMAGE_SIZE", 640, minimum=32),
            device=_get(env, "ITCS_DEVICE", "auto").lower(),
            max_concurrent_inferences=_get_int(env, "ITCS_MAX_CONCURRENT_INFERENCES", 2),
            inference_queue_slots=_get_int(env, "ITCS_INFERENCE_QUEUE_SLOTS", 8),
            max_upload_bytes=_get_int(
                env, "ITCS_MAX_UPLOAD_BYTES", 64 * 1024 * 1024, minimum=1024
            ),
            allowed_origins=_get_list(
                env, "ITCS_ALLOWED_ORIGINS", "http://localhost:5173"
            ),
            return_frame=_get_bool(env, "ITCS_RETURN_FRAME", True),
            max_frame_edge=_get_int(env, "ITCS_MAX_FRAME_EDGE", 1280, minimum=0),
            jpeg_quality=_get_int(env, "ITCS_JPEG_QUALITY", 85, minimum=1),
            priority_min_confidence=_get_float(
                env, "ITCS_PRIORITY_MIN_CONFIDENCE", 0.80, minimum=0.0, maximum=1.0
            ),
            priority_corroborating_sightings=_get_int(
                env, "ITCS_PRIORITY_CORROBORATING_SIGHTINGS", 2
            ),
            priority_corroboration_window_s=_get_float(
                env, "ITCS_PRIORITY_CORROBORATION_WINDOW_S", 10.0, minimum=0.1
            ),
            priority_min_hold_s=_get_float(
                env, "ITCS_PRIORITY_MIN_HOLD_S", 8.0, minimum=0.0
            ),
            priority_max_hold_s=_get_float(
                env, "ITCS_PRIORITY_MAX_HOLD_S", 45.0, minimum=0.1
            ),
            priority_clear_after_s=_get_float(
                env, "ITCS_PRIORITY_CLEAR_AFTER_S", 6.0, minimum=0.1
            ),
            priority_cooldown_s=_get_float(
                env, "ITCS_PRIORITY_COOLDOWN_S", 20.0, minimum=0.0
            ),
            log_level=_get(env, "ITCS_LOG_LEVEL", "INFO").upper(),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Reject combinations that must never reach a running process."""
        if self.jpeg_quality > 100:
            raise ConfigurationError(
                f"ITCS_JPEG_QUALITY must be 1-100, got {self.jpeg_quality}."
            )
        if self.priority_min_hold_s > self.priority_max_hold_s:
            raise ConfigurationError(
                "ITCS_PRIORITY_MIN_HOLD_S must not exceed ITCS_PRIORITY_MAX_HOLD_S."
            )
        if self.inference_queue_slots < self.max_concurrent_inferences:
            raise ConfigurationError(
                "ITCS_INFERENCE_QUEUE_SLOTS must be >= ITCS_MAX_CONCURRENT_INFERENCES."
            )
        if self.detector == "stub" and not self.stub_allowed:
            raise ConfigurationError(
                "ITCS_DETECTOR=stub performs NO detection and is refused while "
                f"ITCS_ENV={self.environment!r}. Set ITCS_ENV=development (or "
                "'test') as a second, deliberate opt-in if you really want the "
                "fake detector."
            )
