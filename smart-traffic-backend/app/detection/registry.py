"""Name -> detector factory registry.

Adding a backend is a two-line change: write a module that implements
:class:`~app.detection.base.Detector` and decorate its factory with
``@register("my-model")``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from app.config import Settings
from app.detection.base import Detector
from app.errors import ConfigurationError

__all__ = ["available_detectors", "build_detector", "register", "resolve_detector_name"]

logger = logging.getLogger(__name__)

DetectorFactory = Callable[[Settings], Detector]

_REGISTRY: dict[str, DetectorFactory] = {}


def register(name: str) -> Callable[[DetectorFactory], DetectorFactory]:
    """Class/function decorator that records a detector factory under ``name``."""

    def decorator(factory: DetectorFactory) -> DetectorFactory:
        if name in _REGISTRY:
            raise ConfigurationError(f"Detector {name!r} is already registered.")
        _REGISTRY[name] = factory
        return factory

    return decorator


def available_detectors() -> list[str]:
    return sorted(_REGISTRY)


def resolve_detector_name(settings: Settings) -> str:
    """Turn ``ITCS_DETECTOR`` (possibly ``auto``) into a concrete registry key.

    ``auto`` prefers the purpose-trained weights when they are present on disk
    and otherwise falls back to the public COCO checkpoint, so a fresh clone
    starts and serves real inference without any manual step. ``auto`` never
    selects ``stub``.
    """
    requested = settings.detector
    if requested != "auto":
        if requested not in _REGISTRY:
            raise ConfigurationError(
                f"Unknown ITCS_DETECTOR={requested!r}. "
                f"Available: {', '.join(available_detectors())}, auto."
            )
        return requested

    if settings.weights_path.is_file():
        logger.info("ITCS_DETECTOR=auto -> 'yolo' (found %s).", settings.weights_path)
        return "yolo"

    logger.warning(
        "ITCS_DETECTOR=auto -> 'coco': no ambulance weights at %s, falling back "
        "to the public %s checkpoint. It detects generic COCO vehicle classes "
        "and CANNOT identify an ambulance.",
        settings.weights_path,
        settings.public_model,
    )
    return "coco"


def build_detector(settings: Settings) -> Detector:
    """Construct the detector selected by ``settings``."""
    name = resolve_detector_name(settings)
    detector = _REGISTRY[name](settings)
    if not detector.performs_real_detection:
        banner = "=" * 72
        logger.critical(
            "%s\n!! DETECTOR %r PERFORMS NO DETECTION -- every result is fabricated.\n"
            "!! Development aid only; refused unless ITCS_ENV is development/test.\n%s",
            banner,
            name,
            banner,
        )
    return detector
