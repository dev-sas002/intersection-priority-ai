"""Domain-level exceptions shared by every layer."""

from __future__ import annotations

__all__ = ["ConfigurationError", "DetectorUnavailableError", "UnreadableVideoError"]


class ConfigurationError(RuntimeError):
    """The process is configured in a way that must not be allowed to start."""


class DetectorUnavailableError(RuntimeError):
    """A detector could not be constructed (missing weights, no network, ...)."""


class UnreadableVideoError(ValueError):
    """The uploaded bytes could not be opened as a video by OpenCV."""
