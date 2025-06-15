"""Value objects exchanged between the detectors and the scanner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Detection", "FrameDetections", "ScanResult"]


@dataclass(frozen=True, slots=True)
class Detection:
    """One box a detector produced for one frame."""

    label: str
    confidence: float
    #: True when the detector's label space genuinely distinguishes ambulances.
    #: The public COCO fallback sets this False, and the API surfaces it.
    is_ambulance_class: bool = False


@dataclass(frozen=True, slots=True)
class FrameDetections:
    """Everything a detector returns for a single frame.

    ``render`` is a zero-argument callable producing the frame with boxes drawn
    on it. It is lazy on purpose: the scanner only ever renders the single
    best frame, so a 100-frame clip does not pay for 100 annotations.
    """

    detections: list[Detection] = field(default_factory=list)
    render: Callable[[], Any] | None = None


@dataclass(frozen=True, slots=True)
class ScanResult:
    """Outcome of scanning one clip. Pure data, no HTTP concepts."""

    ambulance: bool
    confidence: float
    label: str | None
    frames_read: int
    frames_inferred: int
    frames_with_ambulance: int
    best_frame: Any | None = None
    fallback_frame: Any | None = None

    @property
    def frame_to_return(self) -> Any | None:
        return self.best_frame if self.best_frame is not None else self.fallback_frame
