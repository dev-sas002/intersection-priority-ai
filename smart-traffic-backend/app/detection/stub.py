"""A detector that performs no detection at all.

This exists so the HTTP layer can be exercised without torch, weights or a
video codec. It is dangerous by nature, so it is fenced off twice:

1. :func:`app.detection.registry.resolve_detector_name` never selects it --
   ``ITCS_DETECTOR=stub`` has to be typed out.
2. :meth:`app.config.Settings.validate` refuses to build a configuration with
   it unless ``ITCS_ENV`` is ``development`` or ``test``. ``ITCS_ENV`` defaults
   to ``production``, so a single forgotten variable is not enough.

On top of that it sets ``performs_real_detection = False``, which the registry
logs as CRITICAL at startup, ``/health`` reports, and every ``/predict``
response carries as a ``notice``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.config import Settings
from app.detection.base import Detector
from app.detection.registry import register
from app.detection.types import Detection, FrameDetections

__all__ = ["StubDetector", "build_stub"]

STUB_NOTICE = (
    "DETECTION DISABLED: the 'stub' detector fabricates this result and "
    "inspects no pixels. Development and test use only."
)


class StubDetector:
    """Returns a scripted answer for every frame."""

    def __init__(self, *, detect_ambulance: bool = True, confidence: float = 0.9) -> None:
        self.name = "stub"
        self.performs_real_detection = False
        self.identifies_ambulances = False
        self.description = STUB_NOTICE
        self.detect_ambulance = detect_ambulance
        self.confidence = confidence
        self.calls = 0

    def warmup(self) -> None:
        return None

    def detect(self, frames: Sequence[Any]) -> list[FrameDetections]:
        self.calls += len(frames)
        results: list[FrameDetections] = []
        for frame in frames:
            detections = (
                [
                    Detection(
                        label="ambulance",
                        confidence=self.confidence,
                        is_ambulance_class=True,
                    )
                ]
                if self.detect_ambulance
                else []
            )
            results.append(
                FrameDetections(detections=detections, render=lambda f=frame: f)
            )
        return results


@register("stub")
def build_stub(settings: Settings) -> Detector:
    return StubDetector(confidence=max(settings.confidence_threshold, 0.9))
