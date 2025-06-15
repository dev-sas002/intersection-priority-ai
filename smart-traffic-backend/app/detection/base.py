"""The detector seam.

Every detection backend implements :class:`Detector`. The scanner, the HTTP
layer and the tests all talk to this interface only, so swapping YOLOv8 for a
different model (or a remote inference service) means adding one module and one
registry entry -- nothing else changes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from app.detection.types import FrameDetections

__all__ = ["Detector"]


@runtime_checkable
class Detector(Protocol):
    """A batch image detector."""

    #: Registry key, e.g. ``"yolo"``.
    name: str

    #: False for detectors that fabricate results (the ``stub``). Surfaced on
    #: ``/health`` and on every prediction so a fake can never masquerade as
    #: a real one.
    performs_real_detection: bool

    #: False when the label space cannot actually distinguish an ambulance from
    #: another large vehicle (the public COCO fallback).
    identifies_ambulances: bool

    #: Human-readable provenance shown in ``/health``, e.g. the weights path.
    description: str

    def warmup(self) -> None:
        """Load weights and run any first-call initialisation. Called once."""

    def detect(self, frames: Sequence[Any]) -> list[FrameDetections]:
        """Run inference over a batch of frames, one result per input frame."""
