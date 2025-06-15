"""The detection pipeline: frames in, :class:`ScanResult` out.

Deliberately free of FastAPI, of ``UploadFile`` and of any knowledge of HTTP.
It takes a path on disk, walks it with OpenCV and drives whatever
:class:`~app.detection.base.Detector` it was handed, which is what makes
inference testable without spinning up the app.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import cv2

from app.config import Settings
from app.detection.base import Detector
from app.detection.types import Detection, FrameDetections, ScanResult
from app.errors import UnreadableVideoError
from app.imaging import annotate_frame

__all__ = ["VideoScanner"]

logger = logging.getLogger(__name__)

_NO_DETECTION_TEXT = "No Ambulance Detected"
_NO_DETECTION_COLOR = (0, 0, 255)


class VideoScanner:
    """Scans a clip for emergency vehicles using an injected detector."""

    def __init__(self, detector: Detector, settings: Settings) -> None:
        #: Public so a caller (or a test) can swap the backend on a live
        #: scanner without rebuilding the app.
        self.detector = detector
        self._settings = settings

    def scan(self, video_path: str | Path) -> ScanResult:
        """Scan ``video_path``; raises :class:`UnreadableVideoError` if OpenCV
        cannot open it. Always releases the capture."""
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise UnreadableVideoError("Uploaded file could not be read as a video.")
            return self._scan_capture(capture)
        finally:
            capture.release()

    # -- internals ---------------------------------------------------------
    def _scan_capture(self, capture: Any) -> ScanResult:
        settings = self._settings
        reader = _FrameReader(capture, settings.max_frames, settings.frame_stride)
        threshold = settings.confidence_threshold
        required_frames = settings.min_frames_with_ambulance

        frames_inferred = 0
        frames_with_ambulance = 0
        highest_confidence = 0.0
        best_label: str | None = None
        best_frame: Any | None = None
        fallback_frame: Any | None = None

        for batch in _batched(iter(reader), settings.batch_size):
            outputs = self.detector.detect(batch)
            for frame, output in zip(batch, outputs, strict=False):
                frames_inferred += 1
                if fallback_frame is None:
                    fallback_frame = annotate_frame(
                        frame, _NO_DETECTION_TEXT, _NO_DETECTION_COLOR
                    )

                best = _best_ambulance(output, threshold)
                if best is None:
                    continue

                # Count frames, not boxes: several boxes in one frame are still
                # a single frame of evidence.
                frames_with_ambulance += 1
                if best.confidence > highest_confidence:
                    highest_confidence = best.confidence
                    best_label = best.label
                    best_frame = _render(output, frame)

                if frames_with_ambulance >= required_frames:
                    return ScanResult(
                        ambulance=True,
                        confidence=highest_confidence,
                        label=best_label,
                        frames_read=reader.reads,
                        frames_inferred=frames_inferred,
                        frames_with_ambulance=frames_with_ambulance,
                        best_frame=best_frame,
                        fallback_frame=fallback_frame,
                    )

        return ScanResult(
            ambulance=False,
            confidence=0.0,
            label=None,
            frames_read=reader.reads,
            frames_inferred=frames_inferred,
            frames_with_ambulance=frames_with_ambulance,
            best_frame=None,
            fallback_frame=fallback_frame,
        )


class _FrameReader:
    """Pulls frames off a capture, honouring the frame budget and the stride.

    Striding is the cheap scalability lever: sampling every Nth frame covers N
    times more wall-clock video for the same number of forward passes, and an
    ambulance stays in shot for far longer than one frame.
    """

    def __init__(self, capture: Any, max_frames: int, stride: int) -> None:
        self._capture = capture
        self._max_frames = max_frames
        self._stride = stride
        self.reads = 0

    def __iter__(self) -> Iterator[Any]:
        yielded = 0
        index = 0
        while yielded < self._max_frames:
            ret, frame = self._capture.read()
            self.reads += 1
            if not ret:
                return
            if index % self._stride == 0:
                yielded += 1
                yield frame
            index += 1


def _best_ambulance(output: FrameDetections, threshold: float) -> Detection | None:
    """Highest-confidence ambulance-class detection at or above ``threshold``."""
    candidates = [
        detection
        for detection in output.detections
        if detection.is_ambulance_class and detection.confidence >= threshold
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda detection: detection.confidence)


def _render(output: FrameDetections, frame: Any) -> Any:
    """Annotated frame from the detector, falling back to the raw frame."""
    if output.render is None:
        return frame
    try:
        return output.render()
    except Exception:  # pragma: no cover - defensive; a bad plot must not 500
        logger.warning("Detector failed to render an annotated frame.", exc_info=True)
        return frame


def _batched(items: Iterator[Any], size: int) -> Iterator[Sequence[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
