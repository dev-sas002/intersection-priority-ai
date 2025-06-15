"""Detection layer: detectors, the registry that selects one, and the scanner.

Importing this package registers every built-in detector.
"""

# Imported for their @register side effects.
from app.detection import stub as _stub  # noqa: F401
from app.detection import yolo as _yolo  # noqa: F401
from app.detection.base import Detector
from app.detection.registry import available_detectors, build_detector, register
from app.detection.scanner import VideoScanner
from app.detection.types import Detection, FrameDetections, ScanResult

__all__ = [
    "Detection",
    "Detector",
    "FrameDetections",
    "ScanResult",
    "VideoScanner",
    "available_detectors",
    "build_detector",
    "register",
]
