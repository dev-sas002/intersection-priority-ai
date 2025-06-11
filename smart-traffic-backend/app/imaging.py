"""Frame-level helpers: downscaling, JPEG encoding and text overlays.

Kept separate from both the HTTP layer and the detectors so they can be unit
tested against a bare numpy array.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

import cv2

__all__ = ["annotate_frame", "encode_frame", "fit_within"]

logger = logging.getLogger(__name__)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def fit_within(frame: Any, max_edge: int) -> Any:
    """Downscale ``frame`` so its longest edge is at most ``max_edge``.

    Returns the frame unchanged when it already fits or when ``max_edge`` is
    non-positive (the documented "do not resize" value). ``INTER_AREA`` is the
    right filter for shrinking -- it averages the source pixels instead of
    sampling them, so the annotation boxes stay readable.
    """
    if max_edge <= 0:
        return frame
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_edge:
        return frame
    scale = max_edge / longest
    return cv2.resize(
        frame,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def encode_frame(
    frame: Any,
    *,
    max_edge: int = 0,
    quality: int = 85,
) -> str | None:
    """JPEG-encode ``frame`` and return it base64 encoded, or ``None``.

    A raw 4K frame base64-encodes to well over a megabyte of JSON, which
    dominates the response and the memory the server holds per in-flight
    request. ``max_edge`` and ``quality`` bound that cost; see the README's
    design notes for the measured numbers.
    """
    if frame is None:
        return None
    success, buffer = cv2.imencode(
        ".jpg",
        fit_within(frame, max_edge),
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not success:
        logger.warning("Failed to JPEG-encode frame.")
        return None
    return base64.b64encode(buffer).decode("utf-8")


def annotate_frame(
    frame: Any,
    text: str,
    color: tuple[int, int, int],
    scale: float = 1.0,
) -> Any:
    """Return a *copy* of ``frame`` with ``text`` drawn in the top-left corner."""
    labeled_frame = frame.copy()
    cv2.putText(labeled_frame, text, (10, 30), _FONT, scale, color, 2)
    return labeled_frame
