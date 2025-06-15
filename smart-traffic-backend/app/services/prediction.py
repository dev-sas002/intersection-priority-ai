"""The ``/predict`` use case.

This is where the scalability work lives, and none of it belongs in a route
handler:

* the upload is streamed to disk with a hard byte cap, so a large clip never
  sits in the event loop's memory;
* the scan (OpenCV decode + a torch forward pass) is CPU-bound and blocking, so
  it runs on a worker thread instead of stalling the loop;
* a semaphore bounds how many scans run at once, and a second counter bounds
  how many may *wait*, so simultaneous uploads queue politely and the surplus
  is rejected with 503 rather than thrashing the box.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any, BinaryIO

import anyio.to_thread

from app.config import Settings
from app.detection import VideoScanner
from app.detection.stub import STUB_NOTICE
from app.detection.types import ScanResult

__all__ = [
    "PredictionService",
    "ServiceBusyError",
    "UploadTooLargeError",
    "temp_suffix_for",
]

logger = logging.getLogger(__name__)

#: Only plain extensions such as ``.mp4`` are carried over to the temp file
#: name; anything else from the (untrusted) upload filename is discarded.
_SAFE_SUFFIX_RE = re.compile(r"\.[A-Za-z0-9]{1,10}\Z")

_CHUNK_BYTES = 1024 * 1024

_COCO_NOTICE = (
    "The active model has no ambulance class: it reports generic vehicle "
    "classes and cannot confirm that the vehicle is an ambulance. Load the "
    "purpose-trained weights (ITCS_WEIGHTS_PATH) for a real answer."
)


class UploadTooLargeError(ValueError):
    """The upload exceeded ``ITCS_MAX_UPLOAD_BYTES``."""


class ServiceBusyError(RuntimeError):
    """Too many scans already running or queued."""


@dataclass(frozen=True, slots=True)
class PredictionOutcome:
    result: ScanResult
    notice: str | None


class PredictionService:
    """Runs one bounded, concurrency-limited scan per upload."""

    def __init__(self, scanner: VideoScanner, settings: Settings) -> None:
        self._scanner = scanner
        self._settings = settings
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_inferences)
        self._in_flight = 0

    @property
    def scanner(self) -> VideoScanner:
        return self._scanner

    @property
    def slots_free(self) -> int:
        return max(0, self._settings.inference_queue_slots - self._in_flight)

    @property
    def notice(self) -> str | None:
        """Why this detector's answers must not be trusted, if that applies."""
        detector = self._scanner.detector
        if not detector.performs_real_detection:
            return STUB_NOTICE
        if not detector.identifies_ambulances:
            return _COCO_NOTICE
        return None

    async def predict(self, upload: Any, filename: str) -> PredictionOutcome:
        """Persist ``upload`` to a temp file and scan it."""
        if self._in_flight >= self._settings.inference_queue_slots:
            raise ServiceBusyError(
                f"All {self._settings.inference_queue_slots} inference slots are "
                "taken. Retry shortly."
            )

        self._in_flight += 1
        tmp_path: str | None = None
        try:
            tmp_path = await self._spool(upload, temp_suffix_for(filename))
            async with self._semaphore:
                result = await anyio.to_thread.run_sync(self._scanner.scan, tmp_path)
            return PredictionOutcome(result=result, notice=self.notice)
        finally:
            self._in_flight -= 1
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    async def _spool(self, upload: Any, suffix: str) -> str:
        """Stream the upload to a temp file, refusing anything oversized."""
        limit = self._settings.max_upload_bytes
        written = 0
        handle: BinaryIO
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            tmp_path = handle.name
            while True:
                chunk = await upload.read(_CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit:
                    handle.close()
                    os.remove(tmp_path)
                    raise UploadTooLargeError(
                        f"Upload exceeds the {limit} byte limit (ITCS_MAX_UPLOAD_BYTES)."
                    )
                handle.write(chunk)
        logger.debug("Spooled %d bytes to %s", written, tmp_path)
        return tmp_path


def temp_suffix_for(filename: str) -> str:
    """Derive a safe temp-file suffix from an untrusted upload filename."""
    suffix = os.path.splitext(os.path.basename(filename))[1]
    return suffix if _SAFE_SUFFIX_RE.fullmatch(suffix) else ""
