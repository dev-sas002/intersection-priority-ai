"""Upload spooling, the byte cap and the concurrency limits."""

from __future__ import annotations

import asyncio
import os

import pytest

from app.detection.scanner import VideoScanner
from app.detection.types import ScanResult
from app.services.prediction import (
    PredictionService,
    ServiceBusyError,
    UploadTooLargeError,
    temp_suffix_for,
)
from tests.conftest import ScriptedDetector, make_settings


class FakeUpload:
    """Minimal stand-in for Starlette's ``UploadFile``."""

    def __init__(self, data: bytes, chunk: int = 64) -> None:
        self._data = data
        self._chunk = chunk
        self._offset = 0

    async def read(self, size: int = -1) -> bytes:
        size = self._chunk if size < 0 else min(size, self._chunk)
        piece = self._data[self._offset : self._offset + size]
        self._offset += len(piece)
        return piece


def build_service(**overrides) -> PredictionService:
    settings = make_settings(**overrides)
    scanner = VideoScanner(ScriptedDetector([[]]), settings)
    return PredictionService(scanner, settings)


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("clip.mp4", ".mp4"),
        ("CLIP.MOV", ".MOV"),
        ("no_extension", ""),
        ("archive.tar.gz", ".gz"),
        ("", ""),
        ("upload", ""),
        # Path separators must never survive into the temp file name.
        ("../../../etc/passwd", ""),
        ("dir/sub/clip.avi", ".avi"),
        ("clip.this_is_a_very_long_extension", ""),
        ("clip.", ""),
        # Anything but plain alphanumerics is dropped.
        ("clip.m\x00p4", ""),
        ("clip.mp 4", ""),
        ("clip.mp4;rm -rf /", ""),
    ],
)
def test_temp_suffix_for(filename, expected):
    assert temp_suffix_for(filename) == expected


def test_upload_is_streamed_and_scanned(monkeypatch):
    service = build_service()
    seen = {}

    def fake_scan(path):
        seen["path"] = path
        seen["size"] = os.path.getsize(path)
        return ScanResult(False, 0.0, None, 0, 0, 0)

    service.scanner.scan = fake_scan

    outcome = asyncio.run(service.predict(FakeUpload(b"z" * 5000), "clip.mp4"))

    assert outcome.result.ambulance is False
    assert seen["size"] == 5000
    assert seen["path"].endswith(".mp4")
    assert not os.path.exists(seen["path"]), "temp file must not survive the call"


def test_oversized_uploads_are_refused_before_inference():
    service = build_service(ITCS_MAX_UPLOAD_BYTES="2048")
    service.scanner.scan = lambda path: pytest.fail("must not reach the scanner")

    with pytest.raises(UploadTooLargeError, match="ITCS_MAX_UPLOAD_BYTES"):
        asyncio.run(service.predict(FakeUpload(b"z" * 4096), "clip.mp4"))


def test_oversized_uploads_leave_no_temp_file(tmp_path, monkeypatch):
    service = build_service(ITCS_MAX_UPLOAD_BYTES="2048")
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))

    with pytest.raises(UploadTooLargeError):
        asyncio.run(service.predict(FakeUpload(b"z" * 4096), "clip.mp4"))

    assert list(tmp_path.iterdir()) == []


def test_concurrent_scans_are_bounded():
    """Two slots, three simultaneous uploads: at most two run at once."""
    service = build_service(
        ITCS_MAX_CONCURRENT_INFERENCES="2", ITCS_INFERENCE_QUEUE_SLOTS="8"
    )
    peak = {"now": 0, "max": 0}
    lock = __import__("threading").Lock()
    gate = __import__("threading").Event()

    def slow_scan(path):
        with lock:
            peak["now"] += 1
            peak["max"] = max(peak["max"], peak["now"])
        gate.wait(timeout=2.0)
        with lock:
            peak["now"] -= 1
        return ScanResult(False, 0.0, None, 0, 0, 0)

    service.scanner.scan = slow_scan

    async def drive():
        tasks = [
            asyncio.create_task(service.predict(FakeUpload(b"z" * 16), "c.mp4"))
            for _ in range(3)
        ]
        await asyncio.sleep(0.3)
        observed = peak["max"]
        gate.set()
        await asyncio.gather(*tasks)
        return observed

    assert asyncio.run(drive()) == 2


def test_surplus_requests_are_rejected_rather_than_queued_forever():
    service = build_service(
        ITCS_MAX_CONCURRENT_INFERENCES="1", ITCS_INFERENCE_QUEUE_SLOTS="1"
    )
    gate = __import__("threading").Event()

    def slow_scan(path):
        gate.wait(timeout=2.0)
        return ScanResult(False, 0.0, None, 0, 0, 0)

    service.scanner.scan = slow_scan

    async def drive():
        first = asyncio.create_task(service.predict(FakeUpload(b"z" * 16), "a.mp4"))
        await asyncio.sleep(0.2)
        try:
            with pytest.raises(ServiceBusyError, match="inference slots"):
                await service.predict(FakeUpload(b"z" * 16), "b.mp4")
        finally:
            gate.set()
            await first

    asyncio.run(drive())


def test_slots_free_is_reported():
    service = build_service(ITCS_INFERENCE_QUEUE_SLOTS="4")
    assert service.slots_free == 4


def test_notice_is_none_for_a_trustworthy_detector():
    service = build_service()
    assert service.notice is None


def test_notice_calls_out_a_fake_detector():
    service = build_service()
    service.scanner.detector = ScriptedDetector([[]], performs_real_detection=False)
    assert "DETECTION DISABLED" in service.notice


def test_notice_calls_out_a_model_without_an_ambulance_class():
    service = build_service()
    service.scanner.detector = ScriptedDetector([[]], identifies_ambulances=False)
    assert "no ambulance class" in service.notice
