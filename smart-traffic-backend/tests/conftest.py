"""Shared fixtures.

Nothing heavy is ever loaded: the suite needs no camera, no video file, no YOLO
weights and no ultralytics/torch install. ``ITCS_ENV=test`` plus
``ITCS_DETECTOR=stub`` is the only combination that is allowed to skip real
detection, and it is spelled out explicitly here rather than inherited from the
developer's shell.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.detection.types import Detection, FrameDetections  # noqa: E402
from app.main import create_app  # noqa: E402

TEST_ENV = {"ITCS_ENV": "test", "ITCS_DETECTOR": "stub"}


def make_frame(value: int = 0) -> Any:
    """A small, cheap BGR frame, big enough for the overlay text to land on."""
    return np.full((120, 320, 3), value, dtype=np.uint8)


def make_settings(**overrides: str) -> Settings:
    return Settings.from_env({**TEST_ENV, **overrides})


class FakeCapture:
    """Stand-in for ``cv2.VideoCapture`` driven by a list of frames."""

    def __init__(self, frames: Iterable[Any], opened: bool = True) -> None:
        self._frames = list(frames)
        self._index = 0
        self._opened = opened
        self.released = False
        self.reads = 0

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, Any]:
        self.reads += 1
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def release(self) -> None:
        self.released = True


class ScriptedDetector:
    """Detector whose per-frame output is scripted by the test.

    ``script`` is a list of per-frame box specs, each a list of
    ``(label, confidence)`` tuples. The last entry repeats once exhausted.
    """

    def __init__(
        self,
        script: Sequence[Sequence[tuple[str, float]]],
        *,
        ambulance_labels: Sequence[str] = ("ambulance",),
        identifies_ambulances: bool = True,
        performs_real_detection: bool = True,
        render: bool = True,
    ) -> None:
        self.name = "scripted"
        self.performs_real_detection = performs_real_detection
        self.identifies_ambulances = identifies_ambulances
        self.description = "scripted test detector"
        self.script = [list(spec) for spec in script]
        self._ambulance_labels = {label.lower() for label in ambulance_labels}
        self._render = render
        self.calls = 0
        self.batches: list[int] = []
        self.warmed = False

    def warmup(self) -> None:
        self.warmed = True

    def detect(self, frames: Sequence[Any]) -> list[FrameDetections]:
        self.batches.append(len(frames))
        outputs: list[FrameDetections] = []
        for frame in frames:
            spec = self.script[min(self.calls, len(self.script) - 1)]
            self.calls += 1
            detections = [
                Detection(
                    label=label,
                    confidence=confidence,
                    is_ambulance_class=label.lower() in self._ambulance_labels,
                )
                for label, confidence in spec
            ]
            outputs.append(
                FrameDetections(
                    detections=detections,
                    render=(lambda f=frame: f) if self._render else None,
                )
            )
        return outputs


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def client(settings: Settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def scanner(client):
    return client.app.state.scanner


@pytest.fixture
def controller(client):
    return client.app.state.priority_controller


@pytest.fixture
def use_detector(scanner):
    """Install a :class:`ScriptedDetector` on the live scanner."""

    def _install(script, **kwargs) -> ScriptedDetector:
        detector = ScriptedDetector(script, **kwargs)
        scanner.detector = detector
        return detector

    return _install


@pytest.fixture
def use_capture(monkeypatch):
    """Install a :class:`FakeCapture` in place of ``cv2.VideoCapture``."""
    from app.detection import scanner as scanner_module

    def _install(frames, opened: bool = True) -> FakeCapture:
        capture = FakeCapture(frames, opened=opened)
        monkeypatch.setattr(scanner_module.cv2, "VideoCapture", lambda path: capture)
        return capture

    return _install


@pytest.fixture
def video_upload(tmp_path):
    """Build a file on disk to POST to ``/predict``."""

    def _build(name: str = "traffic_video.mp4", content: bytes = b"not-a-real-video"):
        path = tmp_path / name
        path.write_bytes(content)
        return path

    return _build


@pytest.fixture
def post_video(client):
    def _post(path, **data):
        with path.open("rb") as handle:
            return client.post(
                "/predict",
                files={"file": (path.name, handle, "video/mp4")},
                data=data or None,
            )

    return _post


@pytest.fixture(autouse=True)
def _no_ambient_itcs_env(monkeypatch):
    """Stop a developer's exported ITCS_* variables from steering the suite."""
    for key in list(os.environ):
        if key.startswith("ITCS_"):
            monkeypatch.delenv(key, raising=False)
