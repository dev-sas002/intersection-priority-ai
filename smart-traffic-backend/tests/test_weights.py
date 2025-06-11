"""Checkpoint resolution: the missing-weights story, end to end."""

from __future__ import annotations

import io
import urllib.error

import pytest

from app.detection import weights as weights_module
from app.detection.weights import ensure_custom_weights, ensure_public_checkpoint
from app.errors import DetectorUnavailableError


def test_present_weights_are_returned(tmp_path):
    path = tmp_path / "best.pt"
    path.write_bytes(b"checkpoint")
    assert ensure_custom_weights(path) == path


def test_missing_weights_raise_an_actionable_message(tmp_path):
    path = tmp_path / "best.pt"
    with pytest.raises(DetectorUnavailableError) as excinfo:
        ensure_custom_weights(path)

    message = str(excinfo.value)
    # The operator must be told where it looked and what their options are.
    assert str(path) in message
    assert "ITCS_WEIGHTS_PATH" in message
    assert "ITCS_DETECTOR=coco" in message


class _FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def test_public_checkpoint_is_downloaded_once(tmp_path, monkeypatch):
    calls = []

    def fake_urlopen(url, timeout=None):
        calls.append(url)
        return _FakeResponse(b"fake-weights")

    monkeypatch.setattr(weights_module.urllib.request, "urlopen", fake_urlopen)

    first = ensure_public_checkpoint("yolov8n.pt", tmp_path / "cache")
    second = ensure_public_checkpoint("yolov8n.pt", tmp_path / "cache")

    assert first == second
    assert first.read_bytes() == b"fake-weights"
    assert len(calls) == 1, "a cached checkpoint must not be re-downloaded"


def test_download_failure_explains_the_offline_options(tmp_path, monkeypatch):
    def fake_urlopen(url, timeout=None):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(weights_module.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(DetectorUnavailableError) as excinfo:
        ensure_public_checkpoint("yolov8n.pt", tmp_path / "cache")

    assert "ITCS_MODEL_CACHE_DIR" in str(excinfo.value)
    assert not (tmp_path / "cache" / "yolov8n.pt.part").exists()


def test_partial_downloads_are_not_left_behind(tmp_path, monkeypatch):
    def fake_urlopen(url, timeout=None):
        raise OSError("connection reset mid-stream")

    monkeypatch.setattr(weights_module.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(DetectorUnavailableError):
        ensure_public_checkpoint("yolov8n.pt", tmp_path / "cache")

    assert list((tmp_path / "cache").iterdir()) == []
