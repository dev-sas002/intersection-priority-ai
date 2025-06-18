"""Frame encoding and annotation helpers."""

from __future__ import annotations

import base64

import numpy as np

from app import imaging
from app.imaging import annotate_frame, encode_frame, fit_within
from tests.conftest import make_frame


def test_encode_frame_none_returns_none():
    assert encode_frame(None) is None


def test_encode_frame_produces_jpeg():
    encoded = encode_frame(make_frame(128))
    assert isinstance(encoded, str)
    raw = base64.b64decode(encoded)
    assert raw.startswith(b"\xff\xd8") and raw.endswith(b"\xff\xd9")


def test_encode_frame_returns_none_on_encode_failure(monkeypatch):
    monkeypatch.setattr(
        imaging.cv2, "imencode", lambda ext, frame, params=None: (False, None)
    )
    assert encode_frame(make_frame()) is None


def test_lower_quality_produces_a_smaller_payload():
    frame = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    assert len(encode_frame(frame, quality=20)) < len(encode_frame(frame, quality=95))


def test_fit_within_leaves_a_small_frame_alone():
    frame = make_frame()
    assert fit_within(frame, 4096) is frame


def test_fit_within_is_disabled_by_a_non_positive_edge():
    frame = make_frame()
    assert fit_within(frame, 0) is frame
    assert fit_within(frame, -1) is frame


def test_fit_within_caps_the_longest_edge_and_keeps_the_aspect_ratio():
    frame = np.zeros((1080, 3840, 3), dtype=np.uint8)

    resized = fit_within(frame, 1280)

    assert max(resized.shape[:2]) == 1280
    assert resized.shape[0] == round(1080 * 1280 / 3840)


def test_encode_frame_downscales_before_encoding():
    frame = np.random.default_rng(1).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)

    full = encode_frame(frame)
    capped = encode_frame(frame, max_edge=320)

    assert len(capped) < len(full)


def test_annotate_frame_does_not_mutate_input():
    original = make_frame(0)
    untouched = original.copy()

    labeled = annotate_frame(original, "No Ambulance Detected", (0, 0, 255))

    assert np.array_equal(original, untouched)
    assert not np.array_equal(labeled, original)
    assert labeled.shape == original.shape
