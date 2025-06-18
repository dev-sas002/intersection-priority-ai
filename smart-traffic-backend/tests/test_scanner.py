"""The detection pipeline, exercised without FastAPI.

This is the point of splitting the scanner out of the route handler: every
detection rule below is asserted against plain objects.
"""

from __future__ import annotations

import pytest

from app.detection.scanner import VideoScanner
from app.errors import UnreadableVideoError
from tests.conftest import FakeCapture, ScriptedDetector, make_frame, make_settings


def build(frames, script, monkeypatch, **overrides):
    from app.detection import scanner as scanner_module

    capture = FakeCapture(frames, opened=overrides.pop("opened", True))
    monkeypatch.setattr(scanner_module.cv2, "VideoCapture", lambda path: capture)
    detector = ScriptedDetector(script, **overrides.pop("detector_kwargs", {}))
    settings = make_settings(**overrides)
    return VideoScanner(detector, settings), capture, detector


def test_reports_an_ambulance(monkeypatch):
    scanner, _, _ = build([make_frame()] * 5, [[("ambulance", 0.91)]], monkeypatch)
    result = scanner.scan("clip.mp4")
    assert result.ambulance is True
    assert result.confidence == pytest.approx(0.91)
    assert result.label == "ambulance"
    assert result.frame_to_return is not None


def test_reports_no_ambulance(monkeypatch):
    scanner, _, _ = build([make_frame()] * 5, [[]], monkeypatch)
    result = scanner.scan("clip.mp4")
    assert result.ambulance is False
    assert result.confidence == 0.0
    assert result.label is None
    # The negative path still returns the annotated first frame.
    assert result.frame_to_return is not None


def test_detection_below_threshold_is_ignored(monkeypatch):
    scanner, _, _ = build([make_frame()] * 3, [[("ambulance", 0.74)]], monkeypatch)
    result = scanner.scan("clip.mp4")
    assert result.ambulance is False
    assert result.confidence == 0.0


def test_detection_exactly_at_threshold_counts(monkeypatch):
    scanner, _, _ = build([make_frame()] * 3, [[("ambulance", 0.75)]], monkeypatch)
    assert scanner.scan("clip.mp4").ambulance is True


def test_non_ambulance_class_is_ignored(monkeypatch):
    scanner, _, _ = build([make_frame()] * 3, [[("car", 0.99)]], monkeypatch)
    assert scanner.scan("clip.mp4").ambulance is False


def test_highest_confidence_frame_is_reported(monkeypatch):
    scanner, _, _ = build(
        [make_frame(i) for i in range(4)],
        [[("ambulance", 0.80)], [("ambulance", 0.95)], [("ambulance", 0.85)]],
        monkeypatch,
        ITCS_MIN_FRAMES_WITH_AMBULANCE="3",
    )
    result = scanner.scan("clip.mp4")
    assert result.ambulance is True
    assert result.confidence == pytest.approx(0.95)


def test_min_frames_not_reached(monkeypatch):
    scanner, _, _ = build(
        [make_frame()] * 2,
        [[("ambulance", 0.90)]],
        monkeypatch,
        ITCS_MIN_FRAMES_WITH_AMBULANCE="3",
    )
    result = scanner.scan("clip.mp4")
    assert result.ambulance is False
    assert result.confidence == 0.0


def test_multiple_boxes_in_one_frame_count_as_one_frame(monkeypatch):
    """Regression: evidence used to be counted per box rather than per frame."""
    scanner, _, _ = build(
        [make_frame()],
        [[("ambulance", 0.90), ("ambulance", 0.91), ("ambulance", 0.92)]],
        monkeypatch,
        ITCS_MIN_FRAMES_WITH_AMBULANCE="3",
    )
    assert scanner.scan("clip.mp4").ambulance is False


def test_unreadable_video_raises(monkeypatch):
    scanner, capture, _ = build([], [[]], monkeypatch, opened=False)
    with pytest.raises(UnreadableVideoError):
        scanner.scan("broken.mp4")
    assert capture.released is True


def test_video_with_no_frames(monkeypatch):
    scanner, _, _ = build([], [[]], monkeypatch)
    result = scanner.scan("clip.mp4")
    assert result.ambulance is False
    assert result.frame_to_return is None


def test_frame_budget_is_respected(monkeypatch):
    scanner, capture, detector = build(
        [make_frame()] * 50, [[]], monkeypatch, ITCS_MAX_FRAMES="4"
    )
    scanner.scan("clip.mp4")
    assert detector.calls == 4
    assert capture.reads == 4


def test_capture_is_always_released(monkeypatch):
    scanner, capture, _ = build([make_frame()] * 2, [[]], monkeypatch)
    scanner.scan("clip.mp4")
    assert capture.released is True


def test_capture_released_when_the_detector_explodes(monkeypatch):
    scanner, capture, detector = build([make_frame()] * 2, [[]], monkeypatch)

    def boom(frames):
        raise RuntimeError("model blew up")

    detector.detect = boom
    with pytest.raises(RuntimeError):
        scanner.scan("clip.mp4")
    assert capture.released is True


def test_frames_are_inferred_in_batches(monkeypatch):
    scanner, _, detector = build(
        [make_frame()] * 10, [[]], monkeypatch, ITCS_BATCH_SIZE="4", ITCS_MAX_FRAMES="10"
    )
    scanner.scan("clip.mp4")
    assert detector.batches == [4, 4, 2]


def test_stride_samples_the_clip_instead_of_reading_it_all(monkeypatch):
    """Stride 3 over a 9-frame clip: 3 forward passes, 9 reads."""
    scanner, capture, detector = build(
        [make_frame(i) for i in range(9)],
        [[]],
        monkeypatch,
        ITCS_FRAME_STRIDE="3",
        ITCS_MAX_FRAMES="3",
    )
    result = scanner.scan("clip.mp4")
    assert detector.calls == 3
    assert capture.reads == 7
    assert result.frames_inferred == 3


def test_falls_back_to_the_raw_frame_when_the_detector_cannot_render(monkeypatch):
    scanner, _, _ = build(
        [make_frame()],
        [[("ambulance", 0.99)]],
        monkeypatch,
        detector_kwargs={"render": False},
    )
    assert scanner.scan("clip.mp4").frame_to_return is not None


def test_render_failure_does_not_break_the_scan(monkeypatch):
    scanner, _, detector = build([make_frame()], [[("ambulance", 0.99)]], monkeypatch)

    def exploding_detect(frames):
        from app.detection.types import Detection, FrameDetections

        def boom():
            raise RuntimeError("plot failed")

        return [
            FrameDetections(detections=[Detection("ambulance", 0.99, True)], render=boom)
            for _ in frames
        ]

    detector.detect = exploding_detect
    result = scanner.scan("clip.mp4")
    assert result.ambulance is True
    assert result.frame_to_return is not None


def test_coco_candidate_labels_count_as_evidence(monkeypatch):
    """A detector whose ambulance labels are COCO classes still drives the flow."""
    scanner, _, _ = build(
        [make_frame()],
        [[("truck", 0.88)]],
        monkeypatch,
        detector_kwargs={
            "ambulance_labels": ("truck", "bus"),
            "identifies_ambulances": False,
        },
    )
    result = scanner.scan("clip.mp4")
    assert result.ambulance is True
    assert result.label == "truck"
