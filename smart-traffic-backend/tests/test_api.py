"""HTTP contract: /health, /predict and the priority endpoints."""

from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest

from tests.conftest import make_frame


class TestHealth:
    def test_health_is_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["environment"] == "test"

    def test_health_admits_when_detection_is_fake(self, client):
        detector = client.get("/health").json()["detector"]
        assert detector["name"] == "stub"
        assert detector["performs_real_detection"] is False
        assert detector["identifies_ambulances"] is False

    def test_health_reports_free_inference_slots(self, client):
        assert client.get("/health").json()["inference_slots_free"] > 0

    def test_cors_allows_the_configured_origin(self, client, settings):
        origin = settings.allowed_origins[0]
        response = client.get("/health", headers={"Origin": origin})
        assert response.headers.get("access-control-allow-origin") == origin


class TestPredict:
    def test_reports_an_ambulance(
        self, use_capture, use_detector, video_upload, post_video
    ):
        use_capture([make_frame() for _ in range(5)])
        use_detector([[("ambulance", 0.91)]])

        response = post_video(video_upload("ambulance_video.mp4"))

        assert response.status_code == 200
        body = response.json()
        assert body["ambulance"] is True
        assert body["confidence"] == pytest.approx(0.91)
        assert body["label"] == "ambulance"
        assert base64.b64decode(body["frame"])[:2] == b"\xff\xd8"

    def test_reports_no_ambulance(
        self, use_capture, use_detector, video_upload, post_video
    ):
        use_capture([make_frame() for _ in range(5)])
        use_detector([[]])

        body = post_video(video_upload("traffic_video.mp4")).json()

        assert body["ambulance"] is False
        assert body["confidence"] == 0.0
        assert body["frame"]

    def test_filename_alone_never_produces_a_detection(
        self, use_capture, use_detector, video_upload, post_video
    ):
        """Regression: debug mode used to fake a hit from the filename."""
        use_capture([make_frame() for _ in range(30)])
        use_detector([[]])

        body = post_video(video_upload("city_ambulance_run.mp4")).json()

        assert body["ambulance"] is False

    def test_unreadable_video_returns_400(
        self, use_capture, use_detector, video_upload, post_video
    ):
        use_capture([], opened=False)
        use_detector([[]])

        response = post_video(video_upload("broken.mp4"))

        assert response.status_code == 400
        assert "video" in response.json()["detail"].lower()

    def test_video_with_no_frames(
        self, use_capture, use_detector, video_upload, post_video
    ):
        use_capture([])
        use_detector([[]])

        body = post_video(video_upload()).json()

        assert body["ambulance"] is False
        assert body["frame"] is None

    def test_internal_errors_do_not_leak(
        self, use_detector, video_upload, post_video, monkeypatch
    ):
        from app.detection import scanner as scanner_module

        def exploding_videocapture(path):
            raise RuntimeError("decoder blew up")

        monkeypatch.setattr(scanner_module.cv2, "VideoCapture", exploding_videocapture)
        use_detector([[]])

        response = post_video(video_upload())

        assert response.status_code == 500
        assert response.json()["detail"] == "Failed to process video."
        assert "decoder" not in response.text

    def test_temp_file_is_removed(
        self, use_detector, video_upload, post_video, monkeypatch
    ):
        from app.detection import scanner as scanner_module
        from tests.conftest import FakeCapture

        seen = {}
        capture = FakeCapture([make_frame() for _ in range(2)])

        def recording_videocapture(path):
            seen["path"] = path
            assert __import__("os").path.exists(path)
            return capture

        monkeypatch.setattr(scanner_module.cv2, "VideoCapture", recording_videocapture)
        use_detector([[]])

        post_video(video_upload("clip.mp4"))

        import os

        assert capture.released is True
        assert not os.path.exists(seen["path"])

    def test_temp_file_is_removed_on_error(
        self, use_detector, video_upload, post_video, monkeypatch
    ):
        from app.detection import scanner as scanner_module

        seen = {}

        def exploding_videocapture(path):
            seen["path"] = path
            raise RuntimeError("decoder blew up")

        monkeypatch.setattr(scanner_module.cv2, "VideoCapture", exploding_videocapture)
        use_detector([[]])

        post_video(video_upload())

        import os

        assert not os.path.exists(seen["path"])

    def test_requires_a_file(self, client):
        assert client.post("/predict").status_code == 422

    def test_oversized_uploads_are_refused(
        self, use_capture, use_detector, video_upload, post_video, client
    ):
        use_capture([make_frame()])
        use_detector([[]])
        # 64 MiB is the default cap; shrink it rather than uploading that much.
        client.app.state.prediction_service._settings = _with(
            client.app.state.settings, max_upload_bytes=8
        )

        response = post_video(video_upload(content=b"x" * 1024))

        assert response.status_code == 413
        assert "ITCS_MAX_UPLOAD_BYTES" in response.json()["detail"]

    def test_response_carries_the_detector_notice(
        self, use_capture, use_detector, video_upload, post_video
    ):
        """A fabricated result must say so on every single response."""
        use_capture([make_frame()])
        use_detector([[]], performs_real_detection=False)

        body = post_video(video_upload()).json()

        assert body["detector"] == "scripted"
        assert "DETECTION DISABLED" in body["notice"]

    def test_a_model_without_an_ambulance_class_says_so(
        self, use_capture, use_detector, video_upload, post_video
    ):
        use_capture([make_frame()])
        use_detector(
            [[("truck", 0.9)]],
            ambulance_labels=("truck",),
            identifies_ambulances=False,
        )

        body = post_video(video_upload()).json()

        assert body["ambulance"] is True
        assert "no ambulance class" in body["notice"]

    def test_the_returned_frame_is_downscaled_by_default(
        self, use_capture, use_detector, video_upload, post_video
    ):
        """A raw 4K frame is ~1.7 MB of base64; the cap keeps the payload sane."""
        big = np.zeros((2160, 3840, 3), dtype=np.uint8)
        use_capture([big])
        use_detector([[("ambulance", 0.9)]])

        frame = post_video(video_upload()).json()["frame"]

        decoded = np.frombuffer(base64.b64decode(frame), dtype=np.uint8)
        assert max(cv2.imdecode(decoded, cv2.IMREAD_COLOR).shape[:2]) == 1280

    def test_the_frame_can_be_switched_off_entirely(
        self, use_capture, use_detector, video_upload, post_video, client
    ):
        use_capture([make_frame()])
        use_detector([[("ambulance", 0.9)]])
        client.app.state.settings = _with(client.app.state.settings, return_frame=False)

        body = post_video(video_upload()).json()

        assert body["ambulance"] is True
        assert body["frame"] is None

    def test_a_real_ambulance_model_carries_no_notice(
        self, use_capture, use_detector, video_upload, post_video
    ):
        use_capture([make_frame()])
        use_detector([[("ambulance", 0.9)]])

        assert post_video(video_upload()).json()["notice"] is None


class TestPredictFeedsPriority:
    def test_no_approach_means_no_priority_decision(
        self, use_capture, use_detector, video_upload, post_video
    ):
        use_capture([make_frame()])
        use_detector([[("ambulance", 0.99)]])

        assert post_video(video_upload()).json()["priority"] is None

    def test_an_approach_drives_the_controller(
        self, use_capture, use_detector, video_upload, post_video
    ):
        use_capture([make_frame() for _ in range(2)])
        use_detector([[("ambulance", 0.99)]])

        body = post_video(video_upload(), approach="north").json()

        assert body["priority"]["approach"] == "north"
        assert body["priority"]["action"] in {"no_change", "grant"}


class TestPriorityEndpoints:
    def test_state_starts_idle(self, client):
        body = client.get("/priority/state").json()
        assert body["priority_active"] is False
        assert body["active_approach"] is None

    def test_sightings_grant_priority(self, client):
        client.post(
            "/priority/sightings",
            json={"approach": "north", "confidence": 0.95, "at": 0.0},
        )
        response = client.post(
            "/priority/sightings",
            json={"approach": "north", "confidence": 0.95, "at": 1.0},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["action"] == "grant"
        assert body["state"]["active_approach"] == "north"

    def test_confidence_is_validated(self, client):
        response = client.post(
            "/priority/sightings", json={"approach": "north", "confidence": 2.0}
        )
        assert response.status_code == 422

    def test_approach_is_required(self, client):
        assert client.post("/priority/sightings", json={}).status_code == 422

    def test_tick_advances_the_clock(self, client, controller):
        client.post(
            "/priority/sightings",
            json={"approach": "north", "confidence": 0.95, "at": 0.0},
        )
        client.post(
            "/priority/sightings",
            json={"approach": "north", "confidence": 0.95, "at": 1.0},
        )
        assert controller.state().priority_active is True

        controller.tick(now=100.0)

        assert client.get("/priority/state").json()["priority_active"] is False


def _with(settings, **overrides):
    from dataclasses import replace

    return replace(settings, **overrides)
