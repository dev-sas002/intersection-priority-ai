"""Detector selection: the seam, and what ``auto`` picks."""

from __future__ import annotations

import pytest

from app.detection import available_detectors, build_detector, register
from app.detection.registry import resolve_detector_name
from app.detection.stub import StubDetector
from app.errors import ConfigurationError
from tests.conftest import make_settings


def test_builtin_detectors_are_registered():
    assert set(available_detectors()) == {"coco", "stub", "yolo"}


def test_explicit_name_is_honoured():
    assert resolve_detector_name(make_settings()) == "stub"


def test_unknown_detector_is_rejected():
    settings = make_settings(ITCS_DETECTOR="magic")
    with pytest.raises(ConfigurationError, match="Unknown ITCS_DETECTOR"):
        resolve_detector_name(settings)


def test_auto_prefers_custom_weights_when_present(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"pretend-checkpoint")
    settings = make_settings(ITCS_DETECTOR="auto", ITCS_WEIGHTS_PATH=str(weights))
    assert resolve_detector_name(settings) == "yolo"


def test_auto_falls_back_to_the_public_model_when_weights_are_missing(tmp_path):
    settings = make_settings(
        ITCS_DETECTOR="auto", ITCS_WEIGHTS_PATH=str(tmp_path / "absent.pt")
    )
    assert resolve_detector_name(settings) == "coco"


def test_auto_never_selects_the_fake_detector(tmp_path):
    """``auto`` must never silently produce fabricated detections."""
    settings = make_settings(
        ITCS_DETECTOR="auto", ITCS_WEIGHTS_PATH=str(tmp_path / "absent.pt")
    )
    assert resolve_detector_name(settings) != "stub"


def test_build_detector_returns_the_stub_when_asked():
    detector = build_detector(make_settings())
    assert isinstance(detector, StubDetector)
    assert detector.performs_real_detection is False
    assert detector.identifies_ambulances is False


def test_build_detector_logs_critical_for_a_fake_detector(caplog):
    with caplog.at_level("CRITICAL"):
        build_detector(make_settings())
    assert "PERFORMS NO DETECTION" in caplog.text


def test_a_new_backend_needs_only_a_registry_entry():
    """The extensibility seam: one decorator is the whole integration."""

    class Fake:
        name = "unit-test-backend"
        performs_real_detection = True
        identifies_ambulances = True
        description = "registered by a test"

        def warmup(self) -> None:
            return None

        def detect(self, frames):
            return []

    register("unit-test-backend")(lambda settings: Fake())
    try:
        assert "unit-test-backend" in available_detectors()
        settings = make_settings(ITCS_DETECTOR="unit-test-backend")
        assert build_detector(settings).name == "unit-test-backend"
    finally:
        from app.detection import registry

        registry._REGISTRY.pop("unit-test-backend", None)


def test_duplicate_registration_is_refused():
    with pytest.raises(ConfigurationError, match="already registered"):
        register("stub")(lambda settings: StubDetector())
