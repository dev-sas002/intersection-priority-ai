"""Settings parsing, validation and the fake-detector safety fence."""

from __future__ import annotations

import pytest

from app.config import DEFAULT_WEIGHTS_PATH, Settings
from app.errors import ConfigurationError


def test_defaults_are_production_safe():
    settings = Settings.from_env({})
    assert settings.environment == "production"
    assert settings.detector == "auto"
    assert settings.stub_allowed is False
    assert settings.confidence_threshold == 0.75


def test_default_weights_path_is_absolute():
    assert DEFAULT_WEIGHTS_PATH.is_absolute()
    assert DEFAULT_WEIGHTS_PATH.name == "best.pt"


def test_values_are_read_from_the_environment():
    settings = Settings.from_env(
        {
            "ITCS_ENV": "development",
            "ITCS_DETECTOR": "coco",
            "ITCS_CONFIDENCE_THRESHOLD": "0.5",
            "ITCS_MAX_FRAMES": "7",
            "ITCS_FRAME_STRIDE": "3",
            "ITCS_BATCH_SIZE": "4",
            "ITCS_ALLOWED_ORIGINS": "https://a.test, https://b.test",
            "ITCS_COCO_EMERGENCY_CLASSES": "bus",
        }
    )
    assert settings.detector == "coco"
    assert settings.confidence_threshold == 0.5
    assert settings.max_frames == 7
    assert settings.frame_stride == 3
    assert settings.batch_size == 4
    assert settings.allowed_origins == ["https://a.test", "https://b.test"]
    assert settings.coco_emergency_classes == ["bus"]


@pytest.mark.parametrize(
    "env",
    [
        {"ITCS_MAX_FRAMES": "nope"},
        {"ITCS_MAX_FRAMES": "0"},
        {"ITCS_CONFIDENCE_THRESHOLD": "1.5"},
        {"ITCS_CONFIDENCE_THRESHOLD": "-0.1"},
        {"ITCS_IMAGE_SIZE": "8"},
    ],
)
def test_invalid_values_are_rejected(env):
    with pytest.raises(ConfigurationError):
        Settings.from_env(env)


def test_queue_slots_must_cover_concurrency():
    with pytest.raises(ConfigurationError, match="INFERENCE_QUEUE_SLOTS"):
        Settings.from_env(
            {"ITCS_MAX_CONCURRENT_INFERENCES": "4", "ITCS_INFERENCE_QUEUE_SLOTS": "2"}
        )


def test_min_hold_may_not_exceed_max_hold():
    with pytest.raises(ConfigurationError, match="MIN_HOLD"):
        Settings.from_env(
            {"ITCS_PRIORITY_MIN_HOLD_S": "60", "ITCS_PRIORITY_MAX_HOLD_S": "30"}
        )


class TestStubIsFencedOff:
    """The fake detector must be impossible to enable by accident."""

    def test_refused_in_the_default_environment(self):
        with pytest.raises(ConfigurationError, match="NO detection"):
            Settings.from_env({"ITCS_DETECTOR": "stub"})

    def test_refused_in_an_explicit_production_environment(self):
        with pytest.raises(ConfigurationError, match="NO detection"):
            Settings.from_env({"ITCS_DETECTOR": "stub", "ITCS_ENV": "production"})

    def test_refused_in_staging(self):
        with pytest.raises(ConfigurationError):
            Settings.from_env({"ITCS_DETECTOR": "stub", "ITCS_ENV": "staging"})

    @pytest.mark.parametrize("environment", ["development", "test"])
    def test_allowed_only_with_a_second_deliberate_opt_in(self, environment):
        settings = Settings.from_env({"ITCS_DETECTOR": "stub", "ITCS_ENV": environment})
        assert settings.detector == "stub"
        assert settings.stub_allowed is True


def test_device_is_normalised():
    assert Settings.from_env({"ITCS_DEVICE": "CUDA"}).device == "cuda"
    assert Settings.from_env({}).device == "auto"


def test_blank_values_fall_back_to_defaults():
    settings = Settings.from_env({"ITCS_MAX_FRAMES": "   ", "ITCS_ENV": ""})
    assert settings.max_frames == 100
    assert settings.environment == "production"


def test_response_payload_defaults_are_bounded():
    settings = Settings.from_env({})
    assert settings.return_frame is True
    assert settings.max_frame_edge == 1280
    assert settings.jpeg_quality == 85


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("TRUE", True),
        ("1", True),
        ("on", True),
        ("yes", True),
        ("false", False),
        ("0", False),
        ("off", False),
        ("no", False),
    ],
)
def test_return_frame_accepts_the_usual_spellings(raw, expected):
    assert Settings.from_env({"ITCS_RETURN_FRAME": raw}).return_frame is expected


def test_a_nonsense_boolean_is_refused():
    with pytest.raises(ConfigurationError, match="ITCS_RETURN_FRAME must be a boolean"):
        Settings.from_env({"ITCS_RETURN_FRAME": "maybe"})


def test_frame_resizing_can_be_switched_off_with_zero():
    assert Settings.from_env({"ITCS_MAX_FRAME_EDGE": "0"}).max_frame_edge == 0


def test_jpeg_quality_is_bounded_at_both_ends():
    with pytest.raises(ConfigurationError, match="ITCS_JPEG_QUALITY must be >= 1"):
        Settings.from_env({"ITCS_JPEG_QUALITY": "0"})
    with pytest.raises(ConfigurationError, match="ITCS_JPEG_QUALITY must be 1-100"):
        Settings.from_env({"ITCS_JPEG_QUALITY": "101"})
