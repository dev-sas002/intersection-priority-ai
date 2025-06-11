"""The YOLO adapter's pure parts, and its refusal to import torch eagerly."""

from __future__ import annotations

import sys

import pytest

from app.detection import yolo
from app.detection.yolo import CUSTOM_CLASS_MAPPING, build_coco_yolo, resolve_label
from app.errors import DetectorUnavailableError
from tests.conftest import make_settings


def test_ultralytics_is_not_imported_at_module_load():
    """The heavy stack must stay lazy: the suite runs without torch."""
    assert "ultralytics" not in sys.modules
    assert "torch" not in sys.modules


def test_class_mapping_wins_over_model_names():
    assert resolve_label(0, CUSTOM_CLASS_MAPPING, {0: "not-used"}) == "ambulance"


def test_falls_back_to_dict_names():
    assert resolve_label(3, {}, {3: "fire_truck"}) == "fire_truck"


def test_falls_back_to_list_names():
    assert resolve_label(2, {}, ["car", "bus", "van"]) == "van"


@pytest.mark.parametrize("names", [{}, ["car"], None, 42])
def test_unknown_id_degrades_to_its_number(names):
    assert resolve_label(99, {}, names) == "99"


def test_coco_detector_admits_it_cannot_identify_ambulances(tmp_path, monkeypatch):
    monkeypatch.setattr(
        yolo, "ensure_public_checkpoint", lambda name, cache: tmp_path / name
    )
    detector = build_coco_yolo(make_settings(ITCS_COCO_EMERGENCY_CLASSES="truck,bus"))

    assert detector.performs_real_detection is True
    assert detector.identifies_ambulances is False
    assert "no ambulance class" in detector.description


def test_warmup_without_ultralytics_explains_itself(tmp_path, monkeypatch):
    monkeypatch.setattr(
        yolo, "ensure_public_checkpoint", lambda name, cache: tmp_path / name
    )
    detector = build_coco_yolo(make_settings())

    import builtins

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "ultralytics":
            raise ImportError("No module named 'ultralytics'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(DetectorUnavailableError, match=r"requirements-ml\.txt"):
        detector.warmup()


def test_detect_on_an_empty_batch_does_no_work(tmp_path, monkeypatch):
    monkeypatch.setattr(
        yolo, "ensure_public_checkpoint", lambda name, cache: tmp_path / name
    )
    detector = build_coco_yolo(make_settings())
    assert detector.detect([]) == []
