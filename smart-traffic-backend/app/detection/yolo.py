"""Ultralytics YOLO detectors.

Two registry entries share one implementation:

``yolo``
    The purpose-trained single-class ambulance checkpoint. Class ``0`` is an
    ambulance; ``identifies_ambulances`` is True.

``coco``
    The public ``yolov8n.pt`` checkpoint, downloaded on first use. COCO has no
    ambulance class, so this detector reports the configured *candidate*
    emergency classes (``truck``/``bus`` by default) and sets
    ``identifies_ambulances`` to False. It exists so a fresh clone runs real
    inference immediately -- not so the project can claim ambulance accuracy.

``ultralytics`` (and therefore torch) is imported lazily inside
:meth:`_YoloDetector.warmup`, so the test suite and the stub path never need
the heavy stack installed.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from app.config import Settings
from app.detection.base import Detector
from app.detection.registry import register
from app.detection.types import Detection, FrameDetections
from app.detection.weights import ensure_custom_weights, ensure_public_checkpoint
from app.errors import DetectorUnavailableError

__all__ = ["CUSTOM_CLASS_MAPPING", "build_coco_yolo", "build_custom_yolo"]

logger = logging.getLogger(__name__)

#: The purpose-trained checkpoint has one class and does not carry a useful
#: ``names`` table, so the mapping is asserted here.
CUSTOM_CLASS_MAPPING: Mapping[int, str] = {0: "ambulance"}


def resolve_label(
    cls_id: int,
    mapping: Mapping[int, str],
    model_names: Any,
) -> str:
    """Resolve a class id to a label.

    ``mapping`` wins; otherwise fall back to the model's own ``names``, which
    may be a dict or a list and may not contain the id at all.
    """
    if cls_id in mapping:
        return mapping[cls_id]
    if isinstance(model_names, Mapping) and cls_id in model_names:
        return str(model_names[cls_id])
    if isinstance(model_names, (list, tuple)) and 0 <= cls_id < len(model_names):
        return str(model_names[cls_id])
    return str(cls_id)


class _YoloDetector:
    """Shared Ultralytics implementation behind both registry entries."""

    def __init__(
        self,
        *,
        name: str,
        weights: Path,
        settings: Settings,
        class_mapping: Mapping[int, str],
        ambulance_labels: Sequence[str],
        identifies_ambulances: bool,
        description: str,
    ) -> None:
        self.name = name
        self.performs_real_detection = True
        self.identifies_ambulances = identifies_ambulances
        self.description = description
        self._weights = weights
        self._settings = settings
        self._class_mapping = dict(class_mapping)
        self._ambulance_labels = {label.lower() for label in ambulance_labels}
        self._model: Any | None = None

    # -- lifecycle ---------------------------------------------------------
    def warmup(self) -> None:
        if self._model is not None:
            return
        try:
            # Imported here, not at module import, so pytest (and the stub
            # detector) never pull in torch.
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on install
            raise DetectorUnavailableError(
                "The 'ultralytics' package is not installed, so no YOLO detector "
                "can run. Install it with 'pip install -r requirements-ml.txt', "
                "or use ITCS_DETECTOR=stub in a development environment."
            ) from exc

        logger.info("Loading YOLO weights from %s ...", self._weights)
        self._model = YOLO(str(self._weights))
        if self._settings.device != "auto":
            self._model.to(self._settings.device)
        logger.info(
            "Detector %r ready (device=%s, classes=%s).",
            self.name,
            self._settings.device,
            getattr(self._model, "names", None),
        )

    # -- inference ---------------------------------------------------------
    def detect(self, frames: Sequence[Any]) -> list[FrameDetections]:
        if not frames:
            return []
        if self._model is None:
            self.warmup()
        model = self._model

        # A list source makes Ultralytics run one batched forward pass, which
        # is where the GPU win is; on CPU it still saves per-call overhead.
        results = model.predict(
            source=list(frames),
            conf=self._settings.confidence_threshold,
            imgsz=self._settings.image_size,
            verbose=False,
            save=False,
        )
        return [self._to_frame_detections(result) for result in results]

    def _to_frame_detections(self, result: Any) -> FrameDetections:
        model_names = getattr(self._model, "names", None)
        detections: list[Detection] = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = resolve_label(cls_id, self._class_mapping, model_names)
            detections.append(
                Detection(
                    label=label,
                    confidence=float(box.conf[0]),
                    is_ambulance_class=label.lower() in self._ambulance_labels,
                )
            )
        return FrameDetections(detections=detections, render=result.plot)


@register("yolo")
def build_custom_yolo(settings: Settings) -> Detector:
    """The purpose-trained single-class ambulance checkpoint."""
    weights = ensure_custom_weights(settings.weights_path)
    return _YoloDetector(
        name="yolo",
        weights=weights,
        settings=settings,
        class_mapping=CUSTOM_CLASS_MAPPING,
        ambulance_labels=["ambulance"],
        identifies_ambulances=True,
        description=f"Ultralytics YOLO, purpose-trained ambulance weights ({weights}).",
    )


@register("coco")
def build_coco_yolo(settings: Settings) -> Detector:
    """Public COCO checkpoint used as a runnable stand-in."""
    weights = ensure_public_checkpoint(settings.public_model, settings.model_cache_dir)
    candidates = settings.coco_emergency_classes
    return _YoloDetector(
        name="coco",
        weights=weights,
        settings=settings,
        class_mapping={},
        ambulance_labels=candidates,
        identifies_ambulances=False,
        description=(
            f"Public Ultralytics {settings.public_model} (COCO). COCO has no "
            f"ambulance class; {', '.join(candidates)} are treated as candidate "
            "emergency vehicles."
        ),
    )
