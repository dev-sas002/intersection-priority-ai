"""HTTP routes. Thin by design: parse, delegate, map errors, serialise."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app import __version__
from app.api.dependencies import (
    get_controller,
    get_prediction_service,
    get_settings,
)
from app.api.schemas import (
    DetectorInfo,
    HealthResponse,
    PredictionResponse,
    PriorityDecisionResponse,
    PriorityStateResponse,
    SightingRequest,
)
from app.config import Settings
from app.errors import UnreadableVideoError
from app.imaging import encode_frame
from app.priority import PriorityDecision, Sighting, SignalPriorityController
from app.services.prediction import (
    PredictionService,
    ServiceBusyError,
    UploadTooLargeError,
)

__all__ = ["router"]

logger = logging.getLogger(__name__)

router = APIRouter()


def _encoded_frame(frame: object | None, settings: Settings) -> str | None:
    """Base64 JPEG for the response, bounded by the payload settings."""
    if not settings.return_frame:
        return None
    return encode_frame(
        frame,
        max_edge=settings.max_frame_edge,
        quality=settings.jpeg_quality,
    )


def _decision_response(decision: PriorityDecision) -> PriorityDecisionResponse:
    return PriorityDecisionResponse(
        action=decision.action,
        approach=decision.approach,
        reason=decision.reason,
        held_for_s=decision.held_for_s,
        state=PriorityStateResponse(
            priority_active=decision.state.priority_active,
            active_approach=decision.state.active_approach,
            held_for_s=decision.held_for_s,
            cooldowns=dict(decision.state.cooldown_until),
        ),
    )


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health_check(
    settings: Settings = Depends(get_settings),
    service: PredictionService = Depends(get_prediction_service),
) -> HealthResponse:
    """Liveness plus an honest statement of what the loaded model can do."""
    detector = service.scanner.detector
    return HealthResponse(
        status="ok",
        version=__version__,
        environment=settings.environment,
        detector=DetectorInfo(
            name=detector.name,
            performs_real_detection=detector.performs_real_detection,
            identifies_ambulances=detector.identifies_ambulances,
            description=detector.description,
        ),
        inference_slots_free=service.slots_free,
    )


@router.post("/predict", response_model=PredictionResponse, tags=["detection"])
async def predict_video(
    file: UploadFile = File(..., description="Traffic video clip."),
    approach: str | None = Form(
        default=None,
        description=(
            "Optional approach identifier. When given, the result is fed to the "
            "signal-priority controller and its decision is returned."
        ),
    ),
    service: PredictionService = Depends(get_prediction_service),
    controller: SignalPriorityController = Depends(get_controller),
    settings: Settings = Depends(get_settings),
) -> PredictionResponse:
    filename = file.filename or "upload"
    try:
        outcome = await service.predict(file, filename)
    except UploadTooLargeError as exc:
        # 413 Content Too Large; spelled numerically because the Starlette
        # constant was renamed between versions.
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except ServiceBusyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except UnreadableVideoError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("Error while processing video file %s", filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process video.",
        ) from exc

    result = outcome.result
    priority = None
    if approach:
        decision = controller.observe(
            Sighting(
                approach=approach,
                at=time.monotonic(),
                detected=result.ambulance,
                confidence=result.confidence,
            )
        )
        priority = _decision_response(decision)

    return PredictionResponse(
        ambulance=result.ambulance,
        frame=_encoded_frame(result.frame_to_return, settings),
        confidence=result.confidence,
        label=result.label,
        frames_read=result.frames_read,
        frames_inferred=result.frames_inferred,
        detector=service.scanner.detector.name,
        notice=outcome.notice,
        priority=priority,
    )


@router.post(
    "/priority/sightings",
    response_model=PriorityDecisionResponse,
    tags=["priority"],
)
async def record_sighting(
    sighting: SightingRequest,
    controller: SignalPriorityController = Depends(get_controller),
) -> PriorityDecisionResponse:
    """Feed one detection report into the signal-priority state machine.

    Lets a caller that runs its own detector (roadside unit, another service)
    drive the decision layer without going through ``/predict``.
    """
    decision = controller.observe(
        Sighting(
            approach=sighting.approach,
            at=time.monotonic() if sighting.at is None else sighting.at,
            detected=sighting.detected,
            confidence=sighting.confidence,
        )
    )
    return _decision_response(decision)


@router.get("/priority/state", response_model=PriorityStateResponse, tags=["priority"])
async def priority_state(
    controller: SignalPriorityController = Depends(get_controller),
) -> PriorityStateResponse:
    """Current phase, after applying any pending timeout."""
    return _decision_response(controller.tick()).state


@router.post("/priority/tick", response_model=PriorityDecisionResponse, tags=["priority"])
async def priority_tick(
    controller: SignalPriorityController = Depends(get_controller),
) -> PriorityDecisionResponse:
    """Advance the controller's clock without new evidence.

    A caller that has stopped sending sightings still needs the phase to end;
    this is how the timeout rules are driven in a request/response world.
    """
    return _decision_response(controller.tick())
