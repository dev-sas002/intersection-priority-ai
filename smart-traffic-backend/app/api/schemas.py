"""Pydantic request/response models. The HTTP contract lives here only."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.priority.models import PriorityAction

__all__ = [
    "DetectorInfo",
    "HealthResponse",
    "PredictionResponse",
    "PriorityDecisionResponse",
    "PriorityStateResponse",
    "SightingRequest",
]


class DetectorInfo(BaseModel):
    name: str = Field(description="Registry key of the active detector.")
    performs_real_detection: bool = Field(
        description="False for the fake 'stub' detector, which inspects no pixels."
    )
    identifies_ambulances: bool = Field(
        description=(
            "False when the label space cannot distinguish an ambulance from "
            "another large vehicle (the public COCO fallback)."
        )
    )
    description: str


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    environment: str
    detector: DetectorInfo
    inference_slots_free: int


class PriorityStateResponse(BaseModel):
    priority_active: bool
    active_approach: str | None = None
    held_for_s: float | None = None
    cooldowns: dict[str, float] = Field(default_factory=dict)


class PriorityDecisionResponse(BaseModel):
    action: PriorityAction
    approach: str | None = None
    reason: str
    held_for_s: float | None = None
    state: PriorityStateResponse


class PredictionResponse(BaseModel):
    ambulance: bool = Field(
        description="Whether the frame budget contained enough ambulance frames."
    )
    frame: str | None = Field(
        default=None, description="Base64-encoded JPEG of the representative frame."
    )
    confidence: float = 0.0
    label: str | None = Field(
        default=None, description="Label of the highest-confidence detection."
    )
    frames_read: int = 0
    frames_inferred: int = 0
    detector: str
    #: Present whenever the result must not be read as a trustworthy ambulance
    #: detection -- a fake detector, or a model without an ambulance class.
    notice: str | None = None
    priority: PriorityDecisionResponse | None = None


class SightingRequest(BaseModel):
    approach: str = Field(min_length=1, max_length=64, description="Approach identifier.")
    detected: bool = True
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    at: float | None = Field(
        default=None,
        description=(
            "Seconds on the server's monotonic clock. Leave it unset in normal "
            "use; supplying your own values is for replaying a scenario, and "
            "must then be done consistently for every sighting."
        ),
    )
