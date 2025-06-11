"""Value objects for the signal-priority decision layer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "PriorityAction",
    "PriorityDecision",
    "PriorityPolicy",
    "PriorityState",
    "Sighting",
]


class PriorityAction(str, Enum):
    """What the controller is asking the signal owner to do."""

    NO_CHANGE = "no_change"
    #: Start (or restart) an emergency-vehicle priority phase on an approach.
    GRANT = "grant"
    #: Keep the current priority phase running.
    HOLD = "hold"
    #: End the current priority phase and return to normal operation.
    RELEASE = "release"
    #: A qualifying sighting was refused, with a reason.
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class PriorityPolicy:
    """Tuning for :class:`~app.priority.controller.SignalPriorityController`.

    The defaults are conservative on purpose: two corroborating sightings in a
    ten-second window before anything is granted, a minimum hold so the phase
    cannot flap, and a hard ceiling so a stuck detector cannot hold an
    intersection open forever.
    """

    min_confidence: float = 0.80
    corroborating_sightings: int = 2
    corroboration_window_s: float = 10.0
    min_hold_s: float = 8.0
    max_hold_s: float = 45.0
    clear_after_s: float = 6.0
    cooldown_s: float = 20.0


@dataclass(frozen=True, slots=True)
class Sighting:
    """One detection report for one approach at one instant."""

    approach: str
    at: float
    detected: bool
    confidence: float


@dataclass(frozen=True, slots=True)
class PriorityState:
    """The controller's externally visible state."""

    active_approach: str | None = None
    granted_at: float | None = None
    last_sighting_at: float | None = None
    #: Approach -> the time its cooldown expires.
    cooldown_until: tuple[tuple[str, float], ...] = ()

    @property
    def priority_active(self) -> bool:
        return self.active_approach is not None


@dataclass(frozen=True, slots=True)
class PriorityDecision:
    """The controller's answer: an action, why, and the resulting state."""

    action: PriorityAction
    approach: str | None
    reason: str
    state: PriorityState
    #: Seconds the current grant has been held, when one is active.
    held_for_s: float | None = None
