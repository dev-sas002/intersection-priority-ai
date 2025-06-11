"""Signal-priority decision layer.

The project is named after traffic *control*, but detection on its own is only
half of that story. This module closes the gap with the part that is actually
engineering rather than hardware: given a stream of detection sightings over
time, decide whether to ask for an emergency-vehicle priority phase, keep it,
or end it.

It is a pure, deterministic state machine. It drives nothing: it emits
:class:`PriorityDecision` values that a signal controller -- which this project
does not contain and makes no claims about -- would act on. The clock is
injected, so every timing rule is testable without sleeping.

Rules, in order:

1. A sighting below ``min_confidence`` (or with ``detected=False``) is evidence
   of nothing and is discarded.
2. Evidence is corroborated: ``corroborating_sightings`` qualifying sightings
   must land inside ``corroboration_window_s`` before anything is granted. One
   frame of a confused detector cannot open an intersection.
3. An approach in cooldown cannot be granted again until ``cooldown_s`` after
   its last release, which stops a queue of vehicles re-triggering forever.
4. While a grant is active, further sightings on the same approach HOLD it and
   refresh the clear timer.
5. A different approach is DENIED until ``min_hold_s`` has elapsed; after that
   it preempts (release then grant), because an ambulance that has been waiting
   matters more than shaving the tail off the current phase.
6. :meth:`tick` releases on two timeouts: nothing seen for ``clear_after_s``
   (the vehicle has passed) and ``max_hold_s`` as a hard ceiling against a
   stuck or spoofed detector.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable

from app.priority.models import (
    PriorityAction,
    PriorityDecision,
    PriorityPolicy,
    PriorityState,
    Sighting,
)

__all__ = ["SignalPriorityController"]

logger = logging.getLogger(__name__)

Clock = Callable[[], float]


class SignalPriorityController:
    """Decides emergency-vehicle priority for one intersection.

    In-memory and single-instance by design (see the README's Limitations).
    Guarded by a lock so concurrent ``/predict`` requests cannot interleave a
    grant with a release.
    """

    def __init__(
        self,
        policy: PriorityPolicy | None = None,
        clock: Clock = time.monotonic,
    ) -> None:
        self.policy = policy or PriorityPolicy()
        self._clock = clock
        self._lock = threading.Lock()
        self._evidence: dict[str, deque[Sighting]] = defaultdict(deque)
        self._cooldowns: dict[str, float] = {}
        self._active: str | None = None
        self._granted_at: float | None = None
        self._last_sighting_at: float | None = None

    # -- public API --------------------------------------------------------
    def state(self) -> PriorityState:
        with self._lock:
            return self._state()

    def observe(self, sighting: Sighting) -> PriorityDecision:
        """Record a sighting and return the resulting decision."""
        with self._lock:
            return self._observe(sighting)

    def tick(self, now: float | None = None) -> PriorityDecision:
        """Advance time without new evidence; releases on timeout."""
        with self._lock:
            return self._tick(self._clock() if now is None else now)

    def reset(self) -> None:
        """Drop all state. Used by tests and by the ops reset endpoint."""
        with self._lock:
            self._evidence.clear()
            self._cooldowns.clear()
            self._active = None
            self._granted_at = None
            self._last_sighting_at = None

    # -- internals ---------------------------------------------------------
    def _state(self) -> PriorityState:
        return PriorityState(
            active_approach=self._active,
            granted_at=self._granted_at,
            last_sighting_at=self._last_sighting_at,
            cooldown_until=tuple(sorted(self._cooldowns.items())),
        )

    def _decision(
        self,
        action: PriorityAction,
        approach: str | None,
        reason: str,
        now: float,
    ) -> PriorityDecision:
        held = None if self._granted_at is None else max(0.0, now - self._granted_at)
        return PriorityDecision(
            action=action,
            approach=approach,
            reason=reason,
            state=self._state(),
            held_for_s=held,
        )

    def _observe(self, sighting: Sighting) -> PriorityDecision:
        now = sighting.at
        policy = self.policy

        # Timeouts are evaluated first so a late sighting cannot resurrect a
        # phase that should already have ended.
        timeout = self._tick(now)
        if timeout.action is PriorityAction.RELEASE:
            logger.info("Priority released before handling sighting: %s", timeout.reason)

        if not sighting.detected:
            return self._decision(
                PriorityAction.NO_CHANGE, sighting.approach, "no_vehicle_detected", now
            )
        if sighting.confidence < policy.min_confidence:
            return self._decision(
                PriorityAction.NO_CHANGE,
                sighting.approach,
                "below_confidence_floor",
                now,
            )

        window = self._record(sighting, now)

        if self._active == sighting.approach:
            self._last_sighting_at = now
            return self._decision(
                PriorityAction.HOLD, sighting.approach, "vehicle_still_present", now
            )

        if len(window) < policy.corroborating_sightings:
            return self._decision(
                PriorityAction.NO_CHANGE,
                sighting.approach,
                f"awaiting_corroboration_{len(window)}_of_"
                f"{policy.corroborating_sightings}",
                now,
            )

        cooldown_until = self._cooldowns.get(sighting.approach)
        if cooldown_until is not None and now < cooldown_until:
            return self._decision(
                PriorityAction.DENY, sighting.approach, "approach_in_cooldown", now
            )

        reason = "corroborated_emergency_vehicle"
        if self._active is not None:
            held = now - (self._granted_at or now)
            if held < policy.min_hold_s:
                return self._decision(
                    PriorityAction.DENY,
                    sighting.approach,
                    "conflicting_approach_within_min_hold",
                    now,
                )
            reason = "preempted_by_other_approach"
            self._release(now, reason)

        return self._grant(sighting.approach, now, reason)

    def _tick(self, now: float) -> PriorityDecision:
        if self._active is None:
            self._prune_all(now)
            return self._decision(PriorityAction.NO_CHANGE, None, "idle", now)

        policy = self.policy
        granted_at = self._granted_at or now
        held = now - granted_at

        if held >= policy.max_hold_s:
            approach = self._active
            self._release(now, "max_hold_exceeded")
            return self._decision(
                PriorityAction.RELEASE, approach, "max_hold_exceeded", now
            )

        last_seen = self._last_sighting_at or granted_at
        if held >= policy.min_hold_s and (now - last_seen) >= policy.clear_after_s:
            approach = self._active
            self._release(now, "vehicle_cleared")
            return self._decision(
                PriorityAction.RELEASE, approach, "vehicle_cleared", now
            )

        return self._decision(PriorityAction.HOLD, self._active, "holding", now)

    def _record(self, sighting: Sighting, now: float) -> deque[Sighting]:
        window = self._evidence[sighting.approach]
        window.append(sighting)
        self._prune(window, now)
        return window

    def _prune(self, window: deque[Sighting], now: float) -> None:
        cutoff = now - self.policy.corroboration_window_s
        while window and window[0].at < cutoff:
            window.popleft()

    def _prune_all(self, now: float) -> None:
        for window in self._evidence.values():
            self._prune(window, now)

    def _grant(self, approach: str, now: float, reason: str) -> PriorityDecision:
        self._active = approach
        self._granted_at = now
        self._last_sighting_at = now
        self._evidence[approach].clear()
        logger.info("Priority GRANTED to approach %s (%s).", approach, reason)
        return self._decision(PriorityAction.GRANT, approach, reason, now)

    def _release(self, now: float, reason: str) -> None:
        approach = self._active
        if approach is not None:
            self._cooldowns[approach] = now + self.policy.cooldown_s
            self._evidence[approach].clear()
            logger.info("Priority RELEASED from approach %s (%s).", approach, reason)
        self._active = None
        self._granted_at = None
        self._last_sighting_at = None
