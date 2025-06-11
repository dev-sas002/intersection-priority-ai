"""The signal-priority state machine.

The clock is injected, so every timing rule is asserted without sleeping.
"""

from __future__ import annotations

import pytest

from app.priority import (
    PriorityAction,
    PriorityPolicy,
    Sighting,
    SignalPriorityController,
)

POLICY = PriorityPolicy(
    min_confidence=0.80,
    corroborating_sightings=2,
    corroboration_window_s=10.0,
    min_hold_s=8.0,
    max_hold_s=45.0,
    clear_after_s=6.0,
    cooldown_s=20.0,
)


@pytest.fixture
def controller():
    return SignalPriorityController(POLICY, clock=lambda: 0.0)


def see(controller, approach, at, confidence=0.95, detected=True):
    return controller.observe(Sighting(approach, at, detected, confidence))


class TestEvidence:
    def test_a_single_sighting_does_not_grant_priority(self, controller):
        decision = see(controller, "north", at=0.0)
        assert decision.action is PriorityAction.NO_CHANGE
        assert decision.reason == "awaiting_corroboration_1_of_2"
        assert controller.state().priority_active is False

    def test_two_corroborating_sightings_grant_priority(self, controller):
        see(controller, "north", at=0.0)
        decision = see(controller, "north", at=1.0)
        assert decision.action is PriorityAction.GRANT
        assert decision.approach == "north"
        assert decision.state.priority_active is True

    def test_low_confidence_sightings_are_not_evidence(self, controller):
        see(controller, "north", at=0.0, confidence=0.5)
        decision = see(controller, "north", at=1.0, confidence=0.5)
        assert decision.action is PriorityAction.NO_CHANGE
        assert decision.reason == "below_confidence_floor"

    def test_a_negative_sighting_is_not_evidence(self, controller):
        decision = see(controller, "north", at=0.0, detected=False)
        assert decision.action is PriorityAction.NO_CHANGE
        assert decision.reason == "no_vehicle_detected"

    def test_sightings_outside_the_window_do_not_corroborate(self, controller):
        see(controller, "north", at=0.0)
        decision = see(controller, "north", at=30.0)
        assert decision.action is PriorityAction.NO_CHANGE
        assert decision.reason == "awaiting_corroboration_1_of_2"

    def test_evidence_is_per_approach(self, controller):
        see(controller, "north", at=0.0)
        decision = see(controller, "south", at=1.0)
        assert decision.action is PriorityAction.NO_CHANGE
        assert controller.state().priority_active is False


class TestHoldAndRelease:
    def test_further_sightings_hold_the_phase(self, controller):
        see(controller, "north", at=0.0)
        see(controller, "north", at=1.0)
        decision = see(controller, "north", at=3.0)
        assert decision.action is PriorityAction.HOLD
        assert decision.reason == "vehicle_still_present"

    def test_phase_cannot_end_before_the_minimum_hold(self, controller):
        see(controller, "north", at=0.0)
        see(controller, "north", at=1.0)
        # Nothing seen for longer than clear_after_s, but min_hold_s is 8s.
        decision = controller.tick(now=7.9)
        assert decision.action is PriorityAction.HOLD
        assert controller.state().priority_active is True

    def test_phase_releases_once_the_vehicle_has_cleared(self, controller):
        see(controller, "north", at=0.0)
        see(controller, "north", at=1.0)
        decision = controller.tick(now=9.0)
        assert decision.action is PriorityAction.RELEASE
        assert decision.reason == "vehicle_cleared"
        assert controller.state().priority_active is False

    def test_a_stuck_detector_cannot_hold_the_junction_forever(self, controller):
        see(controller, "north", at=0.0)
        see(controller, "north", at=1.0)
        for moment in range(2, 45):
            see(controller, "north", at=float(moment))
        decision = controller.tick(now=46.0)
        assert decision.action is PriorityAction.RELEASE
        assert decision.reason == "max_hold_exceeded"

    def test_held_for_is_reported(self, controller):
        see(controller, "north", at=0.0)
        see(controller, "north", at=1.0)
        decision = see(controller, "north", at=5.0)
        assert decision.held_for_s == pytest.approx(4.0)


class TestArbitration:
    def test_a_competing_approach_is_denied_inside_the_minimum_hold(self, controller):
        see(controller, "north", at=0.0)
        see(controller, "north", at=1.0)
        see(controller, "south", at=2.0)
        decision = see(controller, "south", at=3.0)
        assert decision.action is PriorityAction.DENY
        assert decision.reason == "conflicting_approach_within_min_hold"
        assert controller.state().active_approach == "north"

    def test_a_competing_approach_preempts_after_the_minimum_hold(self, controller):
        see(controller, "north", at=0.0)
        see(controller, "north", at=1.0)
        see(controller, "north", at=6.0)
        see(controller, "south", at=10.0)
        decision = see(controller, "south", at=11.0)
        assert decision.action is PriorityAction.GRANT
        assert decision.approach == "south"
        assert decision.reason == "preempted_by_other_approach"


class TestCooldown:
    def test_a_released_approach_is_in_cooldown(self, controller):
        see(controller, "north", at=0.0)
        see(controller, "north", at=1.0)
        controller.tick(now=9.0)
        see(controller, "north", at=10.0)
        decision = see(controller, "north", at=11.0)
        assert decision.action is PriorityAction.DENY
        assert decision.reason == "approach_in_cooldown"

    def test_cooldown_expires(self, controller):
        see(controller, "north", at=0.0)
        see(controller, "north", at=1.0)
        controller.tick(now=9.0)
        see(controller, "north", at=40.0)
        decision = see(controller, "north", at=41.0)
        assert decision.action is PriorityAction.GRANT

    def test_cooldowns_are_reported_in_the_state(self, controller):
        see(controller, "north", at=0.0)
        see(controller, "north", at=1.0)
        controller.tick(now=9.0)
        assert dict(controller.state().cooldown_until)["north"] == pytest.approx(29.0)


class TestLifecycle:
    def test_tick_on_an_idle_controller_is_a_no_op(self, controller):
        decision = controller.tick(now=100.0)
        assert decision.action is PriorityAction.NO_CHANGE
        assert decision.reason == "idle"

    def test_reset_clears_everything(self, controller):
        see(controller, "north", at=0.0)
        see(controller, "north", at=1.0)
        controller.reset()
        state = controller.state()
        assert state.priority_active is False
        assert state.cooldown_until == ()

    def test_a_late_sighting_cannot_resurrect_an_expired_phase(self, controller):
        see(controller, "north", at=0.0)
        see(controller, "north", at=1.0)
        # One sighting long after the vehicle should have cleared: the phase
        # ends first, and a lone sighting is not enough to re-grant.
        decision = see(controller, "north", at=60.0)
        assert decision.action is PriorityAction.NO_CHANGE
        assert controller.state().priority_active is False
        # ... and the approach it was holding went into cooldown on the way out.
        assert dict(controller.state().cooldown_until)["north"] == pytest.approx(80.0)
