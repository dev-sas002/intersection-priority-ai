# Signal priority: captured session

A walk through `SignalPriorityController` over HTTP, captured on 2026-11-07
against the Docker Compose stack with the default policy: two corroborating
sightings inside ten seconds, a 0.80 confidence floor, an eight-second minimum
hold and release six seconds after the last sighting.

Nothing is edited. `held_for_s` is measured on the server's monotonic clock,
so it stays small until the pause near the end.

The controller only emits decisions. It drives no signal hardware.

```console
$ curl -s localhost:8250/priority/state
{
  "priority_active": false,
  "active_approach": null,
  "held_for_s": null,
  "cooldown_remaining_s": {}
}
$ curl -s -H 'Content-Type: application/json' -d '{"approach":"north","confidence":0.93}' localhost:8250/priority/sightings
{
  "action": "no_change",
  "approach": "north",
  "reason": "awaiting_corroboration_1_of_2",
  "held_for_s": null,
  "state": {
    "priority_active": false,
    "active_approach": null,
    "held_for_s": null,
    "cooldown_remaining_s": {}
  }
}
$ curl -s -H 'Content-Type: application/json' -d '{"approach":"north","confidence":0.95}' localhost:8250/priority/sightings
{
  "action": "grant",
  "approach": "north",
  "reason": "corroborated_emergency_vehicle",
  "held_for_s": 0.0,
  "state": {
    "priority_active": true,
    "active_approach": "north",
    "held_for_s": 0.0,
    "cooldown_remaining_s": {}
  }
}
$ curl -s -H 'Content-Type: application/json' -d '{"approach":"east","confidence":0.99}' localhost:8250/priority/sightings
{
  "action": "no_change",
  "approach": "east",
  "reason": "awaiting_corroboration_1_of_2",
  "held_for_s": 0.04618608399960067,
  "state": {
    "priority_active": true,
    "active_approach": "north",
    "held_for_s": 0.04618608399960067,
    "cooldown_remaining_s": {}
  }
}
$ curl -s -H 'Content-Type: application/json' -d '{"approach":"east","confidence":0.99}' localhost:8250/priority/sightings
{
  "action": "deny",
  "approach": "east",
  "reason": "conflicting_approach_within_min_hold",
  "held_for_s": 0.09145054199962033,
  "state": {
    "priority_active": true,
    "active_approach": "north",
    "held_for_s": 0.09145054199962033,
    "cooldown_remaining_s": {}
  }
}
$ curl -s -H 'Content-Type: application/json' -d '{"approach":"west","confidence":0.4}' localhost:8250/priority/sightings
{
  "action": "no_change",
  "approach": "west",
  "reason": "below_confidence_floor",
  "held_for_s": 0.13748708399998577,
  "state": {
    "priority_active": true,
    "active_approach": "north",
    "held_for_s": 0.13748708399998577,
    "cooldown_remaining_s": {}
  }
}
$ curl -s -w '\nHTTP %{http_code}\n' -H 'Content-Type: application/json' -d '{"approach":"north","at":1e12}' localhost:8250/priority/sightings
{"detail":[{"type":"extra_forbidden","loc":["body","at"],"msg":"Extra inputs are not permitted","input":1000000000000.0}]}
HTTP 422
$ sleep 15
$ curl -s -X POST localhost:8250/priority/tick
{
  "action": "release",
  "approach": "north",
  "reason": "vehicle_cleared",
  "held_for_s": 15.19947371499984,
  "state": {
    "priority_active": false,
    "active_approach": null,
    "held_for_s": null,
    "cooldown_remaining_s": {
      "north": 20.0
    }
  }
}
$ curl -s localhost:8250/priority/state
{
  "priority_active": false,
  "active_approach": null,
  "held_for_s": null,
  "cooldown_remaining_s": {
    "north": 19.944
  }
}
```

## What each step shows

| Step | Action | Rule |
| --- | --- | --- |
| Initial state | | Nothing active, no cooldowns. |
| First `north` sighting | `no_change` | One sighting is not evidence; `reason` counts progress toward corroboration. |
| Second `north` sighting | `grant` | Corroborated inside the 10 s window. |
| First `east` sighting | `no_change` | `east` starts gathering its own evidence. |
| Second `east` sighting | `deny` | Corroborated, but `north` has held for less than `ITCS_PRIORITY_MIN_HOLD_S` (8 s). |
| `west` at 0.4 | `no_change` | Below `ITCS_PRIORITY_MIN_CONFIDENCE`; not recorded at all. |
| `at` in the body | HTTP 422 | Sightings are stamped by the server; a caller-chosen time is refused. |
| `tick` after 15 s | `release` | Held past the minimum and nothing seen for 6 s: `vehicle_cleared`. `held_for_s` is how long the grant lasted. |
| Final state | | `north` is in its 20 s cooldown, shown as seconds remaining. |

The `max_hold_exceeded` release (45 s ceiling) and preemption after the
minimum hold are covered by `tests/test_priority_controller.py` with an
injected clock rather than by waiting.
