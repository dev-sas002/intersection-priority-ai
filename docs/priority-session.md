# Signal priority — captured session

A real walk through `SignalPriorityController` over HTTP, captured on
2026-09-23 against the service on `http://localhost:8250` with the default
policy: two corroborating sightings inside ten seconds, a confidence floor of
0.80, and an eight-second minimum hold.

Nothing here is edited. `held_for_s` is measured against the server's monotonic
clock, which is why the numbers are small — the whole session runs in about a
tenth of a second.

The controller emits decisions. **It drives no signal hardware**, and this
project contains no controller that does.

```console
$ curl -s http://localhost:8250/priority/state | jq
{
    "priority_active": false,
    "active_approach": null,
    "held_for_s": null,
    "cooldowns": {}
}

$ SIGHT='{"approach":"north","confidence":0.95}'
$ curl -s -H "Content-Type: application/json" -d "$SIGHT" http://localhost:8250/priority/sightings | jq
{
    "action": "no_change",
    "approach": "north",
    "reason": "awaiting_corroboration_1_of_2",
    "held_for_s": null,
    "state": {
        "priority_active": false,
        "active_approach": null,
        "held_for_s": null,
        "cooldowns": {}
    }
}

$ # the same sighting again — now it is corroborated
{
    "action": "grant",
    "approach": "north",
    "reason": "corroborated_emergency_vehicle",
    "held_for_s": 0.0,
    "state": {
        "priority_active": true,
        "active_approach": "north",
        "held_for_s": 0.0,
        "cooldowns": {}
    }
}

$ # a competing approach, twice, while "north" is inside its minimum hold
{
    "action": "deny",
    "approach": "east",
    "reason": "conflicting_approach_within_min_hold",
    "held_for_s": 0.04088300000876188,
    "state": {
        "priority_active": true,
        "active_approach": "north",
        "held_for_s": 0.04088300000876188,
        "cooldowns": {}
    }
}

$ # a low-confidence sighting is not evidence at all
{
    "action": "no_change",
    "approach": "west",
    "reason": "below_confidence_floor",
    "held_for_s": 0.073408207972534,
    "state": {
        "priority_active": true,
        "active_approach": "north",
        "held_for_s": 0.073408207972534,
        "cooldowns": {}
    }
}

$ curl -s http://localhost:8250/priority/state | jq
{
    "priority_active": true,
    "active_approach": "north",
    "held_for_s": 0.10594845796003938,
    "cooldowns": {}
}
```

## What each step demonstrates

| Step | Action | Rule being exercised |
| --- | --- | --- |
| Initial state | — | Nothing is active; no approach is in cooldown. |
| First `north` sighting | `no_change` | One sighting is not evidence. `reason` counts the progress: `awaiting_corroboration_1_of_2`. |
| Second `north` sighting | `grant` | Corroborated inside the window, so priority is granted. |
| `east` sightings | `deny` | A competing approach cannot preempt `north` until `ITCS_PRIORITY_MIN_HOLD_S` (8 s) has elapsed, so the phase cannot flap. |
| `west` at 0.4 | `no_change` | Below `ITCS_PRIORITY_MIN_CONFIDENCE`; not recorded as evidence at all. |
| Final state | — | `north` still holds the phase. |

Two rules do not appear above because they are time-driven rather than
evidence-driven, and the suite exercises them with an injected clock instead of
by sleeping:

- **`vehicle_cleared`** — after the minimum hold, nothing seen on the active
  approach for `ITCS_PRIORITY_CLEAR_AFTER_S` releases the phase.
- **`max_hold_exceeded`** — `ITCS_PRIORITY_MAX_HOLD_S` is a hard ceiling, so a
  stuck or spoofed detector cannot hold an intersection open indefinitely.

In a request/response service the clock only advances when a request arrives;
`POST /priority/tick` exists so a caller can drive those timeouts without
sending new evidence.
