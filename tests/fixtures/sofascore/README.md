# Sofascore API specimens

Captured 2026-07-16 from the live API with `curl_cffi` (`impersonate="chrome124"`),
event id `15502599`. All four returned HTTP 200.

These exist because **Sofascore reachability is transient**. The project CLAUDE.md
records a 403 ban (2026-06-11); it had lifted by 2026-07-16, and it can return. A
saved specimen keeps `sofascore_watcher` / `live_sofascore` buildable *and
verifiable* during a ban, instead of parsing against an unverified source.

| File | Endpoint | Verifies |
|---|---|---|
| `live_events.json` | `/sport/football/events/live` | match list, ids, status, scores |
| `event_incidents.json` | `/event/{id}/incidents` | goal events: `incidentType`, `incidentClass`, `isHome`, `time` |
| `event_statistics.json` | `/event/{id}/statistics` | team match stats |
| `event_lineups.json` | `/event/{id}/lineups` | `confirmed`, per-player `statistics` |

## What they do and do not establish

- **Do**: the live field names above are real, present, and current. `incidentClass`
  is the field the reconstructed `goal_type` maps to.
- **Do not**: prove own-goal side-crediting. This specimen has one goal, and it is
  `incidentClass: "regular"`. That `ownGoal` credits the *opposing* side was
  established from the 123-block oracle replay
  (`tests/test_live_reconciliation.py`), not from here.
- **Do not**: substitute for a live Serie A test. Serie A is off-season until
  mid-August; this event is from another competition.

Re-capture rather than hand-edit if the schema is suspected to have moved.
