# Sofascore API specimens

Real Sofascore API responses, captured 2026-07-16 with `curl_cffi`
(`impersonate="chrome124"`), all HTTP 200. Not hand-written mocks.

These exist because **Sofascore reachability is transient**. The project
CLAUDE.md records a 403 ban (2026-06-11); it had lifted by 2026-07-16, and it
can return. A saved specimen keeps `live_sofascore` / `sofascore_watcher`
buildable *and verifiable* during a ban, instead of parsing against an
unverified source.

| File | Endpoint | Event | Why this event |
|---|---|---|---|
| `event_incidents.json` | `/event/{id}/incidents` | **13981681** Sassuolo v Hellas Verona | Serie A, and the oracle holds a block for it — so the parse is checked against real stored output |
| `event_statistics.json` | `/event/{id}/statistics` | **13981681** | Same match. Carries `expectedGoals`, which minor-league matches don't |
| `event_lineups.json` | `/event/{id}/lineups` | **13981681** | Same match. Includes Laurienté, who has a real assist — the only way to tell `goalAssist` from `assists` |
| `event_incidents_var.json` | `/event/{id}/incidents` | **13981716** Genoa v Udinese | The only specimen carrying a `varDecision` incident |
| `live_events.json` | `/sport/football/events/live` | — | Match-list shape; proves the endpoint was live when captured |

`tests/test_live_sofascore.py` maps all of these through the real parsers and
compares against the oracle in `data/live/*.json`.

## What they establish

Every field `live_sofascore` reads, on real data. Specifically:

- The five `incidentType` values that map to events, plus `period` and
  `inGamePenalty`, which are real and **dropped** (no oracle event has either).
- `incidentClass` is the field behind `goal_type` / `card_type` / `decision`.
- `addedTime` is absent on ordinary incidents and `999` on `period` ones.
- `injuryTime` carries no `isHome`.
- `goalAssist` — **not** `assists`, which does not exist.

## What they do NOT establish

- **Own-goal side-crediting.** No specimen contains an own goal. That `ownGoal`
  credits the *opposing* side came from the oracle replay
  (`tests/test_live_reconciliation.py`), which has 4 of them.
- **A live Serie A match.** These are finished matches — Serie A is off-season
  until mid-August. The same endpoints serve both, but nothing here proves the
  live in-play feed behaves identically. One known gap: the finished feed
  carries `injuryTime` incidents the in-play snapshot lacked.

## Maintenance

Re-capture rather than hand-edit if a schema break is suspected. Keep event
13981681 — its value is that the oracle holds a block for the same match, which
is what makes the replay a real test rather than a self-consistency check.
