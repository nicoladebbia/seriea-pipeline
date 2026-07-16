"""Odds snapshot capture and line-movement analysis.

Rebuilt 2026-07-16. The original module was never `git add`ed and was swept
after 2026-06-01 (see AUGUST_RUNBOOK.md §3b). Its behaviour was reconstructed
from surviving artifacts, NOT from memory of the source:

- 1,098 `odds_/bookmakers_/extra_*.json` snapshots in `data/odds_snapshots/`
- 17 logged `analyze_movements` results in `logs/` (with timestamps)
- the last written `data/upcoming/odds_movement.json` (2026-06-01)

What that evidence pins exactly, and what it does not, is documented at each
constant below. Anything not pinned is marked and flagged in the runbook.

Why this module matters: `clv_tracker` reads `bookmakers_*.json` / `extra_*.json`
for closing odds. CLV is the only rock-solid edge signal in this repo, and every
caller here wraps the import in a bare `except`, so a missing module degrades to
"no snapshots, no CLV" *silently*.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
UPCOMING_DIR = DATA_DIR / "upcoming"
SNAPSHOTS_DIR = DATA_DIR / "odds_snapshots"
MOVEMENT_PATH = UPCOMING_DIR / "odds_movement.json"

ODDS_FULL_PATH = UPCOMING_DIR / "odds_full.json"
BOOKMAKERS_PATH = UPCOMING_DIR / "odds_bookmakers.json"
EXTRA_MARKETS_PATH = UPCOMING_DIR / "odds_extra_markets.json"

_OUTCOMES = ("home", "draw", "away")
_TS_FMT = "%Y%m%d_%H%M%S"
_TS_RE = re.compile(r"_(\d{8}_\d{6})\.json$")

# Window over which "opening" odds are taken. UNIQUELY IDENTIFIED by the data:
# swept against all 17 logged runs, 48h is the only candidate with any feasible
# threshold pair -- 24h, 36h and 72h are all refuted (no pair reproduces the
# logs). Independently confirmed per-match: 48h reproduces the stored
# `snapshots_count` (5) and `hours_tracked` (35.8) exactly on all 5 matches of
# the 2026-06-01 run.
MOVEMENT_WINDOW_HOURS = 48.0

# Decimal-odds movement thresholds on max(|dHome|, |dDraw|, |dAway|).
#
# NOT uniquely identified -- these are the natural round values inside the
# feasible ranges found by sweeping the threshold plane against all 17 logged
# runs (data/odds_snapshots + logs):
#     LINE  feasible in [0.09,  0.11]   -> 0.10 chosen (dead centre)
#     STEAM feasible in [0.125, 0.155]  -> 0.15 chosen
# Only 4 of the 17 runs are informative; the other 13 are the degenerate
# all-zero case that any positive threshold satisfies. With these values all 17
# reproduce exactly, including the 20-match 2026-05-06 run (6 line / 5 steam)
# and the 5/5/4 run that pins one match into the (LINE, STEAM] gap.
#
# Residual risk: a true STEAM of e.g. 0.13 would differ from 0.15 only for
# matches moving in that band. `is_steam_move` feeds a 0.8-weight component in
# features/market_intelligence.py, so this is betting-relevant, not cosmetic --
# flagged in AUGUST_RUNBOOK.md §3b for confirmation against a live run.
LINE_MOVE_THRESHOLD = 0.10
STEAM_MOVE_THRESHOLD = 0.15


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return default


def _ts_from_name(path: Path) -> datetime | None:
    m = _TS_RE.search(path.name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), _TS_FMT)
    except ValueError:
        return None


def _iter_snapshots() -> list[tuple[datetime, Path]]:
    """Return (timestamp, path) for every h2h snapshot, oldest first."""
    if not SNAPSHOTS_DIR.exists():
        return []
    out = []
    for p in SNAPSHOTS_DIR.glob("odds_*.json"):
        ts = _ts_from_name(p)
        if ts is not None:
            out.append((ts, p))
    return sorted(out)


def _simple_odds_from_full() -> dict[str, dict[str, float]]:
    """Derive {match: {home, draw, away}} from odds_full.json.

    Mirrors what run_full_pipeline passes in explicitly; used for the bare
    `save_snapshot()` call in betting_unified.py.
    """
    full = _read_json(ODDS_FULL_PATH, {}) or {}
    simple: dict[str, dict[str, float]] = {}
    for key, md in (full.get("matches") or {}).items():
        h2h = md.get("h2h")
        if not h2h:
            continue
        try:
            simple[key] = {o: float(h2h[o]) for o in _OUTCOMES}
        except (KeyError, TypeError, ValueError):
            continue
    return simple


def save_snapshot(odds: dict[str, dict[str, float]] | None = None) -> str | None:
    """Write the three per-run snapshot files, sharing one timestamp.

    Timestamps are LOCAL-naive, deliberately: all 1,098 existing snapshots are
    named in local time, and the movement window compares filename timestamps to
    `datetime.now()`. Switching to UTC here would make new files sort against old
    ones incorrectly and silently corrupt the window. This is a conscious
    deviation from the repo-wide "UTC-aware ISO" rule, scoped to this directory.
    """
    if odds is None:
        odds = _simple_odds_from_full()
    if not odds:
        log.warning("save_snapshot: no odds to snapshot")
        return None

    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    stamp = now.strftime(_TS_FMT)
    iso = now.isoformat()

    with open(SNAPSHOTS_DIR / f"odds_{stamp}.json", "w") as fh:
        json.dump({"timestamp": iso, "matches": odds}, fh, indent=2)
    log.info("Saved odds snapshot: odds_%s.json (%d matches)", stamp, len(odds))

    for label, src in (("bookmakers", BOOKMAKERS_PATH), ("extra", EXTRA_MARKETS_PATH)):
        payload = _read_json(src, {}) or {}
        matches = payload.get("matches")
        if not matches:
            log.warning("save_snapshot: %s has no matches; skipping %s_%s.json", src.name, label, stamp)
            continue
        with open(SNAPSHOTS_DIR / f"{label}_{stamp}.json", "w") as fh:
            json.dump({"timestamp": iso, "matches": matches}, fh, indent=2)
        log.info("Saved %s snapshot: %s_%s.json", "bookmaker" if label == "bookmakers" else label, label, stamp)

    return stamp


def _implied_shift(cur: dict[str, Any], opening: dict[str, Any], outcome: str) -> float:
    """Change in implied probability (1/odds) from opening to current."""
    try:
        return round(1.0 / float(cur[outcome]) - 1.0 / float(opening[outcome]), 4)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return 0.0


def analyze_movements(
    current_odds: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compare current odds to the oldest snapshot within MOVEMENT_WINDOW_HOURS.

    The window is resolved PER MATCH -- the opening price is the oldest snapshot
    that actually contains that match, not the oldest snapshot overall. Matches
    enter the feed at different times, and the per-match rule is what reproduces
    the stored snapshots_count/hours_tracked.
    """
    if current_odds is None:
        current_odds = _simple_odds_from_full()
    if not current_odds:
        return {}

    now = datetime.now()
    cutoff = now - timedelta(hours=MOVEMENT_WINDOW_HOURS)
    history = [(ts, p) for ts, p in _iter_snapshots() if cutoff <= ts <= now]

    # Load once; per-match scans reuse this.
    loaded = [(ts, (_read_json(p, {}) or {}).get("matches") or {}) for ts, p in history]

    movements: dict[str, dict[str, Any]] = {}
    for key, cur in current_odds.items():
        hits = [(ts, m[key]) for ts, m in loaded if key in m]
        if not hits:
            continue
        open_ts, opening = hits[0]
        try:
            deltas = {o: round(float(cur[o]) - float(opening[o]), 4) for o in _OUTCOMES}
        except (KeyError, TypeError, ValueError):
            continue

        biggest = max(abs(v) for v in deltas.values())
        is_line = biggest > LINE_MOVE_THRESHOLD
        is_steam = biggest > STEAM_MOVE_THRESHOLD

        if deltas["home"] > LINE_MOVE_THRESHOLD:
            direction = "home_drifting"
        elif deltas["away"] > LINE_MOVE_THRESHOLD:
            direction = "away_drifting"
        else:
            direction = "stable"

        movements[key] = {
            "match": key,
            "current_home": cur["home"],
            "current_draw": cur["draw"],
            "current_away": cur["away"],
            "opening_home": opening["home"],
            "opening_draw": opening["draw"],
            "opening_away": opening["away"],
            "home_movement": deltas["home"],
            "draw_movement": deltas["draw"],
            "away_movement": deltas["away"],
            # Informational only -- no module reads these (verified by grep), so
            # the exact formula is not pinned by any surviving artifact.
            "implied_prob_shift_home": _implied_shift(cur, opening, "home"),
            "implied_prob_shift_away": _implied_shift(cur, opening, "away"),
            "direction": direction,
            "is_line_move": is_line,
            "is_steam_move": is_steam,
            "snapshots_count": len(hits),
            "hours_tracked": round((now - open_ts).total_seconds() / 3600, 1),
        }

    _save_movement(movements)
    return movements


def _save_movement(movements: dict[str, dict[str, Any]]) -> None:
    """Persist movement analysis for the 9 modules that read odds_movement.json."""
    summary = {
        "total_matches": len(movements),
        "line_moves": sum(1 for m in movements.values() if m["is_line_move"]),
        "steam_moves": sum(1 for m in movements.values() if m["is_steam_move"]),
    }
    UPCOMING_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "analyzed_at": datetime.now().isoformat(),
        "matches": movements,
        "summary": summary,
    }
    with open(MOVEMENT_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)
    log.info("Saved movement analysis: %s", summary)


def run_single_snapshot() -> dict[str, int]:
    """Capture one snapshot and analyse movement. Used by the web scheduler."""
    odds = _simple_odds_from_full()
    if not odds:
        log.warning("run_single_snapshot: no odds available")
        return {"matches": 0, "bookmakers": 0, "steam_moves": 0}

    save_snapshot(odds)
    movements = analyze_movements(odds)
    bookmakers = (_read_json(BOOKMAKERS_PATH, {}) or {}).get("matches") or {}
    return {
        "matches": len(odds),
        "bookmakers": len(bookmakers),
        "steam_moves": sum(1 for m in movements.values() if m.get("is_steam_move")),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    print(json.dumps(run_single_snapshot(), indent=2))
