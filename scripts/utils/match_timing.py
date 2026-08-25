#!/usr/bin/env python3
"""Match timing classification and live-match filtering.

Classifies matches into timing windows based on commence_time and filters
out live/completed matches from odds processing.
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config.settings import DATA_DIR, get_current_season
from config.team_names import normalize_team

log = logging.getLogger(__name__)

# Match window thresholds (hours relative to kickoff)
WINDOW_FAR = 6.0         # >6h = far
WINDOW_APPROACHING = 1.0  # 1-6h = approaching
WINDOW_IMMINENT = 0.0     # 0-1h = imminent (lineups available)
WINDOW_LIVE_END = -2.0    # -2h to 0 = likely live
# < -2h = completed


def classify_match_window(
    commence_time_iso: str,
    now: Optional[datetime] = None,
) -> str:
    """Classify a match into a timing window.

    Args:
        commence_time_iso: ISO 8601 timestamp (e.g. "2026-02-07T19:45:00Z")
        now: Override current time (for testing)

    Returns:
        One of: "far", "approaching", "imminent", "live", "completed"
    """
    if now is None:
        now = datetime.now(timezone.utc)

    try:
        commence_dt = datetime.fromisoformat(
            commence_time_iso.replace("Z", "+00:00")
        )
    except (ValueError, AttributeError):
        log.warning(f"Invalid commence_time: {commence_time_iso}, treating as far")
        return "far"

    hours_until = (commence_dt - now).total_seconds() / 3600

    if hours_until > WINDOW_FAR:
        return "far"
    elif hours_until > WINDOW_APPROACHING:
        return "approaching"
    elif hours_until > WINDOW_IMMINENT:
        return "imminent"
    elif hours_until > WINDOW_LIVE_END:
        return "live"
    else:
        return "completed"


def get_hours_until_kickoff(
    commence_time_iso: str,
    now: Optional[datetime] = None,
) -> float:
    """Return hours until kickoff (negative = already started)."""
    if now is None:
        now = datetime.now(timezone.utc)
    try:
        commence_dt = datetime.fromisoformat(
            commence_time_iso.replace("Z", "+00:00")
        )
        return (commence_dt - now).total_seconds() / 3600
    except (ValueError, AttributeError):
        return 999.0


def filter_prematch_only(matches: Dict) -> Tuple[Dict, int]:
    """Filter out live and completed matches.

    Args:
        matches: Dict of match_key -> match_data, each with "commence_time"

    Returns:
        (filtered_matches, num_filtered)
    """
    filtered = {}
    num_filtered = 0

    for match_key, match_data in matches.items():
        commence = match_data.get("commence_time", "")
        window = classify_match_window(commence)

        # Add window classification to match data
        match_data["match_window"] = window

        if window in ("live", "completed"):
            hours = get_hours_until_kickoff(commence)
            log.warning(
                f"Filtering {window} match: {match_key} "
                f"(commence_time={commence}, {hours:+.1f}h)"
            )
            num_filtered += 1
        else:
            filtered[match_key] = match_data

    if num_filtered:
        log.info(
            f"Filtered {num_filtered} live/completed match(es), "
            f"{len(filtered)} pre-match remaining"
        )

    return filtered, num_filtered


# ---------------------------------------------------------------------------
# Upcoming-fixture sources.
#
# Lives here, not in predict_unified, because TWO independent consumers need it:
# the prediction generator and the scheduler (which decides when the T-30
# pre-kickoff monitors fire). Duplicating a two-league fixture reader is exactly
# how this repo's recurring league-parity bugs get made -- see the "EPL data
# missing where SA has it" catalogue in CLAUDE.md.
#
# Deliberately dependency-light: stdlib plus config (unicodedata / pathlib only).
# The scheduler runs on a timer, so importing this must never pull in numpy,
# pandas, or anything that could trigger a scrape as an import side effect.
# ---------------------------------------------------------------------------

# How far ahead a fixture still counts as "upcoming". The Sofascore source
# carries the WHOLE season (370 per league), so without a horizon every run
# would predict into next May — semantically meaningless, since form and lineup
# signal do not exist that far out, and 740 predictions per run instead of ~20.
# Measured 2026-08-24: 7 and 10 days both yield exactly one matchweek per league
# (10 SA + 10 EPL); 14 yields two. 10 gives a matchweek plus slack for a midweek
# round without pulling in the one after it.
UPCOMING_HORIZON_DAYS = int(os.environ.get("UPCOMING_HORIZON_DAYS", "10"))


def _entry_kickoff(entry: Dict) -> Optional[datetime]:
    """UTC kickoff for a fixture dict, or None if it carries no usable date.

    Returning None (rather than a default) is deliberate: an entry we cannot
    date must be dropped, because "keep what you can't judge" is exactly how a
    frozen row survives every filter downstream.
    """
    for key in ("commence_time", "kickoff", "date", "match_date"):
        raw = entry.get(key)
        if not raw:
            continue
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return None


def _is_future(entry: Dict, now: datetime, horizon_days: Optional[int] = None) -> bool:
    """True for a fixture that has not kicked off and is inside the horizon."""
    ko = _entry_kickoff(entry)
    if ko is None or ko <= now:
        return False
    days = UPCOMING_HORIZON_DAYS if horizon_days is None else horizon_days
    return ko <= now + timedelta(days=days)


def _dedup_key(entry: Dict) -> str:
    """Identity of a fixture, on CANONICAL names.

    Sources spell clubs differently — Sofascore says "AC Milan" where the manual
    file says "Milan". Keying on raw names lets the same match through twice.
    """
    return f"{normalize_team(entry.get('home_team'))}_{normalize_team(entry.get('away_team'))}"


def _sofascore_fixture_files() -> List[Tuple[Path, str]]:
    """The current season's fixture file for every active league.

    Season is derived, never written down — a hardcoded ``2026_2027`` here is
    just next August's silent breakage. Both league files are listed because a
    loader that only opens the Serie A one is this repo's oldest recurring bug.
    """
    season = get_current_season().replace("-", "_")
    base = DATA_DIR / "external" / "sofascore"
    return [
        (base / f"fixtures_{season}.json", "serie_a"),
        (base / f"fixtures_{season}_premier_league.json", "premier_league"),
    ]


def _load_sofascore_fixtures(
    now: datetime,
    files: Optional[List[Tuple[Path, str]]] = None,
    horizon_days: Optional[int] = UPCOMING_HORIZON_DAYS,
) -> List[Dict]:
    """Forward fixtures from the Sofascore season files.

    This is the only fixture source that does NOT depend on the Odds API, which
    is why it leads. Emits the same shape ``manual_matches.json`` uses, so the
    downstream readers (referee_integration, current_form_calculator,
    weather_integration, scheduler) need no change.

    The horizon is applied HERE and defaults to ON. These files carry the whole
    season (~740 fixtures across both leagues), so an unhorizoned read is a
    footgun: the weather fetcher hit it on 2026-08-24 and would have requested
    740 forecasts running into May 2027. A caller that genuinely wants the full
    season must pass ``horizon_days=None`` -- the dangerous option should be the
    one you have to type.
    """
    out: List[Dict] = []
    for path, league in files if files is not None else _sofascore_fixture_files():
        if not path.exists():
            continue
        try:
            with open(path) as f:
                rows = json.load(f)
        except (OSError, ValueError) as e:
            log.warning("Could not read fixtures %s: %s", path.name, e)
            continue
        if not isinstance(rows, list):
            continue
        for r in rows:
            if not isinstance(r, dict):
                continue
            ts = r.get("startTimestamp")
            if not isinstance(ts, (int, float)):
                continue
            ko = datetime.fromtimestamp(ts, tz=timezone.utc)
            if ko <= now:
                continue
            if horizon_days is not None and ko > now + timedelta(days=horizon_days):
                continue
            try:
                home = r["homeTeam"]["name"]
                away = r["awayTeam"]["name"]
            except (KeyError, TypeError):
                continue
            out.append({
                "home_team": normalize_team(home),
                "away_team": normalize_team(away),
                "commence_time": ko.isoformat().replace("+00:00", "Z"),
                "date": ko.date().isoformat(),
                "league": league,
            })
    return out
