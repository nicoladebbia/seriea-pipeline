"""Fetch the upcoming fixture list from The Odds API ``/events``.

Writes ``data/upcoming/matches.json``, the schedule the scheduler refreshes when
it goes stale (``scripts/pipeline/scheduler.py:923``) and the digest falls back
to for leagues with no per-league prediction file
(``scripts/pipeline/notify.py:1946``).

Rebuilt 2026-07-16. The original module was never git-added (the phantom-module
sweep) and, unlike ``live_reconciliation``, left **no oracle**: the only
surviving ``matches.json`` is the synthetic fallback tier — ``source: "manual"``,
templated ``venue`` (``f"{home} Stadium"``), no ``league`` key, and a
``fetched_at`` a week *after* the matches it describes. It records the fallback
firing, not the real fetch, so it could not be replayed. The source was chosen
deliberately instead; ``notify.py:1931`` calls this file the "raw Odds API
schedule", which corroborates the choice.

Why The Odds API ``/events``:

* It is already wired and authenticated here (``odds_fetcher.py:754``), so this
  adds no new dependency, no new credential, and no new ban surface.
* Its ``id`` is the same event id the odds layer already joins on — no
  name-matching layer is needed.
* The listing is billed **0 credits** (``odds_fetcher.py:759``).

Verification status — read this before trusting the module:

* **Schema: verified.** The four fields read here (``id``, ``home_team``,
  ``away_team``, ``commence_time``) are confirmed twice over: a real Odds API
  envelope this codebase fetched and cached
  (``tests/fixtures/odds_api/event_envelope.json``), and the working consumer at
  ``odds_fetcher.py:776-783`` that already extracts the same four from
  ``/events``. ``tests/test_fetch_upcoming_matches.py`` maps the real specimen.
* **Live: NOT verified.** The API key is deactivated (``DEACTIVATED_KEY``, 401)
  for the off-season, so no live ``/events`` call was made. The residual
  unknown is narrow: whether live ``/events`` diverges from the ``/odds``
  envelope the specimen came from. It should not — ``/events`` is that envelope
  minus ``bookmakers`` — but it is unconfirmed until the key is reactivated and
  real fixtures exist (mid-August).

Deliberately NOT emitted: ``venue`` and ``matchweek``. The Odds API supplies
neither, no consumer reads either from this file, and templating them is exactly
the synthetic fallback's fingerprint.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import requests

from config.leagues import ACTIVE_LEAGUES
from config.settings import DATA_DIR
from scripts.data.odds_fetcher import (
    API_BASE_URL,
    API_KEY,
    _resolve_sport_key,
    check_rate_limit,
    normalize_team,
    track_api_call,
)

log = logging.getLogger(__name__)

OUTPUT_PATH = DATA_DIR / "upcoming" / "matches.json"


def _event_to_match(event: dict[str, Any], league: str) -> dict[str, Any] | None:
    """Map one Odds API event onto a matches.json record.

    Returns None if the event lacks the fields the consumers require, rather
    than emitting a half-record with empty team names.
    """
    commence = event.get("commence_time") or ""
    home_raw = event.get("home_team") or ""
    away_raw = event.get("away_team") or ""
    if not (commence and home_raw and away_raw):
        return None

    # commence_time is UTC ISO with a trailing Z (e.g. 2026-04-17T16:30:00Z).
    parsed = datetime.fromisoformat(commence.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return {
        "home_team": normalize_team(home_raw),
        "away_team": normalize_team(away_raw),
        # date/time mirror the previous file's shape. They are UTC, matching
        # commence_time — notify.py compares `commence_time` against a UTC day.
        "date": parsed.strftime("%Y-%m-%d"),
        "time": parsed.strftime("%H:%M"),
        "commence_time": commence,
        "league": league,
        # The event id the odds layer joins on — the reason this source was picked.
        "event_id": event.get("id") or "",
        "source": "odds_api",
    }


def _fetch_league_events(league: str) -> list[dict[str, Any]]:
    """Upcoming events for one league. Empty list on any failure or off-season.

    Mirrors odds_fetcher.py:754-763 — same URL, same rate-limit gate, same
    0-credit accounting.
    """
    sport_key = _resolve_sport_key(league)

    ok, msg = check_rate_limit()
    if not ok:
        log.error("Rate limit: %s", msg)
        return []

    try:
        resp = requests.get(
            f"{API_BASE_URL}/sports/{sport_key}/events",
            params={"apiKey": API_KEY},
            timeout=30,
        )
        resp.raise_for_status()
        events = resp.json()
    except Exception as e:  # noqa: BLE001 - one dead league must not sink the rest
        log.error("Failed to fetch events for %s (%s): %s", league, sport_key, e)
        return []

    # /events listing is billed 0 credits by The Odds API.
    remaining_hdr = resp.headers.get("x-requests-remaining")
    track_api_call(
        credits_remaining=int(remaining_hdr) if remaining_hdr is not None else None,
        estimated_cost=0,
        endpoint=f"events_list_{sport_key}",
    )

    matches = [m for e in (events or []) if (m := _event_to_match(e, league))]
    log.info("%s: %d upcoming events", league, len(matches))
    return matches


def get_upcoming_matches(leagues: list[str] | None = None) -> list[dict[str, Any]]:
    """Upcoming fixtures across ``leagues``, sorted by kickoff.

    An empty list is a legitimate result, not a failure — the off-season returns
    no events. The caller at scheduler.py:925 already guards with ``if matches``.
    """
    out: list[dict[str, Any]] = []
    for league in leagues or ACTIVE_LEAGUES:
        out.extend(_fetch_league_events(league))
    out.sort(key=lambda m: m["commence_time"])
    return out


def save_upcoming_matches(matches: list[dict[str, Any]]) -> None:
    """Write matches.json in the shape notify.py:1950 reads."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "matches": matches,
        "count": len(matches),
    }
    tmp = OUTPUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(OUTPUT_PATH)  # atomic — readers never see a half-written file
    log.info("Wrote %d upcoming matches to %s", len(matches), OUTPUT_PATH)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    matches = get_upcoming_matches()
    save_upcoming_matches(matches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
