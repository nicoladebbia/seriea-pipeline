"""ESPN public match feed — second source for LIVE events and team stats.

Why this exists (2026-09-05): mid-match, every ``api.sofascore.com`` call
answered ``403 {"reason": "challenge"}`` (``server: Varnish``) while the www
site stayed 200 — a third 403 variant, neither the blanket IP deny nor the
Cloudflare fingerprint ban documented in CLAUDE.md. Sofascore match pages carry
incidents but NOT statistics, so with the API tier challenged there was no
Sofascore path to live team stats at all. ESPN's unauthenticated site API served
this exact match (AS Roma vs Atalanta, 40') with possession, shots, shots on
target, blocked shots, corners, fouls, saves, tackles, clearances and cards —
every stat the /live card renders — plus goals, cards and substitutions as
``keyEvents``.

Output is in the SAME shape ``scripts.data.live_sofascore`` produces
(``events`` newest-first, ``statistics`` keyed ``{"home": v, "away": v}``), so
``live_monitor`` and the /live template need no branching. ``player_stats`` is
never produced here: ESPN's boxscore has no per-player live stats.

Specimens (2026-09-05, saved under ``tests/fixtures/espn/``):

* scoreboard ``ita.1``: competitors ``{"homeAway": "home"|"away", "team":
  {"id", "displayName"}}``; ``status.type.state`` in ``pre|in|post``.
* summary ``keyEvents``: ``type.type`` is a slug — ``goal``, ``goal---volley``,
  ``goal---header``, ``yellow-card``, ``substitution``, ``halftime``,
  ``end-regular-time``, plus noise (``kickoff``, ``start-delay``, ``end-delay``,
  ``start-2nd-half``). ``clock.displayValue`` is ``"47'"`` or ``"45'+5'"``.
  ``participants`` on a goal are ``[scorer, assist]``; on a substitution
  ``[player_in, player_out]`` (text: "Cher Ndour replaces Christ Inao Oulaï").
* summary ``boxscore.teams[].statistics``: ``[{"name", "displayValue"}]`` with
  string values (``"63"``, ``"0.3"``).

Verified on 2026-08/09 specimens (14-day sweep of both leagues):

* ``own-goal``: ESPN's ``team`` is the BENEFICIARY ("Own Goal by Redouane
  Halhal, Venezia." carries ``team=AC Milan``). Our convention (Sofascore's,
  relied on by reconciliation and the goal ping) is that ``is_home`` marks the
  SCORER's side and ``goal_type=ownGoal`` credits the opponent — so an ESPN own
  goal is stored with ``is_home`` flipped to the scorer's side.
* ``penalty---scored``: a goal, ``team`` = scoring team. ``penalty---missed``
  is not a goal and is dropped.
* ``red-card``: ``card_type=red``. VAR incidents are dropped (no slug seen).

The header score can trail the ``keyEvents`` feed by a tick; ``score`` is
therefore the per-side max of the header and the score derived from goal
events, so a goal never shows in the list before it shows on the board.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import UTC, datetime
from typing import Any

from curl_cffi import requests as cffi_requests

from config.team_names import normalize_team

log = logging.getLogger(__name__)

_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
LEAGUE_SLUGS = {"serie_a": "ita.1", "premier_league": "eng.1"}
_SCOREBOARD_TTL_S = 60
_TIMEOUT = 15

# ESPN boxscore stat name -> our live_stats key (the /live card reads these).
_STAT_KEYS = {
    "possessionPct": "possession",
    "totalShots": "shots",
    "shotsOnTarget": "shots_on_target",
    "blockedShots": "blocked_shots",
    "wonCorners": "corners",
    "foulsCommitted": "fouls",
    "saves": "saves",
    "offsides": "offsides",
    "yellowCards": "yellow_cards",
    "redCards": "red_cards",
    "totalTackles": "tackles",
    "effectiveClearance": "clearances",
    "accuratePasses": "accurate_passes",
}

_CLOCK_RE = re.compile(r"(\d+)'(?:\+(\d+)')?")

# scoreboard cache: slug -> (fetched_at_monotonic, payload)
_scoreboards: dict[str, tuple[float, dict[str, Any]]] = {}
# The fast live loop calls this every few seconds per match. A refusal (429/403)
# or a connection error pauses ALL ESPN calls for a minute rather than letting
# a 5s loop hammer a throttle into a ban — the Sofascore lesson.
_BACKOFF_S = 60
_backoff_until = 0.0


def _get_json(url: str) -> dict[str, Any] | None:
    global _backoff_until
    if time.monotonic() < _backoff_until:
        return None
    try:
        resp = cffi_requests.get(url, impersonate="chrome124", timeout=_TIMEOUT)
    except Exception as exc:  # noqa: BLE001 - network
        _backoff_until = time.monotonic() + _BACKOFF_S
        log.warning("ESPN request error, pausing %ds: %s", _BACKOFF_S, str(exc)[:80])
        return None
    if resp.status_code == 200:
        return resp.json()
    if resp.status_code in (403, 429, 503):
        _backoff_until = time.monotonic() + _BACKOFF_S
        log.warning("ESPN HTTP %d, pausing %ds (%s)", resp.status_code, _BACKOFF_S, url)
    else:
        log.warning("ESPN HTTP %d for %s", resp.status_code, url)
    return None


def _score_and_clock(comp: dict[str, Any]) -> tuple[list[int] | None, str, str]:
    """([home, away] or None, clock text like "41'" / "HT" / "FT", state pre|in|post)."""
    home, away = _sides(comp.get("competitors") or [])
    try:
        score = [int(home.get("score")), int(away.get("score"))]
    except (TypeError, ValueError):
        score = None
    stype = (comp.get("status") or {}).get("type") or {}
    return score, stype.get("detail") or stype.get("shortDetail") or "", stype.get("state") or ""


def _scoreboard(slug: str) -> dict[str, Any] | None:
    now = time.monotonic()
    cached = _scoreboards.get(slug)
    if cached and now - cached[0] < _SCOREBOARD_TTL_S:
        return cached[1]
    payload = _get_json(f"{_BASE}/{slug}/scoreboard")
    if payload is not None:
        _scoreboards[slug] = (now, payload)
    return payload


def _sides(competitors: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    home = next((c for c in competitors if c.get("homeAway") == "home"), {})
    away = next((c for c in competitors if c.get("homeAway") == "away"), {})
    return home, away


def find_event(home: str, away: str, scoreboard: dict[str, Any]) -> dict[str, Any] | None:
    """The scoreboard event whose sides normalise to (home, away), else None.

    Both name spaces go through ``normalize_team`` ("AS Roma" and "Roma" ->
    "Roma", "Brighton & Hove Albion" and "Brighton and Hove Albion" ->
    "Brighton"). Sides must match in order: a reversed fixture is a different
    match.
    """
    want = (normalize_team(home), normalize_team(away))
    for event in scoreboard.get("events") or []:
        comp = (event.get("competitions") or [{}])[0]
        h, a = _sides(comp.get("competitors") or [])
        got = (
            normalize_team((h.get("team") or {}).get("displayName") or ""),
            normalize_team((a.get("team") or {}).get("displayName") or ""),
        )
        if got == want:
            return event
    return None


def _minute(clock: dict[str, Any] | None) -> tuple[int, int]:
    """("45'+5'") -> (45, 5); ("47'") -> (47, 0); falls back to clock.value."""
    clock = clock or {}
    m = _CLOCK_RE.search(clock.get("displayValue") or "")
    if m:
        return int(m.group(1)), int(m.group(2) or 0)
    try:
        return int(float(clock.get("value") or 0) // 60), 0
    except (TypeError, ValueError):
        return 0, 0


def _participant(event: dict[str, Any], idx: int) -> str:
    parts = event.get("participants") or []
    if idx < len(parts):
        return ((parts[idx].get("athlete") or {}).get("displayName")) or ""
    return ""


def _parse_key_event(ke: dict[str, Any], home_id: str) -> dict[str, Any] | None:
    slug = ((ke.get("type") or {}).get("type") or "").lower()
    minute, added = _minute(ke.get("clock"))
    team_id = str((ke.get("team") or {}).get("id") or "")
    base: dict[str, Any] = {
        "type": "",
        "minute": minute,
        "added_time": added,
        "is_home": (team_id == str(home_id)) if team_id else None,
    }
    if slug.startswith("goal") or "own-goal" in slug or slug == "penalty---scored":
        base["type"] = "goal"
        base["player"] = _participant(ke, 0)
        base["assist"] = _participant(ke, 1)
        base["goal_type"] = (
            "ownGoal" if "own" in slug else "penalty" if "penalty" in slug else "regular"
        )
        if base["goal_type"] == "ownGoal" and base["is_home"] is not None:
            # ESPN credits the beneficiary; we store the scorer's side (see docstring)
            base["is_home"] = not base["is_home"]
        return base
    if "card" in slug:
        base["type"] = "card"
        base["player"] = _participant(ke, 0)
        if "yellow" in slug and "red" in slug:
            base["card_type"] = "yellowRed"
        elif "red" in slug:
            base["card_type"] = "red"
        else:
            base["card_type"] = "yellow"
        return base
    if slug == "substitution":
        base["type"] = "substitution"
        base["player_in"] = _participant(ke, 0)
        base["player_out"] = _participant(ke, 1)
        return base
    if slug == "halftime":
        base.update(type="period", period="ht", is_home=None)
        return base
    if slug in ("end-regular-time", "full-time"):
        base.update(type="period", period="ft", is_home=None)
        return base
    return None  # kickoff, delays, 2nd-half start: not events the card shows


def parse_key_events(key_events: list[dict[str, Any]], home_id: str) -> list[dict[str, Any]]:
    """live_events for one match, **newest first** (the Sofascore convention)."""
    parsed = [e for ke in key_events if (e := _parse_key_event(ke, home_id))]
    parsed.reverse()
    return parsed


def score_from_events(events: list[dict[str, Any]]) -> list[int]:
    """[home, away] counted from goal events; an own goal credits the opponent."""
    home = away = 0
    for e in events:
        if e.get("type") != "goal" or e.get("is_home") is None:
            continue
        credited_home = bool(e["is_home"]) != (e.get("goal_type") == "ownGoal")
        if credited_home:
            home += 1
        else:
            away += 1
    return [home, away]


def _num(value: Any) -> Any:
    if isinstance(value, int | float):
        return value
    s = str(value or "").strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return None


def parse_boxscore(boxscore: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """live_stats for one match from ESPN's team boxscore; absent stats are omitted."""
    home, away = _sides(boxscore.get("teams") or [])
    out: dict[str, dict[str, Any]] = {}
    per_side = {
        "home": {s.get("name"): _num(s.get("displayValue")) for s in home.get("statistics") or []},
        "away": {s.get("name"): _num(s.get("displayValue")) for s in away.get("statistics") or []},
    }
    for espn_key, our_key in _STAT_KEYS.items():
        h, a = per_side["home"].get(espn_key), per_side["away"].get(espn_key)
        if h is None and a is None:
            continue
        out[our_key] = {"home": h, "away": a}
    return out


def fetch_live_data_for_match(home: str, away: str) -> dict[str, Any] | None:
    """Events + team stats for one fixture, or None when ESPN has no such match.

    The league is not needed: every configured league's scoreboard (cached 60s)
    is searched by normalised team names.
    """
    for slug in LEAGUE_SLUGS.values():
        board = _scoreboard(slug)
        if not board:
            continue
        event = find_event(home, away, board)
        if not event:
            continue
        summary = _get_json(f"{_BASE}/{slug}/summary?event={event.get('id')}")
        if not summary:
            return None
        comp = ((summary.get("header") or {}).get("competitions") or [{}])[0]
        home_side, _ = _sides(comp.get("competitors") or [])
        home_id = str((home_side.get("team") or {}).get("id") or "")
        score, clock, state = _score_and_clock(comp)
        events = parse_key_events(summary.get("keyEvents") or [], home_id)
        derived = score_from_events(events)
        if score is None:
            score = derived
        elif derived != score:
            merged = [max(score[0], derived[0]), max(score[1], derived[1])]
            if merged != score:
                log.info("ESPN header score %s trails goal events %s for %s vs %s — using %s",
                         score, derived, home, away, merged)
            score = merged
        return {
            "espn_id": event.get("id"),
            "events": events,
            "statistics": parse_boxscore(summary.get("boxscore") or {}),
            "player_stats": {},
            "fetched": {"events": True, "statistics": True, "player_stats": False},
            "score": score,
            "clock": clock,
            "state": state,
            "source": "espn",
            "fetched_at": datetime.now(UTC).isoformat(),
        }
    return None
