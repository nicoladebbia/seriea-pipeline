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
``live_monitor`` and the /live template need no branching. ``player_stats``
comes from the summary ``rosters`` (shots, on target, goals, assists, fouls
committed/suffered, offsides, cards, saves, goals conceded, own goals — no
minutes, passes, tackles, duels or rating); ``minutes_played`` is derived from
the substitution events and the clock.

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
import unicodedata
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

# ESPN roster stat name -> our live_player_stats key (Sofascore's names, so the
# prop tracker and the substitution tracker read both feeds the same way).
# ESPN has no minutes, passes, tackles, duels or rating per player.
_PLAYER_STAT_KEYS = {
    "totalShots": "shots",
    "shotsOnTarget": "shots_on_target",
    "totalGoals": "goals",
    "goalAssists": "assists",
    "foulsCommitted": "fouls_committed",
    "foulsSuffered": "fouls_drawn",
    "offsides": "offsides",
    "yellowCards": "yellow_cards",
    "redCards": "red_cards",
    "saves": "saves",
    "goalsConceded": "goals_conceded",
    "ownGoals": "own_goals",
}

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


def _scoreboard(slug: str, date: str | None = None) -> dict[str, Any] | None:
    """Today's scoreboard, or one day's (``date`` = YYYYMMDD) for a backfill."""
    now = time.monotonic()
    key = f"{slug}:{date or ''}"
    cached = _scoreboards.get(key)
    if cached and now - cached[0] < _SCOREBOARD_TTL_S:
        return cached[1]
    url = f"{_BASE}/{slug}/scoreboard" + (f"?dates={date}" if date else "")
    payload = _get_json(url)
    if payload is not None:
        _scoreboards[key] = (now, payload)
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


def referee_from_summary(summary: dict[str, Any] | None) -> str | None:
    """The match referee's full name from a summary payload, else None.

    Specimen 2026-09-05 (Fiorentina–Torino, Inter–Napoli): the scoreboard
    event carries no officials; ``summary.gameInfo.officials`` lists
    ``{"fullName": "Davide Massa", "position": {"name": "Referee"}}`` once the
    match is ``post`` and is EMPTY pre-kickoff. Full names, the same space as
    nine seasons of ``matches.parquet`` ("Davide Massa", not "D Massa").
    """
    for official in ((summary or {}).get("gameInfo") or {}).get("officials") or []:
        role = ((official.get("position") or {}).get("name") or "").strip().lower()
        if role == "referee":
            name = (official.get("fullName") or official.get("displayName") or "").strip()
            return name or None
    return None


def match_referee(league: str, date: str, home: str, away: str) -> str | None:
    """Referee of a PLAYED match from ESPN (scoreboard for the day -> event
    -> summary). None when the league has no slug, the day/event is not on
    ESPN, or the match has not been played yet. Fills ground truth: the
    Sofascore fixture list names no referee (0 of 21 finished 2026-27
    fixtures) and worldfootball publishes the season late."""
    return referee_from_summary(_summary_for(league, date, home, away))


def _summary_for(league: str, date: str, home: str, away: str) -> dict[str, Any] | None:
    """Scoreboard for the day -> event -> summary payload, else None."""
    slug = LEAGUE_SLUGS.get(league)
    if not slug or not date:
        return None
    board = _scoreboard(slug, str(date)[:10].replace("-", ""))
    if not board:
        return None
    event = find_event(home, away, board)
    if not event:
        return None
    return _get_json(f"{_BASE}/{slug}/summary?event={event.get('id')}")


def first_half_from_summary(summary: dict[str, Any] | None) -> tuple[int, int] | None:
    """(home, away) goals in the first half of a FINISHED match, else None.

    Goals come from ``keyEvents`` through ``parse_key_events`` (own goals
    already credited to the beneficiary's opponent's side there, so the
    scorer's side is flipped back here); a goal at minute <= 45 — "45'+3'"
    parses to (45, 3) — is first half. A match that is not ``post`` returns
    None: a partial timeline would grade a first-half market on half a half.
    """
    if not summary:
        return None
    comp = ((summary.get("header") or {}).get("competitions") or [{}])[0]
    if (((comp.get("status") or {}).get("type") or {}).get("state") or "").lower() != "post":
        return None
    home_side, _ = _sides(comp.get("competitors") or [])
    home_id = str((home_side.get("team") or home_side).get("id") or "")
    if not home_id:
        return None
    h = a = 0
    for ev in parse_key_events(summary.get("keyEvents") or [], home_id):
        if ev.get("type") != "goal" or ev.get("is_home") is None or int(ev.get("minute") or 0) > 45:
            continue
        credited_home = ev["is_home"] if ev.get("goal_type") != "ownGoal" else not ev["is_home"]
        if credited_home:
            h += 1
        else:
            a += 1
    return h, a


def first_half_score(league: str, date: str, home: str, away: str) -> tuple[int, int] | None:
    """First-half score of a PLAYED match from ESPN, else None. The grader's
    fallback when the Sofascore incidents feed (goal_timeline.parquet) has not
    ingested the match — it stopped at 2026-08-24 under the API challenge."""
    return first_half_from_summary(_summary_for(league, date, home, away))


# ---------------------------------------------------------------------------
# Post-match record: the Sofascore-shaped incidents + team stats the matchday
# updater falls back to while the Sofascore API is challenged (2026-09-05: nine
# of 21 finished Serie A matches had no incidents on disk, six ground-truth rows
# no team stats). Only a summary whose state is ``post`` is used, and it is
# cached per process: a finished match's summary never changes.
# ---------------------------------------------------------------------------

_POST_SUMMARIES: dict[tuple[str, str, str, str], dict[str, Any]] = {}


def _is_post(summary: dict[str, Any] | None) -> bool:
    if not summary:
        return False
    comp = ((summary.get("header") or {}).get("competitions") or [{}])[0]
    return (((comp.get("status") or {}).get("type") or {}).get("state") or "").lower() == "post"


def post_match_summary(league: str, date: str, home: str, away: str) -> dict[str, Any] | None:
    """The ESPN summary of a PLAYED match (state ``post``), else None."""
    key = (league, str(date)[:10], home, away)
    hit = _POST_SUMMARIES.get(key)
    if hit is not None:
        return hit
    summary = _summary_for(*key)
    if not _is_post(summary):
        return None
    _POST_SUMMARIES[key] = summary  # type: ignore[assignment]
    return summary


def _home_id(summary: dict[str, Any]) -> str:
    comp = ((summary.get("header") or {}).get("competitions") or [{}])[0]
    home_side, _ = _sides(comp.get("competitors") or [])
    return str((home_side.get("team") or home_side).get("id") or "")


def _pseudo_player_id(name: str) -> str:
    """ESPN athlete ids are not Sofascore ids. A name-derived key keeps the
    bench-goal join (goal player_id == substitution player_in_id) meaningful
    for ESPN rows and never collides with a numeric Sofascore id — and never
    with another ESPN row through a shared blank."""
    return f"espn:{_fold(name)}" if name else ""


def incident_rows_from_summary(summary: dict[str, Any] | None, match_id: int) -> list[dict[str, Any]]:
    """``match_incidents.parquet`` rows (the Sofascore schema, ``source``
    "espn") from a post-match summary: goals, cards, substitutions. VAR reviews
    and missed penalties are not in ESPN's feed, so the match is NOT
    ``var_checked`` and the rare-event rates leave it out of their denominator.

    Side convention, verified against the parquet (2026-09-05): a Sofascore
    goal row's ``is_home`` is the side CREDITED with the goal, so an own goal by
    a home player is ``is_home=False``. ``parse_key_events`` stores the scorer's
    side for the live card; it is flipped back here.
    """
    if not _is_post(summary):
        return []
    home_id = _home_id(summary)  # type: ignore[arg-type]
    if not home_id:
        return []
    rows: list[dict[str, Any]] = []
    for ev in parse_key_events(summary.get("keyEvents") or [], home_id):  # type: ignore[union-attr]
        kind = ev.get("type")
        if kind not in ("goal", "card", "substitution"):
            continue
        is_home = ev.get("is_home")
        row: dict[str, Any] = {
            "match_id": int(match_id), "incident_type": kind, "incident_class": "",
            "minute": int(ev.get("minute") or 0), "added_time": int(ev.get("added_time") or 0),
            "player_name": "", "player_id": "", "is_home": is_home,
            "player_in_name": None, "player_in_id": None, "card_type": None,
            "goal_type": None, "assist_player": None, "confirmed": None, "source": "espn",
        }
        if kind == "goal":
            goal_type = ev.get("goal_type") or "regular"
            if goal_type == "ownGoal" and is_home is not None:
                row["is_home"] = not is_home
            row.update(incident_class=goal_type, goal_type=goal_type,
                       player_name=ev.get("player") or "",
                       player_id=_pseudo_player_id(ev.get("player") or ""),
                       assist_player=ev.get("assist") or "")
        elif kind == "card":
            card_type = ev.get("card_type") or "yellow"
            row.update(incident_class=card_type, card_type=card_type,
                       player_name=ev.get("player") or "",
                       player_id=_pseudo_player_id(ev.get("player") or ""))
        else:
            row.update(incident_class="regular",
                       player_name=ev.get("player_out") or "",
                       player_id=_pseudo_player_id(ev.get("player_out") or ""),
                       player_in_name=ev.get("player_in") or "",
                       player_in_id=_pseudo_player_id(ev.get("player_in") or ""))
        rows.append(row)
    return rows


def half_time_from_summary(summary: dict[str, Any] | None) -> tuple[int, int] | None:
    """(home, away) at half-time: the competitors' first ``linescores`` entry,
    else counted from the goal events; None unless the match is ``post``."""
    if not _is_post(summary):
        return None
    comp = ((summary.get("header") or {}).get("competitions") or [{}])[0]  # type: ignore[union-attr]
    home_side, away_side = _sides(comp.get("competitors") or [])
    try:
        parts = []
        for side in (home_side, away_side):
            first = (side.get("linescores") or [])[0]
            parts.append(int(float(first.get("displayValue") if first.get("displayValue") not in (None, "") else first.get("value"))))
        return parts[0], parts[1]
    except (IndexError, TypeError, ValueError):
        return first_half_from_summary(summary)


def boxscore_by_side(boxscore: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """{"home": {espn stat name: number}, "away": {...}} — every stat ESPN
    lists, by its own name (``possessionPct``, ``totalShots``, ...)."""
    home, away = _sides(boxscore.get("teams") or [])
    return {
        "home": {s.get("name"): _num(s.get("displayValue")) for s in home.get("statistics") or []},
        "away": {s.get("name"): _num(s.get("displayValue")) for s in away.get("statistics") or []},
    }


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


def _fold(name: str) -> str:
    """Accent-folded lowercase: the roster says "Matìas Soulè", the event feed "Matias Soulè"."""
    return "".join(c for c in unicodedata.normalize("NFKD", name or "") if not unicodedata.combining(c)).lower().strip()


def _parse_roster_entry(entry: dict[str, Any]) -> dict[str, Any]:
    athlete = entry.get("athlete") or {}
    out: dict[str, Any] = {
        "name": athlete.get("displayName") or "",
        "short_name": athlete.get("shortName") or "",
        "position": (entry.get("position") or {}).get("abbreviation") or "",
        "jersey_number": str(entry.get("jersey") or ""),
        "substitute": not bool(entry.get("starter")),
        "subbed_in": bool(entry.get("subbedIn")),
        "subbed_out": bool(entry.get("subbedOut")),
    }
    # A stat ESPN omits is OMITTED, not zero-filled (same contract as Sofascore).
    for item in entry.get("stats") or []:
        key = _PLAYER_STAT_KEYS.get(item.get("name") or "")
        if key:
            val = _num(item.get("displayValue"))
            if val is not None:
                out[key] = val
    return out


def _minutes_played(player: dict[str, Any], events: list[dict[str, Any]], clock_minute: int) -> int | None:
    """Minutes on the pitch derived from the substitution events and the clock.

    ESPN carries no minutes per player. A starter has played the clock; a
    player subbed off played until his substitution; a sub has played since
    his; a red card stops the clock at the card. Stoppage time is ignored;
    None when the clock is unknown or the substitution event cannot be found.
    """
    if clock_minute is None:
        return None
    me = _fold(player["name"])
    # A sent-off player's clock stops at the card.
    end = clock_minute
    for e in events:
        if e.get("type") == "card" and e.get("card_type") == "red" and _fold(e.get("player", "")) == me:
            end = min(end, int(e.get("minute") or clock_minute))
    if player["subbed_in"]:
        for e in events:
            if e.get("type") == "substitution" and _fold(e.get("player_in", "")) == me:
                return max(0, end - int(e.get("minute") or 0))
        return None
    if player["substitute"]:
        return 0
    if player["subbed_out"]:
        for e in events:
            if e.get("type") == "substitution" and _fold(e.get("player_out", "")) == me:
                return min(end, int(e.get("minute") or 0))
        return None
    return end


def parse_rosters(rosters: list[dict[str, Any]], events: list[dict[str, Any]] | None = None,
                  clock_minute: int | None = None) -> dict[str, list[dict[str, Any]]]:
    """live_player_stats for one match: {"home": [...], "away": [...]}."""
    out: dict[str, list[dict[str, Any]]] = {"home": [], "away": []}
    for team in rosters or []:
        side = team.get("homeAway")
        if side not in out:
            continue
        for entry in team.get("roster") or []:
            player = _parse_roster_entry(entry)
            if not player["name"]:
                continue
            minutes = _minutes_played(player, events or [], clock_minute)
            if minutes is not None:
                player["minutes_played"] = minutes
            out[side].append(player)
    return out


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
    out: dict[str, dict[str, Any]] = {}
    per_side = boxscore_by_side(boxscore)
    for espn_key, our_key in _STAT_KEYS.items():
        h, a = per_side["home"].get(espn_key), per_side["away"].get(espn_key)
        if h is None and a is None:
            continue
        out[our_key] = {"home": h, "away": a}
    return out


def fetch_live_data_for_match(home: str, away: str, date: str | None = None) -> dict[str, Any] | None:
    """Events + team stats for one fixture, or None when ESPN has no such match.

    The league is not needed: every configured league's scoreboard (cached 60s)
    is searched by normalised team names.
    """
    for slug in LEAGUE_SLUGS.values():
        board = _scoreboard(slug, date)
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
        clock_minute, _ = _minute({"displayValue": clock})
        if state == "post":
            clock_minute = 90
        elif not _CLOCK_RE.search(clock or ""):
            clock_minute = 45 if clock == "HT" else None
        rosters = summary.get("rosters") or []
        player_stats = parse_rosters(rosters, events, clock_minute) if rosters else {}
        has_players = bool(player_stats.get("home") or player_stats.get("away"))
        return {
            "espn_id": event.get("id"),
            "events": events,
            "statistics": parse_boxscore(summary.get("boxscore") or {}),
            "player_stats": player_stats if has_players else {},
            "fetched": {"events": True, "statistics": True, "player_stats": has_players},
            "score": score,
            "clock": clock,
            "state": state,
            "source": "espn",
            "fetched_at": datetime.now(UTC).isoformat(),
        }
    return None
