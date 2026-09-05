"""Rich live match data from Sofascore — events, team stats, player stats.

Called once per polling cycle by ``scripts/data/live_monitor.py:1140``, which
writes the output onto each match as ``live_events`` / ``live_stats`` /
``live_player_stats`` and feeds ``live_events`` to
``scripts.data.live_reconciliation``. The caller wraps the import in a bare
``except``, so this module's absence was **silent**.

Rebuilt 2026-07-16. Every mapping below was verified against **both** ends:

* **Output oracle** — the 123 persisted blocks in ``data/live/*.json``, written
  by the original (lost) module: 779 events, 50 matches with team stats, 50 with
  player stats.
* **Input specimens** — real Sofascore responses for the *same* Serie A matches
  the oracle covers (``tests/fixtures/sofascore/``, event 13981681 and 13981716).

Verified by replay, not assumed — the notable findings:

* ``assists`` maps from ``goalAssist``, **not** ``assists``. There is no
  ``assists`` key; guessing it would have silently written 0 assists for every
  player. Caught on Laurienté (oracle ``assists=2``, API ``goalAssist=2``).
* ``isHome`` marks the **scorer's** side, so an own goal credits the opponent.
  ``live_reconciliation`` depends on this.
* ``injuryTime`` incidents carry no ``isHome`` — ``is_home`` is None for all 68
  in the oracle.
* ``addedTime`` is absent on ordinary incidents and is the sentinel ``999`` on
  ``period`` incidents. The oracle never contains 999, and never a value above 6.
* ``period`` and ``inGamePenalty`` incidents are **dropped** — no oracle event
  has either type.

⚠️ **Live ceiling.** Verified against real Sofascore responses and replayed
against the oracle, but **not** end-to-end against a live Serie A match: Serie A
is off-season until mid-August, so no fixture exists to poll. The specimens are
finished matches, which serve the same payloads.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any

from scraper import sofascore_events as _ss
from scraper.sofascore_events import _BASE_URL, _get_json, _jitter_delay
from scraper.sofascore_lineups import get_sofascore_match_ids
from scripts.data import live_espn

log = logging.getLogger(__name__)

# Sofascore 403 breaker. One blocked cycle costs ~2 min of backoff (4 attempts
# x 3 endpoints), longer than the poll interval; while it is tripped the cycle
# goes straight to the ESPN fallback and Sofascore is retried after the cooldown.
_BLOCK_COOLDOWN_S = 600
_sofascore_blocked_until = 0.0
# match key -> why this cycle produced nothing (read by live_monitor for the card)
LAST_ERRORS: dict[str, str] = {}

# Sofascore incidentType -> our event type. Anything absent is dropped:
# `period` and `inGamePenalty` both appear in real responses and neither has a
# single instance across the oracle's 779 events.
_INCIDENT_TYPES = {
    "goal": "goal",
    "card": "card",
    "substitution": "substitution",
    "injuryTime": "injury_time",
    "varDecision": "var",
}

# Sofascore statisticsItem key -> our live_stats key. All 12 reproduce the
# oracle exactly for event 13981681 (10 identical; xg and accurate_passes had
# grown by full time, the oracle being a late in-play snapshot).
_STAT_KEYS = {
    "ballPossession": "possession",
    "expectedGoals": "xg",
    "cornerKicks": "corners",
    "fouls": "fouls",
    "offsides": "offsides",
    "yellowCards": "yellow_cards",
    "redCards": "red_cards",
    "hitWoodwork": "hit_woodwork",
    "goalkeeperSaves": "saves",
    "accuratePasses": "accurate_passes",
    "throwIns": "throw_ins",
    "goalKicks": "goal_kicks",
}

# Sofascore player statistics key -> our player_stats key. Covers every one of
# the 24 stat keys in the oracle's 2,339 player records; verified against the
# oracle's own records for event 13981681.
_PLAYER_STAT_KEYS = {
    "goalAssist": "assists",  # NOT "assists" — that key does not exist.
    "totalShots": "shots",
    "rating": "rating",
    "minutesPlayed": "minutes_played",
    "accuratePass": "accurate_passes",
    "totalPass": "total_passes",
    "aerialWon": "aerials_won",
    "totalClearance": "clearances",
    "saves": "saves",
    "touches": "touches",
    "duelWon": "duels_won",
    "duelLost": "duels_lost",
    "totalTackle": "tackles",
    "fouls": "fouls_committed",
    "expectedGoals": "xg",
    "wasFouled": "fouls_drawn",
    "totalCross": "crosses",
    "keyPass": "key_passes",
    "interceptionWon": "interceptions",
    "onTargetScoringAttempt": "shots_on_target",
    "bigChanceCreated": "big_chances_created",
    "goals": "goals",
    "bigChanceMissed": "big_chances_missed",
    "penaltyWon": "penalties_won",
}

# `period` incidents carry addedTime=999 as a "not applicable" sentinel. Those
# are dropped, so this never fires in practice — but a 999 must never reach
# added_time if Sofascore tags another type with it.
_ADDED_TIME_SENTINEL = 999


def _name(node: Any) -> str:
    return (node or {}).get("name") or ""


def _parse_incident(inc: dict[str, Any]) -> dict[str, Any] | None:
    """One Sofascore incident -> one live_events record, or None if dropped."""
    etype = _INCIDENT_TYPES.get(inc.get("incidentType") or "")
    if etype is None:
        return None

    added = inc.get("addedTime") or 0
    if added == _ADDED_TIME_SENTINEL:
        added = 0

    event: dict[str, Any] = {
        "type": etype,
        "minute": inc.get("time") or 0,
        "added_time": added,
        # injuryTime has no isHome; the oracle stores None for all 68 of them.
        "is_home": inc.get("isHome"),
    }

    inc_class = inc.get("incidentClass") or ""
    if etype == "goal":
        event["player"] = _name(inc.get("player"))
        event["assist"] = _name(inc.get("assist1"))
        # regular | penalty | ownGoal — ownGoal credits the OPPOSING side, which
        # live_reconciliation relies on when counting goals per team.
        event["goal_type"] = inc_class
    elif etype == "card":
        event["player"] = _name(inc.get("player"))
        event["card_type"] = inc_class  # yellow | red | yellowRed
    elif etype == "substitution":
        event["player_out"] = _name(inc.get("playerOut"))
        event["player_in"] = _name(inc.get("playerIn"))
    elif etype == "injury_time":
        event["length"] = inc.get("length") or 0
    elif etype == "var":
        event["decision"] = inc_class  # penaltyAwarded | goalAwarded | ...

    return event


def parse_incidents(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """live_events for one match, **newest first**.

    Sofascore returns incidents newest-first and the oracle preserves that order
    — verified: the oracle's first event for 13981681 is the 89' substitution,
    not the 17' card. Do not sort.
    """
    return [e for inc in (payload.get("incidents") or []) if (e := _parse_incident(inc))]


def parse_statistics(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """live_stats for one match, from the ALL period."""
    out: dict[str, dict[str, Any]] = {}
    for period in payload.get("statistics") or []:
        if period.get("period") != "ALL":
            continue
        for group in period.get("groups") or []:
            for item in group.get("statisticsItems") or []:
                key = _STAT_KEYS.get(item.get("key") or "")
                if key and key not in out:
                    out[key] = {"home": item.get("homeValue"), "away": item.get("awayValue")}
    return out


def _parse_player(entry: dict[str, Any]) -> dict[str, Any]:
    player = entry.get("player") or {}
    stats = entry.get("statistics") or {}
    out: dict[str, Any] = {
        "name": player.get("name") or "",
        "short_name": player.get("shortName") or "",
        "position": entry.get("position") or player.get("position") or "",
        "jersey_number": entry.get("jerseyNumber") or player.get("jerseyNumber") or "",
        "substitute": bool(entry.get("substitute")),
    }
    # A stat the API omits is OMITTED, not zero-filled. The oracle is sparse —
    # only 6 keys appear in all 2,339 records, and e.g. `saves` in just 72 —
    # so defaulting to 0 would assert an unrecorded action really was zero.
    for api_key, our_key in _PLAYER_STAT_KEYS.items():
        if api_key in stats:
            out[our_key] = stats[api_key]
    return out


def parse_lineups(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """live_player_stats for one match: {"home": [...], "away": [...]}."""
    return {
        side: [_parse_player(p) for p in (payload.get(side) or {}).get("players") or []]
        for side in ("home", "away")
    }


def fetch_live_data_for_match(sofascore_id: int) -> dict[str, Any]:
    """Events, statistics and player stats for one Sofascore event id.

    Each endpoint is independent — a failure on one leaves the others intact
    rather than losing the whole match.
    """
    data: dict[str, Any] = {
        "sofascore_id": sofascore_id,
        "events": [],
        "statistics": {},
        "player_stats": {},
        # per-field: did the endpoint ANSWER? live_monitor overwrites a field
        # only when it did, so a 403 on /statistics cannot blank good stats.
        "fetched": {"events": False, "statistics": False, "player_stats": False},
        "source": "sofascore",
        "fetched_at": datetime.now(UTC).isoformat(),
    }

    incidents = _get_json(f"{_BASE_URL}/event/{sofascore_id}/incidents")
    if incidents is not None:
        data["events"] = parse_incidents(incidents)
        data["fetched"]["events"] = True

    _jitter_delay()
    stats = _get_json(f"{_BASE_URL}/event/{sofascore_id}/statistics")
    if stats is not None:
        data["statistics"] = parse_statistics(stats)
        data["fetched"]["statistics"] = True

    _jitter_delay()
    lineups = _get_json(f"{_BASE_URL}/event/{sofascore_id}/lineups")
    if lineups is not None:
        data["player_stats"] = parse_lineups(lineups)
        data["fetched"]["player_stats"] = True

    return data


def _answered(data: dict[str, Any] | None) -> bool:
    if not data:
        return False
    flags = data.get("fetched")
    return any(flags.values()) if flags else True


def _espn_fallback(match_key: str) -> dict[str, Any] | None:
    parts = match_key.split(" vs ", 1)
    if len(parts) != 2:
        return None
    try:
        return live_espn.fetch_live_data_for_match(parts[0].strip(), parts[1].strip())
    except Exception as exc:  # noqa: BLE001 - a fallback must not sink the cycle
        log.warning("ESPN fallback failed for %s: %s", match_key, exc)
        return None


def fetch_live_data_for_matches(match_keys: list[str]) -> dict[str, dict[str, Any]]:
    """Live data keyed by match key: Sofascore first, ESPN when Sofascore answers nothing.

    Contract at live_monitor.py (Sofascore block): a match is present in the
    result only when SOME source answered this cycle; an omission preserves the
    match's last good data instead of blanking it, and ``LAST_ERRORS`` says why.
    """
    global _sofascore_blocked_until
    LAST_ERRORS.clear()
    if not match_keys:
        return {}

    now = time.monotonic()
    blocked = now < _sofascore_blocked_until
    id_map: dict[str, int] = {}
    if not blocked:
        # get_sofascore_match_ids takes {match_key: {...}} and falls back to
        # splitting "Home vs Away" when the record carries no team fields.
        id_map = get_sofascore_match_ids({k: {} for k in match_keys}) or {}
        if not id_map:
            log.warning("Sofascore: resolved 0 of %d live match(es)", len(match_keys))
    else:
        log.info("Sofascore: 403 cooldown for %ds more, using ESPN",
                 int(_sofascore_blocked_until - now))

    out: dict[str, dict[str, Any]] = {}
    for match_key in match_keys:
        data: dict[str, Any] | None = None
        why = "Sofascore skipped (403 cooldown)" if blocked else "Sofascore: no id"
        sofascore_id = id_map.get(match_key)
        if sofascore_id is not None:
            try:
                data = fetch_live_data_for_match(sofascore_id)
            except Exception as exc:  # noqa: BLE001 - one bad match must not sink the cycle
                log.warning("Sofascore fetch failed for %s (%s): %s", match_key, sofascore_id, exc)
                why = f"Sofascore error: {str(exc)[:60]}"
            if data is not None and not _answered(data):
                status = getattr(_ss, "_LAST_FAILURE_STATUS", None)
                why = f"Sofascore blocked (HTTP {status})" if status else "Sofascore unreachable"
                if status == 403 and not blocked:
                    _sofascore_blocked_until = time.monotonic() + _BLOCK_COOLDOWN_S
                    log.warning("Sofascore API 403 on every endpoint — cooling down %ds, "
                                "serving live stats/events from ESPN", _BLOCK_COOLDOWN_S)
                data = None
        if data is None:
            espn = _espn_fallback(match_key)
            if espn:
                if sofascore_id is not None:
                    espn["sofascore_id"] = sofascore_id
                data = espn
            else:
                why += ", ESPN: no match"
        if data is None:
            LAST_ERRORS[match_key] = why
            log.warning("No live data for %s this cycle (%s) — keeping last good", match_key, why)
            continue
        out[match_key] = data
    return out
