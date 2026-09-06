"""FotMob — second source for finished-match records (2026-09-06).

Sofascore's API answered ``403 challenge`` from 2026-09-05 and a blanket
``403 Forbidden`` from this IP a day later; under it a finished match lands
score-only and the per-player stats that grade player props, the team stats
and the shot map never arrive. FotMob's site JSON carries all three for both
leagues, with no key:

* ``/api/data/matches?date=YYYYMMDD`` — every match of the day, leagues 55
  (Serie A) and 47 (Premier League), FotMob match ids.
* ``/api/data/matchDetails?matchId=<id>`` — ``content.playerStats`` (minutes,
  shots, SoT, fouls, tackles, duels, passes, touches, xG/xA, keeper stats),
  ``content.shotmap.shots`` (x/y in metres on a 105×68 pitch, xG, xGOT,
  situation, body part, blocked/on-target flags), ``content.stats.Periods``
  (All / FirstHalf / SecondHalf team stats) and ``content.lineup``.

Specimen-verified on Fiorentina–Torino and Man City–Coventry, 2026-09-05
(``tests/fixtures/fotmob_match_*.json`` are trimmed copies). Both endpoints
answered a plain chrome-impersonated GET; in 2024 FotMob briefly required a
signed ``x-mas`` header, so every fetch goes through one breaker and a 403 /
429 / connection failure parks the source for BREAKER_MINUTES — retry storms
are how the Sofascore IP deny was earned (310 logged 403s in two days).

Parsers are PURE and emit rows in the vocabulary of the parquets the pipeline
already reads (``player_match_stats``, ``match_team_stats``,
``all_shots_with_xg``, ``shotmap_stats``), stamped ``source="fotmob"``. A stat
FotMob does not carry is None, never 0 — a false zero is a lie the rolling
windows would average. Player ids are foreign: the parser hands
``(team, name, fotmob_id)`` to a resolver (the chain resolves against the
parquet's own history) and falls back to ``-fotmob_id`` (negative = not a
Sofascore id, can never collide with one).
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.settings import DATA_DIR
from config.team_names import normalize_team

log = logging.getLogger(__name__)

BASE = "https://www.fotmob.com"
LEAGUE_IDS: dict[int, str] = {55: "serie_a", 47: "premier_league"}
BREAKER_MINUTES = 10
TIMEOUT = 25
IMPERSONATE = "chrome124"

# FotMob spellings that ``normalize_team`` does not know. Anything else that
# fails to normalise is REPORTED by the chain, never guessed.
FOTMOB_TEAM_ALIASES: dict[str, str] = {
    "Nottm Forest": "Nottingham Forest",
    "Man Utd": "Man United",
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "AC Milan": "Milan",
    "Hellas Verona": "Verona",
    "Coventry City": "Coventry",
    "Leeds United": "Leeds",
    "Hull City": "Hull",
    "Newcastle United": "Newcastle",
    "Sunderland AFC": "Sunderland",
    "Tottenham Hotspur": "Tottenham",
    "Brighton & Hove Albion": "Brighton",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Sheffield United": "Sheffield Utd",
    "Inter Milan": "Inter",
    "Internazionale": "Inter",
}

# Raw matchDetails payloads. ``data/external/fotmob/matches/`` next door is a
# LEGACY dump of 256-byte ``{"basic": ...}`` stubs from a scraper no longer in
# the tree (2017-26, see DATA_CATALOG) — a different shape, so it is not reused.
RAW_DIR = DATA_DIR / "external" / "fotmob" / "match_details"

_blocked_until: float = 0.0
_last_error: str = ""

Resolver = Callable[[str, str, int], int | None]


def _fold(name: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", name or "")
                   if not unicodedata.combining(c)).lower().strip()


def canonical_team(name: str) -> str:
    """Pipeline team name for a FotMob team name (alias map, then normalize_team)."""
    return normalize_team(FOTMOB_TEAM_ALIASES.get(name or "", name or ""))


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------

def blocked() -> str | None:
    """The reason the source is parked, or None when it may be called."""
    if time.monotonic() < _blocked_until:
        return _last_error or "breaker open"
    return None


def _trip(reason: str) -> None:
    global _blocked_until, _last_error
    _blocked_until = time.monotonic() + BREAKER_MINUTES * 60
    _last_error = reason
    log.warning("FotMob breaker: %s — parked for %d min", reason, BREAKER_MINUTES)


def _get_json(url: str) -> dict[str, Any] | None:
    if blocked():
        return None
    try:
        from curl_cffi import requests as cr
        r = cr.get(url, impersonate=IMPERSONATE, timeout=TIMEOUT)
    except Exception as e:  # noqa: BLE001 - transport failure of any shape
        _trip(f"connect failure: {str(e)[:80]}")
        return None
    if r.status_code in (403, 429) or r.status_code >= 500:
        _trip(f"HTTP {r.status_code} on {url.split('?')[0].rsplit('/', 1)[-1]}")
        return None
    if r.status_code != 200 or "json" not in (r.headers.get("content-type") or ""):
        log.warning("FotMob %s: HTTP %s (%s)", url, r.status_code, r.headers.get("content-type"))
        return None
    try:
        return r.json()
    except ValueError:
        log.warning("FotMob %s: unparseable body", url)
        return None


def matches_for_day(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Our leagues' matches out of a ``/matches?date=`` payload (pure)."""
    out: list[dict[str, Any]] = []
    for lg in (payload or {}).get("leagues") or []:
        league = LEAGUE_IDS.get(lg.get("primaryId") or lg.get("id"))
        if not league:
            continue
        for m in lg.get("matches") or []:
            st = m.get("status") or {}
            out.append({
                "fotmob_id": int(m["id"]),
                "league": league,
                "home": (m.get("home") or {}).get("name") or "",
                "away": (m.get("away") or {}).get("name") or "",
                "home_id": (m.get("home") or {}).get("id"),
                "away_id": (m.get("away") or {}).get("id"),
                "utc_time": st.get("utcTime"),
                "finished": bool(st.get("finished")),
                "cancelled": bool(st.get("cancelled")),
                "score": st.get("scoreStr"),
            })
    return out


def fetch_matches(date: str) -> list[dict[str, Any]] | None:
    """Our leagues' matches on a ``YYYY-MM-DD`` day, None when the source is down."""
    payload = _get_json(f"{BASE}/api/data/matches?date={str(date)[:10].replace('-', '')}")
    if payload is None:
        return None
    return matches_for_day(payload)


def find_match(league: str, home: str, away: str, day_matches: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The day's FotMob match for a fixture, by league + pipeline team names."""
    for m in day_matches:
        if m["league"] != league:
            continue
        if canonical_team(m["home"]) == home and canonical_team(m["away"]) == away:
            return m
    return None


def fetch_match_details(fotmob_id: int) -> dict[str, Any] | None:
    payload = _get_json(f"{BASE}/api/data/matchDetails?matchId={int(fotmob_id)}")
    if payload and (payload.get("general") or {}).get("matchId"):
        return payload
    return None


def raw_path(league: str, season: str, fotmob_id: int) -> Path:
    suffix = "" if league == "serie_a" else f"_{league}"
    return RAW_DIR / f"matches{suffix}" / season / f"{int(fotmob_id)}.json"


def save_raw(payload: dict[str, Any], league: str, season: str, fotmob_id: int) -> Path:
    """The whole payload, as served — nothing is lost to a parser that changes later."""
    p = raw_path(league, season, fotmob_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False))
    tmp.replace(p)
    return p


def load_raw(league: str, season: str, fotmob_id: int) -> dict[str, Any] | None:
    p = raw_path(league, season, fotmob_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# parsers (pure)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MatchContext:
    """Identity of the fixture the rows belong to — Sofascore keyed like every
    other row in these parquets."""
    season: str
    match_id: int              # Sofascore fixture id
    date: str                  # UTC kickoff date, YYYY-MM-DD
    round: int | None
    home_team: str
    away_team: str
    home_score: int | None
    away_score: int | None

    def base(self) -> dict[str, Any]:
        return {"season": self.season, "match_id": self.match_id, "date": self.date,
                "round": self.round, "home_team": self.home_team, "away_team": self.away_team,
                "home_score": self.home_score, "away_score": self.away_score}


def _stat_index(player: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``{title: stat}`` over every group of one playerStats entry."""
    out: dict[str, dict[str, Any]] = {}
    for grp in player.get("stats") or []:
        for title, item in (grp.get("stats") or {}).items():
            stat = (item or {}).get("stat") or {}
            if title not in out:
                out[title] = stat
    return out


def _val(idx: dict[str, dict[str, Any]], title: str) -> Any:
    stat = idx.get(title)
    if stat is None:
        return None
    return stat.get("value")


def _total(idx: dict[str, dict[str, Any]], title: str) -> Any:
    stat = idx.get(title)
    if stat is None:
        return None
    return stat.get("total")


_POSITION = {0: "G", 1: "D", 2: "M", 3: "F"}


def _sides(md: dict[str, Any]) -> tuple[int | None, int | None]:
    g = md.get("general") or {}
    return (g.get("homeTeam") or {}).get("id"), (g.get("awayTeam") or {}).get("id")


def _lineup_index(md: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """fotmob player id -> {is_starter, position, shirt}."""
    out: dict[int, dict[str, Any]] = {}
    lu = (md.get("content") or {}).get("lineup") or {}
    for side in ("homeTeam", "awayTeam"):
        team = lu.get(side) or {}
        for is_starter, key in ((True, "starters"), (False, "subs")):
            for p in team.get(key) or []:
                try:
                    pid = int(p.get("id"))
                except (TypeError, ValueError):
                    continue
                out[pid] = {"is_starter": is_starter,
                            "position": _POSITION.get(p.get("usualPlayingPositionId")),
                            "shirt": p.get("shirtNumber")}
    return out


def _int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def player_rows(md: dict[str, Any], ctx: MatchContext,
                resolve: Resolver | None = None) -> list[dict[str, Any]]:
    """``player_match_stats`` rows for every player with a stat line (an
    unused substitute has none and is skipped, as Sofascore's minutes==0 is)."""
    home_id, away_id = _sides(md)
    lineup = _lineup_index(md)
    rows: list[dict[str, Any]] = []
    for key, p in ((md.get("content") or {}).get("playerStats") or {}).items():
        if not p.get("stats"):
            continue
        idx = _stat_index(p)
        minutes = _int(_val(idx, "Minutes played"))
        if not minutes:
            continue
        try:
            fid = int(p.get("id") or key)
        except (TypeError, ValueError):
            continue
        tid = p.get("teamId")
        if tid == home_id:
            team, opponent, is_home = ctx.home_team, ctx.away_team, True
        elif tid == away_id:
            team, opponent, is_home = ctx.away_team, ctx.home_team, False
        else:
            continue
        name = p.get("name") or ""
        pid = resolve(team, name, fid) if resolve else None
        lu = lineup.get(fid, {})
        # FotMob lists a shot stat only when it is non-zero; the player's own
        # shot map is complete, so an absent count with an empty map is a
        # REAL zero (a prop graded Over 0.5 shots needs the 0, not a None).
        own_shots = [sh for sh in (p.get("shotmap") or []) if isinstance(sh, dict)]
        total_shots = _int(_val(idx, "Total shots"))
        if total_shots is None:
            total_shots = len(own_shots)
        sot = _int(_val(idx, "Shots on target"))
        if sot is None:
            sot = sum(1 for sh in own_shots if sh.get("isOnTarget"))
        blocked_shots = _int(_val(idx, "Blocked shots"))
        if blocked_shots is None:
            blocked_shots = sum(1 for sh in own_shots if sh.get("isBlocked"))
        off_target = _int(_val(idx, "Shots off target"))
        if off_target is None:
            off_target = max(0, total_shots - sot - blocked_shots)
        goals = _int(_val(idx, "Goals"))
        if goals is None:
            goals = sum(1 for sh in own_shots if sh.get("eventType") == "Goal" and not sh.get("isOwnGoal"))
        duels_won = _int(_val(idx, "Duels won"))
        aerial_won = _int(_val(idx, "Aerial duels won"))
        aerial_total = _int(_total(idx, "Aerial duels won"))
        dribbles = _int(_val(idx, "Successful dribbles"))
        rows.append({
            **ctx.base(),
            "team": team, "opponent": opponent, "is_home": is_home,
            "player_id": pid if pid is not None else -fid,
            "player_name": name,
            "position": "G" if p.get("isGoalkeeper") else lu.get("position"),
            "shirt_number": _int(p.get("shirtNumber") or lu.get("shirt")),
            "is_starter": lu.get("is_starter"),
            "minutes": minutes,
            "rating": _float(_val(idx, "FotMob rating")),
            "xg": _float(_val(idx, "Expected goals (xG)")),
            "xgot": _float(_val(idx, "Expected goals on target (xGOT)")),
            "goals": goals,
            "total_shots": total_shots,
            "shots_on_target": sot,
            "shots_off_target": off_target,
            "shots_blocked": blocked_shots,
            "big_chances_created": _int(_val(idx, "Big chances created")),
            "big_chances_missed": _int(_val(idx, "Big chances missed")),
            "xa": _float(_val(idx, "Expected assists (xA)")),
            "assists": _int(_val(idx, "Assists")),
            "key_passes": _int(_val(idx, "Chances created")),
            "accurate_passes": _int(_val(idx, "Accurate passes")),
            "total_passes": _int(_total(idx, "Accurate passes")),
            "accurate_long_balls": _int(_val(idx, "Accurate long balls")),
            "total_long_balls": _int(_total(idx, "Accurate long balls")),
            "accurate_crosses": _int(_val(idx, "Accurate crosses")),
            "total_crosses": _int(_total(idx, "Accurate crosses")),
            "tackles": _int(_val(idx, "Tackles")),
            "interceptions": _int(_val(idx, "Interceptions")),
            "clearances": _int(_val(idx, "Clearances")),
            "ball_recoveries": _int(_val(idx, "Recoveries")),
            "blocks": _int(_val(idx, "Blocks")),
            "last_man_tackle": _int(_val(idx, "Last man tackle")),
            "duels_won": duels_won,
            "duels_lost": _int(_val(idx, "Duels lost")),
            "aerial_won": aerial_won,
            "aerial_lost": (aerial_total - aerial_won) if aerial_total is not None and aerial_won is not None else None,
            "touches": _int(_val(idx, "Touches")),
            "fouls": _int(_val(idx, "Fouls committed")),
            "was_fouled": _int(_val(idx, "Was fouled")),
            "dispossessed": _int(_val(idx, "Dispossessed")),
            "contest_won": dribbles,
            "contest_total": _int(_total(idx, "Successful dribbles")),
            "offsides": _int(_val(idx, "Offsides")),
            "saves": _float(_val(idx, "Saves")),
            "goals_prevented": _float(_val(idx, "Goals prevented")),
            "saved_shots_from_inside_box": _float(_val(idx, "Saves inside box")),
            "keeper_high_claim": _float(_val(idx, "High claim")),
            "keeper_sweeper_total": _float(_val(idx, "Acted as sweeper")),
            "source": "fotmob",
            "player_id_fotmob": fid,
        })
    return rows


_PERIODS = {"All": "ALL", "FirstHalf": "1ST", "SecondHalf": "2ND"}
_FRACTION = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*(?:\((\d+(?:\.\d+)?)%\))?")


def _num_pct(v: Any) -> tuple[float | None, float | None]:
    """``"751 (92%)"`` -> (751, 92); ``12`` -> (12, None); ``"2.12"`` -> (2.12, None)."""
    if v is None:
        return None, None
    if isinstance(v, int | float):
        return float(v), None
    m = _FRACTION.match(str(v))
    if not m:
        return None, None
    return float(m.group(1)), (float(m.group(2)) if m.group(2) else None)


def _period_index(period: dict[str, Any]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for grp in period.get("stats") or []:
        for s in grp.get("stats") or []:
            key = s.get("key")
            vals = s.get("stats")
            if key and isinstance(vals, list) and len(vals) == 2 and key not in out and any(v is not None for v in vals):
                out[key] = vals
    return out


def team_stats_rows(md: dict[str, Any], ctx: MatchContext) -> list[dict[str, Any]]:
    """``match_team_stats`` rows: two per period (ALL / 1ST / 2ND)."""
    periods = ((md.get("content") or {}).get("stats") or {}).get("Periods") or {}
    rows: list[dict[str, Any]] = []
    for fm_period, period in _PERIODS.items():
        pdata = periods.get(fm_period)
        if not pdata:
            continue
        idx = _period_index(pdata)
        if not idx:
            continue

        def side(i: int, idx: dict[str, list[Any]] = idx) -> dict[str, Any]:
            def n(key: str) -> float | None:
                v = idx.get(key)
                return _num_pct(v[i])[0] if v else None

            def pct(key: str) -> float | None:
                v = idx.get(key)
                return _num_pct(v[i])[1] if v else None

            def i_(key: str) -> int | None:
                x = n(key)
                return int(x) if x is not None else None

            duels = idx.get("duel_won")
            duel_pct = None
            if duels and duels[0] is not None and duels[1] is not None and (duels[0] + duels[1]):
                duel_pct = round(100.0 * duels[i] / (duels[0] + duels[1]), 1)
            return {
                "possession": n("BallPossesion"),
                "total_shots": i_("total_shots"),
                "shots_on_target": i_("ShotsOnTarget"),
                "shots_off_target": i_("ShotsOffTarget"),
                "shots_inside_box": i_("shots_inside_box"),
                "shots_outside_box": i_("shots_outside_box"),
                "hit_woodwork": i_("shots_woodwork"),
                "blocked_shots": i_("blocked_shots"),
                "corners": i_("corners"),
                "offsides": i_("Offsides"),
                "fouls": i_("fouls"),
                "big_chances_created": i_("big_chance"),
                "big_chances_missed": i_("big_chance_missed_title"),
                "touches_in_opp_box": i_("touches_opp_box"),
                "accurate_passes": i_("accurate_passes"),
                "total_passes": i_("passes"),
                "accurate_long_balls": i_("long_balls_accurate"),
                "accurate_crosses": i_("accurate_crosses"),
                "duel_won_pct": duel_pct,
                "ground_duels_pct": pct("ground_duels_won"),
                "aerial_duels_pct": pct("aerials_won"),
                "dribbles_pct": pct("dribbles_succeeded"),
                "total_tackles": i_("matchstats.headers.tackles"),
                "interceptions": i_("interceptions"),
                "clearances": i_("clearances"),
                "gk_saves": i_("keeper_saves"),
                "xg": n("expected_goals"),
                "source": "fotmob",
            }

        for i, (team, opponent, is_home) in enumerate(((ctx.home_team, ctx.away_team, True),
                                                         (ctx.away_team, ctx.home_team, False))):
            rows.append({**ctx.base(), "period": period, "team": team, "opponent": opponent,
                         "is_home": is_home, **side(i)})
    return rows


_SHOT_TYPE = {"Goal": "goal", "AttemptSaved": "save", "Miss": "miss", "Post": "post"}
_BODY = {"Header": "head", "RightFoot": "right-foot", "LeftFoot": "left-foot"}
_SITUATION = {"RegularPlay": "regular", "IndividualPlay": "regular", "FastBreak": "fast-break",
              "SetPiece": "set-piece", "FromCorner": "corner", "FreeKick": "free-kick",
              "Penalty": "penalty", "ThrowInSetPiece": "throw-in-set-piece"}
_SET_PIECE = {"set-piece", "corner", "free-kick", "throw-in-set-piece"}
PITCH_X, PITCH_Y = 105.0, 68.0


def shot_rows(md: dict[str, Any], ctx: MatchContext,
              resolve: Resolver | None = None) -> list[dict[str, Any]]:
    """``all_shots_with_xg`` rows. FotMob coordinates are metres with the
    attacked goal at x = 105; Sofascore's are percent of pitch length from the
    goal line (inside the box <= 17) and percent of width (50 = centre), which
    is what ``distance`` / ``angle`` and the inside-box counts assume."""
    home_id, away_id = _sides(md)
    rows: list[dict[str, Any]] = []
    for s in ((md.get("content") or {}).get("shotmap") or {}).get("shots") or []:
        tid = s.get("teamId")
        if tid == home_id:
            is_home, team = True, ctx.home_team
        elif tid == away_id:
            is_home, team = False, ctx.away_team
        else:
            continue
        fx, fy = _float(s.get("x")), _float(s.get("y"))
        x = round((PITCH_X - fx) * 100.0 / PITCH_X, 2) if fx is not None else None
        y = round(fy * 100.0 / PITCH_Y, 2) if fy is not None else None
        if x is not None and y is not None:
            distance = (x ** 2 + (y - 50) ** 2) ** 0.5
            angle = math.degrees(math.atan2(abs(y - 50), x))
        else:
            distance = angle = None
        own_goal = bool(s.get("isOwnGoal"))
        shot_type = "block" if s.get("isBlocked") else _SHOT_TYPE.get(s.get("eventType") or "")
        body = _BODY.get(s.get("shotType") or "", "other")
        situation = _SITUATION.get(s.get("situation") or "")
        minute = _int(s.get("min"))
        name = s.get("playerName") or " ".join(x for x in (s.get("firstName"), s.get("lastName")) if x)
        fid = _int(s.get("playerId"))
        pid = resolve(team, name, fid) if (resolve and fid is not None) else None
        rows.append({
            "season": ctx.season,
            "match_id": str(ctx.match_id),
            "is_home": is_home,
            "player_id": pid if pid is not None else (-fid if fid is not None else None),
            "player_name": name,
            "shot_x": x, "shot_y": y,
            "gm_x": None, "gm_y": None, "gm_z": None,
            "situation": situation,
            "body_part": body,
            "shot_type": shot_type,
            "xg": _float(s.get("expectedGoals")),
            "xgot": _float(s.get("expectedGoalsOnTarget")),
            "is_goal": int(shot_type == "goal" and not own_goal),
            "time": min(minute, 90) if minute is not None else None,
            "distance": distance,
            "angle": angle,
            "is_header": int(body == "head"),
            "is_right": int(body == "right-foot"),
            "is_left": int(body == "left-foot"),
            "is_penalty": int(situation == "penalty"),
            "is_freekick": int(situation == "free-kick"),
            "is_set_piece": int(situation in _SET_PIECE),
            "is_fast_break": int(situation == "fast-break"),
            "xg_predicted": _float(s.get("expectedGoals")),
            "source": "fotmob",
        })
    return rows


def shotmap_stats_rows(shots: list[dict[str, Any]], ctx: MatchContext) -> list[dict[str, Any]]:
    """``shotmap_stats`` rows (two per match) aggregated from ``shot_rows``."""
    import statistics
    rows: list[dict[str, Any]] = []
    for is_home, team, opponent in ((True, ctx.home_team, ctx.away_team), (False, ctx.away_team, ctx.home_team)):
        ts = [s for s in shots if s["is_home"] == is_home]
        base = {**{k: v for k, v in ctx.base().items() if k not in ("home_score", "away_score")},
                "team": team, "opponent": opponent, "is_home": is_home, "source": "fotmob"}
        if not ts:
            rows.append({**base, "shots_total": 0, "shots_on_target": 0, "shots_inside_box": 0,
                         "shots_header": 0, "shots_right_foot": 0, "shots_left_foot": 0,
                         "shots_open_play": 0, "shots_set_piece": 0, "shots_counter": 0,
                         "shots_penalty": 0, "total_xg": 0.0, "total_xgot": 0.0, "avg_shot_xg": 0.0,
                         "max_shot_xg": 0.0, "goals_from_shots": 0, "big_chance_shots": None,
                         "avg_shot_distance": None, "median_shot_distance": None,
                         "shot_distance_std": None, "close_range_pct": 0.0, "shots_hit_post": 0})
            continue
        xgs = [s["xg"] for s in ts if s["xg"] is not None]
        dists = [s["distance"] for s in ts if s["distance"] is not None]
        rows.append({
            **base,
            "shots_total": len(ts),
            "shots_on_target": sum(1 for s in ts if s["shot_type"] in ("save", "goal")),
            "shots_inside_box": sum(1 for s in ts if s["shot_x"] is not None and s["shot_x"] <= 17),
            "shots_header": sum(s["is_header"] for s in ts),
            "shots_right_foot": sum(s["is_right"] for s in ts),
            "shots_left_foot": sum(s["is_left"] for s in ts),
            "shots_open_play": sum(1 for s in ts if s["situation"] == "regular"),
            "shots_set_piece": sum(s["is_set_piece"] for s in ts),
            "shots_counter": sum(s["is_fast_break"] for s in ts),
            "shots_penalty": sum(s["is_penalty"] for s in ts),
            "total_xg": round(sum(xgs), 4) if xgs else 0.0,
            "total_xgot": round(sum(s["xgot"] for s in ts if s["xgot"] is not None), 4),
            "avg_shot_xg": round(sum(xgs) / len(xgs), 4) if xgs else 0.0,
            "max_shot_xg": max(xgs) if xgs else 0.0,
            "goals_from_shots": sum(s["is_goal"] for s in ts),
            "big_chance_shots": None,
            "avg_shot_distance": round(sum(dists) / len(dists), 2) if dists else None,
            "median_shot_distance": round(statistics.median(dists), 2) if dists else None,
            "shot_distance_std": round(statistics.pstdev(dists), 2) if len(dists) > 1 else None,
            "close_range_pct": round(100.0 * sum(1 for d in dists if d <= 12) / len(dists), 1) if dists else 0.0,
            "shots_hit_post": sum(1 for s in ts if s["shot_type"] == "post"),
        })
    return rows


def parse_match(md: dict[str, Any], ctx: MatchContext,
                resolve: Resolver | None = None) -> dict[str, list[dict[str, Any]]]:
    """Every parquet's rows for one match: keys are the parquet base names."""
    shots = shot_rows(md, ctx, resolve)
    return {
        "player_match_stats": player_rows(md, ctx, resolve),
        "match_team_stats": team_stats_rows(md, ctx),
        "all_shots_with_xg": shots,
        "shotmap_stats": shotmap_stats_rows(shots, ctx) if shots else [],
    }


def is_finished(md: dict[str, Any]) -> bool:
    st = (md.get("header") or {}).get("status") or {}
    g = md.get("general") or {}
    return bool(st.get("finished") or g.get("finished")) and not st.get("cancelled")
