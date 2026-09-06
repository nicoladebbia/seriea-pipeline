"""Finished-match record chain — Sofascore, else FotMob, else ESPN (2026-09-06).

The Sofascore ingest (``matchday_updater``) is the spine: when it answers, its
rows are the record. When it does not — ``403 challenge`` since 2026-09-05, a
blanket IP deny a day later — a finished match sat score-only and the player
stats that grade props, the team stats and the shot map never arrived. This
module walks every finished fixture of the season, asks each stat parquet
whether it holds SOFASCORE-backed rows for the match, and fills each missing
component from the next source that answers:

    component            parquet                         fotmob   espn
    player_match_stats   player_match_stats{_lg}.parquet   full     shots/SoT/goals/assists/fouls/cards/saves + minutes
    match_team_stats     match_team_stats{_lg}.parquet     3 periods ALL only (boxscore)
    all_shots_with_xg    all_shots_with_xg{_lg}.parquet    full     —
    shotmap_stats        shotmap_stats{_lg}.parquet        full     —

Rules that keep this honest:
* Every row carries ``source``. A stand-in is NOT coverage: the detector in
  matchday_updater re-fetches the match from Sofascore when it answers again
  and ``_save_merged`` drops the stand-in rows for that match then. Nothing
  is ever overwritten by a weaker source; nothing is lost when a stronger one
  arrives — the raw payload of every source stays on disk
  (``data/external/fotmob/match_details/matches{_lg}/{season}/{id}.json``).
* A source that fails (403 / 429 / connect) is parked by its own breaker for
  the rest of the run; the chain never retries inside a run.
* A team name that does not normalise, a fixture FotMob does not list, a
  payload not marked finished: skipped and NAMED in
  ``data/monitoring/ingest_chain_status.json`` — never guessed.
* Player ids are resolved against the parquet's own history ((team, folded
  name) → unique Sofascore id); unresolved players carry ``-fotmob_id``
  (negative: never a Sofascore id). ESPN rows carry ``-1`` × a stable hash
  of the folded name only when unresolved.

Run: ``python3 -m scripts.data.match_record_chain [--league L] [--season S] [--dry-run]``.
Called by ``run_matchday_update`` after ``heal_from_espn`` on every league pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import DATA_DIR, atomic_write_parquet, get_current_season
from config.team_names import normalize_team

log = logging.getLogger(__name__)

COMPONENTS = ("player_match_stats", "match_team_stats", "all_shots_with_xg", "shotmap_stats")
SOURCE_ORDER = ("fotmob", "espn")
SOFASCORE = "sofascore"
STATUS_FILE = DATA_DIR / "monitoring" / "ingest_chain_status.json"
KICKED_OFF_HOURS = 3.0      # a fixture past kickoff by this much counts as played
LOOKBACK_DAYS = 400         # the whole season; the parquet says what is missing


def _fold(name: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", name or "")
                   if not unicodedata.combining(c)).lower().strip()


# ---------------------------------------------------------------------------
# what is on disk
# ---------------------------------------------------------------------------

def component_path(component: str, league: str) -> Path:
    from scripts.data.matchday_updater import _sofascore_parquet
    if component == "all_shots_with_xg":
        from scripts.data.write_shot_level_xg import out_path
        return out_path(league)
    return _sofascore_parquet(component, league)


def sofascore_backed_ids(path: Path) -> set[str]:
    """Match ids (as str) the parquet holds from Sofascore itself. A row with no
    ``source`` predates the column and IS Sofascore; ``fotmob`` / ``espn`` rows
    are stand-ins and do not count."""
    if not path.exists():
        return set()
    try:
        df = pd.read_parquet(path, columns=["match_id", "source"])
        keep = df["source"].isna() | (df["source"].astype(object) == SOFASCORE)
        ids = df.loc[keep, "match_id"]
    except (KeyError, ValueError):
        ids = pd.read_parquet(path, columns=["match_id"])["match_id"]
    return {str(int(x)) if isinstance(x, int | float) and not pd.isna(x) else str(x) for x in ids.dropna().unique()}


def stand_in_sources(path: Path, match_id: int) -> dict[str, int]:
    """{source: rows} the parquet already holds for the match from stand-ins."""
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(path, columns=["match_id", "source"])
    except (KeyError, ValueError):
        return {}
    m = df["match_id"].astype(str) == str(match_id)
    src = df.loc[m & df["source"].notna(), "source"].astype(str)
    return {k: int(v) for k, v in src.value_counts().items() if k != SOFASCORE}


def finished_fixtures(season: str, league: str, now: float | None = None) -> list[dict[str, Any]]:
    """Fixtures the calendar says are finished, or that kicked off more than
    KICKED_OFF_HOURS ago (the cache cannot flip a status while Sofascore is
    denied); cancelled / postponed excluded."""
    from scripts.data.matchday_updater import _load_fixtures
    now = now or datetime.now(UTC).timestamp()
    out = []
    for f in _load_fixtures(season, league):
        ts = f.get("startTimestamp")
        if not f.get("id") or not ts or ts > now or ts < now - LOOKBACK_DAYS * 86400:
            continue
        st = ((f.get("status") or {}).get("type") or "").lower()
        if st in ("canceled", "cancelled", "postponed"):
            continue
        if st == "finished" or ts <= now - KICKED_OFF_HOURS * 3600:
            out.append(f)
    return out


def sofascore_cached_shotmap(season: str, league: str, match_id: int) -> bool:
    """True when the Sofascore match json on disk carries a shot map: the
    weekly ``write_shot_level_xg`` rebuild will (re)derive the rows from it, so
    the real source is present and a stand-in must not be written."""
    from scripts.data.write_shot_level_xg import _shots_from_cache_file, cache_dir
    p = cache_dir(season, league) / f"{int(match_id)}.json"
    return p.exists() and bool(_shots_from_cache_file(p))


def missing_components(season: str, league: str, fixtures: list[dict[str, Any]]) -> dict[int, list[str]]:
    """{sofascore match id: [components with no Sofascore-backed rows]}.

    ``all_shots_with_xg`` counts the cached Sofascore json as coverage (see
    ``sofascore_cached_shotmap``); ``rebuild_shots_from_cache`` turns those
    into rows before the chain runs."""
    backed = {c: sofascore_backed_ids(component_path(c, league)) for c in COMPONENTS}
    out: dict[int, list[str]] = {}
    for f in fixtures:
        mid = int(f["id"])
        miss = [c for c in COMPONENTS if str(mid) not in backed[c]]
        if "all_shots_with_xg" in miss and sofascore_cached_shotmap(season, league, mid):
            miss.remove("all_shots_with_xg")
        if miss:
            out[mid] = miss
    return out


def rebuild_shots_from_cache(season: str, league: str, fixtures: list[dict[str, Any]]) -> int:
    """Matches whose Sofascore shot map is cached but not yet in the parquet
    get their rows from the cache (zero network) — the weekly job would do it
    days later. Returns the number of such matches before the rebuild."""
    backed = sofascore_backed_ids(component_path("all_shots_with_xg", league))
    pending = [int(f["id"]) for f in fixtures
               if str(int(f["id"])) not in backed and sofascore_cached_shotmap(season, league, int(f["id"]))]
    if pending:
        from scripts.data.write_shot_level_xg import rebuild_from_cache
        log.info("[%s] chain: %d match(es) have a cached Sofascore shot map not yet in the parquet — rebuilding %s",
                 league, len(pending), season)
        rebuild_from_cache(season, league)
    return len(pending)


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------

def fixture_context(f: dict[str, Any], season: str):
    from scraper.fotmob import MatchContext
    from scripts.data.scrape_sofascore import _kickoff_date
    home = normalize_team((f.get("homeTeam") or {}).get("name") or "")
    away = normalize_team((f.get("awayTeam") or {}).get("name") or "")
    rnd = (f.get("roundInfo") or {}).get("round")
    hs = (f.get("homeScore") or {}).get("current")
    as_ = (f.get("awayScore") or {}).get("current")
    return MatchContext(season=season, match_id=int(f["id"]), date=_kickoff_date(f),
                        round=int(rnd) if rnd not in (None, "") else None,
                        home_team=home, away_team=away,
                        home_score=int(hs) if hs is not None else None,
                        away_score=int(as_) if as_ is not None else None)


def build_resolver(league: str):
    """(team, name, foreign id) -> the Sofascore player id the parquet already
    knows for that folded name in that team, when exactly one exists."""
    path = component_path("player_match_stats", league)
    index: dict[tuple[str, str], set[int]] = defaultdict(set)
    if path.exists():
        try:
            cols = ["team", "player_name", "player_id"]
            df = pd.read_parquet(path, columns=cols + ["source"])
            df = df[df["source"].isna() | (df["source"].astype(object) == SOFASCORE)]
        except (KeyError, ValueError):
            df = pd.read_parquet(path, columns=cols)
        df = df[df["player_id"] > 0]
        for team, name, pid in df[["team", "player_name", "player_id"]].drop_duplicates().itertuples(index=False):
            index[(str(team), _fold(str(name)))].add(int(pid))

    def resolve(team: str, name: str, _fid: int) -> int | None:
        ids = index.get((team, _fold(name)))
        return next(iter(ids)) if ids and len(ids) == 1 else None

    return resolve


def _espn_pseudo_id(name: str) -> int:
    h = int(hashlib.blake2b(_fold(name).encode(), digest_size=4).hexdigest(), 16)
    return -(10_000_000_000 + h)


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------

class _DayCache:
    """One FotMob day listing per date per run."""

    def __init__(self) -> None:
        self.days: dict[str, list[dict[str, Any]] | None] = {}

    def matches(self, date: str) -> list[dict[str, Any]] | None:
        from scraper import fotmob
        if date not in self.days:
            self.days[date] = fotmob.fetch_matches(date)
        return self.days[date]


def record_from_fotmob(ctx, league: str, resolve, days: _DayCache) -> tuple[dict[str, list[dict]] | None, str]:
    """(rows per component, reason). Raw payload saved before parsing."""
    from scraper import fotmob
    if fotmob.blocked():
        return None, f"fotmob parked: {fotmob.blocked()}"
    day = days.matches(ctx.date)
    if day is None:
        return None, f"fotmob day listing failed: {fotmob.blocked() or 'no payload'}"
    m = fotmob.find_match(league, ctx.home_team, ctx.away_team, day)
    if not m:
        names = sorted(f"{fotmob.canonical_team(x['home'])} v {fotmob.canonical_team(x['away'])}" for x in day if x["league"] == league)
        return None, f"fotmob lists no {ctx.home_team} v {ctx.away_team} on {ctx.date} (has: {', '.join(names) or 'nothing'})"
    md = fotmob.load_raw(league, ctx.season, m["fotmob_id"])
    if md is None or not fotmob.is_finished(md):
        md = fotmob.fetch_match_details(m["fotmob_id"])
        if md is None:
            return None, f"fotmob matchDetails {m['fotmob_id']} failed: {fotmob.blocked() or 'no payload'}"
        fotmob.save_raw(md, league, ctx.season, m["fotmob_id"])
    if not fotmob.is_finished(md):
        return None, f"fotmob {m['fotmob_id']} not marked finished"
    return fotmob.parse_match(md, ctx, resolve), f"fotmob {m['fotmob_id']}"


_ESPN_POSITIONS = {"G", "D", "M", "F"}


def rows_from_espn_summary(summary: dict[str, Any], ctx, resolve) -> dict[str, list[dict]]:
    """player_match_stats (the subset ESPN carries, minutes from the sub events,
    clock 90) and match_team_stats ALL from the boxscore."""
    from scripts.data import live_espn as le
    events = le.parse_key_events(summary.get("keyEvents") or [], le._home_id(summary))
    rosters = le.parse_rosters(summary.get("rosters") or [], events, clock_minute=90)
    players: list[dict[str, Any]] = []
    for side, is_home in (("home", True), ("away", False)):
        team = ctx.home_team if is_home else ctx.away_team
        opponent = ctx.away_team if is_home else ctx.home_team
        for p in rosters.get(side) or []:
            minutes = p.get("minutes_played")
            if not minutes:
                continue
            pid = resolve(team, p["name"], 0) if resolve else None
            pos = p.get("position") or ""
            players.append({
                **ctx.base(),
                "team": team, "opponent": opponent, "is_home": is_home,
                "player_id": pid if pid is not None else _espn_pseudo_id(p["name"]),
                "player_name": p["name"],
                "position": pos if pos in _ESPN_POSITIONS else None,
                "shirt_number": int(p["jersey_number"]) if str(p.get("jersey_number") or "").isdigit() else None,
                "is_starter": not p.get("substitute"),
                "minutes": int(minutes),
                "goals": p.get("goals"), "assists": p.get("assists"),
                "total_shots": p.get("shots"), "shots_on_target": p.get("shots_on_target"),
                "fouls": p.get("fouls_committed"), "was_fouled": p.get("fouls_drawn"),
                "offsides": p.get("offsides"), "saves": p.get("saves"),
                "source": "espn",
            })
    teams: list[dict[str, Any]] = []
    box = le.parse_boxscore(summary.get("boxscore") or {})
    if box:
        for side, is_home in (("home", True), ("away", False)):
            team = ctx.home_team if is_home else ctx.away_team
            opponent = ctx.away_team if is_home else ctx.home_team

            def g(key: str, _side=side) -> Any:
                v = (box.get(key) or {}).get(_side)
                return v

            teams.append({
                **ctx.base(), "period": "ALL", "team": team, "opponent": opponent, "is_home": is_home,
                "possession": g("possession"), "total_shots": g("shots"), "shots_on_target": g("shots_on_target"),
                "blocked_shots": g("blocked_shots"), "corners": g("corners"), "fouls": g("fouls"),
                "gk_saves": g("saves"), "offsides": g("offsides"), "total_tackles": g("tackles"),
                "clearances": g("clearances"), "accurate_passes": g("accurate_passes"),
                "source": "espn",
            })
    return {"player_match_stats": players, "match_team_stats": teams}


def record_from_espn(ctx, league: str, resolve) -> tuple[dict[str, list[dict]] | None, str]:
    from scripts.data import live_espn as le
    try:
        summary = le.post_match_summary(league, ctx.date, ctx.home_team, ctx.away_team)
    except Exception as e:  # noqa: BLE001 - a source outage, not a failed run
        return None, f"espn failed: {str(e)[:80]}"
    if not summary:
        return None, "espn has no post-match summary"
    return rows_from_espn_summary(summary, ctx, resolve), "espn"


SOURCES = {"fotmob": record_from_fotmob, "espn": record_from_espn}


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------

def _align_dtypes(new_df: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """Cast the incoming columns to the parquet's dtypes where that is lossless
    (a str match_id stays str, an all-None column takes the existing dtype so
    the concat is not a dtype-widening surprise); anything that will not cast
    is left as is."""
    out = new_df.copy()
    for c in out.columns:
        target = existing[c].dtype
        if pd.api.types.is_string_dtype(target) or pd.api.types.is_object_dtype(target):
            if c == "match_id":
                out[c] = out[c].astype(str)
            continue
        try:
            out[c] = out[c].astype(target)
        except (TypeError, ValueError):
            pass
    return out


def write_stand_in(rows: list[dict[str, Any]], path: Path, match_id: int, source: str) -> int:
    """Append stand-in rows for one match, replacing any earlier rows of the SAME
    source for it (idempotent re-fill). Sofascore rows are never touched here —
    the chain only writes a component that has none."""
    if not rows:
        return 0
    new_df = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_parquet(path)
        if "source" not in existing.columns:
            existing["source"] = None
        drop = (existing["match_id"].astype(str) == str(match_id)) & (existing["source"].astype(object) == source)
        existing = existing[~drop]
        for c in new_df.columns:
            if c not in existing.columns:
                existing[c] = None
        for c in existing.columns:
            if c not in new_df.columns:
                new_df[c] = None
        new_df = _align_dtypes(new_df[existing.columns], existing)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(path, combined, index=False)
    return len(new_df)


# ---------------------------------------------------------------------------
# the chain
# ---------------------------------------------------------------------------

def heal_missing(season: str | None = None, league: str = "serie_a", dry_run: bool = False,
                 sources: tuple[str, ...] = SOURCE_ORDER, now: float | None = None,
                 write_status: bool = True) -> dict[str, Any]:
    season = season or get_current_season()
    t0 = time.time()
    fixtures = finished_fixtures(season, league, now)
    shots_rebuilt = 0 if dry_run else rebuild_shots_from_cache(season, league, fixtures)
    missing = missing_components(season, league, fixtures)
    by_id = {int(f["id"]): f for f in fixtures}
    summary: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(), "league": league, "season": season,
        "dry_run": dry_run, "finished_fixtures": len(fixtures), "matches_missing_before": len(missing),
        "matches_filled": 0, "matches_partial": 0, "matches_unfilled": 0, "shots_rebuilt_from_cache": shots_rebuilt,
        "rows_written": {c: 0 for c in COMPONENTS}, "sources": {}, "matches": [],
    }
    if not missing:
        log.info("[%s] chain: every finished fixture has Sofascore rows in all %d parquets", league, len(COMPONENTS))
        if write_status:
            _write_status(summary)
        return summary

    resolve = build_resolver(league)
    days = _DayCache()
    for mid in sorted(missing, key=lambda m: by_id[m].get("startTimestamp") or 0):
        f = by_id[mid]
        ctx = fixture_context(f, season)
        entry: dict[str, Any] = {"match_id": mid, "date": ctx.date, "home": ctx.home_team, "away": ctx.away_team,
                                 "missing_before": list(missing[mid]), "filled": {}, "already": {}, "reasons": []}
        if not ctx.home_team or not ctx.away_team:
            entry["reasons"].append("team name did not normalise")
        todo = list(missing[mid])
        for comp in list(todo):
            have = stand_in_sources(component_path(comp, league), mid)
            if have:
                entry["already"][comp] = have
        for src in sources:
            if not todo or entry["reasons"] and "did not normalise" in entry["reasons"][0]:
                break
            fetch = SOURCES[src]
            record, reason = fetch(ctx, league, resolve, days) if src == "fotmob" else fetch(ctx, league, resolve)
            if record is None:
                entry["reasons"].append(reason)
                continue
            for comp in list(todo):
                rows = record.get(comp) or []
                # a source the parquet already holds for this component: rewrite (idempotent), still counts
                if not rows:
                    continue
                if not dry_run:
                    n = write_stand_in(rows, component_path(comp, league), mid, src)
                    summary["rows_written"][comp] += n
                entry["filled"][comp] = src
                todo.remove(comp)
            entry["reasons"].append(f"{reason}: {', '.join(c for c, s in entry['filled'].items() if s == src) or 'nothing usable'}")
        entry["still_missing"] = todo
        if not todo:
            summary["matches_filled"] += 1
        elif entry["filled"]:
            summary["matches_partial"] += 1
        else:
            summary["matches_unfilled"] += 1
        summary["matches"].append(entry)
        log.info("[%s] chain %s %s v %s: filled %s%s", league, ctx.date, ctx.home_team, ctx.away_team,
                 entry["filled"] or "nothing", f", still missing {todo}" if todo else "")
    try:
        from scraper import fotmob
        summary["sources"]["fotmob"] = fotmob.blocked() or ("ok" if days.days else "not needed")
    except Exception:  # noqa: BLE001
        summary["sources"]["fotmob"] = "import failed"
    summary["elapsed_seconds"] = round(time.time() - t0, 1)
    if write_status:
        _write_status(summary)
    return summary


def _write_status(summary: dict[str, Any]) -> None:
    """One file, one entry per league (the other league's entry is kept)."""
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {}
    if STATUS_FILE.exists():
        try:
            state = json.loads(STATUS_FILE.read_text())
        except (OSError, ValueError):
            state = {}
    state[summary["league"]] = summary
    tmp = STATUS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1, default=str))
    tmp.replace(STATUS_FILE)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--league", default=None, help="serie_a | premier_league (default: both)")
    ap.add_argument("--season", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    leagues = [a.league] if a.league else ["serie_a", "premier_league"]
    rc = 0
    for lg in leagues:
        s = heal_missing(season=a.season, league=lg, dry_run=a.dry_run)
        print(f"{lg}: {s['finished_fixtures']} finished, {s['matches_missing_before']} missing -> "
              f"filled {s['matches_filled']}, partial {s['matches_partial']}, unfilled {s['matches_unfilled']}; "
              f"rows {s['rows_written']}; fotmob {s['sources'].get('fotmob')}")
        for m in s["matches"]:
            print(f"  {m['date']} {m['home']} v {m['away']}: {m['filled'] or '-'} | still {m['still_missing']} | {' / '.join(m['reasons'])}")
        if s["matches_unfilled"]:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
