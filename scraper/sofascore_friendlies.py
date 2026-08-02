#!/usr/bin/env python3
"""Pre-season club-friendly ingest (Sofascore tournament 853).

WHAT THIS IS FOR
----------------
Availability, not performance.  A July friendly tells you almost nothing about
how well a player plays -- no stakes, wildly uneven opposition, rolling
half-time substitutions, trialists in the XI, and squads deliberately short of
match fitness.  It tells you a great deal about *who is available and trusted*:

  - who is fit enough to be named at all
  - which new signing is NOT getting minutes
  - which established starter has been dropped
  - what shape the manager is trialling for matchday 1
  - whose minutes are ramping toward full fitness

That is a genuine prior for matchday-1 XI prediction.  The Sofascore rating is
carried through as ``rating_low_trust`` so that the name itself warns any
downstream join not to compare it with a league rating.

WHY THIS IS A SEPARATE MODULE AND A SEPARATE FILE
-------------------------------------------------
Friendlies must never reach ``lineups.parquet`` or ``player_match_stats.parquet``.
Those feed the training set, and a friendly row landing in them would degrade
the model silently -- nothing downstream would raise.  So this module does NOT
touch ``scraper.sofascore_lineups._SUPPORTED_TOURNAMENT_IDS`` (the league gate);
it keeps its own tournament id and writes its own parquet.

Usage
-----
    python3 -m scraper.sofascore_friendlies --leagues serie_a,premier_league
    python3 -m scraper.sofascore_friendlies --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.leagues import LEAGUE_REGISTRY
from config.settings import get_current_season
from scraper.sofascore_events import _BASE_URL, _get_json, _jitter_delay
from scraper.sofascore_lineups import _normalize_sofascore_team

log = logging.getLogger(__name__)

SOFASCORE_DIR = _PROJECT_ROOT / "data" / "external" / "sofascore"

#: Sofascore's "Club Friendly Games" unique-tournament id.  Deliberately kept
#: out of the league registry so it can never widen the league ingest gate.
FRIENDLY_TOURNAMENT_ID = 853

#: Identity of a row.  Content-owned, never positional -- a back-filled older
#: friendly sorts into the middle, and a positional key would silently
#: reassign every row after it.
_KEY = ["sofascore_event_id", "player_id"]


# ---------------------------------------------------------------------------
# pure parsing helpers (unit-tested without network)
# ---------------------------------------------------------------------------

def _is_friendly(event: dict[str, Any]) -> bool:
    """True only for Sofascore's club-friendly tournament.

    Everything else -- including every league we actually train on -- is
    rejected here.  This is the contamination guard.
    """
    tournament = event.get("tournament") or {}
    unique = tournament.get("uniqueTournament") or {}
    return unique.get("id") == FRIENDLY_TOURNAMENT_ID


def _season_for(ts: int) -> str:
    """Season a friendly belongs to.

    Pre-season friendlies are played in June/July for the season that *starts*
    in August, so the ordinary Aug-1 boundary in ``config.settings`` would file
    them under the season that just ended.
    """
    dt = datetime.fromtimestamp(ts, tz=UTC)
    if dt.month >= 6:
        return f"{dt.year}-{dt.year + 1}"
    return f"{dt.year - 1}-{dt.year}"


def _event_to_meta(
    event: dict[str, Any],
    our_teams: dict[int, tuple[str, str]],
) -> dict[str, Any] | None:
    """Reduce a Sofascore event to the facts we store, or None to skip it.

    Skips: non-friendlies, unfinished matches, and friendlies where neither
    side is a club we track.
    """
    if not _is_friendly(event):
        return None
    if (event.get("status") or {}).get("type") != "finished":
        return None

    home = event.get("homeTeam") or {}
    away = event.get("awayTeam") or {}
    home_id, away_id = home.get("id"), away.get("id")

    if home_id in our_teams:
        our_side, club_id = "home", home_id
    elif away_id in our_teams:
        our_side, club_id = "away", away_id
    else:
        return None

    club, league_key = our_teams[club_id]
    ts = int(event.get("startTimestamp") or 0)

    # Canonical names for clubs we track; opponents we do not track keep their
    # raw Sofascore name (normalising an arbitrary friendly opponent such as
    # "Karlsruher SC" would mangle it more often than it would help).
    def side(team: dict[str, Any]) -> tuple[str | None, bool, str | None]:
        tid = team.get("id")
        if tid in our_teams:
            canonical, lk = our_teams[tid]
            return canonical, True, lk
        return team.get("name"), False, None

    home_name, home_is_ours, home_league = side(home)
    away_name, away_is_ours, away_league = side(away)
    home_country = ((home.get("country") or {}).get("name"))
    away_country = ((away.get("country") or {}).get("name"))

    return {
        "sofascore_event_id": int(event["id"]),
        "match_date": datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d"),
        "season": _season_for(ts),
        "club": club,
        "club_league": league_key,
        "opponent": away_name if our_side == "home" else home_name,
        "is_home": our_side == "home",
        "our_side": our_side,
        "home_team": home_name,
        "away_team": away_name,
        # Both sides can be ours -- Sassuolo v Parma is a real 2026 pre-season
        # fixture, and tagging only one side would lose half of it.
        "home_is_ours": home_is_ours,
        "away_is_ours": away_is_ours,
        "home_league": home_league,
        "away_league": away_league,
        # Ids are what the opponent-profile lookup keys on; names are ambiguous
        # (several "Juventus"/"Arezzo"-shaped clubs exist across tiers).
        "home_id": home.get("id"),
        "away_id": away.get("id"),
        "home_country": home_country,
        "away_country": away_country,
    }


def _parse_lineup(payload: dict[str, Any], meta: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Flatten a /lineups payload into one row per named player, both sides.

    An unused substitute has no ``statistics`` block at all.  He is kept with
    ``minutes_played = 0`` -- "named but did not play" is the signal, and
    dropping him would leave only the players who featured, which is exactly
    the wrong half of the data.
    """
    if not meta:
        return []

    rows: list[dict[str, Any]] = []
    for side in ("home", "away"):
        block = payload.get(side) or {}
        formation = block.get("formation")
        is_home = side == "home"
        team = meta["home_team"] if is_home else meta["away_team"]
        opponent = meta["away_team"] if is_home else meta["home_team"]
        is_ours = meta["home_is_ours"] if is_home else meta["away_is_ours"]
        side_league = meta["home_league"] if is_home else meta["away_league"]
        club_id = meta["home_id"] if is_home else meta["away_id"]
        opp_id = meta["away_id"] if is_home else meta["home_id"]
        opp_country = meta["away_country"] if is_home else meta["home_country"]

        for entry in block.get("players") or []:
            player = entry.get("player") or {}
            stats = entry.get("statistics") or {}
            minutes = int(stats.get("minutesPlayed") or 0)
            rating = stats.get("rating")

            rows.append({
                "sofascore_event_id": meta["sofascore_event_id"],
                "match_date": meta["match_date"],
                "season": meta["season"],
                "club": team,
                "club_id": club_id,
                "opponent": opponent,
                "opponent_id": opp_id,
                "opponent_country": opp_country,
                "is_home": is_home,
                "is_our_club": is_ours,
                "club_league": side_league,
                "formation": formation,
                "player": player.get("name"),
                "player_id": player.get("id"),
                "shirt_number": entry.get("shirtNumber"),
                "position": entry.get("position"),
                "is_starter": not bool(entry.get("substitute")),
                "minutes_played": minutes,
                "was_used": minutes > 0,
                # named rating_low_trust on purpose: a July friendly rating is
                # not comparable to a league rating, and the column name is the
                # only warning a downstream join will ever see.
                "rating_low_trust": float(rating) if rating is not None else None,
            })
    return rows


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------

def _store_path(season: str) -> Path:
    return SOFASCORE_DIR / f"friendlies_{season.replace('-', '_')}.parquet"


#: Columns whose dtype must not drift between season files.  A season with no
#: ratings at all would otherwise store rating_low_trust as object while a
#: populated season stores float64, and reading both at once breaks.
_NULLABLE_FLOATS = ("rating_low_trust", "shirt_number",
                    "opponent_league_id", "opponent_country_priority")
_NULLABLE_STRS = ("club_league", "formation", "position", "player", "opponent",
                  "opponent_country", "opponent_league", "opponent_tier")
_BOOL_COLS = ("is_home", "is_starter", "was_used", "is_our_club",
              "opponent_is_national", "opponent_is_youth")
_INT_COLS = ("sofascore_event_id", "minutes_played", "club_id", "opponent_id")


def _pin_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in _NULLABLE_FLOATS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in _NULLABLE_STRS:
        if col in df:
            df[col] = df[col].astype("object")
    for col in _BOOL_COLS:
        if col in df:
            # infer_objects() before astype: an all-None bool column arrives as
            # object, and fillna->astype on object is deprecated (and silently
            # drifted dtypes between seasons once already).
            df[col] = df[col].fillna(False).infer_objects(copy=False).astype("bool")
    for col in _INT_COLS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
    return df


def _save(rows: Iterable[dict[str, Any]], season: str) -> pd.DataFrame:
    """Merge rows into the season store, de-duplicating on (event, player).

    Last write wins, so a corrected re-scrape updates in place instead of
    appending a second copy.  Saving nothing leaves an existing file untouched.
    """
    path = _store_path(season)
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()

    new = pd.DataFrame(list(rows))
    if new.empty:
        return existing

    merged = pd.concat([existing, new], ignore_index=True) if not existing.empty else new
    merged = _pin_dtypes(merged)
    merged = merged.drop_duplicates(subset=_KEY, keep="last")
    merged = merged.sort_values(["match_date", "sofascore_event_id", "club", "player"])
    merged = merged.reset_index(drop=True)

    # Atomic: this runs on a schedule that can overlap the weekly data refresh,
    # and a reader hitting a half-written parquet would see a torn file. Writing
    # to a sibling then replacing makes the swap indivisible.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp, index=False)
    tmp.replace(path)
    log.info("friendlies: wrote %d rows -> %s", len(merged), path.name)
    return merged


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------

def fetch_club_ids(league_keys: Iterable[str]) -> dict[int, tuple[str, str]]:
    """Resolve {sofascore_team_id: (canonical_name, league_key)} for our clubs."""
    out: dict[int, tuple[str, str]] = {}
    for key in league_keys:
        cfg = LEAGUE_REGISTRY.get(key)
        if cfg is None:
            log.warning("unknown league key %r -- skipping", key)
            continue
        tid = cfg.sofascore_tournament_id

        seasons = _get_json(f"{_BASE_URL}/unique-tournament/{tid}/seasons")
        if not seasons or not seasons.get("seasons"):
            log.warning("%s: no seasons returned", key)
            continue
        season_id = seasons["seasons"][0]["id"]

        teams = _get_json(f"{_BASE_URL}/unique-tournament/{tid}/season/{season_id}/teams")
        if not teams:
            log.warning("%s: no teams returned", key)
            continue
        for t in teams.get("teams", []):
            out[int(t["id"])] = (_normalize_sofascore_team(t["name"]), key)
        _jitter_delay()
    return out


#: Youth / reserve sides play under the parent club's name plus a marker.  A 90
#: minutes against Lazio U20 is not the same evidence as 90 against Man City.
_YOUTH_MARKERS = (" u19", " u20", " u21", " u23", " ii", " b", " primavera",
                  " next gen", " youth", " reserves", " academy")

#: Opponent-profile cache.  One /team/{id} request per DISTINCT opponent, then
#: never again -- opponents repeat across a pre-season and across years.
# Keyed by team id. An opponent promoted or relegated after its first resolution
# would keep its stale league/tier forever, so the cache is STAMPED with the season
# that resolved it and self-invalidates on rollover -- this used to be a manual
# "delete this file every August" note in DATA_CATALOG.md, which is precisely the
# kind of instruction that gets missed. Re-deriving loses nothing: the parquet keeps
# the raw `opponent` name verbatim.
_OPP_CACHE = SOFASCORE_DIR / "friendly_opponent_profiles.json"
_OPP_CACHE_SEASON_KEY = "__season__"


def _is_youth_side(name: str | None) -> bool:
    """Space-padded token match, NOT substring.

    The padding is load-bearing: a bare `" b" in name` would flag "Arminia
    Bielefeld" and "Rosenborg BK".  Matching `" b "` inside `" arminia
    bielefeld "` is False, while "Real Madrid B" -> `" real madrid b "` hits.
    Validated over the 118 distinct opponents actually on disk: 2 flagged
    (Atalanta U23, Lazio U20), zero false positives.
    """
    if not name:
        return False
    low = f" {name.lower()} "
    return any(f"{m} " in low for m in _YOUTH_MARKERS)


def _load_opp_cache() -> dict[str, Any]:
    """Load the cache, discarding it wholesale if it was built in a past season.

    A club's league is a per-season fact, so a profile resolved last August is
    not merely old, it is wrong for anyone promoted or relegated since. The
    season stamp makes that automatic instead of remembered.

    An UNSTAMPED file is treated as current, not as stale: the stamp was added
    2026-08-02, and the file on disk at that point had just been fully rebuilt
    by the Aug 1 backfill (verified -- all six 26/27-promoted clubs already
    carried their NEW league). Discarding it would have thrown away 261 correct
    profiles and re-fetched every one.
    """
    if not _OPP_CACHE.exists():
        return {}
    try:
        cache = json.loads(_OPP_CACHE.read_text())
    except (json.JSONDecodeError, OSError):
        log.warning("opponent cache unreadable -- rebuilding")
        return {}
    if not isinstance(cache, dict):
        log.warning("opponent cache is not a mapping -- rebuilding")
        return {}
    stamped = cache.get(_OPP_CACHE_SEASON_KEY)
    current = get_current_season()
    if stamped is not None and stamped != current:
        log.info(
            "opponent cache was resolved for %s, current season is %s -- "
            "discarding %d profiles so leagues are re-derived",
            stamped, current, len(cache) - 1,
        )
        return {}
    return cache


def _save_opp_cache(cache: dict[str, Any]) -> None:
    """Atomic write -- a truncated cache would silently lose every profile."""
    try:
        _OPP_CACHE.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(cache)
        payload[_OPP_CACHE_SEASON_KEY] = get_current_season()
        tmp = _OPP_CACHE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(_OPP_CACHE)
    except OSError as exc:
        log.warning("could not persist opponent cache: %s", exc)


def fetch_opponent_profile(team_id: int, cache: dict[str, Any]) -> dict[str, Any]:
    """Resolve which competition an opponent actually plays in.

    The friendly itself carries no strength information: "Juventus 2-0 Nice" and
    "Torino 3-0 ACD Pinzolo Valrendena" look identical in the payload.  The
    opponent's OWN primary league is what makes a 90-minute shift legible.
    """
    key = str(team_id)
    if key in cache:
        return cache[key]

    data = _get_json(f"{_BASE_URL}/team/{team_id}")
    _jitter_delay()
    team = (data or {}).get("team") or {}
    put = team.get("primaryUniqueTournament") or {}
    category = put.get("category") or {}

    profile = {
        "opponent_league": put.get("name"),
        "opponent_league_id": put.get("id"),
        "opponent_country_priority": category.get("priority"),
        "opponent_is_national": bool(team.get("national")),
        "opponent_is_youth": _is_youth_side(team.get("name")),
    }
    cache[key] = profile
    return profile


def _opponent_tier(profile: dict[str, Any]) -> str:
    """Coarse, honest bucket -- facts are stored raw so this can be re-derived."""
    if profile.get("opponent_is_youth"):
        return "youth_or_reserve"
    if profile.get("opponent_is_national"):
        return "national_team"
    lid = profile.get("opponent_league_id")
    if lid in {cfg.sofascore_tournament_id for cfg in LEAGUE_REGISTRY.values()}:
        return "top5_league"
    if lid is not None:
        return "other_professional"
    return "lower_or_unknown"


def fetch_team_friendlies(team_id: int, pages: int = 1) -> list[dict[str, Any]]:
    """Recent finished events for a club, filtered to club friendlies.

    One page (30 events) covers the whole current pre-season -- measured on
    Milan 2026-08-01, page 0 spans 2025-12-04 -> 2026-07-25.  Page 1 reaches the
    PREVIOUS pre-season, which is the only way to ever backtest whether this
    signal improves XI accuracy; it is opt-in because it doubles the scrape.
    """
    events: list[dict[str, Any]] = []
    for page in range(max(1, pages)):
        data = _get_json(f"{_BASE_URL}/team/{team_id}/events/last/{page}")
        if not data:
            break
        events.extend(e for e in data.get("events", []) if _is_friendly(e))
        if page + 1 < pages:
            _jitter_delay()
    return events


def scrape_friendlies(
    leagues: Iterable[str] = ("serie_a", "premier_league"),
    dry_run: bool = False,
    pages: int = 1,
) -> pd.DataFrame:
    """Fetch every recent club friendly for our tracked clubs and store it."""
    our_teams = fetch_club_ids(leagues)
    if not our_teams:
        log.error("no club ids resolved -- aborting")
        return pd.DataFrame()
    log.info("resolved %d clubs across %s", len(our_teams), list(leagues))

    seen_events: set[int] = set()
    by_season: dict[str, list[dict[str, Any]]] = {}
    opp_cache = _load_opp_cache()

    for team_id, (_club, _league) in sorted(our_teams.items(), key=lambda kv: kv[1][0]):
        events = fetch_team_friendlies(team_id, pages=pages)
        _jitter_delay()
        for ev in events:
            meta = _event_to_meta(ev, our_teams)
            if not meta or meta["sofascore_event_id"] in seen_events:
                continue
            seen_events.add(meta["sofascore_event_id"])

            payload = _get_json(f"{_BASE_URL}/event/{meta['sofascore_event_id']}/lineups")
            _jitter_delay()
            if not payload:
                log.warning("%s vs %s: no lineups", meta["club"], meta["opponent"])
                continue
            rows = _parse_lineup(payload, meta)

            # Resolve where each opponent actually comes from.  The raw name is
            # kept verbatim on the row; this only ADDS provenance beside it, so
            # a wrong tier can always be re-derived from the stored facts.
            for row in rows:
                opp_id = row.get("opponent_id")
                if not opp_id:
                    continue
                profile = fetch_opponent_profile(int(opp_id), opp_cache)
                row.update(profile)
                row["opponent_tier"] = _opponent_tier(profile)

            by_season.setdefault(meta["season"], []).extend(rows)
            log.info("%-18s vs %-18s %s  (%d players)",
                     meta["club"], meta["opponent"], meta["match_date"], len(rows))

    if not dry_run:
        _save_opp_cache(opp_cache)

    frames = []
    for season, rows in by_season.items():
        if dry_run:
            log.info("[dry-run] %s: %d rows (not written)", season, len(rows))
            frames.append(pd.DataFrame(rows))
        else:
            frames.append(_save(rows, season))
    frames = [fr for fr in frames if not fr.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


#: Club friendlies are a pre-season phenomenon: they cluster in June-August and
#: reappear only around the winter break.  A daily job with no gate would spend
#: ten months fetching an empty list.  Mirrors WINDOW_RANGES in
#: scripts/data/refresh_transfers.py -- same shape, same --force escape hatch.
FRIENDLY_WINDOWS = (
    ((6, 1), (9, 5)),    # summer pre-season, with slack past the first matchweek
    ((12, 15), (1, 10)),  # winter-break friendlies (wraps the year end)
)


def _in_friendly_window(today: date) -> bool:
    for (lo_m, lo_d), (hi_m, hi_d) in FRIENDLY_WINDOWS:
        lo, hi = (lo_m, lo_d), (hi_m, hi_d)
        cur = (today.month, today.day)
        if lo <= hi:
            if lo <= cur <= hi:
                return True
        elif cur >= lo or cur <= hi:   # window wraps 31 Dec
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leagues", default="serie_a,premier_league",
                    help="comma-separated league keys")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch and report without writing the parquet")
    ap.add_argument("--pages", type=int, default=1,
                    help="pages of match history per club; 2 reaches last pre-season")
    ap.add_argument("--force", action="store_true",
                    help="run even outside the pre-season window")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    today = datetime.now(UTC).date()
    if not args.force and not _in_friendly_window(today):
        log.info("outside the friendly window (%s) -- skipping. Use --force to override.",
                 today)
        return 0

    keys = [k.strip() for k in args.leagues.split(",") if k.strip()]
    df = scrape_friendlies(keys, dry_run=args.dry_run, pages=args.pages)

    if df.empty:
        log.warning("no friendly rows produced")
        return 1
    ours = df[df["is_our_club"]] if "is_our_club" in df else df
    log.info("done: %d rows, %d matches, %d tracked-club players",
             len(df), df["sofascore_event_id"].nunique(), ours["player"].nunique())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
