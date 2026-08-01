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
import logging
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.leagues import LEAGUE_REGISTRY
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
                "opponent": opponent,
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
_NULLABLE_FLOATS = ("rating_low_trust", "shirt_number")
_NULLABLE_STRS = ("club_league", "formation", "position", "player", "opponent")


def _pin_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in _NULLABLE_FLOATS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in _NULLABLE_STRS:
        if col in df:
            df[col] = df[col].astype("object")
    for col in ("is_home", "is_starter", "was_used", "is_our_club"):
        if col in df:
            df[col] = df[col].astype("bool")
    for col in ("sofascore_event_id", "minutes_played"):
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

    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)
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


def fetch_team_friendlies(team_id: int) -> list[dict[str, Any]]:
    """Recent finished events for a club, filtered to club friendlies."""
    data = _get_json(f"{_BASE_URL}/team/{team_id}/events/last/0")
    if not data:
        return []
    return [e for e in data.get("events", []) if _is_friendly(e)]


def scrape_friendlies(
    leagues: Iterable[str] = ("serie_a", "premier_league"),
    dry_run: bool = False,
) -> pd.DataFrame:
    """Fetch every recent club friendly for our tracked clubs and store it."""
    our_teams = fetch_club_ids(leagues)
    if not our_teams:
        log.error("no club ids resolved -- aborting")
        return pd.DataFrame()
    log.info("resolved %d clubs across %s", len(our_teams), list(leagues))

    seen_events: set[int] = set()
    by_season: dict[str, list[dict[str, Any]]] = {}

    for team_id, (_club, _league) in sorted(our_teams.items(), key=lambda kv: kv[1][0]):
        events = fetch_team_friendlies(team_id)
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
            by_season.setdefault(meta["season"], []).extend(rows)
            log.info("%-18s vs %-18s %s  (%d players)",
                     meta["club"], meta["opponent"], meta["match_date"], len(rows))

    frames = []
    for season, rows in by_season.items():
        if dry_run:
            log.info("[dry-run] %s: %d rows (not written)", season, len(rows))
            frames.append(pd.DataFrame(rows))
        else:
            frames.append(_save(rows, season))
    frames = [fr for fr in frames if not fr.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leagues", default="serie_a,premier_league",
                    help="comma-separated league keys")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch and report without writing the parquet")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    keys = [k.strip() for k in args.leagues.split(",") if k.strip()]
    df = scrape_friendlies(keys, dry_run=args.dry_run)

    if df.empty:
        log.warning("no friendly rows produced")
        return 1
    ours = df[df["is_our_club"]] if "is_our_club" in df else df
    log.info("done: %d rows, %d matches, %d tracked-club players",
             len(df), df["sofascore_event_id"].nunique(), ours["player"].nunique())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
