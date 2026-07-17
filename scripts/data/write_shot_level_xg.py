"""Persist per-shot rows from cached Sofascore shotmaps to all_shots_with_xg.parquet.

Why this module exists
----------------------
The Sofascore scraper (``scrape_sofascore.py``) fetches the full per-shot shotmap
for every match and caches it under
``data/external/sofascore/matches/{season}/{sofascore_id}.json``, but the only
shot output it *persists* is the per-team AGGREGATE
(``shotmap_stats.parquet``, 2 rows/match). The shot-LEVEL file
``all_shots_with_xg.parquet`` — which ``features/shot_level_xg.py`` and
``features/situational_xg.py`` read — was produced by a one-shot script that
stopped running in Feb 2026. That froze its 2025-26 coverage at 206/380 matches
(last match 2026-02-08) and drove 64 shot features to ~46% NaN for the season.

Nothing about the source changed: the per-shot payload (xg, xgot, coordinates,
situation, body part) is still fetched and cached on every weekly run. This module
re-derives the shot-level rows from that cache — **zero network** — and is meant to
run weekly right after the Sofascore scrape, plus once as a backfill for the gap.

The transform was reverse-engineered from the existing parquet and verified against
it before any row was written:
  - distance = sqrt(x**2 + (y-50)**2),  angle = deg(atan2(|y-50|, x))  [x=8.3,y=51.5
    -> 8.434 / 10.244, matching the stored values exactly]
  - is_set_piece = situation in {corner, set-piece}  — NOT "throw-in-set-piece",
    which the existing file maps to 0. The one-hots below reproduce the file's
    groupby-situation / groupby-body_part means exactly.

xg_predicted note: it is a NaN-fallback for xg, read only where xg is NaN
(``sofascore_features.py``: ``xg.fillna(xg_predicted)``), which never happens for a
season Sofascore serves xg for — all of 2025-26. Rather than fabricate a model
output whose original we could not recover, we set it equal to the observed xg; no
2025-26 feature reads it.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import pandas as pd

from config.settings import DATA_DIR, atomic_write_parquet

log = logging.getLogger(__name__)

SOFA_DIR = DATA_DIR / "external" / "sofascore"

# situation -> is_set_piece. Reverse-engineered from all_shots_with_xg.parquet:
# corner and set-piece map to 1; assisted/regular/fast-break/free-kick/penalty and
# (deliberately) throw-in-set-piece map to 0.
_SET_PIECE = {"corner", "set-piece"}


def _suffix(league: str) -> str:
    return "" if league == "serie_a" else f"_{league}"


def out_path(league: str = "serie_a") -> Path:
    return SOFA_DIR / f"all_shots_with_xg{_suffix(league)}.parquet"


def cache_dir(season: str, league: str = "serie_a") -> Path:
    # EPL shotmaps cache under matches_premier_league/; keep the reader league-aware
    # so a per-league run reads the right cache (never the Serie A one by accident).
    return SOFA_DIR / f"matches{_suffix(league)}" / season


def shot_rows_from_shotmap(shots: list[dict], sofascore_id: str,
                           season: str) -> list[dict]:
    """Map a raw Sofascore shotmap to all_shots_with_xg rows.

    Pure and network-free. ``sofascore_id`` becomes ``match_id`` (the file is
    natively Sofascore-keyed; consumers bridge to canonical via
    match_id_mapping.parquet).
    """
    rows = []
    for s in shots:
        pc = s.get("playerCoordinates") or {}
        gm = s.get("goalMouthCoordinates") or {}
        player = s.get("player") or {}
        x, y = pc.get("x"), pc.get("y")
        situation = s.get("situation")
        body_part = s.get("bodyPart")
        shot_type = s.get("shotType")
        xg = s.get("xg")

        if x is not None and y is not None:
            distance = (x ** 2 + (y - 50) ** 2) ** 0.5
            angle = math.degrees(math.atan2(abs(y - 50), x))
        else:
            distance = angle = None

        rows.append({
            "season": season,
            "match_id": str(sofascore_id),
            "is_home": bool(s.get("isHome")),
            "player_id": player.get("id"),
            "player_name": player.get("name"),
            "shot_x": x,
            "shot_y": y,
            "gm_x": gm.get("x"),
            "gm_y": gm.get("y"),
            "gm_z": gm.get("z"),
            "situation": situation,
            "body_part": body_part,
            "shot_type": shot_type,
            "xg": xg,
            "xgot": s.get("xgot"),
            "is_goal": int(shot_type == "goal"),
            "time": s.get("time"),
            "distance": distance,
            "angle": angle,
            "is_header": int(body_part == "head"),
            "is_right": int(body_part == "right-foot"),
            "is_left": int(body_part == "left-foot"),
            "is_penalty": int(situation == "penalty"),
            "is_freekick": int(situation == "free-kick"),
            "is_set_piece": int(situation in _SET_PIECE),
            "is_fast_break": int(situation == "fast-break"),
            "xg_predicted": xg,  # see module docstring — never read for 2025-26
        })
    return rows


def _shots_from_cache_file(path: Path) -> list[dict]:
    """Pull the shot list out of a cached match json, tolerating both shapes."""
    try:
        d = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        log.warning("unreadable cache json %s: %s", path.name, e)
        return []
    sm = d.get("shotmap")
    if isinstance(sm, dict):
        return sm.get("shotmap", []) or []
    if isinstance(sm, list):
        return sm
    return []


def rebuild_from_cache(season: str, league: str = "serie_a") -> int:
    """Re-derive shot-level rows for one season from the cached shotmaps.

    Idempotent: replaces exactly this season's rows in the target parquet, leaving
    every other season byte-for-byte untouched. Returns the number of matches
    written. The filename stem of each cache file is the authoritative
    Sofascore id (never trust an id inside the json).
    """
    cdir = cache_dir(season, league)
    if not cdir.exists():
        log.error("no cache dir for season %s: %s", season, cdir)
        return 0

    rows: list[dict] = []
    files = sorted(cdir.glob("*.json"))
    empty = 0
    for f in files:
        shots = _shots_from_cache_file(f)
        if not shots:
            empty += 1
            continue
        rows.extend(shot_rows_from_shotmap(shots, f.stem, season))

    if not rows:
        log.warning("no shots parsed for %s (%d cache files, %d empty)",
                    season, len(files), empty)
        return 0

    new_df = pd.DataFrame(rows)
    path = out_path(league)

    if path.exists():
        existing = pd.read_parquet(path)
        existing = existing[existing["season"] != season]
        for c in new_df.columns:
            if c not in existing.columns:
                existing[c] = None
        for c in existing.columns:
            if c not in new_df.columns:
                new_df[c] = None
        new_df = new_df[existing.columns]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    atomic_write_parquet(path, combined, index=False)
    n_matches = new_df["match_id"].nunique()
    log.info(
        "%s: wrote %d shots across %d matches (%d cache files, %d had no shotmap) "
        "-> %s now %d rows",
        season, len(new_df), n_matches, len(files), empty, path.name, len(combined),
    )
    return n_matches


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", required=True, help="e.g. 2025-2026")
    ap.add_argument("--league", default="serie_a")
    args = ap.parse_args(argv)
    rebuild_from_cache(args.season, args.league)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
