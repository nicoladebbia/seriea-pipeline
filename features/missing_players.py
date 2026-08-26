"""Phase 0b.3 — Match-time missing players.

Current `home_suspended_count` is 92% NaN. That column is fed by
weekly-snapshot injuries data which is stale by match-time.

Sofascore raw match JSONs (data/external/sofascore/matches/*/*.json) contain
`home_lineup.missing_players` — a 100%-filled, match-time-accurate list of
absent players with player ID, reason, and for recent seasons the injury
description.

This plugin walks all match JSONs, builds a per-match missing-players count,
then joins via `match_id_mapping.parquet`. Features produced:

  home/away_missing_count           — total missing players
  home/away_missing_doubtful_count  — "doubtful" subset
  home/away_missing_suspended_count — suspension subset (type="missing" w/ reason 4)
  home/away_missing_injury_count    — injury subset (description contains "injury"/"ACL"/"knee"/etc.)
  home/away_key_missing_count       — intersection with top-N xG contributors

Features written at match level (not rolling) — this is a *this-match-only*
signal, not a form signal.

Extractor runs once per build; results cached to
data/parsed/missing_players.parquet for incremental updates.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOFASCORE_MATCHES_DIR = PROJECT_ROOT / "data" / "external" / "sofascore" / "matches"
SOFASCORE_MATCHES_DIRS = (
    (SOFASCORE_MATCHES_DIR, "serie_a"),
    (PROJECT_ROOT / "data" / "external" / "sofascore" / "matches_premier_league", "premier_league"),
)
MAPPING_PATH = PROJECT_ROOT / "data" / "parsed" / "match_id_mapping.parquet"
PLAYER_STATS_PATH = PROJECT_ROOT / "data" / "external" / "sofascore" / "player_match_stats.parquet"
CACHE_PATH = PROJECT_ROOT / "data" / "parsed" / "missing_players.parquet"


def _classify(reason_text: str | None, mp_type: str | None) -> dict[str, bool]:
    """Return flags: injury, suspended, doubtful."""
    txt = (reason_text or "").lower()
    t = (mp_type or "").lower()
    is_doubtful = (t == "doubtful")
    # Sofascore `type` values observed: "missing", "doubtful".
    # Reason descriptions include 'injury', 'suspension', 'knee', 'ACL', 'muscle', etc.
    is_injury = any(k in txt for k in ("injury", "acl", "knee", "muscle", "hamstring",
                                       "ankle", "shoulder", "concussion", "fracture",
                                       "calf", "groin", "thigh", "achilles", "surgery"))
    is_suspended = any(k in txt for k in ("suspension", "suspended", "red card",
                                          "yellow card accumulation", "banned"))
    return {"injury": is_injury, "suspended": is_suspended, "doubtful": is_doubtful}


def _parse_match_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return None

    sofa_id = str(d.get("match_id", path.stem))
    out = {"sofascore_id": sofa_id}
    for side_key, prefix in [("home_lineup", "home"), ("away_lineup", "away")]:
        lineup = d.get(side_key, {}) or {}
        mp_list = lineup.get("missing_players", []) or []
        count = 0
        injury = 0
        suspended = 0
        doubtful = 0
        player_ids: list[int] = []
        for entry in mp_list:
            if not isinstance(entry, dict):
                continue
            count += 1
            tp = entry.get("type")
            desc = entry.get("description")
            flags = _classify(desc, tp)
            injury += int(flags["injury"])
            suspended += int(flags["suspended"])
            doubtful += int(flags["doubtful"])
            pl = entry.get("player", {})
            if isinstance(pl, dict) and "id" in pl:
                player_ids.append(int(pl["id"]))
        out[f"{prefix}_missing_count"] = count
        out[f"{prefix}_missing_injury_count"] = injury
        out[f"{prefix}_missing_suspended_count"] = suspended
        out[f"{prefix}_missing_doubtful_count"] = doubtful
        out[f"{prefix}_missing_player_ids"] = player_ids
    return out


def _walk_match_jsons(seen: dict[str, float] | None = None) -> pd.DataFrame:
    """Parse Sofascore match JSONs -> one row per (sofascore_id, match).

    Walks BOTH league directories. The EPL lives in a sibling top-level dir,
    not a subdirectory, so the old `"premier_league" in season_dir.name` guard
    never matched anything and the EPL was simply never scanned -- it had none
    of these columns at all.

    `seen` maps sofascore_id -> the file mtime that produced the cached row.
    A file is re-parsed when it is unknown or has been rewritten since; Sofascore
    rewrites a match JSON after kickoff as lineups and absences are confirmed,
    so "already cached" is not the same as "still accurate".
    """
    seen = seen or {}
    records = []
    n_parsed = 0
    n_current = 0
    for base, league in SOFASCORE_MATCHES_DIRS:
        if not base.exists():
            log.warning("Missing players: %s not found", base.name)
            continue
        for season_dir in sorted(base.iterdir()):
            if not season_dir.is_dir() or season_dir.name.startswith("."):
                continue
            for json_path in sorted(season_dir.glob("*.json")):
                mtime = json_path.stat().st_mtime
                cached_mtime = seen.get(json_path.stem)
                if cached_mtime is not None and mtime <= cached_mtime:
                    n_current += 1
                    continue
                rec = _parse_match_json(json_path)
                if rec is None:
                    continue
                rec["season"] = season_dir.name.replace("_", "-")  # e.g. 2024-2025
                rec["league"] = league
                rec["source_mtime"] = mtime
                records.append(rec)
                n_parsed += 1
    log.info("Missing players: parsed %d match JSONs (%d already current)", n_parsed, n_current)
    return pd.DataFrame(records)


def _ensure_cache() -> pd.DataFrame:
    """Return the cache, refreshing any match that is new or has been rewritten.

    This used to return the parquet verbatim whenever it existed, so it froze on
    the day it was first built (2026-04-22) and every match played since was
    invisible: 50 of 2025-2026 and all of 2026-2027 were absent, while 1,866 of
    the rows it did hold had been parsed from JSONs Sofascore has since rewritten.

    The watermark lives in the data (`source_mtime` per row), not on the cache
    file. Comparing against the cache file's own mtime would skip forever any
    JSON rewritten between two cache writes.
    """
    cached = pd.read_parquet(CACHE_PATH) if CACHE_PATH.exists() else pd.DataFrame()
    seen: dict[str, float] = {}
    if len(cached) and "source_mtime" in cached.columns:
        seen = dict(zip(cached["sofascore_id"].astype(str), cached["source_mtime"]))

    fresh = _walk_match_jsons(seen)
    if fresh.empty:
        return cached

    # list-valued columns are not parquet-friendly and nothing downstream reads them
    fresh = fresh.drop(columns=[c for c in fresh.columns if c.endswith("_missing_player_ids")])
    fresh["sofascore_id"] = fresh["sofascore_id"].astype(str)
    if len(cached):
        cached["sofascore_id"] = cached["sofascore_id"].astype(str)
        merged = pd.concat([cached, fresh], ignore_index=True)
    else:
        merged = fresh
    merged = merged.drop_duplicates(subset="sofascore_id", keep="last").reset_index(drop=True)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(CACHE_PATH, index=False)
    log.info("Missing players: cache now %d matches (%d refreshed)", len(merged), len(fresh))
    return merged


def add_missing_players_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    df_mp = _ensure_cache()
    if df_mp is None or len(df_mp) == 0:
        log.warning("Missing players: cache empty, skipping")
        return feature_df
    if not MAPPING_PATH.exists():
        log.warning("Missing players: mapping unavailable, skipping")
        return feature_df

    mapping = pd.read_parquet(MAPPING_PATH)[["match_id", "sofascore_id"]].dropna()
    mapping["sofascore_id"] = mapping["sofascore_id"].astype(str)
    df_mp["sofascore_id"] = df_mp["sofascore_id"].astype(str)

    joined = df_mp.merge(mapping, on="sofascore_id", how="inner")
    drop_side_cols = ["season"] if "season" in joined.columns else []
    if drop_side_cols:
        joined = joined.drop(columns=drop_side_cols)

    keep_cols = [c for c in joined.columns
                 if c.startswith("home_missing_") or c.startswith("away_missing_")
                 or c == "match_id"]
    joined = joined[keep_cols]

    before = len(feature_df.columns)
    feature_df = feature_df.merge(joined, on="match_id", how="left")
    added = len(feature_df.columns) - before
    log.info("Missing players: added %d columns", added)
    return feature_df
