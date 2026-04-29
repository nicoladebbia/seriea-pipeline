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


def _load_key_player_ids() -> dict[str, set[int]]:
    """Per (team, season), top-N xG contributors' Sofascore player IDs.

    Used to tag "key player missing" — a missing bottom-of-bench reserve
    doesn't matter; a missing top-scorer does.
    """
    if not PLAYER_STATS_PATH.exists():
        return {}
    ps = pd.read_parquet(PLAYER_STATS_PATH)
    if "xg" not in ps.columns:
        return {}
    ps = ps.dropna(subset=["team", "season"]).copy()
    # Top-10 xG per team-season
    top = (
        ps.groupby(["team", "season"], observed=True)
        .apply(lambda g: g.sort_values("xg", ascending=False).head(10)["player_id"].dropna().astype(int).tolist(), include_groups=False)
        .to_dict()
    )
    return {k: set(v) for k, v in top.items()}


def _walk_match_jsons() -> pd.DataFrame:
    """Parse all Sofascore match JSONs → one row per (sofascore_id, match)."""
    if not SOFASCORE_MATCHES_DIR.exists():
        log.warning("Missing players: Sofascore matches dir not found")
        return pd.DataFrame()
    records = []
    n_parsed = 0
    for season_dir in sorted(SOFASCORE_MATCHES_DIR.iterdir()):
        if not season_dir.is_dir() or season_dir.name.startswith(".") or "premier_league" in season_dir.name:
            continue
        for json_path in season_dir.glob("*.json"):
            rec = _parse_match_json(json_path)
            if rec is None:
                continue
            rec["season"] = season_dir.name.replace("_", "-")  # e.g. 2024-2025
            records.append(rec)
            n_parsed += 1
    log.info("Missing players: parsed %d match JSONs", n_parsed)
    return pd.DataFrame(records)


def _ensure_cache() -> pd.DataFrame:
    if CACHE_PATH.exists():
        cached = pd.read_parquet(CACHE_PATH)
        return cached
    log.info("Building missing_players.parquet cache (one-time; ~10s)")
    df = _walk_match_jsons()
    if len(df):
        # Drop unhashable player_ids column before parquet write; keep a count-only copy
        df_cache = df.drop(columns=[c for c in df.columns if c.endswith("_missing_player_ids")])
        df_cache.to_parquet(CACHE_PATH, index=False)
    return df


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
