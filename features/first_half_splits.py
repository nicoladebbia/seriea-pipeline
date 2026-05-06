"""Phase 0b.4 — First-half / second-half splits from Sofascore match JSONs.

Each Sofascore raw match JSON has team_stats.statistics per period:
  - period "ALL"  (full match — currently extracted to match_team_stats.parquet)
  - period "1ST"  (first half)    ← NOT EXTRACTED
  - period "2ND"  (second half)   ← NOT EXTRACTED

We extract 1ST splits → rolling features. Markets unlocked: FH O/U, HT result,
FH team totals, FH BTTS.

Stats extracted (per half, per side):
  goals-proxy: xg, total shots, shots on target
  discipline: yellow cards, fouls
  possession: ball possession, accurate passes, final third entries
  corners: corner kicks

Output features (home & away versions for each):
  home_fh_xg_roll_{5,10}
  home_fh_shots_roll_{5,10}
  home_fh_sot_roll_{5,10}
  home_fh_cards_roll_{5,10}
  home_fh_corners_roll_{5,10}
  home_fh_possession_roll_{5,10}
  home_fh_goals_prop  — ratio of fh goals to full-match goals (for simulator split)

Cached to data/parsed/first_half_splits.parquet.
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
CACHE_PATH = PROJECT_ROOT / "data" / "parsed" / "first_half_splits.parquet"

# Sofascore key → our canonical stat name
KEY_MAP = {
    "expectedGoals": "xg",
    "totalShotsOnGoal": "total_shots",   # cryptic name but it's TOTAL shots, yes
    "shotsOnGoal": "sot",
    "yellowCards": "yellow_cards",
    "fouls": "fouls",
    "cornerKicks": "corners",
    "ballPossession": "possession",
    "accuratePasses": "accurate_passes",
    "finalThirdEntries": "final_third_entries",
}

WINDOWS = (5, 10)


def _extract_period_stats(match_json: dict, period_label: str = "1ST") -> dict | None:
    """Pull per-team stats for a given period from `team_stats.statistics`."""
    ts = match_json.get("team_stats") or match_json.get("statistics")
    if not isinstance(ts, dict):
        return None
    periods = ts.get("statistics")
    if not isinstance(periods, list):
        return None
    for period in periods:
        if not isinstance(period, dict):
            continue
        if period.get("period") != period_label:
            continue
        home_stats = {}
        away_stats = {}
        for group in period.get("groups", []):
            for stat in group.get("statisticsItems", []):
                key = stat.get("key")
                if key not in KEY_MAP:
                    continue
                canonical = KEY_MAP[key]
                hv = stat.get("homeValue")
                av = stat.get("awayValue")
                if hv is not None:
                    try: home_stats[canonical] = float(hv)
                    except (TypeError, ValueError): pass
                if av is not None:
                    try: away_stats[canonical] = float(av)
                    except (TypeError, ValueError): pass
        return {"home": home_stats, "away": away_stats}
    return None


def _parse_match_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return None
    sofa_id = str(d.get("match_id", path.stem))
    fh = _extract_period_stats(d, "1ST")
    full = _extract_period_stats(d, "ALL")
    if fh is None:
        return None
    rec: dict = {"sofascore_id": sofa_id}
    for side in ("home", "away"):
        for canonical in KEY_MAP.values():
            v = fh.get(side, {}).get(canonical)
            rec[f"{side}_fh_{canonical}"] = v
            if full is not None:
                fv = full.get(side, {}).get(canonical)
                # Goals proxy: xg prop = fh_xg / full_xg
                if canonical == "xg" and fv and fv > 0 and v is not None:
                    rec[f"{side}_fh_xg_prop"] = v / fv
    return rec


def _walk_match_jsons() -> pd.DataFrame:
    if not SOFASCORE_MATCHES_DIR.exists():
        return pd.DataFrame()
    records = []
    for season_dir in sorted(SOFASCORE_MATCHES_DIR.iterdir()):
        if not season_dir.is_dir() or season_dir.name.startswith(".") or "premier_league" in season_dir.name:
            continue
        for json_path in season_dir.glob("*.json"):
            rec = _parse_match_json(json_path)
            if rec is not None:
                records.append(rec)
    return pd.DataFrame(records)


def _ensure_cache() -> pd.DataFrame:
    if CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)
    log.info("Building first_half_splits.parquet cache (one-time; ~20s)")
    df = _walk_match_jsons()
    if len(df):
        df.to_parquet(CACHE_PATH, index=False)
    return df


def _rolling_per_team(per_match_team: pd.DataFrame) -> pd.DataFrame:
    """Reshape per-match (home/away) rows into per-team-match rows and roll."""
    # Build team-match rows from home and away perspective
    home = per_match_team.copy()
    away = per_match_team.copy()
    home["team_is_home"] = True
    away["team_is_home"] = False
    records = []
    for _, row in per_match_team.iterrows():
        for prefix, team_key in [("home", "home_team"), ("away", "away_team")]:
            rec = {
                "match_id": row["match_id"],
                "match_date": row["match_date"],
                "season": row["season"],
                "league": row["league"],
                "team": row[team_key],
                "is_home": (prefix == "home"),
            }
            for canonical in KEY_MAP.values():
                rec[f"fh_{canonical}"] = row.get(f"{prefix}_fh_{canonical}")
            records.append(rec)
    if not records:
        # Empty input → return empty frame with the columns downstream expects
        # rather than raising KeyError on sort_values.
        cols = ["match_id", "match_date", "season", "league", "team", "is_home"]
        cols += [f"fh_{c}" for c in KEY_MAP.values()]
        cols += [f"fh_{c}_roll_{N}" for N in WINDOWS for c in KEY_MAP.values()]
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(records).sort_values(["team", "match_date", "match_id"])

    for N in WINDOWS:
        for canonical in KEY_MAP.values():
            col_src = f"fh_{canonical}"
            col_dst = f"fh_{canonical}_roll_{N}"
            df[col_dst] = (
                df.groupby("team", observed=True)[col_src]
                .transform(lambda s: s.shift(1).rolling(N, min_periods=1).mean())
            )
    return df


def _pivot_home_away(rolled: pd.DataFrame) -> pd.DataFrame:
    records = []
    for mk, group in rolled.groupby("match_id", observed=True):
        home = group[group["is_home"] == True]  # noqa
        away = group[group["is_home"] == False]  # noqa
        rec = {"match_id": mk}
        for N in WINDOWS:
            for canonical in KEY_MAP.values():
                col = f"fh_{canonical}_roll_{N}"
                rec[f"home_{col}"] = float(home[col].iloc[0]) if len(home) and pd.notna(home[col].iloc[0]) else np.nan
                rec[f"away_{col}"] = float(away[col].iloc[0]) if len(away) and pd.notna(away[col].iloc[0]) else np.nan
        records.append(rec)
    return pd.DataFrame(records)


def add_first_half_splits_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    df_fh = _ensure_cache()
    if df_fh is None or len(df_fh) == 0:
        log.warning("First-half splits: cache empty, skipping")
        return feature_df
    if not MAPPING_PATH.exists():
        log.warning("First-half splits: mapping unavailable, skipping")
        return feature_df

    mapping = pd.read_parquet(MAPPING_PATH)[["match_id", "sofascore_id"]].dropna()
    mapping["sofascore_id"] = mapping["sofascore_id"].astype(str)
    df_fh["sofascore_id"] = df_fh["sofascore_id"].astype(str)

    req = {"match_id", "home_team", "away_team", "match_date", "season", "league"}
    if not req.issubset(feature_df.columns):
        log.warning("First-half splits: feature_df missing %s", req - set(feature_df.columns))
        return feature_df

    per_match = df_fh.merge(mapping, on="sofascore_id", how="inner")
    per_match = per_match.merge(feature_df[list(req)].drop_duplicates(subset=["match_id"]),
                                on="match_id", how="inner")
    if per_match.empty:
        log.warning("First-half splits: no matches after mapping+features join "
                    "(df_fh=%d, mapping=%d, features=%d). Cache may be stale or "
                    "match_id_mapping doesn't cover recent fixtures yet.",
                    len(df_fh), len(mapping), len(feature_df))
        return feature_df

    rolled = _rolling_per_team(per_match)
    pivoted = _pivot_home_away(rolled)
    before = len(feature_df.columns)
    feature_df = feature_df.merge(pivoted, on="match_id", how="left")
    log.info("First-half splits: added %d columns", len(feature_df.columns) - before)
    return feature_df
