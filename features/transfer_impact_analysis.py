"""Transfer impact analysis module.

Goes beyond simple transfer counts to measure ACTUAL impact:
- Pre vs post transfer window performance comparison
- New signing integration curves (position-specific)
- Squad disruption from key departures
- January window feature generation

Data sources:
- scraper/transfermarkt.py → data/external/transfermarkt/
- data/parsed/matches.parquet → pre/post performance
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import DATA_DIR

log = logging.getLogger(__name__)

# Position-specific integration curves
# Tuple = effectiveness at game 1, 3, 5, 7+
INTEGRATION_CURVES = {
    "GK": (0.50, 0.80, 0.95, 1.0),   # GK integrates fastest
    "CB": (0.40, 0.70, 0.90, 1.0),
    "DF": (0.40, 0.70, 0.90, 1.0),
    "FB": (0.35, 0.65, 0.85, 1.0),
    "LB": (0.35, 0.65, 0.85, 1.0),
    "RB": (0.35, 0.65, 0.85, 1.0),
    "DM": (0.30, 0.60, 0.85, 1.0),
    "CM": (0.30, 0.60, 0.85, 1.0),
    "MF": (0.30, 0.60, 0.85, 1.0),
    "AM": (0.30, 0.55, 0.80, 1.0),
    "LW": (0.30, 0.55, 0.80, 1.0),
    "RW": (0.30, 0.55, 0.80, 1.0),
    "FW": (0.30, 0.55, 0.80, 1.0),    # FW takes longest
    "CF": (0.30, 0.55, 0.80, 1.0),
}


def analyze_transfer_impact(
    team: str,
    transfer_date: str,
    matches_df: pd.DataFrame,
    window_days: int = 30,
) -> dict:
    """Compare team performance pre vs post transfer window.

    Args:
        team: Team name
        transfer_date: ISO date string (e.g., "2026-01-15")
        matches_df: DataFrame with match data (must have 'date', team cols, scores)
        window_days: Days before/after to compare

    Returns:
        Dict with performance diff fields and confidence indicator.
    """
    try:
        td = pd.Timestamp(transfer_date)
    except Exception:
        return {"impact": 0.0, "confidence": "none", "error": "invalid_date"}

    # Normalize date column
    if "match_date" in matches_df.columns:
        date_col = "match_date"
    elif "date" in matches_df.columns:
        date_col = "date"
    else:
        return {"impact": 0.0, "confidence": "none", "error": "no_date_column"}

    df = matches_df.copy()
    df["_date"] = pd.to_datetime(df[date_col], errors="coerce")

    # Get team's matches
    team_mask = (
        (df.get("home_team", pd.Series(dtype=str)).str.lower() == team.lower()) |
        (df.get("away_team", pd.Series(dtype=str)).str.lower() == team.lower())
    )
    team_df = df[team_mask].copy()

    if team_df.empty:
        return {"impact": 0.0, "confidence": "none", "error": "no_matches"}

    # Pre-window matches
    pre_start = td - timedelta(days=window_days)
    pre = team_df[(team_df["_date"] >= pre_start) & (team_df["_date"] < td)]

    # Post-window matches
    post_end = td + timedelta(days=window_days)
    post = team_df[(team_df["_date"] > td) & (team_df["_date"] <= post_end)]

    if len(pre) < 3 or len(post) < 3:
        return {"impact": 0.0, "confidence": "low", "reason": "insufficient_data"}

    # Compute metrics for each window
    def compute_metrics(subset, team_name):
        goals = []
        xg = []
        points = []
        clean_sheets = 0

        for _, row in subset.iterrows():
            is_home = str(row.get("home_team", "")).lower() == team_name.lower()
            if is_home:
                g = pd.to_numeric(row.get("home_score"), errors="coerce")
                gc = pd.to_numeric(row.get("away_score"), errors="coerce")
                x = pd.to_numeric(row.get("home_xg"), errors="coerce")
            else:
                g = pd.to_numeric(row.get("away_score"), errors="coerce")
                gc = pd.to_numeric(row.get("home_score"), errors="coerce")
                x = pd.to_numeric(row.get("away_xg"), errors="coerce")

            if pd.notna(g):
                goals.append(g)
            if pd.notna(x):
                xg.append(x)
            if pd.notna(g) and pd.notna(gc):
                if g > gc:
                    points.append(3)
                elif g == gc:
                    points.append(1)
                else:
                    points.append(0)
                if gc == 0:
                    clean_sheets += 1

        n = max(len(goals), 1)
        return {
            "goals_per_game": sum(goals) / n if goals else 0,
            "xg_per_game": sum(xg) / max(len(xg), 1) if xg else 0,
            "points_per_game": sum(points) / max(len(points), 1) if points else 0,
            "clean_sheet_pct": clean_sheets / n if goals else 0,
            "matches": n,
        }

    pre_metrics = compute_metrics(pre, team)
    post_metrics = compute_metrics(post, team)

    return {
        "impact": round(post_metrics["points_per_game"] - pre_metrics["points_per_game"], 3),
        "goals_diff": round(post_metrics["goals_per_game"] - pre_metrics["goals_per_game"], 3),
        "xg_diff": round(post_metrics["xg_per_game"] - pre_metrics["xg_per_game"], 3),
        "points_diff": round(post_metrics["points_per_game"] - pre_metrics["points_per_game"], 3),
        "clean_sheet_diff": round(
            post_metrics["clean_sheet_pct"] - pre_metrics["clean_sheet_pct"], 3
        ),
        "pre_matches": pre_metrics["matches"],
        "post_matches": post_metrics["matches"],
        "confidence": "high" if len(pre) >= 5 and len(post) >= 5 else "medium",
    }


def compute_integration_curve(games_since_signing: int, position: str = "MF") -> float:
    """New signing effectiveness over time (position-specific).

    GK integrates fastest (0.50 at game 1), FW takes longest (0.30 at game 1).
    """
    if games_since_signing <= 0:
        return 0.30

    curve = INTEGRATION_CURVES.get(position, INTEGRATION_CURVES["MF"])

    if games_since_signing <= 2:
        return curve[0]
    elif games_since_signing <= 4:
        return curve[1]
    elif games_since_signing <= 6:
        return curve[2]
    else:
        return curve[3]


def compute_squad_disruption(
    departures_df: pd.DataFrame,
    team: str,
    season: str,
) -> float:
    """Measure disruption from key player departures.

    Args:
        departures_df: Transfer DataFrame with transfer_type, team, minutes columns
        team: Team name
        season: Season string

    Returns:
        disruption score 0-1
    """
    if departures_df.empty:
        return 0.0

    team_col = "team" if "team" in departures_df.columns else "squad"
    team_departures = departures_df[
        (departures_df[team_col].str.lower() == team.lower()) &
        (departures_df.get("transfer_type", pd.Series(dtype=str)) == "out")
    ]

    if team_departures.empty:
        return 0.0

    disruption = 0.0
    for _, dep in team_departures.iterrows():
        minutes = pd.to_numeric(dep.get("minutes_played", 0), errors="coerce")
        if pd.isna(minutes):
            minutes = 0

        if minutes > 1500:
            disruption += 0.25  # Key player (starter level)
        elif minutes > 500:
            disruption += 0.10  # Regular rotation player
        else:
            disruption += 0.02  # Minor departure

    return min(disruption, 1.0)


def compute_january_window_features(
    team: str,
    transfers_df: pd.DataFrame,
    matches_df: pd.DataFrame = None,
) -> dict:
    """Compute January window impact features for a team.

    Returns dict with:
    - jan_arrivals: count of January signings
    - jan_spend: January spending (EUR)
    - squad_disruption: key departure penalty (0-1)
    - signing_integration: average integration level of new signings (0.3-1.0)
    """
    if transfers_df.empty:
        return {
            "jan_arrivals": 0,
            "jan_spend": 0,
            "squad_disruption": 0.0,
            "signing_integration": 1.0,
        }

    team_col = "team" if "team" in transfers_df.columns else "squad"

    # Filter to this team's transfers
    team_transfers = transfers_df[
        transfers_df[team_col].str.lower() == team.lower()
    ].copy()

    if team_transfers.empty:
        return {
            "jan_arrivals": 0,
            "jan_spend": 0,
            "squad_disruption": 0.0,
            "signing_integration": 1.0,
        }

    # Filter to January window (Jan 1 - Jan 31)
    if "date" in team_transfers.columns:
        team_transfers["_date"] = pd.to_datetime(team_transfers["date"], errors="coerce")
        jan_mask = team_transfers["_date"].dt.month == 1
        jan_transfers = team_transfers[jan_mask]
    else:
        jan_transfers = team_transfers  # Assume all are January if no date

    # Arrivals
    type_col = "transfer_type" if "transfer_type" in jan_transfers.columns else "direction"
    arrivals = jan_transfers[jan_transfers.get(type_col, pd.Series(dtype=str)) == "in"]
    departures = jan_transfers[jan_transfers.get(type_col, pd.Series(dtype=str)) == "out"]

    # Spend
    fee_col = "fee_eur" if "fee_eur" in arrivals.columns else "fee"
    spend = pd.to_numeric(arrivals.get(fee_col, pd.Series(dtype=float)), errors="coerce").sum()
    if pd.isna(spend):
        spend = 0

    # Squad disruption from departures
    disruption = 0.0
    for _, dep in departures.iterrows():
        minutes = pd.to_numeric(dep.get("minutes_played", 0), errors="coerce")
        if pd.isna(minutes):
            minutes = 0
        if minutes > 1500:
            disruption += 0.25
        elif minutes > 500:
            disruption += 0.10
        else:
            disruption += 0.02
    disruption = min(disruption, 1.0)

    # Average integration level of new signings
    # Estimate games since signing based on current date vs Jan window
    if matches_df is not None and not arrivals.empty:
        # Count team matches since Jan 15 (mid-window estimate)
        jan_mid = pd.Timestamp(f"{datetime.now().year}-01-15")
        if "match_date" in matches_df.columns:
            m_dates = pd.to_datetime(matches_df["match_date"], errors="coerce")
            team_mask = (
                (matches_df.get("home_team", pd.Series(dtype=str)).str.lower() == team.lower()) |
                (matches_df.get("away_team", pd.Series(dtype=str)).str.lower() == team.lower())
            )
            games_since = ((m_dates > jan_mid) & team_mask).sum()
        else:
            games_since = 3  # Default estimate

        integration_scores = []
        for _, arr in arrivals.iterrows():
            pos = arr.get("position", "MF")
            integration_scores.append(compute_integration_curve(games_since, pos))

        avg_integration = sum(integration_scores) / max(len(integration_scores), 1)
    else:
        avg_integration = 0.65  # Mid-range default

    return {
        "jan_arrivals": len(arrivals),
        "jan_spend": int(spend),
        "squad_disruption": round(disruption, 3),
        "signing_integration": round(avg_integration, 3),
    }


def _build_squad_value_lookup(
    tm_dir: Path, file_prefix: str = ""
) -> dict[tuple[str, str], dict]:
    """Build a lookup of squad market value features per (team, season).

    Uses market_values_YYYY_YYYY.parquet files from Transfermarkt.
    Each file has: team, player_name, position, age, market_value_eur, nationality.

    Args:
        tm_dir: Directory containing Transfermarkt parquet files
        file_prefix: League prefix for files (e.g. "premier_league_"). Empty for Serie A.

    Returns dict mapping (team_lower, season) -> {squad_value, squad_avg_value,
    squad_avg_age, squad_depth}.
    """
    lookup: dict[tuple[str, str], dict] = {}

    glob_pattern = f"{file_prefix}market_values_*.parquet"
    for path in sorted(tm_dir.glob(glob_pattern)):
        if "backup" in path.name:
            continue
        # Extract season from filename: market_values_2024_2025.parquet -> 2024-2025
        # or premier_league_market_values_2024_2025.parquet -> 2024-2025
        stem = path.stem
        if file_prefix:
            stem = stem.replace(file_prefix, "", 1)
        parts = stem.replace("market_values_", "").split("_")
        if len(parts) != 2:
            continue
        season = f"{parts[0]}-{parts[1]}"

        try:
            mv = pd.read_parquet(path)
        except Exception:
            continue

        if mv.empty or "team" not in mv.columns or "market_value_eur" not in mv.columns:
            continue

        mv["market_value_eur"] = pd.to_numeric(mv["market_value_eur"], errors="coerce")
        mv["age"] = pd.to_numeric(mv.get("age", pd.Series(dtype=float)), errors="coerce")

        # Filter to realistic ages (source data has noise: ages 1, 99, etc.)
        mv.loc[(mv["age"] < 16) | (mv["age"] > 42), "age"] = np.nan

        # Filter to players with non-zero market value (removes duplicate/noise rows)
        mv_valid = mv[mv["market_value_eur"] > 0].copy()

        for team, grp in mv_valid.groupby("team"):
            vals = grp["market_value_eur"].dropna()
            ages = grp["age"].dropna()
            team_key = team.lower().strip()

            lookup[(team_key, season)] = {
                "squad_value": int(vals.sum()) if len(vals) > 0 else 0,
                "squad_avg_value": int(vals.mean()) if len(vals) > 0 else 0,
                "squad_avg_age": round(float(ages.mean()), 1) if len(ages) > 0 else 0.0,
                "squad_depth": len(grp),
            }

    log.info("Built squad value lookup: %d team-seasons", len(lookup))
    return lookup


def add_transfer_impact_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Add transfer impact features to the match-level DataFrame.

    Adds 16 columns:
    - home_jan_arrivals / away_jan_arrivals
    - home_jan_spend / away_jan_spend
    - home_squad_disruption / away_squad_disruption
    - home_signing_integration / away_signing_integration
    - home_squad_value / away_squad_value
    - home_squad_avg_age / away_squad_avg_age
    - squad_value_diff / squad_value_ratio
    """
    df = feature_df.copy()

    # Initialize with defaults
    for prefix in ("home", "away"):
        df[f"{prefix}_jan_arrivals"] = 0
        df[f"{prefix}_jan_spend"] = 0
        df[f"{prefix}_squad_disruption"] = 0.0
        df[f"{prefix}_signing_integration"] = 1.0
        df[f"{prefix}_squad_value"] = np.nan
        df[f"{prefix}_squad_avg_age"] = np.nan

    # Load transfers
    tm_dir = DATA_DIR / "external" / "transfermarkt"
    if not tm_dir.exists():
        log.info("No Transfermarkt data available — transfer features set to defaults")
        return df

    # Determine league prefix for file lookup (Serie A = no prefix for backward compat)
    league_key = ""
    if "league" in df.columns:
        leagues = df["league"].dropna().unique()
        if len(leagues) == 1:
            league_key = str(leagues[0])
    file_prefix = "" if league_key in ("serie_a", "") else f"{league_key}_"

    # ── Squad market value features ──────────────────────────────────
    squad_lookup = _build_squad_value_lookup(tm_dir, file_prefix=file_prefix)
    if squad_lookup:
        for idx, row in df.iterrows():
            season = row.get("season", "")
            for prefix in ("home", "away"):
                team = str(row.get(f"{prefix}_team", "")).lower().strip()
                sv = squad_lookup.get((team, season))
                if sv:
                    df.at[idx, f"{prefix}_squad_value"] = sv["squad_value"]
                    df.at[idx, f"{prefix}_squad_avg_age"] = sv["squad_avg_age"]

        # Relative squad value features (vectorized)
        h_val = pd.to_numeric(df["home_squad_value"], errors="coerce")
        a_val = pd.to_numeric(df["away_squad_value"], errors="coerce")
        df["squad_value_diff"] = h_val - a_val
        df["squad_value_ratio"] = h_val / a_val.replace(0, np.nan)

        n_sv = df["home_squad_value"].notna().sum()
        log.info("Squad value features: %d/%d matches with data (%.0f%%)",
                 n_sv, len(df), 100 * n_sv / max(len(df), 1))

    # ── January transfer window features ─────────────────────────────
    # Load matches for integration calculation
    matches_path = DATA_DIR / "parsed" / "matches.parquet"
    matches_df = pd.read_parquet(matches_path) if matches_path.exists() else None

    # Process per season
    seasons = df["season"].unique() if "season" in df.columns else []
    transfer_cache: dict[tuple[str, str], dict] = {}

    for season in seasons:
        tr_path = tm_dir / f"{file_prefix}transfers_{season.replace('-', '_')}.parquet"
        if not tr_path.exists():
            continue

        try:
            transfers = pd.read_parquet(tr_path)
        except Exception:
            continue

        season_mask = df["season"] == season
        teams = set(
            df.loc[season_mask, "home_team"].dropna().unique().tolist() +
            df.loc[season_mask, "away_team"].dropna().unique().tolist()
        )

        for team in teams:
            cache_key = (team, season)
            if cache_key not in transfer_cache:
                transfer_cache[cache_key] = compute_january_window_features(
                    team, transfers, matches_df
                )

    # Apply cached features
    for idx, row in df.iterrows():
        season = row.get("season", "")
        for prefix in ("home", "away"):
            team = row.get(f"{prefix}_team", "")
            features = transfer_cache.get((team, season), {})
            if features:
                df.at[idx, f"{prefix}_jan_arrivals"] = features.get("jan_arrivals", 0)
                df.at[idx, f"{prefix}_jan_spend"] = features.get("jan_spend", 0)
                df.at[idx, f"{prefix}_squad_disruption"] = features.get("squad_disruption", 0)
                df.at[idx, f"{prefix}_signing_integration"] = features.get(
                    "signing_integration", 1.0
                )

    n_with_data = (df["home_jan_arrivals"] + df["away_jan_arrivals"] > 0).sum()
    log.info(f"Added transfer impact features ({n_with_data} matches with transfer data, "
             f"{len(squad_lookup)} team-seasons with squad value)")

    return df


def validate_january_transfer_predictions() -> dict:
    """Compare predicted transfer impact vs actual for validation.

    Loads January transfers and compares pre/post team performance.
    Returns correlation and bias metrics.
    """
    tm_dir = DATA_DIR / "external" / "transfermarkt"
    matches_path = DATA_DIR / "parsed" / "matches.parquet"

    if not matches_path.exists():
        return {"error": "No matches data for validation"}

    matches_df = pd.read_parquet(matches_path)

    # Find most recent season with transfer data
    results = []
    for season in ["2024-2025", "2023-2024"]:
        tr_path = tm_dir / f"transfers_{season.replace('-', '_')}.parquet"
        if not tr_path.exists():
            continue

        transfers = pd.read_parquet(tr_path)
        if transfers.empty:
            continue

        # Get teams that made January signings
        team_col = "team" if "team" in transfers.columns else "squad"
        teams = transfers[team_col].unique()

        year = int(season.split("-")[1])
        jan_date = f"{year}-01-15"

        for team in teams:
            impact = analyze_transfer_impact(team, jan_date, matches_df)
            if impact.get("confidence") in ("high", "medium"):
                results.append({
                    "team": team,
                    "season": season,
                    **impact,
                })

    if not results:
        return {"error": "No sufficient data for validation", "results": []}

    # Compute summary statistics
    impacts = [r.get("impact", 0) for r in results]
    return {
        "total_analyzed": len(results),
        "avg_impact": round(np.mean(impacts), 3) if impacts else 0,
        "positive_impact_pct": round(sum(1 for i in impacts if i > 0) / max(len(impacts), 1), 3),
        "results": results,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing transfer impact analysis...")
    print("=" * 60)

    matches_path = DATA_DIR / "parsed" / "matches.parquet"
    if matches_path.exists():
        matches = pd.read_parquet(matches_path)
        print(f"Loaded {len(matches)} matches")

        impact = analyze_transfer_impact("Napoli", "2025-01-15", matches)
        print(f"\nNapoli January 2025 impact: {impact}")

    print(f"\nIntegration curve (FW, game 1): {compute_integration_curve(1, 'FW')}")
    print(f"Integration curve (FW, game 5): {compute_integration_curve(5, 'FW')}")
    print(f"Integration curve (GK, game 1): {compute_integration_curve(1, 'GK')}")

    print("\n" + "=" * 60)
