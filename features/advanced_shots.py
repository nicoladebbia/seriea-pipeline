"""Advanced shot-level features: creation chains, spatial patterns, finishing.

Goes beyond basic shot_quality.py by extracting deeper patterns from
per-shot event data in shots.parquet:

  A. Shot Creation Chain Analysis
     - What types of plays create shots (live pass, dead ball, take-on, etc.)
     - Two-step buildup patterns (e.g., take-on → pass → shot)

  B. Shot Location & Quality Tiers
     - Big chance ratio (xG > 0.15)
     - Close-range ratio (distance < 12 yards)
     - Long-range ratio (distance > 25 yards)
     - xG distribution skewness (few big chances vs many small)

  C. Finishing Efficiency
     - Conversion rate (goals / shots)
     - Goals minus xG (overperformance)
     - Goals minus PSxG (beating the GK beyond post-shot quality)

  D. Body Part Patterns
     - Header ratio and header conversion
     - Preferred foot ratio (left vs right)

  E. Temporal Patterns
     - Early goal threat (shots in first 30 min)
     - Late pressure (shots in last 15 min)

All computed per team per match → rolling 5-match averages → home_*/away_*.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from storage.paths import parsed_path

log = logging.getLogger(__name__)


def _safe_div(a, b, default: float = 0.0) -> float:
    """Safe scalar division."""
    if b == 0 or pd.isna(b):
        return default
    return a / b


def _compute_advanced_shot_metrics(shots: pd.DataFrame) -> pd.DataFrame:
    """Compute advanced shot-level metrics per team per match.

    Returns DataFrame: match_id, team, + ~25 feature columns.
    """
    if "match_id" not in shots.columns or "team" not in shots.columns:
        return pd.DataFrame()

    # Ensure numeric
    for col in ["xg_shot", "psxg_shot", "distance", "minute"]:
        if col in shots.columns:
            shots[col] = pd.to_numeric(shots[col], errors="coerce")

    rows = []
    for (match_id, team), grp in shots.groupby(["match_id", "team"]):
        row: dict = {"match_id": match_id, "team": team}
        n = len(grp)
        if n == 0:
            continue

        xg_vals = grp["xg_shot"].dropna()
        psxg_vals = grp["psxg_shot"].dropna() if "psxg_shot" in grp.columns else pd.Series(dtype=float)
        dist_vals = grp["distance"].dropna() if "distance" in grp.columns else pd.Series(dtype=float)
        minute_vals = grp["minute"].dropna() if "minute" in grp.columns else pd.Series(dtype=float)
        outcomes = grp["outcome"].str.lower().fillna("") if "outcome" in grp.columns else pd.Series(dtype=str)
        body_parts = grp["body_part"].str.lower().fillna("") if "body_part" in grp.columns else pd.Series(dtype=str)
        goals = outcomes.str.contains("goal").sum()

        # -----------------------------------------------------------
        # A. SHOT CREATION CHAIN ANALYSIS
        # -----------------------------------------------------------
        if "sca_1_type" in grp.columns:
            sca1 = grp["sca_1_type"].str.lower().fillna("")
            row["advshot_pct_from_live_pass"] = _safe_div(
                sca1.str.contains("pass.*live|live.*pass").sum(), n
            )
            row["advshot_pct_from_dead_ball"] = _safe_div(
                sca1.str.contains("dead").sum(), n
            )
            row["advshot_pct_from_take_on"] = _safe_div(
                sca1.str.contains("take-on|take on|dribble").sum(), n
            )
            row["advshot_pct_from_shot"] = _safe_div(
                sca1.str.contains("^shot$").sum(), n
            )  # rebound
        else:
            row["advshot_pct_from_live_pass"] = 0.0
            row["advshot_pct_from_dead_ball"] = 0.0
            row["advshot_pct_from_take_on"] = 0.0
            row["advshot_pct_from_shot"] = 0.0

        # Two-step chain: what leads to the assist?
        if "sca_2_type" in grp.columns:
            sca2 = grp["sca_2_type"].str.lower().fillna("")
            has_sca2 = (sca2 != "").sum()
            row["advshot_pct_multi_step_buildup"] = _safe_div(has_sca2, n)
        else:
            row["advshot_pct_multi_step_buildup"] = 0.0

        # -----------------------------------------------------------
        # B. SHOT LOCATION & QUALITY TIERS
        # -----------------------------------------------------------
        if len(xg_vals) > 0:
            row["advshot_big_chance_ratio"] = _safe_div(
                (xg_vals > 0.15).sum(), n
            )
            row["advshot_very_big_chance_ratio"] = _safe_div(
                (xg_vals > 0.30).sum(), n
            )
            row["advshot_speculative_ratio"] = _safe_div(
                (xg_vals < 0.05).sum(), n
            )
            # xG distribution: std captures consistency (low = consistent quality)
            row["advshot_xg_std"] = float(xg_vals.std()) if len(xg_vals) > 1 else 0.0
            # Total xG (already exists but needed for derived features)
            row["advshot_total_xg"] = float(xg_vals.sum())
        else:
            row["advshot_big_chance_ratio"] = 0.0
            row["advshot_very_big_chance_ratio"] = 0.0
            row["advshot_speculative_ratio"] = 0.0
            row["advshot_xg_std"] = 0.0
            row["advshot_total_xg"] = 0.0

        if len(dist_vals) > 0:
            row["advshot_close_range_ratio"] = _safe_div(
                (dist_vals < 12).sum(), n
            )
            row["advshot_long_range_ratio"] = _safe_div(
                (dist_vals > 25).sum(), n
            )
            row["advshot_distance_std"] = float(dist_vals.std()) if len(dist_vals) > 1 else 0.0
        else:
            row["advshot_close_range_ratio"] = 0.0
            row["advshot_long_range_ratio"] = 0.0
            row["advshot_distance_std"] = 0.0

        # -----------------------------------------------------------
        # C. FINISHING EFFICIENCY
        # -----------------------------------------------------------
        row["advshot_conversion_rate"] = _safe_div(goals, n)

        # Goals minus xG (overperformance: positive = clinical)
        if len(xg_vals) > 0:
            row["advshot_goals_minus_xg"] = goals - float(xg_vals.sum())
        else:
            row["advshot_goals_minus_xg"] = 0.0

        # Goals minus PSxG (beating the GK beyond shot quality)
        if len(psxg_vals) > 0:
            row["advshot_goals_minus_psxg"] = goals - float(psxg_vals.sum())
        else:
            row["advshot_goals_minus_psxg"] = 0.0

        # On-target rate (goals + saves vs all shots)
        on_target = outcomes.str.contains("goal|saved|save").sum()
        row["advshot_on_target_rate"] = _safe_div(on_target, n)

        # -----------------------------------------------------------
        # D. BODY PART PATTERNS
        # -----------------------------------------------------------
        if len(body_parts) > 0:
            row["advshot_header_ratio"] = _safe_div(
                body_parts.str.contains("head").sum(), n
            )
            left_foot = body_parts.str.contains("left").sum()
            right_foot = body_parts.str.contains("right").sum()
            foot_total = left_foot + right_foot
            row["advshot_left_foot_ratio"] = _safe_div(left_foot, foot_total, 0.5)
            # Header conversion (goals from headers / header shots)
            header_shots = body_parts.str.contains("head")
            header_goals = (header_shots & outcomes.str.contains("goal")).sum()
            row["advshot_header_conversion"] = _safe_div(
                header_goals, header_shots.sum()
            )
        else:
            row["advshot_header_ratio"] = 0.0
            row["advshot_left_foot_ratio"] = 0.5
            row["advshot_header_conversion"] = 0.0

        # -----------------------------------------------------------
        # E. TEMPORAL PATTERNS
        # -----------------------------------------------------------
        if len(minute_vals) > 0:
            row["advshot_early_threat_ratio"] = _safe_div(
                (minute_vals <= 30).sum(), n
            )
            row["advshot_late_pressure_ratio"] = _safe_div(
                (minute_vals >= 75).sum(), n
            )
        else:
            row["advshot_early_threat_ratio"] = 0.0
            row["advshot_late_pressure_ratio"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_advanced_shot_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add advanced shot-level features to the match-level table.

    Uses shots.parquet for per-shot event analysis, then applies
    rolling 5-match averages (shifted to prevent leakage).

    Columns added: home_advshot_roll5_{stat}, away_advshot_roll5_{stat}.
    """
    shots_path = parsed_path("shots")
    if not shots_path.exists():
        log.warning("shots.parquet missing; skipping advanced shot features")
        return matches

    shots = pd.read_parquet(shots_path)
    df = matches.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values("match_date").reset_index(drop=True)

    # Compute per-team per-match shot metrics
    log.info("Computing advanced shot metrics...")
    shot_metrics = _compute_advanced_shot_metrics(shots)
    if shot_metrics.empty:
        log.warning("No advanced shot metrics computed")
        return df

    # Merge match dates
    shot_metrics = shot_metrics.merge(
        df[["match_id", "match_date"]].drop_duplicates(),
        on="match_id",
        how="left",
    )
    shot_metrics = shot_metrics.sort_values(["team", "match_date"]).reset_index(drop=True)

    feature_cols = [
        c for c in shot_metrics.columns
        if c not in ("match_id", "team", "match_date")
    ]

    # Apply rolling 5-match average with shift(1)
    roll_cols = []
    for stat in feature_cols:
        col_name = f"advshot_roll5_{stat}"
        shot_metrics[col_name] = (
            shot_metrics.groupby("team")[stat]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )
        roll_cols.append(col_name)

    log.info("Computed %d advanced shot rolling features", len(roll_cols))

    # Merge to match level
    for prefix, team_col in [("home", "home_team"), ("away", "away_team")]:
        team_feats = shot_metrics[["match_id", "team"] + roll_cols].copy()
        rename_map = {c: f"{prefix}_{c}" for c in roll_cols}
        team_feats = team_feats.rename(columns=rename_map)
        df = df.merge(
            team_feats.rename(columns={"team": team_col}),
            on=["match_id", team_col],
            how="left",
        )

    added = sum(1 for c in df.columns if "advshot_roll5_" in c)
    log.info("Added %d advanced shot feature columns to match table", added)

    return df
