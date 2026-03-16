"""Shot quality features.

Captures shot-level characteristics beyond xG and shot count:
  - Average shot distance
  - Shot body part distribution (foot/head/other)
  - Shot outcome distribution (goal/saved/blocked/off target)
  - Big chance conversion rate
  - Shots inside box ratio
"""

from __future__ import annotations

import logging

import pandas as pd

from storage.paths import parsed_path

log = logging.getLogger(__name__)


def add_shot_quality_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add shot quality features to the match-level table.

    Uses shots.parquet to compute per-match shot quality metrics,
    then applies rolling 5-match average (shifted).

    Columns added: shot_roll5_{stat} for home and away.
    """
    shots_path = parsed_path("shots")
    if not shots_path.exists():
        log.warning("shots.parquet missing; skipping shot quality features")
        return matches

    shots = pd.read_parquet(shots_path)
    df = matches.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values("match_date").reset_index(drop=True)

    if "match_id" not in shots.columns:
        return df

    # Build team-level shot aggregates per match
    shot_agg = _compute_shot_aggregates(shots)
    if shot_agg.empty:
        return df

    # Merge match dates
    shot_agg = shot_agg.merge(
        df[["match_id", "match_date"]].drop_duplicates(),
        on="match_id",
        how="left",
    )
    shot_agg = shot_agg.sort_values(["team", "match_date"]).reset_index(drop=True)

    feature_cols = [c for c in shot_agg.columns if c not in ("match_id", "team", "match_date")]

    roll_cols = []
    for stat in feature_cols:
        col_name = f"shot_roll5_{stat}"
        shot_agg[col_name] = (
            shot_agg.groupby("team")[stat]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )
        roll_cols.append(col_name)

    # Merge to match level
    for prefix, team_col in [("home", "home_team"), ("away", "away_team")]:
        team_feats = shot_agg[["match_id", "team"] + roll_cols].copy()
        rename_map = {c: f"{prefix}_{c}" for c in roll_cols}
        team_feats = team_feats.rename(columns=rename_map)
        df = df.merge(
            team_feats.rename(columns={"team": team_col}),
            on=["match_id", team_col],
            how="left",
        )

    return df


def _compute_shot_aggregates(shots: pd.DataFrame) -> pd.DataFrame:
    """Compute team-level shot quality metrics per match."""
    if "team" not in shots.columns:
        return pd.DataFrame()

    # Ensure numeric
    for col in ["distance", "xg_shot"]:
        if col in shots.columns:
            shots[col] = pd.to_numeric(shots[col], errors="coerce")

    rows = []
    for (match_id, team), group in shots.groupby(["match_id", "team"]):
        row: dict = {"match_id": match_id, "team": team}
        n = len(group)
        row["total_shots"] = n

        # Average distance
        if "distance" in group.columns:
            row["avg_shot_distance"] = group["distance"].mean()

        # Average xG per shot (shot quality)
        if "xg_shot" in group.columns:
            row["avg_xg_per_shot"] = group["xg_shot"].mean()

        # Body part distribution
        if "body_part" in group.columns:
            bp = group["body_part"].str.lower().fillna("")
            row["pct_shots_foot"] = (bp.str.contains("foot|left|right")).sum() / max(n, 1)
            row["pct_shots_head"] = (bp.str.contains("head")).sum() / max(n, 1)

        # Outcome distribution
        if "outcome" in group.columns:
            outcome = group["outcome"].str.lower().fillna("")
            row["pct_shots_on_target"] = (
                outcome.str.contains("goal|saved|save").sum() / max(n, 1)
            )
            row["pct_shots_blocked"] = (
                outcome.str.contains("block").sum() / max(n, 1)
            )

        rows.append(row)

    return pd.DataFrame(rows)
