"""Sofascore-based advanced player features for EPL.

Mirrors features/advanced_player.py (which uses FBref's 141-col schema)
but reads from the Sofascore 80-col player_match_stats schema instead.

Why this exists: FBref serves EPL match reports with ONLY summary tables
(no passing/possession/defense detail). EPL's data/parsed/player_stats_epl.parquet
has 30 cols vs SA's 141. So advanced_player.py emits zero EPL features.
But Sofascore's per-player-per-match data IS rich for EPL — 94k rows ×
80 cols across 9 seasons. We can build the same kind of derived metrics
from those columns.

Coverage: outputs ~25 metrics per (match, team), which after rolling
shifted over 5 prior matches × 2 sides (home/away) = ~100 EPL features.

This is league-aware: routes to player_match_stats_premier_league.parquet
for EPL, falls back to player_match_stats.parquet for SA (where SA can use
either FBref-rich OR Sofascore — but the FBref-based plugin already produces
~80 metrics for SA so this is mainly an EPL filler).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from features._utils import _safe_div

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Sofascore column dependencies — if any required column is missing, function
# returns empty (graceful no-op).
REQUIRED_COLS = [
    "match_id", "team", "minutes", "rating",
    "total_passes", "accurate_passes",
    "total_shots", "shots_on_target",
    "tackles", "interceptions", "clearances",
    "ball_recoveries", "fouls", "was_fouled",
    "xg", "xa",
]


def _load_sofascore_for_league(league: str | None) -> pd.DataFrame | None:
    """Load Sofascore player stats for the requested league."""
    base = PROJECT_ROOT / "data" / "external" / "sofascore"
    if league == "premier_league":
        path = base / "player_match_stats_premier_league.parquet"
    else:
        path = base / "player_match_stats.parquet"
    if not path.exists():
        log.warning("Sofascore player stats not found: %s", path)
        return None
    df = pd.read_parquet(path)
    log.info(
        "Loaded Sofascore for league=%s: %d rows × %d cols",
        league, len(df), len(df.columns),
    )
    return df


def _compute_sofascore_team_metrics(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-player Sofascore stats to per-team-per-match metrics.

    Returns DataFrame with columns: match_id, team, + ~25 advanced metrics.
    """
    missing = [c for c in REQUIRED_COLS if c not in player_stats.columns]
    if missing:
        log.warning(
            "Sofascore advanced_player skipped — missing required cols: %s",
            missing,
        )
        return pd.DataFrame()

    # Numeric coercion
    numeric = [
        "minutes", "rating", "goals", "assists", "key_passes",
        "total_passes", "accurate_passes", "accurate_long_balls", "total_long_balls",
        "total_crosses", "accurate_crosses",
        "total_shots", "shots_on_target", "shots_off_target", "shots_blocked",
        "tackles", "tackles_won", "interceptions", "clearances", "blocks", "last_man_tackle",
        "ball_recoveries", "duels_won", "duels_lost", "aerial_won", "aerial_lost",
        "carries", "progressive_carries", "carry_distance", "progressive_carry_distance",
        "touches", "possession_lost", "dispossessed", "unsuccessful_touch",
        "fouls", "was_fouled", "offsides",
        "error_to_goal", "error_to_shot",
        "contest_total", "contest_won", "challenge_lost",
        "xg", "xa", "xgot", "big_chances_created", "big_chances_missed",
        "saves", "saved_shots_from_inside_box", "goals_prevented",
        "opp_half_passes", "own_half_passes",
        "total_progression",
    ]
    ps = player_stats.copy()
    for c in numeric:
        if c in ps.columns:
            ps[c] = pd.to_numeric(ps[c], errors="coerce")

    rows = []
    for (match_id, team), grp in ps.groupby(["match_id", "team"], sort=False):
        row: dict = {"match_id": match_id, "team": team}

        # Aggregate raw counts
        total_passes = grp["total_passes"].sum()
        accurate_passes = grp["accurate_passes"].sum()
        total_long = grp.get("total_long_balls", pd.Series(dtype=float)).sum() if "total_long_balls" in grp.columns else 0
        accurate_long = grp.get("accurate_long_balls", pd.Series(dtype=float)).sum() if "accurate_long_balls" in grp.columns else 0
        total_crosses = grp.get("total_crosses", pd.Series(dtype=float)).sum() if "total_crosses" in grp.columns else 0
        accurate_crosses = grp.get("accurate_crosses", pd.Series(dtype=float)).sum() if "accurate_crosses" in grp.columns else 0
        key_passes = grp.get("key_passes", pd.Series(dtype=float)).sum() if "key_passes" in grp.columns else 0
        opp_half_passes = grp.get("opp_half_passes", pd.Series(dtype=float)).sum() if "opp_half_passes" in grp.columns else 0
        own_half_passes = grp.get("own_half_passes", pd.Series(dtype=float)).sum() if "own_half_passes" in grp.columns else 0

        total_carries = grp.get("carries", pd.Series(dtype=float)).sum() if "carries" in grp.columns else 0
        prog_carries = grp.get("progressive_carries", pd.Series(dtype=float)).sum() if "progressive_carries" in grp.columns else 0
        carry_dist = grp.get("carry_distance", pd.Series(dtype=float)).sum() if "carry_distance" in grp.columns else 0
        prog_carry_dist = grp.get("progressive_carry_distance", pd.Series(dtype=float)).sum() if "progressive_carry_distance" in grp.columns else 0

        total_touches = grp["touches"].sum() if "touches" in grp.columns else 0
        possession_lost = grp.get("possession_lost", pd.Series(dtype=float)).sum() if "possession_lost" in grp.columns else 0
        dispossessed = grp.get("dispossessed", pd.Series(dtype=float)).sum() if "dispossessed" in grp.columns else 0
        miscontrols = grp.get("unsuccessful_touch", pd.Series(dtype=float)).sum() if "unsuccessful_touch" in grp.columns else 0

        total_shots = grp["total_shots"].sum()
        shots_on_target = grp["shots_on_target"].sum()
        shots_blocked = grp.get("shots_blocked", pd.Series(dtype=float)).sum() if "shots_blocked" in grp.columns else 0
        big_chances_missed = grp.get("big_chances_missed", pd.Series(dtype=float)).sum() if "big_chances_missed" in grp.columns else 0
        big_chances_created = grp.get("big_chances_created", pd.Series(dtype=float)).sum() if "big_chances_created" in grp.columns else 0

        tackles = grp["tackles"].sum()
        tackles_won = grp.get("tackles_won", pd.Series(dtype=float)).sum() if "tackles_won" in grp.columns else 0
        interceptions = grp["interceptions"].sum()
        clearances = grp["clearances"].sum()
        blocks = grp.get("blocks", pd.Series(dtype=float)).sum() if "blocks" in grp.columns else 0
        recoveries = grp["ball_recoveries"].sum()
        errors = (grp.get("error_to_goal", pd.Series(dtype=float)).sum() if "error_to_goal" in grp.columns else 0) + \
                 (grp.get("error_to_shot", pd.Series(dtype=float)).sum() if "error_to_shot" in grp.columns else 0)
        aerials_won = grp.get("aerial_won", pd.Series(dtype=float)).sum() if "aerial_won" in grp.columns else 0
        aerials_lost = grp.get("aerial_lost", pd.Series(dtype=float)).sum() if "aerial_lost" in grp.columns else 0
        duels_won = grp.get("duels_won", pd.Series(dtype=float)).sum() if "duels_won" in grp.columns else 0
        duels_lost = grp.get("duels_lost", pd.Series(dtype=float)).sum() if "duels_lost" in grp.columns else 0

        contest_total = grp.get("contest_total", pd.Series(dtype=float)).sum() if "contest_total" in grp.columns else 0
        contest_won = grp.get("contest_won", pd.Series(dtype=float)).sum() if "contest_won" in grp.columns else 0

        xg = grp["xg"].sum()
        xa = grp["xa"].sum()
        xgot = grp.get("xgot", pd.Series(dtype=float)).sum() if "xgot" in grp.columns else 0
        goals = grp.get("goals", pd.Series(dtype=float)).sum() if "goals" in grp.columns else 0
        assists = grp.get("assists", pd.Series(dtype=float)).sum() if "assists" in grp.columns else 0

        fouls = grp["fouls"].sum()
        was_fouled = grp["was_fouled"].sum()

        # ---- A. STYLE & BUILDUP ----
        row["sa_pass_accuracy"] = _safe_div(accurate_passes, total_passes)
        row["sa_long_pass_ratio"] = _safe_div(total_long, total_passes)
        row["sa_long_pass_accuracy"] = _safe_div(accurate_long, total_long)
        row["sa_cross_rate"] = _safe_div(total_crosses, total_passes)
        row["sa_cross_accuracy"] = _safe_div(accurate_crosses, total_crosses)
        row["sa_key_passes_per_pass"] = _safe_div(key_passes, total_passes)
        row["sa_opp_half_pass_ratio"] = _safe_div(opp_half_passes, opp_half_passes + own_half_passes, 0.5)
        row["sa_progressive_carry_ratio"] = _safe_div(prog_carries, total_carries)
        row["sa_carry_prog_dist_share"] = _safe_div(prog_carry_dist, prog_carry_dist + carry_dist, 0.5)
        row["sa_carry_per_touch"] = _safe_div(total_carries, total_touches)

        # ---- B. PRESSING & DEFENSE ----
        row["sa_tackle_success_rate"] = _safe_div(tackles_won, tackles)
        row["sa_aerial_dominance"] = _safe_div(aerials_won, aerials_won + aerials_lost, 0.5)
        row["sa_duel_success_rate"] = _safe_div(duels_won, duels_won + duels_lost, 0.5)
        row["sa_recovery_efficiency"] = _safe_div(recoveries, recoveries + errors, 1.0)
        defensive_actions = tackles + interceptions + clearances + blocks
        row["sa_defensive_actions_per_touch"] = _safe_div(defensive_actions, total_touches)
        row["sa_press_intensity"] = _safe_div(tackles + interceptions, total_touches)

        # ---- C. ATTACKING EFFICIENCY ----
        row["sa_shot_accuracy"] = _safe_div(shots_on_target, total_shots)
        row["sa_shot_block_rate"] = _safe_div(shots_blocked, total_shots)
        row["sa_big_chance_conversion"] = _safe_div(
            big_chances_created - big_chances_missed, big_chances_created, 0.5
        ) if big_chances_created > 0 else 0.5
        row["sa_npxg_per_shot"] = _safe_div(max(xg - 0.79 * (grp.get("error_to_shot", pd.Series([0])).sum() if "error_to_shot" in grp.columns else 0), 0), total_shots)
        row["sa_xg_per_shot"] = _safe_div(xg, total_shots)
        row["sa_xa_per_pass"] = _safe_div(xa, total_passes)
        row["sa_take_on_success"] = _safe_div(contest_won, contest_total, 0.5) if contest_total > 0 else 0.5
        row["sa_dispossession_rate"] = _safe_div(dispossessed + miscontrols, total_touches)
        row["sa_finishing_efficiency"] = _safe_div(goals, max(xg, 0.1))  # cap div
        row["sa_xa_xg_ratio"] = _safe_div(xa, max(xg, 0.1))

        # ---- D. DISCIPLINE / GAME-CONTROL ----
        row["sa_foul_rate"] = _safe_div(fouls, total_touches)
        row["sa_fouled_rate"] = _safe_div(was_fouled, total_touches)

        # ---- E. STAR FORM (mean rating of starters with 60+ min) ----
        starters = grp[(grp["minutes"] >= 60)] if "minutes" in grp.columns else grp
        row["sa_star_rating_mean"] = float(starters["rating"].mean()) if len(starters) and "rating" in starters.columns else 6.5

        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def add_advanced_player_sofascore_features(
    matches: pd.DataFrame,
    league: str | None = None,
) -> pd.DataFrame:
    """Add Sofascore-derived advanced player features to match-level table.

    Same kind of metrics as features/advanced_player.py but sourced from
    Sofascore's 80-col schema instead of FBref's 141-col schema. Designed
    primarily for EPL where FBref doesn't provide the rich detail.

    Adds ~25 features per side as `home_sa_roll5_*` / `away_sa_roll5_*` columns
    plus differentials.
    """
    player_stats = _load_sofascore_for_league(league)
    if player_stats is None or player_stats.empty:
        return matches

    df = matches.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values("match_date").reset_index(drop=True)

    log.info("Computing Sofascore team metrics...")
    team_metrics = _compute_sofascore_team_metrics(player_stats)
    if team_metrics.empty:
        log.warning("No Sofascore team metrics produced; skipping")
        return df

    # Bring match dates onto team_metrics
    if "date" in player_stats.columns:
        match_dates = player_stats[["match_id", "date"]].drop_duplicates()
        match_dates["date"] = pd.to_datetime(match_dates["date"], errors="coerce")
        team_metrics = team_metrics.merge(match_dates, on="match_id", how="left")
    else:
        team_metrics["date"] = pd.NaT

    team_metrics = team_metrics.sort_values(["team", "date"]).reset_index(drop=True)

    # Apply rolling 5 with shift(1) per team
    feature_cols = [c for c in team_metrics.columns if c.startswith("sa_")]
    roll_cols: list[str] = []
    for stat in feature_cols:
        roll_name = f"sa_roll5_{stat[3:]}"  # strip 'sa_' prefix; new prefix sa_roll5_
        team_metrics[roll_name] = (
            team_metrics.groupby("team")[stat]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean())
        )
        roll_cols.append(roll_name)
    log.info("Computed %d Sofascore-based rolling metrics", len(roll_cols))

    # Merge back to match level on (team, date) using merge_asof
    df["_match_date"] = pd.to_datetime(df.get("match_date"), errors="coerce")
    lookup_cols = ["team", "date"] + roll_cols
    lookup = team_metrics[lookup_cols].dropna(subset=["date"]).copy()
    lookup = lookup.sort_values(["team", "date"]).reset_index(drop=True)
    lookup = lookup.rename(columns={"date": "_match_date"})

    for prefix, team_col in [("home", "home_team"), ("away", "away_team")]:
        side = df[["_match_date", team_col]].copy()
        side = side.rename(columns={team_col: "team"})
        side = side.dropna(subset=["_match_date"]).reset_index().sort_values("_match_date")

        right = lookup.copy()
        right = right.dropna(subset=["_match_date"]).sort_values("_match_date")

        merged = pd.merge_asof(
            side, right, on="_match_date", by="team", direction="backward",
        ).set_index("index")
        for c in roll_cols:
            if c in merged.columns:
                df.loc[merged.index, f"{prefix}_{c}"] = merged[c].values

    # Differential features
    diffs = {}
    for c in roll_cols:
        h = f"home_{c}"
        a = f"away_{c}"
        if h in df.columns and a in df.columns:
            diffs[f"sa_diff_{c}"] = df[h] - df[a]
    if diffs:
        df = pd.concat([df, pd.DataFrame(diffs, index=df.index)], axis=1)

    df.drop(columns=["_match_date"], errors="ignore", inplace=True)

    n_added = sum(1 for c in df.columns if "_sa_roll5_" in c or c.startswith("sa_diff_"))
    log.info("Added %d Sofascore-based advanced features", n_added)
    return df
