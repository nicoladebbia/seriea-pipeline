"""Advanced player-level features: ratios, style profiles, squad dependency.

Goes far beyond raw count aggregates (team_aggregates.py) by computing
RATIO features that capture tactical identity and squad structure:

  A. Style & Buildup Profile
     - Progressive pass ratio (directness of buildup)
     - Final-third entry method (carry vs pass)
     - Touch distribution in attacking zones
     - Through-ball and switch-play tendencies
     - Pass length preference (direct vs short)

  B. Pressing & Defensive Identity
     - Tackle success rate
     - High press ratio (where on the pitch they press)
     - Aerial dominance
     - Recovery efficiency

  C. Attacking Efficiency
     - Take-on success rate
     - Progressive carry ratio
     - Chance creation per touch
     - xG per attacking-box touch (quality from possession)

  D. Squad Dependency & Concentration
     - Goal-threat concentration (Gini coefficient)
     - Creativity concentration (SCA Gini)
     - Top-scorer goal share
     - Top-creator SCA share

  E. Star Player Rolling Form
     - Top-3 players combined xG+xA form
     - Key player minutes availability ratio

All features are computed per team per match, then rolled into
5-match averages (shift(1) to prevent leakage) and merged
to match level as home_*/away_* columns.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from storage.paths import parsed_path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: safe division
# ---------------------------------------------------------------------------

def _safe_div(a, b, default: float = 0.0):
    """Element-wise a/b, returning default where b is 0 or NaN."""
    if isinstance(a, (pd.Series, np.ndarray)):
        result = np.where((b == 0) | pd.isna(b), default, a / b)
        return pd.Series(result, index=a.index) if isinstance(a, pd.Series) else result
    if b == 0 or pd.isna(b):
        return default
    return a / b


def _gini(values: list[float]) -> float:
    """Compute Gini coefficient for a list of non-negative values.

    Returns 0 (perfect equality) to 1 (maximum concentration).
    """
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2 or arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr)))


# ---------------------------------------------------------------------------
# Core: compute all advanced metrics per team per match
# ---------------------------------------------------------------------------

def _compute_advanced_team_metrics(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Compute advanced ratio features per team per match from player-level data.

    Returns DataFrame with columns: match_id, team, + ~30 feature columns.
    """
    # Convert all numeric columns upfront
    numeric_cols = [
        "minutes", "goals", "assists", "xg", "npxg", "xg_assist", "sca", "gca",
        "shots", "shots_on_target",
        "passes_completed", "passes",
        "passing_passes", "passing_passes_completed",
        "passing_progressive_passes", "passing_passes_into_final_third",
        "passing_passes_into_penalty_area", "passing_passes_long",
        "passing_passes_short", "passing_passes_completed_short",
        "passing_passes_completed_long", "passing_passes_medium",
        "passing_passes_completed_medium",
        "passing_passes_total_distance", "passing_passes_progressive_distance",
        "passing_pass_xa", "passing_assisted_shots",
        "passing_types_through_balls", "passing_types_passes_switches",
        "passing_types_crosses", "passing_types_passes_live",
        "passing_types_passes_dead",
        "possession_touches", "possession_touches_att_pen_area",
        "possession_touches_att_3rd", "possession_touches_def_3rd",
        "possession_touches_mid_3rd", "possession_touches_def_pen_area",
        "possession_carries", "possession_progressive_carries",
        "possession_carries_into_final_third", "possession_carries_into_penalty_area",
        "possession_carries_distance", "possession_carries_progressive_distance",
        "possession_take_ons", "possession_take_ons_won",
        "possession_dispossessed", "possession_miscontrols",
        "possession_passes_received", "possession_progressive_passes_received",
        "defense_tackles", "defense_tackles_won",
        "defense_tackles_att_3rd", "defense_tackles_def_3rd", "defense_tackles_mid_3rd",
        "defense_challenges", "defense_challenges_lost",
        "defense_interceptions", "defense_clearances", "defense_errors",
        "defense_blocked_shots", "defense_blocked_passes",
        "misc_aerials_won", "misc_aerials_lost",
        "misc_ball_recoveries", "misc_fouls", "misc_fouled",
        "misc_cards_yellow", "misc_cards_red",
        "misc_pens_won", "misc_pens_conceded",
    ]

    ps = player_stats.copy()
    for col in numeric_cols:
        if col in ps.columns:
            ps[col] = pd.to_numeric(ps[col], errors="coerce")

    if "match_id" not in ps.columns or "team" not in ps.columns:
        return pd.DataFrame()

    rows = []
    for (match_id, team), grp in ps.groupby(["match_id", "team"]):
        row: dict = {"match_id": match_id, "team": team}

        # ---------------------------------------------------------------
        # A. STYLE & BUILDUP PROFILE
        # ---------------------------------------------------------------
        total_passes = grp["passing_passes"].sum()
        total_passes_completed = grp["passing_passes_completed"].sum()
        prog_passes = grp["passing_progressive_passes"].sum()
        passes_final_third = grp["passing_passes_into_final_third"].sum()
        passes_pen_area = grp["passing_passes_into_penalty_area"].sum()
        passes_long = grp["passing_passes_long"].sum()
        passes_short = grp["passing_passes_short"].sum()
        through_balls = grp["passing_types_through_balls"].sum()
        switches = grp["passing_types_passes_switches"].sum()
        crosses = grp["passing_types_crosses"].sum()

        carries_final_third = grp["possession_carries_into_final_third"].sum()
        carries_pen_area = grp["possession_carries_into_penalty_area"].sum()
        total_carries = grp["possession_carries"].sum()
        prog_carries = grp["possession_progressive_carries"].sum()
        carry_prog_dist = grp["possession_carries_progressive_distance"].sum()
        pass_prog_dist = grp["passing_passes_progressive_distance"].sum()

        total_touches = grp["possession_touches"].sum()
        touches_att_box = grp["possession_touches_att_pen_area"].sum()
        touches_att_3rd = grp["possession_touches_att_3rd"].sum()
        touches_def_3rd = grp["possession_touches_def_3rd"].sum()
        touches_mid_3rd = grp["possession_touches_mid_3rd"].sum()

        # Progressive pass ratio: how direct is their buildup?
        row["adv_progressive_pass_ratio"] = _safe_div(prog_passes, total_passes)

        # Final-third entry: carry vs pass (high = dribble-heavy team)
        final_third_entries = carries_final_third + passes_final_third
        row["adv_carry_vs_pass_final_third"] = _safe_div(
            carries_final_third, final_third_entries, 0.5
        )

        # Touch distribution in attacking zones
        row["adv_touch_att_box_ratio"] = _safe_div(touches_att_box, total_touches)
        row["adv_touch_att_3rd_ratio"] = _safe_div(touches_att_3rd, total_touches)

        # Passing directness: long ball preference
        row["adv_long_pass_ratio"] = _safe_div(passes_long, total_passes)

        # Through-ball rate (incisive passing)
        row["adv_through_ball_rate"] = _safe_div(through_balls, total_passes)

        # Switch-play rate (width usage)
        row["adv_switch_play_rate"] = _safe_div(switches, total_passes)

        # Cross rate
        row["adv_cross_rate"] = _safe_div(crosses, total_passes)

        # Progressive distance: carry vs pass (buildup method)
        total_prog_dist = carry_prog_dist + pass_prog_dist
        row["adv_carry_prog_dist_share"] = _safe_div(carry_prog_dist, total_prog_dist, 0.5)

        # Passes into penalty area per touch (box-entry efficiency)
        row["adv_pen_area_pass_rate"] = _safe_div(passes_pen_area, total_touches)

        # ---------------------------------------------------------------
        # B. PRESSING & DEFENSIVE IDENTITY
        # ---------------------------------------------------------------
        tackles = grp["defense_tackles"].sum()
        tackles_won = grp["defense_tackles_won"].sum()
        tackles_att = grp["defense_tackles_att_3rd"].sum()
        tackles_mid = grp["defense_tackles_mid_3rd"].sum()
        tackles_def = grp["defense_tackles_def_3rd"].sum()
        challenges = grp["defense_challenges"].sum()
        challenges_lost = grp["defense_challenges_lost"].sum()
        interceptions = grp["defense_interceptions"].sum()
        clearances = grp["defense_clearances"].sum()
        blocks_shots = grp["defense_blocked_shots"].sum()
        blocks_passes = grp["defense_blocked_passes"].sum()
        errors = grp["defense_errors"].sum()
        recoveries = grp["misc_ball_recoveries"].sum()
        aerials_won = grp["misc_aerials_won"].sum()
        aerials_lost = grp["misc_aerials_lost"].sum()

        # Tackle success rate
        row["adv_tackle_success_rate"] = _safe_div(tackles_won, tackles)

        # High press ratio: where do they tackle?
        row["adv_high_press_tackle_ratio"] = _safe_div(tackles_att, tackles)
        row["adv_mid_press_tackle_ratio"] = _safe_div(tackles_mid, tackles)

        # Aerial dominance
        total_aerials = aerials_won + aerials_lost
        row["adv_aerial_dominance"] = _safe_div(aerials_won, total_aerials, 0.5)

        # Recovery efficiency (recoveries vs errors)
        row["adv_recovery_efficiency"] = _safe_div(
            recoveries, recoveries + errors, 0.5
        )

        # Defensive action per touch (defensive intensity)
        defensive_actions = tackles + interceptions + clearances + blocks_shots + blocks_passes
        row["adv_defensive_actions_per_touch"] = _safe_div(defensive_actions, total_touches)

        # Challenge success rate (broader than tackles)
        row["adv_challenge_success_rate"] = _safe_div(
            challenges - challenges_lost, challenges, 0.5
        ) if challenges > 0 else 0.5

        # ---------------------------------------------------------------
        # C. ATTACKING EFFICIENCY
        # ---------------------------------------------------------------
        take_ons = grp["possession_take_ons"].sum()
        take_ons_won = grp["possession_take_ons_won"].sum()
        dispossessed = grp["possession_dispossessed"].sum()
        miscontrols = grp["possession_miscontrols"].sum()
        total_sca = grp["sca"].sum()
        total_gca = grp["gca"].sum()
        total_xg = grp["xg"].sum()
        total_npxg = grp["npxg"].sum()
        total_xa = grp["xg_assist"].sum()
        total_shots = grp["shots"].sum()
        total_goals = grp["goals"].sum()

        # Take-on success rate
        row["adv_take_on_success_rate"] = _safe_div(take_ons_won, take_ons, 0.5)

        # Progressive carry ratio
        row["adv_progressive_carry_ratio"] = _safe_div(prog_carries, total_carries)

        # Chance creation per touch
        row["adv_sca_per_touch"] = _safe_div(total_sca, total_touches)

        # Goal creation per touch
        row["adv_gca_per_touch"] = _safe_div(total_gca, total_touches)

        # xG per attacking-box touch (quality from box entries)
        row["adv_xg_per_box_touch"] = _safe_div(total_xg, touches_att_box)

        # xG per shot (shot selection quality)
        row["adv_xg_per_shot"] = _safe_div(total_xg, total_shots)

        # Non-penalty xG per shot
        row["adv_npxg_per_shot"] = _safe_div(total_npxg, total_shots)

        # Carries into box per carry (penetration rate)
        row["adv_box_carry_rate"] = _safe_div(carries_pen_area, total_carries)

        # Ball retention: dispossessions + miscontrols per touch
        row["adv_ball_loss_rate"] = _safe_div(
            dispossessed + miscontrols, total_touches
        )

        # Expected assist per pass (passing threat)
        row["adv_xa_per_pass"] = _safe_div(total_xa, total_passes)

        # ---------------------------------------------------------------
        # D. SQUAD DEPENDENCY & CONCENTRATION
        # ---------------------------------------------------------------
        player_goals = grp["goals"].fillna(0).tolist()
        player_sca = grp["sca"].fillna(0).tolist()
        player_xg = grp["xg"].fillna(0).tolist()
        player_minutes = grp["minutes"].fillna(0).tolist()

        # Gini coefficients (0 = everyone contributes equally, 1 = one player dominates)
        row["adv_goals_gini"] = _gini(player_goals)
        row["adv_sca_gini"] = _gini(player_sca)
        row["adv_xg_gini"] = _gini(player_xg)

        # Top scorer's share of team goals
        row["adv_top_scorer_share"] = (
            _safe_div(max(player_goals), sum(player_goals))
            if player_goals else 0.0
        )

        # Top creator's share of team SCA
        row["adv_top_creator_sca_share"] = (
            _safe_div(max(player_sca), sum(player_sca))
            if player_sca else 0.0
        )

        # Top-3 players' share of total minutes (squad rotation indicator)
        if player_minutes:
            sorted_mins = sorted(player_minutes, reverse=True)
            top3_mins = sum(sorted_mins[:3])
            total_mins = sum(sorted_mins)
            row["adv_top3_minutes_share"] = _safe_div(top3_mins, total_mins)
        else:
            row["adv_top3_minutes_share"] = 0.0

        # ---------------------------------------------------------------
        # E. DISCIPLINE PROFILE
        # ---------------------------------------------------------------
        yellows = grp["misc_cards_yellow"].sum()
        reds = grp["misc_cards_red"].sum()
        fouls = grp["misc_fouls"].sum()
        fouled = grp["misc_fouled"].sum()

        # Fouls committed per touch (aggression rate)
        row["adv_foul_rate"] = _safe_div(fouls, total_touches)

        # Fouls won per touch (draws fouls = attacks well)
        row["adv_fouled_rate"] = _safe_div(fouled, total_touches)

        # Cards per foul (discipline — high = reckless fouls)
        row["adv_cards_per_foul"] = _safe_div(yellows + reds * 2, fouls)

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Star player form: rolling individual performance of top contributors
# ---------------------------------------------------------------------------

def _compute_star_form(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Compute star player rolling form features per team per match.

    For each team, identifies the top contributors (by cumulative xG+xA)
    up to each match and computes their recent form.

    Returns DataFrame with: match_id, team, star_form_*, star_dependency_*
    """
    ps = player_stats.copy()
    for col in ["xg", "xg_assist", "sca", "gca", "goals", "assists", "minutes"]:
        if col in ps.columns:
            ps[col] = pd.to_numeric(ps[col], errors="coerce").fillna(0)

    if "match_date" not in ps.columns:
        return pd.DataFrame()

    ps["match_date"] = pd.to_datetime(ps["match_date"], errors="coerce")
    ps = ps.sort_values("match_date").reset_index(drop=True)

    # Cumulative tracking per player per team
    cum_xg_xa: dict[str, dict[str, float]] = {}  # team -> {player: cum xG+xA}
    player_recent: dict[str, dict[str, list[float]]] = {}  # team -> {player: [last N xG+xA]}

    rows = []
    seen_matches: dict[str, set] = {}  # team -> set of processed match_ids

    for match_id in ps["match_id"].unique():
        match_players = ps[ps["match_id"] == match_id]
        match_date = match_players["match_date"].iloc[0]

        for team in match_players["team"].unique():
            team_grp = match_players[match_players["team"] == team]
            row: dict = {"match_id": match_id, "team": team}

            # Get cumulative stats BEFORE this match
            team_cum = cum_xg_xa.get(team, {})
            team_recent = player_recent.get(team, {})

            if team_cum:
                # Identify top-3 contributors by cumulative xG+xA
                sorted_players = sorted(team_cum.items(), key=lambda x: x[1], reverse=True)
                top3 = sorted_players[:3]
                top3_names = {p for p, _ in top3}

                # Star form: average of top-3's last-5 match xG+xA
                star_recent_vals = []
                for pname in top3_names:
                    recent = team_recent.get(pname, [])
                    if recent:
                        star_recent_vals.append(np.mean(recent[-5:]))

                row["adv_star_form_xg_xa"] = np.mean(star_recent_vals) if star_recent_vals else 0.0

                # Did top-3 play this match?
                match_player_names = set(team_grp["player"].dropna().tolist())
                top3_available = len(top3_names & match_player_names)
                row["adv_star_availability"] = top3_available / max(len(top3), 1)
            else:
                row["adv_star_form_xg_xa"] = 0.0
                row["adv_star_availability"] = 1.0

            rows.append(row)

            # --- Update cumulative trackers WITH this match ---
            if team not in cum_xg_xa:
                cum_xg_xa[team] = {}
            if team not in player_recent:
                player_recent[team] = {}

            for _, p in team_grp.iterrows():
                pname = p.get("player", "")
                if not pname or pd.isna(pname):
                    continue
                xg_val = float(p.get("xg", 0) or 0)
                xa_val = float(p.get("xg_assist", 0) or 0)
                match_output = xg_val + xa_val

                cum_xg_xa[team][pname] = cum_xg_xa[team].get(pname, 0) + match_output
                if pname not in player_recent[team]:
                    player_recent[team][pname] = []
                player_recent[team][pname].append(match_output)
                # Keep only last 10 matches per player
                if len(player_recent[team][pname]) > 10:
                    player_recent[team][pname] = player_recent[team][pname][-10:]

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public API: add_advanced_player_features
# ---------------------------------------------------------------------------

def add_advanced_player_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add advanced player-level ratio and dependency features.

    Uses player_stats.parquet to compute per-team-per-match tactical
    ratios, then applies rolling 5-match averages (shifted).

    Columns added: home_adv_roll5_{stat}, away_adv_roll5_{stat} (~40 features per side).
    """
    ps_path = parsed_path("player_stats")
    if not ps_path.exists():
        log.warning("player_stats.parquet missing; skipping advanced player features")
        return matches

    player_stats = pd.read_parquet(ps_path)
    df = matches.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values("match_date").reset_index(drop=True)

    # --- Part 1: Team-level advanced metrics ---
    log.info("Computing advanced team metrics from player stats...")
    team_metrics = _compute_advanced_team_metrics(player_stats)
    if team_metrics.empty:
        log.warning("No advanced team metrics computed")
        return df

    # --- Part 2: Star player form ---
    log.info("Computing star player form features...")
    star_form = _compute_star_form(player_stats)
    if not star_form.empty:
        team_metrics = team_metrics.merge(
            star_form, on=["match_id", "team"], how="left"
        )

    # --- Merge match dates and apply rolling ---
    team_metrics = team_metrics.merge(
        df[["match_id", "match_date"]].drop_duplicates(),
        on="match_id",
        how="left",
    )
    team_metrics = team_metrics.sort_values(["team", "match_date"]).reset_index(drop=True)

    feature_cols = [
        c for c in team_metrics.columns
        if c not in ("match_id", "team", "match_date")
    ]

    # Apply rolling 5-match average with shift(1) to prevent leakage
    roll_cols = []
    for stat in feature_cols:
        col_name = f"adv_roll5_{stat}"
        team_metrics[col_name] = (
            team_metrics.groupby("team")[stat]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )
        roll_cols.append(col_name)

    log.info("Computed %d advanced rolling features", len(roll_cols))

    # --- Merge back to match level ---
    for prefix, team_col in [("home", "home_team"), ("away", "away_team")]:
        team_feats = team_metrics[["match_id", "team"] + roll_cols].copy()
        rename_map = {c: f"{prefix}_{c}" for c in roll_cols}
        team_feats = team_feats.rename(columns=rename_map)
        df = df.merge(
            team_feats.rename(columns={"team": team_col}),
            on=["match_id", team_col],
            how="left",
        )

    added = sum(1 for c in df.columns if "adv_roll5_" in c)
    log.info("Added %d advanced player feature columns to match table", added)

    return df
