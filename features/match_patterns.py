"""Tier 7-9 — Match pattern features: bogey teams, comeback rate, late goals, set piece dependency.

Features added per side (home_/away_):
  A. Bogey Teams (H2H dominance)
     - h2h_home_dominance, h2h_goals_diff, h2h_btts_rate, h2h_over25_rate

  B. Comeback Rate
     - comeback_rate, ht_lead_hold_rate, losing_at_ht_rate

  C. Late Goals (from Sofascore shot data, rolling 5-match)
     - mp_late_goal_rate, mp_first15_xg_share, mp_last15_xg_share

  D. Set Piece Dependency (from Sofascore shot data, rolling 5-match)
     - mp_set_piece_goal_share, mp_open_play_xg_share, mp_counter_xg_share
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# A. Bogey Teams (H2H dominance)
# ---------------------------------------------------------------------------

def _add_h2h_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add H2H pattern features from prior meetings between the same two teams."""
    df = df.sort_values("match_date").reset_index(drop=True)

    h2h_history: dict[tuple, list] = {}

    cols = {k: [] for k in [
        "h2h_home_dominance", "h2h_goals_diff", "h2h_btts_rate",
        "h2h_over25_rate", "h2h_meetings",
    ]}

    for _, row in df.iterrows():
        home = row.get("home_team", "")
        away = row.get("away_team", "")
        hs = row.get("home_score")
        as_ = row.get("away_score")

        if not home or not away or pd.isna(home) or pd.isna(away):
            for k in cols:
                cols[k].append(None)
            continue

        key = tuple(sorted([home, away]))
        history = h2h_history.get(key, [])

        if len(history) >= 2:
            recent = history[-10:]
            home_wins = sum(1 for m in recent if m["winner"] == home)
            draws = sum(1 for m in recent if m["winner"] == "draw")

            cols["h2h_home_dominance"].append(
                round((home_wins + 0.5 * draws) / len(recent), 3)
            )

            goal_diffs = []
            for m in recent:
                if m["home"] == home:
                    goal_diffs.append(m["hs"] - m["as"])
                else:
                    goal_diffs.append(m["as"] - m["hs"])
            cols["h2h_goals_diff"].append(round(np.mean(goal_diffs), 2))

            btts = sum(1 for m in recent if m["hs"] > 0 and m["as"] > 0)
            cols["h2h_btts_rate"].append(round(btts / len(recent), 3))

            over25 = sum(1 for m in recent if m["hs"] + m["as"] > 2)
            cols["h2h_over25_rate"].append(round(over25 / len(recent), 3))
            cols["h2h_meetings"].append(len(recent))
        else:
            for k in cols:
                cols[k].append(None)

        # Update history
        if not pd.isna(hs) and not pd.isna(as_):
            hs_i, as_i = int(hs), int(as_)
            winner = home if hs_i > as_i else (away if as_i > hs_i else "draw")
            h2h_history.setdefault(key, []).append({
                "home": home, "away": away,
                "hs": hs_i, "as": as_i, "winner": winner,
            })

    for col, vals in cols.items():
        df[col] = vals

    n = df["h2h_meetings"].notna().sum()
    log.info("Added H2H pattern features to %d matches", n)
    return df


# ---------------------------------------------------------------------------
# B. Comeback Rate
# ---------------------------------------------------------------------------

def _add_comeback_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add comeback and HT-lead features using half-time scores."""
    df = df.sort_values("match_date").reset_index(drop=True)

    team_stats: dict[str, dict] = {}

    cols = {k: [] for k in [
        "home_comeback_rate", "away_comeback_rate",
        "home_ht_lead_hold", "away_ht_lead_hold",
        "home_losing_at_ht_rate", "away_losing_at_ht_rate",
    ]}

    for _, row in df.iterrows():
        home = row.get("home_team", "")
        away = row.get("away_team", "")

        for team, prefix in [(home, "home"), (away, "away")]:
            if not team or pd.isna(team):
                cols[f"{prefix}_comeback_rate"].append(None)
                cols[f"{prefix}_ht_lead_hold"].append(None)
                cols[f"{prefix}_losing_at_ht_rate"].append(None)
                continue

            stats = team_stats.get(team)
            if stats and stats["matches"] >= 5:
                n = stats["matches"]
                cols[f"{prefix}_comeback_rate"].append(
                    round(stats["comebacks"] / max(1, stats["losing_at_ht"]), 3)
                )
                cols[f"{prefix}_ht_lead_hold"].append(
                    round(stats["ht_leads_held"] / max(1, stats["ht_leads"]), 3)
                )
                cols[f"{prefix}_losing_at_ht_rate"].append(
                    round(stats["losing_at_ht"] / n, 3)
                )
            else:
                cols[f"{prefix}_comeback_rate"].append(None)
                cols[f"{prefix}_ht_lead_hold"].append(None)
                cols[f"{prefix}_losing_at_ht_rate"].append(None)

        # Update with this match
        hs = row.get("home_score")
        as_ = row.get("away_score")
        hht = row.get("home_ht_score")
        aht = row.get("away_ht_score")

        if pd.isna(hs) or pd.isna(as_) or pd.isna(hht) or pd.isna(aht):
            continue

        hs, as_ = int(hs), int(as_)
        hht, aht = int(hht), int(aht)

        for team, is_home in [(home, True), (away, False)]:
            if not team:
                continue
            if team not in team_stats:
                team_stats[team] = {
                    "matches": 0, "comebacks": 0, "losing_at_ht": 0,
                    "ht_leads": 0, "ht_leads_held": 0,
                }
            s = team_stats[team]
            s["matches"] += 1

            if is_home:
                ht_gf, ht_ga = hht, aht
                ft_gf, ft_ga = hs, as_
            else:
                ht_gf, ht_ga = aht, hht
                ft_gf, ft_ga = as_, hs

            if ht_gf < ht_ga:
                s["losing_at_ht"] += 1
                if ft_gf >= ft_ga:
                    s["comebacks"] += 1
            if ht_gf > ht_ga:
                s["ht_leads"] += 1
                if ft_gf > ft_ga:
                    s["ht_leads_held"] += 1

    for col, vals in cols.items():
        df[col] = vals

    n = df["home_comeback_rate"].notna().sum()
    log.info("Added comeback features to %d matches", n)
    return df


# ---------------------------------------------------------------------------
# C & D. Late Goals + Set Piece Dependency (Sofascore shots)
# ---------------------------------------------------------------------------

def _load_shot_patterns() -> dict:
    """Load per-team per-match shot timing and situation patterns.

    Returns: {(match_id, is_home_bool): {feature_dict}}
    """
    try:
        from features._utils import load_shot_level_xg
        shots = load_shot_level_xg()
    except Exception:
        return {}
    if shots is None:
        return {}

    for col in ["xg", "time", "is_goal"]:
        if col in shots.columns:
            shots[col] = pd.to_numeric(shots[col], errors="coerce").fillna(0)

    patterns = {}

    for (mid, is_home), grp in shots.groupby(["match_id", "is_home"]):
        total_xg = grp["xg"].sum()
        total_goals = int(grp["is_goal"].sum())

        if total_xg <= 0:
            continue

        first_15_xg = grp[grp["time"] <= 15]["xg"].sum()
        last_15_xg = grp[grp["time"] >= 76]["xg"].sum()
        late_goals = int(grp[(grp["time"] >= 76) & (grp["is_goal"] == 1)].shape[0])

        sp_mask = grp["situation"].isin([
            "corner", "set-piece", "free-kick", "throw-in-set-piece",
        ])
        sp_goals = int(grp[sp_mask & (grp["is_goal"] == 1)].shape[0])
        sp_xg = grp[sp_mask]["xg"].sum()
        open_xg = grp[grp["situation"].isin(["regular", "assisted"])]["xg"].sum()
        counter_xg = grp[grp["situation"] == "fast-break"]["xg"].sum()

        patterns[(mid, bool(is_home))] = {
            "first15_xg_share": first_15_xg / total_xg,
            "last15_xg_share": last_15_xg / total_xg,
            "late_goals": late_goals,
            "total_goals": total_goals,
            "sp_goal_share": sp_goals / total_goals if total_goals > 0 else 0,
            "open_xg_share": open_xg / total_xg,
            "counter_xg_share": counter_xg / total_xg,
            "sp_xg_share": sp_xg / total_xg,
        }

    log.info("Loaded shot patterns for %d team-match entries", len(patterns))
    return patterns


def _add_shot_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling late-goal and set-piece features from Sofascore shots."""
    patterns = _load_shot_patterns()
    if not patterns:
        return df

    df = df.sort_values("match_date").reset_index(drop=True)

    # Build per-team time series
    team_history: dict[str, list] = {}

    for _, row in df.iterrows():
        mid = row.get("match_id")
        for team_col, is_home in [("home_team", True), ("away_team", False)]:
            team = row.get(team_col, "")
            if not team or pd.isna(team):
                continue
            p = patterns.get((mid, is_home))
            if p:
                team_history.setdefault(team, []).append({
                    "match_id": mid, **p,
                })

    # Compute rolling 5-match averages (already in chronological order)
    rolling_lookup: dict[tuple, dict] = {}  # (match_id, team) -> features

    feature_keys = [
        "first15_xg_share", "last15_xg_share", "sp_goal_share",
        "open_xg_share", "counter_xg_share", "sp_xg_share",
    ]

    for team, entries in team_history.items():
        for i in range(len(entries)):
            if i >= 3:
                window = entries[max(0, i - 5):i]  # shift(1)
                feats = {}
                for key in feature_keys:
                    vals = [e[key] for e in window]
                    feats[f"mp_{key}"] = round(np.mean(vals), 4)

                # Late goal rate
                total_late = sum(e["late_goals"] for e in window)
                total_goals = sum(e["total_goals"] for e in window)
                feats["mp_late_goal_rate"] = round(
                    total_late / total_goals if total_goals > 0 else 0, 4
                )

                rolling_lookup[(entries[i]["match_id"], team)] = feats

    # Merge to match level
    for prefix, team_col in [("home", "home_team"), ("away", "away_team")]:
        col_data: dict[str, list] = {}
        for _, row in df.iterrows():
            team = row.get(team_col, "")
            mid = row.get("match_id")
            feats = rolling_lookup.get((mid, team), {})
            for k, v in feats.items():
                col_data.setdefault(f"{prefix}_{k}", []).append(v)
            if not feats:
                # Fill with None for all expected columns
                for k in [f"mp_{fk}" for fk in feature_keys] + ["mp_late_goal_rate"]:
                    col_data.setdefault(f"{prefix}_{k}", []).append(None)

        for col, vals in col_data.items():
            if len(vals) == len(df):
                df[col] = vals

    n = sum(1 for c in df.columns if "mp_" in c)
    log.info("Added %d shot pattern features (late goals + set pieces)", n)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_match_pattern_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add all Tier 7-9 match pattern features.

    Adds H2H patterns, comeback rates, late goals, and set piece dependency.
    """
    df = matches.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    log.info("Computing Tier 7-9 match pattern features...")

    df = _add_h2h_pattern_features(df)
    df = _add_comeback_features(df)
    df = _add_shot_pattern_features(df)

    total = sum(1 for c in df.columns if any(
        c.startswith(p) for p in ["h2h_", "home_comeback", "away_comeback",
                                   "home_ht_lead", "away_ht_lead",
                                   "home_losing", "away_losing",
                                   "home_mp_", "away_mp_"]
    ))
    log.info("Total Tier 7-9 features added: %d", total)
    return df
