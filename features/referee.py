"""Referee Intelligence Module - Enhanced referee tendency and bias features.

Referees have measurable tendencies and biases:
- Cards per match (strictness)
- Home team bias (favoring home teams)
- Penalty tendency (vs league average) — sourced from Sofascore shot data
- Team-specific history (harsh to certain teams)
- Big match behavior (changes in high-stakes games)
- xG impact (do matches with this ref tend to be higher/lower scoring?)
- Strictness trend (is the referee getting stricter or more lenient recently?)

These features capture referee patterns that impact match outcomes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _normalize_ws(s: str) -> str:
    """Normalize all whitespace (including \xa0) to regular spaces."""
    import re
    return re.sub(r'\s+', ' ', s).strip() if isinstance(s, str) else s

# Top clubs historically — used for big-match detection
_BIG_MATCH_TEAMS = {
    # Serie A
    "Inter", "Milan", "Juventus", "Napoli", "Roma", "Lazio", "Atalanta",
    # EPL (Big 6 + Newcastle)
    "Arsenal", "Chelsea", "Liverpool", "Man City", "Man United", "Tottenham",
    "Newcastle",
}


def _load_penalty_lookup() -> dict[int, int]:
    """Load penalty counts per match_id from Sofascore shot data."""
    shots_path = _PROJECT_ROOT / "data" / "external" / "sofascore" / "all_shots_with_xg.parquet"
    if not shots_path.exists():
        log.info("No shot data for penalty extraction at %s", shots_path)
        return {}
    try:
        shots = pd.read_parquet(shots_path, columns=["match_id", "is_penalty"])
        shots["is_penalty"] = pd.to_numeric(shots["is_penalty"], errors="coerce").fillna(0)
        pen = shots[shots["is_penalty"] == 1].groupby("match_id").size().to_dict()
        log.info("Loaded penalty data: %d matches with penalties", len(pen))
        return pen
    except Exception as exc:
        log.warning("Failed to load penalty data: %s", exc)
        return {}


def _load_match_xg_lookup() -> dict[int, float]:
    """Load total xG per match from Sofascore player stats."""
    path = _PROJECT_ROOT / "data" / "external" / "sofascore" / "player_match_stats.parquet"
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(path, columns=["match_id", "xg"])
        df["xg"] = pd.to_numeric(df["xg"], errors="coerce").fillna(0)
        return df.groupby("match_id")["xg"].sum().to_dict()
    except Exception:
        return {}


def add_referee_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add referee tendency features to the match-level table.

    Computed from ALL prior matches officiated by the same referee.

    Original columns (8):
      ref_avg_yellows        - avg total yellow cards per match
      ref_avg_reds           - avg total red cards per match
      ref_avg_fouls          - avg total fouls per match
      ref_avg_penalties      - avg penalties awarded per match (NOW from shot data)
      ref_matches_officiated - number of prior matches
      ref_strictness_score   - cards per foul ratio (normalized)
      ref_home_bias          - home win rate vs league average
      ref_home_cards_bias    - cards to away team vs home team

    Tier 5 additions (7):
      ref_avg_total_goals    - avg total goals in this referee's matches
      ref_avg_xg             - avg total xG in this referee's matches
      ref_xg_vs_league       - referee xG vs league avg (positive = higher scoring)
      ref_strictness_trend   - recent 5-match strictness vs career
      ref_big_match_cards    - avg cards in big matches vs normal matches
      ref_home_team_cards    - avg cards given to home team per match
      ref_away_team_cards    - avg cards given to away team per match
    """
    df = matches.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values("match_date").reset_index(drop=True)

    log.info("Adding referee features (Tier 5 enhanced)...")

    # Load external data sources
    penalty_lookup = _load_penalty_lookup()
    xg_lookup = _load_match_xg_lookup()

    # Pre-compute per-match aggregates
    def _safe_col(col: str) -> pd.Series:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0)
        return pd.Series(0, index=df.index)

    df["_total_yellows"] = _safe_col("home_yellow_cards") + _safe_col("away_yellow_cards")
    df["_total_reds"] = _safe_col("home_red_cards") + _safe_col("away_red_cards")
    df["_total_fouls"] = _safe_col("home_fouls") + _safe_col("away_fouls")
    df["_home_yellows"] = _safe_col("home_yellow_cards")
    df["_away_yellows"] = _safe_col("away_yellow_cards")

    # Penalties from Sofascore shot data (FIX: was always 0 before)
    df["_penalties"] = df["match_id"].map(penalty_lookup).fillna(0).astype(float)

    # Total xG from Sofascore player stats
    df["_match_xg"] = df["match_id"].map(xg_lookup).astype(float)

    # Determine match results for bias calculation
    home_score = _safe_col("home_score")
    away_score = _safe_col("away_score")
    df["_home_win"] = (home_score > away_score).astype(float)
    df["_total_goals"] = home_score + away_score

    # Big-match flag: both teams in top clubs
    home_big = df.get("home_team", pd.Series("", index=df.index)).isin(_BIG_MATCH_TEAMS)
    away_big = df.get("away_team", pd.Series("", index=df.index)).isin(_BIG_MATCH_TEAMS)
    df["_is_big_match"] = (home_big & away_big).astype(float)

    # Track referee history incrementally
    ref_stats: dict[str, dict] = {}

    # All 20 output columns (17 original + 3 regression/leniency)
    col_names = [
        "ref_avg_yellows", "ref_avg_reds", "ref_avg_fouls", "ref_avg_penalties",
        "ref_matches_officiated", "ref_strictness_score", "ref_home_bias",
        "ref_home_cards_bias",
        "ref_avg_total_goals", "ref_avg_xg", "ref_xg_vs_league",
        "ref_strictness_trend", "ref_big_match_cards",
        "ref_home_team_cards", "ref_away_team_cards",
        # Referee×team specific bias
        "ref_vs_home_team_bias", "ref_vs_away_team_bias",
        # Regression-to-mean features (red card leniency after strict match)
        "ref_last_match_reds",      # reds given in referee's previous match
        "ref_last_match_cards",     # total cards in referee's previous match
        "ref_regression_signal",    # deviation from mean in last match (negative = was lenient)
    ]
    results: dict[str, list] = {c: [] for c in col_names}

    # Global league averages (expanding, for comparison)
    league_home_win_rate = df["_home_win"].expanding().mean()
    league_avg_xg = df["_match_xg"].expanding().mean()

    for idx, row in df.iterrows():
        referee = row.get("referee", "")
        if not referee or pd.isna(referee) or referee == "":
            for key in results:
                results[key].append(None)
            continue

        stats = ref_stats.get(referee)

        if stats and stats["count"] > 0:
            n = stats["count"]

            # --- Original 8 features ---
            avg_yellows = stats["yellows"] / n
            avg_reds = stats["reds"] / n
            avg_fouls = stats["fouls"] / n
            avg_pens = stats["pens"] / n

            # Strictness: cards per foul
            if stats["fouls"] > 0:
                strictness = (stats["yellows"] + stats["reds"] * 2) / stats["fouls"]
                strictness_score = min(1.0, max(0.0, (strictness - 0.1) / 0.25))
            else:
                strictness_score = 0.5

            # Home bias
            if n >= 5:
                ref_home_rate = stats["home_wins"] / n
                league_avg = league_home_win_rate.iloc[idx] if idx > 0 else 0.45
                home_bias = ref_home_rate - league_avg
            else:
                home_bias = 0.0

            # Home cards bias
            total_hc = stats["home_yellows"]
            total_ac = stats["away_yellows"]
            cards_bias = ((total_ac - total_hc) / (total_hc + total_ac)
                          if (total_hc + total_ac) > 0 else 0.0)

            results["ref_avg_yellows"].append(round(avg_yellows, 2))
            results["ref_avg_reds"].append(round(avg_reds, 3))
            results["ref_avg_fouls"].append(round(avg_fouls, 1))
            results["ref_avg_penalties"].append(round(avg_pens, 3))
            results["ref_matches_officiated"].append(n)
            results["ref_strictness_score"].append(round(strictness_score, 3))
            results["ref_home_bias"].append(round(home_bias, 3))
            results["ref_home_cards_bias"].append(round(cards_bias, 3))

            # --- Tier 5 new features ---

            # Average total goals in this referee's matches
            avg_goals = stats["total_goals"] / n
            results["ref_avg_total_goals"].append(round(avg_goals, 2))

            # Average xG
            xg_n = stats["xg_count"]
            if xg_n > 0:
                avg_xg = stats["total_xg"] / xg_n
                lxg = league_avg_xg.iloc[idx]
                xg_vs = avg_xg - (lxg if not pd.isna(lxg) else 2.6)
            else:
                avg_xg = None
                xg_vs = 0.0
            results["ref_avg_xg"].append(round(avg_xg, 2) if avg_xg is not None else None)
            results["ref_xg_vs_league"].append(round(xg_vs, 3))

            # Strictness trend: last 5 vs career
            rc = stats["recent_cards"]
            rf = stats["recent_fouls"]
            if len(rc) >= 3 and sum(rf[-5:]) > 0:
                recent_s = sum(rc[-5:]) / sum(rf[-5:])
                career_s = strictness if stats["fouls"] > 0 else 0.15
                trend = recent_s - career_s
            else:
                trend = 0.0
            results["ref_strictness_trend"].append(round(trend, 4))

            # Big-match cards modifier
            bm_n = stats["big_match_count"]
            nm_n = n - bm_n
            if bm_n >= 2 and nm_n >= 2:
                bm_cpg = stats["big_match_cards"] / bm_n
                nm_cpg = (stats["yellows"] - stats["big_match_cards"]) / nm_n
                big_mod = bm_cpg / nm_cpg if nm_cpg > 0 else 1.0
            else:
                big_mod = 1.0
            results["ref_big_match_cards"].append(round(big_mod, 3))

            # Per-side card averages
            results["ref_home_team_cards"].append(round(total_hc / n, 2))
            results["ref_away_team_cards"].append(round(total_ac / n, 2))

            # Referee×team specific bias
            # How many cards does this ref give to THIS specific team vs avg?
            ref_avg_per_team = (stats["yellows"]) / (n * 2)  # avg cards per team per match
            tc = stats.get("team_cards", {})
            tm = stats.get("team_matches", {})

            ht = row.get("home_team", "")
            at = row.get("away_team", "")

            if ht and ht in tc and tm.get(ht, 0) >= 2:
                ht_avg = tc[ht] / tm[ht]
                results["ref_vs_home_team_bias"].append(round(ht_avg - ref_avg_per_team, 3))
            else:
                results["ref_vs_home_team_bias"].append(0.0)

            if at and at in tc and tm.get(at, 0) >= 2:
                at_avg = tc[at] / tm[at]
                results["ref_vs_away_team_bias"].append(round(at_avg - ref_avg_per_team, 3))
            else:
                results["ref_vs_away_team_bias"].append(0.0)

            # --- Regression-to-mean features ---
            # Referees who gave reds/many cards last match tend to be more lenient next
            results["ref_last_match_reds"].append(stats["last_match_reds"])
            results["ref_last_match_cards"].append(stats["last_match_cards"])

            # Regression signal: how much did last match deviate from career average?
            # Positive = last match was stricter than usual → expect leniency
            # Negative = last match was lenient → expect more strictness
            if n >= 3:
                career_cards_per_match = (stats["yellows"] + stats["reds"] * 2) / n
                last_deviation = stats["last_match_cards"] - career_cards_per_match
                results["ref_regression_signal"].append(round(last_deviation, 2))
            else:
                results["ref_regression_signal"].append(0.0)

        else:
            for key in results:
                results[key].append(None if key != "ref_matches_officiated" else 0)

        # --- Update stats with this match ---
        if referee not in ref_stats:
            ref_stats[referee] = {
                "yellows": 0, "reds": 0, "fouls": 0, "pens": 0, "count": 0,
                "home_wins": 0, "home_yellows": 0, "away_yellows": 0,
                "total_goals": 0, "total_xg": 0.0, "xg_count": 0,
                "recent_cards": [], "recent_fouls": [],
                "big_match_count": 0, "big_match_cards": 0,
                "team_cards": {},  # {team_name: total_cards}
                "team_matches": {},  # {team_name: match_count}
                "last_match_reds": 0,   # reds in referee's most recent match
                "last_match_cards": 0,  # total cards in referee's most recent match
            }

        s = ref_stats[referee]
        ty = row.get("_total_yellows", 0) or 0
        tr = row.get("_total_reds", 0) or 0
        tf = row.get("_total_fouls", 0) or 0
        s["yellows"] += ty
        s["reds"] += tr
        s["fouls"] += tf
        # Track last match stats for regression-to-mean features
        s["last_match_reds"] = tr
        s["last_match_cards"] = ty + tr * 2  # weighted: red counts as 2
        s["pens"] += row.get("_penalties", 0) or 0
        s["home_wins"] += row.get("_home_win", 0) or 0
        s["home_yellows"] += row.get("_home_yellows", 0) or 0
        s["away_yellows"] += row.get("_away_yellows", 0) or 0
        s["total_goals"] += row.get("_total_goals", 0) or 0
        s["count"] += 1

        # xG tracking
        mxg = row.get("_match_xg")
        if mxg is not None and not pd.isna(mxg):
            s["total_xg"] += mxg
            s["xg_count"] += 1

        # Recent window for trend
        s["recent_cards"].append(ty + tr * 2)
        s["recent_fouls"].append(tf)

        # Big-match tracking
        if row.get("_is_big_match", 0):
            s["big_match_count"] += 1
            s["big_match_cards"] += ty

        # Team-specific card and match tracking
        ht = row.get("home_team", "")
        at = row.get("away_team", "")
        if ht:
            s["team_cards"][ht] = s["team_cards"].get(ht, 0) + (row.get("_home_yellows", 0) or 0)
            s["team_matches"][ht] = s["team_matches"].get(ht, 0) + 1
        if at:
            s["team_cards"][at] = s["team_cards"].get(at, 0) + (row.get("_away_yellows", 0) or 0)
            s["team_matches"][at] = s["team_matches"].get(at, 0) + 1

    # Assign result columns
    for col, values in results.items():
        df[col] = values

    # Cleanup temp columns
    df.drop(
        columns=["_total_yellows", "_total_reds", "_total_fouls",
                 "_home_yellows", "_away_yellows", "_home_win",
                 "_penalties", "_match_xg", "_total_goals", "_is_big_match"],
        inplace=True, errors="ignore",
    )

    n_with = df["ref_matches_officiated"].notna().sum()
    log.info("Added 20 referee features to %d matches", n_with)
    return df


def get_referee_profile(
    referee_name: str,
    matches_df: pd.DataFrame,
    before_date: Optional[pd.Timestamp] = None,
) -> dict:
    """Get detailed profile for a specific referee.

    Args:
        referee_name: Referee's name
        matches_df: DataFrame with match data
        before_date: Only use matches before this date

    Returns:
        Dictionary with referee statistics and tendencies including
        penalty rate, xG impact, big-match behavior, and strictness trend.
    """
    df = matches_df.copy()

    if before_date:
        df["match_date"] = pd.to_datetime(df["match_date"])
        df = df[df["match_date"] < before_date]

    # Normalize whitespace in both the query name and the data column
    norm_name = _normalize_ws(referee_name)
    df["_ref_norm"] = df["referee"].apply(_normalize_ws)
    ref_matches = df[df["_ref_norm"] == norm_name]
    df.drop(columns=["_ref_norm"], inplace=True, errors="ignore")

    if ref_matches.empty:
        return {"referee": referee_name, "matches": 0, "data_available": False}

    n = len(ref_matches)

    def safe_mean(col):
        if col in ref_matches.columns:
            return pd.to_numeric(ref_matches[col], errors="coerce").mean()
        return 0

    total_yellows = safe_mean("home_yellow_cards") + safe_mean("away_yellow_cards")
    total_reds = safe_mean("home_red_cards") + safe_mean("away_red_cards")
    total_fouls = safe_mean("home_fouls") + safe_mean("away_fouls")

    home_score = pd.to_numeric(ref_matches.get("home_score", 0), errors="coerce")
    away_score = pd.to_numeric(ref_matches.get("away_score", 0), errors="coerce")
    home_wins = (home_score > away_score).sum()
    draws = (home_score == away_score).sum()
    away_wins = (home_score < away_score).sum()

    # Penalty data from shots
    pen_lookup = _load_penalty_lookup()
    pen_counts = ref_matches["match_id"].map(pen_lookup).fillna(0)
    avg_pens = pen_counts.mean()

    # xG data
    xg_lookup = _load_match_xg_lookup()
    xg_vals = ref_matches["match_id"].map(xg_lookup).dropna()
    avg_xg = xg_vals.mean() if len(xg_vals) > 0 else None

    # Big-match stats
    home_big = ref_matches.get("home_team", pd.Series("")).isin(_BIG_MATCH_TEAMS)
    away_big = ref_matches.get("away_team", pd.Series("")).isin(_BIG_MATCH_TEAMS)
    big_mask = home_big & away_big
    big_n = big_mask.sum()
    if big_n > 0:
        big_yellows = (
            safe_mean("home_yellow_cards") + safe_mean("away_yellow_cards")
        )
        # Recalculate for big matches only
        bm = ref_matches[big_mask]
        big_avg_y = (
            pd.to_numeric(bm.get("home_yellow_cards", 0), errors="coerce").fillna(0)
            + pd.to_numeric(bm.get("away_yellow_cards", 0), errors="coerce").fillna(0)
        ).mean()
    else:
        big_avg_y = None

    # Strictness trend: last 5 vs career
    recent = ref_matches.tail(5)
    recent_fouls = (
        pd.to_numeric(recent.get("home_fouls", 0), errors="coerce").fillna(0)
        + pd.to_numeric(recent.get("away_fouls", 0), errors="coerce").fillna(0)
    ).sum()
    recent_cards = (
        pd.to_numeric(recent.get("home_yellow_cards", 0), errors="coerce").fillna(0)
        + pd.to_numeric(recent.get("away_yellow_cards", 0), errors="coerce").fillna(0)
    ).sum()
    if recent_fouls > 0 and total_fouls > 0:
        trend = (recent_cards / recent_fouls) - (total_yellows / total_fouls)
    else:
        trend = 0.0

    return {
        "referee": referee_name,
        "matches": n,
        "data_available": True,
        "avg_yellows_per_match": round(total_yellows, 2),
        "avg_reds_per_match": round(total_reds, 3),
        "avg_fouls_per_match": round(total_fouls, 1),
        "avg_penalties_per_match": round(avg_pens, 3),
        "avg_total_goals": round((home_score + away_score).mean(), 2),
        "avg_xg": round(avg_xg, 2) if avg_xg is not None else None,
        "home_win_pct": round(home_wins / n * 100, 1) if n > 0 else 0,
        "draw_pct": round(draws / n * 100, 1) if n > 0 else 0,
        "away_win_pct": round(away_wins / n * 100, 1) if n > 0 else 0,
        "big_matches": int(big_n),
        "big_match_avg_yellows": round(big_avg_y, 2) if big_avg_y is not None else None,
        "strictness_trend_last5": round(trend, 4),
    }


def get_referee_team_history(
    referee_name: str,
    team_name: str,
    matches_df: pd.DataFrame,
    before_date: Optional[pd.Timestamp] = None,
) -> dict:
    """Get referee×team specific card history.

    Returns how many cards this referee typically gives to this team,
    compared to the referee's average and the team's average from other refs.

    Args:
        referee_name: Referee's name
        team_name: Team name
        matches_df: DataFrame with match data
        before_date: Only use matches before this date

    Returns:
        Dictionary with referee-team interaction statistics.
    """
    df = matches_df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    if before_date:
        df = df[df["match_date"] < before_date]

    # Normalize whitespace in referee names (handles \xa0 from parquet data)
    norm_name = _normalize_ws(referee_name)
    df["_ref_norm"] = df["referee"].apply(_normalize_ws)

    # Matches where this referee officiated this team
    team_home = (df["_ref_norm"] == norm_name) & (df["home_team"] == team_name)
    team_away = (df["_ref_norm"] == norm_name) & (df["away_team"] == team_name)

    def _safe(col):
        return pd.to_numeric(df[col], errors="coerce").fillna(0) if col in df.columns else 0

    # Cards given to this team by this referee
    cards_when_home = _safe("home_yellow_cards")[team_home].sum()
    cards_when_away = _safe("away_yellow_cards")[team_away].sum()
    n_matches = team_home.sum() + team_away.sum()

    if n_matches == 0:
        return {
            "referee": referee_name, "team": team_name,
            "matches": 0, "data_available": False,
        }

    avg_cards_from_ref = (cards_when_home + cards_when_away) / n_matches

    # Referee's overall average cards per team per match
    ref_all = df[df["_ref_norm"] == norm_name]
    ref_n = len(ref_all)
    if ref_n > 0:
        ref_total_cards = (
            _safe("home_yellow_cards")[df["_ref_norm"] == norm_name].sum()
            + _safe("away_yellow_cards")[df["_ref_norm"] == norm_name].sum()
        )
        ref_avg_per_team = ref_total_cards / (ref_n * 2)  # 2 teams per match
    else:
        ref_avg_per_team = 0

    # Team's average cards from ALL referees
    team_all_home = df["home_team"] == team_name
    team_all_away = df["away_team"] == team_name
    team_total_cards = (
        _safe("home_yellow_cards")[team_all_home].sum()
        + _safe("away_yellow_cards")[team_all_away].sum()
    )
    team_total_matches = team_all_home.sum() + team_all_away.sum()
    team_avg_cards = team_total_cards / team_total_matches if team_total_matches > 0 else 0

    return {
        "referee": referee_name,
        "team": team_name,
        "matches": int(n_matches),
        "data_available": True,
        "avg_cards_from_this_ref": round(avg_cards_from_ref, 2),
        "ref_avg_cards_per_team": round(ref_avg_per_team, 2),
        "team_avg_cards_all_refs": round(team_avg_cards, 2),
        # Positive = ref gives MORE cards to this team than usual
        "ref_team_bias": round(avg_cards_from_ref - ref_avg_per_team, 3),
        # Positive = team gets MORE cards from this ref than from others
        "team_ref_effect": round(avg_cards_from_ref - team_avg_cards, 3),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing referee module (Tier 5 enhanced)...")
    print("=" * 60)

    from storage.paths import features_path

    feat_path = features_path()
    if feat_path.exists():
        df = pd.read_parquet(feat_path)
        print(f"Loaded {len(df)} matches")

        ref_counts = df[df["referee"].notna()]["referee"].value_counts()
        print(f"\nTop 10 referees by match count:")
        for ref, count in ref_counts.head(10).items():
            print(f"  {ref}: {count} matches")

        if len(ref_counts) > 0:
            top_ref = ref_counts.index[0]
            profile = get_referee_profile(top_ref, df)
            print(f"\nProfile for {top_ref}:")
            for k, v in profile.items():
                print(f"  {k}: {v}")

            # Test referee×team history
            if "home_team" in df.columns:
                teams = df["home_team"].dropna().unique()[:3]
                for team in teams:
                    hist = get_referee_team_history(top_ref, team, df)
                    if hist["data_available"]:
                        print(f"\n  {top_ref} × {team}:")
                        print(f"    matches: {hist['matches']}")
                        print(f"    avg cards: {hist['avg_cards_from_this_ref']}")
                        print(f"    ref bias: {hist['ref_team_bias']}")
                        print(f"    team effect: {hist['team_ref_effect']}")
    else:
        print(f"Features file not found at {feat_path}")
