#!/usr/bin/env python3
"""Advanced Predictions - Mining ALL Available Data!

This script adds predictions for:
1. REFEREE PREDICTIONS - Bias, strictness, card tendencies
2. WEATHER PREDICTIONS - Rain, temperature, wind impacts
3. TIMING PREDICTIONS - First 15 min, last 15 min, Fergie time
4. FORMATION PREDICTIONS - Tactical matchups
5. BIG GAME PREDICTIONS - Top 6 clashes, relegation battles
6. COMBINED CONTEXT - Multi-factor predictions

Using ALL available data that was previously ignored!
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_all_data():
    """Load all available datasets."""
    data = {}

    # Main features
    data["features"] = pd.read_parquet(DATA_DIR / "features" / "features.parquet")
    log.info(f"Loaded features: {len(data['features'])} matches")

    # Shots (for timing)
    try:
        data["shots"] = pd.read_parquet(DATA_DIR / "parsed" / "shots.parquet")
        data["shots"]["minute_num"] = pd.to_numeric(data["shots"]["minute"], errors="coerce")
        log.info(f"Loaded shots: {len(data['shots'])} shots")
    except:
        data["shots"] = None

    # Referee data
    try:
        data["refs"] = pd.read_parquet(DATA_DIR / "external" / "referee" / "referee_assignments.parquet")
        log.info(f"Loaded referee data: {len(data['refs'])} matches")
    except:
        data["refs"] = None

    # Weather data
    try:
        data["weather"] = pd.read_parquet(DATA_DIR / "external" / "weather.parquet")
        log.info(f"Loaded weather data: {len(data['weather'])} matches")
    except:
        data["weather"] = None

    # Lineups (for formation)
    try:
        data["lineups"] = pd.read_parquet(DATA_DIR / "parsed" / "lineups.parquet")
        log.info(f"Loaded lineups: {len(data['lineups'])} entries")
    except:
        data["lineups"] = None

    return data


def analyze_referee_predictions(df: pd.DataFrame, refs_df: pd.DataFrame) -> Dict:
    """Analyze referee-related predictions."""
    results = {}

    log.info("\n" + "=" * 70)
    log.info("REFEREE PREDICTIONS")
    log.info("=" * 70)

    if refs_df is None or df is None:
        log.info("No referee data available")
        return results

    # Merge referee data with main features
    refs_df["match_key"] = refs_df["home_team"] + "_" + refs_df["away_team"] + "_" + refs_df["season"].astype(str)
    df["match_key"] = df["home_team"] + "_" + df["away_team"] + "_" + df["season"].astype(str)

    merged = df.merge(refs_df[["match_key", "referee", "ref_yellows", "ref_reds"]],
                      on="match_key", how="inner", suffixes=("", "_ref"))

    log.info(f"Matches with referee data: {len(merged)}")

    # 1. Referee Card Tendency
    log.info("\n1. REFEREE CARD TENDENCY")
    log.info("-" * 50)

    ref_stats = merged.groupby("referee").agg({
        "match_key": "count",
        "ref_yellows": "mean",
        "home_score": "mean",
        "away_score": "mean",
    }).rename(columns={"match_key": "matches"})

    ref_stats = ref_stats[ref_stats["matches"] >= 20]  # Min 20 matches
    ref_stats["avg_goals"] = ref_stats["home_score"] + ref_stats["away_score"]

    # Classify referees
    avg_yellows = ref_stats["ref_yellows"].mean()
    ref_stats["is_strict"] = (ref_stats["ref_yellows"] > avg_yellows + 0.5).astype(int)

    strict_refs = ref_stats[ref_stats["is_strict"] == 1].index.tolist()
    lenient_refs = ref_stats[ref_stats["is_strict"] == 0].index.tolist()

    log.info(f"Strict referees (avg > {avg_yellows + 0.5:.1f} yellows): {len(strict_refs)}")
    log.info(f"Lenient referees: {len(lenient_refs)}")

    # Analyze strict vs lenient matches
    merged["strict_ref"] = merged["referee"].isin(strict_refs).astype(int)

    strict_matches = merged[merged["strict_ref"] == 1]
    lenient_matches = merged[merged["strict_ref"] == 0]

    if len(strict_matches) > 100 and len(lenient_matches) > 100:
        strict_cards = strict_matches["ref_yellows"].mean()
        lenient_cards = lenient_matches["ref_yellows"].mean()
        strict_goals = (strict_matches["home_score"] + strict_matches["away_score"]).mean()
        lenient_goals = (lenient_matches["home_score"] + lenient_matches["away_score"]).mean()

        log.info(f"\nStrict ref matches: {len(strict_matches)}")
        log.info(f"  Avg yellows: {strict_cards:.2f} vs {lenient_cards:.2f} (lenient)")
        log.info(f"  Avg goals: {strict_goals:.2f} vs {lenient_goals:.2f} (lenient)")

        results["referee_strictness"] = {
            "strict_matches": len(strict_matches),
            "lenient_matches": len(lenient_matches),
            "strict_avg_yellows": strict_cards,
            "lenient_avg_yellows": lenient_cards,
            "strict_avg_goals": strict_goals,
            "lenient_avg_goals": lenient_goals,
            "card_difference": strict_cards - lenient_cards,
            "strict_refs": strict_refs[:10],  # Top 10
        }

        # Prediction: Can we predict high card matches based on referee?
        merged["high_cards"] = (merged["ref_yellows"] >= 5).astype(int)
        base_rate = merged["high_cards"].mean()
        # Recalculate strict matches with the new column
        strict_matches = merged[merged["strict_ref"] == 1]
        strict_high_card_rate = strict_matches["high_cards"].mean() if len(strict_matches) > 0 else 0

        log.info(f"\nHigh card match (5+) base rate: {base_rate:.1%}")
        log.info(f"High card rate with strict ref: {strict_high_card_rate:.1%}")

        results["high_card_prediction"] = {
            "base_rate": base_rate,
            "strict_ref_rate": strict_high_card_rate,
            "lift": strict_high_card_rate - base_rate,
        }

    # 2. Referee Home Bias
    log.info("\n2. REFEREE HOME BIAS")
    log.info("-" * 50)

    merged["home_win"] = (merged["home_score"] > merged["away_score"]).astype(int)
    merged["away_win"] = (merged["away_score"] > merged["home_score"]).astype(int)
    merged["draw"] = (merged["home_score"] == merged["away_score"]).astype(int)

    ref_home_bias = merged.groupby("referee").agg({
        "home_win": "mean",
        "match_key": "count",
    }).rename(columns={"match_key": "matches"})

    ref_home_bias = ref_home_bias[ref_home_bias["matches"] >= 20]
    overall_home_win = merged["home_win"].mean()

    ref_home_bias["home_bias"] = ref_home_bias["home_win"] - overall_home_win

    # Top home-favoring refs
    home_favoring = ref_home_bias.nlargest(5, "home_bias")
    away_favoring = ref_home_bias.nsmallest(5, "home_bias")

    log.info(f"Overall home win rate: {overall_home_win:.1%}")
    log.info("\nMost home-favoring referees:")
    for ref, row in home_favoring.iterrows():
        log.info(f"  {ref}: {row['home_win']:.1%} home wins (+{row['home_bias']:.1%}), {int(row['matches'])} matches")

    log.info("\nMost away-favoring referees:")
    for ref, row in away_favoring.iterrows():
        log.info(f"  {ref}: {row['home_win']:.1%} home wins ({row['home_bias']:.1%}), {int(row['matches'])} matches")

    results["referee_home_bias"] = {
        "overall_home_win_rate": overall_home_win,
        "home_favoring_refs": home_favoring.to_dict(),
        "away_favoring_refs": away_favoring.to_dict(),
    }

    return results


def analyze_weather_predictions(df: pd.DataFrame, weather_df: pd.DataFrame) -> Dict:
    """Analyze weather-related predictions."""
    results = {}

    log.info("\n" + "=" * 70)
    log.info("WEATHER PREDICTIONS")
    log.info("=" * 70)

    if weather_df is None:
        log.info("No weather data available")
        return results

    # Merge weather with features
    merged = df.merge(weather_df, on="match_id", how="inner")
    log.info(f"Matches with weather data: {len(merged)}")

    if len(merged) < 500:
        log.info("Insufficient weather data")
        return results

    merged["total_goals"] = merged["home_score"] + merged["away_score"]

    # 1. Temperature Impact
    log.info("\n1. TEMPERATURE IMPACT")
    log.info("-" * 50)

    # Cold = below 10°C, Hot = above 25°C
    cold_matches = merged[merged["weather_temperature_2m_mean"] < 10]
    hot_matches = merged[merged["weather_temperature_2m_mean"] > 25]
    normal_matches = merged[(merged["weather_temperature_2m_mean"] >= 10) &
                            (merged["weather_temperature_2m_mean"] <= 25)]

    if len(cold_matches) > 100 and len(hot_matches) > 50:
        cold_goals = cold_matches["total_goals"].mean()
        hot_goals = hot_matches["total_goals"].mean()
        normal_goals = normal_matches["total_goals"].mean()

        log.info(f"Cold weather (<10°C): {len(cold_matches)} matches, {cold_goals:.2f} goals/match")
        log.info(f"Normal weather (10-25°C): {len(normal_matches)} matches, {normal_goals:.2f} goals/match")
        log.info(f"Hot weather (>25°C): {len(hot_matches)} matches, {hot_goals:.2f} goals/match")

        results["temperature_impact"] = {
            "cold_matches": len(cold_matches),
            "cold_goals": cold_goals,
            "hot_matches": len(hot_matches),
            "hot_goals": hot_goals,
            "normal_goals": normal_goals,
        }

    # 2. Rain Impact
    log.info("\n2. RAIN IMPACT")
    log.info("-" * 50)

    # Rainy = precipitation > 5mm
    rainy_matches = merged[merged["weather_rain_sum"] > 5]
    dry_matches = merged[merged["weather_rain_sum"] <= 1]

    if len(rainy_matches) > 100 and len(dry_matches) > 500:
        rainy_goals = rainy_matches["total_goals"].mean()
        dry_goals = dry_matches["total_goals"].mean()

        log.info(f"Rainy matches (>5mm): {len(rainy_matches)} matches, {rainy_goals:.2f} goals/match")
        log.info(f"Dry matches (<=1mm): {len(dry_matches)} matches, {dry_goals:.2f} goals/match")

        # Home win rates
        rainy_home_win = (rainy_matches["home_score"] > rainy_matches["away_score"]).mean()
        dry_home_win = (dry_matches["home_score"] > dry_matches["away_score"]).mean()

        log.info(f"Home win rate in rain: {rainy_home_win:.1%}")
        log.info(f"Home win rate when dry: {dry_home_win:.1%}")

        results["rain_impact"] = {
            "rainy_matches": len(rainy_matches),
            "rainy_goals": rainy_goals,
            "dry_matches": len(dry_matches),
            "dry_goals": dry_goals,
            "rainy_home_win": rainy_home_win,
            "dry_home_win": dry_home_win,
        }

    # 3. Wind Impact
    log.info("\n3. WIND IMPACT")
    log.info("-" * 50)

    # High wind = > 30 km/h
    windy_matches = merged[merged["weather_wind_speed_10m_max"] > 30]
    calm_matches = merged[merged["weather_wind_speed_10m_max"] < 15]

    if len(windy_matches) > 50 and len(calm_matches) > 500:
        windy_goals = windy_matches["total_goals"].mean()
        calm_goals = calm_matches["total_goals"].mean()

        log.info(f"Windy matches (>30 km/h): {len(windy_matches)} matches, {windy_goals:.2f} goals/match")
        log.info(f"Calm matches (<15 km/h): {len(calm_matches)} matches, {calm_goals:.2f} goals/match")

        results["wind_impact"] = {
            "windy_matches": len(windy_matches),
            "windy_goals": windy_goals,
            "calm_matches": len(calm_matches),
            "calm_goals": calm_goals,
        }

    return results


def analyze_timing_predictions(df: pd.DataFrame, shots_df: pd.DataFrame) -> Dict:
    """Analyze goal timing predictions."""
    results = {}

    log.info("\n" + "=" * 70)
    log.info("GOAL TIMING PREDICTIONS")
    log.info("=" * 70)

    if shots_df is None:
        log.info("No shots data available")
        return results

    goals = shots_df[shots_df["outcome"] == "Goal"].copy()
    log.info(f"Total goals in shot data: {len(goals)}")

    # 1. First 15 Minutes Analysis
    log.info("\n1. FIRST 15 MINUTES")
    log.info("-" * 50)

    early_goals = goals[goals["minute_num"] <= 15]
    log.info(f"Goals in first 15 min: {len(early_goals)} ({len(early_goals)/len(goals)*100:.1f}%)")

    # Goals per match in first 15 min
    goals_by_match = goals.groupby("match_id").apply(
        lambda x: (x["minute_num"] <= 15).sum()
    ).reset_index(name="early_goals")

    matches_with_early_goal = (goals_by_match["early_goals"] > 0).sum()
    total_matches_in_shots = goals_by_match["match_id"].nunique()

    early_goal_rate = matches_with_early_goal / total_matches_in_shots if total_matches_in_shots > 0 else 0
    log.info(f"Matches with goal in first 15 min: {matches_with_early_goal}/{total_matches_in_shots} ({early_goal_rate:.1%})")

    results["first_15_min"] = {
        "early_goals": len(early_goals),
        "early_goal_pct": len(early_goals) / len(goals) * 100,
        "matches_with_early_goal_rate": early_goal_rate,
    }

    # 2. Late Drama (75-90 min)
    log.info("\n2. LATE DRAMA (75-90 min)")
    log.info("-" * 50)

    late_goals = goals[(goals["minute_num"] >= 75) & (goals["minute_num"] <= 90)]
    log.info(f"Goals in final 15 min: {len(late_goals)} ({len(late_goals)/len(goals)*100:.1f}%)")

    # "Fergie Time" - 85-90 min
    fergie_goals = goals[(goals["minute_num"] >= 85) & (goals["minute_num"] <= 90)]
    log.info(f"Goals in Fergie time (85-90): {len(fergie_goals)} ({len(fergie_goals)/len(goals)*100:.1f}%)")

    results["late_drama"] = {
        "late_goals_75_90": len(late_goals),
        "late_goal_pct": len(late_goals) / len(goals) * 100,
        "fergie_time_goals": len(fergie_goals),
        "fergie_time_pct": len(fergie_goals) / len(goals) * 100,
    }

    # 3. First Half vs Second Half
    log.info("\n3. FIRST HALF vs SECOND HALF")
    log.info("-" * 50)

    first_half_goals = goals[goals["minute_num"] <= 45]
    second_half_goals = goals[goals["minute_num"] > 45]

    log.info(f"First half goals: {len(first_half_goals)} ({len(first_half_goals)/len(goals)*100:.1f}%)")
    log.info(f"Second half goals: {len(second_half_goals)} ({len(second_half_goals)/len(goals)*100:.1f}%)")

    results["half_comparison"] = {
        "first_half_goals": len(first_half_goals),
        "first_half_pct": len(first_half_goals) / len(goals) * 100,
        "second_half_goals": len(second_half_goals),
        "second_half_pct": len(second_half_goals) / len(goals) * 100,
    }

    # 4. Goal Timing by Period
    log.info("\n4. GOALS BY 15-MIN PERIOD")
    log.info("-" * 50)

    periods = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 90)]
    period_stats = {}

    for start, end in periods:
        period_goals = goals[(goals["minute_num"] > start) & (goals["minute_num"] <= end)]
        pct = len(period_goals) / len(goals) * 100
        period_stats[f"{start}-{end}"] = {
            "goals": len(period_goals),
            "pct": pct,
        }
        log.info(f"  {start}-{end} min: {len(period_goals)} goals ({pct:.1f}%)")

    results["goals_by_period"] = period_stats

    return results


def analyze_formation_predictions(df: pd.DataFrame) -> Dict:
    """Analyze formation-related predictions."""
    results = {}

    log.info("\n" + "=" * 70)
    log.info("FORMATION PREDICTIONS")
    log.info("=" * 70)

    # Check if formation data exists
    if "home_formation" not in df.columns or df["home_formation"].isna().all():
        log.info("No formation data available")
        return results

    formation_df = df[df["home_formation"].notna() & df["away_formation"].notna()].copy()
    log.info(f"Matches with formation data: {len(formation_df)}")

    if len(formation_df) < 100:
        log.info("Insufficient formation data")
        return results

    # 1. Formation Win Rates
    log.info("\n1. FORMATION WIN RATES")
    log.info("-" * 50)

    formation_df["home_win"] = (formation_df["home_score"] > formation_df["away_score"]).astype(int)
    formation_df["away_win"] = (formation_df["away_score"] > formation_df["home_score"]).astype(int)
    formation_df["draw"] = (formation_df["home_score"] == formation_df["away_score"]).astype(int)
    formation_df["total_goals"] = formation_df["home_score"] + formation_df["away_score"]

    home_form_stats = formation_df.groupby("home_formation").agg({
        "home_win": ["mean", "count"],
        "total_goals": "mean",
    })
    home_form_stats.columns = ["win_rate", "matches", "avg_goals"]
    home_form_stats = home_form_stats[home_form_stats["matches"] >= 10].sort_values("win_rate", ascending=False)

    log.info("\nBest formations when playing at HOME:")
    for formation, row in home_form_stats.head(5).iterrows():
        log.info(f"  {formation}: {row['win_rate']:.1%} win rate, {int(row['matches'])} matches, {row['avg_goals']:.2f} goals/match")

    away_form_stats = formation_df.groupby("away_formation").agg({
        "away_win": ["mean", "count"],
        "total_goals": "mean",
    })
    away_form_stats.columns = ["win_rate", "matches", "avg_goals"]
    away_form_stats = away_form_stats[away_form_stats["matches"] >= 10].sort_values("win_rate", ascending=False)

    log.info("\nBest formations when playing AWAY:")
    for formation, row in away_form_stats.head(5).iterrows():
        log.info(f"  {formation}: {row['win_rate']:.1%} win rate, {int(row['matches'])} matches")

    results["home_formation_stats"] = home_form_stats.to_dict()
    results["away_formation_stats"] = away_form_stats.to_dict()

    # 2. Formation Matchups
    log.info("\n2. FORMATION MATCHUPS")
    log.info("-" * 50)

    formation_df["matchup"] = formation_df["home_formation"] + " vs " + formation_df["away_formation"]

    matchup_stats = formation_df.groupby("matchup").agg({
        "home_win": "mean",
        "draw": "mean",
        "total_goals": "mean",
        "home_formation": "count",
    }).rename(columns={"home_formation": "matches"})

    matchup_stats = matchup_stats[matchup_stats["matches"] >= 5].sort_values("matches", ascending=False)

    log.info("\nMost common formation matchups:")
    for matchup, row in matchup_stats.head(10).iterrows():
        log.info(f"  {matchup}: {int(row['matches'])} matches, home wins {row['home_win']:.1%}, draws {row['draw']:.1%}")

    results["matchup_stats"] = matchup_stats.head(20).to_dict()

    # 3. Formation Style Analysis
    log.info("\n3. FORMATION STYLE ANALYSIS")
    log.info("-" * 50)

    # Classify formations
    def classify_formation(f):
        if f is None:
            return "Unknown"
        f = str(f)
        # Count defenders (first number)
        try:
            defenders = int(f.split("-")[0])
            if defenders >= 5:
                return "Defensive (5+ back)"
            elif defenders == 4:
                return "Balanced (4 back)"
            else:
                return "Attacking (3 back)"
        except:
            return "Unknown"

    formation_df["home_style"] = formation_df["home_formation"].apply(classify_formation)
    formation_df["away_style"] = formation_df["away_formation"].apply(classify_formation)

    style_matchup = formation_df.groupby(["home_style", "away_style"]).agg({
        "home_win": "mean",
        "total_goals": "mean",
        "home_formation": "count",
    }).rename(columns={"home_formation": "matches"})

    log.info("\nStyle matchup results:")
    for (home_style, away_style), row in style_matchup.iterrows():
        if row["matches"] >= 5:
            log.info(f"  {home_style} vs {away_style}: {int(row['matches'])} matches, home wins {row['home_win']:.1%}, {row['total_goals']:.2f} goals")

    # Convert tuple keys to strings for JSON serialization
    style_dict = {}
    for (home_style, away_style), row in style_matchup.iterrows():
        key = f"{home_style} vs {away_style}"
        style_dict[key] = row.to_dict()
    results["style_matchups"] = style_dict

    return results


def analyze_big_game_predictions(df: pd.DataFrame) -> Dict:
    """Analyze big game predictions - top 6 clashes, relegation battles."""
    results = {}

    log.info("\n" + "=" * 70)
    log.info("BIG GAME PREDICTIONS")
    log.info("=" * 70)

    # Check for league position data
    if "home_league_position" not in df.columns:
        log.info("No league position data available")
        return results

    df = df.copy()
    df["total_goals"] = df["home_score"] + df["away_score"]
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["draw"] = (df["home_score"] == df["away_score"]).astype(int)

    # 1. Top 6 Clashes
    log.info("\n1. TOP 6 CLASHES")
    log.info("-" * 50)

    df["top6_clash"] = ((df["home_league_position"] <= 6) & (df["away_league_position"] <= 6)).astype(int)
    top6_matches = df[df["top6_clash"] == 1]
    other_matches = df[df["top6_clash"] == 0]

    if len(top6_matches) > 50:
        top6_goals = top6_matches["total_goals"].mean()
        other_goals = other_matches["total_goals"].mean()
        top6_draws = top6_matches["draw"].mean()
        other_draws = other_matches["draw"].mean()

        log.info(f"Top 6 clashes: {len(top6_matches)} matches")
        log.info(f"  Avg goals: {top6_goals:.2f} vs {other_goals:.2f} (other)")
        log.info(f"  Draw rate: {top6_draws:.1%} vs {other_draws:.1%} (other)")

        results["top6_clashes"] = {
            "matches": len(top6_matches),
            "avg_goals": top6_goals,
            "draw_rate": top6_draws,
            "other_avg_goals": other_goals,
            "other_draw_rate": other_draws,
        }

    # 2. Relegation Battles
    log.info("\n2. RELEGATION BATTLES")
    log.info("-" * 50)

    df["relegation_battle"] = ((df["home_league_position"] >= 15) & (df["away_league_position"] >= 15)).astype(int)
    relegation_matches = df[df["relegation_battle"] == 1]

    if len(relegation_matches) > 50:
        rel_goals = relegation_matches["total_goals"].mean()
        rel_draws = relegation_matches["draw"].mean()
        rel_home_win = relegation_matches["home_win"].mean()

        log.info(f"Relegation battles: {len(relegation_matches)} matches")
        log.info(f"  Avg goals: {rel_goals:.2f}")
        log.info(f"  Draw rate: {rel_draws:.1%}")
        log.info(f"  Home win rate: {rel_home_win:.1%}")

        results["relegation_battles"] = {
            "matches": len(relegation_matches),
            "avg_goals": rel_goals,
            "draw_rate": rel_draws,
            "home_win_rate": rel_home_win,
        }

    # 3. Top vs Bottom (Mismatch)
    log.info("\n3. TOP vs BOTTOM (MISMATCH)")
    log.info("-" * 50)

    df["mismatch_home_top"] = ((df["home_league_position"] <= 6) & (df["away_league_position"] >= 15)).astype(int)
    df["mismatch_away_top"] = ((df["away_league_position"] <= 6) & (df["home_league_position"] >= 15)).astype(int)

    home_top_matches = df[df["mismatch_home_top"] == 1]
    away_top_matches = df[df["mismatch_away_top"] == 1]

    if len(home_top_matches) > 50:
        home_top_win = home_top_matches["home_win"].mean()
        home_top_goals = home_top_matches["total_goals"].mean()

        log.info(f"Top team at HOME vs bottom: {len(home_top_matches)} matches")
        log.info(f"  Top team (home) win rate: {home_top_win:.1%}")
        log.info(f"  Avg goals: {home_top_goals:.2f}")

        results["top_at_home_vs_bottom"] = {
            "matches": len(home_top_matches),
            "top_team_win_rate": home_top_win,
            "avg_goals": home_top_goals,
        }

    if len(away_top_matches) > 50:
        away_top_win = (away_top_matches["away_score"] > away_top_matches["home_score"]).mean()
        away_top_goals = away_top_matches["total_goals"].mean()

        log.info(f"\nTop team AWAY vs bottom: {len(away_top_matches)} matches")
        log.info(f"  Top team (away) win rate: {away_top_win:.1%}")
        log.info(f"  Avg goals: {away_top_goals:.2f}")

        results["top_away_vs_bottom"] = {
            "matches": len(away_top_matches),
            "top_team_win_rate": away_top_win,
            "avg_goals": away_top_goals,
        }

    # 4. Mid-Table Matches
    log.info("\n4. MID-TABLE MATCHES")
    log.info("-" * 50)

    df["mid_table"] = ((df["home_league_position"] >= 7) & (df["home_league_position"] <= 14) &
                       (df["away_league_position"] >= 7) & (df["away_league_position"] <= 14)).astype(int)
    mid_matches = df[df["mid_table"] == 1]

    if len(mid_matches) > 50:
        mid_goals = mid_matches["total_goals"].mean()
        mid_draws = mid_matches["draw"].mean()

        log.info(f"Mid-table matches: {len(mid_matches)} matches")
        log.info(f"  Avg goals: {mid_goals:.2f}")
        log.info(f"  Draw rate: {mid_draws:.1%}")

        results["mid_table_matches"] = {
            "matches": len(mid_matches),
            "avg_goals": mid_goals,
            "draw_rate": mid_draws,
        }

    return results


def analyze_combined_predictions(df: pd.DataFrame, refs_df: pd.DataFrame, weather_df: pd.DataFrame) -> Dict:
    """Analyze combined multi-factor predictions."""
    results = {}

    log.info("\n" + "=" * 70)
    log.info("COMBINED MULTI-FACTOR PREDICTIONS")
    log.info("=" * 70)

    # Merge all available data
    merged = df.copy()

    if refs_df is not None:
        refs_df["match_key"] = refs_df["home_team"] + "_" + refs_df["away_team"] + "_" + refs_df["season"].astype(str)
        merged["match_key"] = merged["home_team"] + "_" + merged["away_team"] + "_" + merged["season"].astype(str)
        merged = merged.merge(refs_df[["match_key", "referee", "ref_yellows"]],
                              on="match_key", how="left", suffixes=("", "_ref"))

    if weather_df is not None:
        merged = merged.merge(weather_df, on="match_id", how="left")

    merged["total_goals"] = merged["home_score"] + merged["away_score"]
    merged["home_win"] = (merged["home_score"] > merged["away_score"]).astype(int)

    # 1. Derby + Big Stadium + Strict Ref = Card Explosion?
    log.info("\n1. CARD EXPLOSION CONDITIONS")
    log.info("-" * 50)

    if "matchup_competitiveness" in merged.columns and "home_stadium_capacity" in merged.columns:
        # High stakes derby at big stadium
        merged["derby"] = (merged["matchup_competitiveness"] > 0.8).astype(int) if "matchup_competitiveness" in merged.columns else 0
        merged["big_stadium"] = (merged["home_stadium_capacity"] > 50000).astype(int) if "home_stadium_capacity" in merged.columns else 0

        # Card explosion = 6+ yellows
        if "home_yellow_cards" in merged.columns:
            merged["total_yellows"] = merged["home_yellow_cards"].fillna(0) + merged["away_yellow_cards"].fillna(0)
            merged["card_explosion"] = (merged["total_yellows"] >= 6).astype(int)

            base_rate = merged["card_explosion"].mean()

            # Combined factors
            combined_condition = (merged["derby"] == 1) & (merged["big_stadium"] == 1)
            combined_matches = merged[combined_condition]

            if len(combined_matches) > 20:
                combined_rate = combined_matches["card_explosion"].mean()
                log.info(f"Base rate of 6+ cards: {base_rate:.1%}")
                log.info(f"Derby + Big stadium: {combined_rate:.1%} ({len(combined_matches)} matches)")
                log.info(f"Lift: +{combined_rate - base_rate:.1%}")

                results["card_explosion"] = {
                    "base_rate": base_rate,
                    "combined_rate": combined_rate,
                    "matches": len(combined_matches),
                    "lift": combined_rate - base_rate,
                }

    # 2. Rain + Away Underdog = Upset Potential?
    log.info("\n2. RAIN + UNDERDOG UPSET POTENTIAL")
    log.info("-" * 50)

    if "weather_rain_sum" in merged.columns and "elo_diff" in merged.columns:
        # Rainy conditions + big underdog away
        merged["rainy"] = (merged["weather_rain_sum"] > 5).astype(int) if "weather_rain_sum" in merged.columns else 0
        merged["away_underdog"] = (merged["elo_diff"] > 150).astype(int)  # Home team significantly stronger
        merged["away_upset"] = (merged["away_score"] > merged["home_score"]).astype(int)

        # Base upset rate
        underdog_matches = merged[merged["away_underdog"] == 1]
        base_upset_rate = underdog_matches["away_upset"].mean() if len(underdog_matches) > 0 else 0

        # Rainy underdog
        rainy_underdog = merged[(merged["rainy"] == 1) & (merged["away_underdog"] == 1)]

        if len(rainy_underdog) > 20:
            rainy_upset_rate = rainy_underdog["away_upset"].mean()
            log.info(f"Underdog away upset rate (dry): {base_upset_rate:.1%}")
            log.info(f"Underdog away upset rate (rain): {rainy_upset_rate:.1%} ({len(rainy_underdog)} matches)")

            results["rainy_upset"] = {
                "dry_upset_rate": base_upset_rate,
                "rainy_upset_rate": rainy_upset_rate,
                "matches": len(rainy_underdog),
            }

    # 3. Hot Team + Cold Team + Home = Dominant Win?
    log.info("\n3. MOMENTUM DOMINANCE")
    log.info("-" * 50)

    if "home_form_points_3" in merged.columns:
        # Hot home team vs cold away team
        merged["hot_home"] = (merged["home_form_points_3"] >= 7).astype(int)
        merged["cold_away"] = (merged["away_form_points_3"] <= 2).astype(int)
        merged["home_dominant_win"] = (merged["home_score"] >= merged["away_score"] + 2).astype(int)

        momentum_matches = merged[(merged["hot_home"] == 1) & (merged["cold_away"] == 1)]

        if len(momentum_matches) > 50:
            dominant_win_rate = momentum_matches["home_dominant_win"].mean()
            base_dominant_rate = merged["home_dominant_win"].mean()

            log.info(f"Base dominant home win rate (2+ goals): {base_dominant_rate:.1%}")
            log.info(f"Hot home vs cold away dominant rate: {dominant_win_rate:.1%} ({len(momentum_matches)} matches)")

            results["momentum_dominance"] = {
                "base_rate": base_dominant_rate,
                "momentum_rate": dominant_win_rate,
                "matches": len(momentum_matches),
                "lift": dominant_win_rate - base_dominant_rate,
            }

    return results


def main():
    log.info("=" * 70)
    log.info("ADVANCED PREDICTIONS - MINING ALL DATA")
    log.info("=" * 70)

    # Load all data
    data = load_all_data()

    all_results = {}

    # 1. Referee Predictions
    ref_results = analyze_referee_predictions(data["features"], data["refs"])
    all_results["referee"] = ref_results

    # 2. Weather Predictions
    weather_results = analyze_weather_predictions(data["features"], data["weather"])
    all_results["weather"] = weather_results

    # 3. Timing Predictions
    timing_results = analyze_timing_predictions(data["features"], data["shots"])
    all_results["timing"] = timing_results

    # 4. Formation Predictions
    formation_results = analyze_formation_predictions(data["features"])
    all_results["formation"] = formation_results

    # 5. Big Game Predictions
    big_game_results = analyze_big_game_predictions(data["features"])
    all_results["big_game"] = big_game_results

    # 6. Combined Predictions
    combined_results = analyze_combined_predictions(data["features"], data["refs"], data["weather"])
    all_results["combined"] = combined_results

    # Summary
    log.info("\n" + "=" * 70)
    log.info("SUMMARY OF NEW PREDICTIONS")
    log.info("=" * 70)

    categories = [
        ("Referee Predictions", ref_results),
        ("Weather Predictions", weather_results),
        ("Timing Predictions", timing_results),
        ("Formation Predictions", formation_results),
        ("Big Game Predictions", big_game_results),
        ("Combined Predictions", combined_results),
    ]

    for name, results in categories:
        n_predictions = len(results)
        log.info(f"\n{name}: {n_predictions} prediction categories")
        for key in list(results.keys())[:5]:
            log.info(f"  - {key}")

    # Save results
    output_path = DATA_DIR / "models" / "advanced_predictions.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    log.info(f"\n\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
