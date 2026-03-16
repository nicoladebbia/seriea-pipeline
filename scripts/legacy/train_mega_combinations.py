#!/usr/bin/env python3
"""MEGA COMBINATION PREDICTIONS - Advanced Multi-Factor Analysis

This script combines multiple factors for compound accuracy:
1. International Break Impact
2. Rest Days Effect
3. Advanced Multi-Factor Combinations
4. Extreme Condition Predictions
5. Triple/Quadruple Factor Stacking

Professional-level analysis combining referee + weather + formation + momentum!
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_and_merge_data():
    """Load and merge all available data."""
    # Main features
    df = pd.read_parquet(DATA_DIR / "features" / "features.parquet")
    df = df.sort_values(["season", "matchweek", "match_date"]).reset_index(drop=True)

    # Referee data
    try:
        refs = pd.read_parquet(DATA_DIR / "external" / "referee" / "referee_assignments.parquet")
        refs["match_key"] = refs["home_team"] + "_" + refs["away_team"] + "_" + refs["season"].astype(str)
        df["match_key"] = df["home_team"] + "_" + df["away_team"] + "_" + df["season"].astype(str)
        df = df.merge(refs[["match_key", "referee", "ref_yellows", "ref_reds"]],
                      on="match_key", how="left", suffixes=("", "_ref"))
        log.info(f"Merged referee data: {df['referee'].notna().sum()} matches")
    except Exception as e:
        log.warning(f"No referee data: {e}")

    # Weather data
    try:
        weather = pd.read_parquet(DATA_DIR / "external" / "weather.parquet")
        df = df.merge(weather, on="match_id", how="left")
        log.info(f"Merged weather data: {df['weather_rain_sum'].notna().sum()} matches")
    except Exception as e:
        log.warning(f"No weather data: {e}")

    return df


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all derived features for analysis."""
    df = df.copy()

    # Basic outcomes
    df["total_goals"] = df["home_score"] + df["away_score"]
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["away_win"] = (df["away_score"] > df["home_score"]).astype(int)
    df["draw"] = (df["home_score"] == df["away_score"]).astype(int)
    df["home_dominant"] = (df["home_score"] >= df["away_score"] + 2).astype(int)

    # Cards
    if "home_yellow_cards" in df.columns:
        df["total_yellows"] = df["home_yellow_cards"].fillna(0) + df["away_yellow_cards"].fillna(0)
        df["high_cards"] = (df["total_yellows"] >= 5).astype(int)
        df["card_explosion"] = (df["total_yellows"] >= 6).astype(int)

    # Rest days (international break detection)
    df["days_since_last"] = df.groupby("home_team")["match_date"].diff().dt.days
    df["international_break"] = (df["days_since_last"] > 14).astype(int)
    df["short_rest"] = (df["days_since_last"] <= 4).astype(int)
    df["long_rest"] = (df["days_since_last"] >= 10).astype(int)

    # Referee strictness (based on avg yellows)
    if "referee" in df.columns and df["referee"].notna().any():
        ref_avg = df.groupby("referee")["ref_yellows"].mean()
        overall_avg = df["ref_yellows"].mean()
        strict_refs = ref_avg[ref_avg > overall_avg + 0.5].index.tolist()
        df["strict_ref"] = df["referee"].isin(strict_refs).astype(int)

        # Home-favoring refs
        ref_home = df.groupby("referee")["home_win"].mean()
        home_favoring = ref_home[ref_home > 0.50].index.tolist()
        df["home_favoring_ref"] = df["referee"].isin(home_favoring).astype(int)

    # Weather conditions
    if "weather_rain_sum" in df.columns:
        df["rainy"] = (df["weather_rain_sum"] > 5).astype(int)
        df["cold"] = (df["weather_temperature_2m_mean"] < 10).astype(int)
        df["windy"] = (df["weather_wind_speed_10m_max"] > 30).astype(int)

    # Formation style
    if "home_formation" in df.columns and df["home_formation"].notna().any():
        def formation_style(f):
            if pd.isna(f):
                return "unknown"
            try:
                defenders = int(str(f).split("-")[0])
                return "attacking" if defenders == 3 else "balanced" if defenders == 4 else "defensive"
            except:
                return "unknown"
        df["home_style"] = df["home_formation"].apply(formation_style)
        df["away_style"] = df["away_formation"].apply(formation_style)

    # Momentum/Form
    if "home_form_points_3" in df.columns:
        df["hot_home"] = (df["home_form_points_3"] >= 7).astype(int)
        df["cold_home"] = (df["home_form_points_3"] <= 2).astype(int)
        df["hot_away"] = (df["away_form_points_3"] >= 7).astype(int)
        df["cold_away"] = (df["away_form_points_3"] <= 2).astype(int)

    # Derby/Rivalry
    if "matchup_competitiveness" in df.columns:
        df["derby"] = (df["matchup_competitiveness"] > 0.8).astype(int)

    # Stadium size
    if "home_stadium_capacity" in df.columns:
        df["big_stadium"] = (df["home_stadium_capacity"] > 50000).astype(int)

    # Elo mismatch
    if "elo_diff" in df.columns:
        df["home_favorite"] = (df["elo_diff"] > 100).astype(int)
        df["away_favorite"] = (df["elo_diff"] < -100).astype(int)
        df["even_match"] = ((df["elo_diff"] >= -100) & (df["elo_diff"] <= 100)).astype(int)

    return df


def analyze_international_break(df: pd.DataFrame) -> Dict:
    """Analyze international break impact."""
    results = {}

    log.info("\n" + "=" * 70)
    log.info("INTERNATIONAL BREAK IMPACT")
    log.info("=" * 70)

    break_matches = df[df["international_break"] == 1]
    normal_matches = df[df["international_break"] == 0]

    if len(break_matches) > 100:
        break_goals = break_matches["total_goals"].mean()
        normal_goals = normal_matches["total_goals"].mean()
        break_home_win = break_matches["home_win"].mean()
        normal_home_win = normal_matches["home_win"].mean()
        break_draws = break_matches["draw"].mean()
        normal_draws = normal_matches["draw"].mean()

        log.info(f"\nPost-international break matches: {len(break_matches)}")
        log.info(f"  Avg goals: {break_goals:.2f} vs {normal_goals:.2f} (normal)")
        log.info(f"  Home win rate: {break_home_win:.1%} vs {normal_home_win:.1%}")
        log.info(f"  Draw rate: {break_draws:.1%} vs {normal_draws:.1%}")

        # Is there a "rust" effect?
        if "home_favorite" in df.columns:
            break_fav = break_matches[break_matches["home_favorite"] == 1]
            normal_fav = normal_matches[normal_matches["home_favorite"] == 1]

            if len(break_fav) > 30:
                break_fav_win = break_fav["home_win"].mean()
                normal_fav_win = normal_fav["home_win"].mean()
                log.info(f"\nFavorites after break: {break_fav_win:.1%} vs {normal_fav_win:.1%}")
                log.info(f"  Favorites' rust factor: {normal_fav_win - break_fav_win:+.1%}")

        results["international_break"] = {
            "matches": len(break_matches),
            "avg_goals": break_goals,
            "normal_goals": normal_goals,
            "home_win_rate": break_home_win,
            "normal_home_win": normal_home_win,
            "draw_rate": break_draws,
            "goals_diff": break_goals - normal_goals,
        }

    return results


def analyze_rest_days(df: pd.DataFrame) -> Dict:
    """Analyze rest days impact."""
    results = {}

    log.info("\n" + "=" * 70)
    log.info("REST DAYS IMPACT")
    log.info("=" * 70)

    short_rest = df[df["short_rest"] == 1]
    long_rest = df[df["long_rest"] == 1]
    normal_rest = df[(df["short_rest"] == 0) & (df["long_rest"] == 0) & (df["international_break"] == 0)]

    if len(short_rest) > 100 and len(long_rest) > 100:
        log.info(f"\nShort rest (<=4 days): {len(short_rest)} matches")
        log.info(f"Long rest (>=10 days): {len(long_rest)} matches")
        log.info(f"Normal rest: {len(normal_rest)} matches")

        short_goals = short_rest["total_goals"].mean()
        long_goals = long_rest["total_goals"].mean()
        normal_goals = normal_rest["total_goals"].mean()

        short_home_win = short_rest["home_win"].mean()
        long_home_win = long_rest["home_win"].mean()
        normal_home_win = normal_rest["home_win"].mean()

        log.info(f"\nGoals: Short={short_goals:.2f}, Long={long_goals:.2f}, Normal={normal_goals:.2f}")
        log.info(f"Home win: Short={short_home_win:.1%}, Long={long_home_win:.1%}, Normal={normal_home_win:.1%}")

        results["rest_days"] = {
            "short_rest_matches": len(short_rest),
            "long_rest_matches": len(long_rest),
            "short_rest_goals": short_goals,
            "long_rest_goals": long_goals,
            "normal_goals": normal_goals,
            "short_rest_home_win": short_home_win,
            "long_rest_home_win": long_home_win,
        }

    return results


def analyze_triple_combinations(df: pd.DataFrame) -> Dict:
    """Analyze triple factor combinations for compound effects."""
    results = {}

    log.info("\n" + "=" * 70)
    log.info("TRIPLE FACTOR COMBINATIONS")
    log.info("=" * 70)

    # 1. Strict Ref + Derby + Rain = Card Explosion?
    log.info("\n1. STRICT REF + DERBY + RAIN")
    log.info("-" * 50)

    if all(c in df.columns for c in ["strict_ref", "derby", "rainy", "card_explosion"]):
        base_rate = df["card_explosion"].mean()

        # Triple combo
        triple = df[(df["strict_ref"] == 1) & (df["derby"] == 1) & (df["rainy"] == 1)]
        double_strict_derby = df[(df["strict_ref"] == 1) & (df["derby"] == 1)]
        strict_only = df[df["strict_ref"] == 1]

        log.info(f"Base card explosion (6+) rate: {base_rate:.1%}")

        if len(strict_only) > 50:
            log.info(f"Strict ref only: {strict_only['card_explosion'].mean():.1%} ({len(strict_only)} matches)")

        if len(double_strict_derby) > 20:
            log.info(f"Strict + Derby: {double_strict_derby['card_explosion'].mean():.1%} ({len(double_strict_derby)} matches)")

        if len(triple) > 5:
            triple_rate = triple["card_explosion"].mean()
            log.info(f"Strict + Derby + Rain: {triple_rate:.1%} ({len(triple)} matches)")
            log.info(f"  COMPOUND LIFT: +{triple_rate - base_rate:.1%}")

            results["strict_derby_rain_cards"] = {
                "base_rate": base_rate,
                "triple_rate": triple_rate,
                "matches": len(triple),
                "lift": triple_rate - base_rate,
            }

    # 2. Hot Home + Big Stadium + Home-Favoring Ref = Dominant Win?
    log.info("\n2. HOT HOME + BIG STADIUM + FAVORABLE REF")
    log.info("-" * 50)

    if all(c in df.columns for c in ["hot_home", "big_stadium", "home_favoring_ref", "home_dominant"]):
        base_dominant = df["home_dominant"].mean()

        triple = df[(df["hot_home"] == 1) & (df["big_stadium"] == 1) & (df["home_favoring_ref"] == 1)]
        double_hot_stadium = df[(df["hot_home"] == 1) & (df["big_stadium"] == 1)]

        log.info(f"Base dominant home win rate: {base_dominant:.1%}")

        if len(double_hot_stadium) > 30:
            log.info(f"Hot + Big stadium: {double_hot_stadium['home_dominant'].mean():.1%} ({len(double_hot_stadium)} matches)")

        if len(triple) > 10:
            triple_rate = triple["home_dominant"].mean()
            log.info(f"Hot + Big stadium + Fav ref: {triple_rate:.1%} ({len(triple)} matches)")
            log.info(f"  COMPOUND LIFT: +{triple_rate - base_dominant:.1%}")

            results["hot_stadium_ref_dominant"] = {
                "base_rate": base_dominant,
                "triple_rate": triple_rate,
                "matches": len(triple),
                "lift": triple_rate - base_dominant,
            }

    # 3. Cold Away + Long Travel + Rainy = Away Collapse?
    log.info("\n3. COLD AWAY + RAIN + AWAY FAVORITE = UPSET?")
    log.info("-" * 50)

    if all(c in df.columns for c in ["cold_away", "rainy", "away_favorite"]):
        # Away favorite that loses
        away_fav_matches = df[df["away_favorite"] == 1]
        away_fav_loss_rate = 1 - away_fav_matches["away_win"].mean() if len(away_fav_matches) > 0 else 0

        triple = df[(df["cold_away"] == 1) & (df["rainy"] == 1) & (df["away_favorite"] == 1)]

        log.info(f"Away favorite loss rate (overall): {away_fav_loss_rate:.1%}")

        if len(triple) > 5:
            triple_loss = 1 - triple["away_win"].mean()
            log.info(f"Cold away + Rain + Away fav: {triple_loss:.1%} loss rate ({len(triple)} matches)")

            results["cold_rain_fav_upset"] = {
                "base_loss_rate": away_fav_loss_rate,
                "triple_loss_rate": triple_loss,
                "matches": len(triple),
            }

    # 4. Even Match + Derby + Short Rest = Draw?
    log.info("\n4. EVEN MATCH + DERBY + SHORT REST = DRAW?")
    log.info("-" * 50)

    if all(c in df.columns for c in ["even_match", "derby", "short_rest", "draw"]):
        base_draw = df["draw"].mean()

        triple = df[(df["even_match"] == 1) & (df["derby"] == 1) & (df["short_rest"] == 1)]
        double_even_derby = df[(df["even_match"] == 1) & (df["derby"] == 1)]

        log.info(f"Base draw rate: {base_draw:.1%}")

        if len(double_even_derby) > 30:
            log.info(f"Even + Derby: {double_even_derby['draw'].mean():.1%} ({len(double_even_derby)} matches)")

        if len(triple) > 10:
            triple_rate = triple["draw"].mean()
            log.info(f"Even + Derby + Short rest: {triple_rate:.1%} ({len(triple)} matches)")
            log.info(f"  COMPOUND LIFT: +{triple_rate - base_draw:.1%}")

            results["even_derby_short_draw"] = {
                "base_rate": base_draw,
                "triple_rate": triple_rate,
                "matches": len(triple),
                "lift": triple_rate - base_draw,
            }

    return results


def analyze_quadruple_combinations(df: pd.DataFrame) -> Dict:
    """Analyze quadruple factor combinations - the ultimate compound effects."""
    results = {}

    log.info("\n" + "=" * 70)
    log.info("QUADRUPLE FACTOR COMBINATIONS (ULTIMATE PREDICTIONS)")
    log.info("=" * 70)

    # 1. Hot Home + Cold Away + Big Stadium + Home Fav Ref
    log.info("\n1. ULTIMATE HOME ADVANTAGE")
    log.info("-" * 50)

    if all(c in df.columns for c in ["hot_home", "cold_away", "big_stadium", "home_favoring_ref"]):
        base_home_win = df["home_win"].mean()

        quad = df[(df["hot_home"] == 1) & (df["cold_away"] == 1) &
                  (df["big_stadium"] == 1) & (df["home_favoring_ref"] == 1)]

        if len(quad) > 5:
            quad_rate = quad["home_win"].mean()
            quad_dominant = quad["home_dominant"].mean()
            log.info(f"Hot home + Cold away + Big stadium + Fav ref:")
            log.info(f"  {len(quad)} matches")
            log.info(f"  Home win rate: {quad_rate:.1%} (base: {base_home_win:.1%})")
            log.info(f"  Dominant win rate: {quad_dominant:.1%}")
            log.info(f"  ULTIMATE LIFT: +{quad_rate - base_home_win:.1%}")

            results["ultimate_home_advantage"] = {
                "matches": len(quad),
                "home_win_rate": quad_rate,
                "dominant_rate": quad_dominant,
                "base_rate": base_home_win,
                "lift": quad_rate - base_home_win,
            }

    # 2. Strict Ref + Derby + Cold Weather + Big Stadium = Maximum Cards
    log.info("\n2. MAXIMUM CARDS CONDITIONS")
    log.info("-" * 50)

    if all(c in df.columns for c in ["strict_ref", "derby", "cold", "big_stadium", "total_yellows"]):
        base_cards = df["total_yellows"].mean()

        quad = df[(df["strict_ref"] == 1) & (df["derby"] == 1) &
                  (df["cold"] == 1) & (df["big_stadium"] == 1)]

        if len(quad) > 3:
            quad_cards = quad["total_yellows"].mean()
            log.info(f"Strict ref + Derby + Cold + Big stadium:")
            log.info(f"  {len(quad)} matches")
            log.info(f"  Avg cards: {quad_cards:.1f} (base: {base_cards:.1f})")
            log.info(f"  Extra cards: +{quad_cards - base_cards:.1f}")

            results["maximum_cards"] = {
                "matches": len(quad),
                "avg_cards": quad_cards,
                "base_cards": base_cards,
                "extra_cards": quad_cards - base_cards,
            }

    return results


def analyze_formation_combos(df: pd.DataFrame) -> Dict:
    """Analyze formation + other factor combinations."""
    results = {}

    log.info("\n" + "=" * 70)
    log.info("FORMATION COMBINATION PREDICTIONS")
    log.info("=" * 70)

    if "home_style" not in df.columns or df["home_style"].isna().all():
        log.info("No formation data available")
        return results

    formation_df = df[df["home_style"] != "unknown"]

    # 1. 4-back at home vs 3-back away
    log.info("\n1. BALANCED (4-BACK) vs ATTACKING (3-BACK)")
    log.info("-" * 50)

    matchup = formation_df[(formation_df["home_style"] == "balanced") & (formation_df["away_style"] == "attacking")]

    if len(matchup) > 30:
        matchup_win = matchup["home_win"].mean()
        matchup_goals = matchup["total_goals"].mean()
        base_win = formation_df["home_win"].mean()

        log.info(f"4-back home vs 3-back away: {len(matchup)} matches")
        log.info(f"  Home win: {matchup_win:.1%} (base: {base_win:.1%})")
        log.info(f"  Avg goals: {matchup_goals:.2f}")

        # Add other factors
        if "hot_home" in df.columns:
            hot_matchup = matchup[matchup["hot_home"] == 1]
            if len(hot_matchup) > 10:
                hot_win = hot_matchup["home_win"].mean()
                log.info(f"  + Hot home form: {hot_win:.1%} ({len(hot_matchup)} matches)")

        results["balanced_vs_attacking"] = {
            "matches": len(matchup),
            "home_win": matchup_win,
            "base_win": base_win,
            "avg_goals": matchup_goals,
        }

    return results


def analyze_extreme_conditions(df: pd.DataFrame) -> Dict:
    """Find extreme condition predictions with highest confidence."""
    results = {}

    log.info("\n" + "=" * 70)
    log.info("EXTREME CONDITION ANALYSIS")
    log.info("=" * 70)

    # Find combinations with >80% accuracy
    extreme_finds = []

    # Test various combinations
    conditions = [
        ("Hot home + Cold away", lambda d: (d.get("hot_home", 0) == 1) & (d.get("cold_away", 0) == 1), "home_win"),
        ("Derby + Even match", lambda d: (d.get("derby", 0) == 1) & (d.get("even_match", 0) == 1), "draw"),
        ("Strict ref + Derby", lambda d: (d.get("strict_ref", 0) == 1) & (d.get("derby", 0) == 1), "high_cards"),
        ("Big stadium + Home fav", lambda d: (d.get("big_stadium", 0) == 1) & (d.get("home_favorite", 0) == 1), "home_win"),
        ("Rainy + Even match", lambda d: (d.get("rainy", 0) == 1) & (d.get("even_match", 0) == 1), "draw"),
    ]

    for name, condition_fn, target in conditions:
        if target not in df.columns:
            continue

        try:
            mask = condition_fn(df)
            subset = df[mask]

            if len(subset) > 20:
                rate = subset[target].mean()
                base = df[target].mean()
                lift = rate - base

                if rate >= 0.60 and lift >= 0.10:  # High accuracy + significant lift
                    extreme_finds.append({
                        "condition": name,
                        "target": target,
                        "rate": rate,
                        "base": base,
                        "lift": lift,
                        "matches": len(subset),
                    })
                    log.info(f"\n✓ {name} → {target}")
                    log.info(f"  Rate: {rate:.1%} (base: {base:.1%})")
                    log.info(f"  Lift: +{lift:.1%}")
                    log.info(f"  Matches: {len(subset)}")
        except Exception:
            continue

    results["extreme_conditions"] = extreme_finds
    return results


def main():
    log.info("=" * 70)
    log.info("MEGA COMBINATION PREDICTIONS")
    log.info("=" * 70)

    # Load and prepare data
    df = load_and_merge_data()
    log.info(f"\nLoaded {len(df)} matches")

    df = create_derived_features(df)
    log.info(f"Created derived features")

    all_results = {}

    # 1. International Break
    break_results = analyze_international_break(df)
    all_results.update(break_results)

    # 2. Rest Days
    rest_results = analyze_rest_days(df)
    all_results.update(rest_results)

    # 3. Triple Combinations
    triple_results = analyze_triple_combinations(df)
    all_results["triple_combos"] = triple_results

    # 4. Quadruple Combinations
    quad_results = analyze_quadruple_combinations(df)
    all_results["quad_combos"] = quad_results

    # 5. Formation Combos
    formation_results = analyze_formation_combos(df)
    all_results["formation_combos"] = formation_results

    # 6. Extreme Conditions
    extreme_results = analyze_extreme_conditions(df)
    all_results.update(extreme_results)

    # Summary
    log.info("\n" + "=" * 70)
    log.info("SUMMARY - HIGHEST VALUE COMBINATIONS")
    log.info("=" * 70)

    # Find all combos with >60% rate
    high_value = []

    for key, value in all_results.items():
        if isinstance(value, dict):
            if "triple_rate" in value and value.get("triple_rate", 0) > 0.60:
                high_value.append((key, value["triple_rate"], value.get("lift", 0), value.get("matches", 0)))
            elif "rate" in value and value.get("rate", 0) > 0.60:
                high_value.append((key, value["rate"], value.get("lift", 0), value.get("matches", 0)))

    if "extreme_conditions" in all_results:
        for item in all_results["extreme_conditions"]:
            high_value.append((item["condition"], item["rate"], item["lift"], item["matches"]))

    high_value.sort(key=lambda x: x[1], reverse=True)

    log.info("\nTOP COMPOUND PREDICTIONS (>60% accuracy):")
    for name, rate, lift, matches in high_value[:15]:
        log.info(f"  {name}: {rate:.1%} (+{lift:.1%} lift, {matches} matches)")

    # Save results
    output_path = DATA_DIR / "models" / "mega_combinations.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    log.info(f"\n\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
