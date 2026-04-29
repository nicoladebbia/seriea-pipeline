#!/usr/bin/env python3
"""ML MARKET PREDICTIONS — CatBoost predictions for all markets.

Uses the trained CatBoost models (prod_*.cbm) via predict_all_markets() to
generate ML-quality predictions for cards, corners, BTTS, and other markets.

Reuses the FeatureBuilder from ensemble_prediction_engine.py to build feature
rows for upcoming matches, then runs predict_all_markets() on each.

Saves output to:
  - data/upcoming/ml_predictions.json (comprehensive, all 26 market categories)
  - Updates cards_predictions.json, corners_predictions.json, btts_predictions.json
    with ML-quality predictions (replacing heuristic Poisson).

Usage:
    python scripts/ml_market_predictions.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def generate_ml_predictions() -> Dict:
    """Generate ML predictions for all upcoming matches.
    
    Returns dict keyed by match name with predict_all_markets() output.
    """
    from scripts.prediction.ensemble_prediction_engine import (
        EnsemblePredictor, _load_league_matches,
        calculate_all_forms, fetch_all_match_weather, analyze_referee_impact,
    )
    from scripts.models.comprehensive_markets import get_training_features, predict_all_markets
    from config.leagues import ACTIVE_LEAGUES, infer_league

    # Load training feature list
    df_hist = pd.read_parquet(DATA_DIR / "features" / "features.parquet")
    features = get_training_features(df_hist)
    log.info("ML market predictions: %d features required", len(features))

    # Initialize ensemble (just for FeatureBuilder + odds injection)
    ensemble = EnsemblePredictor(live_mode=False)
    ensemble.initialize()

    # Load upcoming matches across all active leagues
    matches: List[Dict] = []
    for _league in ACTIVE_LEAGUES:
        league_matches = _load_league_matches(_league)
        for m in league_matches:
            m.setdefault("league", _league)
        matches.extend(league_matches)
        log.info("Loaded %d upcoming matches for %s", len(league_matches), _league)
    if not matches:
        log.error("No upcoming matches found")
        return {}

    log.info("Found %d upcoming matches across %d leagues", len(matches), len(ACTIVE_LEAGUES))
    
    # Load form data for feature building
    form_data = calculate_all_forms()
    
    all_predictions = {}
    
    for match in matches:
        home = match["home_team"]
        away = match["away_team"]
        match_name = f"{home} vs {away}"
        date = match.get("date", "")
        league = match.get("league") or infer_league(home, away)
        
        try:
            # Build feature row (same as ensemble prediction engine)
            match_features = ensemble.feature_builder.build_match_features(
                home, away, form_data
            )
            if match_features is None:
                log.warning("  %s: could not build features, skipping", match_name)
                continue
            
            # Inject real odds
            match_features = ensemble._inject_odds_into_features(
                match_features, home, away
            )
            
            # Inject referee features if available
            referee_name = match.get("referee", "")
            if referee_name:
                try:
                    from scripts.prediction.referee_integration import get_referee_features_for_prediction
                    ref_feats = get_referee_features_for_prediction(referee_name, home, away)
                    for feat_name, feat_val in ref_feats.items():
                        match_features[feat_name] = feat_val
                except Exception as e:
                    log.debug(f"Failed to inject referee features for {home} vs {away}: {e}")

            # Fill missing features that FeatureBuilder doesn't produce
            # but predict_all_markets() requires (37 contextual/derived features)
            import datetime as _dt
            now = _dt.datetime.now()
            _defaults = {
                "us_coverage": 1, "kickoff_hour": 15, "is_night_match": 0,
                "is_weekend": 1 if now.weekday() >= 5 else 0,
                "any_team_fatigued": 0, "matchweek_avg_goals": 2.61,
                "promoted_vs_established": 0,
                "formation_matchup_home_rate": 0.40,
                "formation_matchup_draw_rate": 0.32,
                "form_diff": 0, "momentum_diff": 0,
                "is_derby": 0, "derby_intensity": 0,
                "derby_home_advantage_boost": 0,
                "league_home_win_rate": 0.46, "league_draw_rate": 0.27,
                "league_avg_goals": 2.62,
                "capacity_ratio": 0, "long_travel": 0, "altitude_advantage": 0,
                "steam_move_flag": 0, "pressing_mismatch": 0,
                "injury_x_elo": 0, "press_mismatch_x_elo": 0,
                "combined_disruption": 0,
                "is_early_kickoff": 0, "is_evening_kickoff": 0,
                "is_primetime": 0, "is_friday_night": 0, "is_monday_night": 0,
                "is_early_season": 1 if now.month in (8, 9) else 0,
                "is_late_season": 1 if now.month in (4, 5) else 0,
                "is_run_in": 1 if now.month in (3, 4, 5) else 0,
                "is_august": 1 if now.month == 8 else 0,
                "is_december": 1 if now.month == 12 else 0,
                "is_january": 1 if now.month == 1 else 0,
                "is_may": 1 if now.month == 5 else 0,
            }
            for col, default in _defaults.items():
                if col not in match_features.columns:
                    match_features[col] = default
            
            # Run predict_all_markets
            preds = predict_all_markets(match_features, features)
            preds["match"] = match_name
            preds["date"] = date
            preds["home_team"] = home
            preds["away_team"] = away
            preds["league"] = league
            
            all_predictions[match_name] = preds
            log.info("  ✅ %s: xG=%.2f-%.2f, BTTS=%.0f%%, Cards=%.1f, Corners=%.1f",
                     match_name, preds.get("home_xg", 0), preds.get("away_xg", 0),
                     preds.get("btts", {}).get("yes", {}).get("prob", 0) * 100,
                     preds.get("cards", {}).get("total_expected", 0),
                     preds.get("corners", {}).get("total_expected", 0))
            
        except Exception as e:
            log.warning("  ❌ %s: %s", match_name, e)
            continue
    
    log.info("Generated ML predictions for %d/%d matches", len(all_predictions), len(matches))
    return all_predictions


def _merge_predictions_by_match(existing_path, new_entries: list) -> list:
    """Merge new entries (list of dicts with 'match' key) into the existing
    file, preserving entries for matches not in the new batch. New entries
    overwrite existing ones for the same match name.
    """
    merged = {}
    if existing_path.exists():
        try:
            existing = json.load(open(existing_path))
            existing_list = existing.get("predictions", existing) if isinstance(existing, dict) else existing
            if isinstance(existing_list, list):
                for e in existing_list:
                    if isinstance(e, dict) and e.get("match"):
                        merged[e["match"]] = e
        except Exception:
            pass
    for e in new_entries:
        if isinstance(e, dict) and e.get("match"):
            merged[e["match"]] = e
    return list(merged.values())


def save_ml_predictions(all_preds: Dict):
    """Save ML predictions to JSON files. Merges with existing per-league outputs
    (which may have been written by the BTTS/corners model for the *other* league).
    """
    from config.leagues import infer_league
    out_dir = DATA_DIR / "upcoming"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _league_for(match_name: str) -> str:
        try:
            home, away = match_name.split(" vs ", 1)
        except ValueError:
            return "serie_a"
        return infer_league(home, away)

    # 1. Save comprehensive ML predictions (overwrite — this file is ML-only)
    ml_path = out_dir / "ml_predictions.json"
    with open(ml_path, "w") as f:
        json.dump(all_preds, f, indent=2, default=str)
    log.info("Saved comprehensive ML predictions to %s", ml_path)

    # 2. cards_predictions.json — merge per-match
    cards_preds = []
    for match_name, preds in all_preds.items():
        cards = preds.get("cards", {})
        if not cards:
            continue
        entry = {
            "match": match_name,
            "league": _league_for(match_name),
            "date": preds.get("date", ""),
            "expected_cards": cards.get("total_expected", 4.5),
            "expected_home_cards": cards.get("home_expected", 2.2),
            "expected_away_cards": cards.get("away_expected", 2.3),
            "source": "CatBoost ML",
        }
        for line in ["3.5", "4.5", "5.5"]:
            over_key = f"over_{line}"
            if over_key in cards:
                entry[f"over_{line.replace('.', '_')}"] = cards[over_key].get("prob", 0)
        cards_preds.append(entry)

    if cards_preds:
        cards_path = out_dir / "cards_predictions.json"
        merged = _merge_predictions_by_match(cards_path, cards_preds)
        with open(cards_path, "w") as f:
            json.dump({"predictions": merged}, f, indent=2)
        log.info("Updated cards_predictions.json: %d ML + %d preserved = %d total",
                 len(cards_preds), len(merged) - len(cards_preds), len(merged))

    # 3. corners_predictions.json — merge per-match
    corners_preds = []
    for match_name, preds in all_preds.items():
        corners = preds.get("corners", {})
        if not corners:
            continue
        entry = {
            "match": match_name,
            "league": _league_for(match_name),
            "date": preds.get("date", ""),
            "expected_corners": corners.get("expected_total", 10.0),
            "expected_home_corners": corners.get("expected_home", 5.0),
            "expected_away_corners": corners.get("expected_away", 5.0),
            "source": "CatBoost ML",
        }
        for line in ["8.5", "9.5", "10.5"]:
            over_key = f"over_{line}"
            if over_key in corners:
                entry[f"over_{line.replace('.', '_')}"] = corners[over_key].get("prob", 0)
        corners_preds.append(entry)

    if corners_preds:
        corners_path = out_dir / "corners_predictions.json"
        merged = _merge_predictions_by_match(corners_path, corners_preds)
        with open(corners_path, "w") as f:
            json.dump({"predictions": merged}, f, indent=2)
        log.info("Updated corners_predictions.json: %d ML + %d preserved = %d total",
                 len(corners_preds), len(merged) - len(corners_preds), len(merged))

    # 4. btts_predictions.json — merge per-match
    btts_preds = []
    for match_name, preds in all_preds.items():
        btts = preds.get("btts", {})
        if not btts:
            continue
        entry = {
            "match": match_name,
            "league": _league_for(match_name),
            "date": preds.get("date", ""),
            "btts_yes": btts.get("yes", {}).get("prob", 0.5),
            "btts_no": btts.get("no", {}).get("prob", 0.5),
            "expected_home_goals": preds.get("home_xg", 1.3),
            "expected_away_goals": preds.get("away_xg", 1.1),
            "source": "CatBoost ML",
        }
        btts_preds.append(entry)

    if btts_preds:
        btts_path = out_dir / "btts_predictions.json"
        merged = _merge_predictions_by_match(btts_path, btts_preds)
        with open(btts_path, "w") as f:
            json.dump({"predictions": merged}, f, indent=2)
        log.info("Updated btts_predictions.json: %d ML + %d preserved = %d total",
                 len(btts_preds), len(merged) - len(btts_preds), len(merged))


def run():
    """Main entry point."""
    log.info("=" * 70)
    log.info("ML MARKET PREDICTIONS — CatBoost for all markets")
    log.info("=" * 70)
    
    preds = generate_ml_predictions()
    if preds:
        save_ml_predictions(preds)
        
        # Summary
        log.info("\nSummary:")
        for match_name, p in preds.items():
            btts_yes = p.get("btts", {}).get("yes", {}).get("prob", 0)
            cards_exp = p.get("cards", {}).get("total_expected", 0)
            corners_exp = p.get("corners", {}).get("total_expected", 0)
            log.info("  %s: BTTS=%.0f%%, Cards=%.1f, Corners=%.1f",
                     match_name, btts_yes * 100, cards_exp, corners_exp)
    
    return preds


if __name__ == "__main__":
    run()
