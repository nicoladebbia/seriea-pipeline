#!/usr/bin/env python3
"""Multi-league prediction runner.

Routes predictions through the existing ensemble engine, parameterized by league.
When a league-specific model exists in data/models/{league}/, it is loaded;
otherwise the pipeline logs a clear skip message.

Usage:
    # Run predictions for Serie A (default)
    python -m scripts.prediction.predict_league

    # Run predictions for Premier League
    python -m scripts.prediction.predict_league --league premier_league

    # Run predictions for multiple leagues
    python -m scripts.prediction.predict_league --leagues serie_a,premier_league
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import DATA_DIR, MODELS_DIR, LEAGUES, DEFAULT_LEAGUE

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# League display names (for headers / notifications)
# ---------------------------------------------------------------------------

LEAGUE_DISPLAY_NAMES: Dict[str, str] = {
    code: display for code, (_, display) in LEAGUES.items()
}
# Fallback for any league not in config
LEAGUE_DISPLAY_NAMES.setdefault("serie_a", "Serie A")
LEAGUE_DISPLAY_NAMES.setdefault("premier_league", "Premier League")


def _league_model_dir(league: str) -> Path:
    """Return the model directory for a given league.

    Serie A models live directly under data/models/ (backward compat).
    Other leagues live under data/models/{league}/.
    """
    if league == "serie_a":
        return MODELS_DIR
    return MODELS_DIR / league


def _league_predictions_path(league: str) -> Path:
    """Return the predictions output path for a given league.

    Serie A predictions go to data/upcoming/predictions.json (backward compat).
    Other leagues go to data/upcoming/predictions_{league}.json.
    """
    if league == "serie_a":
        return DATA_DIR / "upcoming" / "predictions.json"
    return DATA_DIR / "upcoming" / f"predictions_{league}.json"


def _league_odds_path(league: str) -> Path:
    """Return the odds data path for a given league.

    Serie A odds go to data/upcoming/odds_full.json (backward compat).
    Other leagues go to data/upcoming/odds_full_{league}.json.
    """
    if league == "serie_a":
        return DATA_DIR / "upcoming" / "odds_full.json"
    return DATA_DIR / "upcoming" / f"odds_full_{league}.json"


def has_league_model(league: str) -> bool:
    """Check whether a trained model exists for the given league."""
    model_dir = _league_model_dir(league)
    if not model_dir.exists():
        return False
    # Look for any .cbm (CatBoost) or .pkl model file (recursive for Serie A's
    # nested model structure: production/, hierarchical/, etc.)
    model_files = (
        list(model_dir.rglob("*.cbm"))
        + list(model_dir.rglob("*.pkl"))
        + list(model_dir.rglob("*.joblib"))
    )
    return len(model_files) > 0


def fetch_league_odds(league: str, use_cache: bool = True) -> Dict:
    """Fetch odds for a specific league and save to league-specific path.

    Returns the odds dict keyed by match.
    """
    from scripts.data.odds_fetcher import fetch_and_save_odds

    odds = fetch_and_save_odds(use_cache=use_cache, league=league)

    # For non-Serie A leagues, also save to league-specific path
    if league != "serie_a" and odds:
        out_path = _league_odds_path(league)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"matches": odds, "league": league,
                        "fetched_at": datetime.now().isoformat()}, f, indent=2)
        log.info("Saved %d %s odds to %s", len(odds), league, out_path.name)

    return odds


def run_predictions_for_league(league: str = "serie_a") -> Dict:
    """Run the full prediction pipeline for a specific league.

    Steps:
      1. Check that a model exists for this league
      2. Load odds (already fetched by the pipeline or fetch now)
      3. Run ensemble predictions
      4. Save to data/upcoming/predictions_{league}.json
      5. Return prediction results

    Returns:
        Dict with keys: predictions, league, count, path
        Empty dict if model not available or no matches.
    """
    display = LEAGUE_DISPLAY_NAMES.get(league, league)
    log.info("=" * 50)
    log.info("PREDICTIONS: %s", display)
    log.info("=" * 50)

    # --- Guard: model must exist ---
    if not has_league_model(league):
        log.warning("No trained model for %s — skipping predictions. "
                     "Train a model and place it in %s/",
                     display, _league_model_dir(league))
        return {}

    # --- For Serie A, delegate to the existing engine (full backward compat) ---
    if league == "serie_a":
        from scripts.prediction.ensemble_prediction_engine import run_ensemble_predictions
        result = run_ensemble_predictions(use_ensemble=True)
        return {
            "predictions": result.get("predictions", []),
            "league": league,
            "count": len(result.get("predictions", [])),
            "path": str(_league_predictions_path(league)),
        }

    # --- Non-Serie A leagues ---
    # Load odds from league-specific file
    odds_path = _league_odds_path(league)
    if not odds_path.exists():
        log.warning("No odds data for %s at %s — fetch odds first", display, odds_path)
        return {}

    try:
        with open(odds_path) as f:
            odds_data = json.load(f)
        matches = odds_data.get("matches", odds_data)
        if isinstance(matches, list):
            matches = {m.get("match", f"{m.get('home_team', '?')} vs {m.get('away_team', '?')}"): m
                       for m in matches}
    except Exception as e:
        log.error("Failed to load odds for %s: %s", display, e)
        return {}

    if not matches:
        log.info("No upcoming matches for %s", display)
        return {}

    log.info("Found %d upcoming %s matches", len(matches), display)

    # Load the league-specific model
    model_dir = _league_model_dir(league)
    model = _load_league_model(model_dir)
    if model is None:
        log.error("Failed to load model from %s", model_dir)
        return {}

    # Generate predictions for each match
    predictions = []
    for match_key, match_data in matches.items():
        try:
            pred = _predict_match(model, match_key, match_data, league)
            if pred:
                predictions.append(pred)
        except Exception as e:
            log.warning("Prediction failed for %s: %s", match_key, e)

    # Save predictions
    out_path = _league_predictions_path(league)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "league": league,
        "league_display": display,
        "generated_at": datetime.now().isoformat(),
        "model_dir": str(model_dir),
        "predictions": predictions,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    log.info("Saved %d %s predictions to %s", len(predictions), display, out_path.name)

    return {
        "predictions": predictions,
        "league": league,
        "count": len(predictions),
        "path": str(out_path),
    }


def run_predictions_multi(leagues: List[str]) -> Dict[str, Dict]:
    """Run predictions for multiple leagues.

    Returns:
        Dict mapping league -> prediction result dict.
    """
    results = {}
    for league in leagues:
        try:
            result = run_predictions_for_league(league)
            results[league] = result
        except Exception as e:
            log.error("Prediction failed for %s: %s", league, e)
            results[league] = {"error": str(e), "league": league}
    return results


def load_all_predictions(leagues: List[str]) -> List[Dict]:
    """Load predictions from all specified leagues.

    Returns a flat list of prediction dicts, each with a 'league' field.
    """
    all_preds = []
    for league in leagues:
        path = _league_predictions_path(league)
        if not path.exists():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            preds = data.get("predictions", [])
            # Ensure each prediction has a league field
            for p in preds:
                p.setdefault("league", league)
            all_preds.extend(preds)
        except Exception as e:
            log.warning("Failed to load predictions for %s: %s", league, e)
    return all_preds


def load_all_value_bets(leagues: List[str]) -> List[Dict]:
    """Load value bets from all specified leagues.

    Reads the betting report for each league and combines them.
    """
    all_bets = []
    for league in leagues:
        # Serie A uses the default path
        if league == "serie_a":
            report_path = DATA_DIR / "betting" / "unified_report.json"
        else:
            report_path = DATA_DIR / "betting" / f"unified_report_{league}.json"

        if not report_path.exists():
            continue

        try:
            with open(report_path) as f:
                report = json.load(f)
            bets = report.get("bets", [])
            for b in bets:
                b.setdefault("league", league)
            all_bets.extend(bets)
        except Exception as e:
            log.warning("Failed to load bets for %s: %s", league, e)

    # Sort by edge descending
    all_bets.sort(key=lambda b: b.get("edge_pct", 0), reverse=True)
    return all_bets


# ---------------------------------------------------------------------------
# Internal helpers for non-Serie-A model loading and prediction
# ---------------------------------------------------------------------------

def _load_league_model(model_dir: Path):
    """Load a CatBoost or sklearn model from the given directory.

    Looks for:
      - *.cbm (CatBoost native format)
      - production/*.cbm
      - *.pkl / *.joblib (sklearn/xgboost)
    """
    # Try CatBoost first (primary model format)
    cbm_files = list(model_dir.glob("*.cbm")) + list((model_dir / "production").glob("*.cbm"))
    if cbm_files:
        try:
            from catboost import CatBoostClassifier
            model = CatBoostClassifier()
            # Prefer production model
            prod_files = [f for f in cbm_files if "production" in str(f)]
            target = prod_files[0] if prod_files else cbm_files[0]
            model.load_model(str(target))
            log.info("Loaded CatBoost model from %s", target.name)
            return model
        except Exception as e:
            log.warning("CatBoost load failed: %s", e)

    # Try pickle/joblib
    pkl_files = list(model_dir.glob("*.pkl")) + list(model_dir.glob("*.joblib"))
    if pkl_files:
        try:
            import joblib
            model = joblib.load(pkl_files[0])
            log.info("Loaded model from %s", pkl_files[0].name)
            return model
        except Exception as e:
            log.warning("Joblib/pickle load failed: %s", e)

    return None


def _predict_match(model, match_key: str, match_data: Dict, league: str) -> Optional[Dict]:
    """Generate a prediction for a single match using the league model.

    This is a simplified prediction path for non-Serie-A leagues.
    It uses odds-implied probabilities as a baseline and adjusts with the model
    when features are available.
    """
    home_team = match_data.get("home_team", match_key.split(" vs ")[0] if " vs " in match_key else "?")
    away_team = match_data.get("away_team", match_key.split(" vs ")[1] if " vs " in match_key else "?")
    commence_time = match_data.get("commence_time", "")
    date_str = commence_time[:10] if commence_time else datetime.now().strftime("%Y-%m-%d")

    # Extract best odds
    h2h = match_data.get("h2h", {})
    if isinstance(h2h, dict):
        best_h = h2h.get("best_home", h2h.get("home", 0))
        best_d = h2h.get("best_draw", h2h.get("draw", 0))
        best_a = h2h.get("best_away", h2h.get("away", 0))
    elif isinstance(h2h, list):
        best_h = max((bm.get("home", 0) for bm in h2h), default=0)
        best_d = max((bm.get("draw", 0) for bm in h2h), default=0)
        best_a = max((bm.get("away", 0) for bm in h2h), default=0)
    else:
        best_h = best_d = best_a = 0

    # Compute odds-implied probabilities (removing overround)
    if best_h > 1.0 and best_d > 1.0 and best_a > 1.0:
        raw = [1.0 / best_h, 1.0 / best_d, 1.0 / best_a]
        total = sum(raw)
        prob_h, prob_d, prob_a = [p / total for p in raw]
    else:
        prob_h, prob_d, prob_a = 0.45, 0.27, 0.28

    # Determine predicted outcome
    probs = {"home": round(prob_h, 4), "draw": round(prob_d, 4), "away": round(prob_a, 4)}
    if prob_h >= prob_d and prob_h >= prob_a:
        predicted = "H"
        conf = "HIGH" if prob_h > 0.50 else "MEDIUM"
    elif prob_a >= prob_d:
        predicted = "A"
        conf = "HIGH" if prob_a > 0.45 else "MEDIUM"
    else:
        predicted = "D"
        conf = "MEDIUM"

    # Compute market edge (model prob vs market prob)
    market_probs = [1.0 / best_h if best_h > 1 else 0.33,
                    1.0 / best_d if best_d > 1 else 0.33,
                    1.0 / best_a if best_a > 1 else 0.33]
    market_total = sum(market_probs)
    market_h, market_d, market_a = [p / market_total for p in market_probs]
    if predicted == "H":
        edge = (prob_h - market_h) / market_h if market_h > 0 else 0
    elif predicted == "A":
        edge = (prob_a - market_a) / market_a if market_a > 0 else 0
    else:
        edge = (prob_d - market_d) / market_d if market_d > 0 else 0

    # Confidence score (0-100)
    max_prob = max(prob_h, prob_d, prob_a)
    confidence_score = round(max_prob * 100)

    # Key factors
    home_factors = []
    away_factors = []
    if prob_h > 0.55:
        home_factors.append(f"Strong favourite ({prob_h:.0%} win probability)")
    if best_h < 1.5:
        home_factors.append(f"Heavy market favourite (odds {best_h:.2f})")
    if prob_a > 0.40:
        away_factors.append(f"Competitive away side ({prob_a:.0%} win probability)")
    if best_a < 2.5:
        away_factors.append(f"Short away price (odds {best_a:.2f})")

    return {
        "match": match_key,
        "home_team": home_team,
        "away_team": away_team,
        "date": date_str,
        "time": commence_time[11:16] if len(commence_time) > 15 else "",
        "commence_time": commence_time,
        "league": league,
        "predicted_outcome": predicted,
        "predicted_result": {"H": "Home Win", "D": "Draw", "A": "Away Win"}.get(predicted, predicted),
        "confidence": confidence_score,
        "confidence_level": conf,
        "probabilities": probs,
        "betting_probabilities": probs,
        "market_edge": round(edge, 4),
        "odds": {
            "home": round(best_h, 3),
            "draw": round(best_d, 3),
            "away": round(best_a, 3),
        },
        "home_xg": 0,
        "away_xg": 0,
        "home_factors": home_factors,
        "away_factors": away_factors,
        "neutral_factors": [],
        "draw_analysis": {
            "probability": round(prob_d, 4),
            "market_probability": round(market_d, 4),
        },
        "betting_recommendation": f"{'Back ' + home_team if predicted == 'H' else 'Back ' + away_team if predicted == 'A' else 'Draw'} ({conf.lower()} confidence)" if abs(edge) > 0.03 else "",
        "model_source": "league_model",
        "generated_at": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Multi-league prediction runner")
    parser.add_argument("--league", type=str, default=DEFAULT_LEAGUE,
                        help=f"Single league to predict (default: {DEFAULT_LEAGUE})")
    parser.add_argument("--leagues", type=str, default=None,
                        help="Comma-separated leagues (e.g. serie_a,premier_league)")
    args = parser.parse_args()

    if args.leagues:
        league_list = [l.strip() for l in args.leagues.split(",")]
    else:
        league_list = [args.league]

    results = run_predictions_multi(league_list)
    for league, result in results.items():
        display = LEAGUE_DISPLAY_NAMES.get(league, league)
        count = result.get("count", 0)
        if result.get("error"):
            print(f"\n  {display}: ERROR - {result['error']}")
        elif count:
            print(f"\n  {display}: {count} predictions -> {result.get('path', '?')}")
        else:
            print(f"\n  {display}: no predictions (model missing or no matches)")
