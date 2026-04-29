#!/usr/bin/env python3
"""REFEREE INTEGRATION - Critical for 91.5% Accuracy Predictions

This integrates referee assignments with your validated referee bias data.
Referee bias is your BEST predictor:
- Home-favoring refs + Big home fav = 91.5% home win (47 matches)
- Strict ref + Derby = 63.6% high cards (407 matches)

Sources for referee assignments:
1. Manual input (most reliable)
2. Historical pattern analysis (which refs officiate which teams)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR

# Import computed referee features from the training pipeline
try:
    from features.referee import get_referee_profile, get_referee_team_history
    _HAS_REFEREE_MODULE = True
except ImportError:
    _HAS_REFEREE_MODULE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# =============================================================================
# VALIDATED REFEREE CLASSIFICATIONS (8 Seasons of Data)
# =============================================================================

# Home-favoring referees (home win rate significantly above 44.4% average)
HOME_FAVORING_REFS = {
    "Federico Dionisi": {"home_win_rate": 0.581, "lift": 0.166, "matches": 31},
    "Antonio Giua": {"home_win_rate": 0.566, "lift": 0.151, "matches": 53},
    "Simone Sozza": {"home_win_rate": 0.545, "lift": 0.130, "matches": 44},
    "Valerio Marini": {"home_win_rate": 0.538, "lift": 0.123, "matches": 39},
    "Livio Marinelli": {"home_win_rate": 0.529, "lift": 0.114, "matches": 34},
    "Rosario Abisso": {"home_win_rate": 0.524, "lift": 0.109, "matches": 42},
}

# Away-favoring referees (home win rate significantly below average)
AWAY_FAVORING_REFS = {
    "Paolo Mazzoleni": {"home_win_rate": 0.286, "lift": -0.129, "matches": 35},
    "Daniele Doveri": {"home_win_rate": 0.311, "lift": -0.104, "matches": 119},
    "Marco Di Bello": {"home_win_rate": 0.350, "lift": -0.066, "matches": 103},
    "Luca Pairetto": {"home_win_rate": 0.362, "lift": -0.053, "matches": 58},
    "Maurizio Mariani": {"home_win_rate": 0.378, "lift": -0.037, "matches": 90},
}

# Strict referees (high card rate)
STRICT_REFS = {
    "Fabio Maresca": {"avg_cards": 5.42, "high_card_rate": 0.625, "matches": 56},
    "Daniele Orsato": {"avg_cards": 5.21, "high_card_rate": 0.598, "matches": 87},
    "Gianluca Manganiello": {"avg_cards": 5.15, "high_card_rate": 0.583, "matches": 48},
    "Michael Fabbri": {"avg_cards": 5.08, "high_card_rate": 0.571, "matches": 63},
    "Marco Guida": {"avg_cards": 4.98, "high_card_rate": 0.556, "matches": 72},
}

# Lenient referees (low card rate)
LENIENT_REFS = {
    "Gianpaolo Calvarese": {"avg_cards": 3.89, "high_card_rate": 0.389, "matches": 54},
    "Massimiliano Irrati": {"avg_cards": 4.02, "high_card_rate": 0.412, "matches": 49},
    "Piero Giacomelli": {"avg_cards": 4.11, "high_card_rate": 0.428, "matches": 42},
}


def classify_referee(referee_name: str) -> Dict:
    """Classify a referee using computed data from features/referee.py.

    Falls back to hardcoded dicts only if the referee module is unavailable
    or the referee has no computed history.
    """
    if not referee_name:
        return {"bias_type": "unknown", "confidence": "NONE"}

    name = referee_name.strip()

    # --- Data-driven classification (preferred) ---
    if _HAS_REFEREE_MODULE:
        try:
            matches_path = DATA_DIR / "parsed" / "matches.parquet"
            if matches_path.exists():
                matches_df = pd.read_parquet(matches_path)
                profile = get_referee_profile(name, matches_df)
                if profile.get("matches", 0) >= 10:
                    return _classify_from_profile(profile)
        except Exception as e:
            log.debug(f"Data-driven referee classification failed: {e}")

    # --- Fallback: hardcoded dicts ---
    return _classify_from_hardcoded(name)


def _classify_from_profile(profile: Dict) -> Dict:
    """Classify referee from computed profile data.

    Note: get_referee_profile returns home_win_pct in 0-100 scale and
    avg_yellows_per_match as per-match average.
    """
    matches = profile.get("matches", 0)
    # home_win_pct is 0-100 scale from get_referee_profile
    home_win_pct = profile.get("home_win_pct", 44.4) / 100.0
    avg_yellows = profile.get("avg_yellows_per_match", 4.5)
    confidence = "HIGH" if matches >= 50 else "MEDIUM"

    # Home bias: >52% home win rate = home-favoring, <38% = away-favoring
    if home_win_pct >= 0.52:
        return {
            "bias_type": "home_favoring",
            "home_win_lift": round(home_win_pct - 0.444, 3),
            "home_win_rate": round(home_win_pct, 3),
            "matches": matches,
            "avg_yellows": round(avg_yellows, 2),
            "confidence": confidence,
            "source": "computed",
        }
    elif home_win_pct <= 0.38:
        return {
            "bias_type": "away_favoring",
            "home_win_lift": round(home_win_pct - 0.444, 3),
            "home_win_rate": round(home_win_pct, 3),
            "matches": matches,
            "avg_yellows": round(avg_yellows, 2),
            "confidence": confidence,
            "source": "computed",
        }

    # Card strictness: >5.0 avg yellows = strict, <4.0 = lenient
    if avg_yellows >= 5.0:
        return {
            "bias_type": "strict",
            "avg_cards": round(avg_yellows, 2),
            "matches": matches,
            "confidence": confidence,
            "source": "computed",
        }
    elif avg_yellows <= 4.0:
        return {
            "bias_type": "lenient",
            "avg_cards": round(avg_yellows, 2),
            "matches": matches,
            "confidence": confidence,
            "source": "computed",
        }

    return {"bias_type": "neutral", "matches": matches, "confidence": confidence, "source": "computed"}


def _classify_from_hardcoded(name: str) -> Dict:
    """Fallback: classify using hardcoded referee dicts."""
    for ref, data in HOME_FAVORING_REFS.items():
        if ref.lower() in name.lower() or name.lower() in ref.lower():
            return {
                "bias_type": "home_favoring",
                "home_win_lift": data["lift"],
                "home_win_rate": data["home_win_rate"],
                "matches": data["matches"],
                "confidence": "HIGH" if data["matches"] >= 50 else "MEDIUM",
                "source": "hardcoded",
            }
    for ref, data in AWAY_FAVORING_REFS.items():
        if ref.lower() in name.lower() or name.lower() in ref.lower():
            return {
                "bias_type": "away_favoring",
                "home_win_lift": data["lift"],
                "home_win_rate": data["home_win_rate"],
                "matches": data["matches"],
                "confidence": "HIGH" if data["matches"] >= 50 else "MEDIUM",
                "source": "hardcoded",
            }
    for ref, data in STRICT_REFS.items():
        if ref.lower() in name.lower() or name.lower() in ref.lower():
            return {
                "bias_type": "strict",
                "avg_cards": data["avg_cards"],
                "matches": data["matches"],
                "confidence": "HIGH" if data["matches"] >= 50 else "MEDIUM",
                "source": "hardcoded",
            }
    for ref, data in LENIENT_REFS.items():
        if ref.lower() in name.lower() or name.lower() in ref.lower():
            return {
                "bias_type": "lenient",
                "avg_cards": data["avg_cards"],
                "matches": data["matches"],
                "confidence": "MEDIUM",
                "source": "hardcoded",
            }
    return {"bias_type": "neutral", "confidence": "LOW", "source": "hardcoded"}


def get_referee_factors(referee_name: str) -> List[str]:
    """Get prediction factors based on referee classification."""
    classification = classify_referee(referee_name)
    factors = []

    if classification["bias_type"] == "home_favoring":
        factors.append("home_fav_ref")
    elif classification["bias_type"] == "away_favoring":
        factors.append("away_fav_ref")

    if classification["bias_type"] == "strict":
        factors.append("strict_ref")
    elif classification["bias_type"] == "lenient":
        factors.append("lenient_ref")

    return factors


def load_referee_assignments() -> Dict[str, str]:
    """Load referee assignments from manual file.

    Expected format in data/upcoming/referees.json:
    {
        "Inter vs Milan": "Daniele Orsato",
        "Juventus vs Roma": "Marco Di Bello"
    }
    """
    refs_path = DATA_DIR / "upcoming" / "referees.json"

    if refs_path.exists():
        try:
            with open(refs_path) as f:
                data = json.load(f)
            log.info(f"Loaded {len(data)} referee assignments")
            return data
        except Exception as e:
            log.warning(f"Failed to load referee assignments: {e}")

    return {}


def predict_referee_from_history(home_team: str, away_team: str) -> Optional[str]:
    """Predict likely referee based on historical assignments.

    This analyzes which referees typically officiate matches involving these teams.
    """
    try:
        refs_path = DATA_DIR / "external" / "referee" / "referee_assignments.parquet"
        if not refs_path.exists():
            return None

        df = pd.read_parquet(refs_path)

        # Find referees who have officiated these teams
        team_matches = df[
            (df["home_team"] == home_team) | (df["away_team"] == home_team) |
            (df["home_team"] == away_team) | (df["away_team"] == away_team)
        ]

        if len(team_matches) == 0:
            return None

        # Most common referee for these teams
        most_common = team_matches["referee"].value_counts().head(3)

        if len(most_common) > 0:
            return most_common.index[0]

    except Exception as e:
        log.warning(f"Failed to predict referee: {e}")

    return None


def get_referee_features_for_prediction(
    referee_name: str, home_team: str, away_team: str,
) -> Dict[str, float]:
    """Compute the same referee features used in training for a prediction.

    Returns a dict of feature_name -> value matching the columns produced by
    features/referee.py's add_referee_features(), so the ML models see
    consistent data at training and prediction time.
    """
    defaults = {
        "ref_avg_yellows": 0.0, "ref_avg_reds": 0.0, "ref_avg_fouls": 0.0,
        "ref_avg_penalties": 0.0, "ref_matches_officiated": 0.0,
        "ref_strictness_score": 0.0, "ref_home_bias": 0.0,
        "ref_home_cards_bias": 0.0, "ref_avg_total_goals": 0.0,
        "ref_avg_xg": 0.0, "ref_xg_vs_league": 0.0,
        "ref_strictness_trend": 0.0, "ref_big_match_card_modifier": 0.0,
        "ref_home_team_cards": 0.0, "ref_away_team_cards": 0.0,
        "ref_vs_home_team_bias": 0.0, "ref_vs_away_team_bias": 0.0,
    }
    if not referee_name or not _HAS_REFEREE_MODULE:
        return defaults

    try:
        matches_path = DATA_DIR / "parsed" / "matches.parquet"
        if not matches_path.exists():
            return defaults
        matches_df = pd.read_parquet(matches_path)

        profile = get_referee_profile(referee_name, matches_df)
        if profile.get("matches", 0) < 3:
            return defaults

        n = profile["matches"]
        # Profile uses *_per_match keys and home_win_pct in 0-100 scale
        defaults["ref_avg_yellows"] = profile.get("avg_yellows_per_match", 0)
        defaults["ref_avg_reds"] = profile.get("avg_reds_per_match", 0)
        defaults["ref_avg_fouls"] = profile.get("avg_fouls_per_match", 0)
        defaults["ref_avg_penalties"] = profile.get("avg_penalties_per_match", 0)
        defaults["ref_matches_officiated"] = float(n)
        avg_fouls = profile.get("avg_fouls_per_match", 0)
        avg_cards = profile.get("avg_yellows_per_match", 0) + profile.get("avg_reds_per_match", 0) * 2
        defaults["ref_strictness_score"] = round(avg_cards / avg_fouls, 3) if avg_fouls > 0 else 0
        # home_win_pct is 0-100 scale; convert to fraction for bias
        defaults["ref_home_bias"] = round(profile.get("home_win_pct", 44.4) / 100.0 - 0.444, 3)
        defaults["ref_avg_total_goals"] = profile.get("avg_total_goals", 0)
        defaults["ref_avg_xg"] = profile.get("avg_xg", 0) or 0
        defaults["ref_strictness_trend"] = profile.get("strictness_trend_last5", 0) or 0

        # Team-specific bias
        for team, prefix in [(home_team, "home"), (away_team, "away")]:
            hist = get_referee_team_history(referee_name, team, matches_df)
            if hist.get("data_available"):
                defaults[f"ref_{prefix}_team_cards"] = hist.get("avg_cards_from_this_ref", 0)
                defaults[f"ref_vs_{prefix}_team_bias"] = hist.get("ref_team_bias", 0)

    except Exception as e:
        log.debug(f"Failed to compute referee features: {e}")

    return defaults


def analyze_referee_impact(matches: List[Dict]) -> Dict:
    """Analyze referee impact on all upcoming matches.

    Returns enhanced match data with referee factors and computed features.
    """
    assignments = load_referee_assignments()
    results = {}

    for match in matches:
        home = match["home_team"]
        away = match["away_team"]
        key = f"{home} vs {away}"

        referee = assignments.get(key) or match.get("referee")

        if not referee:
            referee = predict_referee_from_history(home, away)
            if referee:
                log.info(f"Predicted referee for {key}: {referee} (from history)")

        if referee:
            classification = classify_referee(referee)
            factors = get_referee_factors(referee)
            computed_features = get_referee_features_for_prediction(referee, home, away)

            results[key] = {
                "referee": referee,
                "classification": classification,
                "factors": factors,
                "computed_features": computed_features,
                "source": "manual" if key in assignments else "predicted",
            }

            log.info(f"{key}: {referee} ({classification['bias_type']}, "
                     f"source={classification.get('source', 'unknown')})")
        else:
            results[key] = {
                "referee": None,
                "classification": {"bias_type": "unknown"},
                "factors": [],
                "computed_features": {},
                "source": "none",
            }

    return results


def create_referee_template(matches: List[Dict]):
    """Create a template file for manual referee entry."""
    template = {}

    for match in matches:
        key = f"{match['home_team']} vs {match['away_team']}"
        template[key] = ""  # Empty for user to fill in

    template_path = DATA_DIR / "upcoming" / "referees.json"

    with open(template_path, "w") as f:
        json.dump(template, f, indent=2)

    log.info(f"Created referee template at {template_path}")
    log.info("Fill in referee names to enable 91.5% accuracy predictions!")

    return template_path


def get_perfect_storm_matches(matches: List[Dict], form_data: Dict, referee_data: Dict) -> List[Dict]:
    """Find matches with 5+ factors (your 95.8-100% accuracy scenarios).

    Perfect storm criteria:
    - Hot home + Cold away + Big stadium + Home favorite + Home-fav ref
    """
    perfect_storms = []

    for match in matches:
        home = match["home_team"]
        away = match["away_team"]
        key = f"{home} vs {away}"

        matchup = form_data.get("matchups", {}).get(key, {})
        ref_info = referee_data.get(key, {})

        # Count all factors
        factors = matchup.get("factors", [])
        ref_factors = ref_info.get("factors", [])
        all_factors = factors + ref_factors

        # Check for perfect storm
        home_factors = [f for f in all_factors if f in [
            "big_stadium", "home_favorite", "big_home_favorite",
            "hot_home", "cold_away", "home_fav_ref"
        ]]

        if len(home_factors) >= 5:
            perfect_storms.append({
                "match": key,
                "factors": home_factors,
                "n_factors": len(home_factors),
                "prediction": "STRONG HOME WIN",
                "confidence": "VERY HIGH"
            })
            log.info(f"🔥 PERFECT STORM: {key} with {len(home_factors)} factors!")

    return perfect_storms


if __name__ == "__main__":
    # Load matches
    matches_path = DATA_DIR / "upcoming" / "manual_matches.json"
    if matches_path.exists():
        with open(matches_path) as f:
            data = json.load(f)
        matches = data.get("matches", [])

        # Create template for referee input
        create_referee_template(matches)

        # Analyze with any existing assignments
        ref_data = analyze_referee_impact(matches)

        print("\n" + "=" * 60)
        print("REFEREE ANALYSIS")
        print("=" * 60)

        for match_key, info in ref_data.items():
            print(f"\n{match_key}")
            if info["referee"]:
                print(f"  Referee: {info['referee']}")
                print(f"  Bias: {info['classification']['bias_type'].upper()}")
                if info["factors"]:
                    print(f"  Factors: {', '.join(info['factors'])}")
            else:
                print("  ⚠️ No referee assigned - add to data/upcoming/referees.json")

        print("\n" + "=" * 60)
        print("To enable 91.5% accuracy predictions:")
        print("1. Edit data/upcoming/referees.json")
        print("2. Add referee names for each match")
        print("3. Re-run predictions")
        print("=" * 60)
    else:
        print("No upcoming matches found")
