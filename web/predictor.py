"""Match prediction service using trained models."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
_DEFAULT_MODELS_DIR = PROJECT_ROOT / "data" / "models" / "universal"
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "features.parquet"


class MatchPredictor:
    """Loads trained model and makes predictions with explanations."""

    def __init__(self, use_no_odds_model: bool = False, league: str | None = None):
        """Initialize the predictor.

        Args:
            use_no_odds_model: If True, use the model trained without odds
                              (better for predicting future matches)
            league: If specified, load per-league model from data/models/{league}/.
                    Falls back to universal if league-specific model doesn't exist.
        """
        self.model = None
        self.feature_names = []
        self.metadata = {}
        self.use_no_odds_model = use_no_odds_model
        # Resolve model directory: per-league if available, else universal
        if league:
            league_dir = PROJECT_ROOT / "data" / "models" / league
            self._models_dir = league_dir if league_dir.exists() else _DEFAULT_MODELS_DIR
        else:
            self._models_dir = _DEFAULT_MODELS_DIR
        self._load_model()

    def _load_model(self):
        """Load the appropriate CatBoost model."""
        try:
            from catboost import CatBoostClassifier

            # Choose model based on configuration
            # For upcoming/future matches, use the pre-match features model
            if self.use_no_odds_model:
                # Try upcoming model first (pre-match features only)
                model_path = self._models_dir / "catboost_upcoming.cbm"
                metadata_path = self._models_dir / "catboost_upcoming_metadata.json"

                if not model_path.exists():
                    # Fall back to no-odds model
                    model_path = self._models_dir / "catboost_no_odds.cbm"
                    metadata_path = self._models_dir / "catboost_no_odds_metadata.json"
            else:
                model_path = self._models_dir / "catboost_latest.cbm"
                metadata_path = self._models_dir / "catboost_metadata.json"

            if not model_path.exists():
                log.warning(f"Model not found at {model_path}")
                # Fall back to the other model
                if self.use_no_odds_model:
                    model_path = self._models_dir / "catboost_latest.cbm"
                    metadata_path = self._models_dir / "catboost_metadata.json"
                else:
                    model_path = self._models_dir / "catboost_no_odds.cbm"
                    metadata_path = self._models_dir / "catboost_no_odds_metadata.json"

            if not model_path.exists():
                log.error("No model found")
                return

            self.model = CatBoostClassifier()
            self.model.load_model(str(model_path))

            # Get feature names directly from the model (more reliable than metadata)
            self.feature_names = list(self.model.feature_names_)

            if metadata_path.exists():
                with open(metadata_path) as f:
                    self.metadata = json.load(f)

            model_type = "no-odds" if "no_odds" in str(model_path) else "standard"
            log.info(f"Loaded {model_type} CatBoost model with {len(self.feature_names)} features")

        except Exception as e:
            log.error(f"Failed to load model: {e}")

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance from the model."""
        if self.model is None or not self.feature_names:
            return {}

        try:
            importances = self.model.get_feature_importance()
            return dict(zip(self.feature_names, importances.tolist()))
        except Exception as e:
            log.error(f"Failed to get feature importance: {e}")
            return {}

    def predict(self, features: pd.DataFrame) -> dict[str, Any]:
        """Make prediction for a single match.

        Returns dict with:
            - prediction: 'H', 'D', or 'A'
            - probabilities: {'H': float, 'D': float, 'A': float}
            - confidence: float (max probability)
            - reasons: list of top contributing features
        """
        if self.model is None:
            return {"error": "Model not loaded"}

        if features.empty:
            return {"error": "No features found for this match"}

        # Work on a copy to avoid modifying original
        df = features.copy()

        # Ensure we have the right features
        missing = [f for f in self.feature_names if f not in df.columns]
        if missing:
            log.warning(f"Missing {len(missing)} features, filling with 0")
            for f in missing:
                df[f] = 0.0

        # Select only model features in correct order and ensure numeric types
        X = df[self.feature_names].fillna(0).astype(float)

        try:
            # Get probabilities
            proba = self.model.predict_proba(X)[0]
            pred_idx = np.argmax(proba)

            # Map to labels (model uses LABEL_MAP: H=0, D=1, A=2)
            labels = ["H", "D", "A"]
            prediction = labels[pred_idx]

            probabilities = {
                "H": float(proba[0]),
                "D": float(proba[1]),
                "A": float(proba[2]),
            }

            # Get top contributing features
            reasons = self._get_prediction_reasons(X.iloc[0], probabilities)

            return {
                "prediction": prediction,
                "probabilities": probabilities,
                "confidence": float(max(proba)),
                "reasons": reasons,
            }

        except Exception as e:
            log.error(f"Prediction failed: {e}")
            return {"error": str(e)}

    def _get_prediction_reasons(
        self, row: pd.Series, probs: dict[str, float]
    ) -> list[dict]:
        """Generate human-readable reasons for the prediction."""
        reasons = []

        # Get feature values and importance
        importance = self.get_feature_importance()
        if not importance:
            return reasons

        # Top 10 most important features
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]

        for feat, imp in top_features:
            if feat not in row.index:
                continue

            val = row[feat]
            if pd.isna(val):
                continue

            reason = self._explain_feature(feat, val, probs)
            if reason:
                reasons.append({
                    "feature": feat,
                    "value": float(val) if isinstance(val, (int, float, np.number)) else str(val),
                    "importance": float(imp),
                    "explanation": reason,
                })

        return reasons[:8]  # Top 8 reasons

    def _explain_feature(self, feat: str, val: float, probs: dict) -> str:
        """Generate human-readable explanation for a feature."""
        explanations = {
            "odds_B365H": lambda v: f"Bookmaker odds suggest {'home favorite' if v < 2.0 else 'home underdog'} (odds: {v:.2f})",
            "odds_B365A": lambda v: f"Bookmaker odds suggest {'away favorite' if v < 2.0 else 'away underdog'} (odds: {v:.2f})",
            "odds_B365D": lambda v: f"Draw odds: {v:.2f}",
            "elo_diff": lambda v: f"Elo rating difference: {v:+.0f} points ({'home stronger' if v > 0 else 'away stronger'})",
            "home_elo": lambda v: f"Home team Elo: {v:.0f}",
            "away_elo": lambda v: f"Away team Elo: {v:.0f}",
            "home_attack_strength": lambda v: f"Home attack strength: {v:.2f}",
            "away_attack_strength": lambda v: f"Away attack strength: {v:.2f}",
            "attack_strength_diff": lambda v: f"Attack strength difference: {v:+.2f}",
            "home_gd_per_match": lambda v: f"Home team avg goal diff: {v:+.2f}",
            "away_gd_per_match": lambda v: f"Away team avg goal diff: {v:+.2f}",
            "league_position_diff": lambda v: f"League position difference: {int(v)} places",
            "matchup_competitiveness": lambda v: f"Matchup competitiveness: {v:.2f} ({'evenly matched' if v > 0.5 else 'one-sided'})",
            "home_roll_10_goals_scored": lambda v: f"Home team goals (last 10): {v:.1f}",
            "away_roll_10_goals_scored": lambda v: f"Away team goals (last 10): {v:.1f}",
            "home_total_shots": lambda v: f"Home team total shots: {v:.0f}",
            "away_total_shots": lambda v: f"Away team total shots: {v:.0f}",
            "home_league_points": lambda v: f"Home team league points: {v:.0f}",
            "away_league_points": lambda v: f"Away team league points: {v:.0f}",
            "home_is_congested": lambda v: f"Home team fixture congestion: {'Yes' if v else 'No'}",
            "away_is_congested": lambda v: f"Away team fixture congestion: {'Yes' if v else 'No'}",
        }

        if feat in explanations:
            try:
                return explanations[feat](val)
            except Exception as e:
                log.debug(f"Failed to generate explanation for feature '{feat}': {e}")

        # Generic explanation for unknown features
        if "home_" in feat:
            return f"Home team {feat.replace('home_', '')}: {val:.2f}"
        elif "away_" in feat:
            return f"Away team {feat.replace('away_', '')}: {val:.2f}"
        elif "_diff" in feat:
            return f"{feat.replace('_diff', '')} difference: {val:+.2f}"

        return None


def get_upcoming_matches() -> list[dict]:
    """Get upcoming and recent matches from the features table (all leagues)."""
    if not FEATURES_PATH.exists():
        return []

    try:
        df = pd.read_parquet(FEATURES_PATH)

        # Parse dates
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

        # Get current season (2025-2026) if available, otherwise latest
        if "season" in df.columns:
            if "2025-2026" in df["season"].values:
                df = df[df["season"] == "2025-2026"]
            else:
                current_season = df["season"].max()
                df = df[df["season"] == current_season]

        # Sort by date descending to get most recent first
        df = df.sort_values("match_date", ascending=False)

        # Get recent matches (last 30)
        matches = []
        for _, row in df.head(30).iterrows():
            matches.append({
                "match_id": str(row.get("match_id", "")),
                "date": row["match_date"].strftime("%Y-%m-%d") if pd.notna(row["match_date"]) else "",
                "home_team": str(row.get("home_team", "Unknown")),
                "away_team": str(row.get("away_team", "Unknown")),
                "season": str(row.get("season", "")),
                "league": str(row.get("league", "")),
                "matchweek": int(row.get("matchweek", 0)) if pd.notna(row.get("matchweek")) else 0,
                "home_score": int(row.get("home_score")) if pd.notna(row.get("home_score")) else None,
                "away_score": int(row.get("away_score")) if pd.notna(row.get("away_score")) else None,
            })

        return matches

    except Exception as e:
        log.error(f"Failed to load matches: {e}")
        return []


def get_match_features(match_id: str) -> pd.DataFrame:
    """Get features for a specific match."""
    if not FEATURES_PATH.exists():
        return pd.DataFrame()

    try:
        df = pd.read_parquet(FEATURES_PATH)
        match = df[df["match_id"] == match_id]
        return match

    except Exception as e:
        log.error(f"Failed to load match features: {e}")
        return pd.DataFrame()
