"""Draw Specialist Model - Specialized model for predicting draws.

Draws are the hardest outcome to predict in football:
- ~25% of Serie A matches end in draws
- Standard models often predict draws at <20% frequency
- Draws have specific patterns that main models miss

This module provides:
1. Draw-specific feature engineering
2. Binary draw classifier (draw vs non-draw)
3. Draw probability adjustment for main ensemble
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

log = logging.getLogger(__name__)


@dataclass
class DrawPrediction:
    """Draw prediction result with supporting features."""
    prob_draw: float
    confidence: float
    key_factors: list[str]
    recommend_draw_bet: bool


def compute_draw_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features specifically useful for draw prediction.

    Draws are more likely when:
    - Teams are evenly matched (similar strength)
    - Both teams are defensive/low-scoring
    - It's a high-stakes match (both teams cautious)
    - Historical H2H has many draws
    - Similar tactical styles (style_similarity high)
    - Match is late in season with relegation/safety decided

    Args:
        df: DataFrame with standard match features

    Returns:
        DataFrame with draw-specific features added
    """
    df = df.copy()

    # Strength/quality matching
    if "home_elo" in df.columns and "away_elo" in df.columns:
        elo_diff = abs(df["home_elo"] - df["away_elo"])
        # More even = higher draw likelihood
        df["draw_elo_factor"] = np.exp(-elo_diff / 200)  # Peaks when diff=0

    if "home_strength_rating" in df.columns and "away_strength_rating" in df.columns:
        strength_diff = abs(df["home_strength_rating"] - df["away_strength_rating"])
        df["draw_strength_factor"] = 1 / (1 + strength_diff)

    # Low-scoring tendency
    if "home_roll_goals_5" in df.columns and "away_roll_goals_5" in df.columns:
        combined_goals = df["home_roll_goals_5"] + df["away_roll_goals_5"]
        df["draw_low_scoring_factor"] = np.exp(-combined_goals / 3)

    if "home_roll_xg_5" in df.columns and "away_roll_xg_5" in df.columns:
        combined_xg = df["home_roll_xg_5"] + df["away_roll_xg_5"]
        df["draw_xg_factor"] = np.exp(-combined_xg / 3)

    # Form convergence (both teams on similar form)
    if "home_form_pts_5" in df.columns and "away_form_pts_5" in df.columns:
        form_diff = abs(df["home_form_pts_5"] - df["away_form_pts_5"])
        df["draw_form_convergence"] = 1 / (1 + form_diff)

    # H2H draw history
    if "h2h_draws" in df.columns and "h2h_total_games" in df.columns:
        df["h2h_draw_rate"] = df["h2h_draws"] / df["h2h_total_games"].clip(lower=1)
    elif "h2h_draws" in df.columns:
        df["h2h_draw_rate"] = df["h2h_draws"] / 10  # Assume 10 games history

    # Style similarity (from tactical module)
    if "style_similarity" in df.columns:
        df["draw_style_factor"] = df["style_similarity"]

    # Safe position flag (nothing to play for = more cautious)
    if "home_safe_position_flag" in df.columns and "away_safe_position_flag" in df.columns:
        df["both_safe"] = (
            df["home_safe_position_flag"].fillna(0) *
            df["away_safe_position_flag"].fillna(0)
        )

    # Defensive teams (low goals conceded)
    if "home_roll_conceded_5" in df.columns and "away_roll_conceded_5" in df.columns:
        low_conceded = (df["home_roll_conceded_5"] < 1.0) & (df["away_roll_conceded_5"] < 1.0)
        df["both_defensive"] = low_conceded.astype(float)

    # Matchup competitiveness
    if "matchup_competitiveness" in df.columns:
        df["draw_competitiveness"] = df["matchup_competitiveness"]

    return df


class DrawSpecialist:
    """Binary classifier specialized for predicting draws."""

    def __init__(
        self,
        class_weight: float = 1.5,  # Weight for draw class
        threshold: float = 0.25,  # Default draw probability threshold
    ):
        self.class_weight = class_weight
        self.threshold = threshold
        self.model: Optional[CatBoostClassifier] = None
        self.feature_names: Optional[list[str]] = None
        self.fitted = False

        # Draw-specific features to prioritize
        self.draw_features = [
            "draw_elo_factor", "draw_strength_factor", "draw_low_scoring_factor",
            "draw_xg_factor", "draw_form_convergence", "h2h_draw_rate",
            "draw_style_factor", "both_safe", "both_defensive",
            "draw_competitiveness", "style_similarity", "matchup_competitiveness",
        ]

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[list[str]] = None,
    ) -> "DrawSpecialist":
        """Train the draw specialist model.

        Args:
            X: Feature DataFrame
            y: Target series with values 'H', 'D', 'A'
            feature_names: Columns to use as features
        """
        # Add draw-specific features
        X = compute_draw_specific_features(X)

        # Determine feature columns
        if feature_names is None:
            feature_names = [c for c in X.columns if X[c].dtype in ['float64', 'int64', 'float32', 'int32']
                           and c not in ['match_id', 'season', 'home_score', 'away_score']]

        # Add any draw-specific features that exist
        for feat in self.draw_features:
            if feat in X.columns and feat not in feature_names:
                feature_names.append(feat)

        self.feature_names = feature_names

        # Binary target: 1 = draw, 0 = not draw
        y_binary = (y == 'D').astype(int)

        # Handle class imbalance
        n_draws = y_binary.sum()
        n_non_draws = len(y_binary) - n_draws
        scale_pos_weight = n_non_draws / max(1, n_draws) * self.class_weight

        log.info(f"Training draw specialist: {n_draws} draws, {n_non_draws} non-draws, weight={scale_pos_weight:.2f}")

        # Use best available library
        if HAS_CATBOOST:
            self.model = CatBoostClassifier(
                iterations=800,
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=5,
                scale_pos_weight=scale_pos_weight,
                verbose=False,
                random_seed=42,
            )
        elif HAS_XGBOOST:
            self.model = XGBClassifier(
                n_estimators=800,
                learning_rate=0.03,
                max_depth=6,
                reg_lambda=5,
                scale_pos_weight=scale_pos_weight,
                verbosity=0,
                random_state=42,
            )
        elif HAS_SKLEARN:
            self.model = GradientBoostingClassifier(
                n_estimators=400,
                learning_rate=0.03,
                max_depth=6,
                random_state=42,
            )
        else:
            raise ImportError("No suitable ML library available (catboost, xgboost, or sklearn required)")

        X_train = X[self.feature_names].fillna(0)
        self.model.fit(X_train, y_binary)
        self.fitted = True

        # Evaluate on training data
        y_pred = self.model.predict(X_train)
        acc = accuracy_score(y_binary, y_pred)
        prec = precision_score(y_binary, y_pred, zero_division=0)
        rec = recall_score(y_binary, y_pred, zero_division=0)
        f1 = f1_score(y_binary, y_pred, zero_division=0)

        log.info(f"Draw specialist training: acc={acc:.3f}, prec={prec:.3f}, rec={rec:.3f}, f1={f1:.3f}")

        return self

    def predict_draw_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of draw for each match.

        Returns:
            Array of draw probabilities (0-1)
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = compute_draw_specific_features(X)
        X_feat = X[self.feature_names].fillna(0)

        # Get probability of class 1 (draw)
        proba = self.model.predict_proba(X_feat)[:, 1]
        return proba

    def adjust_ensemble_probs(
        self,
        X: pd.DataFrame,
        ensemble_probs: np.ndarray,
        adjustment_weight: float = 0.3,
    ) -> np.ndarray:
        """Adjust ensemble probabilities using draw specialist predictions.

        Args:
            X: Feature DataFrame
            ensemble_probs: (n, 3) array from main ensemble [P(A), P(D), P(H)]
            adjustment_weight: How much to weight draw specialist (0-1)

        Returns:
            Adjusted probability array
        """
        draw_probs = self.predict_draw_proba(X)

        # Current draw prob from ensemble (assumes column order A, D, H)
        ensemble_draw = ensemble_probs[:, 1]

        # Blend ensemble and specialist draw probabilities
        adjusted_draw = (
            (1 - adjustment_weight) * ensemble_draw +
            adjustment_weight * draw_probs
        )

        # Redistribute probability mass
        adjusted = ensemble_probs.copy()
        draw_change = adjusted_draw - ensemble_draw

        # Take equally from H and A to maintain relative ordering
        adjusted[:, 0] -= draw_change / 2  # A
        adjusted[:, 1] = adjusted_draw  # D
        adjusted[:, 2] -= draw_change / 2  # H

        # Ensure non-negative and normalize
        adjusted = np.maximum(adjusted, 0.01)
        adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)

        return adjusted

    def get_draw_factors(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get key factors contributing to draw prediction for each match.

        Returns DataFrame with draw probability and key contributing factors.
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = compute_draw_specific_features(X)
        X_feat = X[self.feature_names].fillna(0)

        draw_probs = self.model.predict_proba(X_feat)[:, 1]

        # Get feature importances (handle different library APIs)
        if hasattr(self.model, 'get_feature_importance'):
            importances = self.model.get_feature_importance()
        elif hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            importances = [1.0] * len(self.feature_names)  # Default equal importance
        importance_dict = dict(zip(self.feature_names, importances))

        # For each match, find top contributing features
        results = []
        for i in range(len(X)):
            top_factors = []
            for feat in sorted(self.draw_features, key=lambda f: importance_dict.get(f, 0), reverse=True)[:5]:
                if feat in X_feat.columns:
                    val = X_feat.iloc[i][feat]
                    if pd.notna(val) and val > 0.3:
                        top_factors.append(f"{feat}={val:.2f}")

            results.append({
                "draw_prob": round(draw_probs[i], 3),
                "recommend_draw": draw_probs[i] > self.threshold,
                "key_factors": ", ".join(top_factors[:3]) if top_factors else "none",
            })

        return pd.DataFrame(results)


def train_draw_specialist(
    feature_path: str,
    min_seasons: int = 10,
) -> DrawSpecialist:
    """Train draw specialist on historical data.

    Args:
        feature_path: Path to features.parquet
        min_seasons: Minimum training seasons

    Returns:
        Fitted DrawSpecialist model
    """
    log.info("Training draw specialist model...")

    df = pd.read_parquet(feature_path)
    df = df.dropna(subset=['home_score', 'away_score'])

    # Create target
    df['result'] = df.apply(
        lambda r: 'H' if r['home_score'] > r['away_score']
        else ('A' if r['away_score'] > r['home_score'] else 'D'),
        axis=1
    )

    # Get feature columns
    exclude = ['match_id', 'home_team', 'away_team', 'home_score', 'away_score',
               'match_date', 'season', 'result', 'league']
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64']]

    specialist = DrawSpecialist()
    specialist.fit(df, df['result'], feature_cols)

    return specialist


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing draw specialist module...")
    print("=" * 60)

    # Test feature computation
    sample_df = pd.DataFrame({
        "home_elo": [1500, 1600, 1450],
        "away_elo": [1500, 1400, 1500],
        "home_roll_goals_5": [1.2, 2.0, 0.8],
        "away_roll_goals_5": [1.0, 1.8, 0.6],
        "style_similarity": [0.8, 0.4, 0.9],
    })

    df_with_draw_feats = compute_draw_specific_features(sample_df)

    print("\nDraw-specific features computed:")
    for col in df_with_draw_feats.columns:
        if col.startswith("draw_"):
            print(f"  {col}: {df_with_draw_feats[col].tolist()}")

    print("\n" + "=" * 60)
    print("Draw-specific features help identify:")
    print("  - Evenly matched teams (elo_factor, strength_factor)")
    print("  - Low-scoring matchups (low_scoring_factor, xg_factor)")
    print("  - Similar form and style (form_convergence, style_factor)")
