"""Per-team card rate estimator.

Fits Poisson regressors for home/away yellow+red cards, with multiplicative
scaling by referee strictness (ref_strictness_score feature from Step 13).

Rationale: card rates are weakly predicted by team + opponent attributes but
strongly by referee. A strict ref gives 20-30% more cards; a lax ref does the
opposite. The scaling exploits this without requiring the referee to be a
training feature (which would encode referee identities and overfit).
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

log = logging.getLogger(__name__)


DEFAULT_CARD_FEATURES: tuple[str, ...] = (
    # Team-level tendencies
    "home_roll_5_yellow_cards", "away_roll_5_yellow_cards",
    "home_roll_10_yellow_cards", "away_roll_10_yellow_cards",
    # H2H / derby
    "is_derby",
    # Rolling aggression signals
    "home_roll_5_fouls", "away_roll_5_fouls",
    # Competitive pressure
    "league_position_diff",
)

# Referee strictness scaling: score is normalized 0-1. Anchor 0.5 = neutral.
# A score of 1.0 (strictest) scales up by 30%; 0.0 (most lax) scales down 30%.
REF_STRICTNESS_SCALE = 0.6  # amplitude of the scale factor


class CardRateEstimator:
    """Per-team Poisson rate for cards, ref-scaled at predict time."""

    def __init__(self, features: Iterable[str] = DEFAULT_CARD_FEATURES,
                 alpha: float = 0.5, max_iter: int = 1000,
                 use_ref_scaling: bool = True):
        self.features = list(features)
        self.alpha = alpha
        self.max_iter = max_iter
        self.use_ref_scaling = use_ref_scaling
        self._model_home: PoissonRegressor | None = None
        self._model_away: PoissonRegressor | None = None
        self._usable_features: list[str] = []

    @staticmethod
    def _total_cards_by_side(train_df: pd.DataFrame) -> tuple[pd.Series, pd.Series] | None:
        """Compute yellow + red cards per side if columns exist."""
        needed = {"home_yellow_cards", "away_yellow_cards"}
        if not needed.issubset(train_df.columns):
            return None
        home = train_df["home_yellow_cards"].fillna(0).astype(float)
        away = train_df["away_yellow_cards"].fillna(0).astype(float)
        if "home_red_cards" in train_df.columns:
            home = home + train_df["home_red_cards"].fillna(0).astype(float)
        if "away_red_cards" in train_df.columns:
            away = away + train_df["away_red_cards"].fillna(0).astype(float)
        return home, away

    def fit(self, train_df: pd.DataFrame) -> None:
        totals = self._total_cards_by_side(train_df)
        if totals is None:
            log.warning("CardRateEstimator: card columns missing, skipping fit")
            return
        y_home, y_away = totals

        available = [f for f in self.features if f in train_df.columns]
        if not available:
            log.warning("CardRateEstimator: no usable features")
            return

        train = train_df[train_df["home_yellow_cards"].notna() & train_df["away_yellow_cards"].notna()].copy()
        train = train.dropna(subset=available)
        if len(train) < 100:
            log.warning("CardRateEstimator: insufficient rows (%d)", len(train))
            return

        X = train[available].to_numpy(dtype=float)
        yh = train["home_yellow_cards"].fillna(0).astype(float).values
        if "home_red_cards" in train.columns:
            yh = yh + train["home_red_cards"].fillna(0).astype(float).values
        ya = train["away_yellow_cards"].fillna(0).astype(float).values
        if "away_red_cards" in train.columns:
            ya = ya + train["away_red_cards"].fillna(0).astype(float).values

        self._model_home = PoissonRegressor(alpha=self.alpha, max_iter=self.max_iter).fit(X, yh)
        self._model_away = PoissonRegressor(alpha=self.alpha, max_iter=self.max_iter).fit(X, ya)
        self._usable_features = available
        log.info("CardRateEstimator fit on %d rows, %d features", len(train), len(available))

    def _ref_scale(self, eval_df: pd.DataFrame) -> np.ndarray:
        """Scale factor ∈ [1 - REF_STRICTNESS_SCALE/2, 1 + REF_STRICTNESS_SCALE/2]."""
        n = len(eval_df)
        if not self.use_ref_scaling or "ref_strictness_score" not in eval_df.columns:
            return np.ones(n)
        raw = eval_df["ref_strictness_score"].fillna(0.5).to_numpy(dtype=float)
        # Centered on 0 → +/- amplitude
        return 1.0 + REF_STRICTNESS_SCALE * (raw - 0.5)

    def predict(self, eval_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self._model_home is None or self._model_away is None:
            n = len(eval_df)
            return np.full(n, 2.0), np.full(n, 2.2)  # mean-ish for Serie A
        X = eval_df[self._usable_features].fillna(0.0).to_numpy(dtype=float)
        rate_h = np.clip(self._model_home.predict(X), 0.3, 8.0)
        rate_a = np.clip(self._model_away.predict(X), 0.3, 8.0)
        scale = self._ref_scale(eval_df)
        return rate_h * scale, rate_a * scale

    @property
    def is_fit(self) -> bool:
        return self._model_home is not None and self._model_away is not None
