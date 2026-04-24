"""Per-team corner rate estimator.

Fits Poisson regressors for home_corners and away_corners using features that
correlate with corner generation: attack strength, Sofascore rolling shots and
xG, possession proxies, rolling corner averages.

Training target: corners_home / corners_away from matches.parquet (100% fill
after 2026-04 backfill) or data/external/sofascore/match_team_stats.parquet.

Usage:
    from models.simulator.base_rates.corner_rates import CornerRateEstimator
    est = CornerRateEstimator()
    est.fit(train_df)
    rate_h, rate_a = est.predict(eval_df)
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

log = logging.getLogger(__name__)


# Features used for corner rate prediction. Ordered by priority so missing-feature
# fallback produces a usable but degraded predictor.
DEFAULT_CORNER_FEATURES: tuple[str, ...] = (
    "home_attack_strength", "away_attack_strength",
    "home_defense_strength", "away_defense_strength",
    # Sofascore rolling signals — the main drivers
    "home_ss_roll_xg", "away_ss_roll_xg",
    "home_ss_roll_total_shots", "away_ss_roll_total_shots",
    # Set-piece xG share (from Phase 0b situational_xg) — corner-heavy teams
    "home_xg_share_setpiece_roll_5", "away_xg_share_setpiece_roll_5",
    # Rolling possession — more possession usually means more corners
    # (Sofascore rolling cols are named ss_roll_*)
    "home_ss_roll_possession", "away_ss_roll_possession",
)


class CornerRateEstimator:
    """Per-team Poisson rate estimator for corners."""

    def __init__(self, features: Iterable[str] = DEFAULT_CORNER_FEATURES,
                 alpha: float = 0.5, max_iter: int = 1000):
        self.features = list(features)
        self.alpha = alpha
        self.max_iter = max_iter
        self._model_home: PoissonRegressor | None = None
        self._model_away: PoissonRegressor | None = None
        self._usable_features: list[str] = []

    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit two PoissonRegressors (one for home corners, one for away)."""
        if "home_corners" not in train_df.columns or "away_corners" not in train_df.columns:
            log.warning("CornerRateEstimator: target columns missing, skipping fit")
            return
        available = [f for f in self.features if f in train_df.columns]
        if not available:
            log.warning("CornerRateEstimator: no usable features")
            return

        train = train_df.dropna(subset=available + ["home_corners", "away_corners"])
        if len(train) < 100:
            log.warning("CornerRateEstimator: insufficient training rows (%d)", len(train))
            return

        X = train[available].to_numpy(dtype=float)
        y_home = train["home_corners"].to_numpy(dtype=float)
        y_away = train["away_corners"].to_numpy(dtype=float)

        self._model_home = PoissonRegressor(alpha=self.alpha, max_iter=self.max_iter).fit(X, y_home)
        self._model_away = PoissonRegressor(alpha=self.alpha, max_iter=self.max_iter).fit(X, y_away)
        self._usable_features = available
        log.info("CornerRateEstimator fit on %d rows with %d features", len(train), len(available))

    def predict(self, eval_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (home_corner_rate, away_corner_rate) arrays.

        Clamps to [0.5, 15.0] as defensive bounds (Serie A mean ~4.5 per side).
        """
        if self._model_home is None or self._model_away is None:
            n = len(eval_df)
            # Fallback: league mean
            return np.full(n, 4.5), np.full(n, 4.5)
        X = eval_df[self._usable_features].fillna(0.0).to_numpy(dtype=float)
        rate_h = np.clip(self._model_home.predict(X), 0.5, 15.0)
        rate_a = np.clip(self._model_away.predict(X), 0.5, 15.0)
        return rate_h, rate_a

    @property
    def is_fit(self) -> bool:
        return self._model_home is not None and self._model_away is not None
