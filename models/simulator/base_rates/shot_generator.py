"""Per-team shot + SOT rate estimator.

Shots per match ~ Poisson(rate). Shots-on-target | shots ~ Binomial(conversion).
Goals | SOT ~ Binomial(xG_per_SOT) — but we defer that to the core λ estimator
for now, since goals are what the simulator already samples.

This module exists so shot-quantity markets (team shots O/U, SOT O/U) and
corners that depend on shot rate become derivable from the trial array.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

log = logging.getLogger(__name__)


DEFAULT_SHOT_FEATURES: tuple[str, ...] = (
    "home_attack_strength", "away_attack_strength",
    "home_defense_strength", "away_defense_strength",
    "home_ss_roll_total_shots", "away_ss_roll_total_shots",
    "home_ss_roll_xg", "away_ss_roll_xg",
    "home_fb_roll_sh", "away_fb_roll_sh",  # FBref rolling shots if present
)


class ShotRateEstimator:
    """Poisson rate for total shots per side + empirical SOT/shot ratio."""

    SOT_RATIO_CLAMP = (0.25, 0.55)  # Serie A typical SOT/total ratio bounds

    def __init__(self, features: Iterable[str] = DEFAULT_SHOT_FEATURES,
                 alpha: float = 0.5, max_iter: int = 1000):
        self.features = list(features)
        self.alpha = alpha
        self.max_iter = max_iter
        self._model_home: PoissonRegressor | None = None
        self._model_away: PoissonRegressor | None = None
        self._usable_features: list[str] = []
        self._sot_ratio_home: float = 0.35
        self._sot_ratio_away: float = 0.33

    def fit(self, train_df: pd.DataFrame) -> None:
        # Shot target: matches.parquet columns
        if "home_shots_total" not in train_df.columns or "away_shots_total" not in train_df.columns:
            log.warning("ShotRateEstimator: shot totals missing, skipping fit")
            return

        available = [f for f in self.features if f in train_df.columns]
        if not available:
            log.warning("ShotRateEstimator: no usable features")
            return

        train = train_df.dropna(subset=available + ["home_shots_total", "away_shots_total"])
        if len(train) < 100:
            log.warning("ShotRateEstimator: insufficient rows (%d)", len(train))
            return

        X = train[available].to_numpy(dtype=float)
        yh = train["home_shots_total"].to_numpy(dtype=float)
        ya = train["away_shots_total"].to_numpy(dtype=float)

        self._model_home = PoissonRegressor(alpha=self.alpha, max_iter=self.max_iter).fit(X, yh)
        self._model_away = PoissonRegressor(alpha=self.alpha, max_iter=self.max_iter).fit(X, ya)
        self._usable_features = available

        # Empirical SOT/shot ratios
        if "home_shots_on_target" in train.columns and "away_shots_on_target" in train.columns:
            h_sot = train["home_shots_on_target"].fillna(0).sum()
            a_sot = train["away_shots_on_target"].fillna(0).sum()
            h_tot = max(1.0, train["home_shots_total"].sum())
            a_tot = max(1.0, train["away_shots_total"].sum())
            self._sot_ratio_home = float(np.clip(h_sot / h_tot, *self.SOT_RATIO_CLAMP))
            self._sot_ratio_away = float(np.clip(a_sot / a_tot, *self.SOT_RATIO_CLAMP))

        log.info("ShotRateEstimator fit on %d rows, %d features, SOT ratios home=%.3f away=%.3f",
                 len(train), len(available), self._sot_ratio_home, self._sot_ratio_away)

    def predict(self, eval_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (home_shot_rate, away_shot_rate)."""
        if self._model_home is None or self._model_away is None:
            n = len(eval_df)
            return np.full(n, 12.0), np.full(n, 11.5)
        X = eval_df[self._usable_features].fillna(0.0).to_numpy(dtype=float)
        rate_h = np.clip(self._model_home.predict(X), 3.0, 30.0)
        rate_a = np.clip(self._model_away.predict(X), 3.0, 30.0)
        return rate_h, rate_a

    def sot_ratios(self) -> tuple[float, float]:
        return self._sot_ratio_home, self._sot_ratio_away

    @property
    def is_fit(self) -> bool:
        return self._model_home is not None and self._model_away is not None
