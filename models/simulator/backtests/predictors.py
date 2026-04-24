"""Concrete Predictor adapters used by the harness.

- ProdCatboostBaseline: wraps existing data/models/markets/prod_*.cbm files.
  No retraining; the models are loaded once and predict per-eval-batch.
  Used to produce the baseline report the simulator must beat.

- PoissonFromFeatures: wraps the current features_serie_a.parquet poisson_*
  columns as a predictor. Answers "how does the plain-Poisson λ predictor
  perform if treated as a bettor?" Useful as a sanity-check baseline.

Future: SimulatorPredictor (Phase 1+) will emit joint-simulator probabilities.
It will slot into the same harness without code changes.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoost

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MARKETS_DIR = PROJECT_ROOT / "data" / "models" / "markets"


class _CatboostWrapper:
    def __init__(self, path: Path):
        self.path = path
        self.model = CatBoost()
        self.model.load_model(str(path))

    def align(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = list(self.model.feature_names_)
        missing = [f for f in feats if f not in df.columns]
        if missing:
            df = df.copy()
            for m in missing:
                df[m] = 0.0
        return df[feats].fillna(0.0)

    def predict_prob(self, df: pd.DataFrame) -> np.ndarray:
        X = self.align(df)
        preds = self.model.predict(X, prediction_type="Probability")
        if preds.ndim == 2 and preds.shape[1] == 2:
            return preds[:, 1]
        if preds.ndim == 1:
            return preds
        return preds

    def predict_multiclass(self, df: pd.DataFrame, classes: tuple[str, ...]) -> np.ndarray:
        X = self.align(df)
        preds = self.model.predict(X, prediction_type="Probability")
        if preds.ndim == 1:
            preds = np.column_stack([1 - preds, preds])
        model_classes = list(self.model.classes_) if hasattr(self.model, "classes_") else list(classes)
        # Normalize to strings for matching
        model_classes = [str(c) for c in model_classes]
        out = np.zeros((preds.shape[0], len(classes)))
        for i, target_c in enumerate(classes):
            if str(target_c) in model_classes:
                ci = model_classes.index(str(target_c))
                out[:, i] = preds[:, ci]
        return out


class ProdCatboostBaseline:
    """Adapter: existing prod_*.cbm + prod_1x2.cbm multiclass model."""
    name = "prod_baseline"
    version = "2026-04-21"

    # Mapping of market label -> model filename + kind
    _MARKET_TO_MODEL = {
        "O/U 2.5": ("prod_over_2_5.cbm", "binary"),
        "O/U 1.5": ("prod_over_1_5.cbm", "binary"),
        "O/U 3.5": ("prod_over_3_5.cbm", "binary"),
        "BTTS":    ("prod_btts.cbm", "binary"),
        "1X2":     ("prod_1x2.cbm", "multiclass"),
    }

    def __init__(self):
        self._models: dict[str, _CatboostWrapper] = {}
        for mk, (fname, _) in self._MARKET_TO_MODEL.items():
            p = MARKETS_DIR / fname
            if p.exists():
                try:
                    self._models[mk] = _CatboostWrapper(p)
                except Exception as e:
                    log.warning("Failed to load %s: %s", p.name, e)
            else:
                log.info("Model not present: %s", fname)

    def fit(self, train_df: pd.DataFrame) -> None:
        # Pre-trained; no-op.
        return None

    def supports(self, market_label: str) -> bool:
        return market_label in self._models

    def predict_binary(self, eval_df: pd.DataFrame, market_label: str) -> np.ndarray:
        return self._models[market_label].predict_prob(eval_df)

    def predict_multiclass(self, eval_df: pd.DataFrame, market_label: str) -> np.ndarray:
        return self._models[market_label].predict_multiclass(eval_df, self.classes_for(market_label))

    def classes_for(self, market_label: str) -> tuple[str, ...]:
        if market_label == "1X2":
            return ("H", "D", "A")
        return ()


class PoissonFromFeatures:
    """Treats the existing poisson_* feature columns as the predictor.

    Baseline before any λ-estimator redesign. Produces:
    - P(over X) from poisson_over_{1_5,2_5,3_5}
    - P(BTTS) from poisson_btts
    - P(H/D/A) from poisson_prob_{H,D,A}
    """
    name = "poisson_from_features"
    version = "v1"

    def fit(self, train_df: pd.DataFrame) -> None:
        return None

    def supports(self, market_label: str) -> bool:
        return market_label in {"O/U 1.5", "O/U 2.5", "O/U 3.5", "BTTS", "1X2"}

    def predict_binary(self, eval_df: pd.DataFrame, market_label: str) -> np.ndarray:
        col = {
            "O/U 1.5": "poisson_over_1_5",
            "O/U 2.5": "poisson_over_2_5",
            "O/U 3.5": "poisson_over_3_5",
            "BTTS":    "poisson_btts",
        }[market_label]
        if col not in eval_df.columns:
            return np.array([])
        return eval_df[col].fillna(0.5).to_numpy(dtype=float)

    def predict_multiclass(self, eval_df: pd.DataFrame, market_label: str) -> np.ndarray:
        assert market_label == "1X2"
        cols = ["poisson_prob_H", "poisson_prob_D", "poisson_prob_A"]
        if not all(c in eval_df.columns for c in cols):
            return np.array([]).reshape(0, 3)
        arr = eval_df[cols].fillna(1/3.0).to_numpy(dtype=float)
        # Renormalize row-wise defensively
        row_sums = arr.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        return arr / row_sums

    def classes_for(self, market_label: str) -> tuple[str, ...]:
        return ("H", "D", "A")
