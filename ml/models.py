"""Model wrappers for XGBoost and LightGBM with early stopping support."""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from ml.config import LABEL_MAP, MODEL_EXTENSIONS, MODEL_TYPES, ModelConfig, N_CLASSES

log = logging.getLogger(__name__)


class ModelWrapper:
    """Base class for ML model wrappers."""

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.model: Any = None
        self.feature_names: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "ModelWrapper":
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def get_feature_importance(self) -> Dict[str, float]:
        raise NotImplementedError


def _chronological_val_split(
    X: pd.DataFrame, y, val_fraction: float,
):
    """Split the last `val_fraction` of rows as validation (preserving time order)."""
    n = len(X)
    split = int(n * (1.0 - val_fraction))
    if isinstance(y, pd.Series):
        return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]
    # numpy array
    return X.iloc[:split], y[:split], X.iloc[split:], y[split:]


class XGBoostModel(ModelWrapper):
    """XGBoost multiclass classifier with early stopping."""

    def __init__(self, params: Dict[str, Any] | None = None):
        import xgboost as xgb

        cfg = ModelConfig()
        params = params or cfg.xgb_params
        super().__init__(params)
        self._cfg = cfg
        self.model = xgb.XGBClassifier(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "XGBoostModel":
        y_int = y.map(LABEL_MAP) if y.dtype == object else y
        self.feature_names = list(X.columns)

        # Convert to numpy to avoid XGBoost 3.x + pandas 2.x dtype compat issues
        X_np = X.values.astype(np.float32)
        y_np = y_int.values if hasattr(y_int, 'values') else np.asarray(y_int)

        es_rounds = kwargs.pop("early_stopping_rounds", self._cfg.early_stopping_rounds)

        if es_rounds and len(X) > 200:
            split = int(len(X_np) * (1.0 - self._cfg.early_stopping_val_fraction))
            X_tr, X_val = X_np[:split], X_np[split:]
            y_tr, y_val = y_np[:split], y_np[split:]
            sw = kwargs.pop("sample_weight", None)
            if sw is not None:
                kwargs["sample_weight"] = sw[:split]

            self.model.set_params(early_stopping_rounds=es_rounds)
            self.model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
                **kwargs,
            )
            log.debug(
                "XGBoost early stopped at %d/%d trees",
                self.model.best_iteration, self.model.n_estimators,
            )
        else:
            self.model.fit(X_np, y_np, **kwargs)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X.values.astype(np.float32))

    def get_feature_importance(self) -> Dict[str, float]:
        if self.feature_names is None:
            return {}
        imp = self.model.feature_importances_
        return dict(zip(self.feature_names, imp.tolist()))

    def save(self, path: str) -> None:
        self.model.save_model(path)

    def load(self, path: str) -> None:
        import xgboost as xgb

        self.model = xgb.XGBClassifier()
        self.model.load_model(path)


class LightGBMModel(ModelWrapper):
    """LightGBM multiclass classifier with early stopping."""

    def __init__(self, params: Dict[str, Any] | None = None):
        import lightgbm as lgb

        cfg = ModelConfig()
        params = params or cfg.lgb_params
        super().__init__(params)
        self._cfg = cfg
        self.model = lgb.LGBMClassifier(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "LightGBMModel":
        import lightgbm as lgb

        y_int = y.map(LABEL_MAP) if y.dtype == object else y
        self.feature_names = list(X.columns)

        es_rounds = kwargs.pop("early_stopping_rounds", self._cfg.early_stopping_rounds)

        if es_rounds and len(X) > 200:
            X_tr, y_tr, X_val, y_val = _chronological_val_split(
                X, y_int, self._cfg.early_stopping_val_fraction,
            )
            sw = kwargs.pop("sample_weight", None)
            if sw is not None:
                kwargs["sample_weight"] = sw[: len(X_tr)]

            callbacks = [
                lgb.early_stopping(es_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ]
            self.model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks,
                **kwargs,
            )
            log.debug(
                "LightGBM early stopped at %d/%d trees",
                self.model.best_iteration_, self.model.n_estimators,
            )
        else:
            self.model.fit(X, y_int, **kwargs)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Dict[str, float]:
        if self.feature_names is None:
            return {}
        imp = self.model.feature_importances_
        return dict(zip(self.feature_names, imp.tolist()))

    def save(self, path: str) -> None:
        self.model.booster_.save_model(path)

    def load(self, path: str) -> None:
        import lightgbm as lgb

        self.model = lgb.Booster(model_file=path)


class CatBoostModel(ModelWrapper):
    """CatBoost multiclass classifier with early stopping."""

    def __init__(self, params: Dict[str, Any] | None = None):
        from catboost import CatBoostClassifier

        cfg = ModelConfig()
        params = params or cfg.cb_params
        super().__init__(params)
        self._cfg = cfg
        self.model = CatBoostClassifier(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "CatBoostModel":
        y_int = y.map(LABEL_MAP) if y.dtype == object else y
        self.feature_names = list(X.columns)

        es_rounds = kwargs.pop("early_stopping_rounds", self._cfg.early_stopping_rounds)

        if es_rounds and len(X) > 200:
            X_tr, y_tr, X_val, y_val = _chronological_val_split(
                X, y_int, self._cfg.early_stopping_val_fraction,
            )
            sw = kwargs.pop("sample_weight", None)
            if sw is not None:
                kwargs["sample_weight"] = sw[: len(X_tr)]

            self.model.set_params(early_stopping_rounds=es_rounds)
            self.model.fit(
                X_tr, y_tr,
                eval_set=(X_val, y_val),
                verbose=False,
                **kwargs,
            )
            log.debug(
                "CatBoost early stopped at %d/%d trees",
                self.model.best_iteration_, self.model.get_param("iterations"),
            )
        else:
            self.model.fit(X, y_int, verbose=False, **kwargs)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Dict[str, float]:
        if self.feature_names is None:
            return {}
        imp = self.model.get_feature_importance()
        return dict(zip(self.feature_names, imp.tolist()))

    def save(self, path: str) -> None:
        self.model.save_model(path)

    def load(self, path: str) -> None:
        from catboost import CatBoostClassifier

        self.model = CatBoostClassifier()
        self.model.load_model(path)


_MODEL_REGISTRY: Dict[str, type] = {
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "catboost": CatBoostModel,
}


def get_model(model_type: str, params: Dict[str, Any] | None = None) -> ModelWrapper:
    """Factory function."""
    cls = _MODEL_REGISTRY.get(model_type)
    if cls is None:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(_MODEL_REGISTRY)}")
    return cls(params)
