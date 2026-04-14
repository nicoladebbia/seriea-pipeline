"""Prediction Correction Layer — fixes systematic ensemble biases.

Two mechanisms work in tandem:

1. **StaticCorrector** — Logistic regression trained on 6,500+ OOF predictions
   with match context features. Learns structural biases like "model is
   underconfident above 60%" or "draws are overestimated". Updated monthly
   with retraining.

2. **RollingCorrector** — Exponential moving average of recent prediction errors,
   bucketed by (predicted_class, confidence_band). Adapts week-to-week after
   settlement. The "learning from mistakes" part.

WHY this works: ML models produce well-ranked probabilities but often have
systematic calibration gaps that are stable over time. A lightweight correction
layer can fix these without touching the model itself. The static corrector
handles persistent biases; the rolling corrector handles temporal drift.

Architecture:
  Components → Ensemble → Post-calibration → CalibrationPipeline → [THIS] → Output
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import DATA_DIR, MODELS_DIR

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CorrectionConfig:
    """All tunable parameters for the correction layer."""

    # Static corrector: DISABLED — analysis showed ECE worsened by 59%
    # (0.0113 → 0.0180) while log_loss improved only 0.15%.
    # Keeping code for future improvement; set to True to re-enable.
    static_enabled: bool = False

    # Static corrector: LogisticRegression C candidates (selected via inner CV)
    static_C_candidates: List[float] = field(
        default_factory=lambda: [0.01, 0.1, 0.5, 1.0, 5.0]
    )

    # Rolling corrector: EMA decay factor (0.95 = 20-match half-life)
    rolling_decay: float = 0.95

    # Minimum samples per bucket before rolling correction activates
    rolling_min_samples: int = 15

    # Confidence bands for rolling correction bucketing
    rolling_confidence_bands: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 1.0)]
    )

    # Correction factor clamp range (multiplicative, not additive)
    correction_factor_min: float = 0.70
    correction_factor_max: float = 1.30


# Context features used by the static corrector
CONTEXT_FEATURE_NAMES = [
    "prob_H", "prob_D", "prob_A",
    "confidence",       # max(prob_H, prob_D, prob_A)
    "predicted_class",  # 0=H, 1=D, 2=A
    "elo_gap",          # home_elo - away_elo
    "is_promoted_home",
    "is_promoted_away",
    "matchweek",
    "form_gap",         # home_form_points_5 - away_form_points_5
    "rest_gap",         # home_rest_days - away_rest_days
]

# Default values for missing context features
CONTEXT_DEFAULTS = {
    "elo_gap": 0.0,
    "is_promoted_home": 0.0,
    "is_promoted_away": 0.0,
    "matchweek": 19.0,
    "form_gap": 0.0,
    "rest_gap": 0.0,
}

# Paths
CORRECTION_MODEL_PATH = MODELS_DIR / "universal" / "correction_layer.pkl"
ROLLING_STATE_PATH = MODELS_DIR / "universal" / "rolling_calibration.json"
PREDICTION_LEDGER_PATH = DATA_DIR / "predictions" / "prediction_ledger.jsonl"


# ---------------------------------------------------------------------------
# Static Corrector
# ---------------------------------------------------------------------------

class StaticCorrector:
    """Logistic regression that learns structural probability biases.

    Takes the ensemble's 3-class probabilities + match context as input,
    predicts "corrected" class probabilities. The correction is applied as
    a multiplicative factor (LR output / input) clamped to [0.7, 1.3] to
    prevent wild swings.

    WHY logistic regression: It's the right tool for recalibrating probabilities.
    It learns monotonic, smooth adjustments — exactly what you want for fixing
    "underconfident at high bands" or "draw overestimation". Tree models would
    overfit on 6,500 samples.
    """

    def __init__(self, config: CorrectionConfig | None = None):
        self.config = config or CorrectionConfig()
        self.model = None       # sklearn LogisticRegression
        self.scaler = None      # sklearn StandardScaler
        self.is_fitted = False
        self.train_metrics: Dict[str, float] = {}

    def fit(
        self,
        oof_df: pd.DataFrame,
        features_df: pd.DataFrame | None = None,
    ) -> Dict[str, float]:
        """Train the static corrector on OOF predictions.

        Args:
            oof_df: DataFrame with columns [home_team, match_date, prob_H, prob_D,
                     prob_A, actual] where actual is "H"/"D"/"A".
            features_df: Optional features.parquet to join context features.
                         If None, uses only probability features.

        Returns:
            Dict with training metrics (log_loss_before, log_loss_after, etc.)
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import log_loss
        from sklearn.preprocessing import StandardScaler

        df = oof_df.copy()

        # Build context features
        df["confidence"] = df[["prob_H", "prob_D", "prob_A"]].max(axis=1)
        df["predicted_class"] = df[["prob_H", "prob_D", "prob_A"]].values.argmax(axis=1)

        # Check if context features are already present in the OOF df
        # (e.g., from ensemble._persist_oof_predictions which pre-computes them)
        has_context = all(c in df.columns for c in ["elo_gap", "matchweek"])
        if not has_context and features_df is not None and "home_team" in df.columns:
            df = self._join_context(df, features_df)
        # Always fill missing context cols with defaults (covers partial data or
        # join failures where home_team is missing from OOF)
        for col, default in CONTEXT_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default

        # Encode target
        label_map = {"H": 0, "D": 1, "A": 2}
        y = df["actual"].map(label_map).values
        valid = ~np.isnan(y.astype(float))
        df = df[valid].reset_index(drop=True)
        y = y[valid].astype(int)

        # Prepare feature matrix
        feature_cols = [c for c in CONTEXT_FEATURE_NAMES if c in df.columns]
        X = df[feature_cols].fillna(0.0).values

        # Baseline metrics (before correction)
        probs_before = df[["prob_H", "prob_D", "prob_A"]].values
        ll_before = log_loss(y, probs_before, labels=[0, 1, 2])

        # Walk-forward inner CV to select C
        # Use 60/80 splits (first 60% train, next 20% val, then 80/20)
        n = len(X)
        best_c = 1.0
        best_val_ll = float("inf")

        splits = [
            (0, int(n * 0.60), int(n * 0.60), int(n * 0.80)),
            (0, int(n * 0.80), int(n * 0.80), n),
        ]

        for c in self.config.static_C_candidates:
            val_losses = []
            for tr_start, tr_end, va_start, va_end in splits:
                if va_end <= va_start or tr_end <= tr_start:
                    continue
                X_tr, y_tr = X[tr_start:tr_end], y[tr_start:tr_end]
                X_va, y_va = X[va_start:va_end], y[va_start:va_end]
                probs_va = probs_before[va_start:va_end]

                # Need all 3 classes in training set for multinomial LR
                if len(np.unique(y_tr)) < 3:
                    continue

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_va_s = scaler.transform(X_va)

                lr = LogisticRegression(
                    C=c, solver="lbfgs",
                    max_iter=1000, random_state=42,
                )
                lr.fit(X_tr_s, y_tr)
                lr_proba = lr.predict_proba(X_va_s)

                # Apply as correction factors (LR / input), clamped
                corrected = self._apply_factors(probs_va, lr_proba)
                val_losses.append(log_loss(y_va, corrected, labels=[0, 1, 2]))

            avg_ll = np.mean(val_losses) if val_losses else float("inf")
            if avg_ll < best_val_ll:
                best_val_ll = avg_ll
                best_c = c

        log.info("Static corrector: best C=%.3f (val log_loss=%.4f vs baseline=%.4f)",
                 best_c, best_val_ll, ll_before)

        # Validation gate: don't deploy a corrector that hurts predictions
        if best_val_ll >= ll_before:
            log.warning("Static corrector does NOT improve log_loss (%.4f >= %.4f). "
                        "Skipping — passthrough is better than bad correction.",
                        best_val_ll, ll_before)
            return {
                "skipped": True,
                "reason": "no_improvement",
                "log_loss_before": round(ll_before, 4),
                "best_val_log_loss": round(best_val_ll, 4),
                "n_samples": len(y),
            }

        # Need all 3 classes to train multinomial LR
        if len(np.unique(y)) < 3:
            log.warning("Static corrector: only %d classes in data, need 3. Skipping.",
                        len(np.unique(y)))
            return {"error": "insufficient_classes", "n_classes": int(len(np.unique(y)))}

        # Train final model on all data with best C
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model = LogisticRegression(
            C=best_c, solver="lbfgs",
            max_iter=1000, random_state=42,
        )
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        # Final metrics on full training set (for reporting only)
        lr_proba_all = self.model.predict_proba(X_scaled)
        corrected_all = self._apply_factors(probs_before, lr_proba_all)
        ll_after = log_loss(y, corrected_all, labels=[0, 1, 2])

        # ECE calculation
        ece_before = self._compute_ece(y, probs_before)
        ece_after = self._compute_ece(y, corrected_all)

        self.train_metrics = {
            "log_loss_before": round(ll_before, 4),
            "log_loss_after": round(ll_after, 4),
            "log_loss_improvement": round(ll_before - ll_after, 4),
            "ece_before": round(ece_before, 4),
            "ece_after": round(ece_after, 4),
            "ece_improvement": round(ece_before - ece_after, 4),
            "best_C": best_c,
            "n_samples": len(y),
            "feature_cols": feature_cols,
        }
        log.info("Static corrector trained: ll %.4f → %.4f (Δ=%.4f), ECE %.4f → %.4f",
                 ll_before, ll_after, ll_before - ll_after, ece_before, ece_after)
        return self.train_metrics

    def correct(
        self,
        prob_H: float,
        prob_D: float,
        prob_A: float,
        context: Dict[str, float],
    ) -> Tuple[float, float, float]:
        """Apply static correction to a single prediction.

        Returns corrected (prob_H, prob_D, prob_A).
        """
        if not self.is_fitted:
            return prob_H, prob_D, prob_A

        feature_cols = self.train_metrics.get("feature_cols", CONTEXT_FEATURE_NAMES)
        x = np.array([[context.get(c, CONTEXT_DEFAULTS.get(c, 0.0)) for c in feature_cols]])
        x_scaled = self.scaler.transform(x)

        lr_proba = self.model.predict_proba(x_scaled)[0]
        input_probs = np.array([prob_H, prob_D, prob_A])

        corrected = self._apply_factors(
            input_probs.reshape(1, -1), lr_proba.reshape(1, -1)
        )[0]

        return float(corrected[0]), float(corrected[1]), float(corrected[2])

    def _apply_factors(self, input_probs: np.ndarray, lr_probs: np.ndarray) -> np.ndarray:
        """Compute correction factors and apply with clamping + renormalization.

        Factor = LR_output / input_prob, clamped to [0.7, 1.3].
        Corrected = input_prob * factor, then renormalized to sum to 1.
        """
        cfg = self.config
        # Avoid division by zero
        safe_input = np.clip(input_probs, 1e-6, None)
        factors = lr_probs / safe_input
        factors = np.clip(factors, cfg.correction_factor_min, cfg.correction_factor_max)

        corrected = input_probs * factors
        # Renormalize
        row_sums = corrected.sum(axis=-1, keepdims=True)
        row_sums = np.clip(row_sums, 1e-8, None)
        corrected = corrected / row_sums
        return corrected

    def _join_context(self, df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
        """Join context features from features.parquet onto OOF predictions."""
        # Extract context columns from features_df
        context_cols_map = {
            "home_elo": "home_elo",
            "away_elo": "away_elo",
            "home_is_promoted": "is_promoted_home",
            "away_is_promoted": "is_promoted_away",
            "matchweek": "matchweek",
            "home_form_points_5": "home_form_points_5",
            "away_form_points_5": "away_form_points_5",
            "home_rest_days": "home_rest_days",
            "away_rest_days": "away_rest_days",
        }

        available = {src: tgt for src, tgt in context_cols_map.items()
                     if src in features_df.columns}
        if not available:
            for col, default in CONTEXT_DEFAULTS.items():
                if col not in df.columns:
                    df[col] = default
            return df

        # Build join key from features_df
        feat_ctx = features_df[["home_team", "away_team", "match_date"] +
                               list(available.keys())].copy()
        feat_ctx["match_date"] = pd.to_datetime(feat_ctx["match_date"])
        # Deduplicate to avoid row multiplication during merge
        feat_ctx = feat_ctx.drop_duplicates(subset=["home_team", "match_date"], keep="last")

        # Join on (home_team, match_date)
        df["match_date"] = pd.to_datetime(df["match_date"])
        merged = df.merge(
            feat_ctx, on=["home_team", "match_date"], how="left", suffixes=("", "_feat")
        )

        # Compute derived context features
        if "home_elo" in merged.columns and "away_elo" in merged.columns:
            merged["elo_gap"] = merged["home_elo"].fillna(1500) - merged["away_elo"].fillna(1500)
        else:
            merged["elo_gap"] = 0.0

        if "home_is_promoted" in merged.columns:
            merged["is_promoted_home"] = merged["home_is_promoted"].fillna(0)
        if "away_is_promoted" in merged.columns:
            merged["is_promoted_away"] = merged["away_is_promoted"].fillna(0)

        if "home_form_points_5" in merged.columns and "away_form_points_5" in merged.columns:
            merged["form_gap"] = (merged["home_form_points_5"].fillna(0) -
                                  merged["away_form_points_5"].fillna(0))
        else:
            merged["form_gap"] = 0.0

        if "home_rest_days" in merged.columns and "away_rest_days" in merged.columns:
            merged["rest_gap"] = (merged["home_rest_days"].fillna(0) -
                                  merged["away_rest_days"].fillna(0))
        else:
            merged["rest_gap"] = 0.0

        if "matchweek" not in merged.columns:
            merged["matchweek"] = 19.0

        # Fill remaining defaults
        for col, default in CONTEXT_DEFAULTS.items():
            if col not in merged.columns:
                merged[col] = default

        return merged

    @staticmethod
    def _compute_ece(y: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
        """Expected Calibration Error — measures how well confidence matches accuracy."""
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        accuracies = (predictions == y).astype(float)

        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (confidences > lo) & (confidences <= hi)
            if mask.sum() == 0:
                continue
            avg_conf = confidences[mask].mean()
            avg_acc = accuracies[mask].mean()
            ece += mask.sum() / len(y) * abs(avg_acc - avg_conf)
        return ece


# ---------------------------------------------------------------------------
# Rolling Corrector
# ---------------------------------------------------------------------------

class RollingCorrector:
    """EMA-based correction that adapts from recent prediction errors.

    Buckets predictions by (predicted_class, confidence_band) and tracks an
    exponential moving average of errors (actual_onehot - predicted_probs).
    After settlement, call update() to learn from mistakes. At prediction time,
    call correct() to apply additive EMA corrections.

    WHY EMA: Recent errors matter more than old ones. A decay of 0.95 means
    the last 20 matches have ~64% of the weight. This catches temporal drift
    (e.g., "draws are happening more this season") without overfitting to noise.
    """

    def __init__(self, config: CorrectionConfig | None = None):
        self.config = config or CorrectionConfig()
        # State: {bucket_key: {"ema": [H_err, D_err, A_err], "n": count}}
        self.buckets: Dict[str, Dict[str, Any]] = {}
        # Watermark: tracks how many ledger entries have been processed
        # to avoid replaying already-incorporated entries on next settlement
        self.processed_count: int = 0

    def _get_bucket_key(self, prob_H: float, prob_D: float, prob_A: float) -> str:
        """Map a prediction to its bucket key: '{class}_{conf_lo}_{conf_hi}'."""
        probs = [prob_H, prob_D, prob_A]
        pred_class = ["H", "D", "A"][np.argmax(probs)]
        confidence = max(probs)

        for lo, hi in self.config.rolling_confidence_bands:
            if lo <= confidence < hi or (hi == 1.0 and confidence == 1.0):
                return f"{pred_class}_{lo:.2f}_{hi:.2f}"
        # Fallback — shouldn't happen
        return f"{pred_class}_0.00_1.00"

    def update(
        self,
        prob_H: float,
        prob_D: float,
        prob_A: float,
        actual: str,
        date: str | None = None,
    ) -> None:
        """Update rolling EMA after a match settles.

        Args:
            prob_H/prob_D/prob_A: predicted probabilities
            actual: actual outcome "H"/"D"/"A"
            date: match date (for logging only)
        """
        actual_onehot = {"H": [1, 0, 0], "D": [0, 1, 0], "A": [0, 0, 1]}.get(actual)
        if actual_onehot is None:
            return

        key = self._get_bucket_key(prob_H, prob_D, prob_A)
        predicted = np.array([prob_H, prob_D, prob_A])
        error = np.array(actual_onehot, dtype=float) - predicted
        decay = self.config.rolling_decay

        if key not in self.buckets:
            self.buckets[key] = {"ema": error.tolist(), "n": 1}
        else:
            bucket = self.buckets[key]
            old_ema = np.array(bucket["ema"])
            bucket["ema"] = (decay * old_ema + (1 - decay) * error).tolist()
            bucket["n"] = bucket["n"] + 1

    def correct(
        self,
        prob_H: float,
        prob_D: float,
        prob_A: float,
    ) -> Tuple[float, float, float]:
        """Apply rolling correction to a prediction.

        If the bucket has < min_samples, passthrough (no correction).
        """
        key = self._get_bucket_key(prob_H, prob_D, prob_A)
        bucket = self.buckets.get(key)

        if bucket is None or bucket["n"] < self.config.rolling_min_samples:
            return prob_H, prob_D, prob_A

        ema = np.array(bucket["ema"])
        corrected = np.array([prob_H, prob_D, prob_A]) + ema

        # Clamp to valid probability range, then renormalize
        corrected = np.clip(corrected, 0.01, 0.99)
        corrected = corrected / corrected.sum()

        return float(corrected[0]), float(corrected[1]), float(corrected[2])

    def save(self, path: Path | None = None) -> None:
        """Save rolling state to JSON."""
        path = path or ROLLING_STATE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "buckets": self.buckets,
                "processed_count": self.processed_count,
                "updated_at": datetime.now().isoformat(),
            }, f, indent=2)
        log.info("Saved rolling calibration state (%d buckets, watermark=%d) to %s",
                 len(self.buckets), self.processed_count, path)

    def load(self, path: Path | None = None) -> bool:
        """Load rolling state from JSON. Returns True if loaded."""
        path = path or ROLLING_STATE_PATH
        if not path.exists():
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            self.buckets = data.get("buckets", {})
            self.processed_count = data.get("processed_count", 0)
            log.info("Loaded rolling calibration state (%d buckets, watermark=%d)",
                     len(self.buckets), self.processed_count)
            return True
        except (json.JSONDecodeError, IOError) as e:
            log.warning("Failed to load rolling state: %s", e)
            return False


# ---------------------------------------------------------------------------
# Unified Correction Layer
# ---------------------------------------------------------------------------

class CorrectionLayer:
    """Combines static + rolling correction into a single interface.

    Usage:
        # Training (in ml/training.py):
        layer = CorrectionLayer()
        metrics = layer.train_static(oof_df, features_df)
        layer.save()

        # Inference (in ensemble_prediction_engine.py):
        layer = CorrectionLayer()
        layer.load()
        corrected = layer.correct(prob_H, prob_D, prob_A, context)

        # Settlement (in scheduler.py):
        layer = CorrectionLayer()
        layer.load()
        layer.update_rolling(prob_H, prob_D, prob_A, actual, date)
        layer.save_rolling()
    """

    def __init__(self, config: CorrectionConfig | None = None,
                 league: str | None = None):
        self.config = config or CorrectionConfig()
        self._variant = league or "universal"
        self._model_path = MODELS_DIR / self._variant / "correction_layer.pkl"
        self._rolling_path = MODELS_DIR / self._variant / "rolling_calibration.json"
        self.static = StaticCorrector(self.config)
        self.rolling = RollingCorrector(self.config)
        self.active = False

    def train_static(
        self,
        oof_df: pd.DataFrame,
        features_df: pd.DataFrame | None = None,
    ) -> Dict[str, float]:
        """Train the static corrector. See StaticCorrector.fit()."""
        if not self.config.static_enabled:
            log.info("Static corrector disabled (config.static_enabled=False)")
            return {"status": "disabled"}
        metrics = self.static.fit(oof_df, features_df)
        self.active = self.static.is_fitted
        return metrics

    def correct(
        self,
        prob_H: float,
        prob_D: float,
        prob_A: float,
        context: Dict[str, float] | None = None,
    ) -> Dict[str, Any]:
        """Apply correction: static first, then rolling.

        Returns dict with corrected probs and deltas for transparency.
        """
        orig = (prob_H, prob_D, prob_A)

        # Static correction (structural biases) — disabled by default (ECE regression)
        if self.config.static_enabled and self.static.is_fitted and context is not None:
            prob_H, prob_D, prob_A = self.static.correct(prob_H, prob_D, prob_A, context)

        after_static = (prob_H, prob_D, prob_A)

        # Rolling correction (temporal drift)
        prob_H, prob_D, prob_A = self.rolling.correct(prob_H, prob_D, prob_A)

        return {
            "prob_H": prob_H,
            "prob_D": prob_D,
            "prob_A": prob_A,
            "delta_H": round(prob_H - orig[0], 4),
            "delta_D": round(prob_D - orig[1], 4),
            "delta_A": round(prob_A - orig[2], 4),
            "static_applied": self.static.is_fitted,
            "rolling_applied": (prob_H, prob_D, prob_A) != after_static,
        }

    def update_rolling(
        self,
        prob_H: float,
        prob_D: float,
        prob_A: float,
        actual: str,
        date: str | None = None,
    ) -> None:
        """Update rolling EMA after settlement."""
        self.rolling.update(prob_H, prob_D, prob_A, actual, date)

    def save(self, model_path: Path | None = None, rolling_path: Path | None = None) -> None:
        """Save both static model and rolling state."""
        model_path = model_path or self._model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)

        if self.static.is_fitted:
            with open(model_path, "wb") as f:
                pickle.dump({
                    "model": self.static.model,
                    "scaler": self.static.scaler,
                    "train_metrics": self.static.train_metrics,
                    # feature_cols saved explicitly for ordering safety —
                    # scaler.transform MUST receive features in this exact order
                    "feature_cols": self.static.train_metrics.get(
                        "feature_cols", CONTEXT_FEATURE_NAMES),
                    "config": self.config,
                }, f)
            log.info("Saved static correction model to %s", model_path)

        self.rolling.save(rolling_path or self._rolling_path)

    def save_rolling(self, path: Path | None = None) -> None:
        """Save only rolling state (after settlement updates)."""
        self.rolling.save(path or self._rolling_path)

    def load(self, model_path: Path | None = None, rolling_path: Path | None = None) -> bool:
        """Load static model + rolling state. Graceful fallback — missing = passthrough."""
        model_path = model_path or self._model_path
        loaded_any = False

        if model_path.exists():
            try:
                with open(model_path, "rb") as f:
                    data = pickle.load(f)
                self.static.model = data["model"]
                self.static.scaler = data["scaler"]
                self.static.train_metrics = data.get("train_metrics", {})
                # Restore feature ordering (top-level key takes priority)
                if "feature_cols" in data:
                    self.static.train_metrics["feature_cols"] = data["feature_cols"]
                self.static.is_fitted = True
                loaded_any = True
                log.info("Loaded static correction model (trained on %d samples)",
                         self.static.train_metrics.get("n_samples", 0))
            except Exception as e:
                log.warning("Failed to load static correction model: %s", e)

        if self.rolling.load(rolling_path or self._rolling_path):
            loaded_any = True

        self.active = loaded_any
        return loaded_any


# ---------------------------------------------------------------------------
# Prediction Ledger (append-only log for rolling updates)
# ---------------------------------------------------------------------------

def append_to_ledger(
    home_team: str,
    away_team: str,
    match_date: str,
    prob_H: float,
    prob_D: float,
    prob_A: float,
    predicted: str,
    confidence: float,
    correction_deltas: Dict[str, float] | None = None,
) -> None:
    """Append a prediction to the ledger for later settlement matching."""
    PREDICTION_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "home_team": home_team,
        "away_team": away_team,
        "match_date": match_date,
        "prob_H": round(prob_H, 4),
        "prob_D": round(prob_D, 4),
        "prob_A": round(prob_A, 4),
        "predicted": predicted,
        "confidence": round(confidence, 4),
        "timestamp": datetime.now().isoformat(),
    }
    if correction_deltas:
        entry["correction_deltas"] = correction_deltas

    with open(PREDICTION_LEDGER_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def read_ledger() -> List[Dict]:
    """Read all entries from the prediction ledger."""
    if not PREDICTION_LEDGER_PATH.exists():
        return []
    entries = []
    with open(PREDICTION_LEDGER_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Convert to float, replacing NaN/None with default."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except (TypeError, ValueError):
        return default


def extract_context_features(
    prob_H: float,
    prob_D: float,
    prob_A: float,
    features_dict: Dict[str, Any],
) -> Dict[str, float]:
    """Build context feature dict for the correction layer from prediction + features."""
    probs = [prob_H, prob_D, prob_A]
    home_elo = _safe_float(features_dict.get("home_elo"), 1500)
    away_elo = _safe_float(features_dict.get("away_elo"), 1500)

    return {
        "prob_H": prob_H,
        "prob_D": prob_D,
        "prob_A": prob_A,
        "confidence": max(probs),
        "predicted_class": int(np.argmax(probs)),
        "elo_gap": home_elo - away_elo,
        "is_promoted_home": _safe_float(features_dict.get("home_is_promoted"), 0),
        "is_promoted_away": _safe_float(features_dict.get("away_is_promoted"), 0),
        "matchweek": _safe_float(features_dict.get("matchweek"), 19),
        "form_gap": (_safe_float(features_dict.get("home_form_points_5"), 0) -
                     _safe_float(features_dict.get("away_form_points_5"), 0)),
        "rest_gap": (_safe_float(features_dict.get("home_rest_days"), 0) -
                     _safe_float(features_dict.get("away_rest_days"), 0)),
    }
