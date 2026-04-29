"""Walk-forward CV core library — single source of truth for season-based
walk-forward training and evaluation.

Replaces the duplicate logic in:
  - scripts/models/train_walkforward.py (inline season-loop)
  - scripts/models/retrain_no_odds_catboost.py (TimeSeriesSplitter)
  - ml/walk_forward.py (legacy expanding-window)

This module is a PURE LIBRARY:
  - No file I/O beyond the data the caller passes in.
  - Caller owns reading parquets, writing models, deployment state, etc.
  - Reusable across diagnostic and production trainers.

Status: IMPLEMENTED (2026-04-28). Step 1 of the 5-step migration plan from
docs/2026-04-28_trainer_consolidation_analysis.md. Step 2 (migrate
train_walkforward.py to use this core) is the next session. Step 3+
(migrate production retrain) is multi-session work with paper-trade
validation.

Usage:

    from ml.walkforward_core import (
        WalkForwardConfig, run_walkforward,
    )

    config = WalkForwardConfig(
        eval_seasons=["2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"],
        seeds=[42, 43, 44],
        fit_calibrator=True,
        kind="multiclass",
        n_classes=3,
    )

    report = run_walkforward(
        df=features_df,
        target_builder=lambda d: 1X2_target_series,
        feature_selector=lambda d: feature_list,
        config=config,
    )

    for fold in report.folds:
        print(fold.eval_season, fold.seed, fold.cal_accuracy)
    print("aggregate:", report.aggregate)
"""

from __future__ import annotations

import io
import logging
import pickle
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.isotonic import IsotonicRegression

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardConfig:
    """Configuration for a walk-forward training run.

    Attributes
    ----------
    eval_seasons:
        Seasons to evaluate on. Each becomes one fold. Train data for fold S
        is everything strictly before S, optionally floored by min_train_season.
    seeds:
        Random seeds to iterate over. Mean and stdev across seeds reported in aggregate.
    min_train_season:
        Earliest season allowed in training pool. Empty = use all available.
    min_prior_seasons:
        Minimum number of prior seasons required to attempt a fold.
    fit_calibrator:
        If True, fit per-class isotonic calibrator on the held-out cal-val pool.
    leakage_corr_threshold:
        Refuse to train if any feature has |corr| > this with any plausible target.
    early_stopping_rounds:
        CatBoost early stopping rounds.
    kind:
        "binary" or "multiclass" — determines CatBoost loss function.
    n_classes:
        Number of target classes (2 for binary, 3 for 1X2).
    class_weights:
        Per-class loss weights, e.g. (1.0, 1.5, 1.0) for 1X2 to up-weight draws.
    cal_val_fraction:
        Fraction of the train pool reserved for early-stopping val + calibration.
    iterations / learning_rate / depth / l2_leaf_reg / min_data_in_leaf:
        CatBoost hyperparameters. Defaults match train_walkforward.py.
    """
    eval_seasons: list[str]
    seeds: list[int] = field(default_factory=lambda: [42])
    min_train_season: Optional[str] = None
    min_prior_seasons: int = 5
    fit_calibrator: bool = False
    leakage_corr_threshold: float = 0.5
    early_stopping_rounds: int = 150
    kind: str = "multiclass"
    n_classes: int = 3
    class_weights: Optional[tuple[float, ...]] = None
    cal_val_fraction: float = 0.15
    iterations: int = 2000
    learning_rate: float = 0.02
    depth: int = 6
    l2_leaf_reg: float = 3.0
    min_data_in_leaf: int = 20
    # Post-calibration draw-class boost (architectural fix for draw recall).
    # Multiplied into the calibrated draw probability before renormalization.
    # 0.0 = no boost (original behavior). 0.30 = 30% draw bias.
    draw_boost: float = 0.0


# ---------------------------------------------------------------------------
# Per-fold result
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    eval_season: str
    seed: int
    raw_logloss: float
    raw_accuracy: float
    cal_logloss: Optional[float]
    cal_accuracy: Optional[float]
    ece: float
    brier: float
    n_train: int
    n_eval: int
    feature_count: int
    model_bytes: bytes = b""
    calibrator: Optional[dict] = None
    per_class_accuracy: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardReport:
    folds: list[FoldResult]
    config: WalkForwardConfig
    aggregate: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def detect_leakage(
    df: pd.DataFrame,
    feature_names: list[str],
    threshold: float = 0.5,
) -> list[tuple[str, str, float]]:
    """Scan features for high correlation with plausible 1X2 targets.

    Returns list of (feature_name, target_name, correlation) tuples for
    features that exceed the threshold. Empty list = no leakage detected.

    Targets checked: over_2_5, home_win, draw, btts. Threshold 0.5 catches
    direct leakage (>0.5) while letting through natural quality→outcome
    signals like squad_value_diff (~0.4).
    """
    if not {"home_score", "away_score"}.issubset(df.columns):
        return []
    targets = {
        "over_2_5": ((df["home_score"] + df["away_score"]) >= 3).astype(int),
        "home_win": (df["home_score"] > df["away_score"]).astype(int),
        "draw":     (df["home_score"] == df["away_score"]).astype(int),
        "btts":     ((df["home_score"] > 0) & (df["away_score"] > 0)).astype(int),
    }
    offenders: list[tuple[str, str, float]] = []
    for f in feature_names:
        if f not in df.columns:
            continue
        col = df[f]
        if col.notna().sum() < 100:
            continue
        for tname, y in targets.items():
            try:
                c = float(col.corr(y))
            except Exception:
                continue
            if not np.isnan(c) and abs(c) > threshold:
                offenders.append((f, tname, c))
    offenders.sort(key=lambda x: -abs(x[2]))
    return offenders


def fit_isotonic_calibrator(
    proba: np.ndarray,
    y_true: np.ndarray,
    classes: tuple[str, ...],
) -> dict:
    """Fit per-class isotonic regression calibrator on held-out fold predictions.

    Used when `fit_calibrator=True` in the config. Returns a dict that the
    caller can pickle and apply at inference time.

    Each class gets its own IsotonicRegression mapping raw_proba(class) →
    calibrated_proba(class). At inference, calibrated probabilities are
    re-normalized to sum to 1.
    """
    n_classes = len(classes)
    if proba.ndim == 1:
        # binary case — promote to 2-column
        proba = np.column_stack([1 - proba, proba])
    if proba.shape[1] != n_classes:
        raise ValueError(
            f"Proba shape {proba.shape} doesn't match n_classes={n_classes}"
        )
    # y_true comes in as string labels; convert to one-hot for fitting per-class isotonic
    calibrators: dict[str, IsotonicRegression] = {}
    for i, cls in enumerate(classes):
        y_cls = (y_true == cls).astype(int)
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(proba[:, i], y_cls)
        calibrators[cls] = ir
    return {
        "kind": "per_class_isotonic",
        "classes": list(classes),
        "calibrators": calibrators,
    }


def apply_calibrator(
    proba: np.ndarray,
    calibrator: dict,
    draw_boost: float = 0.0,
) -> np.ndarray:
    """Apply a calibrator dict (from fit_isotonic_calibrator) to raw probabilities.

    Output is normalized so each row sums to 1.

    Optional `draw_boost`: if > 0 and the calibrator includes a "D" class,
    multiply the draw-class calibrated probability by (1 + draw_boost) BEFORE
    renormalizing. This is the architectural fix for the structural draw-class
    underprediction bug — class_weights at training time get cancelled by
    isotonic calibration, so we apply the bias post-calibration instead.
    Recommended values: 0.0 (disabled) or 0.20-0.40 (moderate bias).
    """
    classes = calibrator["classes"]
    cal = calibrator["calibrators"]
    if proba.ndim == 1:
        proba = np.column_stack([1 - proba, proba])
    out = np.zeros_like(proba, dtype=float)
    for i, cls in enumerate(classes):
        out[:, i] = cal[cls].transform(proba[:, i])
    # Optional post-calibration draw boost (architectural fix for draw recall)
    if draw_boost > 0.0 and "D" in classes:
        d_idx = classes.index("D")
        out[:, d_idx] = out[:, d_idx] * (1.0 + draw_boost)
    # Re-normalize
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return out / row_sums


def _ece_multiclass(y_true: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    """Expected Calibration Error for multiclass predictions."""
    if proba.ndim == 1:
        proba = np.column_stack([1 - proba, proba])
    confidences = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)
    bin_edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        in_bin = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if i == bins - 1:
            in_bin = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if in_bin.sum() == 0:
            continue
        bin_acc = accuracies[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (in_bin.sum() / len(y_true)) * abs(bin_acc - bin_conf)
    return float(ece)


def _brier_multiclass(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    """Multiclass Brier score: mean squared error of predicted probabilities vs one-hot truth."""
    one_hot = np.zeros((len(y_true), n_classes))
    for i in range(n_classes):
        one_hot[:, i] = (y_true == i).astype(float)
    return float(np.mean((proba - one_hot) ** 2))


def _logloss(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    """Multiclass log-loss with clipping for numerical stability."""
    eps = 1e-15
    proba_clipped = np.clip(proba, eps, 1 - eps)
    if n_classes == 2:
        if proba_clipped.ndim == 2:
            p = proba_clipped[:, 1]
        else:
            p = proba_clipped
        return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
    # Multiclass
    return float(-np.mean(np.log(proba_clipped[np.arange(len(y_true)), y_true])))


def _fit_one_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: WalkForwardConfig,
    seed: int,
) -> CatBoostClassifier:
    """Train one CatBoost fold with early stopping."""
    params = {
        "iterations": config.iterations,
        "learning_rate": config.learning_rate,
        "depth": config.depth,
        "l2_leaf_reg": config.l2_leaf_reg,
        "min_data_in_leaf": config.min_data_in_leaf,
        "loss_function": "Logloss" if config.kind == "binary" else "MultiClass",
        "early_stopping_rounds": config.early_stopping_rounds,
        "verbose": 0,
        "random_seed": seed,
        "allow_writing_files": False,
    }
    if config.kind == "multiclass":
        params["classes_count"] = config.n_classes
    if config.class_weights is not None:
        params["class_weights"] = list(config.class_weights)
    model = CatBoostClassifier(**params)
    train_pool = Pool(X_train, label=y_train)
    val_pool = Pool(X_val, label=y_val)
    model.fit(train_pool, eval_set=val_pool)
    return model


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_walkforward(
    df: pd.DataFrame,
    target_builder: Callable[[pd.DataFrame], pd.Series],
    feature_selector: Callable[[pd.DataFrame], list[str]],
    config: WalkForwardConfig,
) -> WalkForwardReport:
    """Run walk-forward CV with strict no-leakage train/eval splits.

    For each (eval_season × seed) pair:
      1. Train data = all rows in seasons strictly before eval_season,
         optionally floored by config.min_train_season.
      2. Reserve the last cal_val_fraction of train (chronological) as
         val pool — used for early stopping AND calibrator fitting.
      3. Train CatBoost classifier on the inner train pool.
      4. Evaluate on eval_season rows.
      5. If fit_calibrator: fit isotonic calibrator on the val pool's
         out-of-fold predictions, apply to eval-season predictions.

    Returns WalkForwardReport with FoldResult per (season, seed) and
    an aggregate summary across all folds.

    The caller is responsible for: filtering df by league, dropping rows
    with missing scores (df must have home_score/away_score), and writing
    the per-fold model_bytes/calibrator to disk if desired.
    """
    df = df.copy()
    df = df.sort_values(["season", "match_date"]).reset_index(drop=True)
    df["_target"] = target_builder(df).values

    feature_names = feature_selector(df)

    # Leakage safety net
    if config.leakage_corr_threshold > 0:
        offenders = detect_leakage(df, feature_names, threshold=config.leakage_corr_threshold)
        if offenders:
            msg = "\n".join(
                f"  {f} (corr with {t} = {c:+.3f})" for f, t, c in offenders[:20]
            )
            raise RuntimeError(
                f"Refusing to train — {len(offenders)} feature(s) exceed "
                f"|corr|>{config.leakage_corr_threshold} with a plausible target. "
                f"Add them to your feature_selector's exclusion list or fix the pipeline.\n{msg}"
            )

    all_seasons_sorted = sorted(df["season"].unique())
    folds: list[FoldResult] = []
    classes_str = ("H", "D", "A") if config.kind == "multiclass" else ("0", "1")

    for eval_season in config.eval_seasons:
        prior_seasons = [s for s in all_seasons_sorted if s < eval_season]
        if config.min_train_season:
            prior_seasons = [s for s in prior_seasons if s >= config.min_train_season]
        if len(prior_seasons) < config.min_prior_seasons:
            log.warning(
                "Skipping fold %s — only %d prior seasons (need %d)",
                eval_season, len(prior_seasons), config.min_prior_seasons,
            )
            continue

        train_mask = df["season"].isin(prior_seasons)
        eval_mask = df["season"] == eval_season

        train_df = df.loc[train_mask].copy()
        eval_df = df.loc[eval_mask].copy()

        # Chronological cal/val split on training pool
        train_df = train_df.sort_values("match_date").reset_index(drop=True)
        cal_size = max(1, int(len(train_df) * config.cal_val_fraction))
        inner_train = train_df.iloc[:-cal_size]
        cal_val = train_df.iloc[-cal_size:]

        # CatBoost multiclass requires integer-encoded targets internally,
        # but our target_builder may emit string labels (e.g. 'H'/'D'/'A'
        # for 1X2). Build a stable string→int mapping based on alphabetical
        # order of unique values, so the predict_proba column order is
        # deterministic across folds. Caller-built target can be string or int.
        if config.kind == "multiclass" and inner_train["_target"].dtype == object:
            unique_classes = sorted(df["_target"].astype(str).unique())
            class_to_int = {c: i for i, c in enumerate(unique_classes)}
            int_to_class = {i: c for c, i in class_to_int.items()}
            inner_y = inner_train["_target"].astype(str).map(class_to_int).astype(int)
            cal_y = cal_val["_target"].astype(str).map(class_to_int).astype(int)
            eval_y = eval_df["_target"].astype(str).map(class_to_int).astype(int)
        else:
            unique_classes = None
            int_to_class = None
            inner_y = inner_train["_target"]
            cal_y = cal_val["_target"]
            eval_y = eval_df["_target"]

        X_inner = inner_train[feature_names]
        y_inner = inner_y
        X_cal = cal_val[feature_names]
        y_cal = cal_y
        X_eval = eval_df[feature_names]
        y_eval = eval_y

        for seed in config.seeds:
            model = _fit_one_fold(X_inner, y_inner, X_cal, y_cal, config, seed)
            proba_eval_raw = model.predict_proba(X_eval)
            proba_cal_pool_raw = model.predict_proba(X_cal)

            # Calibration
            calibrator_dict: Optional[dict] = None
            proba_eval_cal: Optional[np.ndarray] = None
            if config.fit_calibrator:
                if config.kind == "multiclass" and int_to_class is not None:
                    cls_tuple = tuple(unique_classes)
                    # Map y_cal back to string labels for the calibrator
                    y_cal_str = np.array([int_to_class[i] for i in y_cal.to_numpy()])
                else:
                    cls_tuple = ("0", "1")
                    y_cal_str = y_cal.astype(str).to_numpy()
                calibrator_dict = fit_isotonic_calibrator(
                    proba_cal_pool_raw, y_cal_str, cls_tuple
                )
                proba_eval_cal = apply_calibrator(
                    proba_eval_raw, calibrator_dict, draw_boost=config.draw_boost
                )

            # Metrics
            y_eval_arr = y_eval.to_numpy()
            if config.kind == "multiclass":
                # y_eval is already int-encoded (from class_to_int map at top of fold).
                # CatBoost predict_proba returns columns in the order of model.classes_,
                # which is the int order our class_to_int produced (0, 1, 2 ...).
                y_eval_int = y_eval_arr.astype(int)
                raw_ll = _logloss(y_eval_int, proba_eval_raw, config.n_classes)
                raw_pred = proba_eval_raw.argmax(axis=1)
                raw_acc = float((raw_pred == y_eval_int).mean())
                raw_ece = _ece_multiclass(y_eval_int, proba_eval_raw)
                raw_brier = _brier_multiclass(y_eval_int, proba_eval_raw, config.n_classes)
                # Per-class recall — compute on CAL predictions when calibrator
                # is fit (since that's the production prediction path), else on raw.
                pca: dict[str, float] = {}
                model_classes = list(model.classes_)
                pca_pred = (
                    proba_eval_cal.argmax(axis=1) if proba_eval_cal is not None else raw_pred
                )
                for i, cls in enumerate(model_classes):
                    label = int_to_class.get(int(cls), str(cls)) if int_to_class else str(cls)
                    cls_mask = (y_eval_int == int(cls))
                    if cls_mask.sum() > 0:
                        pca[label] = float((pca_pred[cls_mask] == int(cls)).mean())
                cal_ll = cal_acc = None
                if proba_eval_cal is not None:
                    cal_ll = _logloss(y_eval_int, proba_eval_cal, config.n_classes)
                    cal_pred = proba_eval_cal.argmax(axis=1)
                    cal_acc = float((cal_pred == y_eval_int).mean())
            else:
                # binary case
                p_pos = proba_eval_raw[:, 1] if proba_eval_raw.ndim == 2 else proba_eval_raw
                raw_ll = _logloss(y_eval_arr.astype(int), p_pos, 2)
                raw_pred = (p_pos >= 0.5).astype(int)
                raw_acc = float((raw_pred == y_eval_arr.astype(int)).mean())
                raw_ece = _ece_multiclass(y_eval_arr.astype(int), proba_eval_raw, bins=10)
                raw_brier = float(np.mean((p_pos - y_eval_arr.astype(int)) ** 2))
                pca = {}
                cal_ll = cal_acc = None
                if proba_eval_cal is not None:
                    p_pos_cal = proba_eval_cal[:, 1] if proba_eval_cal.ndim == 2 else proba_eval_cal
                    cal_ll = _logloss(y_eval_arr.astype(int), p_pos_cal, 2)
                    cal_pred = (p_pos_cal >= 0.5).astype(int)
                    cal_acc = float((cal_pred == y_eval_arr.astype(int)).mean())

            # Serialize model — CatBoost only accepts file paths, so write to
            # a temp file then read bytes back.
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                model.save_model(tmp_path, format="cbm")
                with open(tmp_path, "rb") as f:
                    model_bytes = f.read()
            finally:
                import os
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            calibrator_pickled: Optional[dict] = None
            if calibrator_dict is not None:
                calibrator_pickled = {
                    "kind": calibrator_dict["kind"],
                    "classes": calibrator_dict["classes"],
                    "calibrators_pickle": pickle.dumps(calibrator_dict["calibrators"]),
                }

            folds.append(FoldResult(
                eval_season=eval_season,
                seed=seed,
                raw_logloss=raw_ll,
                raw_accuracy=raw_acc,
                cal_logloss=cal_ll,
                cal_accuracy=cal_acc,
                ece=raw_ece,
                brier=raw_brier,
                n_train=len(inner_train),
                n_eval=len(eval_df),
                feature_count=len(feature_names),
                model_bytes=model_bytes,
                calibrator=calibrator_pickled,
                per_class_accuracy=pca,
            ))
            log.info(
                "  Fold %s seed=%d: raw_ll=%.4f raw_acc=%.4f%s",
                eval_season, seed, raw_ll, raw_acc,
                f" | cal_ll={cal_ll:.4f} cal_acc={cal_acc:.4f}" if cal_ll is not None else "",
            )

    # Aggregate
    aggregate: dict = {}
    if folds:
        accs = [f.cal_accuracy if f.cal_accuracy is not None else f.raw_accuracy for f in folds]
        lls = [f.cal_logloss if f.cal_logloss is not None else f.raw_logloss for f in folds]
        eces = [f.ece for f in folds]
        aggregate = {
            "n_folds": len(folds),
            "mean_accuracy": float(np.mean(accs)),
            "stdev_accuracy": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "mean_log_loss": float(np.mean(lls)),
            "stdev_log_loss": float(np.std(lls, ddof=1)) if len(lls) > 1 else 0.0,
            "mean_ece": float(np.mean(eces)),
            "total_n_eval": sum(f.n_eval for f in folds),
        }

    return WalkForwardReport(folds=folds, config=config, aggregate=aggregate)
