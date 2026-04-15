"""Feature selection: importance-based pruning and odds-excluded variants."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config.settings import MODELS_DIR
from ml.config import (
    LABEL_MAP,
    META_COLS,
    ODDS_COLUMN_PATTERNS,
    ODDS_META_KEEP,
    FeatureConfig,
    ValidationConfig,
    drop_meta,
)
from ml.data import TimeSeriesSplitter

log = logging.getLogger(__name__)


def identify_odds_columns(feature_names: list[str]) -> list[str]:
    """Return feature names that are derived from betting odds."""
    return [
        f for f in feature_names
        if any(f.startswith(p) or f.endswith(p.rstrip("_")) for p in ODDS_COLUMN_PATTERNS)
    ]


def exclude_odds(
    X: pd.DataFrame, feature_names: list[str],
) -> Tuple[pd.DataFrame, list[str]]:
    """Remove raw odds features but keep disagreement/meta features.

    Raw odds (PSH, B365H, pinnacle_home_prob) make the ML model redundant
    with the market predictor. But disagreement features (where Elo disagrees
    with market, where sharp money disagrees with soft) capture WHERE the
    market might be wrong — signal that linear ensemble blending can't express.

    Features in ODDS_META_KEEP are preserved. All other odds-derived features
    are removed.
    """
    odds_cols = identify_odds_columns(feature_names)
    # Keep disagreement/meta features even though they match odds patterns
    actually_excluded = [f for f in odds_cols if f not in ODDS_META_KEEP]
    kept_meta = [f for f in odds_cols if f in ODDS_META_KEEP]
    keep = [f for f in feature_names if f not in set(actually_excluded)]
    # Keep metadata columns (avoid duplicating _has_* already in keep)
    keep_set = set(keep)
    all_keep = keep + [c for c in X.columns if c.startswith("_") and c not in keep_set]
    X_out = X[[c for c in all_keep if c in X.columns]].copy()
    log.info("Excluded %d raw odds features, kept %d meta features (%s), %d total remaining",
             len(actually_excluded), len(kept_meta),
             ", ".join(sorted(kept_meta)) if kept_meta else "none",
             len(keep))
    return X_out, keep


def importance_based_selection(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    model_type: str = "xgboost",
    top_k: int | None = None,
    threshold: float | None = None,
    params: dict | None = None,
) -> Tuple[list[str], Dict[str, float]]:
    """Select features by training on all data and ranking by importance.

    Uses recency-weighted walk-forward importance to avoid both lookahead
    bias AND the fold-0 bias that penalizes modern features (KB#37).

    Modern features (Sofascore 2022+, FBref 2017+) are imputed to 0.0 in
    early folds, making them appear uninformative. To fix this:
    1. Exponential recency weighting: recent folds count much more
    2. Supplementary recent-folds pass: catches features invisible to
       early folds entirely

    Returns (selected_feature_names, importance_dict).
    """
    from ml.tuning import _build_model, _compute_sample_weights, _fit_with_early_stopping

    feat_cfg = FeatureConfig()
    if threshold is None:
        threshold = feat_cfg.importance_threshold

    config = ValidationConfig()
    splitter = TimeSeriesSplitter(config)
    splits = splitter.generate_splits(X["_season"])
    n_folds = len(splits)

    y_int = y.map(LABEL_MAP)

    # Accumulate (importance, weight) pairs per feature
    importance_accum: Dict[str, List[Tuple[float, float]]] = {f: [] for f in feature_names}

    # Compute recency weights: weight = base ^ (n_folds - 1 - fold_idx)
    # Last fold gets weight=1.0, earlier folds get exponentially less
    recency_base = feat_cfg.recency_weight_base
    fold_weights = [
        recency_base ** (n_folds - 1 - i) if recency_base > 1.0 else 1.0
        for i in range(n_folds)
    ]

    for fold_idx, (train_seasons, _) in enumerate(splits):
        train_mask = X["_season"].isin(train_seasons)
        X_tr = drop_meta(X[train_mask])[feature_names]
        y_tr = y_int[train_mask]

        # Per-feature coverage: fraction of non-NaN values in this fold
        coverage = X_tr.notna().mean()

        if params is None:
            from ml.config import ModelConfig
            cfg = ModelConfig()
            p = cfg.xgb_params if model_type == "xgboost" else cfg.lgb_params
        else:
            p = params

        model = _build_model(model_type, p)
        train_seasons_s = X[train_mask]["_season"] if "_season" in X.columns else None
        sw = _compute_sample_weights(y[train_mask], seasons=train_seasons_s)
        _fit_with_early_stopping(model, X_tr, y_tr, model_type, sample_weight=sw)

        w = fold_weights[fold_idx]
        imp = model.feature_importances_
        for feat, score in zip(feature_names, imp):
            if coverage.get(feat, 0) > 0.10:
                importance_accum[feat].append((score, w))

    # Weighted average importance across folds
    avg_importance: Dict[str, float] = {}
    for feat, pairs in importance_accum.items():
        if not pairs:
            avg_importance[feat] = float("nan")
            continue
        scores, weights = zip(*pairs)
        total_w = sum(weights)
        if total_w > 0:
            avg_importance[feat] = float(sum(s * w for s, w in pairs) / total_w)
        else:
            avg_importance[feat] = float("nan")

    # Sort by importance (NaN sorts to bottom)
    sorted_feats = sorted(
        avg_importance.items(),
        key=lambda x: x[1] if not np.isnan(x[1]) else -1,
        reverse=True,
    )

    # Apply selection criteria
    if top_k is not None:
        selected = [f for f, s in sorted_feats[:top_k] if not np.isnan(s)]
    else:
        selected = [f for f, s in sorted_feats if not np.isnan(s) and s > threshold]

    # --- Supplementary recent-folds pass (KB#37 fix) ---
    # Run feature selection on only the last N folds where modern features
    # have real data. Add any features that rank highly in this pass.
    n_recent = feat_cfg.recent_folds_for_supplement
    if n_recent > 0 and n_folds > n_recent:
        recent_splits = splits[-n_recent:]
        recent_importance: Dict[str, List[float]] = {f: [] for f in feature_names}

        for train_seasons, _ in recent_splits:
            train_mask = X["_season"].isin(train_seasons)
            X_tr = drop_meta(X[train_mask])[feature_names]
            y_tr = y_int[train_mask]
            coverage = X_tr.notna().mean()

            if params is None:
                from ml.config import ModelConfig
                cfg = ModelConfig()
                p = cfg.xgb_params if model_type == "xgboost" else cfg.lgb_params
            else:
                p = params

            model = _build_model(model_type, p)
            recent_seasons_s = X[train_mask]["_season"] if "_season" in X.columns else None
            sw = _compute_sample_weights(y[train_mask], seasons=recent_seasons_s)
            _fit_with_early_stopping(model, X_tr, y_tr, model_type, sample_weight=sw)

            imp = model.feature_importances_
            for feat, score in zip(feature_names, imp):
                if coverage.get(feat, 0) > 0.10:
                    recent_importance[feat].append(score)

        # Average recent-only importance
        recent_avg = {
            feat: float(np.mean(scores)) if scores else float("nan")
            for feat, scores in recent_importance.items()
        }
        recent_sorted = sorted(
            recent_avg.items(),
            key=lambda x: x[1] if not np.isnan(x[1]) else -1,
            reverse=True,
        )

        # Add top features from recent pass that aren't already selected
        selected_set = set(selected)
        # Always guarantee at least 15 slots for modern features that only
        # appear in recent folds (e.g., Sofascore stats added from 2017+)
        n_supplement = max(15, feat_cfg.max_features - len(selected))
        n_added = 0
        for feat, score in recent_sorted:
            if n_added >= n_supplement:
                break
            if feat not in selected_set and not np.isnan(score) and score > threshold:
                selected.append(feat)
                selected_set.add(feat)
                n_added += 1
                # Update avg_importance with the recent-only score so
                # correlation pruning can rank it correctly
                avg_importance[feat] = score

        if n_added > 0:
            log.info(
                "Supplementary recent-folds pass added %d modern features "
                "(from last %d folds)",
                n_added, n_recent,
            )

    log.info(
        "Feature selection: %d -> %d features (top_k=%s, threshold=%s, "
        "recency_base=%.1f)",
        len(feature_names), len(selected), top_k, threshold, recency_base,
    )
    return selected, avg_importance


def correlation_pruning(
    X: pd.DataFrame,
    feature_names: list[str],
    importance: Dict[str, float],
    threshold: float | None = None,
) -> list[str]:
    """Remove highly correlated features, keeping the more important one.

    For each pair with |r| > threshold, drop the less important feature.
    """
    feat_cfg = FeatureConfig()
    if threshold is None:
        threshold = feat_cfg.correlation_threshold

    X_feat = X[feature_names].copy()
    corr = X_feat.corr().abs()

    to_drop = set()
    for i in range(len(feature_names)):
        if feature_names[i] in to_drop:
            continue
        for j in range(i + 1, len(feature_names)):
            if feature_names[j] in to_drop:
                continue
            if corr.iloc[i, j] > threshold:
                fi = feature_names[i]
                fj = feature_names[j]
                # Drop the less important one
                if importance.get(fi, 0) >= importance.get(fj, 0):
                    to_drop.add(fj)
                else:
                    to_drop.add(fi)

    selected = [f for f in feature_names if f not in to_drop]
    log.info("Correlation pruning: %d -> %d features (r>%.2f)", len(feature_names), len(selected), threshold)
    return selected


def save_importance_history(
    importance: Dict[str, float],
    selected: List[str],
    variant: str = "universal",
) -> None:
    """Append current run's feature importance to a persistent history file.

    Keeps the last 10 runs. Stored at models/<variant>/feature_importance_history.json.
    """
    history_dir = MODELS_DIR / variant
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / "feature_importance_history.json"

    history: list = []
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning("Could not read importance history; starting fresh")

    sorted_imp = sorted(importance.items(), key=lambda x: -x[1])
    history.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_selected": len(selected),
        "top_20": [[f, round(s, 6)] for f, s in sorted_imp[:20]],
        "importance": {k: round(v, 6) for k, v in importance.items()},
    })

    # Keep last 10 runs
    history = history[-10:]
    history_path.write_text(json.dumps(history, indent=2))
    log.info("Saved feature importance history (%d runs) to %s", len(history), history_path)


def compare_importance(
    variant: str = "universal",
    run_a: int = -2,
    run_b: int = -1,
) -> Dict:
    """Compare feature importance between two historical runs.

    *run_a* and *run_b* are indices into the history list (default: last two).
    Returns dict with rank changes and warnings for large shifts.
    """
    history_path = MODELS_DIR / variant / "feature_importance_history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"No importance history at {history_path}")

    history = json.loads(history_path.read_text())
    if len(history) < 2:
        return {"warning": "Need at least 2 runs to compare"}

    imp_a = history[run_a]["importance"]
    imp_b = history[run_b]["importance"]

    # Rank features in each run
    rank_a = {f: r for r, (f, _) in enumerate(sorted(imp_a.items(), key=lambda x: -x[1]))}
    rank_b = {f: r for r, (f, _) in enumerate(sorted(imp_b.items(), key=lambda x: -x[1]))}

    shared = sorted(set(rank_a) & set(rank_b))
    large_shifts = []
    for f in shared:
        ra, rb = rank_a[f], rank_b[f]
        shift = abs(ra - rb)
        # Flag if rank changed by more than 50% of total features
        if shift > max(len(rank_a), len(rank_b)) * 0.5:
            large_shifts.append({"feature": f, "rank_a": ra, "rank_b": rb, "shift": shift})

    return {
        "run_a_timestamp": history[run_a]["timestamp"],
        "run_b_timestamp": history[run_b]["timestamp"],
        "n_shared_features": len(shared),
        "large_rank_shifts": large_shifts,
        "n_warnings": len(large_shifts),
    }
