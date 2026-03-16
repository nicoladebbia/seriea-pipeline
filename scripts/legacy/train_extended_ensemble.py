#!/usr/bin/env python3
"""EXTENDED ENSEMBLE TRAINING - Dynamic Feature Discovery

Trains xG regression models (home/away) and a classifier using ALL available
ML-safe features from features.parquet, with importance-based selection and
correlation pruning. No hardcoded feature lists — automatically picks up new
features as they're added to the pipeline.

Saves:
  - xg_home.cbm / xg_away.cbm (CatBoost regressors for Poisson-based prediction)
  - catboost_upcoming.cbm (CatBoost classifier for direct 1X2 prediction)
  - extended_model_metadata.json (feature list, CV metrics, importance)
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import accuracy_score, mean_absolute_error, log_loss

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR, MODELS_DIR
from features.build import get_ml_feature_columns
from ml.config import LABEL_MAP, ODDS_COLUMN_PATTERNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# Post-match features that must be excluded (leakage)
POST_MATCH_EXACT = {
    "home_shots_on_target", "away_shots_on_target",
    "home_shots_on_target_count", "away_shots_on_target_count",
    "home_shots_on_target_total", "away_shots_on_target_total",
    "home_total_shots", "away_total_shots",
    "home_crosses", "away_crosses",
    "home_red_cards", "away_red_cards",
    "home_saves_count", "away_saves_count",
    "home_saves", "away_saves",
    "home_corners", "away_corners",
    "home_offsides", "away_offsides",
    "home_touches", "away_touches",
    "home_dribbles", "away_dribbles",
    "home_tackles", "away_tackles",
    "home_interceptions", "away_interceptions",
    "home_fouls", "away_fouls",
    "home_clearances", "away_clearances",
    "home_blocks", "away_blocks",
    "home_possession", "away_possession",
    "home_passing_accuracy", "away_passing_accuracy",
    "attendance", "home_xg", "away_xg",
    "home_cards", "away_cards",
}


def _is_pre_match_feature(col: str) -> bool:
    """Check if a column is available before kickoff."""
    if col in POST_MATCH_EXACT:
        return False
    # Exclude any raw stat columns that aren't rolling averages
    if col.startswith(("home_us_team_shots", "away_us_team_shots")):
        # These are Understat team-level aggregates (pre-match), allow them
        return True
    return True


def _is_odds_feature(col: str) -> bool:
    """Check if a column is derived from betting odds."""
    return any(col.startswith(p) or col == p.rstrip("_") for p in ODDS_COLUMN_PATTERNS)


def discover_features(df: pd.DataFrame, exclude_odds: bool = False) -> List[str]:
    """Dynamically discover all ML-safe pre-match features from features.parquet."""
    ml_cols = get_ml_feature_columns(df)

    # Filter to numeric columns only
    numeric_cols = [c for c in ml_cols
                    if df[c].dtype in ("float64", "float32", "int64", "int32", "int8", "uint8")]

    # Filter to pre-match features
    pre_match = [c for c in numeric_cols if _is_pre_match_feature(c)]

    # Optionally exclude odds
    if exclude_odds:
        pre_match = [c for c in pre_match if not _is_odds_feature(c)]

    # Filter out columns with >80% NaN (too sparse to be useful for xG regression)
    nan_pcts = df[pre_match].isna().mean()
    usable = [c for c in pre_match if nan_pcts[c] < 0.80]

    log.info("Feature discovery: %d ML-safe → %d numeric → %d pre-match → %d usable (<80%% NaN)",
             len(ml_cols), len(numeric_cols), len(pre_match), len(usable))
    return usable


def select_by_importance(
    X: pd.DataFrame, y: pd.Series, feature_names: List[str], top_k: int = 120,
) -> Tuple[List[str], Dict[str, float]]:
    """Select top_k features by XGBoost importance."""
    from xgboost import XGBRegressor

    X_sel = X[feature_names].fillna(0)

    # XGBoost forbids [, ], < and > in feature names — sanitize temporarily
    safe_names = {f: f.replace("<", "_lt_").replace(">", "_gt_").replace("[", "_lb_").replace("]", "_rb_")
                  for f in feature_names}
    X_safe = X_sel.rename(columns=safe_names)

    selector = XGBRegressor(n_estimators=200, max_depth=6, random_state=42, verbosity=0)
    selector.fit(X_safe, y)

    # Map importance back to original names
    reverse_map = {v: k for k, v in safe_names.items()}
    importance = {reverse_map[sn]: float(imp)
                  for sn, imp in zip(X_safe.columns, selector.feature_importances_)}
    sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    selected = [f for f, _ in sorted_feats[:top_k]]

    log.info("Importance selection: %d → %d features", len(feature_names), len(selected))
    return selected, importance


def correlation_pruning(
    X: pd.DataFrame, feature_names: List[str], importance: Dict[str, float],
    threshold: float = 0.85,
) -> List[str]:
    """Remove highly correlated features, keeping the more important one."""
    X_feat = X[feature_names].fillna(0)
    corr = X_feat.corr().abs()

    to_drop = set()
    for i in range(len(feature_names)):
        if feature_names[i] in to_drop:
            continue
        for j in range(i + 1, len(feature_names)):
            if feature_names[j] in to_drop:
                continue
            if corr.iloc[i, j] > threshold:
                fi, fj = feature_names[i], feature_names[j]
                if importance.get(fi, 0) >= importance.get(fj, 0):
                    to_drop.add(fj)
                else:
                    to_drop.add(fi)

    selected = [f for f in feature_names if f not in to_drop]
    log.info("Correlation pruning: %d → %d features (r>%.2f, dropped %d)",
             len(feature_names), len(selected), threshold, len(to_drop))
    return selected


def time_series_split(df: pd.DataFrame, n_splits: int = 5) -> List[Tuple[pd.Index, pd.Index]]:
    """Walk-forward time series splits."""
    seasons = sorted(df["season"].unique())
    splits = []
    min_train = 5

    for i in range(min_train, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]

        train_idx = df[df["season"].isin(train_seasons)].index
        test_idx = df[df["season"] == test_season].index

        if len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits[-n_splits:] if len(splits) > n_splits else splits


def poisson_win_prob(home_xg: float, away_xg: float, max_goals: int = 10) -> Dict[str, float]:
    """Calculate win probabilities from expected goals using Poisson distribution."""
    home_xg = max(0.3, min(4.0, home_xg))
    away_xg = max(0.3, min(4.0, away_xg))

    home_probs = [poisson.pmf(g, home_xg) for g in range(max_goals)]
    away_probs = [poisson.pmf(g, away_xg) for g in range(max_goals)]

    prob_home_win = 0.0
    prob_draw = 0.0
    prob_away_win = 0.0

    for h_goals in range(max_goals):
        for a_goals in range(max_goals):
            prob = home_probs[h_goals] * away_probs[a_goals]
            if h_goals > a_goals:
                prob_home_win += prob
            elif h_goals == a_goals:
                prob_draw += prob
            else:
                prob_away_win += prob

    total = prob_home_win + prob_draw + prob_away_win
    return {
        "H": prob_home_win / total,
        "D": prob_draw / total,
        "A": prob_away_win / total,
    }


def evaluate_high_confidence(y_true: List[int], y_pred: List[int], proba: List[List[float]], threshold: float = 0.55) -> float:
    """Evaluate accuracy on high-confidence predictions only."""
    high_conf_mask = [max(p) >= threshold for p in proba]
    if not any(high_conf_mask):
        return 0.0

    y_true_hc = [y for y, m in zip(y_true, high_conf_mask) if m]
    y_pred_hc = [y for y, m in zip(y_pred, high_conf_mask) if m]

    return accuracy_score(y_true_hc, y_pred_hc)


def load_and_prepare_data(exclude_features=None):
    """Load features.parquet, discover features, select/prune, return prepared data.

    Args:
        exclude_features: optional set of feature names to drop before selection
                          (e.g. drift-detected or high-NaN features from challenger).

    Returns:
        dict with keys: df, X, y_home_goals, y_away_goals, y_result,
                        available_features, importance, splits
    """
    feature_path = DATA_DIR / "features" / "features.parquet"
    df = pd.read_parquet(feature_path)
    df = df.dropna(subset=["home_score", "away_score"])
    log.info("Loaded %d matches with %d total columns", len(df), len(df.columns))

    all_features = discover_features(df, exclude_odds=False)

    # Drop drift/NaN-flagged features if requested
    if exclude_features:
        before = len(all_features)
        all_features = [f for f in all_features if f not in exclude_features]
        dropped = before - len(all_features)
        if dropped:
            log.info("Excluded %d drift/NaN features (%d remaining)", dropped, len(all_features))

    for col in all_features:
        if df[col].isna().any():
            if "elo" in col:
                df[col] = df[col].ffill().fillna(1500.0)
            elif "h2h_" in col and "rate" in col:
                df[col] = df[col].fillna(1/3)
            elif "h2h_" in col:
                df[col] = df[col].fillna(0)
            elif "_roll_" in col or "streak" in col or "momentum" in col:
                df[col] = df[col].fillna(0.0)
            else:
                df[col] = df[col].fillna(0.0)

    selected, importance = select_by_importance(
        df, df["home_score"], all_features, top_k=150,
    )
    selected = correlation_pruning(df, selected, importance, threshold=0.85)

    available_features = selected
    X = df[available_features].fillna(0)

    y_home_goals = df["home_score"]
    y_away_goals = df["away_score"]
    y_result = df.apply(
        lambda r: "H" if r["home_score"] > r["away_score"]
        else ("A" if r["away_score"] > r["home_score"] else "D"),
        axis=1
    )

    splits = time_series_split(df, n_splits=5)

    return {
        "df": df,
        "X": X,
        "y_home_goals": y_home_goals,
        "y_away_goals": y_away_goals,
        "y_result": y_result,
        "available_features": available_features,
        "importance": importance,
        "splits": splits,
    }


def evaluate_xg_cv(data: dict) -> dict:
    """Run walk-forward CV for xG models. Returns metrics dict."""
    from catboost import CatBoostRegressor

    X = data["X"]
    y_home_goals = data["y_home_goals"]
    y_away_goals = data["y_away_goals"]
    y_result = data["y_result"]
    splits = data["splits"]

    all_preds = []
    all_true = []
    all_probs = []
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_home_train = y_home_goals.loc[train_idx]
        y_away_train = y_away_goals.loc[train_idx]
        y_result_test = y_result.loc[test_idx].map(LABEL_MAP)
        n_train = int(len(X_train) * 0.85)

        home_model = CatBoostRegressor(
            iterations=1500, learning_rate=0.02, depth=6,
            l2_leaf_reg=3, min_data_in_leaf=20,
            early_stopping_rounds=150, verbose=0, random_seed=42
        )
        home_model.fit(
            X_train.iloc[:n_train], y_home_train.iloc[:n_train],
            eval_set=(X_train.iloc[n_train:], y_home_train.iloc[n_train:]),
            verbose=False
        )

        away_model = CatBoostRegressor(
            iterations=1500, learning_rate=0.02, depth=6,
            l2_leaf_reg=3, min_data_in_leaf=20,
            early_stopping_rounds=150, verbose=0, random_seed=42
        )
        away_model.fit(
            X_train.iloc[:n_train], y_away_train.iloc[:n_train],
            eval_set=(X_train.iloc[n_train:], y_away_train.iloc[n_train:]),
            verbose=False
        )

        pred_home_xg = home_model.predict(X_test)
        pred_away_xg = away_model.predict(X_test)

        fold_preds = []
        fold_probs = []

        for h_xg, a_xg in zip(pred_home_xg, pred_away_xg):
            probs = poisson_win_prob(h_xg, a_xg)
            fold_probs.append([probs["H"], probs["D"], probs["A"]])
            if probs["H"] >= probs["D"] and probs["H"] >= probs["A"]:
                pred = LABEL_MAP["H"]
            elif probs["A"] >= probs["D"]:
                pred = LABEL_MAP["A"]
            else:
                pred = LABEL_MAP["D"]
            fold_preds.append(pred)

        fold_acc = accuracy_score(y_result_test, fold_preds)
        fold_hc_acc = evaluate_high_confidence(
            y_result_test.tolist(), fold_preds, fold_probs, threshold=0.55
        )
        home_mae = mean_absolute_error(y_home_goals.loc[test_idx], pred_home_xg)
        away_mae = mean_absolute_error(y_away_goals.loc[test_idx], pred_away_xg)

        fold_metrics.append({
            "fold": fold_idx + 1,
            "accuracy": fold_acc,
            "high_conf_accuracy": fold_hc_acc,
            "home_mae": home_mae,
            "away_mae": away_mae,
        })

        log.info("  xG Fold %d: Acc=%.1f%%, HC-Acc=%.1f%%, xG MAE: H=%.2f, A=%.2f",
                 fold_idx + 1, fold_acc * 100, fold_hc_acc * 100, home_mae, away_mae)

        all_preds.extend(fold_preds)
        all_true.extend(y_result_test.tolist())
        all_probs.extend(fold_probs)

    overall_acc = accuracy_score(all_true, all_preds)
    overall_hc_acc = evaluate_high_confidence(all_true, all_preds, all_probs, threshold=0.55)
    overall_logloss = log_loss(all_true, all_probs, labels=[0, 1, 2])

    return {
        "xg_poisson_accuracy": round(overall_acc, 4),
        "xg_poisson_hc_accuracy": round(overall_hc_acc, 4),
        "xg_poisson_logloss": round(overall_logloss, 4),
        "fold_metrics": fold_metrics,
    }


def evaluate_classifier_cv(data: dict) -> dict:
    """Run walk-forward CV for classifier. Returns metrics dict."""
    from catboost import CatBoostClassifier

    X = data["X"]
    y_result = data["y_result"]
    splits = data["splits"]

    classifier_preds = []
    classifier_true = []
    classifier_probs = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train = y_result.loc[train_idx].map(LABEL_MAP)
        y_test = y_result.loc[test_idx].map(LABEL_MAP)
        n_train = int(len(X_train) * 0.85)

        model = CatBoostClassifier(
            iterations=2000, learning_rate=0.02, depth=6,
            l2_leaf_reg=3, min_data_in_leaf=20,
            auto_class_weights="SqrtBalanced",
            early_stopping_rounds=150, verbose=0, random_seed=42
        )
        model.fit(
            X_train.iloc[:n_train], y_train.iloc[:n_train],
            eval_set=(X_train.iloc[n_train:], y_train.iloc[n_train:]),
            verbose=False
        )

        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)

        fold_acc = accuracy_score(y_test, pred)
        fold_hc_acc = evaluate_high_confidence(y_test.tolist(), pred.tolist(), proba.tolist(), threshold=0.55)
        log.info("  Classifier Fold %d: Acc=%.1f%%, HC-Acc=%.1f%%", fold_idx + 1, fold_acc * 100, fold_hc_acc * 100)

        classifier_preds.extend(pred.tolist())
        classifier_true.extend(y_test.tolist())
        classifier_probs.extend(proba.tolist())

    class_acc = accuracy_score(classifier_true, classifier_preds)
    class_hc_acc = evaluate_high_confidence(classifier_true, classifier_preds, classifier_probs, threshold=0.55)
    class_logloss = log_loss(classifier_true, classifier_probs, labels=[0, 1, 2])

    return {
        "classifier_accuracy": round(class_acc, 4),
        "classifier_hc_accuracy": round(class_hc_acc, 4),
        "classifier_logloss": round(class_logloss, 4),
    }


def train_final_models(data: dict, model_dir: Path = None):
    """Train final models on all data and save to model_dir.

    Returns:
        dict with metadata (feature list, importance, metrics placeholders).
    """
    from catboost import CatBoostRegressor, CatBoostClassifier

    if model_dir is None:
        model_dir = MODELS_DIR / "universal"
    model_dir.mkdir(parents=True, exist_ok=True)

    X = data["X"]
    y_home_goals = data["y_home_goals"]
    y_away_goals = data["y_away_goals"]
    y_result = data["y_result"]
    available_features = data["available_features"]

    n_all = int(len(X) * 0.90)

    home_model_final = CatBoostRegressor(
        iterations=1500, learning_rate=0.02, depth=6,
        l2_leaf_reg=3, min_data_in_leaf=20,
        early_stopping_rounds=150, verbose=0, random_seed=42
    )
    home_model_final.fit(
        X.iloc[:n_all], y_home_goals.iloc[:n_all],
        eval_set=(X.iloc[n_all:], y_home_goals.iloc[n_all:]),
        verbose=False
    )

    away_model_final = CatBoostRegressor(
        iterations=1500, learning_rate=0.02, depth=6,
        l2_leaf_reg=3, min_data_in_leaf=20,
        early_stopping_rounds=150, verbose=0, random_seed=42
    )
    away_model_final.fit(
        X.iloc[:n_all], y_away_goals.iloc[:n_all],
        eval_set=(X.iloc[n_all:], y_away_goals.iloc[n_all:]),
        verbose=False
    )

    classifier_final = CatBoostClassifier(
        iterations=2000, learning_rate=0.02, depth=6,
        l2_leaf_reg=3, min_data_in_leaf=20,
        auto_class_weights="SqrtBalanced",
        early_stopping_rounds=150, verbose=0, random_seed=42
    )
    classifier_final.fit(
        X.iloc[:n_all], y_result.iloc[:n_all].map(LABEL_MAP),
        eval_set=(X.iloc[n_all:], y_result.iloc[n_all:].map(LABEL_MAP)),
        verbose=False
    )

    # Save models
    home_model_final.save_model(str(model_dir / "xg_home_extended.cbm"))
    away_model_final.save_model(str(model_dir / "xg_away_extended.cbm"))
    classifier_final.save_model(str(model_dir / "classifier_extended.cbm"))

    home_model_final.save_model(str(model_dir / "xg_home.cbm"))
    away_model_final.save_model(str(model_dir / "xg_away.cbm"))
    classifier_final.save_model(str(model_dir / "catboost_upcoming.cbm"))

    # Feature importance
    xg_importance = dict(zip(available_features, home_model_final.feature_importances_))
    top_xg_features = sorted(xg_importance.items(), key=lambda x: x[1], reverse=True)[:20]

    classifier_importance = dict(zip(available_features, classifier_final.feature_importances_))
    top_classifier_features = sorted(classifier_importance.items(), key=lambda x: x[1], reverse=True)[:20]

    return {
        "available_features": available_features,
        "top_xg_features": top_xg_features,
        "top_classifier_features": top_classifier_features,
        "model_dir": str(model_dir),
    }


def save_metadata(model_dir: Path, available_features: list, cv_metrics: dict,
                  fold_metrics: list, top_xg_features: list, top_classifier_features: list):
    """Save model metadata JSON files."""
    metadata = {
        "model_version": "extended_v2.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_features": len(available_features),
        "feature_names": available_features,
        "cv_metrics": cv_metrics,
        "fold_metrics": fold_metrics,
        "top_xg_features": [f[0] if isinstance(f, tuple) else f for f in top_xg_features],
        "top_classifier_features": [f[0] if isinstance(f, tuple) else f for f in top_classifier_features],
        "label_map": LABEL_MAP,
    }

    with open(model_dir / "extended_model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    for name in ["catboost_upcoming_metadata.json", "catboost_no_odds_metadata.json"]:
        with open(model_dir / name, "w") as f:
            json.dump(metadata, f, indent=2)

    return metadata


def main():
    log.info("=" * 70)
    log.info("EXTENDED ENSEMBLE TRAINING - Dynamic Feature Discovery")
    log.info("=" * 70)

    # Load and prepare data
    data = load_and_prepare_data()
    available_features = data["available_features"]

    log.info("\nDataset statistics:")
    log.info("  Matches: %d, Features: %d", len(data["df"]), len(available_features))
    log.info("  Home goals mean: %.2f", data["y_home_goals"].mean())
    log.info("  Away goals mean: %.2f", data["y_away_goals"].mean())
    y_result = data["y_result"]
    log.info("  Result distribution: H=%d, D=%d, A=%d",
             sum(y_result == 'H'), sum(y_result == 'D'), sum(y_result == 'A'))
    log.info("  Using %d time-series splits", len(data["splits"]))

    # Cross-validate xG models
    log.info("\n" + "=" * 60)
    log.info("TRAINING EXTENDED xG MODELS")
    log.info("=" * 60)

    xg_metrics = evaluate_xg_cv(data)
    log.info("\n  Overall xG-Poisson accuracy: %.1f%%", xg_metrics["xg_poisson_accuracy"] * 100)
    log.info("  High-confidence (>55%%) accuracy: %.1f%%", xg_metrics["xg_poisson_hc_accuracy"] * 100)
    log.info("  Log loss: %.4f", xg_metrics["xg_poisson_logloss"])

    # Cross-validate classifier
    log.info("\n" + "=" * 60)
    log.info("TRAINING EXTENDED CLASSIFIER")
    log.info("=" * 60)

    class_metrics = evaluate_classifier_cv(data)
    log.info("\n  Overall classifier accuracy: %.1f%%", class_metrics["classifier_accuracy"] * 100)
    log.info("  High-confidence accuracy: %.1f%%", class_metrics["classifier_hc_accuracy"] * 100)
    log.info("  Log loss: %.4f", class_metrics["classifier_logloss"])

    # Train final models on all data
    log.info("\n" + "=" * 60)
    log.info("TRAINING FINAL MODELS ON ALL DATA")
    log.info("=" * 60)

    model_dir = MODELS_DIR / "universal"
    train_result = train_final_models(data, model_dir)

    # Save metadata
    cv_metrics = {**xg_metrics, **class_metrics}
    # Remove fold_metrics from cv_metrics since it's separate
    fold_metrics = cv_metrics.pop("fold_metrics", [])
    save_metadata(
        model_dir, available_features, cv_metrics, fold_metrics,
        train_result["top_xg_features"],
        train_result["top_classifier_features"],
    )

    # Summary
    log.info("\n" + "=" * 70)
    log.info("TRAINING COMPLETE - SUMMARY")
    log.info("=" * 70)
    log.info("  Features used: %d (dynamically discovered)", len(available_features))
    log.info("  xG-Poisson CV accuracy: %.1f%%", xg_metrics["xg_poisson_accuracy"] * 100)
    log.info("  xG-Poisson HC accuracy: %.1f%%", xg_metrics["xg_poisson_hc_accuracy"] * 100)
    log.info("  Classifier CV accuracy: %.1f%%", class_metrics["classifier_accuracy"] * 100)
    log.info("  Classifier HC accuracy: %.1f%%", class_metrics["classifier_hc_accuracy"] * 100)

    top_xg_features = train_result["top_xg_features"]
    top_classifier_features = train_result["top_classifier_features"]
    log.info("\nTop 10 xG features:")
    for feat, imp in top_xg_features[:10]:
        log.info("    %s: %.2f", feat, imp)
    log.info("\nTop 10 classifier features:")
    for feat, imp in top_classifier_features[:10]:
        log.info("    %s: %.2f", feat, imp)
    log.info("\nModels saved to: %s", model_dir)


if __name__ == "__main__":
    main()
