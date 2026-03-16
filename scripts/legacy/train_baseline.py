#!/usr/bin/env python3
"""Train baseline CatBoost model with walk-forward validation.

This establishes the accuracy benchmark for the Football Intelligence System.

Usage:
    python scripts/train_baseline.py [--quick]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit

from features.build import get_ml_feature_columns
from storage.paths import features_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Prepare data for training.

    Returns:
        (cleaned_df, feature_columns)
    """
    df = df.copy()

    # Ensure we have result column
    if "result" not in df.columns:
        def get_result(row):
            hs = row.get("home_score")
            as_ = row.get("away_score")
            if pd.isna(hs) or pd.isna(as_):
                return None
            if int(hs) > int(as_):
                return "H"
            elif int(as_) > int(hs):
                return "A"
            return "D"
        df["result"] = df.apply(get_result, axis=1)

    # Filter to valid results
    df = df[df["result"].isin(["H", "D", "A"])].copy()

    # Get ML feature columns
    ml_cols = get_ml_feature_columns(df)

    # Filter to numeric features only
    numeric_cols = []
    for col in ml_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if df[col].dtype in [np.float64, np.int64, float, int]:
                    numeric_cols.append(col)
            except Exception:
                pass

    # Remove features with >20% NaN
    good_cols = []
    for col in numeric_cols:
        nan_rate = df[col].isna().sum() / len(df)
        if nan_rate < 0.20:
            good_cols.append(col)

    log.info(f"Using {len(good_cols)} features (filtered from {len(ml_cols)} ML columns)")

    return df, good_cols


def train_with_time_series_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int = 5,
    quick: bool = False,
) -> dict:
    """Train model with time-series cross-validation.

    Args:
        df: Feature DataFrame
        feature_cols: List of feature column names
        n_splits: Number of CV folds
        quick: Use fewer iterations

    Returns:
        Dictionary with results and trained model
    """
    # Sort by date
    df = df.copy()
    df["_sort_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values("_sort_date").reset_index(drop=True)

    # Prepare X and y
    X = df[feature_cols].fillna(0).astype(float)
    y = df["result"]

    # Time series split
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = []
    all_predictions = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        log.info(f"Fold {fold+1}/{n_splits}: Train={len(X_train)}, Test={len(X_test)}")

        # Train CatBoost
        model = CatBoostClassifier(
            iterations=300 if quick else 800,
            learning_rate=0.05,
            depth=6,
            class_weights={"H": 1.0, "D": 1.3, "A": 1.0},  # Boost draws
            verbose=False,
            random_seed=42,
        )

        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        try:
            ll = log_loss(y_test, y_proba, labels=model.classes_)
        except:
            ll = np.nan

        # Per-class accuracy
        home_mask = y_test == "H"
        draw_mask = y_test == "D"
        away_mask = y_test == "A"

        home_acc = (y_pred[home_mask] == "H").mean() if home_mask.sum() > 0 else 0
        draw_acc = (y_pred[draw_mask] == "D").mean() if draw_mask.sum() > 0 else 0
        away_acc = (y_pred[away_mask] == "A").mean() if away_mask.sum() > 0 else 0

        fold_result = {
            "fold": fold + 1,
            "accuracy": acc,
            "log_loss": ll,
            "home_acc": home_acc,
            "draw_acc": draw_acc,
            "away_acc": away_acc,
            "n_home": home_mask.sum(),
            "n_draw": draw_mask.sum(),
            "n_away": away_mask.sum(),
        }
        results.append(fold_result)

        log.info(f"  Accuracy: {acc:.3f}, H={home_acc:.2f}, D={draw_acc:.2f}, A={away_acc:.2f}")

        # Store predictions
        pred_df = df.iloc[test_idx][["match_id", "match_date", "home_team", "away_team"]].copy()
        pred_df["actual"] = y_test.values
        pred_df["predicted"] = y_pred.flatten() if hasattr(y_pred, 'flatten') else list(y_pred)
        for i, cls in enumerate(model.classes_):
            pred_df[f"prob_{cls}"] = y_proba[:, i]
        all_predictions.append(pred_df)

    # Aggregate results
    results_df = pd.DataFrame(results)
    avg_acc = results_df["accuracy"].mean()
    avg_ll = results_df["log_loss"].mean()

    # Train final model on all data
    log.info("\nTraining final model on all data...")
    final_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        class_weights={"H": 1.0, "D": 1.3, "A": 1.0},
        verbose=False,
        random_seed=42,
    )
    final_model.fit(X, y)

    # Get feature importance
    importance = dict(zip(feature_cols, final_model.get_feature_importance()))

    return {
        "cv_accuracy": avg_acc,
        "cv_log_loss": avg_ll,
        "fold_results": results_df,
        "predictions": pd.concat(all_predictions, ignore_index=True),
        "final_model": final_model,
        "feature_importance": importance,
        "feature_names": feature_cols,
    }


def print_results(results: dict):
    """Print training results summary."""
    print("\n" + "=" * 70)
    print("BASELINE MODEL RESULTS")
    print("=" * 70)

    # Overall metrics
    print(f"\n📊 CROSS-VALIDATION PERFORMANCE ({len(results['fold_results'])} folds)")
    print("-" * 40)
    print(f"  Overall Accuracy: {results['cv_accuracy']*100:.1f}%")
    print(f"  Log Loss: {results['cv_log_loss']:.3f}")

    # Per-class
    fold_df = results['fold_results']
    print(f"\n📈 PER-CLASS ACCURACY")
    print("-" * 40)
    print(f"  Home Win (H): {fold_df['home_acc'].mean()*100:.1f}%")
    print(f"  Draw (D):     {fold_df['draw_acc'].mean()*100:.1f}%")
    print(f"  Away Win (A): {fold_df['away_acc'].mean()*100:.1f}%")

    # Baselines
    print(f"\n📏 VS BASELINES")
    print("-" * 40)
    acc = results['cv_accuracy']
    print(f"  Random (33.3%):        {'✅ +' + f'{(acc-0.333)*100:.1f}%' if acc > 0.333 else '❌'}")
    print(f"  Always Home (45%):     {'✅ +' + f'{(acc-0.45)*100:.1f}%' if acc > 0.45 else '❌'}")
    print(f"  Bookmaker impl (55%):  {'✅ +' + f'{(acc-0.55)*100:.1f}%' if acc > 0.55 else '❌ -' + f'{(0.55-acc)*100:.1f}%'}")
    print(f"  Target (65%):          {'✅ ACHIEVED!' if acc > 0.65 else '❌ Gap: ' + f'{(0.65-acc)*100:.1f}%'}")

    # Top features
    importance = results['feature_importance']
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    print(f"\n🔝 TOP 15 FEATURES")
    print("-" * 40)
    for i, (feat, imp) in enumerate(sorted_imp[:15]):
        print(f"  {i+1:2}. {feat:40} {imp:6.1f}")

    print("\n" + "=" * 70)


def save_model(results: dict, output_dir: Path):
    """Save trained model and metadata."""
    import json

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / "baseline_catboost.cbm"
    results["final_model"].save_model(str(model_path))
    log.info(f"Saved model to {model_path}")

    # Save metadata
    metadata = {
        "cv_accuracy": results["cv_accuracy"],
        "cv_log_loss": results["cv_log_loss"],
        "n_features": len(results["feature_names"]),
        "feature_names": results["feature_names"],
        "top_features": sorted(
            results["feature_importance"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:50],
    }
    metadata_path = output_dir / "baseline_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Saved metadata to {metadata_path}")

    # Save predictions
    pred_path = output_dir / "cv_predictions.parquet"
    results["predictions"].to_parquet(pred_path, index=False)
    log.info(f"Saved predictions to {pred_path}")


def main():
    parser = argparse.ArgumentParser(description="Train baseline model")
    parser.add_argument("--quick", action="store_true", help="Use fewer iterations")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("TRAINING BASELINE MODEL - FOOTBALL INTELLIGENCE SYSTEM")
    log.info("=" * 70)

    # Load features
    feat_path = features_path()
    if not feat_path.exists():
        log.error(f"Features file not found: {feat_path}")
        log.error("Run 'python scripts/rebuild_features.py' first")
        return

    df = pd.read_parquet(feat_path)
    log.info(f"Loaded {len(df)} matches from {feat_path}")

    # Prepare data
    df, feature_cols = prepare_data(df)
    log.info(f"After filtering: {len(df)} matches, {len(feature_cols)} features")

    if len(df) < 100:
        log.error("Not enough data for training (need at least 100 matches)")
        return

    # Train with CV
    results = train_with_time_series_cv(
        df, feature_cols,
        n_splits=5,
        quick=args.quick,
    )

    # Print results
    print_results(results)

    # Save model
    model_dir = Path("data/models/universal")
    save_model(results, model_dir)

    log.info("\n✅ Baseline training complete!")
    log.info(f"   Accuracy: {results['cv_accuracy']*100:.1f}%")
    log.info(f"   Model saved to: {model_dir}")

    return results


if __name__ == "__main__":
    main()
