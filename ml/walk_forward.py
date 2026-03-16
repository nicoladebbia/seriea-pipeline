"""Walk-forward cross-validation for football prediction.

CRITICAL: Standard train/test splits cause data leakage in time-series data.
Walk-forward CV ensures we NEVER use future data to predict past matches.

The walk-forward approach:
1. Train on historical data (e.g., 2005-2020)
2. Purge gap: drop last N matchweeks of training to avoid leakage from
   rolling features that span the train/test boundary
3. Test on future data (e.g., 2021 season)
4. Expand training window, repeat

This mimics real-world prediction: we can only use data from BEFORE the match.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

log = logging.getLogger(__name__)

# Default purge gap: drop this many matchweeks from end of training set
# to prevent rolling-feature leakage across the train/test boundary.
# 2 matchweeks ≈ 20 matches, which covers a 5-match rolling window safely.
DEFAULT_PURGE_MATCHWEEKS = 2


@dataclass
class WalkForwardFold:
    """A single fold in walk-forward CV."""
    fold_number: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_matches: int
    test_matches: int


def walk_forward_split(
    df: pd.DataFrame,
    date_col: str = "match_date",
    min_train_seasons: int = 5,
    test_seasons: int = 1,
    expanding: bool = True,
    purge_matchweeks: int = DEFAULT_PURGE_MATCHWEEKS,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, WalkForwardFold]]:
    """Generate walk-forward train/test splits with purge gap.

    Args:
        df: Feature DataFrame with date column
        date_col: Column containing match dates
        min_train_seasons: Minimum seasons to train on before first test
        test_seasons: Number of seasons per test fold
        expanding: If True, training window expands. If False, rolling window.
        purge_matchweeks: Number of matchweeks to drop from end of training set
            to prevent rolling-feature leakage. Set to 0 to disable.

    Yields:
        (train_df, test_df, fold_info) tuples

    Example with min_train=5, test=1, purge=2:
        Fold 1: Train 2005-2010 (minus last 2 matchweeks), Test 2011
        Fold 2: Train 2005-2011 (minus last 2 matchweeks), Test 2012
        ...
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # Extract seasons
    if "season" not in df.columns:
        # Infer season from date (Aug-Jul cycle)
        df["_season"] = df[date_col].apply(
            lambda d: f"{d.year}-{d.year+1}" if d.month >= 8 else f"{d.year-1}-{d.year}"
        )
    else:
        df["_season"] = df["season"]

    seasons = sorted(df["_season"].unique())
    n_seasons = len(seasons)

    if n_seasons < min_train_seasons + test_seasons:
        raise ValueError(
            f"Need at least {min_train_seasons + test_seasons} seasons, "
            f"but only have {n_seasons}"
        )

    fold_num = 0
    for test_start_idx in range(min_train_seasons, n_seasons, test_seasons):
        fold_num += 1

        # Training seasons
        if expanding:
            train_seasons = seasons[:test_start_idx]
        else:
            train_seasons = seasons[max(0, test_start_idx - min_train_seasons):test_start_idx]

        # Test seasons
        test_end_idx = min(test_start_idx + test_seasons, n_seasons)
        test_seasons_list = seasons[test_start_idx:test_end_idx]

        train_df = df[df["_season"].isin(train_seasons)].copy()
        test_df = df[df["_season"].isin(test_seasons_list)].copy()

        if train_df.empty or test_df.empty:
            continue

        # --- Purge gap: drop last N matchweeks from training set ---
        # This prevents rolling features (e.g. 5-match rolling avg) from
        # leaking information across the train/test boundary.
        purged = 0
        if purge_matchweeks > 0 and not train_df.empty:
            train_dates = sorted(train_df[date_col].dt.date.unique())
            # Each unique date ≈ 1 matchweek (~5 matches per date in Serie A)
            if len(train_dates) > purge_matchweeks:
                cutoff_date = train_dates[-(purge_matchweeks)]
                pre_purge = len(train_df)
                train_df = train_df[train_df[date_col].dt.date < cutoff_date]
                purged = pre_purge - len(train_df)

        if train_df.empty:
            continue

        fold_info = WalkForwardFold(
            fold_number=fold_num,
            train_start=str(train_df[date_col].min().date()),
            train_end=str(train_df[date_col].max().date()),
            test_start=str(test_df[date_col].min().date()),
            test_end=str(test_df[date_col].max().date()),
            train_matches=len(train_df),
            test_matches=len(test_df),
        )

        purge_msg = f", purged {purged}" if purged > 0 else ""
        log.info(
            f"Fold {fold_num}: Train {fold_info.train_start} to {fold_info.train_end} "
            f"({fold_info.train_matches} matches{purge_msg}), "
            f"Test {fold_info.test_start} to {fold_info.test_end} "
            f"({fold_info.test_matches} matches)"
        )

        yield train_df.drop(columns=["_season"]), test_df.drop(columns=["_season"]), fold_info


def walk_forward_matchweek_split(
    df: pd.DataFrame,
    date_col: str = "match_date",
    min_train_matches: int = 1000,
    test_matchweeks: int = 2,
    purge_matchweeks: int = DEFAULT_PURGE_MATCHWEEKS,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, WalkForwardFold]]:
    """Generate walk-forward splits at matchweek granularity with purge gap.

    More fine-grained than season-level splits. Good for recent seasons
    where we want more frequent evaluation points.

    Args:
        df: Feature DataFrame
        date_col: Date column name
        min_train_matches: Minimum matches before first test
        test_matchweeks: Number of matchweeks per test fold (~10 matches each)
        purge_matchweeks: Matchweeks to drop from end of training (leakage prevention)

    Yields:
        (train_df, test_df, fold_info) tuples
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # Group by date to get "matchweeks" (all matches on same/close dates)
    df["_date_group"] = df[date_col].dt.date
    date_groups = sorted(df["_date_group"].unique())

    # ~10 matches per matchweek in Serie A, so test_matchweeks=2 ≈ 20 test matches
    matches_per_fold = test_matchweeks * 10

    # Account for purge gap in minimum training size
    purge_matches = purge_matchweeks * 10  # approximate
    effective_min_train = min_train_matches + purge_matches

    fold_num = 0
    current_train_end = effective_min_train

    while current_train_end + matches_per_fold <= len(df):
        fold_num += 1

        raw_train_df = df.iloc[:current_train_end].copy()
        test_end = min(current_train_end + matches_per_fold, len(df))
        test_df = df.iloc[current_train_end:test_end].copy()

        # Apply purge gap: drop last N matchweeks from training
        if purge_matchweeks > 0:
            train_dates = sorted(raw_train_df[date_col].dt.date.unique())
            if len(train_dates) > purge_matchweeks:
                cutoff = train_dates[-(purge_matchweeks)]
                train_df = raw_train_df[raw_train_df[date_col].dt.date < cutoff]
            else:
                train_df = raw_train_df
        else:
            train_df = raw_train_df

        fold_info = WalkForwardFold(
            fold_number=fold_num,
            train_start=str(train_df[date_col].min().date()),
            train_end=str(train_df[date_col].max().date()),
            test_start=str(test_df[date_col].min().date()),
            test_end=str(test_df[date_col].max().date()),
            train_matches=len(train_df),
            test_matches=len(test_df),
        )

        log.info(
            f"Fold {fold_num}: Train {fold_info.train_matches} matches, "
            f"Test {fold_info.test_matches} matches"
        )

        yield (
            train_df.drop(columns=["_date_group"]),
            test_df.drop(columns=["_date_group"]),
            fold_info
        )

        current_train_end = test_end


@dataclass
class WalkForwardResult:
    """Results from walk-forward cross-validation."""
    fold_number: int
    accuracy: float
    log_loss: float
    brier_score: float
    rps: float
    home_accuracy: float
    draw_accuracy: float
    away_accuracy: float
    n_home: int
    n_draw: int
    n_away: int
    n_total: int
    predictions: pd.DataFrame


def evaluate_walk_forward(
    model: BaseEstimator,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "result",
    min_train_seasons: int = 5,
    test_seasons: int = 1,
    purge_matchweeks: int = DEFAULT_PURGE_MATCHWEEKS,
) -> list[WalkForwardResult]:
    """Run walk-forward CV and evaluate model performance.

    Args:
        model: Scikit-learn compatible classifier
        df: Feature DataFrame
        feature_cols: List of feature column names
        target_col: Target column name
        min_train_seasons: Minimum training seasons
        test_seasons: Seasons per test fold
        purge_matchweeks: Matchweeks to purge from end of training set

    Returns:
        List of WalkForwardResult objects, one per fold
    """
    from sklearn.metrics import accuracy_score, log_loss

    results = []

    from ml.evaluation import ranked_probability_score

    LABEL_TO_INT = {"H": 0, "D": 1, "A": 2}

    for train_df, test_df, fold_info in walk_forward_split(
        df,
        min_train_seasons=min_train_seasons,
        test_seasons=test_seasons,
        purge_matchweeks=purge_matchweeks,
    ):
        # Prepare data
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[target_col]
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df[target_col]

        # Skip if target has invalid values
        y_train = y_train[y_train.isin(["H", "D", "A"])]
        X_train = X_train.loc[y_train.index]
        y_test = y_test[y_test.isin(["H", "D", "A"])]
        X_test = X_test.loc[y_test.index]

        if len(X_train) < 100 or len(X_test) < 10:
            continue

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)

        # Ensure proper class order for log_loss
        classes = model.classes_
        try:
            ll = log_loss(y_test, y_proba, labels=classes)
        except ValueError:
            ll = np.nan

        # Brier score and RPS (need integer labels in H=0,D=1,A=2 order)
        y_int = np.array([LABEL_TO_INT[v] for v in y_test.values])
        # Reorder y_proba columns to match H=0,D=1,A=2 if model classes differ
        class_order = list(classes)
        col_map = [class_order.index(c) for c in ["H", "D", "A"]]
        y_proba_ordered = y_proba[:, col_map]

        # Multiclass Brier
        y_onehot = np.zeros_like(y_proba_ordered)
        y_onehot[np.arange(len(y_int)), y_int] = 1.0
        brier = float(np.mean((y_proba_ordered - y_onehot) ** 2))

        # RPS
        rps = ranked_probability_score(y_int, y_proba_ordered)

        # Per-class accuracy
        home_mask = y_test == "H"
        draw_mask = y_test == "D"
        away_mask = y_test == "A"

        home_acc = (y_pred[home_mask] == "H").mean() if home_mask.sum() > 0 else 0
        draw_acc = (y_pred[draw_mask] == "D").mean() if draw_mask.sum() > 0 else 0
        away_acc = (y_pred[away_mask] == "A").mean() if away_mask.sum() > 0 else 0

        # Build predictions DataFrame
        pred_df = test_df[["match_id", "match_date", "home_team", "away_team"]].copy()
        pred_df["actual"] = y_test.values
        pred_df["predicted"] = y_pred
        for i, cls in enumerate(classes):
            pred_df[f"prob_{cls}"] = y_proba[:, i]

        result = WalkForwardResult(
            fold_number=fold_info.fold_number,
            accuracy=acc,
            log_loss=ll,
            brier_score=brier,
            rps=rps,
            home_accuracy=home_acc,
            draw_accuracy=draw_acc,
            away_accuracy=away_acc,
            n_home=home_mask.sum(),
            n_draw=draw_mask.sum(),
            n_away=away_mask.sum(),
            n_total=len(y_test),
            predictions=pred_df,
        )

        results.append(result)

        log.info(
            f"Fold {fold_info.fold_number}: "
            f"Accuracy={acc:.3f}, LogLoss={ll:.3f}, RPS={rps:.4f}, "
            f"H={home_acc:.2f}, D={draw_acc:.2f}, A={away_acc:.2f}"
        )

    return results


def print_walk_forward_summary(results: list[WalkForwardResult]):
    """Print summary of walk-forward CV results with standard errors and significance."""
    if not results:
        print("No results to summarize")
        return

    from ml.statistical_validation import (
        accuracy_with_ci, binomial_test, BASE_RATE_HOME_WIN, BASE_RATE_RANDOM,
    )

    print("\n" + "=" * 70)
    print("WALK-FORWARD CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    k = len(results)
    accuracies = np.array([r.accuracy for r in results])
    log_losses = np.array([r.log_loss for r in results if not np.isnan(r.log_loss)])
    brier_scores = np.array([r.brier_score for r in results])
    rps_scores = np.array([r.rps for r in results])
    home_accs = np.array([r.home_accuracy for r in results])
    draw_accs = np.array([r.draw_accuracy for r in results])
    away_accs = np.array([r.away_accuracy for r in results])

    # Standard error = std / sqrt(k)
    acc_se = np.std(accuracies, ddof=1) / np.sqrt(k) if k > 1 else 0

    print(f"\nOVERALL PERFORMANCE ({k} folds)")
    print("-" * 50)
    print(f"  Accuracy:  {np.mean(accuracies)*100:.1f}% +/- {acc_se*100:.1f}% (SE)")
    print(f"             (std across folds: {np.std(accuracies)*100:.1f}%)")
    if len(log_losses) > 0:
        ll_se = np.std(log_losses, ddof=1) / np.sqrt(len(log_losses)) if len(log_losses) > 1 else 0
        print(f"  Log Loss:  {np.mean(log_losses):.3f} +/- {ll_se:.3f} (SE)")
    brier_se = np.std(brier_scores, ddof=1) / np.sqrt(k) if k > 1 else 0
    rps_se = np.std(rps_scores, ddof=1) / np.sqrt(k) if k > 1 else 0
    print(f"  Brier:     {np.mean(brier_scores):.4f} +/- {brier_se:.4f} (SE)")
    print(f"  RPS:       {np.mean(rps_scores):.4f} +/- {rps_se:.4f} (SE)")

    # Pooled accuracy with binomial CI (aggregate all folds)
    total_correct = sum(int(r.accuracy * r.n_total + 0.5) for r in results)
    total_n = sum(r.n_total for r in results)
    pooled = accuracy_with_ci(total_correct, total_n)
    print(f"\n  Pooled: {pooled['accuracy']:.1%} "
          f"(95% CI: {pooled['ci_lo']:.1%}-{pooled['ci_hi']:.1%}, n={total_n}) "
          f"[{pooled['reliability']}]")

    # Significance test vs baselines
    sig_random = binomial_test(total_correct, total_n, BASE_RATE_RANDOM)
    sig_home = binomial_test(total_correct, total_n, BASE_RATE_HOME_WIN)
    print(f"\n  vs Random (33.3%): p={sig_random['p_value']:.6f} "
          f"{'** significant' if sig_random['significant_001'] else '* significant' if sig_random['significant_005'] else 'not significant'}")
    print(f"  vs Always-Home (44.4%): p={sig_home['p_value']:.6f} "
          f"{'** significant' if sig_home['significant_001'] else '* significant' if sig_home['significant_005'] else 'not significant'}")

    print(f"\nPER-CLASS ACCURACY")
    print("-" * 50)
    total_h = sum(r.n_home for r in results)
    total_d = sum(r.n_draw for r in results)
    total_a = sum(r.n_away for r in results)
    h_se = np.std(home_accs, ddof=1) / np.sqrt(k) if k > 1 else 0
    d_se = np.std(draw_accs, ddof=1) / np.sqrt(k) if k > 1 else 0
    a_se = np.std(away_accs, ddof=1) / np.sqrt(k) if k > 1 else 0
    print(f"  Home Win (H): {np.mean(home_accs)*100:.1f}% +/- {h_se*100:.1f}% SE ({total_h} samples)")
    print(f"  Draw (D):     {np.mean(draw_accs)*100:.1f}% +/- {d_se*100:.1f}% SE ({total_d} samples)")
    print(f"  Away Win (A): {np.mean(away_accs)*100:.1f}% +/- {a_se*100:.1f}% SE ({total_a} samples)")

    print(f"\nFOLD DETAILS")
    print("-" * 50)
    for r in results:
        print(f"  Fold {r.fold_number}: {r.accuracy*100:.1f}% acc (n={r.n_total}), "
              f"RPS={r.rps:.4f}, "
              f"H={r.home_accuracy*100:.0f}% D={r.draw_accuracy*100:.0f}% A={r.away_accuracy*100:.0f}%")

    # Baseline comparisons with context
    mean_acc = np.mean(accuracies)
    print(f"\nBASELINE COMPARISONS")
    print("-" * 50)
    print(f"  Random guess (33.3%):      {'BEATING' if mean_acc > 0.333 else 'BELOW'} (+{(mean_acc-0.333)*100:.1f}pp)")
    print(f"  Always Home (44.4%):       {'BEATING' if mean_acc > 0.444 else 'BELOW'} ({(mean_acc-0.444)*100:+.1f}pp)")
    print(f"  Bookmaker implied (~52%):  {'BEATING' if mean_acc > 0.52 else 'BELOW'} ({(mean_acc-0.52)*100:+.1f}pp)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with sample data
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from storage.paths import features_path

    print("Testing walk-forward CV module...")

    feat_path = features_path()
    if feat_path.exists():
        df = pd.read_parquet(feat_path)
        print(f"Loaded {len(df)} matches")

        # Test split generation
        print("\nGenerating walk-forward splits:")
        for train_df, test_df, fold in walk_forward_split(df, min_train_seasons=10):
            print(f"  Fold {fold.fold_number}: {fold.train_matches} train, {fold.test_matches} test")
            if fold.fold_number >= 3:
                print("  ... (truncated)")
                break
    else:
        print(f"Features file not found: {feat_path}")
        print("Run 'python scripts/rebuild_features.py' first")
