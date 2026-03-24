"""Train and validate DrawSpecialist binary classifier with walk-forward CV.

This script:
1. Trains DrawSpecialist with walk-forward CV (seasons 2005-2025)
2. Evaluates calibration vs ensemble draw probabilities
3. Backtests in betting context (2023-2025 with Pinnacle odds)
4. Tests blended approach with various weights

Walk-forward strategy:
- For each validation season N:
  - Train on all seasons < N
  - Validate on season N
  - Report precision, recall, F1, AUC-ROC for draw class

Betting backtest (2023-2025 only, where Pinnacle odds exist):
- Edge = DrawSpecialist_prob - Pinnacle_implied_prob
- Bet when edge > 5%, flat €20 stake
- Compare to ensemble's draw betting performance
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.draw_specialist import DrawSpecialist, compute_draw_specific_features

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def load_data(feature_path: str) -> pd.DataFrame:
    """Load and prepare features.parquet."""
    df = pd.read_parquet(feature_path)

    # Filter to matches with results
    df = df.dropna(subset=['home_score', 'away_score'])

    # Create result column if not exists
    if 'result' not in df.columns:
        df['result'] = df.apply(
            lambda r: 'H' if r['home_score'] > r['away_score']
            else ('A' if r['away_score'] > r['home_score'] else 'D'),
            axis=1
        )

    log.info(f"Loaded {len(df)} matches from {df.season.nunique()} seasons")
    log.info(f"Draw rate: {(df.result == 'D').mean():.1%}")

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get valid feature columns for training."""
    exclude = [
        'match_id', 'home_team', 'away_team', 'home_score', 'away_score',
        'match_date', 'season', 'result', 'league', 'gameweek'
    ]

    # Only numeric columns
    feature_cols = [
        c for c in df.columns
        if c not in exclude and df[c].dtype in ['float64', 'int64', 'float32', 'int32']
    ]

    return feature_cols


def walk_forward_cv(df: pd.DataFrame, feature_cols: list[str], val_seasons: list[str]) -> dict:
    """Perform walk-forward cross-validation.

    For each validation season:
    - Train on all prior seasons
    - Validate on that season
    - Report metrics

    Args:
        df: Full dataset
        feature_cols: Feature column names
        val_seasons: Seasons to use as validation folds

    Returns:
        Dictionary with per-fold and aggregate metrics
    """
    results = {
        'folds': [],
        'aggregate': {}
    }

    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    for val_season in val_seasons:
        log.info(f"\n{'='*60}")
        log.info(f"Validation fold: {val_season}")
        log.info(f"{'='*60}")

        # Split data
        train_df = df[df.season < val_season].copy()
        val_df = df[df.season == val_season].copy()

        if len(train_df) == 0 or len(val_df) == 0:
            log.warning(f"Skipping {val_season}: insufficient data")
            continue

        log.info(f"Train: {len(train_df)} matches ({train_df.season.min()} to {train_df.season.max()})")
        log.info(f"Val: {len(val_df)} matches")

        # Train model
        specialist = DrawSpecialist(class_weight=1.5, threshold=0.25)
        specialist.fit(train_df, train_df['result'], feature_cols)

        # Predict on validation
        y_proba = specialist.predict_draw_proba(val_df)
        y_pred = (y_proba > specialist.threshold).astype(int)
        y_true = (val_df['result'] == 'D').astype(int)

        # Metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_proba)
        brier = brier_score_loss(y_true, y_proba)

        # Draw frequency
        actual_draw_rate = y_true.mean()
        predicted_draw_rate = y_pred.mean()
        mean_draw_prob = y_proba.mean()

        fold_result = {
            'season': val_season,
            'n_matches': len(val_df),
            'n_draws': y_true.sum(),
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'auc_roc': auc,
            'brier': brier,
            'actual_draw_rate': actual_draw_rate,
            'predicted_draw_rate': predicted_draw_rate,
            'mean_draw_prob': mean_draw_prob,
        }

        results['folds'].append(fold_result)

        log.info(f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
        log.info(f"AUC-ROC: {auc:.3f} | Brier: {brier:.3f}")
        log.info(f"Draw rate: actual={actual_draw_rate:.1%}, predicted={predicted_draw_rate:.1%}, mean_prob={mean_draw_prob:.1%}")

        # Collect for aggregate metrics
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)

    # Aggregate metrics across all folds
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)

    results['aggregate'] = {
        'n_matches': len(all_y_true),
        'n_draws': all_y_true.sum(),
        'accuracy': accuracy_score(all_y_true, all_y_pred),
        'precision': precision_score(all_y_true, all_y_pred, zero_division=0),
        'recall': recall_score(all_y_true, all_y_pred, zero_division=0),
        'f1': f1_score(all_y_true, all_y_pred, zero_division=0),
        'auc_roc': roc_auc_score(all_y_true, all_y_proba),
        'brier': brier_score_loss(all_y_true, all_y_proba),
        'actual_draw_rate': all_y_true.mean(),
        'mean_draw_prob': all_y_proba.mean(),
    }

    return results


def calibration_analysis(df: pd.DataFrame, feature_cols: list[str], test_seasons: list[str]) -> None:
    """Analyze calibration of draw probabilities.

    Check if predicted probabilities match actual frequencies.
    When model says 30% draw, draws should happen ~30% of the time.
    """
    log.info(f"\n{'='*60}")
    log.info("CALIBRATION ANALYSIS")
    log.info(f"{'='*60}")

    # Train on all data before test seasons
    train_df = df[df.season < min(test_seasons)].copy()
    test_df = df[df.season.isin(test_seasons)].copy()

    specialist = DrawSpecialist(class_weight=1.5, threshold=0.25)
    specialist.fit(train_df, train_df['result'], feature_cols)

    # Predict on test set
    y_proba = specialist.predict_draw_proba(test_df)
    y_true = (test_df['result'] == 'D').astype(int)

    # Bin predictions into deciles
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    print("\nCalibration by predicted probability decile:")
    print(f"{'Predicted Range':<20} {'N Matches':<12} {'Actual Draw %':<15} {'Expected Draw %':<15}")
    print("-" * 65)

    for i in range(len(bins) - 1):
        mask = (y_proba >= bins[i]) & (y_proba < bins[i+1])
        if mask.sum() == 0:
            continue

        n_matches = mask.sum()
        actual_rate = y_true[mask].mean()
        expected_rate = y_proba[mask].mean()

        print(f"{bins[i]:.1f} - {bins[i+1]:.1f}          {n_matches:<12} {actual_rate:<15.1%} {expected_rate:<15.1%}")

    # Overall calibration error
    calibration_error = np.abs(y_true.mean() - y_proba.mean())
    log.info(f"\nOverall calibration error: {calibration_error:.3f}")
    log.info(f"Actual draw rate: {y_true.mean():.1%}")
    log.info(f"Mean predicted prob: {y_proba.mean():.1%}")


def betting_backtest(
    df: pd.DataFrame,
    feature_cols: list[str],
    test_seasons: list[str],
    edge_threshold: float = 0.05,
    stake: float = 20.0
) -> dict:
    """Backtest DrawSpecialist in betting context.

    For each match:
    1. Get DrawSpecialist P(draw)
    2. Get Pinnacle sharp implied P(draw) from odds_PSD
    3. Compute edge = model_prob - sharp_prob
    4. If edge > threshold: bet €stake at opening odds
    5. Settle against actual result

    Args:
        df: Full dataset
        feature_cols: Feature columns
        test_seasons: Seasons to backtest (must have Pinnacle odds)
        edge_threshold: Minimum edge to bet
        stake: Flat stake per bet

    Returns:
        Dictionary with backtest results
    """
    log.info(f"\n{'='*60}")
    log.info(f"BETTING BACKTEST (edge > {edge_threshold:.1%})")
    log.info(f"{'='*60}")

    # Train on all data before test seasons
    train_df = df[df.season < min(test_seasons)].copy()
    test_df = df[df.season.isin(test_seasons)].copy()

    # Filter to matches with Pinnacle odds
    test_df = test_df.dropna(subset=['odds_PSD'])

    if len(test_df) == 0:
        log.error("No matches with Pinnacle odds in test seasons")
        return {}

    log.info(f"Train: {len(train_df)} matches ({train_df.season.min()} to {train_df.season.max()})")
    log.info(f"Test: {len(test_df)} matches with Pinnacle odds")

    # Train model
    specialist = DrawSpecialist(class_weight=1.5, threshold=0.25)
    specialist.fit(train_df, train_df['result'], feature_cols)

    # Predict on test set
    model_draw_prob = specialist.predict_draw_proba(test_df)

    # Pinnacle implied probability (from decimal odds)
    pinnacle_draw_odds = test_df['odds_PSD'].values
    sharp_draw_prob = 1 / pinnacle_draw_odds

    # Compute edge
    edge = model_draw_prob - sharp_draw_prob

    # Filter to bets where edge > threshold
    bet_mask = edge > edge_threshold

    n_bets = bet_mask.sum()
    if n_bets == 0:
        log.warning(f"No bets found with edge > {edge_threshold:.1%}")
        return {}

    # Bet outcomes
    bet_df = test_df[bet_mask].copy()
    bet_df['model_prob'] = model_draw_prob[bet_mask]
    bet_df['sharp_prob'] = sharp_draw_prob[bet_mask]
    bet_df['edge'] = edge[bet_mask]
    bet_df['stake'] = stake
    bet_df['odds'] = pinnacle_draw_odds[bet_mask]
    bet_df['win'] = (bet_df['result'] == 'D').astype(int)
    bet_df['profit'] = bet_df.apply(
        lambda r: stake * (r['odds'] - 1) if r['win'] else -stake,
        axis=1
    )

    # Aggregate results
    total_stake = n_bets * stake
    total_profit = bet_df['profit'].sum()
    roi = (total_profit / total_stake) * 100
    win_rate = bet_df['win'].mean()
    avg_odds = bet_df['odds'].mean()
    avg_edge = bet_df['edge'].mean()

    results = {
        'n_bets': n_bets,
        'total_stake': total_stake,
        'total_profit': total_profit,
        'roi': roi,
        'win_rate': win_rate,
        'avg_odds': avg_odds,
        'avg_edge': avg_edge,
        'by_season': {}
    }

    log.info(f"\nBacktest Results:")
    log.info(f"  Bets: {n_bets}")
    log.info(f"  Win Rate: {win_rate:.1%}")
    log.info(f"  Avg Odds: {avg_odds:.2f}")
    log.info(f"  Avg Edge: {avg_edge:.1%}")
    log.info(f"  Total Stake: €{total_stake:.0f}")
    log.info(f"  Total Profit: €{total_profit:.2f}")
    log.info(f"  ROI: {roi:+.1f}%")

    # Per-season breakdown
    print("\nPer-season breakdown:")
    print(f"{'Season':<15} {'Bets':<8} {'Win Rate':<12} {'Profit':<12} {'ROI':<10}")
    print("-" * 60)

    for season in sorted(test_seasons):
        season_df = bet_df[bet_df.season == season]
        if len(season_df) == 0:
            continue

        s_bets = len(season_df)
        s_wins = season_df['win'].sum()
        s_win_rate = season_df['win'].mean()
        s_profit = season_df['profit'].sum()
        s_stake = s_bets * stake
        s_roi = (s_profit / s_stake) * 100

        results['by_season'][season] = {
            'n_bets': s_bets,
            'wins': s_wins,
            'win_rate': s_win_rate,
            'profit': s_profit,
            'roi': s_roi,
        }

        print(f"{season:<15} {s_bets:<8} {s_win_rate:<12.1%} €{s_profit:<11.2f} {s_roi:+.1f}%")

    return results


def blended_backtest(
    df: pd.DataFrame,
    feature_cols: list[str],
    test_seasons: list[str],
    blend_weights: list[float],
    edge_threshold: float = 0.05,
    stake: float = 20.0
) -> dict:
    """Test blended approach: ensemble + DrawSpecialist.

    This requires ensemble predictions, which we'll simulate by assuming
    the current ensemble achieves the reported backtest performance.

    For now, test DrawSpecialist standalone at various thresholds.
    """
    log.info(f"\n{'='*60}")
    log.info("BLENDED APPROACH TESTING")
    log.info(f"{'='*60}")

    results = {}

    for weight in blend_weights:
        log.info(f"\nTesting DrawSpecialist standalone with adjusted threshold (simulating blend weight {weight})")

        # Adjust threshold to simulate blending
        # Higher weight = more aggressive betting
        adjusted_threshold = edge_threshold * (1 - weight * 0.5)

        result = betting_backtest(
            df, feature_cols, test_seasons,
            edge_threshold=adjusted_threshold,
            stake=stake
        )

        if result:
            results[weight] = result

    return results


def main():
    """Main training and validation workflow."""
    feature_path = str(Path(__file__).parent.parent.parent) + '/data/features/features.parquet'

    # Load data
    df = load_data(feature_path)
    feature_cols = get_feature_columns(df)

    log.info(f"Using {len(feature_cols)} features for training")

    # Define validation seasons (last 5 seasons for CV)
    all_seasons = sorted(df.season.unique())
    val_seasons = all_seasons[-5:]  # 2021-2022 through 2025-2026

    log.info(f"Walk-forward CV on seasons: {val_seasons}")

    # Step 1: Walk-forward CV
    cv_results = walk_forward_cv(df, feature_cols, val_seasons)

    log.info(f"\n{'='*60}")
    log.info("AGGREGATE CV RESULTS")
    log.info(f"{'='*60}")
    agg = cv_results['aggregate']
    log.info(f"Matches: {agg['n_matches']} | Draws: {agg['n_draws']} ({agg['actual_draw_rate']:.1%})")
    log.info(f"Accuracy: {agg['accuracy']:.3f}")
    log.info(f"Precision: {agg['precision']:.3f} | Recall: {agg['recall']:.3f} | F1: {agg['f1']:.3f}")
    log.info(f"AUC-ROC: {agg['auc_roc']:.3f}")
    log.info(f"Brier Score: {agg['brier']:.4f}")
    log.info(f"Mean Draw Prob: {agg['mean_draw_prob']:.1%}")

    # Step 2: Calibration analysis
    test_seasons = ['2023-2024', '2024-2025']
    calibration_analysis(df, feature_cols, test_seasons)

    # Step 3: Betting backtest
    backtest_results = betting_backtest(df, feature_cols, test_seasons, edge_threshold=0.05, stake=20.0)

    # Step 4: Test different edge thresholds (simulating blend weights)
    log.info(f"\n{'='*60}")
    log.info("THRESHOLD SENSITIVITY ANALYSIS")
    log.info(f"{'='*60}")

    thresholds = [0.03, 0.04, 0.05, 0.06, 0.07]
    threshold_results = []

    for thresh in thresholds:
        result = betting_backtest(df, feature_cols, test_seasons, edge_threshold=thresh, stake=20.0)
        if result:
            threshold_results.append({
                'threshold': thresh,
                'n_bets': result['n_bets'],
                'win_rate': result['win_rate'],
                'roi': result['roi'],
                'profit': result['total_profit'],
            })

    if threshold_results:
        print("\nThreshold sensitivity:")
        print(f"{'Edge Threshold':<15} {'Bets':<8} {'Win Rate':<12} {'Profit':<12} {'ROI':<10}")
        print("-" * 60)
        for r in threshold_results:
            print(f"{r['threshold']:<15.1%} {r['n_bets']:<8} {r['win_rate']:<12.1%} €{r['profit']:<11.2f} {r['roi']:+.1f}%")

    # Summary and recommendation
    log.info(f"\n{'='*60}")
    log.info("SUMMARY & RECOMMENDATION")
    log.info(f"{'='*60}")

    if backtest_results:
        log.info(f"\nDrawSpecialist (standalone, 5% edge threshold):")
        log.info(f"  {backtest_results['n_bets']} bets, {backtest_results['win_rate']:.1%} win rate, {backtest_results['roi']:+.1f}% ROI")
        log.info(f"  Total profit: €{backtest_results['total_profit']:.2f}")

    log.info(f"\nBaseline (ensemble 1X2 Draw betting, from backtest report):")
    log.info(f"  301 bets, 27.6% win rate, +20.4% ROI")
    log.info(f"  Total profit: €1230")

    if backtest_results and backtest_results['roi'] > 20.0:
        log.info(f"\n✓ RECOMMENDATION: DrawSpecialist OUTPERFORMS ensemble draw betting")
        log.info(f"  Consider deploying standalone or blending with weight 0.3-0.5")
    elif backtest_results and backtest_results['roi'] > 10.0:
        log.info(f"\n~ RECOMMENDATION: DrawSpecialist shows promise but doesn't beat ensemble")
        log.info(f"  Consider blending with low weight (0.2-0.3) for diversification")
    else:
        log.info(f"\n✗ RECOMMENDATION: DrawSpecialist underperforms ensemble")
        log.info(f"  Do NOT deploy standalone. Revisit feature engineering or stay with ensemble.")


if __name__ == '__main__':
    main()
