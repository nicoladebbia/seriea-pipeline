"""Train DrawSpecialist on full dataset and save for production.

This script trains the final production model using all available data
(2005-2025) and saves it to data/models/universal/draw_specialist.pkl.

Based on validation results:
- 22.7% ROI (479 bets) at 5% edge threshold
- 25.8% ROI (459 bets) at 6% edge threshold
- Outperforms ensemble draw betting by +2.3pp ROI
"""

import logging
import pickle
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.draw_specialist import DrawSpecialist

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def main():
    """Train and save production DrawSpecialist model."""
    log.info("=" * 60)
    log.info("TRAINING PRODUCTION DrawSpecialist MODEL")
    log.info("=" * 60)

    # Load full dataset
    feature_path = '/Users/nicoladebbia/Code_Ideas/Scripts/AI_PREDICTIONS/seriea_pipeline/data/features/features.parquet'
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
    log.info(f"Seasons: {df.season.min()} to {df.season.max()}")
    log.info(f"Draw rate: {(df.result == 'D').mean():.1%}")

    # Get feature columns
    exclude = [
        'match_id', 'home_team', 'away_team', 'home_score', 'away_score',
        'match_date', 'season', 'result', 'league', 'gameweek'
    ]
    feature_cols = [
        c for c in df.columns
        if c not in exclude and df[c].dtype in ['float64', 'int64', 'float32', 'int32']
    ]

    log.info(f"Using {len(feature_cols)} features for training")

    # Train model
    log.info("\nTraining DrawSpecialist...")
    specialist = DrawSpecialist(class_weight=1.5, threshold=0.25)
    specialist.fit(df, df['result'], feature_cols)

    # Save model
    output_path = '/Users/nicoladebbia/Code_Ideas/Scripts/AI_PREDICTIONS/seriea_pipeline/data/models/universal/draw_specialist.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(specialist, f)

    log.info(f"\n✓ Model saved to: {output_path}")

    # Verify model can be loaded
    with open(output_path, 'rb') as f:
        loaded_model = pickle.load(f)

    log.info(f"✓ Model verification: {len(loaded_model.feature_names)} features")
    log.info(f"✓ Model type: {type(loaded_model.model).__name__}")

    # Test prediction on a sample
    sample = df.tail(10)
    probs = loaded_model.predict_draw_proba(sample)
    log.info(f"\nSample predictions (last 10 matches):")
    log.info(f"  Draw probabilities: {probs}")
    log.info(f"  Mean: {probs.mean():.1%}, Min: {probs.min():.1%}, Max: {probs.max():.1%}")

    log.info("\n" + "=" * 60)
    log.info("PRODUCTION MODEL READY")
    log.info("=" * 60)
    log.info("\nDeployment checklist:")
    log.info("  1. ✓ Model trained on full dataset (2005-2025)")
    log.info(f"  2. ✓ Model saved to {output_path}")
    log.info("  3. ✓ Model verified and loadable")
    log.info("  4. [ ] Update betting_unified.py to load model")
    log.info("  5. [ ] Set edge threshold to 6% (or 5% for higher volume)")
    log.info("  6. [ ] Run backtest_auditor to verify integration")
    log.info("  7. [ ] Deploy with €15 stakes initially")
    log.info("\nExpected performance (based on validation):")
    log.info("  - 6% edge threshold: 25.8% ROI, 459 bets per 2 seasons")
    log.info("  - 5% edge threshold: 22.7% ROI, 479 bets per 2 seasons")


if __name__ == '__main__':
    main()
