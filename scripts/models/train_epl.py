#!/usr/bin/env python3
"""Train ML ensemble for the English Premier League.

This is a thin convenience wrapper around train_league_model() with EPL defaults.
All methodology is identical to Serie A: walk-forward CV, Optuna tuning,
3-model ensemble (XGB + LGB + CB), time-decay=0.85, auto draw weights.

Models are saved to data/models/premier_league/ -- completely separate from
Serie A models in data/models/universal/.

Usage:
    # Full training
    python -m scripts.models.train_epl

    # With custom tuning budget
    python -m scripts.models.train_epl --trials 60

    # Dry run: check if EPL data exists in features.parquet
    python -m scripts.models.train_epl --dry-run

    # Exclude odds features (pure performance model)
    python -m scripts.models.train_epl --exclude-odds
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.models.train_league import dry_run, train_league_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

LEAGUE = "premier_league"


def main():
    parser = argparse.ArgumentParser(
        description="Train ML ensemble for the English Premier League",
    )
    parser.add_argument(
        "--trials", type=int, default=40,
        help="Optuna tuning trials per model (default: 40)",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Max features after importance selection",
    )
    parser.add_argument(
        "--corr", type=float, default=None,
        help="Correlation pruning threshold",
    )
    parser.add_argument(
        "--exclude-odds", action="store_true",
        help="Exclude betting-odds-derived features",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Check if EPL data exists, report stats, don't train",
    )

    args = parser.parse_args()

    if args.dry_run:
        results = dry_run(LEAGUE)
        if not results.get("found", False):
            log.warning(
                "EPL data not found in features.parquet. "
                "The scraper agent needs to populate it first. "
                "Run the EPL data pipeline to generate features, then retry."
            )
        return results

    log.info("=" * 60)
    log.info("PREMIER LEAGUE MODEL TRAINING")
    log.info("=" * 60)
    log.info("Training a SEPARATE EPL model (not mixed with Serie A)")
    log.info("Models will be saved to data/models/%s/", LEAGUE)
    log.info("Methodology: same as Serie A (walk-forward CV, Optuna, ensemble)")

    results = train_league_model(
        league=LEAGUE,
        n_tune_trials=args.trials,
        top_k_features=args.top_k,
        corr_threshold=args.corr,
        exclude_odds=args.exclude_odds,
    )

    return results


if __name__ == "__main__":
    main()
