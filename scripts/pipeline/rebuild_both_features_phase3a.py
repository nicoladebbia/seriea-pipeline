"""Phase 3a rebuild: SA + EPL features with xi_quality plugin.

Rebuilds both league parquets. Relies on per-plugin cache for steps 1-25
(unchanged), runs xi_quality + everything downstream from scratch.
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

from features.build import _build_features_for_matches  # noqa: E402
from storage.paths import features_path, parsed_path  # noqa: E402


def main() -> None:
    matches = pd.read_parquet(parsed_path("matches"))
    matches = matches.drop_duplicates(
        subset=["home_team", "away_team", "match_date", "season"], keep="first"
    )
    if "league" not in matches.columns:
        matches["league"] = "serie_a"

    for lg in ["serie_a", "premier_league"]:
        subset = matches[matches["league"] == lg].copy()
        log.info("=" * 72)
        log.info("Rebuilding %s: %d matches", lg, len(subset))
        log.info("=" * 72)
        features = _build_features_for_matches(
            subset, season=None, use_cache=True, league=lg,
        )
        out = features_path(league=lg)
        features.to_parquet(out, index=False)
        log.info("Wrote %s: %d × %d", out, len(features), len(features.columns))
        xi_cols = [c for c in features.columns if c.startswith("home_xi_") or c.startswith("away_xi_")]
        for c in sorted(xi_cols):
            log.info("  %-38s  all=%5.1f%%  2017+=%5.1f%%",
                     c,
                     features[c].notna().mean() * 100,
                     features[features["season"] >= "2017-2018"][c].notna().mean() * 100)


if __name__ == "__main__":
    main()
