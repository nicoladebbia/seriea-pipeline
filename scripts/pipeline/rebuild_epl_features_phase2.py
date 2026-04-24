"""One-shot EPL feature rebuild for Phase 2.

Rebuilds ONLY features_premier_league.parquet, relying on per-plugin cache
to skip re-computing everything. The lineup_xg plugin was version-bumped
to 1.1 (league-aware Sofascore file routing), so that one plugin will
re-run; all other plugins cache-hit.

Run from project root:
    python3 scripts/pipeline/rebuild_epl_features_phase2.py
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
    epl = matches[matches["league"] == "premier_league"].copy()
    log.info("EPL matches: %d", len(epl))

    # use_cache=True: reuse all plugin caches except lineup_xg (v1.1 forces rebuild)
    features = _build_features_for_matches(
        epl, season=None, use_cache=True, league="premier_league",
    )
    out = features_path(league="premier_league")
    features.to_parquet(out, index=False)
    log.info("Wrote %s: %d rows × %d cols", out, len(features), len(features.columns))

    # Fill-rate sanity check on lineup_* columns
    lineup_cols = [c for c in features.columns
                   if c.startswith("home_lineup_") or c.startswith("away_lineup_")
                   or c == "lineup_xg_sum_diff" or c == "lineup_rating_mean_diff"
                   or c == "lineup_xa_sum_diff"]
    log.info("Lineup column fill rates (all seasons):")
    for c in sorted(lineup_cols):
        log.info("  %-40s fill=%5.1f%%", c, features[c].notna().mean() * 100)
    log.info("Lineup column fill rates (2017+ only):")
    modern = features[features["season"] >= "2017-2018"]
    for c in sorted(lineup_cols):
        log.info("  %-40s fill=%5.1f%%", c, modern[c].notna().mean() * 100)


if __name__ == "__main__":
    main()
