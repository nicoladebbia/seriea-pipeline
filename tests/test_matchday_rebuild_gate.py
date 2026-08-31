"""Pin the two mechanisms that killed the first live T-30 ticket (2026-08-28).

Milan vs Venezia, go-live day: the pre-kickoff subprocess was killed by its
120s timeout twice (14:12, 14:29) — sized for a "~25s" flow whose odds
extras loop alone runs ~5 min — so no bets were committed, no order ticket
and no no-action notice fired. Meanwhile the settlement tick (every 5 min)
ran build_features(use_cache=False) — a ~23-minute full rebuild — even when
it ingested zero rows, keeping the box saturated all afternoon.

Two pins:
1. The rebuild gate: zero rows ingested => no feature rebuild.
2. The timeout floor: PRE_KICKOFF_TIMEOUT_SEC must never shrink back below
   the measured runtime of the real flow.
"""

from __future__ import annotations

from scripts.data.matchday_updater import _should_rebuild_features
from scripts.pipeline.scheduler import PRE_KICKOFF_TIMEOUT_SEC


def test_nothing_ingested_skips_the_rebuild():
    assert _should_rebuild_features({"matches_parquet_added": 0, "matches_fetched": 0}) is False
    assert _should_rebuild_features({}) is False  # keys absent on early paths
    assert _should_rebuild_features({"matches_parquet_added": None, "matches_fetched": None}) is False


def test_real_ingest_still_rebuilds():
    assert _should_rebuild_features({"matches_parquet_added": 1, "matches_fetched": 0}) is True
    assert _should_rebuild_features({"matches_parquet_added": 0, "matches_fetched": 2}) is True
    # fallback ingest path bumps matches_parquet_added without matches_fetched
    assert _should_rebuild_features({"matches_parquet_added": 3}) is True


def test_pre_kickoff_timeout_floor():
    """120s killed the commit twice on 2026-08-28. The measured flow is
    ~5-9 min (extras loop + ensemble re-predict). Anyone lowering this back
    below 600s is re-arming the exact go-live-day miss."""
    assert PRE_KICKOFF_TIMEOUT_SEC >= 600
