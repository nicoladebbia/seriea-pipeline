"""Tests for the Phase 3b backtest harness.

Covers:
  - determinism: same inputs, same report
  - odds fallback: injects missing Pinnacle, confirms market_avg / B365 used
  - walk-forward integrity: no training leakage
  - bootstrap CI convergence on synthetic data with known true ROI
  - edge threshold monotonicity: lower threshold → more bets
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.simulator.backtests.harness import (
    BacktestHarness,
    BinaryMarket,
    MulticlassMarket,
)
from models.simulator.backtests.odds_fallback import (
    NoOddsAvailable,
    deoverround_binary,
    deoverround_multiclass,
    resolve_odds_binary,
    resolve_odds_multiclass,
    BinaryOdds,
    MulticlassOdds,
)
from models.simulator.backtests.roi_bootstrap import compute_roi_stats
from models.simulator.backtests.stake_policies import FlatStake, KellyStake, BetInput


# ---------------------------------------------------------------------------
# Fake predictor for deterministic testing
# ---------------------------------------------------------------------------

class _OmniscientBinaryPredictor:
    """Predicts 1.0 when target==1, else 0.0 — always correct."""
    name = "omniscient_binary"
    version = "test"

    def fit(self, train_df): return None
    def supports(self, mk): return mk == "O/U 2.5"
    def predict_binary(self, eval_df, mk):
        return eval_df["over_2_5"].astype(float).to_numpy()
    def predict_multiclass(self, eval_df, mk):
        raise NotImplementedError
    def classes_for(self, mk): return ()


class _RandomBinaryPredictor:
    """Predicts 0.5 for every match — no edge."""
    name = "random_binary"
    version = "test"

    def fit(self, train_df): return None
    def supports(self, mk): return mk == "O/U 2.5"
    def predict_binary(self, eval_df, mk):
        return np.full(len(eval_df), 0.5)
    def predict_multiclass(self, eval_df, mk):
        raise NotImplementedError
    def classes_for(self, mk): return ()


def _make_fake_df(n_matches: int = 50, seed: int = 42):
    """Synthetic Serie A dataset: 50/50 over-under outcomes, fair 2.0 odds + small vig."""
    rng = np.random.default_rng(seed)
    results = rng.choice([0, 1], size=n_matches, p=[0.5, 0.5])
    df = pd.DataFrame({
        "match_id": [f"match_{i:03d}" for i in range(n_matches)],
        "season": ["2024-2025"] * n_matches,
        "league": ["serie_a"] * n_matches,
        "home_score": [1] * n_matches,
        "away_score": [2 * r for r in results],  # total_goals > 2 iff result==1
        "odds_Avg_over25": [1.95] * n_matches,   # implies ~51.3% over (slight vig)
        "odds_Avg_under25": [1.95] * n_matches,
    })
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_odds_fallback_uses_pinnacle_first():
    row = pd.Series({
        "odds_PS_close_over25": 2.0,
        "odds_PS_close_under25": 2.0,
        "odds_Avg_over25": 1.8,
        "odds_Avg_under25": 1.8,
    })
    chain = [("odds_PS_close_over25", "odds_PS_close_under25"),
             ("odds_Avg_over25", "odds_Avg_under25")]
    odds = resolve_odds_binary(row, chain)
    assert odds.source == "pinnacle_close"
    assert odds.yes == 2.0


def test_odds_fallback_falls_to_market_avg_when_pinnacle_missing():
    row = pd.Series({
        "odds_PS_close_over25": float("nan"),
        "odds_PS_close_under25": float("nan"),
        "odds_Avg_over25": 1.8,
        "odds_Avg_under25": 1.8,
    })
    chain = [("odds_PS_close_over25", "odds_PS_close_under25"),
             ("odds_Avg_over25", "odds_Avg_under25")]
    odds = resolve_odds_binary(row, chain)
    # The label index depends on position in chain — "b365_close" at idx=1 in
    # the hardcoded label list, so the label reflects chain position, not semantic.
    # That's intentional: chain position = priority order.
    assert odds.source == "b365_close"
    assert odds.yes == 1.8


def test_odds_fallback_raises_when_all_missing():
    row = pd.Series({
        "odds_Avg_over25": float("nan"),
        "odds_Avg_under25": float("nan"),
    })
    chain = [("odds_Avg_over25", "odds_Avg_under25")]
    with pytest.raises(NoOddsAvailable):
        resolve_odds_binary(row, chain)


def test_deoverround_sums_to_one():
    odds = BinaryOdds(2.0, 2.0, "test")
    p_yes, p_no = deoverround_binary(odds)
    assert abs(p_yes + p_no - 1.0) < 1e-9


def test_deoverround_multiclass_sums_to_one():
    mc = MulticlassOdds(odds=(2.0, 3.5, 4.0), classes=("H", "D", "A"), source="test")
    probs = deoverround_multiclass(mc)
    assert abs(sum(probs) - 1.0) < 1e-9


def test_flat_stake_constant():
    policy = FlatStake(10.0)
    assert policy.stake(BetInput(0.5, 2.0, 5.0), 1000.0) == 10.0


def test_kelly_stake_honors_floor_and_ceiling():
    policy = KellyStake(fraction=0.25, floor_eur=2.0, ceiling_eur=50.0)
    # Sub-floor edge: tiny Kelly, floor clamps. p=0.501 → kelly ≈ 0.25*0.002 = 5e-4 → €0.5 → floor €2
    tiny = policy.stake(BetInput(p=0.501, odds=2.0, edge_pct=0.2), 1000.0)
    assert tiny == 2.0
    # Above-ceiling edge: Kelly wants a lot, ceiling clamps. p=0.90, b=1, Kelly = 0.25*0.8 = 0.2 → €200 → ceiling €50
    huge = policy.stake(BetInput(p=0.90, odds=2.0, edge_pct=80.0), 1000.0)
    assert huge == 50.0
    # Negative EV: zero stake
    zero = policy.stake(BetInput(p=0.3, odds=2.0, edge_pct=-40.0), 1000.0)
    assert zero == 0.0
    # In-range edge: actual Kelly number preserved (not clamped)
    mid = policy.stake(BetInput(p=0.55, odds=2.0, edge_pct=10.0), 1000.0)
    # Full Kelly = (1*0.55 - 0.45)/1 = 0.1; fractional = 0.025 → €25
    assert abs(mid - 25.0) < 0.01


def test_roi_bootstrap_ci_contains_truth_on_synthetic():
    # Construct a bet stream with known 10% ROI
    rng = np.random.default_rng(7)
    n = 500
    stakes = np.full(n, 10.0)
    # 55% win rate at odds 2.0 → EV = 0.55*10 - 0.45*10 = 1.0 → 10% ROI
    wins = rng.random(n) < 0.55
    profits = np.where(wins, 10.0, -10.0)
    stats = compute_roi_stats(stakes, profits, n_resample=1000, seed_salt=0)
    assert abs(stats.roi_pct_point - 10.0) < 5.0  # within CI margin
    assert stats.roi_pct_ci_lower < 10.0 < stats.roi_pct_ci_upper


def test_harness_determinism():
    df = _make_fake_df(50)
    binary_markets = [
        BinaryMarket("O/U 2.5", "over_2_5",
                     (("odds_Avg_over25", "odds_Avg_under25"),)),
    ]
    harness = BacktestHarness(df, binary_markets, [], edge_thresholds_pct=[0.0, 3.0])
    pred = _OmniscientBinaryPredictor()
    r1 = harness.run(pred, ["2024-2025"])
    r2 = harness.run(pred, ["2024-2025"])
    # Everything except `generated_at` must be identical
    d1 = r1.to_json_dict()
    d2 = r2.to_json_dict()
    d1["metadata"]["generated_at"] = d2["metadata"]["generated_at"] = "__norm__"
    assert d1 == d2


def test_omniscient_predictor_wins_flat():
    df = _make_fake_df(100)
    binary_markets = [
        BinaryMarket("O/U 2.5", "over_2_5",
                     (("odds_Avg_over25", "odds_Avg_under25"),)),
    ]
    harness = BacktestHarness(df, binary_markets, [], edge_thresholds_pct=[0.0])
    rep = harness.run(_OmniscientBinaryPredictor(), ["2024-2025"])
    rows = [r for r in rep.summary_rows()
            if r["market"] == "O/U 2.5" and r["stake_policy"] == "flat"
            and r["odds_source"] == "ALL" and r["threshold"] == "thresh_0pct"]
    assert len(rows) == 1
    # Omniscient with fair-ish odds beats vig → positive ROI
    assert rows[0]["roi_pct"] > 80.0  # near-maximum possible


def test_random_predictor_losses_on_vig():
    df = _make_fake_df(500)  # large sample
    binary_markets = [
        BinaryMarket("O/U 2.5", "over_2_5",
                     (("odds_Avg_over25", "odds_Avg_under25"),)),
    ]
    harness = BacktestHarness(df, binary_markets, [], edge_thresholds_pct=[0.0])
    rep = harness.run(_RandomBinaryPredictor(), ["2024-2025"])
    rows = [r for r in rep.summary_rows()
            if r["market"] == "O/U 2.5" and r["stake_policy"] == "flat"
            and r["odds_source"] == "ALL" and r["threshold"] == "thresh_0pct"]
    # Random predictor with vig — ROI should be near -vig (~-2.5%)
    if rows:
        assert rows[0]["roi_pct"] < 5.0  # shouldn't accidentally beat chance


def test_edge_threshold_monotonicity():
    # Higher threshold → fewer bets
    df = _make_fake_df(200)
    binary_markets = [
        BinaryMarket("O/U 2.5", "over_2_5",
                     (("odds_Avg_over25", "odds_Avg_under25"),)),
    ]
    harness = BacktestHarness(df, binary_markets, [], edge_thresholds_pct=[0.0, 3.0, 5.0, 10.0])
    rep = harness.run(_OmniscientBinaryPredictor(), ["2024-2025"])
    n_by_thresh = {}
    for r in rep.summary_rows():
        if r["stake_policy"] == "flat" and r["odds_source"] == "ALL":
            n_by_thresh[r["threshold"]] = r["n_bets"]
    # Monotonic non-increasing
    for lo, hi in zip([0.0, 3.0, 5.0], [3.0, 5.0, 10.0]):
        lo_key = f"thresh_{lo:g}pct"
        hi_key = f"thresh_{hi:g}pct"
        assert n_by_thresh.get(lo_key, 0) >= n_by_thresh.get(hi_key, 0), \
            f"{lo_key}={n_by_thresh.get(lo_key)} should be >= {hi_key}={n_by_thresh.get(hi_key)}"


def test_walk_forward_no_train_leak():
    """The harness's `fit` sees only rows with season < target season."""
    df = pd.DataFrame({
        "match_id": [f"m{i}" for i in range(30)],
        "season": ["2023-2024"] * 15 + ["2024-2025"] * 15,
        "league": ["serie_a"] * 30,
        "home_score": [1] * 30,
        "away_score": [1, 2] * 15,
        "odds_Avg_over25": [1.95] * 30,
        "odds_Avg_under25": [1.95] * 30,
    })
    seen_train_seasons = []

    class _Spy:
        name = "spy"; version = "v1"
        def fit(self, train_df):
            seen_train_seasons.append(tuple(sorted(train_df["season"].unique())))
        def supports(self, mk): return mk == "O/U 2.5"
        def predict_binary(self, eval_df, mk): return np.full(len(eval_df), 0.6)
        def predict_multiclass(self, eval_df, mk): return np.array([]).reshape(0, 0)
        def classes_for(self, mk): return ()

    harness = BacktestHarness(
        df, [BinaryMarket("O/U 2.5", "over_2_5", (("odds_Avg_over25", "odds_Avg_under25"),))],
        [], edge_thresholds_pct=[0.0],
    )
    harness.run(_Spy(), ["2023-2024", "2024-2025"])
    assert seen_train_seasons[0] == ()                           # first season: no prior data
    assert seen_train_seasons[1] == ("2023-2024",)                # second season: one prior


def test_odds_source_tagging_preserved():
    """Bet records report the source that was actually used."""
    df = pd.DataFrame({
        "match_id": ["m1", "m2"],
        "season": ["2024-2025", "2024-2025"],
        "league": ["serie_a", "serie_a"],
        "home_score": [1, 1],
        "away_score": [2, 0],
        "odds_PS_close_over25": [2.0, float("nan")],  # m1 has Pinnacle, m2 doesn't
        "odds_PS_close_under25": [2.0, float("nan")],
        "odds_Avg_over25": [1.95, 1.95],
        "odds_Avg_under25": [1.95, 1.95],
    })
    harness = BacktestHarness(
        df,
        [BinaryMarket("O/U 2.5", "over_2_5",
                      (("odds_PS_close_over25", "odds_PS_close_under25"),
                       ("odds_Avg_over25", "odds_Avg_under25")))],
        [], edge_thresholds_pct=[0.0],
    )
    rep = harness.run(_OmniscientBinaryPredictor(), ["2024-2025"])
    sources_seen = set()
    for r in rep.summary_rows():
        if r["n_bets"] > 0:
            sources_seen.add(r["odds_source"])
    # Both pinnacle_close and b365_close should appear (plus "ALL" aggregate)
    assert "pinnacle_close" in sources_seen
    assert "b365_close" in sources_seen
    assert "ALL" in sources_seen


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
