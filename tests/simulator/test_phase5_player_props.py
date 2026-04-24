"""Tests for Phase 5 player props."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.simulator.base_rates.player_profiles import (
    PlayerProfile,
    PlayerProfileStore,
    _positional_prior,
)
from models.simulator.base_rates.lineup_allocator import allocate_team_shots_to_players
from models.simulator.engine.simulator import simulate_match
from models.simulator.markets import all_player_market_probs


def _fake_player_match_stats(n_players: int = 20, n_matches: int = 15, seed: int = 7):
    """Build a synthetic player_match_stats DataFrame."""
    rng = np.random.default_rng(seed)
    rows = []
    # Striker: high shot rate
    # Defender: low shot rate
    positions = ["F"] * 3 + ["W"] * 4 + ["M"] * 7 + ["D"] * 5 + ["G"] * 1
    positions = positions[:n_players]
    for pid in range(1, n_players + 1):
        pos = positions[pid - 1]
        shot_rate = {"F": 2.5, "W": 1.5, "M": 0.8, "D": 0.3, "G": 0.0}[pos]
        for m in range(n_matches):
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=7 * m)
            mins = int(rng.integers(60, 95))
            total_shots = rng.poisson(shot_rate)
            sot = rng.binomial(total_shots, 0.35) if total_shots > 0 else 0
            goals = rng.binomial(sot, 0.15) if sot > 0 else 0
            xg = float(total_shots * 0.08 + rng.normal(0, 0.05))
            rows.append({
                "player_id": pid, "player_name": f"P{pid}",
                "team": "TeamA",
                "position": pos,
                "date": date, "match_id": 1000 + m,
                "is_starter": True, "minutes": mins,
                "total_shots": total_shots, "shots_on_target": sot,
                "goals": goals, "xg": xg,
                "fouls": rng.poisson(1.0),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PlayerProfileStore
# ---------------------------------------------------------------------------

def test_positional_prior_sane_values():
    for pos, expected_shot in [("F", 2.5), ("W", 1.8), ("M", 0.9), ("D", 0.3), ("G", 0.0)]:
        prior = _positional_prior(pos)
        assert prior.shot_rate_per_90 == expected_shot
        assert prior.is_fallback


def test_profile_store_fits_basic():
    ps = PlayerProfileStore()
    df = _fake_player_match_stats()
    ps.fit(df)
    assert ps.n_profiles == 20
    # Forwards should have higher shot rates than defenders
    forwards = [p for p in ps.all_profiles().values() if p.position == "F"]
    defenders = [p for p in ps.all_profiles().values() if p.position == "D"]
    assert forwards[0].shot_rate_per_90 > defenders[0].shot_rate_per_90


def test_profile_store_fallback_for_low_starts():
    ps = PlayerProfileStore(min_starts=10)
    df = _fake_player_match_stats(n_players=3, n_matches=5)
    ps.fit(df)
    # All players have 5 starts, below min_starts=10 → all fallback
    for prof in ps.all_profiles().values():
        assert prof.is_fallback


def test_profile_store_lookup_returns_positional_for_unknown():
    ps = PlayerProfileStore()
    ps.fit(_fake_player_match_stats())
    prof = ps.lookup(99999)  # unseen player
    assert prof.is_fallback


def test_profile_fits_respect_as_of_date():
    ps = PlayerProfileStore()
    df = _fake_player_match_stats()
    # Truncate to first 3 matches
    cutoff = pd.Timestamp("2024-01-15")
    ps.fit(df, as_of_date=cutoff)
    # With only ~3 matches per player, most should fall below min_starts=3 or near it
    total = len(ps.all_profiles())
    # Stats should exist but may use fewer starts
    for prof in ps.all_profiles().values():
        assert prof.n_starts_used <= 3


# ---------------------------------------------------------------------------
# Lineup allocator
# ---------------------------------------------------------------------------

def test_allocator_sums_to_team_rate():
    ps = PlayerProfileStore()
    ps.fit(_fake_player_match_stats())
    lineup = list(ps.all_profiles().keys())[:11]
    shares = allocate_team_shots_to_players(team_shot_rate=12.0, lineup_player_ids=lineup, profiles=ps)
    total = sum(shares.values())
    assert abs(total - 12.0) < 0.1, f"expected ~12, got {total}"


def test_allocator_skips_goalkeeper():
    ps = PlayerProfileStore()
    ps.fit(_fake_player_match_stats())
    lineup = list(ps.all_profiles().keys())
    shares = allocate_team_shots_to_players(12.0, lineup, ps)
    # GK (position G) should have 0 shots
    for pid, s in shares.items():
        prof = ps.lookup(pid)
        if prof.position == "G":
            assert s == 0.0


def test_allocator_empty_lineup_returns_empty():
    ps = PlayerProfileStore()
    assert allocate_team_shots_to_players(12.0, [], ps) == {}


def test_allocator_zero_rate_returns_empty():
    ps = PlayerProfileStore()
    ps.fit(_fake_player_match_stats())
    assert allocate_team_shots_to_players(0.0, [1, 2, 3], ps) == {}


# ---------------------------------------------------------------------------
# Simulator integration
# ---------------------------------------------------------------------------

def test_simulate_match_without_players_leaves_dicts_empty():
    sim = simulate_match(1.3, 1.1, n_trials=500, seed=42)
    assert sim.player_shots == {}
    assert sim.player_goals == {}


def test_simulate_match_with_player_data_populates_dicts():
    ps = PlayerProfileStore()
    ps.fit(_fake_player_match_stats())
    profiles_dict = ps.all_profiles()
    shares = allocate_team_shots_to_players(
        12.0, list(profiles_dict.keys())[:11], ps
    )
    sim = simulate_match(
        1.3, 1.1, n_trials=5000, seed=42,
        player_profiles_home=profiles_dict,
        player_shot_shares_home=shares,
    )
    assert len(sim.player_shots) > 0
    assert len(sim.player_goals) > 0
    # Shot sum ≈ team rate
    per_trial_sum = sum(sim.player_shots[k] for k in sim.player_shots)
    assert abs(per_trial_sum.mean() - 12.0) < 1.0


def test_p_player_anytime_scorer_in_range():
    ps = PlayerProfileStore()
    ps.fit(_fake_player_match_stats())
    profiles_dict = ps.all_profiles()
    shares = allocate_team_shots_to_players(
        12.0, list(profiles_dict.keys())[:11], ps
    )
    sim = simulate_match(
        1.3, 1.1, n_trials=5000, seed=42,
        player_profiles_home=profiles_dict,
        player_shot_shares_home=shares,
    )
    for pid in shares:
        p = sim.p_player_anytime_scorer(pid)
        assert 0.0 <= p <= 1.0


def test_top_scorers_sorted_descending():
    ps = PlayerProfileStore()
    ps.fit(_fake_player_match_stats())
    profiles_dict = ps.all_profiles()
    # Give all shots to two players to force ordering
    shares = {list(profiles_dict.keys())[0]: 6.0, list(profiles_dict.keys())[1]: 3.0}
    sim = simulate_match(
        1.3, 1.1, n_trials=5000, seed=42,
        player_profiles_home=profiles_dict,
        player_shot_shares_home=shares,
    )
    top = sim.top_scorers(k=2)
    assert len(top) == 2
    assert top[0][1] >= top[1][1]


def test_all_player_market_probs_emits_expected_keys():
    ps = PlayerProfileStore()
    ps.fit(_fake_player_match_stats())
    profiles_dict = ps.all_profiles()
    shares = allocate_team_shots_to_players(
        12.0, list(profiles_dict.keys())[:5], ps
    )
    sim = simulate_match(
        1.3, 1.1, n_trials=2000, seed=42,
        player_profiles_home=profiles_dict,
        player_shot_shares_home=shares,
    )
    out = all_player_market_probs(sim)
    assert len(out) > 0
    # At least one anytime_scorer key
    assert any("anytime_scorer" in k for k in out)
    # All probs in [0,1]
    for k, v in out.items():
        assert 0.0 <= v <= 1.0


def test_p_player_shots_over_plausible():
    ps = PlayerProfileStore()
    ps.fit(_fake_player_match_stats())
    profiles_dict = ps.all_profiles()
    # Striker with high shot share
    strikers = [pid for pid, p in profiles_dict.items() if p.position == "F"]
    shares = {strikers[0]: 3.0}
    sim = simulate_match(
        1.3, 1.1, n_trials=10_000, seed=42,
        player_profiles_home=profiles_dict,
        player_shot_shares_home=shares,
    )
    # Expected 3 shots → P(>0.5) high, P(>2.5) mid, P(>5.5) low
    p_05 = sim.p_player_shots_over(strikers[0], 0.5)
    p_55 = sim.p_player_shots_over(strikers[0], 5.5)
    assert p_05 > 0.8
    assert p_55 < 0.20
