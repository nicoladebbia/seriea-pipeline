"""Tests for the net-squad-delta transfer feature (features/transfer_impact_analysis).

Locks the two subtle behaviors that make this feature correct:
  1. Materiality weighting — end-of-loan RETURNS count at a reduced weight; paid,
     free and fresh loan moves count at full weight.
  2. Loan-return double-count guard — a player who returns from loan AND also
     leaves the same window is counted only on the departure side.

Uses synthetic parquets in a tmp dir so the test never depends on live scrapes.
"""

from __future__ import annotations

import pandas as pd
import pytest

from features.transfer_impact_analysis import (
    _loan_to_permanent_outs,
    _transfer_materiality,
    compute_january_window_features,
    compute_net_squad_delta,
)


def test_materiality_classification() -> None:
    assert _transfer_materiality("End of loan30/06/2026", False)[0] == "loan_return"
    assert _transfer_materiality("End of loan30/06/2026", False)[1] == pytest.approx(0.3)
    assert _transfer_materiality("loan transfer", True) == ("loan_move", 1.0)
    assert _transfer_materiality("€42.75m", False) == ("paid", 1.0)
    assert _transfer_materiality("free transfer", False) == ("free", 1.0)
    assert _transfer_materiality("-", False) == ("free", 1.0)


def _write_transfers(tmp_path, rows) -> None:
    d = tmp_path / "external" / "transfermarkt"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(d / "transfers_2026_2027.parquet", index=False)


def test_loan_return_discounted(tmp_path) -> None:
    # One paid arrival, one end-of-loan return (should count at 0.3x), no departures.
    _write_transfers(tmp_path, [
        {"team": "TestFC", "transfer_type": "in", "player_name": "Paid Signing",
         "age": 25, "fee_text": "€20.00m", "fee_eur": 20e6, "is_loan": False},
        {"team": "TestFC", "transfer_type": "in", "player_name": "Loan Returnee",
         "age": 24, "fee_text": "End of loan30/06/2026", "fee_eur": None, "is_loan": True},
    ])
    d = compute_net_squad_delta(season="2026-2027", tm_dir=tmp_path / "external" / "transfermarkt")
    club = d["testfc"]
    # arrivals_weight = w(paid)*1.0 + w(returnee)*0.3; both unknown players use the
    # 0.15 floor, so 0.15 + 0.15*0.3 = 0.195. Departures 0 → net positive & small.
    assert club["material_in"] == 2
    assert club["material_out"] == 0
    assert club["arrivals_weight"] == pytest.approx(0.15 + 0.15 * 0.3, abs=1e-6)
    assert club["net_squad_delta"] > 0


def test_loan_return_and_leaves_counted_once(tmp_path) -> None:
    # A player returns from loan AND is sold the same window → only the OUT counts.
    _write_transfers(tmp_path, [
        {"team": "TestFC", "transfer_type": "in", "player_name": "Churn Player",
         "age": 23, "fee_text": "End of loan30/06/2026", "fee_eur": None, "is_loan": True},
        {"team": "TestFC", "transfer_type": "out", "player_name": "Churn Player",
         "age": 23, "fee_text": "€5.00m", "fee_eur": 5e6, "is_loan": False},
    ])
    d = compute_net_squad_delta(season="2026-2027", tm_dir=tmp_path / "external" / "transfermarkt")
    club = d["testfc"]
    # The loan-return IN is skipped (player also has an OUT) → 0 arrivals counted.
    assert club["material_in"] == 0
    assert club["arrivals_weight"] == pytest.approx(0.0)
    assert club["material_out"] == 1
    assert club["net_squad_delta"] < 0  # only the departure remains


def test_missing_parquet_returns_empty(tmp_path) -> None:
    assert compute_net_squad_delta(
        season="2099-2100", tm_dir=tmp_path / "external" / "transfermarkt"
    ) == {}


def test_winter_window_excluded_leak_guard(tmp_path) -> None:
    # A summer paid arrival and a WINTER paid arrival for the same club. The
    # net delta is a PRE-SEASON quantity: the winter row must NOT count (a January
    # signing must not retroactively inflate that season's August matches).
    _write_transfers(tmp_path, [
        {"team": "TestFC", "transfer_type": "in", "player_name": "Summer Signing",
         "age": 25, "fee_text": "€20.00m", "fee_eur": 20e6, "is_loan": False,
         "window": "summer"},
        {"team": "TestFC", "transfer_type": "in", "player_name": "January Signing",
         "age": 22, "fee_text": "€30.00m", "fee_eur": 30e6, "is_loan": False,
         "window": "winter"},
    ])
    d = compute_net_squad_delta(
        season="2026-2027", tm_dir=tmp_path / "external" / "transfermarkt"
    )
    club = d["testfc"]
    # Only the summer arrival is counted → material_in == 1, not 2.
    assert club["material_in"] == 1
    assert club["arrivals_weight"] == pytest.approx(0.15)  # unknown player floor


def test_untagged_legacy_file_keeps_all_rows(tmp_path) -> None:
    # A legacy file with NO window column must count all rows (exclude-winter is
    # written with a default so it never zeroes an untagged season).
    _write_transfers(tmp_path, [
        {"team": "TestFC", "transfer_type": "in", "player_name": "A",
         "age": 25, "fee_text": "€10.00m", "fee_eur": 10e6, "is_loan": False},
        {"team": "TestFC", "transfer_type": "in", "player_name": "B",
         "age": 24, "fee_text": "€10.00m", "fee_eur": 10e6, "is_loan": False},
    ])
    d = compute_net_squad_delta(
        season="2026-2027", tm_dir=tmp_path / "external" / "transfermarkt"
    )
    assert d["testfc"]["material_in"] == 2  # both kept, none dropped


def test_season_matched_market_values_no_future_leak(tmp_path) -> None:
    # A 2020-21 delta must be weighted by the 2020-21 market-values file, NOT a
    # later one. Made discriminating with TWO players: the signee's NORMALIZED
    # weight (value / file-max) differs between the season-matched and future
    # files, so reading the wrong file produces a different arrivals_weight.
    d = tmp_path / "external" / "transfermarkt"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"team": "TestFC", "transfer_type": "in", "player_name": "Signee",
         "age": 25, "fee_text": "€5.00m", "fee_eur": 5e6, "is_loan": False},
    ]).to_parquet(d / "transfers_2020_2021.parquet", index=False)
    # 2020-21 file: Signee=2, Other=10 → Signee normalizes to 0.2.
    pd.DataFrame([
        {"player_name": "Signee", "market_value_eur": 2e6},
        {"player_name": "Other Player", "market_value_eur": 10e6},
    ]).to_parquet(d / "market_values_2020_2021.parquet", index=False)
    # 2025-26 file: Signee=8, Other=10 → Signee would normalize to 0.8 if used.
    pd.DataFrame([
        {"player_name": "Signee", "market_value_eur": 8e6},
        {"player_name": "Other Player", "market_value_eur": 10e6},
    ]).to_parquet(d / "market_values_2025_2026.parquet", index=False)

    res = compute_net_squad_delta(season="2020-2021", tm_dir=d)
    # Correct (season-matched) weight = 0.2. A future-file leak would give 0.8.
    assert res["testfc"]["arrivals_weight"] == pytest.approx(0.2)
    assert res["testfc"]["material_in"] == 1


# ---------------------------------------------------------------------------
# Loan-to-permanent double-count fix (bought-back loanee is NOT a departure)
# ---------------------------------------------------------------------------


def test_loan_to_permanent_out_identified() -> None:
    # "End of loan" OUT + a paid IN for the same player = phantom departure.
    tf = pd.DataFrame([
        {"team": "TestFC", "transfer_type": "in", "player_name": "Bought Loanee",
         "fee_text": "€44.00m", "is_loan": False},
        {"team": "TestFC", "transfer_type": "out", "player_name": "Bought Loanee",
         "fee_text": "End of loan30/06/2026", "is_loan": True},
        # a genuine sale (paid OUT, no IN) must NOT be flagged
        {"team": "TestFC", "transfer_type": "out", "player_name": "Real Sale",
         "fee_text": "€10.00m", "is_loan": False},
    ])
    phantom = _loan_to_permanent_outs(tf)
    assert "bought loanee" in phantom
    assert "real sale" not in phantom


def test_bought_back_loanee_not_counted_as_departure(tmp_path) -> None:
    # A player with an "End of loan" OUT AND a paid IN was KEPT — the OUT is
    # phantom and must not subtract from net_squad_delta. Net = arrival only.
    _write_transfers(tmp_path, [
        {"team": "TestFC", "transfer_type": "in", "player_name": "Bought Loanee",
         "age": 24, "fee_text": "€20.00m", "fee_eur": 20e6, "is_loan": False},
        {"team": "TestFC", "transfer_type": "out", "player_name": "Bought Loanee",
         "age": 24, "fee_text": "End of loan30/06/2026", "fee_eur": None, "is_loan": True},
    ])
    d = compute_net_squad_delta(season="2026-2027", tm_dir=tmp_path / "external" / "transfermarkt")
    club = d["testfc"]
    # arrival counted, phantom OUT dropped → net POSITIVE, zero departures
    assert club["material_in"] == 1
    assert club["material_out"] == 0
    assert club["departures_weight"] == pytest.approx(0.0)
    assert club["net_squad_delta"] > 0


# ---------------------------------------------------------------------------
# January (winter) window filter — the live-feature temporal-leak fix
# ---------------------------------------------------------------------------


def test_jan_features_winter_only() -> None:
    # jan_arrivals must count ONLY winter-window rows. A summer signing and a
    # winter signing → jan_arrivals == 1 (the winter one), not 2.
    tf = pd.DataFrame([
        {"team": "TestFC", "transfer_type": "in", "player_name": "Summer Buy",
         "fee_text": "€10.00m", "fee_eur": 10e6, "is_loan": False, "window": "summer"},
        {"team": "TestFC", "transfer_type": "in", "player_name": "Winter Buy",
         "fee_text": "€30.00m", "fee_eur": 30e6, "is_loan": False, "window": "winter"},
    ])
    r = compute_january_window_features("TestFC", tf)
    assert r["jan_arrivals"] == 1
    assert r["jan_spend"] == 30_000_000


def test_jan_features_untagged_file_is_leakfree_zero() -> None:
    # A file with NO window column (predates the backfill) must yield NO winter
    # signal — jan_arrivals == 0 — rather than re-leaking the whole season as
    # January (the old `else: assume all January` bug).
    tf = pd.DataFrame([
        {"team": "TestFC", "transfer_type": "in", "player_name": "A",
         "fee_text": "€10.00m", "fee_eur": 10e6, "is_loan": False},
        {"team": "TestFC", "transfer_type": "in", "player_name": "B",
         "fee_text": "€10.00m", "fee_eur": 10e6, "is_loan": False},
    ])
    r = compute_january_window_features("TestFC", tf)
    assert r["jan_arrivals"] == 0
    assert r["jan_spend"] == 0


def test_jan_disruption_drops_phantom_loan_out() -> None:
    # A bought-back loanee's winter "End of loan" OUT must not inflate
    # squad_disruption (he was kept). With only that phantom OUT → disruption 0.
    tf = pd.DataFrame([
        {"team": "TestFC", "transfer_type": "in", "player_name": "Bought Loanee",
         "fee_text": "€20.00m", "fee_eur": 20e6, "is_loan": False, "window": "winter",
         "minutes_played": 2000},
        {"team": "TestFC", "transfer_type": "out", "player_name": "Bought Loanee",
         "fee_text": "End of loan30/06/2026", "fee_eur": None, "is_loan": True,
         "window": "winter", "minutes_played": 2000},
    ])
    r = compute_january_window_features("TestFC", tf)
    assert r["squad_disruption"] == pytest.approx(0.0)
