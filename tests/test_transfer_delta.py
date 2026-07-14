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
    _transfer_materiality,
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
