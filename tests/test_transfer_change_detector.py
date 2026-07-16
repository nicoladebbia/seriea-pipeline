"""Tests for the transfer change detector (scripts/data/transfer_change_detector).

Locks the behaviors that make an unattended, twice-daily change feed trustworthy:
  1. Cold start seeds the snapshot and reports NOTHING (no phantom "whole league
     signed" on the first run).
  2. An unchanged second run reports nothing (idempotent).
  3. Each real delta is detected: signing, departure, value_change, contract_change.
  4. A sub-threshold value wobble is ignored (VALUE_EPS).

Uses a tmp data dir + a synthetic market_values parquet — never a live scrape.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

import scripts.data.transfer_change_detector as det


def _write_squad(tmp_dir, rows) -> None:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(tmp_dir / "market_values_2026_2027.parquet", index=False)


@pytest.fixture
def tm(tmp_path, monkeypatch):
    d = tmp_path / "external" / "transfermarkt"
    monkeypatch.setattr(det, "TM_DIR", d)
    return d


def _base_rows():
    return [
        {"team": "Napoli", "player_name": "Alpha One", "position": "Goalkeeper",
         "market_value_eur": 10_000_000.0, "contract_until": "2028-06-30"},
        {"team": "Napoli", "player_name": "Beta Two", "position": "Centre-Back",
         "market_value_eur": 20_000_000.0, "contract_until": "2027-06-30"},
    ]


def test_cold_start_seeds_and_reports_nothing(tm):
    _write_squad(tm, _base_rows())
    changes = det.detect_changes("2026-2027")
    assert changes == []
    assert (tm / "squad_snapshot_2026_2027.json").exists()


def test_unchanged_run_is_idempotent(tm):
    _write_squad(tm, _base_rows())
    det.detect_changes("2026-2027")          # seed
    assert det.detect_changes("2026-2027") == []  # no change second run


def test_detects_signing_and_departure(tm):
    _write_squad(tm, _base_rows())
    det.detect_changes("2026-2027")          # seed with Alpha + Beta
    # Beta leaves, a new player Gamma joins
    _write_squad(tm, [
        _base_rows()[0],
        {"team": "Napoli", "player_name": "Gamma Three", "position": "Striker",
         "market_value_eur": 30_000_000.0, "contract_until": "2030-06-30"},
    ])
    changes = det.detect_changes("2026-2027")
    types = {(c["type"], c["player"]) for c in changes}
    assert ("signing", "Gamma Three") in types
    assert ("departure", "Beta Two") in types


def test_detects_value_and_contract_change(tm):
    _write_squad(tm, _base_rows())
    det.detect_changes("2026-2027")
    rows = _base_rows()
    rows[1]["market_value_eur"] = 35_000_000.0   # Beta value up €20m → €35m
    rows[1]["contract_until"] = "2031-06-30"      # and a contract extension
    _write_squad(tm, rows)
    changes = det.detect_changes("2026-2027")
    types = {c["type"] for c in changes}
    assert "value_change" in types
    assert "contract_change" in types


def test_subthreshold_value_wobble_ignored(tm):
    _write_squad(tm, _base_rows())
    det.detect_changes("2026-2027")
    rows = _base_rows()
    rows[0]["market_value_eur"] = 10_000_000.0 + (det.VALUE_EPS - 1)  # below threshold
    _write_squad(tm, rows)
    changes = det.detect_changes("2026-2027")
    assert not any(c["type"] == "value_change" for c in changes)


def test_changelog_is_appended_newest_first(tm):
    _write_squad(tm, _base_rows())
    det.detect_changes("2026-2027")
    _write_squad(tm, [_base_rows()[0]])  # Beta leaves
    det.detect_changes("2026-2027")
    log = json.loads((tm / "transfer_changes_2026_2027.json").read_text())
    assert log and log[0]["type"] == "departure" and log[0]["player"] == "Beta Two"
