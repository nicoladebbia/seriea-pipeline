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


# --- A club that was not in the previous snapshot ------------------------
#
# The cold-start guard only covers the very first run. `previous.get(club, {})`
# turned every LATER change to the club list into a full phantom squad of
# signings. Measured on the live changelog 2026-09-07: nine clubs outside the
# 2026-27 squad set appeared on 2026-07-20 and minted 256 phantom "signings" —
# 36% of the whole feed, all of type `signing`, because this loop iterates
# CURRENT clubs so a club that vanishes emits nothing at all.

# _norm() strips digits, so "P0".."P3" would collapse to ONE key — names must
# differ by letters for the fixture to model distinct players.
_NAMES = ["Aldo Rossi", "Bruno Conti", "Carlo Neri", "Dario Ferri", "Elio Gatti",
          "Fabio Lupo", "Gino Marra", "Hugo Sanna", "Ivo Testa", "Luca Vinci"]


def _club_rows(team: str, n: int = 3):
    names = [f"{_NAMES[i % len(_NAMES)]}{'' if i < len(_NAMES) else chr(97 + i // len(_NAMES))}"
             for i in range(n)]
    assert len({det._norm(x) for x in names}) == n, "fixture names must stay distinct after _norm"
    return [
        {"team": team, "player_name": nm, "position": "Midfield",
         "market_value_eur": 1_000_000.0, "contract_until": "2028-06-30"}
        for nm in names
    ]


def test_a_club_absent_from_the_snapshot_is_not_a_squad_of_signings(tm):
    _write_squad(tm, _base_rows())
    det.detect_changes("2026-2027")                       # seeds Napoli
    _write_squad(tm, _base_rows() + _club_rows("Chievo", 25))
    changes = det.detect_changes("2026-2027")
    # The true positive this guards: the old code returned 25 signings here.
    assert [c for c in changes if c["club"] == "Chievo"] == [], changes[:3]
    assert not any(c["club"] == "Chievo"
                   for c in json.loads((tm / "transfer_changes_2026_2027.json").read_text())
                   ) if (tm / "transfer_changes_2026_2027.json").exists() else True


def test_the_new_club_is_seeded_so_its_next_change_is_real(tm):
    """Seeding must not make the club permanently invisible."""
    _write_squad(tm, _base_rows())
    det.detect_changes("2026-2027")
    _write_squad(tm, _base_rows() + _club_rows("Chievo", 3))
    assert det.detect_changes("2026-2027") == []          # seeded, silent
    _write_squad(tm, _base_rows() + _club_rows("Chievo", 4))   # one real arrival
    changes = det.detect_changes("2026-2027")
    signings = [c for c in changes if c["club"] == "Chievo" and c["type"] == "signing"]
    assert len(signings) == 1, changes
    assert signings[0]["player"] == _NAMES[3]


def test_a_known_club_still_reports_a_real_signing(tm):
    """Positive control: the guard must not suppress ordinary signings."""
    _write_squad(tm, _base_rows())
    det.detect_changes("2026-2027")
    _write_squad(tm, _base_rows() + _club_rows("Napoli", 1))
    changes = det.detect_changes("2026-2027")
    assert [c["type"] for c in changes] == ["signing"]
    assert changes[0]["club"] == "Napoli"
