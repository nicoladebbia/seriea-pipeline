"""Tests for scripts/data/build_player_metadata.py.

The defect this module was written to fix was a *frozen* file: bio data that was
correct on the day it was written and silently wrong every day after. So the
tests here deliberately exercise the state TRANSITIONS — a second run, a newer
match, a birthday — rather than asserting a single snapshot looks right. A test
that only checks "the output parses" would pass against the broken version.

Every test redirects both ``DATA_DIR`` and the module-level ``OUT_PATH``; the
real ``data/features/player_metadata.json`` must never be touched by a test run
(see the ledger-drift incident, where a fixture overwrote the live bankroll).
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from scripts.data import build_player_metadata as bpm


def _player(pid: int, name: str, dob_ts: int, height: int, value: int) -> dict:
    return {
        "player": {
            "id": pid,
            "name": name,
            "position": "M",
            "height": height,
            "dateOfBirthTimestamp": dob_ts,
            "country": {"name": "Italy", "alpha2": "IT"},
            "proposedMarketValueRaw": {"value": value, "currency": "EUR"},
        }
    }


def _write_match(root, season: str, match_id: int, players: list[dict]) -> None:
    d = root / "external" / "sofascore" / "matches" / season
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{match_id}.json").write_text(json.dumps({
        "match_id": match_id,
        "home_lineup": {"starters": players, "substitutes": []},
        "away_lineup": {"starters": [], "substitutes": []},
    }))


def _write_dates(root, rows: list[tuple[int, str]]) -> None:
    p = root / "external" / "sofascore"
    p.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"match_id": str(m), "date": d} for m, d in rows]
    ).to_parquet(p / "player_match_stats.parquet")


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Redirect the module at BOTH the DATA_DIR global and the precomputed OUT_PATH."""
    monkeypatch.setattr(bpm, "DATA_DIR", tmp_path)
    monkeypatch.setattr(bpm, "OUT_PATH", tmp_path / "features" / "player_metadata.json")
    return tmp_path


def test_extracts_bio_fields(env):
    # Buffon: 1978-01-28
    _write_match(env, "2025-2026", 1, [_player(1, "Gigi", 254793600, 192, 5_000_000)])
    _write_dates(env, [(1, "2026-01-10")])

    meta = bpm.build()

    assert set(meta) == {"1"}
    rec = meta["1"]
    assert rec["date_of_birth"] == "1978-01-28"
    assert rec["height"] == 192
    assert rec["nationality"] == "Italy"
    assert rec["country_code"] == "IT"
    assert rec["market_value"] == 5_000_000
    assert rec["market_value_as_of"] == "2026-01-10"


def test_newest_match_wins_for_market_value(env):
    """The mutation this tool must survive: a player's value changes over time.

    Written so it FAILS against a first-write-wins implementation — the older
    match is deliberately walked in an order that does not guarantee the newer
    file is read last.
    """
    _write_match(env, "2024-2025", 10, [_player(7, "Old Name", 946684800, 180, 1_000_000)])
    _write_match(env, "2025-2026", 11, [_player(7, "New Name", 946684800, 180, 9_000_000)])
    _write_dates(env, [(10, "2025-02-01"), (11, "2026-05-01")])

    rec = bpm.build()["7"]

    assert rec["market_value"] == 9_000_000, "must take the value from the NEWEST match"
    assert rec["market_value_as_of"] == "2026-05-01"
    assert rec["name"] == "New Name"


def test_static_fields_survive_a_match_that_omits_them(env):
    """Height/dob are static: a later match missing them must not erase them."""
    _write_match(env, "2024-2025", 20, [_player(3, "A", 946684800, 185, 500_000)])
    thin = {"player": {"id": 3, "name": "A", "position": "D"}}
    _write_match(env, "2025-2026", 21, [thin])
    _write_dates(env, [(20, "2025-02-01"), (21, "2026-05-01")])

    rec = bpm.build()["3"]

    assert rec["height"] == 185
    assert rec["date_of_birth"] == "2000-01-01"


def test_a_later_correction_wins(env):
    """Sofascore corrects player records; the newest match carries the fix.

    Measured on real data: `Honest Ahanor` was served as Italy in older match
    JSONs and Nigeria in newer ones, and `Lorenzo Palmisani`'s height was
    corrected 196 -> 185. A first-non-null-wins rule silently kept the stale
    value forever, which is the same class of bug as the frozen file itself.
    """
    _write_match(env, "2024-2025", 40, [_player(9, "P", 946684800, 196, 1_000)])
    later = _player(9, "P", 946684800, 185, 1_000)
    later["player"]["country"] = {"name": "Nigeria", "alpha2": "NG"}
    _write_match(env, "2025-2026", 41, [later])
    _write_dates(env, [(40, "2025-02-01"), (41, "2026-05-01")])

    rec = bpm.build()["9"]

    assert rec["height"] == 185, "the corrected height must win"
    assert rec["nationality"] == "Nigeria"
    assert rec["country_code"] == "NG"


def test_ordering_is_by_match_date_not_filename(env):
    """A newer match with a numerically smaller id must still win.

    Sofascore match ids are not chronological across seasons, so sorting by
    path or id would reintroduce the stale-value bug on exactly the players
    whose records were most recently corrected.
    """
    _write_match(env, "2024-2025", 999, [_player(11, "Q", 946684800, 170, 100)])
    _write_match(env, "2025-2026", 100, [_player(11, "Q", 946684800, 190, 900)])
    _write_dates(env, [(999, "2025-02-01"), (100, "2026-05-01")])

    rec = bpm.build()["11"]

    assert rec["height"] == 190
    assert rec["market_value"] == 900
    assert rec["market_value_as_of"] == "2026-05-01"


def test_age_is_derived_not_frozen(env):
    """Age must track the calendar. The old file stored an int and rotted."""
    assert bpm._age_from_dob("2000-06-15", on=date(2026, 6, 14)) == 25
    assert bpm._age_from_dob("2000-06-15", on=date(2026, 6, 15)) == 26
    assert bpm._age_from_dob("2000-06-15", on=date(2027, 6, 15)) == 27
    assert bpm._age_from_dob("not-a-date") is None


def test_build_is_idempotent(env):
    _write_match(env, "2025-2026", 30, [_player(5, "B", 946684800, 175, 2_000_000)])
    _write_dates(env, [(30, "2026-03-03")])

    assert bpm.build() == bpm.build()


def test_refuses_to_overwrite_with_an_empty_result(env):
    """A scrape outage must not blank the file every consumer reads."""
    out = env / "features" / "player_metadata.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"1": {"name": "Existing"}}))

    assert bpm.main() == 1, "must exit non-zero when nothing was extracted"
    assert json.loads(out.read_text()) == {"1": {"name": "Existing"}}, "file must be untouched"
