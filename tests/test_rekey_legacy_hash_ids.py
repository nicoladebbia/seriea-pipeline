"""Pins for the legacy FBref-hash -> canonical match_id migration.

The defect being guarded: a hash-vs-canonical mismatch never raises. A
``merge(on="match_id")`` against matches.parquet just returns nothing and the
feature columns come out NaN, which is why seven seasons of adv_roll5_*,
tagg_roll5_* and player_impact stayed empty without a single error in any log.
Every assertion here is therefore about the id STRING, not about whether the
code ran.
"""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.data import rekey_legacy_hash_ids as rk


def _rows(match_id, date, home, away, **extra):
    base = dict(match_date=date, season="2017-2018", **extra)
    return [
        {"match_id": match_id, "team": home, "is_home": True, **base},
        {"match_id": match_id, "team": away, "is_home": False, **base},
    ]


def _frame(*groups):
    return pd.DataFrame([r for g in groups for r in g])


def test_a_hash_id_becomes_the_canonical_date_home_away():
    df = _frame(_rows("0005cd5f", "2017-12-01", "Napoli", "Juventus"))
    out, _ = rk.rekey_frame(df)
    assert set(out["match_id"]) == {"2017-12-01_Napoli_Juventus"}


def test_rows_already_canonical_are_left_alone():
    df = _frame(_rows("2025-08-23_Genoa_Lecce", "2025-08-23", "Genoa", "Lecce"))
    out, s = rk.rekey_frame(df)
    assert set(out["match_id"]) == {"2025-08-23_Genoa_Lecce"}
    assert s["hash_ids"] == 0


def test_running_twice_changes_nothing():
    df = _frame(_rows("0005cd5f", "2017-12-01", "Napoli", "Juventus"))
    once, _ = rk.rekey_frame(df)
    twice, s = rk.rekey_frame(once)
    pd.testing.assert_frame_equal(once, twice)
    assert s["hash_ids"] == 0, "second pass must find nothing left to do"


def test_a_frame_without_dates_uses_the_donor_map():
    """goalkeeper_stats has no match_date and must still be re-keyed."""
    gk = pd.DataFrame([
        {"match_id": "0005cd5f", "team": "Napoli", "is_home": True},
        {"match_id": "0005cd5f", "team": "Juventus", "is_home": False},
    ])
    out, _ = rk.rekey_frame(gk, donor={"0005cd5f": "2017-12-01_Napoli_Juventus"})
    assert set(out["match_id"]) == {"2017-12-01_Napoli_Juventus"}


def test_an_id_the_frame_can_derive_itself_beats_the_donor():
    """A donor may only ADD coverage. If it disagrees, local wins."""
    df = _frame(_rows("0005cd5f", "2017-12-01", "Napoli", "Juventus"))
    out, _ = rk.rekey_frame(df, donor={"0005cd5f": "1999-01-01_Wrong_Wrong"})
    assert set(out["match_id"]) == {"2017-12-01_Napoli_Juventus"}


@pytest.mark.parametrize(
    ("date", "home", "away"),
    [("", "Napoli", "Juventus"), ("2017-12-01", "", "Juventus"), ("2017-12-01", "Napoli", "")],
)
def test_a_blank_field_never_mints_a_half_built_id(date, home, away):
    """"_Napoli_" would look canonical forever and join to nothing."""
    df = _frame(_rows("0005cd5f", date, home, away))
    out, _ = rk.rekey_frame(df)
    assert set(out["match_id"]) == {"0005cd5f"}, "must stay hashed, not become junk"


def test_a_one_sided_match_is_not_rebuilt():
    """Only a home row present -> no away team -> no id can be built."""
    df = pd.DataFrame([
        {"match_id": "0005cd5f", "match_date": "2017-12-01",
         "team": "Napoli", "is_home": True, "season": "2017-2018"},
    ])
    out, _ = rk.rekey_frame(df)
    assert set(out["match_id"]) == {"0005cd5f"}


def test_two_different_matches_do_not_collide():
    df = _frame(
        _rows("aaaaaaaa", "2017-12-01", "Napoli", "Juventus"),
        _rows("bbbbbbbb", "2018-03-04", "Inter", "Milan"),
    )
    out, _ = rk.rekey_frame(df)
    assert set(out["match_id"]) == {
        "2017-12-01_Napoli_Juventus", "2018-03-04_Inter_Milan",
    }


def test_the_reconstruction_stat_reports_partial_coverage():
    """One rebuildable match, one that can't be — must report 0.5, not 1.0."""
    df = _frame(
        _rows("aaaaaaaa", "2017-12-01", "Napoli", "Juventus"),
        _rows("bbbbbbbb", "", "Inter", "Milan"),
    )
    _, s = rk.rekey_frame(df)
    assert s["hash_ids"] == 2
    assert s["rebuilt_ids"] == 1
    assert s["reconstruction"] == pytest.approx(0.5)


def test_a_canonical_id_is_never_rewritten_even_if_its_columns_disagree():
    """Only hash ids are in scope. A canonical id whose own date/team columns
    no longer agree with it (an id minted under an older normalisation, say
    "Inter" vs "Internazionale") must be left EXACTLY as it is — matches.parquet
    is the authority on that id, not this frame's columns. Rewriting it would
    silently unjoin a row that currently joins fine."""
    df = _frame(_rows("2017-12-01_Internazionale_Milan", "2017-12-01", "Inter", "Milan"))
    out, s = rk.rekey_frame(df)
    assert set(out["match_id"]) == {"2017-12-01_Internazionale_Milan"}
    assert s["hash_ids"] == 0
