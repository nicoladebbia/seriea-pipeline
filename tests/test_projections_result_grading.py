"""The /projections page must not grade a fixture against a different match.

`_played_result_row` matched on (home_team, away_team) alone and took the last
such row, so an UNPLAYED fixture inherited the last time those two teams ever
met. Measured live 2026-08-25: Milan v Venezia on 2026-08-28 was displayed
"FT 4-0" — the 2024-09-14 result — and its starred best bet was graded a HIT
against it. 21 of the 22 upcoming fixtures had a historical meeting available
to borrow, so the page was reporting a fabricated track record.
"""

from __future__ import annotations

import pandas as pd

from web.app import _played_result_row

FIXTURE_DATE = "2026-08-28"


def _matches(rows):
    return pd.DataFrame(rows)


HISTORICAL = {
    "home_team": "Milan", "away_team": "Venezia",
    "match_date": pd.Timestamp("2024-09-14"), "home_score": 4.0, "away_score": 0.0,
}


def test_an_unplayed_fixture_gets_no_result_from_an_old_meeting():
    """THE regression: before the fix this returned the 2024-09-14 row."""
    m = _matches([HISTORICAL])
    assert _played_result_row(m, "Milan", "Venezia", FIXTURE_DATE) is None


def test_the_real_result_is_still_attached_once_the_match_is_played():
    """True positive — without this, `return None` would satisfy every other
    test here and silently kill result grading altogether."""
    played = dict(HISTORICAL, match_date=pd.Timestamp(FIXTURE_DATE),
                  home_score=2.0, away_score=1.0)
    m = _matches([HISTORICAL, played])
    row = _played_result_row(m, "Milan", "Venezia", FIXTURE_DATE)
    assert row is not None
    assert int(row.iloc[0]["home_score"]) == 2, "must grade THIS match, not the old one"


def test_a_fixture_scheduled_but_not_yet_scored_gets_no_result():
    scheduled = dict(HISTORICAL, match_date=pd.Timestamp(FIXTURE_DATE),
                     home_score=None, away_score=None)
    m = _matches([HISTORICAL, scheduled])
    assert _played_result_row(m, "Milan", "Venezia", FIXTURE_DATE) is None


def test_an_unparseable_fixture_date_attaches_nothing():
    """A wrong result is worse than no result."""
    m = _matches([HISTORICAL])
    assert _played_result_row(m, "Milan", "Venezia", None) is None
    assert _played_result_row(m, "Milan", "Venezia", "TBD") is None


def test_the_reverse_fixture_is_a_different_match():
    away_leg = {"home_team": "Venezia", "away_team": "Milan",
                "match_date": pd.Timestamp(FIXTURE_DATE),
                "home_score": 3.0, "away_score": 0.0}
    m = _matches([away_leg])
    assert _played_result_row(m, "Milan", "Venezia", FIXTURE_DATE) is None
