"""Season rollover — the August trap.

Every August this repo has broken the same way: a hardcoded `"2025-2026"` goes
stale, the job keeps running, and nothing fails loudly. The fix is two helpers
with deliberately different meanings, and these tests pin the difference.

    get_current_season()          the season to SCRAPE   (calendar)
    latest_season_with_results()  the season to ANALYSE  (played matches)

The mutation these must survive is not "a season rolls over" — it is **next
season's fixtures being ingested with null scores**, which is what makes the
naive `df["season"].max()` wrong. Every test that matters below inserts those
rows, because a suite that only ever sees a settled dataframe stays green while
the helper is broken for its actual job.
"""
from __future__ import annotations

import datetime as _dt

import pandas as pd
import pytest

from config.leagues import PROMOTED_TEAMS
from config.settings import get_current_season, latest_season_with_results


# --------------------------------------------------------------------------
# get_current_season — the calendar side
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "today,expected",
    [
        (_dt.date(2026, 7, 31), "2025-2026"),   # day before rollover
        (_dt.date(2026, 8, 1), "2026-2027"),    # the boundary itself
        (_dt.date(2026, 8, 2), "2026-2027"),
        (_dt.date(2026, 12, 31), "2026-2027"),  # calendar year turns, season does not
        (_dt.date(2027, 1, 1), "2026-2027"),
        (_dt.date(2027, 5, 24), "2026-2027"),   # final matchday
    ],
)
def test_current_season_rolls_on_august_first(monkeypatch, today, expected):
    class _FrozenDate(_dt.date):
        @classmethod
        def today(cls):
            return today

    monkeypatch.setattr("config.settings.date", _FrozenDate)
    assert get_current_season() == expected


# --------------------------------------------------------------------------
# latest_season_with_results — the data side
# --------------------------------------------------------------------------
def _frame(rows):
    return pd.DataFrame(rows, columns=["season", "home_score"])


def test_ignores_next_seasons_unplayed_fixtures():
    """THE test. Fixtures are written before kickoff with a null score, so from
    the moment next season's schedule lands, `.max()` names a season with no
    results and every downstream mean/count reads an empty frame."""
    df = _frame(
        [("2025-2026", 1.0)] * 380
        + [("2026-2027", None)] * 380  # schedule ingested, nothing played yet
    )
    assert df["season"].max() == "2026-2027"          # the naive answer...
    assert latest_season_with_results(df) == "2025-2026"  # ...and the right one


def test_follows_the_new_season_once_matches_are_actually_played():
    """The flip side: it must not pin to the old season forever."""
    df = _frame(
        [("2025-2026", 1.0)] * 380
        + [("2026-2027", 2.0)] * 10      # matchweek 1 played
        + [("2026-2027", None)] * 370    # rest of the schedule pending
    )
    assert latest_season_with_results(df) == "2026-2027"


def test_partially_played_current_season_is_still_current():
    df = _frame([("2025-2026", 1.0)] * 676 + [("2025-2026", None)] * 13)
    assert latest_season_with_results(df) == "2025-2026"


def test_returns_none_rather_than_a_wrong_season():
    """Callers must be able to tell 'nothing played' from 'some season'. The
    historical failure here was a soft empty return that looked like success."""
    assert latest_season_with_results(_frame([("2026-2027", None)] * 380)) is None
    assert latest_season_with_results(_frame([])) is None
    assert latest_season_with_results(pd.DataFrame()) is None
    assert latest_season_with_results(None) is None


def test_missing_season_column_is_none_not_an_exception():
    assert latest_season_with_results(pd.DataFrame({"home_score": [1.0]})) is None


def test_alternate_result_column():
    """Player tables mark participation with `minutes`, not `home_score`."""
    df = pd.DataFrame({
        "season": ["2025-2026"] * 3 + ["2026-2027"] * 2,
        "minutes": [90, 90, 45, None, None],
    })
    assert latest_season_with_results(df, result_col="minutes") == "2025-2026"


def test_absent_result_column_falls_back_to_presence():
    """When the frame has no result column at all, every row counts as played —
    correct for tables that only ever hold completed matches."""
    df = pd.DataFrame({"season": ["2024-2025", "2025-2026"]})
    assert latest_season_with_results(df, result_col="nope") == "2025-2026"


def test_season_ordering_is_lexicographic_safe():
    """YYYY-YYYY sorts correctly as a string across a decade boundary."""
    df = _frame([("2009-2010", 1.0), ("2010-2011", 1.0), ("2025-2026", 1.0)])
    assert latest_season_with_results(df) == "2025-2026"


# --------------------------------------------------------------------------
# The promoted-club data that has to be refreshed alongside the season
# --------------------------------------------------------------------------
@pytest.mark.parametrize("league", ["serie_a", "premier_league"])
def test_active_leagues_have_promotions_for_the_current_season(league):
    """A missing entry here silently treats promoted clubs as ordinary ones."""
    season = get_current_season()
    assert season in PROMOTED_TEAMS[league], (
        f"{league} has no PROMOTED_TEAMS entry for {season} — promoted clubs "
        f"will be scored as if they had league history"
    )
    assert len(PROMOTED_TEAMS[league][season]) == 3, "leagues are 3-up-3-down"


def test_promoted_clubs_are_not_also_last_seasons_clubs():
    """Catches the copy-paste failure of duplicating last season's promotions."""
    for league in ("serie_a", "premier_league"):
        seasons = sorted(PROMOTED_TEAMS[league])
        assert PROMOTED_TEAMS[league][seasons[-1]] != PROMOTED_TEAMS[league][seasons[-2]]
