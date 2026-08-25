"""The off-season dormancy flag on /api/data-freshness.

``offseason_dormant`` is not cosmetic: it gates the ``league_hard`` branch and
forces ``ok: True``, so a wrong flag makes the dashboard's staleness banner
permanently green and the endpoint structurally unable to report a hard
failure. It asked ``matches.parquet`` for upcoming fixtures — a results-only
store that has never held a future-dated row — so it read dormant all year and
was masking a live Serie A HTML outage.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from web.app import _club_leagues_dormant

HORIZON = 14


def _fixture(league: str, days_out: int) -> dict:
    when = datetime.now(timezone.utc) + timedelta(days=days_out)
    return {
        "league": league,
        "date": when.strftime("%Y-%m-%d"),
        "home_team": "A",
        "away_team": "B",
        "commence_time": when.isoformat(),
    }


def _patch(monkeypatch, within_horizon, season=None):
    """Stub the reader. ``season`` is what an UNBOUNDED read returns — the
    readability probe — and defaults to the same fixtures, since a file holding
    fixtures inside the window necessarily holds them at all."""
    import scripts.utils.match_timing as mt

    full = list(within_horizon) if season is None else list(season)

    def fake(_now, *a, horizon_days=14, **k):
        return full if horizon_days is None else list(within_horizon)

    monkeypatch.setattr(mt, "_load_sofascore_fixtures", fake)


def test_a_league_with_fixtures_this_week_is_not_dormant(monkeypatch):
    """MW2 is four days out — neither league is off-season."""
    _patch(monkeypatch, [_fixture("serie_a", 4), _fixture("premier_league", 4)])
    assert _club_leagues_dormant(HORIZON) == {
        "serie_a": False, "premier_league": False
    }


def test_one_league_can_be_dormant_while_the_other_plays(monkeypatch):
    """Serie A and the EPL do not share a calendar; the flag is per-league."""
    _patch(monkeypatch, [_fixture("premier_league", 3)])
    assert _club_leagues_dormant(HORIZON) == {
        "serie_a": True, "premier_league": False
    }


def test_a_genuine_off_season_still_reads_dormant(monkeypatch):
    """The flag must keep doing its real job — June has no club fixtures.

    The season file IS readable (next season is published), it simply holds
    nothing within 14 days. That is the case the suppression exists for.
    """
    _patch(monkeypatch, [], season=[_fixture("serie_a", 60),
                                    _fixture("premier_league", 60)])
    assert _club_leagues_dormant(HORIZON) == {
        "serie_a": True, "premier_league": True
    }


def test_an_unreadable_fixture_source_is_not_reported_as_off_season(monkeypatch):
    """Silence from a broken source must not suppress the alert it should raise.

    This is the distinction the results-store version could not make at all,
    and the one a naive "no fixtures -> dormant" rewrite would lose again:
    dormant forces ``ok: True``, so mistaking a read failure for the off-season
    turns a real outage into a green banner.
    """
    _patch(monkeypatch, [], season=[])
    assert _club_leagues_dormant(HORIZON) == {
        "serie_a": False, "premier_league": False
    }


def test_the_results_store_is_not_the_fixture_source():
    """The regression itself, against real data.

    ``matches.parquet`` holds only played matches, so any dormancy rule reading
    it returns True forever. This asserts the precondition that made the old
    implementation wrong, so the test cannot pass by accident.
    """
    import pandas as pd

    from config.settings import DATA_DIR

    path = DATA_DIR / "parsed" / "matches.parquet"
    if not path.exists():
        import pytest

        pytest.skip("matches.parquet not on disk")
    m = pd.read_parquet(path, columns=["match_date", "home_score"])
    dates = pd.to_datetime(m["match_date"], errors="coerce")
    assert (dates > datetime.now()).sum() == 0, "premise changed: re-check the rule"
    assert m["home_score"].isna().sum() == 0

    # ...and the live flag is nevertheless False, because it reads elsewhere.
    assert _club_leagues_dormant(HORIZON) == {
        "serie_a": False, "premier_league": False
    }
