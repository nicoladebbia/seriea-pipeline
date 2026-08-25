"""The calendar date these scrapers derive is part of a row's IDENTITY.

``matches.parquet`` is keyed by ``{date}_{home}_{away}``, and the Sofascore
stats parquets are joined back to it on that date. ``startTimestamp`` is a UTC
epoch, so the conversion must pin UTC explicitly: a bare
``datetime.fromtimestamp()`` is naive-LOCAL, which made the date a function of
the machine's timezone. Every other consumer in the repo already converts with
an explicit UTC (``build_match_id_mapping``, ``matchday_updater``,
``worldcup/sofascore_fetch``); these three were the outliers.
"""

from __future__ import annotations

import os
import time

import pytest

from scripts.data import scrape_sofascore as ss

# 2025-08-22 02:10 UTC — still 2025-08-21 in any timezone west of UTC.
STRADDLING_TS = 1755826200
EXPECTED_UTC_DATE = "2025-08-22"

FIXTURE = {
    "startTimestamp": STRADDLING_TS,
    "homeTeam": {"name": "Inter", "id": 1},
    "awayTeam": {"name": "Milan", "id": 2},
    "homeScore": {"current": 1},
    "awayScore": {"current": 0},
    "roundInfo": {"round": 1},
}
SHOTMAP = {
    "shotmap": [
        {
            "isHome": True,
            "xg": 0.3,
            "player": {"name": "A", "id": 9},
            "playerCoordinates": {"x": 88, "y": 50},
            "shotType": "goal",
            "situation": "regular",
        }
    ]
}


@pytest.fixture
def west_of_utc():
    """Force a timezone behind UTC so the test fails on a UTC CI box too."""
    old = os.environ.get("TZ")
    os.environ["TZ"] = "America/New_York"
    time.tzset()
    yield
    if old is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = old
    time.tzset()


def test_the_kickoff_date_is_utc_not_machine_local(west_of_utc):
    rows = ss.extract_shotmap_rows(
        {"match_id": "x", "shotmap": SHOTMAP}, FIXTURE, "2025_2026"
    )
    assert rows, "no rows produced — the date branch was not exercised"
    assert {r["date"] for r in rows} == {EXPECTED_UTC_DATE}


def test_a_fixture_with_no_timestamp_yields_an_empty_date(west_of_utc):
    """The miss branch must stay a blank string, not today's date."""
    rows = ss.extract_shotmap_rows(
        {"match_id": "x", "shotmap": SHOTMAP},
        dict(FIXTURE, startTimestamp=0),
        "2025_2026",
    )
    assert rows
    assert {r["date"] for r in rows} == {""}
