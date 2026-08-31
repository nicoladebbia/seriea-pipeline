"""Lineup fetch must self-bound: deadline + blocked-endpoint breaker.

Measured 2026-08-30 08:00: Sofascore 403-blocked the lineups endpoint and every
miss burned the full ~42s retry ladder (2+5+10+20s), so 5 matches = 3.5 min.
The scheduler's 180s subprocess kill then discarded even the lineups that WERE
confirmed. The fetch loop now (a) aborts after 2 consecutive 403-exhausted
fetches and (b) stops at a wall-clock deadline, returning partial results.
"""

import pytest

import scraper.sofascore_events as ss_events
import scraper.sofascore_lineups as ss_lineups
from scraper.sofascore_lineups import fetch_all_lineups

_ODDS = {
    f"Team{i}A vs Team{i}B": {
        "home_team": f"Team{i}A", "away_team": f"Team{i}B",
        "commence_time": "2026-08-30T12:00:00Z",
    }
    for i in range(5)
}
_IDS = {mk: 1000 + i for i, mk in enumerate(_ODDS)}


@pytest.fixture
def _wired(monkeypatch):
    """Common wiring: 5 imminent matches, resolved IDs, no real sleeps."""
    monkeypatch.setattr(ss_lineups, "get_sofascore_match_ids", lambda odds: dict(_IDS))
    monkeypatch.setattr(ss_lineups, "_jitter_delay", lambda *a, **k: None)
    import scripts.utils.match_timing as mt
    monkeypatch.setattr(mt, "classify_match_window", lambda ct: "imminent")
    calls = []
    return calls


def test_breaker_stops_after_two_consecutive_403_exhaustions(_wired, monkeypatch):
    calls = _wired

    def blocked_fetch(match_id):
        calls.append(match_id)
        ss_events._LAST_FAILURE_STATUS = 403  # what _get_json leaves behind
        return None

    monkeypatch.setattr(ss_lineups, "fetch_lineup", blocked_fetch)
    result = fetch_all_lineups(dict(_ODDS))
    assert result == {}
    # 2 blocked fetches trip the breaker; the other 3 matches are never tried
    assert len(calls) == 2


def test_not_yet_published_404_does_not_trip_the_breaker(_wired, monkeypatch):
    calls = _wired

    def unpublished_fetch(match_id):
        calls.append(match_id)
        ss_events._LAST_FAILURE_STATUS = None  # clean 404 / not-confirmed
        return None

    monkeypatch.setattr(ss_lineups, "fetch_lineup", unpublished_fetch)
    result = fetch_all_lineups(dict(_ODDS))
    assert result == {}
    assert len(calls) == 5  # every match still probed


def test_deadline_returns_partial_results(_wired, monkeypatch):
    calls = _wired
    clock = {"t": 0.0}

    def fake_now():
        return clock["t"]

    def slow_confirmed_fetch(match_id):
        calls.append(match_id)
        clock["t"] += 100.0  # each fetch "costs" 100s
        ss_events._LAST_FAILURE_STATUS = None
        return {
            "home_lineup": ["a"] * 11, "away_lineup": ["b"] * 11,
            "home_bench": [], "away_bench": [],
            "home_formation": "4-3-3", "away_formation": "4-4-2",
            "lineup_source": "confirmed", "source_api": "sofascore",
            "match_id_sofascore": match_id,
        }

    monkeypatch.setattr(ss_lineups, "_now", fake_now)
    monkeypatch.setattr(ss_lineups, "fetch_lineup", slow_confirmed_fetch)
    result = fetch_all_lineups(dict(_ODDS), deadline_sec=150.0)
    # t=0 ok (->100), t=100 ok (->200), t=200 > 150 -> stop with 2 partials
    assert len(calls) == 2
    assert len(result) == 2  # partial results RETURNED, not discarded


def test_footballdata_deadline_bounds_the_loop(monkeypatch):
    """Deadline must stop the fd.org loop AND the partial-return log must not crash.

    Pins two things: (a) the 6.5s/match loop stops once the deadline passes,
    (b) the deadline log line references the real accumulator (`confirmed`) —
    the original patch said `len(result)` (NameError swallowed by the caller,
    dropping the whole backup source exactly when its budget ran low).
    Empty odds_data legally bypasses the imminence filter, so the loop runs
    without coupling this test to the team-name normalizers.
    """
    import scraper.footballdata_lineups as fd

    monkeypatch.setenv("FOOTBALLDATA_KEY", "test-key")
    matches = [{"id": i, "utcDate": "2026-08-30T12:00:00Z",
                "homeTeam": {"name": f"H{i}"}, "awayTeam": {"name": f"A{i}"}}
               for i in range(5)]
    monkeypatch.setattr(fd, "_api_get", lambda *a, **k: {"matches": matches})

    seen = []
    real_sleep = fd.time.sleep
    t = {"v": 0.0}
    monkeypatch.setattr(fd.time, "monotonic", lambda: t["v"])

    def fake_sleep(sec):
        seen.append(sec)
        t["v"] += sec

    monkeypatch.setattr(fd.time, "sleep", fake_sleep)
    try:
        result = fd.fetch_lineups_footballdata({}, deadline_sec=10.0)
    finally:
        monkeypatch.setattr(fd.time, "sleep", real_sleep)
    # 6.5s sleep per match: match 0 (t=0->6.5), match 1 (t=6.5->13), then
    # t=13 > 10 -> deadline log fires (must not NameError) -> break.
    assert len(seen) == 2
    assert result == {}  # partial dict returned, no exception propagated
