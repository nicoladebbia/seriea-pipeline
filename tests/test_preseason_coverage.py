"""Pre-season XI coverage — the health check that makes a silent degrade visible.

A club with no pre-season friendly rows is not a bug and nothing raises. It just
means that on matchweek 1 its predicted XI comes from a league table a whole
summer stale — measured at ~11pp worse than the same club with friendlies. The
check exists so that gap is a line in the health report instead of an invisible
accuracy loss.

Both mutations pinned here are ones the FIRST draft got wrong, and both were
invisible on the day it would have been tested:

  1. **The season boundary.** `get_current_season()` rolls 1 August;
     `FRIENDLY_WINDOWS` opens 1 June. Through June and July the calendar helper
     names the season that just ENDED — so a check keyed off it opens the wrong
     parquet AND early-returns "signal retired" against last season's 380 played
     matches. Two of the three pre-season months, permanently green.
  2. **The empty registry.** `fetch_club_ids` logs and CONTINUES on a 403; it
     returns a short dict rather than raising. With `expected` empty,
     `expected - have` is empty too and the check reports FULL coverage exactly
     when Sofascore is blocking us.

Test 1 is therefore written at a frozen July date, not today's.
"""
from __future__ import annotations

import datetime as _dt

import pandas as pd
import pytest

from scraper import sofascore_friendlies as sf
from scripts.pipeline import health_check as hc

LEAGUES = ("serie_a", "premier_league")
SA = [f"SA{i}" for i in range(20)]
EPL = [f"EPL{i}" for i in range(20)]


def _registry(sa=SA, epl=EPL):
    reg = {i: (n, "serie_a") for i, n in enumerate(sa, start=1)}
    reg.update({i: (n, "premier_league") for i, n in enumerate(epl, start=100)})
    return reg


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated DATA_DIR, a live registry of 20+20, no network."""
    monkeypatch.setattr(hc, "DATA_DIR", tmp_path)
    monkeypatch.setattr(hc, "ACTIVE_LEAGUES", LEAGUES, raising=False)
    monkeypatch.setattr("config.leagues.ACTIVE_LEAGUES", LEAGUES)
    monkeypatch.setattr(sf, "fetch_club_ids", lambda keys: _registry())
    (tmp_path / "external" / "sofascore").mkdir(parents=True)
    (tmp_path / "parsed").mkdir(parents=True)
    return tmp_path


def _write_friendlies(root, season, sa=SA, epl=EPL):
    rows = [{"club": c, "club_league": "serie_a", "is_our_club": True} for c in sa]
    rows += [{"club": c, "club_league": "premier_league", "is_our_club": True} for c in epl]
    # An opponent row: is_our_club False, must never count as coverage.
    rows.append({"club": "Some Non-League XI", "club_league": "serie_a",
                 "is_our_club": False})
    path = root / "external" / "sofascore" / f"friendlies_{season.replace('-', '_')}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _write_matches(root, season, played):
    pd.DataFrame({"season": [season] * played,
                  "home_score": [1.0] * played}).to_parquet(
        root / "parsed" / "matches.parquet", index=False)


def _freeze(monkeypatch, day: _dt.date):
    """Freeze BOTH clocks.

    Patching only `hc.datetime` leaves `config.settings.date` on the real today,
    so `get_current_season()` returns the right answer by accident and the July
    boundary test passes against a check that is still keyed off the calendar —
    verified by mutation, this exact fixture let mutation A survive.
    """
    class _D(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return _dt.datetime(day.year, day.month, day.day, 12, 0, tzinfo=tz)

    class _Date(_dt.date):
        @classmethod
        def today(cls):
            return day

    monkeypatch.setattr(hc, "datetime", _D)
    monkeypatch.setattr("config.settings.date", _Date)


# --------------------------------------------------------------------------
# 1. the season boundary — the mutation the first draft failed
# --------------------------------------------------------------------------
def test_july_resolves_next_seasons_parquet_not_the_calendar_season(env, monkeypatch):
    """On 20 July the writer files friendlies under 2026-2027 while
    get_current_season() still says 2025-2026. The check must follow the WRITER."""
    _freeze(monkeypatch, _dt.date(2026, 7, 20))
    _write_friendlies(env, "2026-2027", sa=SA[:-1])       # one SA club missing
    _write_friendlies(env, "2025-2026")                   # last season: complete
    _write_matches(env, "2025-2026", played=380)          # the early-return bait

    got = hc.check_preseason_coverage()
    assert got["season"] == "2026-2027", "opened the season that just ended"
    assert got["status"] == "WARNING"
    assert got["leagues"]["serie_a"]["without_friendlies"] == [SA[-1]]


@pytest.mark.parametrize("day,writer,calendar", [
    (_dt.date(2026, 6, 1), "2026-2027", "2025-2026"),   # window opens, calendar has not rolled
    (_dt.date(2026, 7, 20), "2026-2027", "2025-2026"),  # peak pre-season, two months of divergence
    (_dt.date(2026, 8, 2), "2026-2027", "2026-2027"),   # they finally agree
    (_dt.date(2026, 5, 31), "2025-2026", "2025-2026"),  # before the window
])
def test_the_two_season_helpers_disagree_through_june_and_july(monkeypatch, day,
                                                               writer, calendar):
    """Guards the PREMISE of the boundary test. If these ever converge, the trap
    is gone and the test above stops proving anything — so pin the divergence
    itself, not just the check's behaviour under it."""
    import config.settings as cs

    _freeze(monkeypatch, day)
    assert sf.current_friendly_season(day) == writer
    assert cs.get_current_season() == calendar


def test_played_matches_in_the_friendly_season_retire_the_signal(env, monkeypatch):
    """Once the new season is under way the stale-table problem is over."""
    _freeze(monkeypatch, _dt.date(2026, 8, 30))
    _write_friendlies(env, "2026-2027", sa=SA[:-3])
    _write_matches(env, "2026-2027", played=40)

    got = hc.check_preseason_coverage()
    assert got["status"] == "OK"
    assert "retired" in got["detail"]


def test_a_handful_of_played_matches_does_not_retire_it_yet(env, monkeypatch):
    _freeze(monkeypatch, _dt.date(2026, 8, 20))
    _write_friendlies(env, "2026-2027", sa=SA[:-3])
    _write_matches(env, "2026-2027", played=5)

    assert hc.check_preseason_coverage()["status"] == "WARNING"


# --------------------------------------------------------------------------
# 2. the empty / short registry — silent full-coverage green
# --------------------------------------------------------------------------
@pytest.mark.parametrize("sa_clubs", [[], SA[:3], SA[:17]])
def test_a_short_club_list_is_unknown_never_ok(env, monkeypatch, sa_clubs):
    """fetch_club_ids returns a SHORT DICT on a 403 — it does not raise. An
    unguarded empty `expected` makes `expected - have` empty and reports perfect
    coverage exactly when we are being blocked."""
    _freeze(monkeypatch, _dt.date(2026, 7, 20))
    _write_friendlies(env, "2026-2027")
    monkeypatch.setattr(sf, "fetch_club_ids", lambda keys: _registry(sa=sa_clubs))

    got = hc.check_preseason_coverage()
    assert got["status"] == "UNKNOWN", "reported coverage off an unvalidated list"
    assert "without_friendlies" not in got["leagues"]["serie_a"]


def test_one_short_league_does_not_mask_a_real_gap_in_the_other(env, monkeypatch):
    _freeze(monkeypatch, _dt.date(2026, 7, 20))
    _write_friendlies(env, "2026-2027", epl=EPL[:-2])
    monkeypatch.setattr(sf, "fetch_club_ids", lambda keys: _registry(sa=[]))

    got = hc.check_preseason_coverage()
    assert got["status"] == "UNKNOWN"
    assert got["leagues"]["premier_league"]["without_friendlies"] == sorted(EPL[-2:])


def test_a_raising_club_lookup_is_unknown_not_a_crash(env, monkeypatch):
    _freeze(monkeypatch, _dt.date(2026, 7, 20))
    _write_friendlies(env, "2026-2027")

    def _boom(keys):
        raise RuntimeError("403")
    monkeypatch.setattr(sf, "fetch_club_ids", _boom)

    got = hc.check_preseason_coverage()
    assert got["status"] == "UNKNOWN"
    assert "RuntimeError" in got["detail"]


# --------------------------------------------------------------------------
# ordinary behaviour
# --------------------------------------------------------------------------
def test_full_coverage_is_ok(env, monkeypatch):
    _freeze(monkeypatch, _dt.date(2026, 7, 20))
    _write_friendlies(env, "2026-2027")

    got = hc.check_preseason_coverage()
    assert got["status"] == "OK"
    assert got["leagues"]["serie_a"]["with_friendlies"] == 20


def test_opponent_rows_never_count_as_coverage(env, monkeypatch):
    """The parquet holds both sides of every friendly; only is_our_club rows are
    ours. Counting the opponent would mark a club covered by having PLAYED it."""
    _freeze(monkeypatch, _dt.date(2026, 7, 20))
    path = _write_friendlies(env, "2026-2027", sa=SA[:-1])
    df = pd.read_parquet(path)
    df = pd.concat([df, pd.DataFrame([{"club": SA[-1], "club_league": "serie_a",
                                       "is_our_club": False}])], ignore_index=True)
    df.to_parquet(path, index=False)

    got = hc.check_preseason_coverage()
    assert got["leagues"]["serie_a"]["without_friendlies"] == [SA[-1]]


def test_outside_the_window_the_check_does_nothing(env, monkeypatch):
    """April has no friendlies by design; an alarm every spring is alarm fatigue."""
    _freeze(monkeypatch, _dt.date(2026, 4, 10))
    got = hc.check_preseason_coverage()
    assert got["status"] == "OK"
    assert "outside" in got["detail"]


def test_a_missing_parquet_in_season_is_a_warning_not_a_silent_ok(env, monkeypatch):
    """No file means NOBODY has the signal — the worst case, not the quiet one."""
    _freeze(monkeypatch, _dt.date(2026, 7, 20))
    got = hc.check_preseason_coverage()
    assert got["status"] == "WARNING"
    assert "no friendlies parquet" in got["detail"]


def test_it_never_raises_on_a_malformed_parquet(env, monkeypatch):
    """A monitor that crashes takes down every other check in the report."""
    _freeze(monkeypatch, _dt.date(2026, 7, 20))
    p = env / "external" / "sofascore" / "friendlies_2026_2027.parquet"
    p.write_text("not a parquet")
    assert hc.check_preseason_coverage()["status"] == "UNKNOWN"
