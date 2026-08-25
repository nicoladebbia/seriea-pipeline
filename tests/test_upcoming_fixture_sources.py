#!/usr/bin/env python3
"""Upcoming-fixture loading — the staleness and precedence rules.

The bug this pins (found 2026-08-24): ``load_upcoming_matches`` read only
``manual_matches.json`` and the Odds-API-derived ``upcoming/matches.json``, and
applied **no date filter**. With the Odds API key dead, the second was empty and
the first was frozen on 2026-05-24 — so every daily run happily emitted
predictions for matches played three months earlier. Green logs, wrong fixtures.

The date filter is the half that generalises: without it any source going stale
re-asserts its frozen entries from a different direction. So these tests
exercise the STALE and CONFLICTING cases, not the happy path.

Run with: python3 -m pytest tests/test_upcoming_fixture_sources.py
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Canonical home is scripts/utils/match_timing: the scheduler needs these too,
# and it must not import predict_unified (numpy/pandas + module-level file IO on
# a timer-driven job).
from scripts.utils.match_timing import (  # noqa: E402
    _dedup_key,
    _entry_kickoff,
    _is_future,
    _load_sofascore_fixtures,
)

NOW = datetime(2026, 8, 24, 18, 0, tzinfo=timezone.utc)
PAST = NOW - timedelta(days=92)
SOON = NOW + timedelta(days=4)


def _sofa_row(home, away, ts):
    return {
        "homeTeam": {"name": home},
        "awayTeam": {"name": away},
        "startTimestamp": int(ts.timestamp()),
        "status": {"type": "notstarted"},
    }


# --- the date filter -------------------------------------------------------

def test_a_stale_manual_entry_is_not_upcoming():
    """2026-05-24 was still being predicted on 2026-08-24."""
    assert not _is_future({"commence_time": "2026-05-24T18:45:00Z"}, NOW)


def test_a_real_future_fixture_is_upcoming():
    assert _is_future({"commence_time": "2026-08-28T18:45:00Z"}, NOW)


def test_date_only_entry_still_gets_filtered():
    """Not every source carries commence_time; a bare date must still be judged."""
    assert not _is_future({"date": "2026-05-24"}, NOW)
    assert _is_future({"date": "2026-08-28"}, NOW)


def test_an_undateable_entry_is_dropped_not_kept():
    """Fail closed. Keeping it is how a stale row survives every filter."""
    assert not _is_future({"home_team": "Milan", "away_team": "Venezia"}, NOW)
    assert _entry_kickoff({"home_team": "Milan"}) is None


def test_garbage_timestamps_do_not_raise():
    assert not _is_future({"commence_time": "not-a-date"}, NOW)
    assert not _is_future({"date": ""}, NOW)


def test_a_fixture_beyond_the_horizon_is_not_upcoming():
    """Sofascore carries the whole season; May 2027 is not 'upcoming'."""
    far = (NOW + timedelta(days=200)).isoformat().replace("+00:00", "Z")
    assert not _is_future({"commence_time": far}, NOW)


def test_the_horizon_is_overridable_for_a_wider_sweep():
    far = (NOW + timedelta(days=30)).isoformat().replace("+00:00", "Z")
    assert not _is_future({"commence_time": far}, NOW)
    assert _is_future({"commence_time": far}, NOW, horizon_days=45)


def test_the_horizon_boundary_keeps_the_next_matchweek():
    """A matchweek sits ~4-7 days out; it must never fall outside the default."""
    for d in (1, 4, 7, 10):
        ko = (NOW + timedelta(days=d)).isoformat().replace("+00:00", "Z")
        assert _is_future({"commence_time": ko}, NOW), f"day {d} was excluded"


# --- dedup across sources with different spellings -------------------------

def test_dedup_key_is_normalised_so_sources_actually_collide():
    """Sofascore says 'AC Milan', the manual file says 'Milan'. Raw-name keys
    would treat those as two fixtures and predict the match twice."""
    assert _dedup_key({"home_team": "AC Milan", "away_team": "Venezia"}) == _dedup_key(
        {"home_team": "Milan", "away_team": "Venezia"}
    )
    assert _dedup_key({"home_team": "SSC Napoli", "away_team": "Como"}) == _dedup_key(
        {"home_team": "Napoli", "away_team": "Como"}
    )


def test_different_fixtures_keep_different_keys():
    a = _dedup_key({"home_team": "Milan", "away_team": "Venezia"})
    b = _dedup_key({"home_team": "Venezia", "away_team": "Milan"})
    assert a != b


# --- the sofascore reader --------------------------------------------------

def test_sofascore_reader_drops_played_matches_and_normalises(tmp_path):
    f = tmp_path / "fixtures.json"
    import json

    f.write_text(
        json.dumps(
            [
                _sofa_row("AC Milan", "Venezia", SOON),
                _sofa_row("Inter", "Roma", PAST),  # already played
            ]
        )
    )
    got = _load_sofascore_fixtures(NOW, files=[(f, "serie_a")])
    assert len(got) == 1
    assert got[0]["home_team"] == "Milan"
    assert got[0]["league"] == "serie_a"
    assert got[0]["commence_time"].endswith("Z")


def test_sofascore_reader_tags_each_league_file(tmp_path):
    """The league-parity trap: the EPL file must not inherit 'serie_a'."""
    import json

    sa = tmp_path / "sa.json"
    epl = tmp_path / "epl.json"
    sa.write_text(json.dumps([_sofa_row("AC Milan", "Venezia", SOON)]))
    epl.write_text(json.dumps([_sofa_row("Arsenal", "Chelsea", SOON)]))
    got = _load_sofascore_fixtures(NOW, files=[(sa, "serie_a"), (epl, "premier_league")])
    tags = {g["home_team"]: g["league"] for g in got}
    assert tags["Milan"] == "serie_a"
    assert tags["Arsenal"] == "premier_league"


def test_missing_file_degrades_quietly(tmp_path):
    got = _load_sofascore_fixtures(NOW, files=[(tmp_path / "nope.json", "serie_a")])
    assert got == []


def test_corrupt_file_degrades_quietly(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text("{not json")
    assert _load_sofascore_fixtures(NOW, files=[(f, "serie_a")]) == []


def test_rows_missing_teams_are_skipped_not_fatal(tmp_path):
    import json

    f = tmp_path / "f.json"
    f.write_text(
        json.dumps(
            [
                {"startTimestamp": int(SOON.timestamp())},  # no teams
                _sofa_row("AC Milan", "Venezia", SOON),
            ]
        )
    )
    assert len(_load_sofascore_fixtures(NOW, files=[(f, "serie_a")])) == 1


# --- the freshness monitor must agree with the loader ----------------------

def test_fixture_freshness_check_follows_the_source_the_loader_uses():
    """The monitor watched only the Odds API schedule, so it cried "fixtures
    stale" for 22 days while the source predict_unified actually reads was
    hours old. Blind-to-fixtures is the alert condition, not one cold source."""
    from scripts.pipeline.health_check import _freshest_fixture_source
    from scripts.utils.match_timing import _sofascore_fixture_files

    watched = _freshest_fixture_source()
    loader_sources = {p for p, _ in _sofascore_fixture_files()}
    odds_schedule = watched.parent.name == "upcoming"
    assert watched in loader_sources or odds_schedule, (
        f"monitor watches {watched}, which no fixture loader reads"
    )


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))


# --- the reader's horizon must default ON ----------------------------------
#
# Caught 2026-08-24 by measuring, not by reasoning: `_load_sofascore_fixtures`
# applied no horizon of its own, so the newly-wired weather fetcher asked for
# 740 forecasts running into May 2027. predict_unified happened to be safe only
# because it filtered again downstream. The dangerous option must be the one a
# caller has to type.

def _season_file(tmp_path, now, days):
    import json
    f = tmp_path / "fixtures.json"
    f.write_text(json.dumps([
        _sofa_row(f"Home{d}", f"Away{d}", now + timedelta(days=d)) for d in days
    ]))
    return [(f, "serie_a")]


def test_the_reader_horizons_by_default(tmp_path):
    files = _season_file(tmp_path, NOW, [1, 5, 30, 200])
    got = _load_sofascore_fixtures(NOW, files=files)
    assert len(got) == 2, "a whole-season read escaped the default horizon"


def test_the_full_season_requires_explicitly_asking_for_it(tmp_path):
    files = _season_file(tmp_path, NOW, [1, 5, 30, 200])
    assert len(_load_sofascore_fixtures(NOW, files=files, horizon_days=None)) == 4


def test_a_caller_can_widen_the_horizon_without_unbounding_it(tmp_path):
    files = _season_file(tmp_path, NOW, [1, 5, 30, 200])
    got = _load_sofascore_fixtures(NOW, files=files, horizon_days=45)
    assert len(got) == 3, "explicit horizon not honoured"


def test_the_scheduler_stays_bounded_when_the_file_holds_a_whole_season(tmp_path, monkeypatch):
    """Behavioural, not source-shaped: feed the real reader a season file and
    check what get_kickoff_times actually returns. An inspect.getsource assert
    would pass on a second unbounded call elsewhere and fail on a rename."""
    from scripts.pipeline import scheduler
    from scripts.utils import match_timing

    now = datetime.now(timezone.utc)
    files = _season_file(tmp_path, now, [1, 5, 30, 200])
    monkeypatch.setattr(match_timing, "_sofascore_fixture_files", lambda: files)

    got = {(m["home_team"], m["away_team"]) for m in scheduler.get_kickoff_times()}
    assert ("Home1", "Away1") in got, "the next matchweek fell outside the window"
    assert ("Home200", "Away200") not in got, "scheduler read the whole season"
    assert ("Home30", "Away30") not in got, "scheduler exceeded its 14d horizon"


def test_the_weather_fetcher_only_forecasts_inside_the_horizon(tmp_path, monkeypatch):
    """The measured bug: wiring weather to the shared reader asked for 740
    forecasts running into May 2027. Count the real calls, don't grep the source."""
    from scripts.prediction import weather_integration
    from scripts.utils import match_timing

    now = datetime.now(timezone.utc)
    files = _season_file(tmp_path, now, [1, 5, 30, 200])
    monkeypatch.setattr(match_timing, "_sofascore_fixture_files", lambda: files)

    asked = []
    monkeypatch.setattr(
        weather_integration,
        "get_weather_forecast",
        lambda team, date, time: asked.append((team, date)) or None,
    )
    weather_integration.fetch_all_match_weather()
    assert len(asked) == 2, f"forecast requests not bounded: {len(asked)}"


def test_the_scheduler_dedups_rows_that_carry_no_team_names(tmp_path, monkeypatch):
    """_seen_key normalises to "" for a nameless row, so a naive key would keep
    the FIRST such row and silently drop every later one. Source 2 feeds the
    -3h settlement tail, so that drops settled matches."""
    import json

    from scripts.pipeline import scheduler
    from scripts.utils import match_timing

    monkeypatch.setattr(match_timing, "_sofascore_fixture_files", lambda: [])
    monkeypatch.setattr(scheduler, "DATA_DIR", tmp_path)
    (tmp_path / "upcoming").mkdir()
    soon = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat().replace(
        "+00:00", "Z"
    )
    (tmp_path / "upcoming" / "results.json").write_text(
        json.dumps(
            {
                "results": {
                    "row-a": {"commence_time": soon},
                    "row-b": {"commence_time": soon},
                }
            }
        )
    )
    got = [m for m in scheduler.get_kickoff_times() if m["match"] in ("row-a", "row-b")]
    assert len(got) == 2, f"a nameless row swallowed its siblings: {got}"


# --- model staleness must mean "unseen data", not "old file" ----------------
#
# check_model_freshness flagged STALE purely on file mtime > 30d, so through
# June and July all seven production models sat in the issues list on calendar
# age alone while no football was played. A monitor whose alerts are all noise
# stops being read, and the three real signals were buried under them. The
# metric that actually matters is how many LABELED matches were played after
# the model was trained.

def _matches_parquet(tmp_path, dates):
    import pandas as pd

    d = tmp_path / "parsed"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "match_date": pd.to_datetime(dates),
            "home_score": [1] * len(dates),
            "away_score": [0] * len(dates),
        }
    ).to_parquet(d / "matches.parquet", index=False)
    return d


def _model_dir(tmp_path, mtime):
    md = tmp_path / "models" / "universal"
    md.mkdir(parents=True, exist_ok=True)
    f = md / "catboost_latest.cbm"
    f.write_bytes(b"x")
    import os

    os.utime(f, (mtime, mtime))
    return tmp_path / "models"


def _freshness(tmp_path, monkeypatch, match_dates, model_mtime):
    from scripts.pipeline import health_check as hc

    _matches_parquet(tmp_path, match_dates)
    monkeypatch.setattr(hc, "DATA_DIR", tmp_path)
    monkeypatch.setattr(hc, "MODELS_DIR", _model_dir(tmp_path, model_mtime))
    return hc.check_model_freshness()["catboost_latest"]


def test_an_old_model_with_no_new_results_is_not_stale(tmp_path, monkeypatch):
    """The off-season case: 120 days old, but nothing was played since."""
    trained = datetime(2026, 6, 1, tzinfo=timezone.utc)
    got = _freshness(
        tmp_path, monkeypatch,
        ["2026-05-20", "2026-05-24"],          # all BEFORE training
        trained.timestamp(),
    )
    assert got["unseen_matches"] == 0
    assert got["status"] == "OK", "calendar age alone was treated as staleness"


def test_a_model_that_missed_a_matchweek_is_stale(tmp_path, monkeypatch):
    """...and the check must still do its real job."""
    trained = datetime(2026, 6, 1, tzinfo=timezone.utc)
    got = _freshness(
        tmp_path, monkeypatch,
        [f"2026-08-{d:02d}" for d in range(10, 25)],   # 15 played after training
        trained.timestamp(),
    )
    assert got["unseen_matches"] == 15
    assert got["status"] == "STALE"


def test_a_handful_of_unseen_matches_does_not_trip_it(tmp_path, monkeypatch):
    """Below one matchweek is not worth a retrain alert."""
    trained = datetime(2026, 6, 1, tzinfo=timezone.utc)
    got = _freshness(
        tmp_path, monkeypatch, ["2026-08-10", "2026-08-11"], trained.timestamp()
    )
    assert got["unseen_matches"] == 2
    assert got["status"] == "OK"


def test_an_unmeasurable_model_fails_loud_not_quiet(tmp_path, monkeypatch):
    """No matches.parquet: the count is unknown, so fall back to STALE rather
    than silently reporting a 120-day-old model as healthy."""
    from scripts.pipeline import health_check as hc

    trained = datetime(2026, 6, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(hc, "DATA_DIR", tmp_path)          # no parsed/matches.parquet
    monkeypatch.setattr(hc, "MODELS_DIR", _model_dir(tmp_path, trained.timestamp()))
    got = hc.check_model_freshness()["catboost_latest"]
    assert got["status"] == "STALE"
    assert "unseen_matches" not in got


def test_a_corrupt_matches_parquet_does_not_crash_the_monitor(tmp_path, monkeypatch):
    """The except branch, which the missing-file test never reaches.

    ``_labeled_matches_since_model`` returns early when the parquet is absent,
    so "no matches.parquet" exercises a different path entirely. An unreadable
    or malformed one is what actually enters the handler — and a health monitor
    that raises when its input is corrupt is useless exactly when it matters.
    """
    from scripts.pipeline import health_check as hc

    d = tmp_path / "parsed"
    d.mkdir(parents=True)
    (d / "matches.parquet").write_bytes(b"this is not a parquet file")
    trained = datetime(2026, 6, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(hc, "DATA_DIR", tmp_path)
    monkeypatch.setattr(hc, "MODELS_DIR", _model_dir(tmp_path, trained.timestamp()))

    got = hc.check_model_freshness()["catboost_latest"]
    assert got["status"] == "STALE"
    assert "unseen_matches" not in got


def test_the_training_instant_is_read_as_utc_not_local_time(tmp_path, monkeypatch):
    """``datetime.fromtimestamp(mtime)`` is naive-LOCAL; the parquet is naive-UTC.

    On a machine west of UTC that shifts the training instant backwards — four
    hours on the box this was found on (EDT) — which is enough to pull a whole
    day of fixtures across the boundary and report them unseen. The TZ is
    forced here so the test fails on a UTC CI box too, rather than passing by
    accident of where it runs.
    """
    import os
    import time

    from scripts.pipeline import health_check as hc

    old_tz = os.environ.get("TZ")
    os.environ["TZ"] = "America/New_York"
    time.tzset()
    try:
        # 02:00 UTC on the 1st is still 22:00 on the previous day in New York.
        trained = datetime(2026, 5, 1, 2, 0, tzinfo=timezone.utc)
        _matches_parquet(tmp_path, ["2026-05-01"])       # played the SAME day
        monkeypatch.setattr(hc, "DATA_DIR", tmp_path)
        monkeypatch.setattr(
            hc, "MODELS_DIR", _model_dir(tmp_path, trained.timestamp())
        )
        got = hc.check_model_freshness()["catboost_latest"]
    finally:
        if old_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old_tz
        time.tzset()

    assert got["unseen_matches"] == 0, (
        "the training day was read in local time and slid into the previous day"
    )


# --- team validation must cover both leagues AND survive promotion ----------
#
# validate_team_name checked a hand-typed 20-name Serie A set: frozen on the
# previous season and EPL-blind. It rejected 13 of the 20 live fixtures — the
# season-rollover trap and the league-parity trap in one function.

def test_promoted_clubs_validate():
    """Venezia / Frosinone / Monza came up for 2026-27."""
    from scripts.utils.error_handling import validate_team_name

    for team in ("Venezia", "Frosinone", "Monza"):
        assert validate_team_name(team), f"{team} rejected — list frozen on last season"


def test_premier_league_clubs_validate():
    """The league-parity rule: every team-keyed lookup must know both leagues."""
    from scripts.utils.error_handling import validate_team_name

    for team in ("Crystal Palace", "Man City", "Arsenal", "Coventry City", "Liverpool FC"):
        assert validate_team_name(team), f"{team} rejected — EPL-blind"


def test_validation_still_rejects_what_it_should():
    """Deriving the set from the canonical map must not make it accept anything."""
    from scripts.utils.error_handling import validate_team_name

    for bad in ("", "Not A Team", "Real Madrid FC XI", None, 42):
        assert not validate_team_name(bad), f"{bad!r} wrongly accepted"


def test_validation_rejects_clubs_from_other_leagues():
    """The scope is the two ACTIVE leagues, not every club the map can spell.

    ``TEAM_NAME_MAP`` is the union of five leagues (406 entries), so deriving
    the whitelist from it wholesale let "Real Madrid", "Bayern Munich" and
    "PSG" validate clean — an out-of-league row could then reach the pipeline
    through ``safe_load_matches``. These names are all genuinely IN the map, so
    this fails against that version; a name the map simply cannot spell would
    not have caught it.
    """
    from config.team_names import TEAM_NAME_MAP
    from scripts.utils.error_handling import validate_team_name

    foreign = ("Real Madrid", "Barcelona", "Bayern Munich", "PSG")
    for team in foreign:
        assert team in TEAM_NAME_MAP, f"{team} not in the map — test proves nothing"
        assert not validate_team_name(team), f"{team} wrongly accepted (wrong league)"


def test_validation_rejects_relegated_clubs():
    """A club the map knows but that is not in either 2026-27 roster.

    ``safe_load_matches`` validates UPCOMING fixtures only, so a club that is
    not in this season's league cannot legitimately appear in one.
    """
    from config.team_names import (
        PREMIER_LEAGUE_2026_27,
        SERIE_A_2026_27,
        TEAM_NAME_MAP,
    )
    from scripts.utils.error_handling import validate_team_name

    active = set(SERIE_A_2026_27) | set(PREMIER_LEAGUE_2026_27)
    gone = [c for c in ("Southampton", "Blackpool", "Charlton") if c in TEAM_NAME_MAP]
    assert gone, "no known-inactive club available to test with"
    for team in gone:
        assert team not in active
        assert not validate_team_name(team), f"{team} wrongly accepted (not this season)"


def test_every_live_fixture_passes_validation():
    """The end-to-end claim: 13 of 20 real fixtures used to fail here."""
    from scripts.utils.error_handling import validate_match_data
    from scripts.utils.match_timing import _load_sofascore_fixtures

    fixtures = _load_sofascore_fixtures(datetime.now(timezone.utc))
    if not fixtures:
        import pytest

        pytest.skip("no upcoming fixtures on disk")
    bad = [(m["home_team"], m["away_team"], errs)
           for m in fixtures for ok, errs in [validate_match_data(m)] if not ok]
    assert not bad, f"live fixtures failed validation: {bad}"
