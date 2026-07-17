"""Tests for the rebuilt Sofascore watcher.

The watcher was reconstructed from two artifacts the original left behind, both
committed as fixtures here rather than read out of gitignored ``data/``:

* ``watcher_standings_premier_league_2026_06_01.json`` — its last write.
* ``watcher_last_refresh_2026_06_01.json`` — the heartbeat tick 3055.

The scrape itself is covered by ``tests/test_sofascore_standings.py``; these
tests own the *write policy*, which is the part that was a decision rather than
a move. Nothing here makes a network call or touches the real data dir.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.data import sofascore_watcher as w

FIXTURES = Path(__file__).parent / "fixtures" / "sofascore"
ORACLE = json.loads((FIXTURES / "watcher_standings_premier_league_2026_06_01.json").read_text())
ORACLE_HEARTBEAT = json.loads((FIXTURES / "watcher_last_refresh_2026_06_01.json").read_text())

SPECIMEN = json.loads((FIXTURES / "tournament_standings_serie_a.json").read_text())


@pytest.fixture(autouse=True)
def tmp_data(tmp_path, monkeypatch):
    """Redirect every write off the real data dir.

    The module binds its paths at import time, so monkeypatching DATA_DIR would
    not reach them — the paths themselves have to be replaced.
    """
    monkeypatch.setattr(w, "STANDINGS_DIR", tmp_path / "upcoming")
    monkeypatch.setattr(w, "SOFASCORE_DIR", tmp_path / "sofascore")
    monkeypatch.setattr(w, "TICK_PATH", tmp_path / "sofascore" / ".watcher_tick.json")
    monkeypatch.setattr(w, "HEARTBEAT_PATH", tmp_path / "sofascore" / ".last_refresh.json")
    return tmp_path


@pytest.fixture
def scraped(monkeypatch):
    """Stand in for the scraper, which has its own test module."""

    def _set(payload):
        monkeypatch.setattr(w, "live_standings_via_html", lambda league: payload)

    return _set


def _payload(teams=None):
    """A scraper-shaped payload — all-zero splits, exactly as the HTML yields."""
    teams = teams or ("Arsenal", "Chelsea")
    zero = {"played": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0, "ppg": 0.0}
    return {
        "standings": {
            t: {
                "team": t, "position": i + 1, "played": 10, "wins": 6, "draws": 2,
                "losses": 2, "gf": 18, "ga": 9, "gd": 9, "points": 20,
                "form_last5": "", "home": dict(zero), "away": dict(zero),
                "league": "premier_league",
            }
            for i, t in enumerate(teams)
        },
        "current_matchweek": 10,
        "season": "2025-2026",
        "league": "premier_league",
        "_source": "sofascore_html",
        "_scraped_at": 0.0,
    }


# --------------------------------------------------------------------------
# The deliberate deviation — this is the whole reason the module has a choice
# --------------------------------------------------------------------------


def test_home_and_away_are_omitted_not_zeroed(scraped):
    """The HTML serves one table (type: "total"); splits are unsourceable.

    Emitting them at all — zeroed OR halved like the original — corrupts a real
    record at web/app.py:7528. See test_a_present_block_would_override_* below,
    which pins the mechanism rather than trusting this comment.
    """
    out = w.shape_payload(_payload())

    for team, row in out["standings"].items():
        assert "home" not in row, f"{team} carries a split the HTML cannot source"
        assert "away" not in row
    # ...and nothing else was dropped on the way through.
    assert out["standings"]["Arsenal"]["points"] == 20
    assert out["standings"]["Arsenal"]["form_last5"] == ""  # kept: matches the oracle


def test_a_present_block_would_override_a_real_record(scraped):
    """Reproduce web/app.py:7528's guard against both rejected options.

    `if s_home:` tests the dict's PRESENCE, not whether its values mean
    anything — so a zeroed block and the oracle's halved block are equally
    destructive, and only omission is skipped.
    """
    real_record = {"played": 5, "won": 3, "drawn": 1, "lost": 1}

    def would_override(entry: dict) -> bool:
        s_home = entry.get("home", {})  # web/app.py:7259
        return bool(s_home)             # web/app.py:7528

    assert would_override({"home": {"played": 19, "wins": 0}})  # the oracle's block
    assert would_override({"home": dict.fromkeys(real_record, 0)})  # the zeroed block

    ours = w.shape_payload(_payload())["standings"]["Arsenal"]
    assert not would_override(ours), "the real computed record would be overridden"


def test_the_display_path_degrades_to_a_dash_rather_than_raising():
    """The other consumer of the splits, reproduced: web/app.py:7038.

    Enumerating the readers (2026-07-16) found nine, and every one either
    defaults or guards — this is the one that would raise if someone ever
    tightened it to `entry["home"]["wins"]`. Absent splits must arrive at the
    frontend as played=0, which `teams.html:481` (`hr.played ? … : '-'`)
    renders as a dash. A dash is the honest answer; the original rendered a
    fabricated one.
    """
    def home_rec(entry: dict) -> dict:
        s_home = entry.get("home", {})  # web/app.py:7038
        return {  # web/app.py:7041-7045
            "w": s_home.get("wins", 0), "d": s_home.get("draws", 0),
            "l": s_home.get("losses", 0), "played": s_home.get("played", 0),
            "ppg": s_home.get("ppg", 0),
        }

    ours = w.shape_payload(_payload())["standings"]["Arsenal"]
    assert home_rec(ours) == {"w": 0, "d": 0, "l": 0, "played": 0, "ppg": 0}
    assert not home_rec(ours)["played"], "teams.html:481 renders '-' only on a falsy played"


def test_the_oracle_block_is_internally_incoherent():
    """Why the original's shape is not a contract worth honouring.

    Every oracle team reads played=19, won=0, drawn=0, lost=0. 19 != 0+0+0.
    It is a shape with one field filled by arithmetic, not a record.
    """
    for row in ORACLE["standings"].values():
        h = row["home"]
        assert h["played"] == row["played"] // 2  # ...the halving, and
        assert h["wins"] + h["draws"] + h["losses"] == 0  # ...nothing to back it
        assert h["played"] != h["wins"] + h["draws"] + h["losses"]


def test_the_halving_is_only_pinned_at_a_completed_season():
    """The oracle cannot discriminate played//2 from a real split source.

    Every team sits at played=38 — the one point where home and away are
    necessarily equal. This is why the formula was not carried forward.
    """
    assert {r["played"] for r in ORACLE["standings"].values()} == {38}


# --------------------------------------------------------------------------
# Shape, against the original's own file
# --------------------------------------------------------------------------


def test_top_level_shape_matches_the_oracle(scraped):
    out = w.shape_payload(_payload())
    assert set(out) == set(ORACLE)
    assert out["source"] == ORACLE["source"] == "sofascore_html"


def test_per_team_keys_match_the_oracle_except_the_dropped_splits(scraped):
    ours = w.shape_payload(_payload())["standings"]["Arsenal"]
    theirs = next(iter(ORACLE["standings"].values()))
    assert set(theirs) - set(ours) == {"home", "away"}, "dropped more than intended"
    assert not set(ours) - set(theirs), "invented a key the original did not write"


def test_generated_at_is_tz_aware(scraped):
    """The project rule: every timestamp written here is UTC-aware ISO.

    A naive one silently returns -1 from monitor._iso_age_hours.
    """
    from datetime import datetime

    out = w.shape_payload(_payload())
    assert datetime.fromisoformat(out["generated_at"]).tzinfo is not None
    assert datetime.fromisoformat(ORACLE["generated_at"]).tzinfo is not None


# --------------------------------------------------------------------------
# Failure must not destroy the table
# --------------------------------------------------------------------------


@pytest.mark.parametrize("failure", [{}, {"standings": {}}, None])
def test_a_failed_scrape_leaves_the_existing_file_untouched(scraped, tmp_data, failure):
    """A 403 must not blank the table. The breaker/retry live in the scraper;
    a falsy return here means "no table this tick", never "write an empty one".
    """
    path = tmp_data / "upcoming" / "standings_premier_league.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"standings": {"Arsenal": {"points": 85}}}))

    scraped(failure)
    assert w.refresh_standings("premier_league") == ("failed", 0)
    assert json.loads(path.read_text())["standings"]["Arsenal"]["points"] == 85


def test_an_unplayed_table_never_overwrites_a_real_one(scraped, tmp_data):
    """The hazard the live probe caught, and the reason MW0 is refused.

    Measured 2026-07-16: the EPL page already served the 26/27 table (20 rows,
    every played=0) while get_current_season() still said "2025-2026". A tick
    would have replaced the real 38-played 25/26 final table with an all-zeros
    one carrying the wrong season stamp. The plist was loaded and failing every
    600s at the time, so this was live, not theoretical.
    """
    path = tmp_data / "upcoming" / "standings_premier_league.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"season": "2025-2026", "standings": {"Arsenal": {"played": 38}}}))

    preseason = _payload()
    preseason["current_matchweek"] = 0
    for row in preseason["standings"].values():
        row["played"] = 0
    scraped(preseason)

    assert w.refresh_standings("premier_league") == ("preseason", 0)
    survived = json.loads(path.read_text())
    assert survived["standings"]["Arsenal"]["played"] == 38, "destroyed the real table"
    assert survived["season"] == "2025-2026"


def test_preseason_is_not_reported_as_a_failure(scraped):
    """An empty off-season table is a legitimate result — the same call
    fetch_upcoming_matches makes about an empty fixture list. Reporting it as a
    failure would hold /api/data-freshness's staleness banner red all summer.
    """
    preseason = _payload()
    preseason["current_matchweek"] = 0
    scraped(preseason)

    hb = w.run_tick()
    assert hb["any_failure"] is False
    assert hb["standings_html_failure"] is False
    assert hb["did_standings_refresh"] is False
    assert hb["leagues"]["premier_league"]["status"] == "preseason"


def test_a_kicked_off_season_does_write_over_the_old_table(scraped, tmp_data):
    """The guard is about the matchweek, not the season — so MW1 replacing a
    38-played table proceeds. That drop is correct: it is a real new season.
    """
    path = tmp_data / "upcoming" / "standings_premier_league.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"standings": {"Arsenal": {"played": 38}}}))

    fresh = _payload()
    fresh["current_matchweek"] = 1
    for row in fresh["standings"].values():
        row["played"] = 1
    scraped(fresh)

    assert w.refresh_standings("premier_league") == ("written", 2)
    assert json.loads(path.read_text())["standings"]["Arsenal"]["played"] == 1


def test_a_failed_tick_is_still_a_heartbeat_and_still_exits_zero(scraped):
    """launchd must not see a failed job every time Cloudflare has a mood."""
    scraped({})
    hb = w.run_tick()
    assert hb["any_failure"] is True
    assert hb["standings_html_failure"] is True
    assert hb["standings_json_teams"] == {"premier_league": 0}
    assert w.main() == 0


# --------------------------------------------------------------------------
# Heartbeat + tick
# --------------------------------------------------------------------------


def test_heartbeat_carries_what_the_freshness_endpoint_reads(scraped):
    """/api/data-freshness (web/app.py:2302-2304) reads exactly these."""
    scraped(_payload())
    hb = w.run_tick()
    assert hb["completed_at"] and hb["started_at"]
    assert hb["any_failure"] is False
    assert isinstance(hb["leagues"], dict)
    # ...and the endpoint's own parse of it must survive.
    from datetime import datetime

    assert datetime.fromisoformat(hb["completed_at"].replace("Z", "+00:00"))


def test_heartbeat_does_not_fabricate_work_it_did_not_do(scraped):
    """The original reported incidents_scraped/matches_live; this module does
    neither, so it omits them. A zero would assert "I looked and found none".
    """
    scraped(_payload())
    hb = w.run_tick()
    for fabricated in ("incidents_scraped", "matches_live", "matches_pre_window",
                       "matches_post_window", "did_full_refresh", "api_failure"):
        assert fabricated not in hb
        assert fabricated in ORACLE_HEARTBEAT  # ...and the original did carry it


def test_tick_increments_from_the_stored_counter(scraped, tmp_data):
    w.TICK_PATH.parent.mkdir(parents=True)
    w.TICK_PATH.write_text(json.dumps({"tick": 3055, "updated_at": "x"}))
    assert w.next_tick() == 3056
    assert w.next_tick() == 3057
    assert json.loads(w.TICK_PATH.read_text())["tick"] == 3057


@pytest.mark.parametrize("corrupt", ["", "not json", '{"tick": "banana"}', "{}"])
def test_a_corrupt_tick_file_restarts_the_count_instead_of_crashing(tmp_data, corrupt):
    """Bookkeeping must never take the scrape down."""
    w.TICK_PATH.parent.mkdir(parents=True, exist_ok=True)
    w.TICK_PATH.write_text(corrupt)
    assert w.next_tick() == 1


def test_the_watcher_never_writes_the_serie_a_file(scraped, tmp_data):
    """standings_generator.py owns standings.json — it has the real form and
    real splits this scraper cannot produce. Writing it here would degrade it.
    """
    assert "serie_a" not in w.WATCHED_LEAGUES
    scraped(_payload())
    w.run_tick()
    assert not (tmp_data / "upcoming" / "standings.json").exists()


def test_importing_the_watcher_starts_no_threads():
    """The reason scraper/sofascore_standings.py was extracted at all:
    importing web.app starts the auto-settle loop AND the odds auto-poll.
    """
    import subprocess
    import sys

    code = (
        "import threading;"
        "import scripts.data.sofascore_watcher;"
        "print(threading.active_count())"
    )
    # noqa S603: the "untrusted input" is the literal string above, and a
    # subprocess is the point — thread count is only meaningful in a fresh
    # interpreter that has not already imported half the test suite.
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert out.returncode == 0, out.stderr
    assert int(out.stdout.strip()) == 1, "the watcher spawned a background thread"
