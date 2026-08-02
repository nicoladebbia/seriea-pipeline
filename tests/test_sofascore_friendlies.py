#!/usr/bin/env python3
"""Tests for the pre-season club-friendly ingest.

The point of this scraper is AVAILABILITY, not performance: who is fit enough to
be named, who is being trusted with minutes, and what shape the manager is
trialling.  July ratings against fourth-tier opposition are noise, so the tests
below never assert anything about form -- they guard the two things that can
actually break silently:

  1. Friendlies must NEVER leak into the league tables.  A friendly row landing
     in lineups.parquet / player_match_stats.parquet would quietly contaminate
     the training set, and nothing downstream would raise.
  2. The unused substitute must survive the parse.  "Named but got 0 minutes" is
     the signal; dropping those rows would leave only the players who played,
     which is exactly the wrong half of the data.
"""

from __future__ import annotations

import json

import pytest

from scraper import sofascore_friendlies as f

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    """Redirect the module's write target; it binds the path at import time."""
    monkeypatch.setattr(f, "SOFASCORE_DIR", tmp_path / "sofascore")
    return tmp_path / "sofascore"


def _event(tourn_id=853, tourn_name="Club Friendly Games", eid=999,
           home="Juventus", away="Nice", home_id=2687, away_id=2455,
           ts=1785513600, status="finished"):
    return {
        "id": eid,
        "startTimestamp": ts,
        "status": {"type": status},
        "tournament": {"name": tourn_name, "uniqueTournament": {"id": tourn_id}},
        "homeTeam": {"name": home, "id": home_id},
        "awayTeam": {"name": away, "id": away_id},
        "homeScore": {"current": 2},
        "awayScore": {"current": 0},
    }


def _lineups():
    """One starter who played, one starter subbed off, one unused sub."""
    return {
        "confirmed": True,
        "home": {
            "formation": "3-4-2-1",
            "players": [
                {"player": {"name": "Federico Gatti", "id": 1},
                 "shirtNumber": 4, "substitute": False,
                 "position": "D",
                 "statistics": {"minutesPlayed": 90, "rating": 7.0, "touches": 75}},
                {"player": {"name": "Michele Di Gregorio", "id": 2},
                 "shirtNumber": 29, "substitute": False,
                 "position": "G",
                 "statistics": {"minutesPlayed": 62, "rating": 6.6}},
                # named on the bench, never came on -> no statistics block at all
                {"player": {"name": "Unused Reserve", "id": 3},
                 "shirtNumber": 33, "substitute": True,
                 "position": "M"},
            ],
        },
        "away": {
            "formation": "4-3-3",
            "players": [
                {"player": {"name": "Nice Starter", "id": 10},
                 "shirtNumber": 7, "substitute": False,
                 "position": "F",
                 "statistics": {"minutesPlayed": 90, "rating": 6.9}},
            ],
        },
    }


OURS = {2687: ("Juventus", "serie_a")}


# --------------------------------------------------------------------------
# 1. the contamination guard -- the failure that would be silent
# --------------------------------------------------------------------------

def test_only_the_club_friendly_tournament_is_accepted():
    assert f._is_friendly(_event(tourn_id=853)) is True


@pytest.mark.parametrize("tid,label", [(23, "Serie A"), (17, "Premier League"),
                                       (8, "La Liga"), (35, "Bundesliga")])
def test_league_matches_are_rejected_so_they_cannot_reach_the_friendly_store(tid, label):
    assert f._is_friendly(_event(tourn_id=tid, tourn_name=label)) is False


def test_an_event_with_no_unique_tournament_block_is_rejected_not_crashed():
    ev = _event()
    ev["tournament"] = {"name": "Mystery Cup"}
    assert f._is_friendly(ev) is False


def test_the_friendly_writer_never_targets_a_league_parquet(tmp_store):
    """The store path must be its own file, not lineups/player_match_stats."""
    rows = f._parse_lineup(_lineups(), f._event_to_meta(_event(), OURS))
    f._save(rows, season="2026-2027")
    written = {p.name for p in tmp_store.glob("*.parquet")}
    assert written == {"friendlies_2026_2027.parquet"}
    for forbidden in ("lineups.parquet", "player_match_stats.parquet",
                      "all_shots_with_xg.parquet"):
        assert forbidden not in written


# --------------------------------------------------------------------------
# 2. the availability signal
# --------------------------------------------------------------------------

def test_the_unused_substitute_is_kept_with_zero_minutes():
    """Dropping him would delete the single most useful availability fact."""
    rows = f._parse_lineup(_lineups(), f._event_to_meta(_event(), OURS))
    unused = [r for r in rows if r["player"] == "Unused Reserve"]
    assert len(unused) == 1
    assert unused[0]["minutes_played"] == 0
    assert unused[0]["was_used"] is False
    assert unused[0]["is_starter"] is False


def test_starters_and_substitutes_are_distinguished():
    rows = f._parse_lineup(_lineups(), f._event_to_meta(_event(), OURS))
    by_name = {r["player"]: r for r in rows}
    assert by_name["Federico Gatti"]["is_starter"] is True
    assert by_name["Unused Reserve"]["is_starter"] is False


def test_minutes_are_captured_for_a_player_withdrawn_early():
    rows = f._parse_lineup(_lineups(), f._event_to_meta(_event(), OURS))
    gk = next(r for r in rows if r["player"] == "Michele Di Gregorio")
    assert gk["minutes_played"] == 62
    assert gk["was_used"] is True


def test_the_trialled_formation_is_recorded_per_side():
    rows = f._parse_lineup(_lineups(), f._event_to_meta(_event(), OURS))
    assert {r["formation"] for r in rows if r["is_home"]} == {"3-4-2-1"}
    assert {r["formation"] for r in rows if not r["is_home"]} == {"4-3-3"}


def test_a_missing_statistics_block_does_not_raise():
    payload = _lineups()
    for p in payload["home"]["players"]:
        p.pop("statistics", None)
    rows = f._parse_lineup(payload, f._event_to_meta(_event(), OURS))
    assert all(r["minutes_played"] == 0 for r in rows if r["is_home"])


# --------------------------------------------------------------------------
# 3. our-club tagging
# --------------------------------------------------------------------------

def test_the_event_is_tagged_with_which_of_our_leagues_the_club_belongs_to():
    meta = f._event_to_meta(_event(), OURS)
    assert meta["club"] == "Juventus"
    assert meta["club_league"] == "serie_a"
    assert meta["opponent"] == "Nice"
    assert meta["is_home"] is True


def test_an_away_friendly_still_resolves_our_club_and_the_opponent():
    ev = _event(home="Basel", away="Juventus", home_id=2455, away_id=2687)
    meta = f._event_to_meta(ev, OURS)
    assert meta["club"] == "Juventus"
    assert meta["opponent"] == "Basel"
    assert meta["is_home"] is False


def test_club_names_are_canonical_not_raw_sofascore(tmp_store):
    """Raw names ("AC Milan", "SSC Napoli") break every join back into the
    pipeline, which keys on the canonical form ("Milan", "Napoli")."""
    ours = {2692: ("Milan", "serie_a")}
    ev = _event(home="AC Milan", away="Celtic", home_id=2692, away_id=1111)
    rows = f._parse_lineup(_lineups(), f._event_to_meta(ev, ours))
    home_rows = [r for r in rows if r["is_home"]]
    assert {r["club"] for r in home_rows} == {"Milan"}
    # an untracked opponent keeps its raw name rather than being mangled
    assert {r["club"] for r in rows if not r["is_home"]} == {"Celtic"}


def test_a_friendly_between_two_tracked_clubs_tags_both_sides_as_ours():
    """Sassuolo v Parma is a real 2026 pre-season fixture; tagging only the
    side we happened to iterate from would silently lose half the match."""
    ours = {2793: ("Sassuolo", "serie_a"), 2690: ("Parma", "serie_a")}
    ev = _event(home="Sassuolo", away="Parma", home_id=2793, away_id=2690)
    meta = f._event_to_meta(ev, ours)
    rows = f._parse_lineup(_lineups(), meta)
    assert all(r["is_our_club"] for r in rows)
    assert {r["club_league"] for r in rows} == {"serie_a"}


def test_an_untracked_opponent_is_not_tagged_as_ours():
    rows = f._parse_lineup(_lineups(), f._event_to_meta(_event(), OURS))
    assert all(r["is_our_club"] for r in rows if r["is_home"])
    assert not any(r["is_our_club"] for r in rows if not r["is_home"])
    assert {r["club_league"] for r in rows if not r["is_home"]} == {None}


def test_a_friendly_between_two_clubs_we_do_not_track_is_skipped():
    ev = _event(home="Basel", away="Nice", home_id=2455, away_id=9999)
    assert f._event_to_meta(ev, OURS) is None


def test_an_unfinished_friendly_is_not_written():
    ev = _event(status="notstarted")
    assert f._event_to_meta(ev, OURS) is None


# --------------------------------------------------------------------------
# 4. the mutation the writer must survive -- re-running and back-filling
# --------------------------------------------------------------------------

def test_rerunning_the_same_scrape_does_not_duplicate_rows(tmp_store):
    rows = f._parse_lineup(_lineups(), f._event_to_meta(_event(), OURS))
    first = f._save(rows, season="2026-2027")
    second = f._save(rows, season="2026-2027")
    assert len(second) == len(first)


def test_backfilling_an_older_friendly_merges_without_corrupting_existing_rows(tmp_store):
    """Identity must be (event, player) -- never row position.

    A later-discovered EARLIER friendly gets inserted in the middle once sorted;
    a positional key would silently reassign every row after it.
    """
    newer = f._parse_lineup(_lineups(), f._event_to_meta(_event(eid=999, ts=1785513600), OURS))
    f._save(newer, season="2026-2027")

    older = f._parse_lineup(
        _lineups(), f._event_to_meta(_event(eid=555, ts=1784381400, away="Basel"), OURS)
    )
    merged = f._save(older, season="2026-2027")

    assert set(merged["sofascore_event_id"]) == {999, 555}
    gatti = merged[(merged.player == "Federico Gatti") &
                   (merged.sofascore_event_id == 999)]
    assert len(gatti) == 1
    assert gatti.iloc[0]["minutes_played"] == 90
    assert gatti.iloc[0]["opponent"] == "Nice"


def test_a_corrected_rerun_updates_the_row_rather_than_appending_a_second_one(tmp_store):
    rows = f._parse_lineup(_lineups(), f._event_to_meta(_event(), OURS))
    f._save(rows, season="2026-2027")

    fixed = f._parse_lineup(_lineups(), f._event_to_meta(_event(), OURS))
    for r in fixed:
        if r["player"] == "Michele Di Gregorio":
            r["minutes_played"] = 90
    merged = f._save(fixed, season="2026-2027")

    gk = merged[merged.player == "Michele Di Gregorio"]
    assert len(gk) == 1
    assert gk.iloc[0]["minutes_played"] == 90


def test_dtypes_do_not_drift_between_a_populated_and_an_empty_rating_season(tmp_store):
    """A season with no ratings at all stored rating_low_trust as object while a
    populated season stored float64 -- reading both at once then broke."""
    rated = f._parse_lineup(_lineups(), f._event_to_meta(_event(eid=1), OURS))
    a = f._save(rated, season="2026-2027")

    payload = _lineups()
    for side in ("home", "away"):
        for p in payload[side]["players"]:
            p.pop("statistics", None)
    unrated = f._parse_lineup(payload, f._event_to_meta(_event(eid=2), OURS))
    b = f._save(unrated, season="2025-2026")

    assert a["rating_low_trust"].dtype == b["rating_low_trust"].dtype == "float64"
    assert a["shirt_number"].dtype == b["shirt_number"].dtype == "float64"
    assert a["minutes_played"].dtype == b["minutes_played"].dtype == "int64"


def test_saving_no_rows_leaves_an_existing_file_untouched(tmp_store):
    rows = f._parse_lineup(_lineups(), f._event_to_meta(_event(), OURS))
    before = f._save(rows, season="2026-2027")
    after = f._save([], season="2026-2027")
    assert len(after) == len(before)


# --------------------------------------------------------------------------
# 5. the rating column is carried but explicitly untrusted
# --------------------------------------------------------------------------

def test_rating_is_stored_but_flagged_low_trust_in_the_schema():
    rows = f._parse_lineup(_lineups(), f._event_to_meta(_event(), OURS))
    assert "rating_low_trust" in rows[0]
    assert "rating" not in rows[0], (
        "name the column rating_low_trust so no downstream join treats a July "
        "friendly rating as comparable to a league rating"
    )


# --------------------------------------------------------------------------
# 6. opponent provenance -- "who did we actually play?"
# --------------------------------------------------------------------------
# A friendly result is meaningless without knowing the opposition: "Juventus 2-0
# Nice" and "Torino 3-0 ACD Pinzolo" are identical in the payload.  The raw
# opponent name is always kept verbatim; these columns only ADD context beside
# it, so a wrong bucket can be re-derived from the stored facts.

@pytest.mark.parametrize("name", [
    "Arminia Bielefeld", "Rosenborg BK", "Bologna", "Dolomiti Bellunesi",
])
def test_a_senior_club_is_not_mistaken_for_a_youth_side(name):
    """The space-padding is load-bearing.

    A bare substring test for " b" / " ii" flags Bielefeld, Bellunesi and
    Rosenborg BK -- all senior clubs.  Mislabelling them would silently
    discount a real fixture.
    """
    assert f._is_youth_side(name) is False


@pytest.mark.parametrize("name", ["Atalanta U23", "Lazio U20", "Real Madrid B"])
def test_a_genuine_youth_or_reserve_side_is_flagged(name):
    assert f._is_youth_side(name) is True


def test_opponent_tier_buckets_by_stored_facts_not_by_name():
    top5 = {"opponent_league_id": 34, "opponent_league": "Ligue 1"}
    assert f._opponent_tier(top5) == "top5_league"
    assert f._opponent_tier({"opponent_league_id": 99}) == "other_professional"
    assert f._opponent_tier({}) == "lower_or_unknown"
    assert f._opponent_tier({"opponent_is_national": True}) == "national_team"
    # youth wins over league: Atalanta U23 play in Serie C, not Serie A
    assert f._opponent_tier(
        {"opponent_is_youth": True, "opponent_league_id": 34}
    ) == "youth_or_reserve"


def test_an_opponent_is_looked_up_once_then_served_from_cache(monkeypatch):
    """Opponents repeat across a pre-season; re-fetching each time is waste."""
    calls = []

    def _fake(url, *a, **kw):
        calls.append(url)
        return {"team": {"name": "Nice", "national": False,
                         "primaryUniqueTournament": {
                             "id": 34, "name": "Ligue 1",
                             "category": {"priority": 6}}}}

    monkeypatch.setattr(f, "_get_json", _fake)
    monkeypatch.setattr(f, "_jitter_delay", lambda *a, **kw: None)

    cache: dict = {}
    first = f.fetch_opponent_profile(2455, cache)
    second = f.fetch_opponent_profile(2455, cache)

    assert len(calls) == 1, "second lookup must not hit the network"
    assert first == second
    assert first["opponent_league"] == "Ligue 1"
    assert first["opponent_league_id"] == 34
    assert f._opponent_tier(first) == "top5_league"


def test_a_store_written_before_provenance_existed_upgrades_without_duplicating(tmp_store):
    """The real migration: rows already on disk predate these columns.

    Merging a provenance-carrying row onto a provenance-free one must key on
    (event, player) and must not leave the id columns as object dtype.
    """
    rows = f._parse_lineup(_lineups(), f._event_to_meta(_event(), OURS))
    legacy = [{k: v for k, v in r.items()
               if not k.startswith("opponent_") and k != "club_id"} for r in rows]
    before = f._save(legacy, season="2026-2027")
    assert "opponent_id" not in before.columns

    for r in rows:
        r["opponent_league"] = "Ligue 1"
        r["opponent_tier"] = "top5_league"
    after = f._save(rows, season="2026-2027")

    assert len(after) == len(before), "upgrade must not append a second copy"
    assert after["opponent_id"].dtype == "int64"
    assert after["club_id"].dtype == "int64"
    assert set(after["opponent_tier"].dropna()) == {"top5_league"}
    gatti = after[after.player == "Federico Gatti"]
    assert len(gatti) == 1


# --------------------------------------------------------------------------
# opponent-profile cache — a club's league is a PER-SEASON fact
#
# The cache used to carry a "delete this file every August" note in
# DATA_CATALOG.md. That is a manual step nobody performs, so a club promoted or
# relegated over the summer kept its old league forever and every friendly
# against it was bucketed at the wrong strength tier. The stamp makes it
# automatic; these pin the mutation it exists to survive — a rollover.
# --------------------------------------------------------------------------

@pytest.fixture
def tmp_opp_cache(tmp_path, monkeypatch):
    """_OPP_CACHE is bound at import time, so redirect it explicitly."""
    p = tmp_path / "friendly_opponent_profiles.json"
    monkeypatch.setattr(f, "_OPP_CACHE", p)
    return p


def _stamp(monkeypatch, season):
    monkeypatch.setattr(f, "get_current_season", lambda: season)


def test_a_cache_resolved_last_season_is_discarded(tmp_opp_cache, monkeypatch):
    """THE test: a promoted club must not keep last season's league."""
    _stamp(monkeypatch, "2025-2026")
    f._save_opp_cache({"1": {"opponent_league": "Championship"}})

    _stamp(monkeypatch, "2026-2027")   # August rolls over
    assert f._load_opp_cache() == {}, "stale-season profiles must not survive"


def test_a_cache_resolved_this_season_is_kept(tmp_opp_cache, monkeypatch):
    _stamp(monkeypatch, "2026-2027")
    f._save_opp_cache({"1": {"opponent_league": "Premier League"}})
    got = f._load_opp_cache()
    assert got["1"]["opponent_league"] == "Premier League"


def test_an_unstamped_cache_is_kept_not_thrown_away(tmp_opp_cache, monkeypatch):
    """Files written before the stamp existed are current, not stale —
    discarding them would re-fetch hundreds of correct profiles."""
    tmp_opp_cache.write_text(json.dumps({"1": {"opponent_league": "Serie A"}}))
    _stamp(monkeypatch, "2026-2027")
    assert f._load_opp_cache()["1"]["opponent_league"] == "Serie A"


def test_the_season_stamp_is_written_on_save(tmp_opp_cache, monkeypatch):
    _stamp(monkeypatch, "2026-2027")
    f._save_opp_cache({"7": {"opponent_league": "Serie B"}})
    on_disk = json.loads(tmp_opp_cache.read_text())
    assert on_disk[f._OPP_CACHE_SEASON_KEY] == "2026-2027"


def test_the_stamp_key_cannot_collide_with_a_team_id(tmp_opp_cache, monkeypatch):
    """Team ids are numeric strings; the sentinel must never be one."""
    assert not f._OPP_CACHE_SEASON_KEY.isdigit()
    _stamp(monkeypatch, "2026-2027")
    f._save_opp_cache({"853": {"opponent_league": "Serie A"}})
    got = f._load_opp_cache()
    assert got["853"]["opponent_league"] == "Serie A"


def test_a_corrupt_or_non_mapping_cache_rebuilds_instead_of_raising(tmp_opp_cache,
                                                                   monkeypatch):
    _stamp(monkeypatch, "2026-2027")
    tmp_opp_cache.write_text("[]")
    assert f._load_opp_cache() == {}
    tmp_opp_cache.write_text("{not json")
    assert f._load_opp_cache() == {}
