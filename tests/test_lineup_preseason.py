#!/usr/bin/env python3
"""Tests for the pre-season friendly signal wired into XI prediction.

`_load_current_season_stats` filters to `season.max()`, so between May and the
first league matchweek the predictor reasons off LAST season's completed table.
Measured 2026-08-01 over 17 Serie A clubs: 194 players who featured in a
pre-season friendly had ZERO rows there (summer arrivals, promoted youth) and
could not be picked at all, while 72 players with >=5/10 starts last season
appeared in no friendly whatsoever and were still ranked as nailed-on starters.

The single most important guarantee here is the FIRST test: with no friendly
data the function must behave exactly as it did before, byte for byte. Every
other effect is opt-in on top of that.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.prediction import lineup_predictor as lp

TEAM = "Juventus"


def _stats(n_matches: int = 10, players: int = 14) -> pd.DataFrame:
    """A believable last-season table: 11 regulars plus rotation."""
    rows = []
    for m in range(n_matches):
        for p in range(players):
            rows.append({
                "team": TEAM,
                "match_id": 1000 + m,
                "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=7 * m),
                "player_id": 500 + p,
                "player_name": f"Player {p}",
                "is_starter": p < 11,
                "minutes": 90 if p < 11 else 15,
                "rating": 6.8,
                "position": "M",
                "shirt_number": p + 1,
                "round": m + 1,
            })
    # A rotation player with only 2 league appearances. The Bayesian prior is
    # deliberately weak against a 10-appearance regular (~17% weight) and such a
    # player saturates at 100, so the prior's effect is only observable here.
    for m in (0, 1):
        rows.append({
            "team": TEAM, "match_id": 1000 + m,
            "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=7 * m),
            "player_id": 520, "player_name": "Rotation Guy",
            "is_starter": True, "minutes": 90, "rating": 6.8,
            "position": "M", "shirt_number": 20, "round": m + 1,
        })
    return pd.DataFrame(rows).sort_values("date")


def _signal(players: dict, club_friendlies: int) -> dict:
    return {"players": players, "club_friendlies": club_friendlies}


def _entry(starts, apps, name="New Guy", pos="F", shirt=99):
    return {
        "starts": starts, "appearances": apps,
        "minutes": 60 * apps,
        "start_pct": round(starts / apps * 100, 1) if apps else 0.0,
        "name": name, "position": pos, "shirt_number": shirt,
    }


def _by_name(freq, name):
    return next((p for p in freq if p["name"] == name), None)


# --------------------------------------------------------------------------
# 1. the regression guard -- everything else is built on this
# --------------------------------------------------------------------------

@pytest.mark.parametrize("empty", [None, {}, {"players": {}, "club_friendlies": 0}])
def test_without_friendly_data_the_prediction_is_unchanged(empty):
    """No signal must mean no behaviour change, not a subtly different number.

    This runs for every club that played no tracked friendly, and for the whole
    league from the moment the season starts, so it is the common path.
    """
    df = _stats()
    legacy = lp.get_starter_frequency(df, TEAM, n_matches=10)
    wired = lp.get_starter_frequency(df, TEAM, n_matches=10, preseason=empty)
    assert json.dumps(legacy, sort_keys=True, default=str) == \
           json.dumps(wired, sort_keys=True, default=str)


# --------------------------------------------------------------------------
# 2. the 194 -- players the predictor could not see at all
# --------------------------------------------------------------------------

def test_a_summer_signing_with_no_league_history_can_now_be_picked():
    df = _stats()
    before = lp.get_starter_frequency(df, TEAM, n_matches=10)
    assert _by_name(before, "New Guy") is None, "precondition: invisible today"

    sig = _signal({9001: _entry(starts=4, apps=4)}, club_friendlies=4)
    after = lp.get_starter_frequency(df, TEAM, n_matches=10, preseason=sig)

    signing = _by_name(after, "New Guy")
    assert signing is not None
    assert signing["preseason_only"] is True
    assert signing["is_new_signing"] is True
    assert signing["appearances"] == 0, "zero LEAGUE appearances by definition"
    assert signing["start_pct"] > 50


def test_a_fringe_preseason_player_does_not_outrank_an_established_starter():
    """Friendly minutes are evidence of presence, not of a guaranteed shirt."""
    df = _stats()
    sig = _signal({9001: _entry(starts=1, apps=5)}, club_friendlies=5)
    freq = lp.get_starter_frequency(df, TEAM, n_matches=10, preseason=sig)

    fringe = _by_name(freq, "New Guy")
    regular = _by_name(freq, "Player 0")
    assert fringe["start_pct"] < regular["start_pct"]


def test_a_player_named_in_zero_friendlies_is_not_invented():
    df = _stats()
    sig = _signal({9001: _entry(starts=0, apps=0)}, club_friendlies=4)
    freq = lp.get_starter_frequency(df, TEAM, n_matches=10, preseason=sig)
    assert _by_name(freq, "New Guy") is None


# --------------------------------------------------------------------------
# 3. the 72 -- regulars who have quietly left
# --------------------------------------------------------------------------

def test_a_last_season_regular_absent_from_preseason_is_downgraded():
    df = _stats()
    present = {500 + p: _entry(2, 3, name=f"Player {p}") for p in range(1, 11)}
    sig = _signal(present, club_friendlies=4)   # Player 0 is missing entirely

    before = lp.get_starter_frequency(df, TEAM, n_matches=10)
    after = lp.get_starter_frequency(df, TEAM, n_matches=10, preseason=sig)

    gone = _by_name(after, "Player 0")
    was = _by_name(before, "Player 0")
    assert gone["start_pct"] < was["start_pct"]


def test_absence_carries_no_information_when_the_club_barely_played():
    """One friendly proves nothing -- half the squad sits those out.

    Measured: only 16 of 38 tracked clubs had >=3 friendlies on 2026-08-01, so
    this gate is the difference between a signal and a coin flip.
    """
    df = _stats()
    sig = _signal({500 + p: _entry(1, 1, name=f"Player {p}") for p in range(1, 11)},
                  club_friendlies=lp.PRESEASON_ABSENT_MIN_MATCHES - 1)
    before = lp.get_starter_frequency(df, TEAM, n_matches=10)
    after = lp.get_starter_frequency(df, TEAM, n_matches=10, preseason=sig)
    assert _by_name(after, "Player 0")["start_pct"] == _by_name(before, "Player 0")["start_pct"]


def test_an_unused_substitute_counts_as_present_and_is_not_penalised():
    """Squad presence, not minutes, is what rules out 'he has been sold'."""
    df = _stats()
    named_but_unused = _entry(starts=0, apps=3, name="Player 0")
    others = {500 + p: _entry(2, 3, name=f"Player {p}") for p in range(1, 11)}
    sig = _signal({500: named_but_unused, **others}, club_friendlies=4)

    freq = lp.get_starter_frequency(df, TEAM, n_matches=10, preseason=sig)
    absent_sig = _signal(others, club_friendlies=4)
    absent = lp.get_starter_frequency(df, TEAM, n_matches=10, preseason=absent_sig)

    assert _by_name(freq, "Player 0")["start_pct"] > _by_name(absent, "Player 0")["start_pct"]


# --------------------------------------------------------------------------
# 4. the informed prior is itself shrunk by its own sample size
# --------------------------------------------------------------------------

def test_one_unused_appearance_does_not_collapse_a_players_prior():
    """Paid measurement: before this shrink, a single unused-sub appearance in a
    club's only friendly drove PRIOR_RATE to 0 and cost Milan's Odogu 23 points.
    """
    df = _stats()
    one = _signal({520: _entry(0, 1, name="Rotation Guy")}, club_friendlies=1)
    five = _signal({520: _entry(0, 5, name="Rotation Guy")}, club_friendlies=5)

    base = _by_name(lp.get_starter_frequency(df, TEAM, n_matches=10), "Rotation Guy")["start_pct"]
    thin = _by_name(lp.get_starter_frequency(df, TEAM, 10, preseason=one), "Rotation Guy")["start_pct"]
    thick = _by_name(lp.get_starter_frequency(df, TEAM, 10, preseason=five), "Rotation Guy")["start_pct"]

    assert thick < thin < base, "more friendly evidence must move the prior further"


# --------------------------------------------------------------------------
# 5. the loader degrades to "no signal" rather than raising
# --------------------------------------------------------------------------

def test_a_missing_friendly_store_yields_no_signal(tmp_path, monkeypatch):
    monkeypatch.setattr(lp, "SOFASCORE_DIR", tmp_path)
    assert lp.load_preseason_signal(TEAM) == {}


def test_the_loader_reads_only_our_own_club_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(lp, "SOFASCORE_DIR", tmp_path)
    pd.DataFrame([
        {"season": "2026-2027", "is_our_club": True, "club": TEAM,
         "sofascore_event_id": 1, "player_id": 500, "player": "Player 0",
         "is_starter": True, "minutes_played": 90, "position": "M",
         "shirt_number": 5},
        {"season": "2026-2027", "is_our_club": False, "club": "Nice",
         "sofascore_event_id": 1, "player_id": 700, "player": "Opponent",
         "is_starter": True, "minutes_played": 90, "position": "M",
         "shirt_number": 7},
    ]).to_parquet(tmp_path / "friendlies_2026_2027.parquet")

    sig = lp.load_preseason_signal(TEAM)
    assert set(sig["players"]) == {500}, "opposition players must never enter the pool"
    assert sig["club_friendlies"] == 1
