"""Tests for the rebuilt fetch_upcoming_matches.

The module had no oracle (its only surviving artifact was the synthetic
fallback), so the verification here is different from live_reconciliation's:
the parser is mapped against a **real** cached Odds API envelope
(``tests/fixtures/odds_api/event_envelope.json``), and the output shape is
asserted against what the actual consumers read.

Nothing here makes a network call — the API key is deactivated off-season.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from scripts.data import fetch_upcoming_matches as fum

FIXTURE = Path(__file__).parent / "fixtures" / "odds_api" / "event_envelope.json"


@pytest.fixture
def real_event():
    """A genuine Odds API event envelope this repo fetched and cached."""
    return json.loads(FIXTURE.read_text())


def test_fixture_is_the_real_envelope(real_event):
    """Guard: the specimen must keep the fields that make it evidence.

    If someone replaces it with a hand-written mock missing these, the tests
    below would still pass while verifying nothing.
    """
    assert set(real_event) >= {"id", "sport_key", "commence_time", "home_team", "away_team"}
    assert real_event["sport_key"] == "soccer_italy_serie_a"
    assert "bookmakers" not in real_event, "fixture should be the /events shape"


def test_maps_real_envelope_to_match_record(real_event):
    """The whole point: real API data through the real mapper."""
    m = fum._event_to_match(real_event, "serie_a")

    assert m["home_team"] == "Sassuolo"
    assert m["away_team"] == "Como"
    assert m["commence_time"] == "2026-04-17T16:30:00Z"
    assert m["date"] == "2026-04-17"
    assert m["time"] == "16:30"
    assert m["league"] == "serie_a"
    assert m["source"] == "odds_api"
    # The id the odds layer joins on — the reason this source was chosen.
    assert m["event_id"] == "a2d714226e359e11994632f530a047e8"


def test_does_not_fabricate_venue_or_matchweek(real_event):
    """The synthetic fallback templated venue as f"{home} Stadium".

    The Odds API supplies neither venue nor matchweek. Emitting them would
    reintroduce exactly the fabrication that made the old artifact useless as an
    oracle.
    """
    m = fum._event_to_match(real_event, "serie_a")
    assert "venue" not in m
    assert "matchweek" not in m


def test_incomplete_event_is_dropped_not_half_emitted():
    assert fum._event_to_match({}, "serie_a") is None
    assert fum._event_to_match({"home_team": "Milan"}, "serie_a") is None
    assert (
        fum._event_to_match(
            {"home_team": "Milan", "away_team": "Como", "commence_time": ""}, "serie_a"
        )
        is None
    )


def test_saved_shape_is_what_notify_reads(real_event, tmp_path, monkeypatch):
    """notify.py:1950 reads raw_schedule["matches"], then per-match
    commence_time / home_team / away_team / league."""
    out = tmp_path / "matches.json"
    monkeypatch.setattr(fum, "OUTPUT_PATH", out)

    fum.save_upcoming_matches([fum._event_to_match(real_event, "serie_a")])

    payload = json.loads(out.read_text())
    assert payload["count"] == 1
    assert isinstance(payload["matches"], list)
    mm = payload["matches"][0]
    for key in ("commence_time", "home_team", "away_team", "league"):
        assert mm.get(key), f"consumer reads {key}"
    # The old artifact had no league key, so the digest tagged everything
    # "unknown". Regression guard.
    assert mm["league"] != "unknown"


def test_fetched_at_is_utc_aware(tmp_path, monkeypatch):
    """Repo rule: persisted timestamps are UTC-aware ISO strings."""
    from datetime import datetime

    out = tmp_path / "matches.json"
    monkeypatch.setattr(fum, "OUTPUT_PATH", out)
    fum.save_upcoming_matches([])

    parsed = datetime.fromisoformat(json.loads(out.read_text())["fetched_at"])
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0


def test_empty_result_is_success_not_failure(tmp_path, monkeypatch):
    """Off-season returns no events. That must write a valid empty file, not
    raise — scheduler.py:925 guards with `if matches:` and continues."""
    out = tmp_path / "matches.json"
    monkeypatch.setattr(fum, "OUTPUT_PATH", out)
    monkeypatch.setattr(fum, "_fetch_league_events", lambda league: [])

    matches = fum.get_upcoming_matches(["serie_a"])
    assert matches == []
    fum.save_upcoming_matches(matches)
    assert json.loads(out.read_text())["count"] == 0


def test_a_dead_league_does_not_sink_the_others(real_event, monkeypatch):
    """A 401/timeout on one league must not lose the other's fixtures.

    _fetch_league_events swallows its own errors and returns [], so a dead
    league surfaces here as an empty list.
    """
    good = fum._event_to_match(real_event, "premier_league")
    monkeypatch.setattr(
        fum, "_fetch_league_events", lambda lg: [] if lg == "serie_a" else [good]
    )
    assert fum.get_upcoming_matches(["serie_a", "premier_league"]) == [good]


def test_fetch_league_events_swallows_a_failed_call(monkeypatch):
    """The 401 the deactivated key actually returns must not raise."""

    def boom(*a, **k):
        raise requests.RequestException("401 DEACTIVATED_KEY")

    monkeypatch.setattr(fum.requests, "get", boom)
    monkeypatch.setattr(fum, "check_rate_limit", lambda *a, **k: (True, ""))
    assert fum._fetch_league_events("serie_a") == []


def test_output_sorted_by_kickoff(real_event, monkeypatch):
    later = dict(real_event, commence_time="2026-04-18T16:30:00Z", id="later")
    earlier = dict(real_event, commence_time="2026-04-16T16:30:00Z", id="earlier")
    monkeypatch.setattr(
        fum,
        "_fetch_league_events",
        lambda league: [
            fum._event_to_match(later, league),
            fum._event_to_match(earlier, league),
        ],
    )
    out = fum.get_upcoming_matches(["serie_a"])
    assert [m["event_id"] for m in out] == ["earlier", "later"]
