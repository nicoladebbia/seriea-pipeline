"""Tests for the Betfair -> comparison_odds.json adapter.

Two tiers, split along the verifiable line (see .plans/betfair-adapter-plan.md):

  MECHANICAL (verifiable now): runner->outcome, back-price->price, null handling,
  latest-snapshot selection, incomplete-triple rejection. Tested against
  betfair_feed.py's FIXED output contract.

  NAME JOIN (unverifiable until --probe): tested with DELIBERATELY DIVERGENT names
  (Betfair long-form vs our short-form) — never a circular test where both sides
  already say "Inter". Also asserts the loud-and-counted behaviour on a zero-match
  event: no silent empty/partial output.
"""

from __future__ import annotations

import json

import pytest

from scripts.betting.betfair_to_comparison import (
    _classify_runners,
    _split_event_teams,
    build_comparison_odds,
)

# --- helpers -----------------------------------------------------------------

def _market(event: str, runners: list[dict], *, kickoff="2026-08-24T18:45:00Z"):
    """One market in betfair_feed.py's exact output shape (its lines 162-173)."""
    return {
        "event": event,
        "kickoff": kickoff,
        "snapshots": [
            {"at": "2026-08-24T10:00:00Z", "runners": runners, "total_matched": 1000.0},
        ],
    }


def _runner(name, back, lay=None):
    return {"name": name, "back": back, "lay": lay, "implied_mid": None}


# --- MECHANICAL: event splitting --------------------------------------------

def test_split_event_standard_betfair_separator():
    assert _split_event_teams("Inter v AC Milan") == ("Inter", "AC Milan")


def test_split_event_tolerates_vs_and_caps():
    assert _split_event_teams("Napoli vs Roma") == ("Napoli", "Roma")
    assert _split_event_teams("Lazio V Torino") == ("Lazio", "Torino")


def test_split_event_rejects_non_two_team():
    assert _split_event_teams("") is None
    assert _split_event_teams("Some Outright Market") is None


# --- MECHANICAL: runner classification --------------------------------------

def test_classify_maps_home_draw_away_by_name_not_order():
    # Deliberately put AWAY runner first — order must not decide home/away.
    runners = [
        _runner("AC Milan", 2.10),      # away in event "Inter v AC Milan"
        _runner("The Draw", 3.40),
        _runner("Inter", 2.05),         # home
    ]
    mapped, unclassified = _classify_runners(runners, "Inter", "AC Milan")
    assert mapped == {"home": 2.05, "draw": 3.40, "away": 2.10}
    assert unclassified == []


def test_classify_records_missing_back_price():
    runners = [
        _runner("Inter", 2.05),
        _runner("The Draw", None),      # suspended / thin → no back price
        _runner("AC Milan", 2.10),
    ]
    mapped, unclassified = _classify_runners(runners, "Inter", "AC Milan")
    assert "draw" not in mapped
    assert any("no back price" in u for u in unclassified)


# --- NAME JOIN (unverified-until-probe): DIVERGENT names, not circular -------

def test_join_uses_central_normalizer_on_divergent_longforms(tmp_path):
    """Betfair long-form names must join to our short-form match keys.

    This is the whole point of the join: Betfair says 'AC Milan' / 'Hellas Verona',
    our predictions.json says 'Milan' / 'Verona'. If normalize_team regresses, this
    breaks and (per the loud-accounting contract) writes 0 with a warning.
    """
    store = {
        "markets": {
            "1.111": _market(
                "AC Milan v Hellas Verona",
                [
                    _runner("AC Milan", 1.70),
                    _runner("The Draw", 3.80),
                    _runner("Hellas Verona", 5.20),
                ],
            ),
        }
    }
    in_path = tmp_path / "betfair_odds.json"
    in_path.write_text(json.dumps(store))
    out_path = tmp_path / "comparison_odds.json"

    result = build_comparison_odds(in_path=in_path, out_path=out_path, write=True)

    # Joined to OUR short-form key, not Betfair's long-form event string.
    assert result["book"] == "Betfair"
    assert "Milan vs Verona" in result["odds"], result["odds"]
    assert result["odds"]["Milan vs Verona"]["1x2"] == {
        "home": 1.70, "draw": 3.80, "away": 5.20,
    }
    # And it actually persisted.
    written = json.loads(out_path.read_text())
    assert written["odds"]["Milan vs Verona"]["1x2"]["home"] == 1.70


def test_runner_not_matching_event_teams_is_loud_not_silent(tmp_path, caplog):
    """The real join-failure mode: a runner whose normalized name matches NEITHER
    event-team half (a Betfair spelling our map doesn't know, that also differs
    from the event string). It must be logged as unclassified, leave the 1x2
    triple incomplete, and the market must be SKIPPED — never written partial,
    never silently dropped. This is the unverified-until-probe risk made concrete.
    """
    store = {
        "markets": {
            "1.222": _market(
                "Inter v AC Milan",   # event says Inter/Milan...
                [
                    _runner("Inter", 2.05),
                    _runner("The Draw", 3.40),
                    _runner("Milan FC Reserves", 2.10),  # ...but this runner won't
                                                         # normalize to 'Milan' — a
                                                         # genuine unmapped string
                ],
            ),
        }
    }
    in_path = tmp_path / "betfair_odds.json"
    in_path.write_text(json.dumps(store))
    out_path = tmp_path / "comparison_odds.json"

    with caplog.at_level("WARNING"):
        result = build_comparison_odds(in_path=in_path, out_path=out_path, write=True)

    # The unmatched runner breaks the triple → market skipped, nothing written.
    assert result["odds"] == {}
    # And it was LOUD: unclassified runner warned, incomplete warned, and the
    # "saw markets but wrote 0" join-gap alarm fired.
    assert any("unclassified runners" in r.message for r in caplog.records)
    assert any("incomplete 1x2" in r.message for r in caplog.records)
    assert any("WROTE 0" in r.message for r in caplog.records)


def test_incomplete_triple_skipped_and_logged(tmp_path, caplog):
    """A market missing one of home/draw/away is skipped, warned, and never
    written as a partial (partial 1x2 can't be de-vigged cleanly)."""
    store = {
        "markets": {
            "1.333": _market(
                "Inter v AC Milan",
                [
                    _runner("Inter", 2.05),
                    _runner("The Draw", None),   # missing back → only 2 of 3 usable
                    _runner("AC Milan", 2.10),
                ],
            ),
        }
    }
    in_path = tmp_path / "betfair_odds.json"
    in_path.write_text(json.dumps(store))
    out_path = tmp_path / "comparison_odds.json"

    with caplog.at_level("WARNING"):
        result = build_comparison_odds(in_path=in_path, out_path=out_path, write=True)

    assert result["odds"] == {}  # incomplete → nothing written
    assert any("incomplete 1x2" in r.message for r in caplog.records)
    # and the "saw markets but wrote 0" alarm fired
    assert any("WROTE 0" in r.message for r in caplog.records)


def test_latest_snapshot_wins(tmp_path):
    """When a market has multiple snapshots, the adapter uses the latest (close)."""
    store = {
        "markets": {
            "1.444": {
                "event": "Inter v AC Milan",
                "kickoff": "2026-08-24T18:45:00Z",
                "snapshots": [
                    {"at": "t0", "runners": [
                        _runner("Inter", 3.00), _runner("The Draw", 3.30), _runner("AC Milan", 2.40)]},
                    {"at": "t1", "runners": [   # latest — these prices must win
                        _runner("Inter", 2.05), _runner("The Draw", 3.40), _runner("AC Milan", 2.10)]},
                ],
            }
        }
    }
    in_path = tmp_path / "betfair_odds.json"
    in_path.write_text(json.dumps(store))
    result = build_comparison_odds(in_path=in_path, out_path=tmp_path / "out.json", write=True)
    assert result["odds"]["Inter vs Milan"]["1x2"] == {"home": 2.05, "draw": 3.40, "away": 2.10}


# --- CLEAN SKIP: never raise, never block the pipeline ----------------------

def test_missing_input_file_clean_skip(tmp_path):
    result = build_comparison_odds(
        in_path=tmp_path / "does_not_exist.json", out_path=tmp_path / "out.json", write=False
    )
    assert result == {"book": "Betfair", "odds": {}}


def test_empty_markets_clean_skip(tmp_path):
    in_path = tmp_path / "betfair_odds.json"
    in_path.write_text(json.dumps({"markets": {}}))
    result = build_comparison_odds(in_path=in_path, out_path=tmp_path / "out.json", write=False)
    assert result == {"book": "Betfair", "odds": {}}


def test_garbage_input_clean_skip(tmp_path):
    in_path = tmp_path / "betfair_odds.json"
    in_path.write_text("{not valid json")
    result = build_comparison_odds(in_path=in_path, out_path=tmp_path / "out.json", write=False)
    assert result == {"book": "Betfair", "odds": {}}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
