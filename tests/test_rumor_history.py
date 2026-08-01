"""Rumor-lifecycle store: does it survive the mutations it exists for?

A steady-state test (scrape the same thing twice, still green) would prove
nothing here.  Every test below changes something between runs -- a rumor
appears, disappears, is re-posted under a new URL, or the scraper goes blind --
because those transitions ARE the product.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from scripts.data.rumor_history import (
    annotate_status,
    load_history,
    load_scrape_log,
    record_run,
)

CLUBS = ["Atalanta", "Inter"]


def _rumor(team="Atalanta", player="Sebastiano Esposito",
           club="Cagliari Calcio", url="https://tm/thread/1#post_1", value=15e6):
    return {"team": team, "player_name": player, "current_club": club,
            "age": 24, "market_value_eur": value, "market_value_text": "€15.00m",
            "source_date": "01/08/2026", "source_url": url}


def _df(*rows):
    return pd.DataFrame(list(rows)) if rows else pd.DataFrame()


def _cov(ok=CLUBS, failed=()):
    return {**{t: "ok" for t in ok}, **{t: "fetch_failed" for t in failed}}


def _day(n):
    return datetime(2026, 8, 1, tzinfo=UTC) + timedelta(days=n)


@pytest.fixture()
def tm(tmp_path):
    """Isolated store.  NEVER let a test write the real parquet -- the
    ledger-drift incident in this repo was a test writing its fixture into
    production data."""
    return tmp_path


# ---------------------------------------------------------------- accumulate

def test_first_run_seeds_lifecycle(tm):
    record_run(_df(_rumor()), _cov(), now=_day(0), tm_dir=tm)
    h = load_history(tm)
    assert len(h) == 1
    assert h.iloc[0]["times_seen"] == 1
    assert h.iloc[0]["first_seen"] == h.iloc[0]["last_seen"]


def test_same_rumor_next_day_extends_not_duplicates(tm):
    record_run(_df(_rumor()), _cov(), now=_day(0), tm_dir=tm)
    record_run(_df(_rumor()), _cov(), now=_day(1), tm_dir=tm)
    h = load_history(tm)
    assert len(h) == 1, "a repeated rumor must not mint a second row"
    assert h.iloc[0]["times_seen"] == 2
    assert h.iloc[0]["first_seen"] == _day(0).isoformat()
    assert h.iloc[0]["last_seen"] == _day(1).isoformat()
    assert annotate_status(tm_dir=tm).iloc[0]["days_alive"] == pytest.approx(1.0)


def test_reposted_under_new_url_keeps_one_lifetime(tm):
    """The whole reason source_url is NOT in the key.  Transfermarkt mints a
    new forum post_id for the same rumor; keying on it would reset first_seen
    and destroy the lifetime measurement."""
    record_run(_df(_rumor(url="https://tm/thread/1#post_1")), _cov(),
               now=_day(0), tm_dir=tm)
    record_run(_df(_rumor(url="https://tm/thread/1#post_9999")), _cov(),
               now=_day(3), tm_dir=tm)
    h = load_history(tm)
    assert len(h) == 1
    assert h.iloc[0]["first_seen"] == _day(0).isoformat()
    assert h.iloc[0]["source_url"].endswith("post_9999"), "latest URL wins"
    assert annotate_status(tm_dir=tm).iloc[0]["days_alive"] == pytest.approx(3.0)


def test_attributes_refresh_to_latest(tm):
    record_run(_df(_rumor(value=15e6)), _cov(), now=_day(0), tm_dir=tm)
    record_run(_df(_rumor(value=22e6)), _cov(), now=_day(1), tm_dir=tm)
    assert load_history(tm).iloc[0]["market_value_eur"] == 22e6


def test_mid_sequence_insertion_leaves_existing_rows_intact(tm):
    """Identity must be content-owned.  A rumor appearing between two others
    must not shift anyone else's lifecycle."""
    a, b = _rumor(player="A"), _rumor(player="B")
    record_run(_df(a, b), _cov(), now=_day(0), tm_dir=tm)
    record_run(_df(a, _rumor(player="A2"), b), _cov(), now=_day(1), tm_dir=tm)
    h = load_history(tm).set_index("player_name")
    assert set(h.index) == {"A", "A2", "B"}
    assert h.loc["A", "first_seen"] == h.loc["B", "first_seen"] == _day(0).isoformat()
    assert h.loc["A2", "first_seen"] == _day(1).isoformat()
    assert h.loc["A", "times_seen"] == h.loc["B", "times_seen"] == 2
    assert h.loc["A2", "times_seen"] == 1


# ------------------------------------------------- the coverage distinction

def test_absent_while_club_covered_is_a_real_drop(tm):
    record_run(_df(_rumor()), _cov(), now=_day(0), tm_dir=tm)
    record_run(_df(), _cov(), now=_day(1), tm_dir=tm)   # club fetched, rumor gone
    row = annotate_status(tm_dir=tm).iloc[0]
    assert row["is_dropped"] and not row["is_live"]
    assert row["days_alive"] == pytest.approx(0.0), "died the day after it appeared"


def test_absent_while_scraper_blind_is_NOT_a_drop(tm):
    """The bug this store was designed to avoid.  Four days of 403s must not
    look identical to four days of the rumor being dropped -- those are
    opposite labels for the transfer-prediction question."""
    record_run(_df(_rumor()), _cov(), now=_day(0), tm_dir=tm)
    for d in (1, 2, 3, 4):
        record_run(_df(), _cov(ok=[], failed=CLUBS), now=_day(d), tm_dir=tm)
    row = annotate_status(tm_dir=tm).iloc[0]
    assert not row["is_dropped"], "scraper downtime must never read as a drop"
    assert row["is_live"]


def test_partial_coverage_only_judges_the_clubs_actually_fetched(tm):
    record_run(_df(_rumor(team="Atalanta"), _rumor(team="Inter", player="X")),
               _cov(), now=_day(0), tm_dir=tm)
    # Inter fetched (its rumor is gone); Atalanta 403'd.
    record_run(_df(), _cov(ok=["Inter"], failed=["Atalanta"]), now=_day(1), tm_dir=tm)
    h = annotate_status(tm_dir=tm).set_index("team")
    assert h.loc["Inter", "is_dropped"]
    assert not h.loc["Atalanta", "is_dropped"]


def test_a_dropped_rumor_that_returns_is_live_again(tm):
    record_run(_df(_rumor()), _cov(), now=_day(0), tm_dir=tm)
    record_run(_df(), _cov(), now=_day(1), tm_dir=tm)
    assert annotate_status(tm_dir=tm).iloc[0]["is_dropped"]
    record_run(_df(_rumor()), _cov(), now=_day(2), tm_dir=tm)
    row = annotate_status(tm_dir=tm).iloc[0]
    assert row["is_live"] and row["times_seen"] == 2


def test_days_dark_flags_an_unreliable_verdict(tm):
    record_run(_df(_rumor(team="Atalanta")), _cov(ok=["Atalanta"]), now=_day(0), tm_dir=tm)
    record_run(_df(_rumor(team="Inter", player="X")), _cov(ok=["Inter"]),
               now=_day(6), tm_dir=tm)
    h = annotate_status(tm_dir=tm).set_index("team")
    assert h.loc["Atalanta", "days_dark"] == pytest.approx(6.0)
    assert h.loc["Inter", "days_dark"] == pytest.approx(0.0)


# --------------------------------------------------------------- robustness

def test_total_scrape_failure_preserves_history(tm):
    record_run(_df(_rumor()), _cov(), now=_day(0), tm_dir=tm)
    summ = record_run(None, _cov(ok=[], failed=CLUBS), now=_day(1), tm_dir=tm)
    assert summ["status"] == "failed"
    assert len(load_history(tm)) == 1, "a dead scrape must never erase history"
    assert not annotate_status(tm_dir=tm).iloc[0]["is_dropped"]


def test_rows_from_a_failed_club_are_ignored(tm):
    """Defence in depth: if the scraper ever returns rows for a club it also
    reported as failed, trust the coverage verdict."""
    record_run(_df(_rumor(team="Atalanta")), _cov(ok=["Inter"], failed=["Atalanta"]),
               now=_day(0), tm_dir=tm)
    assert len(load_history(tm)) == 0


def test_scrape_log_appends_one_row_per_run(tm):
    record_run(_df(_rumor()), _cov(), now=_day(0), tm_dir=tm)
    record_run(_df(_rumor()), _cov(ok=["Inter"], failed=["Atalanta"]),
               now=_day(1), tm_dir=tm)
    lg = load_scrape_log(tm)
    assert len(lg) == 2
    assert list(lg["status"]) == ["ok", "partial"]
    assert lg.iloc[1]["failed_teams"] == "Atalanta"
    assert lg.iloc[0]["teams_covered"] == 2


def test_rerunning_the_same_timestamp_is_idempotent(tm):
    record_run(_df(_rumor()), _cov(), now=_day(0), tm_dir=tm)
    record_run(_df(_rumor()), _cov(), now=_day(0), tm_dir=tm)
    h = load_history(tm)
    assert len(h) == 1
    assert annotate_status(tm_dir=tm).iloc[0]["days_alive"] == pytest.approx(0.0)


def test_leagues_do_not_collide(tm):
    record_run(_df(_rumor(team="Atalanta")), _cov(ok=["Atalanta"]),
               season="2026-2027", league="serie_a", now=_day(0), tm_dir=tm)
    record_run(_df(_rumor(team="Atalanta")), _cov(ok=["Atalanta"]),
               season="2026-2027", league="premier_league", now=_day(0), tm_dir=tm)
    assert len(load_history(tm)) == 2


def test_empty_store_annotates_without_raising(tm):
    out = annotate_status(tm_dir=tm)
    assert out.empty
    assert {"days_alive", "is_dropped", "is_live", "days_dark"} <= set(out.columns)


def test_no_tmp_files_left_behind(tm):
    record_run(_df(_rumor()), _cov(), now=_day(0), tm_dir=tm)
    assert not list(tm.glob("*.tmp")), "atomic write must clean up"


def test_scraper_accepts_the_coverage_out_param():
    """Guards the wiring: rumor_history is worthless if the scraper stops
    reporting per-club coverage."""
    import inspect

    from scraper.transfermarkt import scrape_rumors
    assert "coverage" in inspect.signature(scrape_rumors).parameters


def test_scraper_actually_fills_coverage_for_ok_and_failed_clubs(tmp_path, monkeypatch):
    """The signature test above only proves the parameter exists.  This proves
    it is POPULATED, and that a 403 club is recorded as failed rather than
    silently vanishing -- the exact confusion rumor_history must avoid.

    The network is mocked and TM_DIR redirected: calling the real scraper with
    ``only_teams`` would overwrite the production 459-row rumors parquet with
    one club's rows.
    """
    import requests

    from scraper import transfermarkt as tmk

    monkeypatch.setattr(tmk, "TM_DIR", tmp_path)
    monkeypatch.setattr(tmk.time, "sleep", lambda *_: None)
    monkeypatch.setattr(tmk, "_get_league_teams",
                        lambda league: {"Atalanta": ("atalanta", 800),
                                        "Inter": ("inter", 46)})

    class _Resp:
        text = "<html><table class='items'><tbody></tbody></table></html>"

        def raise_for_status(self):
            return None

    def _fake_get(url, **_kw):
        if "inter" in url:
            raise requests.RequestException("403 Forbidden")
        return _Resp()

    monkeypatch.setattr(tmk.requests, "get", _fake_get)

    coverage: dict[str, str] = {}
    tmk.scrape_rumors(season="2026-2027", league="serie_a", coverage=coverage)
    assert coverage == {"Atalanta": "ok", "Inter": "fetch_failed"}, (
        "a club that parsed to ZERO rumors is still covered; a 403 club is not"
    )
