"""The lineup source chain can never fail silently again.

2026-09-05: Sofascore answered 403 "challenge" on every API endpoint, football-data.org
had no key, API-Football's free plan refused the season — and the scheduler, which
runs the fetcher in a subprocess with its output captured, logged only
"Lineup NOT confirmed". Player props were then priced off a predicted XI that
contained a player who was not in the squad.
"""

import json
import logging

import scraper.lineup_fetcher as lf


def test_chain_status_names_every_dead_source(tmp_path, monkeypatch, caplog):
    import scraper.footballdata_lineups as fd
    import scraper.sofascore_events as ss_events
    import scraper.sofascore_lineups as ss

    monkeypatch.setattr(lf, "LINEUP_CHAIN_FILE", tmp_path / "lineup_chain_status.json")
    monkeypatch.setattr(lf, "DATA_DIR", tmp_path)
    import scraper.espn_lineups as es
    monkeypatch.setattr(ss, "fetch_all_lineups", lambda *a, **k: {})
    monkeypatch.setattr(ss_events, "_LAST_FAILURE_STATUS", 403)
    monkeypatch.setattr(es, "fetch_lineups_espn", lambda *a, **k: {})
    monkeypatch.setattr(fd, "fetch_lineups_footballdata", lambda *a, **k: {})
    monkeypatch.delenv("FOOTBALLDATA_KEY", raising=False)
    monkeypatch.delenv("APIFOOTBALL_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)  # the real .env must not leak keys in

    with caplog.at_level(logging.WARNING):
        out = lf.fetch_and_save_lineups({"Roma vs Atalanta": {"commence_time": "2026-09-05T18:45:00Z"}},
                                        deadline_sec=0)
    assert out == {}
    rep = json.loads((tmp_path / "lineup_chain_status.json").read_text())
    assert rep["confirmed"] == [] and rep["n_matches"] == 1
    assert "Sofascore: HTTP 403" in rep["reason"]
    assert "ESPN: XI non ancora pubblicata" in rep["reason"]
    assert "FOOTBALLDATA_KEY non impostata" in rep["reason"]
    assert "APIFOOTBALL_KEY non impostata" in rep["reason"]
    assert any("No lineup source produced a team sheet" in r.message for r in caplog.records)
    assert not (tmp_path / "upcoming" / "confirmed_lineups.json").exists()


def test_chain_status_is_clean_when_a_source_delivers(tmp_path, monkeypatch):
    import scraper.sofascore_lineups as ss

    monkeypatch.setattr(lf, "LINEUP_CHAIN_FILE", tmp_path / "lineup_chain_status.json")
    monkeypatch.setattr(lf, "DATA_DIR", tmp_path)
    sheet = {"home_team": "Roma", "away_team": "Atalanta", "home_lineup": ["a"] * 11, "away_lineup": ["b"] * 11,
             "source_api": "sofascore"}
    monkeypatch.setattr(ss, "fetch_all_lineups", lambda *a, **k: {"Roma vs Atalanta": sheet})
    out = lf.fetch_and_save_lineups({"Roma vs Atalanta": {"commence_time": "2026-09-05T18:45:00Z"}},
                                    deadline_sec=0)
    assert set(out) == {"Roma vs Atalanta"}
    rep = json.loads((tmp_path / "lineup_chain_status.json").read_text())
    assert rep["reason"] is None and rep["confirmed"] == ["Roma vs Atalanta"]
    assert rep["sources"]["sofascore"]["n"] == 1
    saved = json.loads((tmp_path / "upcoming" / "confirmed_lineups.json").read_text())
    assert "Roma vs Atalanta" in saved["matches"]


def test_reason_reads_the_plan_error_when_the_key_exists():
    chain = {"sofascore": {"n": 0, "last_failure_status": 403}, "espn": {"n": 0},
             "football_data": {"key_set": True, "n": 0, "no_lineup_field": True},
             "api_football": {"key_set": True, "n": 0,
                              "error": "{'plan': 'Free plans do not have access to this season'}"}}
    r = lf.lineup_chain_reason(chain)
    assert "football-data.org: piano free senza formazioni" in r
    assert "Free plans do not have access" in r


def test_espn_source_takes_the_second_slot_and_wins_over_football_data(tmp_path, monkeypatch):
    import scraper.espn_lineups as es
    import scraper.footballdata_lineups as fd
    import scraper.sofascore_lineups as ss

    monkeypatch.setattr(lf, "LINEUP_CHAIN_FILE", tmp_path / "lineup_chain_status.json")
    monkeypatch.setattr(lf, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ss, "fetch_all_lineups", lambda *a, **k: {})
    sheet = {"home_team": "Roma", "away_team": "Atalanta", "home_lineup": ["a"] * 11, "away_lineup": ["b"] * 11,
             "home_bench": [], "away_bench": [], "source_api": "espn", "lineup_source": "confirmed"}
    monkeypatch.setattr(es, "fetch_lineups_espn", lambda *a, **k: {"Roma vs Atalanta": sheet})
    monkeypatch.setattr(fd, "fetch_lineups_footballdata", lambda *a, **k: (_ for _ in ()).throw(AssertionError("fd called")))
    monkeypatch.delenv("APIFOOTBALL_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    out = lf.fetch_and_save_lineups({"Roma vs Atalanta": {"commence_time": "2026-09-05T18:45:00Z"}}, deadline_sec=0)
    assert out["Roma vs Atalanta"]["source_api"] == "espn"
    rep = json.loads((tmp_path / "lineup_chain_status.json").read_text())
    assert rep["sources"]["espn"]["n"] == 1 and rep["reason"] is None


def test_espn_parser_on_the_roma_atalanta_specimen(monkeypatch):
    """Shape verified live 2026-09-05 (event 401874781, ita.1): rosters[*].roster[*]
    with `starter`, `athlete.displayName`, `formation`; ESPN spells without
    diacritics (Pasalic). A side whose roster is not released yet is dropped."""
    import scraper.espn_lineups as es

    def roster(team, xi, bench, formation):
        return {"team": {"displayName": team}, "formation": formation,
                "roster": [{"starter": True, "athlete": {"displayName": n}} for n in xi]
                + [{"starter": False, "athlete": {"displayName": n}} for n in bench]}
    summary = {"rosters": [roster("AS Roma", [f"R{i}" for i in range(11)], ["Matìas Soulè"], "3-4-1-2"),
                           roster("Atalanta", ["Marco Carnesecchi", "Éderson"] + [f"A{i}" for i in range(9)],
                                  ["Mario Pasalic", "Giacomo Raspadori"], "4-3-3")]}
    parsed = es.parse_summary_rosters(summary)
    assert set(parsed) == {"Roma", "Atalanta"}          # canonical names, not ESPN's
    assert "Mario Pasalic" in parsed["Atalanta"]["bench"] and parsed["Atalanta"]["formation"] == "4-3-3"
    partial = {"rosters": [roster("AS Roma", ["R1"], [], "")]}
    assert es.parse_summary_rosters(partial) == {}

    board = {"events": [{"id": "401874781", "competitions": [{"competitors": [
        {"homeAway": "home", "team": {"displayName": "AS Roma"}},
        {"homeAway": "away", "team": {"displayName": "Atalanta"}}]}]}]}
    calls = []

    def fake_get(url, params, timeout=15.0):
        calls.append((url.rsplit("/", 1)[-1], params))
        return board if url.endswith("scoreboard") else summary
    monkeypatch.setattr(es, "_get", fake_get)
    from datetime import UTC, datetime
    odds = {"Roma vs Atalanta": {"commence_time": "2026-09-05T18:45:00Z"},
            "Como vs Parma": {"commence_time": "2026-09-14T16:30:00Z"}}     # far: never queried
    out = es.fetch_lineups_espn(odds, now=datetime(2026, 9, 5, 17, 50, tzinfo=UTC))
    assert set(out) == {"Roma vs Atalanta"}
    sheet = out["Roma vs Atalanta"]
    assert sheet["source_api"] == "espn" and sheet["away_lineup"][1] == "Éderson" and len(sheet["away_bench"]) == 2
    assert calls == [("scoreboard", {"dates": "20260905"}), ("summary", {"event": "401874781"})]
    assert es.fetch_lineups_espn(odds, league="ligue_1") == {}   # no ESPN code mapped: nothing, no error


def _health_env(tmp_path, monkeypatch, kickoff_iso, chain=None, sheets=()):
    import scripts.pipeline.health_check as hc
    up = tmp_path / "upcoming"
    up.mkdir(parents=True, exist_ok=True)
    (up / "odds_full.json").write_text(json.dumps({"matches": {"Bologna vs Sassuolo": {"commence_time": kickoff_iso}}}))
    if chain is not None:
        (up / "lineup_chain_status.json").write_text(json.dumps(chain))
    if sheets:
        (up / "confirmed_lineups.json").write_text(json.dumps({"matches": {k: {} for k in sheets}}))
    monkeypatch.setattr(hc, "DATA_DIR", tmp_path)
    return hc


def test_health_is_quiet_when_no_matchday_is_near(tmp_path, monkeypatch):
    from datetime import UTC, datetime
    hc = _health_env(tmp_path, monkeypatch, "2026-09-13T16:00:00Z")
    out = hc.check_lineup_sources(now=datetime(2026, 9, 6, 8, 0, tzinfo=UTC),
                                  probe=lambda: (_ for _ in ()).throw(AssertionError("no probe off-matchday")))
    assert out["status"] == "OK" and out["matchday_near"] is False


def test_health_probes_espn_before_the_sheet_is_due(tmp_path, monkeypatch):
    from datetime import UTC, datetime
    hc = _health_env(tmp_path, monkeypatch, "2026-09-06T16:00:00Z")
    now = datetime(2026, 9, 6, 8, 0, tzinfo=UTC)          # 8h out: probe, nothing due yet
    assert hc.check_lineup_sources(now=now, probe=lambda: 200)["status"] == "OK"
    bad = hc.check_lineup_sources(now=now, probe=lambda: 403)
    assert bad["status"] == "CRITICAL" and "ESPN scoreboard -> 403" in bad["reason"]
    err = hc.check_lineup_sources(now=now, probe=lambda: (_ for _ in ()).throw(OSError("dns")))
    assert err["status"] == "CRITICAL" and "error: dns" in err["reason"]


def test_health_flags_a_due_match_without_a_sheet_with_the_chain_reason(tmp_path, monkeypatch):
    from datetime import UTC, datetime
    chain = {"confirmed": [], "reason": "Sofascore: HTTP 403 (challenge/ban) · ESPN: XI non ancora pubblicata"}
    hc = _health_env(tmp_path, monkeypatch, "2026-09-06T16:00:00Z", chain=chain)
    now = datetime(2026, 9, 6, 15, 20, tzinfo=UTC)         # T-40: due
    out = hc.check_lineup_sources(now=now, probe=lambda: 200)
    assert out["status"] == "CRITICAL" and out["missing_sheets"] == ["Bologna vs Sassuolo"]
    assert "ESPN: XI non ancora pubblicata" in out["reason"]
    # still flagged 2h after kickoff (the sheet never came), quiet after 3h
    assert hc.check_lineup_sources(now=datetime(2026, 9, 6, 18, 0, tzinfo=UTC), probe=lambda: 200)["status"] == "CRITICAL"
    assert hc.check_lineup_sources(now=datetime(2026, 9, 6, 19, 30, tzinfo=UTC), probe=lambda: 200)["status"] == "OK"
    # a sheet on disk (either file) clears it
    hc2 = _health_env(tmp_path, monkeypatch, "2026-09-06T16:00:00Z", chain=chain, sheets=("Bologna vs Sassuolo",))
    assert hc2.check_lineup_sources(now=now, probe=lambda: 200)["status"] == "OK"


def test_health_names_a_fetcher_that_never_ran(tmp_path, monkeypatch):
    from datetime import UTC, datetime
    hc = _health_env(tmp_path, monkeypatch, "2026-09-06T16:00:00Z")   # no chain file at all
    out = hc.check_lineup_sources(now=datetime(2026, 9, 6, 15, 30, tzinfo=UTC), probe=lambda: 200)
    assert out["status"] == "CRITICAL" and "nessun run del fetcher" in out["reason"]


def test_late_sheet_retriggers_the_prediction_update():
    from scripts.pipeline.scheduler import _sheet_landed_after_prediction as f
    assert f({"stages": {"prediction_update": {"triggered_at": "x"}, "lineup_fetch": {"needs_retry": True}}})
    assert not f({"stages": {"lineup_fetch": {"needs_retry": True}}})            # T-30 not run yet: nothing to redo
    assert not f({"stages": {"prediction_update": {"triggered_at": "x"}, "lineup_fetch": {}}})  # sheet was already in
    assert not f({})


def test_every_espn_serie_a_name_maps_to_a_canonical_team():
    """Pinned from ESPN's /ita.1/teams on 2026-09-05 (20 clubs). A rename on
    ESPN's side or a promoted club missing from config/team_names must fail here,
    not silently drop a match from the lineup chain."""
    from config.team_names import normalize_team
    espn = {"AC Milan": "Milan", "AS Roma": "Roma", "Atalanta": "Atalanta", "Bologna": "Bologna",
            "Cagliari": "Cagliari", "Como": "Como", "Fiorentina": "Fiorentina", "Frosinone": "Frosinone",
            "Genoa": "Genoa", "Internazionale": "Inter", "Juventus": "Juventus", "Lazio": "Lazio",
            "Lecce": "Lecce", "Monza": "Monza", "Napoli": "Napoli", "Parma": "Parma", "Sassuolo": "Sassuolo",
            "Torino": "Torino", "Udinese": "Udinese", "Venezia": "Venezia"}
    assert {k: normalize_team(k) for k in espn} == espn


def test_player_stats_coverage_flags_a_finished_match_with_no_stats(tmp_path, monkeypatch):
    """Sofascore challenged -> the evening ingest silently writes nothing ->
    player-prop paper picks never grade. Fixture kickoffs (cached, known ahead)
    vs the dates present in player_match_stats.parquet."""
    from datetime import UTC, datetime

    import pandas as pd

    import scripts.pipeline.health_check as hc
    import scripts.utils.match_timing as mt
    fx = tmp_path / "fixtures.json"
    monkeypatch.setattr(mt, "_sofascore_fixture_files", lambda: [(fx, "serie_a"), (tmp_path / "x.json", "premier_league")])
    monkeypatch.setattr(hc, "DATA_DIR", tmp_path)
    ko = lambda s: int(datetime.fromisoformat(s).replace(tzinfo=UTC).timestamp())  # noqa: E731
    fx.write_text(json.dumps([
        {"startTimestamp": ko("2026-09-04T18:45:00"), "status": {"type": "finished"},
         "homeTeam": {"name": "Como"}, "awayTeam": {"name": "Genoa"}},
        {"startTimestamp": ko("2026-09-05T18:45:00"), "status": {"type": "finished"},
         "homeTeam": {"name": "Roma"}, "awayTeam": {"name": "Atalanta"}},
        {"startTimestamp": ko("2026-09-05T13:00:00"), "status": {"type": "postponed"},
         "homeTeam": {"name": "X"}, "awayTeam": {"name": "Y"}},
        {"startTimestamp": ko("2026-09-13T18:45:00"), "status": {"type": "notstarted"},
         "homeTeam": {"name": "Lazio"}, "awayTeam": {"name": "Milan"}},
    ]))
    pms = tmp_path / "external" / "sofascore" / "player_match_stats.parquet"
    pms.parent.mkdir(parents=True)
    pd.DataFrame({"date": ["2026-09-04"]}).to_parquet(pms)
    # 6h after the Roma match: inside the grace window, only Friday is due -> OK
    assert hc.check_player_stats_coverage(now=datetime(2026, 9, 6, 0, 45, tzinfo=UTC))["status"] == "OK"
    # next morning: Saturday's stats are due and missing -> CRITICAL, postponed fixture ignored
    out = hc.check_player_stats_coverage(now=datetime(2026, 9, 6, 9, 0, tzinfo=UTC))
    assert out["status"] == "CRITICAL" and out["missing_dates"] == ["2026-09-05"] and "1 finished" in out["detail"]
    pd.DataFrame({"date": ["2026-09-04", "2026-09-05"]}).to_parquet(pms)
    assert hc.check_player_stats_coverage(now=datetime(2026, 9, 6, 9, 0, tzinfo=UTC))["status"] == "OK"
