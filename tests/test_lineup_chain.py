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
