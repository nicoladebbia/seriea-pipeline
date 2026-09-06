"""Output contract on the T-30 money path (run_pre_kickoff Step 4 -> Step 5).

The failure this pins: every step of the pre-kickoff run swallows exceptions so
the cycle survives, and until 2026-09-06 a failed Step 4 fell through to Step 5,
which read the PREVIOUS predictions.json off disk and journaled real bets on it.
The contract is checked on the artifact (mtime, coverage), not on return values.
"""
import json
import os
import time

import pytest

import scripts.pipeline.run_full_pipeline as rfp

SA = "Inter vs Udinese"
EPL = "Arsenal vs Chelsea"
ODDS = {
    SA: {"home_team": "Inter", "away_team": "Udinese", "commence_time": "2026-09-06T18:45:00Z"},
    EPL: {"home_team": "Arsenal", "away_team": "Chelsea", "commence_time": "2026-09-06T19:00:00Z"},
}


def _write(path, matches):
    path.write_text(json.dumps({"predictions": [{"match": m} for m in matches]}))


@pytest.fixture
def t30(monkeypatch, tmp_path):
    upcoming = tmp_path / "upcoming"
    upcoming.mkdir()
    monkeypatch.setattr(rfp, "DATA_DIR", tmp_path)
    calls = {"engine": 0, "failures": [], "tickets": 0, "windows": {k: "imminent" for k in ODDS},
             "step4_writes": True, "step4_raises": None}

    import scripts.data.odds_fetcher as odds_fetcher
    monkeypatch.setattr(odds_fetcher, "fetch_and_save_odds", lambda use_cache=True: dict(ODDS))
    monkeypatch.setattr(odds_fetcher, "fetch_pick_markets", lambda hours_ahead=6.5: {"events": {}})
    import scripts.utils.match_timing as match_timing
    # classify is called with commence_time; map it back through the odds entries
    monkeypatch.setattr(match_timing, "classify_match_window",
                        lambda ct, *a, **k: next(calls["windows"][k] for k, v in ODDS.items()
                                                 if v["commence_time"] == ct))
    import scraper.lineup_fetcher as lineup_fetcher
    monkeypatch.setattr(lineup_fetcher, "fetch_and_save_lineups", lambda *a, **k: {})
    import scripts.prediction.ensemble_prediction_engine as engine

    def run_engine(use_ensemble=True):
        if calls["step4_raises"]:
            raise calls["step4_raises"]
        rows = [{"match": k} for k in ODDS]
        if calls["step4_writes"]:
            _write(upcoming / "predictions.json", list(ODDS))
        return {"predictions": rows}
    monkeypatch.setattr(engine, "run_ensemble_predictions", run_engine)
    import scripts.betting.bet_journal as bet_journal
    monkeypatch.setattr(bet_journal, "get_pending_bets", lambda include_superseded=False: [])
    import scripts.betting.betting_unified as bu

    def gen(bankroll):
        calls["engine"] += 1
        return {"summary": {"total_bets": 0, "total_stake": 0.0}, "bets": []}
    monkeypatch.setattr(bu, "generate_unified_report", gen)
    monkeypatch.setattr(bu, "save_report", lambda r: None)
    monkeypatch.setattr(rfp, "_send_t30_ticket",
                        lambda *a, **k: calls.__setitem__("tickets", calls["tickets"] + 1))
    monkeypatch.setattr(rfp, "_notify_t30_failure",
                        lambda stage, reason: calls["failures"].append((stage, reason)))
    monkeypatch.setattr(rfp, "_nag_vpn_for_betfair", lambda: None)
    import scripts.data.betfair_feed as betfair_feed
    monkeypatch.setattr(betfair_feed, "fetch_match_odds", lambda *a, **k: 0)

    _write(upcoming / "goal_predictions.json", list(ODDS))  # morning artifact, may be old
    calls["upcoming"] = upcoming
    calls["bu"] = bu
    return calls


def _age(path, seconds):
    old = time.time() - seconds
    os.utime(path, (old, old))


def test_happy_path_runs_the_engine_once_and_sends_the_ticket(t30):
    rfp.run_pre_kickoff(1000.0)
    assert t30["engine"] == 1 and t30["tickets"] == 1 and t30["failures"] == []


def test_step4_raising_aborts_before_the_engine(t30):
    t30["step4_raises"] = RuntimeError("model file gone")
    assert rfp.run_pre_kickoff(1000.0) is None
    assert t30["engine"] == 0 and t30["tickets"] == 0
    (stage, reason), = t30["failures"]
    assert stage == "aborted before Step 5" and "Step 4 raised (RuntimeError: model file gone)" in reason


def test_predictions_json_from_before_this_run_aborts(t30):
    # Step 4 "succeeds" but writes nothing: yesterday's file is what Step 5 would read.
    _write(t30["upcoming"] / "predictions.json", list(ODDS))
    _age(t30["upcoming"] / "predictions.json", 600)
    t30["step4_writes"] = False
    assert rfp.run_pre_kickoff(1000.0) is None
    assert t30["engine"] == 0
    (_, reason), = t30["failures"]
    assert "predictions.json predates this run by 10 min" in reason


def test_imminent_serie_a_match_missing_from_predictions_aborts(t30, monkeypatch):
    import scripts.prediction.ensemble_prediction_engine as engine

    def only_epl(use_ensemble=True):
        _write(t30["upcoming"] / "predictions.json", [EPL])
        return {"predictions": [{"match": EPL}]}
    monkeypatch.setattr(engine, "run_ensemble_predictions", only_epl)
    assert rfp.run_pre_kickoff(1000.0) is None
    (_, reason), = t30["failures"]
    assert reason == f"predictions.json has no row for {SA}"


def test_missing_epl_row_is_not_a_contract_failure(t30, monkeypatch):
    # predictions.json is Serie A only; a gated league never has a row there.
    import scripts.prediction.ensemble_prediction_engine as engine

    def only_sa(use_ensemble=True):
        _write(t30["upcoming"] / "predictions.json", [SA])
        return {"predictions": [{"match": SA}]}
    monkeypatch.setattr(engine, "run_ensemble_predictions", only_sa)
    _write(t30["upcoming"] / "goal_predictions.json", [SA])
    rfp.run_pre_kickoff(1000.0)
    assert t30["engine"] == 1 and t30["failures"] == []


def test_goal_predictions_missing_aborts_but_old_is_fine(t30):
    _age(t30["upcoming"] / "goal_predictions.json", 10 * 3600)   # morning artifact
    rfp.run_pre_kickoff(1000.0)
    assert t30["engine"] == 1 and t30["failures"] == []
    (t30["upcoming"] / "goal_predictions.json").unlink()
    assert rfp.run_pre_kickoff(1000.0) is None
    (_, reason), = t30["failures"]
    assert reason == "goal_predictions.json missing"


def test_no_serie_a_match_in_play_means_no_contract(t30):
    # Only an EPL match is imminent: nothing Step 5 could commit, stale file is irrelevant.
    t30["windows"][SA] = "distant"
    t30["step4_writes"] = False
    _write(t30["upcoming"] / "predictions.json", [])
    _age(t30["upcoming"] / "predictions.json", 600)
    rfp.run_pre_kickoff(1000.0)
    assert t30["engine"] == 1 and t30["failures"] == []


def test_engine_failure_sends_the_failure_card_not_the_no_action_ticket(t30, monkeypatch):
    def boom(bankroll):
        raise ValueError("odds shape")
    monkeypatch.setattr(t30["bu"], "generate_unified_report", boom)
    rfp.run_pre_kickoff(1000.0)
    assert t30["tickets"] == 0
    (stage, reason), = t30["failures"]
    assert stage == "betting engine" and reason == "ValueError: odds shape"


def test_verify_helper_reads_serie_a_keys_from_the_odds_map():
    keys = rfp._t30_serie_a_keys([SA, EPL, "Nowhere FC vs Nobody"], ODDS)
    assert SA in keys and EPL not in keys
