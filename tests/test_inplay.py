"""In-play paper engine: conditioned simulator, market baseline, picks, journal, settle, backtest."""
import json

import pytest

from scripts.betting import inplay
from scripts.models import goal_process as gp


@pytest.fixture(scope="module")
def prof():
    return gp.load_profile()


# ------------------------------------------------------------- simulator

def test_state_at_kickoff_matches_the_pre_match_simulation(prof):
    pre = gp.market_probs(gp.simulate(1.6, 1.1, prof, n=8000, seed=1))
    st = gp.market_probs(gp.simulate_from_state(1.6, 1.1, 0, (0, 0), prof, n=8000, seed=1))
    for k in ("home_win", "draw", "away_win", "over_2_5"):
        assert abs(pre[k] - st[k]) < 0.02


def test_a_two_goal_lead_late_is_nearly_decided_and_the_board_is_kept(prof):
    p = gp.simulate_from_state(1.6, 1.1, 85, (2, 0), prof, n=4000)
    pr = gp.market_probs(p)
    assert pr["home_win"] > 0.95 and pr["away_win"] < 0.01
    assert int(p["home_final"].min()) >= 2 and int(p["ht_home"].min()) >= 2  # goals on the board stay on every read


def test_market_profile_reproduces_the_market_it_was_read_from(prof):
    h2h = {"home": 0.55, "draw": 0.25, "away": 0.20}
    base = gp.market_profile(h2h, 0.50, prof, n=6000)
    pr = gp.market_probs(gp.simulate_from_state(base["xg_h"], base["xg_a"], 0, (0, 0), prof, k=base["k"], n=8000))
    assert abs(pr["home_win"] - 0.55) < 0.05 and abs(pr["away_win"] - 0.20) < 0.05
    assert abs(pr["over_2_5"] - 0.50) < 0.03


# ------------------------------------------------------------- baseline

def _snap(minute, score, h, d, a, ts="2026-09-05T18:50:00+00:00", totals=None):
    s = {"ts": ts, "min": minute, "score": list(score), "avg_odds": {"home": h, "draw": d, "away": a}}
    if totals:
        s["totals"] = totals
    return s


def test_baseline_prefers_pre_match_odds_then_an_early_level_snapshot(prof):
    e = {"pre_match_odds": {"home": 1.67, "draw": 3.88, "away": 5.17}, "snapshots": [_snap(30, (1, 0), 1.2, 6.0, 12.0)]}
    b = inplay.baseline_for_entry(e, prof, n=2000)
    assert b["h2h_source"] == "pre_match_odds" and b["xg_h"] > b["xg_a"] and e["inplay_baseline"] is b
    e2 = {"snapshots": [_snap(4, (0, 0), 1.7, 3.8, 5.0)]}
    assert inplay.baseline_for_entry(e2, prof, n=2000)["h2h_source"] == "first_snapshot"
    assert inplay.baseline_for_entry({"snapshots": [_snap(30, (1, 0), 1.2, 6.0, 12.0)]}, prof, n=2000) is None
    assert inplay.baseline_for_entry({"snapshots": [_snap(4, (1, 0), 1.2, 6.0, 12.0)]}, prof, n=2000) is None


# ------------------------------------------------------------- picks

def test_picks_are_1x2_only_and_stale_totals_lines_never_qualify(prof):
    base = gp.market_profile({"home": 0.6, "draw": 0.22, "away": 0.18}, 0.55, prof, n=2000)
    # 84th minute at 1-0 with the pre-match "Over 0.5 @ 2.32" still in the feed: fair 1.00, a fake edge
    snap = _snap(84, (1, 0), 1.05, 12.0, 40.0, totals={"all_lines": {"0.5": {"over": 2.32, "under": 1.6}}})
    fair = inplay.fair_for_snapshot(base, snap, prof, n=2000)
    rows = inplay.edges(snap, fair)
    assert rows[0]["market"] == "inplay_ou" and rows[0]["fair"] == 1.0  # it IS the biggest edge...
    pick = inplay.best_pick(rows, 84)
    assert pick is None or pick["market"] == "inplay_1x2"  # ...and it is never picked


def test_best_pick_thresholds_cap_and_minute():
    rows = [{"market": "inplay_1x2", "selection": "Draw", "fair": 0.30, "market_prob": 0.15, "odds": 6.0, "edge": 0.15},
            {"market": "inplay_1x2", "selection": "Away", "fair": 0.09, "market_prob": 0.02, "odds": 40.0, "edge": 0.07},
            {"market": "inplay_1x2", "selection": "Home", "fair": 0.61, "market_prob": 0.83, "odds": 1.2, "edge": -0.22}]
    p = inplay.best_pick(rows, 60)
    assert p["selection"] == "Draw" and p["over_cap"] is True and p["edge_pct"] == 15.0
    assert inplay.best_pick(rows, 86) is None
    rows[0]["edge"] = 0.06
    assert inplay.best_pick(rows, 60)["over_cap"] is False
    rows[0]["edge"] = 0.04
    assert inplay.best_pick(rows, 60) is None  # Away has +7% but fair 9% is inside the noise


# ------------------------------------------------------------- live hook

@pytest.fixture
def journal(tmp_path):
    return tmp_path / "inplay_journal.json"


def _entry():
    return {"commence_time": "2026-09-05T18:45:00Z", "pre_match_odds": {"home": 1.67, "draw": 3.88, "away": 5.17},
            "status": "second_half", "snapshots": []}


def test_on_snapshot_fires_once_per_selection_only_after_a_score_change(journal, monkeypatch, prof):
    monkeypatch.setattr(inplay, "ping_mode", lambda: "on")
    pings = []
    e = _entry()
    s0 = _snap(5, (0, 0), 1.7, 3.8, 5.0, ts="t0")
    e["snapshots"].append(s0)
    assert inplay.on_snapshot("AS Roma vs Atalanta BC", e, s0, prof=prof, journal_path=journal, notify_fn=lambda *a: pings.append(a), n=2000) is None
    assert "fair" in s0 and "best_edge" in s0  # priced even when nothing fires
    # Atalanta score at 81': the books have Roma at 10.0 — the simulator says ~3%, no pick on Roma;
    # force a state where the draw IS mispriced to see the pick path end to end
    s1 = _snap(60, (0, 1), 6.0, 4.5, 1.5, ts="t1")
    e["snapshots"].append(s1)
    rec = inplay.on_snapshot("AS Roma vs Atalanta BC", e, s1, prof=prof, journal_path=journal, notify_fn=lambda *a: pings.append(a), n=2000)
    assert rec is not None and rec["market"] == "inplay_1x2" and rec["bet_id"] and len(pings) == 1
    assert e["inplay_picks"][0]["selection"] == rec["selection"]
    j = json.loads(journal.read_text())
    assert len(j["bets"]) == 1 and next(iter(j["bets"].values()))["pipeline_status"] == "inplay:paper"
    # same state again (no change) → nothing; same selection after another change → blocked by the entry's own list
    s2 = _snap(62, (0, 1), 6.0, 9.0, 1.15, ts="t2")
    e["snapshots"].append(s2)
    assert inplay.on_snapshot("AS Roma vs Atalanta BC", e, s2, prof=prof, journal_path=journal, notify_fn=lambda *a: pings.append(a), n=2000) is None
    assert len(pings) == 1


def test_ping_is_off_by_default_and_a_broken_journal_never_raises(journal, monkeypatch, prof):
    monkeypatch.setattr(inplay, "ping_mode", lambda: "off")
    monkeypatch.setattr(inplay, "journal_pick", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk")))
    pings = []
    e = _entry()
    e["snapshots"].append(_snap(5, (0, 0), 1.7, 3.8, 5.0, ts="t0"))
    s1 = _snap(60, (0, 1), 6.0, 4.5, 1.5, ts="t1")
    e["snapshots"].append(s1)
    rec = inplay.on_snapshot("m", e, s1, prof=prof, journal_path=journal, notify_fn=lambda *a: pings.append(a), n=2000)
    assert rec is not None and rec["bet_id"] is None and pings == []


# ------------------------------------------------------------- settle

def test_settle_grades_1x2_and_takes_clv_from_the_next_snapshot(journal, monkeypatch, prof):
    monkeypatch.setattr(inplay, "ping_mode", lambda: "off")
    e = _entry()
    e["snapshots"] += [_snap(5, (0, 0), 1.7, 3.8, 5.0, ts="t0"), _snap(60, (0, 1), 6.0, 4.5, 1.5, ts="t1")]
    rec = inplay.on_snapshot("AS Roma vs Atalanta BC", e, e["snapshots"][-1], prof=prof, journal_path=journal, n=2000)
    assert rec
    e["snapshots"].append(_snap(61, (0, 1), 5.0, 8.0, 1.2, ts="t2"))
    matchday = {"matches": {"AS Roma vs Atalanta BC": e}}
    assert inplay.settle_for_matchday(matchday, journal_path=journal) == 0  # not finished
    e["status"], e["final_score"] = "completed", [1, 1]
    assert inplay.settle_for_matchday(matchday, journal_path=journal) == 1
    bet = next(iter(json.loads(journal.read_text())["bets"].values()))
    expected = {"Home": "lost", "Draw": "won", "Away": "lost"}[rec["selection"]]
    assert bet["status"] == expected and bet["closing_odds"] == {"Home": 5.0, "Draw": 8.0, "Away": 1.2}[rec["selection"]]
    assert bet["clv_pct"] is not None and e["inplay_picks"][0]["status"] == expected
    assert inplay.settle_for_matchday(matchday, journal_path=journal) == 0  # idempotent


def test_grade_handles_totals_push_and_void():
    assert inplay._grade({"market": "inplay_ou", "side": "over", "line": 2.0}, (1, 1)) == "push"
    assert inplay._grade({"market": "inplay_ou", "side": "under", "line": 2.5}, (1, 1)) == "won"
    assert inplay._grade({"market": "inplay_ou"}, (1, 1)) == "void"
    assert inplay._grade({"market": "inplay_1x2", "selection": "Draw"}, (2, 2)) == "won"


# ------------------------------------------------------------- backtest

def test_backtest_runs_on_a_stored_matchday_and_writes_the_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(inplay, "BACKTEST_PATH", tmp_path / "bt.json")
    day = {"matches": {"AS Roma vs Atalanta BC": {
        "status": "completed", "final_score": [2, 1], "pre_match_odds": {"home": 1.67, "draw": 3.88, "away": 5.17},
        "snapshots": [_snap(4, (0, 0), 1.65, 3.77, 5.07, ts="t0"), _snap(30, (0, 1), 6.0, 9.0, 1.15, ts="t1"),
                      _snap(60, (0, 1), 8.0, 4.0, 1.4, ts="t2", totals={"all_lines": {"0.5": {"over": 2.3, "under": 1.6}}}),
                      _snap(89, (1, 1), 6.5, 1.17, 15.0, ts="t3"), _snap(90, (2, 1), 1.01, 19.0, 101.0, ts="t4")]}}}
    f = tmp_path / "2026-09-05.json"
    f.write_text(json.dumps(day))
    r = inplay.backtest([str(f)], n=1500)
    assert r["matches"] == 1 and r["priced_snapshots"] == 4 and (tmp_path / "bt.json").exists()
    assert r["skill_vs_inplay_market_1x2"] is not None and r["passes_gate"] is False  # n < 200
    for rule in ("state_change", "any_snapshot"):
        assert all(p["market"] == "inplay_1x2" for p in r["sample_picks"][rule])


# ------------------------------------------------------------- red cards, baselines

def test_a_red_card_moves_the_fair_price_against_the_ten_man_side(prof):
    assert (prof.get("red_mult") or {}).get("short"), "profile.json has no red_mult: run goal_process --measure-red"
    level = gp.market_probs(gp.simulate_from_state(1.4, 1.1, 60, (1, 1), prof, n=6000, seed=3))
    home_red = gp.market_probs(gp.simulate_from_state(1.4, 1.1, 60, (1, 1), prof, n=6000, seed=3, red=(1, 0)))
    assert home_red["home_win"] < level["home_win"] - 0.05 and home_red["away_win"] > level["away_win"] + 0.05


def test_reds_at_counts_only_cards_already_shown():
    e = {"live_events": [{"type": "card", "card_type": "yellow", "minute": 10, "is_home": True},
                         {"type": "card", "card_type": "red", "minute": 40, "is_home": True},
                         {"type": "card", "card_type": "yellowRed", "minute": 70, "is_home": False}]}
    assert inplay.reds_at(e, 39) == (0, 0) and inplay.reds_at(e, 40) == (1, 0) and inplay.reds_at(e, 90) == (1, 1)


def test_baseline_falls_back_to_the_closing_snapshot_line(monkeypatch, prof):
    monkeypatch.setattr(inplay, "_closing_line", lambda e, mk: {"home": 1.67, "draw": 3.88, "away": 5.17, "source": "closing_snapshot"})
    e = {"commence_time": "2026-09-05T18:45:00Z", "snapshots": [_snap(30, (1, 0), 1.2, 6.0, 12.0)]}
    b = inplay.baseline_for_entry(e, prof, n=2000, mk="AS Roma vs Atalanta BC")
    assert b and b["h2h_source"] == "closing_snapshot" and b["xg_h"] > b["xg_a"]


def test_model_baseline_uses_archived_xg_and_is_cached_per_variant(monkeypatch, prof):
    monkeypatch.setattr(inplay, "_archived_xg", lambda e, mk: {"xg_h": 2.2, "xg_a": 0.7})
    e = {"pre_match_odds": {"home": 1.67, "draw": 3.88, "away": 5.17}, "snapshots": []}
    m = inplay.baseline_for_entry(e, prof, n=2000, mk="m", baseline="model")
    assert m["baseline"] == "model" and m["xg_h"] == 2.2
    k = inplay.baseline_for_entry(e, prof, n=2000, mk="m", baseline="market")
    assert k["baseline"] == "market" and k["xg_h"] != 2.2
    monkeypatch.setattr(inplay, "_archived_xg", lambda e, mk: None)
    assert inplay.baseline_for_entry({"pre_match_odds": e["pre_match_odds"], "snapshots": []}, prof, n=2000, baseline="model") is None
