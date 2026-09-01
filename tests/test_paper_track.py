"""The EPL deployment gate demands "50+ settled paper bets, CLV+" — but until
run_paper_track existed, load_predictions() dropped a gated league's file
wholesale and NOTHING could ever settle: the bar was unearnable by
construction. These tests pin the paper track's two safety properties:

1. A gated league's candidate lands in the PAPER journal — never the real
   one — with a flat stake and pipeline_status "paper".
2. Settling a paper bet writes only the paper journal; the real journal and
   the bankroll caches derived from it are untouched.

Every path is monkeypatched at module-attribute level (bet_journal reads
JOURNAL_PATH/PAPER_JOURNAL_PATH at call time), so no fixture can reach the
real files — the jul-13 test-leak lesson.
"""

import json

import pytest

import scripts.betting.bet_journal as bj
import scripts.betting.betting_unified as bu

_EPL = "Arsenal vs Chelsea"


@pytest.fixture
def tmp_journals(tmp_path, monkeypatch):
    real = tmp_path / "bet_journal.json"
    paper = tmp_path / "paper_journal.json"
    monkeypatch.setattr(bj, "JOURNAL_PATH", real)
    monkeypatch.setattr(bj, "PAPER_JOURNAL_PATH", paper)
    monkeypatch.setattr(bj, "_JOURNAL_LOCK_PATH", tmp_path / ".journal.lock")
    return real, paper


def _odds(match):
    return {match: {"totals": [{
        "line": 1.5, "over": 1.29, "under": 3.50, "bookmakers_count": 5,
        "all_bookmakers": [
            {"bookmaker": "Pinnacle", "over": 1.28, "under": 3.60},
            {"bookmaker": "bet365", "over": 1.30, "under": 3.40},
            {"bookmaker": "Unibet", "over": 1.29, "under": 3.50},
        ],
    }]}}


def _write_predictions(upcoming, match, date):
    (upcoming / "predictions_premier_league.json").write_text(json.dumps({
        "predictions": [{"match": match, "date": date,
                         "home_team": match.split(" vs ")[0],
                         "away_team": match.split(" vs ")[1]}],
    }))


def test_gated_league_candidate_goes_to_paper_journal_only(
        tmp_journals, tmp_path, monkeypatch):
    real, paper = tmp_journals
    upcoming = tmp_path / "upcoming"
    upcoming.mkdir()
    # 2099-01-03 is a Saturday — clear of the Monday/Friday stake gate.
    _write_predictions(upcoming, _EPL, "2099-01-03")
    monkeypatch.setattr(bu, "UPCOMING", upcoming)
    monkeypatch.setattr(bu, "load_odds_full", lambda: _odds(_EPL))
    monkeypatch.setattr(bu, "load_goal_predictions", lambda: [
        {"match": _EPL, "date": "2099-01-03", "over_1_5": 0.84}])
    # premier_league must be gated for this test to mean anything
    # (rejection-test-needs-a-true-positive rule).
    assert not bu._league_betting_enabled("premier_league")

    n = bu.run_paper_track()

    assert n == 1, "the gated league's candidate must be journaled as paper"
    assert not real.exists(), "the REAL journal must never see a paper bet"
    bets = list(json.loads(paper.read_text())["bets"].values())
    assert len(bets) == 1
    bet = bets[0]
    assert bet["league"] == "premier_league"
    assert bet["stake"] == bu.PAPER_STAKE
    assert bet["pipeline_status"] == "paper"
    assert bet["status"] == "pending"


def test_enabled_league_is_not_paper_tracked(tmp_journals, tmp_path, monkeypatch):
    _, paper = tmp_journals
    upcoming = tmp_path / "upcoming"
    upcoming.mkdir()
    sa = "Inter vs Napoli"
    (upcoming / "predictions.json").write_text(json.dumps({
        "predictions": [{"match": sa, "date": "2099-01-03"}]}))
    monkeypatch.setattr(bu, "UPCOMING", upcoming)
    monkeypatch.setattr(bu, "load_odds_full", lambda: _odds(sa))
    monkeypatch.setattr(bu, "load_goal_predictions", lambda: [
        {"match": sa, "date": "2099-01-03", "over_1_5": 0.84}])

    n = bu.run_paper_track()

    assert n == 0
    assert not paper.exists() or not json.loads(paper.read_text())["bets"]


def test_settle_paper_bets_grades_ou_and_leaves_real_journal_alone(
        tmp_journals):
    real, paper = tmp_journals
    bj.add_bet({
        "match": _EPL, "date": "2099-01-03", "market": "O/U 1.5",
        "selection": "Over 1.5", "odds": 1.29, "stake": 10.0,
        "league": "premier_league", "pipeline_status": "paper",
        "sharp_implied_prob": 0.7377, "edge_pct": 6.1,
    }, journal_path=paper)

    summary = bj.settle_paper_bets({
        _EPL: {"home_score": 2, "away_score": 1, "status": "finished"}})

    assert summary["settled"] == 1 and summary["won"] == 1
    bet = list(json.loads(paper.read_text())["bets"].values())[0]
    assert bet["status"] == "won"
    assert bet["profit"] == pytest.approx(10.0 * 0.29)
    assert bet["clv_pct"] is not None, "CLV must be computed for the bar"
    assert not real.exists(), "settling paper must not create the real journal"


def test_settle_paper_push_and_unknown_market(tmp_journals):
    _, paper = tmp_journals
    bj.add_bet({"match": _EPL, "date": "2099-01-03", "market": "O/U 2.5",
                "selection": "Over 2.5", "odds": 2.4, "stake": 10.0},
               journal_path=paper)
    bj.add_bet({"match": "X vs Y", "date": "2099-01-03", "market": "Corners",
                "selection": "Over 9.5", "odds": 1.9, "stake": 10.0},
               journal_path=paper)

    summary = bj.settle_paper_bets({
        _EPL: {"home_score": 1, "away_score": 1, "total_goals": 2.5,
               "status": "finished"},
        "X vs Y": {"home_score": 1, "away_score": 0, "status": "finished"},
    })

    # exact-line total -> push; unknown market -> left pending, warned
    assert summary["push"] == 1
    bets = {b["market"]: b for b in
            json.loads(paper.read_text())["bets"].values()}
    assert bets["O/U 2.5"]["status"] == "push"
    assert bets["Corners"]["status"] == "pending"
