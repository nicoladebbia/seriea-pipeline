"""Market promotion gate: a paper market earns real stakes by its settled record.

Every test redirects BOTH journals and the state file to tmp_path: the
2026-07-13 ledger drift was a test writing its fixture into the real
bankroll, and this gate writes into the real journal on purpose.
"""
from datetime import UTC, datetime, timedelta

import pytest

from scripts.betting import bet_journal as BJ
from scripts.betting import market_promotion as MP
from scripts.betting import picks as P


@pytest.fixture(autouse=True)
def _isolated_journals(tmp_path, monkeypatch):
    monkeypatch.setattr(BJ, "JOURNAL_PATH", tmp_path / "bet_journal.json")
    monkeypatch.setattr(BJ, "_JOURNAL_LOCK_PATH", tmp_path / ".journal.lock")
    monkeypatch.setattr(P, "PICKS_JOURNAL_PATH", tmp_path / "picks_journal.json")
    monkeypatch.setattr(MP, "STATE_PATH", tmp_path / "market_promotion.json")
    monkeypatch.setattr(P, "GOAL_TIMELINE", tmp_path / "missing.parquet")
    monkeypatch.setattr(P, "PMS_PATH", tmp_path / "missing_pms.parquet")
    monkeypatch.setattr(P, "closing_price_for", lambda bet: None)


def _settled(market, n_won, n_lost, odds=2.0, stake=10.0, placed="2026-09-06T17:00:00+00:00", clv=None, status_extra=None):
    out = []
    for i in range(n_won + n_lost):
        won = i < n_won
        out.append({"market": market, "status": "won" if won else "lost", "stake": stake, "odds": odds,
                    "profit": round(stake * (odds - 1), 2) if won else -stake, "placed_at": placed,
                    "clv_pct": clv})
    for st in (status_extra or []):
        out.append({"market": market, "status": st, "stake": stake, "odds": odds, "profit": 0.0, "placed_at": placed})
    return out


# ---- records and the bar ------------------------------------------------------
def test_market_record_excludes_voids_and_measures_z():
    bets = _settled("player_shots", 35, 25, odds=2.0, status_extra=["voided", "void"])
    rec = MP.market_record(bets)["player_shots"]
    assert rec["n"] == 60 and rec["won"] == 35
    assert rec["roi_pct"] == pytest.approx((35 * 10 - 25 * 10) / 600 * 100, abs=0.1)   # +16.7%
    assert rec["z"] > 1.0
    assert rec["mean_clv_pct"] is None and rec["n_clv"] == 0
    # a demoted market's fresh record starts at record_from
    assert "player_shots" not in MP.market_record(bets, since="2026-09-07T00:00:00+00:00")


def test_bar_names_the_first_unmet_condition():
    ok, why = MP.passes_bar(MP.market_record(_settled("m", 20, 10))["m"])
    assert not ok and why == "30/50 settled"
    ok, why = MP.passes_bar(MP.market_record(_settled("m", 25, 25))["m"])          # ROI 0 at evens
    assert not ok and why.startswith("ROI +0.0%")
    ok, why = MP.passes_bar(MP.market_record(_settled("m", 26, 24))["m"])          # +4% on 50, z ~0.3
    assert not ok and why.startswith("z ")
    ok, why = MP.passes_bar(MP.market_record(_settled("m", 40, 20))["m"])
    assert ok and why == "bar cleared"
    # CLV is required only once 20 real closing prices exist, and then must be > 0
    rec = MP.market_record(_settled("m", 40, 20, clv=-1.0))["m"]
    ok, why = MP.passes_bar(rec)
    assert not ok and why.startswith("CLV -1.00% on 60")
    rec = MP.market_record(_settled("m", 40, 20, clv=+0.8))["m"]
    assert MP.passes_bar(rec)[0]


def _real_engine(market, selection, n_won, n_lost, odds, placed, clv=None):
    out = []
    for b in _settled(market, n_won, n_lost, odds=odds, placed=placed, clv=clv):
        out.append(dict(b, selection=selection, extra=None, pipeline_status="current"))
    return out


def test_incumbents_are_scored_against_the_same_bar_on_their_real_record(tmp_path):
    """O/U Over 1.5 / 2.5 bet real money without ever passing the bar. The
    2026-09-06 real journal: 47 bets at ~1.41, 35 won -> z ~0.6; the gate must
    say it fails, keep a since-go-live record apart, and never treat the
    incumbent as a paper market to promote or demote."""
    legacy = "2026-03-01T17:00:00+00:00"
    live = "2026-09-13T17:00:00+00:00"
    real = (_real_engine("O/U 1.5", "Over 1.5", 35, 12, 1.41, legacy, clv=2.4)
            + _real_engine("O/U 1.5", "Over 1.5", 2, 1, 1.35, live)
            + _real_engine("O/U 2.5", "Over 2.5", 17, 27, 2.47, legacy, clv=4.7)
            + _real_engine("O/U 1.5", "Under 1.5", 0, 3, 3.0, legacy)          # other side: not the incumbent
            + [dict(b, extra={"picks_ref": "x"}, pipeline_status=MP.PIPELINE_STATUS)
               for b in _settled("O/U 1.5", 9, 0, odds=1.41, placed=live)])   # promoted mirror: not the engine
    st = MP.evaluate_promotions([], [], real_all=real, now=datetime(2026, 9, 14, tzinfo=UTC))
    inc = st["incumbents"]
    assert set(inc) == {"ou_over_1_5", "ou_over_2_5"} and st["markets"] == {}
    ou15 = inc["ou_over_1_5"]
    assert ou15["status"] == "incumbent" and ou15["real"]["n"] == 50 and ou15["real"]["won"] == 37
    assert 0 < ou15["real"]["z"] < 1.0 and ou15["real"]["mean_clv_pct"] == 2.4 and ou15["real"]["n_clv"] == 47
    assert not ou15["bar_passed"] and ou15["distance"].startswith("z 0.") and "settled" not in ou15["distance"]
    assert ou15["real_since_live"]["n"] == 3 and ou15["record_span"] == ["2026-03-01", "2026-09-13"]
    assert not ou15["would_demote"]
    ou25 = inc["ou_over_2_5"]
    assert ou25["real"]["n"] == 44 and ou25["real"]["roi_pct"] < 0
    assert ou25["distance"].startswith("44/50 settled; ROI -") and "; z -0." in ou25["distance"]   # every miss, not the first
    assert ou25["real_since_live"]["n"] == 0 and not ou25["would_demote"]
    assert not MP.is_promoted("ou_over_1_5", st)
    # the state file carries it, and a re-run is idempotent
    again = MP.evaluate_promotions([], [], real_all=real, now=datetime(2026, 9, 15, tzinfo=UTC))
    assert again["incumbents"] == inc
    # the default path reads the incumbents from the real journal itself
    for b in real[:3]:
        BJ.add_bet(dict(b, match=f"A{b['odds']} vs B", date="2026-03-01", status="pending", bet_id=None))
    assert MP.evaluate_promotions([], now=datetime(2026, 9, 15, tzinfo=UTC))["incumbents"]["ou_over_1_5"]["real"]["n"] == 0
    card = MP.record_card(st, html=False)
    assert "🏦 Over 1.5 (motore) vera n=50 ROI" in card and "barra NON superata: z " in card
    assert "dal go-live n=3" in card and "🏦 = titolare" in card
    assert "Over 2.5 (motore) vera n=44" in card and "44/50 settled; ROI" in card and "z ≥ 2.5" in card
    assert "nessuna scelta carta" in card


def test_evaluate_promotes_then_demotes_on_the_real_record_and_restarts_the_paper_count(tmp_path):
    now = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
    paper = _settled("player_shots_on_target", 40, 20) + _settled("btts_h1", 10, 10)
    st = MP.evaluate_promotions(paper, [], now=now)
    assert st["markets"]["player_shots_on_target"]["status"] == "promoted"
    assert st["markets"]["player_shots_on_target"]["snapshot"]["n"] == 60
    assert st["markets"]["btts_h1"]["status"] == "paper"
    assert st["markets"]["btts_h1"]["distance"] == "20/50 settled"
    assert MP.is_promoted("player_shots_on_target") and not MP.is_promoted("btts_h1")
    assert (tmp_path / "market_promotion.json").exists()
    # 30 real bets at -33% -> back to paper, and the paper count restarts from now
    real = [dict(b, pipeline_status=MP.PIPELINE_STATUS) for b in _settled("player_shots_on_target", 10, 20, stake=15.0)]
    later = now + timedelta(days=30)
    st = MP.evaluate_promotions(paper, real, now=later)
    row = st["markets"]["player_shots_on_target"]
    assert row["status"] == "paper" and row["reason"].startswith("demoted: real ROI -33.3%")
    assert row["record_from"] == later.isoformat()
    assert row["paper"]["n"] == 0 and row["distance"] == "0/50 settled"   # the old 60 no longer count
    assert not MP.is_promoted("player_shots_on_target")
    # idempotent: a re-run with nothing new changes nothing
    again = MP.evaluate_promotions(paper, real, now=later + timedelta(hours=1))
    assert again["markets"]["player_shots_on_target"]["status"] == "paper"
    assert again["markets"]["player_shots_on_target"]["record_from"] == later.isoformat()


# ---- real stake ---------------------------------------------------------------
def test_promoted_stake_is_half_the_ou_kelly_and_capped():
    # p 0.60 @ 2.02: full Kelly 20.8%; x0.15 x0.5 = 1.56% -> capped at 1.5% of 1000
    assert MP.promoted_stake(0.60, 2.02, 1000.0, kelly_fraction=0.15) == 15.0
    # p 0.55 @ 1.90: full Kelly 5%; x0.075 = 0.375% -> EUR 3.75
    assert MP.promoted_stake(0.55, 1.90, 1000.0, kelly_fraction=0.15) == 3.75
    # below the 0.2% floor -> nothing
    assert MP.promoted_stake(0.51, 1.95, 1000.0, kelly_fraction=0.15) == 0.0
    assert MP.promoted_stake(0.60, 2.02, 0.0, kelly_fraction=0.15) == 0.0


def test_promoted_pick_is_mirrored_into_the_real_journal_and_settled_with_it(monkeypatch):
    monkeypatch.setattr("scripts.betting.bankroll_loader.get_effective_bankroll", lambda: 1000.0)
    now = datetime(2026, 10, 25, 17, 0, tzinfo=UTC)
    lean = {"market_key": "player_shots_on_target", "bet_type": "Tiri in porta", "selection": "Over 1.5",
            "player": "Nico Gonzalez", "team": "Juventus", "probability_pct": 60.0, "implied_pct": 49.5,
            "edge_pct": 8.0, "odds": 2.02, "book": "1xBet", "tier": "A", "source": "player_floors"}
    # (the journal refuses edge_pct > 12: a real LEAN above the cap is flagged and sinks anyway)
    pid = P.journal_lean("Juventus vs Milan", "2026-10-25", lean, "serie_a", placed_at=now)
    # not promoted -> no mirror
    assert P._mirror_if_promoted(pid, {"markets": {}}) is None
    assert BJ.get_pending_bets() == []
    state = {"markets": {"player_shots_on_target": {"status": "promoted"}}}
    rid = P._mirror_if_promoted(pid, state)
    (real,) = BJ.get_pending_bets()
    assert real["bet_id"] == rid and real["stake"] == 15.0 and real["pipeline_status"] == MP.PIPELINE_STATUS
    assert real["extra"]["picks_ref"] == pid and real["extra"]["player"] == "Nico Gonzalez"
    assert real["selection"] == "Nico Gonzalez Over 1.5" and real["odds"] == 2.02
    # a second mirror of the same pick is refused by the journal's own dedup
    assert P._mirror_if_promoted(pid, state) in (None, rid)
    assert len(BJ.get_pending_bets()) == 1
    # the pick settles won -> the real entry settles won on ITS stake
    assert MP.settle_linked(pid, "won", result_score="2-1", closing_odds=1.95) == 1
    (settled,) = BJ.get_settled_bets()
    assert settled["status"] == "won" and settled["profit"] == round(15.0 * 1.02, 2)
    assert settled["closing_odds"] == 1.95 and settled["clv_pct"] == round((1 / 1.95 - 1 / 2.02) * 100, 2)
    # the paper CLV is no longer a fake 0.0: no closing price -> no claim
    from scripts.betting.bet_journal import _load_journal
    paper = _load_journal(P.PICKS_JOURNAL_PATH)["bets"][pid]
    assert paper["sharp_implied_prob"] is None


def test_settle_picks_settles_the_linked_real_bet_and_re_evaluates(monkeypatch):
    monkeypatch.setattr("scripts.betting.bankroll_loader.get_effective_bankroll", lambda: 1000.0)
    now = datetime(2026, 10, 25, 17, 0, tzinfo=UTC)
    lean = {"market_key": "h2h", "bet_type": "1x2 finale", "selection": "1", "probability_pct": 60.0,
            "implied_pct": 50.0, "edge_pct": 9.0, "odds": 2.0, "book": "best of market", "tier": "A"}
    pid = P.journal_lean("Juventus vs Milan", "2026-10-25", lean, "serie_a", placed_at=now)
    assert P._mirror_if_promoted(pid, {"markets": {"h2h": {"status": "promoted"}}})
    calls = []
    monkeypatch.setattr(MP, "evaluate_promotions", lambda *a, **k: calls.append(1))
    summary = P.settle_picks({"Juventus vs Milan": {"home_score": 0, "away_score": 1, "status": "finished",
                                                    "commence_time": "2026-10-25T18:45:00Z"}})
    assert summary["settled"] == 1 and summary["real_settled"] == 1 and calls == [1]
    (real,) = BJ.get_settled_bets()
    assert real["status"] == "lost" and real["profit"] == -real["stake"]


def test_full_time_settler_never_grades_a_promoted_pick(monkeypatch):
    """results_fetcher defaults an unknown market to 'lost': a promoted prop
    in the real journal must be invisible to it."""
    from scripts.data import results_fetcher as RF
    prop = {"bet_id": "x", "match": "Juventus vs Milan", "date": "2026-10-25", "market": "player_shots_on_target",
            "selection": "Nico Gonzalez Over 1.5", "odds": 2.02, "stake": 15.0, "status": "pending",
            "extra": {"picks_ref": "p1"}}
    monkeypatch.setattr(BJ, "get_pending_bets", lambda *a, **k: [prop])
    settled = []
    monkeypatch.setattr(BJ, "settle_bet", lambda *a, **k: settled.append(a) or True)
    out = RF._settle_bets_locked({"Juventus vs Milan": {"home_score": 2, "away_score": 1, "status": "finished"}})
    assert settled == [] and out.get("settled", 0) == 0


# ---- card ----------------------------------------------------------------------
def test_record_card_reads_in_italian_promoted_first():
    assert "Nessuna scelta ancora liquidata" in MP.record_card({"markets": {}})
    st = {"markets": {
        "btts_h1": {"status": "paper", "paper": {"n": 12, "roi_pct": 4.0, "mean_clv_pct": None}, "distance": "12/50 settled"},
        "player_shots_on_target": {"status": "promoted", "paper": {"n": 60, "roi_pct": 16.7, "mean_clv_pct": 1.2},
                                   "real": {"n": 4, "roi_pct": 25.0}},
    }}
    card = MP.record_card(st, html=False)
    lines = card.split("\n")
    assert lines[1].startswith("💰 Tiri in porta giocatore carta n=60 ROI +17% · CLV +1.2% · vera n=4 ROI +25%")
    assert lines[2] == "📝 Goal 1° tempo n=12 ROI +4% · 12/50 settled"
    assert "<b>" not in card and "<b>" in MP.record_card(st, html=True)


def test_bot_record_command_renders(monkeypatch):
    import scripts.pipeline.telegram_bot as tb
    assert "Record mercati" in tb._handle_record()


def test_first_half_pick_grades_from_espn_when_the_timeline_lacks_the_match(monkeypatch, tmp_path):
    """goal_timeline.parquet stopped at 2026-08-24 (Sofascore API challenge):
    every first-half pick would stay pending forever. ESPN's post-match key
    events are the fallback, only for finished matches, once per match."""
    monkeypatch.setattr(P, "GOAL_TIMELINE", tmp_path / "no_timeline.parquet")
    now = datetime(2026, 10, 25, 17, 0, tzinfo=UTC)
    lean = {"market_key": "h2h_h1", "bet_type": "1° tempo 1x2", "selection": "1", "probability_pct": 40.0,
            "implied_pct": 34.0, "edge_pct": 6.0, "odds": 2.9, "book": "Pinnacle", "tier": "A"}
    P.journal_lean("Juventus vs Milan", "2026-10-25", lean, "serie_a", placed_at=now)
    P.journal_lean("Roma vs Lazio", "2026-10-25", dict(lean, selection="2"), "serie_a", placed_at=now)
    calls = []
    monkeypatch.setattr(P, "_first_half_from_espn",
                        lambda league, date, home, away: calls.append((league, date, home, away)) or (1, 0))
    res = {"Juventus vs Milan": {"home_score": 1, "away_score": 2, "status": "finished"},
           "Roma vs Lazio": {"home_score": 0, "away_score": 0, "status": "in_progress"}}
    out = P.settle_picks(res)
    assert out["settled"] == 1 and calls == [("serie_a", "2026-10-25", "Juventus", "Milan")]
    from scripts.betting.bet_journal import get_pending_bets, get_settled_bets
    (won,) = get_settled_bets(journal_path=P.PICKS_JOURNAL_PATH)
    assert won["match"] == "Juventus vs Milan" and won["status"] == "won"      # 1-0 at the break, lost at FT
    assert [b["match"] for b in get_pending_bets(journal_path=P.PICKS_JOURNAL_PATH)] == ["Roma vs Lazio"]


def test_null_simulation_measures_the_sequential_look_not_a_single_one():
    """The gate is re-evaluated after every settlement (settle_picks ->
    evaluate_promotions), so a zero-edge market is promoted the FIRST time it
    crosses. The simulation must reproduce that: first-crossing >= single-look,
    a real edge is caught far more often than the null, and a losing market
    on real stakes is demoted more often than a fair one."""
    null = MP.null_simulation(n_markets=14, max_n=200, sims=600, odds=2.0, edge=0.0, seed=1)
    assert null["first_crossing_by_max_n"] >= null["single_look_at_min"]
    assert 0 < null["single_look_at_min"] < 0.5
    assert null["any_of_n_markets_promoted"] == pytest.approx(
        1 - (1 - null["first_crossing_by_max_n"]) ** 14, abs=1e-3)
    edge = MP.null_simulation(n_markets=14, max_n=200, sims=600, odds=2.0, edge=0.15, seed=1)
    assert edge["first_crossing_by_max_n"] > null["first_crossing_by_max_n"] + 0.3
    vig = MP.null_simulation(n_markets=14, max_n=200, sims=600, odds=2.0, edge=-0.05, seed=1)
    assert vig["demoted_within_real_n"] > null["demoted_within_real_n"]
    assert null["bar"] == MP.PROMOTION_BAR


def test_pick_markets_archive_writes_only_this_cycle_gzipped(tmp_path):
    """PICK_MARKETS_FILE is overwritten every refresh; the archive next to the
    bulk snapshots is what lets a prop record be replayed at another timing."""
    import gzip
    import json
    from datetime import UTC, datetime

    from scripts.data import odds_fetcher as OF
    store = {"events": {"a": {"home": "A", "fetched_at": "x", "bookmakers": []},
                        "b": {"home": "B", "fetched_at": "old", "bookmakers": []}}}
    now = datetime(2026, 9, 6, 15, 0, tzinfo=UTC)
    path = OF._archive_pick_markets(store, ["a", "zzz"], now, "serie_a", snapshot_dir=tmp_path)
    assert path == tmp_path / "pick_markets_20260906_150000.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        got = json.load(fh)
    assert list(got["events"]) == ["a"] and got["league"] == "serie_a"
    assert got["markets"] == list(OF.PICK_EVENT_MARKETS) and got["timestamp"].startswith("2026-09-06T15:00")
    assert OF._archive_pick_markets(store, ["b"], now, "serie_a", snapshot_dir=tmp_path) is not None
    assert OF._archive_pick_markets(store, [], now, "serie_a", snapshot_dir=tmp_path) is None
