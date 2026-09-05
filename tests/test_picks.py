"""Pick engine (scripts/betting/picks.py): every match gets a line, real money
only on VALUE, LEAN is paper, NO EDGE says why.

The per-event specimen below is trimmed from the real Juventus vs AC Milan
response (region eu, 2026-09-05): outcome NAMING is the thing a parser gets
wrong (team names in h2h_h1, 'Juventus/Draw' in HT/FT, 'Juventus:1|AC Milan:0'
in correct score, full player names in `description`).
"""
from datetime import UTC, datetime, timedelta

import pytest

from scripts.betting import picks as P

HOME_RAW, AWAY_RAW = "Juventus", "AC Milan"
EVENT = {
    "home": "Juventus", "away": "Milan", "home_raw": HOME_RAW, "away_raw": AWAY_RAW,
    "commence": "2026-09-06T18:45:00Z", "fetched_at": "2026-09-05T20:00:00+00:00",
    "bookmakers": [
        {"title": "Pinnacle", "markets": [
            {"key": "h2h_h1", "outcomes": [{"name": "AC Milan", "price": 4.51}, {"name": "Juventus", "price": 2.83},
                                           {"name": "Draw", "price": 2.15}]},
            {"key": "totals_h1", "outcomes": [{"name": "Over", "point": 1.0, "price": 2.13},
                                              {"name": "Under", "point": 1.0, "price": 1.74}]},
            {"key": "halftime_fulltime", "outcomes": [{"name": "AC Milan/Juventus", "price": 22.63},
                                                     {"name": "Juventus/Juventus", "price": 3.69},
                                                     {"name": "Draw/Draw", "price": 5.26}]},
            {"key": "correct_score", "outcomes": [{"name": "Juventus:1|AC Milan:0", "price": 7.68},
                                                  {"name": "Juventus:0|AC Milan:2", "price": 19.01}]},
        ]},
        {"title": "GTbets", "markets": [
            {"key": "h2h_h1", "outcomes": [{"name": "AC Milan", "price": 4.25}, {"name": "Juventus", "price": 2.79},
                                           {"name": "Draw", "price": 2.17}]},
            {"key": "totals_h1", "outcomes": [{"name": "Over", "point": 0.5, "price": 1.46},
                                              {"name": "Under", "point": 0.5, "price": 2.64}]},
        ]},
        {"title": "1xBet", "markets": [
            {"key": "player_shots", "outcomes": [
                {"name": "Over", "description": "Goncalo Matias Ramos", "point": 0.5, "price": 1.08},
                {"name": "Over", "description": "Francisco Conceicao", "point": 1.5, "price": 1.36}]},
            {"key": "player_goal_scorer_anytime", "outcomes": [
                {"name": "Yes", "description": "Goncalo Matias Ramos", "price": 3.3}]},
        ]},
        {"title": "William Hill", "markets": [
            {"key": "player_goal_scorer_anytime", "outcomes": [
                {"name": "Yes", "description": "Goncalo Matias Ramos", "price": 3.5}]},
            {"key": "player_assists", "outcomes": [
                {"name": "Over", "description": "Francisco Conceicao", "point": 0.5, "price": 3.75}]},
        ]},
    ],
}
ODDS = {"h2h": {"best_home": 2.18, "best_draw": 3.4, "best_away": 3.9, "home": 2.1, "bookmakers_count": 24},
        "totals": [{"line": 2.5, "best_over": 1.95, "best_under": 1.9, "over": 1.9, "under": 1.85,
                    "bookmakers_count": 13}],
        "commence_time": "2026-09-06T18:45:00Z"}
EXTRA = {"btts": {"best_yes": 1.85, "best_no": 2.02, "bookmakers_count": 1},
         "double_chance": {"1X": {"best": 1.3, "avg": 1.3, "bookmakers_count": 2}},
         "alternate_totals": {"4.5": {"best_over": 6.4, "best_under": 1.12, "bookmakers_count": 1}}}


def _row(bt, sel, p, tier="B", **kw):
    return {"group": "g", "bet_type": bt, "selection": sel, "probability_pct": p, "tier": tier,
            "source": "s", **kw}


# ---- price book -------------------------------------------------------------
def test_price_book_maps_every_specimen_naming_convention():
    book = P.build_price_book(ODDS, EXTRA, EVENT)
    assert book[("h2h", "home", None, None)]["odds"] == 2.18
    assert book[("totals", "over", 2.5, None)]["n_books"] == 13
    assert book[("totals", "over", 4.5, None)]["odds"] == 6.4          # alt totals fill the gap
    assert book[("btts", "yes", None, None)]["odds"] == 1.85
    assert book[("double_chance", "1X", None, None)]["odds"] == 1.3
    # team names -> sides; best across books; Draw kept
    assert book[("h2h_h1", "away", None, None)] == {"odds": 4.51, "book": "Pinnacle", "avg": 4.38, "n_books": 2}
    assert book[("h2h_h1", "draw", None, None)]["odds"] == 2.17
    assert book[("totals_h1", "over", 0.5, None)]["odds"] == 1.46       # .5 and 1.0 lines are distinct keys
    assert book[("totals_h1", "over", 1.0, None)]["odds"] == 2.13
    assert book[("halftime_fulltime", "A/H", None, None)]["odds"] == 22.63
    assert book[("halftime_fulltime", "H/H", None, None)]["odds"] == 3.69
    assert book[("correct_score", "1-0", None, None)]["odds"] == 7.68
    assert book[("correct_score", "0-2", None, None)]["odds"] == 19.01
    assert book[("player_goal_scorer_anytime", "yes", None, "goncalo matias ramos")] == \
        {"odds": 3.5, "book": "William Hill", "avg": 3.4, "n_books": 2}
    assert book[("player_shots", "over", 1.5, "francisco conceicao")]["odds"] == 1.36
    assert book[("player_assists", "over", 0.5, "francisco conceicao")]["odds"] == 3.75


def test_price_book_without_any_artifact_is_empty():
    assert P.build_price_book(None, None, None) == {}


# ---- row -> price key ---------------------------------------------------------
@pytest.mark.parametrize("row,key", [
    (_row("1x2 finale", "2", 30), ("h2h", "away", None, None)),
    (_row("1° tempo 1x2", "X", 30), ("h2h_h1", "draw", None, None)),
    (_row("Under/over", "Under 2.5", 50), ("totals", "under", 2.5, None)),
    (_row("1° tempo under/over", "Over 0.5", 70), ("totals_h1", "over", 0.5, None)),
    (_row("Doppia chance", "X2", 60), ("double_chance", "X2", None, None)),
    (_row("Goal", "Sì", 55), ("btts", "yes", None, None)),
    (_row("Risultato esatto", "2-1", 9), ("correct_score", "2-1", None, None)),
    (_row("Primo tempo / Finale", "D/H", 12), ("halftime_fulltime", "D/H", None, None)),
    (_row("Giocatore marcatore", "Sì", 20, player="Gonçalo Ramos"), ("player_goal_scorer_anytime", "yes", None, "gonçalo ramos".replace("ç", "c"))),
    (_row("Assist giocatore", "Sì", 15, player="Luka Modrić"), ("player_assists", "over", 0.5, "luka modric")),
    (_row("Tiri in porta", "Over 1.5", 30, player="X Y"), ("player_shots_on_target", "over", 1.5, "x y")),
    (_row("Vince o quasi", "Casa 1x sì", 60), None),      # no market on this feed
    (_row("Gol nei primi 15 minuti", "Sì", 20), None),
    (_row("Cartellini Casa", "Over 1.5", 60), None),
])
def test_price_key_for_row(row, key):
    assert P.price_key_for_row(row) == key


# ---- player name join ---------------------------------------------------------
def test_player_matching_accent_subset_and_ambiguity():
    feed = ["Goncalo Matias Ramos", "Francisco Conceicao", "Luka Modric", "Yunus Musah", "Kenan Yildiz"]
    assert P._match_player("Gonçalo Ramos", feed) == "Goncalo Matias Ramos"   # subset of feed tokens
    assert P._match_player("Francisco Conceição", feed) == "Francisco Conceicao"  # accent-folded exact
    assert P._match_player("L. Modric", feed) == "Luka Modric"                # surname + initial
    assert P._match_player("Nobody Here", feed) is None
    assert P._match_player("Kenan Yildiz", feed + ["Kaan Yildiz"]) == "Kenan Yildiz"  # exact beats loose
    assert P._match_player("K. Yildiz", feed + ["Kaan Yildiz"]) is None       # ambiguous -> no price


def test_annotate_rows_prices_only_rows_with_a_real_price():
    book = P.build_price_book(ODDS, EXTRA, EVENT)
    rows = [_row("1x2 finale", "1", 50.0, "A"), _row("Vince o quasi", "Casa 1x sì", 60, "A"),
            _row("Giocatore marcatore", "Sì", 40, "C", player="Gonçalo Ramos"),
            _row("Giocatore marcatore", "Sì", 40, "C", player="Nobody Priced")]
    assert P.annotate_rows(rows, book) == 2
    assert rows[0]["odds"] == 2.18 and rows[0]["edge_pct"] == 9.0 and rows[0]["implied_pct"] == 45.9
    assert "odds" not in rows[1] and "odds" not in rows[3]
    assert rows[2]["odds"] == 3.5 and rows[2]["edge_pct"] == 40.0 and rows[2]["n_books"] == 2


# ---- ranking and labels --------------------------------------------------------
def _cand(edge, tier="B", p=50.0, n_books=5, **kw):
    return {"bet_type": "Under/over", "selection": "Over 2.5", "edge_pct": edge, "tier": tier,
            "probability_pct": p, "n_books": n_books, "odds": 2.0, "book": "b", "implied_pct": 50.0, **kw}


def test_rank_measured_before_big_and_longshots_sink():
    ranked = P.rank_candidates([
        _cand(11.6, "A", p=3.1, n_books=1, selection="Pavlovic SoT Over 1.5"),  # long shot: sinks
        _cand(9.2, "B"), _cand(3.4, "A"), _cand(3.3, "A", n_books=1), _cand(13.0, "A"),
    ], band=(3.0, 7.0))
    order = [(c["edge_pct"], c["tier"]) for c in ranked]
    # credible first: in-band A (multi-book before single-book), then in-band B by edge
    assert order[:3] == [(3.4, "A"), (3.3, "A"), (9.2, "B")]
    assert ranked[1]["thin"] and not ranked[0]["thin"]
    # the two flagged edges sink regardless of size or tier
    assert {c["edge_pct"] for c in ranked[3:]} == {11.6, 13.0}
    assert all(not c["in_band"] for c in ranked[3:])
    assert next(c for c in ranked if c["edge_pct"] == 11.6)["longshot"]
    assert next(c for c in ranked if c["edge_pct"] == 13.0)["overconfident"]


def test_value_comes_from_the_slip_never_recomputed():
    slip = {"selected_bets": [{"match": "A vs B", "market": "O/U 1.5", "selection": "Over 1.5",
                               "best_odds": 1.41, "best_bookmaker": "bk", "edge_pct": 6.52,
                               "model_prob": 0.755, "stake_amount": 16.95}]}
    book = {("totals", "over", 2.5, None): {"odds": 2.0, "book": "b", "avg": 2.0, "n_books": 5}}
    line = P.build_match_pick("A vs B", [_row("Under/over", "Over 2.5", 55.0)], book,
                              slip=slip, candidates={}, band=(3.0, 7.0))
    assert line["label"] == "VALUE" and line["stage"] == "selected"
    assert line["pick"]["odds"] == 1.41 and line["pick"]["edge_pct"] == 6.52 and line["pick"]["stake"] == 16.95
    assert line["lean"]["edge_pct"] == 10.0   # the paper record still rides alongside


def test_candidate_is_value_pending_t30():
    cands = {"candidates": [{"match": "A vs B", "market": "O/U 1.5", "selection": "Over 1.5",
                             "best_odds": 1.5, "edge_pct": 5.0, "model_prob": 0.7}]}
    line = P.build_match_pick("A vs B", [], {}, slip={}, candidates=cands, band=(3.0, 7.0))
    assert line["label"] == "VALUE" and line["stage"] == "candidate" and "T-30" in line["reason"]


def test_lean_and_no_edge_reasons():
    book = {("h2h", "home", None, None): {"odds": 2.0, "book": "b", "avg": 2.0, "n_books": 10},
            ("h2h", "away", None, None): {"odds": 4.0, "book": "b", "avg": 4.0, "n_books": 10}}
    lean = P.build_match_pick("A vs B", [_row("1x2 finale", "1", 52.0, "A"), _row("1x2 finale", "2", 24.0, "A")],
                              book, slip={}, candidates={}, band=(3.0, 7.0))
    assert lean["label"] == "LEAN" and lean["pick"]["selection"] == "1" and lean["pick"]["edge_pct"] == 4.0
    assert "paper" in lean["reason"] and lean["pick"]["in_band"]
    none = P.build_match_pick("A vs B", [_row("1x2 finale", "1", 45.0, "A"), _row("1x2 finale", "2", 20.0, "A")],
                              book, slip={}, candidates={}, band=(3.0, 7.0))
    assert none["label"] == "NO_EDGE" and none["pick"] is None
    assert none["most_probable"]["selection"] == "1" and "no edge" in none["reason"]
    over = P.build_match_pick("A vs B", [_row("1x2 finale", "1", 70.0, "A")], book,
                              slip={}, candidates={}, band=(3.0, 7.0))
    assert over["label"] == "NO_EDGE" and "overconfidence" in over["reason"] and over["n_overconfident"] == 1
    empty = P.build_match_pick("A vs B", [_row("Vince o quasi", "Casa 1x sì", 60.0)], {},
                               slip={}, candidates={}, band=(3.0, 7.0))
    assert empty["label"] == "NO_EDGE" and "no market price" in empty["reason"]


def test_lean_carries_the_engines_rejection_of_the_same_bet():
    slip = {"near_misses": [{"match": "A vs B", "market": "O/U 2.5", "selection": "Over 2.5",
                             "reason": "above_max_edge", "edge_pct": 10.4, "min_edge": 7.0, "max_edge": 10.0}]}
    book = {("totals", "over", 2.5, None): {"odds": 1.78, "book": "b", "avg": 1.7, "n_books": 13}}
    line = P.build_match_pick("A vs B", [_row("Under/over", "Over 2.5", 62.0)], book,
                              slip=slip, candidates={}, band=(3.0, 7.0))
    assert line["label"] == "LEAN"
    assert line["pick"]["engine_note"] == "engine rejected it: above_max_edge at +10.4% (band 7.0-10.0%)"
    assert "engine rejected it" in line["reason"]


# ---- paper journal -------------------------------------------------------------
def test_journal_lean_keeps_two_players_apart_and_carries_extra(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PICKS_JOURNAL_PATH", tmp_path / "picks_journal.json")
    now = datetime(2026, 9, 6, 18, 0, tzinfo=UTC)
    lean_a = {"market_key": "player_shots", "bet_type": "Tiri totali del giocatore", "selection": "Over 0.5",
              "player": "Gonçalo Ramos", "team": "Milan", "probability_pct": 86.2, "implied_pct": 84.7,
              "edge_pct": 1.7, "odds": 1.18, "book": "1xBet", "tier": "A", "source": "player_floors"}
    lean_b = dict(lean_a, player="Francisco Conceição")
    ida = P.journal_lean("Juventus vs Milan", "2026-09-06", lean_a, "serie_a", placed_at=now)
    idb = P.journal_lean("Juventus vs Milan", "2026-09-06", lean_b, "serie_a", placed_at=now)
    assert ida and idb and ida != idb
    from scripts.betting.bet_journal import get_pending_bets
    pend = get_pending_bets(journal_path=P.PICKS_JOURNAL_PATH)
    assert len(pend) == 2
    b = next(x for x in pend if x["bet_id"] == ida)
    assert b["market"] == "player_shots" and b["selection"] == "Gonçalo Ramos Over 0.5"
    assert b["stake"] == 10.0 and b["pipeline_status"] == "pick:lean"
    assert b["extra"] == {"bet_type": "Tiri totali del giocatore", "player": "Gonçalo Ramos", "team": "Milan",
                          "source": "player_floors", "tier": "A", "side": "over", "line": 0.5}


# ---- grading -------------------------------------------------------------------
def _bet(market, selection, **extra):
    return {"market": market, "selection": selection, "extra": extra}


def test_grade_every_market_family():
    ft = {"home_score": 2, "away_score": 1}
    assert P._grade(_bet("h2h", "1"), ft, None, None) == "won"
    assert P._grade(_bet("h2h", "X"), ft, None, None) == "lost"
    assert P._grade(_bet("totals", "Over 2.5", side="over", line=2.5), ft, None, None) == "won"
    assert P._grade(_bet("totals", "Over 3.0", side="over", line=3.0), ft, None, None) == "push"
    assert P._grade(_bet("double_chance", "X2"), ft, None, None) == "lost"
    assert P._grade(_bet("btts", "Sì"), ft, None, None) == "won"
    assert P._grade(_bet("correct_score", "2-1"), ft, None, None) == "won"
    assert P._grade(_bet("h2h_h1", "X"), ft, (0, 0), None) == "won"
    assert P._grade(_bet("h2h_h1", "X"), ft, None, None) is None            # first half unknown yet
    assert P._grade(_bet("totals_h1", "Under 0.5", side="under", line=0.5), ft, (0, 0), None) == "won"
    assert P._grade(_bet("halftime_fulltime", "D/H"), ft, (0, 0), None) == "won"
    assert P._grade(_bet("halftime_fulltime", "D/H"), None, (0, 0), None) is None
    row = {"total_shots": 2, "shots_on_target": 0, "goals": 1, "assists": 0}
    assert P._grade(_bet("player_shots", "X Over 1.5", side="over", line=1.5, player="X"), None, None, row) == "won"
    assert P._grade(_bet("player_shots_on_target", "X Over 0.5", side="over", line=0.5, player="X"), None, None, row) == "lost"
    assert P._grade(_bet("player_goal_scorer_anytime", "X Sì", player="X"), None, None, row) == "won"
    assert P._grade(_bet("player_assists", "X Sì", side="over", line=0.5, player="X"), None, None, row) == "lost"
    assert P._grade(_bet("player_assists", "X Sì", side="over", line=0.5, player="X"), None, None, None) is None
    assert P._grade(_bet("player_to_receive_card", "X Yes"), ft, None, row) is None  # never journaled, never graded


def test_settle_picks_settles_gradable_and_leaves_the_rest_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PICKS_JOURNAL_PATH", tmp_path / "picks_journal.json")
    monkeypatch.setattr(P, "GOAL_TIMELINE", tmp_path / "missing.parquet")
    monkeypatch.setattr(P, "PMS_PATH", tmp_path / "missing_pms.parquet")
    now = datetime(2026, 9, 6, 18, 0, tzinfo=UTC)
    ft_lean = {"market_key": "h2h", "bet_type": "1x2 finale", "selection": "1", "probability_pct": 52.0,
               "implied_pct": 50.0, "edge_pct": 4.0, "odds": 2.0, "book": "best of market", "tier": "A"}
    h1_lean = {"market_key": "h2h_h1", "bet_type": "1° tempo 1x2", "selection": "X", "probability_pct": 40.0,
               "implied_pct": 46.0, "edge_pct": 2.0, "odds": 2.17, "book": "GTbets", "tier": "B"}
    P.journal_lean("Juventus vs Milan", "2026-09-06", ft_lean, "serie_a", placed_at=now)
    P.journal_lean("Juventus vs Milan", "2026-09-06", h1_lean, "serie_a", placed_at=now)
    summary = P.settle_picks({"Juventus vs Milan": {"home_score": 2, "away_score": 1, "status": "finished",
                                                    "commence_time": "2026-09-06T18:45:00Z"}})
    assert summary["settled"] == 1 and summary["won"] == 1 and summary["ungradable"] == 1 and summary["pending"] == 1
    from scripts.betting.bet_journal import get_settled_bets
    (s,) = get_settled_bets(journal_path=P.PICKS_JOURNAL_PATH)
    assert s["market"] == "h2h" and s["profit"] == 10.0 and s["result_score"] == "2-1"


# ---- fetch gating ----------------------------------------------------------------
def test_pick_markets_due_selection_honours_its_own_refresh_window():
    from scripts.data.odds_fetcher import PICK_EVENT_MARKETS, _scorer_events_due
    assert "alternate_team_totals" not in PICK_EVENT_MARKETS and "player_assists" in PICK_EVENT_MARKETS
    now = datetime(2026, 9, 6, 15, 0, tzinfo=UTC)
    evs = [{"id": "a", "commence_time": "2026-09-06T18:45:00Z"},   # 3.75h out: in a 6.5h window
           {"id": "b", "commence_time": "2026-09-07T16:30:00Z"},   # tomorrow: out
           {"id": "c", "commence_time": "2026-09-06T16:00:00Z"}]   # fetched 20 min ago: skipped at 45, due at 15
    store = {"events": {"c": {"fetched_at": (now - timedelta(minutes=20)).isoformat()}}}
    assert [e["id"] for e in _scorer_events_due(evs, store, now, 6.5, 45.0)] == ["a"]
    assert [e["id"] for e in _scorer_events_due(evs, store, now, 6.5, 15.0)] == ["a", "c"]


def test_attach_prices_never_mutates_shared_rows(monkeypatch):
    """Simulator rows are served from goal_process._SERVED_CACHE; a price written
    into them would leak into the next request (and stay when odds vanish)."""
    monkeypatch.setattr(P, "match_price_book", lambda m, league="serie_a": {
        ("h2h", "home", None, None): {"odds": 2.0, "book": "b", "avg": 2.0, "n_books": 10}})
    monkeypatch.setattr(P, "_read", lambda path, default: default)
    shared = [_row("1x2 finale", "1", 55.0, "A")]
    payload = {"markets": shared, "players": []}
    P.attach_prices("A vs B", payload)
    assert payload["markets"][0]["odds"] == 2.0 and payload["n_priced"] == 1
    assert "odds" not in shared[0]
    assert payload["pick"] is None


def test_paper_lean_beside_a_real_bet_is_a_different_bet():
    slip = {"selected_bets": [{"match": "A vs B", "market": "O/U 2.5", "selection": "Over 2.5",
                               "best_odds": 2.0, "edge_pct": 6.0, "model_prob": 0.55, "stake_amount": 10}]}
    book = {("totals", "over", 2.5, None): {"odds": 2.0, "book": "b", "avg": 2.0, "n_books": 5},
            ("h2h", "home", None, None): {"odds": 2.2, "book": "b", "avg": 2.2, "n_books": 5}}
    rows = [_row("Under/over", "Over 2.5", 55.0), _row("1x2 finale", "1", 48.0, "A")]
    line = P.build_match_pick("A vs B", rows, book, slip=slip, candidates={}, band=(3.0, 7.0))
    assert line["label"] == "VALUE" and line["lean"]["selection"] == "1"
    only = P.build_match_pick("A vs B", rows[:1], book, slip=slip, candidates={}, band=(3.0, 7.0))
    assert only["lean"] is None


def test_engine_note_without_numeric_edge_does_not_crash():
    slip = {"near_misses": [{"match": "A vs B", "market": "O/U 2.5", "selection": "Over 2.5", "reason": "x"}]}
    book = {("totals", "over", 2.5, None): {"odds": 1.78, "book": "b", "avg": 1.7, "n_books": 13}}
    line = P.build_match_pick("A vs B", [_row("Under/over", "Over 2.5", 62.0)], book,
                              slip=slip, candidates={}, band=(3.0, 7.0))
    assert line["pick"]["engine_note"] == "engine rejected it: x"


def test_exotic_slot_lists_positive_edges_outside_main_markets_else_the_most_probable_prop():
    book = {("h2h", "home", None, None): {"odds": 2.0, "book": "b", "avg": 2.0, "n_books": 10},
            ("player_assists", "over", 0.5, "federico dimarco"): {"odds": 4.5, "book": "William Hill", "avg": 4.5, "n_books": 1},
            ("player_shots", "over", 0.5, "federico dimarco"): {"odds": 1.2, "book": "1xBet", "avg": 1.2, "n_books": 1},
            ("h2h_h1", "home", None, None): {"odds": 2.9, "book": "Pinnacle", "avg": 2.8, "n_books": 3}}
    rows = [_row("1x2 finale", "1", 53.0, "A", group="Principali"),
            {"group": "Giocatori", "bet_type": "Assist giocatore", "selection": "Sì", "player": "Federico Dimarco",
             "probability_pct": 23.0, "tier": "C", "source": "player_floors"},
            {"group": "Giocatori", "bet_type": "Tiri totali del giocatore", "selection": "Over 0.5",
             "player": "Federico Dimarco", "probability_pct": 80.0, "tier": "A", "source": "player_floors"},
            _row("1° tempo 1x2", "1", 36.0, "A", group="Tempi")]
    line = P.build_match_pick("A vs B", rows, book, slip={}, candidates={}, band=(3.0, 7.0))
    assert line["pick"]["selection"] == "1" and line["pick"]["bet_type"] == "1x2 finale"   # multi-book A wins the headline
    ex = line["exotic"]
    assert [(e["bet_type"], e["edge_pct"]) for e in ex] == [("1° tempo 1x2", 4.4), ("Assist giocatore", 3.5)]
    assert line["n_exotic_positive"] == 2 and "exotic_fallback" not in line
    # without a positive exotic edge, the most probable priced player prop is shown with its price
    rows2 = [rows[0], rows[2]]
    line2 = P.build_match_pick("A vs B", rows2, book, slip={}, candidates={}, band=(3.0, 7.0))
    assert line2["exotic"] == [] and line2["exotic_fallback"]["player"] == "Federico Dimarco"
    # BTTS is a mainstream market, never "insolita"
    goal = [_row("Goal", "No", 53.0, "B", group="Goal")]
    line3 = P.build_match_pick("A vs B", goal, {("btts", "no", None, None): {"odds": 2.0, "book": "b", "avg": 2.0, "n_books": 4}},
                               slip={}, candidates={}, band=(3.0, 7.0))
    assert line3["pick"]["bet_type"] == "Goal" and line3["exotic"] == [] and "exotic_fallback" not in line3
    assert line2["exotic_fallback"]["edge_pct"] == -4.0


def test_slate_prices_player_rows_too(monkeypatch, tmp_path):
    """build_picks must feed player rows to the pick, not just match rows —
    the first live slate silently priced zero player props."""
    import web.match_markets as mm
    monkeypatch.setattr(P, "PICKS_FILE", tmp_path / "picks.json")
    monkeypatch.setattr(P, "_read", lambda path, default: (
        {"predictions": [{"match": "A vs B", "date": "2099-01-01", "league": "serie_a",
                          "home_team": "A", "away_team": "B"}]}
        if getattr(path, "name", "") == "predictions.json" else default))
    monkeypatch.setattr(mm, "assemble_market_inputs", lambda *a, **k: {"pred": {}, "goal_pred": None, "ext": None,
                                                                        "btts": None, "players": None, "sim": None,
                                                                        "halves_gate": None, "kickoff_utc": None,
                                                                        "league": "serie_a"})
    monkeypatch.setattr(mm, "build_match_markets", lambda *a, **k: {
        "markets": [], "players": [{"group": "Giocatori", "bet_type": "Assist giocatore", "selection": "Sì",
                                    "player": "Federico Dimarco", "probability_pct": 23.0, "tier": "C", "source": "s"}]})
    monkeypatch.setattr(P, "build_price_book", lambda *a: {
        ("player_assists", "over", 0.5, "federico dimarco"): {"odds": 4.5, "book": "WH", "avg": 4.5, "n_books": 1}})
    out = P.build_picks("serie_a", journal=False)
    (line,) = out["picks"]
    assert line["label"] == "LEAN" and line["pick"]["player"] == "Federico Dimarco" and line["pick"]["edge_pct"] == 3.5



def test_slate_drops_a_match_that_has_kicked_off(monkeypatch, tmp_path):
    """The /picks card listed Roma–Atalanta 40 minutes into the match
    (2026-09-05): the slate filtered by DATE, not kickoff."""
    from datetime import UTC, datetime

    import web.match_markets as mm
    now = datetime(2026, 9, 5, 19, 30, tzinfo=UTC)
    monkeypatch.setattr(P, "PICKS_FILE", tmp_path / "picks.json")
    monkeypatch.setattr(P, "_read", lambda path, default: {
        "predictions.json": {"predictions": [
            {"match": "Roma vs Atalanta", "date": "2026-09-05", "league": "serie_a", "home_team": "Roma", "away_team": "Atalanta"},
            {"match": "Bologna vs Sassuolo", "date": "2026-09-06", "league": "serie_a", "home_team": "Bologna", "away_team": "Sassuolo"}]},
        "odds_full.json": {"matches": {"Roma vs Atalanta": {"commence_time": "2026-09-05T18:45:00Z"},
                                       "Bologna vs Sassuolo": {"commence_time": "2026-09-06T16:00:00Z"}}},
    }.get(getattr(path, "name", ""), default))
    monkeypatch.setattr(mm, "assemble_market_inputs", lambda *a, **k: {"pred": {}, "goal_pred": None, "ext": None,
                                                                        "btts": None, "players": None, "sim": None,
                                                                        "halves_gate": None, "kickoff_utc": None,
                                                                        "league": "serie_a"})
    monkeypatch.setattr(mm, "build_match_markets", lambda *a, **k: {"markets": [], "players": []})
    monkeypatch.setattr(P, "build_price_book", lambda *a: {})
    out = P.build_picks("serie_a", journal=False, now=now)
    assert [p["match"] for p in out["picks"]] == ["Bologna vs Sassuolo"]


def test_player_rows_come_from_the_official_sheet_when_it_exists(monkeypatch, tmp_path):
    """confirmed_lineups.json wins over the predicted XI; a player missing from
    the team sheet gets no row; every entry says which basis it has."""
    import json

    import pandas as pd

    import scripts.betting.player_predictions as pp
    pms = pd.DataFrame({"player_name": ["Mario Pašalić", "Éderson", "Paulo Dybala"], "player_id": [1, 2, 3],
                        "position": ["M", "M", "F"], "date": pd.to_datetime(["2026-08-31"] * 3)})
    monkeypatch.setattr(pp, "player_engine", lambda: (pms, {}))
    conf = tmp_path / "confirmed.json"
    pred = tmp_path / "predicted.json"
    monkeypatch.setattr(pp, "_CONFIRMED_LINEUPS", conf)
    monkeypatch.setattr(pp, "_LINEUP_PREDICTIONS", pred)
    pred.write_text(json.dumps({"matches": {"Roma vs Atalanta": {
        "home_lineup": {"predicted_xi": [{"name": "Paulo Dybala", "position": "F", "status": "certain", "start_pct": 91.1}]},
        "away_lineup": {"predicted_xi": [{"name": "Mario Pašalić", "position": "M", "status": "likely", "start_pct": 71.7}]}}}}))
    captured = {}

    def fake_predict(home, away, home_xi, away_xi, **kw):
        captured["home"], captured["away"] = home_xi, away_xi
        return {"home_players": [], "away_players": []}
    monkeypatch.setattr(pp, "predict_match_players", fake_predict)

    pp.match_player_floors("Roma vs Atalanta", "Roma", "Atalanta")
    assert [(e["player_name"], e["lineup"], e["xi_status"], e["start_pct"]) for e in captured["away"]] == [
        ("Mario Pašalić", "predicted", "likely", 71.7)]

    xi = ["Ederson"] + [f"Atalanta {i}" for i in range(10)]        # ESPN spelling, no diacritics
    conf.write_text(json.dumps({"matches": {"Roma vs Atalanta": {
        "home_lineup": ["Paulo Dybala"] + [f"Roma {i}" for i in range(10)], "home_bench": [],
        "away_lineup": xi, "away_bench": ["Mario Pasalic"]}}}))
    pp.match_player_floors("Roma vs Atalanta", "Roma", "Atalanta")
    away = {e["player_name"]: e for e in captured["away"]}
    # names come back in the pms spelling, resolved by accent-folded match
    assert away["Éderson"]["lineup"] == "confirmed" and away["Éderson"]["is_starter"] and away["Éderson"]["player_id"] == 2
    assert away["Mario Pašalić"]["is_starter"] is False and away["Mario Pašalić"]["start_pct"] == 0.0
    assert away["Atalanta 3"]["player_id"] is None and away["Atalanta 3"]["position"] == "M"

    # a partial sheet (fewer than MIN_CONFIRMED_XI names) is not an XI: fall back
    conf.write_text(json.dumps({"matches": {"Roma vs Atalanta": {"home_lineup": ["Paulo Dybala"], "away_lineup": ["Éderson"]}}}))
    pp.match_player_floors("Roma vs Atalanta", "Roma", "Atalanta")
    assert captured["away"][0]["lineup"] == "predicted"


def test_uncertain_starter_is_priced_as_a_start_sub_mixture(monkeypatch):
    """A predicted-XI starter at 72% is priced 0.72 × P(starts) + 0.28 × P(sub,
    20'), not at full minutes; a certain starter, a confirmed starter and a
    bench entry are untouched."""
    import pandas as pd

    import scripts.betting.player_predictions as pp
    calls = []

    def fake_predict(*, player_name, is_starter=True, proj_minutes=None, **kw):
        calls.append((player_name, is_starter, proj_minutes))
        p = 0.6 if is_starter else 0.2
        one = {"label": "x", "prob": p, "odds_implied": round(1 / p, 2), "source": "player",
               "calibrated": False, "expected": 2.0 if is_starter else 0.5,
               "split": {"1h": p / 2, "2h": p / 2, "both": None, "exp_1h": 1.0, "exp_2h": 1.0, "timing": "measured"}}
        return {"player_name": player_name, "proj_minutes": 80.0 if is_starter else proj_minutes,
                "markets": {k: dict(one, split=dict(one["split"])) for k in pp.TARGETS}}
    monkeypatch.setattr(pp, "predict_player_markets", fake_predict)
    monkeypatch.setattr(pp, "_get_floor_calibration", lambda pms: None)
    monkeypatch.setattr(pp, "_get_dispersion", lambda pms: None)
    monkeypatch.setattr(pp, "_get_possession", lambda pms, league: (None, None))
    pms = pd.DataFrame({"player_name": [], "player_id": [], "position": [], "date": []})
    away = [
        {"player_name": "Mario Pašalić", "player_id": 1, "position": "M", "is_starter": True, "proj_minutes": 75.0,
         "lineup": "predicted", "xi_status": "likely", "start_pct": 72.0},
        {"player_name": "Éderson", "player_id": 2, "position": "M", "is_starter": True, "proj_minutes": 80.0,
         "lineup": "predicted", "xi_status": "certain", "start_pct": 100.0},
        {"player_name": "Ademola Lookman", "player_id": 3, "position": "F", "is_starter": True, "proj_minutes": None,
         "lineup": "confirmed", "xi_status": "confirmed", "start_pct": 100.0},
        {"player_name": "Nicolò Zaniolo", "player_id": 4, "position": "F", "is_starter": False, "proj_minutes": 12.0,
         "lineup": "predicted", "xi_status": None, "start_pct": 30.0},
    ]
    out = pp.predict_match_players("Roma", "Atalanta", [], away, pms=pms, base_rates={})
    by = {p["player_name"]: p for p in out["away_players"]}
    m = by["Mario Pašalić"]["markets"]["shots_o15"]
    assert m["prob"] == round(0.72 * 0.6 + 0.28 * 0.2, 4) == 0.488
    assert m["expected"] == round(0.72 * 2.0 + 0.28 * 0.5, 4)
    assert m["split"]["1h"] == round(0.72 * 0.3 + 0.28 * 0.1, 4) and m["split"]["timing"] == "measured"
    assert by["Mario Pašalić"]["start_mix"] == 0.72 and by["Mario Pašalić"]["proj_minutes"] == round(0.72 * 80 + 0.28 * 20, 1)
    assert by["Mario Pašalić"]["lineup"] == "predicted" and by["Mario Pašalić"]["start_pct"] == 72.0
    for name in ("Éderson", "Ademola Lookman", "Nicolò Zaniolo"):
        assert "start_mix" not in by[name]
    assert by["Éderson"]["markets"]["shots_o15"]["prob"] == 0.6 and by["Nicolò Zaniolo"]["markets"]["shots_o15"]["prob"] == 0.2
    # exactly one extra (sub-branch) call, for the uncertain starter only
    assert [c for c in calls if c[1] is False and c[2] == pp.SUB_FALLBACK_MINUTES] == [("Mario Pašalić", False, 20.0)]
