"""/api/match-markets assembles every priced market for one match with its tier.

Nothing here computes a probability; the tests pin that (a) each row names the
engine and tier it came from, (b) constants are excluded, not served, (c) the
unbuilt engines are listed as not_built, and (d) missing artifacts degrade to
empty groups, never a 500.
"""
from web.match_markets import NOT_BUILT, build_match_markets

PRED = {"match": "Fiorentina vs Torino", "league": "serie_a", "home_team": "Fiorentina",
        "away_team": "Torino", "probabilities": {"home": 0.349, "draw": 0.348, "away": 0.303},
        "home_xg": 1.14, "away_xg": 0.91, "home_factors": ["cold_away"],
        "away_factors": ["cold_home"], "neutral_factors": [],
        "market_implied": {"home": 0.485, "draw": 0.291, "away": 0.223, "source": "sharp_consensus"},
        "confidence_level": "MEDIUM", "methods_used": ["market", "xg"]}
GOAL = {"match": "Fiorentina vs Torino", "expected_total_goals": 2.15, "over_0_5": 0.884,
        "over_1_5": 0.627, "over_2_5": 0.392, "over_3_5": 0.215, "over_4_5": 0.067}
EXT = {"double_chance": {"1X": {"prob": 0.697}, "X2": {"prob": 0.651}, "12": {"prob": 0.652}},
       "exact_score": [{"score": "1-0", "prob": 0.1468}, {"score": "1-1", "prob": 0.1335}],
       "team_corners": {"home_expected": 5.0, "away_expected": 4.8, "home_over_4_5": {"prob": 0.5595}},
       "red_card": {"yes": {"prob": 0.1393}, "no": {"prob": 0.8607}},
       "first_half": {"result_1x2": {"home": {"prob": 0.2858}, "draw": {"prob": 0.4975}, "away": {"prob": 0.2167}},
                      "over_under": {"over_0.5": {"prob": 0.5858}}},
       "team_cards": {"home_expected": 2.16, "home_over_1_5": {"prob": 0.6356}, "away_over_2_5": {"prob": 0.41}},
       # scalar siblings next to {"prob": …} cells, exactly as the live artifact has them
       "booking_points": {"expected": 45.2, "over_40.5": {"prob": 0.6792}},
       "cards_by_half": {"first_half": {"expected": 2.08, "over_1_5": {"prob": 0.6161}}},
       "team_totals": {"home": {"expected": 1.27, "over_0.5": {"prob": 0.6802}}}}
PLAYERS = {"home_players": [{"player_name": "Kean", "position": "F", "proj_minutes": 84.0,
                             "markets": {"shots_o15": {"prob": 0.71, "label": "Shots Over 1.5"},
                                         "goalscorer": {"prob": 0.31, "label": "Anytime Goalscorer"}}}],
           "away_players": []}


def _build(**over):
    kw = dict(pred=PRED, goal_pred=GOAL, ext=EXT, btts={"btts_yes": 0.395, "btts_no": 0.605},
              engine_bet={"status": "none"}, players=PLAYERS)
    kw.update(over)
    return build_match_markets("Fiorentina vs Torino", **kw)


def _rows(out, bet_type):
    return [r for r in out["markets"] if r["bet_type"] == bet_type]


def test_1x2_is_tier_a_from_the_ensemble_and_keeps_italian_names():
    out = _build()
    r = _rows(out, "1x2 finale")
    assert [x["selection"] for x in r] == ["1", "X", "2"]
    assert all(x["tier"] == "A" and x["source"] == "ensemble" for x in r)
    assert r[1]["probability_pct"] == 34.8


def test_over_under_uses_the_ou_blend_and_flags_the_bet_lines():
    out = _build()
    r = {x["selection"]: x for x in _rows(out, "Under/over")}
    assert r["Over 2.5"]["probability_pct"] == 39.2 and r["Under 2.5"]["probability_pct"] == 60.8
    assert r["Over 2.5"]["bet_line"] is True and r["Over 0.5"]["bet_line"] is False
    assert all(x["tier"] == "B" for x in r.values())


def test_constant_corners_are_excluded_not_served():
    out = _build()
    assert not _rows(out, "Corner")
    ex = [e for e in out["excluded"] if e["bet_type"] == "Corner"]
    assert ex and "CONSTANT" in ex[0]["reason"]


def test_scalar_siblings_in_artifact_cells_are_skipped_not_500():
    out = _build()
    sel = {(r["bet_type"], r["selection"]) for r in out["markets"]}
    assert ("Punti cartellini", "Over 40.5") in sel and ("Punti cartellini", "Expected") not in sel
    assert ("Cartellini 1° tempo", "Over 1.5") in sel
    assert ("Gol Casa", "Over 0.5") in sel and ("Gol Casa", "Expected") not in sel


def test_rare_events_are_tier_c_and_unbuilt_engines_are_listed_not_priced():
    out = _build()
    assert all(x["tier"] == "C" for x in _rows(out, "Espulsione"))
    assert out["not_built"] is NOT_BUILT
    assert not any(n["bet_type"] == "Vince o quasi" for n in out["not_built"])  # built 2026-09-05 (goal_process)
    assert any(n["bet_type"] == "Estro finale" for n in out["not_built"])
    assert not any("Vince" in r["bet_type"] for r in out["markets"])


def test_player_floors_are_tier_a_except_goalscorer():
    out = _build()
    by = {p["bet_type"]: p for p in out["players"]}
    assert by["Tiri totali del giocatore"]["tier"] == "A"
    assert by["Tiri totali del giocatore"]["player"] == "Kean"
    assert by["Giocatore marcatore"]["tier"] == "C"
    assert by["Tiri totali del giocatore"]["contribution_pct"] is None  # placeholder until step 3


def test_missing_artifacts_degrade_to_empty_groups_and_are_named():
    out = _build(pred=None, goal_pred=None, ext=None, btts=None, players=None)
    assert out["markets"] == [] and out["players"] == []
    assert "predictions.json row" in out["missing"]
    assert "goal_predictions.json row" in out["missing"]
    assert out["engine_bet"] == {"status": "none"}


def test_reasoning_is_the_models_own_factors_not_prose():
    out = _build()
    assert "xG 1.14 v 0.91" in out["reasoning"]
    assert "away: cold_home" in out["reasoning"]
    assert any(s.startswith("market implied H/D/A 48.5/29.1/22.3%") for s in out["reasoning"])


def test_endpoint_serves_fixtures_and_survives_a_dead_player_engine(monkeypatch):
    import web.app as appmod
    real = appmod._load_json

    def fake(path, default=None):
        name = getattr(path, "name", str(path))
        if name == "predictions.json":
            return [PRED]
        if name == "goal_predictions.json":
            return [GOAL]
        if name == "extended_markets.json":
            return {"matches": {"Fiorentina vs Torino": EXT}}
        if name == "btts_predictions.json":
            return [{"match": "Fiorentina vs Torino", "btts_yes": 0.395, "btts_no": 0.605}]
        if name == "unified_bet_slip.json":
            return {"near_misses": [{"match": "Fiorentina vs Torino", "market": "O/U 2.5",
                                     "selection": "Over 2.5", "edge_pct": 7.5, "min_edge": 7.0,
                                     "max_edge": 10.0, "gap_pp": 0.0, "reason": "veto_factor:cold_home",
                                     "best_odds": 2.14}]}
        if name in ("odds_full.json", "betting_candidates.json", "predictions_premier_league.json",
                    "lineup_predictions.json"):
            return default if default is not None else {}
        return real(path, default=default)

    monkeypatch.setattr(appmod, "_load_json", fake)
    monkeypatch.setattr(appmod, "_player_engine", lambda: (None, None))
    r = appmod.app.test_client().get("/api/match-markets/fiorentina-vs-torino")
    assert r.status_code == 200, r.data
    d = r.get_json()
    assert d["match"] == "Fiorentina vs Torino"
    assert d["engine_bet"]["status"] == "vetoed" and d["engine_bet"]["reason"] == "veto_factor:cold_home"
    assert [x["probability_pct"] for x in d["markets"] if x["bet_type"] == "1x2 finale"] == [34.9, 34.8, 30.3]
    assert d["players"] == [] and "player floors (no lineup or engine)" in d["missing"]
    assert appmod.app.test_client().get("/api/match-markets/nope-vs-nobody").status_code == 404


def test_simulator_rows_replace_their_poisson_twins_and_add_new_markets():
    """A goal-process row for the same (bet_type, selection) supersedes the
    independent-Poisson artifact row; markets only the simulator prices are
    appended; the 5.5/6.5 exclusion disappears."""
    from web.match_markets import build_match_markets
    ext = {"first_half": {"result_1x2": {"home": {"prob": 0.286}, "draw": {"prob": 0.498}, "away": {"prob": 0.217}}}}
    sim = [
        {"group": "Tempi", "bet_type": "1° tempo 1x2", "selection": "1", "probability_pct": 31.1, "tier": "A", "source": "goal_process"},
        {"group": "Principali", "bet_type": "Vince o quasi", "selection": "Casa 1x sì", "probability_pct": 55.6, "tier": "A", "source": "goal_process"},
        {"group": "Under/over", "bet_type": "Under/over", "selection": "Over 5.5", "probability_pct": 2.6, "tier": "B", "source": "goal_process"},
    ]
    out = build_match_markets("A vs B", pred={"probabilities": {"home": .4, "draw": .3, "away": .3}},
                              goal_pred={"over_2_5": 0.5}, ext=ext, btts=None, engine_bet=None,
                              players=None, league="serie_a", sim=sim)
    ht1 = [r for r in out["markets"] if r["bet_type"] == "1° tempo 1x2" and r["selection"] == "1"]
    assert len(ht1) == 1 and ht1[0]["source"] == "goal_process" and ht1[0]["tier"] == "A"
    # the untouched Poisson X / 2 rows stay
    assert {r["selection"] for r in out["markets"] if r["bet_type"] == "1° tempo 1x2"} == {"1", "X", "2"}
    assert any(r["bet_type"] == "Vince o quasi" for r in out["markets"])
    assert not any(e["bet_type"] == "Under/over" for e in out["excluded"])
    assert not any(n["bet_type"] == "Vince o quasi" for n in out["not_built"])
    assert not any("simulator" in m for m in out["missing"])


def test_no_simulator_rows_is_named_not_silent():
    from web.match_markets import build_match_markets
    out = build_match_markets("A vs B", pred=None, goal_pred={"over_2_5": 0.5}, ext=None, btts=None,
                              engine_bet=None, players=None, league="serie_a", sim=None)
    assert any("simulator" in m for m in out["missing"])
    assert any(e["selection"] == "Over/Under 5.5" for e in out["excluded"])


def test_player_rows_carry_split_and_contribution_with_the_halves_gate():
    from web.match_markets import _split_row
    gate = {"shots_o05": {"passed": True, "skill": 0.03}}
    measured = {"1h": 0.4, "2h": 0.45, "both": 0.18, "timing": "measured"}
    r = _split_row(measured, gate, "shots_o05")
    assert r["tier"] == "A" and r["1h_pct"] == 40.0 and r["both_pct"] == 18.0
    assert _split_row(measured, gate, "sot_o05")["tier"] == "B"                 # not in the gate
    assert _split_row({**measured, "timing": "flat", "both": None}, gate, "shots_o05")["tier"] == "C"
    assert _split_row(None, gate, "shots_o05") is None
