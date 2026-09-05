"""Every post-edge rejection in _make_bet is recorded, so "0 bets" is auditable.

Go-live weekend 2026-08-29..31 produced zero bets and the only artifact was
`TOTAL: 0 value bets` — no record of how far the best candidate was from the
band. The engine now keeps `near_misses` (edge, band, odds, reason) and the slip
carries the closest ten.
"""

from scripts.betting.betting_unified import UnifiedBettingEngine


def _bet(engine, model_p, sharp_p, best_o=2.10, pin_o=2.05, **kw):
    return engine._make_bet(
        match="Inter vs Napoli", date="2026-09-01", market="O/U 2.5",
        selection="Over 2.5", model_p=model_p, sharp_p=sharp_p, best_o=best_o,
        best_bk="bet365", avg_o=pin_o, pin_o=pin_o, count=12, **kw)


def test_below_min_edge_is_recorded_with_gap():
    eng = UnifiedBettingEngine()
    # O/U 2.5 is unshrunk; caller passes line_min_edge 7.0 as the override.
    # model_p 0.60 is the "high" confidence tier -> min_edge 7.0 - 1.5 = 5.5,
    # and _make_bet resolves line_max_edge[2.5] = 10.0 itself — the record
    # must show the band the bet was ACTUALLY judged against.
    assert _bet(eng, 0.60, 0.58, min_edge_override=7.0) is None
    (m,) = eng.near_misses
    assert m["reason"] == "below_min_edge"
    assert m["edge_pct"] == 2.0
    assert m["min_edge"] == 5.5 and m["max_edge"] == 10.0
    assert m["gap_pp"] == 3.5
    assert m["match"] == "Inter vs Napoli" and m["market"] == "O/U 2.5"


def test_above_max_edge_is_recorded():
    eng = UnifiedBettingEngine()
    assert _bet(eng, 0.70, 0.58, min_edge_override=7.0) is None
    (m,) = eng.near_misses
    assert m["reason"] == "above_max_edge"
    # O/U 2.5 max is line_max_edge 10.0 (was market-level 7.0 pre-band-fix)
    assert m["edge_pct"] == 12.0 and m["gap_pp"] == 2.0


def test_dead_zone_odds_recorded_with_zero_gap_when_edge_inside_band():
    eng = UnifiedBettingEngine()
    # edge 6.8 inside [5, 7] but odds 1.80 sit in the 1.5-2.0 dead zone
    assert _bet(eng, 0.648, 0.58, best_o=1.80, pin_o=1.78) is None
    (m,) = eng.near_misses
    assert m["reason"] == "odds_dead_zone_1.5_2.0"
    assert m["gap_pp"] == 0.0


def test_accepted_bet_records_nothing():
    eng = UnifiedBettingEngine()
    bet = _bet(eng, 0.648, 0.58)  # edge 6.8 in band, odds 2.10 (golden zone)
    assert bet is not None, eng.near_misses
    assert eng.near_misses == []


def test_disabled_market_is_not_a_near_miss():
    eng = UnifiedBettingEngine()
    assert eng._make_bet("A vs B", "2026-09-01", "O/U 2.5", "Under 2.5",
                         0.60, 0.50, 2.1, "bk", 2.05, 2.05, 10) is None
    assert eng.near_misses == []  # rejected before any edge existed


def test_top_near_misses_orders_by_gap_then_edge():
    eng = UnifiedBettingEngine()
    # model_p 0.50 = "medium" tier (no adjustment) -> band [7, 10]
    _bet(eng, 0.50, 0.48, min_edge_override=7.0)   # edge 2  -> gap 5.0
    _bet(eng, 0.50, 0.44, min_edge_override=7.0)   # edge 6  -> gap 1.0  <- closest
    _bet(eng, 0.70, 0.58, min_edge_override=7.0)   # edge 12 -> gap 2.0
    top = eng.top_near_misses(2)
    assert [m["gap_pp"] for m in top] == [1.0, 2.0]
    assert top[1]["edge_pct"] == 12.0  # tie on gap -> larger edge first


def _vetoed_pred(*factors):
    return {"match": "Inter vs Napoli", "home_factors": list(factors),
            "away_factors": [], "neutral_factors": []}


def test_factor_veto_no_longer_touches_totals():
    """2026-09-04: 14 of 19 Serie A matches died on the cold_home / away_fav_ref
    veto BEFORE the edge calc — silently. 2026-09-05 (Nicola: "bet more"): the
    veto is 1X2-era evidence (n=8, -33% on 1X2/DC/DNB; n=2, +23% on O/U) and is
    scoped to the result markets. An in-band Over 2.5 with both factors is a bet."""
    eng = UnifiedBettingEngine()
    bet = _bet(eng, 0.648, 0.58, pred=_vetoed_pred("cold_home", "away_fav_ref"))
    assert bet is not None, eng.near_misses
    assert eng.near_misses == []


def test_veto_scope_is_the_result_markets():
    from scripts.betting.betting_unified import _veto_applies
    assert _veto_applies("1X2") and _veto_applies("1X2_Away") and _veto_applies("DC") and _veto_applies("DNB")
    assert not _veto_applies("O/U_Over") and not _veto_applies("O/U_Under") and not _veto_applies("Alt_OU")
    assert not _veto_applies("")


def test_vetoed_result_market_is_recorded_not_silent():
    """Where the veto still applies it must say so in the slip. 1X2/DC/DNB are
    disabled in market_rules, so exercise the gate with 1X2 force-enabled."""
    eng = UnifiedBettingEngine()
    eng.cfg.market_rules["1X2"] = dict(eng.cfg.market_rules["1X2"], enabled=True)
    assert eng._make_bet(
        match="Inter vs Napoli", date="2026-09-01", market="1X2", selection="Home",
        model_p=0.62, sharp_p=0.55, best_o=1.90, best_bk="bet365", avg_o=1.85, pin_o=1.85,
        count=12, pred=_vetoed_pred("cold_home")) is None
    reasons = [m["reason"] for m in eng.near_misses]
    assert "veto_factor:cold_home" in reasons, reasons
