"""The edge band of every ENABLED market x line must be non-empty — and O/U 2.5
must accept its journal-proven [7, 10) window.

Found 2026-09-01: O/U_Over carried line_min_edge {2.5: 7.0} while the
market-wide max_edge_pct was 7.0 -> the O/U 2.5 acceptance band was [7.0, 7.0]
and the ONLY enabled market pair could never fire on that line (a +8.3% edge
died as above_max_edge). The [7,7] band shipped silently because nothing
asserted min < max after per-line overrides. Journal evidence for the window:
edge 7-10% = +12.8% ROI / CLV +4.93 (n=16); edge >=10% = -26.9% ROI (n=17).
"""

from scripts.betting.betting_unified import BettingConfig, UnifiedBettingEngine


def _effective_band(rule, line):
    lo = (rule.get("line_min_edge") or {}).get(line, rule.get("min_edge_pct", 5.0))
    hi = (rule.get("line_max_edge") or {}).get(line, rule.get("max_edge_pct", 7.0))
    return lo, hi


def test_every_enabled_market_line_band_is_nonempty():
    """The empty-band regression guard: after per-line overrides, min < max."""
    rules = BettingConfig().market_rules
    for cat, rule in rules.items():
        if not rule.get("enabled", False):
            continue
        lines = rule.get("allowed_lines") or [None]
        for line in lines:
            lo, hi = _effective_band(rule, line)
            assert lo < hi, (
                f"{cat} line={line}: edge band [{lo}, {hi}] is empty — "
                "every candidate on this line is unconditionally rejected"
            )


def _make(engine, model_p, sharp_p, market="O/U 2.5", selection="Over 2.5",
          best_o=2.45, pin_o=2.40, min_edge_override=7.0):
    # 2026-09-05 is a Saturday (Mon/Fri are stake-gated); odds 2.45 sits
    # outside the 1.5-2.0 dead zone and within [min_odds, max_odds];
    # pin*1.05 >= best_o keeps the Pinnacle-overprice gate quiet.
    return engine._make_bet(
        "Udinese vs Lazio", "2026-09-05", market, selection,
        model_p, sharp_p, best_o, "Pinnacle", best_o, pin_o, 5,
        min_edge_override=min_edge_override,
    )


def test_ou25_edge_in_seven_ten_window_passes():
    engine = UnifiedBettingEngine()
    # unshrunk line: edge = (0.483 - 0.40) * 100 = 8.3pp — the exact shape
    # that was rejected above_max_edge before the line_max_edge fix.
    bet = _make(engine, model_p=0.483, sharp_p=0.40)
    assert bet is not None, (
        f"8.3pp O/U 2.5 edge must pass the [7, 10] band; near_misses="
        f"{engine.near_misses}"
    )


def test_ou25_edge_above_ten_rejected():
    engine = UnifiedBettingEngine()
    # 10.5pp > line_max_edge 10.0 -> overconfidence guard still bites.
    bet = _make(engine, model_p=0.505, sharp_p=0.40)
    assert bet is None
    assert engine.near_misses, "rejection must be recorded as a near miss"
    miss = engine.near_misses[-1]
    assert miss["reason"] == "above_max_edge"
    assert miss["max_edge"] == 10.0, "per-line max (10.0) must be the recorded cap"


def test_ou15_max_edge_unchanged_at_market_cap():
    engine = UnifiedBettingEngine()
    # Line 1.5 has no line_max_edge -> market cap 7.0 still applies.
    # line_shrinkage 0.6: effective edge = 0.6*(0.825-0.70)*100 = 7.5pp > 7.0.
    bet = _make(engine, model_p=0.825, sharp_p=0.70, market="O/U 1.5",
                selection="Over 1.5", best_o=1.30, pin_o=1.28,
                min_edge_override=None)
    assert bet is None
    miss = engine.near_misses[-1]
    assert miss["reason"] == "above_max_edge"
    assert miss["max_edge"] == 7.0
