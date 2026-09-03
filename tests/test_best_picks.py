"""best_pick_per_match — the advisory that ranks every match's best angle.

The advisory must (1) prefer credible in-band edges over out-of-band
mirages, (2) let noise-tier markets top a match only when nothing
validated has a price, (3) skip quarter/whole O/U lines the goals model
has no probability for, and (4) never crash the money path — it's pure.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.betting.betting_unified import best_pick_per_match


def _pred(match="A vs B", ph=0.5, pd=0.3, pa=0.2):
    return {"match": match, "date": "2026-09-05", "time": "18:00",
            "league": "serie_a",
            "probabilities": {"home": ph, "draw": pd, "away": pa}}


def test_in_band_edge_outranks_out_of_band_mirage():
    # Home: p=0.5 @ 2.10 -> +5% (in band). Away: p=0.2 @ 12.0 -> +140% (mirage).
    odds = {"A vs B": {"h2h": {"best_home": 2.10, "best_draw": 3.0,
                               "best_away": 12.0, "bookmakers_count": 20}}}
    picks = best_pick_per_match([_pred()], odds, [], {}, band=(2.0, 12.0))
    assert len(picks) == 1
    best = picks[0]["best"]
    assert best["selection"] == "Home" and best["in_band"]
    # the mirage is still visible in top3, flagged
    mirage = [c for c in picks[0]["top3"] if c["selection"] == "Away"]
    assert mirage and not mirage[0]["in_band"]


def test_noise_tier_only_tops_without_validated_price():
    # No h2h/totals at all; only a BTTS price + noise-tier model prob.
    extra = {"A vs B": {"btts": {"best_yes": 2.2, "best_no": 1.7,
                                 "bookmakers_count": 3}}}
    btts = [{"match": "A vs B", "btts_yes": 0.55, "btts_no": 0.45}]
    picks = best_pick_per_match([_pred()], {}, [], extra, btts)
    assert picks[0]["best"]["tier"] == "noise"
    # ...but with any validated candidate present, noise never wins
    odds = {"A vs B": {"h2h": {"best_home": 2.05, "best_draw": 3.2,
                               "best_away": 5.0, "bookmakers_count": 10}}}
    picks2 = best_pick_per_match([_pred()], odds, [], extra, btts)
    assert picks2[0]["best"]["tier"] != "noise"


def test_ou_quarter_lines_skipped_half_lines_priced():
    gp = [{"match": "A vs B", "over_2_5": 0.55}]
    odds = {"A vs B": {"h2h": {}, "totals": [
        {"line": 2.5, "best_over": 1.95, "best_under": 1.95,
         "bookmakers_count": 5},
        {"line": 2.25, "best_over": 2.04, "best_under": 1.86,
         "bookmakers_count": 5},   # quarter line: no model prob -> skipped
        {"line": 3.0, "best_over": 2.4, "best_under": 1.6,
         "bookmakers_count": 5},   # whole line -> skipped
    ]}}
    picks = best_pick_per_match([_pred()], odds, gp, {})
    markets = {(c["market"], c["selection"]) for c in picks[0]["top3"]}
    assert ("O/U", "Over 2.5") in markets
    assert not any("2.25" in s or "3.0" in s for _, s in markets)


def test_thin_market_guard_and_missing_prob():
    # 1 bookmaker -> excluded; missing model prob -> excluded; no crash
    odds = {"A vs B": {"h2h": {"best_home": 3.0, "bookmakers_count": 1}}}
    picks = best_pick_per_match([_pred()], odds, [], {})
    assert picks == []  # nothing credible to say
    picks2 = best_pick_per_match([{"match": "A vs B", "probabilities": {}}],
                                 odds, [], {})
    assert picks2 == []


def test_every_match_on_the_slate_gets_a_row():
    preds = [_pred("A vs B"), _pred("C vs D", 0.4, 0.3, 0.3)]
    odds = {m: {"h2h": {"best_home": 2.2, "best_draw": 3.3, "best_away": 4.0,
                        "bookmakers_count": 8}} for m in ("A vs B", "C vs D")}
    picks = best_pick_per_match(preds, odds, [], {})
    assert {p["match"] for p in picks} == {"A vs B", "C vs D"}
    for p in picks:
        assert p["best"]["edge_pct"] == p["top3"][0]["edge_pct"]


# ── bot surface: /picks handler + AI budget gate ─────────────────────────


def test_handle_picks_renders_flags_and_tolerates_missing(tmp_path, monkeypatch):
    import json

    import scripts.pipeline.telegram_bot as tb

    monkeypatch.setattr(tb, "PROJECT_ROOT", tmp_path)
    assert "slip" in tb._handle_picks() or "motore" in tb._handle_picks()
    up = tmp_path / "data" / "upcoming"
    up.mkdir(parents=True)
    slip = {"generated_at": "2026-09-03T12:00:00+00:00", "selected_bets": [],
            "best_picks": [
                {"match": "A vs B", "date": "2026-09-05",
                 "best": {"market": "1X2", "selection": "Draw", "odds": 3.4,
                          "edge_pct": 5.0, "model_prob": 0.31,
                          "tier": "validated", "in_band": True}},
                {"match": "C vs D", "date": "2026-09-06",
                 "best": {"market": "1X2", "selection": "Away", "odds": 9.0,
                          "edge_pct": 80.0, "model_prob": 0.2,
                          "tier": "validated", "in_band": False}},
            ]}
    (up / "unified_bet_slip.json").write_text(json.dumps(slip))
    out = tb._handle_picks()
    assert "A vs B" in out and "✅" in out
    assert "C vs D" in out and "⚠️" in out
    assert "fuori banda" in out  # the out-of-band legend appears


def test_ai_budget_gate_caps_and_rolls_over(tmp_path, monkeypatch):
    import json
    from datetime import date

    import scripts.pipeline.telegram_bot as tb

    f = tmp_path / "tg_ai_usage.json"
    monkeypatch.setattr(tb, "_AI_USAGE_FILE", f)
    monkeypatch.setattr(tb, "_AI_DAILY_CALLS", 2)
    assert tb._ai_budget_ok() and tb._ai_budget_ok()
    assert not tb._ai_budget_ok()          # third call blocked
    st = json.loads(f.read_text())
    assert st["calls"] == 2 and st["date"] == date.today().isoformat()
    # yesterday's counter resets
    f.write_text(json.dumps({"date": "2020-01-01", "calls": 99}))
    assert tb._ai_budget_ok()
