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


def test_handle_picks_renders_every_label_and_tolerates_missing(tmp_path, monkeypatch):
    """/picks reads data/upcoming/picks.json (scripts/betting/picks.py) since
    2026-09-05: one line per match, VALUE / LEAN / NO EDGE."""
    import json

    import scripts.pipeline.telegram_bot as tb

    monkeypatch.setattr(tb, "PROJECT_ROOT", tmp_path)
    assert "picks.json" in tb._handle_picks()
    up = tmp_path / "data" / "upcoming"
    up.mkdir(parents=True)
    doc = {"generated_at": "2026-09-05T12:00:00+00:00",
           "counts": {"VALUE": 1, "LEAN": 1, "NO_EDGE": 1},
           "picks": [
               {"match": "A vs B", "date": "2026-09-05", "label": "VALUE", "stage": "selected",
                "pick": {"bet_type": "O/U 1.5", "selection": "Over 1.5", "odds": 1.41, "edge_pct": 6.5,
                         "tier": "engine"},
                "reason": "the betting engine's own selection: real stake, committed at T-30"},
               {"match": "C vs D", "date": "2026-09-06", "label": "LEAN",
                "pick": {"bet_type": "1x2 finale", "selection": "1", "odds": 2.9, "edge_pct": 4.4, "tier": "A",
                         "probability_pct": 36.0, "implied_pct": 34.5, "n_books": 24},
                "alternatives": [{"bet_type": "Under/over", "selection": "Over 2.5", "odds": 1.9, "edge_pct": 2.0,
                                  "tier": "B", "probability_pct": 53.7, "implied_pct": 52.6, "n_books": 9}],
                "exotic": [{"bet_type": "Assist giocatore", "player": "Federico Dimarco", "selection": "Sì",
                            "odds": 4.5, "edge_pct": 3.5, "tier": "C", "probability_pct": 23.0,
                            "implied_pct": 22.2, "n_books": 1}],
                "reason": "model 36.0% vs market 34.5% (best of market, 24 books); edge inside the credible band, paper stake"},
               {"match": "E vs F", "date": "2026-09-06", "label": "NO_EDGE", "pick": None,
                "most_probable": {"bet_type": "1x2 finale", "selection": "1", "odds": 1.5, "edge_pct": -4.0,
                                  "tier": "A"},
                "reason": "most probable is 1 at 64.0%, but the market prices it at 66.7% (1.5): no edge"},
           ]}
    (up / "picks.json").write_text(json.dumps(doc))
    out = tb._handle_picks()
    assert "A vs B" in out and "💰" in out and "VALUE" in out
    assert "📝" in out and "LEAN · solo carta" in out
    assert "Federico Dimarco Sì @ 4.50" in out and "Insolite" in out       # the exotic slot
    assert "Over 2.5 @ 1.90" in out and "Alternative" in out
    assert "E vs F" in out and "➖" in out and "no edge" in out
    assert "solo carta" in out and "tasso base" in out  # legend: LEAN is paper, tiers explained


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


def test_send_message_keeps_formatting_on_a_network_blip(monkeypatch):
    """2026-09-05: a connection reset on chunk 2 of /picks resent it without
    parse_mode and the phone showed literal <b> tags. A network error retries
    with the same params; only an HTTP 400 strips the markup."""
    import scripts.pipeline.telegram_bot as tb

    calls = []
    outcomes = iter(["network", "ok"])

    def fake_request(token, method, params=None, timeout=35):
        st = next(outcomes)
        tb._LAST_TG_STATUS = st
        calls.append(dict(params))
        return {"message_id": 1} if st == "ok" else None

    monkeypatch.setattr(tb, "_tg_request", fake_request)
    monkeypatch.setattr(tb.time, "sleep", lambda s: None)
    assert tb._tg_send_message("t", "c", "<b>x</b>") is True
    assert [c.get("parse_mode") for c in calls] == ["HTML", "HTML"]

    calls.clear()
    outcomes = iter(["http:400", "ok"])
    assert tb._tg_send_message("t", "c", "<b>x</b>") is True
    assert [c.get("parse_mode") for c in calls] == ["HTML", None]

    calls.clear()
    outcomes = iter(["network", "network"])
    assert tb._tg_send_message("t", "c", "<b>x</b>") is False
    assert len(calls) == 2 and all(c.get("parse_mode") == "HTML" for c in calls)


def test_handle_picks_never_lists_the_value_bet_or_the_lean_twice(tmp_path, monkeypatch):
    import json

    import scripts.pipeline.telegram_bot as tb

    monkeypatch.setattr(tb, "PROJECT_ROOT", tmp_path)
    up = tmp_path / "data" / "upcoming"
    up.mkdir(parents=True)
    x2 = {"bet_type": "Doppia chance", "selection": "X2", "odds": 1.4, "edge_pct": 9.6, "tier": "B",
          "probability_pct": 78.0, "implied_pct": 71.4, "n_books": 3}
    doc = {"generated_at": "2026-09-05T12:00:00+00:00", "counts": {"VALUE": 1, "LEAN": 0, "NO_EDGE": 0},
           "picks": [{"match": "Lazio vs Milan", "date": "2026-09-12", "label": "VALUE", "stage": "selected",
                      "pick": {"bet_type": "O/U 1.5", "selection": "Over 1.5", "odds": 1.41, "edge_pct": 6.5,
                               "tier": "engine", "probability_pct": 78.0},
                      "lean": x2,
                      "alternatives": [{"bet_type": "Under/over", "selection": "Over 1.5", "odds": 1.41,
                                        "edge_pct": 10.0, "tier": "B", "probability_pct": 78.0,
                                        "implied_pct": 70.9, "n_books": 3}, x2],
                      "reason": "r"}]}
    (up / "picks.json").write_text(json.dumps(doc))
    out = tb._handle_picks()
    assert out.count("Over 1.5 @ 1.41") == 1 and out.count("X2 @ 1.40") == 1
    assert "Alternative" not in out

