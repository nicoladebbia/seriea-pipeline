"""Component ledger: ex-ante freeze, grading math, rot alarm, gated refit."""
from __future__ import annotations

import json
import math

import pandas as pd
import pytest

import scripts.prediction.component_ledger as cl


def _patch(monkeypatch, tmp_path):
    monkeypatch.setattr(cl, "LEDGER", tmp_path / "ledger.json")
    monkeypatch.setattr(cl, "PREDICTIONS", tmp_path / "preds.json")
    monkeypatch.setattr(cl, "MATCHES", tmp_path / "matches.parquet")
    monkeypatch.setattr(cl, "WEIGHTS_OVERRIDE", tmp_path / "weights.json")
    monkeypatch.setattr(cl, "_nt", lambda s: s)


def _pred_row(home="Genoa", away="Como", date="2026-09-10", ph=0.5):
    return {"league": "serie_a", "home_team": home, "away_team": away,
            "date": date,
            "component_predictions": {
                "ml": {"prob_H": ph, "prob_D": 0.3, "prob_A": round(0.7 - ph, 4)},
                "market": {"prob_H": 0.4, "prob_D": 0.3, "prob_A": 0.3},
            },
            "weights_applied": {"ml": 0.6, "market": 0.4},
            "betting_probabilities": {"home": 0.45, "draw": 0.3, "away": 0.25}}


def test_snapshot_upserts_pre_kickoff_then_freezes(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    kick = 2_000_000.0
    monkeypatch.setattr(cl, "_kickoff_by_pair",
                        lambda: {("Genoa", "Como"): kick})
    cl.PREDICTIONS.write_text(json.dumps({"predictions": [_pred_row(ph=0.5)]}))
    assert cl.snapshot(now_ts=kick - 7200)["stored"] == 1
    # refresh pre-kickoff: the newer forecast wins
    cl.PREDICTIONS.write_text(json.dumps({"predictions": [_pred_row(ph=0.55)]}))
    assert cl.snapshot(now_ts=kick - 600)["stored"] == 1
    # post-kickoff: freeze once, then refuse
    cl.PREDICTIONS.write_text(json.dumps({"predictions": [_pred_row(ph=0.99)]}))
    assert cl.snapshot(now_ts=kick + 60)["frozen"] == 1
    assert cl.snapshot(now_ts=kick + 120)["skipped"] == 1
    led = json.loads(cl.LEDGER.read_text())
    row = led["matches"]["2026-09-10_Genoa_Como"]
    assert row["components"]["ml"]["prob_H"] == 0.55   # last PRE-kickoff value
    assert row["frozen_at"]


def test_snapshot_refuses_post_hoc_row_with_unknown_kickoff(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    monkeypatch.setattr(cl, "_kickoff_by_pair", lambda: {})
    cl.PREDICTIONS.write_text(json.dumps(
        {"predictions": [_pred_row(date="2000-01-01")]}))
    out = cl.snapshot(now_ts=3_000_000.0)
    assert out["stored"] == 0                      # ex-ante or nothing


def test_settle_grades_each_component(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    led = {"matches": {"2026-09-10_Genoa_Como": {
        "date": "2026-09-10", "home": "Genoa", "away": "Como",
        "kickoff_ts": 1.0, "frozen_at": "x", "settled_at": None,
        "components": {"ml": {"prob_H": 0.5, "prob_D": 0.3, "prob_A": 0.2}},
        "ensemble": {"home": 0.5, "draw": 0.3, "away": 0.2},
    }}, "alarm_state": {}}
    cl.LEDGER.write_text(json.dumps(led))
    pd.DataFrame([{"match_date": "2026-09-10", "home_team": "Genoa",
                   "away_team": "Como", "result": "H", "league": "serie_a"}]
                 ).to_parquet(cl.MATCHES)
    assert cl.settle() == 1
    g = json.loads(cl.LEDGER.read_text())["matches"][
        "2026-09-10_Genoa_Como"]["grades"]["ml"]
    assert g["brier"] == round(0.25 + 0.09 + 0.04, 4)
    assert g["log_loss"] == round(-math.log(0.5), 4)
    assert g["correct"] == 1


def _settled_ledger(n, good="market", bad="ml"):
    """n settled rows where `good` nails the outcome and `bad` is noise."""
    matches = {}
    for i in range(n):
        out = "HDA"[i % 3]
        probs_good = {"prob_H": 0.1, "prob_D": 0.1, "prob_A": 0.1}
        probs_good["prob_" + out] = 0.8
        comps = {bad: {"prob_H": 0.34, "prob_D": 0.33, "prob_A": 0.33},
                 good: probs_good,
                 "xg": {"prob_H": 0.34, "prob_D": 0.33, "prob_A": 0.33},
                 "player_xg": {"prob_H": 0.34, "prob_D": 0.33, "prob_A": 0.33},
                 "factor": {"prob_H": 0.34, "prob_D": 0.33, "prob_A": 0.33}}
        matches[f"2026-01-{i:02d}_H{i}_A{i}"] = {
            "date": f"2026-01-{i % 28 + 1:02d}", "home": f"H{i}", "away": f"A{i}",
            "settled_at": "x", "outcome": out, "components": comps,
            "grades": {name: cl._grade(p, out) for name, p in comps.items()},
        }
    return {"matches": matches, "alarm_state": {}}


def test_refit_below_floor_is_report_only(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    cl.LEDGER.write_text(json.dumps(_settled_ledger(30)))
    rep = cl.refit_weights()
    assert rep["status"] == "below-floor"
    assert not cl.WEIGHTS_OVERRIDE.exists()


def test_refit_shifts_weight_to_the_better_component_and_gates(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    cl.LEDGER.write_text(json.dumps(_settled_ledger(150)))
    rep = cl.refit_weights()
    assert rep["status"] == "deployed"
    w = json.loads(cl.WEIGHTS_OVERRIDE.read_text())["weights"]
    assert w["market"] > 0.205                 # weight moved TOWARD the good leg
    assert w["ml"] < 0.605                     # and away from the noise leg
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert rep["holdout_ll_new"] < rep["holdout_ll_current"]


def test_rot_alarm_fires_once_and_clears(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    led = _settled_ledger(60, good="market", bad="ml")
    # degrade ml's RECENT window only: last 20 rows get max-wrong grades
    keys = sorted(led["matches"], key=lambda k: (led["matches"][k]["date"], k))
    for k in keys[-cl.ROLL_RECENT:]:
        led["matches"][k]["grades"]["ml"] = {"brier": 2.0, "log_loss": 6.0,
                                             "correct": 0}
    cl.LEDGER.write_text(json.dumps(led))
    first = cl.rot_alarm()
    assert first and "ml" in first
    assert cl.rot_alarm() is None              # change-gated: no repeat

def test_engine_loader_validates_and_loads(monkeypatch, tmp_path):
    import scripts.prediction.ensemble_prediction_engine as eng
    monkeypatch.setattr(eng, "DATA_DIR", tmp_path)
    (tmp_path / "models").mkdir()
    path = tmp_path / "models" / "ensemble_weights.json"
    load = eng.EnsemblePredictor._load_ledger_weights

    assert load() is None                                    # no file
    good = {"factor": 0.05, "xg": 0.15, "ml": 0.5, "player_xg": 0.05,
            "market": 0.25}
    path.write_text(json.dumps({"weights": good, "n_settled": 120}))
    assert load() == good                                    # valid file loads
    path.write_text(json.dumps({"weights": {**good, "deep": 0.0}}))
    assert load() is None                                    # wrong keys
    path.write_text(json.dumps({"weights": {**good, "ml": 0.97, "market": 0.0,
                                            "xg": 0.0, "player_xg": 0.0,
                                            "factor": 0.03}}))
    assert load() is None                                    # weight > 0.95
    path.write_text(json.dumps({"weights": {k: v * 2 for k, v in good.items()}}))
    assert load() is None                                    # sum != 1
