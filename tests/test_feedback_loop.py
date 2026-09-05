"""Feedback loop: matches.parquet join, advisory clamp, evidence-gated factor steps."""
from __future__ import annotations

import json

import pandas as pd
import pytest

import scripts.analysis.feedback_analyzer as fa
import scripts.models.weight_optimizer as wo


def test_analyzer_joins_archive_against_matches_parquet(monkeypatch, tmp_path):
    monkeypatch.setattr(fa, "ARCHIVE_PATH", tmp_path / "arch.json")
    monkeypatch.setattr(fa, "MATCHES_PATH", tmp_path / "matches.parquet")
    (tmp_path / "arch.json").write_text(json.dumps({
        "Genoa vs Como_2026-09-04": {
            "match": "Genoa vs Como", "home_team": "Genoa", "away_team": "Como",
            "date": "2026-09-04", "predicted_outcome": "AWAY",
            "probabilities": {"home": 0.23, "draw": 0.28, "away": 0.49},
            "component_predictions": {"ml": {"prob_H": 0.23, "prob_D": 0.3,
                                             "prob_A": 0.47}}},
        # EPL row must be excluded by the default serie_a filter
        "Hull vs Aston Villa_2026-09-05": {
            "match": "Hull vs Aston Villa", "home_team": "Hull",
            "away_team": "Aston Villa", "date": "2026-09-05",
            "predicted_outcome": "HOME", "probabilities": {}},
        # unplayed match must not join
        "Inter vs Udinese_2026-09-14": {
            "match": "Inter vs Udinese", "home_team": "Inter",
            "away_team": "Udinese", "date": "2026-09-14",
            "predicted_outcome": "HOME", "probabilities": {}},
    }))
    pd.DataFrame([
        {"match_date": "2026-09-04", "home_team": "Genoa", "away_team": "Como",
         "home_score": 1, "away_score": 4, "result": "A", "league": "serie_a"},
        {"match_date": "2026-09-05", "home_team": "Hull",
         "away_team": "Aston Villa", "home_score": 0, "away_score": 2,
         "result": "A", "league": "premier_league"},
    ]).to_parquet(tmp_path / "matches.parquet")

    rows = fa.match_predictions_to_results()
    assert len(rows) == 1
    assert rows[0]["actual_outcome"] == "AWAY"
    assert rows[0]["league"] == "serie_a"
    assert rows[0]["total_goals"] == 5
    both = fa.match_predictions_to_results(league=None)
    assert {r["league"] for r in both} == {"serie_a", "premier_league"}


def test_analyzer_one_day_date_drift_still_joins(monkeypatch, tmp_path):
    monkeypatch.setattr(fa, "ARCHIVE_PATH", tmp_path / "arch.json")
    monkeypatch.setattr(fa, "MATCHES_PATH", tmp_path / "matches.parquet")
    (tmp_path / "arch.json").write_text(json.dumps({
        "k": {"match": "Roma vs Lazio", "home_team": "Roma", "away_team": "Lazio",
              "date": "2026-09-04", "predicted_outcome": "HOME",
              "probabilities": {}}}))
    pd.DataFrame([{"match_date": "2026-09-05", "home_team": "Roma",
                   "away_team": "Lazio", "home_score": 2, "away_score": 0,
                   "result": "H", "league": "serie_a"}]
                 ).to_parquet(tmp_path / "matches.parquet")
    rows = fa.match_predictions_to_results()
    assert len(rows) == 1 and rows[0]["actual_outcome"] == "HOME"


def test_weight_optimizer_can_never_go_active():
    # "active" is retired: deployment belongs to component_ledger only.
    assert wo._determine_status(2) == "cold_start"
    assert wo._determine_status(19) == "cold_start"
    for n in (20, 30, 100, 10_000):
        assert wo._determine_status(n) == "advisory"


def _analysis(applied, accuracy, n_settled=200):
    return {"n_settled": n_settled,
            "factor_effectiveness": {
                "away_favorite": {"applied": applied, "accuracy": accuracy,
                                  "base_rate": 0.45}}}


def test_factor_step_consumes_its_evidence_no_compounding(monkeypatch, tmp_path):
    monkeypatch.setattr(wo, "FACTOR_ADJ_PATH", tmp_path / "fadj.json")
    a = _analysis(applied=30, accuracy=0.70)      # +25pp over base -> boost
    out1 = wo.compute_factor_decay(a)
    assert out1["details"]["away_favorite"]["action"] == "boost"
    assert out1["multipliers"]["away_favorite"] == 1.1
    (tmp_path / "fadj.json").write_text(json.dumps(out1))

    # Same evidence again (the 2-3x/day pipeline case): must HOLD, not compound
    out2 = wo.compute_factor_decay(a)
    assert out2["details"]["away_favorite"]["action"] == "hold_awaiting_new_data"
    assert out2["multipliers"]["away_favorite"] == 1.1
    (tmp_path / "fadj.json").write_text(json.dumps(out2))

    # 10+ NEW applications: allowed to step again, capped at 1.2
    out3 = wo.compute_factor_decay(_analysis(applied=41, accuracy=0.70))
    assert out3["details"]["away_favorite"]["action"] == "boost"
    assert out3["multipliers"]["away_favorite"] == pytest.approx(1.2, abs=0.02)


def test_factor_below_20_applications_never_moves(monkeypatch, tmp_path):
    monkeypatch.setattr(wo, "FACTOR_ADJ_PATH", tmp_path / "fadj.json")
    out = wo.compute_factor_decay(_analysis(applied=19, accuracy=0.95))
    assert out["details"]["away_favorite"]["action"] == "insufficient_data"
    assert out["multipliers"]["away_favorite"] == 1.0
