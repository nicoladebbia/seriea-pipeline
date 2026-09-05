"""Lessons subsystem: significance-gated generation, shadow grading, honest lifecycle."""
from __future__ import annotations

import json

import pytest

import scripts.pipeline.lesson_writer as lw


def _patch(monkeypatch, tmp_path):
    monkeypatch.setattr(lw, "FEEDBACK_DIR", tmp_path)
    monkeypatch.setattr(lw, "LESSONS_PATH", tmp_path / "lessons.json")


def _audit(team="Roma", n=13, bias=-0.91, se=0.35, mae=1.17):
    return {"per_team_calibration": {team: {
        "matches": n, "xg_mean_bias": bias, "xg_bias_se": se, "xg_mae": mae}},
        "confidence_bands": {}}


def _write_inputs(tmp_path, audit):
    (tmp_path / "prediction_audit.json").write_text(json.dumps(audit))
    (tmp_path / "roi_report.json").write_text(json.dumps({"by_market": {}}))


def test_xg_bias_needs_two_standard_errors(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    # |bias| 0.27 with se 0.30: under 2*se -> rejected (the finishing-luck case)
    _write_inputs(tmp_path, _audit(bias=-0.27, se=0.30))
    out = lw.generate_lessons()
    assert out["new_count"] == 0
    # true positive: |bias| 0.91 with se 0.35 clears 2*se -> lesson created, shrunk
    _write_inputs(tmp_path, _audit(bias=-0.91, se=0.35, n=13))
    out = lw.generate_lessons()
    assert out["new_count"] == 1
    lesson = json.loads(lw.LESSONS_PATH.read_text())["lessons"][0]
    # correction = -bias * n/(n+10) = 0.91 * 13/23 = 0.514 -> capped at 0.5
    assert lesson["correction"]["team_xg_adjust"] == 0.5
    assert "se 0.35" in lesson["evidence"]


def test_xg_bias_fails_closed_without_se_and_below_n_floor(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    _write_inputs(tmp_path, _audit(bias=-1.5, se=0.0))       # huge bias, no se
    assert lw.generate_lessons()["new_count"] == 0
    _write_inputs(tmp_path, _audit(bias=-1.5, se=0.2, n=7))  # n below floor
    assert lw.generate_lessons()["new_count"] == 0


def test_expired_scope_is_rederivable_low_help_rate_is_not(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    dead = {"lessons": [
        {"id": "lesson_001", "type": "xg_bias", "_scope_key": "xg_bias_Roma",
         "active": False, "deactivated_reason": "expired"},
        {"id": "lesson_002", "type": "xg_bias", "_scope_key": "xg_bias_Inter",
         "active": False, "deactivated_reason": "low help rate (0.0%)"},
    ], "metadata": {}}
    lw.LESSONS_PATH.write_text(json.dumps(dead))
    audit = {"per_team_calibration": {
        "Roma": {"matches": 13, "xg_mean_bias": -0.91, "xg_bias_se": 0.35, "xg_mae": 1.1},
        "Inter": {"matches": 13, "xg_mean_bias": -0.86, "xg_bias_se": 0.35, "xg_mae": 1.3},
    }, "confidence_bands": {}}
    _write_inputs(tmp_path, audit)
    out = lw.generate_lessons()
    led = json.loads(lw.LESSONS_PATH.read_text())["lessons"]
    new_scopes = [l["_scope_key"] for l in led if l.get("active")]
    assert new_scopes == ["xg_bias_Roma"]        # expired: reborn; failed: stays dead
    assert out["new_count"] == 1


def test_effectiveness_grades_shadow_vs_base_once_per_match(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    lw.LESSONS_PATH.write_text(json.dumps({"lessons": [
        {"id": "lesson_001", "active": True, "applied_count": 5}], "metadata": {}}))
    shadow = {"base": {"prob_H": 0.40, "prob_D": 0.30, "prob_A": 0.30},
              "adjusted": {"prob_H": 0.55, "prob_D": 0.25, "prob_A": 0.20},
              "ids": ["lesson_001"]}
    # HOME happened; adjusted was closer -> helped
    lw.update_lesson_effectiveness("Roma vs Lazio", "HOME", "HOME",
                                   ["lesson_001"], shadow=shadow)
    l = json.loads(lw.LESSONS_PATH.read_text())["lessons"][0]
    assert l["helped_count"] == 1 and l["graded_count"] == 1
    # the learning loop re-feeds every settled match every run: no double count
    lw.update_lesson_effectiveness("Roma vs Lazio", "HOME", "HOME",
                                   ["lesson_001"], shadow=shadow)
    l = json.loads(lw.LESSONS_PATH.read_text())["lessons"][0]
    assert l["helped_count"] == 1 and l["graded_count"] == 1
    # AWAY happened; adjusted was further -> graded but not helped
    lw.update_lesson_effectiveness("Como vs Roma", "HOME", "AWAY",
                                   ["lesson_001"], shadow=shadow)
    l = json.loads(lw.LESSONS_PATH.read_text())["lessons"][0]
    assert l["helped_count"] == 1 and l["graded_count"] == 2


def test_deactivation_reviews_graded_not_applied(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    lw.LESSONS_PATH.write_text(json.dumps({"lessons": [
        # 40 applications, ZERO graded (the historical broken-tracking shape):
        # must NOT be executed as 0% helpful
        {"id": "lesson_001", "type": "xg_bias", "_scope_key": "a", "active": True,
         "applied_count": 40, "graded_count": 0, "helped_count": 0,
         "expires_after_n": 60},
        # genuinely graded and failing: must be deactivated
        {"id": "lesson_002", "type": "xg_bias", "_scope_key": "b", "active": True,
         "applied_count": 12, "graded_count": 12, "helped_count": 1,
         "expires_after_n": 60},
    ], "metadata": {}}))
    _write_inputs(tmp_path, {"per_team_calibration": {}, "confidence_bands": {}})
    lw.generate_lessons()
    led = {l["id"]: l for l in json.loads(lw.LESSONS_PATH.read_text())["lessons"]}
    assert led["lesson_001"]["active"] is True
    assert led["lesson_002"]["active"] is False
    assert "low help rate" in led["lesson_002"]["deactivated_reason"]


def test_apply_lessons_counts_distinct_matches_and_never_mutates_input():
    from scripts.prediction.ensemble_prediction_engine import EnsemblePredictor
    eng = EnsemblePredictor(live_mode=False)
    eng._lessons_data = {"lessons": [
        {"id": "lesson_001", "type": "xg_bias", "active": True,
         "scope": {"team": "Roma"}, "correction": {"team_xg_adjust": 0.4},
         "confidence": 0.9, "applied_count": 0, "expires_after_n": 20}]}
    probs = {"prob_H": 0.40, "prob_D": 0.30, "prob_A": 0.30}
    frozen = dict(probs)
    out1, ids1 = eng._apply_lessons(dict(probs), "Roma", "Lazio", "HOME",
                                    1.3, 1.1, match_key="2026-09-10_Roma_Lazio")
    assert ids1 == ["lesson_001"]
    assert probs == frozen                               # input untouched
    assert out1["prob_H"] > 0.40                          # +xG raised P(H)
    assert abs(sum(out1[k] for k in ("prob_H", "prob_D", "prob_A")) - 1) < 1e-6
    # same match re-predicted (the 2-3x/day regeneration): count stays 1
    eng._apply_lessons(dict(probs), "Roma", "Lazio", "HOME", 1.3, 1.1,
                       match_key="2026-09-10_Roma_Lazio")
    assert eng._lessons_data["lessons"][0]["applied_count"] == 1
    # a different match: count advances
    eng._apply_lessons(dict(probs), "Roma", "Como", "HOME", 1.3, 1.1,
                       match_key="2026-09-17_Roma_Como")
    assert eng._lessons_data["lessons"][0]["applied_count"] == 2


def test_archive_copies_lesson_fields(monkeypatch, tmp_path):
    import scripts.analysis.performance_dashboard as pdash
    monkeypatch.setattr(pdash, "PREDICTIONS_PATH", tmp_path / "preds.json")
    monkeypatch.setattr(pdash, "PREDICTIONS_ARCHIVE", tmp_path / "arch.json")
    shadow = {"base": {"prob_H": 0.4}, "adjusted": {"prob_H": 0.5},
              "ids": ["lesson_001"]}
    (tmp_path / "preds.json").write_text(json.dumps({"predictions": [{
        "match": "Roma vs Lazio", "home_team": "Roma", "away_team": "Lazio",
        "date": "2026-09-10", "predicted_outcome": "HOME",
        "lessons_applied": ["lesson_001"], "lessons_shadow": shadow}]}))
    assert pdash.archive_predictions() == 1
    row = json.loads((tmp_path / "arch.json").read_text())["Roma vs Lazio_2026-09-10"]
    assert row["lessons_applied"] == ["lesson_001"]
    assert row["lessons_shadow"] == shadow
