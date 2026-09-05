"""print_model_status: the O/U models (the ones that place bets) are the headline."""
from __future__ import annotations

import json
import math
from datetime import UTC, datetime

import scripts.diagnostics.print_model_status as pms


def _ou_meta(line=2.5, with_promotion=True):
    m = {
        "line": line, "n_features": 37, "n_training_rows": 6880, "base_rate": 0.5318,
        "cv_metrics": {"overall_log_loss": 0.6568, "overall_brier": 0.2326,
                       "overall_accuracy": 0.5921, "overall_calibration_gap": 0.0433},
        "eval_metrics": {"log_loss": 0.618, "brier": 0.2143, "calibration_gap": 0.0305},
        "quality_gates": {"log_loss_pass": True, "brier_pass": True, "calibration_pass": False},
        "trained_at": "2026-09-04T20:19:32+00:00",
    }
    if with_promotion:
        m["promotion"] = {
            "promoted": True, "decided_at": "2026-09-05T00:19:32+00:00",
            "reason": "better than incumbent by 0.0081 log-loss",
            "holdout": {"n": 1033, "dates": ["2025-02-27", "2026-09-04"],
                        "naive_log_loss": 0.6975,
                        "incumbent": {"log_loss": 0.6261, "calibration_gap": 0.0515}},
        }
    return m


def test_naive_baselines_match_the_entropy_formula():
    ll, brier = pms.naive_baselines(0.5318)
    b = 0.5318
    assert math.isclose(ll, -(b * math.log(b) + (1 - b) * math.log(1 - b)), abs_tol=1e-4)
    assert math.isclose(brier, b * (1 - b), abs_tol=1e-4)


def test_ou_line_status_reads_promotion_and_marks_bet_lines():
    s = pms.ou_line_status(_ou_meta(2.5))
    assert s["bet"] is True and s["trained_at"] == "2026-09-04"
    assert s["promotion"]["promoted"] is True
    assert s["holdout"]["incumbent"]["log_loss"] == 0.6261
    assert s["cv"]["naive_log_loss"] > s["cv"]["log_loss"]      # the model beats naive
    assert pms.ou_line_status(_ou_meta(3.5))["bet"] is False


def test_pre_gate_metadata_is_flagged_not_hidden():
    s = pms.ou_line_status(_ou_meta(1.5, with_promotion=False))
    assert s["promotion"] is None
    text = "\n".join(pms.render_ou([s], None, ["O/U_Over"]))
    assert "UNCONDITIONALLY" in text


def test_journal_edge_aggregates_settled_ou_bets_only():
    now = datetime(2026, 9, 4, tzinfo=UTC)
    journal = {"bets": {
        "a": {"market": "O/U 1.5", "status": "won", "stake": 10, "profit": 5, "clv_pct": 2.0,
              "placed_at": "2026-08-30T10:00"},
        "b": {"market": "O/U 1.5", "status": "lost", "stake": 10, "profit": -10, "clv_pct": -1.0,
              "placed_at": "2026-05-01T10:00"},
        "c": {"market": "O/U 2.5", "status": "superseded", "stake": 10, "profit": 0,
              "placed_at": "2026-08-31T10:00"},          # unsettled: counted as recent only
        "d": {"market": "1X2", "status": "won", "stake": 10, "profit": 9, "clv_pct": 3.0,
              "placed_at": "2026-08-30T10:00"},          # wrong market prefix
    }}
    e = pms.journal_edge(journal, now=now)
    assert set(e["markets"]) == {"O/U 1.5"}
    r = e["markets"]["O/U 1.5"]
    assert r["n"] == 2 and r["roi"] == -25.0 and r["mean_clv"] == 0.5 and r["clv_pos_share"] == 0.5
    assert r["last_placed"] == "2026-08-30"
    assert e["recent_placed"] == 2                       # a + c within 30 days, b too old


def test_report_puts_ou_before_1x2_and_survives_a_missing_1x2_file(monkeypatch, tmp_path, capsys):
    ou = tmp_path / "over_under"
    ou.mkdir()
    (ou / "ou_2_5_catboost_metadata.json").write_text(json.dumps(_ou_meta(2.5)))
    monkeypatch.setattr(pms, "OU_DIR", ou)
    monkeypatch.setattr(pms, "METADATA", tmp_path / "missing_1x2.json")
    monkeypatch.setattr(pms, "JOURNAL", tmp_path / "missing_journal.json")
    monkeypatch.setattr(pms, "enabled_markets", lambda: ["Alt_OU", "O/U_Over"])
    assert pms.main() == 0
    out = capsys.readouterr().out
    assert out.index("PRIMARY") < out.index("SECONDARY") if "SECONDARY" in out else True
    assert "PRIMARY" in out and "PROMOTED" in out and "1X2 section skipped" in out
    assert "Alt_OU, O/U_Over" in out


def test_report_exits_nonzero_only_when_nothing_is_readable(monkeypatch, tmp_path):
    monkeypatch.setattr(pms, "OU_DIR", tmp_path / "nope")
    monkeypatch.setattr(pms, "METADATA", tmp_path / "nope.json")
    monkeypatch.setattr(pms, "JOURNAL", tmp_path / "nope2.json")
    assert pms.main() == 1
