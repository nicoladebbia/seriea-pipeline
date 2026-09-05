"""over_under_model: ledger-deployed blend weight (fail-soft) and data-derived team strengths."""
from __future__ import annotations

import json

import scripts.models.over_under_model as oum


def test_load_blend_weights_reads_deployed_override(tmp_path):
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"weights": {"2.5": 0.8, "1.5": "0.55"}, "fitted_at": "x"}))
    assert oum.load_blend_weights(p) == {"2.5": 0.8, "1.5": 0.55}


def test_load_blend_weights_ignores_out_of_range_and_garbage(tmp_path):
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"weights": {"2.5": 1.4, "1.5": "abc", "3.5": 0.0}}))
    assert oum.load_blend_weights(p) == {"3.5": 0.0}


def test_load_blend_weights_missing_or_corrupt_is_empty(tmp_path):
    assert oum.load_blend_weights(tmp_path / "nope.json") == {}
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    assert oum.load_blend_weights(p) == {}


def test_goal_prediction_audit_fields_default_empty():
    gp = oum.GoalPrediction(
        match="A vs B", home_team="A", away_team="B", date="2026-09-10",
        expected_home_goals=1.4, expected_away_goals=1.1, expected_total_goals=2.5,
        over_0_5=0.9, over_1_5=0.7, over_2_5=0.5, over_3_5=0.3, over_4_5=0.1,
        factors=[], confidence="LOW", confidence_rank=1,
        home_attack_strength=0.0, away_attack_strength=0.0,
        home_defense_strength=0.0, away_defense_strength=0.0,
    )
    assert gp.ou_ml == {} and gp.ou_poisson == {} and gp.ou_blend_weight == {}
    assert oum.DEFAULT_ML_BLEND_WEIGHT == 0.65


# ---------------------------------------------------------------------------
# team strengths — derived from data, hand-typed dict only as fallback
# ---------------------------------------------------------------------------

def _frame(rows):
    import pandas as pd
    return pd.DataFrame(rows, columns=["league", "season", "home_team", "away_team",
                                       "home_score", "away_score"])


def test_derive_team_strengths_matches_the_documented_formula():
    # A beat B 3-0, then drew 1-1 away: 5 goals over 2 matches → 1.25 per team per match
    m = _frame([("sa", "2025-2026", "A", "B", 3, 0), ("sa", "2025-2026", "B", "A", 1, 1)])
    s = oum.derive_team_strengths(m, min_matches=1)
    assert s["A"] == {"attack_mod": 0.6, "defense_mod": 0.6, "n": 2}     # 2/1.25−1, 1−0.5/1.25
    assert s["B"] == {"attack_mod": -0.6, "defense_mod": -0.6, "n": 2}


def test_derive_team_strengths_shrinks_thin_history_toward_zero():
    m = _frame([("sa", "2025-2026", "A", "B", 3, 0), ("sa", "2025-2026", "B", "A", 1, 1)])
    s = oum.derive_team_strengths(m, min_matches=10)                     # n=2 → ×0.2
    assert s["A"]["attack_mod"] == 0.12 and s["A"]["defense_mod"] == 0.12


def test_derive_team_strengths_windows_seasons_and_splits_leagues():
    m = _frame([
        ("sa", "2020-2021", "A", "B", 9, 0),      # ancient: outside the 3-season window
        ("sa", "2024-2025", "A", "B", 1, 1),
        ("sa", "2025-2026", "A", "B", 1, 1),
        ("sa", "2026-2027", "A", "B", 1, 1),
        ("pl", "2026-2027", "X", "Y", 4, 0),      # other league, its own average
    ])
    s = oum.derive_team_strengths(m, n_seasons=3, min_matches=1)
    assert s["A"] == {"attack_mod": 0.0, "defense_mod": 0.0, "n": 3}    # the 9-0 is excluded
    assert s["X"]["attack_mod"] == 1.0 and s["Y"]["defense_mod"] == -1.0  # judged vs pl avg 2.0


def test_live_table_falls_back_to_the_hand_typed_dict_when_parquet_unreadable(monkeypatch, tmp_path):
    monkeypatch.setattr(oum, "_TEAM_STRENGTHS_LIVE", None)
    monkeypatch.setattr(oum, "DATA_DIR", tmp_path)                       # no parsed/matches.parquet
    assert oum.get_team_strength("Inter") == oum.TEAM_STRENGTHS["Inter"]
    assert oum.get_team_strength("Nowhere FC") == {"attack_mod": 0.0, "defense_mod": 0.0}


def test_live_table_prefers_derived_values(monkeypatch, tmp_path):
    (tmp_path / "parsed").mkdir()
    _frame([("serie_a", "2026-2027", "Frosinone", "Genoa", 2, 0)] * 10).to_parquet(
        tmp_path / "parsed" / "matches.parquet")
    monkeypatch.setattr(oum, "_TEAM_STRENGTHS_LIVE", None)
    monkeypatch.setattr(oum, "DATA_DIR", tmp_path)
    fr = oum.get_team_strength("Frosinone")
    assert fr["attack_mod"] == 1.0 and fr["defense_mod"] == 1.0 and fr["n"] == 10
    assert "Frosinone" not in oum.TEAM_STRENGTHS                         # the typed table never knew it


# --- Step 19 without the pipeline ------------------------------------------


def test_ml_probs_from_predictions_reads_the_classifier_leg_and_skips_rows_without_it():
    import scripts.models.over_under_model as oum

    rows = [
        {"match": "Inter vs Udinese",
         "component_predictions": {"over_under_ml": {"1.5": 0.7873, "2.5": "0.5823"}}},
        {"match": "Lecce vs Monza", "component_predictions": {}},
        {"match": "Genoa vs Frosinone"},
        {"component_predictions": {"over_under_ml": {"2.5": 0.5}}},        # no match key
    ]
    assert oum.ml_probs_from_predictions(rows) == {
        "Inter vs Udinese": {1.5: 0.7873, 2.5: 0.5823},
    }
    assert oum.ml_probs_from_predictions([]) == {}
    assert oum.ml_probs_from_predictions(None) == {}


def test_refresh_from_predictions_feeds_the_blend_and_saves(tmp_path, monkeypatch):
    import json

    import scripts.models.over_under_model as oum

    src = tmp_path / "predictions.json"
    src.write_text(json.dumps({"predictions": [
        {"match": "Inter vs Udinese",
         "component_predictions": {"over_under_ml": {"1.5": 0.8, "2.5": 0.6}}},
    ]}))
    seen, saved = {}, []
    monkeypatch.setattr(oum, "generate_over_under_predictions",
                        lambda ml_ou_probs=None: (seen.setdefault("ml", ml_ou_probs), ["p"], ["b"])[1:])
    monkeypatch.setattr(oum, "save_over_under_predictions", lambda p, b: saved.append((p, b)))

    assert oum.refresh_from_predictions(src) == 1
    assert seen["ml"] == {"Inter vs Udinese": {1.5: 0.8, 2.5: 0.6}}
    assert saved == [(["p"], ["b"])]


def test_refresh_from_predictions_writes_nothing_when_there_is_nothing_to_serve(tmp_path, monkeypatch):
    import scripts.models.over_under_model as oum

    src = tmp_path / "predictions.json"
    src.write_text('{"predictions": []}')
    saved = []
    monkeypatch.setattr(oum, "generate_over_under_predictions", lambda ml_ou_probs=None: ([], []))
    monkeypatch.setattr(oum, "save_over_under_predictions", lambda p, b: saved.append(1))
    assert oum.refresh_from_predictions(src) == 0
    assert saved == []

