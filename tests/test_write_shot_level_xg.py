"""The shot-level transform must reproduce all_shots_with_xg.parquet exactly.

Every mapping asserted here was reverse-engineered from the live file
(groupby-situation / groupby-body_part means, and the distance/angle geometry
checked against a stored row) before the writer was allowed to touch data. The
one non-obvious case — ``throw-in-set-piece`` mapping to is_set_piece=0 — is
pinned because a "reasonable" guess would get it wrong.
"""
from __future__ import annotations

from scripts.data.write_shot_level_xg import shot_rows_from_shotmap


def _shot(**over):
    base = {
        "isHome": True,
        "player": {"id": 123, "name": "Test Player"},
        "playerCoordinates": {"x": 8.3, "y": 51.5, "z": 0},
        "goalMouthCoordinates": {"x": 0, "y": 46.6, "z": 19},
        "situation": "regular",
        "bodyPart": "right-foot",
        "shotType": "miss",
        "xg": 0.11,
        "xgot": None,
        "time": 42,
    }
    base.update(over)
    return base


def _one(**over):
    return shot_rows_from_shotmap([_shot(**over)], "13981650", "2025-2026")[0]


def test_distance_and_angle_match_the_stored_geometry():
    """x=8.3, y=51.5 -> distance 8.434, angle 10.244 in the real file."""
    r = _one()
    assert round(r["distance"], 3) == 8.434
    assert round(r["angle"], 3) == 10.244


def test_missing_coordinates_give_null_geometry_not_a_crash():
    r = _one(playerCoordinates={})
    assert r["distance"] is None and r["angle"] is None


def test_set_piece_is_corner_and_set_piece_only():
    assert _one(situation="corner")["is_set_piece"] == 1
    assert _one(situation="set-piece")["is_set_piece"] == 1
    # the trap: throw-in-set-piece maps to 0 in the real file
    assert _one(situation="throw-in-set-piece")["is_set_piece"] == 0
    assert _one(situation="regular")["is_set_piece"] == 0


def test_situation_one_hots_are_mutually_exclusive_literals():
    pen = _one(situation="penalty")
    assert (pen["is_penalty"], pen["is_freekick"], pen["is_set_piece"], pen["is_fast_break"]) == (1, 0, 0, 0)
    fk = _one(situation="free-kick")
    assert (fk["is_penalty"], fk["is_freekick"]) == (0, 1)
    fb = _one(situation="fast-break")
    assert (fb["is_fast_break"], fb["is_set_piece"]) == (1, 0)


def test_body_part_one_hots():
    assert (_one(bodyPart="head")["is_header"], _one(bodyPart="head")["is_right"]) == (1, 0)
    assert _one(bodyPart="left-foot")["is_left"] == 1
    assert _one(bodyPart="right-foot")["is_right"] == 1
    other = _one(bodyPart="other")
    assert (other["is_header"], other["is_right"], other["is_left"]) == (0, 0, 0)


def test_is_goal_only_for_goal_shot_type():
    assert _one(shotType="goal")["is_goal"] == 1
    for st in ("miss", "save", "block", "post"):
        assert _one(shotType=st)["is_goal"] == 0


def test_xg_predicted_equals_observed_xg_when_present():
    """For seasons Sofascore serves xg, the fallback equals the observed value —
    it is never read, and we do not fabricate a model output for it.
    """
    r = _one(xg=0.37)
    assert r["xg"] == 0.37 and r["xg_predicted"] == 0.37


def test_match_id_is_the_passed_sofascore_id_not_anything_in_the_payload():
    r = _one()
    assert r["match_id"] == "13981650"


def test_coordinates_map_to_the_right_columns():
    r = _one(playerCoordinates={"x": 9.6, "y": 42.9, "z": 0},
             goalMouthCoordinates={"x": 0, "y": 46.6, "z": 19})
    assert (r["shot_x"], r["shot_y"]) == (9.6, 42.9)
    assert (r["gm_x"], r["gm_y"], r["gm_z"]) == (0, 46.6, 19)


def test_blocked_shot_with_no_goalmouth_yields_null_gm_not_a_crash():
    r = _one(goalMouthCoordinates={})
    assert r["gm_x"] is None and r["gm_y"] is None


def test_shot_level_loader_serves_both_leagues_and_tolerates_a_missing_sibling(tmp_path, monkeypatch):
    """Until 2026-09-06 every shot-feature reader opened the Serie A file only, so
    the EPL frame had no shot-level columns. The shared loader concatenates the
    two files (Sofascore ids are disjoint across leagues), passes ``columns``
    through, and a missing sibling is skipped rather than fatal."""
    import pandas as pd

    import features._utils as u
    sa = tmp_path / "all_shots_with_xg.parquet"
    epl = tmp_path / "all_shots_with_xg_premier_league.parquet"
    pd.DataFrame({"match_id": [1, 1], "xg": [0.1, 0.2], "is_penalty": [0, 1]}).to_parquet(sa, index=False)
    monkeypatch.setattr(u, "SHOT_LEVEL_XG_PATHS", (sa, epl))
    only_sa = u.load_shot_level_xg()
    assert len(only_sa) == 2 and set(only_sa["match_id"]) == {1}
    pd.DataFrame({"match_id": [2], "xg": [0.5], "is_penalty": [0]}).to_parquet(epl, index=False)
    both = u.load_shot_level_xg(columns=["match_id", "xg"])
    assert len(both) == 3 and set(both["match_id"]) == {1, 2} and list(both.columns) == ["match_id", "xg"]
    monkeypatch.setattr(u, "SHOT_LEVEL_XG_PATHS", (tmp_path / "nope.parquet",))
    assert u.load_shot_level_xg() is None
