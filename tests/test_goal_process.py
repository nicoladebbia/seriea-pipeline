"""Goal-process simulator (design step 2): minute-resolved goal paths from which
every lead / time-interval market is read. Tests pin the data contract
(timeline bins, own-goal side, 0-0 matches kept), the simulator's invariants,
and that the SAME market functions grade a real path and a simulated one.
"""
import json

import numpy as np
import pandas as pd

from scripts.models import goal_process as gp


# ---------------------------------------------------------------- timeline --
def test_bin_of_maps_regular_and_stoppage_minutes():
    assert gp.bin_of(1, 0) == 0 and gp.bin_of(45, 0) == 44
    assert gp.bin_of(45, 3) == gp.BIN_1H_STOPPAGE            # 45+3'
    assert gp.bin_of(46, 0) == 46 and gp.bin_of(90, 0) == 90
    assert gp.bin_of(90, 5) == gp.BIN_2H_STOPPAGE            # 90+5'
    assert gp.N_BINS == 92 and gp.half_of_bin(gp.BIN_1H_STOPPAGE) == 1 and gp.half_of_bin(46) == 2


def _incidents():
    return pd.DataFrame({
        "match_id": [1, 1, 1, 2, 3, 3],
        "incident_type": ["goal", "goal", "card", "substitution", "goal", "goal"],
        "minute": [10, 45, 60, 70, 90, 12],
        "added_time": [0, 2, 0, 0, 4, 0],
        "is_home": [True, False, True, True, False, True],
        "goal_type": ["regular", "ownGoal", None, None, "penalty", "regular"],
    })


def _mapping():
    return pd.DataFrame({"match_id": ["2024-01-01_A_B", "2024-01-02_C_D", "2024-01-03_E_F"],
                         "sofascore_id": [1, 2, 3], "season": ["2023-2024"] * 3})


def test_timeline_keeps_goal_side_as_recorded_and_keeps_goalless_matches():
    tl, universe = gp.timeline_from_frames(_incidents(), _mapping())
    # own goal at 45+2 is credited to the side Sofascore recorded (verified 99.8% vs scores)
    og = tl[(tl.match_id == "2024-01-01_A_B") & (tl.minute == 45)].iloc[0]
    assert og.side == "away" and og.bin == gp.BIN_1H_STOPPAGE and og.half == 1
    assert set(universe) == {"2024-01-01_A_B", "2024-01-02_C_D", "2024-01-03_E_F"}  # match 2 had no goal
    assert (tl.match_id == "2024-01-02_C_D").sum() == 0
    assert tl.loc[tl.match_id == "2024-01-03_E_F", "bin"].tolist() == [gp.BIN_2H_STOPPAGE, 11]


def test_build_timeline_is_idempotent_and_watermarked(tmp_path, monkeypatch):
    inc_p, map_p, out_p = tmp_path / "inc.parquet", tmp_path / "map.parquet", tmp_path / "tl.parquet"
    _incidents().to_parquet(inc_p)
    _mapping().to_parquet(map_p)
    monkeypatch.setattr(gp, "INCIDENTS_PATH", inc_p)
    monkeypatch.setattr(gp, "MAPPING_PATH", map_p)
    monkeypatch.setattr(gp, "TIMELINE_PATH", out_p)
    tl1, rebuilt1 = gp.build_goal_timeline()
    tl2, rebuilt2 = gp.build_goal_timeline()
    assert rebuilt1 is True and rebuilt2 is False
    pd.testing.assert_frame_equal(tl1.reset_index(drop=True), tl2.reset_index(drop=True))
    # source moved → rebuild
    import os
    import time
    os.utime(inc_p, (time.time() + 5, time.time() + 5))
    _, rebuilt3 = gp.build_goal_timeline()
    assert rebuilt3 is True


# ------------------------------------------------------------- paths/markets --
def test_real_path_and_market_outcomes():
    tl, _ = gp.timeline_from_frames(_incidents(), _mapping())
    paths = gp.paths_from_timeline(tl, ["2024-01-01_A_B"])
    assert paths["home_final"].tolist() == [1] and paths["away_final"].tolist() == [1]
    assert paths["ht_home"].tolist() == [1] and paths["ht_away"].tolist() == [1]  # 45+2 OG is 1st half
    assert paths["max_lead_home"].tolist() == [1] and paths["max_lead_away"].tolist() == [0]
    out = gp.market_outcomes(paths)
    assert out["vince_o_quasi_home_1"].tolist() == [1.0]   # led by 1 at some point
    assert out["vince_o_quasi_home_2"].tolist() == [0.0]
    assert out["home_win"].tolist() == [0.0] and out["draw"].tolist() == [1.0]
    assert out["goal_0_15"].tolist() == [1.0] and out["goal_76_90"].tolist() == [0.0]
    assert out["ht_home"].tolist() == [0.0] and out["ht_draw"].tolist() == [1.0]


def test_simulator_invariants():
    prof = gp.default_profile()
    z = gp.simulate(0.0, 0.0, prof, n=500, seed=1)
    assert z["home_final"].sum() == 0 and z["away_final"].sum() == 0
    a = gp.simulate(1.5, 1.0, prof, n=2000, seed=7)
    b = gp.simulate(1.5, 1.0, prof, n=2000, seed=7)
    assert np.array_equal(a["home_final"], b["home_final"])
    p = gp.market_probs(a)
    assert p["vince_o_quasi_home_1"] >= p["home_win"] >= p["vince_o_quasi_home_2"] - 1e-9 or p["vince_o_quasi_home_1"] >= p["home_win"]
    assert abs(p["home_win"] + p["draw"] + p["away_win"] - 1) < 1e-9
    assert 1.3 < a["home_final"].mean() < 1.7 and 0.85 < a["away_final"].mean() < 1.15


def test_calibration_hits_the_served_over_2_5():
    prof = gp.default_profile()
    s = gp.simulate(1.14, 0.91, prof, n=20000, seed=3, p_over_2_5=0.392)
    assert abs(float((s["home_final"] + s["away_final"] >= 3).mean()) - 0.392) < 0.012
    # the profile total (2.65 default) is above what 0.392 implies, so k pulls DOWN and moves
    assert 0.5 < s["calibration_k"] < 1.0


def test_fit_profile_from_timeline_is_normalised():
    tl, universe = gp.timeline_from_frames(_incidents(), _mapping())
    prof = gp.fit_profile(tl, universe)
    assert abs(sum(prof["hazard"]) - 1.0) < 1e-9 and len(prof["hazard"]) == gp.N_BINS
    assert set(prof["state_mult"]) == {"home", "away"} and set(prof["state_mult"]["home"]) == {"trail", "level", "lead"}
    assert prof["state_mult"]["home"]["level"] == 1.0


def test_served_rows_take_tiers_from_the_live_backtest(tmp_path, monkeypatch):
    """Tier A only where backtest.json says the gate passed; complements inherit;
    1x2 finale and over 2.5 are never served from the simulator."""
    bt = tmp_path / "backtest.json"
    bt.write_text(json.dumps({"gate": {"vince_o_quasi_home_1": {"passed": True, "skill": 0.06},
                                       "goal_both_halves": {"passed": False, "skill": 0.007}}}))
    monkeypatch.setattr(gp, "BACKTEST_PATH", bt)
    monkeypatch.setattr(gp, "PROFILE_PATH", tmp_path / "none.json")
    gp._SERVED_CACHE.clear()
    rows = gp.served_rows(1.3, 1.1, 0.5, n=3000)
    by = {(r["bet_type"], r["selection"]): r for r in rows}
    assert by[("Vince o quasi", "Casa 1x sì")]["tier"] == "A"
    assert by[("Gol in entrambi i tempi", "Sì")]["tier"] == "B" and by[("Gol in entrambi i tempi", "No")]["tier"] == "B"
    assert abs(by[("Gol in entrambi i tempi", "Sì")]["probability_pct"] + by[("Gol in entrambi i tempi", "No")]["probability_pct"] - 100) < 0.11
    assert ("1x2 finale", "1") not in by and ("Under/over", "Over 2.5") not in by
    assert gp.served_rows(0, 1.1, 0.5) == []


def test_served_rows_refuse_a_league_the_gate_never_measured():
    gp._SERVED_CACHE.clear()
    assert gp.served_rows(1.3, 1.1, 0.5, n=500, league="premier_league") == []


def test_rare_event_rates_count_matches_not_events():
    """Two own goals in one match is ONE match with an own goal; a bench goal
    needs the scorer to have come on BEFORE the goal; shots decide box/halfway."""
    inc = pd.DataFrame({
        "match_id": [1, 1, 1, 1, 2, 2],
        "incident_type": ["goal", "goal", "substitution", "goal", "card", "goal"],
        "incident_class": ["ownGoal", "ownGoal", None, "regular", "red", "regular"],
        "minute": [10, 20, 60, 70, 30, 1], "added_time": [0] * 6,
        "player_id": ["9", "9", "", "77", "5", "8"], "player_in_id": ["", "", "77", "", "", ""],
        "is_home": [True] * 6, "card_type": [None, None, None, None, "red", None],
        "goal_type": ["ownGoal", "ownGoal", None, "regular", None, "regular"],
    })
    mp = pd.DataFrame({"match_id": ["2024-01-01_A_B", "2024-01-02_C_D"], "sofascore_id": [1, 2],
                       "season": ["2023-2024"] * 2, "league": ["serie_a"] * 2})
    shots = pd.DataFrame({"match_id": ["2024-01-01_A_B", "2024-01-01_A_B", "2024-01-02_C_D"],
                          "shot_x": [10.0, 30.0, 60.0], "shot_y": [50.0, 50.0, 50.0],
                          "is_goal": [True, True, True], "is_penalty": [True, False, False]})
    r = gp.rare_event_rates(inc, mp, shots, "serie_a")
    assert r["own_goal"] == {"rate": 0.5, "n_matches": 2, "n_events": 1}
    assert r["goal_from_bench"]["n_events"] == 1          # player 77 on at 60', scored at 70'
    assert r["goal_minute_1"]["n_events"] == 1 and r["red_card"]["n_events"] == 1
    assert r["goal_outside_box"]["n_events"] == 2         # x=30 (31.5 m) and x=60; x=10 (10.5 m) is inside
    assert r["goal_beyond_halfway"]["n_events"] == 1      # x=60 → 63 m
    assert r["penalty_awarded"]["n_events"] == 1
    assert gp.rare_event_rates(inc, mp, shots, "premier_league") == {}


def test_rare_event_rates_accept_shots_keyed_on_sofascore_id():
    """all_shots_with_xg.parquet keys on the Sofascore id as a string (measured
    2026-09-05: 0% canonical, 100% sofascore_id); the first live run returned
    0.0 for every shot-based rate because of it."""
    inc = pd.DataFrame({"match_id": [1], "incident_type": ["goal"], "incident_class": ["regular"], "minute": [5],
                        "added_time": [0], "player_id": ["1"], "player_in_id": [""], "is_home": [True],
                        "card_type": [None], "goal_type": ["regular"]})
    mp = pd.DataFrame({"match_id": ["2024-01-01_A_B"], "sofascore_id": [1], "season": ["2023-2024"], "league": ["serie_a"]})
    shots = pd.DataFrame({"match_id": ["1"], "shot_x": [30.0], "shot_y": [50.0], "is_goal": [True], "is_penalty": [False]})
    assert gp.rare_event_rates(inc, mp, shots, "serie_a")["goal_outside_box"]["n_events"] == 1


def test_calibration_outside_reachable_range_is_flagged_not_silent():
    prof = gp.default_profile()
    s = gp.simulate(1.2, 1.0, prof, n=4000, seed=1, p_over_2_5=0.001)
    assert s["calibration_saturated"] is True and s["calibration_k"] == gp.K_BOUNDS[0]
    assert s["calibration_achieved"] > 0.001            # what was actually served, on the record
    ok = gp.simulate(1.2, 1.0, prof, n=4000, seed=1, p_over_2_5=0.45)
    assert ok["calibration_saturated"] is False and abs(ok["calibration_achieved"] - 0.45) < 0.02
    gp._SERVED_CACHE.clear()
    rows = gp.served_rows(1.2, 1.0, 0.001, n=2000)
    assert rows and all(r["calibration_saturated"] for r in rows) and all("key" in r for r in rows)


def test_var_rates_count_only_checked_matches_and_read_the_overturn_semantics():
    """incident_class = the on-field decision, confirmed=False = overturned by VAR
    (verified on disk 2026-09-05). Unchecked matches are not 'no VAR'."""
    inc = pd.DataFrame({
        "match_id": [1, 1, 2, 3, 4],
        "incident_type": ["varDecision", "goal", "var_checked", "goal", "inGamePenalty"],
        "incident_class": ["goalAwarded", "regular", "", "regular", "missed"],
        "minute": [23, 51, 0, 10, 70], "added_time": [0] * 5,
        "player_id": ["9", "8", "", "7", "6"], "player_in_id": [""] * 5, "is_home": [True] * 5,
        "card_type": [None] * 5, "goal_type": [None, "regular", None, "regular", None],
        "confirmed": [False, None, None, None, None],
    })
    mp = pd.DataFrame({"match_id": ["m1", "m2", "m3", "m4"], "sofascore_id": [1, 2, 3, 4],
                       "season": ["2025-2026"] * 4, "league": ["serie_a"] * 4})
    r = gp.rare_event_rates(inc, mp, None, "serie_a")
    # match 3 unchecked → excluded; match 4 fetched (missed penalty, no VAR) → counted as checked
    assert r["var_any"] == {"rate": round(1 / 3, 4), "n_matches": 3, "n_events": 1}
    assert r["var_goal_cancelled"]["n_events"] == 1 and r["var_penalty"]["n_events"] == 0
    assert r["own_goal"]["n_matches"] == 4                                    # non-VAR rates keep the full universe


def test_calibration_saturates_at_the_upper_bound_too_and_warns(caplog):
    import logging
    prof = gp.default_profile()
    with caplog.at_level(logging.WARNING, logger="scripts.models.goal_process"):
        s = gp.simulate(1.2, 1.0, prof, n=4000, seed=1, p_over_2_5=0.999)
    assert s["calibration_saturated"] is True and s["calibration_k"] == gp.K_BOUNDS[1]
    assert s["calibration_achieved"] < 0.999
    assert any("calibration saturated" in r.message for r in caplog.records)
