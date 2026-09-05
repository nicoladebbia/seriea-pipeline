"""E3: per-half split of the player floor markets (2026-09-05)."""
import math

from scripts.betting import player_predictions as pp


def test_half_minutes_follow_starter_or_sub_status():
    assert pp._half_minutes(82, True) == (45.0, 37.0)
    assert pp._half_minutes(30, True) == (30.0, 0.0)
    assert pp._half_minutes(20, False) == (0.0, 20.0)
    assert pp._half_minutes(60, False) == (15.0, 45.0)


def test_split_uses_the_timing_share_and_both_halves_is_a_product():
    s = pp._interval_split(lam90=2.0, k=1, dist="poisson", r=None, proj_minutes=90, is_starter=True, share_1h=0.461)
    lam1, lam2 = 2.0 * 0.5 * 2 * 0.461, 2.0 * 0.5 * 2 * 0.539
    assert math.isclose(s["exp_1h"], round(lam1, 3)) and math.isclose(s["exp_2h"], round(lam2, 3))
    assert math.isclose(s["1h"], round(1 - math.exp(-lam1), 4), abs_tol=1e-4)
    assert math.isclose(s["both"], round((1 - math.exp(-lam1)) * (1 - math.exp(-lam2)), 4), abs_tol=1e-4)
    assert s["timing"] == "measured"
    flat = pp._interval_split(2.0, 2, "poisson", None, 90, True, 0.5)
    assert flat["timing"] == "flat" and flat["both"] is None          # 'nei due tempi' only for the ≥1 lines
    sub = pp._interval_split(2.0, 1, "poisson", None, 20, False, 0.461)
    assert sub["exp_1h"] == 0.0 and sub["1h"] == 0.0


def test_contribution_pct_sums_to_100_per_side_and_fallback_players_get_none(monkeypatch):
    import pandas as pd
    # the fixture carries priors only; calibration / dispersion / possession read raw counts
    monkeypatch.setattr(pp, "_get_floor_calibration", lambda pms: {})
    monkeypatch.setattr(pp, "_get_dispersion", lambda pms: {})
    monkeypatch.setattr(pp, "_get_possession", lambda pms, league: (None, None))
    pms = pd.DataFrame({
        "player_id": [1] * 12 + [2] * 12, "player_name": ["A"] * 12 + ["B"] * 12, "team": ["T"] * 24,
        "opponent": ["U"] * 24, "date": pd.date_range("2025-01-01", periods=12).tolist() * 2,
        "minutes": [90] * 24, "is_starter": [True] * 24, "position": ["F"] * 24,
        **{f"{c}_p90_prior": ([3.0] * 12 + [1.0] * 12) for c in pp._RATE_COLS},
        "prior_n": [12] * 24, "min_prior": [85.0] * 24,
    })
    base_rates = {k: {"F": 0.3, "_overall": 0.3} for k in pp.TARGETS}
    lineup = [{"player_name": "A", "player_id": 1, "position": "F", "is_starter": True, "proj_minutes": 85},
              {"player_name": "B", "player_id": 2, "position": "F", "is_starter": True, "proj_minutes": 85},
              {"player_name": "C", "player_id": 3, "position": "F", "is_starter": True, "proj_minutes": 85}]
    res = pp.predict_match_players("T", "U", lineup, [], pms=pms, base_rates=base_rates)
    shares = [pl["markets"]["shots_o05"]["contribution_pct"] for pl in res["home_players"]]
    assert shares[2] is None and abs(shares[0] + shares[1] - 100.0) < 0.2 and shares[0] == 75.0
    assert res["home_players"][0]["markets"]["shots_o05"]["split"]["1h"] > 0
    assert res["home_players"][2]["markets"]["shots_o05"]["split"] is None


def test_validate_halves_grades_each_half_leak_free_and_gates_on_n(tmp_path):
    """Two starters with fixed priors, one train season and one test season: the
    per-half truth comes from the shot minutes, the base rate from the train
    season only, and n below the gate never passes."""
    import json

    import pandas as pd
    cols = {f"{c}_p90_prior": 0.0 for c in pp._RATE_COLS}
    rows = []
    for season, mids in (("2022-2023", [1, 2]), ("2023-2024", [3, 4])):
        for mid in mids:
            for pid, rate in ((10, 3.0), (11, 0.3)):
                rows.append({**cols, "match_id": mid, "player_id": pid, "season": season, "is_starter": True,
                             "prior_n": 12, "min_prior": 85.0, "total_shots_p90_prior": rate, "shots_on_target_p90_prior": rate / 3})
    pms = pd.DataFrame(rows)
    shots = pd.DataFrame({"match_id": ["3", "3", "4", "1"], "player_id": [10, 10, 10, 10],
                          "time": [10, 70, 20, 5], "shot_type": ["save", "miss", "goal", "miss"]})
    out = tmp_path / "halves.json"
    res = pp.validate_halves("serie_a", test_seasons=("2023-2024",), pms=pms, shots=shots, out_path=out,
                             shares={"total_shots": 0.461, "shots_on_target": 0.5}, min_test_rows=1)
    ph = res["per_half"]
    assert ph["shots_o05:1h"]["n"] == 4 and ph["shots_o05:1h"]["n_events"] == 2      # player 10 shot in 1H in both test matches
    assert ph["shots_o05:2h"]["n_events"] == 1 and ph["shots_o05:both"]["n_events"] == 1
    assert ph["shots_o05:1h"]["base_rate"] == 0.5                                   # 2 of 4 test rows; base from train = 1 of 4
    assert all(not v["passed"] for v in ph.values())                                 # n=4 is below the 200 gate
    assert json.loads(out.read_text())["gate"]["shots_o05"]["passed"] is False
