"""Tests for the World Cup 2026 prediction package (pure logic, no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.worldcup.engine import (
    ELO_HOME_ADV,
    INITIAL_ELO,
    K_CONTINENTAL_FINAL,
    K_FRIENDLY,
    K_QUALIFIER_MAJOR,
    K_WORLD_CUP,
    GoalModel,
    canon_team,
    elo_history,
    expected_score,
    goal_multiplier,
    k_factor,
    one_x_two,
    score_matrix,
)
from scripts.worldcup.players import (
    _squad_candidates,
    build_shares,
    normalize_name,
    p_anytime,
)
from scripts.worldcup.simulate import (
    _rank_group_2026,
    country_of_city,
    parse_slot,
)


class TestEloMath:
    def test_expected_score_symmetry(self) -> None:
        assert expected_score(0.0) == pytest.approx(0.5)
        assert expected_score(200.0) + expected_score(-200.0) == pytest.approx(1.0)

    def test_expected_score_monotone(self) -> None:
        diffs = [-400.0, -100.0, 0.0, 100.0, 400.0]
        scores = [expected_score(d) for d in diffs]
        assert scores == sorted(scores)

    def test_goal_multiplier_table(self) -> None:
        assert goal_multiplier(0) == 1.0
        assert goal_multiplier(1) == 1.0
        assert goal_multiplier(-1) == 1.0
        assert goal_multiplier(2) == 1.5
        assert goal_multiplier(3) == pytest.approx(14.0 / 8.0)
        assert goal_multiplier(5) == pytest.approx(16.0 / 8.0)

    def test_k_factor_classes(self) -> None:
        assert k_factor("FIFA World Cup") == K_WORLD_CUP
        assert k_factor("FIFA World Cup qualification") == K_QUALIFIER_MAJOR
        assert k_factor("UEFA Euro") == K_CONTINENTAL_FINAL
        assert k_factor("Copa América") == K_CONTINENTAL_FINAL
        assert k_factor("UEFA Euro qualification") == K_QUALIFIER_MAJOR
        assert k_factor("UEFA Nations League") == K_QUALIFIER_MAJOR
        assert k_factor("Friendly") == K_FRIENDLY


def _toy_matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "home_team": ["A", "B", "A"],
            "away_team": ["B", "C", "C"],
            "home_score": [2, 0, 1],
            "away_score": [0, 0, 1],
            "tournament": ["Friendly", "Friendly", "Friendly"],
            "city": ["x", "x", "x"],
            "country": ["x", "x", "x"],
            "neutral": [True, True, True],
        }
    )


class TestEloHistory:
    def test_zero_sum_and_prematch_features(self) -> None:
        hist, ratings = elo_history(_toy_matches())
        # First match: both sides start at the initial rating.
        assert hist.loc[0, "elo_home_pre"] == INITIAL_ELO
        assert hist.loc[0, "elo_away_pre"] == INITIAL_ELO
        # Elo is zero-sum: total rating mass is conserved.
        assert sum(ratings.values()) == pytest.approx(INITIAL_ELO * 3)
        # A beat B, so A > initial > B after match 1; A stays above B at the end.
        assert ratings["A"] > ratings["B"]

    def test_winner_gains(self) -> None:
        hist, ratings = elo_history(_toy_matches())
        assert ratings["A"] > INITIAL_ELO
        # Pre-match rating of A in its second match reflects the first win.
        assert hist.loc[2, "elo_home_pre"] > INITIAL_ELO

    def test_home_advantage_in_expectancy(self) -> None:
        # Equal-rated teams, non-neutral: home side is favored in expectancy,
        # so a home win moves ratings LESS than an away win would.
        base = _toy_matches().iloc[:1].copy()
        home_win = base.copy()
        home_win["neutral"] = [False]
        _, r_non_neutral = elo_history(home_win)
        _, r_neutral = elo_history(base)
        gain_non_neutral = r_non_neutral["A"] - INITIAL_ELO
        gain_neutral = r_neutral["A"] - INITIAL_ELO
        assert 0 < gain_non_neutral < gain_neutral
        assert expected_score(ELO_HOME_ADV) > 0.5


class TestGoalModel:
    def test_lam_monotone_in_rating(self) -> None:
        m = GoalModel(b0=0.1, b_diff=0.15, b_home=0.25, b_friendly=0.05)
        weak_vs_strong = m.lam(1400, 1900, at_home=False)
        even = m.lam(1700, 1700, at_home=False)
        strong_vs_weak = m.lam(1900, 1400, at_home=False)
        assert weak_vs_strong < even < strong_vs_weak

    def test_home_term(self) -> None:
        m = GoalModel(b0=0.1, b_diff=0.15, b_home=0.25, b_friendly=0.05)
        assert m.lam(1700, 1700, at_home=True) > m.lam(1700, 1700, at_home=False)


class TestScoreMatrix:
    def test_sums_to_one(self) -> None:
        grid = score_matrix(1.4, 1.1)
        assert grid.sum() == pytest.approx(1.0)

    def test_favorite_more_likely_to_win(self) -> None:
        grid = score_matrix(2.0, 0.8)
        p_home = np.tril(grid, -1).sum()  # home goals > away goals
        p_away = np.triu(grid, 1).sum()
        p_draw = np.trace(grid)
        assert p_home > p_away
        assert p_home + p_away + p_draw == pytest.approx(1.0)

    def test_mean_goals_close_to_lambda(self) -> None:
        grid = score_matrix(1.5, 1.2)
        goals = np.arange(grid.shape[0])
        mean_home = float((grid.sum(axis=1) * goals).sum())
        mean_away = float((grid.sum(axis=0) * goals).sum())
        assert mean_home == pytest.approx(1.5, abs=0.01)
        assert mean_away == pytest.approx(1.2, abs=0.01)

    def test_one_x_two_sums_to_one(self) -> None:
        p_home, p_draw, p_away = one_x_two(1.8, 0.9)
        assert p_home + p_draw + p_away == pytest.approx(1.0)
        assert p_home > p_away


class TestTeamNames:
    def test_fixture_display_names_map_to_dataset_names(self) -> None:
        assert canon_team("USA") == "United States"
        assert canon_team("Korea Republic") == "South Korea"
        assert canon_team("IR Iran") == "Iran"
        assert canon_team("Côte d'Ivoire") == "Ivory Coast"
        assert canon_team("Cabo Verde") == "Cape Verde"
        assert canon_team("Türkiye") == "Turkey"
        assert canon_team("Czechia") == "Czech Republic"
        assert canon_team("Congo DR") == "DR Congo"
        assert canon_team("Brazil") == "Brazil"  # identity for matching names

    def test_unicode_normalization(self) -> None:
        # decomposed ô (o + combining circumflex) must still hit the map
        decomposed = "Côte d'Ivoire"
        assert canon_team(decomposed) == "Ivory Coast"


class TestSlotParsing:
    def test_rank_slot(self) -> None:
        ref = parse_slot("1A")
        assert (ref.kind, ref.rank, ref.groups) == ("rank", 1, ("A",))
        ref = parse_slot("2L")
        assert (ref.kind, ref.rank, ref.groups) == ("rank", 2, ("L",))

    def test_winner_slot(self) -> None:
        assert parse_slot("W74").match_number == 74
        assert parse_slot("Winner 102").match_number == 102

    def test_third_place_pool_slot(self) -> None:
        ref = parse_slot("3ABCDF")
        assert ref.kind == "third"
        assert ref.groups == ("A", "B", "C", "D", "F")

    def test_unparseable_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_slot("Mexico")  # a real team name is not a slot


class TestHostCities:
    def test_country_mapping(self) -> None:
        assert country_of_city("Mexico City") == "Mexico"
        assert country_of_city("Guadalajara") == "Mexico"
        assert country_of_city("Toronto") == "Canada"
        assert country_of_city("Vancouver") == "Canada"
        assert country_of_city("New York/New Jersey") == "United States"
        assert country_of_city("San Francisco Bay Area") == "United States"

    def test_unknown_city_fails_loudly(self) -> None:
        # A feed-side city rename must crash generation, not silently
        # strip a host's advantage.
        with pytest.raises(ValueError):
            country_of_city("Ciudad de México")
        with pytest.raises(ValueError):
            country_of_city("")


class TestGroupRanking2026:
    """2026 regulations: head-to-head BEFORE overall goal difference —
    first World Cup since 1970 to flip this order."""

    def test_h2h_beats_overall_goal_difference(self) -> None:
        # P and Q tied on 6 pts. Q has overall GD +6, P only +1 — but P won
        # the head-to-head, so 2026 rules rank P first (pre-2026 rules
        # would rank Q first).
        teams = ["P", "Q", "R", "S"]
        pts = {"P": 6, "Q": 6, "R": 4, "S": 1}
        gd = {"P": 1, "Q": 6, "R": 0, "S": -7}
        gf = {"P": 2, "Q": 7, "R": 1, "S": 0}
        h2h = {
            ("P", "Q"): (1, 0), ("Q", "P"): (0, 1),
            ("P", "S"): (1, 0), ("S", "P"): (0, 1),
            ("Q", "S"): (5, 0), ("S", "Q"): (0, 5),
            ("P", "R"): (0, 1), ("R", "P"): (1, 0),
            ("Q", "R"): (2, 0), ("R", "Q"): (0, 2),
            ("R", "S"): (0, 0), ("S", "R"): (0, 0),
        }
        elo = {t: 1500.0 for t in teams}
        rng = np.random.default_rng(0)
        ranked = _rank_group_2026(teams, pts, gd, gf, h2h, elo, rng)
        assert ranked == ["P", "Q", "R", "S"]

    def test_fully_level_h2h_falls_to_overall_gd(self) -> None:
        # Two teams drew their h2h match -> tiebreak falls through to
        # overall goal difference.
        teams = ["X", "Y"]
        pts = {"X": 4, "Y": 4}
        gd = {"X": 3, "Y": 1}
        gf = {"X": 4, "Y": 2}
        h2h = {("X", "Y"): (1, 1), ("Y", "X"): (1, 1)}
        elo = {"X": 1500.0, "Y": 1500.0}
        ranked = _rank_group_2026(
            teams, pts, gd, gf, h2h, elo, np.random.default_rng(0)
        )
        assert ranked == ["X", "Y"]

    def test_points_dominate(self) -> None:
        teams = ["A", "B"]
        pts = {"A": 3, "B": 6}
        gd = {"A": 5, "B": -5}
        gf = {"A": 5, "B": 1}
        ranked = _rank_group_2026(
            teams, pts, gd, gf, {}, {"A": 1500.0, "B": 1500.0},
            np.random.default_rng(0),
        )
        assert ranked == ["B", "A"]


class TestDCModel:
    def _synthetic_history(self) -> pd.DataFrame:
        # Strong team S (scores 3, concedes 0) vs weak W, plus a mid team M.
        # 60 matches over 2 years so decay weights stay meaningful.
        rng = np.random.default_rng(11)
        rows = []
        dates = pd.date_range("2023-01-01", periods=60, freq="12D")
        for i, d in enumerate(dates):
            a, b = [("S", "W"), ("S", "M"), ("M", "W")][i % 3]
            score = {("S", "W"): (3, 0), ("S", "M"): (2, 1), ("M", "W"): (2, 1)}[(a, b)]
            rows.append(
                {
                    "date": d,
                    "home_team": a,
                    "away_team": b,
                    "home_score": score[0] + int(rng.integers(0, 2)),
                    "away_score": score[1],
                    "tournament": "Friendly",
                    "city": "x",
                    "country": "x",
                    "neutral": True,
                }
            )
        return pd.DataFrame(rows)

    def test_recovers_team_ordering(self) -> None:
        from scripts.worldcup.engine import fit_dc_model

        hist = self._synthetic_history()
        dc = fit_dc_model(hist, train_start="2023-01-01", reg=0.5, min_matches=10)
        assert dc.att["S"] > dc.att["M"] > dc.att["W"]
        lam_s = dc.lam("S", "W", at_home=False)
        lam_w = dc.lam("W", "S", at_home=False)
        assert lam_s is not None and lam_w is not None
        assert lam_s > 2.0 * lam_w

    def test_unseen_team_returns_none_and_blend_falls_back(self) -> None:
        from scripts.worldcup.engine import blend_lambdas, fit_dc_model

        hist = self._synthetic_history()
        dc = fit_dc_model(hist, train_start="2023-01-01", reg=0.5, min_matches=10)
        assert dc.lam("S", "Atlantis", at_home=False) is None
        assert blend_lambdas(1.7, None, 0.5) == 1.7

    def test_geometric_blend(self) -> None:
        from scripts.worldcup.engine import blend_lambdas

        assert blend_lambdas(2.0, 2.0, 0.5) == pytest.approx(2.0)
        blended = blend_lambdas(1.0, 4.0, 0.5)
        assert blended == pytest.approx(2.0)  # geometric mean


class TestPlayerNames:
    def test_accent_stripping(self) -> None:
        assert normalize_name("Kylian Mbappé") == "kylian mbappe"
        assert normalize_name("Vinícius Júnior") == "vinicius junior"
        # csv is accent-inconsistent (Junya Ito vs Junya Itō) — both normalize equal
        assert normalize_name("Junya Itō") == normalize_name("Junya Ito")

    def test_known_aliases(self) -> None:
        assert normalize_name("Max Arfsten") == "maximilian arfsten"
        assert normalize_name("José Giménez") == "jose maria gimenez"

    def test_token_sorted_handles_name_order_reversal(self) -> None:
        from scripts.worldcup.players import norm_sorted

        # Sofascore lists East-Asian names western-order; squads.json doesn't.
        assert norm_sorted("Heung-min Son") == norm_sorted("Son Heung-min")
        assert norm_sorted("Kylian Mbappé") == norm_sorted("Mbappé Kylian")


class TestStarterDamp:
    def test_presence_mode_deployed_default(self) -> None:
        from scripts.worldcup.players import STARTER_DAMP_FLOOR, starter_damp

        # Absent from recent matches -> the floor, never zero.
        assert starter_damp(None) == STARTER_DAMP_FLOOR
        # Rotated-but-present (>=2 apps) keeps FULL share — warm-up minutes
        # must not punish tournament starters.
        rotated = {"recent_matches": 5.0, "apps": 3.0, "starts": 1.0, "avg_min": 30.0}
        assert starter_damp(rotated) == 1.0
        # One token appearance -> partial confidence.
        fringe = {"recent_matches": 5.0, "apps": 1.0, "starts": 0.0, "avg_min": 12.0}
        assert STARTER_DAMP_FLOOR < starter_damp(fringe) < 1.0
        # Zero appearances despite the team playing -> the floor.
        ghost = {"recent_matches": 5.0, "apps": 0.0, "starts": 0.0, "avg_min": 0.0}
        assert starter_damp(ghost) == STARTER_DAMP_FLOOR

    def test_minutes_mode_proportionality(self) -> None:
        from scripts.worldcup.players import STARTER_DAMP_FLOOR, starter_damp

        full = {"recent_matches": 5.0, "apps": 5.0, "starts": 5.0, "avg_min": 90.0}
        part = {"recent_matches": 5.0, "apps": 3.0, "starts": 1.0, "avg_min": 30.0}
        more = dict(part, avg_min=60.0)
        assert starter_damp(full, mode="minutes") == 1.0
        assert (
            STARTER_DAMP_FLOOR
            < starter_damp(part, mode="minutes")
            < starter_damp(more, mode="minutes")
            <= 1.0
        )


def _toy_scorers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-06-01", "2025-06-01", "2025-06-01", "2025-06-01"]
            ),
            "home_team": ["X"] * 5,
            "away_team": ["Y"] * 5,
            "team": ["X", "X", "X", "X", "Y"],
            "scorer": ["Star Man", "Star Man", "Star Man", "Role Player", "Opponent OG"],
            "minute": [10.0, 20.0, 30.0, 40.0, 50.0],
            "own_goal": [False, False, False, False, False],
            "penalty": [False, False, False, False, False],
            "norm_scorer": [
                "star man", "star man", "star man", "role player", "opponent og",
            ],
        }
    )


class TestScorerShares:
    def test_share_math_and_decay(self) -> None:
        shares = build_shares(
            _toy_scorers(), "X", pd.Timestamp("2025-12-01"), alpha=1.0
        )
        assert "star man" in shares
        # Star Man (3 goals, one recent) outranks Role Player (1 recent goal)
        assert shares["star man"]["share"] > shares["role player"]["share"]
        # Opponent's goal never enters team X's table
        assert "opponent og" not in shares
        # Shrinkage: shares sum strictly below 1
        assert sum(s["share"] for s in shares.values()) < 1.0

    def test_recent_goal_outweighs_old_goal(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2018-01-01", "2025-06-01"]),
                "home_team": ["X", "X"],
                "away_team": ["Y", "Y"],
                "team": ["X", "X"],
                "scorer": ["Old Guy", "New Guy"],
                "minute": [10.0, 20.0],
                "own_goal": [False, False],
                "penalty": [False, False],
                "norm_scorer": ["old guy", "new guy"],
            }
        )
        shares = build_shares(df, "X", pd.Timestamp("2025-12-01"), alpha=0.5)
        assert "new guy" in shares
        # An 8-year-old goal decays below the candidate floor entirely
        assert "old guy" not in shares

    def test_p_anytime_formula(self) -> None:
        assert p_anytime(2.0, 0.3) == pytest.approx(1 - np.exp(-0.6))
        assert p_anytime(100.0, 1.0) == 0.99  # clamped
        assert p_anytime(1.5, 0.0) == 0.0

    def test_own_goals_dropped_by_loader(self, tmp_path) -> None:
        # csv semantics: an OG row names the OPPONENT's player but credits the
        # beneficiary team — the loader must drop the row entirely.
        csv = tmp_path / "goalscorers.csv"
        csv.write_text(
            "date,home_team,away_team,team,scorer,minute,own_goal,penalty\n"
            "2025-06-01,X,Y,X,Real Striker,10,FALSE,FALSE\n"
            "2025-06-01,X,Y,X,Hapless Defender,55,TRUE,FALSE\n"
        )
        from scripts.worldcup.players import load_goalscorers

        df = load_goalscorers(csv)
        shares = build_shares(df, "X", pd.Timestamp("2025-12-01"), alpha=0.5)
        assert "real striker" in shares
        assert "hapless defender" not in shares

    def test_squad_filter(self) -> None:
        shares = {
            "star man": {"share": 0.3, "decayed_goals": 2.5},
            "retired legend": {"share": 0.2, "decayed_goals": 1.5},
        }
        squad = [
            {"name": "Star Man", "position": "FW", "goals": 10},
            {"name": "New Debutant", "position": "FW", "goals": 0},
        ]
        cands = _squad_candidates(squad, shares)
        names = [c["name"] for c in cands]
        assert names == ["Star Man"]  # retired player and no-history player excluded


class TestTiltLambdas:
    def test_tilt_moves_one_x_two_toward_target(self) -> None:
        from scripts.worldcup.engine import one_x_two, tilt_lambdas

        lam_h, lam_a = 1.8, 1.0
        base = one_x_two(lam_h, lam_a)
        # Target: market says the away side is much stronger than the model does.
        target = (0.30, 0.27, 0.43)
        t_h, t_a = tilt_lambdas(lam_h, lam_a, target)
        tilted = one_x_two(t_h, t_a)
        base_err = sum((a - b) ** 2 for a, b in zip(base, target, strict=True))
        tilt_err = sum((a - b) ** 2 for a, b in zip(tilted, target, strict=True))
        assert tilt_err < base_err
        # Total goals preserved by construction (k and 1/k scaling).
        assert t_h * t_a == pytest.approx(lam_h * lam_a, rel=1e-9)

    def test_identity_target_keeps_lambdas(self) -> None:
        from scripts.worldcup.engine import one_x_two, tilt_lambdas

        lam_h, lam_a = 1.5, 1.2
        target = one_x_two(lam_h, lam_a)
        t_h, t_a = tilt_lambdas(lam_h, lam_a, target)
        assert t_h == pytest.approx(lam_h, rel=0.05)
        assert t_a == pytest.approx(lam_a, rel=0.05)


class TestMarketOdds:
    def test_fractional_to_decimal(self) -> None:
        from scripts.worldcup.sofascore_fetch import fractional_to_decimal

        assert fractional_to_decimal("21/50") == pytest.approx(1.42)
        assert fractional_to_decimal("1/1") == pytest.approx(2.0)
        assert fractional_to_decimal("9/2") == pytest.approx(5.5)

    def test_devig_sums_to_one(self) -> None:
        from scripts.worldcup.sofascore_fetch import devig

        implied = devig({"home": 1.42, "draw": 4.8, "away": 8.0})
        assert sum(implied.values()) == pytest.approx(1.0, abs=0.001)
        assert implied["home"] > implied["draw"] > implied["away"]

    def test_parse_1x2(self) -> None:
        from scripts.worldcup.sofascore_fetch import parse_1x2

        payload = {
            "markets": [
                {
                    "marketName": "Full time",
                    "isLive": False,
                    "choices": [
                        {"name": "1", "fractionalValue": "21/50"},
                        {"name": "X", "fractionalValue": "19/5"},
                        {"name": "2", "fractionalValue": "7/1"},
                    ],
                }
            ]
        }
        decs = parse_1x2(payload)
        assert decs == {"home": 1.42, "draw": 4.8, "away": 8.0}
        assert parse_1x2({"markets": []}) is None


class TestScrapeGoalMerge:
    def test_cutoff_bridging_and_per_goal_expansion(self) -> None:
        from scripts.worldcup.players import scrape_goal_rows

        sofa = pd.DataFrame(
            {
                "team": ["USA", "USA", "USA"],
                "date": pd.to_datetime(["2026-05-31", "2026-05-31", "2026-02-01"]),
                "norm_name_sorted": [
                    "christian pulisic", "ghost player", "christian pulisic",
                ],
                "goals": [2.0, 1.0, 1.0],
                "minutes": [90, 90, 90],
            }
        )
        squads = {"USA": [{"name": "Christian Pulisic", "position": "FW"}]}
        rows = scrape_goal_rows(sofa, squads, cutoff=pd.Timestamp("2026-03-31"))
        # Pre-cutoff goal excluded (csv already has it); non-squad player
        # dropped; 2-goal match expands to 2 rows; team canonicalized.
        assert len(rows) == 2
        assert set(rows["norm_scorer"]) == {"christian pulisic"}
        assert set(rows["team"]) == {"United States"}


class TestGrading:
    def _results_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-06-11", "2026-06-12"]),
                "home_team": ["Mexico", "United States"],
                "away_team": ["South Africa", "Paraguay"],
                "home_score": [2, 1],
                "away_score": [0, 1],
                "tournament": ["FIFA World Cup"] * 2,
                "city": ["x"] * 2,
                "country": ["x"] * 2,
                "neutral": [False, False],
            }
        )

    def test_find_result_with_canon_and_date_skew(self) -> None:
        from scripts.worldcup.grading import _find_result

        df = self._results_df()
        # Display name 'USA' must canon to 'United States'; the snapshot
        # date is one day off (UTC-midnight skew) and must still match.
        r = _find_result(df, "USA", "Paraguay", "2026-06-13")
        assert r is not None and (r[0], r[1]) == (1, 1)
        r2 = _find_result(df, "Mexico", "South Africa", "2026-06-11")
        assert r2 is not None and (r2[0], r2[1]) == (2, 0)
        assert _find_result(df, "Mexico", "South Africa", "2026-06-20") is None

    def test_brier_math(self) -> None:
        from scripts.worldcup.grading import _brier

        # Perfect call -> 0; uniform on a home win -> 2*(1/3)^2 + (2/3)^2
        assert _brier((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)) == 0.0
        third = (1 / 3, 1 / 3, 1 / 3)
        assert _brier(third, (1.0, 0.0, 0.0)) == pytest.approx(2 / 3, abs=1e-9)


class TestNinetyMinuteReconstruction:
    def test_penalties_means_draw_at_ninety(self) -> None:
        from scripts.worldcup.grading import _ninety_minute_score

        shootouts = pd.DataFrame(
            {"date": ["2026-06-29"], "home_team": ["France"],
             "away_team": ["Brazil"], "winner": ["France"],
             "first_shooter": ["France"]}
        )
        r = _ninety_minute_score(
            pd.DataFrame(), shootouts, "France", "Brazil",
            pd.Timestamp("2026-06-29"), (4, 3),
        )
        assert r == (0, 0)  # level at 90' regardless of pens score

    def test_et_goals_subtracted(self) -> None:
        from scripts.worldcup.grading import _ninety_minute_score

        scorers = pd.DataFrame(
            {
                "date": ["2026-06-29"] * 3,
                "home_team": ["France"] * 3,
                "away_team": ["Brazil"] * 3,
                "team": ["France", "Brazil", "France"],
                "scorer": ["A", "B", "C"],
                "minute": [40.0, 88.0, 104.0],  # France winner in extra time
                "own_goal": [False, False, False],
                "penalty": [False, False, False],
            }
        )
        r = _ninety_minute_score(
            scorers, pd.DataFrame(), "France", "Brazil",
            pd.Timestamp("2026-06-29"), (2, 1),
        )
        assert r == (1, 1)  # the 104' goal does not count for 1X2 grading

    def test_incomplete_scorer_coverage_refuses(self) -> None:
        from scripts.worldcup.grading import _ninety_minute_score

        scorers = pd.DataFrame(
            {
                "date": ["2026-06-29"],
                "home_team": ["France"],
                "away_team": ["Brazil"],
                "team": ["France"],
                "scorer": ["A"],
                "minute": [40.0],
                "own_goal": [False],
                "penalty": [False],
            }
        )
        r = _ninety_minute_score(
            scorers, pd.DataFrame(), "France", "Brazil",
            pd.Timestamp("2026-06-29"), (2, 1),  # 3 goals, only 1 scorer row
        )
        assert r is None  # refuse to grade rather than grade wrong


@pytest.mark.integration
class TestRealArtifacts:
    """Validations against the real data files (skipped when absent)."""

    @pytest.fixture()
    def fixtures(self) -> list:
        from scripts.worldcup.simulate import FIXTURES_JSON, load_fixtures

        if not FIXTURES_JSON.exists():
            pytest.skip("fixtures.json not present")
        return load_fixtures()

    @pytest.fixture()
    def spec(self) -> dict:
        from scripts.worldcup.simulate import FORMAT_SPEC_JSON, load_format_spec

        if not FORMAT_SPEC_JSON.exists():
            pytest.skip("format_spec.json not present")
        return load_format_spec()

    def test_fixture_shape(self, fixtures: list) -> None:
        assert len(fixtures) == 104
        group = [f for f in fixtures if f["stage"] == "group"]
        assert len(group) == 72
        teams = {t for f in group for t in (f["home"], f["away"])}
        assert len(teams) == 48
        groups = {f["group"] for f in group}
        assert groups == set("ABCDEFGHIJKL")

    def test_annex_c_complete_and_consistent(self, spec: dict) -> None:
        combos = spec["third_place_allocation"]["combinations"]
        assert len(combos) == 495  # C(12, 8)
        seen = set()
        for c in combos:
            adv = frozenset(c["advancing_third_place_groups"])
            assert len(adv) == 8
            assert adv not in seen
            seen.add(adv)
            allocated = {g.lstrip("3") for g in c["allocation"].values()}
            assert allocated == set(adv)

    def test_small_simulation_invariants(self, fixtures: list, spec: dict) -> None:
        from scripts.worldcup.engine import RESULTS_CSV, WorldCupEngine
        from scripts.worldcup.simulate import TournamentSimulator

        if not RESULTS_CSV.exists():
            pytest.skip("international results dataset not present")
        engine = WorldCupEngine.build()
        sim = TournamentSimulator(
            engine, fixtures, spec, rng=np.random.default_rng(7)
        ).run(n_sims=200)
        champ_total = sum(s["champion"] for s in sim.team_stats.values())
        r32_total = sum(s["reach_r32"] for s in sim.team_stats.values())
        assert champ_total == pytest.approx(1.0, abs=1e-9)
        assert r32_total == pytest.approx(32.0, abs=1e-9)
        finalists = sum(s["reach_final"] for s in sim.team_stats.values())
        assert finalists == pytest.approx(2.0, abs=1e-9)

        # Per-knockout-match distributions: every KO match tracked (third
        # place included), each a proper probability distribution.
        ko_numbers = sorted(
            int(f["match_number"]) for f in fixtures if f["stage"] != "group"
        )
        assert sorted(sim.ko_matchup_probs) == ko_numbers
        assert sorted(sim.ko_win_probs) == ko_numbers
        for mn in ko_numbers:
            assert sum(sim.ko_matchup_probs[mn].values()) == pytest.approx(1.0)
            assert sum(sim.ko_win_probs[mn].values()) == pytest.approx(1.0)
            for (a, b), p in sim.ko_matchup_probs[mn].items():
                assert a <= b and 0.0 < p <= 1.0
        # r32 back-compat view unchanged in shape
        assert len(sim.r32_matchup_probs) == 16

        # The predicted bracket built on the same engine + sim is coherent.
        from scripts.worldcup.generate_predictions import build_bracket

        bracket = build_bracket(engine, fixtures, sim, spec)
        teams_48 = {
            t for f in fixtures if f["stage"] == "group"
            for t in (str(f["home"]), str(f["away"]))
        }
        assert bracket["champion"] in teams_48
        assert bracket["third_place_winner"] in teams_48
        assert len(bracket["matches"]) == 32
        for m in bracket["matches"]:
            assert m["pairing_prob"] >= 0.0
            adv = m["prediction"]["advance"]
            assert adv["home"] + adv["away"] == pytest.approx(1.0, abs=1e-3)

        # The simulator must survive knockout.py-resolved fixtures (real
        # names in home/away, original W-refs preserved in slot_home/away) —
        # it replays the whole tournament from the original refs.
        filled = [dict(f) for f in fixtures]
        m_r16 = next(f for f in filled if f["stage"] == "round_of_16")
        m_r16["slot_home"], m_r16["slot_away"] = m_r16["home"], m_r16["away"]
        m_r16["home"], m_r16["away"] = "Spain", "Japan"
        sim_filled = TournamentSimulator(
            engine, filled, spec, rng=np.random.default_rng(11)
        ).run(n_sims=5)
        assert sim_filled.n_sims == 5  # no parse_slot crash on real names

    @staticmethod
    def _stub_engine_and_stats(fixtures: list) -> tuple:
        """Deterministic engine stub + sim marginals for bracket tests that
        must not pay the real engine build."""
        from scripts.worldcup.simulate import SimResult

        class _StubEngine:
            def lambdas(
                self,
                home: str,
                away: str,
                home_at_home: bool = False,
                away_at_home: bool = False,
            ) -> tuple[float, float]:
                # Varies by name so advance probs differ; deterministic.
                return (
                    1.0 + (len(home) % 4) * 0.25 + (0.3 if home_at_home else 0.0),
                    0.8 + (len(away) % 3) * 0.25 + (0.3 if away_at_home else 0.0),
                )

        groups: dict[str, list[str]] = {}
        for f in fixtures:
            if f["stage"] != "group":
                continue
            for t in (str(f["home"]), str(f["away"])):
                if t not in groups.setdefault(str(f["group"]), []):
                    groups[str(f["group"])].append(t)
        team_stats: dict[str, dict[str, float]] = {}
        for letter, teams in groups.items():
            for i, t in enumerate(sorted(teams)):
                team_stats[t] = {
                    "group_winner": 0.6 - 0.15 * i,
                    "group_runner_up": 0.4 - 0.05 * i,
                    "third_qualified": 0.22 - 0.04 * i
                    + 0.001 * (ord(letter) - 65),
                }
        sim = SimResult(
            n_sims=0, team_stats=team_stats, r32_matchup_probs={},
            ko_matchup_probs={}, ko_win_probs={},
        )
        return _StubEngine(), sim

    def test_bracket_coherent_without_real_engine(
        self, fixtures: list, spec: dict
    ) -> None:
        import re as _re
        from collections import Counter

        from scripts.worldcup.generate_predictions import build_bracket

        engine, sim = self._stub_engine_and_stats(fixtures)
        bracket = build_bracket(engine, fixtures, sim, spec)  # type: ignore[arg-type]

        stage_sizes = Counter(m["stage"] for m in bracket["matches"])
        assert stage_sizes == {
            "round_of_32": 16, "round_of_16": 8, "quarter_final": 4,
            "semi_final": 2, "third_place": 1, "final": 1,
        }
        # Coherence: no team occupies two slots in the same round.
        by_stage: dict[str, list[str]] = {}
        for m in bracket["matches"]:
            by_stage.setdefault(m["stage"], []).extend([m["home"], m["away"]])
        for stage, occupants in by_stage.items():
            assert len(occupants) == len(set(occupants)), stage
        # Standings: 12 groups × 4 teams, exactly 8 qualifying thirds.
        assert len(bracket["standings"]) == 12
        assert all(len(s["order"]) == 4 for s in bracket["standings"].values())
        assert sum(s["third_qualifies"] for s in bracket["standings"].values()) == 8
        # Every match advances one of its own sides with a real scoreline.
        for m in bracket["matches"]:
            assert m["advances"] in (m["home"], m["away"])
            assert _re.fullmatch(r"\d+-\d+", m["prediction"]["predicted_score"])
        # The final's winner is the champion; SF losers play the 3rd-place game.
        final = next(m for m in bracket["matches"] if m["stage"] == "final")
        assert bracket["champion"] == final["advances"]
        sfs = [m for m in bracket["matches"] if m["stage"] == "semi_final"]
        sf_losers = {
            m["away"] if m["advances"] == m["home"] else m["home"] for m in sfs
        }
        third = next(m for m in bracket["matches"] if m["stage"] == "third_place")
        assert {third["home"], third["away"]} == sf_losers

    def test_bracket_resolved_fixture_override(
        self, fixtures: list, spec: dict
    ) -> None:
        from scripts.worldcup.generate_predictions import build_bracket

        engine, sim = self._stub_engine_and_stats(fixtures)
        resolved = [dict(f) for f in fixtures]
        m73 = next(f for f in resolved if int(f["match_number"]) == 73)
        # Mirror scripts.worldcup.knockout's fill: original label preserved.
        m73["slot_home"], m73["slot_away"] = m73["home"], m73["away"]
        m73["home"], m73["away"] = "Mexico", "France"  # reality beat the model
        bracket = build_bracket(engine, resolved, sim, spec)  # type: ignore[arg-type]

        b73 = next(m for m in bracket["matches"] if m["match_number"] == 73)
        assert (b73["home"], b73["away"]) == ("Mexico", "France")
        assert b73["resolved"] is True
        assert b73["pairing_prob"] == 1.0
        assert b73["slots"] == "2A vs 2B"  # display keeps the original slots
        # The R16 match fed by 73 must receive 73's advancing team.
        feeder = next(
            m for m in bracket["matches"]
            if 73 in (m["home_source"], m["away_source"])
        )
        side = "home" if feeder["home_source"] == 73 else "away"
        assert feeder[side] == b73["advances"]

        # A filled W-ref round keeps its tree links: resolve an R16 fixture
        # the way knockout.py writes it and the sources must survive.
        m89 = next(f for f in resolved if f["stage"] == "round_of_16")
        m89["slot_home"], m89["slot_away"] = m89["home"], m89["away"]
        m89["home"], m89["away"] = "Spain", "Japan"
        bracket2 = build_bracket(engine, resolved, sim, spec)  # type: ignore[arg-type]
        b89 = next(
            m for m in bracket2["matches"]
            if m["match_number"] == int(m89["match_number"])
        )
        assert (b89["home"], b89["away"]) == ("Spain", "Japan")
        assert b89["resolved"] is True
        assert b89["home_source"] is not None  # tree link parsed from slot_home
        assert b89["away_source"] is not None


class TestResultConditioning:
    """Played results pin the sim (scores), the knockouts (winners) and the
    bracket (reality outranks the engine pick)."""

    @pytest.fixture()
    def fixtures(self) -> list:
        from scripts.worldcup.simulate import FIXTURES_JSON, load_fixtures

        if not FIXTURES_JSON.exists():
            pytest.skip("fixtures.json not present")
        return load_fixtures()

    @pytest.fixture()
    def spec(self) -> dict:
        from scripts.worldcup.simulate import FORMAT_SPEC_JSON, load_format_spec

        if not FORMAT_SPEC_JSON.exists():
            pytest.skip("format_spec.json not present")
        return load_format_spec()

    def test_collect_played_results(self) -> None:
        from scripts.worldcup.knockout import collect_played_results

        fixtures = [
            {"match_number": 1, "stage": "group", "home": "Mexico",
             "away": "South Africa", "date_utc": "2026-06-11T19:00:00Z"},
            {"match_number": 73, "stage": "round_of_32", "home": "2A",
             "away": "2B", "date_utc": "2026-06-28T19:00:00Z"},  # unfilled
            {"match_number": 74, "stage": "round_of_32", "home": "Spain",
             "away": "Japan", "date_utc": "2026-06-28T22:00:00Z"},
            {"match_number": 75, "stage": "round_of_32", "home": "France",
             "away": "Brazil", "date_utc": "2026-06-29T19:00:00Z"},
            {"match_number": 76, "stage": "round_of_32", "home": "England",
             "away": "Ghana", "date_utc": "2026-06-29T22:00:00Z"},
        ]
        results = {
            ("Mexico", "South Africa"): (0, 5),
            ("Spain", "Japan"): (2, 1),
            ("France", "Brazil"): (1, 1),   # level → shootout decides
            ("England", "Ghana"): (0, 0),   # level, shootout row missing
        }
        played = collect_played_results(
            fixtures,
            result_of=lambda h, a, d: results.get((h, a)),
            shootout_winner=lambda h, a, d: "France" if h == "France" else None,
        )
        assert played["scores"] == {1: (0, 5), 74: (2, 1), 75: (1, 1), 76: (0, 0)}
        assert played["winners"] == {74: "Spain", 75: "France"}  # 76 refused

    class _FlatEngine:
        """Every team equal — only pinned reality can separate them."""

        def elo(self, team: str) -> float:
            return 1800.0

        def lambdas(self, home: str, away: str, home_at_home: bool = False,
                    away_at_home: bool = False) -> tuple[float, float]:
            return (1.2, 1.2)

    def test_pinned_group_score_banks_real_points(
        self, fixtures: list, spec: dict
    ) -> None:
        from scripts.worldcup.simulate import TournamentSimulator

        m1 = next(f for f in fixtures if int(f["match_number"]) == 1)
        loser, winner = str(m1["home"]), str(m1["away"])  # pin an away rout
        sim = TournamentSimulator(
            self._FlatEngine(), fixtures, spec,  # type: ignore[arg-type]
            rng=np.random.default_rng(3),
            pinned_scores={1: (0, 5)},
        ).run(n_sims=200)
        s_w, s_l = sim.team_stats[winner], sim.team_stats[loser]
        # 3 banked points + GD+5 vs 0 must dominate flat-strength teammates.
        assert s_w["group_winner"] > s_l["group_winner"] + 0.2
        assert s_w["reach_r32"] > s_l["reach_r32"]
        # Invariants survive conditioning.
        assert sum(t["champion"] for t in sim.team_stats.values()) == pytest.approx(1.0, abs=1e-9)
        assert sum(t["reach_r32"] for t in sim.team_stats.values()) == pytest.approx(32.0, abs=1e-9)

    def test_bogus_pinned_winner_is_ignored(
        self, fixtures: list, spec: dict
    ) -> None:
        from scripts.worldcup.simulate import TournamentSimulator

        sim = TournamentSimulator(
            self._FlatEngine(), fixtures, spec,  # type: ignore[arg-type]
            rng=np.random.default_rng(4),
            pinned_winners={73: "Atlantis"},  # never an entrant
        ).run(n_sims=50)
        assert "Atlantis" not in sim.ko_win_probs[73]
        assert sum(sim.ko_win_probs[73].values()) == pytest.approx(1.0)

    def test_merged_lookups_csv_first_sofascore_bridge(
        self, tmp_path, monkeypatch
    ) -> None:
        import json as _json

        from scripts.worldcup import knockout as ko

        store = tmp_path / "sofa.json"
        store.write_text(_json.dumps({"results": [
            {"event_id": 1, "date": "2026-06-11", "home": "Mexico",
             "away": "South Africa", "home_score": 2, "away_score": 1,
             "winner": "Mexico", "decided_by": "FT", "penalties": None},
            {"event_id": 2, "date": "2026-07-05", "home": "Spain",
             "away": "Japan", "home_score": 1, "away_score": 1,
             "winner": "Japan", "decided_by": "PEN", "penalties": [3, 4]},
        ]}))
        # CSV empty → the scrape bridges, in both orientations, exact date.
        monkeypatch.setattr(ko, "_real_result_lookup",
                            lambda: (lambda h, a, d: None))
        monkeypatch.setattr(ko, "_real_shootout_lookup",
                            lambda: (lambda h, a, d: None))
        rfn = ko._merged_result_lookup(sofa_path=store)
        assert rfn("Mexico", "South Africa", "2026-06-11") == (2, 1)
        assert rfn("South Africa", "Mexico", "2026-06-11") == (1, 2)
        assert rfn("Mexico", "South Africa", "2026-06-12") is None
        sfn = ko._merged_shootout_lookup(sofa_path=store)
        assert sfn("Spain", "Japan", "2026-07-05") == "Japan"  # PEN winner
        assert sfn("Mexico", "South Africa", "2026-06-11") is None  # FT: none
        # The canonical CSV always outranks the scrape.
        monkeypatch.setattr(ko, "_real_result_lookup",
                            lambda: (lambda h, a, d: (9, 9)))
        assert ko._merged_result_lookup(sofa_path=store)(
            "Mexico", "South Africa", "2026-06-11"
        ) == (9, 9)
        # Malformed rows are skipped, never crash the fill.
        store.write_text(_json.dumps({"results": [{"date": "x"}, 42]}))
        assert ko._sofa_results_index(store) == {}

    def test_event_to_result_filters(self) -> None:
        from scripts.worldcup.sofascore_fetch import _event_to_result

        ids = {100: "Mexico", 200: "South Africa"}
        base = {
            "id": 7, "startTimestamp": 1781204400,  # 2026-06-11 19:00 UTC
            "tournament": {"name": "FIFA World Cup 2026, Group A"},
            "status": {"type": "finished"},
            "homeTeam": {"id": 100}, "awayTeam": {"id": 200},
            "homeScore": {"current": 2}, "awayScore": {"current": 1},
            "winnerCode": 1,
        }
        rec = _event_to_result(base, ids, "2026-06-11")
        assert rec == {
            "event_id": 7, "date": "2026-06-11", "home": "Mexico",
            "away": "South Africa", "home_score": 2, "away_score": 1,
            "winner": "Mexico", "decided_by": "FT", "penalties": None,
        }
        # Rejection matrix: live score, wrong tournament, unmapped id,
        # missing score, pre-tournament date.
        assert _event_to_result({**base, "status": {"type": "inprogress"}}, ids, "2026-06-11") is None
        assert _event_to_result({**base, "tournament": {"name": "Friendly"}}, ids, "2026-06-11") is None
        assert _event_to_result({**base, "homeTeam": {"id": 999}}, ids, "2026-06-11") is None
        assert _event_to_result({**base, "homeScore": {}}, ids, "2026-06-11") is None
        assert _event_to_result(base, ids, "2026-06-12") is None
        # Penalties: winner from winnerCode even when the ET score is level.
        pens = {**base, "homeScore": {"current": 1, "penalties": 3},
                "awayScore": {"current": 1, "penalties": 4}, "winnerCode": 2}
        rec2 = _event_to_result(pens, ids, "2026-06-11")
        assert rec2 is not None
        assert (rec2["winner"], rec2["decided_by"], rec2["penalties"]) == (
            "South Africa", "PEN", [3, 4])

    def test_events_from_next_data_shape_walk(self) -> None:
        import json as _json

        from scripts.worldcup.sofascore_fetch import _events_from_next_data

        ev = {"id": 1, "homeTeam": {"id": 100}, "awayTeam": {"id": 200},
              "status": {"type": "finished"}, "homeScore": {"current": 2},
              "awayScore": {"current": 0}, "tournament": {"name": "X"}}
        blob = {"props": {"pageProps": {"deeply": [{"nested": {"events": [ev]}}]}}}
        html = ('<html><script id="__NEXT_DATA__" type="application/json">'
                + _json.dumps(blob) + "</script></html>")
        found = _events_from_next_data(html)
        assert len(found) == 1 and found[0]["id"] == 1  # found by shape
        assert _events_from_next_data("<html>no blob</html>") == []
        assert _events_from_next_data(
            '<script id="__NEXT_DATA__">not json</script>') == []

    def test_bracket_played_match_uses_real_winner(
        self, fixtures: list, spec: dict
    ) -> None:
        from scripts.worldcup.generate_predictions import build_bracket

        engine, sim = TestRealArtifacts._stub_engine_and_stats(fixtures)
        resolved = [dict(f) for f in fixtures]
        m73 = next(f for f in resolved if int(f["match_number"]) == 73)
        m73["slot_home"], m73["slot_away"] = m73["home"], m73["away"]
        m73["home"], m73["away"] = "Mexico", "France"
        played = {"scores": {73: (0, 1)}, "winners": {73: "France"}}
        bracket = build_bracket(
            engine, resolved, sim, spec, played=played  # type: ignore[arg-type]
        )
        b73 = next(m for m in bracket["matches"] if m["match_number"] == 73)
        assert b73["advances"] == "France"  # reality, whatever the engine said
        assert b73["played"] is True
        assert b73["actual_score"] == "0-1"
        feeder = next(
            m for m in bracket["matches"]
            if 73 in (m["home_source"], m["away_source"])
        )
        side = "home" if feeder["home_source"] == 73 else "away"
        assert feeder[side] == "France"
        # A pin whose winner isn't an entrant must not corrupt the walk.
        bad = {"scores": {73: (0, 1)}, "winners": {73: "Atlantis"}}
        bracket2 = build_bracket(
            engine, resolved, sim, spec, played=bad  # type: ignore[arg-type]
        )
        b73b = next(m for m in bracket2["matches"] if m["match_number"] == 73)
        assert b73b["advances"] in ("Mexico", "France")
        assert b73b["actual_score"] == "0-1"  # score still shown


class TestSquadStrength:
    """squad_history parsing/matching + the backtest study's adjustment math."""

    _WIKITEXT = """
==Group A==
===Ecuador===
{{nat fs g start}}
{{nat fs g player|no=1|pos=GK|name=[[Hernán Galíndez]]|age={{birth date and age2|df=y|2022|11|20|1987|3|30}}|caps=12|goals=0|club=[[S.D. Aucas|Aucas]]|clubnat=ECU}}
{{nat fs g player|no=2|pos=DF|name=[[Some Player]]|age={{birth date and age2|df=y|2022|11|20|1997|1|11}}|caps=17|goals=2|club=[[Manchester City F.C.|Manchester City]]|clubnat=ENG}}
{{nat fs g end}}
===Senegal===
{{nat fs g player|no=1|pos=GK|name=[[Édouard Mendy]]|age={{birth date and age2|df=y|2022|11|20|1992|3|1}}|caps=24|goals=0|club=[[Chelsea F.C.|Chelsea]]|clubnat=ENG}}
{{nat fs g player|no=9|pos=FW|name=[[X Y]]|age={{birth date and age2|df=y|2022|11|20|1999|5|5}}|caps=3|goals=1|club={{ill|Casa Sports|fr}}|clubnat=SEN}}
===Player representation===
Some appendix prose with no player rows.
"""

    def test_parse_squads_handles_nested_templates(self) -> None:
        from scripts.worldcup.squad_history import parse_squads

        squads = parse_squads(self._WIKITEXT)
        assert set(squads) == {"Ecuador", "Senegal"}  # appendix heading dropped
        assert squads["Ecuador"] == ["Aucas", "Manchester City"]
        assert squads["Senegal"] == ["Chelsea", "Casa Sports"]  # {{ill}} handled

    def test_club_matcher_cascade(self) -> None:
        from scripts.worldcup.squad_history import ClubMatcher

        elo = {"Man City": 2029.2, "Chelsea": 1900.0, "Arsenal": 1950.0,
               "Arsenal Tula": 1500.0, "Barcelona": 1980.0}
        m = ClubMatcher(elo)
        assert m.match("Chelsea") == "Chelsea"            # exact
        assert m.match("FC Barcelona") == "Barcelona"     # stop-token strip
        assert m.match("Manchester City") == "Man City"   # alias map
        assert m.match("Arsenal Tula") == "Arsenal Tula"  # exact beats prefix
        assert m.match("Casa Sports") is None             # not in ClubElo
        # 'Arsenal' alone: exact normalized hit, never the Tula guess
        assert m.match("Arsenal") == "Arsenal"

    def test_squad_adjustments_math(self) -> None:
        from scripts.worldcup.backtest import _squad_adjustments

        teams = {
            f"T{i}": {"mean_club_elo": 1700 + 50 * i, "coverage_pct": 100.0}
            for i in range(9)
        }
        teams["NoData"] = {"mean_club_elo": None, "coverage_pct": 0.0}
        teams["HalfCov"] = {"mean_club_elo": 1700 + 50 * 8, "coverage_pct": 50.0}
        hist = {"Tournament X": teams}
        adj = _squad_adjustments(hist, "Tournament X")
        assert "NoData" not in adj                       # no fabricated signal
        zs = np.array([adj[f"T{i}"] for i in range(9)])
        assert zs[0] < 0 < zs[-1]                        # ordered by strength
        assert abs(float(np.mean(zs))) < 0.2             # roughly centred
        # coverage shrinks: HalfCov has T8's elo but half the weight
        assert adj["HalfCov"] == pytest.approx(adj["T8"] * 0.5, rel=1e-6)
        # fewer than 8 teams with data -> no signal at all
        assert _squad_adjustments({"Y": dict(list(teams.items())[:5])}, "Y") == {}


class TestAvailability:
    """Player-availability layer: name keys, impact math, lineup overrides."""

    def _xi(self) -> list[dict]:
        # 1 GK, 4 DF, 4 MF, 2 FW — equal €10m values via the values dict
        names = [("gk1", "G")] + [(f"df{i}", "D") for i in range(4)] + [
            (f"mf{i}", "M") for i in range(4)] + [(f"fw{i}", "F") for i in range(2)]
        return [
            {"key": n, "name": n, "position": p, "starts": 5, "recent_matches": 5}
            for n, p in names
        ]

    def _values(self) -> dict[str, float]:
        return {p["key"]: 10e6 for p in self._xi()}

    def test_akey_order_and_hyphen_insensitive(self) -> None:
        from scripts.worldcup.availability import akey

        assert akey("Son Heung-min") == akey("Heung-Min Son")
        assert akey("Min-jae Kim") == akey("Kim Min jae")

    def test_pos_letter_across_sources(self) -> None:
        from scripts.worldcup.availability import pos_letter

        assert pos_letter("GK") == "G"
        assert pos_letter("Goalkeeper") == "G"
        assert pos_letter("Centre-Back") == "D"
        assert pos_letter("Defensive Midfield") == "M"
        assert pos_letter("FW") == "F"
        assert pos_letter("Right Winger") == "F"
        assert pos_letter("") == ""

    def test_no_absences_means_unit_factors(self) -> None:
        from scripts.worldcup.availability import absence_impact

        imp = absence_impact(self._xi(), {}, self._values())
        assert imp["lambda_factor_self"] == 1.0
        assert imp["lambda_factor_opp"] == 1.0
        assert imp["attack_share_out"] == 0.0

    def test_forward_out_hits_own_lambda_most(self) -> None:
        from scripts.worldcup.availability import absence_impact

        imp = absence_impact(self._xi(), {"fw0": 1.0}, self._values())
        assert imp["lambda_factor_self"] < 1.0
        assert imp["attack_share_out"] > imp["defense_share_out"]

    def test_keeper_out_only_bumps_opponent(self) -> None:
        from scripts.worldcup.availability import absence_impact

        imp = absence_impact(self._xi(), {"gk1": 1.0}, self._values())
        assert imp["lambda_factor_self"] == 1.0  # GK attack weight is 0
        assert imp["lambda_factor_opp"] > 1.0

    def test_factors_clamped_when_everyone_is_out(self) -> None:
        from scripts.worldcup.availability import FACTOR_BOUNDS, absence_impact

        deficits = {p["key"]: 1.0 for p in self._xi()}
        imp = absence_impact(self._xi(), deficits, self._values())
        assert imp["lambda_factor_self"] == FACTOR_BOUNDS[0]
        assert imp["lambda_factor_opp"] == FACTOR_BOUNDS[1]

    def test_doubtful_is_half_an_absence(self) -> None:
        from scripts.worldcup.availability import absence_impact

        full = absence_impact(self._xi(), {"fw0": 1.0}, self._values())
        half = absence_impact(self._xi(), {"fw0": 0.5}, self._values())
        assert half["attack_share_out"] == pytest.approx(
            full["attack_share_out"] / 2, abs=1e-4
        )

    def _team_df(self) -> pd.DataFrame:
        rows = []
        for d in pd.date_range("2026-01-01", periods=5, freq="30D"):
            for i in range(14):  # 11 regulars + 3 rotation players
                rows.append(
                    {
                        "date": d,
                        "player_name": f"Player {i}",
                        "position": "F" if i > 11 else ("G" if i == 0 else "M"),
                        "started": i < 11,
                        "minutes": 90.0 if i < 11 else 10.0,
                    }
                )
        return pd.DataFrame(rows)

    def test_expected_xi_is_top_by_starts(self) -> None:
        from scripts.worldcup.availability import expected_xi

        xi = expected_xi(self._team_df())
        assert len(xi) == 11
        names = {p["name"] for p in xi}
        assert names == {f"Player {i}" for i in range(11)}
        assert all(p["starts"] == 5 and p["recent_matches"] == 5 for p in xi)

    def test_confirmed_lineup_overrides_projection(self) -> None:
        from scripts.worldcup.availability import _side_report

        # Player 1 (expected starter) missing from the confirmed XI entirely;
        # Player 2 only on the bench; Player 12 promoted into the XI.
        starters = [f"Player {i}" for i in range(11) if i not in (1, 2)]
        starters += ["Player 12", "Player 13"]
        lineup_side = {"starters": starters, "bench": ["Player 2"], "missing": []}
        report = _side_report(
            "Testland", self._team_df(), lineup_side, True, {}, []
        )
        assert report["lineup_status"] == "confirmed"
        assert report["impact"]["based_on"] == "confirmed_vs_expected"
        assert report["impact"]["lambda_factor_self"] < 1.0
        assert {p["name"] for p in report["xi"]} == set(starters)

    def test_missing_list_drives_status_and_impact(self) -> None:
        from scripts.worldcup.availability import _side_report

        lineup_side = {
            "starters": [],
            "bench": [],
            "missing": [
                {"name": "Player 1", "position": "M", "type": "missing",
                 "reason": "Injury"},
                {"name": "Player 3", "position": "M", "type": "doubtful",
                 "reason": "Other"},
            ],
        }
        report = _side_report(
            "Testland", self._team_df(), lineup_side, False, {}, []
        )
        assert [p["name"] for p in report["out"]] == ["Player 1"]
        assert [p["name"] for p in report["doubtful"]] == ["Player 3"]
        assert report["out"][0]["in_expected_xi"] is True
        assert report["impact"]["based_on"] == "missing_list"
        assert report["impact"]["lambda_factor_self"] < 1.0

    def test_apply_availability_routes_factors_to_both_lambdas(self) -> None:
        from scripts.worldcup.generate_predictions import _apply_availability

        av = {
            "home": {"impact": {"lambda_factor_self": 0.95,
                                "lambda_factor_opp": 1.04},
                     "out": [{"name": "H Star"}], "doubtful": []},
            "away": {"impact": {"lambda_factor_self": 1.0,
                                "lambda_factor_opp": 1.0},
                     "out": [], "doubtful": []},
        }
        lam_h, lam_a, adjusted, summary = _apply_availability(1.5, 1.2, av)
        assert adjusted is True
        # Home attack absences shrink home lambda; home defensive absences
        # (factor_opp 1.04) inflate the AWAY lambda.
        assert lam_h == pytest.approx(1.5 * 0.95)
        assert lam_a == pytest.approx(1.2 * 1.04)
        assert summary is not None and summary["home_out"] == ["H Star"]

    def test_apply_availability_noop_without_news(self) -> None:
        from scripts.worldcup.generate_predictions import _apply_availability

        lam_h, lam_a, adjusted, summary = _apply_availability(1.5, 1.2, None)
        assert (lam_h, lam_a, adjusted, summary) == (1.5, 1.2, False, None)

    def test_real_availability_artifact_shape(self) -> None:
        from scripts.worldcup.availability import AVAILABILITY_JSON

        if not AVAILABILITY_JSON.exists():
            pytest.skip("player_availability.json not present")
        import json

        data = json.loads(AVAILABILITY_JSON.read_text())
        assert 0 < data["alpha"] <= 1
        for sides in data["matches"].values():
            for side in ("home", "away"):
                blob = sides[side]
                imp = blob["impact"]
                assert 0.85 <= imp["lambda_factor_self"] <= 1.0
                assert 1.0 <= imp["lambda_factor_opp"] <= 1.15
                assert len(blob["xi"]) <= 11

    def test_build_match_predictions_applies_team_news(
        self, tmp_path, monkeypatch
    ) -> None:
        """End-to-end: availability JSON on disk -> shifted lambdas + flag."""
        import json

        from scripts.worldcup import generate_predictions as gp
        from scripts.worldcup.engine import RESULTS_CSV, WorldCupEngine
        from scripts.worldcup.simulate import FIXTURES_JSON, load_fixtures

        if not (RESULTS_CSV.exists() and FIXTURES_JSON.exists()):
            pytest.skip("real datasets not present")
        engine = WorldCupEngine.build()
        fixture = next(f for f in load_fixtures() if f["stage"] == "group")
        mn = str(fixture["match_number"])

        # Kill the market blend so the availability effect is isolated
        monkeypatch.setattr(gp, "MARKET_ODDS_JSON", tmp_path / "no_odds.json")
        monkeypatch.setattr(gp, "AVAILABILITY_JSON", tmp_path / "missing.json")
        base = gp.build_match_predictions(engine, [fixture])[0]
        assert base["availability_adjusted"] is False

        av_file = tmp_path / "availability.json"
        av_file.write_text(json.dumps({"matches": {mn: {
            "home": {"impact": {"lambda_factor_self": 0.90,
                                "lambda_factor_opp": 1.00},
                     "out": [{"name": "Star"}], "doubtful": []},
            "away": {"impact": {"lambda_factor_self": 1.00,
                                "lambda_factor_opp": 1.00},
                     "out": [], "doubtful": []},
        }}}))
        monkeypatch.setattr(gp, "AVAILABILITY_JSON", av_file)
        adj = gp.build_match_predictions(engine, [fixture])[0]
        assert adj["availability_adjusted"] is True
        assert adj["home_xg"] == pytest.approx(base["home_xg"] * 0.90, rel=0.01)
        assert adj["probabilities"]["home"] < base["probabilities"]["home"]
        assert adj["availability_impact"]["home_out"] == ["Star"]


class TestBestCombos:
    """Accumulator builder (scripts.worldcup.combos.build_best_combos)."""

    NOW = None  # set in _build to keep datetime import local

    @staticmethod
    def _pred(n, home, away, h, d, a, ko, date, time="19:00"):
        return {
            "match_number": n, "match": f"{home} vs {away}",
            "home_team": home, "away_team": away,
            "date": date, "time": time, "kickoff_utc": ko,
            "probabilities": {"home": h, "draw": d, "away": a},
        }

    def _slate(self):
        # A: heavy home favorite (huge model-vs-market edge)
        # B: mild home favorite (small edge)
        # C: heavy away favorite (edge below the 2pp value bar)
        # D: already kicked off, E: beyond the 48h window — both excluded
        return [
            self._pred(1, "Mexico", "South Africa", 0.70, 0.20, 0.10,
                       "2026-06-15T18:00:00+00:00", "2026-06-15"),
            self._pred(2, "Canada", "Qatar", 0.55, 0.25, 0.20,
                       "2026-06-16T18:00:00+00:00", "2026-06-16"),
            self._pred(3, "Honduras", "Brazil", 0.10, 0.20, 0.70,
                       "2026-06-16T20:00:00+00:00", "2026-06-16", "21:00"),
            self._pred(4, "Italy", "Norway", 0.50, 0.30, 0.20,
                       "2026-06-15T10:00:00+00:00", "2026-06-15"),
            self._pred(5, "Spain", "Ghana", 0.60, 0.25, 0.15,
                       "2026-06-18T12:00:00+00:00", "2026-06-18"),
        ]

    def _market(self):
        return {
            "1": {"odds": {"home": 1.8, "draw": 4.0, "away": 8.5},
                  "implied": {"home": 0.52, "draw": 0.27, "away": 0.21}},
            "2": {"odds": {"home": 2.0, "draw": 3.4, "away": 3.8},
                  "implied": {"home": 0.48, "draw": 0.28, "away": 0.24}},
            "3": {"odds": {"home": 9.0, "draw": 4.4, "away": 1.45},
                  "implied": {"home": 0.10, "draw": 0.21, "away": 0.69}},
        }

    def _build(self, preds, market):
        from datetime import UTC, datetime

        from scripts.worldcup.combos import build_best_combos

        now = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)
        return build_best_combos(preds, market, now=now)

    def test_only_upcoming_window_enters(self) -> None:
        out = self._build(self._slate(), self._market())
        assert out["n_matches"] == 3
        matches = {leg["match"] for c in out["combos"] for leg in c["legs"]}
        assert "Italy vs Norway" not in matches  # already kicked off
        assert "Spain vs Ghana" not in matches   # beyond 48h

    def test_safe_combo_is_double_chance_product(self) -> None:
        out = self._build(self._slate(), self._market())
        safe = next(c for c in out["combos"] if c["key"] == "safe")
        assert all(leg["market"] == "Double Chance" for leg in safe["legs"])
        picks = {leg["match"]: leg["pick"] for leg in safe["legs"]}
        assert picks["Mexico vs South Africa"] == "1X"
        assert picks["Honduras vs Brazil"] == "X2"
        assert safe["combined"]["prob"] == pytest.approx(0.90 * 0.80 * 0.90, abs=1e-3)
        # DC market odds derived from the quoted 1X2 prices
        mex = next(leg for leg in safe["legs"] if leg["match"] == "Mexico vs South Africa")
        assert mex["market_odds"] == pytest.approx(1 / (1 / 1.8 + 1 / 4.0), abs=0.01)

    def test_favorites_combo_straight_top_pick(self) -> None:
        out = self._build(self._slate(), self._market())
        fav = next(c for c in out["combos"] if c["key"] == "favorites")
        assert all(leg["market"] == "1X2" for leg in fav["legs"])
        assert {leg["pick_label"] for leg in fav["legs"]} == {"Mexico", "Canada", "Brazil"}
        assert fav["combined"]["prob"] == pytest.approx(0.70 * 0.55 * 0.70, abs=1e-3)
        # legs come back in kickoff order, not selection order
        assert [leg["match_number"] for leg in fav["legs"]] == [1, 2, 3]

    def test_value_combo_requires_2pp_edge_and_prices_ev(self) -> None:
        out = self._build(self._slate(), self._market())
        val = next(c for c in out["combos"] if c["key"] == "value")
        # Brazil edge is 0.70-0.69=1pp -> below the bar; Mexico+Canada qualify
        assert {leg["pick_label"] for leg in val["legs"]} == {"Mexico", "Canada"}
        assert all(leg["edge"] >= 0.02 for leg in val["legs"])
        cm = val["combined"]
        assert cm["market_odds"] == pytest.approx(1.8 * 2.0, abs=0.01)
        assert cm["ev"] == pytest.approx(0.70 * 0.55 * 3.6 - 1, abs=1e-3)

    def test_no_market_means_no_value_combo(self) -> None:
        out = self._build(self._slate(), {})
        keys = {c["key"] for c in out["combos"]}
        assert keys == {"safe", "favorites"}
        safe = next(c for c in out["combos"] if c["key"] == "safe")
        assert "market_odds" not in safe["combined"]
        assert all(leg["market_odds"] is None for leg in safe["legs"])

    def test_single_match_yields_no_combo(self) -> None:
        out = self._build(self._slate()[:1], self._market())
        assert out["combos"] == []

    def test_malformed_entries_fail_soft(self) -> None:
        broken = [{"match_number": 9}, {"kickoff_utc": "garbage"},
                  self._pred(8, "A", "B", 0.0, 0.0, 0.0, "2026-06-16T18:00:00+00:00", "2026-06-16")]
        out = self._build(broken, {})
        assert out["combos"] == [] and out["n_matches"] == 0

    def test_value_scan_finds_non_favorite_edge(self) -> None:
        # Model loves the DRAW (34% vs implied 26%) while home is the
        # favorite — the value leg must be the draw framing, which the old
        # favorite-only scan could never produce.
        draw_edge = self._pred(6, "Japan", "Senegal", 0.40, 0.34, 0.26,
                               "2026-06-16T15:00:00+00:00", "2026-06-16")
        market = dict(self._market())
        market["6"] = {"odds": {"home": 2.1, "draw": 3.9, "away": 3.4},
                       "implied": {"home": 0.45, "draw": 0.26, "away": 0.29}}
        out = self._build([self._slate()[0], draw_edge], market)
        val = next(c for c in out["combos"] if c["key"] == "value")
        jpn = next(leg for leg in val["legs"] if leg["match"] == "Japan vs Senegal")
        assert jpn["market"] == "1X2" and jpn["pick"] == "X"
        assert jpn["pick_label"] == "Draw"
        assert jpn["ev"] == pytest.approx(0.34 * 3.9 - 1, abs=1e-3)

    def test_value_floor_excludes_lottery_legs(self) -> None:
        # 15% away pick has +5pp edge AND +EV at 8.0 — still excluded:
        # a recommended combo never carries a sub-20% leg. No other framing
        # of this match clears the gates either.
        longshot = self._pred(7, "Egypt", "Iceland", 0.60, 0.25, 0.15,
                              "2026-06-16T15:00:00+00:00", "2026-06-16")
        market = dict(self._market())
        market["7"] = {"odds": {"home": 1.55, "draw": 3.55, "away": 8.0},
                       "implied": {"home": 0.62, "draw": 0.28, "away": 0.10}}
        out = self._build(self._slate()[:2] + [longshot], market)
        val = next(c for c in out["combos"] if c["key"] == "value")
        assert {leg["pick_label"] for leg in val["legs"]} == {"Mexico", "Canada"}

    def test_legs_carry_grading_fields(self) -> None:
        # The archive grades tickets cold — every leg must be self-contained.
        out = self._build(self._slate(), self._market())
        for c in out["combos"]:
            for leg in c["legs"]:
                assert leg["home_team"] and leg["away_team"]
                assert leg["stage"] == "group" and leg["kickoff_utc"]


class TestComboArchive:
    """Ticket snapshots: last write before first kickoff wins, then frozen."""

    @staticmethod
    def _best(prob=0.63):
        leg1 = {"match": "A vs B", "home_team": "A", "away_team": "B",
                "stage": "group", "date": "2026-06-15", "time": "18:00",
                "kickoff_utc": "2026-06-15T18:00:00+00:00", "match_number": 1,
                "market": "Double Chance", "pick": "1X", "pick_label": "A or draw",
                "prob": 0.9, "fair_odds": 1.11, "market_odds": 1.1,
                "edge": 0.01, "ev": -0.01}
        leg2 = {"match": "C vs D", "home_team": "C", "away_team": "D",
                "stage": "group", "date": "2026-06-15", "time": "21:00",
                "kickoff_utc": "2026-06-15T21:00:00+00:00", "match_number": 2,
                "market": "1X2", "pick": "1", "pick_label": "C",
                "prob": 0.7, "fair_odds": 1.43, "market_odds": 1.8,
                "edge": 0.05, "ev": 0.26}
        return {"combos": [{"key": "safe", "title": "Safe", "note": "",
                            "legs": [leg1, leg2],
                            "combined": {"prob": prob, "fair_odds": 1.59,
                                         "market_odds": 1.98, "ev": 0.25}}]}

    def test_last_write_before_first_kickoff_wins(self, tmp_path, monkeypatch) -> None:
        import json

        import scripts.worldcup.combos as cb

        monkeypatch.setattr(cb, "COMBO_ARCHIVE_JSON", tmp_path / "arch.json")
        key = "safe|2026-06-15T18:00:00+00:00"
        assert cb.merge_combo_archive(self._best(0.63), "2026-06-15T10:00:00+00:00") == 1
        assert cb.merge_combo_archive(self._best(0.55), "2026-06-15T14:00:00+00:00") == 1
        arch = json.loads((tmp_path / "arch.json").read_text())
        assert arch[key]["combined"]["prob"] == 0.55  # last pre-kickoff write
        assert arch[key]["first_archived_at"] == "2026-06-15T10:00:00+00:00"
        assert arch[key]["archived_at"] == "2026-06-15T14:00:00+00:00"

    def test_post_kickoff_write_refused(self, tmp_path, monkeypatch) -> None:
        import json

        import scripts.worldcup.combos as cb

        monkeypatch.setattr(cb, "COMBO_ARCHIVE_JSON", tmp_path / "arch.json")
        key = "safe|2026-06-15T18:00:00+00:00"
        assert cb.merge_combo_archive(self._best(0.63), "2026-06-15T10:00:00+00:00") == 1
        # at first-leg kickoff (and any time after) the ticket is frozen
        assert cb.merge_combo_archive(self._best(0.10), "2026-06-15T18:00:00+00:00") == 0
        arch = json.loads((tmp_path / "arch.json").read_text())
        assert arch[key]["combined"]["prob"] == 0.63


class TestComboRecord:
    """Grading settled tickets vs 90' outcomes, ROI at archived odds."""

    @staticmethod
    def _ticket(tier, legs, prob, market_odds, archived="2026-06-15T10:00:00+00:00",
                first_ko="2026-06-15T18:00:00+00:00"):
        combined = {"prob": prob, "fair_odds": round(1 / prob, 2)}
        if market_odds:
            combined["market_odds"] = market_odds
        return {"tier": tier, "title": tier, "legs": legs, "combined": combined,
                "first_kickoff_utc": first_ko, "first_archived_at": archived,
                "archived_at": archived}

    @staticmethod
    def _leg(home, away, market, pick):
        return {"match": f"{home} vs {away}", "home_team": home, "away_team": away,
                "stage": "group", "date": "2026-06-15", "time": "18:00",
                "kickoff_utc": "2026-06-15T18:00:00+00:00", "match_number": 1,
                "market": market, "pick": pick, "pick_label": pick,
                "prob": 0.7, "fair_odds": 1.43, "market_odds": 1.5,
                "edge": 0.03, "ev": 0.05}

    def _run(self, archive, outcomes, tmp_path, monkeypatch):
        import json

        import scripts.worldcup.combos as cb

        monkeypatch.setattr(cb, "COMBO_ARCHIVE_JSON", tmp_path / "arch.json")
        monkeypatch.setattr(cb, "COMBO_RECORD_JSON", tmp_path / "rec.json")
        (tmp_path / "arch.json").write_text(json.dumps(archive))
        return cb.build_combo_record(
            resolve=lambda h, a, d, s: outcomes.get((h, a)))

    def test_hit_miss_roi_and_pending(self, tmp_path, monkeypatch) -> None:
        archive = {
            # both legs hit -> profit = odds - 1
            "safe|k1": self._ticket("safe", [self._leg("A", "B", "Double Chance", "1X"),
                                             self._leg("C", "D", "1X2", "1")],
                                    prob=0.6, market_odds=2.0),
            # one leg misses -> ticket lost, -1 unit
            "favorites|k1": self._ticket("favorites",
                                         [self._leg("A", "B", "1X2", "2"),
                                          self._leg("C", "D", "1X2", "1")],
                                         prob=0.3, market_odds=4.0),
            # a leg unresolved -> ticket pending, not graded
            "value|k1": self._ticket("value", [self._leg("A", "B", "1X2", "1"),
                                               self._leg("E", "F", "1X2", "1")],
                                     prob=0.4, market_odds=3.0),
        }
        outcomes = {("A", "B"): "draw", ("C", "D"): "home"}
        rec = self._run(archive, outcomes, tmp_path, monkeypatch)
        assert rec["n_graded"] == 2
        safe = rec["tiers"]["safe"]
        assert (safe["hits"], safe["n"]) == (1, 1)
        assert safe["profit"] == pytest.approx(1.0)   # 2.0 odds, 1u stake
        assert safe["expected_hit_rate"] == pytest.approx(0.6)
        fav = rec["tiers"]["favorites"]
        assert (fav["hits"], fav["n"]) == (0, 1)
        assert fav["roi"] == pytest.approx(-1.0)
        assert "value" not in rec["tiers"]  # pending, not graded as a miss

    def test_post_kickoff_stamped_ticket_never_grades(self, tmp_path, monkeypatch) -> None:
        archive = {"safe|k1": self._ticket(
            "safe", [self._leg("A", "B", "1X2", "1"), self._leg("C", "D", "1X2", "1")],
            prob=0.5, market_odds=2.5,
            archived="2026-06-15T18:00:00+00:00",  # stamped AT kickoff
        )}
        rec = self._run(archive, {("A", "B"): "home", ("C", "D"): "home"},
                        tmp_path, monkeypatch)
        assert rec["n_graded"] == 0

    def test_leg_hit_matrix(self) -> None:
        from scripts.worldcup.combos import _leg_hit

        assert _leg_hit("1X2", "1", "home") and not _leg_hit("1X2", "1", "draw")
        assert _leg_hit("1X2", "X", "draw") and _leg_hit("1X2", "2", "away")
        assert _leg_hit("Double Chance", "1X", "draw")
        assert _leg_hit("Double Chance", "X2", "away")
        assert not _leg_hit("Double Chance", "X2", "home")
        assert _leg_hit("Double Chance", "12", "home")
        assert not _leg_hit("Double Chance", "12", "draw")


class TestKnockoutFill:
    """Real-results bracket fill (scripts.worldcup.knockout) — injected lookups."""

    A_TEAMS = ["Alpha", "Beta", "Gamma", "Delta"]

    @staticmethod
    def _round_robin(letter: str, teams: list[str], start_mn: int) -> list[dict]:
        out, mn = [], start_mn
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                out.append({"match_number": mn, "stage": "group", "group": letter,
                            "date_utc": "2026-06-20T18:00:00Z",
                            "home": teams[i], "away": teams[j]})
                mn += 1
        return out

    def test_complete_group_fills_rank_slots_partial_group_waits(self) -> None:
        from scripts.worldcup import knockout as ko

        fx = self._round_robin("A", self.A_TEAMS, 1) + \
            self._round_robin("B", ["E1", "E2", "E3", "E4"], 7)
        results = {
            ("Alpha", "Beta"): (2, 0), ("Alpha", "Gamma"): (1, 0),
            ("Alpha", "Delta"): (3, 1), ("Beta", "Gamma"): (2, 1),
            ("Beta", "Delta"): (1, 0), ("Gamma", "Delta"): (1, 0),
            ("E1", "E2"): (1, 0),  # group B: 1 of 6 played
        }
        tables = ko.group_tables(fx, lambda h, a, d: results.get((h, a)))
        assert tables["A"]["complete"] and not tables["B"]["complete"]
        elo = {t: 1500.0 for t in self.A_TEAMS}
        rankings = ko.rank_complete_groups(tables, elo)
        assert rankings["A"] == ["Alpha", "Beta", "Gamma", "Delta"]
        assert "B" not in rankings  # partial groups never rank

        r32 = [{"match_number": 73, "stage": "round_of_32",
                "date_utc": "2026-06-28T19:00:00Z", "home": "1A", "away": "2B"},
               {"match_number": 74, "stage": "round_of_32",
                "date_utc": "2026-06-28T22:00:00Z", "home": "2A", "away": "1B"}]
        changes = ko.resolve_slots(r32, rankings, None, lambda mn: None)
        filled = {(c["match_number"], c["side"]): c["team"]
                  for c in changes if "team" in c}
        assert filled == {(73, "home"): "Alpha", (74, "home"): "Beta"}
        assert r32[0]["home"] == "Alpha" and r32[0]["slot_home"] == "1A"
        assert r32[0]["away"] == "2B"  # group B must wait

    def test_winner_and_loser_slots_with_shootout(self) -> None:
        from scripts.worldcup import knockout as ko

        sf = [{"match_number": 101, "stage": "semi_final",
               "date_utc": "2026-07-14T19:00:00Z", "home": "USA", "away": "Mexico"},
              {"match_number": 102, "stage": "semi_final",
               "date_utc": "2026-07-15T19:00:00Z", "home": "Spain", "away": "France"}]
        finals = [{"match_number": 103, "stage": "third_place",
                   "date_utc": "2026-07-18T19:00:00Z", "home": "L101", "away": "L102"},
                  {"match_number": 104, "stage": "final",
                   "date_utc": "2026-07-19T19:00:00Z", "home": "W101", "away": "W102"}]
        results = {("USA", "Mexico"): (1, 1),    # level FT+ET -> penalties
                   ("Spain", "France"): (2, 1)}  # decided on the score
        decided = ko.make_match_resolver(
            {f["match_number"]: f for f in sf + finals},
            lambda h, a, d: results.get((h, a)),
            # shootouts.csv speaks CANON names — resolver must map back
            lambda h, a, d: "United States" if (h, a) == ("USA", "Mexico") else None)
        ko.resolve_slots(finals, {}, None, decided)
        assert finals[1]["home"] == "USA" and finals[1]["away"] == "Spain"
        assert finals[0]["home"] == "Mexico" and finals[0]["away"] == "France"
        assert finals[0]["slot_home"] == "L101"

    def test_level_feeder_without_shootout_row_stays_open(self) -> None:
        from scripts.worldcup import knockout as ko

        sf = {"match_number": 101, "stage": "semi_final",
              "date_utc": "2026-07-14T19:00:00Z", "home": "USA", "away": "Mexico"}
        final = [{"match_number": 104, "stage": "final",
                  "date_utc": "2026-07-19T19:00:00Z", "home": "W101", "away": "W102"}]
        decided = ko.make_match_resolver(
            {101: sf}, lambda h, a, d: (0, 0), lambda h, a, d: None)
        changes = ko.resolve_slots(final, {}, None, decided)
        assert changes == [] and final[0]["home"] == "W101"

    def test_filled_slot_is_frozen(self) -> None:
        from scripts.worldcup import knockout as ko

        fx = [{"match_number": 73, "stage": "round_of_32",
               "date_utc": "2026-06-28T19:00:00Z",
               "home": "Alpha", "slot_home": "1A", "away": "2B"}]
        # New (different!) rankings must not touch the already-filled side.
        changes = ko.resolve_slots(fx, {"A": ["Zeta", "Eta", "Theta", "Iota"]},
                                   None, lambda mn: None)
        assert changes == [] and fx[0]["home"] == "Alpha"

    def test_third_slot_respects_annex_pool(self) -> None:
        from scripts.worldcup import knockout as ko

        rankings = {"A": ["Alpha", "Beta", "Gamma", "Delta"]}
        fx = [{"match_number": 74, "stage": "round_of_32",
               "date_utc": "2026-06-29T19:00:00Z", "home": "1E", "away": "3ABCDF"}]
        # Annex C assigns group A's third here, A is in the slot pool -> fills
        changes = ko.resolve_slots(fx, rankings, {74: "A"}, lambda mn: None)
        assert fx[0]["away"] == "Gamma"
        assert any(c.get("team") == "Gamma" for c in changes)
        # A letter OUTSIDE the slot pool is a config bug -> loud error, no fill
        fx2 = [{"match_number": 74, "stage": "round_of_32",
                "date_utc": "2026-06-29T19:00:00Z", "home": "1E", "away": "3ABCDF"}]
        changes2 = ko.resolve_slots(fx2, {"G": ["X1", "X2", "X3", "X4"]},
                                    {74: "G"}, lambda mn: None)
        assert fx2[0]["away"] == "3ABCDF"
        assert any("error" in c for c in changes2)

    def test_loser_label_parses(self) -> None:
        ref = parse_slot("L101")
        assert ref.kind == "loser" and ref.match_number == 101
        ref2 = parse_slot("Loser of Match 102")
        assert ref2.kind == "loser" and ref2.match_number == 102


class TestWorldCupDigest:
    """Telegram /wc digest (telegram_bot._handle_worldcup) — injected files."""

    def test_digest_renders_slate_and_combos(self, tmp_path, monkeypatch) -> None:
        import json
        from datetime import UTC, datetime, timedelta
        from zoneinfo import ZoneInfo

        import scripts.worldcup.combos as cb
        from scripts.pipeline.telegram_bot import _handle_worldcup

        now = datetime.now(UTC)
        k1 = (now + timedelta(minutes=30)).replace(microsecond=0)
        k2 = (now + timedelta(hours=30)).replace(microsecond=0)

        def pred(n, home, away, ko):
            return {"match_number": n, "match": f"{home} vs {away}",
                    "home_team": home, "away_team": away,
                    "date": ko.date().isoformat(),
                    "time": ko.strftime("%H:%M"),
                    "kickoff_utc": ko.isoformat(),
                    "probabilities": {"home": 0.6, "draw": 0.25, "away": 0.15}}

        (tmp_path / "preds.json").write_text(json.dumps(
            {"predictions": [pred(1, "Foo", "Bar", k1), pred(2, "Baz", "Qux", k2)]}))
        (tmp_path / "odds.json").write_text(json.dumps({}))
        monkeypatch.setattr(cb, "PREDICTIONS_JSON", tmp_path / "preds.json")
        monkeypatch.setattr(cb, "MARKET_ODDS_JSON", tmp_path / "odds.json")

        text = _handle_worldcup()
        assert "World Cup 2026" in text
        assert "Best combos" in text
        # both matches are inside the 48h combo window -> legs reference them
        assert "Foo or draw" in text and "Baz or draw" in text
        # today-section content mirrors the Rome-date check the digest uses
        rome = ZoneInfo("Europe/Rome")
        if k1.astimezone(rome).date() == now.astimezone(rome).date():
            assert "Foo–Bar" in text
        else:
            assert "No matches today" in text
