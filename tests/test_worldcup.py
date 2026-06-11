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
            parse_slot("L101")  # loser refs are not simulated


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
