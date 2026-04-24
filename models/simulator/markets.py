"""Market probability queries against a MatchSimulation.

Given a simulator output, produce per-market probabilities in a dict that
maps `market_label` → probability (for binary markets) or (classes, probs)
tuple (for multiclass).

The same `market_label` keys used in models.simulator.backtests.harness.
"""
from __future__ import annotations

from models.simulator.engine.simulator import MatchSimulation


def all_player_market_probs(sim: MatchSimulation) -> dict[str, float]:
    """Phase 5 player props. Returns {} if no players simulated."""
    out: dict[str, float] = {}
    if not sim.player_goals:
        return out
    for pid_str in sim.player_goals:
        pid = int(pid_str)
        out[f"player_{pid}_anytime_scorer"] = sim.p_player_anytime_scorer(pid)
        out[f"player_{pid}_shots_over_0_5"] = sim.p_player_shots_over(pid, 0.5)
        out[f"player_{pid}_shots_over_1_5"] = sim.p_player_shots_over(pid, 1.5)
        out[f"player_{pid}_shots_over_2_5"] = sim.p_player_shots_over(pid, 2.5)
        out[f"player_{pid}_sot_over_0_5"] = sim.p_player_sot_over(pid, 0.5)
        out[f"player_{pid}_sot_over_1_5"] = sim.p_player_sot_over(pid, 1.5)
    return out


def all_phase2_market_probs(sim: MatchSimulation) -> dict[str, float]:
    """Phase 2 markets (corners, cards, shots, SOT). Returns {} if arrays
    are not populated — safe to call on a Phase 1 simulation."""
    out: dict[str, float] = {}
    if sim.home_corners is not None and sim.away_corners is not None:
        for line in (7.5, 8.5, 9.5, 10.5, 11.5, 12.5):
            key = f"corners_over_{str(line).replace('.', '_')}"
            out[key] = sim.p_corners_over(line, "both")
        for side in ("home", "away"):
            for line in (3.5, 4.5, 5.5):
                out[f"{side}_corners_over_{str(line).replace('.', '_')}"] = sim.p_corners_over(line, side)
    if sim.home_cards is not None and sim.away_cards is not None:
        for line in (2.5, 3.5, 4.5, 5.5, 6.5):
            key = f"cards_over_{str(line).replace('.', '_')}"
            out[key] = sim.p_cards_over(line, "both")
        for side in ("home", "away"):
            for line in (1.5, 2.5):
                out[f"{side}_cards_over_{str(line).replace('.', '_')}"] = sim.p_cards_over(line, side)
    if sim.home_shots is not None and sim.away_shots is not None:
        for line in (20.5, 22.5, 24.5, 26.5):
            out[f"shots_over_{str(line).replace('.', '_')}"] = sim.p_shots_over(line, "both")
    if sim.home_sot is not None and sim.away_sot is not None:
        for line in (7.5, 8.5, 9.5):
            out[f"sot_over_{str(line).replace('.', '_')}"] = sim.p_sot_over(line, "both")
    return out


def all_goal_market_probs(sim: MatchSimulation) -> dict[str, float | dict]:
    """Every market derivable from goals alone."""
    return {
        # 1X2
        "1X2": {"H": sim.p_home_win(), "D": sim.p_draw(), "A": sim.p_away_win()},
        # Double chance
        "DC_1X": sim.p_double_chance("1X"),
        "DC_X2": sim.p_double_chance("X2"),
        "DC_12": sim.p_double_chance("12"),
        # Draw no bet
        "DNB_home": sim.p_draw_no_bet("home"),
        "DNB_away": sim.p_draw_no_bet("away"),
        # O/U (common lines)
        "O/U 0.5": sim.p_over(0.5),
        "O/U 1.5": sim.p_over(1.5),
        "O/U 2.0": sim.p_over(2.0),
        "O/U 2.5": sim.p_over(2.5),
        "O/U 3.0": sim.p_over(3.0),
        "O/U 3.5": sim.p_over(3.5),
        "O/U 4.5": sim.p_over(4.5),
        # BTTS
        "BTTS": sim.p_btts(yes=True),
        # Clean sheets
        "home_clean_sheet": sim.p_clean_sheet("home"),
        "away_clean_sheet": sim.p_clean_sheet("away"),
        # Asian handicaps (standard lines)
        "AH_home_-2.5": sim.p_handicap(-2.5, "home"),
        "AH_home_-1.5": sim.p_handicap(-1.5, "home"),
        "AH_home_-0.5": sim.p_handicap(-0.5, "home"),
        "AH_home_+0.5": sim.p_handicap(+0.5, "home"),
        "AH_home_+1.5": sim.p_handicap(+1.5, "home"),
        "AH_home_+2.5": sim.p_handicap(+2.5, "home"),
        # Goal ranges
        "goals_0_1": sim.p_total_between(0, 1),
        "goals_2_3": sim.p_total_between(2, 3),
        "goals_4_plus": 1.0 - sim.p_total_between(0, 3),
    }
