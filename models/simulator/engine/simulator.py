"""Core match simulator — draws n_trials joint outcomes per match.

Phase 1 scope: goals only. Phase 2 extends with shots, corners, cards.
Phase 5 extends with per-player props.

Usage:
    from models.simulator.engine.simulator import simulate_match
    sim = simulate_match(lambda_home=1.3, lambda_away=1.1,
                         tau=0.06, n_trials=10_000, seed=42)
    sim.p_over(2.5)  # → 0.51
    sim.p_home_win()  # → 0.42
    sim.p_btts()      # → 0.53

The MatchSimulation dataclass stores the full trial array for downstream
market queries. Adding a new market means adding a counting method on the
trials; no retraining.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .dixon_coles import sample_dc_goals


@dataclass
class MatchSimulation:
    home_goals: np.ndarray        # shape (n_trials,) int
    away_goals: np.ndarray        # shape (n_trials,)
    lambda_home: float             # rate used to generate
    lambda_away: float
    tau: float
    n_trials: int
    seed: int

    # Reserved for Phase 2+ extensions
    home_shots: np.ndarray | None = None
    away_shots: np.ndarray | None = None
    home_sot: np.ndarray | None = None
    away_sot: np.ndarray | None = None
    home_corners: np.ndarray | None = None
    away_corners: np.ndarray | None = None
    home_cards: np.ndarray | None = None
    away_cards: np.ndarray | None = None

    # Phase 2b — first-half splits
    fh_home_goals: np.ndarray | None = None
    fh_away_goals: np.ndarray | None = None

    # Phase 5 — per-player
    player_goals: dict[str, np.ndarray] = field(default_factory=dict)
    player_shots: dict[str, np.ndarray] = field(default_factory=dict)
    player_sot: dict[str, np.ndarray] = field(default_factory=dict)

    # ----- 1X2 markets -----
    def p_home_win(self) -> float:
        return float(np.mean(self.home_goals > self.away_goals))

    def p_draw(self) -> float:
        return float(np.mean(self.home_goals == self.away_goals))

    def p_away_win(self) -> float:
        return float(np.mean(self.home_goals < self.away_goals))

    def p_double_chance(self, which: Literal["1X", "X2", "12"]) -> float:
        if which == "1X":
            return float(np.mean(self.home_goals >= self.away_goals))
        if which == "X2":
            return float(np.mean(self.home_goals <= self.away_goals))
        if which == "12":
            return float(np.mean(self.home_goals != self.away_goals))
        raise ValueError(which)

    def p_draw_no_bet(self, side: Literal["home", "away"]) -> float:
        """Probability of `side` winning among non-draw matches."""
        non_draw = self.home_goals != self.away_goals
        if not non_draw.any():
            return 0.5
        if side == "home":
            return float(np.mean(self.home_goals[non_draw] > self.away_goals[non_draw]))
        return float(np.mean(self.home_goals[non_draw] < self.away_goals[non_draw]))

    # ----- O/U markets -----
    def p_over(self, line: float) -> float:
        total = self.home_goals + self.away_goals
        return float(np.mean(total > line))

    def p_under(self, line: float) -> float:
        total = self.home_goals + self.away_goals
        return float(np.mean(total < line))

    def p_total_between(self, lo: int, hi: int) -> float:
        """P(lo ≤ total ≤ hi) — for goal-range markets."""
        total = self.home_goals + self.away_goals
        return float(np.mean((total >= lo) & (total <= hi)))

    # ----- BTTS + clean sheets -----
    def p_btts(self, yes: bool = True) -> float:
        both = (self.home_goals > 0) & (self.away_goals > 0)
        return float(np.mean(both)) if yes else float(np.mean(~both))

    def p_clean_sheet(self, side: Literal["home", "away"]) -> float:
        if side == "home":
            return float(np.mean(self.away_goals == 0))
        return float(np.mean(self.home_goals == 0))

    # ----- Asian handicap -----
    def p_handicap(self, line: float, side: Literal["home", "away"]) -> float:
        """Probability that `side + line` beats opposite side."""
        if side == "home":
            diff = (self.home_goals + line) - self.away_goals
        else:
            diff = (self.away_goals + line) - self.home_goals
        return float(np.mean(diff > 0))

    # ----- Exact score -----
    def p_exact_score(self, h: int, a: int) -> float:
        return float(np.mean((self.home_goals == h) & (self.away_goals == a)))

    def top_scores(self, k: int = 10) -> list[tuple[tuple[int, int], float]]:
        """Return [(score, probability)] for the k most likely scores."""
        scores = np.array([self.home_goals, self.away_goals]).T
        # Tupleize and count
        uniq, counts = np.unique(scores, axis=0, return_counts=True)
        probs = counts / self.n_trials
        order = np.argsort(-probs)[:k]
        return [((int(uniq[i, 0]), int(uniq[i, 1])), float(probs[i])) for i in order]

    # ----- Phase 2 markets (corners, cards) — available once arrays populated -----
    def p_corners_over(self, line: float, team: Literal["home", "away", "both"] = "both") -> float:
        if self.home_corners is None or self.away_corners is None:
            raise AttributeError("Corner data not populated — run Phase 2 corners_rates first")
        if team == "home":
            return float(np.mean(self.home_corners > line))
        if team == "away":
            return float(np.mean(self.away_corners > line))
        return float(np.mean(self.home_corners + self.away_corners > line))

    def p_cards_over(self, line: float, team: Literal["home", "away", "both"] = "both") -> float:
        if self.home_cards is None or self.away_cards is None:
            raise AttributeError("Card data not populated — run Phase 2 card_rates first")
        if team == "home":
            return float(np.mean(self.home_cards > line))
        if team == "away":
            return float(np.mean(self.away_cards > line))
        return float(np.mean(self.home_cards + self.away_cards > line))

    def p_shots_over(self, line: float, team: Literal["home", "away", "both"] = "both") -> float:
        if self.home_shots is None or self.away_shots is None:
            raise AttributeError("Shot data not populated — pass shot_rate_* to simulate_match")
        if team == "home":
            return float(np.mean(self.home_shots > line))
        if team == "away":
            return float(np.mean(self.away_shots > line))
        return float(np.mean(self.home_shots + self.away_shots > line))

    def p_sot_over(self, line: float, team: Literal["home", "away", "both"] = "both") -> float:
        if self.home_sot is None or self.away_sot is None:
            raise AttributeError("SOT data not populated — pass shot_rate_* to simulate_match")
        if team == "home":
            return float(np.mean(self.home_sot > line))
        if team == "away":
            return float(np.mean(self.away_sot > line))
        return float(np.mean(self.home_sot + self.away_sot > line))

    # ----- Phase 5: per-player markets -----
    def p_player_anytime_scorer(self, player_id: int | str) -> float:
        key = str(int(player_id))
        g = self.player_goals.get(key)
        if g is None:
            return 0.0
        return float(np.mean(g >= 1))

    def p_player_shots_over(self, player_id: int | str, line: float) -> float:
        key = str(int(player_id))
        s = self.player_shots.get(key)
        if s is None:
            return 0.0
        return float(np.mean(s > line))

    def p_player_sot_over(self, player_id: int | str, line: float) -> float:
        key = str(int(player_id))
        sot = self.player_sot.get(key)
        if sot is None:
            return 0.0
        return float(np.mean(sot > line))

    def top_scorers(self, k: int = 3) -> list[tuple[int, float]]:
        """Return [(player_id, P(anytime_scorer))] top-k."""
        rows = []
        for key, g in self.player_goals.items():
            p = float(np.mean(g >= 1))
            rows.append((int(key), p))
        rows.sort(key=lambda x: -x[1])
        return rows[:k]

    def expected_player_goals(self, player_id: int | str) -> float:
        key = str(int(player_id))
        g = self.player_goals.get(key)
        if g is None:
            return 0.0
        return float(np.mean(g))

    # Mean predictors (useful for sanity checks + "expected X" markets)
    def expected_corners(self, team: Literal["home", "away", "both"] = "both") -> float:
        if self.home_corners is None or self.away_corners is None:
            return float("nan")
        if team == "home":
            return float(np.mean(self.home_corners))
        if team == "away":
            return float(np.mean(self.away_corners))
        return float(np.mean(self.home_corners + self.away_corners))

    def expected_cards(self, team: Literal["home", "away", "both"] = "both") -> float:
        if self.home_cards is None or self.away_cards is None:
            return float("nan")
        if team == "home":
            return float(np.mean(self.home_cards))
        if team == "away":
            return float(np.mean(self.away_cards))
        return float(np.mean(self.home_cards + self.away_cards))

    # ----- Consistency check -----
    def assert_consistency(self, tol: float = 1e-6) -> None:
        """P(H) + P(D) + P(A) must equal 1.0 within tol."""
        s = self.p_home_win() + self.p_draw() + self.p_away_win()
        assert abs(s - 1.0) < tol, f"1X2 probs sum to {s}, expected 1"


def _match_seed(match_id: str | None, base_seed: int) -> int:
    """Derive a deterministic seed from match_id — reproducible simulations."""
    if match_id is None:
        return int(base_seed)
    h = hashlib.md5(match_id.encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="big") ^ int(base_seed)


def simulate_match(
    lambda_home: float,
    lambda_away: float,
    tau: float = 0.0,
    n_trials: int = 10_000,
    seed: int = 0,
    match_id: str | None = None,
    corner_rate_home: float | None = None,
    corner_rate_away: float | None = None,
    card_rate_home: float | None = None,
    card_rate_away: float | None = None,
    shot_rate_home: float | None = None,
    shot_rate_away: float | None = None,
    sot_ratio_home: float = 0.35,
    sot_ratio_away: float = 0.33,
    player_profiles_home: dict | None = None,
    player_profiles_away: dict | None = None,
    player_shot_shares_home: dict[int, float] | None = None,
    player_shot_shares_away: dict[int, float] | None = None,
) -> MatchSimulation:
    """Simulate a match.

    Phase 1 args (always required):
      lambda_home, lambda_away: per-team goal Poisson rates
      tau: Dixon-Coles correction (0.0 = independent Poisson)
      n_trials: number of Monte Carlo draws
      seed: base random seed (combined with match_id for per-match determinism)
      match_id: optional ID for seed mixing

    Phase 2 args (optional, populate extended markets):
      corner_rate_*, card_rate_*, shot_rate_* — per-team Poisson rates
      sot_ratio_* — empirical SOT-per-shot ratios (binomial parameter)

    Any Phase 2 rate that is None leaves the corresponding MatchSimulation
    array set to None (downstream market queries raise AttributeError, as
    documented on the dataclass methods).
    """
    actual_seed = _match_seed(match_id, seed)
    rng = np.random.default_rng(actual_seed)
    lh = np.full(n_trials, float(lambda_home))
    la = np.full(n_trials, float(lambda_away))
    if tau != 0.0:
        h, a = sample_dc_goals(lh, la, tau, rng)
    else:
        h = rng.poisson(lh)
        a = rng.poisson(la)

    # Independent Poisson draws for corners, cards, shots (per trial).
    # These are correlated with goals in reality; explicit coupling can be
    # added in Phase 2b if the simple independent draws mispredict joint
    # markets (e.g. "home win AND corners over 9.5").
    home_corners = (rng.poisson(np.full(n_trials, float(corner_rate_home)))
                    if corner_rate_home is not None else None)
    away_corners = (rng.poisson(np.full(n_trials, float(corner_rate_away)))
                    if corner_rate_away is not None else None)
    home_cards = (rng.poisson(np.full(n_trials, float(card_rate_home)))
                  if card_rate_home is not None else None)
    away_cards = (rng.poisson(np.full(n_trials, float(card_rate_away)))
                  if card_rate_away is not None else None)
    home_shots = (rng.poisson(np.full(n_trials, float(shot_rate_home)))
                  if shot_rate_home is not None else None)
    away_shots = (rng.poisson(np.full(n_trials, float(shot_rate_away)))
                  if shot_rate_away is not None else None)
    # SOT: binomial given shots
    home_sot = (rng.binomial(home_shots, float(sot_ratio_home))
                if home_shots is not None else None)
    away_sot = (rng.binomial(away_shots, float(sot_ratio_away))
                if away_shots is not None else None)

    # Phase 5 — per-player sampling. For each player with a shot share,
    # draw shots ~ Poisson(share), then SOT | shots ~ Bin(sot_per_shot),
    # then goals | SOT ~ Bin(goals_per_sot).
    player_shots: dict[str, np.ndarray] = {}
    player_sot: dict[str, np.ndarray] = {}
    player_goals: dict[str, np.ndarray] = {}

    def _simulate_players(shares: dict[int, float] | None, profiles: dict | None) -> None:
        if not shares or not profiles:
            return
        for pid, expected_shots in shares.items():
            if expected_shots <= 0:
                continue
            prof = profiles.get(int(pid))
            if prof is None:
                continue
            # Draw shots ~ Poisson(expected shots)
            s = rng.poisson(np.full(n_trials, float(expected_shots)))
            # SOT given shots
            sot = rng.binomial(s, float(prof.sot_per_shot))
            # Goals given SOT
            g = rng.binomial(sot, float(prof.goals_per_sot))
            key = str(int(pid))
            player_shots[key] = s
            player_sot[key] = sot
            player_goals[key] = g

    _simulate_players(player_shot_shares_home, player_profiles_home)
    _simulate_players(player_shot_shares_away, player_profiles_away)

    return MatchSimulation(
        home_goals=h,
        away_goals=a,
        lambda_home=float(lambda_home),
        lambda_away=float(lambda_away),
        tau=float(tau),
        n_trials=int(n_trials),
        seed=actual_seed,
        home_corners=home_corners,
        away_corners=away_corners,
        home_cards=home_cards,
        away_cards=away_cards,
        home_shots=home_shots,
        away_shots=away_shots,
        home_sot=home_sot,
        away_sot=away_sot,
        player_shots=player_shots,
        player_sot=player_sot,
        player_goals=player_goals,
    )
