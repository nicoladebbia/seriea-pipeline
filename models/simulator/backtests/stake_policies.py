"""Stake policies for the backtest harness.

Three policies, always reported side-by-side:

- Flat(€10) — measures prediction quality independent of bankroll dynamics.
- Kelly(fractional, floor/ceiling, bankroll-growth OFF by default) — measures
  "what would I actually book." Matches the live `betting_unified.py` rules.
- NoStake — shadow-mode only; logs the bet decision without sizing.

Kelly by default keeps bankroll FIXED across the backtest so each bet is
evaluated independently — this isolates the predictor from bankroll-growth
variance. Set `allow_bankroll_growth=True` if you want realistic compounding
(with the caveat that bad predictors blow up and good ones explode, making
comparisons noisier).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class BetInput:
    p: float           # predicted probability of the selected outcome
    odds: float        # decimal odds of the selected outcome
    edge_pct: float    # computed edge in percentage points


class StakePolicy(ABC):
    name: str

    @abstractmethod
    def stake(self, bet: BetInput, bankroll_eur: float) -> float:
        """Return stake in EUR. May be 0 if policy declines the bet."""

    def label(self) -> str:
        return self.name


class FlatStake(StakePolicy):
    name = "flat"

    def __init__(self, amount_eur: float = 10.0):
        self.amount_eur = amount_eur

    def stake(self, bet: BetInput, bankroll_eur: float) -> float:
        return self.amount_eur


class KellyStake(StakePolicy):
    """Fractional Kelly mirroring betting_unified.py production sizing."""
    name = "kelly"

    def __init__(
        self,
        fraction: float = 0.25,        # quarter Kelly
        floor_eur: float = 2.0,
        ceiling_eur: float = 50.0,
    ):
        self.fraction = fraction
        self.floor_eur = floor_eur
        self.ceiling_eur = ceiling_eur

    def stake(self, bet: BetInput, bankroll_eur: float) -> float:
        b = bet.odds - 1.0
        if b <= 0:
            return 0.0
        p = bet.p
        q = 1.0 - p
        kelly_full = (b * p - q) / b
        if kelly_full <= 0:
            return 0.0
        kelly_fractional = kelly_full * self.fraction
        raw_stake = bankroll_eur * kelly_fractional
        return float(max(self.floor_eur, min(self.ceiling_eur, raw_stake)))


class NoStake(StakePolicy):
    """Shadow-mode: decision logged, zero monetary exposure."""
    name = "no_stake"

    def stake(self, bet: BetInput, bankroll_eur: float) -> float:
        return 0.0


DEFAULT_POLICIES: list[StakePolicy] = [FlatStake(10.0), KellyStake(0.25, 2.0, 50.0)]
