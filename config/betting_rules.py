"""Loader and filter logic for `config/betting_rules.json`.

This is the bridge between the walk-forward-validated strategy (saved as JSON
in betting_rules.json) and the existing betting code. The betting engine calls
`should_bet()` to decide whether a candidate bet survives all filters.

No iteration, no hidden state — everything is declarative in the JSON and
the logic here only applies what the JSON says.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

RULES_PATH = Path(__file__).resolve().parent / "betting_rules.json"


@dataclass(frozen=True)
class BetCandidate:
    """Minimal context needed to evaluate a bet against the rules."""
    league: str                 # "serie_a" or "premier_league"
    market: str                 # "1X2", "O/U_2.5", "BTTS", ...
    selection: str              # "H" / "D" / "A" for 1X2, "Over" / "Under" for O/U, etc.
    model_prob: float           # Our walk-forward model's probability for this selection
    odds: float                 # The odds we'd actually bet at (best available)
    odds_source: str            # e.g. "market_max", "pinnacle_close", "b365_open"
    fair_implied_prob: float    # Overround-stripped implied prob from the odds-triplet


@dataclass(frozen=True)
class BetDecision:
    """Output of rules evaluation."""
    accept: bool
    reason: str
    edge_pct: float                       # Model edge vs fair implied, in percentage points
    stake_fraction_of_bankroll: float = 0.0
    # Present only if accept=True; stake cap has been applied.


class BettingRules:
    """Encapsulates the betting_rules.json contract."""

    def __init__(self, rules: Optional[dict] = None):
        self._rules = rules if rules is not None else self._load_default()
        self._version = (self._rules.get("_meta") or {}).get("version", "unknown")

    @classmethod
    def _load_default(cls) -> dict:
        if not RULES_PATH.exists():
            raise FileNotFoundError(
                f"betting_rules.json not found at {RULES_PATH}. "
                "The betting pipeline requires an explicit rules file."
            )
        with open(RULES_PATH) as f:
            return json.load(f)

    @property
    def version(self) -> str:
        return self._version

    def global_config(self) -> dict:
        return self._rules.get("global", {})

    def market_config(self, market: str) -> dict:
        return (self._rules.get("markets") or {}).get(market, {})

    def is_enabled(self, market: str, league: str) -> bool:
        cfg = self.market_config(market)
        return league in (cfg.get("enabled_leagues") or [])

    def kelly_fraction(self) -> float:
        return float(self.global_config().get("staking", {}).get("kelly_fraction", 0.25))

    def max_bet_pct(self) -> float:
        return float(self.global_config().get("staking", {}).get("max_bet_pct_of_bankroll", 0.02))

    def min_bet_pct(self) -> float:
        return float(self.global_config().get("staking", {}).get("min_bet_pct_of_bankroll", 0.001))

    def preferred_odds_source(self) -> str:
        return self.global_config().get("preferred_odds_source", "market_max")

    def paper_trade_active(self, live_bets_for_market: int) -> bool:
        """Return True if the paper-trade gate is still in effect for this
        (league, market) combo.
        """
        g = self.global_config().get("paper_trade_gate", {})
        if not g.get("enabled_by_default", True):
            return False
        require = int(g.get("require_n_live_bets_per_league_market", 100))
        return live_bets_for_market < require

    def evaluate(self, bet: BetCandidate, bankroll: float,
                 live_bets_for_market: int = 0) -> BetDecision:
        """Evaluate one bet candidate against all active rules."""
        edge_pct = (bet.model_prob - bet.fair_implied_prob) / bet.fair_implied_prob * 100.0

        # 1. Market must be enabled for this league.
        if not self.is_enabled(bet.market, bet.league):
            return BetDecision(False, f"market {bet.market} disabled for {bet.league}", edge_pct)

        mcfg = self.market_config(bet.market)

        # 2. Odds source must be the preferred or an allowed fallback.
        preferred = self.preferred_odds_source()
        fallbacks = self.global_config().get("fallback_odds_sources_ordered") or []
        allowed_sources = {preferred, *fallbacks}
        if allowed_sources and bet.odds_source not in allowed_sources:
            return BetDecision(False, f"odds source {bet.odds_source} not in allowed set", edge_pct)

        # 3. Per-league minimum edge (plus any per-class bonus).
        per_league = (mcfg.get("per_league") or {}).get(bet.league, {})
        min_edge = float(per_league.get("min_edge_pct", 0.0))
        class_bonus = 0.0
        class_overrides = mcfg.get("class_overrides") or {}
        if bet.selection in class_overrides:
            class_bonus = float(class_overrides[bet.selection].get("edge_reduction_bonus_pct", 0.0))
        effective_min_edge = min_edge + class_bonus  # bonus is negative to LOWER the threshold
        if edge_pct < effective_min_edge:
            return BetDecision(False, f"edge {edge_pct:.2f}% < threshold {effective_min_edge:.2f}%", edge_pct)

        # 4. Allowed classes (e.g., only bet Draws, or allow all).
        allowed_classes = per_league.get("allowed_classes")
        if allowed_classes is not None and bet.selection not in allowed_classes:
            return BetDecision(False, f"selection {bet.selection} not in allowed classes", edge_pct)

        # 5. Subset filters — odds band and model-prob band.
        subset = mcfg.get("subset_filters") or {}
        odds_band = subset.get("odds_band") or {}
        if "min" in odds_band and bet.odds < float(odds_band["min"]):
            return BetDecision(False, f"odds {bet.odds:.2f} below band min {odds_band['min']}", edge_pct)
        if "max" in odds_band and bet.odds > float(odds_band["max"]):
            return BetDecision(False, f"odds {bet.odds:.2f} above band max {odds_band['max']}", edge_pct)
        prob_band = subset.get("model_prob_band") or {}
        if "min" in prob_band and bet.model_prob < float(prob_band["min"]):
            return BetDecision(False, f"model prob {bet.model_prob:.3f} below band min {prob_band['min']}", edge_pct)
        if "max" in prob_band and bet.model_prob > float(prob_band["max"]):
            return BetDecision(False, f"model prob {bet.model_prob:.3f} above band max {prob_band['max']}", edge_pct)

        # 6. Exclusion filters (hard rejections).
        excl = mcfg.get("exclusion_filters") or {}
        if "longshot_odds_above" in excl and bet.odds > float(excl["longshot_odds_above"]):
            return BetDecision(False, f"odds {bet.odds:.2f} > longshot cap {excl['longshot_odds_above']}", edge_pct)
        if "model_prob_below" in excl and bet.model_prob < float(excl["model_prob_below"]):
            return BetDecision(False, f"model prob {bet.model_prob:.3f} < cap {excl['model_prob_below']}", edge_pct)
        # high-prob-without-strong-edge gate (only reject if prob high AND edge modest)
        if ("model_prob_above_without_strong_edge" in excl
                and bet.model_prob > float(excl["model_prob_above_without_strong_edge"])
                and edge_pct < 10.0):
            return BetDecision(False,
                               f"high prob {bet.model_prob:.3f} without strong edge (edge {edge_pct:.2f}%)",
                               edge_pct)

        # 7. Kelly stake.
        b = bet.odds - 1.0
        if b <= 0:
            return BetDecision(False, f"invalid odds {bet.odds}", edge_pct)
        full_kelly = (b * bet.model_prob - (1 - bet.model_prob)) / b
        full_kelly = max(0.0, full_kelly)
        frac = full_kelly * self.kelly_fraction()
        frac = min(frac, self.max_bet_pct())

        # 8. Paper-trade gate: halve the stake if we haven't hit 100 live bets yet.
        if self.paper_trade_active(live_bets_for_market):
            frac = frac * 0.1  # 10x reduction — paper-trade-lite

        if frac < self.min_bet_pct():
            return BetDecision(False, f"kelly stake {frac*100:.3f}% below min", edge_pct)

        return BetDecision(True, "accepted", edge_pct, stake_fraction_of_bankroll=frac)


# Convenience module-level singleton for callers that want a simple API.
_default_rules: Optional[BettingRules] = None


def get_rules() -> BettingRules:
    global _default_rules
    if _default_rules is None:
        _default_rules = BettingRules()
    return _default_rules


def should_bet(bet: BetCandidate, bankroll: float,
               live_bets_for_market: int = 0) -> BetDecision:
    """One-shot entry point: evaluate a bet against the default loaded rules."""
    return get_rules().evaluate(bet, bankroll, live_bets_for_market)
