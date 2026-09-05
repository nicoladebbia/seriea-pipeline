#!/usr/bin/env python3
"""
UNIFIED BETTING ENGINE v1.0 — Consolidated Professional-Grade System

Merges ALL betting engines into a single, authoritative script:
  - betting_engine.py           (basic 1X2 singles + parlays)
  - advanced_betting_engine.py  (multi-market normalization + portfolio)
  - ultimate_betting_system.py  (sharp odds, Kelly, intelligence, Monte Carlo)
  - deep_betting_engine.py      (Poisson score matrix + isotonic calibration)
  - live_betting.py             (in-play probability adjustments)
  - bankroll_manager.py         (bankroll tracking + flat/Kelly staking)

Architecture:
  1. DATA LAYER      — Load predictions, odds (40+ bookmakers), extended markets
  2. EDGE LAYER      — Model prob vs sharp implied prob, proper overround removal
  3. ODDS LAYER      — Best odds hunting across all bookmakers per selection
  4. STAKE LAYER     — Fractional Kelly (1/4) with caps and variance reduction
  5. INTELLIGENCE    — Steam moves, sharp/soft divergence, odds movement
  6. PORTFOLIO       — Correlation handling, max exposure, two-pass diversification
  7. ACCUMULATORS    — Cross-match parlays from independent DC value bets
  8. MONTE CARLO     — Variance simulation with blended probabilities
  9. OUTPUT          — Professional bet slips with full transparency

Usage:
    python scripts/betting_unified.py                          # Full run
    python scripts/betting_unified.py --bankroll 500           # Custom bankroll
    python scripts/betting_unified.py --min-edge 4.0           # Higher edge threshold
    python scripts/betting_unified.py --kelly 0.15             # More conservative Kelly
    python scripts/betting_unified.py --update-results         # Mark bet results
    python scripts/betting_unified.py --track-record           # Show cumulative P/L
    python scripts/betting_unified.py --json                   # JSON output only
    python scripts/betting_unified.py --dry-run                # No file writes
"""

import sys
import json
import math
import logging
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

from config.settings import DATA_DIR, MODELS_DIR, UPCOMING_DIR as UPCOMING


def _league_betting_enabled(league: str) -> bool:
    """Check if a league has a validated deployment state allowing betting.

    Serie A is always enabled (established pipeline with published metrics).
    All other leagues require a deployment_state.json with betting_enabled=true.
    """
    if league == "serie_a":
        return True
    state_path = MODELS_DIR / league / "deployment_state.json"
    if not state_path.exists():
        return False
    try:
        with open(state_path) as f:
            state = json.load(f)
        return state.get("betting_enabled", False)
    except Exception:
        return False


# =============================================================================
# 0. HELPERS
# =============================================================================

def _fmt_hc(val: float) -> str:
    """Format a handicap value with proper precision and sign.

    0.25 → '+0.25', -1.0 → '-1', 0.5 → '+0.5', -0.75 → '-0.75'
    """
    # Use .2g to drop trailing zeros while keeping precision
    if val == int(val):
        return f"{int(val):+d}"
    elif val * 4 == int(val * 4):  # quarter line (0.25, 0.75)
        return f"{val:+.2f}"
    else:  # half line (0.5, 1.5)
        return f"{val:+.1f}"


# =============================================================================
# 1. CONFIGURATION — BettingConfig dataclass with ALL tunables
# =============================================================================
_LEAGUE_KELLY_DEFAULTS = {
    "serie_a": 0.05,        # Live production — calibrated Apr 2026.
    "premier_league": 0.04, # One notch tighter — pending 50+ EPL settled bets.
}


def _load_league_kelly_fraction(league: str, default: float = 0.05) -> float:
    """Return Kelly fraction for *league*. Read order:

      1. Per-league deployment_state.json[kelly_fraction] (if present)
      2. _LEAGUE_KELLY_DEFAULTS[league]
      3. caller's default
    """
    try:
        state_path = MODELS_DIR / league / "deployment_state.json"
        if state_path.exists():
            import json
            state = json.loads(state_path.read_text())
            kf = state.get("kelly_fraction")
            if kf is not None:
                return float(kf)
    except Exception:
        pass
    return _LEAGUE_KELLY_DEFAULTS.get(league, default)


@dataclass
class BettingConfig:
    """All configurable parameters for the unified betting system.

    Consolidated from ultimate_betting_system.BettingConfig,
    advanced_betting_engine constants, and betting_engine constants.
    """

    # -- Bankroll --
    bankroll: float = 1000.0

    # -- Staking mode --
    # "kelly": fractional Kelly criterion (RECOMMENDED with proper edge filters).
    #   Sizes proportional to edge: bigger bets on better opportunities, smaller on
    #   thin edges. Safe because overconfident edges are blocked upstream:
    #   - max_edge_pct=8% caps model overconfidence (live: >10% = 38% WR, -EUR133)
    #   - Intelligence filter kills bets opposed by sharp money
    #   - Per-market min_edge thresholds ensure genuine value
    # "proportional": flat % per bet (simple fallback)
    staking_mode: str = "kelly"
    proportional_stake_pct: float = 2.0  # flat % when mode=proportional

    # -- Kelly fraction --
    # AUTHORITATIVE FOR LIVE BETTING. This value (0.05) is what production
    # actually uses, set by UnifiedBettingEngine and called from
    # scripts/pipeline/run_full_pipeline.py. The 0.25 in config/betting_rules.json
    # is parallel walkforward infrastructure NOT yet wired into production.
    #
    # 5% Kelly: conservative sizing to reduce variance and drawdown.
    # At 5%, a bet with 5% edge gets ~0.25% of bankroll (EUR 2.50 on 1000).
    # At 5%, a bet with 10% edge gets ~0.75% (EUR 7.50). Scales naturally.
    # Kept conservative at 0.05 given the 2026-04-28 finding of edge erosion
    # on 2025-26 data — see docs/2026-04-28_subset_alpha_findings.md.
    kelly_fraction: float = 0.05

    # -- Edge thresholds (revised Apr 2026 — deep journal analysis, 153 bets) --
    # 3-5% edge = +113% ROI (70% WR). 5-7% = +2%. 7-10% = -1%. 10%+ = -13%.
    # Sweet spot is 3-7%. Anything above 7% is model overconfidence.
    # AUTHORITATIVE FOR LIVE BETTING — the 10% (SA) / 7% (EPL) values in
    # config/betting_rules.json are for the walkforward flow not yet in
    # production.
    min_edge_pct: float = 3.0        # Lowered from 4.0 — 3-5% is the best bucket
    max_edge_pct: float = 7.0        # Lowered from 8.0 — 7-10% edges = -1% ROI
    max_edge_draw_pct: float = 6.0   # Tightened from 7.0
    strong_edge_pct: float = 4.5     # "Strong" value threshold
    elite_edge_pct: float = 6.0      # "Elite" value threshold

    # -- Confidence-stratified edge thresholds --
    # When model is confident (prob > 55%), it's more reliable → lower edge needed.
    # When model is uncertain (prob < 40%), require higher edge for safety.
    confidence_edge_tiers: Dict = field(default_factory=lambda: {
        "high":   {"min_prob": 0.55, "min_edge_adj": -1.5},  # Lower edge threshold by 1.5pp
        "medium": {"min_prob": 0.40, "min_edge_adj":  0.0},  # Default thresholds
        "low":    {"min_prob": 0.00, "min_edge_adj": +2.0},  # Raise edge threshold by 2pp
    })

    # -- Stake limits (% of bankroll) --
    min_stake_pct: float = 0.5       # Don't place bets smaller than this
    max_stake_pct: float = 2.5       # Maximum single bet (pro standard: 2-3%)
    max_match_exposure_pct: float = 8.0   # Max total exposure on one match
    max_daily_exposure_pct: float = 35.0  # Max total exposure for the day
    max_portfolio_exposure_pct: float = 50.0  # Max TOTAL outstanding exposure (including journal pending bets)

    # -- Portfolio limits --
    max_bets_per_match: int = 1      # 1 bet per match — eliminates correlated losses (was 2)
    max_total_bets: int = 20         # Max total bets per round
    max_draw_family_bets: int = 2    # Max draw bets per round — was 3, reduced to manage variance
                                     # (31.7% WR means 6-bet avg losing streaks; cap at 2 to diversify)
    draw_stake_multiplier: float = 0.80  # Reduce draw stakes by 20% vs Kelly recommendation (variance protection)

    # -- Odds limits --
    min_odds: float = 1.25           # Don't bet below this (too low value)
    max_odds: float = 4.0            # Don't bet above this (live audit: all 5 bets >4.0 lost -€130)

    # -- Confidence adjustments --
    model_confidence_weight: float = 0.7  # How much we trust model vs market

    # -- Sharp bookmaker for true probability benchmark --
    sharp_bookmaker: str = "Pinnacle"

    # -- Accumulator / Parlay settings --
    # Phase 8 parlay expansion (Feb 19 2026):
    # Two tiers: Safe parlays (high-prob legs, 70%+ each) and Value parlays (edge-based).
    # Why this works: a parlay of N independent +EV legs is itself +EV.
    # Over 0.5 goals for a team with xG=2.0 has ~85% probability — combining 3-4 such legs
    # gives ~50% combined prob at 2.5-4x combined odds. That's a real edge.
    acca_max_legs: int = 5           # Max legs in any parlay (was 3)
    acca_min_legs: int = 2
    acca_min_qualifying: int = 3     # Minimum qualifying legs to generate any parlay
    acca_min_leg_edge_pct: float = 2.0   # Min edge per leg for value parlays (was 2.5)
    acca_max_per_day: int = 3        # Max parlays per matchday (was 1)

    # Markets to scan for parlay legs even when disabled for singles.
    # BTTS was disabled for singles (negative ROI) but is viable as parlay legs
    # because we only pick high-confidence BTTS Yes (model_prob > 60%).
    acca_parlay_only_markets: List = field(default_factory=lambda: ["BTTS"])

    # Safe parlay settings: combine highest-probability legs from different matches
    acca_safe_min_prob: float = 0.70   # Minimum model prob for safe parlay legs
    acca_safe_max_legs: int = 4        # Max legs in safe parlay

    # -- Monte Carlo simulation --
    monte_carlo_sims: int = 10000

    # -- Odds sanity ranges (from betting_engine + advanced_betting_engine) --
    odds_sanity: Dict = field(default_factory=lambda: {
        "h2h":     {"min": 1.05, "max": 25.0},
        "totals":  {"min": 1.05, "max": 6.0},
        "spreads": {"min": 1.05, "max": 6.0},
        "btts":    {"min": 1.10, "max": 8.0},
        "corners": {"min": 1.10, "max": 10.0},
        "cards":   {"min": 1.10, "max": 10.0},
    })

    # -- Maximum credible value edge (data error above this) --
    max_credible_value_pct: float = 50.0

    # -- Dry run mode (no file writes) --
    dry_run: bool = False

    # -- Per-market rules (multi-market backtest-calibrated, 760 matches 2023-2025) --
    # Each market can be enabled/disabled and has its own edge thresholds.
    #
    # Multi-market backtest evidence (Feb 17 2026):
    #   1X2 Draw:   +16-24% ROI at 4-8% edge. CROWN JEWEL. Consistent both seasons.
    #   1X2 Away:   +3% ROI at 5% edge. Marginal, negative CLV. Monitor.
    #   1X2 Home:   +9-20% ROI at 5-8% edge, but <35 bets. Small sample.
    #   O/U Over:   +25% ROI at 5% edge with blended_40 lineup xG! +12.5% lineup-only.
    #               Team-level xG alone is breakeven (-0.5%). Lineup data is critical.
    #   O/U Under:  -1% to -14% ROI at ALL thresholds and ALL xG sources → DISABLED
    #   Phase 8 live-bet audit (42 bets, Feb 19 2026) — override backtest with LIVE evidence:
    #   AH:         +103.5% ROI live (8 bets, 75% WR) — but 8 bets is noise. Backtest: -20% to -37% ROI → DISABLED
    #   DC:         +39.1% ROI live (7 bets, 85.7% WR) — KEEP
    #   O/U:        +39.1% ROI live (10 bets, 70% WR) — KEEP
    #   1X2:        +34.3% ROI live but 38.5% WR — max_edge now 8% (was 12%)
    #   DNB:        -37.3% ROI live (0% WR, 0 of 2 won) — DISABLE
    #   BTTS:       -7.6% ROI live (no edge visible) — DISABLE
    market_rules: Dict = field(default_factory=lambda: {
        # edge_shrinkage: 1.0 = trust model fully, 0.6 = blend 60% model + 40% market.
        # Prevents overconfident edge claims. O/U has proven calibration (14% ROI).
        # Other markets showed 9.9% claimed → 1.8% actual (overconfident). Use 0.6 if re-enabled.
        "1X2":        {"enabled": False, "min_edge_pct": 6.0,  "max_edge_pct": 8.0, "edge_shrinkage": 0.6},
        "1X2_Away":   {"enabled": False, "min_edge_pct": 7.0,  "max_edge_pct": 7.0, "edge_shrinkage": 0.6},
        "1X2_Draw":   {"enabled": False, "min_edge_pct": 5.0,  "max_edge_pct": 8.0, "edge_shrinkage": 0.6},
        "O/U_Over":   {"enabled": True,  "min_edge_pct": 5.0,  "max_edge_pct": 7.0,    # O/U 1.5: +10.1% ROI (74% WR, 46 bets), realized CLV +2.5%.
                       "allowed_lines": [1.5, 2.5], "kelly_fraction": 0.08,             # 2026-06-01: PER-LINE shrinkage. 1.5 model claims ~8% edge but
                       "line_shrinkage": {1.5: 0.6},                                     # CLV says ~2.5% (3x over-claim) → shrink 1.5 to size Kelly on the
                       "line_min_edge": {2.5: 7.0},                                     # deflated edge. 2.5 UNSHRUNK (best CLV +4.8%, gated at >=7.0).
                       "line_max_edge": {2.5: 10.0}},                                   # 2026-09-01: line 2.5 band was [7,7] = EMPTY (market max 7 met the
                                                                                        # line min 7; +8.3% edge died above_max_edge). Journal: 7-10% edge
                                                                                        # = +12.8% ROI / CLV +4.9 (n=16); >=10% = -26.9% (n=17) → cap 10.
        "O/U_Under":  {"enabled": False, "min_edge_pct": 6.0,  "max_edge_pct": 7.0, "edge_shrinkage": 0.6},
        "AH":         {"enabled": False, "min_edge_pct": 4.0,  "max_edge_pct": 7.0, "edge_shrinkage": 0.6},
        "DC":         {"enabled": False, "min_edge_pct": 4.0,  "max_edge_pct": 7.0, "edge_shrinkage": 0.6},
        "DNB":        {"enabled": False, "min_edge_pct": 5.0,  "max_edge_pct": 7.0, "edge_shrinkage": 0.6},
        "BTTS":       {"enabled": False, "min_edge_pct": 5.0,  "max_edge_pct": 7.0, "edge_shrinkage": 0.6},
        "Alt_OU":     {"enabled": True,  "min_edge_pct": 5.0,  "max_edge_pct": 7.0,
                       "allowed_lines": [1.5, 2.5],                                     # No shrinkage — same proven O/U model
                       "line_min_edge": {2.5: 7.0},                                    # O/U 2.5: higher threshold
                       "line_max_edge": {2.5: 10.0}},                                  # same [7,10] window as O/U_Over 2.5 (band was [7,7] = empty)
        "Alt_AH":     {"enabled": False, "min_edge_pct": 4.0,  "max_edge_pct": 7.0, "edge_shrinkage": 0.6},
        "Corners":    {"enabled": False, "min_edge_pct": 5.0,  "max_edge_pct": 7.0, "edge_shrinkage": 0.6},
        "Cards":      {"enabled": False, "min_edge_pct": 5.0,  "max_edge_pct": 7.0, "edge_shrinkage": 0.6},
    })

    # -- Situational edge adjustments (multi-market backtest-derived, 760 matches) --
    # Value = percentage-point adjustment to min_edge threshold.
    # Negative = lower threshold (easier to trigger) for backtest-profitable situations.
    # Positive = raise threshold (harder to trigger) for backtest-unprofitable situations.
    # Key format: "tag__market_cat__selection" (tag from _get_situational_tags)
    situational_edge_adjustments: Dict = field(default_factory=lambda: {
        # --- Profitable situations: lower the threshold ---
        "derby__1X2__Draw":              -2.0,   # Derby draws: +96% ROI (7 bets)
        "promoted_away__1X2__Draw":      -1.5,   # Promoted away draws: +28% ROI (64 bets)
        "mgr_change__1X2__Draw":         -1.5,   # Manager-change draws: +87-134% ROI
        "rest_advantage__1X2__Draw":     -1.0,   # Rest advantage draws: +42% ROI (33 bets)
        "post_intl__O/U_Over__Over":     -1.0,   # Post-intl break overs: +39% ROI (28 bets)
        "rest_disadvantage__O/U_Over__Over": -1.0,  # Rest disadv overs: +37% ROI (42 bets)
        "promoted_home__O/U_Over__Over": -1.0,   # Promoted home overs: +22-31% ROI (49 bets)
        # --- Unprofitable situations: raise the threshold ---
        "derby__O/U_Over__Over":         +3.0,   # Derby overs: -56% ROI → effectively block
        "midweek__O/U_Over__Over":       +2.0,   # Midweek overs: -25% ROI
        # --- EPL-specific (conservative defaults until backtested) ---
        "big_6_matchup__1X2__Draw":      -1.0,   # Big 6 matchups tend to be tighter
        "big_6_matchup__O/U_Over__Over": +1.0,   # Big 6 matches: often cagey, fewer goals
        "london_derby__1X2__Draw":       -1.5,   # London derbies: higher draw tendency
    })

    @classmethod
    def for_league(cls, league: str = "serie_a") -> "BettingConfig":
        """Return a BettingConfig tuned for the given league.

        Serie A is the default and fully calibrated (0.05 Kelly per Apr 2026
        edge-erosion finding). EPL gets a more conservative 0.04 Kelly
        until 50+ EPL bets have settled and we can calibrate properly.

        Args:
            league: League identifier (e.g., "serie_a", "epl", "la_liga",
                    "bundesliga", "ligue_1").
        """
        cfg = cls()
        league_lower = league.lower().replace(" ", "_")

        # Draw stake multiplier: derived from league draw rate in registry
        try:
            from config.leagues import LEAGUE_REGISTRY
            _lcfg = LEAGUE_REGISTRY.get(league_lower)
            cfg.draw_stake_multiplier = min(0.90, max(0.50, _lcfg.draw_rate * 2.85)) if _lcfg else 0.70
        except Exception:
            cfg.draw_stake_multiplier = 0.70

        # Per-league Kelly fraction: read from per-league deployment_state.json
        # if present, else use league-conservative defaults. Serie A: 0.05 (live
        # production, calibrated). EPL: 0.04 (one notch tighter pending data).
        # TODO: re-evaluate EPL Kelly when 50+ EPL bets have settled.
        cfg.kelly_fraction = _load_league_kelly_fraction(league_lower, default=cfg.kelly_fraction)

        # Situational adjustments: only apply for Serie A (calibrated data)
        if league_lower != "serie_a":
            cfg.situational_edge_adjustments = {}

        return cfg

    @classmethod
    def from_config(cls, bankroll_override: float = None) -> "BettingConfig":
        """Create BettingConfig with bankroll auto-loaded from journal.

        Args:
            bankroll_override: If > 0, use this value instead of auto-loading.
                               None or 0 means auto-load from journal.
        """
        cfg = cls()
        if bankroll_override and bankroll_override > 0:
            cfg.bankroll = bankroll_override
        else:
            try:
                from scripts.betting.bankroll_loader import get_effective_bankroll
                cfg.bankroll = get_effective_bankroll()
            except Exception as e:
                log.critical("BANKROLL LOAD FAILED: %s — REFUSING TO BET", e)
                try:
                    from scripts.pipeline.notify import send_notification
                    send_notification(
                        f"Bankroll load failed: {e}. Betting engine ABORTED.",
                        title="CRITICAL: Bankroll Error",
                        level="critical",
                    )
                except Exception:
                    pass
                raise RuntimeError(f"Cannot load bankroll: {e}") from e
        return cfg


# -- Market priorities (from advanced_betting_engine) --
MARKET_PRIORITY = {
    "1X2": 5,
    "O/U": 4,
    "AH":  3,
    "DC":  3,
    "DNB": 2,
    "BTTS": 2,
    "Corners": 1,
    "Cards": 1,
}

# -- Value thresholds by market (from advanced_betting_engine) --
VALUE_THRESHOLDS = {
    "h2h":     {"strong": 0.15, "moderate": 0.08, "minimum": 0.03},
    "totals":  {"strong": 0.12, "moderate": 0.06, "minimum": 0.03},
    "spreads": {"strong": 0.10, "moderate": 0.05, "minimum": 0.03},
    "btts":    {"strong": 0.12, "moderate": 0.06, "minimum": 0.03},
    "corners": {"strong": 0.12, "moderate": 0.06, "minimum": 0.03},
    "cards":   {"strong": 0.10, "moderate": 0.05, "minimum": 0.03},
}

# -- Stake allocation by value tier (from advanced_betting_engine) --
STAKE_ALLOCATION = {
    "strong":   0.03,   # 3% of bankroll
    "moderate": 0.02,   # 2%
    "minimum":  0.01,   # 1%
}

# -- Flat stakes by confidence (from bankroll_manager) --
FLAT_STAKES = {
    "VERY_HIGH":   0.03,
    "HIGH":        0.02,
    "MEDIUM_HIGH": 0.015,
    "MEDIUM":      0.01,
    "LOW":         0.005,
}

# -- Confidence ranking (from betting_engine) --
CONFIDENCE_RANK = {
    "VERY_HIGH": 5, "VERY HIGH": 5,
    "HIGH": 4,
    "MEDIUM_HIGH": 3, "MEDIUM-HIGH": 3, "MEDIUM HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1,
}


# =============================================================================
# 2. DATA STRUCTURES
# =============================================================================
@dataclass
class ValueBet:
    """A single identified value bet with all metadata.
    (Preserved from ultimate_betting_system.ValueBet)
    """
    match: str
    date: str
    market: str                  # "1X2", "O/U 2.5", "AH -1", "BTTS", "DC", "DNB"
    selection: str               # "Home", "Draw", "Over 2.5", etc.
    model_prob: float            # Our calibrated probability
    sharp_implied_prob: float    # Pinnacle/sharp implied probability (overround removed)
    edge_pct: float              # (model_prob - sharp_implied) / sharp_implied * 100
    raw_edge: float              # model_prob - sharp_implied

    # Odds
    best_odds: float             # Best available odds across all bookmakers
    best_bookmaker: str          # Which bookmaker offers best odds
    avg_odds: float              # Market average odds
    pinnacle_odds: float         # Pinnacle odds (benchmark)
    odds_count: int              # How many bookmakers offer this

    # Staking
    kelly_raw: float = 0.0       # Raw Kelly fraction
    kelly_adj: float = 0.0       # Adjusted Kelly (fractional + caps)
    stake_pct: float = 0.0       # Final stake as % of bankroll
    stake_amount: float = 0.0    # Final stake in currency

    # Expected value
    ev_per_unit: float = 0.0     # Expected value per unit bet
    expected_profit: float = 0.0 # Expected profit on this bet

    # Confidence
    confidence_tier: str = ""    # "ELITE", "STRONG", "STANDARD"
    model_confidence: float = 0.0  # Original model confidence

    # League
    league: str = "serie_a"      # "serie_a", "premier_league", etc.

    # Portfolio
    match_group: str = ""        # For correlation tracking
    is_selected: bool = False    # Final portfolio selection

    # Entry timing (populated by EntryTimingAnalyzer)
    entry_timing: dict = field(default_factory=dict)


@dataclass
class BetSlip:
    """Final output: actionable bet slip."""
    bets: List[ValueBet] = field(default_factory=list)
    total_stake: float = 0.0
    total_ev: float = 0.0
    exposure_pct: float = 0.0
    n_matches: int = 0
    generated_at: str = ""
    bankroll: float = 1000.0


@dataclass
class AccumulatorBet:
    """A multi-leg accumulator (parlay) combining independent value bets.
    (Preserved from ultimate_betting_system.AccumulatorBet)
    """
    legs: List[ValueBet]
    combined_odds: float
    combined_prob: float
    stake_amount: float
    expected_profit: float
    ev_per_unit: float
    n_legs: int
    date: str = ""
    category: str = "value"  # "safe" (high-prob legs) or "value" (edge-based)

    @property
    def potential_profit(self) -> float:
        return round(self.stake_amount * (self.combined_odds - 1), 2)

    @property
    def matches(self) -> str:
        return " + ".join(b.match for b in self.legs)

    @property
    def selections(self) -> str:
        return " + ".join(f"{b.market} {b.selection}" for b in self.legs)


# =============================================================================
# 3. DATA LAYER -- Load all live files
# =============================================================================

# Prediction file → league key. Shared by the real path (load_predictions)
# and the paper track (run_paper_track) so a league can never fall through
# both: enabled leagues bet real, gated leagues bet paper.
PREDICTION_FILES = [
    ("predictions.json", "serie_a"),
    ("predictions_premier_league.json", "premier_league"),
]


def load_predictions() -> List[Dict]:
    """Load ensemble predictions from all validated league files.

    Tags each prediction with a 'league' field and filters out predictions
    from leagues that haven't passed deployment validation.
    """
    all_preds = []
    found_any = False
    for fname, league_key in PREDICTION_FILES:
        p = UPCOMING / fname
        if not p.exists():
            continue
        found_any = True

        # League gate: skip predictions from unvalidated leagues
        if not _league_betting_enabled(league_key):
            log.warning("SKIPPING %s — league '%s' not validated for betting "
                        "(no deployment_state.json with betting_enabled=true)",
                        fname, league_key)
            continue

        try:
            with open(p) as f:
                data = json.load(f)
            preds = data.get("predictions", [])
            # Tag each prediction with its league
            for pred in preds:
                pred.setdefault("league", league_key)
            all_preds.extend(preds)
            log.info("Loaded %d predictions from %s (league=%s)", len(preds), fname, league_key)
        except Exception as e:
            log.warning("Failed to load %s: %s", fname, e)
    if not found_any:
        log.error("No prediction files found")

    # Filter out past matches at load time — prevents stale re-bets on pipeline re-runs
    today_str = datetime.now().strftime("%Y-%m-%d")
    pre_filter = len(all_preds)
    all_preds = [p for p in all_preds if p.get("date", "9999-99-99")[:10] >= today_str]
    if pre_filter > len(all_preds):
        log.info("Filtered %d past-date predictions at load time", pre_filter - len(all_preds))

    return all_preds


def load_odds_full() -> Dict:
    """Load full odds with all bookmakers (all leagues merged)."""
    all_matches = {}
    # Load Serie A odds (default)
    p = UPCOMING / "odds_full.json"
    if p.exists():
        with open(p) as f:
            data = json.load(f)
        all_matches.update(data.get("matches", {}))
    # Merge in other league odds files
    for league_file in UPCOMING.glob("odds_full_*.json"):
        try:
            with open(league_file) as f:
                league_data = json.load(f)
            league_matches = league_data.get("matches", {})
            all_matches.update(league_matches)
            log.info("Loaded %d matches from %s", len(league_matches), league_file.name)
        except Exception as e:
            log.warning("Failed to load %s: %s", league_file.name, e)
    if not all_matches:
        log.warning("No odds data found in any odds_full*.json file")
    return all_matches


def load_simple_odds() -> Dict:
    """Load simple 1X2 odds (fallback, all leagues merged)."""
    all_odds = {}
    for odds_file in [UPCOMING / "odds.json"] + list(UPCOMING.glob("odds_*.json")):
        if odds_file.name.startswith("odds_full") or odds_file.name.startswith("odds_bookmakers"):
            continue
        if odds_file.name.startswith("odds_extra_markets"):
            continue
        if odds_file.name in ("odds_totals.json", "odds_spreads.json", "odds_btts.json",
                              "odds_dnb.json", "odds_movement.json"):
            continue
        if odds_file.exists():
            try:
                with open(odds_file) as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    # Simple odds are flat dicts keyed by match
                    for k, v in data.items():
                        if isinstance(v, dict) and "home" in v:
                            all_odds[k] = v
            except Exception:
                pass
    return all_odds


def load_extended_markets() -> Dict:
    """Load extended market predictions (DC, exact scores, team totals, 1H)."""
    p = UPCOMING / "extended_markets.json"
    if not p.exists():
        return {}
    with open(p) as f:
        data = json.load(f)
    return data.get("matches", {})


def _gate_aux_predictions(rows, allowed: set, label: str):
    """Row-level league gate for the AUXILIARY prediction files.

    goal/btts/cards/corners/margin predictions are merged both-league files with
    NO league field (goal_predictions.json on 2026-08-31: 23 rows = 10 SA + 10 EPL
    + 3 stale). The per-league betting gate lives in load_predictions() only, and
    the O/U scanner — the ONLY enabled market — iterated these files against the
    merged odds with no gate, so a gated-league (EPL) O/U bet would have been
    journaled as Serie A the moment its edge landed in band. Keep only matches
    that survived the gate in load_predictions().
    """
    if not isinstance(rows, list):
        return rows
    kept = [r for r in rows if isinstance(r, dict) and r.get("match") in allowed]
    dropped = [r.get("match") if isinstance(r, dict) else r
               for r in rows if not (isinstance(r, dict) and r.get("match") in allowed)]
    if dropped:
        log.warning("  %s predictions: dropped %d/%d rows with no gated prediction "
                    "(league gated or stale): %s%s", label, len(dropped), len(rows),
                    ", ".join(str(m) for m in dropped[:5]),
                    " ..." if len(dropped) > 5 else "")
    return kept


def load_goal_predictions() -> List[Dict]:
    """Load Poisson goal predictions."""
    p = UPCOMING / "goal_predictions.json"
    if not p.exists():
        return []
    with open(p) as f:
        data = json.load(f)
    return data.get("predictions", [])


PAPER_STAKE = 10.0  # flat — the deployment bar is count + CLV, not sizing


def run_paper_track() -> int:
    """Scan GATED leagues with the production market config and journal the
    candidates as flat-stake PAPER bets in the separate paper journal.

    This is the evidence pump for a league's deployment bar ("50+ settled
    paper bets, CLV+" — premier_league's gated_reason): without it the bar
    is unearnable, because load_predictions() drops a gated league's file
    wholesale and nothing ever settles. Zero real exposure by construction:
    PAPER_JOURNAL_PATH is a different file and bankroll/history/risk derive
    from the real journal only.

    Called from save_bet_slip's non-dry path so paper bets commit on the
    same T-30 timing as real ones (>24h-early bets measured -5% ROI, <24h
    +63% — paper CLV evidence must not be gathered on the losing timing).
    """
    from scripts.betting.bet_journal import (
        PAPER_JOURNAL_PATH, add_bet as journal_add_bet)

    n_journaled = 0
    today_str = datetime.now().strftime("%Y-%m-%d")
    for fname, league_key in PREDICTION_FILES:
        if _league_betting_enabled(league_key):
            continue  # real path covers it
        p = UPCOMING / fname
        if not p.exists():
            continue
        try:
            with open(p) as f:
                preds = json.load(f).get("predictions", [])
        except Exception as e:
            log.warning("Paper track: failed to load %s: %s", fname, e)
            continue
        for pred in preds:
            pred.setdefault("league", league_key)
        preds = [q for q in preds
                 if q.get("date", "9999-99-99")[:10] >= today_str]
        if not preds:
            continue

        allowed = {q.get("match") for q in preds if q.get("match")}
        pred_by_match = {q["match"]: q for q in preds if q.get("match")}
        goal_preds = _gate_aux_predictions(
            load_goal_predictions(), allowed, f"paper:{league_key}")
        if not goal_preds:
            continue

        engine = UnifiedBettingEngine()
        bets = engine.scan_ou_market(goal_preds, load_odds_full(),
                                     pred_by_match=pred_by_match)
        for bet in bets:
            bet_id = journal_add_bet({
                "match": bet.match,
                "date": bet.date,
                "market": bet.market,
                "selection": bet.selection,
                "model_prob": bet.model_prob,
                "sharp_implied_prob": bet.sharp_implied_prob,
                "edge_pct": bet.edge_pct,
                "odds": bet.best_odds,
                "bookmaker": bet.best_bookmaker,
                "avg_odds": bet.avg_odds,
                "pinnacle_odds": bet.pinnacle_odds,
                "stake": PAPER_STAKE,
                "confidence": bet.confidence_tier,
                "league": league_key,
                "pipeline_status": "paper",
            }, journal_path=PAPER_JOURNAL_PATH)
            if bet_id:
                n_journaled += 1
        if bets:
            log.info("Paper track (%s): journaled %d candidate(s)",
                     league_key, len(bets))
    return n_journaled


def load_btts_predictions() -> List[Dict]:
    """Load BTTS model predictions."""
    p = UPCOMING / "btts_predictions.json"
    if not p.exists():
        return []
    with open(p) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("predictions", [])


def load_cards_predictions() -> List[Dict]:
    """Load cards model predictions."""
    p = UPCOMING / "cards_predictions.json"
    if not p.exists():
        return []
    with open(p) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("predictions", [])


def load_corners_predictions() -> List[Dict]:
    """Load corners model predictions."""
    p = UPCOMING / "corners_predictions.json"
    if not p.exists():
        return []
    with open(p) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("predictions", [])


def load_margin_predictions() -> List[Dict]:
    """Load margin predictions with handicap probabilities at every line."""
    p = UPCOMING / "margin_predictions.json"
    if not p.exists():
        return []
    with open(p) as f:
        data = json.load(f)
    return data.get("predictions", [])


def load_extra_market_odds() -> Dict:
    """Load extra market odds (BTTS, DC, alternate totals) with real bookmaker prices.

    Merges Serie A and EPL (and any other league-suffixed) files.
    """
    all_extra: Dict = {}
    for p in [UPCOMING / "odds_extra_markets.json"] + list(UPCOMING.glob("odds_extra_markets_*.json")):
        if not p.exists():
            continue
        try:
            with open(p) as f:
                data = json.load(f)
            matches = data.get("matches", {})
            all_extra.update(matches)
        except Exception:
            log.warning(f"Failed to load extra markets from {p.name}")
    return all_extra


def load_odds_movement() -> Dict:
    """Load odds movement data (steam moves, line movements)."""
    p = UPCOMING / "odds_movement.json"
    if not p.exists():
        return {}
    with open(p) as f:
        data = json.load(f)
    return data.get("matches", {})


def load_bookmaker_analysis() -> Dict:
    """Load sharp/soft bookmaker divergence analysis."""
    p = UPCOMING / "bookmaker_analysis.json"
    if not p.exists():
        return {}
    with open(p) as f:
        data = json.load(f)
    return data.get("matches", {})


def load_all_predictions() -> Dict[str, List[Dict]]:
    """Load all predictions from all models (for match context).
    (From advanced_betting_engine)
    """
    predictions = {}
    loaders = {
        "goals":   UPCOMING / "goal_predictions.json",
        "margins": UPCOMING / "margin_predictions.json",
        "cards":   UPCOMING / "cards_predictions.json",
        "btts":    UPCOMING / "btts_predictions.json",
        "corners": UPCOMING / "corners_predictions.json",
        "main":    UPCOMING / "predictions.json",
    }
    for key, path in loaders.items():
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    predictions[key] = data
                else:
                    predictions[key] = data.get("predictions", [])
            except Exception as e:
                log.warning(f"Failed to load predictions from {path}: {e}")
    return predictions


def load_market_adjustments() -> Dict:
    """Load per-market stake multipliers from ROI feedback loop.
    (From advanced_betting_engine)
    """
    adj_path = DATA_DIR / "feedback" / "market_adjustments.json"
    if not adj_path.exists():
        return {}
    try:
        with open(adj_path) as f:
            data = json.load(f)
        return data.get("market_multipliers", {})
    except Exception:
        return {}


# =============================================================================
# 4. EDGE LAYER -- True probability calculation + edge detection
# =============================================================================
def remove_overround(odds_list: List[float]) -> List[float]:
    """Remove overround from odds to get true implied probabilities.

    Uses multiplicative method (Shin's method simplified):
    true_prob[i] = raw_prob[i] / sum(raw_probs)

    (Preserved exactly from ultimate_betting_system.remove_overround)
    """
    if not odds_list or any(o <= 1 for o in odds_list):
        return [1 / len(odds_list)] * len(odds_list) if odds_list else []

    raw_probs = [1 / o for o in odds_list]
    total = sum(raw_probs)
    if total <= 0:
        return [1 / len(odds_list)] * len(odds_list)

    return [p / total for p in raw_probs]


def find_best_odds(bookmakers: List[Dict], selection_key: str) -> Tuple[float, str, float, int]:
    """Find best odds across all bookmakers for a given selection.

    Returns: (best_odds, best_bookmaker, avg_odds, count)
    (Preserved from ultimate_betting_system.find_best_odds)
    """
    odds_list = []
    best = 0.0
    best_bk = ""

    for bk in bookmakers:
        o = bk.get(selection_key, 0)
        if o and o > 1:
            odds_list.append(o)
            if o > best:
                best = o
                best_bk = bk.get("bookmaker", "Unknown")

    if not odds_list:
        return 0, "", 0, 0

    avg = sum(odds_list) / len(odds_list)
    return best, best_bk, avg, len(odds_list)


def get_pinnacle_odds(bookmakers: List[Dict], selection_key: str) -> float:
    """Get Pinnacle odds for a selection (sharpest benchmark).
    (Preserved from ultimate_betting_system.get_pinnacle_odds)
    """
    for bk in bookmakers:
        if "Pinnacle" in bk.get("bookmaker", ""):
            o = bk.get(selection_key, 0)
            if o and o > 1:
                return o
    return 0


def calculate_kelly(model_prob: float, odds: float, fraction: float = 0.25) -> float:
    """Calculate fractional Kelly criterion stake.

    Kelly formula: f* = (b*p - q) / b
    where b = decimal odds - 1, p = model prob, q = 1-p

    We multiply by fraction (0.25 = quarter Kelly) for safety.
    Returns fraction of bankroll to stake (e.g. 0.03 = 3%).
    Returns 0 for any invalid input (NaN, inf, out of range).
    """
    # Reject NaN/inf/invalid inputs
    if (not isinstance(model_prob, (int, float)) or not isinstance(odds, (int, float))
            or math.isnan(model_prob) or math.isnan(odds)
            or math.isinf(model_prob) or math.isinf(odds)):
        return 0

    if odds <= 1 or model_prob <= 0 or model_prob >= 1:
        return 0

    b = odds - 1
    p = model_prob
    q = 1 - p

    kelly = (b * p - q) / b
    if kelly <= 0 or math.isnan(kelly):
        return 0

    # Apply fraction and floor at 0.001 (0.1%)
    result = kelly * fraction
    return result if result >= 0.001 else 0


def calculate_ev(model_prob: float, odds: float) -> float:
    """Calculate expected value per unit staked.

    EV = p * (odds - 1) - (1 - p)
    Positive EV = profitable bet.

    (Preserved from ultimate_betting_system.calculate_ev)
    """
    if odds <= 1 or model_prob <= 0:
        return -1
    return model_prob * (odds - 1) - (1 - model_prob)


def estimate_odds(probability: float) -> float:
    """Estimate decimal odds from probability (adds ~10% bookmaker margin).
    (From betting_engine.estimate_odds)
    """
    if probability <= 0 or probability >= 1:
        return 2.0
    fair_odds = 1 / probability
    return round(fair_odds * 0.90, 2)


# =============================================================================
# 5. UNIFIED BETTING ENGINE CLASS
# =============================================================================
class UnifiedBettingEngine:
    """Unified betting engine consolidating all market scanners, portfolio
    optimization, accumulator generation, intelligence filters, and output.

    Primary logic preserved from ultimate_betting_system.py (the most
    sophisticated), with unique features merged from the other engines.
    """

    def __init__(self, cfg: BettingConfig = None):
        self.cfg = cfg or BettingConfig()
        self.all_bets: List[ValueBet] = []
        self.selected: List[ValueBet] = []
        self.accumulators: List[AccumulatorBet] = []
        self.slip: Optional[BetSlip] = None
        # Rejections recorded AFTER an edge was computed — makes "0 bets" auditable.
        self.near_misses: List[Dict] = []

    def top_near_misses(self, n: int = 10) -> List[Dict]:
        """Closest-to-band rejections first (gap_pp 0 = inside the edge band,
        rejected by another gate), then by descending edge. Factor-vetoed
        candidates sort last: they are informative, not near."""
        return sorted(self.near_misses,
                      key=lambda m: (str(m.get("reason", "")).startswith("veto_factor"),
                                     m["gap_pp"], -m["edge_pct"]))[:n]

    def _log_near_misses(self, n: int = 5) -> None:
        top = self.top_near_misses(n)
        if not top:
            return
        log.info("  Near misses (%d rejected after edge calc; closest %d):",
                 len(self.near_misses), len(top))
        for m in top:
            log.info("    %-32s %-10s %-10s edge %+.1f%% band [%.1f, %.1f] odds %.2f (pin %s) -> %s",
                     m["match"][:32], m["market"], str(m["selection"])[:10], m["edge_pct"],
                     m["min_edge"], m["max_edge"], m["best_odds"],
                     m["pinnacle_odds"] if m["pinnacle_odds"] else "-", m["reason"])

    @staticmethod
    def _get_betting_probs(pred: Dict) -> Dict:
        """Get betting-calibrated probabilities from a prediction.

        Prefers 'betting_probabilities' (raw ensemble probs without draw boost
        or temperature scaling) over 'probabilities' (accuracy-tuned).
        This prevents inflated draw edges from bleeding into bet selection.
        """
        bp = pred.get("betting_probabilities")
        if bp and bp.get("home", 0) > 0 and bp.get("draw", 0) > 0:
            return bp
        return pred.get("probabilities", {})

    # -----------------------------------------------------------------
    # Shared bet constructor (from ultimate_betting_system._make_bet)
    # -----------------------------------------------------------------
    def _get_market_category(self, market_str: str, selection: str = "") -> str:
        """Map market string (e.g., 'AH -1', 'O/U 2.5') to config category.

        For O/U markets, uses the selection to distinguish Over vs Under,
        returning 'O/U_Over' or 'O/U_Under' for separate enable/disable control.
        Falls back to 'O/U_Over' (the profitable side) if selection is ambiguous.
        """
        m = market_str.upper()
        if m.startswith("1X2"):
            sel_up = selection.upper()
            if "DRAW" in sel_up:
                return "1X2_Draw"
            if "AWAY" in sel_up:
                return "1X2_Away"
            return "1X2"
        elif m.startswith("O/U") or m.startswith("OVER") or m.startswith("UNDER"):
            sel = selection.upper()
            if "UNDER" in sel:
                return "O/U_Under"
            return "O/U_Over"
        elif m.startswith("AH"):
            return "AH"
        elif m.startswith("DC") or m.startswith("DOUBLE"):
            return "DC"
        elif m.startswith("DNB") or m.startswith("DRAW NO"):
            return "DNB"
        elif m.startswith("BTTS") or m.startswith("BOTH"):
            return "BTTS"
        elif "CORNER" in m:
            return "Corners"
        elif "CARD" in m:
            return "Cards"
        return "1X2"  # fallback

    # -----------------------------------------------------------------
    # Situational tagging (Phase 3 — backtest-derived edge adjustments)
    # -----------------------------------------------------------------
    # ⚠️ The ONLY hand-maintained duplicate of config.leagues.PROMOTED_TEAMS.
    # features/creative_factors.py imports the real one; this file forked it and
    # added spelling variants. That fork is why this goes stale every August. It
    # should read config.leagues and normalise names, but that is a betting-path
    # refactor and does not belong in a season-rollover commit. Until then:
    # update BOTH together.
    # 2026-2027 measured 2026-08-02 against the live Sofascore club lists, with
    # spelling variants included because this file joins on raw names.
    _PROMOTED_TEAMS = {
        "2023-2024": {"Cagliari", "Frosinone", "Genoa"},
        "2024-2025": {"Como", "Parma", "Venezia"},
        "2025-2026": {"Sassuolo", "Pisa", "Cremonese"},
        "2026-2027": {"Frosinone", "Monza", "Venezia"},
    }
    _EPL_PROMOTED_TEAMS = {
        "2023-2024": {"Burnley", "Sheffield United", "Luton Town"},
        "2024-2025": {"Leicester City", "Ipswich Town", "Southampton"},
        "2025-2026": {"Leeds", "Leeds United", "Burnley", "Sunderland"},
        "2026-2027": {"Coventry City", "Coventry", "Hull", "Hull City",
                      "Ipswich", "Ipswich Town"},
    }
    _EPL_BIG_6 = {
        "Arsenal", "Chelsea", "Liverpool", "Man City", "Manchester City",
        "Man United", "Manchester United", "Tottenham", "Tottenham Hotspur",
    }
    _LONDON_CLUBS = {
        "Arsenal", "Chelsea", "Tottenham", "Tottenham Hotspur",
        "West Ham", "West Ham United", "Crystal Palace",
        "Fulham", "Brentford",
    }

    def _get_situational_tags(self, pred: Dict) -> List[str]:
        """Compute situational tags from prediction data.

        Tags mirror the backtest_multimarket.py tag_match() categories.
        Used to apply per-situation edge adjustments in _make_bet().

        Tags produced:
          derby, midweek, promoted_home, promoted_away,
          rest_advantage, rest_disadvantage, post_intl
        """
        tags = []
        match_date = pred.get("date", "")
        home = pred.get("home_team", pred.get("match", "").split(" vs ")[0])
        away = pred.get("away_team", pred.get("match", "").split(" vs ")[-1])

        # Derby — from neutral_factors (set by identify_all_factors)
        neutral = pred.get("neutral_factors", [])
        if "derby" in neutral:
            tags.append("derby")

        # Midweek — Tue/Wed/Thu (dayofweek 1,2,3)
        if match_date:
            try:
                from datetime import datetime as dt
                d = dt.strptime(match_date[:10], "%Y-%m-%d")
                if d.weekday() in (1, 2, 3):
                    tags.append("midweek")
            except (ValueError, TypeError):
                pass

        # Promoted team (current season — both leagues)
        season = self._current_season(match_date)
        promoted = self._PROMOTED_TEAMS.get(season, set()) | self._EPL_PROMOTED_TEAMS.get(season, set())
        if home in promoted:
            tags.append("promoted_home")
        if away in promoted:
            tags.append("promoted_away")

        # EPL Big 6 matchup
        if home in self._EPL_BIG_6 and away in self._EPL_BIG_6:
            tags.append("big_6_matchup")

        # London derby
        if home in self._LONDON_CLUBS and away in self._LONDON_CLUBS:
            tags.append("london_derby")

        # Rest advantage / disadvantage — from situational_context
        # (injected by ensemble_prediction_engine from features.parquet)
        ctx = pred.get("situational_context", {})
        rest_adv = ctx.get("rest_advantage")
        if rest_adv is not None and abs(rest_adv) >= 2:
            tags.append("rest_advantage" if rest_adv > 0 else "rest_disadvantage")

        # Post-international break — either team returning from break
        if ctx.get("home_post_intl_break") or ctx.get("away_post_intl_break"):
            tags.append("post_intl")

        # Manager change — check factors for 'mgr_change' hint
        home_factors = pred.get("home_factors", [])
        away_factors = pred.get("away_factors", [])
        if any("mgr_change" in f for f in home_factors + away_factors + neutral):
            tags.append("mgr_change")

        return tags

    @staticmethod
    def _current_season(date_str: str) -> str:
        """Derive season string from match date (e.g., '2025-2026')."""
        try:
            from datetime import datetime as dt
            d = dt.strptime(date_str[:10], "%Y-%m-%d")
            year = d.year
            # Serie A season: Aug-May. Jan-May belongs to previous year's season.
            if d.month <= 7:
                return f"{year - 1}-{year}"
            return f"{year}-{year + 1}"
        except (ValueError, TypeError):
            return "2025-2026"

    def _apply_situational_adjustment(self, min_edge: float, market_cat: str,
                                       selection: str, tags: List[str]) -> float:
        """Adjust min_edge threshold based on situational tags.

        Returns the adjusted min_edge (lower = easier to trigger, higher = harder).
        """
        adj_map = self.cfg.situational_edge_adjustments
        total_adj = 0.0
        applied = []

        # Normalize selection: "Over 2.5" → "Over", "Under 3.5" → "Under", etc.
        sel_norm = selection.split()[0] if selection else selection

        for tag in tags:
            key = f"{tag}__{market_cat}__{sel_norm}"
            if key in adj_map:
                total_adj += adj_map[key]
                applied.append((key, adj_map[key]))

        if applied:
            # Floor: never reduce below 50% of the configured market min_edge.
            # This prevents tag stacking from completely bypassing market-specific
            # thresholds (e.g., draw at 10% shouldn't drop below 5%).
            floor = max(1.0, min_edge * 0.5)
            new_edge = max(floor, min_edge + total_adj)
            log.debug("Situational adj: %s → min_edge %.1f%% → %.1f%% (floor=%.1f%%, adjustments: %s)",
                      tags, min_edge, new_edge, floor, applied)
            return new_edge
        return min_edge

    def _make_bet(self, match, date, market, selection, model_p, sharp_p,
                  best_o, best_bk, avg_o, pin_o, count, confidence=0,
                  max_edge_override=None, min_edge_override=None,
                  pred=None) -> Optional[ValueBet]:
        """Shared bet construction with mandatory +EV gate.
        Returns None if the bet is not +EV at best available odds.
        Uses per-market edge thresholds from BettingConfig.market_rules,
        adjusted by situational context when pred is provided.
        """
        cfg = self.cfg

        # Per-market edge thresholds (multi-market backtest-calibrated)
        market_cat = self._get_market_category(market, selection)
        rules = cfg.market_rules.get(market_cat, {})

        # Check if this specific market side is enabled (e.g., O/U_Under disabled)
        if not rules.get("enabled", True):
            return None

        min_edge = min_edge_override if min_edge_override is not None else rules.get("min_edge_pct", cfg.min_edge_pct)
        max_edge = rules.get("max_edge_pct", cfg.max_edge_pct)

        # Per-line max_edge override (mirrors line_min_edge / line_shrinkage).
        # Without it a line_min_edge raise can meet the market-wide cap and
        # produce an EMPTY band: O/U 2.5 sat at [7.0, 7.0] and every candidate
        # was rejected (a +8.3% edge died as above_max_edge on 2026-09-01).
        # Journal evidence for the 2.5 window: edge 7-10% = +12.8% ROI / CLV
        # +4.9 (n=16); edge >=10% = -26.9% ROI (n=17) -> cap belongs at 10.
        if max_edge_override is None:
            line_max_edge = rules.get("line_max_edge")
            if line_max_edge:
                try:
                    _line = float(str(market).split()[-1])
                    if _line in line_max_edge:
                        max_edge_override = line_max_edge[_line]
                except (ValueError, IndexError):
                    pass

        # Apply situational adjustments (Phase 3 backtest-derived)
        if pred is not None:
            tags = self._get_situational_tags(pred)
            if tags:
                min_edge = self._apply_situational_adjustment(
                    min_edge, market_cat, selection, tags
                )

        # Veto factors that are statistically losing — on 1X2/DC/DNB (journal:
        # n=8, -33% ROI). Applied to every market via this shared builder; the
        # O/U evidence is n=2. Resolved AFTER the edge calc below so the slip
        # records it (2026-09-04: 14/19 SA matches vanished here silently).
        _VETO_FACTORS = {"cold_home", "away_fav_ref"}
        _veto_hits: set = set()
        if pred:
            _all_factors = set(
                (pred.get("neutral_factors") or []) +
                (pred.get("home_factors") or []) +
                (pred.get("away_factors") or [])
            )
            _veto_hits = _all_factors & _VETO_FACTORS

        # Validate inputs — reject NaN/inf before any calculation
        if (any(math.isnan(v) or math.isinf(v) for v in [model_p, sharp_p, best_o] if isinstance(v, float))
                or model_p <= 0 or model_p >= 1 or sharp_p <= 0):
            return None

        # Edge shrinkage: blend model toward market to combat overconfidence.
        # Per-market shrinkage factor (1.0 = trust model fully, 0.5 = blend 50/50).
        # Other markets were overconfident (9.9% claimed → 1.8% actual).
        # Per-line override (line_shrinkage) wins over the market-level value:
        # e.g. O/U 1.5 over-claims edge ~3x (8% claimed vs +2.5% realized CLV) so
        # it is shrunk, while O/U 2.5 (best CLV) stays unshrunk. The line is parsed
        # from the market string ("O/U 1.5" → 1.5).
        shrinkage = rules.get("edge_shrinkage", 1.0)
        line_shrinkage = rules.get("line_shrinkage")
        if line_shrinkage:
            try:
                _line = float(str(market).split()[-1])
                shrinkage = line_shrinkage.get(_line, shrinkage)
            except (ValueError, IndexError):
                pass
        effective_model_p = shrinkage * model_p + (1 - shrinkage) * sharp_p

        raw_edge = effective_model_p - sharp_p
        # Use absolute edge (model_p - implied_p) in percentage points.
        # This matches the dashboard display and ensures thresholds are
        # consistent across all markets (draws aren't penalized for having
        # lower base probabilities).
        edge_pct = raw_edge * 100

        def _miss(reason: str) -> None:
            """Record a post-edge rejection (closes over the CURRENT min/max_edge)."""
            _max = max_edge_override if max_edge_override is not None else max_edge
            if edge_pct < min_edge:
                gap = min_edge - edge_pct
            elif edge_pct > _max:
                gap = edge_pct - _max
            else:
                gap = 0.0
            self.near_misses.append({
                "match": match, "date": date, "market": market, "selection": selection,
                "model_prob": round(float(model_p), 4), "sharp_prob": round(float(sharp_p), 4),
                "effective_model_prob": round(float(effective_model_p), 4),
                "edge_pct": round(edge_pct, 2), "min_edge": round(min_edge, 2),
                "max_edge": round(_max, 2), "gap_pp": round(gap, 2),
                "best_odds": float(best_o), "pinnacle_odds": float(pin_o) if pin_o else None,
                "reason": reason,
            })
            return None

        # Hard cap removed — now handled by per-market max_edge (7.0%) at line below.
        # Previously: edges >8% hit 37.7% WR. Now capped at 7% via market_rules.

        if _veto_hits:
            return _miss("veto_factor:" + ",".join(sorted(_veto_hits)))

        # Odds-range gating: 1.5-2.0 is a dead zone (live: 15 bets, 40% WR, -EUR121).
        if 1.5 <= best_o < 2.0:
            return _miss("odds_dead_zone_1.5_2.0")  # -44% ROI live. No edge overcomes it.

        # Confidence-stratified edge: adjust min_edge based on model probability
        # High confidence (prob > 55%) = more reliable = lower edge needed
        # Low confidence (prob < 40%) = uncertain = higher edge needed
        for tier_name in ("high", "medium", "low"):
            tier = cfg.confidence_edge_tiers.get(tier_name, {})
            if model_p >= tier.get("min_prob", 0):
                adj = tier.get("min_edge_adj", 0)
                min_edge = max(2.0, min_edge + adj)  # Floor at 2% even with adjustment
                break

        if edge_pct < min_edge:
            return _miss("below_min_edge")
        # Market-specific edge cap (backtest-calibrated)
        effective_max = max_edge_override if max_edge_override is not None else max_edge
        if edge_pct > effective_max:
            return _miss("above_max_edge")  # Live-proven: edges above cap are model overconfidence
        if raw_edge < 0.02:
            return _miss("raw_edge_below_2pct")  # Minimum 2% absolute edge
        if best_o < cfg.min_odds or best_o > cfg.max_odds:
            return _miss("odds_outside_range")

        # Require Pinnacle benchmark — can't verify edge without sharp reference
        if not pin_o or pin_o <= 1:
            return _miss("no_pinnacle_reference")  # No Pinnacle data = no confidence in edge reality
        # Reject if best odds >5% above Pinnacle (at/below = +15% ROI, above = -1.4%)
        if best_o > pin_o * 1.05:
            return _miss("best_odds_above_pinnacle_5pct")

        # Golden zone: 2.0-2.5 odds = +38.2% ROI (best range). Lower edge threshold.
        if 2.0 <= best_o <= 2.5:
            min_edge = max(2.0, min_edge - 0.5)

        # Away preference: away bets = +13.1% ROI vs home -0.5%. Lower threshold.
        sel_upper = selection.upper().strip() if isinstance(selection, str) else ""
        if any(kw in sel_upper for kw in ("AWAY", "2", "X2")):
            min_edge = max(2.0, min_edge - 0.5)

        # Re-check min_edge after adjustments
        if edge_pct < min_edge:
            return _miss("below_min_edge")

        ev = calculate_ev(model_p, best_o)
        if ev <= 0:
            return _miss("negative_ev")  # HARD GATE: never recommend a -EV bet

        # Per-market Kelly fraction (proven markets get higher fraction)
        market_kelly = rules.get("kelly_fraction", cfg.kelly_fraction)
        # Per-league Kelly tightening: scale by league_kelly / cfg.kelly_fraction.
        # SA stays at 1.0 (cfg default = SA value). EPL = 0.04/0.05 = 0.8x.
        # TODO: re-evaluate EPL Kelly once 50+ EPL bets have settled.
        _bet_league = pred.get("league", "serie_a") if pred else "serie_a"
        _league_kelly = _load_league_kelly_fraction(_bet_league, default=cfg.kelly_fraction)
        if cfg.kelly_fraction > 0 and _league_kelly != cfg.kelly_fraction:
            market_kelly = market_kelly * (_league_kelly / cfg.kelly_fraction)
        kelly_raw = calculate_kelly(model_p, best_o, fraction=1.0)
        kelly_adj = calculate_kelly(model_p, best_o, fraction=market_kelly)

        if cfg.staking_mode == "proportional":
            stake_pct = min(cfg.proportional_stake_pct, cfg.max_stake_pct)
        else:
            # Kelly fractional — sizes proportional to perceived edge.
            stake_pct = min(kelly_adj * 100, cfg.max_stake_pct)
            if stake_pct < 0.20:
                return _miss("kelly_below_0.2pct")  # edge too thin (lowered from 0.30)

            # Edge-quality multiplier: data shows 3-5% edges = +113% ROI,
            # 5-7% = +2% ROI. Boost proven range, reduce marginal.
            if 3.0 <= edge_pct <= 5.0:
                stake_pct *= 1.2   # 20% boost for sweet spot edges
            elif edge_pct > 5.0:
                stake_pct *= 0.85  # 15% reduction for marginal edges

            # Re-apply cap after multiplier
            stake_pct = min(stake_pct, cfg.max_stake_pct)

        # NO artificial floor — if Kelly says 0.25%, stake 0.25%.
        # The old 0.5% floor was oversizing thin edges by 67%+.

        # Apply draw stake reduction for variance management
        # Draws have high edge but low win rate (31.7%) → brutal losing streaks.
        # Reduce stake by draw_stake_multiplier to protect bankroll psychology.
        # Multiplier derived from league draw rate: higher draw rate = less reduction.
        if market_cat == "1X2_Draw" and hasattr(cfg, "draw_stake_multiplier"):
            _bet_league = pred.get("league", "serie_a") if pred else "serie_a"
            try:
                from config.leagues import LEAGUE_REGISTRY
                _lcfg = LEAGUE_REGISTRY.get(_bet_league)
                # Scale: draw_rate 0.28 → mult 0.80, draw_rate 0.22 → mult 0.65
                _draw_mult = min(0.90, max(0.50, _lcfg.draw_rate * 2.85)) if _lcfg else cfg.draw_stake_multiplier
            except Exception:
                _draw_mult = cfg.draw_stake_multiplier
            stake_pct *= _draw_mult

        # Day-of-week gate: block Monday (-40.75% ROI, 21 bets) and Friday (-9.16% ROI, 18 bets)
        try:
            from datetime import datetime as _dt
            _match_dt = _dt.strptime(date[:10], "%Y-%m-%d")
            _dow = _match_dt.weekday()
            if _dow == 0:  # Monday: -40.75% ROI, 33% WR → full block
                return None
            if _dow == 4:  # Friday: -9.16% ROI despite 61% WR (bad odds profile) → full block
                return None
        except (ValueError, TypeError):
            log.warning("Day-of-week check failed for date=%s — allowing bet", date)

        tier = ("ELITE" if edge_pct >= cfg.elite_edge_pct else
                "STRONG" if edge_pct >= cfg.strong_edge_pct else "STANDARD")

        # Extract league from prediction context
        _league = pred.get("league", "serie_a") if pred else "serie_a"

        return ValueBet(
            match=match, date=date, market=market, selection=selection,
            model_prob=round(model_p, 4), sharp_implied_prob=round(sharp_p, 4),
            edge_pct=round(edge_pct, 2), raw_edge=round(raw_edge, 4),
            best_odds=best_o, best_bookmaker=best_bk, avg_odds=round(avg_o, 2),
            pinnacle_odds=pin_o, odds_count=count,
            kelly_raw=round(kelly_raw, 4), kelly_adj=round(kelly_adj, 4),
            stake_pct=round(stake_pct, 2),
            stake_amount=round(cfg.bankroll * stake_pct / 100, 2),
            ev_per_unit=round(ev, 4),
            expected_profit=round(cfg.bankroll * stake_pct / 100 * ev, 2),
            confidence_tier=tier, model_confidence=confidence,
            league=_league,
            match_group=match,
        )

    # -----------------------------------------------------------------
    # MARKET SCANNERS
    # -----------------------------------------------------------------
    def scan_1x2_market(self, predictions: List[Dict],
                        odds_full: Dict) -> List[ValueBet]:
        """Scan 1X2 market for value bets.
        BACKTEST-PROVEN: Only Draw bets are profitable in 1X2.
        """
        bets = []
        for pred in predictions:
            match = pred["match"]
            date = pred.get("date", "")
            probs = self._get_betting_probs(pred)
            if not probs or match not in odds_full:
                continue

            h2h = odds_full[match].get("h2h", {})
            if not h2h:
                continue
            bookmakers = h2h.get("all_bookmakers", [])
            if not bookmakers:
                continue

            pin_h = get_pinnacle_odds(bookmakers, "home")
            pin_d = get_pinnacle_odds(bookmakers, "draw")
            pin_a = get_pinnacle_odds(bookmakers, "away")
            if not all([pin_h, pin_d, pin_a]):
                pin_h = h2h.get("home", 0)
                pin_d = h2h.get("draw", 0)
                pin_a = h2h.get("away", 0)
            if not all([pin_h > 1, pin_d > 1, pin_a > 1]):
                continue

            true_probs = remove_overround([pin_h, pin_d, pin_a])

            # Scan all three 1X2 outcomes based on market rules
            outcomes = [
                ("1X2",       "Home", probs.get("home", 0), true_probs[0], "home", pin_h, None),
                ("1X2_Away",  "Away", probs.get("away", 0), true_probs[2], "away", pin_a, None),
                ("1X2_Draw",  "Draw", probs.get("draw", 0), true_probs[1], "draw", pin_d, self.cfg.max_edge_draw_pct),
            ]

            for market_cat, selection, model_p, sharp_p, bk_key, pin_odds, max_edge_ovr in outcomes:
                rule = self.cfg.market_rules.get(market_cat, {})
                if not rule.get("enabled", False):
                    continue
                if model_p <= 0:
                    continue

                best_o, best_bk, avg_o, count = find_best_odds(bookmakers, bk_key)
                bet = self._make_bet(
                    match, date, "1X2", selection, model_p, sharp_p,
                    best_o, best_bk, avg_o, pin_odds, count,
                    confidence=pred.get("confidence", 0),
                    max_edge_override=max_edge_ovr,
                    pred=pred)
                if bet:
                    bets.append(bet)
        return bets

    def scan_ou_market(self, goal_preds: List[Dict],
                       odds_full: Dict,
                       pred_by_match: Dict = None) -> List[ValueBet]:
        """Scan Over/Under markets for value.

        When `pred_by_match` is given it is the set of matches that passed the
        per-league betting gate: any goal prediction outside it is skipped
        (goal_predictions.json is a merged both-league file — see
        _gate_aux_predictions). Passing None disables that check (unit tests).
        """
        gate = pred_by_match is not None
        pred_by_match = pred_by_match or {}
        bets = []
        for gp in goal_preds:
            match = gp["match"]
            date = gp.get("date", "")
            if gate and match not in pred_by_match:
                continue  # league-gated or stale — never price it
            if match not in odds_full:
                continue

            totals = odds_full[match].get("totals", [])
            if not totals:
                continue

            # Get allowed lines from market rules (if specified)
            ou_over_rules = self.cfg.market_rules.get("O/U_Over", {})
            ou_under_rules = self.cfg.market_rules.get("O/U_Under", {})
            allowed_lines_over = ou_over_rules.get("allowed_lines")  # None = all lines
            allowed_lines_under = ou_under_rules.get("allowed_lines")
            line_min_edge_over = ou_over_rules.get("line_min_edge", {})
            line_min_edge_under = ou_under_rules.get("line_min_edge", {})

            for total in totals:
                line = total.get("line", 0)
                bookmakers = total.get("all_bookmakers", [])
                if not bookmakers:
                    continue
                # Require 3+ bookmakers for reliable market consensus.
                # With 1-2 books, there's no sharp line to benchmark against.
                if total.get("bookmakers_count", len(bookmakers)) < 3:
                    continue

                # Map line to probability. over_X_Y already IS the served blend
                # (w_ml × O/U CatBoost + (1 − w_ml) × Poisson, made in
                # over_under_model; legs audited in ou_ml / ou_poisson). A former
                # "prefer ml_over_under" branch here read a key nothing ever wrote.
                key = f"over_{str(line).replace('.', '_')}"
                model_over = gp.get(key, 0)

                if model_over <= 0:
                    continue
                model_under = 1 - model_over

                pin_over = get_pinnacle_odds(bookmakers, "over")
                pin_under = get_pinnacle_odds(bookmakers, "under")
                if not pin_over or not pin_under or pin_over <= 1 or pin_under <= 1:
                    pin_over = total.get("over", 0)
                    pin_under = total.get("under", 0)
                if pin_over <= 1 or pin_under <= 1:
                    continue

                true_probs = remove_overround([pin_over, pin_under])

                match_pred = pred_by_match.get(match)
                for side_idx, (sel_name, sel_key, model_p, allowed_lines, lme) in enumerate([
                    (f"Over {line}", "over", model_over, allowed_lines_over, line_min_edge_over),
                    (f"Under {line}", "under", model_under, allowed_lines_under, line_min_edge_under),
                ]):
                    # Skip lines not in allowed list (CLV-driven line filtering)
                    if allowed_lines is not None and line not in allowed_lines:
                        continue
                    best_o, best_bk, avg_o, count = find_best_odds(bookmakers, sel_key)
                    # Use real bookmaker count from totals entry when available
                    # (synthetic alt-totals entries have 1 in all_bookmakers but
                    # bookmakers_count reflects the real market depth)
                    real_count = total.get("bookmakers_count", count)
                    if real_count > count:
                        count = real_count
                    pin_o = [pin_over, pin_under][side_idx]
                    # Per-line min_edge override (e.g., O/U 2.5 needs higher edge than 1.5)
                    line_edge = lme.get(line) if lme else None
                    bet = self._make_bet(
                        match, date, f"O/U {line}", sel_name, model_p,
                        true_probs[side_idx], best_o, best_bk, avg_o, pin_o,
                        count, min_edge_override=line_edge, pred=match_pred)
                    if bet:
                        bets.append(bet)
        return bets

    def scan_ah_market(self, predictions: List[Dict], odds_full: Dict,
                       extended: Dict,
                       margin_preds: List[Dict] = None) -> List[ValueBet]:
        """Scan Asian Handicap / Spread market for value.
        Uses ML-calibrated handicap_probs when available, falls back to norm.cdf.
        """
        bets = []
        margin_preds = margin_preds or []
        margin_by_match = {mp["match"]: mp for mp in margin_preds}

        for pred in predictions:
            match = pred["match"]
            date = pred.get("date", "")
            home_xg = pred.get("home_xg", 0)
            away_xg = pred.get("away_xg", 0)
            if match not in odds_full:
                continue

            spreads = odds_full[match].get("spreads", [])
            if not spreads:
                continue

            ml_margin = margin_by_match.get(match, {})
            handicap_probs = ml_margin.get("handicap_probs", {})

            for spread in spreads:
                line = spread.get("line", 0)
                home_hc = spread.get("home_handicap", 0)
                away_hc = spread.get("away_handicap", 0)
                bookmakers = spread.get("all_bookmakers", [])
                if not bookmakers:
                    continue

                # Try ML handicap probs first
                line_key = str(home_hc) if home_hc != 0 else "0"
                ml_probs = handicap_probs.get(line_key, {})
                if not ml_probs:
                    for fmt in [f"{home_hc:.2f}", f"{home_hc:.1f}", f"{home_hc:+.1f}", f"{home_hc:.0f}"]:
                        ml_probs = handicap_probs.get(fmt, {})
                        if ml_probs:
                            break

                if ml_probs and "home" in ml_probs and "away" in ml_probs:
                    home_ah_prob = ml_probs["home"]
                    away_ah_prob = ml_probs["away"]
                else:
                    try:
                        from scipy.stats import norm
                        xg_diff = (home_xg or 1.3) - (away_xg or 1.1)
                        adjusted_diff = xg_diff + home_hc
                        home_ah_prob = norm.cdf(adjusted_diff / 1.3)
                        away_ah_prob = 1 - home_ah_prob
                    except ImportError:
                        continue

                pin_home = get_pinnacle_odds(bookmakers, "home")
                pin_away = get_pinnacle_odds(bookmakers, "away")
                if not pin_home or not pin_away or pin_home <= 1 or pin_away <= 1:
                    pin_home = spread.get("home_odds", 0)
                    pin_away = spread.get("away_odds", 0)
                if pin_home <= 1 or pin_away <= 1:
                    continue

                true_probs = remove_overround([pin_home, pin_away])

                for side_idx, (sel_name, sel_key, model_p) in enumerate([
                    (f"Home {_fmt_hc(home_hc)}", "home", home_ah_prob),
                    (f"Away {_fmt_hc(away_hc)}", "away", away_ah_prob),
                ]):
                    best_o, best_bk, avg_o, count = find_best_odds(bookmakers, sel_key)
                    pin_o = [pin_home, pin_away][side_idx]
                    bet = self._make_bet(
                        match, date, f"AH {line}", sel_name, model_p,
                        true_probs[side_idx], best_o, best_bk, avg_o, pin_o,
                        count)
                    if bet:
                        bets.append(bet)
        return bets

    def scan_dc_market(self, predictions: List[Dict], extended: Dict,
                       odds_full: Dict,
                       extra_odds: Dict = None) -> List[ValueBet]:
        """Scan Double Chance market for value.
        Uses REAL DC bookmaker odds from extra_odds when available.
        Falls back to derived DC from best 1X2 odds otherwise.
        """
        bets = []
        extra_odds = extra_odds or {}

        for pred in predictions:
            match = pred["match"]
            date = pred.get("date", "")
            probs = self._get_betting_probs(pred)
            if not probs or match not in odds_full:
                continue

            h2h = odds_full[match].get("h2h", {})
            bookmakers = h2h.get("all_bookmakers", []) if h2h else []
            if not bookmakers:
                continue

            ph = probs.get("home", 0)
            pd_ = probs.get("draw", 0)
            pa = probs.get("away", 0)

            ext = extended.get(match, {})
            ext_dc = ext.get("double_chance", {})

            pin_h = get_pinnacle_odds(bookmakers, "home")
            pin_d = get_pinnacle_odds(bookmakers, "draw")
            pin_a = get_pinnacle_odds(bookmakers, "away")
            if not all([pin_h > 1, pin_d > 1, pin_a > 1]):
                continue
            true_1x2 = remove_overround([pin_h, pin_d, pin_a])

            real_dc = extra_odds.get(match, {}).get("double_chance", {})

            for dc_name, model_p, sharp_p, ext_key in [
                ("1X (Home or Draw)", ph + pd_, true_1x2[0] + true_1x2[1], "1X"),
                ("X2 (Draw or Away)", pd_ + pa, true_1x2[1] + true_1x2[2], "X2"),
            ]:
                # Blend with extended Poisson DC prob if available
                if ext_dc and ext_key in ext_dc:
                    ext_p = ext_dc[ext_key].get("prob", 0)
                    if ext_p > 0:
                        model_p = 0.6 * ext_p + 0.4 * model_p

                # Try real DC odds first
                real_found = False
                if real_dc and ext_key in real_dc:
                    dc_data = real_dc[ext_key]
                    best_o = dc_data.get("best", 0)
                    avg_o = dc_data.get("avg", 0)
                    n_books = dc_data.get("bookmakers_count", 0)
                    if best_o > 1:
                        bet = self._make_bet(
                            match, date, "DC", dc_name, model_p, sharp_p,
                            best_o, f"DC bookmaker (real)", avg_o, avg_o, n_books,
                            confidence=pred.get("confidence", 0))
                        if bet:
                            bets.append(bet)
                        real_found = True

                # Derived DC bets DISABLED: when real DC odds are unavailable,
                # the fallback used fair price (1/sharp_prob) as best_odds.
                # No bookmaker offers vig-free odds — this created phantom edges
                # of ~5-8% that don't exist in reality. CLV data confirmed all
                # derived DC bets had -6.5% to -9.6% adverse selection.
                # Only bet DC when real bookmaker odds are available (above).
        return bets

    def scan_btts_market(self, btts_preds: List[Dict],
                         extra_odds: Dict) -> List[ValueBet]:
        """Scan BTTS market — DISABLED 2026-05-06.

        btts_predictions.json is written by ml_market_predictions.py which
        falls back to constants (btts_yes=0.5148 for every match) because
        data/models/markets/*.cbm were deleted in the 2026-04-27 cleanup.
        Comparing real bookmaker BTTS odds against a constant 51.48% model
        probability generated bets with no real edge calculation.

        Walkforward HAS a BTTS model at
        data/models/walkforward/serie_a/btts/season_2024-2025.cbm but it
        hasn't been backtested or wired in yet. Re-enable only after a
        held-out backtest with skill_score > 0.02 AND ECE < 0.05.

        See CLAUDE.md "Match Intelligence shows corners/cards" symptom block.
        """
        log.info("  BTTS: skipped (predictions are constants — see CLAUDE.md)")
        return []

    def scan_dnb_market(self, predictions: List[Dict],
                        extra_odds: Dict) -> List[ValueBet]:
        """Scan Draw-No-Bet market using real bookmaker odds."""
        bets = []
        pred_by_match = {p["match"]: p for p in predictions}

        for match_key, extra in extra_odds.items():
            dnb = extra.get("draw_no_bet", {})
            if not dnb:
                continue
            pred = pred_by_match.get(match_key, {})
            if not pred:
                continue
            probs = self._get_betting_probs(pred)
            if not probs:
                continue

            date = pred.get("date", "")
            ph = probs.get("home", 0)
            pd_ = probs.get("draw", 0)
            pa = probs.get("away", 0)
            if pd_ >= 1:
                continue

            dnb_home_prob = ph / (1 - pd_) if pd_ < 1 else 0
            dnb_away_prob = pa / (1 - pd_) if pd_ < 1 else 0

            best_home = dnb.get("best_home", 0)
            best_away = dnb.get("best_away", 0)
            avg_home = dnb.get("home", 0)
            avg_away = dnb.get("away", 0)
            n_books = dnb.get("bookmakers_count", 0)

            if not all([avg_home > 1, avg_away > 1]):
                continue
            true_probs = remove_overround([avg_home, avg_away])

            for side_idx, (sel_name, model_p, best_o) in enumerate([
                ("DNB Home", dnb_home_prob, best_home),
                ("DNB Away", dnb_away_prob, best_away),
            ]):
                if best_o <= 1 or model_p <= 0:
                    continue
                avg_o = [avg_home, avg_away][side_idx]
                bet = self._make_bet(
                    match_key, date, "DNB", sel_name, model_p,
                    true_probs[side_idx], best_o, "DNB book",
                    avg_o, avg_o, n_books)
                if bet:
                    bets.append(bet)
        return bets

    def scan_alt_totals_market(self, goal_preds: List[Dict],
                               extra_odds: Dict,
                               extended: Dict) -> List[ValueBet]:
        """Scan alternate O/U totals using REAL bookmaker odds from extra_odds."""
        bets = []
        gp_by_match = {gp["match"]: gp for gp in goal_preds}

        for match_key, extra in extra_odds.items():
            alt_totals = extra.get("alternate_totals", {})
            if not alt_totals:
                continue

            gp = gp_by_match.get(match_key, {})
            ext = extended.get(match_key, {})
            ext_ou = ext.get("over_under", {})
            home_xg = gp.get("expected_home_goals", 1.3)
            away_xg = gp.get("expected_away_goals", 1.1)
            total_xg = home_xg + away_xg
            date = gp.get("date", "")

            for line_str, line_data in alt_totals.items():
                try:
                    line = float(line_str)
                except (ValueError, TypeError):
                    continue
                # Skip non-standard lines
                if line != int(line) + 0.5 and line != int(line):
                    continue

                best_over = line_data.get("best_over", 0)
                best_under = line_data.get("best_under", 0)
                avg_over = line_data.get("over", 0)
                avg_under = line_data.get("under", 0)
                n_books = line_data.get("bookmakers_count", 0)

                if not all([best_over > 1, best_under > 1, avg_over > 1, avg_under > 1]):
                    continue

                # Get model probability
                over_key = f"over_{line_str}"
                model_over = 0
                if ext_ou and over_key in ext_ou:
                    model_over = ext_ou[over_key].get("prob", 0)
                if model_over <= 0:
                    gp_key = f"over_{line_str.replace('.', '_')}"
                    model_over = gp.get(gp_key, 0)
                if model_over <= 0:
                    try:
                        from scipy.stats import poisson
                        model_over = 1 - sum(
                            poisson.pmf(k, total_xg) for k in range(int(line) + 1))
                    except ImportError:
                        continue

                model_under = 1 - model_over
                true_probs = remove_overround([avg_over, avg_under])

                alt_ou_rules = self.cfg.market_rules.get("Alt_OU", {})
                for side_idx, (sel_name, model_p, best_o) in enumerate([
                    (f"Over {line_str}", model_over, best_over),
                    (f"Under {line_str}", model_under, best_under),
                ]):
                    # Respect allowed_lines for Alt_OU (same as main O/U)
                    a_lines = alt_ou_rules.get("allowed_lines")
                    if a_lines is not None and line not in a_lines:
                        continue
                    avg_o = [avg_over, avg_under][side_idx]
                    lme = alt_ou_rules.get("line_min_edge", {})
                    line_edge = lme.get(line) if lme else None
                    # Pass the max too: the market string resolves to O/U_Over
                    # inside _make_bet, so Alt_OU's own line_max_edge would
                    # otherwise never be consulted (dead config).
                    lxe = alt_ou_rules.get("line_max_edge", {})
                    line_max = lxe.get(line) if lxe else None
                    bet = self._make_bet(
                        match_key, date, f"O/U {line_str}", sel_name, model_p,
                        true_probs[side_idx], best_o, "Alt totals book",
                        avg_o, avg_o, n_books, min_edge_override=line_edge,
                        max_edge_override=line_max)
                    if bet:
                        bets.append(bet)
        return bets

    def scan_alt_spreads_market(self, predictions: List[Dict],
                                extra_odds: Dict,
                                margin_preds: List[Dict]) -> List[ValueBet]:
        """Scan alternate Asian Handicap lines from extra_odds with real bookmaker odds."""
        bets = []
        pred_by_match = {p["match"]: p for p in predictions}
        margin_by_match = {mp["match"]: mp for mp in (margin_preds or [])}

        for match_key, extra in extra_odds.items():
            alt_spreads = extra.get("alternate_spreads", {})
            if not alt_spreads:
                continue
            pred = pred_by_match.get(match_key, {})
            if not pred:
                continue

            date = pred.get("date", "")
            home_xg = pred.get("home_xg", 1.3) or 1.3
            away_xg = pred.get("away_xg", 1.1) or 1.1
            ml_margin = margin_by_match.get(match_key, {})
            handicap_probs = ml_margin.get("handicap_probs", {})

            for line_str, line_data in alt_spreads.items():
                try:
                    line = float(line_str)
                except (ValueError, TypeError):
                    continue

                best_home = line_data.get("best_home", 0)
                best_away = line_data.get("best_away", 0)
                avg_home = line_data.get("home", 0)
                avg_away = line_data.get("away", 0)
                n_books = line_data.get("bookmakers_count", 0)

                has_home = best_home > 1 and avg_home > 1
                has_away = best_away > 1 and avg_away > 1
                if not has_home and not has_away:
                    continue

                ml_probs = handicap_probs.get(line_str, {})
                if not ml_probs:
                    for fmt in [f"{line:.2f}", f"{line:.1f}", f"{line:+.1f}", f"{line:.0f}"]:
                        ml_probs = handicap_probs.get(fmt, {})
                        if ml_probs:
                            break

                if ml_probs and "home" in ml_probs and "away" in ml_probs:
                    home_ah_prob = ml_probs["home"]
                    away_ah_prob = ml_probs["away"]
                else:
                    try:
                        from scipy.stats import norm
                        xg_diff = home_xg - away_xg
                        adjusted_diff = xg_diff + line
                        home_ah_prob = norm.cdf(adjusted_diff / 1.3)
                        away_ah_prob = 1 - home_ah_prob
                    except ImportError:
                        continue

                if has_home and has_away:
                    true_probs = remove_overround([avg_home, avg_away])
                elif has_home:
                    true_probs = [1 / avg_home, 1 - 1 / avg_home]
                else:
                    true_probs = [1 - 1 / avg_away, 1 / avg_away]

                sides = []
                if has_home:
                    sides.append((f"Home {_fmt_hc(line)}", "home", home_ah_prob,
                                  best_home, avg_home, 0))
                if has_away:
                    sides.append((f"Away {_fmt_hc(-line)}", "away", away_ah_prob,
                                  best_away, avg_away, 1))

                for sel_name, sel_key, model_p, best_o, avg_o, idx in sides:
                    bet = self._make_bet(
                        match_key, date, f"AH {line}", sel_name, model_p,
                        true_probs[idx], best_o, "Alt spread book",
                        avg_o, avg_o, n_books)
                    if bet:
                        bets.append(bet)
        return bets

    def scan_cards_market(self, cards_preds: List[Dict]) -> List[ValueBet]:
        """Cards market scanner.
        HONEST ASSESSMENT: Without real bookmaker odds, we cannot identify
        genuine value. Returns empty list until real cards odds are available.
        """
        log.info("  Cards: skipped (no real bookmaker odds)")
        return []

    def scan_corners_market(self, corners_preds: List[Dict]) -> List[ValueBet]:
        """Corners market scanner.
        HONEST ASSESSMENT: Without real bookmaker odds, we cannot identify
        genuine value. Returns empty list until real corners odds are available.
        """
        log.info("  Corners: skipped (no real bookmaker odds)")
        return []

    # -----------------------------------------------------------------
    # INTELLIGENCE LAYER (steam moves + sharp/soft divergence)
    # -----------------------------------------------------------------
    @staticmethod
    def _bet_to_side(bet: ValueBet) -> str:
        """Map a bet's selection to 'home', 'draw', or 'away'."""
        sel = bet.selection.lower()
        if bet.market == "1X2":
            if "home" in sel:
                return "home"
            elif "draw" in sel:
                return "draw"
            else:
                return "away"
        elif bet.market == "DC":
            if "1x" in sel:
                return "home"
            elif "x2" in sel:
                return "away"
            else:
                return "home"
        elif "ah" in bet.market.lower():
            return "home" if "home" in sel else "away"
        return "neutral"

    def apply_intelligence_filters(self, all_bets: List[ValueBet],
                                   odds_movement: Dict,
                                   bookmaker_analysis: Dict) -> List[ValueBet]:
        """Apply market intelligence to adjust bet confidence and filter bad bets.

        Three signals:
        1. CLV KILL-SWITCH: a market whose rolling 50-bet CLV is negative has
           stopped beating the close — no new bets until the window recovers
        2. STEAM MOVES: If sharp money moves against our bet -> penalize heavily
        3. SHARP/SOFT DIVERGENCE: If sharps agree with us -> boost; disagree -> penalize

        (Steam/divergence preserved from ultimate_betting_system; CLV gate 2026-06-11)
        """
        from scripts.betting.clv_tracker import get_market_clv_gate

        filtered = []
        clv_gates: Dict[tuple, Dict] = {}  # (market, league) -> gate, one read per run

        for bet in all_bets:
            gate_key = (bet.market, bet.league)
            if gate_key not in clv_gates:
                try:
                    clv_gates[gate_key] = get_market_clv_gate(bet.market, bet.league)
                except Exception as e:  # gate must never break bet generation
                    clv_gates[gate_key] = {"blocked": False, "reason": f"gate error: {e}"}
            gate = clv_gates[gate_key]
            if gate.get("blocked"):
                print(f"  CLV KILL-SWITCH: skipping {bet.market} {bet.selection} "
                      f"({bet.match}) — {gate['reason']}")
                continue

            match = bet.match
            penalty = 1.0
            intel_notes = []

            # -- STEAM MOVE CHECK --
            movement = odds_movement.get(match, {})
            if movement:
                is_steam = movement.get("is_steam_move", False)
                is_line_move = movement.get("is_line_move", False)
                direction = movement.get("direction", "stable")

                if is_steam or is_line_move:
                    our_side = self._bet_to_side(bet)
                    move_against = False
                    if our_side == "home" and "home_drifting" in direction:
                        move_against = True
                    elif our_side == "away" and "away_drifting" in direction:
                        move_against = True
                    elif our_side == "draw" and "away_strengthening" in direction:
                        move_against = True

                    if move_against:
                        if is_steam:
                            # Hard reject: steam moves against = sharp money
                            # says we're wrong. Live data: -EUR217 in 2-day
                            # steam crash. Don't fade sharp money.
                            intel_notes.append("STEAM_AGAINST_REJECTED")
                            penalty = 0  # Will be caught by penalty <= 0.2 check below
                        else:
                            penalty *= 0.5
                            intel_notes.append("LINE_MOVE_AGAINST")
                    elif not move_against and is_steam:
                        penalty *= 1.15
                        intel_notes.append("STEAM_WITH")

            # -- SHARP/SOFT DIVERGENCE CHECK --
            ba = bookmaker_analysis.get(match, {})
            if ba and ba.get("has_sharp_data"):
                divergence = ba.get("divergence", 0)
                sharp_dir = ba.get("sharp_direction", "neutral")
                our_side = self._bet_to_side(bet)

                if divergence > 0.02:
                    if sharp_dir == our_side:
                        penalty *= 1.10
                        intel_notes.append("SHARP_AGREES")
                    elif sharp_dir != "neutral" and sharp_dir != our_side:
                        penalty *= 0.7
                        intel_notes.append("SHARP_DISAGREES")

                sharp_probs = ba.get("sharp_consensus", {})
                if sharp_probs:
                    sharp_p_for_side = sharp_probs.get(
                        f"prob_{our_side[0].upper()}", 0)
                    if sharp_p_for_side > 0:
                        model_vs_sharp = bet.model_prob - sharp_p_for_side
                        if model_vs_sharp > 0.08:
                            intel_notes.append("MODEL_MUCH_HIGHER_THAN_SHARP")
                        elif model_vs_sharp < -0.05:
                            penalty *= 0.5
                            intel_notes.append("MODEL_BELOW_SHARP")

            # Apply penalty
            if penalty <= 0.2:
                continue
            if penalty != 1.0:
                if self.cfg.staking_mode == "proportional":
                    # Proportional: flat sizing, don't scale by penalty.
                    # Kill threshold: if combined penalties drop below 0.5, remove entirely.
                    if penalty < 0.5:
                        continue
                else:
                    # Kelly staking: scale stake by penalty
                    bet.stake_pct = round(bet.stake_pct * penalty, 2)
                    bet.stake_amount = round(bet.stake_amount * penalty, 2)
                    bet.expected_profit = round(bet.expected_profit * penalty, 2)
                    if bet.stake_pct < 0.3:
                        continue

            bet._intel_notes = intel_notes       # type: ignore
            bet._intel_penalty = round(penalty, 2)  # type: ignore
            filtered.append(bet)

        return filtered

    # -----------------------------------------------------------------
    # COMPOSITE CONFIDENCE SCORE
    # -----------------------------------------------------------------
    @staticmethod
    def compute_confidence_score(bet: ValueBet) -> float:
        """Compute a 0-100 composite score based on what ACTUALLY predicts profit.

        Rebuilt from journal analysis (153 bets). Old weights (confidence=25%)
        had zero predictive value. New weights reflect empirical profit drivers.

        Components:
          Edge sweet spot (0-30):  3-5% edges score highest (inverted above 5%)
          Odds zone quality (0-25): 2.0-2.5 = max, 1.25-1.5 = good, rest = low
          Market liquidity (0-20):  more bookmakers = more reliable odds
          Kelly signal (0-15):      raw Kelly fraction strength
          Pinnacle gap (0-10):      best_odds vs pinnacle — lower gap = sharper line
        """
        # Edge sweet spot: 3-5% = best bucket (+113% ROI). Above 5% = marginal.
        edge = bet.raw_edge * 100  # Convert to pct
        if 3.0 <= edge <= 5.0:
            edge_pts = 30  # Maximum — sweet spot
        elif edge < 3.0:
            edge_pts = edge * 10  # 0-30 linear
        else:
            # 5-7%: declining quality. 5% = 25pts, 7% = 15pts.
            edge_pts = max(10, 30 - (edge - 5.0) * 7.5)

        # Odds zone quality: 2.0-2.5 = +38.2% ROI (best), 1.25-1.5 = +4.3%
        odds = bet.best_odds
        if 2.0 <= odds <= 2.5:
            odds_zone_pts = 25  # Golden zone
        elif 2.5 < odds <= 3.0:
            odds_zone_pts = 18  # Good
        elif 1.25 <= odds < 1.5:
            odds_zone_pts = 15  # Decent but low value
        elif 3.0 < odds <= 4.0:
            odds_zone_pts = 12  # Higher risk
        else:
            odds_zone_pts = 5   # Outside proven ranges

        # Liquidity: more bookmakers = more efficient market = more reliable edge
        liq_pts = min(20, bet.odds_count * 0.7)

        # Kelly signal: higher raw Kelly = stronger perceived edge
        kelly_pts = min(15, bet.kelly_raw * 100)

        # Pinnacle gap: LOWER is better (we want to be AT or BELOW Pinnacle)
        if bet.pinnacle_odds > 1 and bet.best_odds > 1:
            gap = (bet.best_odds - bet.pinnacle_odds) / bet.pinnacle_odds
            # gap < 0 = we're below Pinnacle (good). gap > 0 = above (risky).
            pin_pts = max(0, min(10, 10 - gap * 200))
        else:
            pin_pts = 0

        return round(edge_pts + odds_zone_pts + liq_pts + kelly_pts + pin_pts, 1)

    # -----------------------------------------------------------------
    # PORTFOLIO OPTIMIZATION (two-pass diversity + selection)
    # -----------------------------------------------------------------
    @staticmethod
    def _market_group(market: str) -> str:
        """Map a market string to its broad group."""
        if market == "1X2":
            return "1X2"
        elif market.startswith("O/U"):
            return "O/U"
        elif market.startswith("AH"):
            return "AH"
        elif market == "DC":
            return "DC"
        elif market == "DNB":
            return "DNB"
        elif market == "BTTS":
            return "BTTS"
        elif market.startswith("Cards"):
            return "Cards"
        elif market.startswith("Corners"):
            return "Corners"
        return market

    def _get_pending_journal_exposure(self, current_matches: set = None) -> float:
        """Load pending bets from journal and return total stake as % of bankroll.

        Args:
            current_matches: set of match keys being re-evaluated this run.
                Bets for these matches will be excluded (they'll be replaced).
        """
        try:
            import json
            journal_path = DATA_DIR / "betting" / "bet_journal.json"
            if not journal_path.exists():
                return 0.0
            with open(journal_path) as f:
                journal = json.load(f)
            bets = journal.get("bets", {})
            if not isinstance(bets, dict):
                return 0.0
            current_matches = current_matches or set()
            total_pending = sum(
                v.get("stake", 0) or 0
                for v in bets.values()
                if isinstance(v, dict) and v.get("status") == "pending"
                and v.get("match", "") not in current_matches
            )
            if self.cfg.bankroll > 0:
                return (total_pending / self.cfg.bankroll) * 100
            return 0.0
        except Exception:
            return 0.0

    def optimize_portfolio(self, all_bets: List[ValueBet]) -> List[ValueBet]:
        """Two-pass portfolio construction for maximum diversification.

        PASS 1 -- Guaranteed slots: pick the single best bet from each market type
        PASS 2 -- Fill remaining slots by composite score, with hard cap 50%

        (Preserved from ultimate_betting_system.optimize_portfolio)
        """
        cfg = self.cfg
        if not all_bets:
            return []

        # Score all bets
        for bet in all_bets:
            bet._score = self.compute_confidence_score(bet)  # type: ignore

        # Pre-seed exposure with existing pending journal bets to prevent
        # total portfolio over-exposure across multiple pipeline runs.
        # Exclude matches being re-evaluated (they'll be replaced in journal).
        current_matches = {bet.match_group for bet in all_bets}
        existing_exposure_pct = self._get_pending_journal_exposure(current_matches)
        portfolio_room = max(0.0, cfg.max_portfolio_exposure_pct - existing_exposure_pct)
        effective_daily_cap = min(cfg.max_daily_exposure_pct, portfolio_room)
        if existing_exposure_pct > 0:
            print(f"  Portfolio guard: {existing_exposure_pct:.1f}% already pending in journal "
                  f"(excl {len(current_matches)} current matches), "
                  f"room for {portfolio_room:.1f}% more (cap {cfg.max_portfolio_exposure_pct}%)")

        selected = []
        match_count = {}
        match_exposure = {}
        market_count = {}
        total_exposure = 0.0
        match_selections = {}
        draw_family_count = 0  # Track 1X2 Draw + DC + DNB bets across all matches

        # Pre-seed match_selections from pending journal bets to prevent
        # contradictions across multiple pipeline runs.
        try:
            import json as _json
            _jpath = DATA_DIR / "betting" / "bet_journal.json"
            if _jpath.exists():
                with open(_jpath) as _jf:
                    _jdata = _json.load(_jf)
                for _jb in (_jdata.get("bets", {}) or {}).values():
                    if not isinstance(_jb, dict) or _jb.get("status") != "pending":
                        continue
                    _jm = _jb.get("match", "")
                    if _jm in current_matches:
                        continue  # Will be replaced by current run
                    _jmkt = _jb.get("market", "")
                    _jsel = _jb.get("selection", "")
                    match_selections.setdefault(_jm, set()).add((_jmkt, _jsel.split()[0]))
                    if _jmkt in ("DNB", "DC") or (_jmkt == "1X2" and "DRAW" in _jsel.upper()):
                        draw_family_count += 1
        except Exception:
            pass  # Non-critical — worst case we allow a dupe

        def _is_draw_family(bet) -> bool:
            """Check if bet is draw-sensitive (profits from or is hedged by draw)."""
            if bet.market == "DNB":
                return True
            if bet.market == "DC":
                return True
            if bet.market == "1X2" and "DRAW" in bet.selection.upper():
                return True
            return False

        def _can_add(bet):
            nonlocal draw_family_count
            m = bet.match_group
            if match_count.get(m, 0) >= cfg.max_bets_per_match:
                return False
            if match_exposure.get(m, 0) + bet.stake_pct > cfg.max_match_exposure_pct:
                return False
            if total_exposure + bet.stake_pct > effective_daily_cap:
                return False
            # Draw-family cap: prevent correlation blow-up from too many draw-sensitive bets
            if _is_draw_family(bet) and draw_family_count >= cfg.max_draw_family_bets:
                return False
            existing = match_selections.get(m, set())
            if bet.market == "1X2" and any(mk == "1X2" for mk, _ in existing):
                return False
            if bet.market.startswith("O/U") and any(mk == bet.market for mk, _ in existing):
                return False
            if bet.market.startswith("AH") and any(mk == bet.market for mk, _ in existing):
                return False
            if bet.market == "DC" and any(mk == "DC" for mk, _ in existing):
                return False
            # DC + DNB on same match is redundant (both profit from same team winning)
            if bet.market == "DC" and any(mk == "DNB" for mk, _ in existing):
                return False
            if bet.market == "DNB" and any(mk == "DC" for mk, _ in existing):
                return False
            # 1X2 Draw contradicts DNB (Draw = push on DNB, opposite bet directions)
            if bet.market == "1X2" and "DRAW" in bet.selection.upper():
                if any(mk == "DNB" for mk, _ in existing):
                    return False
            if bet.market == "DNB" and any(
                mk == "1X2" and "DRAW" in sel.upper() for mk, sel in existing
            ):
                return False
            return True

        def _accept(bet):
            nonlocal total_exposure, draw_family_count
            m = bet.match_group
            bet.is_selected = True
            selected.append(bet)
            match_count[m] = match_count.get(m, 0) + 1
            match_exposure[m] = match_exposure.get(m, 0) + bet.stake_pct
            sel_type = (bet.market, bet.selection.split()[0])
            match_selections.setdefault(m, set()).add(sel_type)
            market_count[bet.market] = market_count.get(bet.market, 0) + 1
            total_exposure += bet.stake_pct
            if _is_draw_family(bet):
                draw_family_count += 1

        # Categorize bets by broad market type
        market_groups = {}
        for bet in all_bets:
            key = self._market_group(bet.market)
            market_groups.setdefault(key, []).append(bet)
        for key in market_groups:
            market_groups[key].sort(key=lambda b: b._score, reverse=True)  # type: ignore

        # QUALITY-FIRST SELECTION (replaced 2-pass diversity system)
        # With most markets disabled (only O/U 1.5 + DC active), diversity
        # is less important than quality. Take the best bets by score.
        max_per_market = max(3, int(cfg.max_total_bets * 0.50))
        all_sorted = sorted(all_bets, key=lambda b: b._score, reverse=True)  # type: ignore

        # Track team exposure: max 3 bets involving any single team
        team_counts: dict[str, int] = {}
        MAX_TEAM_EXPOSURE = 3

        for bet in all_sorted:
            if len(selected) >= cfg.max_total_bets:
                break

            # Market concentration cap
            mtype = self._market_group(bet.market)
            current_count = sum(1 for s in selected
                                if self._market_group(s.market) == mtype)
            if current_count >= max_per_market:
                continue

            # Team correlation check: extract teams from match name
            match = bet.match
            teams = [t.strip() for t in match.split(" vs ")] if " vs " in match else []
            if teams:
                if any(team_counts.get(t, 0) >= MAX_TEAM_EXPOSURE for t in teams):
                    continue

            if _can_add(bet):
                _accept(bet)
                for t in teams:
                    team_counts[t] = team_counts.get(t, 0) + 1

        return selected

    # -----------------------------------------------------------------
    # ACCUMULATOR GENERATION
    # -----------------------------------------------------------------
    def generate_accumulators(self, selected_singles: List[ValueBet],
                              parlay_only_bets: Optional[List[ValueBet]] = None,
                              ) -> List[AccumulatorBet]:
        """Generate parlays in two tiers: Safe (high-prob) and Value (edge-based).

        Why parlays work when done right:
        A parlay of N independent +EV legs is itself +EV. The combined EV equals
        the product of individual EVs, minus margin compounding. The key constraints:
        1. Each leg must individually be +EV (model_prob × best_odds > 1)
        2. Legs MUST be from different matches (independence assumption)
        3. Safe parlays: pick legs with model_prob > 70% — things like Over 0.5 goals
           for a team with xG=2.0, or BTTS Yes when both teams have xG > 1.0.
           Combined odds are modest (2-5x) but hit rate is high (~40-55%).
        4. Value parlays: pick legs with highest edge per unit, 2-3 legs max.

        Eligible markets for parlay legs:
        - DC (proven: +39% ROI in backtest)
        - O/U Over (strong for low lines like Over 0.5, Over 1.5)
        - BTTS Yes (parlay-only: disabled for singles but viable as high-prob legs)
        - AH (best single-market ROI: +103%)
        - Alt_OU Over (Over 0.5, Over 1.5 for high-xG teams)
        - 1X2 (only heavy favorites with model_prob >= 65%)
        """
        cfg = self.cfg
        if not selected_singles:
            return []

        # Pool all available legs: portfolio singles + parlay-only bets
        all_available = list(selected_singles)
        if parlay_only_bets:
            all_available.extend(parlay_only_bets)

        accumulators: List[AccumulatorBet] = []

        # --- TIER 1: SAFE PARLAYS ---
        # High-probability legs from any eligible market, 3-4 legs from different matches.
        # The idea: Over 0.5 at @1.15 with 87% prob + BTTS Yes @1.80 with 62% prob + ...
        # Combined odds 2.5-4.0x with 40-55% combined prob → real edge.
        safe_legs = [b for b in all_available
                     if b.model_prob >= cfg.acca_safe_min_prob
                     and self._is_parlay_eligible(b)
                     and b.model_prob * b.best_odds > 1.0]  # Must be individually +EV

        # One best leg per match (highest probability)
        safe_by_match: Dict[str, ValueBet] = {}
        for b in sorted(safe_legs, key=lambda x: x.model_prob, reverse=True):
            if b.match not in safe_by_match:
                safe_by_match[b.match] = b

        safe_independent = sorted(safe_by_match.values(),
                                  key=lambda b: b.model_prob, reverse=True)

        safe_used_matches = set()
        if len(safe_independent) >= 3:
            # Build safe parlay: top N highest-probability legs from different matches
            n_safe = min(cfg.acca_safe_max_legs, len(safe_independent))
            safe_parlay_legs = safe_independent[:n_safe]

            combo_odds = 1.0
            combo_prob = 1.0
            for leg in safe_parlay_legs:
                combo_odds *= leg.best_odds
                combo_prob *= leg.model_prob

            ev = combo_prob * (combo_odds - 1) - (1 - combo_prob)
            if ev > 0:
                # Stake: moderate (1-2% bankroll) since these are high-probability
                stake = round(cfg.bankroll * 0.015, 2)  # 1.5% of bankroll
                stake = max(stake, cfg.bankroll * cfg.min_stake_pct / 100)
                stake = min(stake, cfg.bankroll * cfg.max_stake_pct / 100 * 0.4)

                acc = AccumulatorBet(
                    legs=safe_parlay_legs,
                    combined_odds=round(combo_odds, 2),
                    combined_prob=round(combo_prob, 4),
                    stake_amount=stake,
                    expected_profit=round(stake * ev, 2),
                    ev_per_unit=round(ev, 4),
                    n_legs=n_safe,
                    date=safe_parlay_legs[0].date,
                    category="safe",
                )
                accumulators.append(acc)
                safe_used_matches = {b.match for b in safe_parlay_legs}
                log.info("  Safe parlay: %d legs @ %.2f combined odds (%.1f%% prob, EV=%.3f)",
                         n_safe, combo_odds, combo_prob * 100, ev)

        # --- TIER 2: VALUE PARLAYS ---
        # Edge-based selection from portfolio singles (expanded from DC-only to all eligible)
        value_legs = [b for b in selected_singles
                      if b.edge_pct >= cfg.acca_min_leg_edge_pct
                      and self._is_parlay_eligible(b)
                      and b.model_prob * b.best_odds > 1.0]

        # One best leg per match (highest EV)
        value_by_match: Dict[str, ValueBet] = {}
        for b in value_legs:
            if b.match not in value_by_match:
                value_by_match[b.match] = b
            elif b.ev_per_unit > value_by_match[b.match].ev_per_unit:
                value_by_match[b.match] = b

        # Exclude matches already used in safe parlay
        value_independent = sorted(
            [b for b in value_by_match.values() if b.match not in safe_used_matches],
            key=lambda b: b.ev_per_unit, reverse=True)

        if len(value_independent) < cfg.acca_min_qualifying:
            return accumulators

        value_used = set()
        # Generate doubles from top EV pairs
        for i in range(len(value_independent)):
            if len(accumulators) >= cfg.acca_max_per_day:
                break
            for j in range(i + 1, len(value_independent)):
                if len(accumulators) >= cfg.acca_max_per_day:
                    break
                b1, b2 = value_independent[i], value_independent[j]
                if b1.match in value_used or b2.match in value_used:
                    continue

                legs = [b1, b2]
                combo_odds = b1.best_odds * b2.best_odds
                combo_prob = b1.model_prob * b2.model_prob
                ev = combo_prob * (combo_odds - 1) - (1 - combo_prob)
                if ev <= 0.03:
                    continue

                stake = round(min(b1.stake_amount, b2.stake_amount) * 0.5, 2)
                stake = max(stake, cfg.bankroll * cfg.min_stake_pct / 100)
                stake = min(stake, cfg.bankroll * cfg.max_stake_pct / 100 * 0.5)

                acc = AccumulatorBet(
                    legs=legs,
                    combined_odds=round(combo_odds, 2),
                    combined_prob=round(combo_prob, 4),
                    stake_amount=stake,
                    expected_profit=round(stake * ev, 2),
                    ev_per_unit=round(ev, 4),
                    n_legs=2,
                    date=b1.date or b2.date,
                    category="value",
                )
                accumulators.append(acc)
                value_used.add(b1.match)
                value_used.add(b2.match)

        # Try one triple if enough independent bets remain
        if len(value_independent) >= 6 and len(accumulators) < cfg.acca_max_per_day:
            remaining = [b for b in value_independent if b.match not in value_used]
            if len(remaining) >= 3:
                legs = remaining[:3]
                combo_odds = 1.0
                combo_prob = 1.0
                for leg in legs:
                    combo_odds *= leg.best_odds
                    combo_prob *= leg.model_prob
                combo_odds = round(combo_odds, 2)

                ev = combo_prob * (combo_odds - 1) - (1 - combo_prob)
                if ev > 0:
                    stake = round(min(lg.stake_amount for lg in legs) * 0.3, 2)
                    stake = max(stake, cfg.bankroll * cfg.min_stake_pct / 100)

                    acc = AccumulatorBet(
                        legs=legs,
                        combined_odds=combo_odds,
                        combined_prob=round(combo_prob, 4),
                        stake_amount=stake,
                        expected_profit=round(stake * ev, 2),
                        ev_per_unit=round(ev, 4),
                        n_legs=3,
                        date=legs[0].date,
                        category="value",
                    )
                    accumulators.append(acc)

        return accumulators

    @staticmethod
    def _is_parlay_eligible(bet: ValueBet) -> bool:
        """Check if a bet qualifies as a parlay leg.

        Eligible: DC, O/U Over, BTTS Yes, AH, Alt O/U Over, 1X2 favorites.
        Not eligible: O/U Under (low prob), BTTS No (defeats purpose),
                      DNB (redundant with 1X2), Under lines (low-variance legs).
        """
        market = bet.market
        sel = bet.selection

        # DC: proven parlay performer (85.7% WR, +39% ROI in backtest)
        if market == "DC":
            return True
        # O/U Over: great parlay leg especially at low lines (Over 0.5, Over 1.5)
        if market.startswith("O/U") and "Over" in sel:
            return True
        # BTTS Yes: viable parlay leg when both teams expected to score
        if market == "BTTS" and "Yes" in sel:
            return True
        # AH: best single-market ROI (+103% in backtest)
        if market.startswith("AH"):
            return True
        # Alt AH
        if market.startswith("Alt_AH") or "AH" in market:
            return True
        # 1X2: only heavy favorites for parlays (adds modest odds safely)
        if market == "1X2" and bet.model_prob >= 0.65:
            return True
        return False

    # -----------------------------------------------------------------
    # MONTE CARLO SIMULATION
    # -----------------------------------------------------------------
    def monte_carlo_simulation(self, slip: BetSlip) -> Optional[Dict]:
        """Simulate portfolio outcomes to show expected distribution.

        For each simulation, each bet wins/loses based on blended probability
        (60% model + 40% sharp) for realistic simulation.

        (Preserved from ultimate_betting_system.monte_carlo_simulation)
        """
        n_sims = self.cfg.monte_carlo_sims
        rng = random.Random(42)

        if not slip.bets:
            return None

        results = []
        for _ in range(n_sims):
            pnl = 0
            for bet in slip.bets:
                sim_prob = 0.6 * bet.model_prob + 0.4 * bet.sharp_implied_prob
                if rng.random() < sim_prob:
                    pnl += bet.stake_amount * (bet.best_odds - 1)
                else:
                    pnl -= bet.stake_amount
            results.append(pnl)

        results.sort()
        p5 = results[int(n_sims * 0.05)]
        p25 = results[int(n_sims * 0.25)]
        p50 = results[int(n_sims * 0.50)]
        p75 = results[int(n_sims * 0.75)]
        p95 = results[int(n_sims * 0.95)]
        avg = sum(results) / n_sims
        profit_pct = sum(1 for r in results if r > 0) / n_sims * 100

        return {
            "n_sims": n_sims,
            "profit_probability_pct": round(profit_pct, 1),
            "avg_pnl": round(avg, 2),
            "p5": round(p5, 2),
            "p25": round(p25, 2),
            "median": round(p50, 2),
            "p75": round(p75, 2),
            "p95": round(p95, 2),
            "worst": round(results[0], 2),
            "best": round(results[-1], 2),
        }

    # -----------------------------------------------------------------
    # BET SLIP GENERATION
    # -----------------------------------------------------------------
    def generate_bet_slip(self, selected: List[ValueBet]) -> BetSlip:
        """Generate final bet slip with all details."""
        return BetSlip(
            bets=selected,
            total_stake=round(sum(b.stake_amount for b in selected), 2),
            total_ev=round(sum(b.expected_profit for b in selected), 2),
            exposure_pct=round(sum(b.stake_pct for b in selected), 2),
            n_matches=len(set(b.match for b in selected)),
            generated_at=datetime.now().isoformat(),
            bankroll=self.cfg.bankroll,
        )

    # -----------------------------------------------------------------
    # MAIN RUN METHOD
    # -----------------------------------------------------------------
    def run(self) -> Tuple[BetSlip, List[ValueBet]]:
        """Run the complete unified betting engine pipeline.

        Returns: (bet_slip, all_value_bets)
        """
        cfg = self.cfg

        # Use real-time bankroll instead of static initial
        try:
            from scripts.betting.bankroll_loader import get_effective_bankroll
            live_bankroll = get_effective_bankroll()
            if live_bankroll > 0:
                cfg.bankroll = live_bankroll
                log.info("Using live bankroll: €%.2f", live_bankroll)
        except Exception as e:
            log.warning("Failed to load live bankroll, using default: %s", e)

        log.info("=" * 70)
        log.info("UNIFIED BETTING ENGINE v1.0 -- Consolidated Professional-Grade")
        log.info("Bankroll: %.0f | Kelly: %.0f%% | Min Edge: %.0f%%",
                 cfg.bankroll, cfg.kelly_fraction * 100, cfg.min_edge_pct)
        log.info("=" * 70)

        # -- Load ALL data sources --
        log.info("\nLoading live data...")
        predictions = load_predictions()
        odds_full = load_odds_full()
        extended = load_extended_markets()
        goal_preds = load_goal_predictions()
        btts_preds = load_btts_predictions()
        cards_preds = load_cards_predictions()
        corners_preds = load_corners_predictions()
        extra_odds = load_extra_market_odds()
        movement = load_odds_movement()
        bk_analysis = load_bookmaker_analysis()
        margin_preds = load_margin_predictions()

        # -- Filter out past matches (prevent betting on already-played games) --
        today_str = datetime.now().strftime("%Y-%m-%d")
        pre_filter = len(predictions)
        predictions = [
            p for p in predictions
            if p.get("date", "9999-99-99")[:10] >= today_str
        ]
        if pre_filter > len(predictions):
            log.warning("  Filtered out %d past matches (date < %s)",
                        pre_filter - len(predictions), today_str)

        # Auxiliary prediction files carry BOTH leagues and no league field —
        # gate them to the matches that passed the per-league gate above.
        _allowed = {p.get("match") for p in predictions}
        goal_preds = _gate_aux_predictions(goal_preds, _allowed, "goal")
        btts_preds = _gate_aux_predictions(btts_preds, _allowed, "btts")
        cards_preds = _gate_aux_predictions(cards_preds, _allowed, "cards")
        corners_preds = _gate_aux_predictions(corners_preds, _allowed, "corners")
        margin_preds = _gate_aux_predictions(margin_preds, _allowed, "margin")

        log.info("  Predictions:    %d matches", len(predictions))
        log.info("  Odds (40+ bk):  %d matches", len(odds_full))
        log.info("  Extended mkts:  %d matches", len(extended))
        log.info("  Goal preds:     %d matches", len(goal_preds))
        log.info("  BTTS preds:     %d matches", len(btts_preds))
        log.info("  Cards preds:    %d matches", len(cards_preds))
        log.info("  Corners preds:  %d matches", len(corners_preds))
        log.info("  Margin preds:   %d matches (ML handicap probs)", len(margin_preds))
        log.info("  Extra odds:     %d matches (BTTS, DC, alt totals, DNB)",
                 len(extra_odds))
        log.info("  Odds movement:  %d matches (steam/line moves)", len(movement))
        log.info("  Sharp/soft:     %d matches (bookmaker analysis)", len(bk_analysis))

        if not predictions or not odds_full:
            log.error("Missing critical data. Run prediction pipeline first.")
            empty_slip = BetSlip(generated_at=datetime.now().isoformat(),
                                 bankroll=cfg.bankroll)
            return empty_slip, []

        # -- Merge alternate totals from extra_odds into odds_full --
        # The per-event endpoint returns lines 0.5-4.5+ that the bulk endpoint misses.
        # Convert alternate_totals format → totals array format so scan_ou_market sees them.
        if extra_odds:
            alt_added = 0
            for match_key, em in extra_odds.items():
                if match_key not in odds_full:
                    continue
                alt_totals = em.get("alternate_totals", {})
                if not alt_totals:
                    continue
                existing_lines = {t.get("line") for t in odds_full[match_key].get("totals", [])}
                for line_str, entry in alt_totals.items():
                    line = float(line_str)
                    if line in existing_lines:
                        continue  # Already have this line from bulk endpoint
                    # Only include standard half-lines the model can price
                    if line not in (0.5, 1.5, 2.5, 3.5, 4.5):
                        continue
                    over_avg = entry.get("over", 0)
                    under_avg = entry.get("under", 0)
                    if over_avg <= 1 or under_avg <= 1:
                        continue
                    best_over = entry.get("best_over", over_avg)
                    best_under = entry.get("best_under", under_avg)
                    bk_count = entry.get("bookmakers_count", 1)
                    # Require minimum 3 bookmakers for a meaningful market consensus.
                    # With 1-2 books, the "average" is just one bookie's opinion — not
                    # a sharp line, so edge calculations are unreliable.
                    if bk_count < 3:
                        continue
                    # Build synthetic totals entry matching scan_ou_market format
                    synth = {
                        "line": line,
                        "over": over_avg,
                        "under": under_avg,
                        "best_over": best_over,
                        "best_under": best_under,
                        "bookmakers_count": bk_count,
                        "all_bookmakers": [
                            {"bookmaker": "Alt totals book", "over": best_over, "under": best_under}
                        ],
                    }
                    odds_full[match_key].setdefault("totals", []).append(synth)
                    alt_added += 1
            if alt_added:
                log.info("Merged %d alternate total lines into O/U scanner", alt_added)

        # -- Scan markets for value (respecting per-market enable/disable) --
        mr = cfg.market_rules

        # Auto-kill losing markets from risk controls
        try:
            from scripts.betting.risk_controls import check_market_health, _load_settled_bets, RiskConfig
            _health = check_market_health(_load_settled_bets(), RiskConfig())
            for _killed in _health.get("kill_markets", []):
                _mkt = _killed.split("(")[0].strip()
                for _rule_key in list(mr.keys()):
                    if _mkt.lower() in _rule_key.lower():
                        mr[_rule_key]["enabled"] = False
                        log.warning("Auto-kill: %s disabled (%s)", _rule_key, _killed)
        except Exception:
            pass
        self.near_misses = []
        log.info("\nScanning markets for value (market rules applied)...")
        all_bets = []

        # Build prediction lookup for situational tagging in scanners
        pred_by_match = {p["match"]: p for p in predictions}

        # Run 1X2 scanner if ANY 1X2 variant (Home, Away, Draw) is enabled
        _1x2_home_on = mr.get("1X2", {}).get("enabled", False)
        _1x2_away_on = mr.get("1X2_Away", {}).get("enabled", False)
        _1x2_draw_on = mr.get("1X2_Draw", {}).get("enabled", False)
        if _1x2_home_on or _1x2_away_on or _1x2_draw_on:
            bets_1x2 = self.scan_1x2_market(predictions, odds_full)
            enabled_parts = []
            if _1x2_home_on: enabled_parts.append("Home")
            if _1x2_away_on: enabled_parts.append("Away")
            if _1x2_draw_on: enabled_parts.append("Draw")
            log.info("  1X2:      %d value bets (%s enabled)", len(bets_1x2), "+".join(enabled_parts))
            all_bets.extend(bets_1x2)
        else:
            log.info("  1X2:      DISABLED (all variants off)")

        ou_over_on = mr.get("O/U_Over", {}).get("enabled", True)
        ou_under_on = mr.get("O/U_Under", {}).get("enabled", True)
        if ou_over_on or ou_under_on:
            bets_ou = self.scan_ou_market(goal_preds, odds_full, pred_by_match)
            n_over = sum(1 for b in bets_ou if "Over" in b.selection)
            n_under = sum(1 for b in bets_ou if "Under" in b.selection)
            log.info("  O/U Over: %d value bets (min_edge=%.0f%%)%s",
                     n_over,
                     mr.get("O/U_Over", {}).get("min_edge_pct", cfg.min_edge_pct),
                     "" if ou_over_on else " DISABLED")
            log.info("  O/U Undr: %d value bets%s",
                     n_under,
                     "" if ou_under_on else " DISABLED (backtest: -1%% to -14%% ROI)")
            all_bets.extend(bets_ou)
        else:
            log.info("  O/U:      DISABLED (both Over and Under)")

        if mr.get("AH", {}).get("enabled", True):
            bets_ah = self.scan_ah_market(predictions, odds_full, extended,
                                          margin_preds=margin_preds)
            log.info("  AH:       %d value bets", len(bets_ah))
            all_bets.extend(bets_ah)
        else:
            log.info("  AH:       DISABLED (backtest: -27%% ROI at all thresholds)")

        if mr.get("DC", {}).get("enabled", True):
            bets_dc = self.scan_dc_market(predictions, extended, odds_full,
                                          extra_odds=extra_odds)
            log.info("  DC:       %d value bets", len(bets_dc))
            all_bets.extend(bets_dc)
        else:
            log.info("  DC:       DISABLED")

        parlay_extra = []  # Legs from disabled markets, eligible for parlays only

        if mr.get("BTTS", {}).get("enabled", True):
            bets_btts = self.scan_btts_market(btts_preds, extra_odds)
            log.info("  BTTS:     %d value bets", len(bets_btts))
            all_bets.extend(bets_btts)
        else:
            # BTTS disabled for singles but scan for parlay legs if configured
            if btts_preds and "BTTS" in getattr(cfg, 'acca_parlay_only_markets', []):
                _btts_parlay = self.scan_btts_market(btts_preds, extra_odds)
                parlay_extra.extend(_btts_parlay)
                log.info("  BTTS:     DISABLED for singles (%d legs for parlays)",
                         len(_btts_parlay))
            else:
                log.info("  BTTS:     DISABLED")

        # Alt O/U: DISABLED — alternate totals are now merged into odds_full
        # before scan_ou_market (see "Merge alternate totals" block above).
        # Running scan_alt_totals_market would duplicate those bets AND bypass
        # the 3-bookmaker minimum check. scan_ou_market handles all lines now.
        log.info("  Alt O/U:  merged into O/U scanner (no separate scan)")

        if mr.get("DNB", {}).get("enabled", True):
            bets_dnb = self.scan_dnb_market(predictions, extra_odds)
            log.info("  DNB:      %d value bets", len(bets_dnb))
            all_bets.extend(bets_dnb)
        else:
            log.info("  DNB:      DISABLED")

        if mr.get("Alt_AH", {}).get("enabled", True):
            bets_alt_ah = self.scan_alt_spreads_market(predictions, extra_odds,
                                                       margin_preds)
            log.info("  Alt AH:   %d value bets", len(bets_alt_ah))
            all_bets.extend(bets_alt_ah)
        else:
            log.info("  Alt AH:   DISABLED (backtest: AH model miscalibrated)")

        if mr.get("Cards", {}).get("enabled", True):
            bets_cards = self.scan_cards_market(cards_preds)
            all_bets.extend(bets_cards)

        if mr.get("Corners", {}).get("enabled", True):
            bets_corners = self.scan_corners_market(corners_preds)
            all_bets.extend(bets_corners)

        log.info("\n  TOTAL: %d value bets across %d market types",
                 len(all_bets), len(set(b.market for b in all_bets)) if all_bets else 0)

        # -- Apply intelligence filters --
        pre_intel = len(all_bets)
        if movement or bk_analysis:
            log.info("\nApplying market intelligence filters...")
            all_bets = self.apply_intelligence_filters(
                all_bets, movement, bk_analysis)
            removed = pre_intel - len(all_bets)
            adjusted = sum(1 for b in all_bets
                           if hasattr(b, '_intel_penalty') and b._intel_penalty != 1.0)
            log.info("  Removed: %d bets (sharp money against)", removed)
            log.info("  Adjusted: %d bets (stake modified)", adjusted)
            log.info("  Remaining: %d value bets", len(all_bets))

        self.all_bets = all_bets
        self._log_near_misses()

        # Advisory: the best-priced angle for EVERY match on the slate,
        # regardless of whether anything cleared the betting bar. Never
        # enters the money path.
        try:
            self.best_picks = best_pick_per_match(
                predictions, odds_full, goal_preds, extra_odds, btts_preds,
                band=(cfg.min_edge_pct, cfg.max_edge_pct))
            log.info("  Best-pick advisory: %d matches ranked", len(self.best_picks))
        except Exception as e:
            self.best_picks = []
            log.warning("  Best-pick advisory failed (bets unaffected): %s", e)

        # -- Portfolio optimization --
        log.info("\nOptimizing portfolio...")
        selected = self.optimize_portfolio(all_bets)
        log.info("  Selected: %d bets from %d candidates",
                 len(selected), len(all_bets))
        self.selected = selected

        # -- Generate accumulators (parlays) --
        accumulators = self.generate_accumulators(selected,
                                                   parlay_only_bets=parlay_extra)
        if accumulators:
            n_safe = sum(1 for a in accumulators if a.category == "safe")
            n_value = len(accumulators) - n_safe
            log.info("  Parlays: %d generated (%d safe, %d value)", len(accumulators), n_safe, n_value)
            for acc in accumulators:
                log.info("    [%s] %d-fold @ %.2f (%.1f%% prob): %s",
                         acc.category.upper(), acc.n_legs, acc.combined_odds,
                         acc.combined_prob * 100, acc.selections)
        self.accumulators = accumulators

        # -- Generate bet slip --
        slip = self.generate_bet_slip(selected)
        slip._accumulators = accumulators  # type: ignore
        self.slip = slip

        # -- Entry timing: AVOID = hard block, WAIT = keep but flag --
        try:
            from scripts.betting.entry_timer import EntryTimingAnalyzer
            analyzer = EntryTimingAnalyzer()
            if analyzer.load_snapshots() > 0:
                to_remove = []
                for bet in selected:
                    sel = bet.selection.lower()
                    if "home" in sel or "1x" in sel:
                        outcome = "home"
                    elif "draw" in sel:
                        outcome = "draw"
                    elif "away" in sel or "x2" in sel:
                        outcome = "away"
                    else:
                        continue
                    rec = analyzer.recommend_entry(
                        bet.match, outcome, model_prob=bet.model_prob,
                    )
                    bet.entry_timing = rec
                    if rec.get("action") == "AVOID":
                        log.info("  TIMING BLOCK: %s %s — AVOID (%s)",
                                 bet.match, bet.selection, rec.get("reason", ""))
                        to_remove.append(bet)
                    elif rec.get("action") == "WAIT":
                        log.info("  TIMING: %s %s — WAIT (%s)",
                                 bet.match, bet.selection, rec.get("reason", ""))
                # Remove AVOID bets from selected
                if to_remove:
                    selected = [b for b in selected if b not in to_remove]
                    log.info("Removed %d bets flagged AVOID by entry timer", len(to_remove))
                    # Rebuild slip
                    slip = BetSlip(
                        bets=selected,
                        total_stake=sum(b.stake_amount for b in selected),
                        expected_profit=sum(b.expected_profit for b in selected),
                        match_count=len(set(b.match for b in selected)),
                    )
        except Exception as e:
            log.debug("Entry timing enrichment skipped: %s", e)

        return slip, all_bets


def best_pick_per_match(predictions: List[Dict], odds_full: Dict,
                        goal_preds: List[Dict], extra_odds: Dict,
                        btts_preds: List[Dict] = None,
                        band: Tuple[float, float] = (2.0, 12.0)) -> List[Dict]:
    """One ranked best pick for EVERY match on the gated slate — advisory only.

    Scans every market where we hold BOTH a model probability and a real
    price: 1X2 (ensemble), O/U half-integer lines (dedicated goals model),
    DC/DNB (arithmetic on the ensemble), BTTS (noise-tier model, labeled).
    Ranks by edge = p*best_odds - 1. Tiers: VALIDATED (1X2, O/U) and DERIVED
    (DC, DNB — same ensemble, thinner market) compete on edge; NOISE (BTTS)
    can only top a match when nothing validated has a price, and is labeled.

    Edges are only credible inside the betting band: live results showed
    edges past ~max_edge are model overconfidence (>10% edge = 38% WR),
    so a pick is `in_band` when band[0] <= edge <= band[1]. In-band picks
    outrank out-of-band ones regardless of raw edge size; an out-of-band
    "monster edge" surfaces last and flagged, never as the headline.

    INFORMATIONAL ONLY: this list never touches the money path — journaled
    bets still come exclusively from the edge-gated scanners above. It exists
    so every match shows its best-priced angle even when nothing clears the
    betting bar.
    """
    gp_by_match = {g.get("match"): g for g in (goal_preds or [])}
    btts_by_match = {b.get("match"): b for b in (btts_preds or [])}
    picks = []
    for pred in predictions:
        match = pred.get("match")
        probs = pred.get("probabilities") or {}
        ph, pd_, pa = (probs.get("home"), probs.get("draw"), probs.get("away"))
        if None in (ph, pd_, pa):
            continue
        cands = []

        def cand(market, sel, prob, odds, tier, books=99):
            if not odds or odds <= 1.01 or prob is None or books < 2:
                return
            cands.append({"market": market, "selection": sel,
                          "model_prob": round(float(prob), 4),
                          "odds": round(float(odds), 2),
                          "edge_pct": round((float(prob) * float(odds) - 1) * 100, 1),
                          "tier": tier})

        om = odds_full.get(match) or {}
        h2h = om.get("h2h") or {}
        cand("1X2", "Home", ph, h2h.get("best_home"), "validated",
             h2h.get("bookmakers_count", 0))
        cand("1X2", "Draw", pd_, h2h.get("best_draw"), "validated",
             h2h.get("bookmakers_count", 0))
        cand("1X2", "Away", pa, h2h.get("best_away"), "validated",
             h2h.get("bookmakers_count", 0))

        xm = extra_odds.get(match) or {}
        gp = gp_by_match.get(match) or {}
        # O/U: dedicated goals model carries over_0_5..over_4_5 → half lines only
        seen_lines = set()
        def ou_cand(line, best_over, best_under, books):
            if line in seen_lines or (line * 2) % 2 != 1:
                return
            key = f"over_{str(line).replace('.', '_')}"
            p_over = gp.get(key)
            if p_over is None:
                return
            seen_lines.add(line)
            cand("O/U", f"Over {line}", p_over, best_over, "validated", books)
            cand("O/U", f"Under {line}", 1 - p_over, best_under, "validated", books)
        for t in om.get("totals") or []:
            try:
                ou_cand(float(t.get("line")), t.get("best_over"),
                        t.get("best_under"), t.get("bookmakers_count", 0))
            except (TypeError, ValueError):
                continue
        for line_s, t in (xm.get("alternate_totals") or {}).items():
            try:
                ou_cand(float(line_s), t.get("best_over"), t.get("best_under"),
                        t.get("bookmakers_count", 0))
            except (TypeError, ValueError):
                continue

        dc = xm.get("double_chance") or {}
        for sel, prob in (("1X", ph + pd_), ("X2", pd_ + pa), ("12", ph + pa)):
            d = dc.get(sel) or {}
            cand("DC", sel, prob, d.get("best"), "derived",
                 d.get("bookmakers_count", 0))
        dnb = xm.get("draw_no_bet") or {}
        if ph + pa > 0:
            cand("DNB", "Home", ph / (ph + pa), dnb.get("best_home"),
                 "derived", dnb.get("bookmakers_count", 0))
            cand("DNB", "Away", pa / (ph + pa), dnb.get("best_away"),
                 "derived", dnb.get("bookmakers_count", 0))

        bt = btts_by_match.get(match) or {}
        bto = xm.get("btts") or {}
        cand("BTTS", "Yes", bt.get("btts_yes"), bto.get("best_yes"),
             "noise", bto.get("bookmakers_count", 0))
        cand("BTTS", "No", bt.get("btts_no"), bto.get("best_no"),
             "noise", bto.get("bookmakers_count", 0))

        if not cands:
            continue
        lo, hi = band
        for c in cands:
            c["in_band"] = lo <= c["edge_pct"] <= hi
        trusted = [c for c in cands if c["tier"] != "noise"]
        pool = trusted or cands
        # in-band first (credible edges), then by closeness to the band from
        # below (a +1% edge beats a +150% mirage), then raw edge
        pool.sort(key=lambda c: (
            not c["in_band"],
            -c["edge_pct"] if c["in_band"] else abs(c["edge_pct"] - hi),
        ))
        picks.append({
            "match": match,
            "date": pred.get("date"),
            "time": pred.get("time"),
            "league": pred.get("league"),
            "best": pool[0],
            "top3": pool[:3],
            "n_markets_scanned": len(cands),
        })
    picks.sort(key=lambda x: (not x["best"]["in_band"],
                              -x["best"]["edge_pct"] if x["best"]["in_band"]
                              else abs(x["best"]["edge_pct"])))
    return picks


# =============================================================================
# 6. BET HISTORY TRACKER
# =============================================================================
def load_bet_history() -> List[Dict]:
    """Load bet history from file."""
    path = UPCOMING / "bet_history.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def save_bet_history(history: List[Dict]):
    """Save bet history."""
    from config.settings import atomic_write_json
    path = UPCOMING / "bet_history.json"
    atomic_write_json(path, history)


def record_bets(slip: BetSlip) -> int:
    """Record current bet slip to history for tracking."""
    history = load_bet_history()
    count = 0
    existing_ids = {h["id"] for h in history}
    for bet in slip.bets:
        entry = {
            "id": f"{bet.date}_{bet.match}_{bet.market}_{bet.selection.upper()}".replace(" ", "_"),
            "date": bet.date,
            "match": bet.match,
            "market": bet.market,
            "selection": bet.selection,
            "model_prob": bet.model_prob,
            "edge_pct": bet.edge_pct,
            "best_odds": bet.best_odds,
            "best_bookmaker": bet.best_bookmaker,
            "stake": bet.stake_amount,
            "ev_per_unit": bet.ev_per_unit,
            "placed_at": slip.generated_at,
            "result": None,
            "profit": None,
        }
        if entry["id"] not in existing_ids:
            history.append(entry)
            count += 1
    save_bet_history(history)

    # Also write to unified journal
    try:
        from scripts.betting.bet_journal import add_bet as journal_add_bet
        for bet in slip.bets:
            journal_add_bet({
                "match": bet.match,
                "date": bet.date,
                "market": bet.market,
                "selection": bet.selection,
                "model_prob": bet.model_prob,
                "sharp_implied_prob": bet.sharp_implied_prob,
                "edge_pct": bet.edge_pct,
                "odds": bet.best_odds,
                "bookmaker": bet.best_bookmaker,
                "avg_odds": bet.avg_odds,
                "pinnacle_odds": bet.pinnacle_odds,
                "stake": bet.stake_amount,
                "confidence": bet.confidence_tier,
                "placed_at": slip.generated_at,
                "league": getattr(bet, 'league', 'serie_a'),
            })
    except Exception as e:
        log.debug("Failed to write to bet journal: %s", e)

    return count


def get_track_record(history: List[Dict]) -> Dict:
    """Compute cumulative track record from history."""
    settled = [h for h in history if h.get("result") is not None]
    if not settled:
        return {"n_settled": 0, "n_pending": len(history)}
    wins = sum(1 for h in settled if h["result"] == "WIN")
    losses = sum(1 for h in settled if h["result"] == "LOSS")
    total_staked = sum(h.get("stake", 0) for h in settled)
    total_profit = sum(h.get("profit", 0) for h in settled
                       if h.get("profit") is not None)
    return {
        "n_settled": len(settled),
        "n_pending": len(history) - len(settled),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / max(len(settled), 1), 3),
        "total_staked": round(total_staked, 2),
        "total_profit": round(total_profit, 2),
        "roi": round(total_profit / max(total_staked, 1) * 100, 2),
    }


def _check_drawdown(history: List[Dict]):
    """Check for excessive drawdown and warn user."""
    settled = [h for h in history if h.get("result") is not None]
    if len(settled) < 5:
        return

    running = 0
    peak = 0
    max_dd = 0
    for h in settled:
        running += h.get("profit", 0)
        peak = max(peak, running)
        dd = peak - running
        max_dd = max(max_dd, dd)

    total_staked = sum(h.get("stake", 0) for h in settled)
    dd_pct = max_dd / max(total_staked / len(settled) * 10, 1) * 100

    if max_dd > 0:
        print(f"\n  Drawdown Analysis:")
        print(f"    Current P/L: {running:+.1f}")
        print(f"    Peak P/L:    {peak:+.1f}")
        print(f"    Max Drawdown: {max_dd:.1f}")

    if dd_pct > 30:
        print(f"    WARNING: Significant drawdown detected ({dd_pct:.0f}%).")
        print(f"    Consider reducing stakes by 50% until recovery.")

    streak = 0
    max_streak = 0
    for h in settled:
        if h.get("result") == "LOSS":
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    if max_streak >= 5:
        print(f"    Losing streak: {max_streak} consecutive losses detected.")
        print(f"    This is within normal variance for value betting. Stay disciplined.")


# =============================================================================
# 7. RESULTS UPDATER
# =============================================================================
def update_results():
    """Interactive CLI to mark bet results after match day."""
    history = load_bet_history()
    pending = [h for h in history if h.get("result") is None]

    if not pending:
        print("No pending bets to update.")
        return

    print(f"\n{'=' * 70}")
    print(f"  BET RESULTS UPDATER -- {len(pending)} pending bets")
    print(f"{'=' * 70}")

    updated = 0
    for i, bet in enumerate(pending):
        print(f"\n  [{i+1}/{len(pending)}] {bet['match']} | "
              f"{bet['market']} -> {bet['selection']}")
        print(f"         Odds: {bet['best_odds']:.2f} | Stake: {bet['stake']:.2f}")

        while True:
            r = input("         Result (W=win, L=loss, P=push, S=skip): ").strip().upper()
            if r in ("W", "L", "P", "S"):
                break
            print("         Invalid. Enter W, L, P, or S")

        if r == "S":
            continue
        elif r == "W":
            bet["result"] = "WIN"
            bet["profit"] = round(bet["stake"] * (bet["best_odds"] - 1), 2)
            updated += 1
        elif r == "L":
            bet["result"] = "LOSS"
            bet["profit"] = -bet["stake"]
            updated += 1
        elif r == "P":
            bet["result"] = "PUSH"
            bet["profit"] = 0
            updated += 1

    if updated > 0:
        save_bet_history(history)
        print(f"\n  Updated {updated} bets. Recalculating track record...")
        record = get_track_record(history)
        if record.get("n_settled", 0) > 0:
            print(f"  Record: {record['wins']}W / {record['losses']}L "
                  f"({record['win_rate']:.0%}) | ROI: {record['roi']:+.1f}% | "
                  f"P/L: {record['total_profit']:+.1f}")

        # CLV tracking integration
        try:
            from scripts.betting.clv_tracker import track_clv_for_settled_bets, get_clv_summary
            settled = [h for h in history if h.get("result") in ("WIN", "LOSS")]
            if settled:
                clv_result = track_clv_for_settled_bets(settled)
                if clv_result.get("new_tracked", 0) > 0:
                    print(f"\n  CLV computed for {clv_result['new_tracked']} new bets")
                summary = get_clv_summary()
                if summary["total_tracked"] > 0:
                    print(f"     Running CLV: {summary['running_clv_pct']:+.2f}% "
                          f"({summary['positive_clv_count']}/{summary['total_tracked']} positive)")
        except Exception as e:
            log.warning(f"Failed to compute CLV for settled bets: {e}")

    _check_drawdown(history)


# =============================================================================
# 8. OUTPUT / DISPLAY
# =============================================================================
def print_bet_slip(engine: UnifiedBettingEngine, slip: BetSlip,
                   all_value_bets: List[ValueBet]):
    """Print beautiful, actionable bet slip to console."""
    print("\n" + "=" * 95)
    print("  UNIFIED BETTING ENGINE v1.0 -- Consolidated Professional-Grade")
    print("=" * 95)
    print(f"  Generated: {slip.generated_at}")
    print(f"  Bankroll:  {slip.bankroll:.0f}")
    print(f"  Value bets scanned: {len(all_value_bets)} | Selected: {len(slip.bets)}")

    # Track record
    history = load_bet_history()
    record = get_track_record(history)
    if record.get("n_settled", 0) > 0:
        print(f"\n  TRACK RECORD: {record['wins']}W/{record['losses']}L "
              f"({record['win_rate']:.0%} win rate) | "
              f"ROI: {record['roi']:+.1f}% | P/L: {record['total_profit']:+.1f}")
    if record.get("n_pending", 0) > 0:
        print(f"     {record['n_pending']} pending bets awaiting results")

    print("=" * 95)

    if not slip.bets:
        print("\n  NO VALUE BETS FOUND -- Market is efficient today.")
        print("  Recommendation: SKIP this round. Preserve bankroll.")
        return

    # Group by date
    by_date = {}
    for b in slip.bets:
        d = b.date or "Unknown"
        by_date.setdefault(d, []).append(b)

    num = 0
    for date in sorted(by_date.keys()):
        bets_on_day = by_date[date]
        day_stake = sum(b.stake_amount for b in bets_on_day)
        day_ev = sum(b.expected_profit for b in bets_on_day)

        print(f"\n  {'=' * 93}")
        print(f"  |  {date}  --  {len(bets_on_day)} bet(s), "
              f"{day_stake:.0f} staked, {day_ev:+.0f} expected")
        print(f"  {'=' * 93}")

        for bet in sorted(bets_on_day, key=lambda b: b.ev_per_unit, reverse=True):
            num += 1
            score = engine.compute_confidence_score(bet)
            potential = bet.stake_amount * (bet.best_odds - 1)

            tier_icon = {"ELITE": "[E]", "STRONG": "[S]", "STANDARD": "[.]"}.get(
                bet.confidence_tier, "[.]")

            print(f"\n  {tier_icon} BET #{num}: {bet.match}")
            print(f"     {bet.market} -> {bet.selection}")
            print(f"     {'=' * 63}")
            print(f"     Model: {bet.model_prob:.1%}  vs  Market: {bet.sharp_implied_prob:.1%}"
                  f"  ->  Edge: {bet.raw_edge:+.1%}")
            print(f"     Odds:  {bet.best_odds:.2f} @ {bet.best_bookmaker}"
                  f"  (Pinnacle: {bet.pinnacle_odds:.2f}, {bet.odds_count} books)")
            print(f"     Stake: {bet.stake_amount:.2f} ({bet.stake_pct:.1f}%)"
                  f"  |  Kelly: {bet.kelly_adj:.2%}")
            print(f"     EV: {bet.ev_per_unit:+.2%}/unit  |  "
                  f"Win: +{potential:.0f}  |  Lose: -{bet.stake_amount:.0f}")
            print(f"     Confidence: {score:.0f}/100 [{bet.confidence_tier}]")

            intel = getattr(bet, '_intel_notes', [])
            penalty = getattr(bet, '_intel_penalty', 1.0)
            if intel:
                notes_str = " ".join(intel)
                adj_str = f" (stake x{penalty})" if penalty != 1.0 else ""
                print(f"     Intel: {notes_str}{adj_str}")

            # Entry timing recommendation
            timing = getattr(bet, 'entry_timing', {})
            if timing and timing.get("action"):
                action = timing["action"]
                emoji = {"ENTER_NOW": "\u2705", "WAIT": "\u26a0\ufe0f", "AVOID": "\u274c"}.get(action, "")
                conf = timing.get("confidence", 0)
                print(f"     Timing: {emoji} {action} ({conf:.0%}) — {timing.get('reason', '')[:80]}")

    # Accumulators
    accumulators = getattr(slip, '_accumulators', [])
    if accumulators:
        print(f"\n  {'=' * 93}")
        print(f"  |  ACCUMULATORS -- {len(accumulators)} parlay(s)")
        print(f"  {'=' * 93}")

        for idx, acc in enumerate(accumulators, 1):
            leg_type = {2: "DOUBLE", 3: "TREBLE"}.get(acc.n_legs, f"{acc.n_legs}-FOLD")
            print(f"\n  ACCA #{idx}: {leg_type} @ {acc.combined_odds:.2f}")
            for li, leg in enumerate(acc.legs, 1):
                print(f"     Leg {li}: {leg.match} -> {leg.market} "
                      f"{leg.selection} @ {leg.best_odds:.2f}")
            print(f"     {'=' * 63}")
            print(f"     Combined prob: {acc.combined_prob:.1%}"
                  f"  |  EV: {acc.ev_per_unit:+.2%}/unit")
            print(f"     Stake: {acc.stake_amount:.2f}"
                  f"  |  Win: +{acc.potential_profit:.0f}"
                  f"  |  Lose: -{acc.stake_amount:.0f}")

    # Portfolio Summary
    n = len(slip.bets)
    print(f"\n{'=' * 95}")
    print(f"  PORTFOLIO SUMMARY")
    print(f"  {'=' * 91}")
    print(f"  Bets: {n} across {slip.n_matches} matches")
    print(f"  Stake: {slip.total_stake:.0f} ({slip.exposure_pct:.1f}% of bankroll)")
    print(f"  Expected P/L: {slip.total_ev:+.0f} "
          f"(ROI: {slip.total_ev / max(slip.total_stake, 1) * 100:+.1f}%)")

    max_loss = slip.total_stake
    best_case = sum(b.stake_amount * (b.best_odds - 1) for b in slip.bets)
    print(f"  Risk: -{max_loss:.0f} (worst) to +{best_case:.0f} (best)")

    mkt_mix = {}
    for b in slip.bets:
        mkt_mix[b.market] = mkt_mix.get(b.market, 0) + 1
    mix_str = ", ".join(f"{k}: {v}" for k, v in sorted(mkt_mix.items()))
    print(f"  Markets: {mix_str}")

    avg_score = sum(engine.compute_confidence_score(b) for b in slip.bets) / n
    print(f"  Avg Confidence: {avg_score:.0f}/100")
    print(f"  Avg Odds: {sum(b.best_odds for b in slip.bets) / n:.2f}")

    print(f"\n  {'=' * 91}")
    print("  RULES:")
    print("  1. Bet ONLY at the listed bookmaker and odds (or better)")
    print("  2. If odds drop below Pinnacle -> skip the bet (edge gone)")
    print("  3. Never exceed the stake shown -- discipline = profit")
    print("  4. Record results with: python betting_unified.py --update-results")
    print(f"  {'=' * 91}")


def print_monte_carlo(mc: Dict):
    """Print Monte Carlo simulation results."""
    if not mc:
        return
    print(f"\n  MONTE CARLO SIMULATION ({mc['n_sims']:,} runs)")
    print(f"  {'=' * 60}")
    print(f"  Probability of profit:  {mc['profit_probability_pct']:.1f}%")
    print(f"  Average P/L:            {mc['avg_pnl']:+.0f}")
    print(f"  +-------------------------------------------------")
    print(f"  |  5th percentile:   {mc['p5']:+.0f}  (worst realistic)")
    print(f"  |  25th percentile:  {mc['p25']:+.0f}")
    print(f"  |  Median:           {mc['median']:+.0f}")
    print(f"  |  75th percentile:  {mc['p75']:+.0f}")
    print(f"  |  95th percentile:  {mc['p95']:+.0f}  (best realistic)")
    print(f"  +-------------------------------------------------")
    print(f"  Absolute worst:  {mc['worst']:+.0f}  |  Absolute best: {mc['best']:+.0f}")


# =============================================================================
# 9. JSON OUTPUT / SAVE
# =============================================================================
def save_bet_slip(slip: BetSlip, all_value: List[ValueBet],
                  accumulators: List[AccumulatorBet] = None,
                  dry_run: bool = False,
                  near_misses: List[Dict] = None,
                  best_picks: List[Dict] = None) -> Optional[Path]:
    """Save bet slip to JSON for tracking."""
    output = {
        "generated_at": slip.generated_at,
        "bankroll": slip.bankroll,
        "summary": {
            "total_bets": len(slip.bets),
            "total_stake": slip.total_stake,
            "expected_profit": slip.total_ev,
            "expected_roi_pct": round(
                slip.total_ev / max(slip.total_stake, 1) * 100, 2),
            "exposure_pct": slip.exposure_pct,
            "matches_covered": slip.n_matches,
        },
        "selected_bets": [asdict(b) for b in slip.bets],
        "all_value_bets_found": len(all_value),
        "rejected_bets": [asdict(b) for b in all_value if not b.is_selected],
        # closest post-edge rejections — why "0 bets" is 0 bets
        "near_misses": list(near_misses or []),
        # advisory best angle per match (all markets, edge-ranked, tiered) —
        # informational, never journaled
        "best_picks": list(best_picks or []),
    }

    if accumulators:
        output["accumulators"] = []
        for acc in accumulators:
            output["accumulators"].append({
                "n_legs": acc.n_legs,
                "combined_odds": acc.combined_odds,
                "combined_prob": acc.combined_prob,
                "stake_amount": acc.stake_amount,
                "expected_profit": acc.expected_profit,
                "ev_per_unit": acc.ev_per_unit,
                "potential_profit": acc.potential_profit,
                "category": acc.category,
                "legs": [asdict(leg) for leg in acc.legs],
            })

    if dry_run:
        log.info("DRY RUN: would save bet slip to %s",
                 UPCOMING / "unified_bet_slip.json")
        return None

    from config.settings import atomic_write_json
    path = UPCOMING / "unified_bet_slip.json"
    atomic_write_json(path, output)
    log.info("Saved bet slip to %s", path)

    # Auto-record placement odds for CLV tracking
    try:
        from scripts.betting.clv_tracker import record_bet_placement
        n_recorded = record_bet_placement(path)
        if n_recorded:
            log.info("CLV: recorded %d bet placements", n_recorded)
    except Exception as e:
        log.debug(f"Failed to record bet placement for CLV tracking: {e}")

    # Also write to unified journal
    try:
        from scripts.betting.bet_journal import add_bet as journal_add_bet
        for bet in slip.bets:
            journal_add_bet({
                "match": bet.match,
                "date": bet.date,
                "market": bet.market,
                "selection": bet.selection,
                "model_prob": bet.model_prob,
                "sharp_implied_prob": bet.sharp_implied_prob,
                "edge_pct": bet.edge_pct,
                "odds": bet.best_odds,
                "bookmaker": bet.best_bookmaker,
                "avg_odds": bet.avg_odds,
                "pinnacle_odds": bet.pinnacle_odds,
                "stake": bet.stake_amount,
                "confidence": bet.confidence_tier,
                "placed_at": slip.generated_at,
                "league": getattr(bet, 'league', 'serie_a'),
            })
        log.info("Journal: recorded %d bets", len(slip.bets))
    except Exception as e:
        log.debug("Failed to write to bet journal: %s", e)

    # Paper track: gated leagues journal their candidates on the same
    # commit timing (flat stakes, separate journal, zero real exposure).
    try:
        n_paper = run_paper_track()
        if n_paper:
            log.info("Paper track: %d bet(s) journaled", n_paper)
    except Exception as e:
        log.warning("Paper track failed: %s", e)

    return path


# =============================================================================
# BACKWARD-COMPATIBLE WRAPPERS
# (for callers that used advanced_betting_engine / betting_engine)
# =============================================================================

def generate_unified_report(bankroll: float = None) -> Dict:
    """Backward-compatible wrapper matching advanced_betting_engine.generate_unified_report().

    Creates a UnifiedBettingEngine, runs the full scan, and returns a report
    dict with 'summary', 'bets', and 'accumulators' keys.

    If bankroll is None or 0, auto-loads from bet journal (journal-derived balance).
    """
    cfg = BettingConfig.from_config(bankroll_override=bankroll)
    engine = UnifiedBettingEngine(cfg)
    slip, all_bets = engine.run()
    bets_data = []
    for b in (slip.bets if slip else []):
        bets_data.append({
            "match": b.match,
            "date": b.date,
            "market": b.market,
            "selection": b.selection,
            "model_prob": b.model_prob,
            "best_odds": b.best_odds,
            "best_bookmaker": b.best_bookmaker,
            "edge_pct": b.edge_pct,
            "ev_per_unit": b.ev_per_unit,
            "stake_amount": b.stake_amount,
            "confidence": b.confidence_tier,
        })
    accs_data = []
    for a in engine.accumulators:
        accs_data.append({
            "legs": [{"match": l.match, "market": l.market, "selection": l.selection,
                       "odds": l.best_odds} for l in a.legs],
            "combined_odds": a.combined_odds,
            "stake": a.stake_amount,
        })
    total_stake = sum(b.get("stake_amount", 0) for b in bets_data)
    total_potential = sum(
        b.get("stake_amount", 0) * (b.get("best_odds", 1) - 1)
        for b in bets_data
    )
    return {
        "summary": {
            "total_bets": len(bets_data),
            "recommended_bets": len(bets_data),
            "total_accumulators": len(accs_data),
            "bankroll": cfg.bankroll,
            "total_stake": round(total_stake, 2),
            "total_potential_profit": round(total_potential, 2),
        },
        "bets": bets_data,
        "accumulators": accs_data,
        "_slip": slip,
        "_engine": engine,
    }


def save_report(report: Dict):
    """Backward-compatible wrapper matching advanced_betting_engine.save_report()."""
    slip = report.get("_slip")
    engine = report.get("_engine")
    if slip and engine:
        save_bet_slip(slip, engine.all_bets, near_misses=engine.top_near_misses(40),
                      best_picks=getattr(engine, 'best_picks', []))
    else:
        # No slip/engine attached: nothing to journal. The old fallback wrote
        # data/betting/unified_report.json — a legacy file that froze in
        # February and was still read as "current bets" by a dozen consumers
        # (removed 2026-08-31). Never resurrect it.
        log.warning("save_report called without a slip — nothing written")


def generate_betting_slip(include_parlays: bool = True) -> Dict:
    """Backward-compatible wrapper matching betting_engine.generate_betting_slip()."""
    cfg = BettingConfig()
    engine = UnifiedBettingEngine(cfg)
    slip, all_bets = engine.run()
    if not slip or not slip.bets:
        return {}
    bets_data = []
    for b in slip.bets:
        bets_data.append({
            "match": b.match,
            "date": b.date,
            "market": b.market,
            "selection": b.selection,
            "model_prob": b.model_prob,
            "best_odds": b.best_odds,
            "best_bookmaker": b.best_bookmaker,
            "edge_pct": b.edge_pct,
            "ev_per_unit": b.ev_per_unit,
            "stake_amount": b.stake_amount,
            "confidence": b.confidence_tier,
        })
    total_stake = sum(b.stake_amount for b in slip.bets)
    result = {
        "summary": {
            "recommended_bets": len(bets_data),
            "total_bets": len(bets_data),
            "total_stake": round(total_stake, 2),
        },
        "bets": bets_data,
        "total_bets": len(bets_data),
        "total_stake": round(total_stake, 2),
    }
    if include_parlays and engine.accumulators:
        result["parlays"] = [{
            "legs": [{"match": l.match, "selection": l.selection,
                       "market": l.market, "model_prob": round(l.model_prob, 3),
                       "odds": l.best_odds} for l in a.legs],
            "combined_odds": a.combined_odds,
            "combined_prob": round(a.combined_prob, 4),
            "stake": a.stake_amount,
            "category": a.category,
            "ev_per_unit": a.ev_per_unit,
        } for a in engine.accumulators]
    return result


# =============================================================================
# 10. CLI ENTRY POINT
# =============================================================================
def main():
    """Main entry point with full CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified Betting Engine v1.0 -- Consolidated Professional-Grade",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/betting_unified.py                    # Full run
  python scripts/betting_unified.py --bankroll 500     # Custom bankroll
  python scripts/betting_unified.py --min-edge 4.0     # Higher threshold
  python scripts/betting_unified.py --kelly 0.15       # More conservative
  python scripts/betting_unified.py --update-results   # Mark bet results
  python scripts/betting_unified.py --track-record     # Show P/L history
  python scripts/betting_unified.py --json             # JSON output only
  python scripts/betting_unified.py --dry-run          # No file writes
        """)

    parser.add_argument("--bankroll", type=float, default=0,
                        help="Bankroll amount (default: 0 = auto-load from journal)")
    parser.add_argument("--min-edge", type=float, default=2.0,
                        help="Minimum edge %% (default: 2.0)")
    parser.add_argument("--max-edge", type=float, default=12.0,
                        help="Maximum edge %% (default: 12.0)")
    parser.add_argument("--kelly", type=float, default=0.10,
                        help="Kelly fraction (default: 0.10)")
    parser.add_argument("--max-stake", type=float, default=2.5,
                        help="Max stake %% of bankroll (default: 2.5)")
    parser.add_argument("--staking", choices=["kelly", "proportional"], default="kelly",
                        help="Staking mode: kelly (edge-scaled, recommended) or proportional (flat %%)")
    parser.add_argument("--prop-pct", type=float, default=2.0,
                        help="Proportional stake %% of bankroll (default: 2.0)")
    parser.add_argument("--max-bets", type=int, default=20,
                        help="Max total bets (default: 20)")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON only (no pretty print)")
    parser.add_argument("--dry-run", action="store_true",
                        help="No file writes")
    parser.add_argument("--update-results", action="store_true",
                        help="Enter results update mode (mark bets WIN/LOSS)")
    parser.add_argument("--track-record", action="store_true",
                        help="Show cumulative track record")
    parser.add_argument("--no-monte-carlo", action="store_true",
                        help="Skip Monte Carlo simulation")

    args = parser.parse_args()

    # -- Dispatch special modes --
    if args.update_results:
        update_results()
        return 0

    if args.track_record:
        history = load_bet_history()
        record = get_track_record(history)
        if record.get("n_settled", 0) > 0:
            print(f"\nTrack Record: {record['wins']}W/{record['losses']}L "
                  f"({record['win_rate']:.0%}) | ROI: {record['roi']:+.1f}% | "
                  f"P/L: {record['total_profit']:+.1f} | "
                  f"Staked: {record['total_staked']:.0f}")
        else:
            print(f"\nNo settled bets yet. {record.get('n_pending', 0)} pending.")
        _check_drawdown(history)
        return 0

    # -- Build config from args (auto-load bankroll if not specified) --
    cfg = BettingConfig.from_config(
        bankroll_override=args.bankroll if args.bankroll > 0 else None,
    )
    cfg.kelly_fraction = args.kelly
    cfg.staking_mode = args.staking
    cfg.proportional_stake_pct = args.prop_pct
    cfg.min_edge_pct = args.min_edge
    cfg.max_edge_pct = args.max_edge
    cfg.max_stake_pct = args.max_stake
    cfg.max_total_bets = args.max_bets
    cfg.dry_run = args.dry_run

    # -- Run engine --
    engine = UnifiedBettingEngine(cfg)
    slip, all_bets = engine.run()

    if not slip.bets:
        # Still persist the slip: near_misses and the best-pick advisory are
        # exactly what a 0-bet day needs on disk (dashboard, /picks, digest).
        if not cfg.dry_run:
            save_bet_slip(slip, all_bets,
                          near_misses=engine.top_near_misses(40),
                          best_picks=getattr(engine, 'best_picks', []))
        if args.json:
            print(json.dumps({"bets": [], "message": "No value bets found"},
                             indent=2))
        else:
            print("\nNo value bets found. Market is efficient today.")
        return 0

    # -- Monte Carlo --
    mc_results = None
    if not args.no_monte_carlo:
        mc_results = engine.monte_carlo_simulation(slip)

    # -- Output --
    if args.json:
        output = {
            "generated_at": slip.generated_at,
            "bankroll": slip.bankroll,
            "summary": {
                "total_bets": len(slip.bets),
                "total_stake": slip.total_stake,
                "expected_profit": slip.total_ev,
                "exposure_pct": slip.exposure_pct,
                "matches_covered": slip.n_matches,
            },
            "bets": [asdict(b) for b in slip.bets],
            "accumulators": [
                {
                    "n_legs": acc.n_legs,
                    "combined_odds": acc.combined_odds,
                    "combined_prob": round(acc.combined_prob, 4),
                    "stake_amount": acc.stake_amount,
                    "ev_per_unit": acc.ev_per_unit,
                    "category": acc.category,
                    "legs": [asdict(leg) for leg in acc.legs],
                }
                for acc in engine.accumulators
            ],
        }
        if mc_results:
            output["monte_carlo"] = mc_results
        print(json.dumps(output, indent=2, default=str))
    else:
        print_bet_slip(engine, slip, all_bets)
        if mc_results:
            print_monte_carlo(mc_results)

        # Value bets NOT selected (for reference)
        rejected = [b for b in all_bets if not b.is_selected]
        if rejected:
            print(f"\n  {len(rejected)} additional value bets found but not selected "
                  f"(portfolio limits):")
            for b in rejected[:5]:
                print(f"     - {b.match} | {b.market} {b.selection} | "
                      f"edge={b.edge_pct:.1f}% | EV={b.ev_per_unit:+.4f} | "
                      f"odds={b.best_odds:.2f}")
            if len(rejected) > 5:
                print(f"     ... and {len(rejected) - 5} more")

    # -- Save --
    if not cfg.dry_run:
        save_bet_slip(slip, all_bets, engine.accumulators,
                      near_misses=engine.top_near_misses(40),
                      best_picks=getattr(engine, 'best_picks', []))
        n_recorded = record_bets(slip)
        log.info("Recorded %d bets to history tracker", n_recorded)

        # CLV tracking
        try:
            from scripts.betting.clv_tracker import record_bet_placement, get_clv_summary
            from scripts.data.odds_tracker import save_snapshot
            save_snapshot()
            n_clv = record_bet_placement()
            if n_clv:
                log.info("Recorded %d bet placements for CLV tracking", n_clv)
            summary = get_clv_summary()
            if summary["total_tracked"] > 0:
                print(f"\n  CLV TRACKING ({summary['total_tracked']} previous bets)")
                print(f"     Running CLV: {summary['running_clv_pct']:+.2f}%")
                print(f"     Positive CLV rate: {summary['positive_rate']:.1%}")
        except Exception as e:
            log.debug(f"Failed to display CLV tracking summary: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
