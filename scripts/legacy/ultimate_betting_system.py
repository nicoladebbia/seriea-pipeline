#!/usr/bin/env python3
"""
██╗   ██╗██╗  ████████╗██╗███╗   ███╗ █████╗ ████████╗███████╗
██║   ██║██║  ╚══██╔══╝██║████╗ ████║██╔══██╗╚══██╔══╝██╔════╝
██║   ██║██║     ██║   ██║██╔████╔██║███████║   ██║   █████╗  
██║   ██║██║     ██║   ██║██║╚██╔╝██║██╔══██║   ██║   ██╔══╝  
╚██████╔╝███████╗██║   ██║██║ ╚═╝ ██║██║  ██║   ██║   ███████╗
 ╚═════╝ ╚══════╝╚═╝   ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝
           BETTING ENGINE v2.0 — Professional Grade

Reads all live prediction + odds files. Scans every market for value.
Applies fractional Kelly staking, correlation handling, portfolio limits.
Outputs actionable bet slips ranked by expected value.

Architecture:
  1. DATA LAYER  — Load predictions, odds (40+ bookmakers), extended markets
  2. EDGE LAYER  — Model prob vs sharp implied prob, proper overround removal
  3. ODDS LAYER  — Best odds hunting across all bookmakers per selection
  4. STAKE LAYER — Fractional Kelly (1/4) with caps and variance reduction
  5. PORTFOLIO   — Correlation handling, max exposure per match, diversification
  6. OUTPUT      — Professional bet slips with full transparency

Backtested: +13.1% ROI (2023-24), +4.8% ROI (2024-25) on draw value bets alone.
"""
import sys, json, logging, math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

from config.settings import DATA_DIR

UPCOMING = DATA_DIR / "upcoming"

# =============================================================================
# CONFIGURATION — Tunable parameters
# =============================================================================
@dataclass
class BettingConfig:
    """All configurable parameters for the betting system."""
    # Bankroll
    bankroll: float = 1000.0
    
    # Kelly fraction (1/4 Kelly = conservative, industry standard)
    kelly_fraction: float = 0.25
    
    # Edge thresholds (walk-forward calibrated for T=1.289, validated 2023-2025)
    min_edge_pct: float = 5.0       # Minimum edge (walk-forward: +0.6%/+11.2% per season)
    max_edge_pct: float = 15.0      # Max edge (T=1.289 softens probs, widens edge dist)
    max_edge_draw_pct: float = 15.0 # Draw-specific max edge (aligned with general max)
    strong_edge_pct: float = 4.0    # "Strong" value threshold (%)
    elite_edge_pct: float = 5.5     # "Elite" value threshold (%)
    
    # Stake limits (% of bankroll)
    min_stake_pct: float = 0.5      # Don't place bets smaller than this
    max_stake_pct: float = 5.0      # Maximum single bet (even if Kelly says more)
    max_match_exposure_pct: float = 8.0   # Max total exposure on one match
    max_daily_exposure_pct: float = 35.0  # Max total exposure for the day
    
    # Portfolio limits
    max_bets_per_match: int = 3     # Max correlated bets on same match
    max_total_bets: int = 20        # Max total bets per round
    
    # Odds limits
    min_odds: float = 1.25          # Don't bet below this (too low value)
    max_odds: float = 15.0          # Don't bet above this (too unlikely)
    
    # Confidence adjustments
    model_confidence_weight: float = 0.7  # How much we trust our model vs market
    
    # Sharp bookmaker for true probability benchmark
    sharp_bookmaker: str = "Pinnacle"


# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class ValueBet:
    """A single identified value bet with all metadata."""
    match: str
    date: str
    market: str                  # "1X2", "O/U 2.5", "AH -1", "BTTS", "DC", "Exact"
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
    ev_per_unit: float = 0.0     # Expected value per €1 bet
    expected_profit: float = 0.0 # Expected profit on this bet
    
    # Confidence
    confidence_tier: str = ""    # "ELITE", "STRONG", "STANDARD"
    model_confidence: float = 0.0  # Original model confidence
    
    # Portfolio
    match_group: str = ""        # For correlation tracking
    is_selected: bool = False    # Final portfolio selection


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
    """A multi-leg accumulator (parlay) combining independent value bets."""
    legs: List[ValueBet]
    combined_odds: float
    combined_prob: float       # product of individual model probs
    stake_amount: float
    expected_profit: float
    ev_per_unit: float
    n_legs: int
    date: str = ""

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
# 1. DATA LAYER — Load all live files
# =============================================================================
def load_predictions() -> List[Dict]:
    """Load ensemble predictions."""
    p = UPCOMING / "predictions.json"
    if not p.exists():
        log.error("No predictions.json found")
        return []
    with open(p) as f:
        data = json.load(f)
    return data.get("predictions", [])


def load_odds_full() -> Dict:
    """Load full odds with all bookmakers."""
    p = UPCOMING / "odds_full.json"
    if not p.exists():
        log.error("No odds_full.json found")
        return {}
    with open(p) as f:
        data = json.load(f)
    return data.get("matches", {})


def load_extended_markets() -> Dict:
    """Load extended market predictions (DC, exact scores, team totals, 1H)."""
    p = UPCOMING / "extended_markets.json"
    if not p.exists():
        return {}
    with open(p) as f:
        data = json.load(f)
    return data.get("matches", {})


def load_goal_predictions() -> List[Dict]:
    """Load Poisson goal predictions."""
    p = UPCOMING / "goal_predictions.json"
    if not p.exists():
        return []
    with open(p) as f:
        data = json.load(f)
    return data.get("predictions", [])


def load_simple_odds() -> Dict:
    """Load simple 1X2 odds."""
    p = UPCOMING / "odds.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def load_btts_predictions() -> List[Dict]:
    """Load BTTS model predictions."""
    p = UPCOMING / "btts_predictions.json"
    if not p.exists():
        return []
    with open(p) as f:
        data = json.load(f)
    # Handle both raw list and dict-wrapped formats
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


def load_extra_market_odds() -> Dict:
    """Load extra market odds (BTTS, DC, alternate totals) with real bookmaker prices."""
    p = UPCOMING / "odds_extra_markets.json"
    if not p.exists():
        return {}
    with open(p) as f:
        data = json.load(f)
    return data.get("matches", {})


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


def load_margin_predictions() -> List[Dict]:
    """Load margin predictions with handicap probabilities at every line."""
    p = UPCOMING / "margin_predictions.json"
    if not p.exists():
        return []
    with open(p) as f:
        data = json.load(f)
    return data.get("predictions", [])


# =============================================================================
# 2. EDGE LAYER — True probability calculation + edge detection
# =============================================================================
def remove_overround(odds_list: List[float]) -> List[float]:
    """Remove overround from odds to get true implied probabilities.
    
    Uses multiplicative method (Shin's method simplified):
    true_prob[i] = raw_prob[i] / sum(raw_probs)
    """
    if not odds_list or any(o <= 1 for o in odds_list):
        return [1/len(odds_list)] * len(odds_list) if odds_list else []
    
    raw_probs = [1/o for o in odds_list]
    total = sum(raw_probs)
    if total <= 0:
        return [1/len(odds_list)] * len(odds_list)
    
    return [p / total for p in raw_probs]


def find_best_odds(bookmakers: List[Dict], selection_key: str) -> Tuple[float, str, float, int]:
    """Find best odds across all bookmakers for a given selection.
    
    Returns: (best_odds, best_bookmaker, avg_odds, count)
    """
    odds_list = []
    best = 0
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
    """Get Pinnacle odds for a selection (sharpest benchmark)."""
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
    """
    if odds <= 1 or model_prob <= 0 or model_prob >= 1:
        return 0
    
    b = odds - 1
    p = model_prob
    q = 1 - p
    
    kelly = (b * p - q) / b
    if kelly <= 0:
        return 0
    
    # Apply fraction and floor at 0.001 (0.1%)
    result = kelly * fraction
    return result if result >= 0.001 else 0


def calculate_ev(model_prob: float, odds: float) -> float:
    """Calculate expected value per unit staked.
    
    EV = p * (odds - 1) - (1 - p)
    Positive EV = profitable bet.
    """
    if odds <= 1 or model_prob <= 0:
        return -1
    return model_prob * (odds - 1) - (1 - model_prob)


# =============================================================================
# 3. MARKET SCANNER — Scan all markets for value
# =============================================================================
def _make_bet(match, date, market, selection, model_p, sharp_p,
              best_o, best_bk, avg_o, pin_o, count, cfg, confidence=0,
              max_edge_override=None) -> Optional[ValueBet]:
    """Shared bet construction with mandatory +EV gate.
    Returns None if the bet is not +EV at best available odds."""
    raw_edge = model_p - sharp_p
    edge_pct = (raw_edge / sharp_p * 100) if sharp_p > 0 else 0
    
    if edge_pct < cfg.min_edge_pct:
        return None
    # Market-specific edge cap (backtest-calibrated)
    effective_max = max_edge_override if max_edge_override is not None else cfg.max_edge_pct
    if edge_pct > effective_max:
        return None  # Backtest-proven: edges above cap are model overconfidence
    if raw_edge < 0.02:
        return None  # Minimum 2% absolute edge (prevents noise on low-prob events)
    if best_o < cfg.min_odds or best_o > cfg.max_odds:
        return None
    
    ev = calculate_ev(model_p, best_o)
    if ev <= 0:
        return None  # HARD GATE: never recommend a -EV bet
    
    kelly_raw = calculate_kelly(model_p, best_o, fraction=1.0)
    kelly_adj = calculate_kelly(model_p, best_o, fraction=cfg.kelly_fraction)
    stake_pct = min(kelly_adj * 100, cfg.max_stake_pct)
    if stake_pct < 0.15:
        return None  # Edge too small — Kelly says < 0.15% of bankroll
    if stake_pct < cfg.min_stake_pct:
        stake_pct = cfg.min_stake_pct  # Floor at minimum (bet is still +EV)
    
    tier = "ELITE" if edge_pct >= cfg.elite_edge_pct else \
           "STRONG" if edge_pct >= cfg.strong_edge_pct else "STANDARD"
    
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
        match_group=match,
    )


def scan_1x2_market(predictions: List[Dict], odds_full: Dict, cfg: BettingConfig) -> List[ValueBet]:
    """Scan 1X2 market for value bets across all matches."""
    bets = []
    
    for pred in predictions:
        match = pred["match"]
        date = pred.get("date", "")
        probs = pred.get("probabilities", {})
        
        if not probs or match not in odds_full:
            continue
        
        odds_data = odds_full[match]
        h2h = odds_data.get("h2h", {})
        if not h2h:
            continue
        
        bookmakers = h2h.get("all_bookmakers", [])
        if not bookmakers:
            continue
        
        # Get Pinnacle odds for sharp benchmark
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
        
        # BACKTEST-PROVEN: Only Draw bets are profitable in 1X2.
        # H/A bets lose money at every edge level (model 2x overconfident).
        # Draw bets: +2.8% ROI over 360 bets in 2024-25 backtest.
        model_d = probs.get("draw", 0)
        if model_d <= 0:
            continue
        
        best_o, best_bk, avg_o, count = find_best_odds(bookmakers, "draw")
        
        bet = _make_bet(match, date, "1X2", "Draw", model_d, true_probs[1],
                       best_o, best_bk, avg_o, pin_d, count, cfg,
                       confidence=pred.get("confidence", 0),
                       max_edge_override=cfg.max_edge_draw_pct)
        if bet:
            bets.append(bet)
    
    return bets


def scan_ou_market(goal_preds: List[Dict], odds_full: Dict, cfg: BettingConfig) -> List[ValueBet]:
    """Scan Over/Under markets for value."""
    bets = []
    
    for gp in goal_preds:
        match = gp["match"]
        date = gp.get("date", "")
        
        if match not in odds_full:
            continue
        
        totals = odds_full[match].get("totals", [])
        if not totals:
            continue
        
        for total in totals:
            line = total.get("line", 0)
            bookmakers = total.get("all_bookmakers", [])
            
            if not bookmakers:
                continue
            
            # Map line to our probability keys
            prob_key_over = f"over_{str(line).replace('.', '_')}"
            model_over = gp.get(prob_key_over, 0)
            
            # Common lines
            if line == 2.5:
                model_over = gp.get("over_2_5", 0)
            elif line == 1.5:
                model_over = gp.get("over_1_5", 0)
            elif line == 3.5:
                model_over = gp.get("over_3_5", 0)
            elif line == 0.5:
                model_over = gp.get("over_0_5", 0)
            elif line == 4.5:
                model_over = gp.get("over_4_5", 0)
            
            if model_over <= 0:
                continue
            
            model_under = 1 - model_over
            
            # Get Pinnacle benchmark
            pin_over = get_pinnacle_odds(bookmakers, "over")
            pin_under = get_pinnacle_odds(bookmakers, "under")
            
            if not pin_over or not pin_under or pin_over <= 1 or pin_under <= 1:
                pin_over = total.get("over", 0)
                pin_under = total.get("under", 0)
            
            if pin_over <= 1 or pin_under <= 1:
                continue
            
            true_probs = remove_overround([pin_over, pin_under])
            
            for side_idx, (sel_name, sel_key, model_p) in enumerate([
                (f"Over {line}", "over", model_over),
                (f"Under {line}", "under", model_under),
            ]):
                best_o, best_bk, avg_o, count = find_best_odds(bookmakers, sel_key)
                pin_o = [pin_over, pin_under][side_idx]
                
                bet = _make_bet(match, date, f"O/U {line}", sel_name, model_p,
                               true_probs[side_idx], best_o, best_bk, avg_o, pin_o,
                               count, cfg)
                if bet:
                    bets.append(bet)
    
    return bets


def scan_ah_market(predictions: List[Dict], odds_full: Dict, 
                   extended: Dict, cfg: BettingConfig,
                   margin_preds: List[Dict] = None) -> List[ValueBet]:
    """Scan Asian Handicap / Spread market for value.
    
    Uses ML-calibrated handicap_probs from margin_predictions.json when available.
    Falls back to norm.cdf(xG_diff) only when ML probs are missing.
    """
    bets = []
    margin_preds = margin_preds or []
    
    # Index margin predictions by match name
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
        
        # Get ML margin predictions for this match
        ml_margin = margin_by_match.get(match, {})
        handicap_probs = ml_margin.get("handicap_probs", {})
        
        for spread in spreads:
            line = spread.get("line", 0)
            home_hc = spread.get("home_handicap", 0)
            away_hc = spread.get("away_handicap", 0)
            bookmakers = spread.get("all_bookmakers", [])
            
            if not bookmakers:
                continue
            
            # Try ML handicap probs first (much better calibrated)
            line_key = str(home_hc) if home_hc != 0 else "0"
            # Also try without sign for positive values
            ml_probs = handicap_probs.get(line_key, {})
            if not ml_probs:
                # Try alternate key formats
                for fmt in [f"{home_hc:.1f}", f"{home_hc:+.1f}", f"{home_hc:.0f}"]:
                    ml_probs = handicap_probs.get(fmt, {})
                    if ml_probs:
                        break
            
            if ml_probs and "home" in ml_probs and "away" in ml_probs:
                home_ah_prob = ml_probs["home"]
                away_ah_prob = ml_probs["away"]
            else:
                # Fallback: norm.cdf with xG diff
                from scipy.stats import norm
                xg_diff = (home_xg or 1.3) - (away_xg or 1.1)
                adjusted_diff = xg_diff + home_hc
                home_ah_prob = norm.cdf(adjusted_diff / 1.3)
                away_ah_prob = 1 - home_ah_prob
            
            # Get Pinnacle benchmark
            pin_home = get_pinnacle_odds(bookmakers, "home")
            pin_away = get_pinnacle_odds(bookmakers, "away")
            
            if not pin_home or not pin_away or pin_home <= 1 or pin_away <= 1:
                pin_home = spread.get("home_odds", 0)
                pin_away = spread.get("away_odds", 0)
            
            if pin_home <= 1 or pin_away <= 1:
                continue
            
            true_probs = remove_overround([pin_home, pin_away])
            
            for side_idx, (sel_name, sel_key, model_p) in enumerate([
                (f"Home {home_hc:+.1f}", "home", home_ah_prob),
                (f"Away {away_hc:+.1f}", "away", away_ah_prob),
            ]):
                best_o, best_bk, avg_o, count = find_best_odds(bookmakers, sel_key)
                pin_o = [pin_home, pin_away][side_idx]
                
                bet = _make_bet(match, date, f"AH {line}", sel_name, model_p,
                               true_probs[side_idx], best_o, best_bk, avg_o, pin_o,
                               count, cfg)
                if bet:
                    bets.append(bet)
    
    return bets


def scan_dc_market(predictions: List[Dict], extended: Dict, 
                   odds_full: Dict, cfg: BettingConfig,
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
        probs = pred.get("probabilities", {})
        
        if not probs or match not in odds_full:
            continue
        
        h2h = odds_full[match].get("h2h", {})
        bookmakers = h2h.get("all_bookmakers", []) if h2h else []
        
        if not bookmakers:
            continue
        
        ph = probs.get("home", 0)
        pd_ = probs.get("draw", 0)
        pa = probs.get("away", 0)
        
        # Extended Poisson DC probs (more calibrated)
        ext = extended.get(match, {})
        ext_dc = ext.get("double_chance", {})
        
        # Get Pinnacle as sharp benchmark
        pin_h = get_pinnacle_odds(bookmakers, "home")
        pin_d = get_pinnacle_odds(bookmakers, "draw")
        pin_a = get_pinnacle_odds(bookmakers, "away")
        if not all([pin_h > 1, pin_d > 1, pin_a > 1]):
            continue
        true_1x2 = remove_overround([pin_h, pin_d, pin_a])
        
        # Check for REAL DC odds from extra_odds
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
            
            # Try to find real DC odds first (keys: "1X", "X2", "12")
            real_found = False
            if real_dc and ext_key in real_dc:
                dc_data = real_dc[ext_key]
                best_o = dc_data.get("best", 0)
                avg_o = dc_data.get("avg", 0)
                n_books = dc_data.get("bookmakers_count", 0)
                if best_o > 1:
                    bet = _make_bet(match, date, "DC", dc_name, model_p,
                                   sharp_p, best_o, f"DC bookmaker (real)",
                                   avg_o, avg_o, n_books, cfg,
                                   confidence=pred.get("confidence", 0))
                    if bet:
                        bets.append(bet)
                    real_found = True
            
            # Fallback: derive DC from 1X2
            if not real_found:
                dc_sharp_odds = 1 / sharp_p if sharp_p > 0 else 99
                n_books = len(bookmakers)
                bet = _make_bet(match, date, "DC", dc_name, model_p, sharp_p,
                               round(dc_sharp_odds * 1.00, 2),
                               f"DC (from {n_books} 1X2 books)",
                               round(dc_sharp_odds, 2), round(dc_sharp_odds, 2),
                               n_books, cfg,
                               confidence=pred.get("confidence", 0))
                if bet:
                    bets.append(bet)
    
    return bets


# =============================================================================
# 3b. NEW MARKET SCANNERS — BTTS, Cards, Corners
# =============================================================================
def scan_btts_market(btts_preds: List[Dict], extra_odds: Dict,
                     cfg: BettingConfig) -> List[ValueBet]:
    """Scan BTTS market using model predictions + real bookmaker odds."""
    bets = []
    
    for bp in btts_preds:
        match = bp["match"]
        date = bp.get("date", "")
        
        # Get real BTTS odds from extra markets
        extra = extra_odds.get(match, {})
        btts_odds = extra.get("btts", {})
        
        if not btts_odds:
            continue  # No real bookmaker odds → skip
        
        for sel_name, model_p, odds_key, pin_key in [
            ("BTTS Yes", bp.get("btts_yes", 0), "best_yes", "yes"),
            ("BTTS No", bp.get("btts_no", 0), "best_no", "no"),
        ]:
            if model_p <= 0:
                continue
            
            best_o = btts_odds.get(odds_key, 0)
            avg_o = btts_odds.get(pin_key, 0)
            n_books = btts_odds.get("bookmakers_count", 0)
            
            if best_o <= 1 or avg_o <= 1:
                continue
            
            # Sharp implied prob from average odds (overround removed)
            yes_avg = btts_odds.get("yes", 0)
            no_avg = btts_odds.get("no", 0)
            if yes_avg <= 1 or no_avg <= 1:
                continue
            true_probs = remove_overround([yes_avg, no_avg])
            sharp_p = true_probs[0] if sel_name == "BTTS Yes" else true_probs[1]
            
            bet = _make_bet(match, date, "BTTS", sel_name, model_p, sharp_p,
                           best_o, f"Best BTTS book", avg_o, avg_o, n_books, cfg)
            if bet:
                bets.append(bet)
    
    return bets


def scan_alt_totals_market(goal_preds: List[Dict], extra_odds: Dict,
                           extended: Dict, cfg: BettingConfig) -> List[ValueBet]:
    """Scan alternate O/U totals using REAL bookmaker odds from extra_odds.
    
    These have real odds at lines 0.5-5.5 (much more than standard totals).
    Uses ML O/U probs from extended_markets when available, else Poisson.
    """
    bets = []
    from scipy.stats import poisson
    
    # Index goal predictions by match
    gp_by_match = {gp["match"]: gp for gp in goal_preds}
    
    for match_key, extra in extra_odds.items():
        alt_totals = extra.get("alternate_totals", {})
        if not alt_totals:
            continue
        
        # Get model O/U probs
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
            
            # Skip non-standard lines (quarter lines like 1.75, 2.25)
            if line != int(line) + 0.5 and line != int(line):
                continue
            
            best_over = line_data.get("best_over", 0)
            best_under = line_data.get("best_under", 0)
            avg_over = line_data.get("over", 0)
            avg_under = line_data.get("under", 0)
            n_books = line_data.get("bookmakers_count", 0)
            
            if not all([best_over > 1, best_under > 1, avg_over > 1, avg_under > 1]):
                continue
            
            # Get model probability — prefer ML from extended_markets
            over_key = f"over_{line_str}"
            under_key = f"under_{line_str}"
            
            # Try extended markets ML probs first
            model_over = 0
            if ext_ou and over_key in ext_ou:
                model_over = ext_ou[over_key].get("prob", 0)
            
            # Fallback to goal_predictions
            if model_over <= 0:
                gp_key = f"over_{line_str.replace('.', '_')}"
                model_over = gp.get(gp_key, 0)
            
            # Fallback to Poisson
            if model_over <= 0:
                model_over = 1 - sum(poisson.pmf(k, total_xg) for k in range(int(line) + 1))
            
            model_under = 1 - model_over
            
            # Sharp benchmark from average odds
            true_probs = remove_overround([avg_over, avg_under])
            
            for side_idx, (sel_name, model_p, best_o) in enumerate([
                (f"Over {line_str}", model_over, best_over),
                (f"Under {line_str}", model_under, best_under),
            ]):
                avg_o = [avg_over, avg_under][side_idx]
                
                bet = _make_bet(match_key, date, f"O/U {line_str}", sel_name, model_p,
                               true_probs[side_idx], best_o, "Alt totals book",
                               avg_o, avg_o, n_books, cfg)
                if bet:
                    bets.append(bet)
    
    return bets


def scan_dnb_market(predictions: List[Dict], extra_odds: Dict,
                    cfg: BettingConfig) -> List[ValueBet]:
    """Scan Draw-No-Bet market using real bookmaker odds.
    
    DNB = bet on a team to win, stake returned if draw.
    Equivalent to AH 0.0 but with real dedicated odds.
    """
    bets = []
    
    pred_by_match = {p["match"]: p for p in predictions}
    
    for match_key, extra in extra_odds.items():
        dnb = extra.get("draw_no_bet", {})
        if not dnb:
            continue
        
        pred = pred_by_match.get(match_key, {})
        if not pred:
            continue
        
        probs = pred.get("probabilities", {})
        if not probs:
            continue
        
        date = pred.get("date", "")
        ph = probs.get("home", 0)
        pd_ = probs.get("draw", 0)
        pa = probs.get("away", 0)
        
        # DNB probability = P(win) / (1 - P(draw))
        # Because stake is returned on draw
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
            
            bet = _make_bet(match_key, date, "DNB", sel_name, model_p,
                           true_probs[side_idx], best_o, "DNB book",
                           avg_o, avg_o, n_books, cfg)
            if bet:
                bets.append(bet)
    
    return bets


def scan_alt_spreads_market(predictions: List[Dict], extra_odds: Dict,
                            margin_preds: List[Dict],
                            cfg: BettingConfig) -> List[ValueBet]:
    """Scan alternate Asian Handicap lines from extra_odds with real bookmaker odds.
    
    Uses ML handicap_probs from margin_predictions for model probability.
    """
    bets = []
    from scipy.stats import norm
    
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
            
            # Get real odds
            best_home = line_data.get("best_home", 0)
            best_away = line_data.get("best_away", 0)
            avg_home = line_data.get("home", 0)
            avg_away = line_data.get("away", 0)
            n_books = line_data.get("bookmakers_count", 0)
            
            # Need at least one side with valid odds
            has_home = best_home > 1 and avg_home > 1
            has_away = best_away > 1 and avg_away > 1
            if not has_home and not has_away:
                continue
            
            # Get ML handicap probs
            ml_probs = handicap_probs.get(line_str, {})
            if not ml_probs:
                for fmt in [f"{line:.1f}", f"{line:+.1f}", f"{line:.0f}"]:
                    ml_probs = handicap_probs.get(fmt, {})
                    if ml_probs:
                        break
            
            if ml_probs and "home" in ml_probs and "away" in ml_probs:
                home_ah_prob = ml_probs["home"]
                away_ah_prob = ml_probs["away"]
            else:
                xg_diff = home_xg - away_xg
                adjusted_diff = xg_diff + line  # line is from home perspective
                home_ah_prob = norm.cdf(adjusted_diff / 1.3)
                away_ah_prob = 1 - home_ah_prob
            
            # Build sharp benchmark from available odds
            if has_home and has_away:
                true_probs = remove_overround([avg_home, avg_away])
            elif has_home:
                true_probs = [1 / avg_home, 1 - 1 / avg_home]
            else:
                true_probs = [1 - 1 / avg_away, 1 / avg_away]
            
            sides = []
            if has_home:
                sides.append((f"Home {line:+.1f}", "home", home_ah_prob, best_home, avg_home, 0))
            if has_away:
                sides.append((f"Away {-line:+.1f}", "away", away_ah_prob, best_away, avg_away, 1))
            
            for sel_name, sel_key, model_p, best_o, avg_o, idx in sides:
                bet = _make_bet(match_key, date, f"AH {line}", sel_name, model_p,
                               true_probs[idx], best_o, "Alt spread book",
                               avg_o, avg_o, n_books, cfg)
                if bet:
                    bets.append(bet)
    
    return bets


def scan_cards_market(cards_preds: List[Dict], cfg: BettingConfig) -> List[ValueBet]:
    """Cards market scanner.
    
    HONEST ASSESSMENT: Our cards model is a Poisson heuristic (base_rate × referee ×
    team_discipline). Bookmakers use the same approach with better data. Without real
    bookmaker odds to compare against, we cannot identify genuine value.
    
    Returns empty list until real cards odds are available via API or manual input.
    """
    log.info("  Cards: skipped (no real bookmaker odds — heuristic model has no provable edge)")
    return []


def scan_corners_market(corners_preds: List[Dict], cfg: BettingConfig) -> List[ValueBet]:
    """Corners market scanner.
    
    HONEST ASSESSMENT: Our corners model is a Poisson heuristic (team_corners +
    factors). Bookmakers adjust per-match too. Without real bookmaker odds,
    we cannot identify genuine value.
    
    Returns empty list until real corners odds are available.
    """
    log.info("  Corners: skipped (no real bookmaker odds — heuristic model has no provable edge)")
    return []


# =============================================================================
# 3c. INTELLIGENCE LAYER — Steam moves + Sharp/Soft divergence
# =============================================================================
def apply_intelligence_filters(all_bets: List[ValueBet],
                                odds_movement: Dict,
                                bookmaker_analysis: Dict) -> List[ValueBet]:
    """Apply market intelligence to adjust bet confidence and filter bad bets.
    
    Two signals:
    1. STEAM MOVES: If sharp money moves against our bet → penalize heavily
    2. SHARP/SOFT DIVERGENCE: If sharps agree with us → boost; disagree → penalize
    
    Returns the same list with adjusted stakes and confidence.
    Bets that fail intelligence checks are removed entirely.
    """
    filtered = []
    
    for bet in all_bets:
        match = bet.match
        penalty = 1.0  # Multiplier: 1.0 = no change, 0.5 = halve stake, 0 = skip
        intel_notes = []
        
        # ── STEAM MOVE CHECK ──
        movement = odds_movement.get(match, {})
        if movement:
            is_steam = movement.get("is_steam_move", False)
            is_line_move = movement.get("is_line_move", False)
            direction = movement.get("direction", "stable")
            
            if is_steam or is_line_move:
                # Determine if the move is AGAINST our bet
                our_side = _bet_to_side(bet)
                move_against = False
                
                if our_side == "home" and "home_drifting" in direction:
                    move_against = True  # Home odds rising = money against home
                elif our_side == "away" and "away_drifting" in direction:
                    move_against = True
                elif our_side == "draw" and "away_strengthening" in direction:
                    move_against = True  # Away getting shorter often hurts draw
                
                if move_against:
                    if is_steam:
                        penalty *= 0.3  # Steam against us = severe penalty
                        intel_notes.append("STEAM_AGAINST")
                    else:
                        penalty *= 0.6  # Line move against = moderate penalty
                        intel_notes.append("LINE_MOVE_AGAINST")
                elif not move_against and is_steam:
                    penalty *= 1.15  # Steam WITH us = slight boost
                    intel_notes.append("STEAM_WITH")
        
        # ── SHARP/SOFT DIVERGENCE CHECK ──
        ba = bookmaker_analysis.get(match, {})
        if ba and ba.get("has_sharp_data"):
            divergence = ba.get("divergence", 0)
            sharp_dir = ba.get("sharp_direction", "neutral")
            our_side = _bet_to_side(bet)
            
            if divergence > 0.02:  # Significant divergence (>2%)
                if sharp_dir == our_side:
                    penalty *= 1.10  # Sharps agree with us
                    intel_notes.append("SHARP_AGREES")
                elif sharp_dir != "neutral" and sharp_dir != our_side:
                    penalty *= 0.7  # Sharps disagree
                    intel_notes.append("SHARP_DISAGREES")
            
            # Also check if sharp consensus probability is close to our model
            sharp_probs = ba.get("sharp_consensus", {})
            if sharp_probs:
                sharp_p_for_side = sharp_probs.get(f"prob_{our_side[0].upper()}", 0)
                if sharp_p_for_side > 0:
                    model_vs_sharp = bet.model_prob - sharp_p_for_side
                    if model_vs_sharp > 0.08:
                        intel_notes.append("MODEL_MUCH_HIGHER_THAN_SHARP")
                    elif model_vs_sharp < -0.05:
                        penalty *= 0.5  # Our model much lower than sharp = bad sign
                        intel_notes.append("MODEL_BELOW_SHARP")
        
        # Apply penalty
        if penalty <= 0.2:
            continue  # Skip this bet entirely
        
        if penalty != 1.0:
            bet.stake_pct = round(bet.stake_pct * penalty, 2)
            bet.stake_amount = round(bet.stake_amount * penalty, 2)
            bet.expected_profit = round(bet.expected_profit * penalty, 2)
            if bet.stake_pct < 0.3:
                continue  # Too small after penalty
        
        # Store intel notes on the bet for display
        bet._intel_notes = intel_notes  # type: ignore
        bet._intel_penalty = round(penalty, 2)  # type: ignore
        filtered.append(bet)
    
    return filtered


def _bet_to_side(bet: ValueBet) -> str:
    """Map a bet's selection to 'home', 'draw', or 'away' for movement checking."""
    sel = bet.selection.lower()
    if bet.market == "1X2":
        if "home" in sel:
            return "home"
        elif "draw" in sel:
            return "draw"
        else:
            return "away"
    elif bet.market == "DC":
        if "1x" in sel.lower():
            return "home"  # 1X leans home
        elif "x2" in sel.lower():
            return "away"  # X2 leans away
        else:
            return "home"
    elif "ah" in bet.market.lower():
        if "home" in sel:
            return "home"
        else:
            return "away"
    # For O/U, BTTS, Cards, Corners — movement check is less relevant
    return "neutral"


# =============================================================================
# 4. COMPOSITE CONFIDENCE SCORE
# =============================================================================
def compute_confidence_score(bet: ValueBet) -> float:
    """Compute a 0-100 composite confidence score.
    
    Components:
      Edge strength (0-30):   raw_edge mapped to 0-30 points
      Model confidence (0-25): ensemble confidence level
      Market liquidity (0-20): number of bookmakers offering this
      Odds quality (0-15):     best_odds vs pinnacle gap (bigger = more value)
      Kelly signal (0-10):     raw Kelly fraction strength
    """
    # Edge strength: raw_edge of 0.05 = 15pts, 0.10 = 25pts, 0.15+ = 30pts
    edge_pts = min(30, bet.raw_edge * 200)
    
    # Model confidence: map 0-1 to 0-25
    conf = bet.model_confidence if isinstance(bet.model_confidence, float) else 0
    conf_pts = conf * 25
    
    # Liquidity: 1 bookmaker = 2pts, 10 = 10pts, 30+ = 20pts
    liq_pts = min(20, bet.odds_count * 0.7)
    
    # Odds quality: how much better is best vs pinnacle
    if bet.pinnacle_odds > 1 and bet.best_odds > 1:
        odds_gap = (bet.best_odds - bet.pinnacle_odds) / bet.pinnacle_odds
        odds_pts = min(15, odds_gap * 200)
    else:
        odds_pts = 0
    
    # Kelly signal: raw Kelly of 0.05 = 5pts, 0.10 = 8pts, 0.15+ = 10pts
    kelly_pts = min(10, bet.kelly_raw * 70)
    
    return round(edge_pts + conf_pts + liq_pts + odds_pts + kelly_pts, 1)


def _market_group(market: str) -> str:
    """Map a market string to its broad group for portfolio diversification."""
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


# =============================================================================
# 5. PORTFOLIO OPTIMIZATION — Two-pass diversity + selection
# =============================================================================
def _can_add(bet, selected, match_count, match_exposure, match_selections,
             total_exposure, cfg):
    """Check if a bet can be added to the portfolio."""
    m = bet.match_group
    
    if match_count.get(m, 0) >= cfg.max_bets_per_match:
        return False
    if match_exposure.get(m, 0) + bet.stake_pct > cfg.max_match_exposure_pct:
        return False
    if total_exposure + bet.stake_pct > cfg.max_daily_exposure_pct:
        return False
    
    # Mutual exclusivity on same match
    existing = match_selections.get(m, set())
    if bet.market == "1X2":
        if any(mk == "1X2" for mk, _ in existing):
            return False
    if bet.market.startswith("O/U"):
        if any(mk == bet.market for mk, _ in existing):
            return False
    if bet.market.startswith("AH"):
        if any(mk == bet.market for mk, _ in existing):
            return False
    if bet.market == "DC":
        if any(mk == "DC" for mk, _ in existing):
            return False
    
    return True


def _accept(bet, selected, match_count, match_exposure, match_selections,
            market_count):
    """Mark bet as selected and update tracking dicts. Returns new total_exposure delta."""
    m = bet.match_group
    bet.is_selected = True
    selected.append(bet)
    match_count[m] = match_count.get(m, 0) + 1
    match_exposure[m] = match_exposure.get(m, 0) + bet.stake_pct
    sel_type = (bet.market, bet.selection.split()[0])
    match_selections.setdefault(m, set()).add(sel_type)
    market_count[bet.market] = market_count.get(bet.market, 0) + 1
    return bet.stake_pct


def optimize_portfolio(all_bets: List[ValueBet], cfg: BettingConfig) -> List[ValueBet]:
    """Two-pass portfolio construction for maximum diversification.
    
    PASS 1 — Guaranteed slots: pick the single best bet from each market type
             (1X2, O/U, AH, DC). This ensures we always diversify.
    PASS 2 — Fill remaining slots by composite score, with a hard cap of
             max 50% of total bets from any single market.
    """
    if not all_bets:
        return []
    
    # Score all bets
    for bet in all_bets:
        bet._score = compute_confidence_score(bet)  # type: ignore
    
    selected = []
    match_count = {}
    match_exposure = {}
    market_count = {}
    total_exposure = 0.0
    match_selections = {}
    
    # Categorize bets by broad market type
    market_groups = {}  # "1X2", "O/U", "AH", "DC", "BTTS", "Cards", "Corners"
    for bet in all_bets:
        key = _market_group(bet.market)
        market_groups.setdefault(key, []).append(bet)
    
    # Sort each group by score
    for key in market_groups:
        market_groups[key].sort(key=lambda b: b._score, reverse=True)  # type: ignore
    
    # ── PASS 1: Guaranteed best from each market ──
    priority_order = ["1X2", "O/U", "AH", "DC", "DNB", "BTTS", "Cards", "Corners"]
    for mtype in priority_order:
        if mtype not in market_groups:
            continue
        for bet in market_groups[mtype]:
            if _can_add(bet, selected, match_count, match_exposure,
                        match_selections, total_exposure, cfg):
                delta = _accept(bet, selected, match_count, match_exposure,
                                match_selections, market_count)
                total_exposure += delta
                break  # One per market in pass 1
    
    # ── PASS 2: Fill remaining by composite score ──
    max_per_market = max(3, int(cfg.max_total_bets * 0.50))  # Hard cap 50%
    
    all_sorted = sorted(all_bets, key=lambda b: b._score, reverse=True)  # type: ignore
    for bet in all_sorted:
        if len(selected) >= cfg.max_total_bets:
            break
        if bet.is_selected:
            continue
        
        # Market cap
        mtype = _market_group(bet.market)
        current_count = sum(1 for s in selected if _market_group(s.market) == mtype)
        if current_count >= max_per_market:
            continue
        
        if _can_add(bet, selected, match_count, match_exposure,
                    match_selections, total_exposure, cfg):
            delta = _accept(bet, selected, match_count, match_exposure,
                            match_selections, market_count)
            total_exposure += delta
    
    return selected


# =============================================================================
# 5a. ACCUMULATOR GENERATION — Cross-match parlays
# =============================================================================
def generate_accumulators(selected_singles: List[ValueBet],
                          cfg: BettingConfig,
                          max_legs: int = 3,
                          min_legs: int = 2,
                          min_qualifying: int = 3) -> List[AccumulatorBet]:
    """Generate accumulator bets from independent singles on different matches.

    Rules (backtest-calibrated v2):
    - Only combine bets from DIFFERENT matches (no correlated legs)
    - Only DC bets (most consistent edge across both seasons)
    - Each leg must have edge >= 2.5% (above noise threshold)
    - Minimum 3 qualifying singles needed (strong matchday signal)
    - Stake = half of the smallest single leg stake (conservative)
    - Max 1 accumulator per matchday (avoid over-exposure)
    - Combined EV must exceed 0.03 (meaningful +EV)
    """
    if not selected_singles:
        return []

    # Filter to DC bets with edge >= 2.5% (above noise, proven consistent)
    eligible = [b for b in selected_singles
                if b.market == "DC" and b.edge_pct >= 2.5]

    # Group by match to ensure independence
    by_match = {}
    for b in eligible:
        if b.match not in by_match:
            by_match[b.match] = b
        elif b.ev_per_unit > by_match[b.match].ev_per_unit:
            by_match[b.match] = b  # Keep best EV per match

    independent = sorted(by_match.values(), key=lambda b: b.ev_per_unit, reverse=True)

    if len(independent) < min_qualifying:
        return []

    accumulators = []
    used_matches = set()

    # Generate doubles from top EV pairs
    for i in range(len(independent)):
        if len(accumulators) >= 1:  # Max 1 accumulator per day
            break
        for j in range(i + 1, len(independent)):
            if len(accumulators) >= 1:
                break
            b1, b2 = independent[i], independent[j]

            # Skip if either match already used in an accumulator
            if b1.match in used_matches or b2.match in used_matches:
                continue

            legs = [b1, b2]
            combo_odds = b1.best_odds * b2.best_odds
            combo_prob = b1.model_prob * b2.model_prob

            # EV check: combined must be meaningfully +EV
            ev = combo_prob * (combo_odds - 1) - (1 - combo_prob)
            if ev <= 0.03:
                continue

            # Stake: half of smallest leg stake (conservative)
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
            )
            accumulators.append(acc)
            used_matches.add(b1.match)
            used_matches.add(b2.match)

    # Try one triple if we have 6+ independent bets and room
    if len(independent) >= 6 and len(accumulators) < 1:
        remaining = [b for b in independent if b.match not in used_matches]
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
                stake = round(min(l.stake_amount for l in legs) * 0.3, 2)
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
                )
                accumulators.append(acc)

    return accumulators


# =============================================================================
# 5. OUTPUT — Professional bet slips
# =============================================================================
def generate_bet_slip(selected: List[ValueBet], cfg: BettingConfig) -> BetSlip:
    """Generate final bet slip with all details."""
    slip = BetSlip(
        bets=selected,
        total_stake=round(sum(b.stake_amount for b in selected), 2),
        total_ev=round(sum(b.expected_profit for b in selected), 2),
        exposure_pct=round(sum(b.stake_pct for b in selected), 2),
        n_matches=len(set(b.match for b in selected)),
        generated_at=datetime.now().isoformat(),
        bankroll=cfg.bankroll,
    )
    return slip


# =============================================================================
# BET HISTORY TRACKER
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
    path = UPCOMING / "bet_history.json"
    with open(path, "w") as f:
        json.dump(history, f, indent=2, default=str)


def record_bets(slip: BetSlip):
    """Record current bet slip to history for tracking."""
    history = load_bet_history()
    for bet in slip.bets:
        entry = {
            "id": f"{bet.date}_{bet.match}_{bet.market}_{bet.selection}".replace(" ", "_"),
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
        existing_ids = {h["id"] for h in history}
        if entry["id"] not in existing_ids:
            history.append(entry)
    save_bet_history(history)
    return len(slip.bets)


def get_track_record(history: List[Dict]) -> Dict:
    """Compute cumulative track record from history."""
    settled = [h for h in history if h.get("result") is not None]
    if not settled:
        return {"n_settled": 0, "n_pending": len(history)}
    wins = sum(1 for h in settled if h["result"] == "WIN")
    losses = sum(1 for h in settled if h["result"] == "LOSS")
    total_staked = sum(h.get("stake", 0) for h in settled)
    total_profit = sum(h.get("profit", 0) for h in settled if h.get("profit") is not None)
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


def print_bet_slip(slip: BetSlip, all_value_bets: List[ValueBet]):
    """Print beautiful, actionable bet slip to console."""
    
    print("\n" + "=" * 95)
    print("  ULTIMATE BETTING ENGINE v3.3 — Backtest-Calibrated")
    print("=" * 95)
    print(f"  Generated: {slip.generated_at}")
    print(f"  Bankroll:  €{slip.bankroll:.0f}")
    print(f"  Value bets scanned: {len(all_value_bets)} | Selected: {len(slip.bets)}")
    
    # Track record
    history = load_bet_history()
    record = get_track_record(history)
    if record.get("n_settled", 0) > 0:
        print(f"\n  📈 TRACK RECORD: {record['wins']}W/{record['losses']}L "
              f"({record['win_rate']:.0%} win rate) | "
              f"ROI: {record['roi']:+.1f}% | P/L: €{record['total_profit']:+.1f}")
    if record.get("n_pending", 0) > 0:
        print(f"     {record['n_pending']} pending bets awaiting results")
    
    print("=" * 95)
    
    if not slip.bets:
        print("\n  ⚠️  NO VALUE BETS FOUND — Market is efficient today.")
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
        
        print(f"\n  ┌{'=' * 93}")
        print(f"  │  📅  {date}  —  {len(bets_on_day)} bet(s), €{day_stake:.0f} staked, €{day_ev:+.0f} expected")
        print(f"  └{'=' * 93}")
        
        for bet in sorted(bets_on_day, key=lambda b: b.ev_per_unit, reverse=True):
            num += 1
            score = compute_confidence_score(bet)
            potential = bet.stake_amount * (bet.best_odds - 1)
            
            # Tier indicator
            tier_icon = {"ELITE": "🔥", "STRONG": "💪", "STANDARD": "📊"}.get(bet.confidence_tier, "•")
            
            print(f"\n  {tier_icon} BET #{num}: {bet.match}")
            print(f"     {bet.market} → {bet.selection}")
            print(f"     ────────────────────────────────────────────────────────────")
            print(f"     Model: {bet.model_prob:.1%}  vs  Market: {bet.sharp_implied_prob:.1%}  →  Edge: {bet.raw_edge:+.1%}")
            print(f"     Odds:  {bet.best_odds:.2f} @ {bet.best_bookmaker}  (Pinnacle: {bet.pinnacle_odds:.2f}, {bet.odds_count} books)")
            print(f"     Stake: €{bet.stake_amount:.2f} ({bet.stake_pct:.1f}%)  |  Kelly: {bet.kelly_adj:.2%}")
            print(f"     EV: {bet.ev_per_unit:+.2%}/unit  |  Win: +€{potential:.0f}  |  Lose: -€{bet.stake_amount:.0f}")
            print(f"     Confidence: {score:.0f}/100 [{bet.confidence_tier}]")
            # Intelligence notes
            intel = getattr(bet, '_intel_notes', [])
            penalty = getattr(bet, '_intel_penalty', 1.0)
            if intel:
                icons = {"STEAM_WITH": "🟢", "SHARP_AGREES": "🟢",
                         "STEAM_AGAINST": "🔴", "LINE_MOVE_AGAINST": "🟡",
                         "SHARP_DISAGREES": "🟡", "MODEL_BELOW_SHARP": "🔴",
                         "MODEL_MUCH_HIGHER_THAN_SHARP": "⚡"}
                notes_str = " ".join(f"{icons.get(n, '•')}{n}" for n in intel)
                adj_str = f" (stake ×{penalty})" if penalty != 1.0 else ""
                print(f"     Intel: {notes_str}{adj_str}")
    
    # Accumulators (if any)
    accumulators = getattr(slip, '_accumulators', [])
    if accumulators:
        print(f"\n  {'=' * 93}")
        print(f"  │  🎯  ACCUMULATORS — {len(accumulators)} parlay(s)")
        print(f"  {'=' * 93}")
        
        for idx, acc in enumerate(accumulators, 1):
            leg_type = {2: "DOUBLE", 3: "TREBLE"}.get(acc.n_legs, f"{acc.n_legs}-FOLD")
            print(f"\n  🎲 ACCA #{idx}: {leg_type} @ {acc.combined_odds:.2f}")
            for li, leg in enumerate(acc.legs, 1):
                print(f"     Leg {li}: {leg.match} → {leg.market} {leg.selection} @ {leg.best_odds:.2f}")
            print(f"     ────────────────────────────────────────────────────────────")
            print(f"     Combined prob: {acc.combined_prob:.1%}  |  EV: {acc.ev_per_unit:+.2%}/unit")
            print(f"     Stake: €{acc.stake_amount:.2f}  |  Win: +€{acc.potential_profit:.0f}  |  Lose: -€{acc.stake_amount:.0f}")

    # Portfolio Summary
    print(f"\n{'=' * 95}")
    print(f"  PORTFOLIO SUMMARY")
    print(f"  {'=' * 91}")
    
    n = len(slip.bets)
    print(f"  Bets: {n} across {slip.n_matches} matches")
    print(f"  Stake: €{slip.total_stake:.0f} ({slip.exposure_pct:.1f}% of bankroll)")
    print(f"  Expected P/L: €{slip.total_ev:+.0f} (ROI: {slip.total_ev / max(slip.total_stake, 1) * 100:+.1f}%)")
    
    max_loss = slip.total_stake
    best_case = sum(b.stake_amount * (b.best_odds - 1) for b in slip.bets)
    print(f"  Risk: -€{max_loss:.0f} (worst) to +€{best_case:.0f} (best)")
    
    # Market mix
    mkt_mix = {}
    for b in slip.bets:
        mkt_mix[b.market] = mkt_mix.get(b.market, 0) + 1
    mix_str = ", ".join(f"{k}: {v}" for k, v in sorted(mkt_mix.items()))
    print(f"  Markets: {mix_str}")
    
    avg_score = sum(compute_confidence_score(b) for b in slip.bets) / n
    print(f"  Avg Confidence: {avg_score:.0f}/100")
    print(f"  Avg Odds: {sum(b.best_odds for b in slip.bets) / n:.2f}")
    
    print(f"\n  {'=' * 91}")
    print("  RULES:")
    print("  1. Bet ONLY at the listed bookmaker and odds (or better)")
    print("  2. If odds drop below Pinnacle → skip the bet (edge gone)")
    print("  3. Never exceed the stake shown — discipline = profit")
    print("  4. Record results in bet_history.json after each match day")
    print(f"  {'=' * 91}")


def save_bet_slip(slip: BetSlip, all_value: List[ValueBet]):
    """Save bet slip to JSON for tracking."""
    output = {
        "generated_at": slip.generated_at,
        "bankroll": slip.bankroll,
        "summary": {
            "total_bets": len(slip.bets),
            "total_stake": slip.total_stake,
            "expected_profit": slip.total_ev,
            "expected_roi_pct": round(slip.total_ev / max(slip.total_stake, 1) * 100, 2),
            "exposure_pct": slip.exposure_pct,
            "matches_covered": slip.n_matches,
        },
        "selected_bets": [asdict(b) for b in slip.bets],
        "all_value_bets_found": len(all_value),
        "rejected_bets": [asdict(b) for b in all_value if not b.is_selected],
    }
    
    path = UPCOMING / "ultimate_bet_slip.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Saved bet slip to %s", path)
    
    # Auto-record placement odds for CLV tracking
    try:
        from scripts.betting.clv_tracker import record_bet_placement
        n_recorded = record_bet_placement(path)
        if n_recorded:
            log.info("CLV: recorded %d bet placements for closing line tracking", n_recorded)
    except Exception as e:
        log.debug("CLV recording skipped: %s", e)
    
    return path


# =============================================================================
# MAIN
# =============================================================================
def main(bankroll: float = 1000.0):
    """Run the ultimate betting engine."""
    cfg = BettingConfig(bankroll=bankroll)
    
    log.info("=" * 70)
    log.info("ULTIMATE BETTING ENGINE v3.3 — Backtest-Calibrated + Full Markets")
    log.info("Bankroll: €%.0f | Kelly: %.0f%% | Min Edge: %.0f%%",
             cfg.bankroll, cfg.kelly_fraction*100, cfg.min_edge_pct)
    log.info("=" * 70)
    
    # ── Load ALL data sources ──
    log.info("\n▶ Loading live data...")
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
    
    log.info("  Predictions:    %d matches", len(predictions))
    log.info("  Odds (40+ bk):  %d matches", len(odds_full))
    log.info("  Extended mkts:  %d matches", len(extended))
    log.info("  Goal preds:     %d matches", len(goal_preds))
    log.info("  BTTS preds:     %d matches", len(btts_preds))
    log.info("  Cards preds:    %d matches", len(cards_preds))
    log.info("  Corners preds:  %d matches", len(corners_preds))
    log.info("  Margin preds:   %d matches (ML handicap probs)", len(margin_preds))
    log.info("  Extra odds:     %d matches (BTTS, DC, alt totals, DNB)", len(extra_odds))
    log.info("  Odds movement:  %d matches (steam/line moves)", len(movement))
    log.info("  Sharp/soft:     %d matches (bookmaker analysis)", len(bk_analysis))
    
    if not predictions or not odds_full:
        log.error("Missing critical data. Run prediction pipeline first.")
        return
    
    # ── Scan ALL markets for value ──
    log.info("\n▶ Scanning all markets for value...")
    all_bets = []
    
    # 1X2
    bets_1x2 = scan_1x2_market(predictions, odds_full, cfg)
    log.info("  1X2:      %d value bets", len(bets_1x2))
    all_bets.extend(bets_1x2)
    
    # O/U
    bets_ou = scan_ou_market(goal_preds, odds_full, cfg)
    log.info("  O/U:      %d value bets", len(bets_ou))
    all_bets.extend(bets_ou)
    
    # Asian Handicap (with ML margin predictions)
    bets_ah = scan_ah_market(predictions, odds_full, extended, cfg,
                             margin_preds=margin_preds)
    log.info("  AH:       %d value bets", len(bets_ah))
    all_bets.extend(bets_ah)
    
    # Double Chance
    bets_dc = scan_dc_market(predictions, extended, odds_full, cfg, extra_odds=extra_odds)
    log.info("  DC:       %d value bets", len(bets_dc))
    all_bets.extend(bets_dc)
    
    # BTTS (with real bookmaker odds)
    bets_btts = scan_btts_market(btts_preds, extra_odds, cfg)
    log.info("  BTTS:     %d value bets", len(bets_btts))
    all_bets.extend(bets_btts)
    
    # Alternate totals (real odds from extra_odds + ML probs)
    bets_alt_ou = scan_alt_totals_market(goal_preds, extra_odds, extended, cfg)
    log.info("  Alt O/U:  %d value bets", len(bets_alt_ou))
    all_bets.extend(bets_alt_ou)
    
    # Draw-No-Bet (real odds from extra_odds)
    bets_dnb = scan_dnb_market(predictions, extra_odds, cfg)
    log.info("  DNB:      %d value bets", len(bets_dnb))
    all_bets.extend(bets_dnb)
    
    # Alternate spreads (real odds + ML handicap probs)
    bets_alt_ah = scan_alt_spreads_market(predictions, extra_odds, margin_preds, cfg)
    log.info("  Alt AH:   %d value bets", len(bets_alt_ah))
    all_bets.extend(bets_alt_ah)
    
    # Cards
    bets_cards = scan_cards_market(cards_preds, cfg)
    log.info("  Cards:    %d value bets", len(bets_cards))
    all_bets.extend(bets_cards)
    
    # Corners
    bets_corners = scan_corners_market(corners_preds, cfg)
    log.info("  Corners:  %d value bets", len(bets_corners))
    all_bets.extend(bets_corners)
    
    log.info("\n  TOTAL: %d value bets across %d market types", len(all_bets),
             len(set(b.market for b in all_bets)))
    
    # ── Apply intelligence filters (steam moves + sharp/soft) ──
    pre_intel = len(all_bets)
    if movement or bk_analysis:
        log.info("\n▶ Applying market intelligence filters...")
        all_bets = apply_intelligence_filters(all_bets, movement, bk_analysis)
        removed = pre_intel - len(all_bets)
        adjusted = sum(1 for b in all_bets if hasattr(b, '_intel_penalty') and b._intel_penalty != 1.0)
        log.info("  Removed: %d bets (sharp money against)", removed)
        log.info("  Adjusted: %d bets (stake modified by intelligence)", adjusted)
        log.info("  Remaining: %d value bets", len(all_bets))
    
    # ── Portfolio optimization ──
    log.info("\n▶ Optimizing portfolio...")
    selected = optimize_portfolio(all_bets, cfg)
    log.info("  Selected: %d bets from %d candidates", len(selected), len(all_bets))
    
    # Generate accumulators from selected singles
    accumulators = generate_accumulators(selected, cfg)
    if accumulators:
        log.info("  Accumulators: %d parlays generated", len(accumulators))
        for acc in accumulators:
            log.info("    %d-fold @ %.2f: %s", acc.n_legs, acc.combined_odds, acc.selections)

    # Generate and display bet slip
    slip = generate_bet_slip(selected, cfg)
    slip._accumulators = accumulators  # type: ignore
    print_bet_slip(slip, all_bets)
    
    # Monte Carlo variance analysis
    monte_carlo_simulation(slip)
    
    # Value bets NOT selected (for reference)
    rejected = [b for b in all_bets if not b.is_selected]
    if rejected:
        print(f"\n  📝 {len(rejected)} additional value bets found but not selected (portfolio limits):")
        for b in rejected[:5]:
            print(f"     • {b.match} | {b.market} {b.selection} | edge={b.edge_pct:.1f}% | EV={b.ev_per_unit:+.4f} | odds={b.best_odds:.2f}")
        if len(rejected) > 5:
            print(f"     ... and {len(rejected) - 5} more")
    
    # Save and record
    save_bet_slip(slip, all_bets)
    n_recorded = record_bets(slip)
    log.info("Recorded %d bets to history tracker", n_recorded)
    
    # ── CLV Tracking: record placements + save odds snapshot ──
    try:
        from scripts.betting.clv_tracker import record_bet_placement, get_clv_summary
        from scripts.data.odds_tracker import save_snapshot
        
        # Save odds snapshot for future closing line comparison
        save_snapshot()
        
        # Record bet placements for CLV tracking
        n_clv = record_bet_placement()
        if n_clv:
            log.info("Recorded %d bet placements for CLV tracking", n_clv)
        
        # Show CLV summary from previous bets (if any)
        summary = get_clv_summary()
        if summary["total_tracked"] > 0:
            print(f"\n  📊 CLV TRACKING ({summary['total_tracked']} previous bets)")
            print(f"     Running CLV: {summary['running_clv_pct']:+.2f}%")
            print(f"     Positive CLV rate: {summary['positive_rate']:.1%}")
            if summary['running_clv_pct'] > 0.5:
                print(f"     ✅ Beating closing lines — edge is REAL")
            elif summary['running_clv_pct'] > -0.5:
                print(f"     ⚠️  Roughly matching closing lines — edge uncertain")
            else:
                print(f"     ❌ Negative CLV — market moves against us")
    except Exception as e:
        log.debug(f"CLV tracking: {e}")
    
    return slip, all_bets


# =============================================================================
# MONTE CARLO SIMULATION — Variance & expected outcomes
# =============================================================================
def monte_carlo_simulation(slip: BetSlip, n_sims: int = 10000):
    """Simulate portfolio outcomes to show expected distribution.
    
    For each simulation, each bet wins/loses based on model_prob.
    Reports percentile outcomes and probability of profit.
    """
    import random
    random.seed(42)
    
    if not slip.bets:
        return
    
    results = []
    for _ in range(n_sims):
        pnl = 0
        for bet in slip.bets:
            # Use blended prob (60% model + 40% sharp) for realistic simulation
            # Pure model_prob overestimates win rate (known draw bias)
            sim_prob = 0.6 * bet.model_prob + 0.4 * bet.sharp_implied_prob
            if random.random() < sim_prob:
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
    
    print(f"\n  MONTE CARLO SIMULATION ({n_sims:,} runs)")
    print(f"  {'─' * 60}")
    print(f"  Probability of profit:  {profit_pct:.1f}%")
    print(f"  Average P/L:            €{avg:+.0f}")
    print(f"  ┌────────────────────────────────────────────")
    print(f"  │  5th percentile:   €{p5:+.0f}  (worst realistic)")
    print(f"  │  25th percentile:  €{p25:+.0f}")
    print(f"  │  Median:           €{p50:+.0f}")
    print(f"  │  75th percentile:  €{p75:+.0f}")
    print(f"  │  95th percentile:  €{p95:+.0f}  (best realistic)")
    print(f"  └────────────────────────────────────────────")
    
    worst = results[0]
    best = results[-1]
    print(f"  Absolute worst:  €{worst:+.0f}  |  Absolute best: €{best:+.0f}")
    
    # Break-even analysis
    be_wins = 0
    for bet in slip.bets:
        if bet.best_odds > 1:
            be_wins += 1 / bet.best_odds
    print(f"  Break-even wins needed: {be_wins:.1f} of {len(slip.bets)} bets")


# =============================================================================
# RESULTS UPDATER — Mark bets as WIN/LOSS after matches
# =============================================================================
def update_results():
    """Interactive CLI to mark bet results after match day."""
    history = load_bet_history()
    pending = [h for h in history if h.get("result") is None]
    
    if not pending:
        print("No pending bets to update.")
        return
    
    print(f"\n{'=' * 70}")
    print(f"  BET RESULTS UPDATER — {len(pending)} pending bets")
    print(f"{'=' * 70}")
    
    updated = 0
    for i, bet in enumerate(pending):
        print(f"\n  [{i+1}/{len(pending)}] {bet['match']} | {bet['market']} → {bet['selection']}")
        print(f"         Odds: {bet['best_odds']:.2f} | Stake: €{bet['stake']:.2f}")
        
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
                  f"({record['win_rate']:.0%}) | ROI: {record['roi']:+.1f}% | P/L: €{record['total_profit']:+.1f}")
        
        # Auto-compute CLV for settled bets
        try:
            from scripts.betting.clv_tracker import track_clv_for_settled_bets, get_clv_summary
            settled = [h for h in history if h.get("result") in ("WIN", "LOSS")]
            if settled:
                clv_result = track_clv_for_settled_bets(settled)
                if clv_result.get("new_tracked", 0) > 0:
                    print(f"\n  📊 CLV computed for {clv_result['new_tracked']} new bets")
                summary = get_clv_summary()
                if summary["total_tracked"] > 0:
                    print(f"     Running CLV: {summary['running_clv_pct']:+.2f}% "
                          f"({summary['positive_clv_count']}/{summary['total_tracked']} positive)")
                    if summary['running_clv_pct'] > 0.5:
                        print(f"     ✅ Beating closing lines — edge is REAL")
                    elif summary['running_clv_pct'] > -0.5:
                        print(f"     ⚠️  Roughly matching closing lines")
                    else:
                        print(f"     ❌ Negative CLV — review strategy")
        except Exception as e:
            log.debug(f"CLV tracking: {e}")
    
    # Drawdown check
    _check_drawdown(history)


def _check_drawdown(history: List[Dict]):
    """Check for excessive drawdown and warn user."""
    settled = [h for h in history if h.get("result") is not None]
    if len(settled) < 5:
        return
    
    # Running P/L
    running = 0
    peak = 0
    max_dd = 0
    for h in settled:
        running += h.get("profit", 0)
        peak = max(peak, running)
        dd = peak - running
        max_dd = max(max_dd, dd)
    
    total_staked = sum(h.get("stake", 0) for h in settled)
    dd_pct = max_dd / max(total_staked / len(settled) * 10, 1) * 100  # DD as % of ~10-bet bankroll
    
    if max_dd > 0:
        print(f"\n  Drawdown Analysis:")
        print(f"    Current P/L: €{running:+.1f}")
        print(f"    Peak P/L:    €{peak:+.1f}")
        print(f"    Max Drawdown: €{max_dd:.1f}")
    
    if dd_pct > 30:
        print(f"    ⚠️  WARNING: Significant drawdown detected ({dd_pct:.0f}%).")
        print(f"    Consider reducing stakes by 50% until recovery.")
    
    # Losing streak
    streak = 0
    max_streak = 0
    for h in settled:
        if h.get("result") == "LOSS":
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    
    if max_streak >= 5:
        print(f"    ⚠️  Losing streak: {max_streak} consecutive losses detected.")
        print(f"    This is within normal variance for value betting. Stay disciplined.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ultimate Betting Engine v3.3")
    parser.add_argument("--bankroll", type=float, default=1000.0, help="Bankroll in €")
    parser.add_argument("--min-edge", type=float, default=3.0, help="Minimum edge %%")
    parser.add_argument("--kelly", type=float, default=0.25, help="Kelly fraction")
    parser.add_argument("--update-results", action="store_true",
                        help="Enter results update mode (mark bets as WIN/LOSS)")
    parser.add_argument("--track-record", action="store_true",
                        help="Show cumulative track record")
    args = parser.parse_args()
    
    if args.update_results:
        update_results()
    elif args.track_record:
        history = load_bet_history()
        record = get_track_record(history)
        if record.get("n_settled", 0) > 0:
            print(f"\nTrack Record: {record['wins']}W/{record['losses']}L "
                  f"({record['win_rate']:.0%}) | ROI: {record['roi']:+.1f}% | "
                  f"P/L: €{record['total_profit']:+.1f} | "
                  f"Staked: €{record['total_staked']:.0f}")
        else:
            print(f"\nNo settled bets yet. {record.get('n_pending', 0)} pending.")
        _check_drawdown(history)
    else:
        main(bankroll=args.bankroll)
