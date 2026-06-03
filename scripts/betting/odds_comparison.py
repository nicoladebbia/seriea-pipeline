"""Odds-comparison engine — the brain of the betting edge.

Takes the model's probability for an outcome and a bookmaker's offered price,
and decides whether there is value. Source-agnostic: works against Betfair,
Sisal, Pinnacle, or a sample file — it only needs (model_prob, book_odds).

This is deliberately separate from the live odds FETCHING (Betfair client,
which needs credentials). The engine is pure and fully testable offline.

Core definitions:
  fair_odds   = 1 / model_prob              (what odds the model thinks are fair)
  implied_p   = 1 / book_odds               (the book's implied probability, with vig)
  edge        = model_prob - implied_p      (probability edge; >0 = model sees value)
  edge_pct    = edge * 100
  ev          = model_prob * (book_odds - 1) - (1 - model_prob)   (expected value per 1u)
  value       = book_odds / fair_odds - 1   (how much the price beats fair; >0 = value)

A bet is flagged only when ALL hold:
  - ev > 0                         (positive expected value — the hard floor)
  - edge_pct >= min_edge_pct       (clears the noise threshold for the market)
  - book_odds within [min_odds, max_odds]   (avoid junk prices)

NOTE: a flagged edge is NECESSARY but not SUFFICIENT to bet. Calibration of the
model probability matters (a market the backtest called "noise" can show a fake
edge against an efficient price). Per-market min_edge encodes how much we trust
the model there. The honest workflow: only the who-wins family (1X2/DC) earned
real calibration skill; goal-quantity markets need a HIGHER bar before trusting
a flagged edge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Per-market config derived from the 2026-06-03 calibration backtest
# (.plans/projection-backtest-results.md). Each market has:
#   min_edge : pp edge required to flag value (higher = less model trust)
#   trust    : "skill" (real calibration skill) | "noise" (at base rate — be skeptical)
# Who-wins family showed real skill (+0.010 held-out); goal-quantity markets were
# at the noise floor — a flagged edge there is more likely model error than real value,
# so they get a high bar AND a 'noise' warning the UI surfaces.
MARKET_CONFIG = {
    "1x2":            {"min_edge": 3.0, "trust": "skill"},
    "double_chance":  {"min_edge": 3.0, "trust": "skill"},
    "winning_margin": {"min_edge": 6.0, "trust": "noise"},   # direction has weak skill; magnitude noisy
    "htft":           {"min_edge": 6.0, "trust": "noise"},
    "over_under":     {"min_edge": 6.0, "trust": "noise"},   # goal-quantity = noise floor
    "btts":           {"min_edge": 7.0, "trust": "noise"},
    "multigol":       {"min_edge": 7.0, "trust": "noise"},
    "team_totals":    {"min_edge": 7.0, "trust": "noise"},
    "exact_score":    {"min_edge": 9.0, "trust": "noise"},   # high variance
    "odd_even":       {"min_edge": 99.0, "trust": "noise"},  # coinflip → effectively off
}
GLOBAL_MIN_EDGE = 5.0
MIN_ODDS = 1.20
MAX_ODDS = 15.0

# Staking — mirrors production betting_unified.py caps.
KELLY_FRACTION = 0.05      # fractional Kelly (production default)
MAX_STAKE_PCT = 2.5        # max single bet as % of bankroll
MIN_STAKE_PCT = 0.20       # below this, edge too thin → no stake
NOISE_KELLY_MULT = 0.5     # halve the stake on 'noise'-trust markets (less model trust)


def kelly_stake_pct(model_prob: float, odds: float, trust: str = "skill") -> float:
    """Recommended stake as % of bankroll, fractional Kelly with production caps.

    Reuses the same Kelly math as betting_unified. Noise-trust markets get half
    stake (we trust the model's probability less there). Returns 0 if too thin.
    """
    from scripts.betting.betting_unified import calculate_kelly
    frac = KELLY_FRACTION * (NOISE_KELLY_MULT if trust == "noise" else 1.0)
    kelly = calculate_kelly(model_prob, odds, fraction=frac)  # fraction of bankroll
    stake_pct = kelly * 100
    stake_pct = min(stake_pct, MAX_STAKE_PCT)
    if stake_pct < MIN_STAKE_PCT:
        return 0.0
    return round(stake_pct, 2)


def remove_vig(prices: dict) -> dict:
    """De-vig a set of COMPLEMENTARY outcomes: normalize implied probs to sum to 1.

    Converts {outcome: odds} → {outcome: fair_prob} with the book's margin removed.
    IMPORTANT: the outcomes passed in must be mutually exclusive and exhaustive
    (e.g. home/draw/away, or over_2.5/under_2.5). Passing a mixed set (multiple O/U
    lines together) double-counts and is WRONG — group by line-pair first via
    devig_market(). Uses the simple proportional (multiplicative) method.
    """
    implied = {k: 1.0 / o for k, o in prices.items() if o and o > 1}
    total = sum(implied.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in implied.items()}


def devig_market(market: str, prices: dict) -> dict:
    """De-vig a full market's prices, grouping complementary outcomes correctly.

    O/U markets are de-vigged per line-pair (over_X/under_X); two-way markets
    (btts yes/no) and the 3-way 1x2 as single groups. Markets without a clean
    complementary structure (exact_score, multigol independent ranges, htft) are
    NOT de-vigged here — for those the raw 1/odds is used (a flagged edge is
    already treated skeptically via the 'noise' trust flag).
    """
    if market == "over_under":
        out = {}
        lines = sorted({k.split("_", 1)[1] for k in prices if "_" in k})
        for ln in lines:
            pair = {k: prices[k] for k in (f"over_{ln}", f"under_{ln}") if k in prices}
            if len(pair) == 2:
                out.update(remove_vig(pair))
        return out
    if market in ("1x2", "btts", "double_chance", "odd_even", "team_totals"):
        # team_totals: each side's over/under is a pair, but keys are home_over_X etc.
        if market == "team_totals":
            out = {}
            for side in ("home", "away"):
                for ln in ("0.5", "1.5", "2.5"):
                    pair = {k: prices[k] for k in (f"{side}_over_{ln}", f"{side}_under_{ln}") if k in prices}
                    if len(pair) == 2:
                        out.update(remove_vig(pair))
            return out
        # double_chance outcomes overlap (1X, X2, 12 are NOT mutually exclusive) →
        # don't normalize across them; de-vig is not well-defined, use raw.
        if market == "double_chance":
            return {}
        return remove_vig(prices)
    # exact_score, multigol, htft, winning_margin: no clean complement → raw implied
    return {}


@dataclass
class EdgeResult:
    market: str
    outcome: str
    model_prob: float
    fair_odds: float
    book_odds: float
    book: str
    implied_prob: float       # de-vigged (the book's TRUE implied prob)
    raw_implied: float        # with vig (1/book_odds)
    edge_pct: float           # model_prob - de-vigged implied, in pp
    ev: float
    value_pct: float
    is_value: bool
    trust: str = "skill"      # "skill" | "noise" — model's calibration on this market
    stake_pct: float = 0.0    # recommended stake as % of bankroll (fractional Kelly)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "market": self.market, "outcome": self.outcome,
            "model_prob": round(self.model_prob, 4),
            "fair_odds": round(self.fair_odds, 2),
            "book_odds": round(self.book_odds, 2),
            "book": self.book,
            "implied_prob": round(self.implied_prob, 4),
            "raw_implied": round(self.raw_implied, 4),
            "edge_pct": round(self.edge_pct, 2),
            "ev": round(self.ev, 4),
            "value_pct": round(self.value_pct, 2),
            "is_value": self.is_value,
            "trust": self.trust,
            "stake_pct": self.stake_pct,
            "reason": self.reason,
        }


def compare_one(market: str, outcome: str, model_prob: float, book_odds: float,
                book: str = "", devig_implied: Optional[float] = None,
                min_edge_pct: Optional[float] = None,
                min_odds: float = MIN_ODDS, max_odds: float = MAX_ODDS) -> Optional[EdgeResult]:
    """Compare one model probability against one book price. Returns EdgeResult or None.

    devig_implied: the book's de-vigged implied prob for this outcome (preferred).
                   If None, falls back to raw 1/book_odds (vig-inflated).
    """
    if model_prob is None or book_odds is None:
        return None
    if not (0 < model_prob < 1) or book_odds <= 1:
        return None

    cfg = MARKET_CONFIG.get(market, {})
    trust = cfg.get("trust", "skill")
    if min_edge_pct is None:
        min_edge_pct = cfg.get("min_edge", GLOBAL_MIN_EDGE)

    fair_odds = 1.0 / model_prob
    raw_implied = 1.0 / book_odds
    implied_p = devig_implied if devig_implied is not None else raw_implied
    edge_pct = (model_prob - implied_p) * 100          # edge vs DE-VIGGED price
    ev = model_prob * (book_odds - 1) - (1 - model_prob)
    value_pct = (book_odds / fair_odds - 1) * 100

    reasons = []
    is_value = True
    if ev <= 0:
        is_value = False; reasons.append("EV<=0")
    if edge_pct < min_edge_pct:
        is_value = False; reasons.append(f"edge<{min_edge_pct}")
    if not (min_odds <= book_odds <= max_odds):
        is_value = False; reasons.append("odds out of range")

    # recommended stake (only meaningful for value bets)
    stake = kelly_stake_pct(model_prob, book_odds, trust=trust) if is_value else 0.0

    return EdgeResult(
        market=market, outcome=outcome, model_prob=model_prob, fair_odds=fair_odds,
        book_odds=book_odds, book=book, implied_prob=implied_p, raw_implied=raw_implied,
        edge_pct=edge_pct, ev=ev, value_pct=value_pct, is_value=is_value, trust=trust,
        stake_pct=stake, reason="OK" if is_value else "; ".join(reasons),
    )


def compare_match(projection: dict, book_odds: dict, book: str = "") -> list[EdgeResult]:
    """Compare a full per-match projection against a book's odds for that match.

    projection: a single entry from /api/projections (has the market sub-dicts).
    book_odds:  {market_key: {outcome_key: price}} for the same match.
                Outcome keys must match the projection's keys (see odds_keys.py mapping
                if a book uses different names).
    Returns all EdgeResults (value and non-value) so the caller can show context.
    """
    results: list[EdgeResult] = []

    # map projection market → {outcome: model_prob} for EVERY market the page projects
    def market_probs(market: str):
        if market == "1x2":
            p = projection.get("probabilities", {})
            return {"home": p.get("home"), "draw": p.get("draw"), "away": p.get("away")}
        if market == "btts":
            b = projection.get("btts", {})
            return {"yes": b.get("yes", {}).get("prob"), "no": b.get("no", {}).get("prob")}
        if market == "double_chance":
            return {k: v.get("prob") for k, v in projection.get("double_chance", {}).items()}
        if market == "multigol":
            return {r["range"]: r["prob"] for r in projection.get("multi_goal", [])}
        if market == "exact_score":
            return {r["score"]: r["prob"] for r in projection.get("exact_score", [])}
        if market == "htft":
            return {r["combo"]: r["prob"] for r in projection.get("htft", [])}
        if market == "over_under":
            # derive total-goals O/U from the multigol ranges (independent Poisson on total)
            # multi_goal has cumulative-ish ranges; reconstruct over_X from "X+" entries
            mg = {r["range"]: r["prob"] for r in projection.get("multi_goal", [])}
            ou = {}
            # "3+" → over 2.5 ; "4+" → over 3.5 ; over 1.5 = 1 - P(0-1)
            if "0-1" in mg:
                ou["over_1.5"] = max(0.0, 1 - mg["0-1"]); ou["under_1.5"] = mg["0-1"]
            if "3+" in mg:
                ou["over_2.5"] = mg["3+"]; ou["under_2.5"] = max(0.0, 1 - mg["3+"])
            if "4+" in mg:
                ou["over_3.5"] = mg["4+"]; ou["under_3.5"] = max(0.0, 1 - mg["4+"])
            return ou
        # generic {outcome: {prob, fair_odds}} dict markets (winning_margin, team_totals, odd_even, ...)
        md = projection.get(market)
        if isinstance(md, dict):
            return {k: v.get("prob") for k, v in md.items() if isinstance(v, dict) and "prob" in v}
        return {}

    for market, outcomes in book_odds.items():
        probs = market_probs(market)
        if not probs:
            continue
        # de-vig the book's prices for THIS market (grouped by complementary outcomes;
        # returns {} for markets where de-vig isn't well-defined → raw 1/odds is used)
        devig = devig_market(market, outcomes)
        for outcome, price in outcomes.items():
            mp = probs.get(outcome)
            if mp is None:
                continue
            r = compare_one(market, outcome, mp, price, book=book,
                            devig_implied=devig.get(outcome))
            if r is not None:
                results.append(r)
    return results


def best_value_bets(results: list[EdgeResult], top_n: int = 10) -> list[EdgeResult]:
    """Return only the value bets, sorted by edge descending."""
    vals = [r for r in results if r.is_value]
    vals.sort(key=lambda r: -r.edge_pct)
    return vals[:top_n]
