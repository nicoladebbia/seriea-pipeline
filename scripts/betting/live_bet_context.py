#!/usr/bin/env python3
"""Live Bet Context — answers 'what bets/parlays does the user have on this match?'

Central module that ties pending bets + parlays to live match events,
generating human-readable commentary for each market type.

Usage:
    from scripts.betting.live_bet_context import get_match_bet_context
    ctx = get_match_bet_context("Cagliari vs Napoli", home_score=0, away_score=1, minute=34)
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "data"
PARLAY_REPORT_PATH = DATA_DIR / "betting" / "parlay_report.json"


def get_match_bet_context(match_key: str, home_score: int = 0,
                          away_score: int = 0, minute: int = None) -> dict:
    """Build full bet context for a match: pending bets, parlay legs, commentary.

    Args:
        match_key: "Home vs Away" format
        home_score: current home goals
        away_score: current away goals
        minute: current match minute (None if pre-kickoff)

    Returns:
        Dict with match, has_bets, total_stake, and per-bet details including
        parlay involvement and human-readable commentary.
    """
    pending = _load_pending_bets()
    parlay_map = _load_parlay_map()

    match_bets = _find_match_bets(match_key, pending)

    # If no pending bets found, check settled bets too (FT notification
    # fires after auto-settle has already marked them as won/lost)
    if not match_bets:
        match_bets = _load_match_bets_including_settled(match_key)

    bets_out = []
    total_stake = 0.0
    for bet in match_bets:
        market = bet.get("market", "")
        selection = bet.get("selection", "")
        odds = bet.get("odds", 0)
        stake = bet.get("stake", 0)
        confidence = bet.get("confidence", "")
        total_stake += stake

        commentary = _generate_commentary(market, selection, home_score, away_score, minute)

        # Use journal status if already settled, otherwise compute from score
        journal_status = bet.get("status", "")
        if journal_status == "won":
            is_winning = True
        elif journal_status == "lost":
            is_winning = False
        elif journal_status == "push":
            is_winning = None
        else:
            is_winning = _check_winning(market, selection, home_score, away_score)

        parlay_legs = _match_bet_to_parlays(bet, parlay_map, match_key)

        bets_out.append({
            "market": market,
            "selection": selection,
            "odds": odds,
            "stake": stake,
            "confidence": confidence,
            "commentary": commentary,
            "is_winning": is_winning,
            "parlay_legs": parlay_legs,
        })

    return {
        "match": match_key,
        "has_bets": bool(bets_out),
        "total_stake": round(total_stake, 2),
        "bets": bets_out,
    }


# ─── Internal helpers ────────────────────────────────────────────────────────

def _load_pending_bets() -> List[Dict]:
    """Load pending bets from the bet journal (single source of truth)."""
    try:
        from scripts.betting.bet_journal import get_pending_bets
        return get_pending_bets()
    except Exception as e:
        log.warning("Failed to load pending bets from journal: %s", e)
        return []


def _load_match_bets_including_settled(match_key: str) -> List[Dict]:
    """Load ALL bets for a match — pending AND recently settled.

    At full-time, bets may already be settled by auto-settle before the
    FT notification fires. We need to include them to show correct P&L.
    """
    try:
        journal_path = DATA_DIR / "betting" / "bet_journal.json"
        if not journal_path.exists():
            return []
        with open(journal_path) as f:
            journal = json.load(f)
        bets = journal.get("bets", {})
        if isinstance(bets, dict):
            bets = list(bets.values())

        norm_target = _fuzzy_match_key(match_key)
        matched = []
        for b in bets:
            if b.get("status") in ("pending", "won", "lost", "push"):
                if _fuzzy_match_key(b.get("match", "")) == norm_target:
                    matched.append(b)
        return matched
    except Exception as e:
        log.warning("Failed to load match bets: %s", e)
        return []


def _load_parlay_map() -> Dict[str, List[dict]]:
    """Build reverse index: normalized match key -> list of parlay leg info."""
    if not PARLAY_REPORT_PATH.exists():
        return {}
    try:
        with open(PARLAY_REPORT_PATH) as f:
            report = json.load(f)
    except Exception as e:
        log.warning("Failed to load parlay report: %s", e)
        return {}

    result: Dict[str, List[dict]] = {}
    categories = report.get("categories", {})
    for cat_name, parlays in categories.items():
        for parlay in parlays:
            parlay_id = parlay.get("id", "???")
            legs = parlay.get("legs", [])
            combined_odds = parlay.get("combined_odds", 0)
            for i, leg in enumerate(legs):
                leg_match = leg.get("match", "")
                norm_key = _fuzzy_match_key(leg_match)
                entry = {
                    "parlay_id": parlay_id,
                    "category": cat_name,
                    "leg_index": i,
                    "total_legs": len(legs),
                    "combined_odds": combined_odds,
                    "leg_market": leg.get("market", ""),
                    "leg_selection": leg.get("selection", ""),
                }
                result.setdefault(norm_key, []).append(entry)
    return result


def _find_match_bets(match_key: str, pending: List[Dict]) -> List[Dict]:
    """Find all pending bets for a given match using fuzzy name matching."""
    norm_target = _fuzzy_match_key(match_key)
    matched = []
    for bet in pending:
        bet_match = bet.get("match", "")
        if _fuzzy_match_key(bet_match) == norm_target:
            matched.append(bet)
    return matched


def _match_bet_to_parlays(bet: dict, parlay_map: dict,
                          match_key: str) -> List[dict]:
    """Find parlay legs that include this bet (by match + market + selection)."""
    norm_key = _fuzzy_match_key(match_key)
    candidates = parlay_map.get(norm_key, [])
    if not candidates:
        return []

    bet_market = _normalize_market(bet.get("market", ""))
    bet_selection = bet.get("selection", "").lower().strip()

    matched = []
    for leg in candidates:
        leg_market = _normalize_market(leg.get("leg_market", ""))
        leg_selection = leg.get("leg_selection", "").lower().strip()
        if leg_market == bet_market and leg_selection == bet_selection:
            matched.append({
                "parlay_id": leg["parlay_id"],
                "category": leg["category"],
                "leg_index": leg["leg_index"],
                "total_legs": leg["total_legs"],
                "combined_odds": leg["combined_odds"],
            })
    return matched


def _generate_commentary(market: str, selection: str, home_score: int,
                         away_score: int, minute: int = None) -> str:
    """Generate human-readable status text for a bet given current score."""
    total = home_score + away_score
    remaining = max(0, 90 - minute) if minute else None
    time_str = f" in {remaining} min" if remaining is not None else ""
    norm_market = _normalize_market(market)

    # Over/Under
    if norm_market == "totals":
        line = _extract_line(selection)
        if line is not None:
            if "over" in selection.lower():
                needed = line - total + 1
                if total > line:
                    return "COVERED"
                elif needed <= 0:
                    return "COVERED"
                else:
                    return f"needs {needed:.0f} more goal{'s' if needed != 1 else ''}{time_str}"
            elif "under" in selection.lower():
                if total > line:
                    return "BUSTED"
                elif total == line:
                    return f"one more goal busts it{time_str}"
                else:
                    margin = line - total
                    return f"ON TRACK — {margin:.0f} goal margin{time_str}"

    # 1X2 / Match result
    if norm_market in ("h2h", "1x2"):
        sel = selection.lower().strip()
        if sel in ("home", "1"):
            if home_score > away_score:
                return "WINNING"
            elif home_score == away_score:
                return f"DRAWING — need a home goal{time_str}"
            else:
                return f"LOSING — need comeback{time_str}"
        elif sel in ("draw", "x"):
            if home_score == away_score:
                return "ON TRACK"
            else:
                return f"LOSING — need equaliser{time_str}"
        elif sel in ("away", "2"):
            if away_score > home_score:
                return "WINNING"
            elif home_score == away_score:
                return f"DRAWING — need an away goal{time_str}"
            else:
                return f"LOSING — need comeback{time_str}"

    # Double Chance
    if norm_market == "double_chance":
        sel = selection.upper().strip()
        if sel == "1X":
            if home_score >= away_score:
                return "ON TRACK"
            else:
                return f"LOSING — need equaliser{time_str}"
        elif sel == "X2":
            if away_score >= home_score:
                return "ON TRACK"
            else:
                return f"LOSING — need equaliser{time_str}"
        elif sel == "12":
            if home_score != away_score:
                return "ON TRACK"
            else:
                return f"DRAWING — need a goal from either side{time_str}"

    # BTTS
    if norm_market == "btts":
        sel = selection.lower().strip()
        home_scored = home_score > 0
        away_scored = away_score > 0
        if sel == "yes":
            if home_scored and away_scored:
                return "HIT"
            elif home_scored:
                return f"need away team to score{time_str}"
            elif away_scored:
                return f"need home team to score{time_str}"
            else:
                return f"need both teams to score{time_str}"
        elif sel == "no":
            if home_scored and away_scored:
                return "BUSTED"
            elif not home_scored and not away_scored:
                return "ON TRACK — clean sheet both sides"
            else:
                return f"ON TRACK — one side scoreless{time_str}"

    # Asian Handicap / Spreads
    if norm_market == "spreads":
        line = _extract_line(selection)
        if line is not None:
            is_home = not selection.lower().startswith("away")
            if is_home:
                adjusted = home_score + line - away_score
            else:
                adjusted = away_score + line - home_score
            if adjusted > 0:
                return f"WINNING by adjusted margin {adjusted:+.1f}"
            elif adjusted == 0:
                return "PUSH territory"
            else:
                return f"LOSING by adjusted margin {adjusted:+.1f}{time_str}"

    return "in play" if minute else "pending"


def _check_winning(market: str, selection: str, home_score: int,
                   away_score: int) -> Optional[bool]:
    """Quick check: is this bet currently winning? None if unclear."""
    commentary = _generate_commentary(market, selection, home_score, away_score)
    if any(w in commentary for w in ("WINNING", "ON TRACK", "COVERED", "HIT")):
        return True
    if any(w in commentary for w in ("LOSING", "BUSTED")):
        return False
    return None


def _fuzzy_match_key(match_str: str) -> str:
    """Normalize a match string for comparison: 'Cagliari vs Napoli' -> 'cagliari_napoli'."""
    try:
        from config.team_names import normalize_team
    except ImportError:
        normalize_team = lambda x: x

    parts = re.split(r'\s+vs\.?\s+', match_str.strip(), maxsplit=1)
    if len(parts) == 2:
        home = normalize_team(parts[0].strip()).lower()
        away = normalize_team(parts[1].strip()).lower()
        return f"{home}_{away}"
    return match_str.strip().lower().replace(" ", "_")


def _normalize_market(market: str) -> str:
    """Map various market name variants to canonical form."""
    m = market.lower().strip()
    if m in ("h2h", "1x2", "match_result", "moneyline"):
        return "h2h"
    if m in ("totals", "o/u", "over_under", "over/under"):
        return "totals"
    if m in ("spreads", "ah", "asian_handicap", "handicap"):
        return "spreads"
    if m in ("btts", "both_teams_to_score"):
        return "btts"
    if m in ("double_chance", "dc"):
        return "double_chance"
    if m in ("draw_no_bet", "dnb"):
        return "draw_no_bet"
    return m


def _extract_line(selection: str) -> Optional[float]:
    """Extract numeric line from selection string, e.g. 'Over 2.5' -> 2.5."""
    match = re.search(r'[-+]?\d+\.?\d*', selection)
    if match:
        return float(match.group())
    return None
