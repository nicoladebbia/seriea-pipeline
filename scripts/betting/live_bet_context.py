#!/usr/bin/env python3
"""Live Bet Context — answers 'what bets/parlays does the user have on this match?'

Central module that ties pending bets + parlays to live match events,
generating human-readable commentary for each market type.

Handles the REAL journal shapes (verified 2026-08-31 against bet_journal.json):
    market '1X2'      selection 'Home' / 'Draw' / 'Away'
    market 'O/U 2.5'  selection 'Over 2.5' / 'Under 2.0' / 'Alt Over 3.25'
    market 'DC'       selection '1X (Home or Draw)'
    market 'BTTS'     selection 'BTTS No'
    market 'DNB'      selection 'DNB Home'
    market 'AH 0.25'  selection 'Away -0.2'   (selection line is ROUNDED —
                       the true magnitude lives in the market string)
    market 'spreads'  selection 'HOME +0'
    market 'PARLAY'   selection '3-leg parlay'
Plus structural support for player props (shots on target, shots, passes,
tackles) via live Sofascore player stats, for the day those are activated.

Usage:
    from scripts.betting.live_bet_context import get_match_bet_context
    ctx = get_match_bet_context("Cagliari vs Napoli", home_score=0, away_score=1, minute=34)
"""

import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

from scripts.utils.parsing import extract_line

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "data"
PARLAY_REPORT_PATH = DATA_DIR / "betting" / "parlay_report.json"


def get_match_bet_context(match_key: str, home_score: int = 0,
                          away_score: int = 0, minute: int = None,
                          player_stats: dict = None) -> dict:
    """Build full bet context for a match: pending bets, parlay legs, commentary.

    Args:
        match_key: "Home vs Away" format
        home_score: current home goals
        away_score: current away goals
        minute: current match minute (None if pre-kickoff)
        player_stats: live Sofascore player stats {"home": [...], "away": [...]}
            (optional — enables live commentary on player-prop bets)

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

        commentary = _generate_commentary(market, selection, home_score,
                                          away_score, minute,
                                          player_stats=player_stats)

        # Use journal status if already settled, otherwise compute from score
        journal_status = bet.get("status", "")
        if journal_status == "won":
            is_winning = True
        elif journal_status == "lost":
            is_winning = False
        elif journal_status in ("push", "voided", "void"):
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
            if b.get("status") in ("pending", "won", "lost", "push", "voided", "void"):
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


# ─── Line / side resolution ──────────────────────────────────────────────────

def resolve_bet_line(market: str, selection: str) -> Optional[float]:
    """The true line for a totals/AH bet.

    The journal rounds AH selection lines to 1dp ('AH 0.25' -> 'Away -0.2'),
    so for handicaps the magnitude comes from the MARKET string and the sign
    from the SELECTION. Totals prefer the selection ('Alt Over 3.25' on
    market 'O/U 2.5'), falling back to the market.
    """
    sel_line = extract_line(selection)
    mkt_line = extract_line(market)
    norm = _normalize_market(market)
    if norm == "spreads":
        if sel_line is not None and mkt_line is not None:
            # Same handicap magnitude, selection just rounded -> trust market
            if abs(abs(mkt_line) - abs(sel_line)) <= 0.06:
                return abs(mkt_line) if sel_line >= 0 else -abs(mkt_line)
        return sel_line if sel_line is not None else mkt_line
    # Totals and everything else: selection first, market fallback
    return sel_line if sel_line is not None else mkt_line


def _line_fraction(line: float) -> int:
    """0, 25, 50 or 75 — the fractional part of a betting line in cents."""
    return int(round((abs(line) - math.floor(abs(line) + 1e-9)) * 100))


def _sel_side(selection: str) -> str:
    """'home' / 'away' / 'draw' / '' from a selection's tokens."""
    for tok in re.findall(r"[a-z0-9]+", selection.lower()):
        if tok in ("home", "1"):
            return "home"
        if tok in ("away", "2"):
            return "away"
        if tok in ("draw", "x"):
            return "draw"
    return ""


# ─── Player props (structural — activates when such bets exist) ──────────────

_PLAYER_PROP_PATTERNS = [
    (re.compile(r"shots?\s*on\s*target|\bsot\b", re.I), "shots_on_target", "shots on target"),
    (re.compile(r"\bshots?\b", re.I), "shots", "shots"),
    (re.compile(r"\bpass(es)?\b", re.I), "total_passes", "passes"),
    (re.compile(r"\btackles?\b", re.I), "tackles", "tackles"),
    (re.compile(r"\bassists?\b", re.I), "assists", "assists"),
]


def _player_prop_stat(market: str, selection: str):
    """(stat_key, label) if this bet is a player prop, else None."""
    blob = f"{market} {selection}"
    for pattern, stat_key, label in _PLAYER_PROP_PATTERNS:
        if pattern.search(blob):
            return stat_key, label
    return None


def _fold(s: str) -> str:
    """Lowercase + strip accents: 'Martínez' -> 'martinez'."""
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if not unicodedata.combining(c))


def _find_player_stat(player_stats: dict, selection: str, stat_key: str):
    """(player_name, value) for the player named in the selection, else None."""
    if not player_stats:
        return None
    # Player name = the selection text before Over/Under, minus stat words
    name_part = re.split(r"\b(over|under)\b", selection, flags=re.I)[0]
    name_part = re.sub(
        r"shots?\s*on\s*target|\bsot\b|\bshots?\b|\bpass(es)?\b|\btackles?\b|\bassists?\b",
        "", name_part, flags=re.I).strip(" -:").strip()
    if not name_part:
        return None
    target = _fold(name_part)
    for side in ("home", "away"):
        for p in player_stats.get(side, []) or []:
            pname = _fold(p.get("name") or "")
            if pname and (target in pname or pname in target):
                return p.get("name"), p.get(stat_key)
    return None


# ─── Commentary ──────────────────────────────────────────────────────────────

def _generate_commentary(market: str, selection: str, home_score: int,
                         away_score: int, minute: int = None,
                         player_stats: dict = None) -> str:
    """Human-readable live status for a bet given the current score.

    Vocabulary contract (consumed by _check_winning and the tests):
    winning states contain WINNING / ON TRACK / COVERED / HIT / HALF WON;
    losing states contain LOSING / BUSTED / HALF LOSS; refund states
    contain PUSH or 'stake refunded'.
    """
    total = home_score + away_score
    remaining = max(0, 90 - minute) if minute else None
    time_str = f" in {remaining} min" if remaining is not None else ""
    norm_market = _normalize_market(market)

    # Player props (shots on target, shots, passes, ...) — live stat vs line
    prop = _player_prop_stat(market, selection)
    if prop:
        stat_key, label = prop
        line = resolve_bet_line(market, selection)
        found = _find_player_stat(player_stats, selection, stat_key)
        if line is None:
            return "player prop — no line parsed"
        direction = "over" if "under" not in selection.lower() else "under"
        if found and found[1] is not None:
            name, value = found
            value = float(value)
            if direction == "over":
                if value > line:
                    return f"HIT — {name} has {value:g} {label} (line {line:g})"
                needed = math.floor(line) + 1 - value
                return (f"{name}: {value:g} {label} — "
                        f"needs {needed:g} more{time_str}")
            else:
                if value > line:
                    return f"BUSTED — {name} at {value:g} {label} (line {line:g})"
                return f"ON TRACK — {name} at {value:g} {label}, line {line:g}{time_str}"
        return f"player prop ({label} {direction} {line:g}) — live stat not tracked yet"

    # Over/Under (goals)
    if norm_market == "totals":
        line = resolve_bet_line(market, selection)
        if line is not None:
            frac = _line_fraction(line)
            floor_line = int(math.floor(line + 1e-9))
            if "over" in selection.lower():
                # Over X.75 at exactly X+1 goals: half won, half pushes
                if frac == 75 and total == floor_line + 1:
                    return (f"HALF WON — half (Over {floor_line}.5) won, half "
                            f"(Over {floor_line + 1}) refunds; one more goal wins fully{time_str}")
                if total > line:
                    return "COVERED"
                needed = floor_line + 1 - total
                if frac == 0 and total == floor_line:
                    return f"PUSH zone — one more goal wins, no more goals = stake refunded{time_str}"
                if frac == 25 and total == floor_line:
                    return (f"needs 1 more goal{time_str} — at FT here: half refunded, half lost")
                return f"needs {needed} more goal{'s' if needed != 1 else ''}{time_str}"
            elif "under" in selection.lower():
                # Under X.75 at exactly X+1 goals: half lost, half pushes
                if frac == 75 and total == floor_line + 1:
                    return (f"HALF LOSS — half (Under {floor_line}.5) lost, "
                            f"half (Under {floor_line + 1}) refunds")
                if total > line:
                    return "BUSTED"
                if frac == 0 and total == floor_line:
                    return f"PUSH zone — no more goals = stake refunded, one more loses{time_str}"
                if frac == 25 and total == floor_line:
                    return f"HALF WON zone — at FT here: half won, half refunded; a goal busts it{time_str}"
                if frac == 50 and total == floor_line:
                    return f"one more goal busts it{time_str}"
                margin = line - total
                return f"ON TRACK — {margin:g} goal margin{time_str}"

    # 1X2 / Match result
    if norm_market == "h2h":
        side = _sel_side(selection)
        if side == "home":
            if home_score > away_score:
                return "WINNING"
            elif home_score == away_score:
                return f"DRAWING — need a home goal{time_str}"
            else:
                return f"LOSING — need comeback{time_str}"
        elif side == "draw":
            if home_score == away_score:
                return "ON TRACK"
            else:
                return f"LOSING — need equaliser{time_str}"
        elif side == "away":
            if away_score > home_score:
                return "WINNING"
            elif home_score == away_score:
                return f"DRAWING — need an away goal{time_str}"
            else:
                return f"LOSING — need comeback{time_str}"

    # Double Chance — selection like '1X (Home or Draw)': match the token
    if norm_market == "double_chance":
        sel_tok = ""
        for tok in re.findall(r"[A-Z0-9]+", selection.upper()):
            if tok in ("1X", "X2", "12"):
                sel_tok = tok
                break
        if sel_tok == "1X":
            if home_score >= away_score:
                return "ON TRACK"
            return f"LOSING — need equaliser{time_str}"
        elif sel_tok == "X2":
            if away_score >= home_score:
                return "ON TRACK"
            return f"LOSING — need equaliser{time_str}"
        elif sel_tok == "12":
            if home_score != away_score:
                return "ON TRACK"
            return f"DRAWING — need a goal from either side{time_str}"

    # Draw No Bet — 'DNB Home' / 'DNB Away'
    if norm_market == "draw_no_bet":
        side = _sel_side(selection)
        lead = home_score - away_score if side == "home" else away_score - home_score
        if side in ("home", "away"):
            if lead > 0:
                return "WINNING"
            elif lead == 0:
                return f"PUSH zone — draw = stake refunded; need a goal to win{time_str}"
            else:
                return f"LOSING — need comeback (draw refunds){time_str}"

    # BTTS — 'BTTS Yes' / 'BTTS No'
    if norm_market == "btts":
        toks = re.findall(r"[a-z]+", selection.lower())
        home_scored = home_score > 0
        away_scored = away_score > 0
        if "yes" in toks:
            if home_scored and away_scored:
                return "HIT"
            elif home_scored:
                return f"need away team to score{time_str}"
            elif away_scored:
                return f"need home team to score{time_str}"
            else:
                return f"need both teams to score{time_str}"
        elif "no" in toks:
            if home_scored and away_scored:
                return "BUSTED"
            elif not home_scored and not away_scored:
                return "ON TRACK — clean sheet both sides"
            else:
                return f"ON TRACK — one side scoreless{time_str}"

    # Asian Handicap / Spreads
    if norm_market == "spreads":
        line = resolve_bet_line(market, selection)
        if line is not None:
            side = _sel_side(selection) or "home"
            if side == "home":
                adjusted = home_score + line - away_score
            else:
                adjusted = away_score + line - home_score
            adjusted = round(adjusted, 2)
            if adjusted >= 0.5:
                return f"WINNING by adjusted margin {adjusted:+g}"
            elif abs(adjusted - 0.25) < 0.01:
                return "HALF WON position — half won, half refunds at this score"
            elif adjusted == 0:
                return f"PUSH position — stake refunded at this score{time_str}"
            elif abs(adjusted + 0.25) < 0.01:
                return f"HALF LOSS position — half refunded, half lost at this score{time_str}"
            else:
                return f"LOSING by adjusted margin {adjusted:+g}{time_str}"

    # Parlay container rows — legs are tracked individually
    if norm_market == "parlay":
        return "parlay — legs tracked individually"

    return "in play" if minute else "pending"


def _check_winning(market: str, selection: str, home_score: int,
                   away_score: int) -> Optional[bool]:
    """Quick check: is this bet currently winning? None if unclear/push."""
    commentary = _generate_commentary(market, selection, home_score, away_score)
    if any(w in commentary for w in ("PUSH", "stake refunded")):
        return None
    if any(w in commentary for w in ("WINNING", "ON TRACK", "COVERED", "HIT", "HALF WON")):
        return True
    if any(w in commentary for w in ("LOSING", "BUSTED", "HALF LOSS")):
        return False
    return None


def _fuzzy_match_key(match_str: str) -> str:
    """Normalize a match string for comparison: 'Cagliari vs Napoli' -> 'cagliari_napoli'."""
    try:
        from config.team_names import normalize_team
    except ImportError:
        normalize_team = lambda x: x  # noqa: E731

    parts = re.split(r'\s+vs\.?\s+', match_str.strip(), maxsplit=1)
    if len(parts) == 2:
        home = normalize_team(parts[0].strip()).lower()
        away = normalize_team(parts[1].strip()).lower()
        return f"{home}_{away}"
    return match_str.strip().lower().replace(" ", "_")


def _normalize_market(market: str) -> str:
    """Map market name variants (incl. real journal shapes) to canonical form."""
    m = market.lower().strip()
    if m in ("h2h", "1x2", "match_result", "moneyline"):
        return "h2h"
    if m in ("totals", "o/u", "over_under", "over/under") or m.startswith(("o/u ", "alt o/u")):
        return "totals"
    if m in ("spreads", "ah", "asian_handicap", "handicap") or m.startswith(("ah ", "ah-", "ah+")):
        return "spreads"
    if m in ("btts", "both_teams_to_score") or m.startswith("btts"):
        return "btts"
    if m in ("double_chance", "dc"):
        return "double_chance"
    if m in ("draw_no_bet", "dnb"):
        return "draw_no_bet"
    if m.startswith("parlay"):
        return "parlay"
    return m
