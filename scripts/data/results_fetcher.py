#!/usr/bin/env python3
"""AUTO-RESULTS FETCHER - Automated Result Tracking via Odds API /scores

Fetches completed match scores from The Odds API, auto-settles pending bets,
and tracks P&L in bankroll.json. Critical for measuring ROI.

API Cost: 1 credit per call (uses daysFrom=3 to catch recent results)

Usage:
    python scripts/results_fetcher.py              # Fetch & settle
    python scripts/results_fetcher.py --history     # Show bet history
    python scripts/results_fetcher.py --roi         # Show ROI summary
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

API_BASE_URL = "https://api.the-odds-api.com/v4"
SERIE_A_KEY = "soccer_italy_serie_a"

# File paths
HISTORY_FILE = DATA_DIR / "betting" / "history.json"
BANKROLL_FILE = DATA_DIR / "betting" / "bankroll.json"
RESULTS_FILE = DATA_DIR / "upcoming" / "results.json"

# Team name normalization — use the canonical mapping from config/team_names.py
from config.team_names import normalize_team
from config.api_keys import get_odds_api_key


def fetch_scores(days_from: int = 3) -> List[Dict]:
    """Fetch completed match scores from Odds API.

    Args:
        days_from: Number of days back to check (1 credit, 2 if daysFrom used)

    Returns:
        List of completed match results.
    """
    if not HAS_REQUESTS:
        log.error("requests library not available")
        return []

    api_key = get_odds_api_key()
    if not api_key:
        log.error("ODDS_API_KEY not set")
        return []

    url = f"{API_BASE_URL}/sports/{SERIE_A_KEY}/scores/"
    params = {
        "apiKey": api_key,
        "daysFrom": days_from,
    }

    try:
        log.info(f"Fetching scores (last {days_from} days)...")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        remaining = int(response.headers.get("x-requests-remaining", 0))
        log.info(f"Fetched {len(data)} events | Credits remaining: {remaining}")

        # Track API usage. results_fetcher uses /scores endpoint; The Odds API bills
        # scores at 1 credit (or 2 with days_from=historical scores). No region multiplier.
        try:
            from scripts.data.odds_fetcher import track_api_call
            est = 2 if days_from else 1
            remaining_int = int(remaining) if remaining not in (None, "?") else None
            track_api_call(
                credits_remaining=remaining_int,
                estimated_cost=est,
                endpoint=f"scores_{'historical' if days_from else 'live'}",
            )
        except ImportError:
            pass

        return data

    except requests.exceptions.HTTPError as e:
        log.error(f"HTTP error fetching scores: {e}")
        return []
    except Exception as e:
        log.error(f"Failed to fetch scores: {e}")
        return []


def parse_scores(raw_scores: List[Dict]) -> Dict[str, Dict]:
    """Parse raw API scores into match results.

    Returns dict mapping "Home vs Away" to result data.
    """
    results = {}

    for event in raw_scores:
        if not event.get("completed"):
            continue

        home_team = normalize_team(event.get("home_team", ""))
        away_team = normalize_team(event.get("away_team", ""))

        if not home_team or not away_team:
            continue

        match_key = f"{home_team} vs {away_team}"

        scores = event.get("scores", [])
        home_score = None
        away_score = None

        for score in scores:
            name = normalize_team(score.get("name", ""))
            if name == home_team:
                home_score = int(score.get("score", 0))
            elif name == away_team:
                away_score = int(score.get("score", 0))

        if home_score is None or away_score is None:
            continue

        # Determine result
        if home_score > away_score:
            result = "HOME"
        elif away_score > home_score:
            result = "AWAY"
        else:
            result = "DRAW"

        results[match_key] = {
            "match": match_key,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "result": result,
            "total_goals": home_score + away_score,
            "btts": home_score > 0 and away_score > 0,
            "commence_time": event.get("commence_time", ""),
            "completed": True,
        }

    return results


def save_results(results: Dict[str, Dict]) -> Path:
    """Save parsed results to file."""
    output = {
        "fetched_at": datetime.now().isoformat(),
        "results": results,
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    log.info(f"Saved {len(results)} results to {RESULTS_FILE}")
    return RESULTS_FILE


def _load_history() -> List[Dict]:
    """Load bet history as a flat list of bet dicts.

    Handles both plain list format and structured dict format.
    Normalizes field names: 'outcome' -> 'status', uppercase -> lowercase.
    """
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE) as f:
                data = json.load(f)
            if isinstance(data, list):
                entries = data
            elif isinstance(data, dict):
                # Handle structured format: {"settled_bets": [...], ...}
                entries = data.get("settled_bets", data.get("bets", []))
            else:
                entries = []
            # Normalize: ensure every entry has lowercase 'status' field
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                if "status" not in entry and "outcome" in entry:
                    entry["status"] = entry["outcome"].lower()
                elif "status" in entry and entry["status"] == entry["status"].upper():
                    entry["status"] = entry["status"].lower()
                # Normalize profit field name
                if "profit" not in entry and "profit_loss" in entry:
                    entry["profit"] = entry["profit_loss"]
            return [e for e in entries if isinstance(e, dict)]
        except Exception as e:
            log.warning(f"Failed to load bet history: {e}")
    return []


def _save_history(history: List[Dict]):
    """Save bet history (atomic write to prevent corruption on crash).

    DEPRECATED (2026-04-23): history.json is now a derived cache regenerated
    from the journal by scripts.betting.ledger.rebuild_caches(). This
    function is kept only as a fallback if the ledger import fails mid-
    settlement. Do not call directly — let _settle_bets_locked do it.
    """
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = HISTORY_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(history, f, indent=2)
    tmp.rename(HISTORY_FILE)


def _load_bankroll() -> Dict:
    """Load bankroll data, ensuring all required keys exist."""
    defaults = {
        "initial_balance": 1000.0,
        "current_balance": 1000.0,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "total_deposited": 1000.0,
        "total_withdrawn": 0.0,
        "peak_balance": 1000.0,
        "lowest_balance": 1000.0,
    }
    if BANKROLL_FILE.exists():
        try:
            with open(BANKROLL_FILE) as f:
                data = json.load(f)
            # Backfill missing keys from defaults
            for k, v in defaults.items():
                if k not in data:
                    data[k] = v
            # Ensure peak/lowest are consistent with current balance
            bal = data["current_balance"]
            data["peak_balance"] = max(data["peak_balance"], bal)
            data["lowest_balance"] = min(data["lowest_balance"], bal)
            return data
        except Exception as e:
            log.warning(f"Failed to load bankroll data: {e}")
    return defaults


def _save_bankroll(bankroll: Dict):
    """Save bankroll data (atomic write to prevent corruption on crash).

    DEPRECATED (2026-04-23): bankroll.json is derived from the journal by
    scripts.betting.ledger.rebuild_caches(). This function is a fallback
    only. The old `peak_balance/lowest_balance` mutations in callers were
    the root cause of drift between journal and bankroll.
    """
    bankroll["updated_at"] = datetime.now().isoformat()
    BANKROLL_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = BANKROLL_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(bankroll, f, indent=2)
    tmp.rename(BANKROLL_FILE)


def _rebuild_bankroll_from_history():  # noqa: D401
    # DEPRECATED (2026-04-23): bankroll.json is now regenerated from the
    # journal via scripts.betting.ledger.rebuild_caches(). This function
    # computes from history.json which is itself a derived cache, making it
    # second-hand and drift-prone. Only called as a fallback if the ledger
    # import fails.
    """Rebuild bankroll.json from history.json — the single source of truth.

    Replays all settled bets chronologically to compute correct balance,
    peak, and lowest values. Called after settlement to prevent drift.
    """
    history = _load_history()
    initial = 1000.0
    if BANKROLL_FILE.exists():
        try:
            with open(BANKROLL_FILE) as f:
                initial = json.load(f).get("initial_balance", 1000.0)
        except Exception:
            pass

    running = initial
    peak = initial
    lowest = initial
    for bet in history:
        status = bet.get("outcome", bet.get("status", "")).upper().lower()
        if status in ("won", "lost", "push"):
            profit = bet.get("profit_loss", bet.get("profit", 0))
            running += profit
            peak = max(peak, running)
            lowest = min(lowest, running)

    # Count pending bets from journal (if available) instead of hardcoding 0
    pending_count = 0
    pending_stakes = 0.0
    try:
        from scripts.betting.bet_journal import get_pending_bets
        pending = get_pending_bets(include_superseded=False)
        pending_count = len(pending)
        pending_stakes = sum(b.get("stake", 0) or 0 for b in pending)
    except Exception:
        pass

    _save_bankroll({
        "initial_balance": initial,
        "current_balance": round(running, 2),
        "peak_balance": round(peak, 2),
        "lowest_balance": round(lowest, 2),
        "pending_bets": pending_count,
        "pending_stakes": round(pending_stakes, 2),
    })


def settle_bets(results: Dict[str, Dict]) -> Dict:
    """Auto-settle pending bets against actual results.

    Reads pending bets from journal first, falls back to unified_report.json.
    Updates history.json, bankroll.json, and journal.

    Uses advisory file locking to prevent concurrent settlement runs from
    double-counting profits.

    Returns settlement summary.
    """
    import fcntl
    _settle_lock_path = DATA_DIR / "betting" / ".settle.lock"
    _settle_lock_path.parent.mkdir(parents=True, exist_ok=True)
    _lock_fd = open(_settle_lock_path, "w")
    try:
        fcntl.flock(_lock_fd, fcntl.LOCK_EX)
        return _settle_bets_locked(results)
    finally:
        fcntl.flock(_lock_fd, fcntl.LOCK_UN)
        _lock_fd.close()


def _settle_bets_locked(results: Dict[str, Dict]) -> Dict:
    """Inner settlement logic — must be called under settle lock."""
    # Try journal first for pending bets
    journal_bets = []
    use_journal = False
    try:
        from scripts.betting.bet_journal import get_pending_bets, settle_bet as journal_settle
        journal_bets = get_pending_bets()
        if journal_bets:
            use_journal = True
            log.info("Loaded %d pending bets from journal", len(journal_bets))
    except Exception as e:
        log.debug("Journal not available: %s", e)

    if use_journal:
        # Map journal bets to the format expected by settlement logic
        bets = []
        # A superseded bet is a pipeline artifact (replaced by a later slip, no
        # stake). The scores API reaches 3 days back, so one older than that can
        # never resolve here: 12 of them from Feb–Apr re-warned "No result found"
        # every 15-min cycle for five months.
        reach_cutoff = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        n_unreachable = 0
        for jb in journal_bets:
            if jb.get("status") == "superseded" and (jb.get("date") or "9999") < reach_cutoff:
                n_unreachable += 1
                continue
            if (jb.get("extra") or {}).get("picks_ref"):
                # a promoted pick (player prop / first-half market): graded by
                # picks.settle_picks with the linked paper entry. This grader
                # defaults an unknown market to "lost" — never let it see one.
                continue
            bets.append({
                "match": jb.get("match", ""),
                "date": jb.get("date", ""),
                "market": jb.get("market", ""),
                "selection": jb.get("selection", ""),
                "odds": jb.get("odds", 1.0),
                "stake": jb.get("stake", 0.0),
                "confidence": jb.get("confidence", "MEDIUM"),
                "value_pct": jb.get("edge_pct", 0),
                "_bet_id": jb.get("bet_id"),  # carry through for journal update
            })
        if n_unreachable:
            log.debug("Skipping %d superseded bets older than %s (no stake, out of the API's reach)",
                      n_unreachable, reach_cutoff)
    else:
        # The journal is the ONLY settlement source (CLAUDE.md ledger rules).
        # The old fallback settled bets out of data/betting/unified_report.json,
        # a report that froze in February — removed 2026-08-31.
        log.warning("Settlement without the journal is not supported — nothing settled")
        return {"settled": 0, "pending": 0}
    # Dedup: if pipeline re-ran and placed multiple bets on same match+market,
    # only settle the newest one (by bet_id / placed_at) to prevent double P&L.
    if use_journal and len(bets) > 1:
        from collections import defaultdict
        _match_market_groups = defaultdict(list)
        for _b in bets:
            _key = f"{_b['match']}_{_b.get('market', 'h2h')}"
            _match_market_groups[_key].append(_b)
        deduped = []
        for _key, _group in _match_market_groups.items():
            if len(_group) > 1:
                # Keep newest by bet_id (lexicographically, includes date prefix)
                _newest = max(_group, key=lambda b: str(b.get("_bet_id", "") or ""))
                deduped.append(_newest)
                log.info("Dedup: kept %s, skipped %d older bets for %s",
                         _newest.get("_bet_id"), len(_group) - 1, _key)
            else:
                deduped.append(_group[0])
        bets = deduped

    history = _load_history()
    bankroll = _load_bankroll()

    settled_count = 0
    won_count = 0
    push_count = 0
    total_profit = 0.0

    # Track already-settled bets to avoid double settlement.
    # Key includes selection to avoid collapsing different bets on same match+market
    # (e.g., Over 2.5 and Under 2.5 on the same match).
    # Normalize: uppercase selection, h2h → 1X2 to prevent case/market dupes.
    def _norm_market(m):
        return "1X2" if m.upper() in ("H2H", "1X2", "MONEYLINE") else m

    settled_matches = {
        h.get("match", "") + "_" + _norm_market(h.get("market", "h2h")) + "_" + h.get("selection", "").upper()
        for h in history if h.get("status") in ("won", "lost", "push")
    }

    for bet in bets:
        match_key = bet.get("match", "")
        market = bet.get("market", "h2h")
        selection = bet.get("selection", "")
        bet_id = f"{match_key}_{_norm_market(market)}_{selection.upper()}"

        if bet_id in settled_matches:
            continue

        result = results.get(match_key)
        # Fuzzy match: try normalized team names if exact match fails
        if not result:
            _TEAM_ALIASES = {
                "AC Milan": "Milan", "Milan": "AC Milan",
                "Hellas Verona": "Verona", "Verona": "Hellas Verona",
                "AS Roma": "Roma", "Roma": "AS Roma",
                "SPAL 2013": "SPAL", "SPAL": "SPAL 2013",
                "Parma Calcio 1913": "Parma", "Parma": "Parma Calcio 1913",
            }
            for rk, rv in results.items():
                # Try swapping team aliases in the result key
                rk_norm = rk
                for alias_from, alias_to in _TEAM_ALIASES.items():
                    rk_norm = rk_norm.replace(alias_from, alias_to)
                if rk_norm == match_key or rk == match_key.replace(" vs ", " - "):
                    result = rv
                    log.info("Fuzzy match: %s → %s", match_key, rk)
                    break
        if not result:
            # Log warning for past matches not found in results
            bet_date = bet.get("date", "")
            if bet_date and bet_date <= datetime.now().strftime("%Y-%m-%d"):
                log.warning("No result found for past bet: %s (date: %s, market: %s)",
                           match_key, bet_date, market)
            continue

        # Determine outcome: "won", "lost", or "push"
        selection = bet.get("selection", "")
        odds = bet.get("odds", 1.0)
        stake = bet.get("stake", 0.0)
        outcome = "lost"  # default

        # Handle edge cases: postponed, abandoned, walkover matches
        match_status = result.get("status", "").lower()
        if match_status in ("postponed", "cancelled", "suspended"):
            outcome = "void"
            log.info("Match %s was %s — voiding bet", match_key, match_status)
        elif match_status == "abandoned":
            # Most bookmakers void if abandoned before 70-75 minutes
            minutes_played = result.get("minutes_played", 0)
            if minutes_played < 70:
                outcome = "void"
                log.info("Match %s abandoned at %d' — voiding bet", match_key, minutes_played)
            else:
                log.info("Match %s abandoned at %d' — settling on score", match_key, minutes_played)
        elif match_status == "walkover":
            outcome = "void"
            log.info("Match %s was walkover — voiding bet", match_key)

        sel_upper = selection.upper().strip()
        mkt_upper = market.upper().strip()
        home_score = result.get("home_score", 0)
        away_score = result.get("away_score", 0)
        total_goals = result.get("total_goals", home_score + away_score)
        btts = result.get("btts", home_score > 0 and away_score > 0)
        match_result = result.get("result", "")  # "HOME", "DRAW", "AWAY"

        # Skip market evaluation for voided matches
        if outcome == "void":
            pass
        # ── 1X2 / h2h ──
        elif mkt_upper in ("H2H", "1X2"):
            if sel_upper in ("HOME", "1"):
                outcome = "won" if home_score > away_score else "lost"
            elif sel_upper in ("DRAW", "X"):
                outcome = "won" if home_score == away_score else "lost"
            elif sel_upper in ("AWAY", "2"):
                outcome = "won" if away_score > home_score else "lost"
            elif selection == match_result:
                outcome = "won"

        # ── Totals / O/U ──
        elif mkt_upper.startswith("O/U") or mkt_upper == "TOTALS":
            if "OVER" in sel_upper:
                line = float(sel_upper.split()[-1]) if len(sel_upper.split()) > 1 else 2.5
                if total_goals > line:
                    outcome = "won"
                elif total_goals == line:
                    outcome = "push"
            elif "UNDER" in sel_upper:
                line = float(sel_upper.split()[-1]) if len(sel_upper.split()) > 1 else 2.5
                if total_goals < line:
                    outcome = "won"
                elif total_goals == line:
                    outcome = "push"

        # ── BTTS ──
        elif mkt_upper == "BTTS":
            if "YES" in sel_upper:
                outcome = "won" if btts else "lost"
            else:
                outcome = "won" if not btts else "lost"

        # ── Asian Handicap / Spreads ──
        elif mkt_upper.startswith("AH") or mkt_upper in ("SPREADS", "HANDICAP"):
            parts = sel_upper.split()
            side = parts[0] if parts else ""
            line = 0.0
            if len(parts) > 1:
                try:
                    line = float(parts[-1])
                except ValueError:
                    line = 0.0
            goal_diff = home_score - away_score
            if side == "AWAY":
                adjusted_diff = -goal_diff + line
            else:
                adjusted_diff = goal_diff + line
            if adjusted_diff > 0:
                outcome = "won"
            elif adjusted_diff == 0:
                outcome = "push"
            else:
                outcome = "lost"

        # ── Double Chance ──
        elif mkt_upper == "DC":
            if "1X" in sel_upper or "HOME OR DRAW" in sel_upper:
                outcome = "won" if home_score >= away_score else "lost"
            elif "X2" in sel_upper or "DRAW OR AWAY" in sel_upper:
                outcome = "won" if away_score >= home_score else "lost"
            elif "12" in sel_upper or "HOME OR AWAY" in sel_upper:
                outcome = "won" if home_score != away_score else "lost"

        # ── Draw No Bet ──
        elif mkt_upper == "DNB":
            if "HOME" in sel_upper:
                if home_score > away_score:
                    outcome = "won"
                elif home_score == away_score:
                    outcome = "push"
                else:
                    outcome = "lost"
            elif "AWAY" in sel_upper:
                if away_score > home_score:
                    outcome = "won"
                elif home_score == away_score:
                    outcome = "push"
                else:
                    outcome = "lost"

        # Calculate profit based on outcome
        if outcome == "void":
            profit = 0.0  # stake refunded, match not played/completed
        elif outcome == "won":
            profit = stake * (odds - 1)
        elif outcome == "push":
            profit = 0.0  # stake refunded, no gain
        else:
            profit = -stake

        # Record in history
        # match_kickoff_at = actual kickoff time from Odds API (audit field for
        # downstream sort: bets are grouped by match, not by grading batch).
        # Delegated to ledger.validate_commence_time so every settler uses the
        # same staleness guard (single place to tune the 365-day threshold).
        from scripts.betting.ledger import validate_commence_time
        raw_commence = result.get("commence_time", "")
        date_for_history, commence_iso = validate_commence_time(
            raw_commence, fallback_date=bet.get("date", "")
        )
        commence_iso = commence_iso or ""

        history_entry = {
            "match": match_key,
            "date": date_for_history,
            "market": market,
            "selection": selection,
            "odds": odds,
            "stake": round(stake, 2),
            "status": outcome,
            "profit": round(profit, 2),
            "result": f"{result['home_score']}-{result['away_score']}",
            "settled_at": datetime.now().isoformat(),
            "match_kickoff_at": commence_iso or None,
            "confidence": bet.get("confidence", "MEDIUM"),
            "value_pct": bet.get("value_pct", 0),
        }
        history.append(history_entry)

        # Update journal if available
        if use_journal and bet.get("_bet_id"):
            try:
                journal_settle(
                    bet["_bet_id"],
                    status=outcome,
                    result_score=f"{result['home_score']}-{result['away_score']}",
                    profit=round(profit, 2),
                    match_kickoff_at=commence_iso or None,
                )
            except Exception as e:
                log.error("Failed to update journal for bet %s: %s", bet.get("_bet_id"), e)

        # Update bankroll
        bankroll["current_balance"] = round(bankroll["current_balance"] + profit, 2)
        bankroll["peak_balance"] = max(bankroll["peak_balance"], bankroll["current_balance"])
        bankroll["lowest_balance"] = min(bankroll["lowest_balance"], bankroll["current_balance"])

        settled_count += 1
        total_profit += profit
        if outcome == "won":
            won_count += 1
        elif outcome == "push":
            push_count += 1

        status_emoji = "W" if outcome == "won" else "P" if outcome == "push" else "L"
        log.info(f"  [{status_emoji}] {match_key} | {selection} @ {odds:.2f} | "
                f"{'+'if profit > 0 else ''}{profit:.2f}")

    # Delegate ALL cache writes to the ledger — single source of truth.
    # The ledger reads bet_journal.json (updated per-bet above via journal_settle)
    # and atomically regenerates bankroll.json + history.json from it.
    # This replaces the old triple-write path (history.append + _save_history
    # + _rebuild_bankroll_from_history + bankroll_loader.update_bankroll_json)
    # which was the root cause of bankroll.json drift.
    try:
        from scripts.betting import ledger
        ledger.rebuild_caches()
    except Exception as e:
        log.error("ledger cache rebuild failed: %s", e)
        # Fallback: preserve old behavior so we at least have *something*
        _save_history(history)
        _rebuild_bankroll_from_history()
    # Reload bankroll so summary has the rebuilt (accurate) balance
    bankroll = _load_bankroll()

    lost_count = settled_count - won_count - push_count
    summary = {
        "settled": settled_count,
        "won": won_count,
        "lost": lost_count,
        "push": push_count,
        "profit": round(total_profit, 2),
        "new_balance": bankroll["current_balance"],
        "pending": len(bets) - settled_count,
    }

    log.info(f"Settlement: {settled_count} bets settled | "
            f"{won_count}W {lost_count}L {push_count}P | "
            f"P&L: {'+'if total_profit > 0 else ''}{total_profit:.2f} | "
            f"Balance: {bankroll['current_balance']:.2f}")

    return summary


def get_roi_summary() -> Dict:
    """Calculate comprehensive ROI summary from bet history."""
    history = _load_history()
    bankroll = _load_bankroll()

    if not history:
        return {"message": "No bet history yet"}

    total_bets = len(history)
    wins = sum(1 for h in history if h.get("status") == "won")
    losses = sum(1 for h in history if h.get("status") == "lost")
    pushes = sum(1 for h in history if h.get("status") == "push")
    total_staked = sum(h.get("stake", 0) for h in history)
    total_profit = sum(h.get("profit", 0) for h in history)
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
    # Win rate excludes pushes (they don't count as wins or losses)
    decided = wins + losses
    win_rate = round(wins / decided * 100, 1) if decided > 0 else 0

    # By market
    by_market = {}
    for h in history:
        m = h.get("market", "h2h")
        if m not in by_market:
            by_market[m] = {"bets": 0, "wins": 0, "losses": 0, "pushes": 0, "profit": 0, "staked": 0}
        by_market[m]["bets"] += 1
        by_market[m]["staked"] += h.get("stake", 0)
        by_market[m]["profit"] += h.get("profit", 0)
        if h.get("status") == "won":
            by_market[m]["wins"] += 1
        elif h.get("status") == "push":
            by_market[m]["pushes"] += 1
        else:
            by_market[m]["losses"] += 1

    for m, data in by_market.items():
        data["roi"] = round(data["profit"] / data["staked"] * 100, 1) if data["staked"] > 0 else 0

    return {
        "total_bets": total_bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": win_rate,
        "total_staked": round(total_staked, 2),
        "total_profit": round(total_profit, 2),
        "roi": round(roi, 1),
        "current_balance": bankroll["current_balance"],
        "initial_balance": bankroll["initial_balance"],
        "by_market": by_market,
    }


def fetch_and_settle() -> Dict:
    """Main function: fetch results and auto-settle bets.

    Returns settlement summary.
    """
    log.info("=" * 60)
    log.info("AUTO-RESULTS FETCHER")
    log.info("=" * 60)

    # Warn about orphaned bets that the API can't reach (daysFrom max is 3)
    try:
        from scripts.betting.bet_journal import get_pending_bets
        pending = get_pending_bets(include_superseded=False)
        cutoff = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        orphaned = [b for b in pending
                    if b.get("date", "9999") < cutoff]
        if orphaned:
            matches = [f"{b['match']} ({b['date']})" for b in orphaned[:5]]
            log.warning("Found %d orphaned bets older than 3 days (API can't reach): %s",
                       len(orphaned), ", ".join(matches))
            log.warning("These bets need manual settlement or a cached results lookup")
    except Exception:
        pass

    raw_scores = fetch_scores(days_from=3)
    if not raw_scores:
        log.warning("No scores data received")
        return {"settled": 0, "message": "No scores available"}

    # Parse results
    results = parse_scores(raw_scores)
    completed = {k: v for k, v in results.items() if v.get("completed")}
    log.info(f"Found {len(completed)} completed matches")

    if not completed:
        return {"settled": 0, "message": "No completed matches"}

    # Save results
    save_results(results)

    # Settle bets
    summary = settle_bets(completed)

    return summary


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Auto-Results Fetcher")
    parser.add_argument("--history", action="store_true", help="Show bet history")
    parser.add_argument("--roi", action="store_true", help="Show ROI summary")
    args = parser.parse_args()

    if args.history:
        history = _load_history()
        if not history:
            print("No bet history yet")
        else:
            print(f"\nBet History ({len(history)} bets):")
            print(f"{'Match':<25} {'Selection':>10} {'Odds':>6} {'Stake':>7} {'Result':>7} {'P&L':>8}")
            print("-" * 70)
            for h in history[-20:]:  # Show last 20
                s = h.get("status", "lost")
                status = "W" if s == "won" else "P" if s == "push" else "L"
                profit = h.get("profit", 0)
                print(f"  {h.get('match', '?'):<23} {h.get('selection', ''):>10} "
                      f"{h.get('odds', 0):>6.2f} ${h.get('stake', 0):>6.2f} "
                      f"  [{status}]  {'+'if profit > 0 else ''}{profit:>7.2f}")

    elif args.roi:
        roi = get_roi_summary()
        if "message" in roi:
            print(roi["message"])
        else:
            print(f"\n{'='*50}")
            print(f" ROI SUMMARY")
            print(f"{'='*50}")
            print(f"  Total Bets: {roi['total_bets']}")
            pushes = roi.get('pushes', 0)
            push_str = f"-{pushes}P" if pushes else ""
            print(f"  Record: {roi['wins']}W-{roi['losses']}L{push_str} ({roi['win_rate']}%)")
            print(f"  Total Staked: ${roi['total_staked']:.2f}")
            print(f"  Total Profit: {'+'if roi['total_profit'] > 0 else ''}${roi['total_profit']:.2f}")
            print(f"  ROI: {roi['roi']:+.1f}%")
            print(f"  Balance: ${roi['current_balance']:.2f} (started ${roi['initial_balance']:.2f})")

            if roi["by_market"]:
                print(f"\n  By Market:")
                for m, data in roi["by_market"].items():
                    print(f"    {m}: {data['bets']} bets, {data['wins']}W, "
                          f"ROI: {data['roi']:+.1f}%")

    else:
        summary = fetch_and_settle()
        print(f"\nSettlement: {summary.get('settled', 0)} bets settled")
        if summary.get("settled", 0) > 0:
            print(f"  Won: {summary['won']}, Lost: {summary['lost']}")
            print(f"  P&L: {'+'if summary['profit'] > 0 else ''}${summary['profit']:.2f}")
            print(f"  New Balance: ${summary['new_balance']:.2f}")
