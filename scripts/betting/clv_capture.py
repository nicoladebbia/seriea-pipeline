#!/usr/bin/env python3
"""CLV AUTO-CAPTURE — Capture closing odds at kickoff and compute CLV.

CLV (Closing Line Value) = (bet_odds / closing_odds) - 1
Positive CLV means you consistently beat the closing line, which is the
strongest predictor of long-term profitability.

Flow:
  1. Load pending bets from journal
  2. Fetch current odds snapshot (Odds API)
  3. For each pending bet, find matching market odds
  4. Store as closing_odds + compute clv_pct
  5. Update journal

Runs on every pre-kickoff cycle and REPLACES the stored price each time while
kickoff is still ahead, so what ends up in the journal is the last pre-kickoff
quote — the close. Once kickoff has passed the stored price is frozen; a bet
that was never captured before kickoff still takes a post-match price as a
last-resort proxy.

API Cost: ~4 credits (same as odds snapshot)

Usage:
    python -m scripts.betting.clv_capture              # Capture CLV for pending bets
    python -m scripts.betting.clv_capture --dry-run    # Show matches without writing
    python -m scripts.betting.clv_capture --from-cache # Use cached odds (no API call)
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR, atomic_write_json
from scripts.utils.match_timing import now_utc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Sharp bookmakers (used for closing line if available)
SHARP_BOOKMAKERS = {"Pinnacle", "Pinnacle Sports", "BetCRIS", "CRIS", "Matchbook"}

# Cached odds file candidates (in priority order)
_ODDS_FILES = [
    "odds_full.json",       # Primary: unified odds with all bookmakers
    "odds_data.json",       # Legacy format
    "odds.json",            # Minimal format
]


def _load_cached_odds() -> Dict:
    """Load odds from cached files and merge extra markets.

    odds_full.json has: {"matches": {"Match Key": {h2h, totals, spreads}}}
    odds_extra_markets.json has: {"matches": {"Match Key": {btts, double_chance, draw_no_bet, ...}}}

    We merge both so CLV can be captured for all market types.
    """
    odds = {}

    # Load primary odds (h2h, totals, spreads)
    for fname in _ODDS_FILES:
        path = DATA_DIR / "upcoming" / fname
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)

                if "matches" in data and isinstance(data["matches"], dict):
                    odds = data["matches"]
                elif isinstance(data, dict) and any(
                    isinstance(v, dict) and "h2h" in v for v in data.values()
                ):
                    odds = data
                else:
                    continue

                log.info("Loaded primary odds: %s (%d matches)", fname, len(odds))
                break
            except Exception as e:
                log.warning("Failed to load %s: %s", fname, e)
                continue

    # Merge extra markets (DC, BTTS, DNB, alternate spreads/totals) — all leagues
    extra_paths = [DATA_DIR / "upcoming" / "odds_extra_markets.json"] + \
        list((DATA_DIR / "upcoming").glob("odds_extra_markets_*.json"))
    for extra_path in extra_paths:
        if not extra_path.exists():
            continue
        try:
            with open(extra_path) as f:
                extra = json.load(f)
            extra_matches = extra.get("matches", {})
            merged = 0
            for match_key, extra_data in extra_matches.items():
                if match_key not in odds:
                    odds[match_key] = {}
                for mkt_key in ("btts", "double_chance", "draw_no_bet",
                                "alternate_totals", "alternate_spreads",
                                "team_totals_raw"):
                    if mkt_key in extra_data:
                        odds[match_key][mkt_key] = extra_data[mkt_key]
                        merged += 1
            log.info("Merged extra markets from %s: %d entries from %d matches",
                     extra_path.name, merged, len(extra_matches))
        except Exception as e:
            log.warning("Failed to load extra markets from %s: %s", extra_path.name, e)

    if not odds:
        log.warning("No cached odds found")
    return odds


def _find_sharp_odds(bookmakers: List[Dict], selection_key: str) -> Tuple[float, str] | None:
    """Find sharp bookmaker odds for a selection, falling back to market average.

    Args:
        bookmakers: List of {"bookmaker": str, "home": float, "draw": float, "away": float, ...}
        selection_key: Key to look up, e.g. "home", "draw", "away", "over", "under"

    Returns:
        (odds, book) — the sharp book's price and its name, else the market mean
        tagged "market_mean". The NAME is load-bearing: a mean of many books is a
        different reference from a sharp quote, and comparing one against the other
        would read as line movement when nothing moved. The journal stores it so
        `clv_move_pct` can refuse the mixed comparison.
    """
    # Try sharp bookmakers first
    for bm in bookmakers:
        if bm.get("bookmaker") in SHARP_BOOKMAKERS:
            val = bm.get(selection_key, 0)
            if val and val > 1.0:
                return val, bm["bookmaker"]

    # Fall back to market average
    vals = [bm.get(selection_key, 0) for bm in bookmakers if bm.get(selection_key, 0) > 1.0]
    if vals:
        return round(sum(vals) / len(vals), 3), "market_mean"

    return None


def _line_key(line: float) -> str:
    """alternate_totals is keyed by the line as it was written: "1.5", "2.0"."""
    return str(int(line)) if float(line).is_integer() else str(line)


def _match_entry(bet: Dict, odds_data: Dict) -> Dict | None:
    """The odds entry for this bet's fixture, or None."""
    match_key = bet.get("match", "")
    entry = odds_data.get(match_key)
    if not entry:
        parts = match_key.split(" vs ")
        if len(parts) == 2:
            entry = odds_data.get(f"{parts[1]} vs {parts[0]}")
    return entry or None


def _is_pre_kickoff(entry: Dict) -> bool:
    """True only when this entry's kickoff is provably still ahead of us.

    A price read after kickoff is not a closing price — it is an in-play or
    stale quote. Unknown or unparseable `commence_time` reads as NOT pre-kickoff
    so an unknown-vintage price can never overwrite a stored close (fail closed).
    """
    raw = (entry or {}).get("commence_time")
    if not raw:
        return False
    try:
        ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts > now_utc()


def _match_bet_to_odds(bet: Dict, odds_data: Dict) -> Tuple[float, str] | None:
    """Match a pending bet to current odds.

    Returns (closing_odds, source_description) or None if no match.
    """
    market = bet.get("market", "").upper()
    selection = bet.get("selection", "").upper()

    match_odds = _match_entry(bet, odds_data)
    if not match_odds:
        return None

    # ── 1X2 ──
    if market in ("1X2", "H2H"):
        h2h = match_odds.get("h2h", {})
        bookmakers = h2h.get("all_bookmakers", [])

        if "HOME" in selection or selection == "1":
            sel_key = "home"
        elif "DRAW" in selection or selection == "X":
            sel_key = "draw"
        elif "AWAY" in selection or selection == "2":
            sel_key = "away"
        else:
            return None

        # Use sharp bookmaker or average
        sharp = _find_sharp_odds(bookmakers, sel_key)
        if sharp:
            closing, book = sharp
            return closing, f"h2h.{sel_key} ({len(bookmakers)} bookmakers) [{book}]"

        # Fall back to the summary field
        fallback = h2h.get(sel_key) or h2h.get(f"best_{sel_key}")
        if fallback and fallback > 1.0:
            return fallback, f"h2h.{sel_key} (summary)"

    # ── O/U (totals) ──
    elif "O/U" in market or "OVER" in selection or "UNDER" in selection:
        totals = match_odds.get("totals", [])

        # Find the right line (default 2.5)
        line = 2.5
        # Try to extract line from market name, e.g. "O/U 2.5", "O/U 1.5"
        for part in market.split():
            try:
                line = float(part)
                break
            except ValueError:
                continue
        # Also check selection for line
        for part in selection.split():
            try:
                line = float(part)
                break
            except ValueError:
                continue

        # The bulk feed carries only the headline lines (2.0 / 2.25 / 2.5). O/U 1.5
        # — the line this system bets most — lives in alternate_totals, so a capture
        # that reads `totals` alone never matches the money market at all.
        if not any(abs(t.get("line", 0) - line) < 0.01 for t in totals):
            alt = (match_odds.get("alternate_totals") or {}).get(_line_key(line))
            if alt:
                side = "over" if "OVER" in selection else "under" if "UNDER" in selection else None
                if side:
                    sharp = _find_sharp_odds(alt.get("all_bookmakers", []), side)
                    if sharp:
                        closing, book = sharp
                        n = len(alt.get("all_bookmakers", []))
                        return closing, f"alt_totals.{line}.{side} ({n} bm) [{book}]"
                    # the MEAN across books, never best_*: a max over N books is
                    # an extreme, not a line, and comparing our taken price to the
                    # best price available at the close makes CLV negative by
                    # construction
                    fallback = alt.get(side) or alt.get(f"best_{side}")
                    if fallback and fallback > 1.0:
                        return fallback, f"alt_totals.{line}.{side} (summary)"
            return None

        for total in totals:
            if abs(total.get("line", 0) - line) < 0.01:
                bookmakers = total.get("all_bookmakers", [])
                if "OVER" in selection:
                    sharp = _find_sharp_odds(bookmakers, "over")
                    if sharp:
                        closing, book = sharp
                        return closing, f"totals.{line}.over ({len(bookmakers)} bm) [{book}]"
                    fallback = total.get("over") or total.get("best_over")
                    if fallback and fallback > 1.0:
                        return fallback, f"totals.{line}.over (summary)"
                elif "UNDER" in selection:
                    sharp = _find_sharp_odds(bookmakers, "under")
                    if sharp:
                        closing, book = sharp
                        return closing, f"totals.{line}.under ({len(bookmakers)} bm) [{book}]"
                    fallback = total.get("under") or total.get("best_under")
                    if fallback and fallback > 1.0:
                        return fallback, f"totals.{line}.under (summary)"
                break

    # ── DC (double chance) ──
    elif market == "DC":
        dc = match_odds.get("double_chance", {})
        if "1X" in selection or "HOME OR DRAW" in selection:
            dc_key = "1X"
        elif "X2" in selection or "DRAW OR AWAY" in selection:
            dc_key = "X2"
        elif "12" in selection or "HOME OR AWAY" in selection:
            dc_key = "12"
        else:
            return None

        dc_entry = dc.get(dc_key, {})

        # Try sharp bookmaker from all_bookmakers (uses "odds" key, not "home"/"draw")
        bms = dc_entry.get("all_bookmakers", [])
        sharp = _find_sharp_odds(bms, "odds")
        if sharp:
            closing, book = sharp
            return closing, f"dc.{dc_key} ({len(bms)} bookmakers) [{book}]"

        best = dc_entry.get("best")
        if best and best > 1.0:
            return best, f"dc.{dc_key} (best)"
        avg = dc_entry.get("avg")
        if avg and avg > 1.0:
            return avg, f"dc.{dc_key} (avg)"

    # ── DNB ──
    elif market == "DNB":
        dnb = match_odds.get("draw_no_bet", {})
        if "HOME" in selection:
            val = dnb.get("best_home") or dnb.get("home")
        elif "AWAY" in selection:
            val = dnb.get("best_away") or dnb.get("away")
        else:
            return None
        if val and val > 1.0:
            return val, f"dnb.{selection}"

    # ── BTTS ──
    elif market == "BTTS":
        btts = match_odds.get("btts", {})
        if "YES" in selection:
            val = btts.get("best_yes") or btts.get("yes")
        else:
            val = btts.get("best_no") or btts.get("no")
        if val and val > 1.0:
            return val, f"btts.{selection}"

    # ── AH (spreads) ──
    elif "AH" in market or "SPREAD" in market:
        spreads = match_odds.get("spreads", [])
        if not spreads:
            return None

        # Parse the target line from the market name (e.g. "AH 1.75", "AH -0.25")
        target_line = None
        for part in market.split():
            try:
                target_line = abs(float(part))
                break
            except ValueError:
                continue
        # Also try from selection (e.g. "Home +1.8", "Away +0.2")
        if target_line is None:
            for part in selection.replace("+", "").replace("-", "").split():
                try:
                    target_line = float(part)
                    break
                except ValueError:
                    continue

        # Find the closest matching line
        if target_line is not None:
            best_spread = min(spreads, key=lambda s: abs(s.get("line", 0) - target_line))
            if abs(best_spread.get("line", 0) - target_line) <= 0.5:
                bookmakers = best_spread.get("all_bookmakers", [])
                if "HOME" in selection:
                    sharp = _find_sharp_odds(bookmakers, "home")
                    if sharp:
                        closing, book = sharp
                        return closing, f"spreads.{best_spread['line']}.home ({len(bookmakers)} bm) [{book}]"
                elif "AWAY" in selection:
                    sharp = _find_sharp_odds(bookmakers, "away")
                    if sharp:
                        closing, book = sharp
                        return closing, f"spreads.{best_spread['line']}.away ({len(bookmakers)} bm) [{book}]"
        else:
            # No line parsed — try first available spread
            spread = spreads[0]
            bookmakers = spread.get("all_bookmakers", [])
            if "HOME" in selection:
                sharp = _find_sharp_odds(bookmakers, "home")
                if sharp:
                    closing, book = sharp
                    return closing, f"spreads.{spread.get('line')}.home ({len(bookmakers)} bm) [{book}]"
            elif "AWAY" in selection:
                sharp = _find_sharp_odds(bookmakers, "away")
                if sharp:
                    closing, book = sharp
                    return closing, f"spreads.{spread.get('line')}.away ({len(bookmakers)} bm) [{book}]"

    return None


def capture_clv(dry_run: bool = False, from_cache: bool = False) -> Dict:
    """Main CLV capture flow.

    Returns summary: {captured, skipped, no_match, errors}
    """
    from scripts.betting.bet_journal import get_pending_bets, update_clv

    # Step 1: Get pending bets
    pending = get_pending_bets()
    if not pending:
        log.info("No pending bets — nothing to capture CLV for")
        return {"captured": 0, "pending": 0}

    log.info("Found %d pending bets", len(pending))

    # Step 2: Fetch current odds
    odds_data = {}
    if from_cache:
        odds_data = _load_cached_odds()
        if not odds_data:
            return {"captured": 0, "error": "no_cached_odds"}
    else:
        try:
            from scripts.data.odds_fetcher import fetch_snapshot
            odds_data = fetch_snapshot()
            log.info("Fetched fresh odds snapshot (%d matches)", len(odds_data))
        except Exception as e:
            log.error("Failed to fetch odds: %s — falling back to cache", e)
            odds_data = _load_cached_odds()
            if not odds_data:
                return {"captured": 0, "error": str(e)}

    if not odds_data:
        return {"captured": 0, "error": "no_odds_data"}

    # Step 3: Match bets to odds and capture CLV
    captured = 0
    skipped = 0
    no_match = 0

    for bet in pending:
        bet_id = bet.get("bet_id", "?")
        bet_odds = bet.get("odds", 0)

        result = _match_bet_to_odds(bet, odds_data)
        if result is None:
            no_match += 1
            log.debug("No match for bet %s (%s %s)",
                      bet_id, bet.get("match"), bet.get("selection"))
            continue

        # The CLOSING line is the LAST price before kickoff, not the first one we
        # happened to see. This loop used to skip any bet that already carried a
        # clv_pct, so a bet journalled at T-30 had its "close" read from the very
        # snapshot it was priced on: 58 of 172 real rows ended up with
        # closing_odds EXACTLY equal to their own entry sharp price, and the line
        # movement the stake ladder gates on was structurally zero on precisely
        # the T-30 bets the design wants. So: overwrite while kickoff is still
        # ahead, and freeze the stored price once it has passed.
        if bet.get("closing_odds") is not None and not _is_pre_kickoff(
            _match_entry(bet, odds_data) or {}
        ):
            skipped += 1
            continue

        closing_odds, source = result
        clv_pct = round(((bet_odds / closing_odds) - 1.0) * 100, 2) if closing_odds > 1.0 else 0.0

        if dry_run:
            log.info("DRY RUN: %s | %s %s | bet=%.2f close=%.2f CLV=%+.1f%% [%s]",
                     bet.get("match"), bet.get("market"), bet.get("selection"),
                     bet_odds, closing_odds, clv_pct, source)
        else:
            update_clv(bet_id, closing_odds=closing_odds, clv_pct=clv_pct, closing_source=source)
            log.info("CLV captured: %s | %s %s | bet=%.2f close=%.2f CLV=%+.1f%% [%s]",
                     bet.get("match"), bet.get("market"), bet.get("selection"),
                     bet_odds, closing_odds, clv_pct, source)

        captured += 1

    # Save CLV history
    if captured > 0 and not dry_run:
        _append_clv_history(captured, pending)

    summary = {
        "captured": captured,
        "skipped": skipped,
        "no_match": no_match,
        "total_pending": len(pending),
        "dry_run": dry_run,
        "timestamp": now_utc().isoformat(),
    }

    log.info("CLV capture: %d captured, %d frozen (kickoff passed), %d no match",
             captured, skipped, no_match)
    return summary


def _append_clv_history(n_captured: int, bets: List[Dict]):
    """Append CLV capture event to history for tracking over time."""
    history_path = DATA_DIR / "betting" / "clv_history.json"

    history = {"captures": [], "bets": []}
    if history_path.exists():
        try:
            with open(history_path) as f:
                existing = json.load(f)
            if isinstance(existing, dict):
                history = existing
                if "captures" not in history:
                    history["captures"] = []
            elif isinstance(existing, list):
                history["captures"] = existing
        except Exception:
            pass

    history["captures"].append({
        "timestamp": now_utc().isoformat(),
        "captured": n_captured,
        "bets_count": len(bets),
    })

    atomic_write_json(history_path, history, indent=2)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Capture CLV for pending bets")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show matches without writing")
    parser.add_argument("--from-cache", action="store_true",
                        help="Use cached odds instead of fetching fresh")
    args = parser.parse_args()

    summary = capture_clv(dry_run=args.dry_run, from_cache=args.from_cache)
    print(json.dumps(summary, indent=2))
