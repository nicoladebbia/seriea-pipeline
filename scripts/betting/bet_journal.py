"""Unified Bet Journal — single source of truth for all bets.

Stores bets from placement through settlement in a single JSON file
at data/betting/bet_journal.json. Replaces the fragmented 5-file system.

Schema per bet:
    bet_id, match, date, market, selection,
    model_prob, sharp_implied_prob, edge_pct,
    odds, bookmaker, avg_odds, pinnacle_odds,
    stake, confidence, factors,
    status (pending/won/lost/push/void),
    result_score, profit,
    closing_odds, clv_pct,
    placed_at, settled_at

Usage:
    python scripts/bet_journal.py pending       # List pending bets
    python scripts/bet_journal.py settled       # List settled bets
    python scripts/bet_journal.py stats         # P&L summary
    python scripts/bet_journal.py report        # Full weekly report
    python scripts/bet_journal.py migrate       # One-time migration from legacy
    python scripts/bet_journal.py settle        # Fetch results + auto-settle
"""

import contextlib
import fcntl
import json
import logging
import shutil
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# Canonical settled statuses. On-disk data uses "voided" (past tense); we
# accept "void" as a legacy alias on write and canonicalize to "voided".
_SETTLED_STATUSES = ("won", "lost", "push", "voided")
_VALID_SETTLE_INPUTS = ("won", "lost", "push", "void", "voided")


def _canon_settle_status(s: str) -> str:
    """Normalize 'void' -> 'voided' so the journal has one spelling on disk."""
    return "voided" if s == "void" else s

# Resolve paths relative to project root
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
DATA_DIR = _PROJECT_ROOT / "data"
JOURNAL_PATH = DATA_DIR / "betting" / "bet_journal.json"
# PAPER track: a gated league's candidates are journaled here with flat
# stakes so it can earn its deployment bar (50+ settled, CLV+) without a
# cent of real exposure. Deliberately a SEPARATE file: bankroll.json,
# history.json, risk state and every ROI/drift monitor derive from
# JOURNAL_PATH only, so paper entries can never contaminate real P&L.
PAPER_JOURNAL_PATH = DATA_DIR / "betting" / "paper_journal.json"

# Legacy file paths for migration
LEGACY_FILES = {
    "placed_bets_log": DATA_DIR / "betting" / "placed_bets_log.json",
    "placed_bets": DATA_DIR / "betting" / "placed_bets.json",
    "bet_history": DATA_DIR / "upcoming" / "bet_history.json",
    "history": DATA_DIR / "betting" / "history.json",
    "clv_history": DATA_DIR / "betting" / "clv_history.json",
}


# =============================================================================
# FILE LOCKING — prevents race conditions between pipeline and auto_settle
# =============================================================================

_JOURNAL_LOCK_PATH = DATA_DIR / "betting" / ".journal.lock"


def _with_journal_lock(func):
    """Decorator: acquire advisory file lock before journal read/write."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        _JOURNAL_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_JOURNAL_LOCK_PATH, "w") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                return func(*args, **kwargs)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
    return wrapper



@contextlib.contextmanager
def journal_lock():
    """Context-manager form of the journal lock. Use from external modules
    (e.g. scripts.betting.ledger) that need to hold the lock across multiple
    journal reads/writes without per-function-decorator granularity.
    """
    _JOURNAL_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = open(_JOURNAL_LOCK_PATH, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


# =============================================================================
# JOURNAL I/O
# =============================================================================

def _load_journal(journal_path: Optional[Path] = None) -> Dict:
    """Load the journal file. Returns dict with 'metadata' and 'bets' keys."""
    journal_path = journal_path or JOURNAL_PATH
    if journal_path.exists():
        try:
            with open(journal_path) as f:
                data = json.load(f)
            # Ensure required structure
            if "bets" not in data:
                data["bets"] = {}
            if "metadata" not in data:
                data["metadata"] = {}
            return data
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Failed to load journal: %s", e)
    return {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "version": 1,
        },
        "bets": {},
    }


def _save_journal(journal: Dict, journal_path: Optional[Path] = None):
    """Save journal to disk atomically."""
    journal_path = journal_path or JOURNAL_PATH
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal["metadata"]["updated_at"] = datetime.now().isoformat()
    journal["metadata"]["total_bets"] = len(journal["bets"])

    # Write to temp file then rename for atomicity
    tmp_path = journal_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(journal, f, indent=2, default=str)
    tmp_path.rename(journal_path)


def get_clv_lookup() -> Dict[str, float]:
    """Build a CLV lookup table by market type.

    Returns dict mapping market_type → average CLV%.
    Used to gate bets: if a market type has historically negative CLV,
    we're betting into adverse moves and should block.

    Markets with <5 settled bets return None (insufficient data).
    """
    journal = _load_journal()
    from collections import defaultdict
    market_clvs: Dict[str, list] = defaultdict(list)

    for bet in journal["bets"].values():
        if bet.get("status") not in ("won", "lost", "push"):
            continue
        clv = bet.get("clv_pct")
        if clv is None:
            continue
        market = (bet.get("market") or "unknown").upper()
        # Normalize market names
        if market in ("H2H", "MONEYLINE"):
            market = "1X2"
        market_clvs[market].append(clv)

    result = {}
    for market, clvs in market_clvs.items():
        if len(clvs) >= 5:
            result[market] = round(sum(clvs) / len(clvs), 2)
        # else: insufficient data, don't include

    return result


def _compute_clv(bet: Dict) -> None:
    """Compute Closing Line Value for a settled bet.

    CLV measures whether we got better odds than the sharp market implied.
    Positive CLV = we beat the market (good). Negative = we didn't (bad).

    Two methods, in priority order:
    1. If closing_odds available: CLV = (1/placed_odds) - (1/closing_odds)
       (we got better odds than closing line)
    2. If sharp_implied_prob available: CLV = (1/placed_odds) - sharp_implied_prob
       (we got better odds than Pinnacle's implied probability)

    Stored as percentage (e.g., 2.5 means +2.5% edge over closing/sharp).
    """
    placed_odds = bet.get("odds")
    if not placed_odds or placed_odds <= 1.0:
        return

    placed_implied = 1.0 / placed_odds

    # Method 1: closing odds (best — actual closing line)
    closing_odds = bet.get("closing_odds")
    if closing_odds and closing_odds > 1.0:
        closing_implied = 1.0 / closing_odds
        bet["clv_pct"] = round((closing_implied - placed_implied) * 100, 2)
        return

    # Method 2: sharp implied probability (Pinnacle at placement time)
    sharp_prob = bet.get("sharp_implied_prob")
    if sharp_prob and sharp_prob > 0:
        # CLV = sharp thinks outcome is X% likely, we got odds implying Y%
        # If sharp says 45% but our odds imply 40%, CLV = +5% (we got value)
        bet["clv_pct"] = round((sharp_prob - placed_implied) * 100, 2)
        return


def backfill_clv() -> Dict:
    """Backfill CLV for all settled bets that have the required data.

    Returns dict with counts of bets updated.
    """
    journal = _load_journal()
    updated = 0

    for bet_id, bet in journal["bets"].items():
        if bet.get("status") not in ("won", "lost", "push"):
            continue
        if bet.get("clv_pct") is not None:
            continue  # Already has CLV

        _compute_clv(bet)
        if bet.get("clv_pct") is not None:
            updated += 1

    if updated > 0:
        _save_journal(journal)
        log.info("Backfilled CLV for %d settled bets", updated)

    return {"updated": updated, "total_settled": sum(
        1 for b in journal["bets"].values()
        if b.get("status") in ("won", "lost", "push")
    )}


def _generate_bet_id(date: str, match: str, market: str, selection: str) -> str:
    """Generate deterministic bet_id for deduplication.

    Normalizes market names so h2h/H2H/1X2/moneyline all produce the same ID.
    """
    m = market.strip().upper()
    if m in ("H2H", "1X2", "MONEYLINE"):
        m = "h2h"  # Canonical: always lowercase h2h
    elif "TOTALS" in m or "O/U" in m:
        # Preserve line info (e.g., "O/U 2.5" → "OU_2.5")
        m = market.strip().upper().replace(" ", "_").replace("/", "")
    else:
        m = market.strip().upper().replace(" ", "_").replace("/", "")
    parts = [
        date.strip(),
        match.strip().replace(" ", "_"),
        m,
        selection.strip().upper().replace(" ", "_"),
    ]
    return "_".join(parts)


# =============================================================================
# CRUD OPERATIONS
# =============================================================================

MAX_EDGE_PCT = 12.0  # Defense-in-depth: reject bets above this edge regardless of caller

# =============================================================================
# MODEL VERSION STAMPING — auto-tags every new bet with the deployed model
# =============================================================================

_MODEL_VERSION_CACHE: Dict = {"mtime": None, "data": None}
_DEPLOYMENT_STATE_PATH = DATA_DIR / "models" / "deployment_state.json"
_GIT_SHA_CACHE: Optional[str] = None


def _get_git_sha() -> Optional[str]:
    """Current HEAD short SHA. Cached per-process. None if not a git repo / git missing."""
    global _GIT_SHA_CACHE
    if _GIT_SHA_CACHE is not None:
        return _GIT_SHA_CACHE or None
    try:
        import subprocess
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode().strip()
        _GIT_SHA_CACHE = sha
        return sha
    except Exception:
        _GIT_SHA_CACHE = ""  # sentinel = "tried, failed"
        return None


def _get_model_snapshot() -> Dict:
    """Read deployment_state.json and return (model_version, deployed_at, model_accuracy).

    Cached per-process; re-read if the file's mtime changes (so a retrain between
    two add_bet calls in a long-running process picks up the new version).
    """
    if not _DEPLOYMENT_STATE_PATH.exists():
        return {"model_version": "unknown", "model_deployed_at": None, "model_accuracy": None}
    try:
        mtime = _DEPLOYMENT_STATE_PATH.stat().st_mtime
    except OSError:
        return {"model_version": "unknown", "model_deployed_at": None, "model_accuracy": None}

    if _MODEL_VERSION_CACHE["mtime"] == mtime and _MODEL_VERSION_CACHE["data"] is not None:
        return _MODEL_VERSION_CACHE["data"]

    try:
        with open(_DEPLOYMENT_STATE_PATH) as f:
            ds = json.load(f)
    except Exception as e:
        log.debug("deployment_state.json unreadable: %s", e)
        return {"model_version": "unknown", "model_deployed_at": None, "model_accuracy": None}

    snapshot = {
        "model_version": ds.get("model_version") or "unknown",
        "model_deployed_at": ds.get("deployed_at"),
        "model_accuracy": (ds.get("metrics") or {}).get("accuracy"),
    }
    _MODEL_VERSION_CACHE["mtime"] = mtime
    _MODEL_VERSION_CACHE["data"] = snapshot
    return snapshot


def _stamp_model_version(entry: Dict) -> None:
    """Stamp the bet entry with model_version, model_deployed_at, model_accuracy, git_sha.

    Idempotent: if the entry already has these fields (e.g. a caller pre-set them),
    they're preserved. Only missing fields are filled.
    """
    snapshot = _get_model_snapshot()
    entry.setdefault("model_version", snapshot["model_version"])
    entry.setdefault("model_deployed_at", snapshot["model_deployed_at"])
    entry.setdefault("model_accuracy_at_placement", snapshot["model_accuracy"])
    sha = _get_git_sha()
    if sha:
        entry.setdefault("git_sha", sha)



@_with_journal_lock
def add_bet(bet_data: Dict, journal_path: Optional[Path] = None) -> str:
    """Add a bet to the journal. Returns bet_id.

    Deduplicates by bet_id. If a bet with the same ID already exists
    and is pending, updates odds/stake. If settled, skips.

    Enforces edge cap as defense-in-depth — no bet with edge > MAX_EDGE_PCT
    can enter the journal regardless of which pathway produced it.

    Required fields: match, date, market, selection, odds, stake
    """
    edge = bet_data.get("edge_pct")
    if edge is not None and abs(edge) > MAX_EDGE_PCT:
        log.warning(
            "Rejected bet %s: edge %.1f%% exceeds cap %.1f%%",
            bet_data.get("match", "unknown"), edge, MAX_EDGE_PCT,
        )
        return ""

    if not bet_data.get("date"):
        bet_data["date"] = datetime.now().strftime("%Y-%m-%d")

    journal = _load_journal(journal_path)

    bet_id = _generate_bet_id(
        bet_data.get("date", ""),
        bet_data.get("match", ""),
        bet_data.get("market", ""),
        bet_data.get("selection", ""),
    )

    if bet_id in journal["bets"]:
        existing = journal["bets"][bet_id]
        if existing.get("status") != "pending":
            log.debug("Skipping settled bet %s", bet_id)
            return bet_id
    else:
        # Extra safety: check for same match+selection with different bet_id
        # (catches market name variants that produce different IDs)
        match_norm = bet_data.get("match", "").strip()
        sel_norm = bet_data.get("selection", "").strip().upper()
        for existing_id, existing_bet in journal["bets"].items():
            if (existing_bet.get("match", "").strip() == match_norm and
                existing_bet.get("selection", "").strip().upper() == sel_norm and
                existing_bet.get("status") not in ("superseded",)):
                log.warning("Duplicate blocked: %s %s already exists as %s",
                           match_norm, sel_norm, existing_id)
                return existing_id

    if bet_id in journal["bets"]:
        existing = journal["bets"][bet_id]
        # Update pending bet with latest pipeline data.
        # Always update odds/stake from pipeline re-runs — staking config
        # changes (e.g., Kelly → proportional) must propagate to journal.
        for key in ("model_prob", "edge_pct", "avg_odds",
                     "pinnacle_odds", "bookmaker", "confidence", "factors",
                     "sharp_implied_prob", "pipeline_status",
                     "league", "odds", "stake"):
            if key in bet_data and bet_data[key] is not None:
                existing[key] = bet_data[key]
        existing["updated_at"] = datetime.now().isoformat()
        log.debug("Updated pending bet %s", bet_id)
    else:
        # New bet entry
        entry = {
            "bet_id": bet_id,
            "match": bet_data.get("match", ""),
            "date": bet_data.get("date", ""),
            "league": bet_data.get("league", "serie_a"),
            "market": bet_data.get("market", ""),
            "selection": bet_data.get("selection", ""),
            "model_prob": bet_data.get("model_prob"),
            "sharp_implied_prob": bet_data.get("sharp_implied_prob"),
            "edge_pct": bet_data.get("edge_pct"),
            "odds": bet_data.get("odds"),
            "bookmaker": bet_data.get("bookmaker"),
            "avg_odds": bet_data.get("avg_odds"),
            "pinnacle_odds": bet_data.get("pinnacle_odds"),
            "stake": bet_data.get("stake"),
            "confidence": bet_data.get("confidence"),
            "factors": bet_data.get("factors"),
            "status": "pending",
            "result_score": None,
            "profit": None,
            "closing_odds": None,
            "clv_pct": None,
            "placed_at": bet_data.get("placed_at", datetime.now().isoformat()),
            "settled_at": None,
            "pipeline_status": bet_data.get("pipeline_status"),
        }
        # Auto-tag model version so post-retrain audits can filter by model.
        _stamp_model_version(entry)
        journal["bets"][bet_id] = entry
        log.debug("Added new bet %s (model=%s)", bet_id, entry.get("model_version"))

    _save_journal(journal, journal_path)
    return bet_id


def get_pending_bets(match_date: str = None, include_superseded: bool = True,
                     journal_path: Optional[Path] = None) -> List[Dict]:
    """Get all unsettled bets, optionally filtered by match date.

    Args:
        match_date: Optional date filter (YYYY-MM-DD).
        include_superseded: If True, also return superseded bets (pipeline
            re-ran and generated different bets, but originals may have
            already been placed by the user).
    """
    journal = _load_journal(journal_path)
    settleable = {"pending", "superseded"} if include_superseded else {"pending"}
    pending = []
    for bet in journal["bets"].values():
        if bet.get("status") not in settleable:
            continue
        if match_date and bet.get("date") != match_date:
            continue
        pending.append(bet)
    return sorted(pending, key=lambda b: (b.get("date") or "", b.get("match") or ""))


def get_settled_bets(journal_path: Optional[Path] = None) -> List[Dict]:
    """Get all settled bets (won/lost/push/void)."""
    journal = _load_journal(journal_path)
    settled = []
    for bet in journal["bets"].values():
        if bet.get("status") in _SETTLED_STATUSES:
            settled.append(bet)
    # `settled_at` may be None for legacy/manually-corrected entries —
    # `or ""` coalesces to a sortable empty string instead of raising
    # TypeError when sorting against str-typed entries.
    return sorted(settled, key=lambda b: b.get("settled_at") or "")


def settle_paper_bets(results: Dict[str, Dict]) -> Dict:
    """Grade pending PAPER bets against fetched results and settle them in
    the paper journal. Flat-stake P&L; bankroll/history caches untouched.

    The paper journal only receives what run_paper_track() journals — O/U
    markets today — but 1X2 grading is included for when more markets earn
    enablement. An unknown market is left pending with a loud warning,
    never silently mis-graded (the priced path and the graded path must
    enumerate the same market set — 2026-08-31 lesson).
    """
    pending = get_pending_bets(journal_path=PAPER_JOURNAL_PATH)
    summary = {"settled": 0, "won": 0, "push": 0, "voided": 0,
               "pending": len(pending)}
    for bet in pending:
        match = bet.get("match", "")
        res = results.get(match)
        if res is None:
            for rk, rv in results.items():
                if (rk == match.replace(" vs ", " - ")
                        or rk.replace(" - ", " vs ") == match):
                    res = rv
                    break
        if res is None:
            continue

        status = (res.get("status") or "").lower()
        hs, as_ = res.get("home_score"), res.get("away_score")
        outcome = None
        score_str = None
        if status in ("postponed", "cancelled", "suspended", "walkover"):
            outcome = "void"
        elif hs is None or as_ is None:
            continue
        else:
            score_str = f"{int(hs)}-{int(as_)}"
            total = res.get("total_goals", hs + as_)
            m = (bet.get("market") or "").upper().strip()
            sel = (bet.get("selection") or "").upper().strip()
            if m.startswith("O/U") or m == "TOTALS":
                try:
                    line = float(sel.split()[-1])
                except (ValueError, IndexError):
                    line = 2.5
                if total == line:
                    outcome = "push"
                elif "OVER" in sel:
                    outcome = "won" if total > line else "lost"
                elif "UNDER" in sel:
                    outcome = "won" if total < line else "lost"
            elif m in ("1X2", "H2H"):
                if sel in ("HOME", "1"):
                    outcome = "won" if hs > as_ else "lost"
                elif sel in ("DRAW", "X"):
                    outcome = "won" if hs == as_ else "lost"
                elif sel in ("AWAY", "2"):
                    outcome = "won" if as_ > hs else "lost"
            if outcome is None:
                log.warning("Paper settle: unknown market %r on %s — left "
                            "pending", bet.get("market"), match)
                continue

        stake = float(bet.get("stake") or 0)
        odds = float(bet.get("odds") or 0)
        if outcome == "won" and odds <= 1.0:
            log.warning("Paper settle: bet %s has no usable odds (%r) — left "
                        "pending", bet.get("bet_id"), bet.get("odds"))
            continue
        profit = {"won": round(stake * (odds - 1), 2),
                  "lost": -stake}.get(outcome, 0.0)
        # Results dicts carry "commence_time" (results_fetcher.parse_scores),
        # never "kickoff_at" — same key the real settler reads.
        if settle_bet(bet.get("bet_id", ""), outcome, result_score=score_str,
                      profit=profit, match_kickoff_at=res.get("commence_time") or None,
                      journal_path=PAPER_JOURNAL_PATH):
            summary["settled"] += 1
            summary["pending"] -= 1
            if outcome == "won":
                summary["won"] += 1
            elif outcome == "push":
                summary["push"] += 1
            elif outcome == "void":
                summary["voided"] += 1
    if summary["settled"]:
        log.info("Paper settle: %(settled)d settled (%(won)d W, %(push)d P, "
                 "%(voided)d V), %(pending)d pending", summary)
    return summary


def get_paper_track_stats(league: str = None) -> Dict:
    """The gated-league deployment bar, measured: settled count, W/L, flat
    ROI, mean CLV — read from the paper journal only."""
    settled = get_settled_bets(journal_path=PAPER_JOURNAL_PATH)
    pending = get_pending_bets(journal_path=PAPER_JOURNAL_PATH,
                               include_superseded=False)
    if league:
        settled = [b for b in settled if b.get("league") == league]
        pending = [b for b in pending if b.get("league") == league]
    wl = [b for b in settled if b.get("status") in ("won", "lost")]
    staked = sum(float(b.get("stake") or 0) for b in wl)
    profit = sum(float(b.get("profit") or 0) for b in settled)
    clvs = [float(b["clv_pct"]) for b in settled
            if b.get("clv_pct") is not None]
    return {
        "league": league or "all",
        "n_settled": len(settled),
        "n_won": sum(1 for b in wl if b["status"] == "won"),
        "n_lost": sum(1 for b in wl if b["status"] == "lost"),
        "n_pending": len(pending),
        "roi_pct": round(profit / staked * 100, 2) if staked else None,
        "profit": round(profit, 2),
        "mean_clv_pct": round(sum(clvs) / len(clvs), 2) if clvs else None,
        "n_clv": len(clvs),
    }


@_with_journal_lock
def settle_bet(bet_id: str, status: str, result_score: str = None,
               profit: float = None,
               match_kickoff_at: str | None = None,
               journal_path: Optional[Path] = None) -> bool:
    """Mark a bet as won/lost/push/void.

    Args:
        bet_id: journal key
        status: won / lost / push / void
        result_score: "home-away" string, e.g. "2-1"
        profit: signed profit amount (negative on loss)
        match_kickoff_at: ISO kickoff timestamp from the match source.
            Stored as a separate field from `settled_at` — the latter is the
            audit timestamp of when our grader ran; the former is when the
            match kicked off (used for chronological sorting).

    Returns True if bet was found and updated, False otherwise.
    """
    if status not in _VALID_SETTLE_INPUTS:
        log.error("Invalid status: %s", status)
        return False
    status = _canon_settle_status(status)

    journal = _load_journal(journal_path)
    if bet_id not in journal["bets"]:
        log.warning("Bet %s not found in journal", bet_id)
        return False

    bet = journal["bets"][bet_id]
    # Superseded bets: allow settling ONLY if no replacement is already settled
    # (user may have placed the original bet before the pipeline superseded it).
    if bet.get("status") == "superseded":
        replacement_id = bet.get("superseded_by")
        if replacement_id:
            replacement = journal["bets"].get(replacement_id)
            if replacement is None:
                log.warning("Bet %s superseded_by %s but replacement not found in journal",
                            bet_id, replacement_id)
                # Allow settling — replacement may have been deleted
            elif replacement.get("status") in _SETTLED_STATUSES:
                log.debug("Bet %s superseded by %s (already settled) — skipping",
                          bet_id, replacement_id)
                return False
        elif not replacement_id:
            log.warning("Bet %s is superseded but has no superseded_by field", bet_id)
        # No replacement settled yet — allow settling the original
    elif bet.get("status") != "pending":
        log.debug("Bet %s cannot be settled (status=%s)", bet_id, bet.get("status"))
        return False

    bet["status"] = status
    bet["result_score"] = result_score
    bet["profit"] = profit
    bet["settled_at"] = datetime.now().isoformat()
    if match_kickoff_at:
        bet["match_kickoff_at"] = match_kickoff_at

    # Compute CLV (Closing Line Value) from stored data
    # CLV = placed_implied_prob - sharp_implied_prob (positive = we got better odds than sharp market)
    _compute_clv(bet)

    _save_journal(journal, journal_path)
    log.info("Settled %s as %s (profit: %s, clv: %s%%)",
             bet_id, status, profit, bet.get("clv_pct"))
    return True


def repair_settlements(dry_run: bool = True) -> Dict:
    """Re-derive correct outcomes from stored result_score and fix wrong statuses.

    Some bets were incorrectly settled (e.g., marked 'lost' when they should be
    'won') due to a dedup collision in results_fetcher that prevented correction.
    This function re-computes the correct outcome for every settled bet that has
    a result_score, and fixes any mismatches.

    Args:
        dry_run: If True, only report what would change. If False, apply fixes.

    Returns:
        Dict with 'checked', 'wrong', 'fixed' counts and 'details' list.
    """
    journal = _load_journal()
    stats = {"checked": 0, "wrong": 0, "fixed": 0, "details": []}

    for bet_id, bet in journal["bets"].items():
        if bet.get("status") not in ("won", "lost", "push"):
            continue
        result_score = bet.get("result_score")
        if not result_score or "-" not in str(result_score):
            continue

        stats["checked"] += 1
        parts = str(result_score).split("-")
        try:
            home_score = int(parts[0].strip())
            away_score = int(parts[1].strip())
        except (ValueError, IndexError):
            continue

        selection = bet.get("selection", "")
        market = bet.get("market", "")
        sel_upper = selection.upper().strip()
        mkt_upper = market.upper().strip()
        total_goals = home_score + away_score
        btts = home_score > 0 and away_score > 0
        expected = "lost"  # default

        # 1X2
        if mkt_upper in ("H2H", "1X2"):
            if sel_upper in ("HOME", "1"):
                expected = "won" if home_score > away_score else "lost"
            elif sel_upper in ("DRAW", "X"):
                expected = "won" if home_score == away_score else "lost"
            elif sel_upper in ("AWAY", "2"):
                expected = "won" if away_score > home_score else "lost"

        # O/U
        elif mkt_upper.startswith("O/U") or mkt_upper == "TOTALS":
            if "OVER" in sel_upper:
                line = float(sel_upper.split()[-1]) if len(sel_upper.split()) > 1 else 2.5
                if total_goals > line:
                    expected = "won"
                elif total_goals == line:
                    expected = "push"
            elif "UNDER" in sel_upper:
                line = float(sel_upper.split()[-1]) if len(sel_upper.split()) > 1 else 2.5
                if total_goals < line:
                    expected = "won"
                elif total_goals == line:
                    expected = "push"

        # BTTS
        elif mkt_upper == "BTTS":
            if "YES" in sel_upper:
                expected = "won" if btts else "lost"
            else:
                expected = "won" if not btts else "lost"

        # DC
        elif mkt_upper == "DC":
            if "1X" in sel_upper or "HOME OR DRAW" in sel_upper:
                expected = "won" if home_score >= away_score else "lost"
            elif "X2" in sel_upper or "DRAW OR AWAY" in sel_upper:
                expected = "won" if away_score >= home_score else "lost"
            elif "12" in sel_upper or "HOME OR AWAY" in sel_upper:
                expected = "won" if home_score != away_score else "lost"

        # DNB
        elif mkt_upper == "DNB":
            if "HOME" in sel_upper:
                if home_score > away_score:
                    expected = "won"
                elif home_score == away_score:
                    expected = "push"
            elif "AWAY" in sel_upper:
                if away_score > home_score:
                    expected = "won"
                elif home_score == away_score:
                    expected = "push"

        # AH / Spreads
        elif mkt_upper.startswith("AH") or mkt_upper in ("SPREADS", "HANDICAP"):
            parts_sel = sel_upper.split()
            side = parts_sel[0] if parts_sel else ""
            line = 0.0
            if len(parts_sel) > 1:
                try:
                    line = float(parts_sel[-1])
                except ValueError:
                    line = 0.0
            goal_diff = home_score - away_score
            adjusted = (-goal_diff + line) if side == "AWAY" else (goal_diff + line)
            if adjusted > 0:
                expected = "won"
            elif adjusted == 0:
                expected = "push"

        current = bet.get("status")
        if current != expected:
            stats["wrong"] += 1
            detail = {
                "bet_id": bet_id,
                "match": bet.get("match"),
                "market": market,
                "selection": selection,
                "result_score": result_score,
                "was": current,
                "should_be": expected,
            }
            stats["details"].append(detail)

            if not dry_run:
                # Fix the status and recalculate profit
                odds = bet.get("odds", 1.0)
                stake = bet.get("stake", 0.0)
                if expected == "won":
                    profit = round(stake * (odds - 1), 2)
                elif expected == "push":
                    profit = 0.0
                else:
                    profit = round(-stake, 2)

                bet["status"] = expected
                bet["profit"] = profit
                bet["repaired_at"] = datetime.now().isoformat()
                bet["repair_note"] = f"Was '{current}', fixed to '{expected}'"
                stats["fixed"] += 1

    if not dry_run and stats["fixed"] > 0:
        _save_journal(journal)
        log.info("Repaired %d/%d wrong settlements", stats["fixed"], stats["wrong"])

    return stats


def update_clv(bet_id: str, closing_odds: float, clv_pct: float = None) -> bool:
    """Store CLV data for a bet.

    If clv_pct is not provided, computes it from odds and closing_odds.
    CLV = (bet_odds / closing_odds) - 1
    """
    journal = _load_journal()
    if bet_id not in journal["bets"]:
        log.warning("Bet %s not found for CLV update", bet_id)
        return False

    bet = journal["bets"][bet_id]
    bet["closing_odds"] = closing_odds

    if clv_pct is not None:
        bet["clv_pct"] = clv_pct
    elif bet.get("odds") and closing_odds and closing_odds > 1.0:
        bet["clv_pct"] = round(((bet["odds"] / closing_odds) - 1.0) * 100, 2)

    _save_journal(journal)
    return True


# =============================================================================
# STATISTICS
# =============================================================================

def get_journal_stats() -> Dict:
    """Aggregate P&L, ROI, CLV, by-market breakdown."""
    journal = _load_journal()
    bets = list(journal["bets"].values())

    if not bets:
        return {"total_bets": 0, "message": "No bets in journal"}

    pending = [b for b in bets if b.get("status") == "pending"]
    settled = [b for b in bets if b.get("status") in _SETTLED_STATUSES]
    won = [b for b in settled if b["status"] == "won"]
    lost = [b for b in settled if b["status"] == "lost"]
    pushes = [b for b in settled if b["status"] == "push"]

    total_staked = sum(b.get("stake", 0) or 0 for b in settled)
    total_profit = sum(b.get("profit", 0) or 0 for b in settled)
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0.0

    # CLV stats
    clv_bets = [b for b in bets if b.get("clv_pct") is not None]
    avg_clv = (sum(b["clv_pct"] for b in clv_bets) / len(clv_bets)) if clv_bets else 0.0
    positive_clv = sum(1 for b in clv_bets if b["clv_pct"] > 0)

    # By market
    by_market = {}
    for b in settled:
        m = b.get("market", "unknown")
        if m not in by_market:
            by_market[m] = {"bets": 0, "wins": 0, "losses": 0, "pushes": 0,
                            "profit": 0.0, "staked": 0.0}
        by_market[m]["bets"] += 1
        by_market[m]["profit"] += b.get("profit", 0) or 0
        by_market[m]["staked"] += b.get("stake", 0) or 0
        if b["status"] == "won":
            by_market[m]["wins"] += 1
        elif b["status"] == "lost":
            by_market[m]["losses"] += 1
        elif b["status"] == "push":
            by_market[m]["pushes"] += 1

    # Bets contested or settled today -- the settlement card and day wrap
    # read this key (it was read for months before it existed; see 2026-08-28).
    _today = datetime.now().strftime("%Y-%m-%d")
    settled_today = [
        b for b in settled
        if (b.get("date") or "").startswith(_today)
        or (str(b.get("settled_at") or ""))[:10] == _today
    ]

    # Trailing W/L streak over decisive results (pushes/voids neither
    # break nor extend a streak). Consumed by the scheduler's post-settlement
    # loss-streak alert -- these keys were read there long before they existed.
    decisive = sorted(
        (b for b in settled if b["status"] in ("won", "lost")),
        key=lambda b: b.get("settled_at") or b.get("date") or "",
    )
    current_streak = 0
    streak_loss = 0.0
    recent_losses = []
    if decisive:
        last_status = decisive[-1]["status"]
        run = []
        for b in reversed(decisive):
            if b["status"] != last_status:
                break
            run.append(b)
        current_streak = len(run) if last_status == "won" else -len(run)
        if last_status == "lost":
            streak_loss = round(sum(b.get("stake", 0) or 0 for b in run), 2)
            recent_losses = list(reversed(run[:5]))  # most recent last

    return {
        "total_bets": len(bets),
        "pending": len(pending),
        "settled": len(settled),
        "won": len(won),
        "lost": len(lost),
        "pushes": len(pushes),
        "total_staked": round(total_staked, 2),
        "total_profit": round(total_profit, 2),
        "roi_pct": round(roi, 2),
        "clv_avg_pct": round(avg_clv, 2),
        "clv_positive": positive_clv,
        "clv_total": len(clv_bets),
        "by_market": by_market,
        "current_streak": current_streak,
        "streak_loss": streak_loss,
        "recent_losses": recent_losses,
        "settled_today": settled_today,
    }


# =============================================================================
# FILL TIER -- what was actually placed at the book (two-tier ledger)
# =============================================================================
# The journal row records what the ENGINE committed (model tier: stake, odds,
# edge -- never rewritten). These functions annotate the SAME row with what
# happened at the book. They never create rows, never touch stake/odds/status,
# and never alter settlement math -- verified-vs-journal comparisons are
# computed by readers (proof-of-edge card) from the annotations.

FILL_STATUSES = ("placed", "missed", "unverified")


@_with_journal_lock
def mark_bet_fill(bet_id: str, fill_status: str,
                  filled_odds: float | None = None) -> dict:
    """Annotate an existing journal row with its real-world fill state.

    fill_status: "placed" (bet was placed at the book; filled_odds records the
    actual price, defaulting to the committed odds), "missed" (never placed),
    or "unverified" (kickoff passed with no confirmation -- set by the sweep).
    Re-marking is allowed (a mis-tap is corrected by tapping again), EXCEPT
    that the sweep's "unverified" never overwrites an explicit answer.
    """
    if fill_status not in FILL_STATUSES:
        return {"ok": False, "error": f"invalid fill_status: {fill_status}"}
    journal = _load_journal()
    bet = journal["bets"].get(bet_id)
    if bet is None:
        return {"ok": False, "error": f"no such bet: {bet_id}"}
    if fill_status == "unverified" and bet.get("fill_status") in ("placed", "missed"):
        return {"ok": True, "bet_id": bet_id, "fill_status": bet["fill_status"],
                "note": "explicit answer stands"}
    bet["fill_status"] = fill_status
    if fill_status == "placed":
        try:
            odds = float(filled_odds) if filled_odds else float(bet.get("odds") or 0)
        except (TypeError, ValueError):
            odds = float(bet.get("odds") or 0)
        if odds > 0:
            bet["filled_odds"] = round(odds, 2)
    else:
        bet.pop("filled_odds", None)
    bet["fill_updated_at"] = datetime.now().isoformat()
    _save_journal(journal)
    log.info("Fill recorded: %s -> %s%s", bet_id, fill_status,
             f" @ {bet.get('filled_odds')}" if bet.get("filled_odds") else "")
    return {"ok": True, "bet_id": bet_id, "fill_status": fill_status,
            "filled_odds": bet.get("filled_odds")}


@_with_journal_lock
def sweep_unverified_fills(bet_ids) -> int:
    """After kickoff: any of these bets with no fill answer is flagged
    "unverified" -- it stays in the journal (model tier intact) but drops out
    of verified ROI/CLV. Explicit placed/missed answers are never overwritten.
    Returns the number of rows flagged.
    """
    journal = _load_journal()
    flagged = 0
    now = datetime.now().isoformat()
    for bid in bet_ids or []:
        bet = journal["bets"].get(bid)
        if bet is None or bet.get("fill_status"):
            continue
        bet["fill_status"] = "unverified"
        bet["fill_updated_at"] = now
        flagged += 1
    if flagged:
        _save_journal(journal)
        log.info("Fill sweep: %d bet(s) flagged unverified", flagged)
    return flagged


# =============================================================================
# MIGRATION FROM LEGACY FILES
# =============================================================================

def migrate_from_legacy() -> Dict:
    """One-time import from 5 legacy files.

    Reads placed_bets_log.json, placed_bets.json, bet_history.json,
    history.json, and clv_history.json. Deduplicates by (date, match,
    market, selection). Merges settlement and CLV data.

    Backs up legacy files to data/betting/legacy/ first.
    """
    journal = _load_journal()
    stats = {"sources_read": 0, "total_imported": 0, "duplicates_merged": 0,
             "settlements_applied": 0, "clv_applied": 0, "errors": []}

    # Intermediate store: dedup_key -> best record
    merged: Dict[str, Dict] = {}

    def _dedup_key(date: str, match: str, market: str, selection: str) -> str:
        return _generate_bet_id(date, match, market, selection)

    def _richness(rec: Dict) -> int:
        """Count non-None fields — prefer richer records."""
        return sum(1 for v in rec.values() if v is not None)

    # --- Source 1: placed_bets_log.json (14 records) ---
    path = LEGACY_FILES["placed_bets_log"]
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            for b in data:
                key = _dedup_key(b.get("date", ""), b.get("match", ""),
                                 b.get("market", ""), b.get("selection", ""))
                rec = {
                    "match": b.get("match", ""),
                    "date": b.get("date", ""),
                    "market": _normalize_market(b.get("market", "")),
                    "selection": b.get("selection", ""),
                    "model_prob": b.get("our_probability"),
                    "edge_pct": b.get("value_pct"),
                    "odds": b.get("odds"),
                    "stake": b.get("stake"),
                    "confidence": b.get("confidence"),
                    "factors": b.get("factors"),
                    "placed_at": b.get("recorded_at"),
                    "status": b.get("status", "pending"),
                }
                if key not in merged or _richness(rec) > _richness(merged[key]):
                    merged[key] = rec
                else:
                    stats["duplicates_merged"] += 1
            stats["sources_read"] += 1
            log.info("Read %d bets from placed_bets_log.json", len(data))
        except Exception as e:
            stats["errors"].append(f"placed_bets_log: {e}")

    # --- Source 2: placed_bets.json (37 records) ---
    path = LEGACY_FILES["placed_bets"]
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            bets_list = data.get("bets", []) if isinstance(data, dict) else data
            for b in bets_list:
                key = _dedup_key(b.get("date", ""), b.get("match", ""),
                                 b.get("market", ""), b.get("selection", ""))
                rec = {
                    "match": b.get("match", ""),
                    "date": b.get("date", ""),
                    "market": _normalize_market(b.get("market", "")),
                    "selection": b.get("selection", ""),
                    "model_prob": b.get("model_prob"),
                    "sharp_implied_prob": b.get("sharp_implied_prob"),
                    "edge_pct": b.get("edge_pct"),
                    "odds": b.get("best_odds"),
                    "bookmaker": b.get("best_bookmaker"),
                    "avg_odds": b.get("avg_odds"),
                    "pinnacle_odds": b.get("pinnacle_odds"),
                    "stake": b.get("stake_amount"),
                    "placed_at": b.get("placed_at"),
                    "status": "pending",
                }
                if key not in merged or _richness(rec) > _richness(merged[key]):
                    old = merged.get(key)
                    merged[key] = rec
                    # Preserve fields from previous source that this one lacks
                    if old:
                        stats["duplicates_merged"] += 1
                        for k in ("confidence", "factors"):
                            if rec.get(k) is None and old.get(k) is not None:
                                rec[k] = old[k]
                else:
                    # Keep richer existing, but overlay any new fields
                    existing = merged[key]
                    for k, v in rec.items():
                        if v is not None and existing.get(k) is None:
                            existing[k] = v
                    stats["duplicates_merged"] += 1
            stats["sources_read"] += 1
            log.info("Read %d bets from placed_bets.json", len(bets_list))
        except Exception as e:
            stats["errors"].append(f"placed_bets: {e}")

    # --- Source 3: bet_history.json (37 records, in data/upcoming/) ---
    path = LEGACY_FILES["bet_history"]
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            bets_list = data if isinstance(data, list) else data.get("bets", [])
            for b in bets_list:
                key = _dedup_key(b.get("date", ""), b.get("match", ""),
                                 b.get("market", ""), b.get("selection", ""))
                rec = {
                    "match": b.get("match", ""),
                    "date": b.get("date", ""),
                    "market": _normalize_market(b.get("market", "")),
                    "selection": b.get("selection", ""),
                    "model_prob": b.get("model_prob"),
                    "edge_pct": b.get("edge_pct"),
                    "odds": b.get("best_odds"),
                    "bookmaker": b.get("best_bookmaker"),
                    "stake": b.get("stake"),
                    "placed_at": b.get("placed_at"),
                    "status": "pending",
                }
                if key not in merged or _richness(rec) > _richness(merged[key]):
                    old = merged.get(key)
                    merged[key] = rec
                    if old:
                        stats["duplicates_merged"] += 1
                        for k in ("confidence", "factors", "avg_odds",
                                   "pinnacle_odds", "sharp_implied_prob", "bookmaker"):
                            if rec.get(k) is None and old.get(k) is not None:
                                rec[k] = old[k]
                else:
                    existing = merged[key]
                    for k, v in rec.items():
                        if v is not None and existing.get(k) is None:
                            existing[k] = v
                    stats["duplicates_merged"] += 1
            stats["sources_read"] += 1
            log.info("Read %d bets from bet_history.json", len(bets_list))
        except Exception as e:
            stats["errors"].append(f"bet_history: {e}")

    # --- Source 4: history.json (3 settled records) ---
    path = LEGACY_FILES["history"]
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            for b in data:
                key = _dedup_key(b.get("date", ""), b.get("match", ""),
                                 b.get("market", ""), b.get("selection", ""))
                if key in merged:
                    rec = merged[key]
                    rec["status"] = b.get("status", rec.get("status", "pending"))
                    rec["profit"] = b.get("profit")
                    rec["result_score"] = b.get("result")
                    rec["settled_at"] = b.get("settled_at")
                    if rec.get("odds") is None:
                        rec["odds"] = b.get("odds")
                    if rec.get("stake") is None:
                        rec["stake"] = b.get("stake")
                    stats["settlements_applied"] += 1
                else:
                    # Bet only in history — add it
                    merged[key] = {
                        "match": b.get("match", ""),
                        "date": b.get("date", ""),
                        "market": _normalize_market(b.get("market", "")),
                        "selection": b.get("selection", ""),
                        "odds": b.get("odds"),
                        "stake": b.get("stake"),
                        "status": b.get("status", "lost"),
                        "profit": b.get("profit"),
                        "result_score": b.get("result"),
                        "settled_at": b.get("settled_at"),
                    }
                    stats["settlements_applied"] += 1
            stats["sources_read"] += 1
            log.info("Read %d settlements from history.json", len(data))
        except Exception as e:
            stats["errors"].append(f"history: {e}")

    # --- Source 5: clv_history.json (11 records) ---
    path = LEGACY_FILES["clv_history"]
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            clv_bets = data.get("bets", []) if isinstance(data, dict) else data
            for b in clv_bets:
                # CLV records don't have date — match by (match, market, selection)
                # Try all keys to find matching bet
                found = False
                for key, rec in merged.items():
                    if (rec.get("match") == b.get("match") and
                            rec.get("selection") == b.get("selection") and
                            _normalize_market(rec.get("market", "")).upper() ==
                            _normalize_market(b.get("market", "")).upper()):
                        rec["closing_odds"] = b.get("closing_odds")
                        rec["clv_pct"] = b.get("clv_pct")
                        stats["clv_applied"] += 1
                        found = True
                        break
                if not found:
                    log.debug("CLV record unmatched: %s %s %s",
                              b.get("match"), b.get("market"), b.get("selection"))
            stats["sources_read"] += 1
            log.info("Applied %d CLV records from clv_history.json",
                     stats["clv_applied"])
        except Exception as e:
            stats["errors"].append(f"clv_history: {e}")

    # Write all merged bets to journal
    for key, rec in merged.items():
        bet_id = key
        # Don't overwrite already-settled journal bets
        if bet_id in journal["bets"] and \
                journal["bets"][bet_id].get("status") != "pending":
            continue

        entry = {
            "bet_id": bet_id,
            "match": rec.get("match", ""),
            "date": rec.get("date", ""),
            "market": rec.get("market", ""),
            "selection": rec.get("selection", ""),
            "model_prob": rec.get("model_prob"),
            "sharp_implied_prob": rec.get("sharp_implied_prob"),
            "edge_pct": rec.get("edge_pct"),
            "odds": rec.get("odds"),
            "bookmaker": rec.get("bookmaker"),
            "avg_odds": rec.get("avg_odds"),
            "pinnacle_odds": rec.get("pinnacle_odds"),
            "stake": rec.get("stake"),
            "confidence": rec.get("confidence"),
            "factors": rec.get("factors"),
            "status": rec.get("status", "pending"),
            "result_score": rec.get("result_score"),
            "profit": rec.get("profit"),
            "closing_odds": rec.get("closing_odds"),
            "clv_pct": rec.get("clv_pct"),
            "placed_at": rec.get("placed_at"),
            "settled_at": rec.get("settled_at"),
        }
        journal["bets"][bet_id] = entry

    stats["total_imported"] = len(journal["bets"])

    # Backup legacy files
    _backup_legacy_files()

    _save_journal(journal)
    return stats


def _normalize_market(market: str) -> str:
    """Normalize market names across legacy sources.

    Legacy sources use: 'h2h', '1X2', 'totals', 'O/U 2.5', 'spreads', etc.
    Normalize to consistent format.
    """
    m = market.strip()
    m_upper = m.upper()
    if m_upper in ("H2H", "1X2", "MONEYLINE"):
        return "1X2"
    if m_upper in ("TOTALS", "OU", "OVER/UNDER") or m_upper.startswith("O/U"):
        return m if m_upper.startswith("O/U") else "O/U 2.5"
    if m_upper in ("SPREADS", "HANDICAP", "AH"):
        return "spreads"
    return m


def _backup_legacy_files():
    """Copy legacy files to data/betting/legacy/ for safety."""
    backup_dir = DATA_DIR / "betting" / "legacy"
    backup_dir.mkdir(parents=True, exist_ok=True)

    for name, path in LEGACY_FILES.items():
        if path.exists():
            dest = backup_dir / path.name
            if not dest.exists():
                shutil.copy2(path, dest)
                log.info("Backed up %s to %s", path.name, dest)


# =============================================================================
# WEEKLY REPORT
# =============================================================================

def generate_report(days: int = 7) -> str:
    """Generate a comprehensive weekly report.

    Covers: bankroll change, record, by-market, by-confidence,
    by-edge-bucket, CLV summary.
    """
    journal = _load_journal()
    bets = list(journal["bets"].values())

    if not bets:
        return "No bets in journal."

    # Date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    end_str = end_date.strftime("%b %-d, %Y")
    start_str = start_date.strftime("%b %-d, %Y")

    # Split bets
    all_settled = [b for b in bets if b.get("status") in ("won", "lost", "push")]
    all_pending = [b for b in bets if b.get("status") == "pending"]

    # Period filter for settled bets
    period_settled = []
    for b in all_settled:
        sa = b.get("settled_at", "")
        if sa:
            try:
                settled_date = datetime.fromisoformat(sa).date()
                if start_date <= settled_date <= end_date:
                    period_settled.append(b)
            except (ValueError, TypeError):
                pass

    # If no period settled, show all-time
    if not period_settled:
        period_settled = all_settled
        start_str = "All time"

    # Bankroll (from bankroll.json for accuracy)
    bankroll_path = DATA_DIR / "betting" / "bankroll.json"
    initial_balance = 1000.0
    current_balance = 1000.0
    if bankroll_path.exists():
        try:
            with open(bankroll_path) as f:
                br = json.load(f)
            initial_balance = br.get("initial_balance", 1000.0)
            current_balance = br.get("current_balance", 1000.0)
        except (json.JSONDecodeError, OSError):
            pass

    # Core stats
    wins = [b for b in period_settled if b["status"] == "won"]
    losses = [b for b in period_settled if b["status"] == "lost"]
    pushes = [b for b in period_settled if b["status"] == "push"]
    decided = len(wins) + len(losses)
    win_rate = (len(wins) / decided * 100) if decided > 0 else 0.0
    total_staked = sum(b.get("stake", 0) or 0 for b in period_settled)
    total_profit = sum(b.get("profit", 0) or 0 for b in period_settled)
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
    net_change = current_balance - initial_balance
    net_pct = (net_change / initial_balance * 100) if initial_balance > 0 else 0.0

    lines = []
    lines.append(f"WEEKLY BETTING REPORT ({start_str} - {end_str})")
    lines.append("\u2501" * 50)
    lines.append("")
    lines.append(f"BANKROLL:  ${initial_balance:,.2f} \u2192 ${current_balance:,.2f} "
                 f"({'+' if net_change >= 0 else ''}{net_change:,.2f}, "
                 f"{'+' if net_pct >= 0 else ''}{net_pct:.1f}%)")
    lines.append(f"RECORD:    {len(wins)}W-{len(losses)}L-{len(pushes)}P "
                 f"({win_rate:.0f}% win rate)")
    lines.append(f"STAKED:    ${total_staked:,.2f} | "
                 f"ROI: {'+' if roi >= 0 else ''}{roi:.1f}%")
    lines.append(f"PENDING:   {len(all_pending)} bets")

    # By market
    lines.append("")
    lines.append("BY MARKET:")
    by_market = {}
    for b in period_settled:
        m = b.get("market", "unknown")
        if m not in by_market:
            by_market[m] = {"bets": 0, "wins": 0, "losses": 0, "pushes": 0,
                            "profit": 0.0, "staked": 0.0}
        by_market[m]["bets"] += 1
        by_market[m]["profit"] += b.get("profit", 0) or 0
        by_market[m]["staked"] += b.get("stake", 0) or 0
        if b["status"] == "won":
            by_market[m]["wins"] += 1
        elif b["status"] == "lost":
            by_market[m]["losses"] += 1
        elif b["status"] == "push":
            by_market[m]["pushes"] += 1

    for m, s in sorted(by_market.items()):
        m_roi = (s["profit"] / s["staked"] * 100) if s["staked"] > 0 else 0.0
        rec = f"{s['wins']}W-{s['losses']}L"
        if s["pushes"]:
            rec += f"-{s['pushes']}P"
        lines.append(f"  {m:<12} {s['bets']} {'bet' if s['bets'] == 1 else 'bets':<5} "
                     f"{rec:<10} {'+' if s['profit'] >= 0 else ''}${s['profit']:,.2f}  "
                     f"{'+' if m_roi >= 0 else ''}{m_roi:.1f}% ROI")

    # By confidence
    lines.append("")
    lines.append("BY CONFIDENCE:")
    by_conf = {}
    for b in period_settled:
        c = (b.get("confidence") or "UNKNOWN").upper()
        if c not in by_conf:
            by_conf[c] = {"bets": 0, "profit": 0.0, "staked": 0.0}
        by_conf[c]["bets"] += 1
        by_conf[c]["profit"] += b.get("profit", 0) or 0
        by_conf[c]["staked"] += b.get("stake", 0) or 0

    for c in ("ELITE", "HIGH", "STRONG", "MEDIUM-HIGH", "MEDIUM", "STANDARD", "UNKNOWN"):
        if c in by_conf:
            s = by_conf[c]
            c_roi = (s["profit"] / s["staked"] * 100) if s["staked"] > 0 else 0.0
            lines.append(f"  {c:<14} {s['bets']} {'bet' if s['bets'] == 1 else 'bets':<5} "
                         f"{'+' if s['profit'] >= 0 else ''}${s['profit']:,.2f}  "
                         f"{'+' if c_roi >= 0 else ''}{c_roi:.0f}% ROI")

    # By edge bucket
    lines.append("")
    lines.append("BY EDGE BUCKET:")
    edge_buckets = {"0-10%": [], "10-20%": [], "20%+": []}
    for b in period_settled:
        edge = b.get("edge_pct") or 0
        if edge < 10:
            edge_buckets["0-10%"].append(b)
        elif edge < 20:
            edge_buckets["10-20%"].append(b)
        else:
            edge_buckets["20%+"].append(b)

    for bucket, bucket_bets in edge_buckets.items():
        n = len(bucket_bets)
        profit = sum(b.get("profit", 0) or 0 for b in bucket_bets)
        if n == 0:
            lines.append(f"  {bucket:<8} 0 bets")
        else:
            lines.append(f"  {bucket:<8} {n} {'bet' if n == 1 else 'bets':<5} "
                         f"{'+' if profit >= 0 else ''}${profit:,.2f}")

    # CLV summary
    clv_bets = [b for b in bets if b.get("clv_pct") is not None]
    if clv_bets:
        avg_clv = sum(b["clv_pct"] for b in clv_bets) / len(clv_bets)
        positive = sum(1 for b in clv_bets if b["clv_pct"] > 0)
        lines.append("")
        flag = " [OK]" if avg_clv > 0 else " [!]"
        lines.append(f"CLV: {avg_clv:+.2f}% avg ({positive}/{len(clv_bets)} positive){flag}")

    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def _print_bets_table(bets: List[Dict], title: str):
    """Print a table of bets."""
    print(f"\n{title} ({len(bets)} bets)")
    print("-" * 90)
    if not bets:
        print("  (none)")
        return

    print(f"  {'Date':<12} {'Match':<28} {'Market':<10} {'Selection':<18} "
          f"{'Odds':>6} {'Stake':>7} {'Status':<6}")
    print(f"  {'-'*12} {'-'*28} {'-'*10} {'-'*18} {'-'*6} {'-'*7} {'-'*6}")
    for b in bets:
        status = b.get("status", "?")
        profit_str = ""
        if status != "pending" and b.get("profit") is not None:
            profit_str = f" ({'+' if b['profit'] >= 0 else ''}{b['profit']:.2f})"
        odds_str = f"{b.get('odds', 0):.2f}" if b.get("odds") else "?"
        stake_str = f"${b.get('stake', 0):.0f}" if b.get("stake") else "?"
        print(f"  {b.get('date', '?'):<12} {b.get('match', '?'):<28} "
              f"{b.get('market', '?'):<10} {b.get('selection', '?'):<18} "
              f"{odds_str:>6} {stake_str:>7} {status:<6}{profit_str}")


def main():
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Bet Journal")
    parser.add_argument("command", nargs="?", default="stats",
                        choices=["pending", "settled", "stats", "report",
                                 "migrate", "settle"],
                        help="Command to run")
    parser.add_argument("--days", type=int, default=7,
                        help="Days for report period (default: 7)")
    args = parser.parse_args()

    if args.command == "pending":
        bets = get_pending_bets()
        _print_bets_table(bets, "PENDING BETS")

    elif args.command == "settled":
        bets = get_settled_bets()
        _print_bets_table(bets, "SETTLED BETS")

    elif args.command == "stats":
        stats = get_journal_stats()
        print(f"\n{'='*50}")
        print("BET JOURNAL STATS")
        print(f"{'='*50}")
        print(f"  Total bets:    {stats.get('total_bets', 0)}")
        print(f"  Pending:       {stats.get('pending', 0)}")
        print(f"  Settled:       {stats.get('settled', 0)} "
              f"({stats.get('won', 0)}W-{stats.get('lost', 0)}L-"
              f"{stats.get('pushes', 0)}P)")
        print(f"  Total staked:  ${stats.get('total_staked', 0):,.2f}")
        print(f"  Total profit:  ${stats.get('total_profit', 0):+,.2f}")
        print(f"  ROI:           {stats.get('roi_pct', 0):+.2f}%")
        if stats.get("clv_total"):
            print(f"  CLV:           {stats.get('clv_avg_pct', 0):+.2f}% avg "
                  f"({stats.get('clv_positive', 0)}/{stats.get('clv_total', 0)} positive)")

    elif args.command == "report":
        print(generate_report(days=args.days))

    elif args.command == "migrate":
        print("\nMigrating from legacy files...")
        result = migrate_from_legacy()
        print(f"\nMigration complete:")
        print(f"  Sources read:       {result['sources_read']}")
        print(f"  Total imported:     {result['total_imported']}")
        print(f"  Duplicates merged:  {result['duplicates_merged']}")
        print(f"  Settlements applied: {result['settlements_applied']}")
        print(f"  CLV records applied: {result['clv_applied']}")
        if result["errors"]:
            print(f"  Errors: {result['errors']}")
        print(f"\nJournal saved to: {JOURNAL_PATH}")

        # Show quick stats
        stats = get_journal_stats()
        print(f"\nPost-migration stats:")
        print(f"  Total bets: {stats['total_bets']} "
              f"({stats['pending']} pending, {stats['settled']} settled)")
        print(f"  P&L: ${stats['total_profit']:+,.2f} | ROI: {stats['roi_pct']:+.2f}%")

    elif args.command == "settle":
        print("\nFetching results and settling bets...")
        try:
            from scripts.data.results_fetcher import fetch_and_settle
            summary = fetch_and_settle()
            print(f"\nSettlement complete:")
            print(f"  Settled: {summary.get('settled', 0)}")
            print(f"  Won: {summary.get('won', 0)}")
            print(f"  Lost: {summary.get('lost', 0)}")
            print(f"  Push: {summary.get('push', 0)}")
            print(f"  Profit: ${summary.get('profit', 0):+,.2f}")
            print(f"  Balance: ${summary.get('new_balance', 0):,.2f}")
        except Exception as e:
            print(f"Settlement failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
