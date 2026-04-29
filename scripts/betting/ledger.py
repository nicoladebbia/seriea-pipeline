"""Canonical bet ledger — single source of truth.

This module owns the financial truth of the betting system. All reads and
writes of bet state MUST go through this module.

Authoritative file: data/betting/bet_journal.json
    - Only ACTIVE bets (status ∈ pending/won/lost/push/voided) contribute to
      financial aggregates.
    - Status='superseded' means the bet was replaced by a later re-bet; it
      does NOT contribute to P&L. Retained only for audit.

Derived caches (regenerated atomically on every write):
    - data/betting/bankroll.json: current_balance, peak_balance,
      lowest_balance, pending_bets, pending_stakes, updated_at.
    - data/betting/history.json: settled bets only, sorted by settled_at.

Public API:
    settle_bet(bet_id, status, profit, ...)  — update a pending bet
    add_bet(bet) — register a new bet (delegates to bet_journal.record_bet)
    supersede_bet(bet_id, new_bet_id) — mark old bet replaced

    get_balance() -> float
    get_peak() -> float
    get_lowest() -> float
    get_roi(window=None) -> float
    get_history_view() -> list[dict]    # read-only projection
    get_settled(status=None) -> list[dict]
    get_pending() -> list[dict]
    rebuild_caches() -> dict            # regenerate bankroll.json + history.json
                                          from journal; use for one-shot repair
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
JOURNAL_PATH = DATA_DIR / "betting" / "bet_journal.json"
BANKROLL_PATH = DATA_DIR / "betting" / "bankroll.json"
HISTORY_PATH = DATA_DIR / "betting" / "history.json"
_LEDGER_LOCK_PATH = DATA_DIR / "betting" / ".ledger.lock"

# Settled statuses — only these count toward realized P&L
_SETTLED_STATUSES = ("won", "lost", "push", "voided")
# Active statuses — contribute to aggregates (settled OR in-flight)
_ACTIVE_STATUSES = _SETTLED_STATUSES + ("pending",)


# ---------------------------------------------------------------------------
# Atomic I/O
# ---------------------------------------------------------------------------
def _atomic_write(path: Path, data: Any) -> None:
    """Atomic JSON write via temp file + rename. Safe against crashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


@contextlib.contextmanager
def _ledger_lock() -> Iterator[None]:
    """Process-level advisory lock. Serializes concurrent ledger writes
    (e.g. launchd settlement + dashboard auto-settle firing together).
    """
    _LEDGER_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(_LEDGER_LOCK_PATH, "w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def _read_journal() -> dict:
    if not JOURNAL_PATH.exists():
        return {"metadata": {"initial_bankroll": 1000.0}, "bets": {}}
    return json.loads(JOURNAL_PATH.read_text())


# ---------------------------------------------------------------------------
# Iteration helpers
# ---------------------------------------------------------------------------
def _iter_bets(journal: dict) -> Iterable[dict]:
    """Yield bet dicts regardless of dict- or list-shaped journal."""
    bets = journal.get("bets", {})
    if isinstance(bets, dict):
        for bet_id, b in bets.items():
            # Guarantee bet_id on the bet dict even if the key is the only location
            b = {**b, "_bet_id": b.get("bet_id") or b.get("_bet_id") or bet_id}
            yield b
    else:
        for b in bets:
            yield b


# ---------------------------------------------------------------------------
# Read API — derives everything from the journal
# ---------------------------------------------------------------------------
def get_initial_bankroll() -> float:
    j = _read_journal()
    md = j.get("metadata", {}) or {}
    return float(md.get("initial_bankroll", 1000.0))


def get_settled(status: str | None = None) -> list[dict]:
    """Return settled bets (optionally filtered by status).

    NOTE: excludes 'superseded' — those are audit-only and do not contribute
    to P&L.
    """
    j = _read_journal()
    out = []
    for b in _iter_bets(j):
        s = b.get("status")
        if status:
            if s == status:
                out.append(b)
        elif s in _SETTLED_STATUSES:
            out.append(b)
    return out


def get_pending() -> list[dict]:
    j = _read_journal()
    return [b for b in _iter_bets(j) if b.get("status") == "pending"]


def get_superseded() -> list[dict]:
    """Audit-only view of bets that were replaced by later re-bets."""
    j = _read_journal()
    return [b for b in _iter_bets(j) if b.get("status") == "superseded"]


def _settled_profit_sum(journal: dict | None = None) -> float:
    journal = journal if journal is not None else _read_journal()
    total = 0.0
    for b in _iter_bets(journal):
        if b.get("status") in _SETTLED_STATUSES:
            total += float(b.get("profit") or 0)
    return round(total, 2)


def _settled_chronological(journal: dict | None = None) -> list[dict]:
    """Settled bets sorted oldest → newest by (settled_at, match_kickoff_at)."""
    journal = journal if journal is not None else _read_journal()
    settled = [b for b in _iter_bets(journal) if b.get("status") in _SETTLED_STATUSES]
    settled.sort(key=lambda b: (b.get("settled_at") or "", b.get("match_kickoff_at") or ""))
    return settled


def get_balance() -> float:
    """Current bankroll = initial + sum(settled profit)."""
    j = _read_journal()
    initial = float(j.get("metadata", {}).get("initial_bankroll", 1000.0))
    return round(initial + _settled_profit_sum(j), 2)


def get_peak() -> float:
    """Running maximum balance computed from journal in settled_at order."""
    j = _read_journal()
    initial = float(j.get("metadata", {}).get("initial_bankroll", 1000.0))
    running = initial
    peak = initial
    for b in _settled_chronological(j):
        running += float(b.get("profit") or 0)
        if running > peak:
            peak = running
    return round(peak, 2)


def get_lowest() -> float:
    """Running minimum balance computed from journal in settled_at order."""
    j = _read_journal()
    initial = float(j.get("metadata", {}).get("initial_bankroll", 1000.0))
    running = initial
    lowest = initial
    for b in _settled_chronological(j):
        running += float(b.get("profit") or 0)
        if running < lowest:
            lowest = running
    return round(lowest, 2)


def get_drawdown_pct() -> float:
    peak = get_peak()
    current = get_balance()
    return round((peak - current) / peak * 100, 2) if peak else 0.0


def get_roi(window: int | None = None) -> float:
    """Rolling ROI over the N most-recent-settled bets (by settled_at desc).

    If window is None: all-time ROI = (balance - initial) / initial * 100.
    If window is an int: sum(profit[-N])/sum(stake[-N])*100 — matches the
    semantics the health-monitor expects.
    """
    j = _read_journal()
    if window is None:
        initial = float(j.get("metadata", {}).get("initial_bankroll", 1000.0))
        current = get_balance()
        return round((current - initial) / initial * 100, 2) if initial else 0.0
    settled = _settled_chronological(j)
    slice_ = settled[-window:] if window > 0 else settled
    staked = sum(float(b.get("stake") or 0) for b in slice_)
    profit = sum(float(b.get("profit") or 0) for b in slice_)
    return round(profit / staked * 100, 2) if staked > 0 else 0.0


def get_history_view() -> list[dict]:
    """Return a list-shaped view of all settled bets for legacy consumers
    that expect history.json layout. Reads from journal only.
    """
    view = []
    for b in _settled_chronological():
        view.append({
            "id": b.get("_bet_id"),
            "match": b.get("match"),
            "date": b.get("date"),
            "market": b.get("market"),
            "selection": b.get("selection"),
            "odds": b.get("odds"),
            "stake": b.get("stake"),
            "status": b.get("status"),
            "profit": b.get("profit"),
            "result": b.get("result_score"),
            "settled_at": b.get("settled_at"),
            "match_kickoff_at": b.get("match_kickoff_at"),
            "confidence": b.get("confidence", "MEDIUM"),
            "value_pct": b.get("edge_pct") or b.get("value_pct") or 0,
        })
    return view


# ---------------------------------------------------------------------------
# Cache regeneration — bankroll.json + history.json are derived
# ---------------------------------------------------------------------------
def rebuild_caches() -> dict:
    """Regenerate bankroll.json and history.json from the journal.

    Lock order: journal_lock (outer) -> ledger_lock (inner). Every journal-
    mutating function in this module follows the same order to prevent
    deadlock with bet_journal.py's _with_journal_lock writers.

    Safe to call repeatedly; idempotent. Returns a summary dict.
    """
    from scripts.betting.bet_journal import journal_lock
    with journal_lock(), _ledger_lock():
        j = _read_journal()
        initial = float(j.get("metadata", {}).get("initial_bankroll", 1000.0))
        settled = _settled_chronological(j)
        running = initial
        peak = initial
        lowest = initial
        for b in settled:
            running += float(b.get("profit") or 0)
            if running > peak:
                peak = running
            if running < lowest:
                lowest = running
        current = round(running, 2)

        pending = [b for b in _iter_bets(j) if b.get("status") == "pending"]
        pending_stakes = round(sum(float(b.get("stake") or 0) for b in pending), 2)

        bankroll = {
            "initial_balance": round(initial, 2),
            "current_balance": current,
            "available_balance": round(current - pending_stakes, 2),
            "peak_balance": round(peak, 2),
            "lowest_balance": round(lowest, 2),
            "pending_bets": len(pending),
            "pending_stakes": pending_stakes,
            "updated_at": datetime.now().isoformat(),
        }
        _atomic_write(BANKROLL_PATH, bankroll)

        history = get_history_view()
        _atomic_write(HISTORY_PATH, history)

        return {
            "bankroll": bankroll,
            "history_rows": len(history),
            "pending": len(pending),
        }


# ---------------------------------------------------------------------------
# Write API — delegates to bet_journal and refreshes caches
# ---------------------------------------------------------------------------
def settle_bet(
    bet_id: str,
    status: str,
    result_score: str | None = None,
    profit: float | None = None,
    match_kickoff_at: str | None = None,
) -> bool:
    """Mark a pending bet as settled. Delegates to bet_journal.settle_bet
    then rebuilds the bankroll + history caches.
    """
    from scripts.betting.bet_journal import settle_bet as _journal_settle

    ok = _journal_settle(
        bet_id,
        status=status,
        result_score=result_score,
        profit=profit,
        match_kickoff_at=match_kickoff_at,
    )
    if ok:
        try:
            rebuild_caches()
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "ledger: cache rebuild failed after settle_bet %s: %s", bet_id, e
            )
    return ok


def supersede_bet(bet_id: str, new_bet_id: str | None = None) -> bool:
    """Mark a pending bet as superseded (replaced by a later re-bet).

    Superseded bets are EXCLUDED from P&L. The backlink helps audit; if the
    new_bet_id is unknown, leave it None. Lock order: journal_lock (outer)
    -> ledger_lock (inner).
    """
    from scripts.betting.bet_journal import journal_lock
    with journal_lock():
        j = _read_journal()
        bets = j.get("bets", {})
        if not isinstance(bets, dict) or bet_id not in bets:
            return False
        bet = bets[bet_id]
        if bet.get("status") != "pending":
            return False
        bet["status"] = "superseded"
        bet["superseded_at"] = datetime.now().isoformat()
        if new_bet_id:
            bet["superseded_by"] = new_bet_id
        md = j.setdefault("metadata", {})
        md["updated_at"] = datetime.now().isoformat()
        _atomic_write(JOURNAL_PATH, j)
    rebuild_caches()
    return True



def supersede_many(
    replacements: "list[tuple[str, str | None]]",
    reason: str | None = None,
) -> int:
    """Batch-supersede multiple bets in one locked write.

    replacements: list of (old_bet_id, new_bet_id_or_None) tuples.
    Every old bet must be status='pending' or the entry is skipped (never
    overwrites a settled bet).

    Returns the number of bets actually superseded. Rebuilds caches once at
    the end. Lock order: journal_lock (outer) -> ledger_lock (inner).
    """
    if not replacements:
        return 0
    marked = 0
    now_iso = datetime.now().isoformat()
    from scripts.betting.bet_journal import journal_lock
    with journal_lock():
        j = _read_journal()
        bets = j.get("bets", {})
        if not isinstance(bets, dict):
            return 0
        for old_id, new_id in replacements:
            bet = bets.get(old_id)
            if not bet or bet.get("status") != "pending":
                continue
            bet["status"] = "superseded"
            bet["superseded_at"] = now_iso
            if new_id:
                bet["superseded_by"] = new_id
            if reason:
                bet["superseded_reason"] = reason
            marked += 1
        if marked:
            j.setdefault("metadata", {})["updated_at"] = now_iso
            _atomic_write(JOURNAL_PATH, j)
    if marked:
        # rebuild_caches acquires its own (journal_lock, ledger_lock) pair
        rebuild_caches()
    return marked


# ---------------------------------------------------------------------------
# Settlement validation — shared by every settler to prevent the stale-
# commence_time bug (Odds API returned prior-season kickoff in 5 EPL bets
# on 2026-04-23).
# ---------------------------------------------------------------------------
def validate_commence_time(commence_iso: str, fallback_date: str | None = None,
                           stale_threshold_days: int = 365) -> tuple[str, str | None]:
    """Inspect commence_time from an external settlement source.

    Returns (date_yyyymmdd, trusted_commence_iso_or_None).
    If commence_time is more than `stale_threshold_days` old, returns the
    fallback_date and None (indicating the commence_time is untrustworthy
    and should not be stored as match_kickoff_at).
    """
    import logging
    _log = logging.getLogger(__name__)
    if not commence_iso:
        return (fallback_date or ""), None
    try:
        dt = datetime.fromisoformat(commence_iso.replace("Z", "+00:00"))
        days_back = (datetime.now() - dt.replace(tzinfo=None)).days
        if days_back > stale_threshold_days:
            _log.warning(
                "commence_time %s is %d days old — treating as stale cache. "
                "Using fallback_date=%s instead.",
                commence_iso, days_back, fallback_date,
            )
            return (fallback_date or commence_iso[:10]), None
    except (ValueError, TypeError) as e:
        _log.debug("commence_time parse failed: %s", e)
    return commence_iso[:10], commence_iso



# ---------------------------------------------------------------------------
# Invariant harness — confirms single-source-of-truth at any moment
# ---------------------------------------------------------------------------
def verify_invariants() -> dict:
    """Verify that journal (truth) ↔ bankroll.json (cache) ↔ history.json
    (cache) agree exactly.

    Returns a dict with:
      ok: bool — True iff every invariant passes
      violations: list[str] — human-readable descriptions of each failure
      journal, bankroll, history: computed snapshots for audit

    Callers (health-monitor, CLI, CI) should raise / alert when ok is False.
    """
    violations: list[str] = []

    # Source of truth: derive from journal directly
    j = _read_journal()
    initial = float(j.get("metadata", {}).get("initial_bankroll", 1000.0))
    settled = _settled_chronological(j)
    running = initial
    peak = initial
    lowest = initial
    for b in settled:
        running += float(b.get("profit") or 0)
        if running > peak:
            peak = running
        if running < lowest:
            lowest = running
    journal_current = round(running, 2)
    journal_peak = round(peak, 2)
    journal_lowest = round(lowest, 2)
    journal_settled_count = len(settled)

    # bankroll.json cache
    if BANKROLL_PATH.exists():
        try:
            bk = json.loads(BANKROLL_PATH.read_text())
        except Exception as e:
            violations.append(f"bankroll.json unreadable: {e}")
            bk = {}
    else:
        violations.append("bankroll.json missing")
        bk = {}

    if bk:
        if round(bk.get("current_balance", -1), 2) != journal_current:
            violations.append(
                f"bankroll.current_balance ({bk.get('current_balance')}) "
                f"!= journal-derived ({journal_current})"
            )
        if round(bk.get("peak_balance", -1), 2) != journal_peak:
            violations.append(
                f"bankroll.peak_balance ({bk.get('peak_balance')}) "
                f"!= journal-derived ({journal_peak})"
            )
        if round(bk.get("lowest_balance", -1), 2) != journal_lowest:
            violations.append(
                f"bankroll.lowest_balance ({bk.get('lowest_balance')}) "
                f"!= journal-derived ({journal_lowest})"
            )
        if round(bk.get("initial_balance", -1), 2) != round(initial, 2):
            violations.append(
                f"bankroll.initial_balance ({bk.get('initial_balance')}) "
                f"!= journal metadata ({initial})"
            )

    # history.json cache
    if HISTORY_PATH.exists():
        try:
            hist = json.loads(HISTORY_PATH.read_text())
        except Exception as e:
            violations.append(f"history.json unreadable: {e}")
            hist = []
    else:
        violations.append("history.json missing")
        hist = []

    if isinstance(hist, list):
        if len(hist) != journal_settled_count:
            violations.append(
                f"history.json rows ({len(hist)}) "
                f"!= journal settled count ({journal_settled_count})"
            )
        hist_profit = round(sum(float(b.get("profit") or 0) for b in hist), 2)
        journal_profit = round(_settled_profit_sum(j), 2)
        if hist_profit != journal_profit:
            violations.append(
                f"history.json profit sum ({hist_profit}) "
                f"!= journal profit sum ({journal_profit})"
            )
        # Date-quality invariants
        bad_dates = [b for b in hist if (b.get("date") or "").startswith(("2024", "2025"))]
        if bad_dates:
            violations.append(
                f"history.json has {len(bad_dates)} entries with pre-2026 dates "
                f"(settlement cache bug recurring?)"
            )
        # Duplicate-key invariant
        import re as _re
        def _nsel(s):
            s = (s or "").upper().strip().replace("OVER ", "O").replace("UNDER ", "U")
            return _re.sub(r"\s+", " ", s)
        def _nmkt(m):
            m = (m or "").lower().strip()
            if m in ("totals",) or m.startswith("o/u"):
                return "ou"
            if m == "1x2":
                return "h2h"
            return m
        from collections import Counter
        dup_keys = Counter(
            ((b.get("match") or "").strip(),
             (b.get("date") or "").strip(),
             _nmkt(b.get("market")),
             _nsel(b.get("selection")))
            for b in hist if b.get("status") in ("won", "lost", "push", "voided")
        )
        dups = [k for k, c in dup_keys.items() if c > 1]
        if dups:
            violations.append(
                f"history.json has {len(dups)} duplicate logical keys "
                f"(superseded bets leaking into history?)"
            )

    # Status-value invariant: no "void" (should always be canonicalized to "voided")
    legacy_void = [b for b in _iter_bets(j) if b.get("status") == "void"]
    if legacy_void:
        violations.append(
            f"journal has {len(legacy_void)} bets with legacy status='void' "
            f"(should be 'voided')"
        )

    # Superseded audit-trail invariant (only warn, not block): every
    # superseded bet after 2026-04-24 should carry superseded_at.
    missing_supat = [
        b for b in _iter_bets(j)
        if b.get("status") == "superseded"
        and not b.get("superseded_at")
        and (b.get("placed_at") or "") >= "2026-04-24"
    ]
    if missing_supat:
        violations.append(
            f"{len(missing_supat)} newer-than-2026-04-24 superseded bets lack "
            f"superseded_at (audit chain broken)"
        )

    return {
        "ok": not violations,
        "violations": violations,
        "journal": {
            "current_balance": journal_current,
            "peak_balance": journal_peak,
            "lowest_balance": journal_lowest,
            "initial_balance": round(initial, 2),
            "settled_count": journal_settled_count,
        },
        "bankroll": bk,
        "history_rows": len(hist) if isinstance(hist, list) else None,
    }
