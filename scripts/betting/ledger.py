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
    get_metrics(journal=None, now=None) -> dict   # THE payload every surface renders;
                                          nothing outside this module recomputes a number
    ensure_initial_bankroll_metadata()  # persist initial_bankroll once (no more 1000 literal)
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
from datetime import UTC, datetime, timedelta
from datetime import date as _date
from pathlib import Path
from typing import Any, Iterable, Iterator
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
JOURNAL_PATH = DATA_DIR / "betting" / "bet_journal.json"
BANKROLL_PATH = DATA_DIR / "betting" / "bankroll.json"
HISTORY_PATH = DATA_DIR / "betting" / "history.json"
_LEDGER_LOCK_PATH = DATA_DIR / "betting" / ".ledger.lock"
# Legacy snapshot written by features/bankroll_manager.py — NOT a ledger cache,
# but readers still open it, so verify_invariants() checks it until retired.
STATE_JSON_PATH = DATA_DIR / "bankroll" / "state.json"
# External alert sources consumed by get_metrics()
HEALTH_STATUS_PATH = DATA_DIR / "monitoring" / "health_status.json"
CANDIDATES_PATH = DATA_DIR / "upcoming" / "betting_candidates.json"
T30_MARKER_PATH = DATA_DIR / "pipeline" / "t30_ticket_state.json"

# One betting day = one fixture-calendar day in Italy. Bets belong to the match
# they are on (the journal `date` field), never to the clock that settled them.
DAY_TZ = "Europe/Rome"
DEFAULT_INITIAL_BANKROLL = 1000.0  # documented default; its use is an ALERT, never silent
METRICS_DEFINITIONS_VERSION = 1

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
# get_metrics() — THE one computation every surface renders from
# ---------------------------------------------------------------------------
def _initial_from(journal: dict) -> tuple[float, bool]:
    """(initial_bankroll, found). Absence is reported by the caller as an alert."""
    md = journal.get("metadata", {}) or {}
    v = md.get("initial_bankroll")
    if v is None:
        return DEFAULT_INITIAL_BANKROLL, False
    return float(v), True


def _bet_day(b: dict) -> str | None:
    """Betting-day key: the match `date` (fixture calendar). Fallbacks keep
    old rows countable, never the settlement clock first."""
    for k in ("date", "match_kickoff_at", "settled_at"):
        v = b.get(k)
        if v:
            return str(v)[:10]
    return None


def _parse_dt(v: str | None) -> datetime | None:
    if not v:
        return None
    try:
        dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _period(rows: list[dict]) -> dict:
    won = sum(1 for b in rows if b.get("status") == "won")
    lost = sum(1 for b in rows if b.get("status") == "lost")
    return {"n": len(rows), "won": won, "lost": lost,
            "pnl": round(sum(float(b.get("profit") or 0) for b in rows), 2),
            "staked": round(sum(float(b.get("stake") or 0) for b in rows), 2)}


def _group(rows: list[dict], key: str) -> dict:
    out: dict[str, dict] = {}
    for b in rows:
        k = str(b.get(key) or "unknown")
        g = out.setdefault(k, {"n": 0, "won": 0, "lost": 0, "push": 0, "voided": 0,
                               "staked": 0.0, "profit": 0.0})
        g["n"] += 1
        g["won"] += b.get("status") == "won"
        g["lost"] += b.get("status") == "lost"
        g["push"] += b.get("status") == "push"
        g["voided"] += b.get("status") in ("voided", "void")
        g["staked"] += float(b.get("stake") or 0)
        g["profit"] += float(b.get("profit") or 0)
    for g in out.values():
        g["staked"] = round(g["staked"], 2)
        g["profit"] = round(g["profit"], 2)
        g["roi_pct"] = round(g["profit"] / g["staked"] * 100, 2) if g["staked"] else 0.0
    return out


def _risk_gate_alerts(metrics: dict) -> list[dict]:
    """Risk gates as alerts. Lazy import: risk_controls -> bankroll_loader -> ledger."""
    try:
        from scripts.betting.risk_controls import check_risk_gates
        gates = check_risk_gates(bankroll=metrics["bankroll"]["current"])
    except (ImportError, OSError, ValueError, KeyError, TypeError) as e:
        # never let an alert source take the payload down
        return [{"level": "WARNING", "source": "risk", "message": f"risk gates unavailable: {e}"}]
    if gates.get("allow_betting", True):
        return []
    return [{"level": "CRITICAL", "source": "risk",
             "message": f"BETTING PAUSED — {gates.get('reason') or 'risk gate tripped'}"}]


def _health_alerts() -> list[dict]:
    if not HEALTH_STATUS_PATH.exists():
        return []
    try:
        h = json.loads(HEALTH_STATUS_PATH.read_text())
    except (OSError, ValueError) as e:
        return [{"level": "WARNING", "source": "health", "message": f"health_status.json unreadable: {e}"}]
    out = []
    for item in h.get("issues", []) or []:
        try:
            level, msg = item[0], item[1]
        except (IndexError, TypeError, KeyError):
            continue
        out.append({"level": str(level), "source": "health", "message": str(msg)})
    return out


def _settlement_lag_alerts(pending: list[dict], now: datetime, grace_hours: float = 3.0) -> list[dict]:
    """A pending bet whose match finished more than `grace_hours` ago is money
    the settlement chain has not closed out."""
    out = []
    for b in pending:
        ko = _parse_dt(b.get("match_kickoff_at"))
        if ko and now - ko > timedelta(hours=grace_hours):
            hrs = (now - ko).total_seconds() / 3600
            out.append({"level": "WARNING", "source": "settlement_lag",
                        "message": f"{b.get('match')} kicked off {hrs:.0f}h ago, bet still pending"})
    return out


def _funnel(now: datetime, today: str) -> tuple[dict, list[dict]]:
    """T-30 conversion funnel for today: candidates -> tickets -> fills.
    Alerts once any candidate's T-30 moment has passed with zero tickets —
    the exact shape of the 2026-08-28 miss."""
    funnel = {"date": today, "candidates_n": 0, "tickets_n": 0, "filled_n": 0}
    alerts: list[dict] = []
    cands: list[dict] = []
    try:
        if CANDIDATES_PATH.exists():
            d = json.loads(CANDIDATES_PATH.read_text())
            rows = d.get("bets") or d.get("candidates") or []
            cands = [c for c in rows if not c.get("date") or str(c.get("date"))[:10] == today]
    except (OSError, ValueError, AttributeError):
        cands = []
    funnel["candidates_n"] = len(cands)
    try:
        if T30_MARKER_PATH.exists():
            mk = json.loads(T30_MARKER_PATH.read_text())
            if mk.get("date") == today:
                funnel["tickets_n"] = len(mk.get("bets", {}) or {})
    except (OSError, ValueError, AttributeError):
        funnel["tickets_n"] = 0  # unreadable marker reads as "no tickets"
    if funnel["candidates_n"] and funnel["tickets_n"] == 0:
        kos = [_parse_dt(c.get("commence_time") or c.get("_kickoff") or c.get("kickoff")) for c in cands]
        kos = [k for k in kos if k]
        if kos and now >= min(kos) - timedelta(minutes=30):
            alerts.append({"level": "CRITICAL", "source": "t30_funnel",
                           "message": f"T-30 funnel: {funnel['candidates_n']} candidate(s) today, "
                                      f"0 ticket(s) sent after the first T-30 moment"})
    return funnel, alerts


def get_metrics(
    journal: dict | None = None,
    now: datetime | None = None,
    rolling_window: int = 50,
    period_days: int = 7,
    include_alerts: bool = True,
) -> dict:
    """Every money/performance number the system shows, computed ONCE.

    Definitions (see .plans/ledger-metrics-plan.md, 2026-08-28):
      ROI            = profit / stake (pushes and voids count in turnover at 0 profit).
                       bankroll_growth_pct = (current - initial) / initial — a different
                       number, never labelled ROI.
      rolling        = the last `rolling_window` settled bets by settled_at, `rolling_n`
                       says how many actually exist.
      betting day    = the match `date` (fixture calendar, Europe/Rome). today /
                       last_betting_day / last_7_days all use it. Settlement clocks
                       never move a bet between days.
      win rate       = won / (won + lost) — decisive only, named so.
      streaks        = streak_decisive (pushes/voids skipped; the alert) and
                       non_win_run (pushes/voids extend; the risk gate) — two names,
                       one source, so they can no longer disagree about "the streak".
      peak / lowest  = along the settled_at-ordered equity curve; readers never
                       max/min them with live values. drawdown_pct is current
                       (off peak), max_drawdown_pct is historical.
      CLV            = per-bet clv_pct only, never a summary cache.
      fill tier      = verified P&L recomputed at filled_odds for fill_status=placed.
    `journal` and `now` are injectable so tests freeze every clock.
    """
    j = journal if journal is not None else _read_journal()
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    rome = ZoneInfo(DAY_TZ)
    today_d: _date = now.astimezone(rome).date()
    today = today_d.isoformat()

    initial, initial_found = _initial_from(j)
    settled = _settled_chronological(j)
    pending = [b for b in _iter_bets(j) if b.get("status") == "pending"]

    # --- equity curve -------------------------------------------------------
    running = peak = lowest = initial
    max_dd = 0.0
    for b in settled:
        running += float(b.get("profit") or 0)
        peak = max(peak, running)
        lowest = min(lowest, running)
        if peak > 0:
            max_dd = max(max_dd, (peak - running) / peak * 100)
    current = round(running, 2)
    pending_stakes = round(sum(float(b.get("stake") or 0) for b in pending), 2)
    bankroll = {
        "initial": round(initial, 2),
        "current": current,
        "available": round(current - pending_stakes, 2),
        "pending_stakes": pending_stakes,
        "pending_n": len(pending),
        "peak": round(peak, 2),
        "lowest": round(lowest, 2),
        "drawdown_pct": round((peak - current) / peak * 100, 2) if peak else 0.0,
        "max_drawdown_pct": round(max_dd, 2),
        "bankroll_growth_pct": round((current - initial) / initial * 100, 2) if initial else 0.0,
    }

    # --- ROI (on stake) -------------------------------------------------------
    staked = sum(float(b.get("stake") or 0) for b in settled)
    profit = sum(float(b.get("profit") or 0) for b in settled)
    window = settled[-rolling_window:] if rolling_window > 0 else settled
    w_staked = sum(float(b.get("stake") or 0) for b in window)
    w_profit = sum(float(b.get("profit") or 0) for b in window)
    roi = {
        "all_time_pct": round(profit / staked * 100, 2) if staked else 0.0,
        "all_time_n": len(settled),
        "all_time_staked": round(staked, 2),
        "all_time_profit": round(profit, 2),
        "rolling_pct": round(w_profit / w_staked * 100, 2) if w_staked else 0.0,
        "rolling_n": len(window),
        "rolling_window": rolling_window,
        "by_market": _group(settled, "market"),
        "by_league": _group(settled, "league"),
    }

    # --- record ---------------------------------------------------------------
    counts = {s: sum(1 for b in settled if b.get("status") == s) for s in _SETTLED_STATUSES}
    decisive = counts["won"] + counts["lost"]
    record = {**counts, "settled_n": len(settled),
              "win_rate_decisive": round(counts["won"] / decisive * 100, 2) if decisive else 0.0}

    # --- streaks (settled_at order, most recent last) -------------------------
    streak_decisive = 0
    streak_loss = 0.0
    for b in reversed(settled):
        s = b.get("status")
        if s not in ("won", "lost"):
            continue
        if streak_decisive == 0:
            streak_decisive = 1 if s == "won" else -1
        elif (s == "won") == (streak_decisive > 0):
            streak_decisive += 1 if s == "won" else -1
        else:
            break
        if s == "lost":
            streak_loss += -float(b.get("profit") or 0)
    non_win_run = 0
    for b in reversed(settled):
        if b.get("status") == "won":
            break
        non_win_run += 1
    streak = {"streak_decisive": streak_decisive,
              "streak_loss_eur": round(streak_loss, 2) if streak_decisive < 0 else 0.0,
              "non_win_run": non_win_run}

    # --- periods (betting day = match date, Europe/Rome) ----------------------
    by_day: dict[str, list[dict]] = {}
    for b in settled:
        d = _bet_day(b)
        if d:
            by_day.setdefault(d, []).append(b)
    days = [(today_d - timedelta(days=i)).isoformat() for i in range(period_days - 1, -1, -1)]
    per_day = [{"date": d, **_period(by_day.get(d, []))} for d in days]
    last7_rows = [b for d in days for b in by_day.get(d, [])]
    last_day = max(by_day) if by_day else None
    periods = {
        "today": {"date": today, **_period(by_day.get(today, []))},
        "last_betting_day": ({"date": last_day, **_period(by_day[last_day])} if last_day
                             else {"date": None, **_period([])}),
        "last_7_days": {"from": days[0], "to": days[-1], **_period(last7_rows), "per_day": per_day},
    }

    # --- CLV (per-bet values only) -------------------------------------------
    clv_rows = [b for b in settled if b.get("clv_pct") is not None]
    clv_vals = [float(b["clv_pct"]) for b in clv_rows]
    clv = {
        "n": len(clv_vals),
        "avg_pct": round(sum(clv_vals) / len(clv_vals), 2) if clv_vals else None,
        "positive_rate": round(sum(1 for v in clv_vals if v > 0) / len(clv_vals) * 100, 2) if clv_vals else None,
        "by_market": {},
    }
    for mk in {str(b.get("market")) for b in clv_rows}:
        vals = [float(b["clv_pct"]) for b in clv_rows if str(b.get("market")) == mk]
        clv["by_market"][mk] = {"n": len(vals), "avg_pct": round(sum(vals) / len(vals), 2)}

    # --- fill tier (verified at filled odds) ---------------------------------
    annotated = [b for b in settled if b.get("fill_status")]
    placed = [b for b in annotated if b.get("fill_status") == "placed"]
    v_staked = v_profit = 0.0
    for b in placed:
        stake = float(b.get("stake") or 0)
        fo = float(b.get("filled_odds") or b.get("odds") or 0)
        v_staked += stake
        if b.get("status") == "won":
            v_profit += stake * (fo - 1)
        elif b.get("status") == "lost":
            v_profit -= stake
    fill = {
        "verified_n": len(placed),
        "verified_roi_pct": round(v_profit / v_staked * 100, 2) if v_staked else None,
        "verified_profit": round(v_profit, 2),
        "fill_rate": round(len(placed) / len(annotated) * 100, 2) if annotated else None,
        "missed_n": sum(1 for b in annotated if b.get("fill_status") == "missed"),
        "unverified_n": sum(1 for b in annotated if b.get("fill_status") == "unverified"),
    }

    metrics = {
        "bankroll": bankroll, "roi": roi, "record": record, "streak": streak,
        "periods": periods, "clv": clv, "fill": fill,
        "funnel": {"date": today, "candidates_n": 0, "tickets_n": 0, "filled_n": 0},
        "alerts": [],
        "meta": {"computed_at": now.isoformat(), "day_tz": DAY_TZ,
                 "definitions_version": METRICS_DEFINITIONS_VERSION,
                 "rolling_window": rolling_window, "period_days": period_days},
    }

    # --- alerts: one list ----------------------------------------------------
    alerts: list[dict] = []
    if not initial_found:
        alerts.append({"level": "WARNING", "source": "ledger",
                       "message": f"journal metadata has no initial_bankroll — using the "
                                  f"{DEFAULT_INITIAL_BANKROLL:.0f} default (run ensure_initial_bankroll_metadata)"})
    alerts += _settlement_lag_alerts(pending, now)
    funnel, funnel_alerts = _funnel(now, today)
    metrics["funnel"] = funnel
    alerts += funnel_alerts
    if include_alerts:
        alerts += _risk_gate_alerts(metrics)
        alerts += _health_alerts()
    metrics["alerts"] = alerts
    return metrics


def ensure_initial_bankroll_metadata(default: float = DEFAULT_INITIAL_BANKROLL) -> bool:
    """Persist initial_bankroll into journal metadata if absent. Returns True
    if it wrote. Idempotent; never overwrites an existing value."""
    from scripts.betting.bet_journal import journal_lock
    with journal_lock(), _ledger_lock():
        j = _read_journal()
        md = j.setdefault("metadata", {})
        if md.get("initial_bankroll") is not None:
            return False
        md["initial_bankroll"] = float(default)
        md["initial_bankroll_set_at"] = datetime.now(UTC).isoformat()
        _atomic_write(JOURNAL_PATH, j)
        return True


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

        # state.json: legacy snapshot some readers still hold (Phase 3 retires
        # it). Refreshed here so the ledger is its ONLY writer.
        if STATE_JSON_PATH.exists():
            try:
                st = json.loads(STATE_JSON_PATH.read_text())
            except (OSError, ValueError):
                st = {}
            st["current_bankroll"] = current
            st["peak_bankroll"] = round(peak, 2)
            st["initial_bankroll"] = round(initial, 2)
            st["last_updated"] = datetime.now().isoformat()
            _atomic_write(STATE_JSON_PATH, st)

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

    # Legacy snapshot bankroll/state.json — not a ledger cache, but still read by
    # web/app.py, web/advisor.py, parlay_generator, learning_loop (2026-08-28 it
    # said 1015.35 / 1278.06 against journal 1024.17 / 1253.62 and nobody noticed).
    if STATE_JSON_PATH.exists():
        try:
            st = json.loads(STATE_JSON_PATH.read_text())
        except (OSError, ValueError) as e:
            violations.append(f"bankroll/state.json unreadable: {e}")
            st = {}
        for key, truth in (("current_bankroll", journal_current), ("peak_bankroll", journal_peak)):
            if key in st and round(float(st.get(key) or 0), 2) != truth:
                violations.append(
                    f"bankroll/state.json {key} ({st.get(key)}) != journal-derived ({truth})"
                )

    # initial_bankroll must be recorded, not defaulted
    if (j.get("metadata", {}) or {}).get("initial_bankroll") is None:
        violations.append("journal metadata lacks initial_bankroll (defaulting silently to 1000)")

    # The payload every surface renders must agree with this derivation
    try:
        m = get_metrics(journal=j, include_alerts=False)
        if m["bankroll"]["current"] != journal_current or m["bankroll"]["peak"] != journal_peak:
            violations.append(
                f"get_metrics() bankroll ({m['bankroll']['current']}/{m['bankroll']['peak']}) "
                f"!= invariant derivation ({journal_current}/{journal_peak})"
            )
    except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:
        violations.append(f"get_metrics() failed: {e}")

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
