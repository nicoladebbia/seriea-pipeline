#!/usr/bin/env python3
"""AUTO-SETTLE ORCHESTRATOR — Fetch results, settle bets, track P&L, alert on drift.

Single entry point for daily post-match processing. Designed for cron or manual use.

What it does:
  1. Fetches completed match scores via Odds API
  2. Auto-settles pending bets in the journal
  3. Appends settlement snapshot to P&L time-series (pnl_history.json)
  4. Computes rolling performance metrics (ROI, win rate, CLV)
  5. Compares rolling ROI against backtest baselines
  6. Generates a daily report with alert flags

Usage:
    python -m scripts.betting.auto_settle              # Fetch & settle & report
    python -m scripts.betting.auto_settle --report      # Report only (no API call)
    python -m scripts.betting.auto_settle --days 5      # Look back 5 days
    python -m scripts.betting.auto_settle --dry-run     # Show what would settle, don't write

API Cost: 1-2 credits per call (same as results_fetcher)
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ─── Paths ───
BETTING_DIR = DATA_DIR / "betting"
PNL_HISTORY_FILE = BETTING_DIR / "pnl_history.json"
DRIFT_ALERTS_FILE = BETTING_DIR / "drift_alerts.json"

# ─── Backtest baselines (from docs/baselines.md) ───
BASELINE_ROI = 12.7          # 2-season backtest ROI %
BASELINE_ACCURACY = 53.8     # 2-season backtest accuracy %
ROI_WARN_THRESHOLD = 5.0     # Warn if rolling ROI drops below this
ROI_ALERT_THRESHOLD = 0.0    # Alert if rolling ROI drops below this
MIN_BETS_FOR_ALERT = 20      # Don't alert until enough bets for statistical relevance
ROLLING_WINDOW = 50          # Rolling window for metrics (last N bets)


# =============================================================================
# P&L HISTORY
# =============================================================================

def _load_pnl_history() -> List[Dict]:
    """Load P&L time-series."""
    if PNL_HISTORY_FILE.exists():
        try:
            with open(PNL_HISTORY_FILE) as f:
                return json.load(f)
        except Exception as e:
            log.warning("Failed to load P&L history: %s", e)
    return []


def _save_pnl_history(history: List[Dict]):
    """Save P&L time-series."""
    PNL_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PNL_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def append_pnl_snapshot(settlement_summary: Dict, stats: Dict) -> Dict:
    """Append a snapshot to the P&L time-series after settlement.

    Each snapshot records the state of the betting system at a point in time,
    enabling cumulative P&L charts and performance tracking.
    """
    history = _load_pnl_history()

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "date": datetime.now().strftime("%Y-%m-%d"),
        # Settlement details
        "settled_this_run": settlement_summary.get("settled", 0),
        "won_this_run": settlement_summary.get("won", 0),
        "lost_this_run": settlement_summary.get("lost", 0),
        "push_this_run": settlement_summary.get("push", 0),
        "profit_this_run": settlement_summary.get("profit", 0.0),
        # Cumulative state
        "total_bets": stats.get("total_bets", 0),
        "total_settled": stats.get("settled", 0),
        "total_pending": stats.get("pending", 0),
        "total_won": stats.get("won", 0),
        "total_lost": stats.get("lost", 0),
        "total_pushes": stats.get("pushes", 0),
        "total_staked": stats.get("total_staked", 0.0),
        "cumulative_profit": stats.get("total_profit", 0.0),
        "roi_pct": stats.get("roi_pct", 0.0),
        "clv_avg_pct": stats.get("clv_avg_pct", 0.0),
        # Balance
        "balance": settlement_summary.get("new_balance", 0.0),
    }

    history.append(snapshot)
    _save_pnl_history(history)
    return snapshot


# =============================================================================
# ROLLING METRICS
# =============================================================================

def compute_rolling_metrics(window: int = ROLLING_WINDOW) -> Dict:
    """Compute rolling performance over the last N settled bets.

    Returns rolling ROI, win rate, avg edge, avg CLV, and streak info.
    """
    try:
        from scripts.betting.bet_journal import get_settled_bets
        settled = get_settled_bets()
    except Exception:
        return {"error": "Could not load settled bets"}

    if not settled:
        return {"n_bets": 0, "message": "No settled bets"}

    recent = settled[-window:]
    n = len(recent)

    total_staked = sum(b.get("stake", 0) or 0 for b in recent)
    total_profit = sum(b.get("profit", 0) or 0 for b in recent)
    wins = sum(1 for b in recent if b.get("status") == "won")
    losses = sum(1 for b in recent if b.get("status") == "lost")
    pushes = sum(1 for b in recent if b.get("status") == "push")

    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
    decided = wins + losses
    win_rate = (wins / decided * 100) if decided > 0 else 0.0

    # Average edge at time of bet
    edges = [b.get("edge_pct", 0) or 0 for b in recent if b.get("edge_pct")]
    avg_edge = (sum(edges) / len(edges)) if edges else 0.0

    # CLV
    clv_vals = [b.get("clv_pct", 0) for b in recent if b.get("clv_pct") is not None]
    avg_clv = (sum(clv_vals) / len(clv_vals)) if clv_vals else None
    positive_clv_pct = (
        sum(1 for c in clv_vals if c > 0) / len(clv_vals) * 100
    ) if clv_vals else None

    # Current streak
    streak_type = None
    streak_len = 0
    for b in reversed(recent):
        s = b.get("status")
        if s == "push":
            continue
        if streak_type is None:
            streak_type = s
            streak_len = 1
        elif s == streak_type:
            streak_len += 1
        else:
            break

    # By market
    by_market = {}
    for b in recent:
        m = b.get("market", "unknown")
        if m not in by_market:
            by_market[m] = {"bets": 0, "profit": 0.0, "staked": 0.0}
        by_market[m]["bets"] += 1
        by_market[m]["profit"] += b.get("profit", 0) or 0
        by_market[m]["staked"] += b.get("stake", 0) or 0
    for m in by_market:
        s = by_market[m]["staked"]
        by_market[m]["roi"] = round(by_market[m]["profit"] / s * 100, 1) if s > 0 else 0

    return {
        "window": window,
        "n_bets": n,
        "total_staked": round(total_staked, 2),
        "total_profit": round(total_profit, 2),
        "roi_pct": round(roi, 2),
        "win_rate": round(win_rate, 1),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "avg_edge": round(avg_edge, 2),
        "avg_clv": round(avg_clv, 2) if avg_clv is not None else None,
        "positive_clv_pct": round(positive_clv_pct, 1) if positive_clv_pct is not None else None,
        "streak": f"{streak_len}{streak_type[0].upper()}" if streak_type else "—",
        "by_market": by_market,
    }


# =============================================================================
# DRIFT DETECTION
# =============================================================================

def check_performance_drift(rolling: Dict) -> List[Dict]:
    """Compare rolling metrics against backtest baselines.

    Returns list of alert dicts: {level, metric, actual, baseline, message}
    Levels: INFO, WARNING, CRITICAL
    """
    alerts = []
    n_bets = rolling.get("n_bets", 0)

    if n_bets < MIN_BETS_FOR_ALERT:
        alerts.append({
            "level": "INFO",
            "metric": "sample_size",
            "message": f"Only {n_bets} settled bets — need {MIN_BETS_FOR_ALERT}+ for drift alerts",
        })
        return alerts

    roi = rolling.get("roi_pct", 0)

    # ROI drift
    if roi < ROI_ALERT_THRESHOLD:
        alerts.append({
            "level": "CRITICAL",
            "metric": "roi",
            "actual": roi,
            "baseline": BASELINE_ROI,
            "threshold": ROI_ALERT_THRESHOLD,
            "message": f"Rolling ROI {roi:+.1f}% is NEGATIVE (baseline: {BASELINE_ROI}%)",
        })
    elif roi < ROI_WARN_THRESHOLD:
        alerts.append({
            "level": "WARNING",
            "metric": "roi",
            "actual": roi,
            "baseline": BASELINE_ROI,
            "threshold": ROI_WARN_THRESHOLD,
            "message": f"Rolling ROI {roi:+.1f}% below warning threshold {ROI_WARN_THRESHOLD}% (baseline: {BASELINE_ROI}%)",
        })
    else:
        alerts.append({
            "level": "INFO",
            "metric": "roi",
            "actual": roi,
            "baseline": BASELINE_ROI,
            "message": f"Rolling ROI {roi:+.1f}% — on track (baseline: {BASELINE_ROI}%)",
        })

    # Win rate drift
    win_rate = rolling.get("win_rate", 0)
    decided = rolling.get("wins", 0) + rolling.get("losses", 0)
    if decided >= 20:
        # For 1X2 bets, backtest win rate was ~46% at 5% edge threshold
        # Below 35% is concerning (more than 2 SD below expected for small samples)
        if win_rate < 35.0:
            alerts.append({
                "level": "WARNING",
                "metric": "win_rate",
                "actual": win_rate,
                "message": f"Win rate {win_rate:.1f}% is low (expected ~46%)",
            })

    # CLV drift — positive CLV is the #1 indicator of genuine edge
    avg_clv = rolling.get("avg_clv")
    if avg_clv is not None:
        if avg_clv < -2.0:
            alerts.append({
                "level": "CRITICAL",
                "metric": "clv",
                "actual": avg_clv,
                "message": f"Average CLV {avg_clv:+.1f}% — betting into market moves, likely no real edge",
            })
        elif avg_clv < 0:
            alerts.append({
                "level": "WARNING",
                "metric": "clv",
                "actual": avg_clv,
                "message": f"Average CLV {avg_clv:+.1f}% — slightly negative, monitor closely",
            })
        else:
            alerts.append({
                "level": "INFO",
                "metric": "clv",
                "actual": avg_clv,
                "message": f"Average CLV {avg_clv:+.1f}% — positive (good sign)",
            })

    # Losing streak detection
    streak = rolling.get("streak", "")
    if streak and streak.endswith("L"):
        try:
            streak_len = int(streak[:-1])
            if streak_len >= 8:
                alerts.append({
                    "level": "WARNING",
                    "metric": "streak",
                    "actual": streak_len,
                    "message": f"Current losing streak of {streak_len} — consider pausing if > 10",
                })
        except ValueError:
            pass

    return alerts


def _save_drift_alerts(alerts: List[Dict]):
    """Persist drift alerts for external monitoring."""
    data = {
        "checked_at": datetime.now().isoformat(),
        "alerts": alerts,
        "has_critical": any(a["level"] == "CRITICAL" for a in alerts),
        "has_warning": any(a["level"] == "WARNING" for a in alerts),
    }
    DRIFT_ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DRIFT_ALERTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


# =============================================================================
# DAILY REPORT
# =============================================================================

def print_daily_report(
    settlement_summary: Optional[Dict],
    rolling: Dict,
    alerts: List[Dict],
    pnl_snapshot: Optional[Dict] = None,
):
    """Print a concise daily report to stdout."""
    print()
    print("=" * 60)
    print("  DAILY BETTING REPORT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Settlement results
    if settlement_summary and settlement_summary.get("settled", 0) > 0:
        s = settlement_summary
        print(f"\n  TODAY'S SETTLEMENTS: {s['settled']} bets")
        print(f"    Won: {s['won']}  Lost: {s['lost']}  Push: {s.get('push', 0)}")
        profit = s.get("profit", 0)
        print(f"    P&L: {'+'if profit >= 0 else ''}${profit:.2f}")
        print(f"    Balance: ${s.get('new_balance', 0):.2f}")
    elif settlement_summary:
        pending = settlement_summary.get("pending", 0)
        print(f"\n  No bets settled today ({pending} still pending)")
    else:
        print("\n  (Report only — no settlement run)")

    # Rolling performance
    n = rolling.get("n_bets", 0)
    if n > 0:
        print(f"\n  ROLLING PERFORMANCE (last {rolling.get('window', ROLLING_WINDOW)} bets, actual {n}):")
        print(f"    ROI:      {rolling['roi_pct']:+.1f}%  (baseline: {BASELINE_ROI}%)")
        print(f"    Win Rate: {rolling['win_rate']:.1f}%")
        push_str = f"-{rolling['pushes']}P" if rolling.get('pushes') else ''
        print(f"    Record:   {rolling['wins']}W-{rolling['losses']}L{push_str}")
        print(f"    Profit:   {'+'if rolling['total_profit'] >= 0 else ''}"
              f"${rolling['total_profit']:.2f} on ${rolling['total_staked']:.2f} staked")
        print(f"    Avg Edge: {rolling['avg_edge']:.1f}%")
        if rolling.get("avg_clv") is not None:
            print(f"    Avg CLV:  {rolling['avg_clv']:+.1f}%  "
                  f"({rolling.get('positive_clv_pct', 0):.0f}% positive)")
        else:
            print(f"    Avg CLV:  N/A (no closing odds tracked)")
        print(f"    Streak:   {rolling.get('streak', '—')}")

        # By market
        by_market = rolling.get("by_market", {})
        if by_market:
            print(f"\n    By Market:")
            for m, d in sorted(by_market.items(), key=lambda x: -x[1]["bets"]):
                print(f"      {m:<8}  {d['bets']:>3} bets  ROI: {d['roi']:+.1f}%  "
                      f"P&L: {'+'if d['profit'] >= 0 else ''}${d['profit']:.2f}")

    # Alerts
    critical = [a for a in alerts if a["level"] == "CRITICAL"]
    warnings = [a for a in alerts if a["level"] == "WARNING"]
    infos = [a for a in alerts if a["level"] == "INFO"]

    if critical or warnings:
        print(f"\n  {'!'*50}")
        print(f"  ALERTS:")
        for a in critical:
            print(f"    [CRITICAL] {a['message']}")
        for a in warnings:
            print(f"    [WARNING]  {a['message']}")
        print(f"  {'!'*50}")
    else:
        print(f"\n  STATUS: All metrics within acceptable range")
        for a in infos:
            if a["metric"] != "sample_size":
                print(f"    [OK] {a['message']}")

    # P&L trend (from history)
    pnl_history = _load_pnl_history()
    if len(pnl_history) >= 3:
        recent_3 = pnl_history[-3:]
        profits = [s.get("profit_this_run", 0) for s in recent_3]
        dates = [s.get("date", "?") for s in recent_3]
        print(f"\n  RECENT RUNS:")
        for d, p in zip(dates, profits):
            emoji = "+" if p >= 0 else ""
            print(f"    {d}: {emoji}${p:.2f}")

    print()
    print("=" * 60)


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def run(
    days_from: int = 3,
    dry_run: bool = False,
    report_only: bool = False,
) -> Dict:
    """Main orchestration: settle → track → check → report.

    Args:
        days_from: Days to look back for results (API param)
        dry_run: If True, show what would settle without writing
        report_only: If True, skip settlement, just show current state

    Returns:
        Dict with settlement_summary, rolling_metrics, drift_alerts
    """
    settlement_summary = None
    pnl_snapshot = None
    completed_results = {}  # Shared across steps for parlay settlement

    # ── Step 1: Settle ──
    if not report_only:
        try:
            from scripts.data.results_fetcher import fetch_scores, parse_scores, save_results, settle_bets

            log.info("Fetching scores (last %d days)...", days_from)
            raw_scores = fetch_scores(days_from=days_from)

            if raw_scores:
                results = parse_scores(raw_scores)
                completed = {k: v for k, v in results.items() if v.get("completed")}
                completed_results = completed
                log.info("Found %d completed matches", len(completed))

                if completed:
                    save_results(results)

                    if dry_run:
                        # Show what would settle without writing
                        from scripts.betting.bet_journal import get_pending_bets
                        pending = get_pending_bets()
                        would_settle = []
                        for bet in pending:
                            match_key = bet.get("match", "")
                            if match_key in completed:
                                would_settle.append(bet)
                        settlement_summary = {
                            "settled": 0,
                            "would_settle": len(would_settle),
                            "pending": len(pending),
                            "dry_run": True,
                        }
                        if would_settle:
                            log.info("DRY RUN — would settle %d bets:", len(would_settle))
                            for b in would_settle:
                                r = completed[b["match"]]
                                log.info("  %s | %s @ %.2f | Score: %d-%d",
                                         b["match"], b.get("selection", "?"),
                                         b.get("odds", 0),
                                         r["home_score"], r["away_score"])
                    else:
                        settlement_summary = settle_bets(completed)
                else:
                    settlement_summary = {"settled": 0, "pending": 0, "message": "No completed matches"}
            else:
                settlement_summary = {"settled": 0, "message": "No scores from API"}

        except Exception as e:
            log.error("Settlement failed: %s", e)
            settlement_summary = {"settled": 0, "error": str(e)}

    # ── Step 2: Track P&L ──
    try:
        from scripts.betting.bet_journal import get_journal_stats
        stats = get_journal_stats()

        if settlement_summary and settlement_summary.get("settled", 0) > 0 and not dry_run:
            pnl_snapshot = append_pnl_snapshot(settlement_summary, stats)
            log.info("P&L snapshot saved (cumulative: $%.2f, ROI: %.1f%%)",
                     stats.get("total_profit", 0), stats.get("roi_pct", 0))
            # Update bankroll.json from journal (derived, not source of truth)
            try:
                from scripts.betting.bankroll_loader import compute_current_bankroll, update_bankroll_json
                balance_info = compute_current_bankroll()
                update_bankroll_json(balance_info)
            except Exception as e:
                log.warning("Failed to update bankroll.json: %s", e)
    except Exception as e:
        log.warning("P&L tracking failed: %s", e)
        stats = {}

    # ── Step 2b: Settle parlays ──
    if completed_results and not dry_run:
        try:
            from scripts.betting.parlay_tracker import settle_parlays
            n_parlay_settled = settle_parlays(completed_results)
            if n_parlay_settled > 0:
                log.info("Settled %d parlay(s) in tracker", n_parlay_settled)
        except Exception as e:
            log.warning("Parlay tracker settlement failed: %s", e)

    # ── Step 3: Rolling metrics ──
    rolling = compute_rolling_metrics()

    # ── Step 4: Drift check ──
    alerts = check_performance_drift(rolling)
    _save_drift_alerts(alerts)

    # ── Step 5: Report ──
    print_daily_report(settlement_summary, rolling, alerts, pnl_snapshot)

    return {
        "settlement": settlement_summary,
        "rolling": rolling,
        "alerts": alerts,
        "pnl_snapshot": pnl_snapshot,
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto-settle orchestrator: settle bets, track P&L, check drift"
    )
    parser.add_argument("--days", type=int, default=3,
                        help="Days to look back for results (default: 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would settle without writing")
    parser.add_argument("--report", action="store_true",
                        help="Report only — skip settlement, show current state")
    args = parser.parse_args()

    result = run(
        days_from=args.days,
        dry_run=args.dry_run,
        report_only=args.report,
    )

    # Exit code: non-zero if critical alerts
    if any(a["level"] == "CRITICAL" for a in result.get("alerts", [])):
        sys.exit(2)
    elif any(a["level"] == "WARNING" for a in result.get("alerts", [])):
        sys.exit(1)
    sys.exit(0)
