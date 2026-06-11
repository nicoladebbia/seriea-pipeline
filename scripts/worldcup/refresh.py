"""World Cup matchday refresh — the whole live loop in one command.

Steps (each fail-soft: a dead source skips, the rest proceed):
1. Re-download results.csv + goalscorers.csv (martj42 upstream)
2. Market odds refresh (skipped if fetched < ODDS_MIN_AGE_H ago)
3. Confirmed lineups + missing/doubtful players for fixtures within 48h
4. Played-match player stats appended to the scrape parquet
5. Player availability report (expected XI + team-news lambda factors)
6. Regenerate: team predictions (+ sim) -> player predictions -> track record
7. Kickstart the dashboard so it serves the new artifacts

Run: python3 -m scripts.worldcup.refresh [--force-odds]
Scheduled via ~/Library/LaunchAgents/com.seriea-pipeline.wc-refresh.plist
(every 2h; exits instantly outside the tournament window). RunAtLoad is
deliberately false — see the wake-storm lesson in CLAUDE.md.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from scripts.worldcup.engine import atomic_write_json, read_json_safe

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "worldcup"
STATE_JSON = DATA_DIR / "refresh_state.json"

TOURNAMENT_START = "2026-06-10"
TOURNAMENT_END = "2026-07-20"
ODDS_MIN_AGE_H = 5.0
PYTHON = sys.executable

CSV_SOURCES = {
    "international_results.csv": "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
    "international_goalscorers.csv": "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv",
}


def _log(msg: str) -> None:
    print(f"[{datetime.now(UTC).isoformat(timespec='seconds')}] {msg}", flush=True)


def _state() -> dict:
    # fail-soft: a truncated state file must never crash-loop the automation
    return dict(read_json_safe(STATE_JSON, {}))  # type: ignore[arg-type]


def _save_state(state: dict) -> None:
    atomic_write_json(STATE_JSON, state)


def _run(module: str, *args: str, timeout: int = 1200) -> bool:
    cmd = [PYTHON, "-m", module, *args]
    try:
        # noqa-justified: argv is a fixed list of our own module names, no
        # shell, no user input anywhere in the chain.
        r = subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        _log(f"{module} {' '.join(args)} -> FAILED ({type(exc).__name__}) — continuing")
        return False
    tail = (r.stdout + r.stderr).strip().splitlines()[-2:]
    _log(f"{module} {' '.join(args)} -> rc={r.returncode} | " + " | ".join(tail))
    return r.returncode == 0


def refresh_csvs() -> None:
    import urllib.request

    # A hung connection would otherwise block forever — and launchd never
    # starts overlapping same-label runs, so one hang = silence for the rest
    # of the tournament. 120s cap on every socket op in this download.
    socket.setdefaulttimeout(120)
    for fname, url in CSV_SOURCES.items():
        target = DATA_DIR / fname
        tmp = target.with_suffix(".tmp")
        try:
            urllib.request.urlretrieve(url, tmp)  # noqa: S310 — fixed https URLs
            if tmp.stat().st_size > 1_000_000:  # sanity: both files are MBs
                tmp.replace(target)
                _log(f"csv refreshed: {fname} ({target.stat().st_size:,} bytes)")
            else:
                _log(f"csv SKIPPED (suspiciously small): {fname}")
        except Exception as exc:  # noqa: BLE001 — fail-soft per step
            _log(f"csv refresh failed for {fname}: {exc}")
        finally:
            tmp.unlink(missing_ok=True)
    socket.setdefaulttimeout(None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-odds", action="store_true")
    args = parser.parse_args()

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    if not (TOURNAMENT_START <= today <= TOURNAMENT_END):
        _log(f"outside tournament window ({today}) — nothing to do")
        return

    _log("=== WC matchday refresh start ===")
    state = _state()
    refresh_csvs()

    last_odds = state.get("last_odds_fetch", "1970-01-01T00:00:00+00:00")
    age_h = (
        datetime.now(UTC) - datetime.fromisoformat(last_odds)
    ).total_seconds() / 3600
    if args.force_odds or age_h >= ODDS_MIN_AGE_H:
        if _run("scripts.worldcup.sofascore_fetch", "--odds"):
            state["last_odds_fetch"] = datetime.now(UTC).isoformat()
    else:
        _log(f"odds skipped (fetched {age_h:.1f}h ago, min {ODDS_MIN_AGE_H}h)")

    _run("scripts.worldcup.sofascore_fetch", "--lineups")
    _run("scripts.worldcup.sofascore_fetch", "--refresh-stats")
    # Same-night final scores — bridges martj42's publish lag so the
    # knockout fill and the result-pinned simulation see reality first.
    _run("scripts.worldcup.sofascore_fetch", "--results")
    _run("scripts.worldcup.availability")
    _run("scripts.worldcup.knockout")
    _run("scripts.worldcup.generate_predictions", "--sims", "10000")
    _run("scripts.worldcup.players")
    _run("scripts.worldcup.grading")
    _run("scripts.worldcup.combos")

    # Morning Telegram digest (slate + best combos): once per UTC day, not
    # before 07:00 UTC (09:00 Italy) — a post-midnight tick must not ping.
    today = datetime.now(UTC).date().isoformat()
    if state.get("last_wc_digest") != today and datetime.now(UTC).hour >= 7:
        if _run("scripts.pipeline.telegram_bot", "--wc-digest", timeout=120):
            state["last_wc_digest"] = today

    r = subprocess.run(  # noqa: S603 — fixed argv, absolute path
        ["/bin/launchctl", "kickstart", "-k",
         f"gui/{os.getuid()}/com.seriea-pipeline.web-dashboard"],
        capture_output=True, timeout=60,
    )
    _log(f"dashboard kickstart rc={r.returncode}")

    state["last_run"] = datetime.now(UTC).isoformat()
    _save_state(state)
    _log("=== WC matchday refresh done ===")


if __name__ == "__main__":
    main()
