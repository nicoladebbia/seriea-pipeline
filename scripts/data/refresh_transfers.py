"""Serie A transfer-window daily refresh.

Pulls the current 2026-27 window each day so the model's net-squad-delta
feature and the /transfers dashboard stay current:
1. Confirmed transfers  -> transfers_2026_2027.parquet   (feeds the model)
2. Squad market values   -> market_values_2026_2027.parquet (talent weights)
3. Rumors                -> rumors_2026_2027.parquet       (display-only)

Fail-soft: a dead/blocked source logs and is skipped; the rest proceed.
Window-gated: exits instantly outside the transfer window so a stray launchd
fire off-season costs nothing.

Run: python3 -m scripts.data.refresh_transfers [--season 2026-2027] [--force]
Scheduled via ~/Library/LaunchAgents/com.seriea-pipeline.transfer-refresh.plist
(daily 06:00; RunAtLoad deliberately false — wake-storm lesson in CLAUDE.md).
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime

# Serie A 2026-27 summer window (approx). Winter window is Jan; the feature only
# needs to be current when the season is live, so we cover summer + a winter tail.
WINDOW_RANGES = [
    ("2026-06-01", "2026-09-05"),   # summer window + a few days slack
    ("2027-01-01", "2027-02-05"),   # winter window
]


def _in_window(today: date) -> bool:
    for lo, hi in WINDOW_RANGES:
        if date.fromisoformat(lo) <= today <= date.fromisoformat(hi):
            return True
    return False


def _log(msg: str) -> None:
    print(f"[{datetime.now(UTC).isoformat()}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2026-2027")
    ap.add_argument("--league", default="serie_a")
    ap.add_argument("--force", action="store_true",
                    help="run even outside the transfer window")
    args = ap.parse_args()

    today = datetime.now(UTC).date()
    if not args.force and not _in_window(today):
        _log(f"outside transfer window ({today}) — skipping. Use --force to override.")
        return 0

    _log(f"=== Serie A transfer refresh start ({args.season}) ===")

    from scraper.transfermarkt import (
        current_league_teams,
        scrape_rumors,
        scrape_squad_market_values,
        scrape_transfers,
    )

    teams = current_league_teams(args.season, args.league)
    if teams:
        _log(f"resolved {len(teams)} current {args.league} clubs")
    else:
        _log("could not resolve current clubs — falling back to full team map")

    # 1. Confirmed transfers (feeds the model)
    try:
        df = scrape_transfers(season=args.season, league=args.league, only_teams=teams)
        _log(f"confirmed transfers: {len(df)} rows")
    except Exception as e:  # noqa: BLE001 — fail-soft; one dead source must not kill the rest
        _log(f"confirmed transfers FAILED: {type(e).__name__}: {e}")

    # 2. Squad market values (talent weights for the delta)
    try:
        mv = scrape_squad_market_values(
            season=args.season, league=args.league, only_teams=teams
        )
        _log(f"market values: {len(mv)} rows")
    except Exception as e:  # noqa: BLE001 — fail-soft
        _log(f"market values FAILED: {type(e).__name__}: {e}")

    # 3. Rumors (display-only — NEVER feeds the model)
    try:
        rm = scrape_rumors(season=args.season, league=args.league, only_teams=teams)
        _log(f"rumors: {len(rm)} rows")
    except Exception as e:  # noqa: BLE001
        _log(f"rumors FAILED: {type(e).__name__}: {e}")

    # 4. Second source — Wikipedia (exact date + independent cross-check). Serie A
    #    only; display + data-quality, never fed to the model directly.
    try:
        from scraper.wiki_transfers import enrich_transfers_with_wiki, scrape_wiki_transfers
        wk = scrape_wiki_transfers(season=args.season)
        _log(f"wikipedia transfers: {len(wk)} rows")
        # 5. Merge Wikipedia's date + confidence flag onto the TM spine.
        enriched = enrich_transfers_with_wiki(season=args.season)
        if enriched is not None and "n_sources" in enriched.columns:
            dual = int((enriched["n_sources"] == 2).sum())
            _log(f"cross-source enrichment: {dual}/{len(enriched)} TM rows dual-confirmed")
    except Exception as e:  # noqa: BLE001 — a dead second source must not kill the refresh
        _log(f"wikipedia/enrichment FAILED: {type(e).__name__}: {e}")

    # 6. Change detection — diff the fresh squad against the last snapshot and log
    #    every signing / departure / value-change / contract-change. First run
    #    seeds the snapshot and logs nothing (no cold-start phantom signings).
    try:
        from scripts.data.transfer_change_detector import detect_changes
        changes = detect_changes(season=args.season)
        if changes:
            _log(f"CHANGES DETECTED: {len(changes)} — " + "; ".join(
                f"{c['type']}:{c.get('player')}" for c in changes[:6]
            ) + (" …" if len(changes) > 6 else ""))
        else:
            _log("no squad changes since last snapshot")
    except Exception as e:  # noqa: BLE001 — change log is best-effort, never blocks
        _log(f"change detection FAILED: {type(e).__name__}: {e}")

    _log("=== Serie A transfer refresh done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
