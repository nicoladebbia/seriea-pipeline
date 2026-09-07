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

# Window months, derived from the season -- NOT written as dates. The literals
# here used to be ("2026-06-01", "2026-09-05") and ("2027-01-01", "2027-02-05"):
# correct for exactly one season, then silently wrong, and the failure mode is a
# job that skips forever while its log says nothing is wrong. Same annual fuse as
# the season-in-a-filename trap in CLAUDE.md.
#   summer: 1 Jun -> 5 Sep of the season's FIRST year (a few days of slack past
#           deadline day), winter: 1 Jan -> 5 Feb of its SECOND year.
_SUMMER = ((6, 1), (9, 5))
_WINTER = ((1, 1), (2, 5))


def _window_ranges(season: str) -> list[tuple[date, date]]:
    """Concrete (start, end) dates for `season` ("2026-2027")."""
    y1, y2 = (int(part) for part in season.split("-")[:2])
    (sm, sd), (em, ed) = _SUMMER
    (wm, wd), (xm, xd) = _WINTER
    return [
        (date(y1, sm, sd), date(y1, em, ed)),
        (date(y2, wm, wd), date(y2, xm, xd)),
    ]


def _in_window(today: date, season: str = "2026-2027") -> bool:
    return any(lo <= today <= hi for lo, hi in _window_ranges(season))


def _log(msg: str) -> None:
    print(f"[{datetime.now(UTC).isoformat()}] {msg}", flush=True)


def _run_change_detection(args) -> None:
    """Diff the fresh squad against the last snapshot and log every
    signing / departure / value-change / contract-change. First run seeds the
    snapshot and logs nothing (no cold-start phantom signings).

    Called on BOTH paths: value and contract changes do not stop when the
    window closes, and this is what /rosters renders as "Recent Changes".
    """
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2026-2027")
    ap.add_argument("--league", default="serie_a")
    ap.add_argument("--force", action="store_true",
                    help="run even outside the transfer window")
    args = ap.parse_args()

    today = datetime.now(UTC).date()
    # Outside the window, squad MEMBERSHIP is frozen -- that is what a closed
    # window means -- but market values and contract dates keep moving, and the
    # change detector reports both. Skipping the whole run froze /rosters'
    # "Recent Changes" from deadline day until January (observed: the panel's
    # newest entry was 2026-09-02 while the job ran twice a day saying nothing
    # was wrong) and left the squad-value model features on a stale snapshot for
    # four months. So: full run inside the window, squad + change detection only
    # outside it. The window-specific sources (transfers, rumors, Wikipedia,
    # Capology) are the expensive ones and they genuinely have nothing to say.
    in_window = args.force or _in_window(today, args.season)
    if in_window:
        _log(f"=== Serie A transfer refresh start ({args.season}) ===")
    else:
        _log(f"=== Serie A squad-only refresh ({args.season}) — "
             f"outside transfer window ({today}); use --force for the full run ===")

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

    # 2. Squad market values (talent weights for the delta).
    #    Runs in EVERY mode: values and contract dates move year-round, and the
    #    change detector below is what feeds /rosters' "Recent Changes".
    try:
        mv = scrape_squad_market_values(
            season=args.season, league=args.league, only_teams=teams
        )
        _log(f"market values: {len(mv)} rows")
    except Exception as e:  # noqa: BLE001 — fail-soft
        _log(f"market values FAILED: {type(e).__name__}: {e}")

    if not in_window:
        # Squad + change detection only. Everything below this point is
        # window-specific: no confirmed transfers are registered, no rumors are
        # live, and Wikipedia's window page is closed.
        _run_change_detection(args)
        _log("=== Serie A squad-only refresh done ===")
        return 0

    # 1. Confirmed transfers (feeds the model)
    try:
        df = scrape_transfers(season=args.season, league=args.league, only_teams=teams)
        _log(f"confirmed transfers: {len(df)} rows")
    except Exception as e:  # noqa: BLE001 — fail-soft; one dead source must not kill the rest
        _log(f"confirmed transfers FAILED: {type(e).__name__}: {e}")

    # 3. Rumors (display-only — NEVER feeds the model).
    #    rumors_<season>.parquet is OVERWRITTEN each run, so it is
    #    survivorship-biased and useless retrospectively. record_run() folds the
    #    same rows into the append-only rumor_history.parquet, plus a per-club
    #    coverage log so a later absence means "dropped", not "scraper blind".
    coverage: dict[str, str] = {}
    try:
        rm = scrape_rumors(season=args.season, league=args.league,
                           only_teams=teams, coverage=coverage)
        _log(f"rumors: {len(rm)} rows")
    except Exception as e:  # noqa: BLE001
        _log(f"rumors FAILED: {type(e).__name__}: {e}")
        rm = None
    try:
        from scripts.data.rumor_history import record_run
        summ = record_run(rm, coverage, season=args.season, league=args.league)
        _log(f"rumor history: +{summ['new_rumors']} new, {summ['updated_rumors']} refreshed, "
             f"{summ['total_tracked']} tracked, coverage "
             f"{summ['teams_covered']}/{summ['teams_expected']} ({summ['status']})")
    except Exception as e:  # noqa: BLE001 — history is additive; never block the refresh
        _log(f"rumor history FAILED: {type(e).__name__}: {e}")

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

    # 5b. Capology salary ESTIMATES (display-only, /rosters). Capology 429-throttles
    #     a fast sweep and wages change per-signing not per-day, so this runs WEEKLY
    #     (Mondays) — not on every twice-daily fire — unless --force. Its own parquet;
    #     NEVER a model feature; the number is a labeled estimate, never official.
    if args.force or today.weekday() == 0:  # Monday
        try:
            from scraper.capology_salaries import save_salaries
            sal = save_salaries(season=args.season)
            if not sal.empty:
                _log(f"capology salaries: {len(sal)} players across {sal['team'].nunique()} clubs")
            else:
                _log("capology salaries: nothing scraped (throttled/blocked) — kept prior parquet")
        except Exception as e:  # noqa: BLE001 — a dead salary source must not kill the refresh
            _log(f"capology salaries FAILED: {type(e).__name__}: {e}")
    else:
        _log("capology salaries: skipped (weekly step, runs Mondays; use --force to override)")

    _run_change_detection(args)

    _log("=== Serie A transfer refresh done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
