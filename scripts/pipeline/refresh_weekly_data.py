"""Weekly data refresh orchestrator — runs every Monday at 04:00.

Refreshes all external data sources that feed the feature pipeline:
  1. Download any missing FBref match report HTMLs
  2. Re-parse FBref HTMLs into player_stats, lineups, events, goalkeeper_stats, shots
  3. Refresh Sofascore scrapes
  4. Refresh Understat scrapes
  5. Refresh referee assignments
  6. Backfill weather for new matches

All steps are offline (no Odds API cost). On failure for one source, other
sources still run (independent try/except per step).
"""
from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

from config.settings import get_current_season  # noqa: E402  (needs sys.path above)

log = logging.getLogger(__name__)

# Derived, not written down — this is a SCRAPE target, so the calendar season is
# correct even before matchweek 1. It used to be a hardcoded string with a
# "update this at the start of each new season" comment, which is exactly the
# kind of instruction that gets missed: the job keeps running and silently
# re-scrapes a finished season. If you need the season to ANALYSE rather than
# fetch, use latest_season_with_results() instead.
CURRENT_SEASON = get_current_season()


def run(cmd: list[str], step: str, timeout: int = 3600) -> bool:
    """Run a subprocess; log on success/failure; return True if exit 0."""
    log.info("=== %s ===", step)
    log.info("  cmd: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd, cwd=str(PROJECT), capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0:
            # Print last few lines of stdout for context
            tail = "\n".join(result.stdout.strip().splitlines()[-5:])
            log.info("  ✓ OK\n%s", tail)
            return True
        log.error("  ✗ FAILED (exit %d)\n%s", result.returncode, result.stderr[-2000:])
        return False
    except subprocess.TimeoutExpired:
        log.error("  ✗ TIMEOUT after 3600s")
        return False
    except Exception as e:
        log.error("  ✗ EXCEPTION: %s", e)
        return False


def step_watch_fbref_shots(py: str) -> bool:
    """Run the shots parser as a WATCH for good news, not as a required step.

    FBref removed the ``shots_all`` table from its match reports starting with
    2025-26. Confirmed source-side 2026-07-16 (absent from all 380 cached
    2025-26 reports, present in 2024-25, and a scrolled re-fetch came back
    larger and still lacked it), and re-measured 2026-08-25 against all 8
    cached 2026-27 reports: still absent. The parser itself is healthy —
    ``--season 2024-2025`` exits 0 on the same code path.

    It used to sit in the Step 3 loop, so this job exited 1 EVERY week on a
    dead upstream it can do nothing about. A permanently red signal is worse
    than no signal: it trains you to ignore the one week a real step breaks,
    because the failure looks identical. The exit code now reflects only the
    steps that can actually succeed.

    Nothing downstream is lost by not parsing it. ``parsed/shots.parquet`` has
    no live reader (DATA_CATALOG.md, "data/parsed/shots.parquet") — every shot
    feature reads ``all_shots_with_xg.parquet``, which Step 4b rebuilds from
    the Sofascore cache. What 2025-26+ genuinely loses is FBref's
    shot-creating-action chain and ``psxg_shot``, neither of which any current
    feature consumes.

    Returns True when the table is BACK, which is why this runs at all: it is
    cheap (reads already-cached HTML, no network) and it is the only thing
    that would ever tell us the upstream recovered. Logged loudly on purpose —
    a True here means go re-wire the shot-creation features.
    """
    ok = run(
        [py, "-m", "scripts.data.parse_all_shots", "--season", CURRENT_SEASON, "--append"],
        "Parse shots (watch — FBref dropped shots_all in 2025-26)",
    )
    if ok:
        log.warning(
            "  !! FBref shots_all appears to be BACK for %s — parsed/shots.parquet "
            "wrote rows. This has been dead since 2025-26; re-check whether the "
            "shot-creating-action features are worth re-wiring.", CURRENT_SEASON,
        )
    else:
        log.info(
            "  (expected) no shots_all table in %s reports — FBref removed it in "
            "2025-26. Not counted as a failure; shot features read "
            "all_shots_with_xg.parquet, rebuilt in Step 4b.", CURRENT_SEASON,
        )
    return ok


def step_refresh_fbref_fixtures() -> bool:
    """Refresh fbref fixtures.html so scrape_fbref_missing sees the latest match list."""
    log.info("=== FBref fixtures.html refresh ===")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("refresh_fixtures", PROJECT / "scripts" / "pipeline" / "_refresh_fbref_fixtures.py")
        # If that file doesn't exist, skip (we store it under scripts/pipeline/)
        if spec is None or not (PROJECT / "scripts" / "pipeline" / "_refresh_fbref_fixtures.py").exists():
            log.info("  skipped (no refresher script)")
            return True
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main(CURRENT_SEASON)
        log.info("  ✓ OK")
        return True
    except Exception as e:
        log.error("  ✗ %s", e)
        return False


def main() -> int:
    import time as _t
    _t0 = _t.time()
    log_path = PROJECT / "logs" / "weekly-data-refresh.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, mode="a"), logging.StreamHandler()],
    )

    log.info("")
    log.info("=" * 70)
    log.info("WEEKLY DATA REFRESH — %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 70)

    results = {}
    # Observations that must NOT gate the exit code — see step_watch_fbref_shots.
    watches: dict[str, bool] = {}
    py = sys.executable

    # --- Step 1: Refresh FBref fixtures.html (so we discover new matches) ---
    results["fbref_fixtures"] = step_refresh_fbref_fixtures()

    # --- Step 2: Download any new FBref match HTMLs (headless) ---
    # 300s, not the default 3600: FBref is Cloudflare-blocked weekly and this
    # step otherwise hangs the full hour for nothing — the Sofascore fallback
    # (step below) covers the data either way. Fail fast, move on.
    results["fbref_htmls"] = run(
        [py, "-m", "scripts.data.scrape_fbref_missing", "--season", CURRENT_SEASON, "--headless"],
        "FBref match HTML download",
        timeout=300,
    )

    # --- Step 3: Re-parse FBref into 4 parquets (all append current season) ---
    # parse_all_shots is deliberately NOT here — see step_watch_fbref_shots().
    for parser_name, label in [
        ("parse_all_player_stats", "player_stats"),
        ("parse_all_lineups", "lineups"),
        ("parse_all_events", "events"),
        ("parse_all_goalkeeper_stats", "goalkeeper_stats"),
    ]:
        results[f"parse_{label}"] = run(
            [py, "-m", f"scripts.data.{parser_name}", "--season", CURRENT_SEASON, "--append"],
            f"Parse {label}",
        )

    # --- Step 3b: shots — a WATCH, not a required step ---
    watches["shots_all_restored"] = step_watch_fbref_shots(py)

    # --- Step 4: Sofascore refresh (per active league) ---
    from config.leagues import ACTIVE_LEAGUES
    sofascore_results = {}
    for _league in ACTIVE_LEAGUES:
        sofascore_results[_league] = run(
            [py, "-m", "scripts.data.scrape_sofascore", "--league", _league, "--season", CURRENT_SEASON],
            f"Sofascore refresh ({_league})",
        )
    results["sofascore"] = all(sofascore_results.values())

    # --- Step 4b: Shot-level xG (re-derive all_shots_with_xg from the Sofascore cache) ---
    # Step 4 fetches every match's full shotmap but only persists the per-team AGGREGATE
    # (shotmap_stats.parquet). This rebuilds the shot-LEVEL file that features/shot_level_xg
    # and features/situational_xg read. Its original one-shot writer stopped in Feb 2026,
    # freezing 2025-26 coverage at 206/380 and sending 64 shot features to ~46% NaN — this
    # step keeps it current. Zero network: it reads the json Step 4 just cached.
    # all_shots_with_xg.parquet is Serie A only — the shot plugins read that single
    # file (features/shot_level_xg.py), and EPL matches have never been in it. EPL
    # shot features are a separate, un-built feature; looping ACTIVE_LEAGUES here
    # would only mint an all_shots_with_xg_premier_league.parquet that nothing reads.
    results["shot_level_xg"] = run(
        [py, "-m", "scripts.data.write_shot_level_xg", "--league", "serie_a", "--season", CURRENT_SEASON],
        "Shot-level xG rebuild (serie_a)",
    )

    # --- Step 5: Understat refresh (best-effort — scraper may fail on schema changes) ---
    try:
        from scraper.understat_scraper import scrape_understat_xg
        import pandas as pd
        df = scrape_understat_xg()
        if df is not None and len(df) > 0:
            for col in df.columns:
                try:
                    df[[col]].to_parquet("/dev/null")
                except Exception:
                    df = df.drop(columns=[col])
            df.to_parquet(PROJECT / "data" / "external" / "understat" / "matches_xg.parquet", index=False)
            log.info("  ✓ Understat: %d rows", len(df))
            results["understat"] = True
        else:
            log.warning("  Understat: no data returned")
            results["understat"] = False
    except Exception as e:
        log.error("  ✗ Understat: %s", e)
        results["understat"] = False

    # --- Step 6: Referee refresh (per active league; SA has scraper, EPL falls back) ---
    try:
        from scraper.referee import scrape_all_referee_assignments
        import pandas as pd
        ref_dfs = []
        for _league in ACTIVE_LEAGUES:
            cache = PROJECT / "data" / "external" / "referee" / f"referee_assignments_{_league}.parquet"
            if cache.exists():
                cache.unlink()
            try:
                new_df = scrape_all_referee_assignments(seasons=[CURRENT_SEASON], league=_league)
                if len(new_df) > 0:
                    ref_dfs.append(new_df)
                    log.info("  ✓ Referees (%s): %d rows", _league, len(new_df))
            except Exception as e:
                log.warning("  ✗ Referees (%s): %s", _league, e)
        if ref_dfs:
            combined_path = PROJECT / "data" / "external" / "referee" / "referee_assignments.parquet"
            new_combined = pd.concat(ref_dfs, ignore_index=True)
            if combined_path.exists():
                old = pd.read_parquet(combined_path)
                other = old[old["season"] != CURRENT_SEASON] if "season" in old.columns else old
                combined = pd.concat([other, new_combined], ignore_index=True)
            else:
                combined = new_combined
            combined.to_parquet(combined_path, index=False)
            log.info("  ✓ Referees: %d rows total", len(combined))
            results["referees"] = True
        else:
            results["referees"] = False
    except Exception as e:
        log.error("  ✗ Referees: %s", e)
        results["referees"] = False

    # --- Step 7: Weather backfill ---
    try:
        import pandas as pd
        from scraper.weather import fetch_weather_for_matches
        m = pd.read_parquet(PROJECT / "data" / "parsed" / "matches.parquet")
        m = m[m["league"].isin(ACTIVE_LEAGUES) & (m["season"] == CURRENT_SEASON)]
        w_path = PROJECT / "data" / "external" / "weather.parquet"
        w = pd.read_parquet(w_path) if w_path.exists() else pd.DataFrame()
        existing_ids = set(w["match_id"]) if len(w) else set()
        to_fetch = m[~m["match_id"].isin(existing_ids)]
        if len(to_fetch) > 0:
            log.info("  Weather: fetching %d new matches", len(to_fetch))
            new_w = fetch_weather_for_matches(to_fetch)
            if new_w is not None and len(new_w) > 0:
                combined = pd.concat([w, new_w], ignore_index=True).drop_duplicates(subset=["match_id"], keep="last")
                combined.to_parquet(w_path, index=False)
                log.info("  ✓ Weather: %d → %d rows", len(w), len(combined))
                results["weather"] = True
            else:
                results["weather"] = False
        else:
            log.info("  Weather: already up to date")
            results["weather"] = True
    except Exception as e:
        log.error("  ✗ Weather: %s", e)
        results["weather"] = False

    # --- Step 8: Rebuild match_id_mapping (derived, depends on Sofascore/Understat refresh) ---
    results["match_id_mapping"] = run(
        [py, "-m", "scripts.data.build_match_id_mapping"],
        "match_id_mapping rebuild",
    )

    # --- Step 9: Sofascore → FBref fallback (fills gaps when FBref is Cloudflare-blocked) ---
    # Runs AFTER mapping rebuild so sofa_id → canonical lookup is fresh.
    results["sofascore_fallback"] = run(
        [py, "-m", "scripts.data.fallback_sofascore_to_fbref", "--season", CURRENT_SEASON],
        "Sofascore → FBref fallback fill",
    )

    # --- Summary ---
    log.info("")
    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    for step, ok in results.items():
        log.info("  %s %s", "✓" if ok else "✗", step)
    n_ok = sum(results.values())
    n_total = len(results)
    log.info("  %d/%d steps OK", n_ok, n_total)
    # Printed, never counted: a watch firing false is the expected state.
    for name, fired in watches.items():
        log.info("  %s watch: %s", "!!" if fired else "--", name)

    # --- Scheduler notification ---
    try:
        from scripts.pipeline.notify import notify_scheduler_run
        failed_steps = [k for k, v in results.items() if not v]
        details = {"Steps OK": f"{n_ok}/{n_total}"}
        if failed_steps:
            details["Failed"] = ", ".join(failed_steps)
        status = "success" if n_ok == n_total else ("warn" if n_ok > 0 else "fail")
        notify_scheduler_run(
            name="weekly-data-refresh",
            status=status,
            duration_sec=_t.time() - _t0,
            details=details,
            error=None if n_ok == n_total else f"{n_total - n_ok} step(s) failed",
        )
    except Exception as e:
        log.debug("Notify failed: %s", e)

    return 0 if n_ok == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
