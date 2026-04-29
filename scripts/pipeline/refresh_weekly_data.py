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

log = logging.getLogger(__name__)

CURRENT_SEASON = "2025-2026"  # Update this at start of each new season


def run(cmd: list[str], step: str) -> bool:
    """Run a subprocess; log on success/failure; return True if exit 0."""
    log.info("=== %s ===", step)
    log.info("  cmd: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd, cwd=str(PROJECT), capture_output=True, text=True, timeout=3600,
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
    py = sys.executable

    # --- Step 1: Refresh FBref fixtures.html (so we discover new matches) ---
    results["fbref_fixtures"] = step_refresh_fbref_fixtures()

    # --- Step 2: Download any new FBref match HTMLs (headless) ---
    results["fbref_htmls"] = run(
        [py, "-m", "scripts.data.scrape_fbref_missing", "--season", CURRENT_SEASON, "--headless"],
        "FBref match HTML download",
    )

    # --- Step 3: Re-parse FBref into 5 parquets (all append current season) ---
    for parser_name, label in [
        ("parse_all_player_stats", "player_stats"),
        ("parse_all_lineups", "lineups"),
        ("parse_all_events", "events"),
        ("parse_all_goalkeeper_stats", "goalkeeper_stats"),
        ("parse_all_shots", "shots"),
    ]:
        results[f"parse_{label}"] = run(
            [py, "-m", f"scripts.data.{parser_name}", "--season", CURRENT_SEASON, "--append"],
            f"Parse {label}",
        )

    # --- Step 4: Sofascore refresh ---
    results["sofascore"] = run(
        [py, "-m", "scripts.data.scrape_sofascore", "--league", "serie_a", "--season", CURRENT_SEASON],
        "Sofascore refresh",
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

    # --- Step 6: Referee refresh (remove cache to force re-scrape) ---
    try:
        cache = PROJECT / "data" / "external" / "referee" / "referee_assignments_serie_a.parquet"
        if cache.exists():
            cache.unlink()
        from scraper.referee import scrape_all_referee_assignments
        import pandas as pd
        new_df = scrape_all_referee_assignments(seasons=[CURRENT_SEASON], league="serie_a")
        if len(new_df) > 0:
            combined_path = PROJECT / "data" / "external" / "referee" / "referee_assignments.parquet"
            if combined_path.exists():
                old = pd.read_parquet(combined_path)
                other = old[old["season"] != CURRENT_SEASON] if "season" in old.columns else old
                combined = pd.concat([other, new_df], ignore_index=True)
            else:
                combined = new_df
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
        m = m[(m["league"] == "serie_a") & (m["season"] == CURRENT_SEASON)]
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
