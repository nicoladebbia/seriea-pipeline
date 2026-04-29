#!/usr/bin/env python3
"""FULL BETTING PIPELINE RUNNER - 32-Step Intelligence Pipeline

Runs the complete betting AI pipeline with all models and intelligence systems.

Pipeline Steps:
 1. Fetch multi-market odds
 2. Sync matches from Odds API
 3. Refresh squad rosters (API-Football / Football-Data.org)
 4. Save per-bookmaker data
 5. Bookmaker analysis (sharp/soft divergence)
 6. Track odds movement (enhanced with per-bookmaker snapshots)
 7. Cross-market correlation analysis
 8. Market intelligence aggregation
 9. Fetch confirmed lineups (pre-kickoff)
10. Fetch injury data
11. Ensemble predictions (enhanced with confirmed lineups)
11b. Formation predictions (SofaScore position-based)
11c. Match stats predictions (shots, cards, possession, fouls)
12. Generate current standings
13. Generate H2H records
14. Extended markets (double chance, team totals, exact score, etc.)
15. Player props (goalscorer predictions)
16. Sentiment analysis (Perplexity AI)
17. Player analysis
18. Intelligence adjustments (enhanced with market intel)
19. Over/Under model
20. Handicap model
21. Cards model
22. BTTS & Corners model
23. Betting engine
24. Advanced multi-market engine
25. Parlay generator (correlation-adjusted multi-leg bets)
26. AI reasoning for bets (GPT-4o-mini)
27. Italian market standards
28. Results, auto-settle & CLV tracking
29. Feedback analysis & weight optimization
30. Performance dashboard

Usage:
    python scripts/run_full_pipeline.py                 # Full run
    python scripts/run_full_pipeline.py --quick         # Skip API calls (use cache)
    python scripts/run_full_pipeline.py --bankroll 2000 # Custom bankroll
    python scripts/run_full_pipeline.py --snapshot-only # Only fetch + snapshot (Steps 1-6, ~6 credits)
    python scripts/run_full_pipeline.py --pre-kickoff   # Pre-kickoff mode (~25s, confirmed lineups)
    python scripts/run_full_pipeline.py --live-monitor  # Live monitor loop (every 15 min)
    python scripts/run_full_pipeline.py --live-once     # Single live poll (2 API calls)
    python scripts/run_full_pipeline.py --live-status   # Show live monitoring status
"""

import fcntl
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load .env before any module that needs API keys
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass  # dotenv not required if env vars set externally

from config.settings import DATA_DIR
from scripts.utils.logging_config import (
    PipelineLogger,
    log_pipeline_start,
    log_pipeline_end,
    log_step,
    log_step_complete,
)

# Initialize centralized logging
log = PipelineLogger("pipeline")


def _get_json_encoder():
    """Get numpy-safe JSON encoder."""
    import numpy as np

    class _Encoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.generic)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    return _Encoder


def _validate_data_gate(name: str, data, min_threshold: int, required: bool = True) -> bool:
    """Check data completeness after a pipeline step.

    Returns True if data passes. If required=True and data is insufficient,
    raises RuntimeError to abort the pipeline.
    """
    import pandas as pd
    if data is None:
        count = 0
    elif isinstance(data, dict):
        count = len(data)
    elif isinstance(data, (list, pd.DataFrame)):
        count = len(data)
    else:
        count = 1 if data else 0

    if count < min_threshold:
        msg = f"Data gate FAILED: {name} has {count} items (min: {min_threshold})"
        if required:
            log.critical(msg)
            raise RuntimeError(msg)
        else:
            log.warning(msg)
            return False
    log.info("Data gate OK: %s has %d items", name, count)
    return True


def _update_placed_bets_log(settlement: dict):
    """Update the persistent bet log with settlement results."""
    log_path = DATA_DIR / "betting" / "placed_bets_log.json"
    if not log_path.exists():
        return

    try:
        with open(log_path) as f:
            bet_log = json.load(f)
    except json.JSONDecodeError as e:
        log.warning("Corrupt bet log JSON at %s: %s", log_path, e)
        return
    except OSError as e:
        log.warning("Cannot read bet log at %s: %s", log_path, e)
        return

    # Load history to cross-reference settled bets
    history_path = DATA_DIR / "betting" / "history.json"
    settled_keys = set()
    if history_path.exists():
        try:
            with open(history_path) as f:
                history = json.load(f)
            hist_list = (history if isinstance(history, list)
                         else history.get("settled_bets", history.get("bets", [])))
            for h in hist_list:
                key = (h.get("match", ""), h.get("market", ""), h.get("selection", ""))
                settled_keys.add((key, h.get("status", "")))
        except Exception as e:
            log.warning(f"Failed to load bet history for settlement: {e}")

    updated = 0
    for bet in bet_log:
        if bet.get("status") != "pending":
            continue
        key = (bet.get("match", ""), bet.get("market", ""), bet.get("selection", ""))
        for skey, status in settled_keys:
            if skey == key and status in ("won", "lost"):
                bet["status"] = status
                bet["settled_at"] = datetime.now().isoformat()
                updated += 1
                break

    if updated > 0:
        with open(log_path, "w") as f:
            json.dump(bet_log, f, indent=2)


def _archive_bets(bets: list, generated_at: str):
    """Archive recommended bets to a persistent log.

    The unified_report.json gets overwritten each pipeline run, so bets
    must be saved to a durable log for settlement tracking.
    Appends only new bets (deduped by match+market+selection).
    """
    archive_path = DATA_DIR / "betting" / "placed_bets_log.json"
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if archive_path.exists():
        try:
            with open(archive_path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = []

    # Index existing bets by key for fast lookup
    existing_by_key = {}
    for i, b in enumerate(existing):
        key = (b.get("match", ""), b.get("market", ""), b.get("selection", ""), b.get("date", ""))
        existing_by_key[key] = i

    new_count = 0
    updated_count = 0
    for bet in bets:
        key = (bet.get("match", ""), bet.get("market", ""), bet.get("selection", ""), bet.get("date", ""))
        if key not in existing_by_key:
            bet_record = {**bet, "recorded_at": generated_at, "status": "pending"}
            existing.append(bet_record)
            existing_by_key[key] = len(existing) - 1
            new_count += 1
        else:
            # Update odds/stake if changed (only for pending bets)
            idx = existing_by_key[key]
            old = existing[idx]
            if old.get("status") == "pending":
                old_odds = old.get("odds", old.get("fair_odds", 0))
                new_odds = bet.get("odds", bet.get("fair_odds", 0))
                if old_odds != new_odds:
                    old["odds"] = new_odds
                    old["stake"] = bet.get("stake", old.get("stake"))
                    old["updated_at"] = generated_at
                    updated_count += 1

    changed = new_count + updated_count
    if changed > 0:
        with open(archive_path, "w") as f:
            json.dump(existing, f, indent=2)
        parts = []
        if new_count:
            parts.append(f"{new_count} new")
        if updated_count:
            parts.append(f"{updated_count} updated")
        print(f"  Archived {' + '.join(parts)} bets to persistent log ({len(existing)} total)")

    # Also write to unified journal + mark stale bets from previous runs
    try:
        from scripts.betting.bet_journal import add_bet as journal_add_bet, _load_journal
        from scripts.betting.bet_journal import _generate_bet_id
        from scripts.betting import ledger as _ledger

        # Compute logical-key -> new_bet_id for the supersede backlink
        current_bet_ids: set = set()
        logical_key_to_new_id: dict = {}
        for bet in bets:
            bid = _generate_bet_id(bet.get("date", ""), bet.get("match", ""),
                                   bet.get("market", ""), bet.get("selection", ""))
            current_bet_ids.add(bid)
            logical_key = (bet.get("date", ""), bet.get("match", ""),
                           bet.get("market", ""), bet.get("selection", ""))
            logical_key_to_new_id[logical_key] = bid

        # Read-only snapshot to build the supersede list; the actual mutation
        # is delegated to ledger.supersede_many which owns the lock + rebuild.
        journal_snapshot = _load_journal()
        current_bet_dates = {bet.get("date", "") for bet in bets if bet.get("date")}
        replacements: list = []
        for bid, b in journal_snapshot.get("bets", {}).items():
            if (isinstance(b, dict) and b.get("status") == "pending"
                    and bid not in current_bet_ids
                    and b.get("date", "") in current_bet_dates):
                lk = (b.get("date", ""), b.get("match", ""),
                      b.get("market", ""), b.get("selection", ""))
                replacements.append((bid, logical_key_to_new_id.get(lk)))

        stale_count = _ledger.supersede_many(
            replacements, reason="replaced_by_newer_pipeline_output"
        )

        journal_count = 0
        for bet in bets:
            journal_add_bet({
                "match": bet.get("match", ""),
                "date": bet.get("date", ""),
                "market": bet.get("market", ""),
                "selection": bet.get("selection", ""),
                "model_prob": bet.get("model_prob", bet.get("our_probability")),
                "edge_pct": bet.get("edge_pct", bet.get("value_pct")),
                "odds": bet.get("best_odds", bet.get("odds")),
                "stake": bet.get("stake_amount", bet.get("stake")),
                "confidence": bet.get("confidence_tier", bet.get("confidence")),
                "factors": bet.get("factors"),
                "placed_at": generated_at,
                "pipeline_status": "current",
            })
            journal_count += 1
        if journal_count:
            print(f"  Journal: recorded {journal_count} bets ({stale_count} old marked stale)")
    except Exception as e:
        pass  # Don't break pipeline if journal fails


def banner(text: str):
    """Print a banner."""
    width = 70
    print("\n" + "=" * width)
    print(f" {text}")
    print("=" * width)


import time as _time
_step_times = {}

def step(num: int, total: int, name: str):
    """Print step indicator with timing of previous step."""
    # Log duration of previous step
    if _step_times:
        prev_name, prev_start = _step_times.get("_last", ("", 0))
        if prev_start > 0:
            elapsed = _time.time() - prev_start
            if elapsed > 30:
                log.info(f"Step '{prev_name}' took {elapsed:.1f}s (SLOW)")
            elif elapsed > 5:
                log.info(f"Step '{prev_name}' took {elapsed:.1f}s")
    _step_times["_last"] = (name, _time.time())
    print(f"\n[{num}/{total}] {name}...")
    log.info(f"Starting step {num}/{total}: {name}")


def _is_data_stale(filepath: Path, max_age_hours: float) -> bool:
    """Check if a file/directory is older than max_age_hours.

    For directories, checks the newest file inside.
    """
    if not filepath.exists():
        return True
    try:
        if filepath.is_dir():
            files = list(filepath.glob("*"))
            if not files:
                return True
            newest = max(f.stat().st_mtime for f in files if f.is_file())
            mtime = datetime.fromtimestamp(newest)
        else:
            mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
        age = datetime.now() - mtime
        return age > timedelta(hours=max_age_hours)
    except Exception:
        return True


def _run_market_analysis(odds: dict, summary: dict):
    """Run bookmaker analysis, odds tracking, cross-market, and market intelligence.

    These are all LOCAL computations on already-fetched odds data — zero API credits.
    They produce the bookmaker_analysis.json, odds_movement.json, cross_market_signals.json,
    and market_intelligence.json files that the dashboard reads.
    """
    # Bookmaker analysis (sharp vs soft consensus)
    print(f"    Bookmaker analysis...")
    try:
        from features.bookmaker_analysis import run_bookmaker_analysis
        ba = run_bookmaker_analysis()
        if ba:
            sharp = sum(1 for m in ba.values() if m.get("has_sharp_data"))
            print(f"      {len(ba)} matches ({sharp} with sharp data)")
            summary["bookmaker_analysis_updated"] = True
        else:
            print(f"      No bookmaker data")
    except Exception as e:
        print(f"      Bookmaker analysis warning: {e}")
        log.debug(f"Incremental bookmaker analysis: {e}")

    # Odds movement tracking
    print(f"    Odds movement...")
    try:
        from scripts.data.odds_tracker import save_snapshot, analyze_movements
        if odds:
            simple_odds = {}
            for mk, md in odds.items():
                h2h = md.get("h2h")
                if h2h:
                    simple_odds[mk] = {"home": h2h["home"], "draw": h2h["draw"], "away": h2h["away"]}
            if simple_odds:
                save_snapshot(simple_odds)
                movements = analyze_movements(simple_odds)
                n_steam = sum(1 for m in movements.values() if m.get("is_steam_move"))
                print(f"      {len(movements)} matches tracked{f', {n_steam} STEAM' if n_steam else ''}")
                summary["odds_movement_updated"] = True
    except Exception as e:
        print(f"      Odds movement warning: {e}")
        log.debug(f"Incremental odds movement: {e}")

    # Cross-market analysis
    print(f"    Cross-market signals...")
    try:
        from features.cross_market_analysis import run_cross_market_analysis
        cm = run_cross_market_analysis()
        if cm:
            anomalies = sum(1 for r in cm.values() if r.get("has_anomaly"))
            print(f"      {len(cm)} matches{f', {anomalies} anomalies' if anomalies else ''}")
            summary["cross_market_updated"] = True
    except Exception as e:
        print(f"      Cross-market warning: {e}")
        log.debug(f"Incremental cross-market: {e}")

    # Market intelligence aggregation
    print(f"    Market intelligence...")
    try:
        from features.market_intelligence import run_market_intelligence
        mi = run_market_intelligence()
        if mi:
            actionable = sum(1 for r in mi.values() if r.get("composite_score", 0) > 0.3)
            print(f"      {len(mi)} matches{f', {actionable} actionable' if actionable else ''}")
            summary["market_intelligence_updated"] = True
    except Exception as e:
        print(f"      Market intelligence warning: {e}")
        log.debug(f"Incremental market intelligence: {e}")

    # Re-merge EPL data into shared files (Serie A steps above overwrite them)
    print(f"    EPL market data merge...")
    try:
        from scripts.prediction.generate_epl_supplementary import (
            generate_epl_bookmaker_analysis, generate_epl_cross_market_signals,
            generate_epl_market_intelligence, _load_json, _merge_matches, _save_json,
            get_epl_predictions, UPCOMING_DIR,
        )
        if get_epl_predictions():
            bk = generate_epl_bookmaker_analysis()
            cm = generate_epl_cross_market_signals()
            mi_epl = generate_epl_market_intelligence(bk, cm)
            for fname, data, ts_key in [
                ("bookmaker_analysis.json", bk, "analyzed_at"),
                ("cross_market_signals.json", cm, "analyzed_at"),
                ("market_intelligence.json", mi_epl, "analyzed_at"),
            ]:
                existing = _load_json(UPCOMING_DIR / fname)
                merged = _merge_matches(existing, data, ts_key)
                _save_json(UPCOMING_DIR / fname, merged)
            print(f"      {len(mi_epl)} EPL matches merged")
        else:
            print(f"      No EPL predictions, skipping")
    except Exception as e:
        print(f"      EPL market merge warning: {e}")
        log.debug(f"EPL market merge: {e}")


def _run_supplementary_analysis(summary: dict):
    """Run player analysis, lineup predictions, and sentiment — local computations."""

    # Player analysis
    print(f"    Player analysis...")
    try:
        from scripts.analysis.player_analyzer import analyze_all_upcoming_matches as analyze_players
        results = analyze_players()
        if results:
            print(f"      {len(results)} matches analyzed")
            summary["player_analysis_updated"] = True
    except Exception as e:
        print(f"      Player analysis warning: {e}")
        log.debug(f"Incremental player analysis: {e}")

    # Lineup predictions
    print(f"    Lineup predictions...")
    try:
        from scripts.prediction.lineup_predictor import generate_lineup_predictions
        lp = generate_lineup_predictions()
        if lp:
            print(f"      {lp.get('match_count', 0)} matches, {lp.get('team_count', 0)} teams")
            summary["lineup_predictions_updated"] = True
    except Exception as e:
        print(f"      Lineup predictions warning: {e}")
        log.debug(f"Incremental lineup predictions: {e}")

    # Sentiment analysis (uses Perplexity API — skip if no API key configured)
    print(f"    Sentiment...")
    try:
        from scripts.prediction.sentiment_analyzer import analyze_all_upcoming_matches as analyze_sentiment
        results = analyze_sentiment()
        if results:
            print(f"      {len(results)} matches")
            summary["sentiment_updated"] = True
    except Exception as e:
        print(f"      Sentiment warning: {e}")
        log.debug(f"Incremental sentiment: {e}")


def run_incremental(bankroll: float = 1000.0, leagues: list = None) -> Dict:
    """Run an incremental pipeline refresh.

    Refreshes everything needed for accurate predictions:
    1. Fetch results from Odds API /scores — settle newly completed matches
    2. If odds are stale (>4h), fetch fresh odds + run market analysis chain
    3. If market/supplementary data is stale (>6h), refresh that too
    4. Run predictions for new/unseen fixtures
    5. Archive predictions, update pipeline state

    Market analysis (bookmaker, cross-market, intelligence) and supplementary
    data (player analysis, lineups, sentiment) are LOCAL computations on
    already-fetched data — zero additional API credits.

    Returns a summary dict suitable for the web UI.
    """
    import time as _time
    from scripts.pipeline.pipeline_state import (
        load_state, save_state, get_new_results, get_new_fixtures,
        needs_odds_refresh, mark_predicted, mark_settled, update_timestamp,
    )

    start_time = _time.time()
    state = load_state()

    banner("INCREMENTAL REFRESH")
    print(f"\n  Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Previously predicted: {len(state.get('predicted_matches', []))} matches")
    print(f"  Previously settled: {len(state.get('settled_matches', []))} matches")

    summary = {
        "status": "running",
        "new_results": [],
        "new_predictions": [],
        "settled_bets": [],
        "credits_used": 0,
        "errors": [],
        "odds_updated": False,
        "predictions_updated": False,
        "market_analysis_updated": False,
    }

    # ── Step 1: Fetch results and settle new bets ──
    print(f"\n[1/6] Fetching results & settling bets...")
    try:
        from scripts.data.results_fetcher import fetch_scores, parse_scores, save_results, settle_bets

        raw_scores = fetch_scores(days_from=3)
        summary["credits_used"] += 2

        if raw_scores:
            all_results = parse_scores(raw_scores)
            completed = {k: v for k, v in all_results.items() if v.get("completed")}

            # Filter to only new results
            new_results = get_new_results(completed, state)
            print(f"  Found {len(completed)} completed matches, {len(new_results)} new")

            if new_results:
                save_results(all_results)

                # Load existing predictions to check accuracy
                preds_archive = {}
                archive_path = DATA_DIR / "upcoming" / "predictions_archive.json"
                if archive_path.exists():
                    try:
                        with open(archive_path) as f:
                            preds_archive = json.load(f)
                    except Exception:
                        pass

                for match_key, result in new_results.items():
                    result_entry = {
                        "match": match_key,
                        "score": f"{result['home_score']}-{result['away_score']}",
                        "result": result["result"],
                    }
                    # Check our prediction accuracy
                    for archive_key, pred in preds_archive.items():
                        if pred.get("match") == match_key:
                            result_entry["our_prediction"] = pred.get("predicted_outcome", "")
                            result_entry["correct"] = (
                                pred.get("predicted_outcome", "").upper() == result["result"]
                            )
                            break
                    summary["new_results"].append(result_entry)

                # Settle bets against new results
                settlement = settle_bets(new_results)
                if settlement.get("settled", 0) > 0:
                    print(f"  Settled {settlement['settled']} bets: "
                          f"{settlement['won']}W {settlement['lost']}L")
                    summary["settled_bets"] = [{
                        "settled": settlement["settled"],
                        "won": settlement.get("won", 0),
                        "lost": settlement.get("lost", 0),
                        "push": settlement.get("push", 0),
                        "profit": settlement.get("profit", 0),
                        "new_balance": settlement.get("new_balance", 0),
                    }]

                # Mark as settled in state
                dates = [r.get("commence_time", "")[:10] for r in new_results.values()]
                state = mark_settled(state, list(new_results.keys()), dates)

                # Ingest new match data and regenerate standings
                print(f"  Ingesting match data & updating standings...")
                try:
                    from scripts.data.matchday_updater import run_matchday_update
                    update_result = run_matchday_update(
                        rebuild_features=True,
                        regenerate_standings=True,
                    )
                    fetched = update_result.get("matches_fetched", 0)
                    if fetched:
                        print(f"  Ingested {fetched} matches, standings + features rebuilt")
                        summary["standings_updated"] = True
                        summary["features_rebuilt"] = True
                    else:
                        print(f"  No new Sofascore data yet (matches may still be in progress)")
                except Exception as e:
                    print(f"  Matchday update warning: {e}")
                    log.warning(f"Incremental matchday update: {e}")
            else:
                print(f"  No new results to process")
        else:
            print(f"  No scores data received")
    except Exception as e:
        print(f"  Results fetch error: {e}")
        summary["errors"].append(f"Results: {e}")
        log.warning(f"Incremental results error: {e}")

    state = update_timestamp(state, "last_results_fetch")

    # ── Step 2: Refresh odds if stale ──
    odds = {}
    odds_were_refreshed = False
    print(f"\n[2/6] Checking odds freshness...")
    if needs_odds_refresh(state, max_age_hours=4.0):
        print(f"  Odds are stale, fetching fresh...")
        try:
            from scripts.data.odds_fetcher import fetch_and_save_odds

            odds = fetch_and_save_odds(use_cache=False)
            summary["credits_used"] += 2
            odds_were_refreshed = True
            summary["odds_updated"] = True
            print(f"  Fetched odds for {len(odds)} matches")
            state = update_timestamp(state, "last_odds_fetch")
        except Exception as e:
            print(f"  Odds fetch error: {e}")
            summary["errors"].append(f"Odds: {e}")
            log.warning(f"Incremental odds error: {e}")
    else:
        print(f"  Odds are fresh, using cached data")
        # Load cached odds
        odds_path = DATA_DIR / "upcoming" / "odds_full.json"
        if odds_path.exists():
            try:
                with open(odds_path) as f:
                    cached = json.load(f)
                odds = cached.get("matches", cached) if isinstance(cached, dict) else {}
                print(f"  Loaded {len(odds)} matches from cache")
            except Exception:
                pass

    # Fetch odds for extra leagues (if specified)
    extra_leagues = [l for l in (leagues or []) if l != "serie_a"]
    if extra_leagues and odds_were_refreshed:
        try:
            from scripts.prediction.predict_league import fetch_league_odds, LEAGUE_DISPLAY_NAMES
            for league in extra_leagues:
                display = LEAGUE_DISPLAY_NAMES.get(league, league)
                try:
                    league_odds = fetch_league_odds(league, use_cache=False)
                    summary["credits_used"] += 2
                    print(f"  {display}: fetched odds for {len(league_odds)} matches")
                except Exception as e:
                    print(f"  {display}: odds fetch error: {e}")
                    log.warning(f"Incremental {league} odds error: {e}")
        except ImportError:
            pass

    # ── Step 3: Market analysis chain (if odds refreshed or analysis stale) ──
    market_intel_path = DATA_DIR / "upcoming" / "market_intelligence.json"
    market_stale = _is_data_stale(market_intel_path, max_age_hours=6.0)

    if odds_were_refreshed or market_stale:
        print(f"\n[3/6] Running market analysis chain (0 API credits)...")
        if odds_were_refreshed:
            print(f"  Triggered by: fresh odds fetched")
        else:
            print(f"  Triggered by: market data >6h stale")
        _run_market_analysis(odds, summary)
        summary["market_analysis_updated"] = True
    else:
        print(f"\n[3/6] Market analysis is fresh, skipping")

    # ── Step 4: Supplementary analysis (if data stale) ──
    player_path = DATA_DIR / "upcoming" / "player_analysis.json"
    lineup_path = DATA_DIR / "upcoming" / "lineup_predictions.json"
    supplementary_stale = (
        _is_data_stale(player_path, max_age_hours=12.0) or
        _is_data_stale(lineup_path, max_age_hours=12.0)
    )

    if supplementary_stale or odds_were_refreshed:
        print(f"\n[4/6] Refreshing supplementary analysis (0 API credits)...")
        _run_supplementary_analysis(summary)
    else:
        print(f"\n[4/6] Supplementary analysis is fresh, skipping")

    # ── Step 4b: Auto-refresh stale data sources ──
    # Injuries: refresh if >48h stale (uses Football-Data API, 0 credits)
    injuries_path = DATA_DIR / "upcoming" / "injuries.json"
    if _is_data_stale(injuries_path, max_age_hours=48.0):
        print(f"\n[4b/6] Auto-refreshing stale injuries (>48h old)...")
        try:
            from scraper.injuries import get_current_injuries
            injuries = get_current_injuries(auto_scrape=True)
            if injuries:
                print(f"  Refreshed injuries for {len(injuries)} teams")
                summary["injuries_refreshed"] = True
        except Exception as e:
            print(f"  Injury refresh warning: {e}")
            log.warning(f"Incremental injury refresh: {e}")

    # Features: rebuild daily (local computation, 0 credits) so rolling windows
    # stay fresh — was 72h, dropped to 24h after finding that 3-day stale rolling
    # windows were a measurable accuracy tax on upcoming-match predictions.
    features_path = DATA_DIR / "features" / "features.parquet"
    if _is_data_stale(features_path, max_age_hours=24.0):
        print(f"  Auto-rebuilding stale features (>24h old)...")
        try:
            from features.build import build_features
            build_features()
            print(f"  Features rebuilt")
            summary["features_rebuilt"] = True
        except Exception as e:
            print(f"  Feature rebuild warning: {e}")
            log.warning(f"Incremental feature rebuild: {e}")

    # ── Step 5: Predict new fixtures (or refresh stale predictions) ──
    print(f"\n[5/6] Running predictions...")
    try:
        if odds:
            new_fixtures = get_new_fixtures(odds, state)
            print(f"  {len(new_fixtures)} new fixtures to predict (of {len(odds)} total)")

            # Also regenerate if odds were refreshed and predictions are stale (>3h)
            predictions_stale = _is_data_stale(
                DATA_DIR / "upcoming" / "predictions.json", max_age_hours=3.0
            )
            should_regenerate = bool(new_fixtures) or (odds_were_refreshed and predictions_stale)

            if should_regenerate:
                if not new_fixtures:
                    print(f"  Odds refreshed & predictions >3h old — regenerating with fresh odds")

                # Archive existing predictions first
                try:
                    from scripts.analysis.performance_dashboard import archive_predictions
                    archive_predictions()
                except Exception:
                    pass

                # Run ensemble predictions (runs for ALL matches but we track which are new)
                from scripts.prediction.ensemble_prediction_engine import run_ensemble_predictions
                result = run_ensemble_predictions(use_ensemble=True)
                predictions = result.get("predictions", [])
                print(f"  Generated {len(predictions)} predictions")
                summary["predictions_updated"] = True

                # Track new predictions in summary
                new_match_keys = set(new_fixtures.keys()) if new_fixtures else set()
                for pred in predictions:
                    match = pred.get("match", "")
                    if match in new_match_keys:
                        summary["new_predictions"].append({
                            "match": match,
                            "prediction": pred.get("predicted_outcome", ""),
                            "confidence": pred.get("confidence_level", ""),
                            "home_prob": pred.get("probabilities", {}).get("home", 0),
                            "draw_prob": pred.get("probabilities", {}).get("draw", 0),
                            "away_prob": pred.get("probabilities", {}).get("away", 0),
                        })

                # Run betting engine on fresh predictions
                try:
                    from scripts.betting.betting_unified import BetSlip, record_bets
                    from scripts.betting.betting_unified import load_predictions as load_betting_preds
                    log.info("Betting recommendations generated via predict_unified pipeline step")
                except ImportError:
                    pass

                # Mark all current fixtures as predicted
                pred_keys = [p.get("match", "") for p in predictions]
                pred_dates = [p.get("date", "") for p in predictions]
                state = mark_predicted(state, pred_keys, pred_dates)
            else:
                print(f"  All fixtures already predicted & predictions fresh, nothing to do")
        else:
            print(f"  No odds data available, skipping predictions")
    except Exception as e:
        print(f"  Prediction error: {e}")
        summary["errors"].append(f"Predictions: {e}")
        log.warning(f"Incremental prediction error: {e}")

    # Run predictions for extra leagues
    if extra_leagues:
        print(f"\n[5b/6] Multi-league predictions ({', '.join(extra_leagues)})...")
        try:
            from scripts.prediction.predict_league import run_predictions_for_league, LEAGUE_DISPLAY_NAMES
            for league in extra_leagues:
                display = LEAGUE_DISPLAY_NAMES.get(league, league)
                try:
                    result = run_predictions_for_league(league)
                    count = result.get("count", 0)
                    if count:
                        print(f"  {display}: {count} predictions")
                    else:
                        print(f"  {display}: skipped (no model or no matches)")
                except Exception as e:
                    print(f"  {display}: prediction error: {e}")
                    log.warning(f"Incremental {league} prediction error: {e}")
        except ImportError:
            pass

    # ── Step 6: Update state and generate dashboard ──
    print(f"\n[6/6] Updating state & dashboard...")
    state = update_timestamp(state, "last_run")
    save_state(state)

    try:
        from scripts.analysis.performance_dashboard import generate_dashboard
        generate_dashboard()
        print(f"  Dashboard regenerated")
    except Exception as e:
        print(f"  Dashboard warning: {e}")

    elapsed = _time.time() - start_time
    summary["status"] = "completed"
    summary["elapsed_seconds"] = round(elapsed, 1)

    banner("INCREMENTAL REFRESH COMPLETE")
    print(f"\n  Elapsed: {elapsed:.1f}s")
    print(f"  Credits used: ~{summary['credits_used']} (API) + local analysis")
    print(f"  New results: {len(summary['new_results'])}")
    print(f"  New predictions: {len(summary['new_predictions'])}")
    refreshed = []
    if summary["odds_updated"]: refreshed.append("odds")
    if summary["market_analysis_updated"]: refreshed.append("market")
    if summary.get("player_analysis_updated"): refreshed.append("players")
    if summary.get("lineup_predictions_updated"): refreshed.append("lineups")
    if refreshed:
        print(f"  Refreshed: {', '.join(refreshed)}")
    if summary["settled_bets"]:
        sb = summary["settled_bets"][0]
        print(f"  Settled bets: {sb['settled']} ({sb['won']}W/{sb['lost']}L)")
    if summary["errors"]:
        print(f"  Errors: {len(summary['errors'])}")

    return summary


def run_pre_kickoff(bankroll: float = 1000.0):
    """Run a focused pre-kickoff pipeline for imminent matches.

    Quick 5-step flow (~25 seconds):
    1. Quick odds refresh (use cache if recent)
    2. Classify match windows, filter live
    3. Fetch confirmed lineups for imminent matches
    4. Regenerate ensemble predictions (with confirmed lineups)
    5. Regenerate betting recommendations
    """
    start_time = time.time()
    total = 5

    banner("PRE-KICKOFF PIPELINE - 5 STEPS")
    print(f"\n  Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode: Pre-kickoff (confirmed lineups)")
    print(f"  Bankroll: ${bankroll:.2f}")

    # Step 1: Quick odds refresh
    step(1, total, "Quick Odds Refresh")
    odds = {}
    try:
        from scripts.data.odds_fetcher import fetch_and_save_odds
        odds = fetch_and_save_odds(use_cache=True)
        print(f"  Loaded odds for {len(odds)} matches")
    except Exception as e:
        print(f"  Odds warning: {e}")

    # Step 2: Classify match windows
    step(2, total, "Classifying Match Windows")
    try:
        from scripts.utils.match_timing import classify_match_window
        imminent = []
        approaching = []
        for match_key, data in odds.items():
            ct = data.get("commence_time", "")
            window = classify_match_window(ct)
            if window == "imminent":
                imminent.append(match_key)
            elif window == "approaching":
                approaching.append(match_key)
        print(f"  Imminent (0-1h): {len(imminent)} matches")
        print(f"  Approaching (1-6h): {len(approaching)} matches")
        if imminent:
            for m in imminent:
                print(f"    >> {m}")
        if not imminent and not approaching:
            print(f"  No matches approaching kickoff - nothing to update")
            elapsed = time.time() - start_time
            banner(f"PRE-KICKOFF COMPLETE ({elapsed:.1f}s)")
            return None
    except Exception as e:
        print(f"  Window classification warning: {e}")

    # Step 3: Fetch confirmed lineups
    step(3, total, "Fetching Confirmed Lineups")
    confirmed = {}
    try:
        from scraper.lineup_fetcher import fetch_and_save_lineups
        confirmed = fetch_and_save_lineups()
        if confirmed:
            print(f"  Confirmed lineups for {len(confirmed)} match(es):")
            for mk, ld in confirmed.items():
                print(f"    {mk}: {ld.get('home_formation', '?')} vs {ld.get('away_formation', '?')}")
        else:
            print(f"  No confirmed lineups yet (may be too early)")
    except Exception as e:
        print(f"  Lineup fetch: {e}")

    # Step 4: Regenerate predictions
    step(4, total, "Regenerating Ensemble Predictions")
    predictions = []
    try:
        from scripts.prediction.ensemble_prediction_engine import run_ensemble_predictions
        result = run_ensemble_predictions(use_ensemble=True)
        predictions = result.get("predictions", [])
        confirmed_count = sum(1 for p in predictions if p.get("lineup_source") == "confirmed")
        print(f"  Generated {len(predictions)} predictions")
        if confirmed_count:
            print(f"  {confirmed_count} with confirmed lineups (boosted player_xg)")
    except Exception as e:
        print(f"  Prediction error: {e}")

    # Step 5: Regenerate betting recommendations
    step(5, total, "Regenerating Betting Recommendations")
    report = None
    try:
        from scripts.betting.betting_unified import generate_unified_report, save_report
        report = generate_unified_report(bankroll)
        save_report(report)
        print(f"  Generated {report['summary']['total_bets']} optimized bets")
        print(f"  Total stake: ${report['summary']['total_stake']:.2f}")

        if report.get("bets"):
            _archive_bets(report["bets"], report.get("generated_at", ""))
    except Exception as e:
        print(f"  Betting engine warning: {e}")

    elapsed = time.time() - start_time
    banner(f"PRE-KICKOFF COMPLETE ({elapsed:.1f}s)")
    print(f"\n  Matches updated: {len(predictions)}")
    if confirmed:
        print(f"  Confirmed lineups: {len(confirmed)}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return report


def _run_parallel_data_collection(quick: bool = False, total_steps: int = 32):
    """Run Steps 1-10 using parallel thread groups for faster data collection.

    Group A (sequential chain): Steps 1->2->4->5->6 (odds chain)
    Group B (independent): Steps 3, 9, 10
    Group C (waits for Group A odds analysis): Steps 7->8

    Returns:
        Tuple of (odds, market_intel_results) for downstream use.
    """
    # Shared state (thread-safe via GIL for simple assignments)
    shared = {
        "odds": {},
        "market_intel_results": {},
        "timings": {},  # per-group timing
    }

    # Thread-safe output lock (prevents interleaved print lines)
    _print_lock = threading.Lock()

    def ts_print(*args, **kwargs):
        """Thread-safe print."""
        with _print_lock:
            print(*args, **kwargs)

    def ts_step(num, total, name):
        """Thread-safe step indicator."""
        with _print_lock:
            print(f"\n[{num}/{total}] {name}...")
            log.info(f"Starting step {num}/{total}: {name}")

    # Synchronization events
    odds_ready = threading.Event()        # Set after Step 1 (odds fetched)
    odds_analysis_ready = threading.Event()  # Set after Step 6 (analysis done)

    # ---- Group A: Odds chain (Steps 1->2->4->5->6) ----
    def group_a():
        t0 = time.time()
        # Step 1: Fetch odds
        ts_step(1, total_steps, "Fetching Multi-Market Odds")
        try:
            from scripts.data.odds_fetcher import fetch_and_save_odds, get_usage_summary
            shared["odds"] = fetch_and_save_odds(use_cache=quick)
            usage = get_usage_summary()
            ts_print(f"  Fetched odds for {len(shared['odds'])} matches")
            ts_print(f"  API credits: {usage.get('api_remaining', '?')} remaining")
        except Exception as e:
            ts_print(f"  Odds fetch warning: {e}")
            log.warning(f"Odds fetch error: {e}")
        odds_ready.set()

        odds = shared["odds"]

        # Step 2: Sync matches
        ts_step(2, total_steps, "Syncing Matches from Odds API")
        try:
            from scripts.data.odds_fetcher import sync_matches_from_odds
            if odds:
                synced = sync_matches_from_odds(odds)
                ts_print(f"  Synced {len(synced)} matches from Odds API")
            else:
                ts_print(f"  No odds data to sync (using existing matches)")
        except Exception as e:
            ts_print(f"  Match sync warning: {e}")
            log.warning(f"Match sync error: {e}")

        # Step 4: Validate per-bookmaker data
        ts_step(4, total_steps, "Saving Per-Bookmaker Data")
        try:
            bm_path = DATA_DIR / "upcoming" / "odds_bookmakers.json"
            if bm_path.exists():
                with open(bm_path) as f:
                    bm_data = json.load(f)
                bookmaker_count = len(bm_data.get("matches", {}))
                if bookmaker_count > 0:
                    total_entries = sum(
                        len(v.get("h2h", []))
                        for v in bm_data.get("matches", {}).values()
                    )
                    ts_print(f"  Per-bookmaker data: {bookmaker_count} matches, {total_entries} bookmaker entries")
                else:
                    ts_print(f"  No per-bookmaker data available")
            else:
                ts_print(f"  odds_bookmakers.json not found (odds fetcher may not have run)")
        except Exception as e:
            ts_print(f"  Per-bookmaker data warning: {e}")
            log.warning(f"Per-bookmaker data error: {e}")

        # Step 5: Bookmaker analysis
        ts_step(5, total_steps, "Running Bookmaker Analysis (Sharp/Soft)")
        try:
            from features.bookmaker_analysis import run_bookmaker_analysis
            ba_results = run_bookmaker_analysis()
            if ba_results:
                sharp_count = sum(1 for m in ba_results.values() if m.get("has_sharp_data"))
                actionable = sum(1 for m in ba_results.values() if m.get("divergence_actionable"))
                ts_print(f"  Analyzed {len(ba_results)} matches ({sharp_count} with sharp data)")
                if actionable > 0:
                    ts_print(f"  {actionable} matches with actionable sharp-soft divergence (>3%)")
            else:
                ts_print(f"  No bookmaker data for analysis")
        except Exception as e:
            ts_print(f"  Bookmaker analysis warning: {e}")
            log.warning(f"Bookmaker analysis error: {e}")

        # Step 6: Track odds movement
        ts_step(6, total_steps, "Tracking Odds Movement")
        try:
            from scripts.data.odds_tracker import save_snapshot, analyze_movements
            if odds:
                simple_odds = {}
                for mk, md in odds.items():
                    h2h = md.get("h2h")
                    if h2h:
                        simple_odds[mk] = {"home": h2h["home"], "draw": h2h["draw"], "away": h2h["away"]}
                if simple_odds:
                    save_snapshot(simple_odds)
                    movements = analyze_movements(simple_odds)
                    n_moves = sum(1 for m in movements.values() if m.get("is_line_move"))
                    n_steam = sum(1 for m in movements.values() if m.get("is_steam_move"))
                    ts_print(f"  Saved odds snapshot & analyzed {len(movements)} matches")
                    if n_steam > 0:
                        ts_print(f"  {n_steam} STEAM MOVES detected (sharp money!)")
                    elif n_moves > 0:
                        ts_print(f"  {n_moves} notable line movements detected")
                    else:
                        ts_print(f"  No significant line movements")
            else:
                ts_print(f"  No odds data to track")
        except Exception as e:
            ts_print(f"  Odds tracking warning: {e}")
            log.warning(f"Odds tracking error: {e}")

        odds_analysis_ready.set()
        shared["timings"]["group_a"] = time.time() - t0

    # ---- Group B: Independent steps (3, 9, 10) ----
    def group_b():
        t0 = time.time()
        # Step 3: Refresh squads
        ts_step(3, total_steps, "Refreshing Squad Data")
        try:
            from scraper.squad_fetcher import fetch_and_save_squads
            squad_data = fetch_and_save_squads()
            teams_count = len(squad_data.get("teams", {}))
            total_players = sum(len(t.get("players", [])) for t in squad_data.get("teams", {}).values())
            source = squad_data.get("source", "unknown")
            ts_print(f"  Squad data: {teams_count} teams, {total_players} players (source: {source})")
        except Exception as e:
            ts_print(f"  Squad refresh warning: {e}")
            log.warning(f"Squad refresh error: {e}")

        # Step 9: Fetch lineups
        ts_step(9, total_steps, "Fetching Confirmed Lineups")
        try:
            from scraper.lineup_fetcher import fetch_and_save_lineups
            confirmed = fetch_and_save_lineups()
            if confirmed:
                ts_print(f"  Fetched confirmed lineups for {len(confirmed)} match(es)")
                for mk, ld in confirmed.items():
                    ts_print(f"    {mk}: {ld.get('home_formation', '?')} vs {ld.get('away_formation', '?')}")
            else:
                ts_print(f"  No confirmed lineups available (normal if >1h before kickoff)")
        except Exception as e:
            ts_print(f"  Lineup fetch info: {e}")
            log.debug(f"Lineup fetch: {e}")

        # Step 10: Fetch injuries
        ts_step(10, total_steps, "Fetching Injury Data")
        try:
            from scraper.injuries import get_current_injuries
            injuries_df = get_current_injuries(auto_scrape=not quick)
            if not injuries_df.empty:
                n_teams = injuries_df["team"].nunique() if "team" in injuries_df.columns else 0
                ts_print(f"  Loaded {len(injuries_df)} injuries across {n_teams} teams")
            else:
                ts_print(f"  No injury data available (run with --full to scrape)")
        except Exception as e:
            ts_print(f"  Injury data warning: {e}")
            log.warning(f"Injury data error: {e}")
        shared["timings"]["group_b"] = time.time() - t0

    # ---- Group C: Cross-market + market intel (waits for Group A) ----
    def group_c():
        t0 = time.time()
        odds_analysis_ready.wait(timeout=120)

        # Step 7: Cross-market analysis
        ts_step(7, total_steps, "Running Cross-Market Analysis")
        try:
            from features.cross_market_analysis import run_cross_market_analysis
            cm_results = run_cross_market_analysis()
            if cm_results:
                off = sum(1 for r in cm_results.values() if r.get("offensive_dominance", {}).get("detected"))
                dfs = sum(1 for r in cm_results.values() if r.get("defensive_signal", {}).get("detected"))
                ts_print(f"  Analyzed {len(cm_results)} matches")
                if off > 0:
                    ts_print(f"  {off} offensive dominance signals")
                if dfs > 0:
                    ts_print(f"  {dfs} defensive signals")
            else:
                ts_print(f"  No odds data for cross-market analysis")
        except Exception as e:
            ts_print(f"  Cross-market analysis warning: {e}")
            log.warning(f"Cross-market analysis error: {e}")

        # Step 8: Market intelligence
        ts_step(8, total_steps, "Aggregating Market Intelligence")
        try:
            from features.market_intelligence import run_market_intelligence
            shared["market_intel_results"] = run_market_intelligence()
            mi = shared["market_intel_results"]
            if mi:
                actionable = sum(1 for r in mi.values() if r.get("composite_score", 0) > 0.3)
                steam = sum(1 for r in mi.values() if r.get("steam_detected"))
                ts_print(f"  Aggregated intelligence for {len(mi)} matches")
                if actionable > 0:
                    ts_print(f"  {actionable} matches with actionable signals")
                if steam > 0:
                    ts_print(f"  {steam} steam moves detected")
            else:
                ts_print(f"  No data for market intelligence aggregation")
        except Exception as e:
            ts_print(f"  Market intelligence warning: {e}")
            log.warning(f"Market intelligence error: {e}")
        shared["timings"]["group_c"] = time.time() - t0

    # Execute groups in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(group_a): "Group A (odds chain)",
            executor.submit(group_b): "Group B (squads/lineups/injuries)",
            executor.submit(group_c): "Group C (cross-market/intel)",
        }
        for future in as_completed(futures):
            group_name = futures[future]
            try:
                future.result()
            except Exception as e:
                log.warning(f"{group_name} failed: {e}")
                ts_print(f"  WARNING: {group_name} encountered error: {e}")

    # Report per-group timing
    timings = shared["timings"]
    if timings:
        ts_print(f"  Timing: Group A {timings.get('group_a', 0):.1f}s, "
                 f"Group B {timings.get('group_b', 0):.1f}s, "
                 f"Group C {timings.get('group_c', 0):.1f}s")

    # Data quality gates — abort early if critical data is missing
    _validate_data_gate("odds", shared["odds"], min_threshold=1, required=True)

    return shared["odds"], shared["market_intel_results"]


_pipeline_lock_fd = None

def run_pipeline(quick: bool = False, bankroll: float = 1000.0, snapshot_only: bool = False,
                 no_parallel: bool = False, run_challenger: bool = False,
                 leagues: list = None):
    """Run the complete betting pipeline."""
    global _pipeline_lock_fd

    # Acquire exclusive lock to prevent concurrent pipeline runs
    lock_path = DATA_DIR / "pipeline.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    _pipeline_lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(_pipeline_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _pipeline_lock_fd.write(f"PID {os.getpid()} since {datetime.now().isoformat()}\n")
        _pipeline_lock_fd.flush()
    except BlockingIOError:
        print("  Another pipeline is already running. Exiting.")
        log.warning("Pipeline lock held by another process, aborting")
        _pipeline_lock_fd.close()
        _pipeline_lock_fd = None
        return

    start_time = time.time()

    total_steps = 6 if snapshot_only else 33
    mode_label = "Snapshot-only" if snapshot_only else ("Quick (cached)" if quick else "Full (fresh data)")

    banner(f"FULL BETTING PIPELINE - {total_steps} STEPS")
    print(f"\n  Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode: {mode_label}")
    print(f"  Bankroll: ${bankroll:.2f}")

    # Notify pipeline start
    try:
        from scripts.pipeline.notify import notify_pipeline_start
        notify_pipeline_start()
    except Exception:
        pass

    # Archive previous predictions before they get overwritten
    try:
        from scripts.analysis.performance_dashboard import archive_predictions
        n = archive_predictions()
        if n:
            print(f"  Archived {n} previous predictions for tracking")
    except Exception as e:
        log.debug(f"Prediction archive skipped: {e}")

    odds = {}
    predictions = []
    sentiment_results = []
    player_results = []
    market_intel_results = {}
    report = None

    # =========================================================================
    # Steps 1-10: Data Collection (Parallel or Sequential)
    # =========================================================================
    if not no_parallel and not snapshot_only:
        banner("PARALLEL DATA COLLECTION (Steps 1-10)")
        dc_start = time.time()
        odds, market_intel_results = _run_parallel_data_collection(
            quick=quick, total_steps=total_steps,
        )
        dc_elapsed = time.time() - dc_start
        print(f"\n  Data collection completed in {dc_elapsed:.1f}s (parallel)")

        # Snapshot-only check (odds already fetched)
        if snapshot_only:
            elapsed = time.time() - start_time
            banner("SNAPSHOT COMPLETE")
            print(f"\n  Elapsed: {elapsed:.1f}s | Steps: 1-6 | Matches: {len(odds)}")
            print(f"  Run 3-4x daily (48h before matchday) for temporal depth")
            return None
    else:
        # Sequential mode (original behavior, also used for snapshot_only)
        step(1, total_steps, "Fetching Multi-Market Odds")
        try:
            from scripts.data.odds_fetcher import fetch_and_save_odds, get_usage_summary

            odds = fetch_and_save_odds(use_cache=quick)
            usage = get_usage_summary()
            print(f"  Fetched odds for {len(odds)} matches")
            print(f"  API credits: {usage.get('api_remaining', '?')} remaining")
        except Exception as e:
            print(f"  Odds fetch warning: {e}")
            log.warning(f"Odds fetch error: {e}")

        # CRITICAL: If no odds at all, abort — can't compute edges without market data
        if not odds:
            log.error("CRITICAL: No odds data available. Cannot compute betting edges.")
            try:
                from scripts.pipeline.notify import send_notification
                send_notification(
                    "Pipeline aborted: no odds data. Check Odds API key/quota.",
                    title="CRITICAL: No Odds Data",
                    level="critical",
                )
            except Exception:
                pass
            print("  ABORTING: No odds data — cannot generate bet recommendations")
            return summary

        step(2, total_steps, "Syncing Matches from Odds API")
        try:
            from scripts.data.odds_fetcher import sync_matches_from_odds
            if odds:
                synced = sync_matches_from_odds(odds)
                print(f"  Synced {len(synced)} matches from Odds API")
            else:
                print(f"  No odds data to sync (using existing matches)")
        except Exception as e:
            print(f"  Match sync warning: {e}")
            log.warning(f"Match sync error: {e}")

        # Step 2b: Refresh historical (settled-match) odds in matches.parquet.
        # The football-data.co.uk source went stale Feb 23 2026; backfill from
        # Odds API historical endpoint runs weekly (Mondays) to keep CSVs fresh,
        # then import_seriea_odds.py merges any new CSV rows into matches.parquet.
        # The merge is idempotent so safe to run daily even when no new CSV data.
        step(2.5, total_steps, "Refreshing Historical Odds (CSV → matches.parquet)")
        try:
            import datetime as _dt, subprocess as _sp
            # Weekly backfill: only on Mondays AND only if Odds API quota allows.
            # If quota burned out, skip silently — the daily import below still picks
            # up whatever is already in the CSV from the last successful backfill.
            if _dt.datetime.now().weekday() == 0:  # Monday
                since = (_dt.date.today() - _dt.timedelta(days=10)).isoformat()
                print(f"  Weekly backfill: Odds API since {since}...")
                try:
                    proc = _sp.run(
                        [sys.executable, "-m", "scripts.data.backfill_historical_odds",
                         "--league", "serie_a", "--since", since],
                        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=900,
                    )
                    if proc.returncode == 0:
                        print(f"  Backfill OK")
                    else:
                        last = (proc.stderr or proc.stdout or "")[-200:]
                        print(f"  Backfill warning: {last}")
                except Exception as _e:
                    print(f"  Backfill skipped: {_e}")

            # Daily merge: CSV → matches.parquet (cheap, no API calls)
            from scripts import import_seriea_odds as _isa
            _matched = _isa.update_parquet_with_odds()
            print(f"  Imported odds: {_matched}/{_isa.TOTAL_SA_ROWS or '?'} SA rows have odds")
        except Exception as e:
            print(f"  Historical odds refresh warning: {e}")
            log.warning(f"Historical odds refresh error: {e}")

        step(3, total_steps, "Refreshing Squad Data")
        try:
            from scraper.squad_fetcher import fetch_and_save_squads
            squad_data = fetch_and_save_squads()
            teams_count = len(squad_data.get("teams", {}))
            total_players = sum(len(t.get("players", [])) for t in squad_data.get("teams", {}).values())
            source = squad_data.get("source", "unknown")
            print(f"  Squad data: {teams_count} teams, {total_players} players (source: {source})")
        except Exception as e:
            print(f"  Squad refresh warning: {e}")
            log.warning(f"Squad refresh error: {e}")

        step(4, total_steps, "Saving Per-Bookmaker Data")
        try:
            bm_path = DATA_DIR / "upcoming" / "odds_bookmakers.json"
            if bm_path.exists():
                with open(bm_path) as f:
                    bm_data = json.load(f)
                bookmaker_count = len(bm_data.get("matches", {}))
                if bookmaker_count > 0:
                    total_entries = sum(
                        len(v.get("h2h", []))
                        for v in bm_data.get("matches", {}).values()
                    )
                    print(f"  Per-bookmaker data: {bookmaker_count} matches, {total_entries} bookmaker entries")
                else:
                    print(f"  No per-bookmaker data available")
            else:
                print(f"  odds_bookmakers.json not found (odds fetcher may not have run)")
        except Exception as e:
            print(f"  Per-bookmaker data warning: {e}")
            log.warning(f"Per-bookmaker data error: {e}")

        step(5, total_steps, "Running Bookmaker Analysis (Sharp/Soft)")
        try:
            from features.bookmaker_analysis import run_bookmaker_analysis
            ba_results = run_bookmaker_analysis()
            if ba_results:
                sharp_count = sum(1 for m in ba_results.values() if m.get("has_sharp_data"))
                actionable = sum(1 for m in ba_results.values() if m.get("divergence_actionable"))
                print(f"  Analyzed {len(ba_results)} matches ({sharp_count} with sharp data)")
                if actionable > 0:
                    print(f"  {actionable} matches with actionable sharp-soft divergence (>3%)")
            else:
                print(f"  No bookmaker data for analysis")
        except Exception as e:
            print(f"  Bookmaker analysis warning: {e}")
            log.warning(f"Bookmaker analysis error: {e}")

        step(6, total_steps, "Tracking Odds Movement")
        try:
            from scripts.data.odds_tracker import save_snapshot, analyze_movements
            if odds:
                simple_odds = {}
                for mk, md in odds.items():
                    h2h = md.get("h2h")
                    if h2h:
                        simple_odds[mk] = {"home": h2h["home"], "draw": h2h["draw"], "away": h2h["away"]}
                if simple_odds:
                    save_snapshot(simple_odds)
                    movements = analyze_movements(simple_odds)
                    n_moves = sum(1 for m in movements.values() if m.get("is_line_move"))
                    n_steam = sum(1 for m in movements.values() if m.get("is_steam_move"))
                    print(f"  Saved odds snapshot & analyzed {len(movements)} matches")
                    if n_steam > 0:
                        print(f"  {n_steam} STEAM MOVES detected (sharp money!)")
                    elif n_moves > 0:
                        print(f"  {n_moves} notable line movements detected")
                    else:
                        print(f"  No significant line movements")
            else:
                print(f"  No odds data to track")
        except Exception as e:
            print(f"  Odds tracking warning: {e}")
            log.warning(f"Odds tracking error: {e}")

        # ---- SNAPSHOT-ONLY MODE: Exit after Step 6 ----
        if snapshot_only:
            elapsed = time.time() - start_time
            banner("SNAPSHOT COMPLETE")
            print(f"\n  Elapsed: {elapsed:.1f}s | Steps: 1-6 | Matches: {len(odds)}")
            print(f"  Run 3-4x daily (48h before matchday) for temporal depth")
            return None

        step(7, total_steps, "Running Cross-Market Analysis")
        try:
            from features.cross_market_analysis import run_cross_market_analysis
            cm_results = run_cross_market_analysis()
            if cm_results:
                off = sum(1 for r in cm_results.values() if r.get("offensive_dominance", {}).get("detected"))
                dfs = sum(1 for r in cm_results.values() if r.get("defensive_signal", {}).get("detected"))
                print(f"  Analyzed {len(cm_results)} matches")
                if off > 0:
                    print(f"  {off} offensive dominance signals")
                if dfs > 0:
                    print(f"  {dfs} defensive signals")
            else:
                print(f"  No odds data for cross-market analysis")
        except Exception as e:
            print(f"  Cross-market analysis warning: {e}")
            log.warning(f"Cross-market analysis error: {e}")

        step(8, total_steps, "Aggregating Market Intelligence")
        try:
            from features.market_intelligence import run_market_intelligence
            market_intel_results = run_market_intelligence()
            if market_intel_results:
                actionable = sum(1 for r in market_intel_results.values() if r.get("composite_score", 0) > 0.3)
                steam = sum(1 for r in market_intel_results.values() if r.get("steam_detected"))
                print(f"  Aggregated intelligence for {len(market_intel_results)} matches")
                if actionable > 0:
                    print(f"  {actionable} matches with actionable signals")
                if steam > 0:
                    print(f"  {steam} steam moves detected")
            else:
                print(f"  No data for market intelligence aggregation")
        except Exception as e:
            print(f"  Market intelligence warning: {e}")
            log.warning(f"Market intelligence error: {e}")

        step(9, total_steps, "Fetching Confirmed Lineups")
        try:
            from scraper.lineup_fetcher import fetch_and_save_lineups
            confirmed = fetch_and_save_lineups()
            if confirmed:
                print(f"  Fetched confirmed lineups for {len(confirmed)} match(es)")
                for mk, ld in confirmed.items():
                    print(f"    {mk}: {ld.get('home_formation', '?')} vs {ld.get('away_formation', '?')}")
            else:
                print(f"  No confirmed lineups available (normal if >1h before kickoff)")
        except Exception as e:
            print(f"  Lineup fetch info: {e}")
            log.debug(f"Lineup fetch: {e}")

        step(10, total_steps, "Fetching Injury Data")
        try:
            from scraper.injuries import get_current_injuries
            # Always try to scrape if data is >48h old
            _inj_stale = _is_data_stale(DATA_DIR / "external" / "injuries", 48)
            injuries_df = get_current_injuries(auto_scrape=_inj_stale or not quick)
            if not injuries_df.empty:
                n_teams = injuries_df["team"].nunique() if "team" in injuries_df.columns else 0
                print(f"  Loaded {len(injuries_df)} injuries across {n_teams} teams")
            else:
                print(f"  No injury data available")
        except Exception as e:
            print(f"  Injury data warning: {e}")
            log.warning(f"Injury data error: {e}")

        # Auto-refresh Sofascore if >3 days stale (runs as subprocess — takes 10-20 min)
        try:
            _ss_dir = DATA_DIR / "external" / "sofascore"
            _ss_files = list(_ss_dir.glob("*.parquet")) if _ss_dir.exists() else []
            _ss_age = max(((datetime.now().timestamp() - f.stat().st_mtime) / 3600 for f in _ss_files), default=999)
            if _ss_age > 72:  # >3 days
                print(f"  Sofascore data stale ({_ss_age:.0f}h) — starting background refresh...")
                import subprocess as _sp
                _sp.Popen(
                    [sys.executable, "-m", "scripts.data.scrape_sofascore", "--season", "2025-2026"],
                    cwd=str(PROJECT_ROOT),
                    stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
                )
                print(f"  Sofascore refresh started in background")
        except Exception as e:
            print(f"  Sofascore refresh skipped: {e}")

        # Auto-refresh Understat if >5 days stale
        try:
            _us_path = DATA_DIR / "parsed" / "understat_players.parquet"
            _us_age = (datetime.now().timestamp() - _us_path.stat().st_mtime) / 3600 if _us_path.exists() else 999
            if _us_age > 120:  # >5 days
                print(f"  Understat data stale ({_us_age:.0f}h) — refreshing...")
                from scraper.understat_scraper import scrape_understat_xg
                _us_df = scrape_understat_xg()
                if _us_df is not None and len(_us_df) > 0:
                    # Drop columns that can't serialize to Parquet (empty struct types)
                    for _col in _us_df.columns:
                        try:
                            _us_df[[_col]].to_parquet("/dev/null")
                        except Exception:
                            _us_df = _us_df.drop(columns=[_col])
                    _us_df.to_parquet(str(_us_path), index=False)
                    print(f"  Understat refreshed ({len(_us_df)} rows)")
                else:
                    print(f"  Understat scraper returned no data")
        except Exception as e:
            print(f"  Understat refresh skipped: {e}")

    # =========================================================================
    # Step 10a: Fetch Weather Data for Upcoming Matches
    # =========================================================================
    try:
        from scraper.weather import fetch_weather_for_matches
        import pandas as _pd
        _fixtures_path = DATA_DIR / "upcoming" / "fixtures.json"
        if _fixtures_path.exists():
            import json as _json
            with open(_fixtures_path) as _f:
                _fix = _json.load(_f)
            if _fix:
                _match_df = _pd.DataFrame(_fix) if isinstance(_fix, list) else _pd.DataFrame()
                if not _match_df.empty:
                    _weather = fetch_weather_for_matches(_match_df)
                    if _weather is not None and not _weather.empty:
                        _weather.to_json(DATA_DIR / "upcoming" / "weather.json", orient="records", indent=2)
                        print(f"  Weather: fetched for {len(_weather)} matches")
    except Exception as e:
        print(f"  Weather data warning: {e}")
        log.warning(f"Weather fetch error: {e}")

    # =========================================================================
    # Step 10b: Refresh Referee Assignments
    # =========================================================================
    try:
        from scripts.prediction.referee_integration import load_referee_assignments
        _refs = load_referee_assignments()
        if _refs:
            import json as _json
            _ref_path = DATA_DIR / "upcoming" / "referees.json"
            from config.settings import atomic_write_json as _awj
            _awj(_ref_path, _refs)
            print(f"  Referees: {len(_refs)} assignments loaded")
    except Exception as e:
        print(f"  Referee data warning: {e}")
        log.warning(f"Referee fetch error: {e}")

    # =========================================================================
    # Step 10c: Fetch Odds for Extra Leagues
    # =========================================================================
    extra_leagues = [l for l in (leagues or []) if l != "serie_a"]
    if extra_leagues:
        print(f"\n  Fetching odds for extra leagues: {', '.join(extra_leagues)}")
        try:
            from scripts.prediction.predict_league import fetch_league_odds, LEAGUE_DISPLAY_NAMES
            for league in extra_leagues:
                display = LEAGUE_DISPLAY_NAMES.get(league, league)
                try:
                    league_odds = fetch_league_odds(league, use_cache=quick)
                    print(f"    {display}: {len(league_odds)} matches")
                except Exception as e:
                    print(f"    {display}: odds error: {e}")
                    log.warning(f"{league} odds fetch error: {e}")
        except ImportError:
            pass

    # =========================================================================
    # Step 11: Generate Ensemble Predictions (Enhanced with injury adjustments)
    # =========================================================================
    step(11, total_steps, "Generating Ensemble Predictions")
    try:
        from scripts.prediction.ensemble_prediction_engine import run_ensemble_predictions

        result = run_ensemble_predictions(use_ensemble=True)
        predictions = result.get("predictions", [])
        methods = result.get("methods_available", ["factor"])
        print(f"  Generated {len(predictions)} ensemble predictions")
        print(f"  Methods: {', '.join(methods)}")
        # Report injury adjustments
        injury_adj_count = sum(1 for p in predictions if p.get("injury_adjustments"))
        if injury_adj_count > 0:
            print(f"  {injury_adj_count} predictions with injury xG adjustments")
    except Exception as e:
        print(f"  Ensemble failed, falling back to factor-only: {e}")
        log.warning(f"Ensemble error: {e}, falling back")
        try:
            from scripts.prediction.predict_unified import run_predictions
            result = run_predictions()
            predictions = result.get("predictions", [])
            print(f"  Generated {len(predictions)} factor-only predictions (fallback)")
        except Exception as e2:
            print(f"  Prediction error: {e2}")
            log.warning(f"Prediction error: {e2}")

    # =========================================================================
    # Step 11b: Formation Predictions
    # =========================================================================
    step(11, total_steps, "Predicting Formations (SofaScore)")
    try:
        from scripts.prediction.formation_predictor import add_formation_predictions
        predictions = add_formation_predictions(predictions)
        formed = sum(1 for p in predictions if p.get("predicted_home_formation"))
        print(f"  Added formation predictions for {formed} matches")
    except Exception as e:
        print(f"  Formation prediction warning: {e}")
        log.warning(f"Formation prediction error: {e}")

    # =========================================================================
    # Step 11c: Predicted Match Stats
    # =========================================================================
    step(11, total_steps, "Predicting Match Stats (shots, cards, possession)")
    try:
        from scripts.prediction.match_stats_predictor import add_match_stats_predictions
        predictions = add_match_stats_predictions(predictions)
        has_stats = sum(1 for p in predictions if p.get("predicted_stats"))
        print(f"  Added predicted stats for {has_stats} matches")
    except Exception as e:
        print(f"  Match stats prediction warning: {e}")
        log.warning(f"Match stats prediction error: {e}")

    # Save enriched predictions back to predictions.json (preserving wrapper dict)
    if predictions:
        preds_path = DATA_DIR / "upcoming" / "predictions.json"
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        # Re-read the original wrapper dict to preserve metadata
        try:
            with open(preds_path) as f:
                wrapper = json.load(f)
            if isinstance(wrapper, dict):
                wrapper["predictions"] = predictions
            else:
                wrapper = {"predictions": predictions}
        except Exception:
            wrapper = {"predictions": predictions}
        # Preserve league field through enrichment (infer from team names when missing)
        from config.leagues import infer_league as _infer_league
        for _p in predictions:
            _p.setdefault("league", _infer_league(_p.get("home_team"), _p.get("away_team")))
        from config.settings import atomic_write_json as _awj
        _awj(preds_path, wrapper, cls=_get_json_encoder())
        print(f"  Saved enriched predictions ({len(predictions)} matches)")

    # =========================================================================
    # Step 11d: Multi-League Predictions (non-Serie-A leagues)
    # =========================================================================
    extra_leagues = [l for l in (leagues or []) if l != "serie_a"]
    if extra_leagues:
        step(11, total_steps, f"Multi-League Predictions ({', '.join(extra_leagues)})")
        try:
            from scripts.prediction.predict_league import run_predictions_for_league, LEAGUE_DISPLAY_NAMES
            for league in extra_leagues:
                display = LEAGUE_DISPLAY_NAMES.get(league, league)
                result = run_predictions_for_league(league)
                if result.get("count", 0) > 0:
                    print(f"  {display}: {result['count']} predictions")
                elif result.get("error"):
                    print(f"  {display}: ERROR - {result['error']}")
                else:
                    print(f"  {display}: skipped (no model or no matches)")
        except Exception as e:
            print(f"  Multi-league prediction warning: {e}")
            log.warning(f"Multi-league prediction error: {e}")

    # =========================================================================
    # Step 12: Generate Current Standings
    # =========================================================================
    step(12, total_steps, "Generating Current Standings")
    try:
        from scripts.prediction.standings_generator import generate_current_standings
        standings = generate_current_standings()
        print(f"  Generated standings for {standings.get('team_count', 0)} teams")
    except Exception as e:
        print(f"  Standings warning: {e}")
        log.warning(f"Standings error: {e}")

    # =========================================================================
    # Step 13: Generate H2H for Upcoming Matches
    # =========================================================================
    step(13, total_steps, "Generating H2H Records")
    try:
        from scripts.prediction.h2h_generator import generate_h2h_for_upcoming
        h2h = generate_h2h_for_upcoming()
        print(f"  Generated H2H for {h2h.get('match_count', 0)} Serie A matches")
        # Generate H2H for extra leagues (merges into same file)
        for league in extra_leagues:
            try:
                h2h_extra = generate_h2h_for_upcoming(league=league)
                extra_count = len(h2h_extra.get("h2h", {})) - h2h.get("match_count", 0)
                print(f"  Generated H2H for {extra_count} {league} matches")
            except Exception as e:
                print(f"  H2H {league} warning: {e}")
    except Exception as e:
        print(f"  H2H warning: {e}")
        log.warning(f"H2H error: {e}")

    # =========================================================================
    # Step 14: Extended Markets (Double Chance, Team Totals, etc.)
    # =========================================================================
    step(14, total_steps, "Generating Extended Markets")
    try:
        from scripts.betting.extended_markets import generate_extended_markets
        ext = generate_extended_markets()
        print(f"  Generated extended markets for {ext.get('match_count', 0)} matches")
    except Exception as e:
        print(f"  Extended markets warning: {e}")
        log.warning(f"Extended markets error: {e}")

    # =========================================================================
    # Step 15: Player Props (Goalscorer Predictions)
    # =========================================================================
    step(15, total_steps, "Generating Player Props")
    try:
        from scripts.betting.player_props import generate_player_props
        pp = generate_player_props()
        total_players = sum(len(m["players"]) for m in pp.get("matches", {}).values())
        print(f"  Generated player props for {pp.get('match_count', 0)} matches ({total_players} players)")
    except Exception as e:
        print(f"  Player props warning: {e}")
        log.warning(f"Player props error: {e}")

    # =========================================================================
    # Step 16: Sentiment Analysis (Perplexity AI)
    # =========================================================================
    step(16, total_steps, "Running Sentiment Analysis (Perplexity AI)")
    try:
        from scripts.prediction.sentiment_analyzer import analyze_all_upcoming_matches

        sentiment_results = analyze_all_upcoming_matches()
        home_edge = sum(1 for s in sentiment_results if s.sentiment_edge == "home")
        away_edge = sum(1 for s in sentiment_results if s.sentiment_edge == "away")
        contrarian = sum(1 for s in sentiment_results if s.contrarian_opportunity)
        print(f"  Analyzed sentiment for {len(sentiment_results)} matches")
        print(f"  Edge: {home_edge} home, {away_edge} away")
        if contrarian > 0:
            print(f"  {contrarian} contrarian opportunities detected!")
    except Exception as e:
        print(f"  Sentiment analysis warning: {e}")
        log.warning(f"Sentiment analysis error: {e}")

    # =========================================================================
    # Step 17: Player Analysis
    # =========================================================================
    step(17, total_steps, "Running Player Analysis")
    try:
        from scripts.analysis.player_analyzer import analyze_all_upcoming_matches as analyze_players

        player_results = analyze_players()
        home_adv = sum(1 for p in player_results if p.home_strength > p.away_strength)
        away_adv = sum(1 for p in player_results if p.away_strength > p.home_strength)
        print(f"  Analyzed players for {len(player_results)} matches")
        print(f"  Squad advantage: {home_adv} home, {away_adv} away")
    except Exception as e:
        print(f"  Player analysis warning: {e}")
        log.warning(f"Player analysis error: {e}")

    # =========================================================================
    # Step 18: Apply Intelligence Adjustments (Enhanced with market intel)
    # =========================================================================
    step(18, total_steps, "Applying Intelligence Adjustments")
    try:
        from scripts.prediction.intelligence_integrator import apply_all_intelligence

        pred_path = DATA_DIR / "upcoming" / "predictions.json"
        with open(pred_path) as f:
            pred_data = json.load(f)

        # Build lookup maps by match name
        sentiment_map = {s.match: s for s in sentiment_results}
        player_map = {p.match: p for p in player_results}

        adjusted_count = 0
        market_intel_count = 0
        for pred in pred_data.get("predictions", []):
            match_name = pred.get("match", "")
            sentiment = sentiment_map.get(match_name)
            player = player_map.get(match_name)
            intel = market_intel_results.get(match_name)

            if sentiment or player or intel:
                apply_all_intelligence(
                    pred,
                    sentiment=sentiment,
                    player_analysis=player,
                    market_intel=intel,
                )
                adjusted_count += 1
                if intel:
                    market_intel_count += 1

        pred_data["intelligence_applied"] = True
        pred_data["intelligence_sources"] = []
        if sentiment_results:
            pred_data["intelligence_sources"].append("web_search_sentiment")
        if player_results:
            pred_data["intelligence_sources"].append("player_analysis")
        if market_intel_count > 0:
            pred_data["intelligence_sources"].append("market_intelligence")

        # Preserve league field through enrichment (file-level league wins, then team inference)
        from config.leagues import infer_league as _infer_league
        _file_lg = pred_data.get("league")
        for _p in pred_data.get("predictions", []):
            _p.setdefault("league", _file_lg or _infer_league(_p.get("home_team"), _p.get("away_team")))
        from config.settings import atomic_write_json as _awj2
        _awj2(pred_path, pred_data, cls=_get_json_encoder())

        print(f"  Applied intelligence to {adjusted_count}/{len(pred_data.get('predictions', []))} predictions")
        if sentiment_results:
            print(f"  Perplexity sentiment: {len(sentiment_results)} matches")
        if player_results:
            print(f"  Player analysis: {len(player_results)} matches")
        if market_intel_count > 0:
            print(f"  Market intelligence: {market_intel_count} matches")
    except Exception as e:
        print(f"  Intelligence integration warning: {e}")
        log.warning(f"Intelligence integration error: {e}")

    # =========================================================================
    # Step 18b: EPL Supplementary Data (bookmaker, cross-market, sentiment,
    #           weather, referees — mirrors Serie A supplementary data)
    # =========================================================================
    if extra_leagues and "premier_league" in extra_leagues:
        print(f"\n  Generating EPL supplementary data (bookmaker, cross-market, sentiment, weather, referees)...")
        try:
            from scripts.prediction.generate_epl_supplementary import generate_all_epl_supplementary
            generate_all_epl_supplementary()
        except Exception as e:
            print(f"  EPL supplementary data warning: {e}")
            log.warning(f"EPL supplementary data error: {e}")

    # =========================================================================
    # Step 19: Over/Under model
    # =========================================================================
    step(19, total_steps, "Running Over/Under Model")
    try:
        from scripts.models.over_under_model import (
            generate_over_under_predictions,
            save_over_under_predictions
        )

        # Extract ML O/U probabilities from ensemble predictions (step 11)
        ml_ou_probs = {}
        for p in predictions:
            ou_ml = p.get("component_predictions", {}).get("over_under_ml", {})
            if ou_ml:
                ml_ou_probs[p["match"]] = {float(k): v for k, v in ou_ml.items()}
        if ml_ou_probs:
            print(f"  ML O/U probs available for {len(ml_ou_probs)} matches")

        goal_preds, ou_bets = generate_over_under_predictions(ml_ou_probs=ml_ou_probs)
        save_over_under_predictions(goal_preds, ou_bets)
        recommended_ou = [b for b in ou_bets if b.recommendation == "BET"]
        print(f"  Generated {len(goal_preds)} goal predictions")
        print(f"  Found {len(recommended_ou)} recommended O/U bets")
    except Exception as e:
        print(f"  O/U model warning: {e}")
        log.warning(f"O/U model error: {e}")

    # =========================================================================
    # Step 20: Handicap model
    # =========================================================================
    step(20, total_steps, "Running Handicap Model")
    try:
        from scripts.models.handicap_model import (
            generate_handicap_predictions,
            save_handicap_predictions
        )

        margin_preds, hc_bets = generate_handicap_predictions()
        save_handicap_predictions(margin_preds, hc_bets)
        recommended_hc = [b for b in hc_bets if b.recommendation == "BET"]
        print(f"  Generated {len(margin_preds)} margin predictions")
        print(f"  Found {len(recommended_hc)} recommended handicap bets")
    except Exception as e:
        print(f"  Handicap model warning: {e}")
        log.warning(f"Handicap model error: {e}")

    # =========================================================================
    # Step 21: Cards model
    # =========================================================================
    step(21, total_steps, "Running Cards Model")
    try:
        from scripts.models.cards_model import (
            generate_cards_predictions,
            save_cards_predictions
        )

        cards_preds, cards_bets = generate_cards_predictions()
        save_cards_predictions(cards_preds, cards_bets)
        recommended_cards = [b for b in cards_bets if b.recommendation == "BET"]
        print(f"  Generated {len(cards_preds)} cards predictions")
        print(f"  Found {len(recommended_cards)} recommended cards bets")
    except Exception as e:
        print(f"  Cards model warning: {e}")
        log.warning(f"Cards model error: {e}")

    # =========================================================================
    # Step 22: BTTS & Corners model
    # =========================================================================
    step(22, total_steps, "Running BTTS & Corners Model")
    try:
        from scripts.models.btts_corners_model import (
            generate_all_predictions,
            save_predictions
        )

        btts_preds, corners_preds, bc_bets = generate_all_predictions()
        save_predictions(btts_preds, corners_preds, bc_bets)
        recommended_bc = [b for b in bc_bets if b.recommendation == "BET"]
        print(f"  Generated {len(btts_preds)} BTTS, {len(corners_preds)} corners predictions")
        print(f"  Found {len(recommended_bc)} recommended BTTS/corners bets")
    except Exception as e:
        print(f"  BTTS/Corners model warning: {e}")
        log.warning(f"BTTS/Corners model error: {e}")

    # =========================================================================
    # Step 22b: ML Market Predictions (CatBoost — all markets)
    # =========================================================================
    step(22, total_steps, "Running ML Market Predictions (CatBoost)")
    try:
        from scripts.models.ml_market_predictions import generate_ml_predictions, save_ml_predictions
        ml_preds = generate_ml_predictions()
        if ml_preds:
            save_ml_predictions(ml_preds)
            print(f"  Generated CatBoost ML predictions for {len(ml_preds)} matches")
            print(f"  Markets: 1X2, O/U, BTTS, Cards, Corners, HT, DC, Exact Score")
    except Exception as e:
        print(f"  ML predictions warning: {e}")
        log.warning(f"ML predictions error: {e}")

    # =========================================================================
    # Step 22c: Player-Level Predictions (CatBoost per-player markets)
    # =========================================================================
    step(22, total_steps, "Running Player-Level Predictions")
    try:
        from scripts.betting.player_predictions import (
            load_player_data, build_player_features, predict_player_markets
        )
        import json as _json

        pms = load_player_data()
        pms = build_player_features(pms)

        # Get upcoming lineups from Sofascore data
        lineups_path = DATA_DIR / "upcoming" / "lineups.json"
        player_preds_all = {}

        if lineups_path.exists():
            with open(lineups_path) as f:
                lineups = _json.load(f)

            for match_key, match_data in lineups.items():
                match_preds = {"home_players": [], "away_players": []}
                for side, is_home in [("home", True), ("away", False)]:
                    players = match_data.get(f"{side}_lineup", [])
                    for p in players:
                        pred = predict_player_markets(
                            player_name=p.get("player_name", ""),
                            player_id=p.get("player_id", 0),
                            team=match_data.get(f"{side}_team", ""),
                            opponent=match_data.get(f"{'away' if is_home else 'home'}_team", ""),
                            position=p.get("position", "M"),
                            is_home=is_home,
                            pms=pms,
                        )
                        if pred and pred.get("markets"):
                            match_preds[f"{side}_players"].append(pred)
                player_preds_all[match_key] = match_preds

            out_path = DATA_DIR / "upcoming" / "player_predictions.json"
            from config.settings import atomic_write_json as _awj3
            _awj3(out_path, player_preds_all)
            total_players = sum(
                len(m.get("home_players", [])) + len(m.get("away_players", []))
                for m in player_preds_all.values()
            )
            print(f"  Generated player predictions for {total_players} players across {len(player_preds_all)} matches")
        else:
            # Fallback: use latest Sofascore data for current season starters
            print("  No lineups.json found — skipping player predictions (run lineup scraper first)")
    except Exception as e:
        print(f"  Player predictions warning: {e}")
        log.warning(f"Player predictions error: {e}")

    # =========================================================================
    # Step 22d: Player Prop Odds & Value Scanner
    # =========================================================================
    step(22, total_steps, "Fetching Player Prop Odds & Scanning for Value")
    try:
        from scripts.betting.player_prop_odds import (
            fetch_player_prop_odds, save_player_prop_odds,
            scan_for_value, save_value_bets
        )
        props = fetch_player_prop_odds(use_cache=quick)
        if props:
            save_player_prop_odds(props)
            value_bets = scan_for_value(props)
            if value_bets:
                save_value_bets(value_bets)
                print(f"  Fetched player props for {len(props)} matches")
                print(f"  Found {len(value_bets)} player prop value bets (avg edge: "
                      f"{sum(v['edge_pct'] for v in value_bets)/len(value_bets):+.1f}%)")
            else:
                print(f"  Fetched player props for {len(props)} matches — no value bets found")
        else:
            print("  No player prop odds available")
    except Exception as e:
        print(f"  Player prop odds warning: {e}")
        log.warning(f"Player prop odds error: {e}")

    # =========================================================================
    # Step 23: Betting Engine
    # =========================================================================
    step(23, total_steps, "Running Betting Engine")
    try:
        from scripts.betting.betting_unified import generate_betting_slip

        slip = generate_betting_slip(include_parlays=False)
        if slip:
            print(f"  Generated betting slip with {slip['summary']['recommended_bets']} bets")
    except Exception as e:
        print(f"  Betting engine warning: {e}")
        log.warning(f"Betting engine error: {e}")

    # =========================================================================
    # Step 24: Advanced Betting Engine
    # When BETTING_DRY_RUN_FROM_MORNING=true, this step generates CANDIDATES
    # only — no journal write, no archive. Bets commit at T-30 via
    # run_pre_kickoff() so they price against fresh odds + confirmed lineups.
    # See journal analysis 2026-04-25: 91% of bets were placed >24h before
    # kickoff at -5% ROI; the few <24h bets ran +63% mean ROI on n=12.
    # =========================================================================
    candidate_only = os.environ.get("BETTING_DRY_RUN_FROM_MORNING", "false").lower() in ("1", "true", "yes")
    step_label = "Generating Bet Candidates (T-30 will commit)" if candidate_only else "Running Advanced Multi-Market Engine"
    step(24, total_steps, step_label)
    try:
        from scripts.betting.betting_unified import (
            generate_unified_report,
            save_report,
            BettingConfig,
            UnifiedBettingEngine,
        )

        if candidate_only:
            cfg = BettingConfig.from_config(bankroll_override=bankroll)
            cfg.dry_run = True
            engine = UnifiedBettingEngine(cfg)
            slip, _ = engine.run()
            n_candidates = len(slip.bets) if slip else 0
            total_stake = sum(b.stake_amount for b in (slip.bets if slip else []))
            print(f"  Generated {n_candidates} candidates (NOT journaled — T-30 will commit)")
            print(f"  If committed: ${total_stake:.2f} total stake")

            # Persist candidates so the T-30 stage can re-evaluate them
            import json
            candidates_payload = {
                "generated_at": datetime.now().isoformat(),
                "candidates": [
                    {
                        "match": b.match, "date": b.date, "market": b.market,
                        "selection": b.selection, "model_prob": b.model_prob,
                        "best_odds": b.best_odds, "best_bookmaker": b.best_bookmaker,
                        "edge_pct": b.edge_pct, "ev_per_unit": b.ev_per_unit,
                        "stake_amount": b.stake_amount,
                        "confidence_tier": b.confidence_tier,
                    } for b in (slip.bets if slip else [])
                ],
            }
            cand_path = Path("data/upcoming/betting_candidates.json")
            cand_path.parent.mkdir(parents=True, exist_ok=True)
            cand_path.write_text(json.dumps(candidates_payload, indent=2, default=str))
            report = None  # downstream parlay/etc steps treat as no-op
        else:
            report = generate_unified_report(bankroll)
            save_report(report)
            print(f"  Generated unified report with {report['summary']['total_bets']} optimized bets")
            print(f"  Total stake: ${report['summary']['total_stake']:.2f}")
            print(f"  Potential profit: ${report['summary']['total_potential_profit']:.2f}")

            # Archive bets to persistent log so they survive report overwrites
            if report.get("bets"):
                _archive_bets(report["bets"], report.get("generated_at", ""))
    except Exception as e:
        print(f"  Advanced engine warning: {e}")
        log.warning(f"Advanced engine error: {e}")
        report = None

    # =========================================================================
    # Step 25: Parlay Generator
    # =========================================================================
    step(25, total_steps, "Generating Parlay Recommendations")
    try:
        from scripts.betting.parlay_generator import generate_parlay_report
        parlay_report = generate_parlay_report(bankroll=bankroll)
        regenerated = parlay_report.get("regenerated", True)
        top_picks = parlay_report.get("top_picks", [])
        if regenerated:
            print(f"  Generated {parlay_report['total_parlays']} parlays from {parlay_report['total_legs_available']} value legs")
            if top_picks:
                print(f"  Top {len(top_picks)} picks selected")
        else:
            print(f"  Cached: {parlay_report['total_parlays']} parlays (inputs unchanged)")

        # Notify parlays ready (skips if not regenerated)
        try:
            from scripts.pipeline.notify import notify_parlays_ready
            notify_parlays_ready(parlay_report=parlay_report)
        except Exception:
            pass
    except Exception as e:
        print(f"  Parlay generator warning: {e}")
        log.warning(f"Parlay generator error: {e}")

    # =========================================================================
    # Step 25b: CLV Tracking (Closing Line Value)
    # =========================================================================
    step(25, total_steps, "CLV Tracking — Recording Placements & Computing CLV")
    try:
        from scripts.betting.clv_tracker import record_bet_placement, track_clv_from_slip, get_clv_summary
        n_recorded = record_bet_placement()
        if n_recorded:
            print(f"  Recorded {n_recorded} new bet placements")
        clv_result = track_clv_from_slip()
        if clv_result.get("new_tracked", 0) > 0:
            print(f"  Tracked CLV for {clv_result['new_tracked']} bets")
        summary = get_clv_summary()
        if summary["total_tracked"] > 0:
            print(f"  Running CLV: {summary['running_clv_pct']:+.2f}% "
                  f"({summary['positive_clv_count']}/{summary['total_tracked']} positive)")
        else:
            print(f"  No CLV data yet (will populate after matchday)")
    except Exception as e:
        print(f"  CLV tracking warning: {e}")
        log.warning(f"CLV tracking error: {e}")

    # =========================================================================
    # Step 26: AI Reasoning for Bets (GPT-4o-mini)
    # =========================================================================
    step(26, total_steps, "Generating AI Reasoning (GPT-4o-mini)")
    try:
        from scripts.prediction.ai_reasoning import generate_bet_reasoning

        if report and report.get("bets"):
            match_contexts = {}
            for pred in predictions:
                match_name = pred.get("match", "")
                match_contexts[match_name] = {
                    "probabilities": pred.get("probabilities", {}),
                    "home_xg": pred.get("home_xg"),
                    "away_xg": pred.get("away_xg"),
                    "confidence_level": pred.get("confidence_level", "MEDIUM"),
                    "intelligence_adjustments": pred.get("intelligence_adjustments", []),
                    "market_intelligence": pred.get("market_intelligence", {}),
                }
                for s in sentiment_results:
                    if s.match == match_name:
                        match_contexts[match_name]["home_sentiment"] = s.home_composite
                        match_contexts[match_name]["away_sentiment"] = s.away_composite
                        match_contexts[match_name]["sentiment_reasoning"] = s.reasoning[:500]

            reasoning_count = 0
            for bet in report["bets"]:
                match_name = bet.get("match", "")
                context = match_contexts.get(match_name)
                try:
                    reasoning = generate_bet_reasoning(bet, context)
                    bet["ai_reasoning"] = reasoning.get("analysis", "")
                    bet["ai_confidence"] = reasoning.get("confidence_level", "MEDIUM")
                    reasoning_count += 1
                except Exception as re:
                    log.debug(f"Reasoning failed for {match_name}: {re}")

            if reasoning_count > 0:
                from scripts.betting.betting_unified import save_report as resave
                resave(report)

            print(f"  Generated AI reasoning for {reasoning_count}/{len(report['bets'])} bets")
        else:
            print(f"  No bets to analyze")
    except Exception as e:
        print(f"  AI reasoning warning: {e}")
        log.warning(f"AI reasoning error: {e}")

    # =========================================================================
    # Step 27: Apply Italian Market Standards
    # =========================================================================
    step(27, total_steps, "Applying Italian Market Standards")
    try:
        from scripts.betting.italian_market_standards import apply_all_italian_standards

        results = apply_all_italian_standards()
        total_converted = sum(r.get("filtered_bets", 0) for r in results.values())
        print(f"  Applied Italian standards to {len(results)} files")
        print(f"  {total_converted} bets formatted for SISAL/Italian markets")
    except Exception as e:
        print(f"  Italian standards warning: {e}")
        log.warning(f"Italian standards error: {e}")

    # =========================================================================
    # Step 28: Fetch Results, Auto-Settle & CLV Tracking
    # =========================================================================
    step(28, total_steps, "Fetching Results, Auto-Settling & CLV Tracking")
    try:
        from scripts.data.results_fetcher import fetch_and_settle, get_roi_summary

        settlement = fetch_and_settle()
        if settlement.get("settled", 0) > 0:
            print(f"  Settled {settlement['settled']} bets: "
                  f"{settlement['won']}W {settlement['lost']}L")
            print(f"  P&L: {'+'if settlement['profit'] > 0 else ''}${settlement['profit']:.2f}")
            print(f"  Balance: ${settlement['new_balance']:.2f}")

            # Mark settled bets in persistent log too
            _update_placed_bets_log(settlement)
        else:
            print(f"  No pending bets to settle (matches not yet completed)")

        # Show lifetime P&L summary
        roi = get_roi_summary()
        if roi.get("total_bets", 0) > 0:
            print(f"  Lifetime: {roi['total_bets']} bets, "
                  f"ROI: {roi['roi']:+.1f}%, "
                  f"Balance: ${roi['current_balance']:.2f}")

        # Show persistent bet log stats
        log_path = DATA_DIR / "betting" / "placed_bets_log.json"
        if log_path.exists():
            with open(log_path) as f:
                bet_log = json.load(f)
            pending = sum(1 for b in bet_log if b.get("status") == "pending")
            settled = sum(1 for b in bet_log if b.get("status") in ("won", "lost"))
            if pending > 0 or settled > 0:
                print(f"  Bet log: {len(bet_log)} total, {pending} pending, {settled} settled")

        # CLV tracking
        try:
            from scripts.betting.clv_tracker import track_clv_for_settled_bets, get_clv_summary

            history_path = DATA_DIR / "betting" / "history.json"
            if history_path.exists():
                with open(history_path) as f:
                    history = json.load(f)
                hist_list = (history if isinstance(history, list)
                             else history.get("settled_bets", history.get("bets", [])))
                settled_bets = [
                    b for b in hist_list
                    if isinstance(b, dict) and b.get("status") in ("won", "lost")
                ]
                if settled_bets:
                    clv_result = track_clv_for_settled_bets(settled_bets)
                    if clv_result.get("total_tracked", 0) > 0:
                        print(f"  CLV: {clv_result['running_clv_pct']:+.2f}% "
                              f"(positive rate: {clv_result['positive_rate']:.0%})")
        except Exception as clv_e:
            log.debug(f"CLV tracking: {clv_e}")

    except Exception as e:
        print(f"  Results fetcher warning: {e}")
        log.warning(f"Results fetcher error: {e}")

    # =========================================================================
    # Step 29: Feedback Analysis & Weight Optimization
    # =========================================================================
    step(29, total_steps, "Feedback Analysis & Weight Optimization")
    try:
        from scripts.analysis.feedback_analyzer import run_feedback_analysis
        from scripts.models.weight_optimizer import run_weight_optimization

        fb_analysis = run_feedback_analysis()
        n_settled = fb_analysis.get("n_settled", 0)
        overall = fb_analysis.get("overall", {})
        print(f"  Settled predictions: {n_settled}")

        if n_settled > 0:
            print(f"  Accuracy: {overall.get('accuracy', 0):.1%}, "
                  f"Mean Brier: {overall.get('mean_brier', 0):.4f}")

            # Best method
            method_brier = fb_analysis.get("method_brier_scores", {})
            if method_brier:
                best = min(method_brier.items(), key=lambda x: x[1]["mean_brier"])
                print(f"  Best method: {best[0]} (Brier: {best[1]['mean_brier']:.4f})")

            # Intelligence effectiveness
            intel = fb_analysis.get("intelligence_effectiveness", {})
            for source, data in intel.items():
                print(f"  Intelligence [{source}]: {data['verdict']} "
                      f"(helped {data['helped']}/{data['total_applications']})")

        # Run weight optimization
        opt_result = run_weight_optimization()
        weights_info = opt_result.get("weights", {})
        print(f"  Weight status: {weights_info.get('status', 'unknown')}")

        # Factor adjustments
        factor_info = opt_result.get("factor_adjustments", {})
        n_adjusted = sum(1 for d in factor_info.get("details", {}).values()
                         if d.get("action") in ("decay", "boost"))
        if n_adjusted:
            print(f"  Factor adjustments: {n_adjusted} factors modified")

        # Calibration
        cal_info = opt_result.get("calibration", {})
        cal_status = cal_info.get("status", "unknown")
        if cal_status == "active":
            print(f"  Calibration: {cal_info.get('bias', 'unknown')} "
                  f"(error: {cal_info.get('mean_calibration_error', 0):.3f})")
        else:
            print(f"  Calibration: {cal_status}")

    except Exception as e:
        print(f"  Feedback analysis warning: {e}")
        log.warning(f"Feedback analysis error: {e}")

    # =========================================================================
    # Step 29b: Learning Loop (Parallel Analysis Agents)
    # =========================================================================
    step(29, total_steps, "Learning Loop (Prediction Audit + Drift + ROI)")
    try:
        from scripts.pipeline.learning_loop import run_learning_loop
        ll_result = run_learning_loop()
        agents_ok = sum(1 for v in ll_result.values() if v.get("status") == "ok")
        print(f"  Learning loop: {agents_ok}/3 agents completed successfully")
        for agent, data in ll_result.items():
            if data.get("status") == "ok":
                print(f"    {agent}: {data.get('summary', 'done')}")
            else:
                print(f"    {agent}: {data.get('error', 'failed')}")
    except Exception as e:
        print(f"  Learning loop warning: {e}")
        log.warning(f"Learning loop error: {e}")

    # =========================================================================
    # Step 30: Performance Dashboard
    # =========================================================================
    step(30, total_steps, "Performance Dashboard")
    try:
        from scripts.analysis.performance_dashboard import generate_dashboard, print_dashboard
        dashboard = generate_dashboard()
        pa = dashboard.get("prediction_accuracy", {})
        bp = dashboard.get("betting_performance", {})
        print(f"  Verified predictions: {pa.get('total_verified', 0)} "
              f"(accuracy: {pa.get('accuracy', 0):.1%})")
        print(f"  Betting P&L: ${bp.get('profit', 0):+.2f} "
              f"(ROI: {bp.get('roi', 0):+.1%})")
        fd = dashboard.get("feature_drift", {})
        print(f"  Feature health: {fd.get('healthy', 0)}/{fd.get('total_checked', 0)}")
    except Exception as e:
        print(f"  Dashboard warning: {e}")
        log.warning(f"Dashboard error: {e}")

    # =========================================================================
    # Step 31 (optional): Model Challenger
    # =========================================================================
    if run_challenger:
        step(31, total_steps, "Model Challenger (Retrain + Validate + Swap)")
        try:
            from scripts.models.model_challenger import run_model_challenger
            ch_result = run_model_challenger()
            decision = ch_result.get("decision", "unknown")
            print(f"  Challenger decision: {decision}")
            if decision == "swap":
                print(f"  New model metrics: {ch_result.get('new_metrics', {})}")
            elif decision == "keep":
                print(f"  Current model still better. Reason: {ch_result.get('reason', '')}")
        except Exception as e:
            print(f"  Model challenger warning: {e}")
            log.warning(f"Model challenger error: {e}")

    # =========================================================================
    # Step 32: Generate Unified Report
    # =========================================================================
    step(32, total_steps, "Unified Report (consolidate all match intelligence)")
    try:
        from scripts.prediction.generate_unified_report import generate_unified_report
        ur = generate_unified_report()
        match_count = len(ur.get("matches", {}))
        print(f"  Consolidated {match_count} matches into unified_report.json")
        avg_sources = sum(
            m.get("_data_sources", 0)
            for m in ur.get("matches", {}).values()
        ) / max(match_count, 1)
        print(f"  Average data sources per match: {avg_sources:.1f}")
    except Exception as e:
        print(f"  Unified report warning: {e}")
        log.warning(f"Unified report error: {e}")

    # =========================================================================
    # Final Summary
    # =========================================================================
    elapsed = time.time() - start_time

    banner("PIPELINE COMPLETE")
    print(f"\n  Elapsed Time: {elapsed:.1f} seconds")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Notify pipeline complete
    try:
        from scripts.pipeline.notify import notify_pipeline_done
        n_preds = len(predictions) if predictions else 0
        n_vbets = report["summary"].get("total_bets", 0) if report else 0
        notify_pipeline_done(n_predictions=n_preds, n_value_bets=n_vbets, elapsed_sec=elapsed)
    except Exception:
        pass

    # Morning briefing — send if this is the morning run (before noon)
    try:
        if datetime.now().hour < 12:
            from scripts.pipeline.notify import notify_morning_briefing
            notify_morning_briefing()
    except Exception:
        pass

    if report:
        print("\n" + "=" * 70)
        print(" FINAL BETTING SUMMARY")
        print("=" * 70)

        summary = report["summary"]
        total_stake = summary.get('total_stake', 0)
        stake_pct = total_stake / bankroll * 100 if bankroll else 0
        total_profit = summary.get('total_potential_profit', 0)

        print(f"\n  Bankroll: ${bankroll:.2f}")
        print(f"  Total Bets: {summary.get('total_bets', 0)}")
        print(f"  Total Stake: ${total_stake:.2f} ({stake_pct:.1f}%)")
        print(f"  Potential Profit: ${total_profit:.2f}")

        bets = report.get("bets", [])
        if bets:
            avg_edge = sum(b.get('edge_pct', b.get('value_pct', 0)) for b in bets) / len(bets)
            print(f"  Average Edge: {avg_edge:+.1f}%")

            print(f"\n  RECOMMENDED BETS:")
            print(f"  {'Match':<25} {'Selection':>15} {'Odds':>6} {'Stake':>8}")
            print("  " + "-" * 60)

            for bet in bets:
                match = bet.get('match', '')
                sel = bet.get('selection', '')
                odds = bet.get('best_odds', bet.get('odds', 0))
                stake = bet.get('stake_amount', bet.get('stake', 0))
                print(f"  {match:<25} {sel:>15} {odds:>6.2f} ${stake:>7.2f}")

    # Output files
    print("\n" + "=" * 70)
    print(" OUTPUT FILES")
    print("=" * 70)

    output_files = [
        "data/quality_report.json",
        "data/quality/correlation_report.json",
        "data/squads/current_squads.json",
        "data/upcoming/odds.json",
        "data/upcoming/odds_totals.json",
        "data/upcoming/odds_spreads.json",
        "data/upcoming/odds_btts.json",
        "data/upcoming/odds_dnb.json",
        "data/upcoming/odds_bookmakers.json",
        "data/upcoming/bookmaker_analysis.json",
        "data/upcoming/odds_movement.json",
        "data/upcoming/cross_market_signals.json",
        "data/upcoming/market_intelligence.json",
        "data/upcoming/predictions.json",
        "data/upcoming/sentiment_analysis.json",
        "data/upcoming/player_analysis.json",
        "data/upcoming/goal_predictions.json",
        "data/upcoming/margin_predictions.json",
        "data/upcoming/cards_predictions.json",
        "data/upcoming/btts_predictions.json",
        "data/upcoming/corners_predictions.json",
        "data/upcoming/over_under_bets.json",
        "data/upcoming/handicap_bets.json",
        "data/upcoming/cards_bets.json",
        "data/upcoming/btts_corners_bets.json",
        "data/upcoming/standings.json",
        "data/upcoming/h2h_upcoming.json",
        "data/upcoming/extended_markets.json",
        "data/upcoming/unified_report.json",
        "data/upcoming/player_props.json",
        "data/upcoming/results.json",
        "data/betting/betting_slip.json",
        "data/betting/unified_report.json",
        "data/betting/history.json",
        "data/betting/bankroll.json",
        "data/betting/clv_history.json",
        "data/upcoming/predictions_archive.json",
        "data/feedback/analysis.json",
        "data/feedback/optimized_weights.json",
        "data/feedback/factor_adjustments.json",
        "data/feedback/calibration_curve.json",
        "data/performance_dashboard.json",
    ]

    base_path = Path(__file__).parent.parent.parent

    for f in output_files:
        path = base_path / f
        if path.exists():
            size = path.stat().st_size
            print(f"  {f} ({size:,} bytes)")
        else:
            print(f"  {f} (not found)")

    print("\n" + "=" * 70)
    print(" NEXT STEPS")
    print("=" * 70)
    print("\n  1. Review the betting recommendations above")
    print("  2. Compare with bookmaker odds for value")
    print("  3. Place bets with recommended stakes")
    print("  4. Track results with: python scripts/performance_tracker.py")
    print("  5. Run --snapshot-only 3-4x daily for temporal depth")

    # Post-pipeline data validation — catch issues before they reach users
    try:
        from scripts.utils.data_validator import validate_pipeline_output
        print("\n" + "=" * 70)
        print(" DATA VALIDATION")
        print("=" * 70)
        validation = validate_pipeline_output()
        n_crit = len(validation.get("critical", []))
        n_warn = len(validation.get("warning", []))
        if n_crit:
            print(f"\n  ⚠️  {n_crit} CRITICAL issues found — check logs!")
            try:
                from scripts.pipeline.notify import notify
                notify(
                    f"Pipeline completed but {n_crit} data issues detected: "
                    + "; ".join(validation["critical"][:3]),
                    title="Data Validation Warning",
                    level="warning",
                )
            except Exception:
                pass
        elif n_warn:
            print(f"\n  {n_warn} warnings (non-critical)")
        else:
            print("\n  All checks passed ✓")
    except Exception as e:
        log.warning("Data validation skipped: %s", e)

    # Release pipeline lock
    if _pipeline_lock_fd:
        try:
            fcntl.flock(_pipeline_lock_fd, fcntl.LOCK_UN)
            _pipeline_lock_fd.close()
        except Exception:
            pass

    return report


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Full Betting Pipeline")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode (use cached data)")
    parser.add_argument("--bankroll", type=float, default=0,
                       help="Bankroll amount (default: 0 = auto-load from journal)")
    parser.add_argument("--snapshot-only", action="store_true",
                       help="Only run Steps 1-6 (fetch + snapshot + squads, ~6 credits)")
    parser.add_argument("--pre-kickoff", action="store_true",
                       help="Pre-kickoff mode: fetch lineups + quick predictions (~25s)")
    parser.add_argument("--live-monitor", action="store_true",
                       help="Live match monitor: poll scores + odds every 15 min")
    parser.add_argument("--live-once", action="store_true",
                       help="Live monitor: single poll (2 API calls)")
    parser.add_argument("--live-status", action="store_true",
                       help="Show current live monitoring status")
    parser.add_argument("--no-parallel", action="store_true",
                       help="Disable parallel data collection (debug mode)")
    parser.add_argument("--challenger", action="store_true",
                       help="Run model challenger after pipeline (retrain + validate + swap)")
    parser.add_argument("--incremental", action="store_true",
                       help="Incremental refresh: only new results, fresh odds if stale, new predictions (~4 credits)")
    from config.leagues import ACTIVE_LEAGUES as _AL
    _default_leagues = ",".join(_AL)
    parser.add_argument("--leagues", type=str, default=_default_leagues,
                       help=f"Comma-separated leagues to run (default: {_default_leagues})")
    args = parser.parse_args()

    # Auto-load bankroll from journal if not explicitly set
    if args.bankroll <= 0:
        try:
            from scripts.betting.bankroll_loader import get_effective_bankroll
            args.bankroll = get_effective_bankroll()
        except Exception as e:
            log.warning(f"Failed to auto-load bankroll: {e} — using 1000")
            args.bankroll = 1000.0

    # Parse leagues list
    leagues = [l.strip() for l in args.leagues.split(",") if l.strip()]
    if not leagues:
        leagues = ["serie_a"]

    if args.live_monitor:
        from scripts.data.live_monitor import watch_loop
        watch_loop()
    elif args.live_once:
        from scripts.data.live_monitor import poll_once
        poll_once()
    elif args.live_status:
        from scripts.data.live_monitor import show_status
        show_status()
    elif args.pre_kickoff:
        run_pre_kickoff(bankroll=args.bankroll)
    elif args.incremental:
        run_incremental(bankroll=args.bankroll, leagues=leagues)
    else:
        run_pipeline(
            quick=args.quick, bankroll=args.bankroll,
            snapshot_only=args.snapshot_only,
            no_parallel=args.no_parallel,
            run_challenger=args.challenger,
            leagues=leagues,
        )


if __name__ == "__main__":
    main()
