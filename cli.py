"""CLI entry point for the Serie A pipeline."""

import logging
import sys

import click

from config.settings import SEASONS
from ml.config import FeatureConfig, MODEL_TYPES, TuningConfig
from storage.paths import ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("seriea")


@click.group()
def main():
    """Serie A pipeline: feature engineering, models, retraining."""
    ensure_dirs()


# ---------------------------------------------------------------------------
# Scraping commands
# ---------------------------------------------------------------------------


@main.command()
@click.option("--season", default=SEASONS[-1], help="Season to scrape (e.g. 2024-2025)")
def scrape_fixtures(season: str):
    """Scrape the fixture list for a season from FBref."""
    from scraper.fixtures import scrape_season_fixtures

    log.info("Scraping fixtures for %s", season)
    df = scrape_season_fixtures(season)
    log.info("Scraped %d fixtures", len(df))


@main.command()
@click.option("--season", default=SEASONS[-1], help="Season to fetch odds for")
def fetch_odds(season: str):
    """Download betting odds from football-data.co.uk."""
    from scraper.odds import download_odds

    log.info("Downloading betting odds for %s", season)
    df = download_odds(season)
    log.info("Downloaded %d match odds", len(df))


@main.command()
def fetch_weather():
    """Fetch historical weather data for all parsed matches."""
    import pandas as pd
    from scraper.weather import fetch_weather_for_matches
    from storage.paths import parsed_path

    matches_path = parsed_path("matches")
    if not matches_path.exists():
        log.error("matches.parquet not found. It is produced by scripts/pipeline/refresh_weekly_data.py (weekly-data-refresh).")
        return

    matches = pd.read_parquet(matches_path)
    log.info("Fetching weather for %d matches", len(matches))
    weather = fetch_weather_for_matches(matches)
    log.info("Weather data: %d rows x %d columns", len(weather), len(weather.columns))


@main.command()
@click.option("--season", default=SEASONS[-1], help="Season to scrape")
def fetch_transfers(season: str):
    """Scrape transfer and market value data from Transfermarkt."""
    from scraper.transfermarkt import scrape_squad_market_values, scrape_transfers

    log.info("Scraping market values for %s", season)
    mv = scrape_squad_market_values(season)
    log.info("Market values: %d players", len(mv))

    log.info("Scraping transfers for %s", season)
    tr = scrape_transfers(season)
    log.info("Transfers: %d records", len(tr))


@main.command()
def import_historical():
    """Import historical match data from football-data.co.uk (2020-2024)."""
    from scraper.historical import import_all_historical

    log.info("Importing historical match data from football-data.co.uk")
    df = import_all_historical()
    log.info("Total matches in dataset: %d", len(df))


@main.command()
@click.option("--season", default=None, help="Season to scrape (default: all)")
def fetch_referees(season: str | None):
    """Scrape referee assignments from worldfootball.net."""
    from scraper.referee import scrape_all_referee_assignments

    seasons = [season] if season else None
    log.info("Scraping referee assignments%s", f" for {season}" if season else " for all seasons")
    df = scrape_all_referee_assignments(seasons=seasons)
    log.info("Referee assignments: %d records", len(df))


# ---------------------------------------------------------------------------
# Parsing commands
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Feature engineering commands
# ---------------------------------------------------------------------------


@main.command()
@click.option("--season", default=None, help="Season to build features for (default: all)")
def features(season: str | None):
    """Build ML-ready feature table from parsed data."""
    from features.build import build_features

    log.info("Building features%s", f" for {season}" if season else "")
    df = build_features(season=season)
    log.info("Feature table: %d rows x %d columns", len(df), len(df.columns))


# ---------------------------------------------------------------------------
# Pipeline commands
# ---------------------------------------------------------------------------


@main.command()
def status():
    """Show pipeline status: parsed-table and feature counts."""
    from storage.paths import parsed_path, features_path
    import pandas as pd

    for table in ["matches", "player_stats", "goalkeeper_stats", "shots", "lineups", "events"]:
        p = parsed_path(table)
        if p.exists():
            df = pd.read_parquet(p)
            click.echo(f"  {table}: {len(df)} rows x {len(df.columns)} cols")
        else:
            click.echo(f"  {table}: not yet created")

    fp = features_path()
    if fp.exists():
        df = pd.read_parquet(fp)
        click.echo(f"  features: {len(df)} rows x {len(df.columns)} cols")
    else:
        click.echo(f"  features: not yet created")


# ---------------------------------------------------------------------------
# ML commands
# ---------------------------------------------------------------------------


@main.group()
def ml():
    """Machine learning: train, evaluate, predict."""
    pass


@ml.command()
@click.option("--model", "model_types", multiple=True, default=MODEL_TYPES,
              help="Model type(s) to train")
@click.option("--no-validate", is_flag=True, help="Skip walk-forward CV")
def train(model_types: tuple, no_validate: bool):
    """Train universal models on all seasons."""
    from ml.training import train_universal

    results = train_universal(
        model_types=list(model_types),
        validate=not no_validate,
    )
    for key, val in results.items():
        if key.endswith("_path"):
            click.echo(f"  {key}: {val}")


@ml.command()
@click.option("--season", default=None, help="Season for rich model (default: latest)")
@click.option("--model", "model_types", multiple=True, default=MODEL_TYPES,
              help="Model type(s) to train")
def train_rich(season: str | None, model_types: tuple):
    """Train rich models using all features for one season."""
    from ml.training import train_rich as _train_rich

    results = _train_rich(season=season, model_types=list(model_types))
    for key, val in results.items():
        if key.endswith("_path"):
            click.echo(f"  {key}: {val}")


@ml.command()
@click.option("--variant", default="universal", help="Model variant")
@click.option("--model", default="xgboost", help="Model type")
def evaluate(variant: str, model: str):
    """Evaluate a trained model on the last season."""
    import pandas as pd
    from ml.data import DataLoader
    from ml.evaluation import compute_metrics, print_report
    from ml.persistence import load_model
    from ml.training import _strip_meta
    from storage.paths import features_path

    wrapper, metadata = load_model(variant, model)
    fp = str(features_path())
    loader = DataLoader(fp)

    if variant == "universal":
        X, y, _ = loader.get_universal_dataset()
    else:
        X, y, _ = loader.get_rich_dataset()

    last_season = sorted(X["_season"].unique())[-1]
    mask = X["_season"] == last_season
    X_test = _strip_meta(X[mask])
    y_test = y[mask]

    # Align features
    for col in metadata["feature_names"]:
        if col not in X_test.columns:
            X_test[col] = float("nan")
    X_test = X_test[metadata["feature_names"]]

    y_proba = wrapper.predict_proba(X_test)
    metrics = compute_metrics(y_test, y_proba)
    click.echo(print_report(metrics))


@ml.command()
@click.option("--variant", default="universal", help="Model variant")
@click.option("--model", default="xgboost", help="Model type")
@click.option("--top-n", default=20, help="Number of top features to show")
def importance(variant: str, model: str, top_n: int):
    """Show top feature importances for a trained model."""
    from ml.persistence import load_model

    _, metadata = load_model(variant, model)
    imp = metadata.get("feature_importance", {})
    if not imp:
        click.echo("No feature importance data available.")
        return

    sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:top_n]
    click.echo(f"\nTop {top_n} features ({variant}/{model}):")
    click.echo("-" * 50)
    for feat, score in sorted_imp:
        click.echo(f"  {feat:<40} {score:.4f}")


@ml.command()
def backtest():
    """Run full walk-forward backtest and display results."""
    from ml.training import train_universal

    click.echo("Running walk-forward backtest (universal features)...")
    results = train_universal(model_types=list(MODEL_TYPES), validate=True)

    for mt in MODEL_TYPES:
        cv_key = f"{mt}_cv"
        if cv_key in results:
            cv = results[cv_key]
            click.echo(f"\n=== {mt.upper()} Walk-Forward CV ===")
            click.echo(cv.to_string(index=False))
            click.echo(f"\nAverages:")
            avg = cv[["accuracy", "log_loss", "brier_score"]].mean()
            click.echo(f"  Accuracy:    {avg['accuracy']:.4f}")
            click.echo(f"  Log Loss:    {avg['log_loss']:.4f}")
            click.echo(f"  Brier Score: {avg['brier_score']:.4f}")


@ml.command()
@click.option("--trials", default=None, type=int, help="Optuna trials per model (default: from config)")
@click.option("--top-k", default=None, type=int, help="Keep top K features (default: all non-zero)")
@click.option("--corr-threshold", default=None, type=float, help="Correlation pruning threshold (default: from config)")
def optimize(trials: int | None, top_k: int | None, corr_threshold: float | None):
    """Full optimized pipeline: feature selection + tuning + calibration + ensemble."""
    from ml.training import train_optimized

    tuning_cfg = TuningConfig()
    display_trials = trials if trials is not None else tuning_cfg.n_trials
    click.echo(f"Running full optimization ({display_trials} trials per model)...")
    results = train_optimized(
        n_tune_trials=trials,
        top_k_features=top_k,
        corr_threshold=corr_threshold,
    )

    click.echo(f"\n{'=' * 60}")
    click.echo("OPTIMIZATION RESULTS")
    click.echo(f"{'=' * 60}")
    click.echo(f"Features: {results.get('n_features_original', '?')} -> {results.get('n_features_selected', '?')}")

    for mt in MODEL_TYPES:
        ts = results.get(f"{mt}_tune_score")
        if ts:
            click.echo(f"{mt} tuned best CV log_loss: {ts:.4f}")
        cv_key = f"{mt}_cv"
        if cv_key in results:
            avg = results[cv_key][["accuracy", "log_loss", "brier_score"]].mean()
            click.echo(f"{mt} tuned CV avg: acc={avg['accuracy']:.4f}  ll={avg['log_loss']:.4f}  brier={avg['brier_score']:.4f}")

    if "ensemble_cv" in results:
        ens = results["ensemble_cv"]
        avg = ens[["ensemble_accuracy", "ensemble_log_loss", "ensemble_brier"]].mean()
        click.echo(f"\nEnsemble CV avg: acc={avg['ensemble_accuracy']:.4f}  ll={avg['ensemble_log_loss']:.4f}  brier={avg['ensemble_brier']:.4f}")

        # Show per-fold comparison
        click.echo(f"\nPer-fold comparison:")
        display_cols = ["fold", "test", "ensemble_accuracy", "ensemble_log_loss"]
        for mt in MODEL_TYPES:
            display_cols.extend([f"{mt}_accuracy", f"{mt}_log_loss"])
        click.echo(ens[display_cols].to_string(index=False))

    for key, val in results.items():
        if key.endswith("_path"):
            click.echo(f"  {key}: {val}")


@ml.command()
@click.option("--full", is_flag=True, help="Force full Optuna retrain (2-3 hours)")
@click.option("--quick", is_flag=True, help="Force quick retrain (~15-20 min)")
@click.option("--check", is_flag=True, help="Just check matchweek status")
@click.option("--force", is_flag=True, help="Force retrain even if matchweek not complete")
@click.option("--dry-run", is_flag=True, help="Compare only, don't promote")
def retrain(full: bool, quick: bool, check: bool, force: bool, dry_run: bool):
    """Retrain models — auto-triggered when a matchweek completes."""
    from scripts.pipeline.weekly_retrain import (
        auto_retrain,
        forced_retrain,
        get_matchweek_status,
    )

    if check:
        from scripts.pipeline.weekly_retrain import MATCHES_PER_MATCHWEEK
        status = get_matchweek_status()
        if "error" in status:
            click.echo(f"Error: {status['error']}")
            return
        click.echo(f"Completed matchweeks: {status['completed_matchweeks']} ({status['total_matches']} matches)")
        if status["matches_in_progress"]:
            click.echo(f"MW {status['completed_matchweeks'] + 1} in progress: "
                        f"{status['matches_in_progress']}/{MATCHES_PER_MATCHWEEK} played")
        click.echo(f"Last retrained: MW {status['last_retrained_matchweek']}")
        click.echo(f"Needs retrain: {'YES' if status['needs_retrain'] else 'NO'}")
        return

    # --full / --quick go through forced_retrain, never the ensemble-only trainers
    # directly. The auxiliary models (catboost_no_odds, xG, draw detector, the O/U
    # classifiers behind the enabled bets) and the prediction refresh run in the
    # caller, and the retrain gate must be stamped.
    if full:
        click.echo("Starting FULL retrain (Optuna + feature selection, ~2-3 hours)...")
        result = forced_retrain("full", dry_run=dry_run)
    elif quick:
        click.echo("Starting QUICK retrain (reuse hyperparams, ~15-20 min)...")
        result = forced_retrain("quick", dry_run=dry_run)
    else:
        click.echo("Checking matchweek status...")
        result = auto_retrain(dry_run=dry_run, force=force)
        if result.get("action") == "skipped":
            click.echo(f"Skipped: {result.get('reason')}")
            return

    status = "PROMOTED" if result.get("promoted") else "NOT PROMOTED"
    click.echo(f"\nResult: {status}")
    if result.get("matchweek"):
        click.echo(f"Matchweek: {result['matchweek']}")
    click.echo(f"Reason: {result.get('comparison', result.get('error', '?'))}")
    if result.get("new_metrics"):
        m = result["new_metrics"]
        click.echo(f"New LL: {m.get('ensemble_log_loss', 0):.4f}")
    ou = (result.get("auxiliary_models") or {}).get("over_under") or {}
    if ou:
        ou_lines = ", ".join(f"{k} {v}" for k, v in ou.get("lines", {}).items())
        click.echo(f"O/U classifiers: {ou_lines or ou.get('error', '?')}")
    if result.get("predictions_refreshed_for"):
        click.echo(f"Predictions refreshed for: {', '.join(result['predictions_refreshed_for'])}")


@ml.command()
@click.option("--version", default=None, help="Archive version to restore (default: latest)")
def rollback(version: str | None):
    """Rollback models to a previous archived version."""
    from scripts.pipeline.weekly_retrain import rollback as do_rollback

    success = do_rollback(version)
    if success:
        click.echo("Rollback complete.")
    else:
        click.echo("Rollback failed.")


@ml.command()
def retrain_history():
    """Show retrain history (promotions, metrics, comparisons)."""
    from scripts.pipeline.weekly_retrain import METRICS_HISTORY

    if not METRICS_HISTORY.exists():
        click.echo("No retrain history yet. Run `seriea ml retrain` first.")
        return

    import json
    click.echo(f"{'Date':<22} {'Mode':<8} {'Promoted':<10} {'LL Old':<10} {'LL New':<10} {'Reason'}")
    click.echo("-" * 85)
    with open(METRICS_HISTORY) as f:
        for line in f:
            entry = json.loads(line)
            ts = entry.get("timestamp", "?")[:19]
            mode = entry.get("mode", "?")
            promoted = "YES" if entry.get("promoted") else "NO"
            old_m = entry.get("current_metrics") or {}
            new_m = entry.get("new_metrics") or {}
            old = f"{old_m.get('ensemble_log_loss', 0):.4f}" if old_m else "N/A"
            new = f"{new_m.get('ensemble_log_loss', 0):.4f}" if new_m else "N/A"
            reason = entry.get("comparison", entry.get("error", "?"))[:30]
            click.echo(f"{ts:<22} {mode:<8} {promoted:<10} {old:<10} {new:<10} {reason}")


@ml.command()
def ablation():
    """Compare models with and without betting odds features."""
    from ml.feature_selection import exclude_odds
    from ml.training import train_universal
    from ml.data import DataLoader
    from storage.paths import features_path

    fp = str(features_path())
    loader = DataLoader(fp)
    X, y, feature_names = loader.get_universal_dataset()

    # With odds (baseline)
    click.echo("=== WITH betting odds ===")
    results_odds = train_universal(
        model_types=["xgboost"], validate=True, use_sample_weights=True,
    )

    # Without odds
    _, no_odds_feats = exclude_odds(X, feature_names)
    click.echo(f"\n=== WITHOUT betting odds ({len(no_odds_feats)} features) ===")
    results_no_odds = train_universal(
        model_types=["xgboost"], validate=True, use_sample_weights=True,
        feature_names_override=no_odds_feats,
    )

    click.echo(f"\n{'=' * 60}")
    click.echo("ABLATION COMPARISON")
    click.echo(f"{'=' * 60}")

    for label, res in [("With odds", results_odds), ("Without odds", results_no_odds)]:
        cv = res.get("xgboost_cv")
        if cv is not None:
            avg = cv[["accuracy", "log_loss", "brier_score", "f1_D"]].mean()
            click.echo(f"{label}: acc={avg['accuracy']:.4f}  ll={avg['log_loss']:.4f}  brier={avg['brier_score']:.4f}  f1_D={avg['f1_D']:.4f}")


if __name__ == "__main__":
    # Allow running as a script from the seriea_pipeline directory
    main()
