"""Track 1 — Shadow predictor runner for upcoming Serie A fixtures.

Runs the full simulator (Phase 0 λ + Phase 2 corners/cards/shots + Phase 5
player profiles) on every upcoming fixture from data/upcoming/matches.json,
writes predictions BEFORE kickoff to data/upcoming/simulator_shadow_log.json.

Output schema (one entry per fixture):
{
  "run_id": "YYYY-MM-DD_HHMMSS",
  "generated_at": ISO-UTC,
  "fixtures": [
    {
      "match_key": "YYYY-MM-DD_Home_Away",
      "home_team": str,
      "away_team": str,
      "match_date": str,
      "kickoff_utc": str,
      "lambda_home": float, "lambda_away": float, "tau": float,
      "goal_markets": {
        "1X2": {"H": p, "D": p, "A": p},
        "over_0_5": p, "over_1_5": p, "over_2_5": p, "over_3_5": p, "over_4_5": p,
        "btts": p,
        "home_clean_sheet": p, "away_clean_sheet": p,
        ...
      },
      "corner_markets": {"corners_over_8_5": p, "corners_over_9_5": p, ...},
      "card_markets": {"cards_over_3_5": p, "cards_over_4_5": p, ...},
      "shot_markets": {...},
      "top_scorers": [{"player_id": int, "player_name": str, "prob": p}, ...],
      "expected": {
        "home_goals": float, "away_goals": float,
        "corners_total": float, "cards_total": float, "shots_total": float,
      }
    }
  ]
}

Every prediction is deterministic per (match_key, seed) so re-running produces
the same output — critical for shadow-mode integrity.

Usage:
    python3 scripts/prediction/simulate_upcoming.py
    python3 scripts/prediction/simulate_upcoming.py --matches data/upcoming/matches.json \
                                                     --output data/upcoming/simulator_shadow_log.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.simulator.base_rates.card_rates import CardRateEstimator
from models.simulator.base_rates.corner_rates import CornerRateEstimator
from models.simulator.base_rates.lineup_allocator import allocate_team_shots_to_players
from models.simulator.base_rates.player_profiles import (
    PlayerProfileStore,
    load_player_match_stats,
)
from models.simulator.base_rates.shot_generator import ShotRateEstimator
from models.simulator.engine.dixon_coles import fit_tau_mle
from models.simulator.engine.simulator import simulate_match
from models.simulator.markets import (
    all_phase2_market_probs,
    all_player_market_probs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "features_serie_a.parquet"
UPCOMING_MATCHES_PATH = PROJECT_ROOT / "data" / "upcoming" / "matches.json"
SHADOW_LOG_PATH = PROJECT_ROOT / "data" / "upcoming" / "simulator_shadow_log.json"

N_TRIALS = 10_000
SEED = 42

# λ estimator features (Phase 0b winner: Arm K — G + missing_players)
LAMBDA_FEATURES = [
    "home_attack_strength", "away_attack_strength",
    "home_defense_strength", "away_defense_strength",
    "league_avg_goals",
    "home_ss_roll_xg", "away_ss_roll_xg",
    "home_ss_roll_xgot", "away_ss_roll_xgot",
    "home_ss_roll_total_shots", "away_ss_roll_total_shots",
    "home_us_team_xg", "away_us_team_xg",
    "home_us_team_npxg", "away_us_team_npxg",
    "home_missing_count", "away_missing_count",
    "home_missing_injury_count", "away_missing_injury_count",
    "home_missing_suspended_count", "away_missing_suspended_count",
]


def _to_match_key(date_str: str, home: str, away: str) -> str:
    """YYYY-MM-DD_Home_Away — matches canonical match_id convention."""
    return f"{date_str}_{home}_{away}".replace(" ", "")


def _lineup_fallback(team: str, profiles: PlayerProfileStore, raw_stats: pd.DataFrame,
                     before_date: pd.Timestamp, max_lookback_days: int = 35) -> list[int]:
    """Top-11 by recent minutes (fallback when we don't have confirmed XI)."""
    if "date" not in raw_stats.columns:
        return []
    df = raw_stats.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    recent = df[(df["team"] == team)
                & (df["date"] < before_date)
                & (df["is_starter"] == True)]  # noqa: E712
    if len(recent) == 0:
        return []
    window = recent[recent["date"] >= before_date - pd.Timedelta(days=max_lookback_days)]
    if len(window) == 0:
        window = recent
    by_player = window.groupby("player_id", observed=True)["minutes"].sum()
    top = by_player.sort_values(ascending=False).head(11)
    return top.index.tolist()


def _fit_all_models(features_df: pd.DataFrame, raw_stats: pd.DataFrame) -> dict:
    """Fit λ regressors + Phase 2 rate estimators + player profiles."""
    log.info("Fitting λ estimator on Serie A pre-current-season data")
    train = features_df.dropna(subset=["home_score", "away_score"]).copy()
    available = [f for f in LAMBDA_FEATURES if f in train.columns]
    log.info("  Using %d/%d λ features", len(available), len(LAMBDA_FEATURES))
    train_lambda = train.dropna(subset=available + ["home_score", "away_score"])
    if len(train_lambda) < 100:
        raise SystemExit(f"Insufficient λ training rows ({len(train_lambda)})")

    from sklearn.linear_model import PoissonRegressor
    X = train_lambda[available].to_numpy(dtype=float)
    m_home = PoissonRegressor(alpha=0.5, max_iter=1000).fit(X, train_lambda["home_score"])
    m_away = PoissonRegressor(alpha=0.5, max_iter=1000).fit(X, train_lambda["away_score"])
    lh_pred = np.clip(m_home.predict(X), 0.1, 6.0)
    la_pred = np.clip(m_away.predict(X), 0.1, 6.0)
    tau = fit_tau_mle(
        train_lambda["home_score"].to_numpy(dtype=int),
        train_lambda["away_score"].to_numpy(dtype=int),
        lh_pred, la_pred,
    )
    log.info("  Fit τ=%.4f", tau)

    corner_est = CornerRateEstimator()
    corner_est.fit(train)

    card_est = CardRateEstimator()
    card_est.fit(train)

    shot_est = ShotRateEstimator()
    shot_est.fit(train)

    profiles = PlayerProfileStore(min_starts=3, rolling_window=10)
    profiles.fit(raw_stats)

    return {
        "lambda_home": m_home,
        "lambda_away": m_away,
        "lambda_features": available,
        "tau": tau,
        "corner_est": corner_est,
        "card_est": card_est,
        "shot_est": shot_est,
        "profiles": profiles,
    }


def _predict_lambdas(models: dict, feature_row: pd.DataFrame) -> tuple[float, float]:
    feats = models["lambda_features"]
    X = feature_row[feats].fillna(0.0).to_numpy(dtype=float)
    lh = float(np.clip(models["lambda_home"].predict(X)[0], 0.1, 6.0))
    la = float(np.clip(models["lambda_away"].predict(X)[0], 0.1, 6.0))
    return lh, la


def _find_feature_row(features_df: pd.DataFrame, home: str, away: str,
                       match_date: pd.Timestamp) -> pd.DataFrame | None:
    """Most recent row for this matchup. Falls back to synthetic if none exists."""
    mask = ((features_df["home_team"] == home)
            & (features_df["away_team"] == away)
            & (features_df["match_date"] == match_date))
    matched = features_df[mask]
    if len(matched) > 0:
        return matched.iloc[[0]]
    # Try same home team — most recent row (will use rolling features for this team)
    same_home = features_df[features_df["home_team"] == home].sort_values("match_date")
    same_home = same_home[same_home["match_date"] < match_date]
    if len(same_home) > 0:
        return same_home.iloc[[-1]]
    return None


def run(matches_path: Path, output_path: Path, only_future: bool = True) -> dict:
    log.info("Loading upcoming fixtures from %s", matches_path)
    with open(matches_path) as f:
        upcoming = json.load(f)
    fixtures = upcoming.get("matches", [])
    log.info("  %d fixtures in source file", len(fixtures))

    if only_future:
        today = datetime.now(timezone.utc).date()
        kept = []
        for fx in fixtures:
            date_str = fx.get("date") or fx.get("match_date", "")
            try:
                fx_date = pd.to_datetime(date_str).date()
                if fx_date >= today:
                    kept.append(fx)
            except Exception:
                kept.append(fx)  # keep if date unparseable
        log.info("  %d fixtures on or after %s (filtered)", len(kept), today)
        fixtures = kept

    log.info("Loading features + player stats")
    features_df = pd.read_parquet(FEATURES_PATH)
    features_df = features_df[features_df["league"] == "serie_a"].copy()
    features_df["match_date"] = pd.to_datetime(features_df["match_date"], errors="coerce")
    log.info("  features: %d rows × %d cols", *features_df.shape)

    raw_stats = load_player_match_stats()
    if raw_stats is None:
        log.warning("player_match_stats.parquet missing — no player-prop predictions")
        raw_stats = pd.DataFrame()

    models = _fit_all_models(features_df, raw_stats)

    pid_to_name = {}
    if len(raw_stats) > 0:
        pid_to_name = dict(zip(raw_stats["player_id"], raw_stats["player_name"]))

    results = []
    for fx in fixtures:
        home = fx["home_team"]
        away = fx["away_team"]
        date_str = fx.get("date") or fx.get("match_date", "")
        kickoff = fx.get("commence_time", "")
        match_date = pd.to_datetime(date_str, errors="coerce")

        feature_row = _find_feature_row(features_df, home, away, match_date)
        if feature_row is None:
            log.warning("No feature row for %s vs %s (%s) — skipping", home, away, date_str)
            continue

        # Override with upcoming match's home/away teams (feature row may be from an
        # older match where the team was different-side).
        row = feature_row.iloc[0].copy()
        row["home_team"] = home
        row["away_team"] = away

        # λ
        lh, la = _predict_lambdas(models, feature_row)

        # Phase 2 rates
        corner_est = models["corner_est"]
        card_est = models["card_est"]
        shot_est = models["shot_est"]
        rate_ch, rate_ca = corner_est.predict(feature_row) if corner_est.is_fit else (None, None)
        rate_kh, rate_ka = card_est.predict(feature_row) if card_est.is_fit else (None, None)
        rate_sh, rate_sa = shot_est.predict(feature_row) if shot_est.is_fit else (None, None)
        sot_h, sot_a = shot_est.sot_ratios() if shot_est.is_fit else (0.35, 0.33)

        # Phase 5 player shares
        profiles = models["profiles"]
        lineup_home = _lineup_fallback(home, profiles, raw_stats, match_date) if len(raw_stats) else []
        lineup_away = _lineup_fallback(away, profiles, raw_stats, match_date) if len(raw_stats) else []
        profiles_dict = profiles.all_profiles() if profiles.n_profiles else {}
        if rate_sh is not None and len(lineup_home) >= 8:
            shares_h = allocate_team_shots_to_players(float(rate_sh[0]), lineup_home, profiles)
        else:
            shares_h = {}
        if rate_sa is not None and len(lineup_away) >= 8:
            shares_a = allocate_team_shots_to_players(float(rate_sa[0]), lineup_away, profiles)
        else:
            shares_a = {}

        match_key = _to_match_key(date_str, home, away)

        sim = simulate_match(
            lambda_home=lh, lambda_away=la,
            tau=models["tau"], n_trials=N_TRIALS, seed=SEED,
            match_id=match_key,
            corner_rate_home=float(rate_ch[0]) if rate_ch is not None else None,
            corner_rate_away=float(rate_ca[0]) if rate_ca is not None else None,
            card_rate_home=float(rate_kh[0]) if rate_kh is not None else None,
            card_rate_away=float(rate_ka[0]) if rate_ka is not None else None,
            shot_rate_home=float(rate_sh[0]) if rate_sh is not None else None,
            shot_rate_away=float(rate_sa[0]) if rate_sa is not None else None,
            sot_ratio_home=sot_h, sot_ratio_away=sot_a,
            player_profiles_home=profiles_dict if shares_h else None,
            player_profiles_away=profiles_dict if shares_a else None,
            player_shot_shares_home=shares_h if shares_h else None,
            player_shot_shares_away=shares_a if shares_a else None,
        )

        # Package all markets
        goal_markets = {
            "1X2": {
                "H": round(sim.p_home_win(), 4),
                "D": round(sim.p_draw(), 4),
                "A": round(sim.p_away_win(), 4),
            },
            "double_chance_1X": round(sim.p_double_chance("1X"), 4),
            "double_chance_X2": round(sim.p_double_chance("X2"), 4),
            "double_chance_12": round(sim.p_double_chance("12"), 4),
            "draw_no_bet_home": round(sim.p_draw_no_bet("home"), 4),
            "draw_no_bet_away": round(sim.p_draw_no_bet("away"), 4),
            "over_0_5": round(sim.p_over(0.5), 4),
            "over_1_5": round(sim.p_over(1.5), 4),
            "over_2_5": round(sim.p_over(2.5), 4),
            "over_3_5": round(sim.p_over(3.5), 4),
            "over_4_5": round(sim.p_over(4.5), 4),
            "btts": round(sim.p_btts(yes=True), 4),
            "home_clean_sheet": round(sim.p_clean_sheet("home"), 4),
            "away_clean_sheet": round(sim.p_clean_sheet("away"), 4),
            # Asian handicaps
            "AH_home_-1.5": round(sim.p_handicap(-1.5, "home"), 4),
            "AH_home_-0.5": round(sim.p_handicap(-0.5, "home"), 4),
            "AH_home_+0.5": round(sim.p_handicap(+0.5, "home"), 4),
            "AH_home_+1.5": round(sim.p_handicap(+1.5, "home"), 4),
        }
        top_exact = [{"score": f"{h}-{a}", "prob": round(p, 4)}
                     for (h, a), p in sim.top_scores(k=12)]

        phase2_markets = all_phase2_market_probs(sim)
        phase2_markets = {k: round(v, 4) for k, v in phase2_markets.items()}

        top_scorers = []
        if sim.player_goals:
            for pid, p in sim.top_scorers(k=8):
                top_scorers.append({
                    "player_id": int(pid),
                    "player_name": pid_to_name.get(pid, f"player_{pid}"),
                    "anytime_scorer_prob": round(p, 4),
                })

        expected = {
            "home_goals": round(lh, 3),
            "away_goals": round(la, 3),
            "corners_total": round(sim.expected_corners("both"), 2) if sim.home_corners is not None else None,
            "corners_home": round(sim.expected_corners("home"), 2) if sim.home_corners is not None else None,
            "corners_away": round(sim.expected_corners("away"), 2) if sim.home_corners is not None else None,
            "cards_total": round(sim.expected_cards("both"), 2) if sim.home_cards is not None else None,
            "shots_total": round(float((sim.home_shots + sim.away_shots).mean()), 2) if sim.home_shots is not None else None,
        }

        results.append({
            "match_key": match_key,
            "home_team": home,
            "away_team": away,
            "match_date": date_str,
            "kickoff_utc": kickoff,
            "lambda_home": round(lh, 4),
            "lambda_away": round(la, 4),
            "tau": round(models["tau"], 4),
            "n_trials": N_TRIALS,
            "seed": SEED,
            "goal_markets": goal_markets,
            "top_exact_scores": top_exact,
            "corner_card_shot_markets": phase2_markets,
            "top_scorers": top_scorers,
            "expected": expected,
        })

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_fixtures": len(results),
        "seed": SEED,
        "n_trials": N_TRIALS,
        "tau_fitted": round(models["tau"], 4),
        "fixtures": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str))
    log.info("Wrote %d fixture predictions to %s", len(results), output_path)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matches", default=str(UPCOMING_MATCHES_PATH))
    ap.add_argument("--output", default=str(SHADOW_LOG_PATH))
    args = ap.parse_args()

    payload = run(Path(args.matches), Path(args.output))
    print(f"\n=== Shadow predictions for {payload['n_fixtures']} fixtures ===\n")
    for fx in payload["fixtures"]:
        gm = fx["goal_markets"]
        exp = fx["expected"]
        print(f"{fx['home_team']:12s} vs {fx['away_team']:12s}  ({fx['match_date']})")
        print(f"  1X2: H={gm['1X2']['H']:.1%}  D={gm['1X2']['D']:.1%}  A={gm['1X2']['A']:.1%}")
        print(f"  O/U 2.5: over={gm['over_2_5']:.1%}   BTTS: {gm['btts']:.1%}")
        print(f"  Expected goals: {exp['home_goals']:.2f} - {exp['away_goals']:.2f}   corners: {exp['corners_total']}   cards: {exp['cards_total']}")
        if fx["top_scorers"]:
            top3 = fx["top_scorers"][:3]
            names = "  ".join(f"{p['player_name']}({p['anytime_scorer_prob']:.0%})" for p in top3)
            print(f"  Top scorers: {names}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
