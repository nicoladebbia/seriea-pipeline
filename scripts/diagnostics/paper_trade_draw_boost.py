#!/usr/bin/env python3
"""Paper-trade harness for the H4 draw-boost fix.

Compares predictions WITH and WITHOUT the post-calibration draw boost on
the latest 150 settled matches. Uses stored opening (Max) and closing
(Pinnacle close, with fallback) odds from matches.parquet.

Output: side-by-side ROI, hit rate, draw participation, per-bet log.

Methodology:
  - Use the new walkforward_core to score each match (one-fold-back model).
  - For each match, compute model probabilities both with boost=0.0 and
    boost=0.30.
  - Apply a fixed betting rule: bet on the highest-edge outcome if edge ≥ 5%.
  - Stake: fractional Kelly (0.25× full Kelly) capped at 2% bankroll.
  - Place at OPENING odds (best across books = `odds_MaxH/D/A`).
  - Settle at CLOSING odds (Pinnacle close if available, else Avg open as proxy).

This is a REPRODUCIBLE measurement, not a recommendation. The result tells
us whether the boost would have made/lost money on the most recent 150
matches if we had been betting them at retail prices.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.walkforward_core import (  # noqa: E402
    WalkForwardConfig,
    run_walkforward,
    apply_calibrator,
)
from ml.feature_selection import exclude_odds  # noqa: E402
from scripts.models.train_walkforward import LEAKY_COLUMNS, META_COLUMNS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

EDGE_THRESHOLD = 0.05  # bet only when model edge over implied prob ≥ 5%
KELLY_FRACTION = 0.25  # 0.25 × full Kelly
MAX_STAKE_FRACTION = 0.02  # cap at 2% of bankroll per bet
INITIAL_BANKROLL = 1000.0
N_MATCHES = 150


def build_1x2_target(df: pd.DataFrame) -> pd.Series:
    res = []
    for h, a in zip(df["home_score"], df["away_score"]):
        if h > a:
            res.append("H")
        elif h < a:
            res.append("A")
        else:
            res.append("D")
    return pd.Series(res, index=df.index)


def select_features(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [
        c for c in numeric
        if c not in META_COLUMNS and c not in LEAKY_COLUMNS and not c.startswith("_")
    ]
    feats = [f for f in feats if not df[feats].isna().all(axis=0).get(f, False)]
    _, kept = exclude_odds(df[feats].copy(), feats)
    return kept


def kelly_stake(prob: float, decimal_odds: float, bankroll: float) -> float:
    """Fractional Kelly stake with safety cap."""
    if decimal_odds <= 1.0 or prob <= 0:
        return 0.0
    b = decimal_odds - 1.0  # net odds
    q = 1.0 - prob
    full_kelly = (b * prob - q) / b
    if full_kelly <= 0:
        return 0.0
    stake_fraction = min(full_kelly * KELLY_FRACTION, MAX_STAKE_FRACTION)
    return stake_fraction * bankroll


def implied_prob(odds: float) -> float:
    """Decimal odds → implied probability (with vig)."""
    return 1.0 / odds if odds > 1.0 else 0.0


def select_bet(model_probs: dict, opening_odds: dict) -> Optional[tuple[str, float, float]]:
    """Pick the best edge from H/D/A.

    Returns (outcome, model_prob, opening_decimal_odds) or None if no bet.
    """
    best = None
    for outcome in ["H", "D", "A"]:
        odds = opening_odds.get(outcome, 0.0)
        if odds <= 1.01:
            continue
        market_prob = implied_prob(odds)
        edge = model_probs[outcome] - market_prob
        if edge < EDGE_THRESHOLD:
            continue
        score = edge  # rank by raw edge
        if best is None or score > best[3]:
            best = (outcome, model_probs[outcome], odds, edge)
    if best is None:
        return None
    return best[0], best[1], best[2]


def settle_bet(outcome_pick: str, actual_result: str, stake: float,
               opening_odds: float, closing_odds: float) -> dict:
    """Compute P/L at both opening (we bet at this) and closing (sharp benchmark).

    P/L for actual money is at OPENING price (what we'd have paid).
    Closing price is informational — tells us if the bet was sharp.
    """
    won = (outcome_pick == actual_result)
    pnl_opening = stake * (opening_odds - 1.0) if won else -stake
    return {
        "won": won,
        "pnl_opening": pnl_opening,
        "stake": stake,
        "opening_odds": opening_odds,
        "closing_odds": closing_odds,
    }


def main() -> int:
    log.info("Loading data...")
    matches = pd.read_parquet(PROJECT_ROOT / "data" / "parsed" / "matches.parquet")
    matches["match_date"] = pd.to_datetime(matches["match_date"])
    settled = matches.dropna(subset=["home_score", "away_score"]).copy()

    # Latest N matches (Serie A only — same as model is trained for)
    latest = settled[
        (settled["league"] == "serie_a")
        & settled["odds_MaxH"].notna()
    ].sort_values("match_date", ascending=False).head(N_MATCHES).copy()
    latest = latest.sort_values("match_date").reset_index(drop=True)
    log.info("Selected %d latest SA matches (%s → %s)",
             len(latest), latest["match_date"].min().date(), latest["match_date"].max().date())

    # Build features for these matches by loading the full feature parquet
    # and filtering. The 150 matches are the most recent — they live in the
    # 2024-2025 / 2025-2026 seasons, which the model HAS seen during training.
    # For honest paper-trade we need to use a model that was trained STRICTLY
    # BEFORE these matches — so we use the existing walkforward artifact for
    # each match's season.
    log.info("Loading features...")
    feats = pd.read_parquet(PROJECT_ROOT / "data" / "features" / "features_serie_a.parquet")
    feats["match_date"] = pd.to_datetime(feats["match_date"])
    target_match_keys = set(zip(latest["home_team"], latest["away_team"], latest["match_date"]))
    feat_match = feats[
        feats.apply(lambda r: (r["home_team"], r["away_team"], r["match_date"]) in target_match_keys, axis=1)
    ].copy()
    log.info("Found %d/%d matches with features", len(feat_match), len(latest))

    # Honest scoring: use walkforward_core with the eval seasons covering the
    # 150 matches. Model trains strictly before each eval season — fair test.
    eval_seasons = sorted(feat_match["season"].unique().tolist())
    log.info("Eval seasons covering the 150 matches: %s", eval_seasons)

    config_no_boost = WalkForwardConfig(
        eval_seasons=eval_seasons,
        seeds=[42],
        min_prior_seasons=5,
        fit_calibrator=True,
        kind="multiclass",
        n_classes=3,
        cal_val_fraction=0.15,
        draw_boost=0.0,
    )
    config_with_boost = WalkForwardConfig(
        eval_seasons=eval_seasons,
        seeds=[42],
        min_prior_seasons=5,
        fit_calibrator=True,
        kind="multiclass",
        n_classes=3,
        cal_val_fraction=0.15,
        draw_boost=0.30,
    )

    df_for_walkforward = feats.dropna(subset=["home_score", "away_score"]).copy()

    log.info("Running walkforward WITHOUT draw boost (baseline)...")
    report_no = run_walkforward(
        df=df_for_walkforward, target_builder=build_1x2_target,
        feature_selector=select_features, config=config_no_boost,
    )
    log.info("Running walkforward WITH draw_boost=0.30...")
    report_with = run_walkforward(
        df=df_for_walkforward, target_builder=build_1x2_target,
        feature_selector=select_features, config=config_with_boost,
    )

    # The walkforward_core doesn't expose per-match probabilities, only per-fold
    # aggregates. To paper-trade per-match we need the per-match probas.
    # Approach: re-run inference using the trained models, applying the boost
    # as appropriate.
    # We'll instead use the per-fold MODEL ARTIFACTS the run produced, run
    # inference per match, and apply boost variations.
    log.info("Generating per-match probabilities for paper-trade...")

    # Map eval_season → fold result
    fold_by_season_no: dict[str, object] = {}
    fold_by_season_with: dict[str, object] = {}
    for f in report_no.folds:
        fold_by_season_no[f.eval_season] = f
    for f in report_with.folds:
        fold_by_season_with[f.eval_season] = f

    feature_names = select_features(df_for_walkforward)
    bets_no_boost = []
    bets_with_boost = []
    bankroll_no = bankroll_with = INITIAL_BANKROLL

    for _, row in latest.iterrows():
        season = row["season"]
        if season not in fold_by_season_no:
            continue
        # Match feature row
        feat_row = feat_match[
            (feat_match["home_team"] == row["home_team"])
            & (feat_match["away_team"] == row["away_team"])
            & (feat_match["match_date"] == row["match_date"])
        ]
        if feat_row.empty:
            continue
        X = feat_row[feature_names].iloc[[0]]

        # Load both fold models from their bytes
        from catboost import CatBoostClassifier
        import tempfile, os, pickle as pk
        fold_no = fold_by_season_no[season]
        fold_with = fold_by_season_with[season]
        model_no = CatBoostClassifier()
        with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as tmp:
            tmp.write(fold_no.model_bytes)
            tmp_path = tmp.name
        try:
            model_no.load_model(tmp_path)
        finally:
            os.unlink(tmp_path)

        # Both fold models trained from same data → same model. Predict once.
        proba_raw = model_no.predict_proba(X)[0]

        # Apply calibrators with and without boost
        cal_dict_no = {
            "kind": fold_no.calibrator["kind"],
            "classes": fold_no.calibrator["classes"],
            "calibrators": pk.loads(fold_no.calibrator["calibrators_pickle"]),
        }
        cal_dict_with = {
            "kind": fold_with.calibrator["kind"],
            "classes": fold_with.calibrator["classes"],
            "calibrators": pk.loads(fold_with.calibrator["calibrators_pickle"]),
        }
        proba_cal_no = apply_calibrator(proba_raw.reshape(1, -1), cal_dict_no, draw_boost=0.0)[0]
        proba_cal_with = apply_calibrator(proba_raw.reshape(1, -1), cal_dict_with, draw_boost=0.30)[0]

        # Map alphabetical [A, D, H] → dict
        classes = cal_dict_no["classes"]
        probs_no = {cls: float(proba_cal_no[i]) for i, cls in enumerate(classes)}
        probs_with = {cls: float(proba_cal_with[i]) for i, cls in enumerate(classes)}

        # Opening odds (Max), Closing odds (PS close, fallback to Avg)
        opening = {
            "H": float(row["odds_MaxH"]) if pd.notna(row["odds_MaxH"]) else 0.0,
            "D": float(row["odds_MaxD"]) if pd.notna(row["odds_MaxD"]) else 0.0,
            "A": float(row["odds_MaxA"]) if pd.notna(row["odds_MaxA"]) else 0.0,
        }
        if pd.notna(row.get("odds_PS_close_H")):
            closing = {
                "H": float(row["odds_PS_close_H"]),
                "D": float(row["odds_PS_close_D"]),
                "A": float(row["odds_PS_close_A"]),
            }
        else:
            closing = {
                "H": float(row["odds_AvgH"]) if pd.notna(row.get("odds_AvgH")) else opening["H"],
                "D": float(row["odds_AvgD"]) if pd.notna(row.get("odds_AvgD")) else opening["D"],
                "A": float(row["odds_AvgA"]) if pd.notna(row.get("odds_AvgA")) else opening["A"],
            }

        # Actual outcome
        h_score, a_score = int(row["home_score"]), int(row["away_score"])
        if h_score > a_score:
            actual = "H"
        elif h_score < a_score:
            actual = "A"
        else:
            actual = "D"

        # Pick + settle for each variant
        for variant_name, probs, bankroll, log_list in [
            ("no_boost", probs_no, bankroll_no, bets_no_boost),
            ("with_boost", probs_with, bankroll_with, bets_with_boost),
        ]:
            pick = select_bet(probs, opening)
            if pick is None:
                continue
            outcome, model_p, op_odds = pick
            stake = kelly_stake(model_p, op_odds, bankroll)
            if stake < 0.01:
                continue
            settled = settle_bet(outcome, actual, stake, op_odds, closing[outcome])
            log_list.append({
                "match_date": row["match_date"].date().isoformat(),
                "match": f"{row['home_team']} vs {row['away_team']}",
                "actual": actual,
                "pick": outcome,
                "model_prob": model_p,
                "stake": settled["stake"],
                "opening_odds": settled["opening_odds"],
                "closing_odds": settled["closing_odds"],
                "won": settled["won"],
                "pnl_opening": settled["pnl_opening"],
            })
            if variant_name == "no_boost":
                bankroll_no += settled["pnl_opening"]
            else:
                bankroll_with += settled["pnl_opening"]

    # Report
    print()
    print("=" * 80)
    print("PAPER TRADE RESULTS — latest 150 SA matches")
    print("=" * 80)
    for variant_name, bets, final_bankroll in [
        ("WITHOUT draw boost (baseline)", bets_no_boost, bankroll_no),
        ("WITH draw_boost=0.30 (H4 fix)", bets_with_boost, bankroll_with),
    ]:
        if not bets:
            print(f"\n{variant_name}: NO BETS PLACED (no edge ≥ {EDGE_THRESHOLD*100:.0f}%)")
            continue
        df_b = pd.DataFrame(bets)
        n_bets = len(df_b)
        wins = int(df_b["won"].sum())
        total_staked = df_b["stake"].sum()
        total_pnl = df_b["pnl_opening"].sum()
        roi = (total_pnl / total_staked) * 100 if total_staked > 0 else 0
        n_draws_picked = int((df_b["pick"] == "D").sum())
        n_draws_actual = int(((df_b["pick"] == "D") & (df_b["actual"] == "D")).sum())
        pct_return = (final_bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100
        print(f"\n{variant_name}")
        print(f"  Bets placed:       {n_bets}")
        print(f"  Wins / losses:     {wins} / {n_bets - wins}  ({wins/n_bets*100:.1f}% hit rate)")
        print(f"  Total staked:      €{total_staked:.2f}")
        print(f"  Total P/L:         €{total_pnl:+.2f}")
        print(f"  Flat ROI:          {roi:+.2f}%")
        print(f"  Bankroll:          €{INITIAL_BANKROLL:.0f} → €{final_bankroll:.2f}  ({pct_return:+.2f}%)")
        print(f"  Draw bets picked:  {n_draws_picked}  (of which {n_draws_actual} won = {n_draws_actual/n_draws_picked*100 if n_draws_picked else 0:.1f}%)")
        # Per-pick breakdown
        for pick in ["H", "D", "A"]:
            sub = df_b[df_b["pick"] == pick]
            if len(sub) == 0:
                continue
            sub_wins = int(sub["won"].sum())
            sub_pnl = sub["pnl_opening"].sum()
            sub_stake = sub["stake"].sum()
            sub_roi = (sub_pnl / sub_stake) * 100 if sub_stake > 0 else 0
            print(f"    {pick} picks: {len(sub)} bets, {sub_wins} wins ({sub_wins/len(sub)*100:.0f}%), ROI {sub_roi:+.1f}%")

    print()
    print("=" * 80)
    print("Saving per-bet logs to data/diagnostics/")
    print("=" * 80)
    out_dir = PROJECT_ROOT / "data" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    if bets_no_boost:
        pd.DataFrame(bets_no_boost).to_csv(out_dir / "paper_trade_no_boost.csv", index=False)
    if bets_with_boost:
        pd.DataFrame(bets_with_boost).to_csv(out_dir / "paper_trade_with_boost.csv", index=False)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
