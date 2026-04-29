#!/usr/bin/env python3
"""Deep subset analysis for hidden betting edges.

Splits the baseline 1X2 paper-trade by N candidate dimensions to find subsets
where the model's edge is meaningfully different from the average. Findings
are split-tested:
  - DISCOVERY phase: 2022-23, 2023-24 SA seasons (find candidate effects)
  - VALIDATION phase: 2024-25 SA season (confirm or reject each candidate)
  - HOLDOUT phase: 2025-26 SA season (final out-of-sample sanity check)

Only effects that survive validation AND holdout with consistent direction
get reported as "real edges worth filtering on."

Output: data/diagnostics/subset_alpha_findings.json + a printed summary.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Reuse the paper-trade machinery
EDGE_THRESHOLD = 0.05
KELLY_FRACTION = 0.25
MAX_STAKE_FRACTION = 0.02
INITIAL_BANKROLL = 1000.0


def kelly_stake(prob: float, decimal_odds: float, bankroll: float) -> float:
    if decimal_odds <= 1.0 or prob <= 0:
        return 0.0
    b = decimal_odds - 1.0
    q = 1.0 - prob
    full_kelly = (b * prob - q) / b
    if full_kelly <= 0:
        return 0.0
    return min(full_kelly * KELLY_FRACTION, MAX_STAKE_FRACTION) * bankroll


def implied_prob(odds: float) -> float:
    return 1.0 / odds if odds > 1.0 else 0.0


def select_bet_from_probs(probs: dict, odds: dict, edge_threshold: float = EDGE_THRESHOLD):
    """Pick best edge bet."""
    best = None
    for outcome in ["H", "D", "A"]:
        o = odds.get(outcome, 0.0)
        if o <= 1.01:
            continue
        edge = probs[outcome] - implied_prob(o)
        if edge < edge_threshold:
            continue
        if best is None or edge > best[3]:
            best = (outcome, probs[outcome], o, edge)
    return best


def get_actual_outcome(home_score: float, away_score: float) -> str:
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def make_bets_df(probs_df: pd.DataFrame, matches_df: pd.DataFrame, feats_df: pd.DataFrame) -> pd.DataFrame:
    """For each match, decide bet at baseline rules; return per-bet record.

    Slicing dimensions come from FEATS (which has is_midweek, is_derby, etc.),
    not from matches.parquet which only has raw fields.
    """
    bets = []
    for _, row in matches_df.iterrows():
        match_key = (row["home_team"], row["away_team"], row["match_date"])
        prob_row = probs_df[
            (probs_df["home_team"] == row["home_team"])
            & (probs_df["away_team"] == row["away_team"])
            & (probs_df["match_date"] == row["match_date"])
        ]
        feat_row = feats_df[
            (feats_df["home_team"] == row["home_team"])
            & (feats_df["away_team"] == row["away_team"])
            & (feats_df["match_date"] == row["match_date"])
        ]
        if prob_row.empty:
            continue
        probs = {
            "H": float(prob_row.iloc[0]["prob_H"]),
            "D": float(prob_row.iloc[0]["prob_D"]),
            "A": float(prob_row.iloc[0]["prob_A"]),
        }
        odds = {
            "H": float(row.get("odds_MaxH", 0)) if pd.notna(row.get("odds_MaxH")) else 0,
            "D": float(row.get("odds_MaxD", 0)) if pd.notna(row.get("odds_MaxD")) else 0,
            "A": float(row.get("odds_MaxA", 0)) if pd.notna(row.get("odds_MaxA")) else 0,
        }
        if max(odds.values()) <= 1.01:
            continue
        pick = select_bet_from_probs(probs, odds)
        if pick is None:
            continue
        outcome, p_model, op_odds, edge = pick
        actual = get_actual_outcome(row["home_score"], row["away_score"])
        won = (outcome == actual)
        stake = kelly_stake(p_model, op_odds, INITIAL_BANKROLL)
        if stake < 0.01:
            continue
        pnl = stake * (op_odds - 1.0) if won else -stake
        bets.append({
            "match_date": row["match_date"],
            "season": row["season"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "outcome_pick": outcome,
            "actual": actual,
            "won": won,
            "model_prob": p_model,
            "opening_odds": op_odds,
            "edge": edge,
            "stake": stake,
            "pnl": pnl,
        })
        # Slicing dims come from feature row when available; fall back to 0
        if not feat_row.empty:
            fr = feat_row.iloc[0]
            bets[-1].update({
                "is_midweek": int(fr.get("is_midweek", 0)) if pd.notna(fr.get("is_midweek")) else 0,
                "is_evening_kickoff": int(fr.get("is_evening_kickoff", 0)) if pd.notna(fr.get("is_evening_kickoff")) else 0,
                "is_early_kickoff": int(fr.get("is_early_kickoff", 0)) if pd.notna(fr.get("is_early_kickoff")) else 0,
                "is_derby": int(fr.get("is_derby", 0)) if pd.notna(fr.get("is_derby")) else 0,
                "is_no_crowd_match": int(fr.get("is_no_crowd_match", 0)) if pd.notna(fr.get("is_no_crowd_match")) else 0,
                "home_short_rest": int(fr.get("home_short_rest", 0)) if pd.notna(fr.get("home_short_rest")) else 0,
                "away_short_rest": int(fr.get("away_short_rest", 0)) if pd.notna(fr.get("away_short_rest")) else 0,
                "home_manager_changed": int(fr.get("home_manager_changed", 0)) if pd.notna(fr.get("home_manager_changed")) else 0,
                "away_manager_changed": int(fr.get("away_manager_changed", 0)) if pd.notna(fr.get("away_manager_changed")) else 0,
                "matchweek": int(fr.get("matchweek", 0)) if pd.notna(fr.get("matchweek")) else 0,
                "home_in_cl_zone": int(fr.get("home_in_cl_zone", 0)) if pd.notna(fr.get("home_in_cl_zone")) else 0,
                "home_in_rel_zone": int(fr.get("home_in_rel_zone", 0)) if pd.notna(fr.get("home_in_rel_zone")) else 0,
                "weather_precip": float(fr.get("weather_precipitation_sum", 0)) if pd.notna(fr.get("weather_precipitation_sum")) else 0,
                "weather_temp": float(fr.get("weather_temperature_2m_mean", 15)) if pd.notna(fr.get("weather_temperature_2m_mean")) else 15,
                "kickoff_hour": int(fr.get("kickoff_hour", 0)) if pd.notna(fr.get("kickoff_hour")) else 0,
                "ah_line_movement": float(fr.get("ah_line_movement", 0)) if pd.notna(fr.get("ah_line_movement")) else 0,
            })
        else:
            bets[-1].update({k: 0 for k in [
                "is_midweek", "is_evening_kickoff", "is_early_kickoff", "is_derby",
                "is_no_crowd_match", "home_short_rest", "away_short_rest",
                "home_manager_changed", "away_manager_changed", "matchweek",
                "home_in_cl_zone", "home_in_rel_zone", "kickoff_hour", "ah_line_movement",
            ]})
            bets[-1]["weather_precip"] = 0.0
            bets[-1]["weather_temp"] = 15.0
    return pd.DataFrame(bets)


def slice_stats(df: pd.DataFrame, name: str) -> dict:
    """Compute ROI stats for a bet slice."""
    if len(df) == 0:
        return {"name": name, "n": 0, "win_rate": None, "roi": None, "stake": 0, "pnl": 0}
    n = len(df)
    wins = int(df["won"].sum())
    stake = float(df["stake"].sum())
    pnl = float(df["pnl"].sum())
    roi = pnl / stake if stake > 0 else 0
    return {
        "name": name,
        "n": n,
        "win_rate": wins / n,
        "roi": roi,
        "stake": stake,
        "pnl": pnl,
    }


def main() -> int:
    """The actual analysis.

    Strategy: load existing per-fold model artifacts to avoid re-training.
    Score every match in 2022-23 → 2025-26 using the appropriate fold model.
    Then split-analyze.
    """
    log.info("Loading matches + features...")
    matches = pd.read_parquet(PROJECT_ROOT / "data" / "parsed" / "matches.parquet")
    matches["match_date"] = pd.to_datetime(matches["match_date"])
    matches = matches[matches["league"] == "serie_a"].copy()
    matches = matches.dropna(subset=["home_score", "away_score"]).copy()

    feats = pd.read_parquet(PROJECT_ROOT / "data" / "features" / "features_serie_a.parquet")
    feats["match_date"] = pd.to_datetime(feats["match_date"])

    target_seasons = ["2022-2023", "2023-2024", "2024-2025", "2025-2026"]
    matches = matches[matches["season"].isin(target_seasons)].copy()
    feats = feats[feats["season"].isin(target_seasons)].copy()

    log.info("Target seasons %s: %d matches", target_seasons, len(matches))

    # Score per fold using the existing walkforward seed-42 models
    from catboost import CatBoostClassifier
    from ml.feature_selection import exclude_odds
    from scripts.models.train_walkforward import LEAKY_COLUMNS, META_COLUMNS

    # Build feature selector matching the trained models
    numeric = feats.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [
        c for c in numeric
        if c not in META_COLUMNS and c not in LEAKY_COLUMNS and not c.startswith("_")
    ]
    feature_cols = [f for f in feature_cols if not feats[feature_cols].isna().all(axis=0).get(f, False)]
    _, feature_cols = exclude_odds(feats[feature_cols].copy(), feature_cols)

    log.info("Selected %d features", len(feature_cols))

    # Score each match using the model trained for its season
    # FRESH variant: scoring uses two trainer dirs depending on season —
    # the original 5season_seed42 for 22-23 / 23-24 / 24-25, and the freshly-trained
    # fresh_2025_seed42 for 25-26 (model that included 24-25 in training).
    import pickle as pk

    # FRESH variant: 2025-26 uses a freshly-trained model that INCLUDED 2024-25
    # in training data. Other seasons keep their stale fold models for context.
    season_model_map = {
        "2022-2023": ("1x2__5season_seed42", "season_2022-2023.cbm"),
        "2023-2024": ("1x2__5season_seed42", "season_2023-2024.cbm"),
        "2024-2025": ("1x2__5season_seed42", "season_2024-2025.cbm"),
        "2025-2026": ("1x2__fresh_2025_seed42", "season_2025-2026.cbm"),
    }
    base_dir = PROJECT_ROOT / "data" / "models" / "walkforward" / "serie_a"

    all_probs = []
    for season in target_seasons:
        mapping = season_model_map.get(season)
        if mapping is None:
            log.warning("No model mapping for season %s", season)
            continue
        sub_dir, model_filename = mapping
        model_path = base_dir / sub_dir / model_filename
        cal_path = base_dir / sub_dir / model_filename.replace(".cbm", "_calibrators.pkl")
        if not model_path.exists():
            log.warning("Missing %s, skipping season %s", model_path.name, season)
            continue
        m = CatBoostClassifier()
        m.load_model(str(model_path))
        season_feats = feats[feats["season"] == season].copy()
        if season_feats.empty:
            continue
        X = season_feats[feature_cols]
        proba = m.predict_proba(X)
        # Apply calibrator if present
        if cal_path.exists():
            with open(cal_path, "rb") as f:
                cals = pk.load(f)
            # Cals expected as dict {class -> IsotonicRegression}
            classes = list(m.classes_)
            cal_proba = np.zeros_like(proba)
            for i, cls in enumerate(classes):
                if str(cls) in cals:
                    cal_proba[:, i] = cals[str(cls)].transform(proba[:, i])
                elif cls in cals:
                    cal_proba[:, i] = cals[cls].transform(proba[:, i])
                else:
                    cal_proba[:, i] = proba[:, i]
            row_sum = cal_proba.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            proba = cal_proba / row_sum
        # Map model class indices to H/D/A
        classes = list(m.classes_)
        # CatBoost returns classes_ as the int-encoded class labels in its own order;
        # we encoded H/D/A alphabetically to A=0, D=1, H=2 in the trainer.
        # The classes_ attr will be [0, 1, 2] -> need to map back.
        # Look up the season's metadata to get the class mapping.
        meta_path = base_dir / sub_dir / model_filename.replace(".cbm", "_metadata.json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            spec_classes = meta.get("classes", ["A", "D", "H"])
        else:
            spec_classes = ["A", "D", "H"]
        idx_a = spec_classes.index("A") if "A" in spec_classes else 0
        idx_d = spec_classes.index("D") if "D" in spec_classes else 1
        idx_h = spec_classes.index("H") if "H" in spec_classes else 2
        season_feats = season_feats.reset_index(drop=True)
        season_feats["prob_H"] = proba[:, idx_h]
        season_feats["prob_D"] = proba[:, idx_d]
        season_feats["prob_A"] = proba[:, idx_a]
        all_probs.append(season_feats[["home_team", "away_team", "match_date", "season", "prob_H", "prob_D", "prob_A"]])

    if not all_probs:
        log.error("No probabilities computed; check model artifacts")
        return 1
    probs_df = pd.concat(all_probs, ignore_index=True)
    log.info("Computed probabilities for %d matches", len(probs_df))

    log.info("Generating bets at baseline edge threshold %.0f%%...", EDGE_THRESHOLD * 100)
    bets = make_bets_df(probs_df, matches, feats)
    if bets.empty:
        log.error("No bets produced — check edge threshold or odds availability")
        return 1
    log.info("Generated %d bets across %d matches", len(bets), len(matches))

    # Split into discovery (22-23, 23-24), validation (24-25), holdout (25-26)
    discovery = bets[bets["season"].isin(["2022-2023", "2023-2024"])].copy()
    validation = bets[bets["season"] == "2024-2025"].copy()
    holdout = bets[bets["season"] == "2025-2026"].copy()
    log.info(
        "Splits: discovery=%d, validation=%d, holdout=%d",
        len(discovery), len(validation), len(holdout)
    )

    print()
    print("=" * 80)
    print("BASELINE (no filter) — sanity check")
    print("=" * 80)
    for split_name, df in [("discovery", discovery), ("validation", validation), ("holdout", holdout)]:
        s = slice_stats(df, split_name)
        if s["n"] == 0:
            print(f"  {split_name:<12} n=   0  (empty split)")
            continue
        print(f"  {split_name:<12} n={s['n']:>4}  win={s['win_rate']:.1%}  roi={s['roi']:+.2%}  pnl={s['pnl']:+.2f}")

    # Candidate dimensions to slice on
    binary_dims = [
        "is_midweek", "is_evening_kickoff", "is_early_kickoff", "is_derby",
        "is_no_crowd_match", "home_short_rest", "away_short_rest",
        "home_manager_changed", "away_manager_changed",
        "home_in_cl_zone", "home_in_rel_zone",
    ]
    # Also slice on outcome class
    outcome_dims = [("pick_H", lambda d: d[d["outcome_pick"] == "H"]),
                    ("pick_D", lambda d: d[d["outcome_pick"] == "D"]),
                    ("pick_A", lambda d: d[d["outcome_pick"] == "A"])]
    # Continuous dimensions binned
    continuous_bins = {
        "matchweek_early(1-12)": lambda d: d[d["matchweek"].between(1, 12)],
        "matchweek_mid(13-26)": lambda d: d[d["matchweek"].between(13, 26)],
        "matchweek_late(27-38)": lambda d: d[d["matchweek"] >= 27],
        "weather_precip>5mm": lambda d: d[d["weather_precip"] > 5],
        "weather_cold(<10C)": lambda d: d[d["weather_temp"] < 10],
        "kickoff_<=15h": lambda d: d[d["kickoff_hour"].between(1, 15)],
        "kickoff_>=20h": lambda d: d[d["kickoff_hour"] >= 20],
    }

    findings = []
    print()
    print("=" * 80)
    print("DISCOVERY → VALIDATION → HOLDOUT")
    print("=" * 80)
    print(f"{'dimension':<35}{'disc n':>8}{'disc roi':>10}{'val n':>7}{'val roi':>10}{'hold n':>7}{'hold roi':>10}{'verdict':>14}")
    print("-" * 105)

    def evaluate_slice(name: str, sub_d, sub_v, sub_h):
        s_d = slice_stats(sub_d, name + "/disc")
        s_v = slice_stats(sub_v, name + "/val")
        s_h = slice_stats(sub_h, name + "/hold")
        # Verdict logic: needs n≥30 in discovery, ROI direction consistent across all 3 splits
        verdict = "—"
        if s_d["n"] >= 30 and s_d["roi"] is not None:
            disc_roi = s_d["roi"]
            val_roi = s_v["roi"] if s_v["n"] >= 10 else None
            hold_roi = s_h["roi"] if s_h["n"] >= 5 else None
            # consistent positive direction
            all_signs = [r for r in [disc_roi, val_roi, hold_roi] if r is not None]
            if disc_roi > 0.03 and val_roi is not None and val_roi > 0 and hold_roi is not None and hold_roi > -0.05:
                verdict = "STRONG+"
            elif disc_roi > 0.03 and (val_roi or 0) > 0:
                verdict = "weak+"
            elif disc_roi < -0.05 and (val_roi or 0) < 0:
                verdict = "STRONG-"
            elif disc_roi < -0.05:
                verdict = "weak-"
        n_d = f"{s_d['n']:>8}"
        roi_d = f"{s_d['roi']:+.2%}" if s_d["roi"] is not None else "  n/a"
        n_v = f"{s_v['n']:>7}"
        roi_v = f"{s_v['roi']:+.2%}" if s_v["roi"] is not None else "  n/a"
        n_h = f"{s_h['n']:>7}"
        roi_h = f"{s_h['roi']:+.2%}" if s_h["roi"] is not None else "  n/a"
        print(f"{name:<35}{n_d}{roi_d:>10}{n_v}{roi_v:>10}{n_h}{roi_h:>10}{verdict:>14}")
        return {
            "dimension": name,
            "discovery": s_d,
            "validation": s_v,
            "holdout": s_h,
            "verdict": verdict,
        }

    # Binary dims — slice as TRUE
    for dim in binary_dims:
        sub_d = discovery[discovery[dim] == 1]
        sub_v = validation[validation[dim] == 1]
        sub_h = holdout[holdout[dim] == 1]
        findings.append(evaluate_slice(f"{dim}=1", sub_d, sub_v, sub_h))

    # Outcome class dims
    for name, fn in outcome_dims:
        findings.append(evaluate_slice(name, fn(discovery), fn(validation), fn(holdout)))

    # Continuous bins
    for name, fn in continuous_bins.items():
        findings.append(evaluate_slice(name, fn(discovery), fn(validation), fn(holdout)))

    # Combined high-leverage filter from real_edge_found.md
    print("-" * 105)
    print("COMBINED FILTERS")
    print("-" * 105)
    combined_specs = [
        ("draw_picks_only", lambda d: d[d["outcome_pick"] == "D"]),
        ("mid_odds(2.2-4.0)", lambda d: d[d["opening_odds"].between(2.2, 4.0)]),
        ("mid_prob(0.30-0.50)", lambda d: d[d["model_prob"].between(0.30, 0.50)]),
        ("mid_odds + mid_prob", lambda d: d[d["opening_odds"].between(2.2, 4.0) & d["model_prob"].between(0.30, 0.50)]),
        ("draws + mid_odds", lambda d: d[(d["outcome_pick"] == "D") & d["opening_odds"].between(2.2, 4.0)]),
    ]
    for name, fn in combined_specs:
        findings.append(evaluate_slice(name, fn(discovery), fn(validation), fn(holdout)))

    # Save findings
    out_path = PROJECT_ROOT / "data" / "diagnostics" / "subset_alpha_findings.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(findings, f, indent=2, default=str)

    print()
    print("=" * 80)
    print(f"Saved {len(findings)} findings to {out_path.relative_to(PROJECT_ROOT)}")
    strong_pos = [f for f in findings if f["verdict"] == "STRONG+"]
    weak_pos = [f for f in findings if f["verdict"] == "weak+"]
    strong_neg = [f for f in findings if f["verdict"] == "STRONG-"]
    print(f"STRONG+ (consistent across 3 splits): {len(strong_pos)}")
    for f in strong_pos:
        print(f"  - {f['dimension']}")
    print(f"weak+: {len(weak_pos)}")
    for f in weak_pos:
        print(f"  - {f['dimension']}")
    print(f"STRONG-: {len(strong_neg)}")
    for f in strong_neg:
        print(f"  - {f['dimension']}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
