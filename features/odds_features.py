"""Derived odds features for the ML training pipeline.

Computes second-order features from the raw odds and implied probabilities
already merged in Step 30:
  - Sharp-soft divergence (Pinnacle vs market average)
  - Asian handicap normalization
  - Market-implied goal totals
  - Best-available probability (coalesced Pinnacle/market)
  - Odds consistency measures
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def add_odds_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ~14 derived odds features to the feature table.

    All features are NaN-safe: rows without odds data simply get NaN values
    (which the column-cleanup step in build.py will handle if coverage is too low).
    """
    df = df.copy()
    added = 0

    # --- Best-available probabilities (coalesce Pinnacle → market) ---
    for outcome in ("home", "draw", "away"):
        pin_col = f"pinnacle_{outcome}_prob"
        mkt_col = f"market_{outcome}_prob"
        best_col = f"{outcome}_prob_best"
        if pin_col in df.columns or mkt_col in df.columns:
            pin = df[pin_col] if pin_col in df.columns else pd.Series(np.nan, index=df.index)
            mkt = df[mkt_col] if mkt_col in df.columns else pd.Series(np.nan, index=df.index)
            df[best_col] = pin.fillna(mkt)
            added += 1

    # Best O/U probability
    for side in ("over", "under"):
        pin_col = f"pinnacle_ou_{side}_prob"
        mkt_col = f"market_ou_{side}_prob"
        best_col = f"ou_{side}_prob_best"
        if pin_col in df.columns or mkt_col in df.columns:
            pin = df[pin_col] if pin_col in df.columns else pd.Series(np.nan, index=df.index)
            mkt = df[mkt_col] if mkt_col in df.columns else pd.Series(np.nan, index=df.index)
            df[best_col] = pin.fillna(mkt)
            added += 1

    # --- Sharp-soft divergence (Pinnacle - market average) ---
    for outcome in ("home", "draw", "away"):
        pin_col = f"pinnacle_{outcome}_prob"
        mkt_col = f"market_{outcome}_prob"
        div_col = f"sharp_soft_{outcome}_div"
        if pin_col in df.columns and mkt_col in df.columns:
            df[div_col] = (df[pin_col] - df[mkt_col]).round(4)
            added += 1

    # O/U sharp-soft divergence
    pin_ou = "pinnacle_ou_over_prob"
    mkt_ou = "market_ou_over_prob"
    if pin_ou in df.columns and mkt_ou in df.columns:
        df["sharp_soft_ou_div"] = (df[pin_ou] - df[mkt_ou]).round(4)
        added += 1

    # --- Asian handicap normalization ---
    ah_col = "odds_AHh"
    if ah_col in df.columns:
        ah = pd.to_numeric(df[ah_col], errors="coerce")
        df["ah_line_normalized"] = ah.clip(-3, 3).round(2)
        df["ah_line_abs"] = ah.abs().round(2)
        added += 2

    # --- Odds home favourite flag ---
    if "home_prob_best" in df.columns and "away_prob_best" in df.columns:
        df["odds_home_fav"] = (df["home_prob_best"] > df["away_prob_best"]).astype(float)
        # Set NaN where both probs are missing
        both_missing = df["home_prob_best"].isna() & df["away_prob_best"].isna()
        df.loc[both_missing, "odds_home_fav"] = np.nan
        added += 1

    # --- Market-implied goal total ---
    # Linear approximation: lambda ~= 2.5 + 2.0 * (p_over - 0.5)
    # Accurate within +/- 0.1 goals for p in [0.3, 0.7]
    if "ou_over_prob_best" in df.columns:
        p = df["ou_over_prob_best"]
        df["market_goal_total"] = (2.5 + 2.0 * (p - 0.5)).round(2)
        added += 1

    # --- Odds consistency (agreement between sharp and soft books) ---
    # 1 = perfect agreement, lower = larger divergence
    if "pinnacle_home_prob" in df.columns and "market_home_prob" in df.columns:
        pin = df["pinnacle_home_prob"]
        mkt = df["market_home_prob"]
        max_prob = pd.concat([pin, mkt], axis=1).max(axis=1)
        df["odds_consistency"] = (1 - (pin - mkt).abs() / max_prob.clip(lower=0.01)).round(4)
        added += 1

    if "pinnacle_ou_over_prob" in df.columns and "market_ou_over_prob" in df.columns:
        pin = df["pinnacle_ou_over_prob"]
        mkt = df["market_ou_over_prob"]
        max_prob = pd.concat([pin, mkt], axis=1).max(axis=1)
        df["ou_consistency"] = (1 - (pin - mkt).abs() / max_prob.clip(lower=0.01)).round(4)
        added += 1

    # --- Line Velocity: Opening → Closing odds movement ---
    # Positive velocity = odds shortened (more money on that outcome)
    # Negative velocity = odds drifted (money moved away)
    # Computed as: (1/closing - 1/opening) = implied prob change

    # Pinnacle line velocity (sharpest signal)
    for outcome, open_col, close_col in [
        ("home", "odds_PSH", "odds_PSCH"),
        ("draw", "odds_PSD", "odds_PSCD"),
        ("away", "odds_PSA", "odds_PSCA"),
    ]:
        if open_col in df.columns and close_col in df.columns:
            o = pd.to_numeric(df[open_col], errors="coerce")
            c = pd.to_numeric(df[close_col], errors="coerce")
            # Implied probability change
            df[f"line_vel_pin_{outcome}"] = ((1 / c) - (1 / o)).round(4)
            added += 1

    # Market average line velocity
    for outcome, open_col, close_col in [
        ("home", "odds_AvgH", "odds_AvgCH"),
        ("draw", "odds_AvgD", "odds_AvgCD"),
        ("away", "odds_AvgA", "odds_AvgCA"),
    ]:
        if open_col in df.columns and close_col in df.columns:
            o = pd.to_numeric(df[open_col], errors="coerce")
            c = pd.to_numeric(df[close_col], errors="coerce")
            df[f"line_vel_mkt_{outcome}"] = ((1 / c) - (1 / o)).round(4)
            added += 1

    # Over/Under line velocity (Pinnacle)
    for side, open_col, close_col in [
        ("over", "odds_P>2.5", "odds_PC>2.5"),
        ("under", "odds_P<2.5", "odds_PC<2.5"),
    ]:
        if open_col in df.columns and close_col in df.columns:
            o = pd.to_numeric(df[open_col], errors="coerce")
            c = pd.to_numeric(df[close_col], errors="coerce")
            df[f"line_vel_ou_{side}"] = ((1 / c) - (1 / o)).round(4)
            added += 1

    # Asian Handicap line movement
    if "odds_AHh" in df.columns and "odds_AHCh" in df.columns:
        ah_open = pd.to_numeric(df["odds_AHh"], errors="coerce")
        ah_close = pd.to_numeric(df["odds_AHCh"], errors="coerce")
        df["ah_line_movement"] = (ah_close - ah_open).round(2)
        added += 1

    # Composite steam move indicator: large Pinnacle home velocity
    if "line_vel_pin_home" in df.columns:
        vel = df["line_vel_pin_home"].abs()
        df["steam_move_flag"] = (vel > 0.03).astype(float)  # >3% implied prob shift
        df.loc[df["line_vel_pin_home"].isna(), "steam_move_flag"] = np.nan
        added += 1

    log.info("Added %d odds-derived features (incl. line velocity)", added)
    return df
