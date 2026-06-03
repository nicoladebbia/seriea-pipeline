"""Comparative team-stat markets — "who makes more corners / fouls / cards".

The tipster bet markets like "who makes more corners", "who makes more fouls",
"1X on corners" (home makes more or equal). These are NOT goal markets and the
score-range engine can't produce them. This module models each team's stat as
a Poisson count, opponent-adjusted with a home-advantage term, and computes:
  P(home > away), P(tie), P(away > home)
via the Skellam distribution (the exact difference of two Poissons).

Honesty rule (validated 2026-06-03 backtest):
- FOULS has real per-match signal (skill +0.016 opponent-adjusted) → show the model %.
- CORNERS, CARDS are low-differentiation (the per-match prediction is no better than
  the base rate) → fall back to the BASE RATE, labelled 'low differentiation', because
  a worse-than-base per-match number is anti-informative.
Each market is tagged confidence: "signal" | "low_diff" so the UI shows which to trust.

This produces PROBABILITIES (the user's goal), not betting edges.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import skellam

# Validated per-stat: does the opponent-adjusted model beat the base rate?
# (skill > 0.01 on the 2026-06-03 walk-forward backtest). If not → show base rate.
STAT_CONFIG = {
    "fouls":        {"confidence": "signal",   "skill": 0.016},
    "corners":      {"confidence": "low_diff",  "skill": -0.052, "base_home_more": 0.554, "base_tie": 0.091},
    "yellow_cards": {"confidence": "low_diff",  "skill": -0.072, "base_home_more": 0.32, "base_tie": 0.18},
    "shots_on_target_count": {"confidence": "low_diff", "skill": 0.0, "base_home_more": 0.55, "base_tie": 0.08},
    "shots_total":  {"confidence": "low_diff",  "skill": 0.0, "base_home_more": 0.56, "base_tie": 0.05},
}

# Friendly market labels (Italian/tipster style)
STAT_LABELS = {
    "fouls": "Falli (chi ne fa di più)",
    "corners": "Corner (chi ne fa di più)",
    "yellow_cards": "Cartellini gialli (chi ne prende di più)",
    "shots_on_target_count": "Tiri in porta (chi ne fa di più)",
    "shots_total": "Tiri totali (chi ne fa di più)",
}


def _skellam_outcomes(exp_home: float, exp_away: float) -> dict:
    """P(home>away), P(tie), P(away>home) for two Poisson counts."""
    eh = max(0.1, exp_home)
    ea = max(0.1, exp_away)
    p_tie = float(skellam.pmf(0, eh, ea))
    p_home_more = float(1 - skellam.cdf(0, eh, ea))
    p_away_more = float(skellam.cdf(-1, eh, ea))
    s = p_home_more + p_tie + p_away_more
    if s > 0:
        p_home_more, p_tie, p_away_more = p_home_more / s, p_tie / s, p_away_more / s
    return {"home_more": p_home_more, "tie": p_tie, "away_more": p_away_more}


def comparative_market(stat: str, exp_home: float, exp_away: float) -> dict:
    """One comparative market with confidence handling.

    exp_home/exp_away: opponent-adjusted expected counts for this match.
    Returns probabilities + the derived 1X/X2/12 (DC-style) for the stat, plus a
    confidence flag. For 'low_diff' stats, falls back to the validated base rate so
    we never show a per-match number that's worse than guessing.
    """
    cfg = STAT_CONFIG.get(stat, {"confidence": "low_diff"})
    conf = cfg["confidence"]

    if conf == "signal":
        o = _skellam_outcomes(exp_home, exp_away)
        source = "model"
    else:
        # low differentiation → honest base rate (model per-match was worse than base)
        bhm = cfg.get("base_home_more", 0.50)
        bt = cfg.get("base_tie", 0.08)
        o = {"home_more": bhm, "tie": bt, "away_more": max(0.0, 1 - bhm - bt)}
        source = "base_rate"

    hm, tie, am = o["home_more"], o["tie"], o["away_more"]
    return {
        "stat": stat,
        "label": STAT_LABELS.get(stat, stat),
        "confidence": conf,         # "signal" | "low_diff"
        "source": source,           # "model" | "base_rate"
        "exp_home": round(exp_home, 2),
        "exp_away": round(exp_away, 2),
        # who makes more
        "home_more": round(hm, 4),
        "tie": round(tie, 4),
        "away_more": round(am, 4),
        # DC-style (1X = home more or equal, X2 = away more or equal, 12 = not tie)
        "home_more_or_eq": round(hm + tie, 4),
        "away_more_or_eq": round(am + tie, 4),
        "not_tie": round(hm + am, 4),
    }


def compute_expected_counts(home_team: str, away_team: str, matches_df,
                            stats=("corners", "fouls", "yellow_cards"),
                            window: int = 10) -> dict:
    """Opponent-adjusted expected counts per stat for an upcoming match.

    For each stat: exp_home = league_avg * (home_for/L) * (away_against/L), which
    blends home's generation rate with away's concession rate (pre-match, no leak).
    Returns {stat: (exp_home, exp_away)} for stats with enough history, else skipped.
    """
    import numpy as np
    import pandas as pd

    df = matches_df.sort_values("match_date")

    def team_rate(team, stat):
        games = df[(df["home_team"] == team) | (df["away_team"] == team)].tail(window)
        if len(games) < 3:
            return None, None
        fors, against = [], []
        for _, r in games.iterrows():
            if r["home_team"] == team:
                fors.append(r.get(f"home_{stat}")); against.append(r.get(f"away_{stat}"))
            else:
                fors.append(r.get(f"away_{stat}")); against.append(r.get(f"home_{stat}"))
        fors = [x for x in fors if pd.notna(x)]
        against = [x for x in against if pd.notna(x)]
        return (np.mean(fors) if fors else None, np.mean(against) if against else None)

    # empirical first-half share of each stat (corners/fouls/cards mostly even
    # across halves; ~0.46 reflects slightly fewer events in the opening period)
    HALF_SHARE = 0.46

    out = {}
    for stat in stats:
        hcol, acol = f"home_{stat}", f"away_{stat}"
        if hcol not in df.columns or acol not in df.columns:
            continue
        L = df[hcol].add(df[acol], fill_value=0).div(2).tail(200).mean()
        if not L or L <= 0:
            continue
        hf, ha = team_rate(home_team, stat)
        af, aa = team_rate(away_team, stat)
        if None in (hf, ha, af, aa):
            continue
        exp_h = L * (hf / L) * (aa / L)
        exp_a = L * (af / L) * (ha / L)
        out[stat] = (float(exp_h), float(exp_a))
        # first-half variant (only for stats where it's a real Sisal market — corners)
        if stat == "corners":
            out[f"{stat}_1h"] = (float(exp_h * HALF_SHARE), float(exp_a * HALF_SHARE))
    return out


def all_comparative_markets(expected: dict) -> list:
    """All comparative markets for a match.

    expected: {stat: (exp_home, exp_away)} opponent-adjusted expectations.
    Returns a list of comparative_market dicts (full-match). First-half variants
    can be added by passing half-scaled expectations under e.g. 'corners_1h'.
    """
    out = []
    for stat, (eh, ea) in expected.items():
        base = stat.replace("_1h", "")
        if base not in STAT_CONFIG:
            continue
        m = comparative_market(base, eh, ea)
        if stat.endswith("_1h"):
            m["period"] = "1h"
            m["label"] += " (1° tempo)"
        out.append(m)
    return out
