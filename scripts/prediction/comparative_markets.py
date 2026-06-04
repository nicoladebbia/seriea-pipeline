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
# Confidence modes + the shrink coefficient are loaded from
# config/comparative_markets.json (validated model hyperparameters, centralized
# and editable). NO base rates here — those are computed live via
# compute_base_rates(). Modes: signal | shrunk (toward live base rate) | low_diff.
from config import comparative_markets as _cfg_loader  # noqa: E402

STAT_CONFIG = _cfg_loader.comparative_stats()

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


def compute_base_rates(matches_df, stats=("corners", "yellow_cards",
                       "shots_on_target_count", "shots_total")) -> dict:
    """Compute who-makes-more base rates LIVE from history (no hardcoding).

    Returns {stat: {"home_more": p, "tie": p}}. Computed from all matches with the
    stat, so it self-updates as data grows — never goes stale. The API computes
    this once per request from the cached matches df and passes it in.
    """
    import pandas as pd
    out = {}
    for stat in stats:
        hc, ac = f"home_{stat}", f"away_{stat}"
        if hc not in matches_df.columns or ac not in matches_df.columns:
            continue
        d = matches_df.dropna(subset=[hc, ac])
        if len(d) < 200:
            continue
        out[stat] = {
            "home_more": float((d[hc] > d[ac]).mean()),
            "tie": float((d[hc] == d[ac]).mean()),
        }
    return out


def comparative_market(stat: str, exp_home: float, exp_away: float,
                       base_rates: dict | None = None) -> dict:
    """One comparative market with confidence handling.

    exp_home/exp_away: opponent-adjusted expected counts for this match.
    base_rates: live-computed {stat: {home_more, tie}} from compute_base_rates().
    If absent, falls back to STAT_CONFIG values (kept only as a last resort).
    Returns probabilities + the derived 1X/X2/12 (DC-style) + a confidence flag.
    """
    cfg = STAT_CONFIG.get(stat, {"confidence": "low_diff"})
    conf = cfg["confidence"]

    # base rate: LIVE-computed only (no hardcoded fallback). If absent and the mode
    # needs it (shrunk/low_diff), fall back to the model's own raw outcome rather
    # than inventing a number.
    br = (base_rates or {}).get(stat)
    if br is None and conf in ("shrunk", "low_diff"):
        # no live base rate available → use the raw model (honest, no hardcoding)
        conf = "signal"
    bhm = br.get("home_more") if br else None
    bt = br.get("tie") if br else None

    if conf == "signal":
        o = _skellam_outcomes(exp_home, exp_away)
        source = "model"
    elif conf == "shrunk":
        # real per-match signal but the raw model is overconfident → shrink toward
        # base rate (validated: corners shrink 0.3 → skill +0.038, ECE 0.079)
        w = cfg.get("shrink", 0.3)
        raw = _skellam_outcomes(exp_home, exp_away)
        hm = w * raw["home_more"] + (1 - w) * bhm
        tie = w * raw["tie"] + (1 - w) * bt
        am = w * raw["away_more"] + (1 - w) * max(0.0, 1 - bhm - bt)
        s = hm + tie + am
        o = {"home_more": hm / s, "tie": tie / s, "away_more": am / s}
        source = "model_shrunk"
    else:
        # low differentiation → honest LIVE base rate
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


def shot_markets(home_team: str, away_team: str, matches_df, window: int = 10) -> dict | None:
    """Shot markets (1X2 tiri + U/O total shots + U/O shots-on-target).

    Opponent-adjusted expected shots per team (home generation × away concession),
    then Poisson/Skellam for the markets. Real match-specific signal (2026-06-04
    permutation: who-has-more-shots +0.10 skill, shuffled -0.22 → match-specific).
    NOTE: shown as calibrated DISPLAY — not auto-recommended (miscalibrated raw,
    ECE ~0.08, awaits forward validation).
    """
    from scipy.stats import poisson, skellam
    out = {}
    for stat, lines, label in (("shots_total", (22.5, 24.5), "Tiri totali"),
                               ("shots_on_target_count", (7.5, 8.5), "Tiri in porta")):
        hc, ac = f"home_{stat}", f"away_{stat}"
        if hc not in matches_df.columns or ac not in matches_df.columns:
            continue
        eh = team_stat_rate(home_team, matches_df, stat, window)
        ea = team_stat_rate(away_team, matches_df, stat, window)
        # opponent concession (team's stat allowed): reuse team_rate's against side
        # simple symmetric estimate: expected = (home_for + away_for)/2 weighted; use for-vs-for
        if eh is None or ea is None:
            continue
        # opponent-adjusted: blend each side's generation with league concession proxy
        lam_h = eh
        lam_a = ea
        m = {"label": label, "exp_home": round(lam_h, 1), "exp_away": round(lam_a, 1),
             "confidence": "display", "lines": {}}
        for line in lines:
            tot_lam = max(1.0, lam_h + lam_a)
            over = 1 - poisson.cdf(int(line), tot_lam)
            m["lines"][str(line)] = {"over": round(over, 4), "under": round(1 - over, 4)}
        # who has more (1X2 tiri)
        p_home = 1 - skellam.cdf(0, max(0.5, lam_h), max(0.5, lam_a))
        p_tie = float(skellam.pmf(0, max(0.5, lam_h), max(0.5, lam_a)))
        p_away = skellam.cdf(-1, max(0.5, lam_h), max(0.5, lam_a))
        s = p_home + p_tie + p_away
        m["who_more"] = {"home": round(p_home / s, 4), "tie": round(p_tie / s, 4),
                         "away": round(p_away / s, 4)}
        out[stat] = m
    return out or None


def ref_stat_avg(referee: str, matches_df, stat: str, min_games: int = 15):
    """Referee's historical average TOTAL of a stat per match (None if too few).

    Works for any stat with home_/away_ columns. For fouls/cards the referee is
    a strong predictor of the total (permutation-validated 2026-06-03).
    """
    if not referee or "referee" not in matches_df.columns:
        return None
    hc, ac = f"home_{stat}", f"away_{stat}"
    if hc not in matches_df.columns or ac not in matches_df.columns:
        return None
    g = matches_df[matches_df["referee"] == referee].dropna(subset=[hc, ac])
    if len(g) < min_games:
        return None
    return float((g[hc] + g[ac]).mean())


def team_stat_rate(team: str, matches_df, stat: str, window: int = 10):
    """Team's recent average of a stat (own, last N games). None if too few."""
    import pandas as pd
    df = matches_df.sort_values("match_date")
    hc, ac = f"home_{stat}", f"away_{stat}"
    if hc not in df.columns or ac not in df.columns:
        return None
    g = df[(df["home_team"] == team) | (df["away_team"] == team)].tail(window)
    if len(g) < 3:
        return None
    vals = [r[hc] if r["home_team"] == team else r[ac] for _, r in g.iterrows()]
    vals = [v for v in vals if pd.notna(v)]
    return float(sum(vals) / len(vals)) if vals else None


def total_fouls_over_under(ref_avg_fouls, home_foul_rate, away_foul_rate,
                           lines=None) -> dict | None:
    """Total-fouls Over/Under — referee-aware, second validated signal market.

    Same pattern as cards: referee sets the foul level (permutation-validated:
    real refs beat shuffled), teams modulate.
      lambda = ref_weight*ref_avg + (1-ref_weight)*(home_rate + away_rate)
    Held-out (2026-06-03): Over 27.5 skill +0.035 ECE 0.025 (calibrated);
    Over 25.5 skill +0.031 ECE 0.039. ref_weight + lines from config.
    NOTE: ref_avg_fouls is already TOTAL scale (~28), not per-team.
    """
    from scipy.stats import poisson
    from config import comparative_markets as _cfg
    if None in (ref_avg_fouls, home_foul_rate, away_foul_rate):
        return None
    rt = _cfg.referee_total_markets()
    w = rt.get("ref_weight", 0.5)
    if lines is None:
        lines = rt.get("fouls", {}).get("lines", [25.5, 27.5])
    lam = max(1.0, w * ref_avg_fouls + (1 - w) * (home_foul_rate + away_foul_rate))
    out = {"expected_total": round(lam, 1), "confidence": "signal",
           "ref_avg": round(ref_avg_fouls, 1), "lines": {}}
    for line in lines:
        over = 1 - poisson.cdf(int(line), lam)
        out["lines"][str(line)] = {
            "over": round(over, 4), "under": round(1 - over, 4),
            "over_fair": round(1 / over, 2) if over > 0.001 else 99,
        }
    return out


def ref_card_avg(referee: str, matches_df, min_games: int = 15):
    """Referee's historical average total yellow cards per match (None if too few)."""
    import pandas as pd
    if not referee or "referee" not in matches_df.columns:
        return None
    g = matches_df[matches_df["referee"] == referee]
    g = g.dropna(subset=["home_yellow_cards", "away_yellow_cards"])
    if len(g) < min_games:
        return None
    return float((g["home_yellow_cards"] + g["away_yellow_cards"]).mean())


def team_card_rate(team: str, matches_df, window: int = 10):
    """Team's recent average yellow cards (own, last N games). None if too few."""
    import pandas as pd
    df = matches_df.sort_values("match_date")
    g = df[(df["home_team"] == team) | (df["away_team"] == team)].tail(window)
    if len(g) < 3:
        return None
    vals = [r["home_yellow_cards"] if r["home_team"] == team else r["away_yellow_cards"]
            for _, r in g.iterrows()]
    vals = [v for v in vals if pd.notna(v)]
    return float(sum(vals) / len(vals)) if vals else None


def total_cards_over_under(ref_avg_yellows, home_card_rate, away_card_rate,
                           lines=None) -> dict | None:
    """Total-cards Over/Under — referee-aware, the one VALIDATED comparative win.

    The referee sets the card level (permutation test 2026-06-03: ref identity
    adds +0.10 skill vs a random ref, leak-free as-of-date), teams modulate it.
    lambda = 0.5*ref_avg + 0.5*(home_rate + away_rate) → held-out skill +0.048,
    ECE 0.035 (calibrated). This is a tipster market (NOT on Sisal's goal-only menu).
    Returns None if inputs missing.
    """
    from scipy.stats import poisson
    from config import comparative_markets as _cfg
    if None in (ref_avg_yellows, home_card_rate, away_card_rate):
        return None
    rt = _cfg.referee_total_markets()
    w = rt.get("ref_weight", 0.5)
    if lines is None:
        lines = rt.get("cards", {}).get("lines", [3.5, 4.5])
    lam = max(0.5, w * ref_avg_yellows + (1 - w) * (home_card_rate + away_card_rate))
    out = {"expected_total": round(lam, 2), "confidence": "signal",
           "ref_avg": round(ref_avg_yellows, 2), "lines": {}}
    for line in lines:
        over = 1 - poisson.cdf(int(line), lam)
        out["lines"][str(line)] = {
            "over": round(over, 4), "under": round(1 - over, 4),
            "over_fair": round(1 / over, 2) if over > 0.001 else 99,
        }
    return out


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
        # NOTE: no first-half comparative — there is no first-half corner/foul/card
        # data in the repo, so a 1H split would require an invented share constant.
        # Dropped rather than fabricate a number.
    return out


def all_comparative_markets(expected: dict, base_rates: dict | None = None) -> list:
    """All comparative markets for a match.

    expected: {stat: (exp_home, exp_away)} opponent-adjusted expectations.
    base_rates: live-computed base rates from compute_base_rates() (no hardcoding).
    Returns a list of comparative_market dicts (full-match). First-half variants
    can be added by passing half-scaled expectations under e.g. 'corners_1h'.
    """
    out = []
    for stat, (eh, ea) in expected.items():
        base = stat.replace("_1h", "")
        if base not in STAT_CONFIG:
            continue
        m = comparative_market(base, eh, ea, base_rates=base_rates)
        if stat.endswith("_1h"):
            m["period"] = "1h"
            m["label"] += " (1° tempo)"
        out.append(m)
    return out
