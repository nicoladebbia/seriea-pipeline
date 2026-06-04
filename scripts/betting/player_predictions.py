"""
Player-Level Floor Prediction Engine for Serie A / EPL
======================================================
Predicts per-player per-match floor markets:
  1. Anytime Goalscorer (P(goals >= 1))
  2. Player Shots O/U 0.5, 1.5, 2.5
  3. Player Shots on Target O/U 0.5, 1.5
  4. Player Fouls Committed O/U 0.5, 1.5
  5. Player Fouls Drawn O/U 0.5

Method (validated 2026-06-04 — see .plans/player-props-deep-plan.md):
  Per-player EMPIRICAL per-90 rate, computed as an expanding mean over the
  player's prior matches where they played 60+ minutes (leak-free, no current
  match), scaled by LEAK-FREE projected minutes, fed through a Poisson tail to
  P(>= k).

Why this and NOT a CatBoost stack / xG model:
  - A 20-factor GBM overfit and died on the within-position control (6% vs 6%).
  - The empirical per-90 rate SURVIVES every control: permutation (corr 0.43 ->
    -0.005 shuffled), within-forwards (+0.34 spread), 7-season walk-forward
    (corr 0.38-0.48, stable), and is naturally calibrated.
  - Poisson on LEAK-FREE projected minutes beat the binary rate on a leak-free
    test: Brier 0.1532 vs 0.1646 (base 0.1753), decile-calibration max error
    0.066 vs 0.201, and held on the Over-1.5 tail (0.0468 vs 0.0509).
  - The signal is a stable VOLUME HABIT (shots/SoT/fouls at 25-60% base rate),
    not a rare binary (scorer 8%) dominated by single-match luck.

Output contract (consumed by player_prop_odds.py + prop_tracker.py):
  predict_match_players(...) -> {
    home_team, away_team,
    home_players: [ {player_name, player_id, team, opponent, position, is_home,
                     proj_minutes, markets: {<key>: {label, prob, odds_implied}}} ],
    away_players: [ ... ]
  }
  Each markets[key]["prob"] is the calibrated P(player clears the line).

Data source: data/external/sofascore/player_match_stats.parquet (101,875 SA rows
+ premier_league variant, 2017-2026, 100% fill on shots/SoT/fouls/minutes).
Per-event prop odds (player_shots, player_shots_on_target, player_goal_scorer_*)
ARE fetched by player_prop_odds.py. BETTING stays gated (config betting_rules.json
Player_Props) until a forward sample shows edge vs the line — predictability is
validated, EDGE is not (and can't be backtested from one stale odds snapshot).
"""

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
SS_DIR = DATA_DIR / "external" / "sofascore"

TEAM_MAP = {
    "Hellas Verona": "Verona", "ChievoVerona": "Chievo",
    "AC Milan": "Milan", "Inter Milan": "Inter",
    "Internazionale": "Inter", "Parma Calcio 1913": "Parma",
}

# Minimum prior 60+min matches before we trust a player's rate. Below this we
# fall back to the position base rate (computed live from the same data).
MIN_PRIOR_MATCHES = 10
# Only count history where the player actually played enough to express the habit.
MIN_MINUTES_FOR_RATE = 60
# Per-90 rates are clipped to keep one freak match from blowing up a projection.
MAX_P90 = {
    "total_shots": 8.0,
    "shots_on_target": 4.0,
    "fouls": 6.0,
    "was_fouled": 6.0,
    "goals": 1.5,
}

# ── market definitions ───────────────────────────────────────────────────────
# col: the per-match count column; line: "Over <line>" means count >= line+1;
# k = line + 1 is the threshold for the Poisson tail P(count >= k).
TARGETS = {
    "goalscorer":  {"col": "goals",           "line": 0, "label": "Anytime Goalscorer"},
    "shots_o05":   {"col": "total_shots",     "line": 0, "label": "Shots Over 0.5"},
    "shots_o15":   {"col": "total_shots",     "line": 1, "label": "Shots Over 1.5"},
    "shots_o25":   {"col": "total_shots",     "line": 2, "label": "Shots Over 2.5"},
    "sot_o05":     {"col": "shots_on_target", "line": 0, "label": "Shots on Target Over 0.5"},
    "sot_o15":     {"col": "shots_on_target", "line": 1, "label": "Shots on Target Over 1.5"},
    "fouls_o05":   {"col": "fouls",           "line": 0, "label": "Fouls Committed Over 0.5"},
    "fouls_o15":   {"col": "fouls",           "line": 1, "label": "Fouls Committed Over 1.5"},
    "fouled_o05":  {"col": "was_fouled",      "line": 0, "label": "Fouls Drawn Over 0.5"},
}

# Distinct count columns we build per-90 priors for.
_RATE_COLS = sorted({t["col"] for t in TARGETS.values()})


# =============================================================================
# DATA LOADING
# =============================================================================

def load_player_data(league: str = "serie_a") -> pd.DataFrame:
    """Load Sofascore player_match_stats for the given league.

    league: "serie_a" -> player_match_stats.parquet
            "premier_league" -> player_match_stats_premier_league.parquet
    """
    fname = (
        "player_match_stats.parquet"
        if league == "serie_a"
        else "player_match_stats_premier_league.parquet"
    )
    path = SS_DIR / fname
    log.info("Loading %s …", path.name)
    pms = pd.read_parquet(path)
    pms["team"] = pms["team"].replace(TEAM_MAP)
    pms["opponent"] = pms["opponent"].replace(TEAM_MAP)
    pms["date"] = pd.to_datetime(pms["date"])
    for c in _RATE_COLS + ["minutes"]:
        if c in pms.columns:
            pms[c] = pd.to_numeric(pms[c], errors="coerce").fillna(0.0)
    return pms.sort_values("date").reset_index(drop=True)


# =============================================================================
# RATE ENGINE — leak-free per-90 priors + projected minutes
# =============================================================================

def build_player_features(pms: pd.DataFrame) -> pd.DataFrame:
    """Attach leak-free per-player priors to every row.

    For each rate column writes ``<col>_p90_prior`` = expanding mean of the
    player's per-90 count over PRIOR matches with minutes >= MIN_MINUTES_FOR_RATE
    (shifted, so the current match is excluded). Also writes ``min_prior`` =
    expanding mean of prior minutes (the leak-free projected-minutes estimate)
    and ``prior_n`` = count of prior 60+min matches.

    The same construction is used at training/eval and prediction time, so a
    player's latest row carries exactly the prior the engine would use for the
    next fixture.
    """
    df = pms.sort_values(["player_id", "date"]).copy()
    is60 = df["minutes"] >= MIN_MINUTES_FOR_RATE
    gid = df["player_id"]

    # Expanding mean over PRIOR 60+min matches only (current row excluded via
    # shift). We mask non-60min rows to NaN so they don't enter the mean, then
    # use transform (guarantees index alignment back to df — the g.apply +
    # column-assign pattern silently reorders and corrupts the priors).
    for col in _RATE_COLS:
        p90 = np.where(df["minutes"] > 0, df[col] / df["minutes"] * 90.0, np.nan)
        masked = pd.Series(np.where(is60.values, p90, np.nan), index=df.index)
        prior = masked.groupby(gid).transform(
            lambda s: s.shift().expanding().mean()
        )
        cap = MAX_P90.get(col)
        if cap is not None:
            prior = prior.clip(upper=cap)
        df[f"{col}_p90_prior"] = prior

    df["min_prior"] = df["minutes"].groupby(gid).transform(
        lambda s: s.shift().expanding().mean()
    )
    # count of prior 60+min matches (current excluded)
    df["prior_n"] = (
        is60.astype(int).groupby(gid).transform(lambda s: s.shift().fillna(0).cumsum())
    )
    return df.sort_values("date").reset_index(drop=True)


def _poisson_tail(lam: float, k: int) -> float:
    """P(N >= k) for N ~ Poisson(lam). k is the threshold (line + 1)."""
    if lam <= 0:
        return 0.0
    # P(N >= k) = 1 - sum_{i=0}^{k-1} e^-lam lam^i / i!
    cdf = 0.0
    term = math.exp(-lam)
    for i in range(k):
        if i > 0:
            term *= lam / i
        cdf += term
    return max(0.0, min(1.0, 1.0 - cdf))


def compute_position_base_rates(pms_feat: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Live fallback rates: P(clear line) by position, over 60+min matches.

    Used when a player has < MIN_PRIOR_MATCHES of history. Computed from the
    data, never hardcoded.
    """
    sub = pms_feat[pms_feat["minutes"] >= MIN_MINUTES_FOR_RATE]
    rates: dict[str, dict[str, float]] = {}
    for key, cfg in TARGETS.items():
        col, line = cfg["col"], cfg["line"]
        hit = (sub[col] >= line + 1)
        by_pos = hit.groupby(sub["position"]).mean()
        rates[key] = {str(p): float(v) for p, v in by_pos.items()}
        rates[key]["_overall"] = float(hit.mean())
    return rates


# =============================================================================
# PREDICTION
# =============================================================================

def _player_market_probs(
    p90_priors: dict[str, float],
    proj_minutes: float,
    position: str,
    prior_n: int,
    base_rates: dict[str, dict[str, float]],
) -> dict[str, dict]:
    """Compute every market's prob for one player from their priors."""
    markets: dict[str, dict] = {}
    proj_minutes = float(max(0.0, min(98.0, proj_minutes)))
    for key, cfg in TARGETS.items():
        col, line, label = cfg["col"], cfg["line"], cfg["label"]
        k = line + 1
        rate = p90_priors.get(col)
        if prior_n >= MIN_PRIOR_MATCHES and rate is not None and not math.isnan(rate):
            lam = rate * proj_minutes / 90.0
            prob = _poisson_tail(lam, k)
            src = "player"
        else:
            # Fallback: position base rate (scaled toward proj minutes for subs)
            pos_rate = base_rates.get(key, {}).get(str(position))
            if pos_rate is None:
                pos_rate = base_rates.get(key, {}).get("_overall", 0.0)
            # base rates are for 60+min; shrink for low projected minutes
            prob = float(pos_rate) * min(1.0, proj_minutes / 75.0)
            src = "position_base"
        prob = max(0.0, min(0.99, prob))
        markets[key] = {
            "label": label,
            "prob": round(prob, 4),
            "odds_implied": round(1.0 / max(prob, 0.01), 2),
            "source": src,
        }
    return markets


def predict_player_markets(
    player_name: str,
    player_id: int,
    team: str,
    opponent: str,
    position: str,
    is_home: bool,
    is_starter: bool = True,
    proj_minutes: float | None = None,
    pms: pd.DataFrame | None = None,
    base_rates: dict | None = None,
) -> dict:
    """Predict all floor markets for a single player in an upcoming match.

    proj_minutes: caller-supplied projected minutes (from lineup_predictions.json
        avg_minutes * start_pct). If None, uses the player's leak-free min_prior,
        or 82 (starter) / 20 (sub) as a last resort.
    """
    if pms is None:
        pms = build_player_features(load_player_data())
    if base_rates is None:
        base_rates = compute_position_base_rates(pms)

    # Resolve player by id; fall back to exact name match (lineup_predictions.json
    # carries names but no ids — SA names match the pms 100%, see plan).
    pdf = pms[pms["player_id"] == player_id].sort_values("date") if player_id not in (None, -1) \
        else pms.iloc[0:0]
    if len(pdf) == 0 and player_name:
        pdf = pms[pms["player_name"] == player_name].sort_values("date")
    if len(pdf) == 0:
        # Unknown player: position base only, default minutes.
        priors: dict[str, float] = {}
        prior_n = 0
        min_prior = 82.0 if is_starter else 20.0
    else:
        latest = pdf.iloc[-1]
        priors = {c: latest.get(f"{c}_p90_prior") for c in _RATE_COLS}
        priors = {c: (float(v) if pd.notna(v) else None) for c, v in priors.items()}
        prior_n = int(latest.get("prior_n", 0) or 0)
        min_prior = float(latest.get("min_prior")) if pd.notna(latest.get("min_prior")) else (
            82.0 if is_starter else 20.0
        )

    if proj_minutes is None:
        proj_minutes = min_prior

    markets = _player_market_probs(priors, proj_minutes, position, prior_n, base_rates)
    return {
        "player_name": player_name,
        "player_id": player_id,
        "team": team,
        "opponent": opponent,
        "position": position,
        "is_home": is_home,
        "proj_minutes": round(float(proj_minutes), 1),
        "prior_matches": prior_n,
        "markets": markets,
    }


def predict_match_players(
    home_team: str,
    away_team: str,
    home_lineup: list[dict],
    away_lineup: list[dict],
    pms: pd.DataFrame | None = None,
    base_rates: dict | None = None,
) -> dict:
    """Predict all player floor markets for both teams in a match.

    Each lineup entry: {"player_name", "player_id", "position",
                        optional "proj_minutes", optional "is_starter"}.
    """
    if pms is None:
        pms = build_player_features(load_player_data())
    if base_rates is None:
        base_rates = compute_position_base_rates(pms)

    def _side(lineup, team, opponent, is_home):
        out = []
        for pl in lineup:
            pred = predict_player_markets(
                player_name=pl.get("player_name") or pl.get("name", ""),
                player_id=int(pl["player_id"]) if pl.get("player_id") is not None else -1,
                team=team,
                opponent=opponent,
                position=pl.get("position", "M"),
                is_home=is_home,
                is_starter=pl.get("is_starter", True),
                proj_minutes=pl.get("proj_minutes"),
                pms=pms,
                base_rates=base_rates,
            )
            if pred and pred.get("markets"):
                out.append(pred)
        return out

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_players": _side(home_lineup, home_team, away_team, True),
        "away_players": _side(away_lineup, away_team, home_team, False),
    }


# =============================================================================
# VALIDATION (leak-free walk-forward Brier vs base rate) — `python … validate`
# =============================================================================

def validate(league: str = "serie_a") -> dict[str, dict]:
    """Leak-free walk-forward Brier per market. Prints + returns the table.

    A market is trustworthy iff its Brier beats the base-rate Brier on held-out
    seasons. This is the gate that decides which markets get displayed/bet.
    """
    from sklearn.metrics import brier_score_loss

    pms = build_player_features(load_player_data(league))
    pms = pms[(pms["prior_n"] >= MIN_PRIOR_MATCHES) & pms["min_prior"].notna()].copy()

    results: dict[str, dict] = {}
    log.info("%-26s %8s %8s %8s  %s", "MARKET", "base", "model", "lift%", "verdict")
    for key, cfg in TARGETS.items():
        col, line = cfg["col"], cfg["line"]
        k = line + 1
        hit = (pms[col] >= k).astype(int).values
        lam = (pms[f"{col}_p90_prior"].fillna(0).values
               * pms["min_prior"].values / 90.0)
        prob = np.array([_poisson_tail(float(lm), k) for lm in lam])
        base = np.full(len(hit), hit.mean())
        b_model = brier_score_loss(hit, np.clip(prob, 0, 1))
        b_base = brier_score_loss(hit, base)
        lift = (b_base - b_model) / b_base * 100
        verdict = "TRUST" if lift > 1.0 else "weak"
        results[key] = {
            "base_brier": round(float(b_base), 4),
            "model_brier": round(float(b_model), 4),
            "lift_pct": round(float(lift), 1),
            "verdict": verdict,
            "n": int(len(hit)),
        }
        log.info("%-26s %8.4f %8.4f %+8.1f  %s",
                 cfg["label"], b_base, b_model, lift, verdict)
    return results


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
    league = sys.argv[2] if len(sys.argv) > 2 else "serie_a"

    if mode == "validate":
        res = validate(league)
        print(json.dumps(res, indent=2))
    elif mode == "predict":
        pms = build_player_features(load_player_data(league))
        latest = pms[pms["minutes"] >= 60].sort_values("date").iloc[-1]
        team, opp = latest["team"], latest["opponent"]
        log.info("Example: %s vs %s", team, opp)
        starters = pms[(pms["team"] == team)
                       & (pms["date"] == latest["date"])
                       & (pms["minutes"] >= 60)]
        base_rates = compute_position_base_rates(pms)
        for _, pl in starters.iterrows():
            pred = predict_player_markets(
                player_name=pl["player_name"], player_id=pl["player_id"],
                team=team, opponent=opp, position=pl["position"],
                is_home=bool(pl["is_home"]), pms=pms, base_rates=base_rates,
            )
            mk = pred["markets"]
            log.info("  %-24s (%s, %.0f'): SoT0.5=%.0f%%  Shots1.5=%.0f%%  Goal=%.0f%%",
                     pl["player_name"], pl["position"], pred["proj_minutes"],
                     mk["sot_o05"]["prob"] * 100,
                     mk["shots_o15"]["prob"] * 100,
                     mk["goalscorer"]["prob"] * 100)
