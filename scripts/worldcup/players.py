"""World Cup 2026 anytime-goalscorer model (player layer).

Method (the international analog of scripts/betting/player_predictions.py's
empirical-rate construction — no GBM, no fitted weights):

    share_i = decayed player goals / (decayed team credited goals + ALPHA)
    P(anytime in match m) = 1 - exp(-lambda_team(m) * share_i)

where goals come from data/worldcup/international_goalscorers.csv (own goals
excluded — the csv credits them to the beneficiary team with the OPPONENT's
player named), decayed with a ~2-year half-life, and lambda_team comes from
the backtested team engine (predictions.json at generation time; the
walk-forward GLM in backtest mode).

Honesty constraints baked in:
- goalscorers.csv is a SAMPLE, not a ledger: whole matches lack scorer rows
  in 2024-26 (~55-80% goal coverage) and the file lags results by ~2 months.
  Shares are ratio-robust to match-level gaps; totals are not — this module
  never derives team goal totals from it. Squad players with no csv history
  are not shown (coverage gap, e.g. recent debutants) — documented in the UI.
- ALPHA is selected on WC 2018 ONLY; WC 2022 / Euro 2020 / Euro 2024 /
  Copa 2024 are untouched holdouts. Gate mirrors player_predictions.py's
  validate(): display only if Brier beats the equal-share baseline by > 1%.

Run:  python3 -m scripts.worldcup.players --backtest   (writes player_model_metadata.json)
      python3 -m scripts.worldcup.players              (writes player_predictions.json)
Live numbers: data/worldcup/player_model_metadata.json — never quote from docs.
"""

from __future__ import annotations

import argparse
import json
import math
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.worldcup.engine import (
    DATA_DIR,
    canon_team,
    elo_history,
    fit_goal_model,
    load_results,
)

GOALSCORERS_CSV = DATA_DIR / "international_goalscorers.csv"
SQUADS_JSON = DATA_DIR / "squads.json"
PREDICTIONS_JSON = DATA_DIR / "predictions.json"
SIMULATION_JSON = DATA_DIR / "simulation.json"
PLAYER_PREDICTIONS_JSON = DATA_DIR / "player_predictions.json"
PLAYER_METADATA_JSON = DATA_DIR / "player_model_metadata.json"

# Scraped sources (2026-06 scrape; see DATA_CATALOG.md §6b)
SOFA_PARQUET = DATA_DIR / "sofascore_intl_player_stats.parquet"
TM_JSON = DATA_DIR / "transfermarkt_values.json"
CLUBELO_JSON = DATA_DIR / "clubelo_squad_strength.json"
SCRAPE_METADATA_JSON = DATA_DIR / "scrape_model_metadata.json"
TEAM_CONTEXT_JSON = DATA_DIR / "team_context.json"
CONFIRMED_LINEUPS_JSON = DATA_DIR / "confirmed_lineups.json"

# Confirmed-lineup share factors: mechanical minute expectations, not fitted
# parameters (a bench player averages ~25-30 of 90 minutes when used; a
# player outside the matchday squad can only feature if plans change).
LINEUP_STARTER = 1.0
LINEUP_BENCH = 0.35
LINEUP_ABSENT = 0.05

MIN_SHOT_PRIOR_APPS = 3  # min played appearances before a shots-floor estimate
STARTER_DAMP_FLOOR = 0.10  # absent-from-recent-matches dampening floor
RECENT_WINDOW = 5  # team matches defining "recent" minutes/starts

HALF_LIFE_DAYS = 730.0  # ~2 years — international careers turn over fast
ALPHA = 4.0  # shrinkage (decayed-goal units); selected on WC 2018, see metadata
MIN_DECAYED_GOALS = 0.25  # candidate floor: at least a quarter-goal of decayed credit
MIN_SHARE = 0.01
DISPLAY_TOP_N = 5
DISPLAY_MIN_PROB = 0.05
GATE_MIN_LIFT = 0.01  # mirror player_predictions.validate(): trust iff lift > 1%

# Known squad-name -> csv-name breaks (normalized forms), from the squad scout.
NAME_ALIASES = {
    "max arfsten": "maximilian arfsten",
    "jose gimenez": "jose maria gimenez",
}

# (label, csv tournament string, group window) — WC 2018 is the ALPHA
# selection set; the other four are untouched holdouts.
SELECTION = ("World Cup 2018", "FIFA World Cup", "2018-06-14", "2018-06-28")
HOLDOUTS = [
    ("World Cup 2022", "FIFA World Cup", "2022-11-20", "2022-12-02"),
    ("Euro 2020", "UEFA Euro", "2021-06-11", "2021-06-23"),
    ("Euro 2024", "UEFA Euro", "2024-06-14", "2024-06-26"),
    ("Copa América 2024", "Copa América", "2024-06-20", "2024-07-02"),
]


def normalize_name(name: str) -> str:
    """Accent-stripped lowercase form — the csv itself is accent-inconsistent."""
    decomposed = unicodedata.normalize("NFKD", name.strip().lower())
    flat = "".join(c for c in decomposed if not unicodedata.combining(c))
    return NAME_ALIASES.get(flat, flat)


def load_goalscorers(path: Path = GOALSCORERS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["scorer"])
    df = df[~df["own_goal"].astype(bool)]  # OG rows name the opponent's player
    df["norm_scorer"] = df["scorer"].map(normalize_name)
    return df


def load_squads(path: Path = SQUADS_JSON) -> dict[str, list[dict[str, Any]]]:
    with open(path, encoding="utf-8") as f:
        squads: dict[str, list[dict[str, Any]]] = json.load(f)
    return squads


def build_shares(
    scorers: pd.DataFrame,
    team_canon: str,
    as_of: pd.Timestamp,
    *,
    alpha: float = ALPHA,
    half_life_days: float = HALF_LIFE_DAYS,
) -> dict[str, dict[str, float]]:
    """Decayed goal share per (normalized) scorer for one team as of a date.

    Both numerator and denominator come from the same csv sample, so shares
    survive the dataset's match-level coverage gaps far better than totals.
    """
    rows = scorers[(scorers["team"] == team_canon) & (scorers["date"] < as_of)]
    if rows.empty:
        return {}
    age_days = (as_of - rows["date"]).dt.days.to_numpy(dtype=np.float64)
    weights = np.float_power(0.5, age_days / half_life_days)
    frame = pd.DataFrame(
        {"norm_scorer": rows["norm_scorer"].to_numpy(), "w": weights}
    )
    per_player = frame.groupby("norm_scorer")["w"].sum()
    team_total = float(per_player.sum())
    out: dict[str, dict[str, float]] = {}
    for norm_name, decayed in per_player.items():
        share = float(decayed) / (team_total + alpha)
        if float(decayed) >= MIN_DECAYED_GOALS and share >= MIN_SHARE:
            out[str(norm_name)] = {
                "share": share,
                "decayed_goals": float(decayed),
            }
    return out


def p_anytime(lam_team: float, share: float) -> float:
    """P(player scores >= 1) given team goal mean and the player's share."""
    return float(min(0.99, 1.0 - math.exp(-lam_team * share)))


# --------------------------------------------------------- scraped sources --


def norm_sorted(name: str) -> str:
    """Token-sorted normalized name — handles East-Asian name-order reversal
    between sources ('Heung-min Son' vs 'Son Heung-min')."""
    return " ".join(sorted(normalize_name(name).split()))


def load_sofa(path: Path = SOFA_PARQUET) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"].str[:10])
    return df.sort_values(["team", "date"], kind="stable")


def minutes_profile(
    team_df: pd.DataFrame, before: pd.Timestamp | None = None
) -> dict[str, dict[str, float]]:
    """Per-player recent-availability profile from one team's scraped matches:
    minutes/starts over the team's last RECENT_WINDOW matches. Players absent
    from those matches get an implicit zero profile (handled by callers)."""
    sub = team_df if before is None else team_df[team_df["date"] < before]
    if sub.empty:
        return {}
    recent_dates = sorted(sub["date"].unique())[-RECENT_WINDOW:]
    recent = sub[sub["date"].isin(recent_dates)]
    n_matches = len(recent_dates)
    out: dict[str, dict[str, float]] = {}
    for key, grp in recent.groupby("norm_name_sorted"):
        out[str(key)] = {
            "recent_matches": float(n_matches),
            "apps": float((grp["minutes"] > 0).sum()),
            "starts": float(grp["started"].sum()),
            "avg_min": float(grp["minutes"].fillna(0).sum() / n_matches),
        }
    return out


def starter_damp(
    profile: dict[str, float] | None, mode: str = "presence"
) -> float:
    """Multiplicative share dampener in [STARTER_DAMP_FLOOR, 1]. Only ever
    DOWN-weights; a player absent from all recent matches gets the floor.

    mode='minutes'  — proportional to recent average minutes (avg_min/75).
    mode='presence' — penalize only genuine absence (< 2 appearances in the
    recent window); rotated-but-present players keep full share. Deployed
    mode is 'presence': pre-tournament friendlies systematically rotate
    starters, so warm-up minutes under-represent tournament minutes for
    exactly the players most likely to start — see scrape_model_metadata.json
    for both modes' validation numbers.
    """
    if profile is None:
        return STARTER_DAMP_FLOOR
    if mode == "minutes":
        return float(min(1.0, max(STARTER_DAMP_FLOOR, profile["avg_min"] / 75.0)))
    apps = profile.get("apps", 0.0)
    if apps >= 2.0:
        return 1.0
    if apps >= 1.0:
        return 0.6
    return STARTER_DAMP_FLOOR


def shot_rate_profile(
    team_df: pd.DataFrame, before: pd.Timestamp | None = None
) -> dict[str, dict[str, float]]:
    """Per-player international shots-per-90 from played appearances."""
    sub = team_df if before is None else team_df[team_df["date"] < before]
    played = sub[sub["minutes"] > 0]
    out: dict[str, dict[str, float]] = {}
    for key, grp in played.groupby("norm_name_sorted"):
        minutes = float(grp["minutes"].sum())
        if len(grp) < MIN_SHOT_PRIOR_APPS or minutes < 90:
            continue
        out[str(key)] = {
            "shots_p90": float(grp["shots"].fillna(0).sum() * 90.0 / minutes),
            "sot_p90": float(
                grp["shots_on_target"].fillna(0).sum() * 90.0 / minutes
            ),
            "prior_apps": float(len(grp)),
            "avg_min": float(grp["minutes"].mean()),
        }
    return out


def _poisson_tail_ge1(lam: float) -> float:
    return float(min(0.99, 1.0 - math.exp(-max(lam, 0.0))))


def scrape_goal_rows(
    sofa: pd.DataFrame,
    squads: dict[str, list[dict[str, Any]]],
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Convert scraped per-player goals AFTER the goalscorers-csv cutoff into
    pseudo goalscorer rows (one row per goal), so recent warm-up form enters
    the share tables. Keys are bridged through the squad list so they land on
    the same normalize_name(squad name) keys the candidate join uses; scraped
    players who aren't in a 2026 squad are dropped (they can't be displayed
    anyway). Strictly post-cutoff dates make double-counting with the csv
    impossible until the csv catches up — refresh the cutoff when it does."""
    bridge: dict[tuple[str, str], str] = {}
    for team, roster in squads.items():
        for p in roster:
            bridge[(team, norm_sorted(str(p["name"])))] = normalize_name(
                str(p["name"])
            )
    recent = sofa[(sofa["date"] > cutoff) & (sofa["goals"].fillna(0) > 0)]
    rows: list[dict[str, Any]] = []
    for r in recent.itertuples(index=False):
        key = bridge.get((str(r.team), str(r.norm_name_sorted)))
        if key is None:
            continue
        for _ in range(int(r.goals)):
            rows.append(
                {
                    "date": r.date,
                    "team": canon_team(str(r.team)),
                    "norm_scorer": key,
                }
            )
    return pd.DataFrame(rows, columns=["date", "team", "norm_scorer"])


def scorers_with_scrape_merge(
    scorers: pd.DataFrame,
    squads: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    """goalscorers.csv + post-csv scraped goal events (when the scrape exists)."""
    if not SOFA_PARQUET.exists():
        return scorers
    cutoff = scorers["date"].max()
    extra = scrape_goal_rows(load_sofa(), squads, cutoff)
    if extra.empty:
        return scorers
    return pd.concat(
        [scorers[["date", "team", "norm_scorer"]], extra], ignore_index=True
    )


def validate_scrape() -> dict[str, Any]:
    """Walk-forward validation INSIDE the scraped window: each team's first
    matches predict its later ones (eval = matches with >=5 priors).

    Gate A — starter-dampened scorer shares vs the share-only model.
    Gate B — shots>=0.5 floor from intl shots/90 vs a position base rate.
    Outcomes come from the scrape itself (per-player goals/shots), so the
    goalscorers.csv coverage gaps don't contaminate grading.
    """
    sofa = load_sofa()
    scorers = load_goalscorers()
    squads = load_squads()

    squad_keys: dict[str, dict[str, dict[str, Any]]] = {
        team: {norm_sorted(str(p["name"])): p for p in roster}
        for team, roster in squads.items()
    }
    # share tables keyed by sorted-name for joining against the scrape
    shares_sorted: dict[str, dict[str, float]] = {}
    for team, roster in squads.items():
        shares = build_shares(
            scorers, canon_team(team), pd.Timestamp("2026-06-09")
        )
        bridged: dict[str, float] = {}
        for p in roster:
            c = shares.get(normalize_name(str(p["name"])))
            if c is not None:
                bridged[norm_sorted(str(p["name"]))] = c["share"]
        shares_sorted[team] = bridged

    a_base: list[float] = []
    a_damp: list[float] = []
    a_presence: list[float] = []
    a_y: list[float] = []
    b_model: list[float] = []
    b_base_probs: list[float] = []
    b_y: list[float] = []

    # Position base rate for shots>=1, fit on each team's PRIOR matches pooled
    # globally (first half of every team's window).
    prior_pool = []
    for _team, team_df in sofa.groupby("team"):
        dates = sorted(team_df["date"].unique())
        prior_pool.append(team_df[team_df["date"].isin(dates[:5])])
    pool = pd.concat(prior_pool)
    pool = pool[pool["minutes"] > 0]
    pos_base = {
        str(pos): float((grp["shots"].fillna(0) >= 1).mean())
        for pos, grp in pool.groupby("position")
    }
    overall_base = float((pool["shots"].fillna(0) >= 1).mean())

    for team, team_df in sofa.groupby("team"):
        team = str(team)
        dates = sorted(team_df["date"].unique())
        if len(dates) < 6:
            continue
        for eval_date in dates[5:]:
            match_rows = team_df[team_df["date"] == eval_date]
            prior = team_df[team_df["date"] < eval_date]
            lam_team = float(
                prior.groupby("date")["our_score"].first().mean()
            )
            profile = minutes_profile(team_df, before=eval_date)
            rates = shot_rate_profile(team_df, before=eval_date)
            scored = set(
                match_rows[match_rows["goals"].fillna(0) > 0]["norm_name_sorted"]
            )

            # Gate A universe: squad scorer-candidates (deployment universe).
            for key, share in shares_sorted.get(team, {}).items():
                a_base.append(p_anytime(lam_team, share))
                a_damp.append(
                    p_anytime(
                        lam_team,
                        share * starter_damp(profile.get(key), mode="minutes"),
                    )
                )
                a_presence.append(
                    p_anytime(
                        lam_team,
                        share * starter_damp(profile.get(key), mode="presence"),
                    )
                )
                a_y.append(1.0 if key in scored else 0.0)

            # Gate B universe: players who PLAYED in the eval match with a
            # prior rate estimate (rate quality test, conditional on playing).
            played = match_rows[match_rows["minutes"] > 0]
            for row in played.itertuples(index=False):
                key = str(row.norm_name_sorted)
                rate = rates.get(key)
                if rate is None or key not in squad_keys.get(team, {}):
                    continue
                proj_min = profile.get(key, {}).get("avg_min", 60.0)
                lam_shots = rate["shots_p90"] * max(proj_min, 20.0) / 90.0
                b_model.append(_poisson_tail_ge1(lam_shots))
                b_base_probs.append(pos_base.get(str(row.position), overall_base))
                b_y.append(1.0 if (row.shots or 0) >= 1 else 0.0)

    def brier(p: list[float], y: list[float]) -> float:
        pa, ya = np.array(p), np.array(y)
        return float(((pa - ya) ** 2).mean())

    gate_a_lift = 1.0 - brier(a_damp, a_y) / brier(a_base, a_y)
    gate_presence_lift = 1.0 - brier(a_presence, a_y) / brier(a_base, a_y)
    gate_b_lift = 1.0 - brier(b_model, b_y) / brier(b_base_probs, b_y)
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": (
            "walk-forward inside the scraped window (last ~5 matches per team "
            "evaluated, first ~5 as priors); outcomes from the scrape itself"
        ),
        "starter_dampening": {
            "n": len(a_y),
            "actual_rate": round(float(np.mean(a_y)), 4),
            "brier_share_only": round(brier(a_base, a_y), 5),
            "brier_dampened_minutes": round(brier(a_damp, a_y), 5),
            "brier_dampened_presence": round(brier(a_presence, a_y), 5),
            "lift_minutes": round(gate_a_lift, 4),
            "lift": round(gate_presence_lift, 4),
            "deployed_mode": "presence",
            "deployment_note": (
                "minutes-mode validated higher in-window but the eval window "
                "is regular internationals where rotation persists; the WC "
                "deployment context reverses warm-up rotation, so the "
                "presence mode (absence-only penalty) is deployed"
            ),
            "passed": bool(gate_presence_lift > GATE_MIN_LIFT),
        },
        "shots_floor": {
            "n": len(b_y),
            "actual_rate": round(float(np.mean(b_y)), 4),
            "brier_model": round(brier(b_model, b_y), 5),
            "brier_position_base": round(brier(b_base_probs, b_y), 5),
            "lift": round(gate_b_lift, 4),
            "passed": bool(gate_b_lift > GATE_MIN_LIFT),
            "scope_note": "validated conditional on playing >=1 minute",
        },
    }
    SCRAPE_METADATA_JSON.write_text(json.dumps(metadata, indent=2))
    return metadata


# ------------------------------------------------------------------ backtest --


def _evaluate_tournament(
    hist: pd.DataFrame,
    scorers: pd.DataFrame,
    label: str,
    tournament: str,
    start: str,
    end: str,
    *,
    alpha: float,
) -> dict[str, Any] | None:
    """Brier of share-model vs equal-share baseline on one tournament's
    group-stage player-match universe (players with pre-tournament credit)."""
    model = fit_goal_model(hist, train_end=start)
    window = hist[
        (hist["tournament"] == tournament)
        & (hist["date"] >= pd.Timestamp(start))
        & (hist["date"] <= pd.Timestamp(end))
    ]
    if len(window) < 10:
        return None
    as_of = pd.Timestamp(start)

    shares_cache: dict[str, dict[str, dict[str, float]]] = {}
    preds: list[float] = []
    base: list[float] = []
    outcomes: list[float] = []

    for row in window.itertuples(index=False):
        for team, at_home in (
            (row.home_team, not row.neutral),
            (row.away_team, False),
        ):
            if team not in shares_cache:
                shares_cache[team] = build_shares(
                    scorers, team, as_of, alpha=alpha
                )
            candidates = shares_cache[team]
            if not candidates:
                continue
            elo_t = row.elo_home_pre if team == row.home_team else row.elo_away_pre
            elo_o = row.elo_away_pre if team == row.home_team else row.elo_home_pre
            lam = model.lam(elo_t, elo_o, at_home=at_home, major=True)
            match_scorers = set(
                scorers[
                    (scorers["team"] == team)
                    & (scorers["date"] == row.date)
                ]["norm_scorer"]
            )
            eq_share = sum(c["share"] for c in candidates.values()) / len(candidates)
            for norm_name, c in candidates.items():
                preds.append(p_anytime(lam, c["share"]))
                base.append(p_anytime(lam, eq_share))
                outcomes.append(1.0 if norm_name in match_scorers else 0.0)

    p = np.array(preds)
    b = np.array(base)
    y = np.array(outcomes)
    brier = float(((p - y) ** 2).mean())
    brier_base = float(((b - y) ** 2).mean())
    return {
        "label": label,
        "n_player_matches": int(len(y)),
        "actual_rate": round(float(y.mean()), 4),
        "predicted_mean": round(float(p.mean()), 4),
        "brier": round(brier, 5),
        "brier_equal_share": round(brier_base, 5),
        "lift_vs_equal_share": round(1.0 - brier / brier_base, 4),
    }


def run_backtest() -> dict[str, Any]:
    df = load_results()
    hist, _ = elo_history(df)
    scorers = load_goalscorers()

    # ALPHA selection on WC 2018 only.
    selection_runs = {}
    for alpha in (4.0, 8.0, 16.0):
        r = _evaluate_tournament(hist, scorers, *SELECTION, alpha=alpha)
        if r is not None:
            selection_runs[alpha] = r
    # Select by lift, not raw Brier: the candidate floor depends on alpha, so
    # each alpha is scored on a slightly different universe — lift is
    # internally normalized against the same-universe baseline.
    best_alpha = max(
        selection_runs, key=lambda a: selection_runs[a]["lift_vs_equal_share"]
    )

    per_tournament = []
    for holdout in HOLDOUTS:
        r = _evaluate_tournament(hist, scorers, *holdout, alpha=best_alpha)
        if r is not None:
            per_tournament.append(r)

    n_total = sum(r["n_player_matches"] for r in per_tournament)
    pooled_lift = round(
        sum(r["lift_vs_equal_share"] * r["n_player_matches"] for r in per_tournament)
        / n_total,
        4,
    )
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "method": "decayed goal share x team lambda -> 1-exp(-lam*share)",
        "alpha_selected_on": SELECTION[0],
        "alpha_grid": {str(a): selection_runs[a] for a in selection_runs},
        "alpha": best_alpha,
        "half_life_days": HALF_LIFE_DAYS,
        "per_tournament": per_tournament,
        "overall": {
            "n_player_matches": n_total,
            "lift_vs_equal_share": pooled_lift,
            "predicted_mean": round(
                sum(r["predicted_mean"] * r["n_player_matches"] for r in per_tournament)
                / n_total,
                4,
            ),
            "actual_rate": round(
                sum(r["actual_rate"] * r["n_player_matches"] for r in per_tournament)
                / n_total,
                4,
            ),
        },
        "gate": {
            "rule": f"Brier lift over equal-share baseline > {GATE_MIN_LIFT:.0%} on untouched holdouts",
            "passed": bool(pooled_lift > GATE_MIN_LIFT),
        },
        "caveats": (
            "goalscorers.csv is a sample (~55-80% match coverage 2024-26, lags "
            "~2 months); outcomes graded against the same sample. Historical "
            "squads unavailable, so the backtest universe is 'players with "
            "pre-tournament scoring credit' — the deployed display set adds a "
            "current-squad filter on top."
        ),
    }
    PLAYER_METADATA_JSON.write_text(json.dumps(metadata, indent=2))
    return metadata


# ---------------------------------------------------------------- generation --


def _squad_candidates(
    squad: list[dict[str, Any]],
    shares: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Join squad players against the team's share table by normalized name."""
    out = []
    for player in squad:
        norm = normalize_name(str(player["name"]))
        c = shares.get(norm)
        if c is None:
            continue
        out.append(
            {
                "name": player["name"],
                "position": player.get("position", ""),
                "share": c["share"],
                "decayed_goals": c["decayed_goals"],
                "career_goals": int(player.get("goals", 0)),
            }
        )
    return out


def generate(sims_meta_required: bool = True) -> dict[str, Any]:
    with open(PREDICTIONS_JSON, encoding="utf-8") as f:
        team_preds = json.load(f)
    with open(SIMULATION_JSON, encoding="utf-8") as f:
        sim = json.load(f)
    squads = load_squads()
    scorers = load_goalscorers()
    gate: dict[str, Any] = {}
    alpha = ALPHA
    if PLAYER_METADATA_JSON.exists():
        meta = json.loads(PLAYER_METADATA_JSON.read_text())
        gate = meta.get("gate", {})
        alpha = float(meta.get("alpha", ALPHA))
    elif sims_meta_required:
        raise SystemExit("Run --backtest first: player gate metadata missing.")

    # THE GATE IS ENFORCED HERE, not just recorded: a failing backtest ships
    # an empty player layer (API and UI render nothing), with the gate dict
    # kept in the payload so the page can say why.
    if not gate.get("passed", False):
        payload_empty = {
            "generated_at": datetime.now(UTC).isoformat(),
            "method": "GATED OFF — backtest gate failed, no predictions shipped",
            "coverage_note": "",
            "gate": gate,
            "per_match": {},
            "golden_boot": [],
        }
        PLAYER_PREDICTIONS_JSON.write_text(json.dumps(payload_empty, indent=2))
        return payload_empty

    # Scraped context: recent minutes (starter dampening — gate-checked),
    # market values, club strength. All optional: absent file -> absent layer.
    scrape_gates: dict[str, Any] = {}
    if SCRAPE_METADATA_JSON.exists():
        scrape_gates = json.loads(SCRAPE_METADATA_JSON.read_text())
    damp_on = bool(
        scrape_gates.get("starter_dampening", {}).get("passed", False)
    )
    profiles: dict[str, dict[str, dict[str, float]]] = {}
    if damp_on and SOFA_PARQUET.exists():
        sofa = load_sofa()
        profiles = {
            str(team): minutes_profile(team_df)
            for team, team_df in sofa.groupby("team")
        }
    tm_values: dict[str, dict[str, int]] = {}
    if TM_JSON.exists():
        tm_raw = json.loads(TM_JSON.read_text())
        for team, blob in tm_raw.items():
            # skip the _meta provenance key (its 'players' field is a count)
            if isinstance(blob, dict) and isinstance(blob.get("players"), list):
                tm_values[team] = {
                    norm_sorted(str(p["name"])): int(p["value_eur"])
                    for p in blob["players"]
                }

    as_of = pd.Timestamp.now(tz=None).normalize() + pd.Timedelta(days=1)
    # Fold post-csv scraped goals (June warm-ups) into the share source —
    # same validated estimator, fresher data. See scrape_goal_rows.
    scorers_merged = scorers_with_scrape_merge(scorers, squads)
    shares_by_team: dict[str, list[dict[str, Any]]] = {}
    for display_name, squad in squads.items():
        shares = build_shares(
            scorers_merged, canon_team(display_name), as_of, alpha=alpha
        )
        candidates = _squad_candidates(squad, shares)
        team_profile = profiles.get(display_name, {})
        team_tm = tm_values.get(display_name, {})
        for c in candidates:
            key = norm_sorted(str(c["name"]))
            prof = team_profile.get(key)
            c["share_raw"] = c["share"]  # pre-damp, for lineup overrides
            if damp_on:
                c["share"] = c["share"] * starter_damp(prof)
            c["recent"] = (
                {
                    "starts": int(prof["starts"]),
                    "matches": int(prof["recent_matches"]),
                    "avg_min": round(prof["avg_min"], 0),
                }
                if prof is not None
                else None
            )
            value = team_tm.get(key)
            c["value_eur"] = value
        shares_by_team[display_name] = sorted(candidates, key=lambda c: -c["share"])

    lineups: dict[str, Any] = {}
    if CONFIRMED_LINEUPS_JSON.exists():
        lineups = json.loads(CONFIRMED_LINEUPS_JSON.read_text())

    per_match: dict[str, dict[str, Any]] = {}
    team_lambdas: dict[str, list[float]] = {}
    for p in team_preds["predictions"]:
        entry: dict[str, Any] = {}
        lineup = lineups.get(str(p["match_number"]))
        lineup_ok = bool(lineup and lineup.get("confirmed"))
        for side, lam_key in (("home", "home_xg"), ("away", "away_xg")):
            team = p[f"{side}_team"]
            lam = float(p[lam_key])
            team_lambdas.setdefault(team, []).append(lam)
            starters: set[str] = set()
            bench: set[str] = set()
            if lineup_ok and lineup is not None:
                side_lineup = lineup.get(side, {})
                starters = {norm_sorted(n) for n in side_lineup.get("starters", [])}
                bench = {norm_sorted(n) for n in side_lineup.get("bench", [])}
            rows = []
            for c in shares_by_team.get(team, [])[: DISPLAY_TOP_N * 3]:
                share = c["share"]
                confirmed_xi: bool | None = None
                if lineup_ok:
                    # Confirmed lineup is FACT — it overrides presence damping.
                    key = norm_sorted(str(c["name"]))
                    factor = (
                        LINEUP_STARTER if key in starters
                        else LINEUP_BENCH if key in bench
                        else LINEUP_ABSENT
                    )
                    share = c["share_raw"] * factor
                    confirmed_xi = key in starters
                prob = p_anytime(lam, share)
                rows.append(
                    {
                        "name": c["name"],
                        "position": c["position"],
                        "prob": round(prob, 4),
                        "fair_odds": round(1.0 / max(prob, 0.01), 2),
                        "window_goals": round(c["decayed_goals"], 1),
                        "recent": c.get("recent"),
                        "value_eur": c.get("value_eur"),
                        "confirmed_xi": confirmed_xi,
                    }
                )
            rows.sort(key=lambda r: -float(r["prob"]))
            rows = [r for r in rows if r["prob"] >= DISPLAY_MIN_PROB][:DISPLAY_TOP_N]
            entry[side] = rows
        per_match[str(p["match_number"])] = entry

    # Golden-boot watch: E[goals] = share x E[team tournament goals],
    # KO-match scoring approximated by the team's mean group-stage lambda.
    boot: list[dict[str, Any]] = []
    stats_by_team = {t["team"]: t for t in sim["teams"]}
    for team, candidates in shares_by_team.items():
        lams = team_lambdas.get(team)
        s = stats_by_team.get(team)
        if not lams or s is None:
            continue
        # E[KO matches]; SF losers play the third-place match (reach_sf -
        # reach_final mass). KO scoring uses the mean group lambda (approx).
        exp_ko = (
            s["reach_r32"] + s["reach_r16"] + s["reach_qf"]
            + s["reach_sf"] + s["reach_final"]
            + (s["reach_sf"] - s["reach_final"])
        )
        exp_team_goals = sum(lams) + float(np.mean(lams)) * exp_ko
        for c in candidates[:6]:
            boot.append(
                {
                    "name": c["name"],
                    "team": team,
                    "position": c["position"],
                    "expected_goals": round(c["share"] * exp_team_goals, 2),
                    "recent": c.get("recent"),
                    "value_eur": c.get("value_eur"),
                }
            )
    boot.sort(key=lambda r: -float(r["expected_goals"]))
    boot = boot[:20]

    # Team-level context for the UI (display only; club-Elo gated on coverage).
    context: dict[str, Any] = {}
    if TM_JSON.exists():
        tm_raw = json.loads(TM_JSON.read_text())
        for team in squads:
            blob = tm_raw.get(team)
            if isinstance(blob, dict) and blob.get("total_value_eur"):
                context.setdefault(team, {})["squad_value_eur"] = int(
                    blob["total_value_eur"]
                )
    if CLUBELO_JSON.exists():
        ce_raw = json.loads(CLUBELO_JSON.read_text())
        for team in squads:
            blob = ce_raw.get(team)
            if isinstance(blob, dict) and blob.get("coverage_pct", 0) >= 50:
                context.setdefault(team, {})["mean_club_elo"] = round(
                    float(blob["mean_club_elo"])
                )
                context[team]["club_elo_coverage_pct"] = round(
                    float(blob["coverage_pct"])
                )
    TEAM_CONTEXT_JSON.write_text(json.dumps(context, indent=2, ensure_ascii=False))

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "method": (
            "decayed international goal share (2y half-life, shrinkage "
            f"alpha={ALPHA:g}) x backtested team lambda -> 1-exp(-lam*share); "
            "2026 squad-filtered"
            + ("; starter-dampened by recent minutes (gate-passed)" if damp_on else "")
        ),
        "coverage_note": (
            "Scorer history is a sample: the source csv misses whole recent "
            "matches (~55-80% goal coverage 2024-26) and lags ~2 months. "
            "Players without recorded international goals are not shown."
        ),
        "gate": gate,
        "scrape_gates": scrape_gates,
        "per_match": per_match,
        "golden_boot": boot,
    }
    PLAYER_PREDICTIONS_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False)
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--validate-scrape", action="store_true")
    args = parser.parse_args()

    if args.validate_scrape:
        md = validate_scrape()
        for gate_name in ("starter_dampening", "shots_floor"):
            g = md[gate_name]
            print(
                f"{gate_name}: n={g['n']}  actual={g['actual_rate']:.3f}  "
                f"lift={g['lift']:+.1%}  -> {'PASSED' if g['passed'] else 'FAILED'}"
            )
        return

    if args.backtest:
        md = run_backtest()
        print(f"alpha selected: {md['alpha']} (on {md['alpha_selected_on']})")
        print(
            f"{'tournament':<22}{'n':>7}{'pred':>8}{'actual':>8}"
            f"{'brier':>9}{'eq-brier':>9}{'lift':>8}"
        )
        for r in md["per_tournament"]:
            print(
                f"{r['label']:<22}{r['n_player_matches']:>7}"
                f"{r['predicted_mean']:>8.3f}{r['actual_rate']:>8.3f}"
                f"{r['brier']:>9.4f}{r['brier_equal_share']:>9.4f}"
                f"{r['lift_vs_equal_share']:>8.1%}"
            )
        o = md["overall"]
        print(
            f"{'OVERALL':<22}{o['n_player_matches']:>7}"
            f"{o['predicted_mean']:>8.3f}{o['actual_rate']:>8.3f}"
            f"{'':>9}{'':>9}{o['lift_vs_equal_share']:>8.1%}"
        )
        print(
            f"GATE {'PASSED' if md['gate']['passed'] else 'FAILED'}: {md['gate']['rule']}"
        )
    else:
        payload = generate()
        n_matches = len(payload["per_match"])
        print(f"player predictions: {n_matches} matches -> {PLAYER_PREDICTIONS_JSON}")
        print("\nGolden-boot watch (top 10 expected goals):")
        for r in payload["golden_boot"][:10]:
            print(
                f"  {r['name']:<24}{r['team']:<16}{r['expected_goals']:>5.2f}"
            )


if __name__ == "__main__":
    main()
