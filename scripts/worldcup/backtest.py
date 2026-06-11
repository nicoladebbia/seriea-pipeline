"""Held-out backtest + variant selection for the World Cup engine.

Variants compared (all walk-forward — every fit uses only matches BEFORE the
tournament being evaluated):
- glm_base   : Elo-difference Poisson GLM, no tournament-environment term
- glm_major  : same GLM predicting with the major-final covariate ON
- dc_reg{R}  : Dixon-Coles attack/defense MLE (L2 shrinkage R)
- ens_w{W}   : geometric blend, W on the GLM(major) side, 1-W on DC

Selection happens on the DEV tournaments (WC 2018 + Euro 2020) ONLY; the
reported numbers and the ship gate come from the UNTOUCHED finals
(WC 2022, Euro 2024, Copa América 2024). Group-stage windows only — knockout
scores include extra time in the source data.

Run: python3 -m scripts.worldcup.backtest
Writes: data/worldcup/model_metadata.json — the live source for every
performance number shown anywhere (never quote numbers from docs).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from scripts.worldcup.engine import (
    DATA_DIR,
    PRODUCTION_DC_REG,
    PRODUCTION_WEIGHT_GLM,
    TRAIN_START,
    DCModel,
    GoalModel,
    blend_lambdas,
    elo_history,
    fit_dc_model,
    fit_goal_model,
    load_results,
    one_x_two,
    score_matrix,
    tilt_lambdas,
)

METADATA_JSON = DATA_DIR / "model_metadata.json"
HISTORICAL_ODDS_JSON = DATA_DIR / "historical_odds.json"
CLUBELO_HISTORY_JSON = DATA_DIR / "clubelo_history.json"
SQUAD_STUDY_JSON = DATA_DIR / "squad_strength_study.json"
# Elo points added per unit of coverage-weighted squad-strength z-score.
SQUAD_K_GRID = (0.0, 10.0, 20.0, 35.0, 50.0, 75.0, 100.0)

# (label, tournament string in dataset, group-stage window [start, end])
DEV_TOURNAMENTS: list[tuple[str, str, str, str]] = [
    ("World Cup 2018", "FIFA World Cup", "2018-06-14", "2018-06-28"),
    ("Euro 2020", "UEFA Euro", "2021-06-11", "2021-06-23"),
]
FINAL_TOURNAMENTS: list[tuple[str, str, str, str]] = [
    ("World Cup 2022", "FIFA World Cup", "2022-11-20", "2022-12-02"),
    ("Euro 2024", "UEFA Euro", "2024-06-14", "2024-06-26"),
    ("Copa América 2024", "Copa América", "2024-06-20", "2024-07-02"),
]

DC_REG_GRID = (1.0, 2.0, 5.0)
ENS_WEIGHT_GRID = (0.3, 0.5, 0.7)

PredictFn = Callable[[Any], tuple[float, float]]


def ece_score(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error, equal-width bins, pooled one-vs-rest."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(probs, bins[1:-1])
    total = len(probs)
    err = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        err += mask.sum() / total * abs(probs[mask].mean() - outcomes[mask].mean())
    return float(err)


def _make_predictors(
    hist: pd.DataFrame, start: str
) -> dict[str, PredictFn]:
    """All variant lambda-providers, each fit strictly pre-tournament."""
    glm: GoalModel = fit_goal_model(hist, train_start=TRAIN_START, train_end=start)
    dcs: dict[float, DCModel] = {
        reg: fit_dc_model(hist, train_start=TRAIN_START, train_end=start, reg=reg)
        for reg in DC_REG_GRID
    }

    def glm_fn(row: Any, *, major: bool) -> tuple[float, float]:
        at_home = not row.neutral
        return (
            glm.lam(row.elo_home_pre, row.elo_away_pre, at_home=at_home, major=major),
            glm.lam(row.elo_away_pre, row.elo_home_pre, at_home=False, major=major),
        )

    def dc_fn(row: Any, reg: float) -> tuple[float | None, float | None]:
        at_home = not row.neutral
        return (
            dcs[reg].lam(row.home_team, row.away_team, at_home=at_home),
            dcs[reg].lam(row.away_team, row.home_team, at_home=False),
        )

    predictors: dict[str, PredictFn] = {
        "glm_base": lambda r: glm_fn(r, major=False),
        "glm_major": lambda r: glm_fn(r, major=True),
    }
    for reg in DC_REG_GRID:
        def dc_only(r: Any, _reg: float = reg) -> tuple[float, float]:
            gh, ga = glm_fn(r, major=True)
            dh, da = dc_fn(r, _reg)
            return (dh if dh is not None else gh, da if da is not None else ga)
        predictors[f"dc_reg{reg:g}"] = dc_only
    for w in ENS_WEIGHT_GRID:
        for reg in DC_REG_GRID:
            def ens(r: Any, _w: float = w, _reg: float = reg) -> tuple[float, float]:
                gh, ga = glm_fn(r, major=True)
                dh, da = dc_fn(r, _reg)
                return blend_lambdas(gh, dh, _w), blend_lambdas(ga, da, _w)
            predictors[f"ens_w{w:g}_reg{reg:g}"] = ens
    return predictors


def _evaluate(
    hist: pd.DataFrame,
    label: str,
    tournament: str,
    start: str,
    end: str,
    predict_fn: PredictFn,
    train: pd.DataFrame,
) -> dict[str, Any] | None:
    window = hist[
        (hist["tournament"] == tournament)
        & (hist["date"] >= pd.Timestamp(start))
        & (hist["date"] <= pd.Timestamp(end))
    ]
    if len(window) < 10:
        return None

    base_pool = train[
        (train["date"] >= pd.Timestamp(TRAIN_START))
        & train["neutral"]
        & ~train["tournament"].str.contains("Friendly", case=False)
    ]
    outcomes_pool = np.sign(base_pool["home_score"] - base_pool["away_score"])
    base_rates = np.array(
        [
            (outcomes_pool == 1).mean(),
            (outcomes_pool == 0).mean(),
            (outcomes_pool == -1).mean(),
        ]
    )
    base_over = float(
        ((base_pool["home_score"] + base_pool["away_score"]) > 2.5).mean()
    )

    probs = np.zeros((len(window), 3))
    p_over = np.zeros(len(window))
    y = np.zeros((len(window), 3))
    y_over = np.zeros(len(window))
    elo_pick_correct = np.zeros(len(window))

    for i, row in enumerate(window.itertuples(index=False)):
        lam_h, lam_a = predict_fn(row)
        probs[i] = one_x_two(lam_h, lam_a)
        grid = score_matrix(lam_h, lam_a)
        totals = np.add.outer(np.arange(grid.shape[0]), np.arange(grid.shape[1]))
        p_over[i] = float(grid[totals > 2.5].sum())

        if row.home_score > row.away_score:
            y[i, 0] = 1.0
            elo_pick_correct[i] = float(row.elo_home_pre >= row.elo_away_pre)
        elif row.home_score == row.away_score:
            y[i, 1] = 1.0
        else:
            y[i, 2] = 1.0
            elo_pick_correct[i] = float(row.elo_away_pre > row.elo_home_pre)
        y_over[i] = float(row.home_score + row.away_score > 2.5)

    acc = float((probs.argmax(axis=1) == y.argmax(axis=1)).mean())
    brier = float(((probs - y) ** 2).sum(axis=1).mean())
    brier_base = float(((base_rates[None, :] - y) ** 2).sum(axis=1).mean())
    acc_base = float((y.argmax(axis=1) == int(base_rates.argmax())).mean())
    eps = 1e-12
    logloss = float(-(np.log(np.clip((probs * y).sum(axis=1), eps, 1))).mean())

    brier_over = float(((p_over - y_over) ** 2).mean())
    brier_over_base = float(((base_over - y_over) ** 2).mean())

    return {
        "label": label,
        "n_matches": int(len(window)),
        "accuracy": round(acc, 4),
        "accuracy_base_rate": round(acc_base, 4),
        "accuracy_higher_elo": round(float(elo_pick_correct.mean()), 4),
        "brier": round(brier, 4),
        "brier_base_rate": round(brier_base, 4),
        "skill_score": round(1.0 - brier / brier_base, 4),
        "log_loss": round(logloss, 4),
        "ece_1x2": round(ece_score(probs.ravel(), y.ravel()), 4),
        "ou25_brier": round(brier_over, 4),
        "ou25_brier_base": round(brier_over_base, 4),
        "ou25_skill": round(1.0 - brier_over / brier_over_base, 4),
        "ou25_ece": round(ece_score(p_over, y_over), 4),
    }


def _weighted(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    n = sum(r["n_matches"] for r in rows)
    out: dict[str, Any] = {"n_matches": n}
    for key in keys:
        out[key] = round(sum(r[key] * r["n_matches"] for r in rows) / n, 4)
    return out


_SUMMARY_KEYS = (
    "accuracy",
    "accuracy_base_rate",
    "accuracy_higher_elo",
    "brier",
    "brier_base_rate",
    "skill_score",
    "log_loss",
    "ece_1x2",
    "ou25_skill",
    "ou25_ece",
)


def _load_market_lookup() -> dict[tuple[str, str, str], dict[str, float]]:
    """Historical closing odds keyed by (canon_home, canon_away, date)."""
    if not HISTORICAL_ODDS_JSON.exists():
        return {}
    blob = json.loads(HISTORICAL_ODDS_JSON.read_text())
    out: dict[tuple[str, str, str], dict[str, float]] = {}
    for entry in blob.values():
        for row in entry.get("rows", []):
            out[(row["home"], row["away"], row["date"])] = row["implied"]
    return out


def _market_for(
    lookup: dict[tuple[str, str, str], dict[str, float]], row: Any
) -> dict[str, float] | None:
    date = pd.Timestamp(row.date)
    for delta in (0, 1, -1):
        key = (
            str(row.home_team),
            str(row.away_team),
            (date + pd.Timedelta(days=delta)).strftime("%Y-%m-%d"),
        )
        if key in lookup:
            return lookup[key]
    return None


def _evaluate_blend(
    hist: pd.DataFrame,
    tournament: str,
    start: str,
    end: str,
    predict_fn: PredictFn,
    lookup: dict[tuple[str, str, str], dict[str, float]],
    weight_model: float,
    *,
    use_tilt: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    """1X2 probs + outcomes on the odds-covered subset of one tournament.
    weight_model=1 -> pure model (subset reference); 0 -> pure market."""
    window = hist[
        (hist["tournament"] == tournament)
        & (hist["date"] >= pd.Timestamp(start))
        & (hist["date"] <= pd.Timestamp(end))
    ]
    probs: list[tuple[float, float, float]] = []
    ys: list[tuple[float, float, float]] = []
    for row in window.itertuples(index=False):
        market = _market_for(lookup, row)
        if market is None:
            continue
        lam_h, lam_a = predict_fn(row)
        p_model = one_x_two(lam_h, lam_a)
        p_mkt = (market["home"], market["draw"], market["away"])
        blended = tuple(
            weight_model * a + (1.0 - weight_model) * b
            for a, b in zip(p_model, p_mkt, strict=True)
        )
        if use_tilt:
            t_h, t_a = tilt_lambdas(lam_h, lam_a, blended)  # type: ignore[arg-type]
            blended = one_x_two(t_h, t_a)
        probs.append(blended)  # type: ignore[arg-type]
        if row.home_score > row.away_score:
            ys.append((1.0, 0.0, 0.0))
        elif row.home_score == row.away_score:
            ys.append((0.0, 1.0, 0.0))
        else:
            ys.append((0.0, 0.0, 1.0))
    if not probs:
        return None
    return np.array(probs), np.array(ys)


def _blend_metrics(probs: np.ndarray, ys: np.ndarray) -> dict[str, float]:
    eps = 1e-12
    return {
        "n_matches": int(len(ys)),
        "accuracy": round(float((probs.argmax(1) == ys.argmax(1)).mean()), 4),
        "brier": round(float(((probs - ys) ** 2).sum(1).mean()), 4),
        "log_loss": round(
            float(-(np.log(np.clip((probs * ys).sum(1), eps, 1))).mean()), 4
        ),
        "ece_1x2": round(ece_score(probs.ravel(), ys.ravel()), 4),
    }


def run_blend_validation(
    hist: pd.DataFrame, selected_variant: str
) -> dict[str, Any] | None:
    """Market-blend validation: weight selected on DEV, judged on FINAL.
    Returns None when no historical odds are available."""
    lookup = _load_market_lookup()
    if not lookup:
        return None

    def collect(
        tournaments: list[tuple[str, str, str, str]], w: float, *, tilt: bool
    ) -> dict[str, float] | None:
        all_p, all_y = [], []
        for _label, tournament, start, end in tournaments:
            predictors = _make_predictors(hist, start)
            res = _evaluate_blend(
                hist, tournament, start, end, predictors[selected_variant],
                lookup, w, use_tilt=tilt,
            )
            if res is not None:
                all_p.append(res[0])
                all_y.append(res[1])
        if not all_p:
            return None
        return _blend_metrics(np.concatenate(all_p), np.concatenate(all_y))

    dev_grid: dict[str, dict[str, float]] = {}
    for w in (1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.0):
        m = collect(DEV_TOURNAMENTS, w, tilt=False)
        if m is not None:
            dev_grid[f"w{w:g}"] = m
    if not dev_grid:
        return None
    best_key = min(dev_grid, key=lambda k: dev_grid[k]["brier"])
    best_w = float(best_key[1:])

    final_model = collect(FINAL_TOURNAMENTS, 1.0, tilt=False)
    final_market = collect(FINAL_TOURNAMENTS, 0.0, tilt=False)
    final_blend = collect(FINAL_TOURNAMENTS, best_w, tilt=False)
    final_tilt = collect(FINAL_TOURNAMENTS, best_w, tilt=True)
    if final_model is None or final_blend is None:
        return None

    passed = bool(final_blend["brier"] < final_model["brier"])
    tilt_ok = bool(
        final_tilt is not None
        and final_tilt["brier"] <= final_model["brier"]
    )
    return {
        "weight_model": best_w,
        "selected_on": "DEV (WC 2018 + Euro 2020), odds-covered subset",
        "dev_grid": dev_grid,
        "final_model_subset": final_model,
        "final_market": final_market,
        "final_blend": final_blend,
        "final_blend_tilted": final_tilt,
        "gate": {
            "rule": "blend Brier < model Brier on untouched finals (odds subset); tilt must not regress",
            "passed": passed and tilt_ok,
        },
    }


def run_backtest() -> dict[str, Any]:
    df = load_results()
    hist, _ = elo_history(df)

    # --- DEV phase: every variant on WC 2018 + Euro 2020 ---------------------
    dev_results: dict[str, list[dict[str, Any]]] = {}
    for label, tournament, start, end in DEV_TOURNAMENTS:
        train = hist[hist["date"] < pd.Timestamp(start)]
        predictors = _make_predictors(hist, start)
        for name, fn in predictors.items():
            r = _evaluate(hist, label, tournament, start, end, fn, train)
            if r is not None:
                dev_results.setdefault(name, []).append(r)

    dev_summary = {
        name: _weighted(rows, _SUMMARY_KEYS) for name, rows in dev_results.items()
    }
    selected = min(dev_summary, key=lambda name: dev_summary[name]["brier"])

    # --- FINAL phase: selected variant + glm_base reference on the untouched
    # tournaments ------------------------------------------------------------
    final_rows: dict[str, list[dict[str, Any]]] = {selected: [], "glm_base": []}
    for label, tournament, start, end in FINAL_TOURNAMENTS:
        train = hist[hist["date"] < pd.Timestamp(start)]
        predictors = _make_predictors(hist, start)
        for name in final_rows:
            r = _evaluate(hist, label, tournament, start, end, predictors[name], train)
            if r is not None:
                final_rows[name].append(r)

    overall = _weighted(final_rows[selected], _SUMMARY_KEYS)
    reference = _weighted(final_rows["glm_base"], _SUMMARY_KEYS)

    blend = run_blend_validation(hist, selected)

    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "data_through": str(df["date"].max().date()),
        "train_start": TRAIN_START,
        "holdout_scope": (
            "FINAL = untouched WC 2022 / Euro 2024 / Copa 2024 group stages; "
            "variant selected on DEV (WC 2018 + Euro 2020) only"
        ),
        "selected_variant": selected,
        "dev_summary": dev_summary,
        "overall": overall,
        "reference_glm_base": reference,
        "market_blend": blend,
        "per_tournament": final_rows[selected],
        "gate": {
            "rule": "skill_score > 0 vs neutral-competitive base rate on the untouched finals",
            "passed": bool(overall["skill_score"] > 0),
        },
        "notes": (
            "1X2 skill markets: who-wins family only. Goal-quantity markets "
            "(O/U, BTTS, multigol, exact score) are display-grade unless "
            "ou25_skill > 0 with ece < 0.05 — mirrors the Serie A finding."
        ),
    }
    METADATA_JSON.write_text(json.dumps(metadata, indent=2))
    return metadata


def _squad_adjustments(
    squad_history: dict[str, Any], label: str
) -> dict[str, float]:
    """team -> coverage-weighted z-score of squad mean club Elo, within one
    tournament's participants.

    Teams whose squads have no club-Elo data (non-European leagues) get NO
    adjustment rather than a fabricated one; partially covered squads are
    shrunk toward zero by their coverage share.
    """
    rows = {
        t: v
        for t, v in squad_history.get(label, {}).items()
        if isinstance(v, dict) and not t.startswith("_")
    }
    xs = {
        t: float(v["mean_club_elo"])
        for t, v in rows.items()
        if v.get("mean_club_elo") is not None
    }
    if len(xs) < 8:
        return {}
    vals = np.array(list(xs.values()))
    sd = float(vals.std())
    if sd < 1e-9:
        return {}
    mu = float(vals.mean())
    return {
        t: ((x - mu) / sd) * (float(rows[t].get("coverage_pct", 0.0)) / 100.0)
        for t, x in xs.items()
    }


def run_squad_strength_study() -> dict[str, Any]:
    """Does club-based squad strength add signal on top of international Elo?

    One new dial on a FIXED model family (the production variant: geometric
    ensemble, w=PRODUCTION_WEIGHT_GLM on GLM-major, DC reg=PRODUCTION_DC_REG):
    the GLM leg's Elo inputs become elo_pre + k * z(squad club Elo) * coverage,
    from data/worldcup/clubelo_history.json (scripts.worldcup.squad_history).
    The DC leg is team-name based and unchanged.

    k is selected on the DEV tournaments only; the verdict compares the
    selected k against k=0 (= production) on the UNTOUCHED finals. This
    writes a study file, NOT model metadata — deployment is a separate,
    human decision.
    """
    df = load_results()
    hist, _ = elo_history(df)
    squad_history = json.loads(CLUBELO_HISTORY_JSON.read_text())

    def predictors_for(label: str, start: str) -> dict[float, PredictFn]:
        glm = fit_goal_model(hist, train_start=TRAIN_START, train_end=start)
        dc = fit_dc_model(
            hist, train_start=TRAIN_START, train_end=start, reg=PRODUCTION_DC_REG
        )
        adj = _squad_adjustments(squad_history, label)

        def make(k: float) -> PredictFn:
            def fn(row: Any) -> tuple[float, float]:
                ah = k * adj.get(row.home_team, 0.0)
                aa = k * adj.get(row.away_team, 0.0)
                at_home = not row.neutral
                gh = glm.lam(
                    row.elo_home_pre + ah, row.elo_away_pre + aa,
                    at_home=at_home, major=True,
                )
                ga = glm.lam(
                    row.elo_away_pre + aa, row.elo_home_pre + ah,
                    at_home=False, major=True,
                )
                dh = dc.lam(row.home_team, row.away_team, at_home=at_home)
                da = dc.lam(row.away_team, row.home_team, at_home=False)
                lam_h = blend_lambdas(gh, dh, PRODUCTION_WEIGHT_GLM)
                lam_a = blend_lambdas(ga, da, PRODUCTION_WEIGHT_GLM)
                return lam_h, lam_a

            return fn

        return {k: make(k) for k in SQUAD_K_GRID}

    dev_rows: dict[float, list[dict[str, Any]]] = {k: [] for k in SQUAD_K_GRID}
    adj_stats: dict[str, dict[str, Any]] = {}
    for label, _tournament, _start, _end in DEV_TOURNAMENTS + FINAL_TOURNAMENTS:
        adj = _squad_adjustments(squad_history, label)
        adj_stats[label] = {
            "teams_with_signal": len(adj),
            "mean_abs_z_weighted": round(
                float(np.mean(np.abs(list(adj.values())))), 3
            ) if adj else 0.0,
        }
    for label, tournament, start, end in DEV_TOURNAMENTS:
        train = hist[hist["date"] < pd.Timestamp(start)]
        fns = predictors_for(label, start)
        for k, fn in fns.items():
            r = _evaluate(hist, label, tournament, start, end, fn, train)
            if r is not None:
                dev_rows[k].append(r)
    dev_summary = {
        f"k{k:g}": _weighted(rows, _SUMMARY_KEYS) for k, rows in dev_rows.items()
    }
    k_best = min(SQUAD_K_GRID, key=lambda k: dev_summary[f"k{k:g}"]["brier"])

    final_ks = sorted({k_best, 0.0})
    final_rows: dict[float, list[dict[str, Any]]] = {k: [] for k in final_ks}
    for label, tournament, start, end in FINAL_TOURNAMENTS:
        train = hist[hist["date"] < pd.Timestamp(start)]
        fns = predictors_for(label, start)
        for k in final_rows:
            r = _evaluate(hist, label, tournament, start, end, fns[k], train)
            if r is not None:
                final_rows[k].append(r)

    overall_best = _weighted(final_rows[k_best], _SUMMARY_KEYS)
    overall_prod = _weighted(final_rows[0.0], _SUMMARY_KEYS)
    improves = bool(
        k_best != 0.0 and overall_best["brier"] < overall_prod["brier"]
    )

    study = {
        "generated_at": datetime.now(UTC).isoformat(),
        "data_through": str(df["date"].max().date()),
        "design": (
            "Fixed production variant (ens geometric, "
            f"w={PRODUCTION_WEIGHT_GLM} GLM-major, DC reg={PRODUCTION_DC_REG}); "
            "GLM Elo inputs += k * z(squad mean club Elo) * coverage. "
            "k selected on DEV (WC18+Euro20) by Brier; verdict = selected k "
            "vs k=0 on the untouched finals (WC22+Euro24+Copa24)."
        ),
        "squad_signal": adj_stats,
        "dev_summary": dev_summary,
        "selected_k": k_best,
        "final_selected": overall_best,
        "final_production_k0": overall_prod,
        "final_per_tournament": {
            f"k{k:g}": rows for k, rows in final_rows.items()
        },
        "delta": {
            "brier": round(overall_best["brier"] - overall_prod["brier"], 4),
            "skill_score": round(
                overall_best["skill_score"] - overall_prod["skill_score"], 4
            ),
            "accuracy": round(
                overall_best["accuracy"] - overall_prod["accuracy"], 4
            ),
            "ece_1x2": round(
                overall_best["ece_1x2"] - overall_prod["ece_1x2"], 4
            ),
        },
        "verdict": {
            "improves_on_untouched_finals": improves,
            "note": (
                "Study only — nothing deploys from this file. Deployment "
                "would additionally need a 2026 wiring decision (engine "
                "rating adjustment at predict time) and a re-run of the "
                "full backtest gate."
            ),
        },
    }
    SQUAD_STUDY_JSON.write_text(json.dumps(study, indent=2))
    return study


def _print_squad_study(study: dict[str, Any]) -> None:
    print(f"Squad-strength study · data through {study['data_through']}")
    print("\nDEV k-grid (WC18 + Euro20), sorted by Brier:")
    print(f"{'k':>6}{'n':>5}{'acc':>8}{'brier':>9}{'skill':>8}{'ece':>7}")
    for name, s in sorted(study["dev_summary"].items(), key=lambda kv: kv[1]["brier"]):
        marker = " <== selected" if name == f"k{study['selected_k']:g}" else ""
        print(
            f"{name:>6}{s['n_matches']:>5}{s['accuracy']:>8.3f}"
            f"{s['brier']:>9.4f}{s['skill_score']:>8.3f}{s['ece_1x2']:>7.3f}{marker}"
        )
    b, p = study["final_selected"], study["final_production_k0"]
    print("\nFINAL (untouched WC22 + Euro24 + Copa24):")
    print(f"{'variant':<22}{'n':>5}{'acc':>8}{'brier':>9}{'skill':>8}{'ece':>7}")
    print(
        f"{'k=' + format(study['selected_k'], 'g') + ' (selected)':<22}"
        f"{b['n_matches']:>5}{b['accuracy']:>8.3f}{b['brier']:>9.4f}"
        f"{b['skill_score']:>8.3f}{b['ece_1x2']:>7.3f}"
    )
    print(
        f"{'k=0 (production)':<22}{p['n_matches']:>5}{p['accuracy']:>8.3f}"
        f"{p['brier']:>9.4f}{p['skill_score']:>8.3f}{p['ece_1x2']:>7.3f}"
    )
    d = study["delta"]
    print(
        f"\ndelta (selected - production): brier {d['brier']:+.4f} · "
        f"skill {d['skill_score']:+.4f} · acc {d['accuracy']:+.4f} · "
        f"ece {d['ece_1x2']:+.4f}"
    )
    print(f"improves on untouched finals: "
          f"{study['verdict']['improves_on_untouched_finals']}")
    print(f"wrote {SQUAD_STUDY_JSON}")


def main() -> None:
    if "--squad-study" in sys.argv:
        _print_squad_study(run_squad_strength_study())
        return
    md = run_backtest()
    print(f"Data through: {md['data_through']}")
    print("\nDEV selection (WC 2018 + Euro 2020), sorted by Brier:")
    print(f"{'variant':<18}{'n':>4}{'acc':>8}{'brier':>8}{'skill':>8}{'ece':>7}")
    for name, s in sorted(md["dev_summary"].items(), key=lambda kv: kv[1]["brier"]):
        marker = " <== selected" if name == md["selected_variant"] else ""
        print(
            f"{name:<18}{s['n_matches']:>4}{s['accuracy']:>8.3f}"
            f"{s['brier']:>8.4f}{s['skill_score']:>8.3f}{s['ece_1x2']:>7.3f}{marker}"
        )
    print(f"\nFINAL (untouched: WC22 + Euro24 + Copa24) — {md['selected_variant']}:")
    print(
        f"{'tournament':<22}{'n':>4}{'acc':>8}{'elo':>8}{'brier':>8}{'skill':>8}{'ece':>7}"
    )
    for r in md["per_tournament"]:
        print(
            f"{r['label']:<22}{r['n_matches']:>4}{r['accuracy']:>8.3f}"
            f"{r['accuracy_higher_elo']:>8.3f}{r['brier']:>8.4f}"
            f"{r['skill_score']:>8.3f}{r['ece_1x2']:>7.3f}"
        )
    o, ref = md["overall"], md["reference_glm_base"]
    print(
        f"{'OVERALL (selected)':<22}{o['n_matches']:>4}{o['accuracy']:>8.3f}"
        f"{o['accuracy_higher_elo']:>8.3f}{o['brier']:>8.4f}"
        f"{o['skill_score']:>8.3f}{o['ece_1x2']:>7.3f}"
    )
    print(
        f"{'OVERALL (glm_base)':<22}{ref['n_matches']:>4}{ref['accuracy']:>8.3f}"
        f"{ref['accuracy_higher_elo']:>8.3f}{ref['brier']:>8.4f}"
        f"{ref['skill_score']:>8.3f}{ref['ece_1x2']:>7.3f}"
    )
    print(f"GATE {'PASSED' if md['gate']['passed'] else 'FAILED'}: {md['gate']['rule']}")

    blend = md.get("market_blend")
    if blend:
        print(f"\nMARKET BLEND (w_model={blend['weight_model']:g}, dev-selected):")
        for name in ("final_model_subset", "final_market", "final_blend", "final_blend_tilted"):
            m = blend.get(name)
            if m:
                print(
                    f"  {name:<22} n={m['n_matches']:>3}  acc={m['accuracy']:.3f}  "
                    f"brier={m['brier']:.4f}  ece={m['ece_1x2']:.3f}"
                )
        g = blend["gate"]
        print(f"  BLEND GATE {'PASSED' if g['passed'] else 'FAILED'}: {g['rule']}")
    else:
        print("\nMARKET BLEND: no historical odds available — not evaluated")


if __name__ == "__main__":
    main()
