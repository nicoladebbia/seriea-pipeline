"""Per-player scoring/assist rate priors from Sofascore player_match_stats.

Validated walk-forward on held-out 2025-26+ before wiring (2026-09-04):
P(goal) Brier skill +0.073 vs base rate, P(assist) +0.020, calibration flat
per decile once the era moment-match multiplier is applied (c≈0.98 / 0.99).
Two data facts the numbers rest on, both measured that day:

- xG exists only from 2022-23 (0% filled before). Mixing pre-era rows in as
  zeros deflated every prior 4x — the era filter is load-bearing.
- Within the era a NaN xg row means "no shots taken", not "missing": 0 goals
  among 18,496 NaN-xg appearances. So NaN imputes to 0 here, deliberately.

The cache records the parquet's mtime and rebuilds when the source moves —
never a build-once snapshot (the repo's cache-freeze rule).
"""
from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "external" / "sofascore" / "player_match_stats.parquet"
CACHE = ROOT / "data" / "fantacalcio" / "player_rates.json"

ERA_START = "2022-07-01"   # first season with Sofascore xG
SHRINK_MIN = 900.0         # pseudo-minutes toward the league per-90 mean


def _fit_c(lam, hits) -> float:
    """Moment-match multiplier: mean(1 - exp(-c*lam)) == realized rate.

    Plain bisection — the mean is monotone in c, no scipy in the prod path.
    """
    import numpy as np
    lam = np.asarray(lam, dtype=float)
    target = float(np.mean(np.asarray(hits, dtype=float)))
    lo, hi = 0.2, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if float(np.mean(1 - np.exp(-mid * lam))) < target:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 4)


def _build(df: pd.DataFrame) -> dict:
    """Pure aggregation: era frame -> rates payload. Testable without disk."""
    d = df.dropna(subset=["date"]).copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d[(d["date"] >= ERA_START) & (d["minutes"].fillna(0) > 0)]
    for c in ("xg", "goals", "assists"):
        d[c] = d[c].fillna(0.0)
    league_xg90 = float(d.xg.sum() / d.minutes.sum() * 90)
    league_ast90 = float(d.assists.sum() / d.minutes.sum() * 90)
    g = d.groupby("player_name").agg(minutes=("minutes", "sum"),
                                     xg=("xg", "sum"),
                                     assists=("assists", "sum"),
                                     n_app=("minutes", "size"))
    g["xg90"] = (g.xg + league_xg90 * SHRINK_MIN / 90) \
        / (g.minutes + SHRINK_MIN) * 90
    g["ast90"] = (g.assists + league_ast90 * SHRINK_MIN / 90) \
        / (g.minutes + SHRINK_MIN) * 90
    g["min_pa"] = g.minutes / g.n_app
    m = d.merge(g[["xg90", "ast90"]], left_on="player_name", right_index=True)
    c_goal = _fit_c(m.xg90 * m.minutes / 90, (m.goals > 0).astype(int))
    c_assist = _fit_c(m.ast90 * m.minutes / 90, (m.assists > 0).astype(int))
    players = {name: {"xg90": round(float(r.xg90), 4),
                      "ast90": round(float(r.ast90), 4),
                      "min_pa": round(float(r.min_pa), 1),
                      "n_app": int(r.n_app)}
               for name, r in g.iterrows()}
    return {"built_at": datetime.now(UTC).isoformat(),
            "era_start": ERA_START,
            "c_goal": c_goal, "c_assist": c_assist,
            "league_xg90": round(league_xg90, 4),
            "league_ast90": round(league_ast90, 4),
            "players": players}


def load_rates(refresh: bool = False) -> dict | None:
    """Watermark-cached rates; rebuilds when the parquet's mtime moves."""
    try:
        src_mtime = SOURCE.stat().st_mtime
    except OSError:
        return None
    if not refresh:
        try:
            cached = json.loads(CACHE.read_text())
            if cached.get("source_mtime") == src_mtime:
                return cached
        except (OSError, ValueError):
            pass
    try:
        df = pd.read_parquet(SOURCE, columns=["date", "player_name", "minutes",
                                              "xg", "goals", "assists"])
    except (OSError, ValueError):
        return None
    payload = _build(df)
    payload["source_mtime"] = src_mtime
    tmp = CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False))
    tmp.replace(CACHE)
    return payload


def p_from_lam(lam: float) -> float:
    """P(at least one event) under Poisson with per-match rate lam."""
    return round(1.0 - math.exp(-max(lam, 0.0)), 3)


if __name__ == "__main__":
    r = load_rates(refresh=True)
    if r:
        print(f"players={len(r['players'])} c_goal={r['c_goal']} "
              f"c_assist={r['c_assist']} league_xg90={r['league_xg90']}")
