"""Live per-market probability calibration.

The 2017+ backtest showed several markets OVERCLAIM — they say 70% but only ~56%
happens (away-win, O/U 2.5, shot O/U). Calibration fits an isotonic map per market
on HISTORICAL outcomes so the displayed probability matches the real hit rate. Held-
out validated: away-win gap −12% → −2%.

Calibration is honest regardless of market-selection bias — it's measured per-market
on reliability, which can't be gamed by which markets we chose to show. The maps are
computed LIVE from history (no hardcoded shrink factors), cached, and refit if the
data changes.

A market is identified by a key; raw model prob → calibrated prob via that market's
isotonic map. If a market has no fitted map (too few samples), the raw prob is
returned unchanged.
"""
from __future__ import annotations

import numpy as np

_CACHE: dict = {"maps": None, "mtime": 0.0}


def _eh_home_cover(d, line):
    """Raw P(home covers a +line European handicap) per match, from the production
    compute_european_handicap function — so the calibration map is fit on the same
    probability the live API serves."""
    from scripts.betting import extended_markets as EM
    key = f"home_{'+' if line > 0 else ''}{line}"
    out = []
    hxg = d["poisson_home_xg"].values
    axg = d["poisson_away_xg"].values
    for i in range(len(d)):
        try:
            eh = EM.compute_european_handicap(max(.05, hxg[i]), max(.05, axg[i]))
            out.append((eh.get(key, {}).get("home", {}) or {}).get("prob", float("nan")))
        except Exception:
            out.append(float("nan"))
    return np.array(out, dtype=float)


def _build_maps(matches_path):
    """Fit an isotonic calibration map per market from 2017+ history."""
    import pandas as pd
    from sklearn.isotonic import IsotonicRegression
    from scipy.stats import poisson
    from ml.poisson import poisson_1x2

    d = pd.read_parquet(matches_path)
    d = d.dropna(subset=["home_score", "away_score", "poisson_home_xg", "poisson_away_xg"])
    d = d[pd.to_datetime(d["match_date"], errors="coerce") >= "2017-01-01"].reset_index(drop=True)
    if len(d) < 500:
        return {}
    hs = d["home_score"].values.astype(int)
    as_ = d["away_score"].values.astype(int)
    tot = hs + as_
    P = np.array([poisson_1x2(max(.05, d["poisson_home_xg"].iloc[i]),
                              max(.05, d["poisson_away_xg"].iloc[i])) for i in range(len(d))])
    lam = d["poisson_home_xg"].values + d["poisson_away_xg"].values

    # market_key -> (raw_prob_array, outcome_array)
    series = {
        "1x2_home": (P[:, 0], (hs > as_).astype(float)),
        "1x2_draw": (P[:, 1], (hs == as_).astype(float)),
        "1x2_away": (P[:, 2], (as_ > hs).astype(float)),
        "ou_1.5": (1 - poisson.cdf(1, lam), (tot > 1.5).astype(float)),
        "ou_2.5": (1 - poisson.cdf(2, lam), (tot > 2.5).astype(float)),
        "ou_3.5": (1 - poisson.cdf(3, lam), (tot > 3.5).astype(float)),
        "btts": ((1 - np.exp(-d["poisson_home_xg"].values)) * (1 - np.exp(-d["poisson_away_xg"].values)),
                 ((hs > 0) & (as_ > 0)).astype(float)),
        # double chance — sums of the 1X2 Poisson outcomes; backtest showed ECE 0.075
        # (inherits real 1X2 skill but the displayed % was off). Calibrate each leg.
        "dc_1x": (P[:, 0] + P[:, 1], (hs >= as_).astype(float)),
        "dc_x2": (P[:, 1] + P[:, 2], (as_ >= hs).astype(float)),
        "dc_12": (P[:, 0] + P[:, 2], (hs != as_).astype(float)),
        # team totals + multigol — ECE sweep found these miscalibrated (0.066-0.073),
        # held-out validated (fit 2017-23, ECE drops -0.03 to -0.05 on 2024-25 unseen).
        "tt_home_o1.5": (1 - poisson.cdf(1, d["poisson_home_xg"].values), (hs > 1.5).astype(float)),
        "tt_home_o2.5": (1 - poisson.cdf(2, d["poisson_home_xg"].values), (hs > 2.5).astype(float)),
        "tt_away_o1.5": (1 - poisson.cdf(1, d["poisson_away_xg"].values), (as_ > 1.5).astype(float)),
        "tt_away_o2.5": (1 - poisson.cdf(2, d["poisson_away_xg"].values), (as_ > 2.5).astype(float)),
        "multigol_2_3": (1 - poisson.cdf(1, lam) - (1 - poisson.cdf(3, lam)), ((tot >= 2) & (tot <= 3)).astype(float)),
        # european handicap (home outcome at +1/+2 lines) — sweep found home_+1
        # miscalibrated (held-out ECE 0.075 -> 0.034); -1/-2 lines were already fine.
        # raw prob = P(home covers): derived from the same Poisson score grid.
        "eh_home_+1": (_eh_home_cover(d, +1), ((hs - as_ + 1) > 0).astype(float)),
        "eh_home_+2": (_eh_home_cover(d, +2), ((hs - as_ + 2) > 0).astype(float)),
    }
    maps = {}
    for key, (raw, y) in series.items():
        m = np.isfinite(raw) & np.isfinite(y)
        if m.sum() < 300:
            continue
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw[m], y[m])
        maps[key] = iso
    return maps


def _get_maps(matches_path):
    import os
    try:
        mtime = os.path.getmtime(matches_path)
    except OSError:
        return {}
    if _CACHE["maps"] is None or _CACHE["mtime"] != mtime:
        _CACHE["maps"] = _build_maps(matches_path)
        _CACHE["mtime"] = mtime
    return _CACHE["maps"]


def calibrate(market_key: str, raw_prob: float, matches_path) -> float:
    """Return the calibrated probability for a market, or raw if no map exists."""
    if raw_prob is None:
        return raw_prob
    maps = _get_maps(matches_path)
    iso = maps.get(market_key)
    if iso is None:
        return raw_prob
    try:
        return float(np.clip(iso.predict([raw_prob])[0], 0.001, 0.999))
    except Exception:
        return raw_prob


def calibrated_markets() -> set:
    """Which market keys have a calibration map (for UI badging)."""
    maps = _CACHE.get("maps") or {}
    return set(maps.keys())
