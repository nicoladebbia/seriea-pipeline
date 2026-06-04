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
