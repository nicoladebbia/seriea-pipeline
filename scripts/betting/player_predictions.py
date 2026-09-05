"""
Player-Level Floor Prediction Engine for Serie A / EPL
======================================================
Predicts per-player per-match floor markets:
  1. Anytime Goalscorer (P(goals >= 1))
  2. Player Shots O/U 0.5, 1.5, 2.5
  3. Player Shots on Target O/U 0.5, 1.5
  4. Player Fouls Committed O/U 0.5, 1.5
  5. Player Fouls Drawn O/U 0.5
  6. Player Tackles O/U 0.5, 1.5, 2.5
  7. Player Passes Completed O/U 19.5, 29.5, 39.5 — negative binomial tail
     (measured per-player over-dispersion 4.25x vs Poisson; tackles at 1.12
     stay Poisson like fouls)
  8. Player Duels Won O/U 2.5, 4.5 (dispersion 1.29 — Poisson + isotonic)
  9. Player Interceptions O/U 0.5, 1.5 (dispersion 1.07 — Poisson)

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
from scipy import stats as _sps

from config.team_names import normalize_team_safe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
SS_DIR = DATA_DIR / "external" / "sofascore"

def _canon(s: pd.Series) -> pd.Series:
    """Canonicalise a column of team names.

    This was a hand-typed six-entry dict covering Serie A only, which put the
    two documented traps of this repo into one production engine. Team names
    reach here straight from the Sofascore parquets, but the fixture side (and
    therefore every lookup below) uses CANONICAL names, so any club the dict
    did not know became a key nobody queries:

    * "Liverpool FC" and "Liverpool" existed side by side as SEPARATE keys, so
      the possession prior for Liverpool was built from part of its history and
      the 2025-26 rows sat under a key no fixture ever asks for.
    * "Coventry City" meant newly-promoted Coventry had no prior at all, so
      every Coventry player silently lost the possession adjustment.

    ``normalize_team_safe`` is a strict superset of the six entries (verified
    name by name) and knows both active leagues, so it cannot go stale on a
    promotion the way the literal did.

    The reverse risk of a fuzzy normaliser is OVER-collapse: fusing two distinct
    clubs into one key would merge their priors, the mirror of the bug above.
    Audited across all four stats parquets and every team-ish column — the only
    multi-source group is "Liverpool"/"Liverpool FC", which is the intended
    merge. ``test_distinct_clubs_do_not_fuse`` pins the shared-name pairs.

    Mapped over the DISTINCT values rather than per cell: these frames run to
    ~100k rows over a few dozen clubs, and it keeps the collapse set auditable.
    """
    uniq = s.dropna().astype(str).unique()
    lookup = {v: normalize_team_safe(v) for v in uniq}
    return s.map(lambda v: lookup.get(v, v) if isinstance(v, str) else v)

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
    "assists": 1.0,
    "accurate_passes": 95.0,   # p99 = 89
    "tackles": 7.0,            # p99 = 6
    "duels_won": 14.0,         # p99 = 13
    "interceptions": 6.0,      # p99 = 5
}

# ── market definitions ───────────────────────────────────────────────────────
# col: the per-match count column; line: "Over <line>" means count >= line+1;
# k = line + 1 is the threshold for the Poisson tail P(count >= k).
TARGETS = {
    "goalscorer":  {"col": "goals",           "line": 0, "label": "Anytime Goalscorer"},
    # 2026-09-05: same rate engine as goalscorer, same tier C (declared, not
    # measured); priced against player_assists on the per-event feed.
    "assists":     {"col": "assists",         "line": 0, "label": "Anytime Assist"},
    "shots_o05":   {"col": "total_shots",     "line": 0, "label": "Shots Over 0.5"},
    "shots_o15":   {"col": "total_shots",     "line": 1, "label": "Shots Over 1.5"},
    "shots_o25":   {"col": "total_shots",     "line": 2, "label": "Shots Over 2.5"},
    "sot_o05":     {"col": "shots_on_target", "line": 0, "label": "Shots on Target Over 0.5"},
    "sot_o15":     {"col": "shots_on_target", "line": 1, "label": "Shots on Target Over 1.5"},
    "fouls_o05":   {"col": "fouls",           "line": 0, "label": "Fouls Committed Over 0.5"},
    "fouls_o15":   {"col": "fouls",           "line": 1, "label": "Fouls Committed Over 1.5"},
    "fouled_o05":  {"col": "was_fouled",      "line": 0, "label": "Fouls Drawn Over 0.5"},
    "tackles_o05": {"col": "tackles",         "line": 0, "label": "Tackles Over 0.5"},
    "tackles_o15": {"col": "tackles",         "line": 1, "label": "Tackles Over 1.5"},
    "tackles_o25": {"col": "tackles",         "line": 2, "label": "Tackles Over 2.5"},
    "passes_o195": {"col": "accurate_passes", "line": 19, "label": "Passes Completed Over 19.5", "dist": "nbinom"},
    "passes_o295": {"col": "accurate_passes", "line": 29, "label": "Passes Completed Over 29.5", "dist": "nbinom"},
    "passes_o395": {"col": "accurate_passes", "line": 39, "label": "Passes Completed Over 39.5", "dist": "nbinom"},
    "duels_o25":      {"col": "duels_won",     "line": 2, "label": "Duels Won Over 2.5"},
    "duels_o45":      {"col": "duels_won",     "line": 4, "label": "Duels Won Over 4.5"},
    "intercepts_o05": {"col": "interceptions", "line": 0, "label": "Interceptions Over 0.5"},
    "intercepts_o15": {"col": "interceptions", "line": 1, "label": "Interceptions Over 1.5"},
}

# Distinct count columns we build per-90 priors for.
_RATE_COLS = sorted({t["col"] for t in TARGETS.values()})
# Columns whose targets declare an over-dispersed (negative binomial) tail.
_NBINOM_COLS = sorted({t["col"] for t in TARGETS.values() if t.get("dist") == "nbinom"})

# ---- per-interval split (E3, 2026-09-05) --------------------------------------
# A player's expected events in a half = per-90 rate × minutes ON THE PITCH in
# that half × the league's within-match timing share for that stat.
# Measured before building (Serie A 2017-27):
#   • starters average 44.8' in the 1st half and 37.3' in the 2nd (35% are subbed
#     off); subs average 21.4' and enter in the 1st half 2.9% of the time — so the
#     on-pitch split follows from starter/sub status + projected minutes alone.
#   • shots: 46.1% fall in the 1st half. Across 322 players with ≥80 shots the
#     std of a player's own 1st-half share is 0.058 vs 0.056 binomial noise at
#     that n — a player's shot-minute profile carries NO information beyond the
#     league share, so no per-player timing term is fitted (it would be noise).
#   • fouls, tackles, duels, passes, interceptions have no minute stamp in any
#     catalog file → a flat 50/50 timing share, declared on the row
#     (`timing: "flat"`), never validated; only shots / SoT halves are backtested.
_SHOTS_PATH = SS_DIR / "all_shots_with_xg.parquet"
_HALF_SHARE_CACHE: dict = {}


def _get_half_shares() -> dict[str, float]:
    """{col: share of events in the 1st half}. Measured from the shot events for
    total_shots / shots_on_target; every other column is 0.5 (flat, declared)."""
    if _HALF_SHARE_CACHE:
        return _HALF_SHARE_CACHE
    shares = {c: 0.5 for c in _RATE_COLS}
    try:
        sh = pd.read_parquet(_SHOTS_PATH, columns=["time", "shot_type"])
        shares["total_shots"] = float((sh["time"] <= 45).mean())
        on = sh[sh["shot_type"].isin(["goal", "save"])]
        shares["shots_on_target"] = float((on["time"] <= 45).mean())
    except (OSError, KeyError, ValueError):
        pass
    _HALF_SHARE_CACHE.update(shares)
    return _HALF_SHARE_CACHE


def _half_minutes(proj_minutes: float, is_starter: bool) -> tuple[float, float]:
    """Expected minutes on the pitch in (1st half, 2nd half)."""
    m = float(max(0.0, min(98.0, proj_minutes)))
    if is_starter:
        h1 = min(m, 45.0)
        return h1, m - h1
    h2 = min(m, 45.0)
    return m - h2, h2


def _interval_split(lam90: float, k: int, dist: str, r: float | None,
                    proj_minutes: float, is_starter: bool, share_1h: float) -> dict:
    """P(≥k) in each half and P(≥1 in BOTH halves) from a 90-minute expected
    count `lam90` (already scaled to proj_minutes). Halves are independent
    Poisson thinnings of the match process, so 'nei due tempi' is a product."""
    m1, m2 = _half_minutes(proj_minutes, is_starter)
    total = max(m1 + m2, 1e-9)
    # timing weight: a share of 0.461 over 45' means the 1st-half hazard is
    # 0.922× flat and the 2nd-half hazard 1.078× flat
    lam1 = lam90 * (m1 / total) * (2.0 * share_1h)
    lam2 = lam90 * (m2 / total) * (2.0 * (1.0 - share_1h))
    return {
        "1h": round(_count_tail(lam1, k, dist, r), 4),
        "2h": round(_count_tail(lam2, k, dist, r), 4),
        "both": round(_count_tail(lam1, 1, dist, r) * _count_tail(lam2, 1, dist, r), 4) if k == 1 else None,
        "exp_1h": round(lam1, 3), "exp_2h": round(lam2, 3),
        "timing": "measured" if abs(share_1h - 0.5) > 1e-9 else "flat",
    }


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
    pms["team"] = _canon(pms["team"])
    pms["opponent"] = _canon(pms["opponent"])
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


def _count_tail(lam: float, k: int, dist: str = "poisson", r: float | None = None) -> float:
    """P(N >= k) for a count with mean lam. k is the threshold (line + 1).

    dist: "poisson" (default) or "nbinom" with dispersion r (var = lam + lam²/r).
    nbinom without a fitted r falls back to Poisson rather than guessing one.
    """
    if lam <= 0:
        return 0.0
    if dist == "nbinom" and r is not None and r > 0:
        p = r / (r + lam)
        return float(max(0.0, min(1.0, _sps.nbinom.sf(k - 1, r, p))))
    # Poisson: P(N >= k) = 1 - sum_{i=0}^{k-1} e^-lam lam^i / i!  (stable series,
    # no scipy on the hot path)
    cdf = 0.0
    term = math.exp(-lam)
    for i in range(k):
        if i > 0:
            term *= lam / i
        cdf += term
    return max(0.0, min(1.0, 1.0 - cdf))


def fit_dispersion(pms: pd.DataFrame) -> dict[str, float]:
    """Method-of-moments NB dispersion r per over-dispersed column.

    Parameterization: var = m + m²/r (bigger r → closer to Poisson). Per-player
    var/mean over 60+min rows for players with ≥10 such matches; r_i = m_i² /
    (var_i − m_i); column r = median over players, clipped to [1, 50]. If the
    column is near-Poisson overall (median var/mean ≤ 1.1) we return the cap —
    fitting tiny excesses would just amplify noise.
    """
    sub = pms[pms["minutes"] >= MIN_MINUTES_FOR_RATE]
    out: dict[str, float] = {}
    for col in _NBINOM_COLS:
        g = sub.groupby("player_id")[col].agg(["mean", "var", "count"])
        g = g[(g["count"] >= 10) & (g["mean"] > 0) & g["var"].notna()]
        if not len(g):
            out[col] = 50.0
            continue
        if float((g["var"] / g["mean"]).median()) <= 1.1:
            out[col] = 50.0
            continue
        excess = g["var"] - g["mean"]
        ok = excess > 1e-9
        r_i = g.loc[ok, "mean"] ** 2 / excess[ok]
        out[col] = float(np.clip(float(np.median(r_i)), 1.0, 50.0))
    return out


_DISP_CACHE: dict = {"disp": None, "mtime": 0.0}


def _get_dispersion(pms_feat: pd.DataFrame) -> dict[str, float]:
    """Cached dispersion fits, rebuilt only when the data changes."""
    path = SS_DIR / "player_match_stats.parquet"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    if _DISP_CACHE["disp"] is None or _DISP_CACHE["mtime"] != mtime:
        _DISP_CACHE["disp"] = fit_dispersion(pms_feat)
        _DISP_CACHE["mtime"] = mtime
    return _DISP_CACHE["disp"]


# ── possession context — passes adjustment (validated 2026-06-11) ────────────
# lam_passes *= (clip(exp_match_poss / player_poss_ctx, 0.6, 1.6)) ** 0.5.
# Ablation: beta=0.5 beat the raw habit rate on passes_o295/o395 in 8/8
# season cells (delta -0.0002..-0.0019 Brier); beta=1.0 was mixed. Applied to
# accurate_passes only — tackles showed no validated possession link. Any
# missing prior (new team, missing parquet, unknown player) degrades to
# factor 1.0 = the plain validated habit rate.
_POSS_BETA = 0.5
_POSS_RATIO_CLIP = (0.6, 1.6)
_POSS_CACHE: dict = {"key": None, "team": None, "player": None}


def _get_possession(
    pms: pd.DataFrame, league: str = "serie_a",
) -> tuple[dict[str, float] | None, dict[int, float] | None]:
    """(team → latest leak-free possession prior, player_id → latest context).

    Both are expanding means over PRIOR matches only (shifted), mirroring the
    per-90 priors. Returns (None, None) when the team-stats parquet is missing.
    """
    fname = (
        "match_team_stats.parquet"
        if league == "serie_a"
        else "match_team_stats_premier_league.parquet"
    )
    path = SS_DIR / fname
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None, None
    key = (league, mtime)
    if _POSS_CACHE["key"] == key:
        return _POSS_CACHE["team"], _POSS_CACHE["player"]
    mts = pd.read_parquet(path)
    mts = mts[mts["period"] == "ALL"][["match_id", "team", "possession", "date"]].copy()
    mts["team"] = _canon(mts["team"])
    mts["date"] = pd.to_datetime(mts["date"])
    mts = mts.sort_values("date")
    mts["poss_prior"] = mts.groupby("team")["possession"].transform(
        lambda s: s.shift().expanding().mean()
    )
    latest = mts.dropna(subset=["poss_prior"]).drop_duplicates("team", keep="last")
    team_prior = dict(zip(latest["team"], latest["poss_prior"].astype(float)))

    joined = pms.merge(
        mts[["match_id", "team", "possession"]], on=["match_id", "team"], how="inner"
    ).sort_values("date")
    joined["poss_ctx"] = joined.groupby("player_id")["possession"].transform(
        lambda s: s.shift().expanding().mean()
    )
    latest_p = joined.dropna(subset=["poss_ctx"]).drop_duplicates("player_id", keep="last")
    player_ctx = dict(zip(latest_p["player_id"].astype(int), latest_p["poss_ctx"].astype(float)))

    _POSS_CACHE.update(key=key, team=team_prior, player=player_ctx)
    return team_prior, player_ctx


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
# CALIBRATION — light isotonic on the markets the ECE diagnostic flags as off
# =============================================================================
# Measured 2026-06-04: 6/9 floor markets already empirically calibrated (ECE
# <0.03) — fitting isotonic on those adds noise. Only the low "at least one"
# bars over-predict slightly (Poisson zero-inflation: 1-exp(-λ) overstates
# "≥1" when real counts have more zeros than Poisson). We fit maps for ALL
# markets out-of-fold but only APPLY where the raw ECE warrants it; markets
# already OK get an identity-ish map so the round-trip is a no-op.
_CALIB_CACHE: dict = {"maps": None, "mtime": 0.0}
# Markets whose RAW prob the ECE diagnostic flagged (>0.03). Others pass through.
# 2026-06-11: passes (all lines) + low tackles bars flagged — NB fixes dispersion
# but not zero-inflation (tactical benchings / early subs); isotonic mops up.
_CALIBRATE_MARKETS = {
    "shots_o05", "fouls_o05", "fouled_o05",
    "tackles_o05", "tackles_o15",
    "passes_o195", "passes_o295", "passes_o395",
    "duels_o25", "duels_o45", "intercepts_o05",
}


def _build_floor_calibration(pms_feat) -> dict:
    """Fit a per-market isotonic map (raw Poisson prob -> empirical hit rate).

    Out-of-fold by construction: priors are leak-free (prior-only), so the raw
    prob never saw the current match's outcome. Fitted only for the flagged
    markets; returns {market_key: IsotonicRegression}.
    """
    from sklearn.isotonic import IsotonicRegression

    m = pms_feat[(pms_feat["prior_n"] >= MIN_PRIOR_MATCHES)
                 & pms_feat["min_prior"].notna()]
    disp = _get_dispersion(pms_feat)
    maps: dict = {}
    for key in _CALIBRATE_MARKETS:
        cfg = TARGETS[key]
        col, line = cfg["col"], cfg["line"]
        dist = cfg.get("dist", "poisson")
        k = line + 1
        lam = (m[f"{col}_p90_prior"].fillna(0).to_numpy()
               * m["min_prior"].to_numpy() / 90.0)
        raw = np.array([_count_tail(float(x), k, dist, disp.get(col)) for x in lam])
        hit = (m[col] >= k).astype(int).to_numpy()
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw, hit)
        maps[key] = iso
    return maps


def _get_floor_calibration(pms_feat) -> dict:
    """Cached calibration maps, rebuilt only when the data changes."""
    path = SS_DIR / "player_match_stats.parquet"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    if _CALIB_CACHE["maps"] is None or _CALIB_CACHE["mtime"] != mtime:
        _CALIB_CACHE["maps"] = _build_floor_calibration(pms_feat)
        _CALIB_CACHE["mtime"] = mtime
    return _CALIB_CACHE["maps"]


# =============================================================================
# PREDICTION
# =============================================================================

def _player_market_probs(
    p90_priors: dict[str, float],
    proj_minutes: float,
    position: str,
    prior_n: int,
    base_rates: dict[str, dict[str, float]],
    calib: dict | None = None,
    disp: dict[str, float] | None = None,
    is_starter: bool = True,
) -> dict[str, dict]:
    """Compute every market's prob for one player from their priors.

    calib: optional {market_key: IsotonicRegression} from _get_floor_calibration.
    Applied only to player-rate probs on the flagged markets (the low "≥1" bars
    that over-predict). Position-fallback probs aren't calibrated (different
    process; they already shrink for minutes).
    """
    markets: dict[str, dict] = {}
    proj_minutes = float(max(0.0, min(98.0, proj_minutes)))
    for key, cfg in TARGETS.items():
        col, line, label = cfg["col"], cfg["line"], cfg["label"]
        dist = cfg.get("dist", "poisson")
        k = line + 1
        rate = p90_priors.get(col)
        calibrated = False
        split = None
        lam = None
        if prior_n >= MIN_PRIOR_MATCHES and rate is not None and not math.isnan(rate):
            lam = rate * proj_minutes / 90.0
            prob = _count_tail(lam, k, dist, (disp or {}).get(col))
            src = "player"
            if calib and key in calib:
                prob = float(calib[key].predict([prob])[0])
                calibrated = True
            split = _interval_split(lam, k, dist, (disp or {}).get(col), proj_minutes, is_starter,
                                    _get_half_shares().get(col, 0.5))
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
            "calibrated": calibrated,
            "expected": round(lam, 3) if lam is not None else None,   # 90' expected count, feeds contribution %
            "split": split,                                             # None on the position-base fallback
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
    calib: dict | None = None,
    disp: dict[str, float] | None = None,
    poss_factor: float | None = None,
) -> dict:
    """Predict all floor markets for a single player in an upcoming match.

    proj_minutes: caller-supplied projected minutes (from lineup_predictions.json
        avg_minutes * start_pct). If None, uses the player's leak-free min_prior,
        or 82 (starter) / 20 (sub) as a last resort.
    poss_factor: optional possession multiplier for the passes rate (computed by
        predict_match_players from team priors + player context; None = 1.0).
    """
    if pms is None:
        pms = build_player_features(load_player_data())
    if base_rates is None:
        base_rates = compute_position_base_rates(pms)
    if calib is None:
        calib = _get_floor_calibration(pms)
    if disp is None:
        disp = _get_dispersion(pms)

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
    if poss_factor and priors.get("accurate_passes"):
        priors["accurate_passes"] = priors["accurate_passes"] * float(poss_factor)

    markets = _player_market_probs(priors, proj_minutes, position, prior_n, base_rates, calib, disp,
                                   is_starter=is_starter)
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
    league: str = "serie_a",
) -> dict:
    """Predict all player floor markets for both teams in a match.

    Each lineup entry: {"player_name", "player_id", "position",
                        optional "proj_minutes", optional "is_starter"}.
    """
    if pms is None:
        pms = build_player_features(load_player_data(league))
    if base_rates is None:
        base_rates = compute_position_base_rates(pms)
    calib = _get_floor_calibration(pms)
    disp = _get_dispersion(pms)
    team_prior, player_ctx = _get_possession(pms, league)

    def _side(lineup, team, opponent, is_home):
        tp = (team_prior or {}).get(team)
        op = (team_prior or {}).get(opponent)
        exp_poss = 100.0 * tp / (tp + op) if (tp and op) else None
        out = []
        for pl in lineup:
            pid = pl.get("player_id")
            poss_factor = None
            if exp_poss is not None and pid is not None:
                ctx = (player_ctx or {}).get(int(pid))
                if ctx and ctx > 0:
                    ratio = float(np.clip(exp_poss / ctx, *_POSS_RATIO_CLIP))
                    poss_factor = ratio ** _POSS_BETA
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
                calib=calib,
                disp=disp,
                poss_factor=poss_factor,
            )
            if pred and pred.get("markets"):
                pred["lineup"] = pl.get("lineup")
                pred["xi_status"] = pl.get("xi_status")
                pred["start_pct"] = pl.get("start_pct")
                out.append(pred)
        # contribution %: this player's expected 90' count over the side's total,
        # per stat column (shots, fouls, …). Only players with a rate count;
        # a position-base fallback has no expected count and gets None.
        for col in _RATE_COLS:
            keys = [k for k, t in TARGETS.items() if t["col"] == col]
            tot = sum((pl["markets"][keys[0]].get("expected") or 0.0) for pl in out) if keys else 0.0
            for pl in out:
                e = pl["markets"][keys[0]].get("expected") if keys else None
                share = round(100.0 * e / tot, 1) if (e is not None and tot > 0) else None
                for k in keys:
                    pl["markets"][k]["contribution_pct"] = share
        return out

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_players": _side(home_lineup, home_team, away_team, True),
        "away_players": _side(away_lineup, away_team, home_team, False),
    }


# =============================================================================
# VALIDATION of the per-half split (E3) — shots / SoT only, the two stats with
# a minute stamp. `python -m scripts.betting.player_predictions validate-halves`
# =============================================================================

_HALVES_BACKTEST_PATH = DATA_DIR / "models" / "player_floors" / "halves_backtest.json"
_HALVES_MARKETS = ("shots_o05", "shots_o15", "sot_o05")
_HALVES_SKILL_GATE = 0.02
_HALVES_N_GATE = 200


def validate_halves(league: str = "serie_a", test_seasons=("2023-2024", "2024-2025", "2025-2026"),
                    pms: pd.DataFrame | None = None, shots: pd.DataFrame | None = None,
                    out_path: Path | None = None, shares: dict[str, float] | None = None,
                    min_test_rows: int = 50) -> dict:
    """Walk-forward Brier of P(≥k in the 1st half) / P(≥k in the 2nd half) /
    P(≥1 in both) against the base rate of the seasons before the test season.
    Inputs are leak-free: the player's per-90 prior and `min_prior` as the
    projected minutes (what the page would have served), starters only.
    Ground truth: shots per half from all_shots_with_xg (Sofascore match id)."""
    if pms is None:
        pms = build_player_features(load_player_data(league))
    sh = shots if shots is not None else pd.read_parquet(_SHOTS_PATH, columns=["match_id", "player_id", "time", "shot_type"])
    sh = sh.copy()
    sh["match_id"] = pd.to_numeric(sh["match_id"], errors="coerce").astype("Int64")
    sh["on"] = sh["shot_type"].isin(["goal", "save"])
    per = sh.groupby(["match_id", "player_id"]).agg(
        s1=("time", lambda t: int((t <= 45).sum())), s2=("time", lambda t: int((t > 45).sum())),
        o1=("on", lambda o: int((o & (sh.loc[o.index, "time"] <= 45)).sum())),
        o2=("on", lambda o: int((o & (sh.loc[o.index, "time"] > 45)).sum()))).reset_index()
    rows = pms[(pms["is_starter"] == True) & (pms["prior_n"] >= MIN_PRIOR_MATCHES)].copy()  # noqa: E712
    rows = rows.merge(per, on=["match_id", "player_id"], how="left").fillna({"s1": 0, "s2": 0, "o1": 0, "o2": 0})
    shares = shares if shares is not None else _get_half_shares()
    out: dict = {}
    for mk in _HALVES_MARKETS:
        col, k = TARGETS[mk]["col"], TARGETS[mk]["line"] + 1
        a1, a2 = ("s1", "s2") if col == "total_shots" else ("o1", "o2")
        acc = {h: {"y": [], "p": [], "pb": []} for h in ("1h", "2h", "both")}
        for ts in test_seasons:
            train = rows[rows["season"] < ts]
            test = rows[rows["season"] == ts]
            if len(test) < min_test_rows or not len(train):
                continue
            base = {"1h": float((train[a1] >= k).mean()), "2h": float((train[a2] >= k).mean()),
                    "both": float(((train[a1] >= 1) & (train[a2] >= 1)).mean())}
            for _, r in test.iterrows():
                rate = r.get(f"{col}_p90_prior")
                if rate is None or pd.isna(rate):
                    continue
                mins = float(r["min_prior"]) if pd.notna(r.get("min_prior")) else 82.0
                sp = _interval_split(rate * mins / 90.0, k, "poisson", None, mins, True, shares.get(col, 0.5))
                truth = {"1h": float(r[a1] >= k), "2h": float(r[a2] >= k), "both": float((r[a1] >= 1) and (r[a2] >= 1))}
                for h in acc:
                    if sp[h] is None:
                        continue
                    acc[h]["y"].append(truth[h])
                    acc[h]["p"].append(sp[h])
                    acc[h]["pb"].append(base[h])
        for h, a in acc.items():
            if not a["y"]:
                continue
            y, ph, pb = np.asarray(a["y"]), np.asarray(a["p"]), np.asarray(a["pb"])
            brier, bb = float(np.mean((ph - y) ** 2)), float(np.mean((pb - y) ** 2))
            skill = float(1 - brier / bb) if bb > 0 else 0.0
            n_ev = int(y.sum())
            out[f"{mk}:{h}"] = {"n": int(len(y)), "n_events": n_ev, "base_rate": float(y.mean()),
                                "brier": round(brier, 5), "brier_base": round(bb, 5), "skill": round(skill, 4),
                                "passed": bool(skill >= _HALVES_SKILL_GATE and min(n_ev, len(y) - n_ev) >= _HALVES_N_GATE)}
    # market-level gate (what the page reads): every served half passes
    gate = {mk: {"passed": all(out.get(f"{mk}:{h}", {}).get("passed") for h in ("1h", "2h")),
                 "skill": min(out.get(f"{mk}:{h}", {}).get("skill", 0.0) for h in ("1h", "2h"))}
            for mk in _HALVES_MARKETS if f"{mk}:1h" in out}
    result = {"generated_at": pd.Timestamp.utcnow().isoformat(), "league": league, "test_seasons": list(test_seasons),
              "half_shares": {c: round(v, 4) for c, v in shares.items() if c in ("total_shots", "shots_on_target")},
              "per_half": out, "gate": gate}
    out_path = out_path or _HALVES_BACKTEST_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return result


# =============================================================================
# VALIDATION (leak-free walk-forward Brier vs base rate) — `python … validate`
# =============================================================================

def validate(league: str = "serie_a") -> dict[str, dict]:
    """Leak-free walk-forward Brier per market. Prints + returns the table.

    A market is trustworthy iff its Brier beats the base-rate Brier on held-out
    seasons. This is the gate that decides which markets get displayed/bet.
    """
    from sklearn.metrics import brier_score_loss

    pms_all = build_player_features(load_player_data(league))
    disp = fit_dispersion(pms_all)
    pms = pms_all[(pms_all["prior_n"] >= MIN_PRIOR_MATCHES)
                  & pms_all["min_prior"].notna()].copy()
    if disp:
        log.info("NB dispersion fits: %s", {c: round(r, 1) for c, r in disp.items()})

    results: dict[str, dict] = {}
    log.info("%-26s %8s %8s %8s  %s", "MARKET", "base", "model", "lift%", "verdict")
    for key, cfg in TARGETS.items():
        col, line = cfg["col"], cfg["line"]
        dist = cfg.get("dist", "poisson")
        k = line + 1
        hit = (pms[col] >= k).astype(int).values
        lam = (pms[f"{col}_p90_prior"].fillna(0).values
               * pms["min_prior"].values / 90.0)
        prob = np.array([_count_tail(float(lm), k, dist, disp.get(col)) for lm in lam])
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



# =============================================================================
# FIXTURE PRICING HELPERS (moved from web/app.py 2026-09-05)
# =============================================================================
_PLAYER_ENGINE_CACHE: dict = {"pms": None, "base_rates": None, "mtime": 0.0}
_PMS_PATH = DATA_DIR / "external" / "sofascore" / "player_match_stats.parquet"
_LINEUP_PREDICTIONS = DATA_DIR / "upcoming" / "lineup_predictions.json"
_CONFIRMED_LINEUPS = DATA_DIR / "upcoming" / "confirmed_lineups.json"
MIN_CONFIRMED_XI = 7  # a confirmed side with fewer names is a partial parse, not an XI


def player_engine():
    """Cached (pms-with-priors, base_rates). Rebuilds only when the parquet
    changes: building priors over 100k rows is ~1s, never per request."""
    try:
        mtime = _PMS_PATH.stat().st_mtime
    except OSError:
        return None, None
    if _PLAYER_ENGINE_CACHE["pms"] is None or _PLAYER_ENGINE_CACHE["mtime"] != mtime:
        pms = build_player_features(load_player_data("serie_a"))
        _PLAYER_ENGINE_CACHE["pms"] = pms
        _PLAYER_ENGINE_CACHE["base_rates"] = compute_position_base_rates(pms)
        _PLAYER_ENGINE_CACHE["mtime"] = mtime
    return _PLAYER_ENGINE_CACHE["pms"], _PLAYER_ENGINE_CACHE["base_rates"]


def lineup_entries(side_lineup: dict) -> list:
    """Flatten a lineup_predictions side into [{player_name, player_id, position,
    proj_minutes, is_starter}], using start_pct * avg_minutes for projected mins."""
    out = []
    for grp, starter in (("predicted_xi", True), ("bench", False)):
        for p in side_lineup.get(grp, []) or []:
            avg_min = float(p.get("avg_minutes") or (82.0 if starter else 20.0))
            start_pct = float(p.get("start_pct") or (100.0 if starter else 0.0))
            proj_min = avg_min if starter else avg_min * (start_pct / 100.0)
            out.append({
                "player_name": p.get("name") or p.get("player_name", ""),
                "player_id": p.get("player_id"),
                "position": p.get("position", "M"),
                "proj_minutes": proj_min,
                "is_starter": starter,
                # the basis of the row: a PREDICTED XI, with the predictor's own
                # status/start% so a consumer can say "expected, not official"
                "lineup": "predicted",
                "xi_status": p.get("status"),
                "start_pct": start_pct,
            })
    return out


def confirmed_entries(names: list, bench: list, pms) -> list:
    """Entries from an OFFICIAL team sheet (confirmed_lineups.json): starters and
    bench by name, id/position resolved from pms when the name is known there.
    A player not on the sheet gets no row at all — Pašalić was priced from a
    predicted XI while not in the squad (2026-09-05)."""
    latest = pms.sort_values("date").groupby("player_name").tail(1).set_index("player_name") if pms is not None else None
    out = []
    for grp, starter in ((names, True), (bench, False)):
        for name in grp or []:
            row = latest.loc[name] if (latest is not None and name in latest.index) else None
            out.append({
                "player_name": name,
                "player_id": int(row["player_id"]) if row is not None else None,
                "position": str(row["position"]) if row is not None else "M",
                "proj_minutes": None,
                "is_starter": starter,
                "lineup": "confirmed",
                "xi_status": "confirmed",
                "start_pct": 100.0 if starter else 0.0,
            })
    return out


def recent_xi(pms, team: str) -> list:
    """Fallback likely-XI: the team's most-recent match starters from pms
    (names + ids straight from sofascore so they resolve)."""
    tdf = pms[pms["team"] == team]
    if not len(tdf):
        return []
    last_date = tdf["date"].max()
    last = tdf[(tdf["date"] == last_date) & (tdf["minutes"] >= 60)]
    return [{
        "player_name": r["player_name"], "player_id": r["player_id"],
        "position": r["position"], "is_starter": True,
        "proj_minutes": None,  # engine uses the player's leak-free min_prior
        "lineup": "recent", "xi_status": None, "start_pct": None,
    } for _, r in last.iterrows()]


def match_player_floors(match_key: str, home: str, away: str, league: str = "serie_a"):
    """Player floor markets for one fixture, or None when there is no engine
    (parquet missing) or no XI to price. The OFFICIAL sheet from
    confirmed_lineups.json when it holds this match, else the predicted XI from
    lineup_predictions, else the most recent starters. Every row says which
    (`lineup`). Serie A only: the priors are Serie A."""
    if league != "serie_a":
        return None
    pms, base_rates = player_engine()
    if pms is None:
        return None
    try:
        confirmed = (json.loads(_CONFIRMED_LINEUPS.read_text()).get("matches") or {}).get(match_key)
    except (OSError, ValueError, AttributeError):
        confirmed = None
    try:
        lineups = json.loads(_LINEUP_PREDICTIONS.read_text())
    except (OSError, ValueError):
        lineups = {}
    lp = (lineups.get("matches", {}) if isinstance(lineups, dict) else {}).get(match_key)
    if confirmed and len(confirmed.get("home_lineup") or []) >= MIN_CONFIRMED_XI \
            and len(confirmed.get("away_lineup") or []) >= MIN_CONFIRMED_XI:
        home_xi = confirmed_entries(confirmed.get("home_lineup"), confirmed.get("home_bench"), pms)
        away_xi = confirmed_entries(confirmed.get("away_lineup"), confirmed.get("away_bench"), pms)
    elif lp:
        home_xi = lineup_entries(lp.get("home_lineup", {}))
        away_xi = lineup_entries(lp.get("away_lineup", {}))
    else:
        home_xi = recent_xi(pms, home)
        away_xi = recent_xi(pms, away)
    if not home_xi and not away_xi:
        return None
    return predict_match_players(home, away, home_xi, away_xi, pms=pms,
                                 base_rates=base_rates, league=league)

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
    league = sys.argv[2] if len(sys.argv) > 2 else "serie_a"

    if mode == "validate":
        res = validate(league)
        print(json.dumps(res, indent=2))
    elif mode == "validate-halves":
        res = validate_halves(league)
        print(json.dumps({"half_shares": res["half_shares"], "gate": res["gate"]}, indent=1))
        for k, v in res["per_half"].items():
            print(f"{k:16s} n={v['n']:6d} base={v['base_rate']:.3f} brier={v['brier']:.4f} base_brier={v['brier_base']:.4f} skill={v['skill']:+.4f} {'PASS' if v['passed'] else '-'}")
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
