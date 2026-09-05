"""Goal-process simulator: minute-resolved goal paths for lead / interval markets.

Design step 2 of .plans/every-market-design.md. A time-inhomogeneous bivariate
Poisson process: per side, per minute-bin, rate = k · xG_side · hazard[bin] ·
state_mult[side][score state]. Sampling N paths gives, from ONE sample set,
every market that depends on WHEN a goal happens or on a LEAD existing at any
moment: Vince o quasi (1x sì / 2x sì), Tempi, Minuti, gol in entrambi i tempi,
primo tempo / finale, and the totals tail (Under/over 5.5, 6.5).

Data contract
  data/parsed/goal_timeline.parquet  one row per goal: canonical match_id,
      sofascore_id, season, side (home/away = the scoring side as Sofascore
      records it; own goals INCLUDED on the beneficiary side, verified 99.8%
      vs final scores on 6,330 matches, 2026-09-05), minute, added_time, bin,
      half, goal_type, source_mtime (incidents parquet mtime → watermark).
  data/models/goal_process/profile.json   fitted hazard + state multipliers
  data/models/goal_process/backtest.json  THE live source of skill numbers;
      never quote a number for this engine from a doc.

Bins: 0..44 = 1st-half minutes 1..45, 45 = 1st-half stoppage (45+x'),
46..90 = 2nd-half minutes 46..90, 91 = 2nd-half stoppage (90+x'). Regular
time plus stoppage only, exactly as the bookmaker settles.

Calibration: when the O/U blend's P(over 2.5) is supplied, a scalar k on both
intensities is solved so the simulated P(total ≥ 3) equals it. One clock: the
simulator never disagrees with the money model on the line that is bet.

v1 limits (measured or declared, never hidden): league-level hazard (no
per-team minute profile), no red-card effect, score-state multipliers carry a
selection bias (leading teams are better teams). The backtest gate decides
which markets are served as tier A; everything else is tier B.
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd

from config.settings import DATA_DIR

log = logging.getLogger(__name__)

INCIDENTS_PATH = DATA_DIR / "external" / "sofascore" / "match_incidents.parquet"
MAPPING_PATH = DATA_DIR / "parsed" / "match_id_mapping.parquet"
TIMELINE_PATH = DATA_DIR / "parsed" / "goal_timeline.parquet"
MODEL_DIR = DATA_DIR / "models" / "goal_process"
PROFILE_PATH = MODEL_DIR / "profile.json"
BACKTEST_PATH = MODEL_DIR / "backtest.json"
RARE_PATH = MODEL_DIR / "rare_events.json"
SHOTS_PATH = DATA_DIR / "external" / "sofascore" / "all_shots_with_xg.parquet"

N_BINS = 92
BIN_1H_STOPPAGE = 45
BIN_2H_STOPPAGE = 91
K_BOUNDS = (0.25, 4.0)   # calibration scalar range; outside it the total is not believable
SKILL_GATE = 0.02      # 1 − Brier/Brier_base, same floor as every other model here
N_GATE = 200


# ============================================================ timeline (data) ==
def bin_of(minute: int, added_time: int) -> int:
    minute = int(minute)
    added_time = int(added_time or 0)
    if minute <= 45:
        return BIN_1H_STOPPAGE if (minute == 45 and added_time > 0) else max(0, minute - 1)
    if minute >= 90 and added_time > 0:
        return BIN_2H_STOPPAGE
    return min(90, max(46, minute))


def half_of_bin(b: int) -> int:
    return 1 if b <= BIN_1H_STOPPAGE else 2


def timeline_from_frames(incidents: pd.DataFrame, mapping: pd.DataFrame):
    """(timeline rows for goals, universe of canonical match_ids that have ANY
    incident row — so 0-0 matches are part of the population, not dropped)."""
    extra = [c for c in ("season", "league") if c in mapping.columns]
    mp = mapping[["match_id", "sofascore_id", *extra]].dropna(subset=["sofascore_id"]).copy()
    mp["sofascore_id"] = mp["sofascore_id"].astype("int64")
    inc = incidents.copy()
    inc["sofascore_id"] = inc["match_id"].astype("int64")
    inc = inc.drop(columns=["match_id"]).merge(mp, on="sofascore_id", how="inner")
    uni = inc.drop_duplicates("match_id")[["match_id", *extra]].sort_values("match_id").reset_index(drop=True)
    universe = uni if "league" in extra else uni["match_id"].tolist()
    g = inc[inc["incident_type"] == "goal"].copy()
    g["side"] = np.where(g["is_home"].astype(bool), "home", "away")
    g["added_time"] = g["added_time"].fillna(0).astype(int)
    g["minute"] = g["minute"].astype(int)
    g["bin"] = [bin_of(m, a) for m, a in zip(g["minute"], g["added_time"])]
    g["half"] = [half_of_bin(b) for b in g["bin"]]
    cols = ["match_id", "sofascore_id", "side", "minute", "added_time", "bin", "half", "goal_type"]
    for c in reversed(extra):
        cols.insert(2, c)
    tl = g[cols].sort_values("match_id", kind="stable").reset_index(drop=True)  # source order kept inside a match
    return tl, universe


def build_goal_timeline(force: bool = False):
    """Returns (timeline, rebuilt). Rebuilds only when the incidents parquet is
    newer than the watermark stored IN the timeline (source_mtime column) —
    never compares against the cache's own mtime (CLAUDE.md cache rule)."""
    src_mtime = os.stat(INCIDENTS_PATH).st_mtime
    if TIMELINE_PATH.exists() and not force:
        cached = pd.read_parquet(TIMELINE_PATH)
        wm = cached.attrs.get("source_mtime") if hasattr(cached, "attrs") else None
        if wm is None and "source_mtime" in cached.columns and len(cached):
            wm = float(cached["source_mtime"].iloc[0])
        if wm is not None and wm >= src_mtime:
            return cached, False
    tl, universe = timeline_from_frames(pd.read_parquet(INCIDENTS_PATH), pd.read_parquet(MAPPING_PATH))
    tl["source_mtime"] = src_mtime
    TIMELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tl.to_parquet(TIMELINE_PATH, index=False)
    # the universe (incl. goalless matches) travels beside the timeline
    uni = universe if isinstance(universe, pd.DataFrame) else pd.DataFrame({"match_id": universe})
    uni.to_parquet(TIMELINE_PATH.with_name("goal_timeline_universe.parquet"), index=False)
    return tl, True


def load_universe(league: str | None = "serie_a") -> list:
    """Canonical ids with ANY incident row, one league at a time — the timeline
    holds BOTH leagues (mapping is 7,998 SA + 7,999 EPL) and a profile fitted on
    the mix is neither league's (first fit did exactly that, 2026-09-05)."""
    p = TIMELINE_PATH.with_name("goal_timeline_universe.parquet")
    if not p.exists():
        return []
    u = pd.read_parquet(p)
    if league and "league" in u.columns:
        u = u[u["league"] == league]
    return u["match_id"].tolist()


# ================================================================== paths ==
def _empty_paths(n: int) -> dict:
    return {"goals_h": np.zeros((n, N_BINS), dtype=np.int8), "goals_a": np.zeros((n, N_BINS), dtype=np.int8)}


def _finish_paths(p: dict) -> dict:
    gh, ga = p["goals_h"].astype(np.int32), p["goals_a"].astype(np.int32)
    ch, ca = np.cumsum(gh, axis=1), np.cumsum(ga, axis=1)
    lead = ch - ca
    p["home_final"], p["away_final"] = ch[:, -1], ca[:, -1]
    p["ht_home"], p["ht_away"] = ch[:, BIN_1H_STOPPAGE], ca[:, BIN_1H_STOPPAGE]
    p["max_lead_home"] = np.maximum(lead.max(axis=1), 0)
    p["max_lead_away"] = np.maximum((-lead).max(axis=1), 0)
    p["lead_at_15"] = lead[:, 14]
    any_goal = (gh + ga) > 0
    first = np.where(any_goal.any(axis=1), any_goal.argmax(axis=1), -1)
    p["first_goal_bin"] = first
    rows = np.arange(len(first))
    fh = np.where(first >= 0, gh[rows, np.maximum(first, 0)] > 0, False)
    p["first_goal_side"] = np.where(first < 0, 0, np.where(fh, 1, 2))  # 0 none, 1 home, 2 away
    p["goals_1h"] = ch[:, BIN_1H_STOPPAGE] + ca[:, BIN_1H_STOPPAGE]
    p["goals_2h"] = (p["home_final"] + p["away_final"]) - p["goals_1h"]
    tot = gh + ga
    p["goal_0_15"] = tot[:, 0:15].sum(axis=1) > 0
    p["goal_76_90"] = tot[:, 76:91].sum(axis=1) > 0          # minutes 76..90 (bins 76..90)
    p["goal_2h_stoppage"] = tot[:, BIN_2H_STOPPAGE] > 0
    return p


def paths_from_timeline(tl: pd.DataFrame, match_ids: list) -> dict:
    """The REAL paths, in the same array layout the simulator produces, so one
    market function grades both."""
    idx = {m: i for i, m in enumerate(match_ids)}
    p = _empty_paths(len(match_ids))
    sub = tl[tl["match_id"].isin(idx)]
    for m, side, b in zip(sub["match_id"], sub["side"], sub["bin"]):
        arr = p["goals_h"] if side == "home" else p["goals_a"]
        arr[idx[m], b] += 1
    p["match_id"] = list(match_ids)
    return _finish_paths(p)


# ================================================================= profile ==
def default_profile() -> dict:
    """Flat hazard with the measured Serie A stoppage mass (1H 2.1%, 2H 6.3% of
    goals, 3,388 matches, 2026-09-05) and no score-state effect. Used when no
    fitted profile exists; the fitted one replaces it."""
    w = np.ones(N_BINS)
    w[BIN_1H_STOPPAGE] = 0.0
    w[BIN_2H_STOPPAGE] = 0.0
    w = w / w.sum() * (1 - 0.021 - 0.063)
    w[BIN_1H_STOPPAGE] = 0.021
    w[BIN_2H_STOPPAGE] = 0.063
    return {"hazard": w.tolist(), "state_mult": {"home": {"trail": 1.0, "level": 1.0, "lead": 1.0},
                                                  "away": {"trail": 1.0, "level": 1.0, "lead": 1.0}},
            "total": {"mu": 2.65, "beta": 0.0, "xt_mean": 2.68}, "fitted_on": None, "n_matches": 0}


def fit_profile(tl: pd.DataFrame, universe: list, seasons: list | None = None,
                xg: pd.DataFrame | None = None) -> dict:
    """Hazard = smoothed share of goals per bin; state multipliers = goals per
    exposure-bin in (trail / level / lead) relative to level, per side;
    total = a one-slope regression of real goals on the frame's xG sum.

    Why the total is NOT the xG sum: measured 2026-09-05 on 2023-26 (n=1,135),
    corr(xg_home+xg_away, goals) = 0.06 while corr(xg_home−xg_away, goal diff)
    = 0.37. The frame's Poisson xG carries the RATIO between the sides and
    almost nothing about the TOTAL (std 0.77, range 0.2–7.2, all noise), so
    feeding it as intensity lost to the base rate on every totals market
    (over_2_5 skill −0.068, over_5_5 −0.22). The simulator therefore takes the
    split from xG and the total from this regression (≈ base rate, beta≈0.05),
    and in serving the O/U CatBoost's P(over 2.5) rescales the total again."""
    # `universe` must already be scoped (league, seasons): it is the population,
    # goalless matches included. Intersecting it with the timeline would drop
    # every 0-0 and inflate every base rate, so `seasons` only trims the goals.
    if seasons is not None and "season" in tl.columns:
        tl = tl[tl["season"].isin(seasons)]
    mids = list(universe)
    if not mids:
        return default_profile()
    p = paths_from_timeline(tl, mids)
    tot = (p["goals_h"] + p["goals_a"]).astype(float)
    counts = tot.sum(axis=0)
    # 5-bin moving average inside each regular half; stoppage bins untouched
    sm = counts.copy()
    for lo, hi in ((0, 45), (46, 91)):
        seg = counts[lo:hi]
        k = np.ones(5) / 5
        sm[lo:hi] = np.convolve(np.pad(seg, 2, mode="edge"), k, mode="valid")
    hazard = sm / sm.sum() if sm.sum() > 0 else np.asarray(default_profile()["hazard"])
    # score state before each bin
    ch = np.cumsum(p["goals_h"], axis=1)
    ca = np.cumsum(p["goals_a"], axis=1)
    lead_before = np.zeros_like(ch)
    lead_before[:, 1:] = (ch - ca)[:, :-1]
    mult = {}
    for side, goals, sign in (("home", p["goals_h"], 1), ("away", p["goals_a"], -1)):
        rel = sign * lead_before
        out = {}
        for name, mask in (("trail", rel < 0), ("level", rel == 0), ("lead", rel > 0)):
            expo = (mask * hazard[None, :]).sum()          # hazard-weighted exposure
            out[name] = float(goals[mask].sum() / expo) if expo > 0 else float("nan")
        lvl = out["level"] if out["level"] and out["level"] == out["level"] else 1.0
        mult[side] = {k: (round(v / lvl, 4) if v == v else 1.0) for k, v in out.items()}
        mult[side]["level"] = 1.0
    total = default_profile()["total"]
    if xg is not None and len(xg):
        sub = xg[xg["match_id"].isin(mids)]
        pos = {m: i for i, m in enumerate(mids)}
        rows = [pos[m] for m in sub["match_id"]]
        xt = (sub["poisson_home_xg"] + sub["poisson_away_xg"]).to_numpy(float)
        g = (p["home_final"][rows] + p["away_final"][rows]).astype(float)
        if len(xt) > 50 and xt.std() > 0:
            beta = float(np.cov(xt, g, bias=True)[0, 1] / xt.var())
            total = {"mu": round(float(g.mean()), 4), "beta": round(beta, 4), "xt_mean": round(float(xt.mean()), 4)}
    return {"hazard": hazard.tolist(), "state_mult": mult, "total": total,
            "fitted_on": seasons, "n_matches": len(mids)}


def load_profile() -> dict:
    if PROFILE_PATH.exists():
        return json.loads(PROFILE_PATH.read_text())
    return default_profile()


# =============================================================== simulate ==
def _simulate_k(xg_h: float, xg_a: float, prof: dict, n: int, seed: int, k: float) -> dict:
    rng = np.random.default_rng(seed)
    w = np.asarray(prof["hazard"], dtype=float)
    mh, ma = prof["state_mult"]["home"], prof["state_mult"]["away"]
    p = _empty_paths(n)
    lead = np.zeros(n, dtype=np.int32)
    tot = prof.get("total")
    if tot and (xg_h + xg_a) > 0:
        xt = xg_h + xg_a
        lam_tot = max(0.2, tot["mu"] + tot["beta"] * (xt - tot["xt_mean"]))
        xg_h, xg_a = lam_tot * xg_h / xt, lam_tot * xg_a / xt
    mh_arr = np.array([mh["trail"], mh["level"], mh["lead"]])
    ma_arr = np.array([ma["trail"], ma["level"], ma["lead"]])
    for b in range(N_BINS):
        sh = np.sign(lead) + 1          # 0 trail, 1 level, 2 lead (home view)
        sa = 2 - sh                     # away view is the mirror
        lam_h = k * xg_h * w[b] * mh_arr[sh]
        lam_a = k * xg_a * w[b] * ma_arr[sa]
        gh = rng.poisson(lam_h) if xg_h > 0 else np.zeros(n, dtype=np.int64)
        ga = rng.poisson(lam_a) if xg_a > 0 else np.zeros(n, dtype=np.int64)
        p["goals_h"][:, b] = gh
        p["goals_a"][:, b] = ga
        lead += gh - ga
    return p


def simulate(xg_h: float, xg_a: float, prof: dict | None = None, n: int = 20000, seed: int = 0,
             p_over_2_5: float | None = None) -> dict:
    """Sample n paths. With p_over_2_5, solve the scalar k (bisection, same seed
    each evaluation) so P(total ≥ 3) matches the served O/U number."""
    prof = prof or load_profile()
    k = 1.0
    saturated = False
    achieved = None
    if p_over_2_5 is not None and (xg_h + xg_a) > 0:
        def _over(kk: float) -> float:
            pp = _simulate_k(xg_h, xg_a, prof, n, seed, kk)
            return float(((pp["goals_h"].sum(axis=1) + pp["goals_a"].sum(axis=1)) >= 3).mean())
        lo, hi = K_BOUNDS
        over_lo, over_hi = _over(lo), _over(hi)
        if p_over_2_5 <= over_lo or p_over_2_5 >= over_hi:
            # Outside the physically sensible range (k in K_BOUNDS spans roughly
            # P(over 2.5) 0.08..0.93 at a 2.7-goal total). Pin to the bound and
            # SAY so: a silent pin would serve every lead market off a total
            # the O/U model never asked for.
            k = lo if p_over_2_5 <= over_lo else hi
            achieved = over_lo if k == lo else over_hi
            saturated = True
            log.warning("goal_process calibration saturated: target P(over 2.5)=%.3f outside [%.3f, %.3f] at k in %s",
                        p_over_2_5, over_lo, over_hi, K_BOUNDS)
        else:
            for _ in range(18):
                mid = (lo + hi) / 2
                over = _over(mid)
                if abs(over - p_over_2_5) < 0.003:
                    lo = hi = mid
                    achieved = over
                    break
                if over < p_over_2_5:
                    lo = mid
                else:
                    hi = mid
            k = (lo + hi) / 2
    p = _simulate_k(xg_h, xg_a, prof, n, seed, k)
    p["calibration_k"] = float(k)
    p["calibration_saturated"] = saturated
    p["calibration_target"] = p_over_2_5
    p["calibration_achieved"] = achieved
    return _finish_paths(p)


# ================================================================ markets ==
def market_outcomes(p: dict) -> dict:
    """0/1 arrays per market, one per path. Works on real and simulated paths."""
    hf, af = p["home_final"], p["away_final"]
    hw, aw = hf > af, af > hf
    o = {
        "home_win": hw, "draw": hf == af, "away_win": aw,
        "vince_o_quasi_home_1": hw | (p["max_lead_home"] >= 1),
        "vince_o_quasi_home_2": hw | (p["max_lead_home"] >= 2),
        "vince_o_quasi_away_1": aw | (p["max_lead_away"] >= 1),
        "vince_o_quasi_away_2": aw | (p["max_lead_away"] >= 2),
        "ht_home": p["ht_home"] > p["ht_away"], "ht_draw": p["ht_home"] == p["ht_away"], "ht_away": p["ht_away"] > p["ht_home"],
        "1h_over_0_5": p["goals_1h"] >= 1, "1h_over_1_5": p["goals_1h"] >= 2,
        "2h_over_0_5": p["goals_2h"] >= 1, "2h_over_1_5": p["goals_2h"] >= 2,
        "goal_both_halves": (p["goals_1h"] >= 1) & (p["goals_2h"] >= 1),
        "goal_0_15": p["goal_0_15"], "goal_76_90": p["goal_76_90"], "goal_2h_stoppage": p["goal_2h_stoppage"],
        "lead_15_home": p["lead_at_15"] > 0, "lead_15_level": p["lead_at_15"] == 0, "lead_15_away": p["lead_at_15"] < 0,
        "first_goal_home": p["first_goal_side"] == 1, "first_goal_away": p["first_goal_side"] == 2, "no_goal": p["first_goal_side"] == 0,
        "over_5_5": (hf + af) >= 6, "over_6_5": (hf + af) >= 7,
        "over_2_5": (hf + af) >= 3,
    }
    hd = np.where(hw, 0, np.where(aw, 2, 1))
    ht = np.where(o["ht_home"], 0, np.where(o["ht_away"], 2, 1))
    for i, a in enumerate("HDA"):
        for j, b in enumerate("HDA"):
            o[f"htft_{a}{b}"] = (ht == i) & (hd == j)
    return {k: v.astype(float) for k, v in o.items()}


def market_probs(p: dict) -> dict:
    return {k: float(v.mean()) for k, v in market_outcomes(p).items()}


# Italian names exactly as the bookmaker lists them AND exactly as the Poisson
# artifact rows already served by web/match_markets.py spell them, so a
# simulator row REPLACES the independent-Poisson row for the same bet instead
# of sitting beside it. (group, bet_type, selection, complement_selection)
MARKET_LABELS = {
    "vince_o_quasi_home_1": ("Principali", "Vince o quasi", "Casa 1x sì", None),
    "vince_o_quasi_home_2": ("Principali", "Vince o quasi", "Casa 2x sì", None),
    "vince_o_quasi_away_1": ("Principali", "Vince o quasi", "Ospite 1x sì", None),
    "vince_o_quasi_away_2": ("Principali", "Vince o quasi", "Ospite 2x sì", None),
    "ht_home": ("Tempi", "1° tempo 1x2", "1", None), "ht_draw": ("Tempi", "1° tempo 1x2", "X", None), "ht_away": ("Tempi", "1° tempo 1x2", "2", None),
    "1h_over_0_5": ("Tempi", "1° tempo under/over", "Over 0.5", "Under 0.5"), "1h_over_1_5": ("Tempi", "1° tempo under/over", "Over 1.5", "Under 1.5"),
    "2h_over_0_5": ("Tempi", "2° tempo under/over", "Over 0.5", "Under 0.5"), "2h_over_1_5": ("Tempi", "2° tempo under/over", "Over 1.5", "Under 1.5"),
    "goal_both_halves": ("Goal", "Gol in entrambi i tempi", "Sì", "No"),
    "goal_0_15": ("Minuti", "Gol nei primi 15 minuti", "Sì", "No"),
    "goal_76_90": ("Minuti", "Gol dal 76' al 90'", "Sì", "No"),
    "goal_2h_stoppage": ("Minuti", "Gol nel recupero 2° tempo", "Sì", "No"),
    "lead_15_home": ("Minuti", "Minuti 1x2 (15')", "1", None), "lead_15_level": ("Minuti", "Minuti 1x2 (15')", "X", None), "lead_15_away": ("Minuti", "Minuti 1x2 (15')", "2", None),
    "first_goal_home": ("Goal", "Prima squadra a segnare", "Casa", None), "first_goal_away": ("Goal", "Prima squadra a segnare", "Ospite", None), "no_goal": ("Goal", "Prima squadra a segnare", "Nessuno", None),
    "over_5_5": ("Under/over", "Under/over", "Over 5.5", "Under 5.5"), "over_6_5": ("Under/over", "Under/over", "Over 6.5", "Under 6.5"),
}
for _a in "HDA":
    for _b in "HDA":
        MARKET_LABELS[f"htft_{_a}{_b}"] = ("Tempi", "Primo tempo / Finale", f"{_a}/{_b}", None)
# Markets the simulator prices but does NOT serve: 1x2 finale belongs to the
# ensemble (and the simulator's draw FAILS the gate: skill −0.022), over 2.5 is
# the O/U model's own number fed back in.
NOT_SERVED = {"home_win", "draw", "away_win", "over_2_5"}


def _gate() -> dict:
    return (json.loads(BACKTEST_PATH.read_text()).get("gate") or {}) if BACKTEST_PATH.exists() else {}


def served_rows(xg_h: float, xg_a: float, p_over_2_5: float | None, n: int = 10000, seed: int = 0,
                league: str = "serie_a") -> list:
    """Rows for web/match_markets.py: every labelled market with its tier from
    the live backtest.json (A if the gate passed, else B; no backtest → B).
    Complement selections (Under x.5, No) are 1 − p and inherit the tier.
    Serie A only: the profile and the gate were fitted and measured on Serie A,
    and a tier A on an EPL row would claim a measurement that never happened."""
    if league != "serie_a" or not xg_h or not xg_a or xg_h <= 0 or xg_a <= 0:
        return []
    key = (round(float(xg_h), 3), round(float(xg_a), 3), None if p_over_2_5 is None else round(float(p_over_2_5), 4), n, seed)
    if key in _SERVED_CACHE:
        return _SERVED_CACHE[key]
    prof = load_profile()
    sim = simulate(float(xg_h), float(xg_a), prof, n=n, seed=seed, p_over_2_5=p_over_2_5)
    probs = market_probs(sim)
    gate = _gate()
    rows = []
    for mk, (group, bet_type, sel, comp) in MARKET_LABELS.items():
        if mk in NOT_SERVED:
            continue
        g = gate.get(mk) or {}
        tier = "A" if g.get("passed") else "B"
        base = {"group": group, "bet_type": bet_type, "tier": tier, "source": "goal_process",
                "skill": g.get("skill"), "calibration_k": round(sim["calibration_k"], 3),
                "calibration_saturated": bool(sim["calibration_saturated"])}
        rows.append({**base, "key": mk, "selection": sel, "probability_pct": round(probs[mk] * 100, 1)})
        if comp:
            rows.append({**base, "key": mk + "__not", "selection": comp,
                         "probability_pct": round((1 - probs[mk]) * 100, 1)})
    if len(_SERVED_CACHE) > 256:
        _SERVED_CACHE.clear()
    _SERVED_CACHE[key] = rows
    return rows


_SERVED_CACHE: dict = {}


# ============================================================ rare events ==
# Speciali match: no per-match model exists or is warranted for these; the
# honest number is the league base rate over recent seasons, served as tier C
# with its n. Sources: match_incidents (goals, cards, substitutions) and
# all_shots_with_xg (coordinates: x = 0..100 along the pitch length with the
# attacked goal at x = 0, y = 0..100 across; verified 2026-09-05 by
# reconstructing the parquet's own `distance` column within 1.1 m median).
_PITCH_X, _PITCH_Y = 1.05, 0.68          # metres per unit
_BOX_DEPTH, _BOX_HALF_WIDTH = 16.5, 20.16
RARE_LABELS = {
    "goal_minute_1": ("Speciali match", "Gol nel primo minuto", "Sì"),
    "goal_outside_box": ("Speciali match", "Gol da fuori area", "Sì"),
    "goal_beyond_halfway": ("Speciali match", "Gol da oltre metà campo", "Sì"),
    "goal_from_bench": ("Speciali match", "Gol dalla panchina", "Sì"),
    "own_goal": ("Speciali match", "Autogol", "Sì"),
    "penalty_awarded": ("Speciali match", "Rigore", "Sì"),
    "red_card": ("Speciali match", "Espulsione", "Sì"),
    # VAR (2026-09-05): Sofascore `varDecision` rows, kept by the incidents parser
    # since the same day and backfilled with --var-backfill. incident_class is the
    # ON-FIELD decision under review and confirmed=False means it was OVERTURNED
    # (verified on 100 matches: goalAwarded+False → 0/18 have the goal in the goal
    # list; penaltyNotAwarded+False → 10/12 are followed by a penalty goal).
    "var_any": ("Speciali match", "Intervento VAR", "Sì"),
    "var_goal_cancelled": ("Speciali match", "Gol annullato dal VAR", "Sì"),
    "var_goal_given": ("Speciali match", "Gol convalidato dal VAR", "Sì"),
    "var_penalty": ("Speciali match", "Rigore VAR", "Sì"),
    "var_penalty_cancelled": ("Speciali match", "Rigore annullato dal VAR", "Sì"),
    "var_red_card": ("Speciali match", "Espulsione VAR", "Sì"),
}


def rare_event_rates(incidents: pd.DataFrame, mapping: pd.DataFrame, shots: pd.DataFrame | None,
                     league: str = "serie_a", seasons: list | None = None) -> dict:
    """Share of matches with at least one event, per market, over `seasons`
    (default: every season present). Returns {market: {rate, n_matches, n_events}}."""
    mp = mapping[["match_id", "sofascore_id", "season", "league"]].dropna(subset=["sofascore_id"]).copy()
    mp["sofascore_id"] = mp["sofascore_id"].astype("int64")
    inc = incidents.rename(columns={"match_id": "sofascore_id"}).copy()
    inc["sofascore_id"] = inc["sofascore_id"].astype("int64")
    inc = inc.merge(mp, on="sofascore_id", how="inner")
    inc = inc[inc["league"] == league]
    if seasons is not None:
        inc = inc[inc["season"].isin(seasons)]
    n = int(inc["match_id"].nunique())
    if n == 0:
        return {}
    goals = inc[inc["incident_type"] == "goal"]
    subs = inc[inc["incident_type"] == "substitution"][["match_id", "player_in_id", "minute"]].rename(
        columns={"player_in_id": "player_id", "minute": "sub_minute"})
    bench = goals.merge(subs, on=["match_id", "player_id"], how="inner")
    bench = bench[bench["minute"] >= bench["sub_minute"]]
    var = inc[inc["incident_type"] == "varDecision"]
    if "confirmed" not in var.columns:
        var = var.assign(confirmed=None)
    overturned = var[var["confirmed"] == False]  # noqa: E712 - nullable object column
    hits = {
        "goal_minute_1": goals[goals["minute"] == 1]["match_id"],
        "goal_from_bench": bench["match_id"],
        "own_goal": goals[goals["goal_type"] == "ownGoal"]["match_id"],
        "red_card": inc[inc["card_type"].isin(["red", "yellowRed"])]["match_id"],
    }
    # VAR markets only over the matches the backfill has CHECKED (a VAR row or the
    # var_checked marker); unchecked matches would read as "no VAR" and deflate the rate
    checked = inc[inc["incident_type"].isin(["varDecision", "var_checked"])]["match_id"].unique()
    n_var = int(len(checked))
    var_hits = {
        "var_any": var["match_id"],
        "var_goal_cancelled": overturned[overturned["incident_class"] == "goalAwarded"]["match_id"],
        "var_goal_given": overturned[overturned["incident_class"] == "goalNotAwarded"]["match_id"],
        "var_penalty": overturned[overturned["incident_class"] == "penaltyNotAwarded"]["match_id"],
        "var_penalty_cancelled": overturned[overturned["incident_class"] == "penaltyAwarded"]["match_id"],
        "var_red_card": var[var["incident_class"] == "redCardGiven"]["match_id"],
    }
    if shots is not None and len(shots):
        # all_shots_with_xg is keyed on the Sofascore id (as a string), not the canonical id
        sh = shots.copy()
        sid = pd.to_numeric(sh["match_id"], errors="coerce")
        if sid.notna().all() and not sid.isin(set(inc["match_id"])).any():
            canon = dict(zip(inc["sofascore_id"], inc["match_id"]))
            sh["match_id"] = sid.astype("int64").map(canon)
        sh = sh[sh["match_id"].isin(set(inc["match_id"]))]
        depth = sh["shot_x"] * _PITCH_X
        lateral = (sh["shot_y"] - 50).abs() * _PITCH_Y
        outside = ~((depth <= _BOX_DEPTH) & (lateral <= _BOX_HALF_WIDTH))
        is_goal = sh["is_goal"].astype(bool)
        hits["goal_outside_box"] = sh[is_goal & outside]["match_id"]
        hits["goal_beyond_halfway"] = sh[is_goal & (depth >= 52.5)]["match_id"]
        hits["penalty_awarded"] = sh[sh["is_penalty"].astype(bool)]["match_id"]   # taken, scored or not
    else:
        hits["penalty_awarded"] = goals[goals["goal_type"] == "penalty"]["match_id"]  # scored only
    out = {}
    for k, ids in hits.items():
        m = int(ids.nunique())
        out[k] = {"rate": round(m / n, 4), "n_matches": n, "n_events": m}
    if n_var:
        for k, ids in var_hits.items():
            m = int(ids.nunique())
            out[k] = {"rate": round(m / n_var, 4), "n_matches": n_var, "n_events": m}
    return out


def build_rare_events(seasons_back: int = 3) -> dict:
    """Serie A base rates over the last `seasons_back` complete seasons plus
    the current one, written to rare_events.json with the season list."""
    mp = pd.read_parquet(MAPPING_PATH)
    sa_seasons = sorted(mp[mp["league"] == "serie_a"]["season"].dropna().unique())
    seasons = sa_seasons[-(seasons_back + 1):]
    shots = pd.read_parquet(SHOTS_PATH) if SHOTS_PATH.exists() else None
    rates = rare_event_rates(pd.read_parquet(INCIDENTS_PATH), mp, shots, "serie_a", seasons)
    out = {"generated_at": pd.Timestamp.utcnow().isoformat(), "league": "serie_a",
           "seasons": list(seasons), "rates": rates}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RARE_PATH.write_text(json.dumps(out, indent=2))
    return out


def rare_event_rows(league: str = "serie_a") -> list:
    """Tier C rows (base rate only) for the Speciali match group."""
    if league != "serie_a" or not RARE_PATH.exists():
        return []
    d = json.loads(RARE_PATH.read_text())
    rows = []
    for k, (group, bet_type, sel) in RARE_LABELS.items():
        r = d.get("rates", {}).get(k)
        if not r:
            continue
        if r["n_events"] == 0:   # below the table's resolution: say so on the row, never print 0.0% as a price
            sel = f"{sel} (0 in {r['n_matches']} matches)"
        rows.append({"group": group, "bet_type": bet_type, "selection": sel,
                     "probability_pct": round(r["rate"] * 100, 1), "tier": "C", "source": "rare_event_table",
                     "n_matches": r["n_matches"], "n_events": r["n_events"], "seasons": d.get("seasons")})
    return rows


# =============================================================== backtest ==
def backtest(test_seasons=("2023-2024", "2024-2025", "2025-2026"), n: int = 3000, seed: int = 0) -> dict:
    """Walk-forward by season: profile fitted on seasons BEFORE the test season,
    intensities = the feature frame's pre-match Poisson xG for that match,
    outcomes = the real path. Brier vs the training-season base rate."""
    tl, _ = build_goal_timeline()
    universe = load_universe("serie_a")
    fx = pd.read_parquet(DATA_DIR / "features" / "features_serie_a.parquet",
                         columns=["match_id", "season", "poisson_home_xg", "poisson_away_xg"])
    fx = fx[fx["match_id"].isin(universe)].dropna()
    season_of = dict(zip(fx["match_id"], fx["season"]))
    tl = tl.copy()
    tl["season"] = tl["match_id"].map(season_of)
    uni_season = {m: season_of.get(m) for m in universe}
    per_market: dict = {}
    for ts in test_seasons:
        train_seasons = sorted({s for s in fx["season"].unique() if s < ts})
        train_ids = [m for m, s in uni_season.items() if s in train_seasons]
        test_rows = fx[fx["season"] == ts]
        if len(test_rows) < 20 or not train_ids:
            continue
        prof = fit_profile(tl, train_ids, seasons=None, xg=fx)
        base = market_probs(paths_from_timeline(tl, train_ids))
        real = market_outcomes(paths_from_timeline(tl, test_rows["match_id"].tolist()))
        preds = {k: [] for k in real}
        for i, (_mid, xh, xa) in enumerate(zip(test_rows["match_id"], test_rows["poisson_home_xg"], test_rows["poisson_away_xg"])):
            pr = market_probs(simulate(float(xh), float(xa), prof, n=n, seed=seed + i))
            for k in preds:
                preds[k].append(pr[k])
        for k in real:
            y = real[k]
            ph = np.asarray(preds[k])
            pb = np.full_like(y, base[k])
            acc = per_market.setdefault(k, {"y": [], "p": [], "pb": []})
            acc["y"].append(y)
            acc["p"].append(ph)
            acc["pb"].append(pb)
        log.info("backtest season %s: %d matches, profile on %d", ts, len(test_rows), len(train_ids))
    out = {"generated_at": pd.Timestamp.utcnow().isoformat(), "test_seasons": list(test_seasons),
           "n_sims": n, "gate": {}, "skill_gate": SKILL_GATE, "n_gate": N_GATE}
    for k, acc in per_market.items():
        y = np.concatenate(acc["y"])
        ph = np.concatenate(acc["p"])
        pb = np.concatenate(acc["pb"])
        brier = float(np.mean((ph - y) ** 2))
        brier_base = float(np.mean((pb - y) ** 2))
        skill = float(1 - brier / brier_base) if brier_base > 0 else 0.0
        n_events = int(y.sum())
        out["gate"][k] = {"n": int(len(y)), "n_events": n_events, "base_rate": float(y.mean()),
                          "brier": round(brier, 5), "brier_base": round(brier_base, 5),
                          "skill": round(skill, 4),
                          "passed": bool(skill >= SKILL_GATE and min(n_events, len(y) - n_events) >= N_GATE)}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    BACKTEST_PATH.write_text(json.dumps(out, indent=2))
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build-timeline", action="store_true")
    ap.add_argument("--fit-profile", action="store_true", help="fit on every season and write profile.json")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--rare-events", action="store_true", help="write rare_events.json (Speciali match base rates)")
    ap.add_argument("--sims", type=int, default=3000)
    ap.add_argument("--simulate", nargs=2, type=float, metavar=("XG_HOME", "XG_AWAY"))
    ap.add_argument("--over25", type=float, default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.build_timeline:
        tl, rebuilt = build_goal_timeline(force=True)
        print(f"timeline rows={len(tl)} matches={tl['match_id'].nunique()} universe={len(load_universe())} rebuilt={rebuilt}")
    if a.fit_profile:
        tl, _ = build_goal_timeline()
        fx = pd.read_parquet(DATA_DIR / "features" / "features_serie_a.parquet",
                             columns=["match_id", "poisson_home_xg", "poisson_away_xg"]).dropna()
        prof = fit_profile(tl, load_universe("serie_a"), xg=fx)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        PROFILE_PATH.write_text(json.dumps(prof, indent=1))
        print(json.dumps({"n_matches": prof["n_matches"], "total": prof["total"], "state_mult": prof["state_mult"],
                          "stoppage_mass": [round(prof["hazard"][BIN_1H_STOPPAGE], 4), round(prof["hazard"][BIN_2H_STOPPAGE], 4)]}))
    if a.backtest:
        out = backtest(n=a.sims)
        for k, g in sorted(out["gate"].items(), key=lambda kv: -kv[1]["skill"]):
            print(f"{k:24s} n={g['n']:5d} base={g['base_rate']:.3f} brier={g['brier']:.4f} base_brier={g['brier_base']:.4f} skill={g['skill']:+.4f} {'PASS' if g['passed'] else '-'}")
    if a.rare_events:
        out = build_rare_events()
        print(json.dumps({"seasons": out["seasons"], **{k: v["rate"] for k, v in out["rates"].items()}}))
    if a.simulate:
        for r in served_rows(a.simulate[0], a.simulate[1], a.over25):
            print(f"{r['group']:12s} {r['bet_type']:28s} {r['selection']:14s} {r['probability_pct']:6.1f}%  {r['tier']}  k={r['calibration_k']}")


if __name__ == "__main__":
    main()
