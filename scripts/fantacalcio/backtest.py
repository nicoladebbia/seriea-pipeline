"""Walk-forward backtest of the auction projection against official fantacalcio points.

The board's projection is built from Understat xG/xA. The *truth* it is trying to predict
is the fantavoto actually awarded by fantacalcio.it -- a different scorer, with its own
assist rules. This module measures the gap, one season at a time, using only information
that existed before each auction.

Two things it deliberately does NOT do:
  * peek at the target season. `outfield_projection` blends early-season minutes when the
    season is under way; here the history is truncated below `ty`, so `early` is zero and
    `proj_min` falls back to prior-season minutes. The live 2026-27 board has strictly MORE
    information than any backtest fold, so these numbers are a floor, not an estimate.
  * condition on playing. The universe is whoever appeared the season BEFORE the auction --
    the pool you actually bid from. Players who then vanished score zero rather than
    dropping out, because losing a player to the bench is a real cost of a bad pick.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fantacalcio import build_auction_board as B  # noqa: E402
from scripts.fantacalcio import namematch as NM  # noqa: E402
from scripts.fantacalcio import voti  # noqa: E402

SEASONS = ["2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
           "2022-23", "2023-24", "2024-25", "2025-26"]


def _yr(season: str) -> int:
    return int(season[:4])


def truth(season: str) -> pd.DataFrame:
    """Official season totals. `bonus_total` is exactly what season_bonus predicts."""
    d = voti.fetch(season).copy()
    d = d[d.nome.notna()]
    d["pg"] = pd.to_numeric(d.pg, errors="coerce").fillna(0.0)
    d["bonus_per_app"] = d.mfv - d.mv
    d["bonus_total"] = d.bonus_per_app * d.pg
    d["fanta_total"] = d.mfv * d.pg
    return d[["nome", "R", "team", "pg", "mv", "mfv",
              "bonus_per_app", "bonus_total", "fanta_total"]]


def team_map(codes, us_teams) -> dict:
    """fantacalcio 3-letter code -> Understat club name, by prefix, uniqueness-checked."""
    out, clash = {}, []
    for c in codes:
        if not isinstance(c, str):
            continue
        hits = [t for t in us_teams
                if any(tok.startswith(c.lower()) for tok in NM.norm(t).split())]
        if len(hits) == 1:
            out[c] = hits[0]
        elif len(hits) > 1:
            clash.append((c, hits))
    return out, clash


def link(pred: pd.DataFrame, tru: pd.DataFrame, tmap: dict) -> pd.DataFrame:
    """Attach each fantacalcio row to an Understat projection row."""
    idx = NM.build_index(pred.player.tolist(), pred.last_team.tolist())
    rows = []
    for r in tru.itertuples():
        sur, ini = NM.parse_listone(r.nome)
        m, sc = NM.match_one(sur, ini, tmap.get(r.team), idx)
        rows.append({"nome": r.nome, "player": m["full"] if m else None, "score": sc})
    return pd.DataFrame(rows)


def fold(ty: int, verbose: bool = False) -> dict:
    """One auction: predict season `ty` knowing only seasons < ty."""
    tgt, prev = f"{ty}-{str(ty+1)[2:]}", f"{ty-1}-{str(ty)[2:]}"
    u = pd.read_parquet(ROOT / "data/parsed/understat_players.parquet")
    u = u[u.league == "ITA-Serie A"].copy()
    u["yr"] = u.season.astype(str).str[:4].astype(int)
    pred = B.outfield_projection(ty, src=u[u.yr < ty])

    t_now, t_prev = truth(tgt), truth(prev)
    us_teams = sorted(u[u.yr == ty - 1].team.map(B.NT).dropna().unique())
    tmap, clash = team_map(t_prev.team.dropna().unique(), us_teams)

    # Universe = the pool that existed at auction time, outfield only.
    pool = t_prev[(t_prev.pg > 0) & (t_prev.R.isin(["D", "C", "A"]))].copy()
    lk = link(pred, pool, tmap)
    pool = pool.merge(lk, on="nome", how="left")
    feat = ["player", "season_bonus", "proj_min", "bonus90", "np_g90", "ast90",
            "pt90", "pconv", "yel90", "red90"]
    pool = pool.merge(pred[feat], on="player", how="left")
    pool = pool.rename(columns={"bonus_total": "prev_bonus", "fanta_total": "prev_fanta",
                                "pg": "prev_pg"})
    # What they actually did in the target season; absent = 0.
    nxt = t_now[["nome", "bonus_total", "fanta_total", "pg"]].rename(
        columns={"bonus_total": "act_bonus", "fanta_total": "act_fanta", "pg": "act_pg"})
    pool = pool.merge(nxt, on="nome", how="left")
    for c in ("act_bonus", "act_fanta", "act_pg"):
        pool[c] = pool[c].fillna(0.0)
    cov = float(pool.season_bonus.notna().mean())
    pool["model"] = pool.season_bonus.fillna(0.0)

    # Official prior form: decay-weighted bonus per appearance over every season BEFORE ty,
    # weighted by games. This is the fantacalcio scorer's own verdict, not Understat's.
    hist = []
    for s_ in SEASONS:
        if _yr(s_) >= ty:
            continue
        h = truth(s_).copy()
        h["w"] = B.DECAY.get(ty - _yr(s_), 0.05) * h.pg
        hist.append(h[["nome", "bonus_per_app", "w", "pg"]])
    H = pd.concat(hist)
    H = H[H.pg > 0]
    off = H.groupby("nome").apply(
        lambda d: pd.Series({"off_rate": np.average(d.bonus_per_app, weights=d.w),
                             "off_w": d.w.sum()}), include_groups=False).reset_index()
    pool = pool.merge(off, on="nome", how="left")
    n90 = pool.proj_min.fillna(0.0) / 90.0
    pool["app90"] = n90
    pool["off_pred"] = pool.off_rate.fillna(0.0) * n90
    pool["c_goal"] = (pool.np_g90.fillna(0) + pool.pt90.fillna(0) * pool.pconv.fillna(0.78)) * n90
    pool["c_ast"] = pool.ast90.fillna(0) * n90
    pool["c_pmiss"] = pool.pt90.fillna(0) * (1 - pool.pconv.fillna(0.78)) * n90
    pool["c_yel"] = pool.yel90.fillna(0) * n90
    pool["c_red"] = pool.red90.fillna(0) * n90

    def sp(col):
        d = pool[[col, "act_bonus"]].dropna()
        return float(d[col].corr(d.act_bonus, method="spearman"))

    res = {"ty": ty, "target": tgt, "n": len(pool), "coverage": cov,
           "clash": clash,
           "sp_model": sp("model"), "sp_prev_bonus": sp("prev_bonus"),
           "sp_prev_fanta": sp("prev_fanta")}
    # Decision metric: take the top-K by each ranker, compare realised bonus.
    for K in (30, 60, 120):
        for tag, col in (("model", "model"), ("prev", "prev_bonus")):
            top = pool.nlargest(K, col)
            res[f"top{K}_{tag}"] = float(top.act_bonus.mean())
            res[f"top{K}_{tag}_played"] = float((top.act_pg >= 10).mean())
    if verbose:
        print(f"  clash={clash}")
    return res, pool


FEATS = ["c_goal", "c_ast", "c_pmiss", "c_yel", "c_red", "app90"]


def _fit(tr: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = np.column_stack([tr[c].values for c in cols] + [np.ones(len(tr))])
    y = tr.act_bonus.values
    return np.linalg.lstsq(X, y, rcond=None)[0]


def _apply(te: pd.DataFrame, cols: list[str], beta: np.ndarray) -> np.ndarray:
    X = np.column_stack([te[c].values for c in cols] + [np.ones(len(te))])
    return X @ beta


CANDS = {
    "current":   None,                       # the shipped bonus90 formula, unfitted
    "refit":     FEATS,                      # same components, coefficients fitted to truth
    "refit+off": FEATS + ["off_pred"],       # ...plus fantacalcio's own prior verdict
    "off_only":  ["off_pred"],
}


def compare() -> int:
    """Walk-forward: every candidate is fitted only on seasons strictly before the fold."""
    pools = {}
    for ty in range(2018, 2026):
        _, pl = fold(ty)
        pools[ty] = pl.assign(ty=ty)
        print(f"  loaded {ty}", flush=True)

    rows = []
    for ty in range(2020, 2026):                  # need >=2 prior folds to fit
        tr = pd.concat([pools[t] for t in pools if t < ty]).fillna(0.0)
        te = pools[ty].fillna(0.0)
        rec = {"ty": ty, "n": len(te)}
        for name, cols in CANDS.items():
            pred = te.model.values if cols is None else _apply(te, cols, _fit(tr, cols))
            te = te.assign(**{f"p_{name}": pred})
            rec[f"sp_{name}"] = float(pd.Series(pred).corr(
                pd.Series(te.act_bonus.values), method="spearman"))
            for K in (30, 60):
                rec[f"t{K}_{name}"] = float(
                    te.nlargest(K, f"p_{name}").act_bonus.mean())
        rows.append(rec)
    d = pd.DataFrame(rows)
    print("\n=== walk-forward candidate comparison ("
          f"{d.ty.min()}-{d.ty.max()}, {len(d)} folds) ===")
    hdr = f"{'candidate':<12}{'spearman':>10}{'top30':>9}{'top60':>9}   per-season spearman"
    print(hdr)
    print("-" * len(hdr))
    for name in CANDS:
        per = " ".join(f"{v:.3f}" for v in d[f"sp_{name}"])
        print(f"{name:<12}{d[f'sp_{name}'].mean():>10.3f}"
              f"{d[f't30_{name}'].mean():>9.2f}{d[f't60_{name}'].mean():>9.2f}   {per}")
    base = d["sp_current"]
    print("\nwins vs current (spearman, per fold):")
    for name in CANDS:
        if name == "current":
            continue
        w = int((d[f"sp_{name}"] > base).sum())
        print(f"  {name:<12} {w}/{len(d)} folds   "
              f"mean delta {d[f'sp_{name}'].mean()-base.mean():+.4f}")
    d.to_csv(ROOT / "data/fantacalcio/backtest_candidates.csv", index=False)
    return 0


def _price_ladder() -> dict:
    """Per-role sorted list of this year's real listone prices, used as the price SHAPE
    for historical folds. Absolute credit levels are league-invariant (500 credits,
    10 managers), so the ladder transfers; only who sits on which rung changes."""
    L = B.load_listone(ROOT / "data/fantacalcio/listone_2026_2027.xlsx")
    col = "market_price" if "market_price" in L.columns else (
        "quota" if "quota" in L.columns else L.select_dtypes("number").columns[-1])
    return {r: sorted(pd.to_numeric(g[col], errors="coerce").dropna().tolist(),
                      reverse=True)
            for r, g in L.groupby("R")}


def _squad_score(sq: list, act: dict) -> float:
    """Realised points. You field your best XI each week, so slots are assigned by what
    the player ACTUALLY did, not by what he was projected to do."""
    tot = 0.0
    by_role = {}
    for p in sq:
        by_role.setdefault(p["R"], []).append(act.get(p["id"], 0.0))
    for role, vals in by_role.items():
        w = B.SLOT_W[role]
        for i, v in enumerate(sorted(vals, reverse=True)):
            tot += v * (w[i] if i < len(w) else 0.12)
    return tot


def simulate(budget: int = 475, roster: dict | None = None) -> int:
    """Head-to-head auction replay: my board vs what the room does, scored on reality."""
    roster = roster or {"D": 8, "C": 8, "A": 6}
    ladder = _price_ladder()
    out = []
    for ty in range(2018, 2026):
        _, pl = fold(ty)
        pl = pl[pl.R.isin(roster)].copy().fillna({"model": 0.0, "prev_bonus": 0.0,
                                                  "prev_fanta": 0.0, "act_bonus": 0.0})
        pl["id"] = np.arange(len(pl))
        # Prices: the ROOM sets them, and the room ranks on last season's fantapoints.
        pl["market_price"] = 1.0
        for role, g in pl.groupby("R"):
            rungs = ladder.get(role, [])
            order = g.sort_values("prev_fanta", ascending=False).index
            vals = [rungs[i] if i < len(rungs) else 1 for i in range(len(order))]
            pl.loc[order, "market_price"] = vals
        pl["status"] = "OK"
        act = dict(zip(pl.id, pl.act_bonus))

        res = {"ty": ty, "target": f"{ty}-{str(ty+1)[2:]}"}
        for tag, col in (("model", "model"), ("room", "prev_bonus"),
                         ("oracle", "act_bonus")):
            F = pl.assign(season_points=pl[col], model_price=pl[col])
            plan = B.optimise(F, budget, roster)
            res[tag] = _squad_score(plan["squad"], act)
            res[f"{tag}_spend"] = sum(plan["spend"].values())
        res["edge_pct"] = (res["model"] / res["room"] - 1) * 100 if res["room"] else 0.0
        res["capture"] = (res["model"] / res["oracle"] * 100) if res["oracle"] else 0.0
        out.append(res)
        print(f"{res['target']}  model={res['model']:7.1f}  room={res['room']:7.1f}  "
              f"oracle={res['oracle']:7.1f}   edge={res['edge_pct']:+6.1f}%  "
              f"capture={res['capture']:5.1f}%")
    d = pd.DataFrame(out)
    w = int((d.model > d.room).sum())
    print(f"\n=== {len(d)} simulated auctions, roster {roster}, {budget} credits ===")
    print(f"  model beats the room in {w}/{len(d)} seasons")
    print(f"  mean realised: model {d.model.mean():.1f} vs room {d.room.mean():.1f}"
          f"   ({d.edge_pct.mean():+.1f}% per season, median {d.edge_pct.median():+.1f}%)")
    print(f"  mean capture of perfect foresight: {d.capture.mean():.1f}%")
    d.to_csv(ROOT / "data/fantacalcio/backtest_auctions.csv", index=False)
    return 0


def main() -> int:
    rows = []
    for ty in range(2018, 2026):
        r, _ = fold(ty)
        rows.append(r)
        print(f"{r['target']}  n={r['n']:4d}  match={r['coverage']:.0%}  "
              f"spearman model={r['sp_model']:.3f} prev={r['sp_prev_bonus']:.3f}  "
              f"top30 bonus model={r['top30_model']:6.1f} prev={r['top30_prev']:6.1f}")
    d = pd.DataFrame(rows)
    print("\n=== walk-forward mean over", len(d), "seasons ===")
    print(f"  spearman   model {d.sp_model.mean():.3f}   "
          f"prev-season-bonus {d.sp_prev_bonus.mean():.3f}   "
          f"prev-season-fanta {d.sp_prev_fanta.mean():.3f}")
    for K in (30, 60, 120):
        print(f"  top-{K:<3d} realised bonus   model {d[f'top{K}_model'].mean():6.2f}   "
              f"prev {d[f'top{K}_prev'].mean():6.2f}   "
              f"(played>=10: {d[f'top{K}_model_played'].mean():.0%} vs "
              f"{d[f'top{K}_prev_played'].mean():.0%})")
    d.drop(columns=["clash"]).to_csv(ROOT / "data/fantacalcio/backtest_folds.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(
        compare() if "--compare" in sys.argv
        else simulate() if "--simulate" in sys.argv
        else main())
