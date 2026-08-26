#!/usr/bin/env python3
"""Build the Fantacalcio 2026-27 auction board: projections, credit values, optimal roster.

Reads the official Fantacalcio.it listone (roles + quotazioni + FVM), multi-season
Understat player-seasons, FBref goalkeeper match stats, and the Transfermarkt live
squad file. Writes data/fantacalcio/auction_board.json for the /fantacalcio page.

Projection is validated by scripts/fantacalcio/backtest.py -- a walk-forward replay of
eight past auctions, each using only information that existed before it. Run that, do not
quote numbers from here; the payload reads its CSVs at build time so the page can never
show a figure the backtest does not currently support.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.team_names import normalize_team as NT  # noqa: E402
from scripts.fantacalcio import voti  # noqa: E402
from scripts.fantacalcio.namematch import build_index, match_one, norm, parse_listone  # noqa: E402

TARGET_YEAR = 2026
DECAY = {0: 1.00, 1: 0.62, 2: 0.34, 3: 0.18, 4: 0.09}
ALIAS_U = {"Zambo Anguissa": "Franck Zambo",
           # Verified against the TM squad file 2026-08-26: the surname match alone
           # picks the wrong person for these two (Hamari Traore, Afonso Moreira).
           "Traorè Hj.": "Hamed Junior Traore",
           "Moreira": "Diego Moreira"}
# Listone rows that must NOT match Understat at all: the real player has no big-5 record
# and the surname alone would hand him someone else's career (Ale Gomes != André Gomes).
BLOCK_U = {"Gomes"}
ALIAS_TM = {"Zambo Anguissa": "Frank Anguissa"}
# weight of the k-th best player at a role in your weekly XI
SLOT_W = {
    "P": [1.0, 0.05, 0.05],
    "D": [1.0, 1.0, 1.0, 0.50, 0.12, 0.12, 0.12, 0.12],
    "C": [1.0, 1.0, 1.0, 1.0, 0.50, 0.12, 0.12, 0.12],
    "A": [1.0, 1.0, 0.60, 0.12, 0.12, 0.12],
}
STARTERS = {"P": 1.0, "D": 3.5, "C": 4.5, "A": 2.6}
# The 6.0 every fielded player earns is FREE for outfield roles: in a 10-team league
# with 25-man rosters your bench supplies it too, so it cancels in any comparison, and
# `season_bonus` already scales with minutes. It does NOT cancel in goal, where only one
# keeper plays and a keeper with no minutes would otherwise "concede nothing" and outrank
# every real starter.
BASE_VOTE = {"P": 6.0, "D": 0.0, "C": 0.0, "A": 0.0}

# Cross-league conversion, measured on the 164 players who moved from a big-5 league to
# Serie A with >=600 minutes on both sides (2017-2025): SA rate / foreign rate, pooled.
# Per-league splits (EPL x1.05, La Liga x0.77 for npxg) sit on n=17-63 each -- that spread
# is noise plus who-moves selection, so per-league factors would overfit; pooled n=137-146.
# Applied as a data transform on the foreign ROWS so the projection formula stays one
# formula. Penalties inherit the 0.93 through goals/xg -- slightly conservative, accepted.
FOREIGN_CONV = {"goals": 0.93, "np_goals": 0.93, "xg": 0.93, "np_xg": 0.93,
                "assists": 1.00, "xa": 1.00, "yellow_cards": 1.05, "red_cards": 1.05}
# A newcomer's foreign minutes count this much toward expected Serie A minutes. Movers who
# established themselves kept ~0.99 of their minutes (median), but that conditions on
# succeeding; shaded to the pessimistic side of the p40 (0.837) for the ones we cannot see.
FOREIGN_MIN_W = 0.85
# Validated on 2018-2025 debutants (>=600 foreign minutes the season before): ranking
# realized Serie A bonus, foreign-record projection beats the market-value fallback 4/6
# folds, mean spearman 0.554 vs 0.493, and is stable where mv swings (0.52-0.65 vs
# 0.32-0.70). That measurement is why this tier exists.


def load_listone(path: Path) -> pd.DataFrame:
    d = pd.read_excel(path, sheet_name="Tutti", header=1)
    d = d[d.Id.notna()].copy()
    d["Id"] = d.Id.astype(int)
    for c in ("Qt.A", "Qt.I", "FVM"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["team"] = d.Squadra.map(NT)
    return d


def outfield_projection(ty: int = TARGET_YEAR,
                        src: pd.DataFrame | None = None) -> pd.DataFrame:
    # `src` exists so backtest.py can hand in a history truncated before `ty`. Reading the
    # parquet unconditionally would leak the full target season into `cur`/`early`.
    u = pd.read_parquet(ROOT / "data/parsed/understat_players.parquet") if src is None else src
    u = u.copy()
    u["yr"] = u.season.astype(str).str[:4].astype(int)
    u = u[u.yr >= ty - 4]
    # Foreign league rows enter the SAME aggregation, converted at the measured
    # FOREIGN_CONV rates -- the formula below never needs to know where a row came from.
    # Serie A membership still matters in exactly three places, kept explicit under
    # `is_sa`: current-season pace (`cur`, rounds_played), mw_min, and sa_min_recent,
    # which defines who is "blind" and must stay a Serie A fact.
    u["is_sa"] = u.league == "ITA-Serie A"
    cols = ["minutes", "goals", "xg", "np_goals", "np_xg", "assists", "xa",
            "yellow_cards", "red_cards", "matches"]
    for c in cols:
        # float64, not the parquet's nullable Int64 -- the foreign-conversion multiply
        # below writes scaled floats back into these columns.
        u[c] = pd.to_numeric(u[c], errors="coerce").fillna(0.0).astype("float64")
    u["team"] = u.team.map(NT)
    for c, f in FOREIGN_CONV.items():
        u.loc[~u.is_sa, c] = u.loc[~u.is_sa, c] * f
    u["pens_scored"] = (u.goals - u.np_goals).clip(lower=0)
    u["pens_taken"] = (u.xg - u.np_xg).clip(lower=0) / 0.76
    u["w"] = (ty - u.yr).map(DECAY).fillna(0.05)

    # Minutes: foreign minutes predict Serie A minutes at FOREIGN_MIN_W on the euro.
    u["min_w"] = np.where(u.is_sa, u.minutes, u.minutes * FOREIGN_MIN_W)
    piv = u.pivot_table(index="player", columns="yr", values="min_w", aggfunc="sum").fillna(0)
    piv_sa = (u[u.is_sa].pivot_table(index="player", columns="yr", values="minutes",
                                     aggfunc="sum").reindex(piv.index).fillna(0))
    l1 = piv.get(ty - 1, pd.Series(0.0, index=piv.index))
    l2 = piv.get(ty - 2, pd.Series(0.0, index=piv.index))
    cur = piv_sa.get(ty, pd.Series(0.0, index=piv.index)).fillna(0)
    _rp = u.loc[(u.yr == ty) & u.is_sa, "matches"].max()
    rounds_played = max(int(_rp) if pd.notna(_rp) else 0, 1)

    agg = ["minutes", "np_goals", "np_xg", "assists", "xa", "yellow_cards",
           "red_cards", "pens_scored", "pens_taken"]
    W = u.groupby("player").apply(
        lambda d: pd.Series({c: (d[c] * d.w).sum() for c in agg}), include_groups=False)
    W["last_team"] = u[u.is_sa].sort_values("yr").groupby("player").team.last()
    W["sa_min_recent"] = u[(u.yr >= ty - 3) & u.is_sa].groupby("player").minutes.sum()
    W["sa_min_recent"] = W.sa_min_recent.fillna(0)
    W["fo_min_recent"] = u[(u.yr >= ty - 3) & ~u.is_sa].groupby("player").minutes.sum()
    W["fo_min_recent"] = W.fo_min_recent.fillna(0)
    fo_recent = u[(u.yr >= ty - 3) & ~u.is_sa]
    W["fo_league"] = (fo_recent.groupby(["player", "league"]).minutes.sum()
                      .reset_index().sort_values("minutes")
                      .drop_duplicates("player", keep="last").set_index("player").league)

    m90 = (W.minutes / 90).replace(0, np.nan)
    fw = (m90 / (m90 + 40)).clip(upper=0.60)          # finishing-skill credit
    W["np_g90"] = ((W.np_xg / m90) + fw * ((W.np_goals - W.np_xg) / m90)).clip(lower=0).fillna(0)
    W["ast90"] = ((W.xa / m90) + fw * ((W.assists - W.xa) / m90)).clip(lower=0).fillna(0)
    W["pt90"] = (W.pens_taken / m90).fillna(0).clip(lower=0)
    W["pconv"] = np.where(W.pens_taken > 0.5, (W.pens_scored / W.pens_taken).clip(0, 1), 0.78)
    W["yel90"] = (W.yellow_cards / m90).fillna(0)
    W["red90"] = (W.red_cards / m90).fillna(0)
    W["bonus90"] = (3 * (W.np_g90 + W.pt90 * W.pconv) + W.ast90
                    - 3 * W.pt90 * (1 - W.pconv) - 0.5 * W.yel90 - 1.0 * W.red90)

    base = (0.85 * l1 + 0.15 * l2).reindex(W.index).fillna(0)
    early = (cur / rounds_played * 38).reindex(W.index).fillna(0)
    hh, he = base > 0, early > 0
    W["proj_min"] = np.where(hh & he, 0.65 * base + 0.35 * early,
                      np.where(hh, base, np.where(he, 0.85 * early, 0.0))).clip(0, 3230)
    W["season_bonus"] = W.bonus90 * W.proj_min / 90
    W["mw_min"] = cur.reindex(W.index).fillna(0)
    return W.reset_index()


def gk_projection(ty: int = TARGET_YEAR, clean_sheet_bonus: float = 0.0,
                  src: pd.DataFrame | None = None) -> pd.DataFrame:
    g = pd.read_parquet(ROOT / "data/parsed/goalkeeper_stats.parquet") if src is None else src.copy()
    g["yr"] = g.season.astype(str).str[:4].astype(int)
    g = g[g.yr >= ty - 4].copy()
    for c in ("minutes", "gk_goals_against", "gk_saves"):
        g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0.0)
    g["team"] = g.team.map(NT)
    g["cs"] = ((g.gk_goals_against == 0) & (g.minutes >= 60)).astype(float)
    g["w"] = (ty - g.yr).map(DECAY).fillna(0.05)

    piv = g.pivot_table(index="player", columns="yr", values="minutes", aggfunc="sum").fillna(0)
    l1 = piv.get(ty - 1, pd.Series(0.0, index=piv.index))
    l2 = piv.get(ty - 2, pd.Series(0.0, index=piv.index))
    cur = piv.get(ty, pd.Series(0.0, index=piv.index))
    _r = g[g.yr == ty].groupby("team").size().max()
    rounds = max(int(_r) if pd.notna(_r) else 0, 1)

    W = g.groupby("player").apply(lambda d: pd.Series({
        "minutes": (d.minutes * d.w).sum(),
        "ga": (d.gk_goals_against * d.w).sum(),
        "cs": (d.cs * d.w).sum(),
    }), include_groups=False)
    W["last_team"] = g.sort_values("yr").groupby("player").team.last()
    W["sa_min_recent"] = g[g.yr >= ty - 3].groupby("player").minutes.sum()
    W["sa_min_recent"] = W.sa_min_recent.fillna(0)
    m90 = (W.minutes / 90).replace(0, np.nan)
    # shrink goals-against per 90 toward the league mean (keepers behind bad defences
    # regress; a single good season is not a skill claim)
    lg = (W.ga.sum() / (W.minutes.sum() / 90))
    k = 12.0
    W["ga90"] = ((W.ga + lg * k) / (m90 + k)).fillna(lg)
    W["cs_rate"] = ((W.cs + 0.28 * k) / (m90 + k)).fillna(0.28)
    W["bonus90"] = -W.ga90 + clean_sheet_bonus * W.cs_rate

    base = (0.85 * l1 + 0.15 * l2).reindex(W.index).fillna(0)
    early = (cur / rounds * 38).reindex(W.index).fillna(0)
    hh, he = base > 0, early > 0
    W["proj_min"] = np.where(hh & he, 0.65 * base + 0.35 * early,
                      np.where(hh, base, np.where(he, 0.85 * early, 0.0))).clip(0, 3420)
    W["season_bonus"] = W.bonus90 * W.proj_min / 90
    W["mw_min"] = cur.reindex(W.index).fillna(0)
    return W.reset_index()


def assemble(listone: pd.DataFrame, out: pd.DataFrame, gk: pd.DataFrame) -> pd.DataFrame:
    """Join every listone row to its projection and to the live Transfermarkt squad."""
    tm = pd.read_parquet(ROOT / "data/external/transfermarkt/market_values_2026_2027.parquet")
    tm["team"] = tm.team.map(NT)
    wk = pd.read_parquet(ROOT / "data/external/transfermarkt/wiki_transfers_2026_2027.parquet")

    tm_idx = build_index(tm.player_name, tm.team)
    proj = {"P": gk.set_index("player"), "OUT": out.set_index("player")}
    o_idx = sorted(build_index(out.player, out.last_team),
                   key=lambda r: -proj["OUT"].sa_min_recent.get(r["full"], 0))
    g_idx = sorted(build_index(gk.player, gk.last_team),
                   key=lambda r: -proj["P"].sa_min_recent.get(r["full"], 0))

    rows = []
    for _, r in listone.iterrows():
        sur, ini = parse_listone(r.Nome)
        d = {"id": int(r.Id), "nome": r.Nome, "R": r.R, "RM": str(r.RM), "team": r.team,
             "fvm": float(r.FVM), "qt": float(r["Qt.A"])}

        alias_tm = ALIAS_TM.get(r.Nome)
        hit = None
        if alias_tm is not None:
            sub = tm[tm.player_name == alias_tm]
            if len(sub):
                hit = {"full": alias_tm, "team": sub.iloc[0].team}
        if hit is None:
            hit, _ = match_one(sur, ini, r.team, tm_idx)

        if hit is None:
            w = wk[wk.player_name.map(
                lambda x, _s=sur: isinstance(x, str) and norm(x).split()[-1] == _s.split()[-1])]
            w = w[w.from_club.map(
                lambda x, _t=r.team: isinstance(x, str) and norm(x)[:5] == norm(_t)[:5])]
            d.update(status="DEPARTED" if len(w) else "UNVERIFIED", age=None, mv=None,
                     joined=None,
                     note=(f"→ {w.iloc[0].to_club} {w.iloc[0].transfer_date}" if len(w) else ""))
        else:
            t = tm[tm.player_name == hit["full"]].iloc[0]
            d.update(status="OK" if hit["team"] == r.team else "TEAM_MISMATCH", note="",
                     age=int(t.age) if pd.notna(t.age) else None,
                     mv=float(t.market_value_eur) if pd.notna(t.market_value_eur) else None,
                     joined=str(t.joined_date)[:10])

        idx, tbl = (g_idx, proj["P"]) if r.R == "P" else (o_idx, proj["OUT"])
        alias_u = ALIAS_U.get(r.Nome)
        hu = {"full": alias_u} if alias_u else None
        mscore = 9.0 if alias_u else 0.0
        if hu is None and r.Nome not in BLOCK_U:
            hu, mscore = match_one(sur, ini, r.team, idx)
        if hu is not None and hu["full"] in tbl.index:
            p = tbl.loc[hu["full"]]
            d.update(src_name=hu["full"], sa_min=float(p.sa_min_recent),
                     proj_min=float(p.proj_min), bonus90=float(p.bonus90),
                     season_bonus=float(p.season_bonus), mw_min=float(p.mw_min),
                     src_team=(None if pd.isna(p.last_team) else str(p.last_team)),
                     ga90=(float(p.ga90) if "ga90" in p.index else np.nan),
                     fo_min=float(p.get("fo_min_recent", 0.0) or 0.0),
                     fo_league=(None if pd.isna(p.get("fo_league")) else str(p.get("fo_league"))),
                     fo_proj_min=np.nan)
        else:
            d.update(src_name=None, sa_min=0.0, proj_min=np.nan, bonus90=np.nan,
                     season_bonus=np.nan, mw_min=0.0, src_team=None, ga90=np.nan,
                     fo_min=0.0, fo_league=None, fo_proj_min=np.nan)
        # Keepers live in a Serie A-only stats table, so a keeper arriving from abroad
        # (Vicario) looks blind to it. His minutes are still knowable: the outfield table
        # carries every league's minutes for him. Ask it, and let fill_blind use the
        # answer for est_m -- the mv fit keeps the bonus, the record keeps the minutes.
        if r.R == "P" and d["sa_min"] < 270:
            ho, _ = match_one(sur, ini, r.team, o_idx)
            if ho is not None and ho["full"] in proj["OUT"].index:
                po = proj["OUT"].loc[ho["full"]]
                if float(po.get("fo_min_recent", 0.0) or 0.0) >= 600:
                    d.update(fo_min=float(po.fo_min_recent),
                             fo_league=(None if pd.isna(po.get("fo_league"))
                                        else str(po.get("fo_league"))),
                             fo_proj_min=float(po.proj_min))

        j = pd.to_datetime(d.get("joined"), errors="coerce")
        d["new"] = bool(pd.notna(j) and j >= pd.Timestamp(f"{TARGET_YEAR}-06-01"))
        d["blind"] = bool(d["sa_min"] < 270)
        # Foreign-informed: no meaningful Serie A record, but a real one abroad. These
        # rows keep their converted projection -- fill_blind must not overwrite it.
        d["foreign"] = bool(d["blind"] and d["fo_min"] >= 600)
        d["_mscore"] = float(mscore)
        rows.append(d)
    out_df = pd.DataFrame(rows)
    # A source player claimed by two listone rows means one of them is wrong (two real
    # players share a surname). Keep the better-scoring claim; blank the other rather
    # than shipping one man's record under two names.
    out_df["_teamhit"] = (out_df.src_team.notna() & (out_df.src_team == out_df.team)).astype(int)
    dup = out_df[out_df.src_name.notna()].sort_values(
        ["_teamhit", "_mscore"], ascending=[False, False])
    loser = dup.duplicated(subset=["src_name"], keep="first")
    bad = dup[loser].index
    out_df.loc[bad, ["src_name", "proj_min", "bonus90", "season_bonus", "ga90"]] = \
        [None, np.nan, np.nan, np.nan, np.nan]
    out_df.loc[bad, "sa_min"] = 0.0
    out_df.loc[bad, "blind"] = True
    out_df.loc[bad, ["fo_min"]] = 0.0
    out_df.loc[bad, "foreign"] = False
    out_df.loc[bad, "dup_dropped"] = True
    out_df["dup_dropped"] = out_df.get("dup_dropped", pd.Series(False, index=out_df.index)).fillna(False)
    return out_df.drop(columns=["_mscore", "_teamhit"])


def fill_blind(F: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Players with no recent Serie A minutes get a market-value prior, clearly flagged.

    Both halves are estimated: minutes AND bonus rate. Estimating bonus alone was a bug
    that handed a 14-point season to players with zero projected minutes, because a
    per-90 rate multiplied by nothing is still nothing. Leaving them at NaN is also
    wrong -- the optimiser would treat a EUR 50m striker as worthless -- so we estimate
    both, shrink hard toward the role median, and mark every such row `estimated`.
    """
    F = F.copy()
    F["log_mv"] = np.log10(F.mv.fillna(0).clip(lower=1e5))
    F["estimated"] = False
    fitinfo = {}
    for role in ("P", "D", "C", "A"):
        tr = F[(F.R == role) & F.season_bonus.notna() & F.mv.notna() & (~F.blind)
               & (F.proj_min > 0)]
        # A foreign-informed OUTFIELD row already has a measured projection; estimating
        # over it would replace a validated number with the R^2~0.05 fit this tier
        # exists to escape. Keepers stay in `te` (their bonus is unknowable from the
        # foreign row) but take their minutes from the record below.
        te = (F.R == role) & (F.season_bonus.isna() | F.blind) & F.mv.notna() \
            & ~(F.foreign & (F.R != "P"))
        if len(tr) < 12 or not te.any():
            fitinfo[role] = {"n": int(len(tr)), "r2_bonus": None, "r2_min": None}
            continue
        x = tr.log_mv.values
        bb, ab = np.polyfit(x, tr.bonus90.values, 1)
        bm, am = np.polyfit(x, tr.proj_min.values, 1)
        def _r2(y, pred):
            return round(float(1 - ((y - pred) ** 2).sum()
                                / max(((y - y.mean()) ** 2).sum(), 1e-9)), 3)
        fitinfo[role] = {"n": int(len(tr)),
                         "r2_bonus": _r2(tr.bonus90.values, ab + bb * x),
                         "r2_min": _r2(tr.proj_min.values, am + bm * x)}
        xt = F.loc[te, "log_mv"]
        est_b = ab + bb * xt
        est_m = (am + bm * xt).clip(0, 3230)
        med_b, med_m = float(tr.bonus90.median()), float(tr.proj_min.median())
        # 60/40 shrink toward the role median: this is a prior, not a measurement
        F.loc[te, "bonus90"] = 0.6 * est_b + 0.4 * med_b
        F.loc[te, "proj_min"] = 0.6 * est_m + 0.4 * med_m
        # ...except minutes for a keeper with a real record abroad: those are measured.
        kf = te & (F.R == "P") & F.fo_proj_min.notna() & (F.fo_proj_min > 0)
        F.loc[kf, "proj_min"] = F.loc[kf, "fo_proj_min"].clip(0, 3230)
        F.loc[te, "estimated"] = True
    F["season_bonus"] = F.bonus90 * F.proj_min / 90
    F["season_bonus"] = F.season_bonus.fillna(0.0)
    F["proj_min"] = F.proj_min.fillna(0.0)
    return F, fitinfo


def apply_injuries(F: pd.DataFrame) -> pd.DataFrame:
    """Overlay the newest Transfermarkt injury snapshot onto projected minutes.

    This is a fact layer, not a model: a player out until late October misses a known
    number of rounds, and paying his full projection ignores it. The join demands
    surname AND team agreement, so a shared surname on another club cannot leak. A
    player out with no return date gets the note and no haircut -- an unknowable length
    is not license to invent one.
    """
    import glob
    import pathlib
    snaps = sorted(glob.glob(str(ROOT / "data/external/injuries/injuries_2026-*.parquet")))
    snaps = [x for x in snaps if "premier_league" not in x]
    F = F.copy()
    F["inj_note"] = None
    if not snaps:
        return F
    snap = pd.read_parquet(snaps[-1])
    snap_date = pathlib.Path(snaps[-1]).stem.replace("injuries_", "")
    season_start = pd.Timestamp(f"{TARGET_YEAR}-08-23")
    today = pd.Timestamp(snap_date)
    for r in snap[snap.is_currently_out].itertuples():
        team = NT(r.team) or r.team
        toks = norm(r.player_name).split()
        if not toks:
            continue
        cand = F[(F.team == team)
                 & F.nome.map(lambda x, t=toks: norm(x).split()[0] in t)]
        if len(cand) > 1:
            # Two same-surname teammates (both Sulemanas at Atalanta). The snapshot has
            # the full given name, the listone an initial -- let it decide.
            def _ini_ok(nome, t=toks):
                _, ini = parse_listone(nome)
                return bool(ini) and any(g.startswith(ini[0]) for g in t)
            cand = cand[cand.nome.map(_ini_ok)]
        if len(cand) != 1:
            continue                       # still ambiguous, or absent: do nothing
        i = cand.index[0]
        ret = pd.to_datetime(r.expected_return, errors="coerce")
        if pd.notna(ret) and ret > today:
            missed = min(max(float((ret - season_start).days) / 7.4, 0.0), 38.0)
            F.loc[i, "proj_min"] = float(F.loc[i, "proj_min"]) * (38.0 - missed) / 38.0
            F.loc[i, "inj_note"] = (f"{r.injury_type} — out until ~{str(ret.date())}, "
                                    f"~{missed:.0f} rounds ({snap_date} snapshot)")
        else:
            # No return date is usually unknowable -- except when the injury TYPE is the
            # information. A cruciate or Achilles rupture is a 6-month rehab by medical
            # consensus; leaving those minutes whole priced a torn-ACL defender into the
            # squad. Everything else stays note-only rather than inventing a number.
            severe = any(k in str(r.injury_type).lower()
                         for k in ("cruciate", "achilles", "acl"))
            if severe:
                ret = today + pd.Timedelta(days=180)
                missed = min(max(float((ret - season_start).days) / 7.4, 0.0), 38.0)
                F.loc[i, "proj_min"] = float(F.loc[i, "proj_min"]) * (38.0 - missed) / 38.0
                F.loc[i, "inj_note"] = (f"{r.injury_type} — no return date given; severe "
                                        f"class, assumed ~6 months (~{missed:.0f} rounds) "
                                        f"({snap_date} snapshot)")
            else:
                F.loc[i, "inj_note"] = (f"{r.injury_type} — currently out, no return date "
                                        f"({snap_date} snapshot)")
    F["season_bonus"] = F.bonus90 * F.proj_min / 90
    F["season_bonus"] = F.season_bonus.fillna(0.0)
    return F


# Per-team minutes cap by fantacalcio role, in SLOTS of the eleven. P is analytic: one
# keeper, never substituted, exactly 1/11 of minutes. D/C/A come from the nine seasons of
# voti appearance shares renormalized onto the remaining ten slots. Appearance shares
# overstate attackers (they get subbed most), which makes the A cap slightly GENEROUS --
# deliberately so: this pass only ever squeezes groups over cap, so a generous cap
# under-corrects and can never punish a plausible allocation.
_OUT = {"D": 0.358, "C": 0.365, "A": 0.204}
ROLE_SLOTS = {"P": 1.0, **{r: v / sum(_OUT.values()) * 10.0 for r, v in _OUT.items()}}
TEAM_MINUTES = 38 * 90 * 1.05               # stoppage-inclusive, per slot


def normalize_team_minutes(F: pd.DataFrame) -> pd.DataFrame:
    """A club can only hand out ~39,500 minutes; the per-player projection does not know
    that. Every player transferred INTO a stacked side keeps the minutes he had at his old
    club, so before this pass Milan's roster summed to 132% of a season and eight clubs
    exceeded 110%. This is an accounting identity, not a model: any (team, role) group
    over its share of the cap is squeezed proportionally back onto it. Proportional, so
    the ranking WITHIN the group is untouched -- only the impossible cross-team totals
    move. Groups under the cap are left alone: a thin squad is thin data, not a violation.
    Runs after the injury haircut on purpose -- an injured man's freed minutes soften the
    squeeze on his teammates.
    """
    F = F.copy()
    live = F.status != "DEPARTED"
    for (team, role), grp in F[live].groupby(["team", "R"]):
        cap = ROLE_SLOTS[role] * TEAM_MINUTES
        tot = grp.proj_min.sum()
        if tot > cap:
            F.loc[grp.index, "proj_min"] = grp.proj_min * (cap / tot)
    F["season_bonus"] = F.bonus90 * F.proj_min / 90
    F["season_bonus"] = F.season_bonus.fillna(0.0)
    return F


VOTI_SEASONS = ["2025-26", "2024-25", "2023-24", "2022-23"]


def attach_voti(F: pd.DataFrame) -> pd.DataFrame:
    """Media voto and availability, joined on NAME ONLY.

    Roles are reassigned every summer (Neres C->A, Cambiaghi A->C, Cancellieri A->C), so a
    (name, role) join silently drops exactly the players whose role change makes them
    interesting. Shrinkage uses the role he will PLAY, from the listone.
    """
    hist = voti.load(VOTI_SEASONS)
    g = voti.project(hist, VOTI_SEASONS)
    mu = {k.upper(): v for k, v in voti.role_means(hist, VOTI_SEASONS).items()}
    F = F.merge(g[["nome", "mv_raw", "pg_w", "pg_last"]], on="nome", how="left")
    F["mv_hat"] = voti.shrink(F.mv_raw, F.pg_w, F.R, mu)
    F["p_play"] = np.clip(F.proj_min.fillna(0) / (38 * 90), 0, 0.97)
    F.loc[F.status == "DEPARTED", ["mv_hat", "p_play"]] = np.nan
    return F, mu


# ---------------------------------------------------------------- modificatore difesa
# Computed on the raw VOTO, not the fantavoto. Bands are the two shapes actually in use;
# Leghe lets an admin set the values, so this is parameterised rather than assumed.
MOD_TABLES = {
    "three": [(6.00, 1), (6.50, 3), (7.00, 6)],
    "six": [(6.00, 1), (6.005, 2), (6.255, 3), (6.505, 4), (6.755, 5), (7.005, 6)],
}
# Weekly within-player voto spread, measured by regressing squared adjacent-season mean
# differences on (1/n1 + 1/n2): the slope is sampling variance, the intercept real talent
# drift. Season aggregates alone understate this badly.
MOD_SD = {"P": 0.350, "D": 0.393, "C": 0.730, "A": 0.478}
# Voto d'ufficio when you cannot field four defenders: first 5.0, then 4.5. This penalty is
# the entire reason defensive DEPTH has value under the modifier -- without it the model
# happily buys two stars and six players who never take the field.
MOD_OFFICE = (5.0, 4.5, 4.5)


def modifier_points(gk_mv: float, gk_p: float, d_mv, d_p, table, mu_p: float,
                    weeks: int = 38, sims: int = 3000, seed: int = 7) -> tuple:
    """Expected modifier points per season for one keeper + a set of defenders.

    Monte Carlo because the quantity is an order statistic (best 3 of those who actually
    played) passed through a step function -- there is no useful closed form, and a
    point estimate at the mean sits on the wrong side of a band boundary half the time.
    """
    rng = np.random.default_rng(seed)
    n = weeks * sims
    d_mv = np.asarray(d_mv, dtype=float)
    d_p = np.asarray(d_p, dtype=float)
    if len(d_mv) == 0:
        return 0.0, 0.0
    played = rng.random(n) < gk_p
    gv = np.where(played, rng.normal(gk_mv, MOD_SD["P"], n),
                  rng.normal(mu_p - 0.10, MOD_SD["P"], n))
    avail = rng.random((n, len(d_mv))) < d_p
    votes = rng.normal(d_mv, MOD_SD["D"], (n, len(d_mv)))
    votes = np.where(avail, votes, -np.inf)
    best = np.sort(votes, axis=1)[:, ::-1][:, :3]
    if best.shape[1] < 3:                       # fewer than three defenders owned at all
        best = np.pad(best, ((0, 0), (0, 3 - best.shape[1])), constant_values=-np.inf)
    best = np.where(np.isinf(best), np.asarray(MOD_OFFICE)[None, :], best)
    avg = (gv + best.sum(axis=1)) / 4.0
    pts = np.zeros_like(avg)
    for thr, val in table:
        pts = np.where(avg >= thr, val, pts)
    return float(pts.mean() * weeks), float(np.median(avg))


def _unit_of(plan: dict, F: pd.DataFrame) -> tuple:
    """The keeper and defenders a plan actually bought, as (gk_row, d_rows)."""
    ids = [int(x["id"]) for x in plan["squad"]]
    sub = F[F["id"].isin(ids)]
    gk = sub[sub.R == "P"].sort_values("season_points", ascending=False)
    return (gk.iloc[0] if len(gk) else None), sub[sub.R == "D"]


def modifier_credit(F: pd.DataFrame, plan: dict, table, mu: dict,
                    teams: int) -> tuple:
    """Per-player modifier credit, linearised around the defence the plan actually picked.

    The modifier is a property of a SET, so it cannot go into a knapsack directly. What can
    is its local derivative: measure dM/d(unit mean voto) by re-running the Monte Carlo with
    every vote shifted +/- 0.05, then a player who occupies one of the four counting slots
    moves that mean by (his voto - the replacement's) / 4.
    """
    gk, ds = _unit_of(plan, F)
    if gk is None or ds.empty:
        return pd.Series(0.0, index=F.index), 0.0, 0.0
    args = (gk.mv_hat, gk.p_play, ds.mv_hat.values, ds.p_play.values, table, mu["P"])
    m0, med = modifier_points(*args)
    hi, _ = modifier_points(gk.mv_hat + 0.05, gk.p_play, ds.mv_hat.values + 0.05,
                            ds.p_play.values, table, mu["P"])
    lo, _ = modifier_points(gk.mv_hat - 0.05, gk.p_play, ds.mv_hat.values - 0.05,
                            ds.p_play.values, table, mu["P"])
    fprime = (hi - lo) / 0.10

    # Baseline is the marginal STARTING defender in the league, not the role mean: that is
    # who you actually displace by owning a better one.
    dpool = F[(F.R == "D") & F.mv_hat.notna() & (F.proj_min.fillna(0) > 900)]
    k = min(len(dpool) - 1, teams * 4 - 1)
    base_d = float(dpool.mv_hat.sort_values(ascending=False).iloc[k]) if k >= 0 else mu["D"]
    kp = F[(F.R == "P") & F.mv_hat.notna() & (F.proj_min.fillna(0) > 1200)]
    k2 = min(len(kp) - 1, teams - 1)
    base_p = float(kp.mv_hat.sort_values(ascending=False).iloc[k2]) if k2 >= 0 else mu["P"]

    base = np.where(F.R == "P", base_p, base_d)
    credit = fprime * (F.mv_hat - base) / 4.0 * F.p_play.fillna(0.0)
    credit = credit.where(F.R.isin(["D", "P"]), 0.0).fillna(0.0)
    return credit, m0, fprime


def value_credits(F: pd.DataFrame, teams: int, budget: int,
                  roster: dict, extra: pd.Series | None = None) -> pd.DataFrame:
    """Value over the marginal STARTER, scaled so the league's credits clear the market."""
    F = F.copy()
    # Total expected fantapoints, base vote included. Bonus alone inverts goalkeepers
    # (one who never plays concedes nothing) and rewards players with no minutes at all.
    # The 6.0 base largely cancels in VOR because the replacement plays too, but it
    # correctly zeroes anyone who does not take the field.
    # Keepers: goals PREVENTED versus a replacement keeper, over the games they play.
    # Raw goals-against inverts (a keeper who never plays concedes nothing) and total
    # points inflates (your backup supplies the 6.0 too). This does neither, and lands
    # on the same points-above-replacement scale as outfield bonus.
    gkp = F[(F.R == "P") & F.ga90.notna() & (F.proj_min >= 900)].sort_values("ga90")
    repl_ga90 = float(gkp.ga90.iloc[teams - 1]) if len(gkp) >= teams else float(
        gkp.ga90.median() if len(gkp) else 1.35)
    F["repl_ga90"] = np.where(F.R == "P", repl_ga90, np.nan)
    F["season_points"] = np.where(
        F.R == "P", (F.proj_min / 90.0) * (repl_ga90 - F.ga90.fillna(repl_ga90)),
        F.season_bonus)
    # The modifier credit has to land HERE, not before the call: this function rebuilds
    # season_points from season_bonus every time, so anything added upstream is silently
    # discarded. (It was -- every lambda in the sweep returned a byte-identical squad.)
    if extra is not None:
        F["season_points"] = F["season_points"] + extra.reindex(F.index).fillna(0.0)
    F.loc[F.status == "DEPARTED", ["season_points", "season_bonus"]] = np.nan
    repl = {}
    for role, st in STARTERS.items():
        # Replacement level is set by players we actually have data on. A blind-fit row
        # cannot define the bar it is about to be measured against.
        pool = F[(F.R == role) & F.season_points.notna()
                 & ~F.estimated].season_points.sort_values(ascending=False)
        n = int(round(st * teams))
        repl[role] = float(pool.iloc[n]) if len(pool) > n else float(pool.min())
    F["replacement"] = F.R.map(repl)
    # A blind-fit row has R^2 0.05-0.31: its "bonus" is regression noise, not a signal.
    # Cap it at replacement so it can never out-bid a real player for a starter slot;
    # it stays available as 1-credit bench filler, valued honestly at replacement.
    est = F.estimated & F.season_points.notna()
    F.loc[est, "season_points"] = np.minimum(
        F.loc[est, "season_points"], F.loc[est, "replacement"])
    F["vor"] = (F.season_points - F.replacement).clip(lower=0)

    total = teams * budget
    n_rostered = teams * sum(roster.values())
    surplus = total - n_rostered
    F["market_price"] = 1.0
    # Market price: FVM scaled so the league's whole budget clears, role by role, on the
    # same share of credits the market itself puts into each role.
    for role in roster:
        m = (F.R == role) & (F.status != "DEPARTED")
        share = F.loc[m, "fvm"].sum() / F.loc[F.status != "DEPARTED", "fvm"].sum()
        pot = surplus * share
        fs = F.loc[m, "fvm"].sum()
        F.loc[m, "market_price"] = 1 + F.loc[m, "fvm"] * (pot / fs if fs > 0 else 0)

    # Model price by RANK-MATCH, not by rescaling VOR. VOR is concentrated in ~110
    # players while the market spreads the same credits over 250, so scaling the two
    # sums against each other inflates every good player 3x and floors everyone else --
    # the ranking survives but the credit numbers become unbiddable. Instead: take the
    # role's own price ladder and hand the k-th best player the k-th highest price.
    # Total credit mass is then identical to the market's by construction, and the edge
    # is a rank displacement quoted in real credits.
    F["model_price"] = 1.0
    for role in roster:
        m = F.index[(F.R == role) & (F.status != "DEPARTED")]
        sub = F.loc[m]
        ladder = np.sort(sub.market_price.values)[::-1]
        order = sub.season_points.fillna(-1e9).sort_values(ascending=False).index
        F.loc[order, "model_price"] = ladder[: len(order)]

    F.loc[F.status == "DEPARTED", ["model_price", "market_price"]] = np.nan
    F["edge"] = F.model_price - F.market_price
    F["edge_ratio"] = F.model_price / F.market_price.replace(0, np.nan)
    return F


def optimise(F: pd.DataFrame, budget: int, roster: dict,
             exclude: frozenset = frozenset(), owned: tuple = ()) -> dict:
    """Pick the 25-man squad maximising expected weekly XI points at market prices.

    Two stages, both exact: a 0/1 knapsack per role over (players chosen, credits spent),
    then a max-plus combination that splits the budget across the four roles. The k-th
    best player at a role is weighted by how often he actually makes your XI (SLOT_W),
    so the optimiser buys starters and fills the bench at the floor price.
    """
    NEG = -1e18
    per_role, tables = {}, {}
    for role, slots in roster.items():
        pool = F[(F.R == role) & F.model_price.notna() & (F.status != "DEPARTED")].copy()
        if exclude:
            pool = pool[~pool["id"].isin(exclude)]
        pool = pool.sort_values("season_points", ascending=False).reset_index(drop=True)
        pool["cost"] = pool.market_price.round().clip(lower=1).astype(int)
        w = SLOT_W[role]
        n = len(pool)
        # dp[j][b]: best score with EXACTLY j players costing EXACTLY b credits.
        # A snapshot per item is kept so reconstruction cannot pick the same player
        # twice -- a back-pointer array alone gets overwritten by later items.
        hist = np.full((n + 1, slots + 1, budget + 1), NEG, dtype=np.float64)
        hist[0, 0, 0] = 0.0
        for i in range(n):
            c = int(pool.cost.iat[i]); sb = float(pool.season_points.iat[i])
            cur = hist[i]
            nxt = cur.copy()
            if c <= budget:
                for j in range(1, slots + 1):
                    cand = cur[j - 1][: budget + 1 - c] + sb * w[j - 1]
                    tgt = nxt[j][c:]
                    better = cand > tgt
                    if better.any():
                        idx = np.nonzero(better)[0]
                        nxt[j][c + idx] = cand[idx]
            hist[i + 1] = nxt
        per_role[role] = hist[n][slots]
        tables[role] = (pool, hist, slots)

    roles = list(roster)
    cum = np.full(budget + 1, NEG); cum[0] = 0.0
    picks = []
    for role in roles:
        arr = per_role[role]
        nxt = np.full(budget + 1, NEG)
        pick = np.zeros(budget + 1, dtype=np.int32)
        for k in range(budget + 1):
            if arr[k] <= NEG / 2:
                continue
            cand = cum[: budget + 1 - k] + arr[k]
            tgt = nxt[k:]
            better = cand > tgt
            if better.any():
                idx = np.nonzero(better)[0]
                nxt[k + idx] = cand[idx]
                pick[k + idx] = k
        cum = nxt
        picks.append(pick)

    best_b = int(np.argmax(cum))
    if cum[best_b] <= NEG / 2:
        return {"squad": [], "spend": {r: 0 for r in roles}, "score": 0.0}
    spend, b = {}, best_b
    for role, pick in zip(reversed(roles), reversed(picks)):
        k = int(pick[b]); spend[role] = k; b -= k

    squad = []
    for role in roles:
        pool, hist, slots = tables[role]
        b, j, i = spend[role], slots, len(pool)
        chosen = []
        while i > 0 and j > 0:
            if hist[i][j][b] > hist[i - 1][j][b]:      # item i-1 was taken here
                p = pool.iloc[i - 1]
                chosen.append(p)
                b -= int(p.cost); j -= 1
            i -= 1
        for rank, p in enumerate(sorted(chosen, key=lambda x: -x.season_points)):
            squad.append({**p.to_dict(), "rank": rank + 1,
                          "slot_w": SLOT_W[role][rank] if rank < len(SLOT_W[role]) else 0.12})
    return {"squad": squad, "spend": spend, "score": float(cum[best_b])}


def marginal_rate(F: pd.DataFrame, budget: int, roster: dict, base: float) -> float:
    """Points bought by one more credit, at this budget. The exchange rate between the
    two currencies -- without it 'he is worth 4 points more' cannot become a price."""
    up = optimise(F, budget + 10, roster)["score"]
    return max((up - base) / 10.0, 1e-6)


def walkaway_price(F: pd.DataFrame, budget: int, roster: dict, pid: int,
                   cache: dict | None = None,
                   exclude: frozenset = frozenset()) -> int:
    """The highest price at which this squad still wants him -- bid up to and INCLUDING
    this number, never one credit more.

    Exact, by perturbation: set his price to p, re-solve the whole board, ask whether he
    survived. Monotone in p (raising a price never makes a player more attractive), so a
    binary search is sound. This deliberately does not use a force-in: `optimise`'s
    `owned` parameter is dead, and a price nudge needs no such machinery.
    """
    lo, hi = 0, budget

    def keeps(p: int) -> bool:
        key = (pid, p, exclude)
        if cache is not None and key in cache:
            return cache[key]
        Fx = F.copy()
        Fx.loc[Fx.id == pid, "market_price"] = p
        got = pid in {int(x["id"])
                      for x in optimise(Fx, budget, roster, exclude=exclude)["squad"]}
        if cache is not None:
            cache[key] = got
        return got

    if not keeps(1):
        return 0                       # not worth a single credit to this squad
    if keeps(hi):
        return hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if keeps(mid):
            lo = mid
        else:
            hi = mid
    return lo


def bidding_table(F: pd.DataFrame, plan: dict, budget: int, roster: dict,
                  extra_per_role: int = 40) -> dict:
    """Walk-away price and need for everyone worth a decision: the squad, plus the players
    just outside it who are the realistic fallbacks when one is stolen."""
    base = plan["score"]
    ids = [int(x["id"]) for x in plan["squad"]]
    want = set(ids)
    for role in roster:
        pool = F[(F.R == role) & F.model_price.notna() & (F.status != "DEPARTED")]
        want |= set(pool.nlargest(extra_per_role, "season_points").id.astype(int))
    rate = marginal_rate(F, budget, roster, base)
    out = {}
    for pid in sorted(want):
        wa = walkaway_price(F, budget, roster, pid)
        if pid in ids:
            need = max(0.0, base - optimise(F, budget, roster,
                                            exclude=frozenset([pid]))["score"])
        else:
            need = 0.0
        out[int(pid)] = {"walkaway": int(wa), "need": round(float(need), 2),
                         "in_squad": pid in want and pid in ids}
    return {"rate": round(float(rate), 4), "players": out}


def _backtest_summary() -> dict:
    """Read the walk-forward backtest off disk. Never hard-code these -- a stale literal
    on an auction page is exactly the failure this repo has paid for before."""
    out = {"source": "scripts/fantacalcio/backtest.py", "folds": [], "auctions": []}
    fp = ROOT / "data/fantacalcio/backtest_folds.csv"
    ap_ = ROOT / "data/fantacalcio/backtest_auctions.csv"
    if fp.exists():
        d = pd.read_csv(fp)
        out["folds"] = [{"season": r.target, "n": int(r.n),
                         "match_rate": round(float(r.coverage), 3),
                         "model_spearman": round(float(r.sp_model), 3),
                         "naive_spearman": round(float(r.sp_prev_bonus), 3)}
                        for r in d.itertuples()]
        out["spearman_model"] = round(float(d.sp_model.mean()), 3)
        out["spearman_naive"] = round(float(d.sp_prev_bonus.mean()), 3)
    if ap_.exists():
        d = pd.read_csv(ap_)
        out["auctions"] = [{"season": r.target, "model": round(float(r.model), 1),
                            "room": round(float(r.room), 1),
                            "oracle": round(float(r.oracle), 1),
                            "edge_pct": round(float(r.edge_pct), 1)}
                           for r in d.itertuples()]
        out["auction_wins"] = int((d.model > d.room).sum())
        out["auction_n"] = int(len(d))
        out["auction_edge_median_pct"] = round(float(d.edge_pct.median()), 1)
        out["auction_capture_pct"] = round(float(d.capture.mean()), 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument("--teams", type=int, default=10)
    ap.add_argument("--mod-difesa", choices=["three", "six", "off"], default="three",
                    help="modificatore di difesa band table (Leghe lets admins set values)")
    ap.add_argument("--clean-sheet-bonus", type=float, default=0.0,
                    help="+points for a goalkeeper clean sheet if your league enables it")
    ap.add_argument("--listone", type=Path,
                    default=ROOT / "data/fantacalcio/listone_2026_2027.xlsx")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data/fantacalcio/auction_board.json")
    a = ap.parse_args()
    roster = {"P": 3, "D": 8, "C": 8, "A": 6}

    listone = load_listone(a.listone)
    out = outfield_projection()
    gk = gk_projection(clean_sheet_bonus=a.clean_sheet_bonus)
    F = assemble(listone, out, gk)
    F, fitinfo = fill_blind(F)
    F = apply_injuries(F)
    F = normalize_team_minutes(F)
    F, role_mu = attach_voti(F)

    mod_table = MOD_TABLES.get(a.mod_difesa)
    if mod_table is not None:
        # The modifier needs FOUR defenders on the pitch, not three-and-a-half: the whole
        # bonus is forfeited if you cannot field four, and the voto d'ufficio for a missing
        # one (5.0, then 4.5) is a band-killer. Widen the starter ladder to match.
        SLOT_W["D"] = [1.0, 1.0, 1.0, 1.0, 0.35, 0.12, 0.12, 0.12]
        STARTERS["D"] = 4.3

    F = value_credits(F, a.teams, a.budget, roster)
    plan = optimise(F, a.budget, roster)
    mod_info = {"table": a.mod_difesa, "points": 0.0, "iterations": []}
    F["mod_credit"] = 0.0

    if mod_table is not None:
        # A linear per-player credit cannot see that the modifier is a THRESHOLD on the best
        # three: five 6.2 defenders are worth more together than the sum of their individual
        # credits, because as a group they carry the average across a band. So do not trust
        # the linearisation's scale -- sweep it, and score every squad it produces by the
        # honest quantity: raw bonus points plus the Monte Carlo modifier of the defence it
        # actually bought. The sweep is the correction; the argmax is the answer.
        # Measured 2026-08-26: the sweep's argmax (lam=0.5) beat lam=0 by 0.06 points out
        # of 370 -- stable across three MC seeds, but 0.06 is not a preference, and it
        # churned six of the cheap bench slots for it. Crediting the modifier into
        # season_points also propagates into model_price and into every need_loss, so the
        # numbers you BID on would carry a Monte Carlo term. Pinned to zero: the modifier's
        # real effect is the widened four-starter ladder above, which applies regardless,
        # and the true modifier points are still measured and reported below.
        best = None
        for lam in (0.0,):
            credit, m0, fprime = modifier_credit(F, plan, mod_table, role_mu, a.teams)
            F["mod_credit"] = credit * lam
            F = value_credits(F, a.teams, a.budget, roster, extra=F["mod_credit"])
            cand = optimise(F, a.budget, roster)
            gk, ds = _unit_of(cand, F)
            true_m, med = (modifier_points(gk.mv_hat, gk.p_play, ds.mv_hat.values,
                                           ds.p_play.values, mod_table, role_mu["P"])
                           if gk is not None and not ds.empty else (0.0, 0.0))
            # Strip the credit back out before adding the real thing, or it is counted twice.
            cr = float(F.set_index("id").loc[
                [int(x["id"]) for x in cand["squad"]], "mod_credit"].sum())
            total = cand["score"] - cr + true_m
            mod_info["iterations"].append(
                {"lambda": lam, "raw_score": round(cand["score"], 2),
                 "modifier_pts": round(true_m, 1), "median_avg": round(med, 3),
                 "true_total": round(total, 1), "dM_dvote": round(fprime, 1),
                 "spend_D": cand["spend"].get("D"), "spend_P": cand["spend"].get("P")})
            if best is None or total > best[0]:
                best = (total, cand, true_m, med, lam, F["mod_credit"].copy())
        # Re-price the whole board at the winning lambda so prices match the plan on show.
        total, plan, true_m, med, lam, credit_col = best
        F["mod_credit"] = credit_col
        F = value_credits(F, a.teams, a.budget, roster, extra=F["mod_credit"])
        plan = optimise(F, a.budget, roster)
        gk, ds = _unit_of(plan, F)
        true_m, med = modifier_points(gk.mv_hat, gk.p_play, ds.mv_hat.values,
                                      ds.p_play.values, mod_table, role_mu["P"])
        cr = float(F.set_index("id").loc[
            [int(x["id"]) for x in plan["squad"]], "mod_credit"].sum())
        mod_info.update(points=round(true_m, 1), median_avg=round(med, 3),
                        chosen_lambda=lam,
                        true_total=round(plan["score"] - cr + true_m, 1))

    # How much do I actually NEED each pick? Re-solve the whole board with him removed:
    # the score you lose is the only honest answer, because it already accounts for the
    # money freed up and who is still on the shelf to spend it on. A star with a close
    # substitute is worth less to you than his price suggests; one with none is worth more.
    bid = bidding_table(F, plan, a.budget, roster)
    wa_of = {k: v["walkaway"] for k, v in bid["players"].items()}

    marg = {}
    base_ids = {int(s_["id"]) for s_ in plan["squad"]}
    for s_ in plan["squad"]:
        pid = int(s_["id"])
        alt = optimise(F, a.budget, roster, exclude=frozenset({pid}))
        alt_ids = {int(x["id"]) for x in alt["squad"]}
        # The fallback chain has to be CUMULATIVE. Banning one player at a time answers
        # "who replaces him if everyone else is still on the shelf" -- which is never the
        # situation. In a real room the run continues: he goes, then your replacement goes,
        # then the next one. So ban each answer before asking for the next.
        banned, chain = {pid}, []
        for _ in range(3):
            step = optimise(F, a.budget, roster, exclude=frozenset(banned))
            if not step["squad"]:
                break
            fresh = [x for x in step["squad"]
                     if int(x["id"]) not in base_ids and int(x["id"]) not in banned
                     and x["R"] == s_["R"]]
            if not fresh:
                break
            pick = max(fresh, key=lambda z: float(z["season_points"]))
            # His cap must be measured in the state you would actually be bidding in --
            # with the men above him in the chain already gone. The clean-board number is
            # wrong here by construction: it prices him against a shelf that no longer has
            # the player you just lost.
            wa_ctx = walkaway_price(F, a.budget, roster, int(pick["id"]),
                                    exclude=frozenset(banned))
            chain.append({
                "nome": pick["nome"], "team": pick["team"],
                "buy": int(round(float(pick["market_price"]))),
                "max": int(round(float(pick["model_price"]))),
                "walkaway": int(wa_ctx),
                "walkaway_clean": int(wa_of.get(int(pick["id"]), 0)),
                "loss_here": round(plan["score"] - step["score"], 2)})
            banned.add(int(pick["id"]))
        marg[pid] = {
            "loss": round(plan["score"] - alt["score"], 2),
            "replacements": chain,
            "n_changed": len(alt_ids - base_ids),
        }

    def clean(rec: dict) -> dict:
        o = {}
        for k, v in rec.items():
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = None if np.isnan(v) else round(float(v), 3)
            elif isinstance(v, (np.bool_,)):
                v = bool(v)
            elif isinstance(v, float) and np.isnan(v):
                v = None
            o[k] = v
        return o

    keep = ["id", "nome", "R", "RM", "team", "fvm", "qt", "status", "note", "age", "mv",
            "joined", "new", "blind", "foreign", "fo_min", "fo_league", "inj_note", "estimated",
            "dup_dropped", "sa_min", "mw_min", "proj_min",
            "bonus90", "ga90", "repl_ga90", "season_bonus", "season_points", "replacement", "vor", "model_price",
            "market_price", "edge", "edge_ratio",
            "mv_raw", "mv_hat", "pg_w", "p_play", "mod_credit"]
    players = [clean({k: r.get(k) for k in keep}) for _, r in F.iterrows()]
    squad = [clean({**{k: s.get(k) for k in keep}, "rank": s["rank"],
                    "slot_w": s["slot_w"],
                    "need_loss": marg[int(s["id"])]["loss"],
                    "walkaway": int(wa_of.get(int(s["id"]), 0)),
                    "alternatives": marg[int(s["id"])]["replacements"]})
             for s in plan["squad"]]

    payload = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "settings": {"budget": a.budget, "teams": a.teams, "roster": roster,
                     "clean_sheet_bonus": a.clean_sheet_bonus,
                     "mod_difesa": a.mod_difesa,
                     "mod_table": MOD_TABLES.get(a.mod_difesa),
                     "mod_sd": MOD_SD, "mod_office": list(MOD_OFFICE),
                     "role_mu_P": round(float(role_mu.get("P", 6.17)), 3),
                     "slot_w": SLOT_W,
                     "starters": STARTERS},
        "validation": {**_backtest_summary(), "blind_fit": fitinfo},
        "bidding": bid,
        "counts": {
            "listone": int(len(F)),
            "departed": int((F.status == "DEPARTED").sum()),
            "blind": int((F.blind & ~F.foreign).sum()),
            "foreign": int(F.foreign.sum()),
            "estimated": int(F.estimated.sum()),
            "new": int(F.new.sum()),
        },
        "replacement": {k: round(float(v), 2) for k, v in
                        F.groupby("R").replacement.first().items()},
        "plan": {"spend": plan["spend"], "score": round(plan["score"], 1)},
        "modifier": mod_info,
        "squad": squad,
        "players": players,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=1))
    tot = sum(round(s["market_price"]) for s in squad if s.get("market_price"))
    print(f"wrote {a.out}  players={len(players)} squad={len(squad)} "
          f"spend={tot}/{a.budget} score={plan['score']:.1f}")
    print("  spend by role:", plan["spend"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
