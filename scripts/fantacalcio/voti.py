"""Italian media voto from fantacalcio.it, projected forward for the auction board.

The modificatore di difesa is computed on the raw *voto* -- not the fantavoto, and
emphatically not a Sofascore rating (Serie A defenders average ~6.80 on Sofascore and
~6.05 on the Italian scale; feeding the former into a table whose top band starts at
7.00 prices every defence at maximum bonus).  So this module goes to the primary source.

Verified specimens 2026-08-26: Svilar 38 pg / mv 6.26, Dimarco 7 goals 17 assists,
Malen 18 pg / 14 goals -- the last cross-checked against Understat independently.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from curl_cffi import requests as creq

CACHE = Path("data/fantacalcio/voti")
URL = "https://www.fantacalcio.it/statistiche-serie-a/{season}/riepilogo/1"

# Same shape as the board's own recency decay, so the two agree on what "recent" means.
DECAY = {0: 1.00, 1: 0.62, 2: 0.34, 3: 0.18}

# Empirical-Bayes shrinkage strength, in games. A defender with 3 appearances at 7.0 is
# noise; k=12 pulls him most of the way back to his role mean, a 34-game regular barely
# moves. Chosen as roughly a third of a season -- the point where mv stops being sampling.
SHRINK_K = 12.0


def _parse(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for tr in soup.select("tr.player-row"):
        name_el = tr.select_one("a.player-name") or tr.select_one("th.player-name")
        role_el = tr.select_one("span.role")
        if name_el is None:
            continue
        rec = {
            "nome": name_el.get_text(strip=True),
            "R": (role_el.get("data-value") or "").upper() if role_el else None,
        }
        for td in tr.select("td[data-col-key]"):
            rec[td["data-col-key"]] = td.get_text(strip=True)
        out.append(rec)
    df = pd.DataFrame(out)
    for c in ("mv", "mfv"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False),
                                  errors="coerce")
    for c in ("pg", "gol", "ass", "amm", "esp", "gs", "rp"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "sq" in df.columns:
        df = df.rename(columns={"sq": "team"})
    return df


def fetch(season: str, refresh: bool = False) -> pd.DataFrame:
    """One season of per-player voto stats, cached to parquet."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"stats_{season.replace('-', '_')}.parquet"
    if p.exists() and not refresh:
        return pd.read_parquet(p)
    r = creq.get(URL.format(season=season), impersonate="chrome124", timeout=40)
    r.raise_for_status()
    df = _parse(r.text)
    # A schema break shows up as an empty frame or a missing mv column, not an exception.
    if df.empty or "mv" not in df.columns or df.mv.notna().sum() < 100:
        raise RuntimeError(f"statistiche parse looks broken for {season}: {df.shape}")
    df["season"] = season
    df.to_parquet(p, index=False)
    return df


def load(seasons: list[str], refresh: bool = False) -> pd.DataFrame:
    return pd.concat([fetch(s, refresh) for s in seasons], ignore_index=True)


def project(hist: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    """Decay-weighted media voto per player, un-shrunk. Keyed on NAME ONLY.

    Roles are reassigned every summer -- Neres was C in 2025-26 and is A on the 2026-27
    listone, likewise Cambiaghi, Cancellieri, Rodriguez Je. Joining vote history on
    (name, role) silently drops every player who was re-roled, which is exactly the set of
    players whose role change makes them interesting. Shrinkage needs a role prior, but it
    must be the role he will PLAY, so that happens at merge time in the board builder.
    """
    h = hist[hist.mv.notna() & hist.pg.notna() & (hist.pg > 0)].copy()
    idx = {s: i for i, s in enumerate(seasons)}
    h["w"] = h.season.map(lambda s: DECAY.get(idx.get(s, 99), 0.0))
    h = h[h.w > 0].copy()
    h["wn"] = h.w * h.pg

    g = h.groupby("nome", as_index=False).apply(
        lambda d: pd.Series({
            "mv_raw": np.average(d.mv, weights=d.wn),
            "pg_w": d.wn.sum(),
            "pg_last": d.loc[d.w.idxmax(), "pg"],
            "R_last": d.loc[d.w.idxmax(), "R"],
        }), include_groups=False)
    return g


def shrink(mv_raw, pg_w, role, role_mu, k: float = SHRINK_K):
    """Empirical-Bayes pull toward the mean of the role the player will actually play.

    A keeper's 6.30 and a defender's 6.05 are different baselines and must not share a
    prior. De Luca's 7.0 on one appearance is sampling noise, and without this he buys
    himself a starter slot.
    """
    mu = np.asarray([role_mu.get(r, np.nan) for r in role], dtype=float)
    pg = np.nan_to_num(np.asarray(pg_w, dtype=float))
    mv = np.asarray(mv_raw, dtype=float)
    out = (pg * mv + k * mu) / (pg + k)
    return np.where(np.isnan(mv), mu, out)


def role_means(hist: pd.DataFrame, seasons: list[str]) -> dict:
    """Games-weighted mean voto by role, on the SEASON's own role labels."""
    h = hist[hist.mv.notna() & hist.pg.notna() & (hist.pg > 0)].copy()
    idx = {s: i for i, s in enumerate(seasons)}
    h["w"] = h.season.map(lambda s: DECAY.get(idx.get(s, 99), 0.0))
    h = h[h.w > 0]
    h = h.assign(wn=h.w * h.pg)
    return h.groupby("R").apply(
        lambda d: float(np.average(d.mv, weights=d.wn)), include_groups=False).to_dict()


def role_sds(hist: pd.DataFrame) -> dict:
    """Week-to-week spread of a single player's voto, by role.

    Season means understate it badly, so widen the cross-sectional spread of season
    averages. Anchored at 0.45 as a floor -- a real weekly voto swings more than that.
    """
    h = hist[hist.mv.notna()]
    sd = h.groupby("R").mv.std().to_dict()
    return {r: max(0.45, float(v) * 1.8) for r, v in sd.items()}
