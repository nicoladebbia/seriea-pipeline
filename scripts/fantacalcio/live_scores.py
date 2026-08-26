"""Per-round Serie A voti and fantavoti, parsed from fantacalcio.it.

The scoring table below is not copied from the rules page -- it was RECONCILED against
936 of 942 published fantavoti in round 1 of 2025-26 (voto + bonuses == fantavoto to the
cent). The six that did not reconcile were the sv sentinel described below. Two things
that reconciliation caught, which reading the rules page would not have:

  * "Player of the match" appears as a bonus column but is worth ZERO in the fantavoto.
    Weighting it +1 put every match's best player one point too high -- 9 players a round.
  * Cards are NOT bonus columns. They live in the class of the grade span
    (`player-grade yellow-card`), so a parser that only reads `player-bonus` misses every
    booking silently.

The sv sentinel: a player with no vote is emitted as `data-value="55"`, not as an empty
value. Whole ratings render bare ("6" is 6.0) and halves carry a comma ("6,5"), so 55 is
simply out of the range a voto can occupy. Counted per round it matches the sv total
exactly (33/33, 36/36, 33/33 across rounds 1, 19 and 38). We therefore reject anything
outside [0, 10] rather than special-casing the literal -- a 55.0 voto reaching a team
total would be a silent catastrophe, and the range check cannot be fooled by a new
sentinel.

Three `.pill` blocks per player are three independent voti providers. They genuinely
disagree (143 of 314 rows in round 1), so which one you read is a real choice, not a
formality; `LIST_DEFAULT` is Fantacalcio Italia, which is what FantaLeghe uses.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "fantacalcio" / "voti"
URL = "https://www.fantacalcio.it/voti-fantacalcio-serie-a/{season}/{rnd}"

LIST_DEFAULT = 0          # 0 = Fantacalcio Italia, 1 = Statistico, 2 = Gazzetta

# Verified by reconciliation, not by reading the rules page. See module docstring.
BONUS = {
    "Gol segnati": 3.0,
    "Rigori segnati": 3.0,
    "Rigori parati": 3.0,
    "Assist": 1.0,
    "Gol subiti": -1.0,
    "Autoreti": -2.0,
    "Rigori sbagliati": -3.0,
    "Player of the match": 0.0,
}
CARD = {"yellow-card": -0.5, "red-card": -1.0}

# A round that parses fewer rows than this is treated as NOT PLAYED, never as "played and
# nobody appeared". A real round is ~310. The floor exists because an unplayed round and a
# broken selector both return HTTP 200 with zero rows, and only one of them is benign.
MIN_ROWS = 100

_ROW = re.compile(r"<tr>(.*?)</tr>", re.S)
_HREF = re.compile(r"/serie-a/squadre/([^/]+)/([^/]+)/(\d+)/")
_ROLE = re.compile(r'class="role"\s*data-value="([^"]*)"')
_PILL = re.compile(r'<div class="pill">(.*?)</div>', re.S)
_GRADE = re.compile(r'class="player-grade([^"]*)"\s*data-value="([^"]*)"')
_FANTA = re.compile(r'player-fanta-grade"\s*data-value="([^"]*)"')
_BONUS = re.compile(r'player-bonus[^>]*data-value="([^"]*)"[^>]*title="([^"]*)"')


def _dec(raw: str) -> float | None:
    """Italian decimal. No range guard -- a fantavoto is legitimately >10 or negative."""
    try:
        return float(raw.replace(",", "."))
    except ValueError:
        return None


def _voto(raw: str) -> float | None:
    """A voto, range-guarded. Anything outside [0, 10] is the sv sentinel, not a rating.

    The guard belongs HERE and nowhere else. Applying it to the fantavoto silently
    discards every double-figure haul and falls back to our own arithmetic -- which is
    precisely the substitution this module exists to avoid.
    """
    v = _dec(raw)
    return v if v is not None and 0.0 <= v <= 10.0 else None


def parse(html: str, lst: int = LIST_DEFAULT) -> pd.DataFrame:
    rows = []
    for chunk in _ROW.findall(html):
        m = _HREF.search(chunk)
        if not m:
            continue
        pills = _PILL.findall(chunk)
        if lst >= len(pills):
            continue
        g = _GRADE.search(pills[lst])
        f = _FANTA.search(pills[lst])
        if not g:
            continue
        voto = _voto(g.group(2))
        cards = sum(CARD.get(c, 0.0) for c in g.group(1).split())
        bon = {t: (_dec(v) or 0.0) for v, t in _BONUS.findall(chunk)}
        points = sum(BONUS.get(k, 0.0) * v for k, v in bon.items())
        role = _ROLE.search(chunk)
        published = _dec(f.group(1)) if f else None
        rows.append({
            "pid": int(m.group(3)), "slug": m.group(2), "team": m.group(1),
            "role": role.group(1).upper() if role else "?",
            "voto": voto, "cards": cards, "bonus": points,
            # The published fantavoto wins when the player has a voto; ours is only the
            # fallback, so a bonus column we do not know about cannot cost points silently.
            "fantavoto": None if voto is None
            else (published if published is not None else voto + cards + points),
            "played": voto is not None,
        })
    return pd.DataFrame(rows)


def fetch_round(season: str, rnd: int, refresh: bool = False) -> pd.DataFrame | None:
    """One round. Returns None when the round has not been played.

    Cached on disk because a played round never changes once the voti settle, and the
    tracker re-reads every round of the season on each run.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"round_{season.replace('-', '_')}_{rnd:02d}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    from curl_cffi import requests as rq
    r = rq.get(URL.format(season=season, rnd=rnd), impersonate="chrome124", timeout=30)
    if r.status_code != 200:
        return None
    df = parse(r.text)
    if len(df) < MIN_ROWS:
        return None
    df.to_parquet(path, index=False)
    return df


def played_rounds(season: str, upto: int = 38, refresh: bool = False) -> list[int]:
    """Rounds with published voti. Stops at the first unplayed round.

    Deliberately stops rather than scanning all 38: a mid-season gap would mean a schema
    break, and continuing past it would report the break as "those rounds were not played".
    """
    out = []
    for rnd in range(1, upto + 1):
        if fetch_round(season, rnd, refresh) is None:
            break
        out.append(rnd)
    return out
