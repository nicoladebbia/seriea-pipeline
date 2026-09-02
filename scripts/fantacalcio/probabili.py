"""Probable lineups from fantacalcio.it, per player, for the coming giornata.

Structure verified against a live specimen 2026-09-02 (spec saved that session):
20 ``card team-card`` blocks (one per team), each with ``h6 team-name``,
``h6 team-formation``, a ``player-list starters`` ul (11 player links) and a
``player-list reserves`` ul; ballottaggio percentages live in ``ballot-list``
blocks OUTSIDE the team cards, so they are parsed page-wide and overlaid.
Player hrefs carry the same pid the voti pages and the auction board use, so
the join is by id, never by name.

The XI advisor uses this as its p(plays) source when a player is listed —
a page that says "titolare" beats an appearance-rate prior. The mapping
constants below are HEURISTICS, not measurements: after ~5 rounds compare
probabili status vs who actually got a voto and fit them properly.
"""
from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "fantacalcio" / "probabili.json"
URL = "https://www.fantacalcio.it/probabili-formazioni-serie-a"
CACHE_TTL_H = 6.0

INDISP_CACHE = ROOT / "data" / "fantacalcio" / "indisponibili.json"
INDISP_URL = "https://www.fantacalcio.it/indisponibili-serie-a"
# Availability tiers BELOW a probabili listing (pid-exact fresh beats
# name-matched). HEURISTICS with a refit path: every value lands in
# p_play_src, so pred_ledger's per-source calibration grades each tier
# against who actually got a voto once ~5 rounds are reconciled.
P_SUSPENDED = 0.02        # squalificato — deterministic
P_OUT = 0.05              # infortunato, prose reads as out
P_DOUBT = 0.35            # infortunato, prose reads as doubtful
P_NEWS_CAP = 0.60         # title-bound headline risk only — never a zeroing
# Default for an injured-list row is OUT — the 2026-09-02 specimen (43 real
# notes) shows most entries are hard outs ("forfait per domenica", "out
# nella 3a", "salter\xe0"), and anyone genuinely playing is rescued by his
# probabili listing anyway. Doubt needs an explicit THIS-round-hope marker;
# "da valutare"/"recuperabile" alone talk about the return DATE, not this
# match (the Geubbels trap: "terr\xe0 fuori luned\xec... tempi di recupero
# da valutare").
_DOUBT_RX = re.compile(
    r"in dubbio|ci prova|prover\xe0 a esserci|da valutare quotidianamente"
    r"|dovrebbe farcela|corsa contro il tempo|possibile convocazione"
    r"|recupero lampo|si tenta il recupero|verso la convocazione", re.I)
_TEAM_SPAN = re.compile(r'<span class="team-name">([^<]+)</span>')
_INDISP_ITEM = re.compile(
    r'<strong class="item-name">([^<]+)</strong>\s*'
    r'<div class="item-description">(.*?)</div>', re.S)

# p(gets a voto) by probabili status. Heuristic constants — see module docstring.
P_STARTER = 0.88          # listed titolare, no ballottaggio
P_RESERVE = 0.15          # listed panchina, no ballottaggio
BALLOT_CLAMP = (0.05, 0.95)

_CARD = re.compile(r'class="card team-card')
_NAME = re.compile(r'class="h6 team-name"[^>]*>\s*([^<]+)')
_FORM = re.compile(r'class="h6 team-formation"[^>]*>\s*([^<]+)')
_PLAYER = re.compile(
    r'/serie-a/squadre/[^/"]+/([^/"]+)/(\d+)"[^>]*>(.*?)</a>', re.S)
_SPAN = re.compile(r"<span>([^<]+)</span>")
_BALLOT = re.compile(r"ballot-list(.*?)</ul>", re.S)
_BALLOT_ROW = re.compile(
    r'/(\d+)"[^>]*>\s*<span>[^<]+</span>\s*</a>\s*'
    r'<strong class="percentage">(\d+)', re.S)
_MATCHWEEK = re.compile(r'class="matchweek">\s*(\d+)')

# A parse that yields fewer than this is a schema break, not a quiet week.
MIN_TEAMS = 16
MIN_STARTERS = 150


def _players_in(block: str) -> list[dict]:
    out = []
    for slug, pid, rest in _PLAYER.findall(block):
        m = _SPAN.search(rest)
        if m:
            out.append({"pid": int(pid), "slug": slug, "nome": m.group(1).strip()})
    return out


def parse(html: str) -> dict | None:
    """None on schema break — callers must fall back to the cache, never to {}."""
    starts = [m.start() for m in _CARD.finditer(html)]
    teams: dict = {}
    n_starters = 0
    for i, s in enumerate(starts):
        card = html[s:starts[i + 1] if i + 1 < len(starts) else s + 25000]
        name = _NAME.search(card)
        st_i, rs_i = card.find("player-list starters"), card.find("player-list reserves")
        if not name or st_i < 0 or rs_i < 0:
            continue
        starters = _players_in(card[st_i:rs_i])
        # Bound the reserves to their own </ul>: the card slice continues with
        # other player lists (indisponibili, squalificati, cross-team strips)
        # whose links would otherwise be swallowed as fake reserves — and a
        # fake reserve can overwrite a real starter in status_by_pid. Measured
        # 2026-09-02: 243 pids double-listed, Genoa starters inside Como.
        rs_end = card.find("</ul>", rs_i)
        reserves = _players_in(card[rs_i:rs_end if rs_end > 0 else len(card)])
        form = _FORM.search(card)
        teams[name.group(1).strip()] = {
            "formation": form.group(1).strip() if form else None,
            "starters": starters, "reserves": reserves,
        }
        n_starters += len(starters)
    if len(teams) < MIN_TEAMS or n_starters < MIN_STARTERS:
        return None
    ballots: dict = {}
    for block in _BALLOT.findall(html):
        for pid, pct in _BALLOT_ROW.findall(block):
            ballots[int(pid)] = int(pct)
    mw = _MATCHWEEK.search(html)
    return {"fetched_at": datetime.now(UTC).isoformat(),
            "matchweek": int(mw.group(1)) if mw else None,
            "teams": teams, "ballots": ballots}


def fetch_probabili(refresh: bool = False) -> dict | None:
    """Cached fetch. On any failure the last good cache is served — the page
    disappears for hours around deadline sometimes, and stale-but-real beats
    empty."""
    cached = None
    try:
        cached = json.loads(CACHE.read_text())
    except (OSError, ValueError):
        pass
    if cached and not refresh:
        try:
            age_h = (datetime.now(UTC)
                     - datetime.fromisoformat(cached["fetched_at"])).total_seconds() / 3600
            if age_h < CACHE_TTL_H:
                return cached
        except (KeyError, ValueError):
            pass
    try:
        from curl_cffi import requests as rq
        r = rq.get(URL, impersonate="chrome124", timeout=30)
        if r.status_code == 200:
            data = parse(r.text)
            if data is not None:
                CACHE.parent.mkdir(parents=True, exist_ok=True)
                CACHE.write_text(json.dumps(data, indent=1, ensure_ascii=False))
                return data
    except Exception:
        pass
    return cached


def parse_indisponibili(html: str) -> dict | None:
    """Per-club injured/suspended lists from the indisponibili page.

    Structure (specimen verified 2026-09-02): 20 <span class="team-name">
    blocks, each with Infortunati / Squalificati / Diffidati columns of
    <strong class="item-name"> + prose description. Names only — no pids —
    so consumers must match name WITHIN the club. Diffidati are skipped:
    the tracker's own card ledger is measured, the page is not needed.
    Sentinel: >=15 club blocks including Inter, else None (schema break).
    """
    spans = list(_TEAM_SPAN.finditer(html))
    teams: dict[str, list[dict]] = {}
    for i, m in enumerate(spans):
        block = html[m.end():spans[i + 1].start() if i + 1 < len(spans)
                     else len(html)]
        items: list[dict] = []
        for col in block.split('<div class="col">')[1:]:
            if "Squalificati" in col.split("<ul")[0]:
                cat = "squalificato"
            elif "Infortunati" in col.split("<ul")[0]:
                cat = "infortunato"
            else:
                continue                      # Diffidati or ad column
            for nome, desc in _INDISP_ITEM.findall(col):
                nome = unescape(nome)
                note = unescape(re.sub(r"<[^>]+>", "", desc))
                note = re.sub(r"\s+", " ", note).strip()
                status = cat
                if cat == "infortunato" and _DOUBT_RX.search(note):
                    status = "infortunato_dubbio"
                items.append({"nome": nome.strip(), "status": status,
                              "note": note})
        teams[m.group(1).strip()] = items
    if len(teams) < 15 or "Inter" not in teams:
        return None
    return {"fetched_at": datetime.now(UTC).isoformat(), "teams": teams}


def fetch_indisponibili(refresh: bool = False) -> dict | None:
    """Cached fetch, same contract as fetch_probabili: 6h TTL, on any
    failure the last good cache is served (stale-but-real beats empty)."""
    cached = None
    try:
        cached = json.loads(INDISP_CACHE.read_text())
    except (OSError, ValueError):
        pass
    if cached and not refresh:
        try:
            age_h = (datetime.now(UTC)
                     - datetime.fromisoformat(cached["fetched_at"])
                     ).total_seconds() / 3600
            if age_h < CACHE_TTL_H:
                return cached
        except (KeyError, ValueError):
            pass
    try:
        from curl_cffi import requests as rq
        r = rq.get(INDISP_URL, impersonate="chrome124", timeout=30)
        if r.status_code == 200:
            data = parse_indisponibili(r.text)
            if data is not None:
                INDISP_CACHE.parent.mkdir(parents=True, exist_ok=True)
                INDISP_CACHE.write_text(
                    json.dumps(data, indent=1, ensure_ascii=False))
                return data
    except Exception:
        pass
    return cached


def status_by_pid(data: dict | None) -> dict[int, dict]:
    """pid -> {"status": "starter"|"reserve", "ballot_pct": int|None, "team": name}."""
    out: dict[int, dict] = {}
    if not data:
        return out
    ballots = {int(k): v for k, v in (data.get("ballots") or {}).items()}
    for tname, t in (data.get("teams") or {}).items():
        for kind in ("starters", "reserves"):
            for p in t.get(kind, []):
                pid = int(p["pid"])
                # A starter listing is never downgraded by a later reserve
                # listing of the same pid (stale caches may still carry the
                # pre-2026-09-02 cross-card bleed).
                if kind == "reserves" and out.get(pid, {}).get("status") == "starter":
                    continue
                out[pid] = {
                    "status": "starter" if kind == "starters" else "reserve",
                    "ballot_pct": ballots.get(pid),
                    "team": tname,
                }
    return out


def p_play_override(pid: int, model_p: float, by_pid: dict[int, dict]) -> tuple[float, str]:
    """(p_play, source). Probabili wins when the player is listed; the model
    keeps players the page does not know about (it lists ~25 per club)."""
    info = by_pid.get(pid)
    if not info:
        return model_p, "model"
    pct = info.get("ballot_pct")
    if pct is not None:
        lo, hi = BALLOT_CLAMP
        return min(max(pct / 100.0, lo), hi), "ballottaggio"
    if info["status"] == "starter":
        return P_STARTER, "probabili"
    return P_RESERVE, "probabili"
