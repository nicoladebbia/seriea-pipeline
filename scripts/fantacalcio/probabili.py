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

# Second probabili source: SosFanta's per-giornata page. Independent
# editorial desk, same shape of signal (per-player titolarita %). Names
# only, no pids (join by name within the club, like the indisponibili
# page). Specimen verified live 2026-09-03: 10 match blocks, 20 teams,
# Titolari/Ballottaggi/Panchina/In dubbio/Indisponibili sections, pct
# badges per player (Paz N. 95, Diao 55), ballots as "A - B" 60-40 pairs.
# NOTE the similarly-named /news/probabili-formazioni-serie-a/ URL is a
# FOSSIL — it serves a 2015 article (datePublished checked); only the
# lista-formazioni path is the live product.
SOSFANTA_CACHE = ROOT / "data" / "fantacalcio" / "sosfanta.json"
SOSFANTA_URL = ("https://www.sosfanta.com/lista-formazioni/"
                "probabili-formazioni-serie-a/")

# Penalty takers, same publisher and card markup as the probabili page.
# Specimen verified live 2026-09-03 (freshness by CONTENT, not names alone:
# Dovbyk under Bologna, Kean under Como, Krstovic under Atalanta — all
# summer-2026 moves listed at the NEW club). Player hrefs carry pids, so
# the join is by id. The page renders one ranked "Rigori" <ol> per team.
RIGORISTI_CACHE = ROOT / "data" / "fantacalcio" / "rigoristi.json"
RIGORISTI_URL = "https://www.fantacalcio.it/rigoristi-serie-a"
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
_PCT = re.compile(r'aria-valuenow="(\d{1,3})"')
_BALLOT = re.compile(r"ballot-list(.*?)</ul>", re.S)
_BALLOT_ROW = re.compile(
    r'/(\d+)"[^>]*>\s*<span>[^<]+</span>\s*</a>\s*'
    r'<strong class="percentage">(\d+)', re.S)
_MATCHWEEK = re.compile(r'class="matchweek">\s*(\d+)')

# A parse that yields fewer than this is a schema break, not a quiet week.
MIN_TEAMS = 16
MIN_STARTERS = 150


def _players_in(block: str) -> list[dict]:
    """Each <li> also carries the page's own titolarita progress bar
    (aria-valuenow, verified live 2026-09-02: Bijlow 90, reserves 5) — the
    per-player measurement that replaces the flat P_STARTER/P_RESERVE."""
    out = []
    ms = list(_PLAYER.finditer(block))
    for j, m in enumerate(ms):
        sp = _SPAN.search(m.group(3))
        if not sp:
            continue
        tail = block[m.end():ms[j + 1].start() if j + 1 < len(ms)
                     else len(block)]
        pm = _PCT.search(tail)
        row = {"pid": int(m.group(2)), "slug": m.group(1),
               "nome": sp.group(1).strip()}
        if pm:
            row["pct"] = int(pm.group(1))
        out.append(row)
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


_RIG_COL = re.compile(
    r'<header class="primary">Rigori</header>(.*?)</ol>', re.S)


def parse_rigoristi(html: str) -> dict | None:
    """None on schema break — callers must fall back to the cache, never {}.

    {"teams": {team: [{"pid", "nome", "rank"}, ...]}} — rank is 1-based
    list position (the page's own ordering; #1 is the designated taker)."""
    teams: dict[str, list[dict]] = {}
    for card in html.split('class="card team-card"')[1:]:
        tn = _TEAM_SPAN.search(card)
        col = _RIG_COL.search(card)
        if not tn or not col:
            continue
        takers = []
        for k, m in enumerate(_PLAYER.finditer(col.group(1))):
            sp = _SPAN.search(m.group(3))
            if sp:
                takers.append({"pid": int(m.group(2)),
                               "nome": unescape(sp.group(1)).strip(),
                               "rank": k + 1})
        teams[unescape(tn.group(1)).strip()] = takers
    if len(teams) < MIN_TEAMS or sum(1 for t in teams.values() if t) < MIN_TEAMS:
        return None
    return {"fetched_at": datetime.now(UTC).isoformat(), "teams": teams}


def rigoristi_by_pid(data: dict | None) -> dict[int, int]:
    """pid -> rank (1 = designated taker)."""
    out: dict[int, int] = {}
    for takers in ((data or {}).get("teams") or {}).values():
        for t in takers:
            out[int(t["pid"])] = int(t["rank"])
    return out


_SF_TEAM = re.compile(
    r'<h2 class="[^"]*truncate">([^<]+)</h2>\s*'
    r'<span class="[^"]*text-primary">([^<]+)</span>', re.S)
_SF_TIME = re.compile(r'<time datetime="([^"]+)"')
_SF_H3 = re.compile(
    r"<h3[^>]*>\s*(Titolari|Ballottaggi|Panchina|In dubbio|Indisponibili)"
    r"\s*</h3>")
_SF_UL = re.compile(r'<ul class="flex flex-col min-w-0">(.*?)</ul>', re.S)
_SF_PLAYER = re.compile(
    r'>\s*(\d{1,3})%\s*</span>\s*<span[^>]*truncate">([^<]+)</span>')
_SF_BALLOT = re.compile(
    r'>\s*(\d{1,3})%\s*</span>\s*<span aria-hidden="true">-</span>\s*'
    r'<span[^>]*>\s*(\d{1,3})%\s*</span>\s*</div>\s*'
    r'<span[^>]*truncate">\s*([^<]+?)\s*</span>', re.S)
_SF_NAME = re.compile(r'<span[^>]*truncate">\s*([^<]+?)\s*</span>')
_SF_STATUS = re.compile(r'title="([^"]+)"')
_SF_NOTE = re.compile(r'text-\[#7b809a\]">([^<]*)</span>')


def parse_sosfanta(html: str) -> dict | None:
    """None on schema break — callers must fall back to the cache, never {}.

    Match blocks are segmented by header h2 pairs (home, away — each h2 is
    followed by its formation span, which is what separates a team header
    from nav h2s). Every section is a grid of two <ul>s, ul[0]=home,
    ul[1]=away (verified: Como players carry items-end/text-right)."""
    heads = list(_SF_TEAM.finditer(html))
    if len(heads) < 2 or len(heads) % 2:
        return None
    teams: dict = {}
    matches: list[dict] = []
    n_pct = 0
    for k in range(0, len(heads), 2):
        block = html[heads[k].start():
                     heads[k + 2].start() if k + 2 < len(heads) else len(html)]
        pair = [unescape(heads[k + i].group(1)).strip() for i in (0, 1)]
        forms = [heads[k + i].group(2).strip() for i in (0, 1)]
        tm = _SF_TIME.search(block)
        kick = tm.group(1) if tm else None
        matches.append({"home": pair[0], "away": pair[1], "kickoff": kick})
        sides = [{"formation": forms[i], "kickoff": kick, "players": {},
                  "out": [], "doubt": []} for i in (0, 1)]
        secs = list(_SF_H3.finditer(block))
        for si, sm in enumerate(secs):
            chunk = block[sm.end():secs[si + 1].start() if si + 1 < len(secs)
                          else len(block)]
            uls = _SF_UL.findall(chunk)[:2]
            for side, ul in zip(sides, uls):
                if sm.group(1) in ("Titolari", "Panchina"):
                    for pct, nome in _SF_PLAYER.findall(ul):
                        side["players"].setdefault(
                            unescape(nome).strip(), int(pct))
                        n_pct += 1
                elif sm.group(1) == "Ballottaggi":
                    for p1, p2, names in _SF_BALLOT.findall(ul):
                        parts = unescape(names).split(" - ")
                        if len(parts) == 2:
                            for nome, pct in zip(parts, (p1, p2)):
                                side["players"].setdefault(
                                    nome.strip(), int(pct))
                else:                       # In dubbio / Indisponibili
                    for li in ul.split("<li")[1:]:
                        nm = _SF_NAME.search(li)
                        if not nm:
                            continue
                        st = _SF_STATUS.search(li)
                        nt = _SF_NOTE.search(li)
                        row = {"nome": unescape(nm.group(1)).strip(),
                               "status": (st.group(1).lower()
                                          if st else "indisponibile"),
                               "note": unescape(nt.group(1)).strip()
                                       if nt else ""}
                        key = ("doubt" if sm.group(1) == "In dubbio"
                               else "out")
                        side[key].append(row)
        for name, side in zip(pair, sides):
            teams[name] = side
    if len(teams) < MIN_TEAMS or n_pct < MIN_STARTERS:
        return None
    return {"fetched_at": datetime.now(UTC).isoformat(),
            "matches": matches, "teams": teams}


def feed_age_h(data: dict | None) -> float | None:
    """Hours since the feed was actually FETCHED (None when unknown). The
    caches below serve stale-forever on failure by design, so every consumer
    that shows a human the data must also show this age — a wedged scraper
    otherwise looks exactly like a quiet news day."""
    try:
        return (datetime.now(UTC)
                - datetime.fromisoformat(data["fetched_at"])
                ).total_seconds() / 3600
    except (TypeError, KeyError, ValueError):
        return None


def _cached_feed(url: str, cache: Path, parse_fn,
                 refresh: bool = False) -> dict | None:
    """One cached-fetch contract for every feed in this module: 6h TTL, and
    on ANY failure (network, 403, schema break) the last good cache is
    served — the pages disappear for hours around deadline sometimes, and
    stale-but-real beats empty. feed_age_h is how consumers show staleness."""
    cached = None
    try:
        cached = json.loads(cache.read_text())
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
        r = rq.get(url, impersonate="chrome124", timeout=30)
        if r.status_code == 200:
            data = parse_fn(r.text)
            if data is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(data, indent=1,
                                            ensure_ascii=False))
                return data
    except Exception:
        pass
    return cached


def fetch_probabili(refresh: bool = False) -> dict | None:
    return _cached_feed(URL, CACHE, parse, refresh)


def fetch_sosfanta(refresh: bool = False) -> dict | None:
    return _cached_feed(SOSFANTA_URL, SOSFANTA_CACHE, parse_sosfanta, refresh)


def fetch_rigoristi(refresh: bool = False) -> dict | None:
    return _cached_feed(RIGORISTI_URL, RIGORISTI_CACHE, parse_rigoristi,
                        refresh)


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
                # OPEN VERIFICATION: every Squalificati column on the
                # 2026-09-02 specimen was empty ("Nessuno"), so the
                # non-empty markup is asserted from the Infortunati twin,
                # not observed. Low exposure: SA bans are independently
                # measured by the discipline ledger (voti cards).
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
    return _cached_feed(INDISP_URL, INDISP_CACHE, parse_indisponibili,
                        refresh)


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
                    "pct": p.get("pct"),
                    "team": tname,
                }
    return out


def p_play_override(pid: int, model_p: float, by_pid: dict[int, dict]) -> tuple[float, str]:
    """(p_play, source). Probabili wins when the player is listed; the model
    keeps players the page does not know about (it lists ~25 per club)."""
    info = by_pid.get(pid)
    if not info:
        return model_p, "model"
    lo, hi = BALLOT_CLAMP
    pct = info.get("ballot_pct")
    if pct is not None:
        return min(max(pct / 100.0, lo), hi), "ballottaggio"
    # The page's own per-player titolarita bar: a measurement, not a flat
    # constant — its own ledger bucket ("titolarita") so calibration can
    # judge it separately from the P_STARTER/P_RESERVE fallback.
    if info.get("pct") is not None:
        return min(max(info["pct"] / 100.0, lo), hi), "titolarita"
    if info["status"] == "starter":
        return P_STARTER, "probabili"
    return P_RESERVE, "probabili"
