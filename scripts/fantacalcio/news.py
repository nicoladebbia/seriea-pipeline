"""Player news for the saved squad, aggregated from several papers' RSS feeds.

Feeds were specimen-probed live 2026-09-02 (all 200 with ``<item>`` payloads).
Items are matched to roster players by SURNAME with word boundaries — a
listone name like "Adams A." matches "Adams" as a word, never as a substring
(so "Rrahmani" can't be hit by "Rahm"). This is display-only signal for the
page: nothing downstream computes on it, so a missed match costs a headline,
not a lineup.

Retention: items younger than RETENTION_DAYS, deduped by link, newest first.
"""
from __future__ import annotations

import html as html_mod
import json
import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "fantacalcio" / "news.json"

FEEDS = {
    "Gazzetta": "https://www.gazzetta.it/rss/calcio.xml",
    "Corriere dello Sport": "https://www.corrieredellosport.it/rss/calcio",
    "Tuttosport": "https://www.tuttosport.com/rss/calcio",
}
RETENTION_DAYS = 14
MAX_ITEMS = 200


def _surname(nome: str) -> str:
    """Listone display name -> the surname token worth matching.

    "Adams A." -> "Adams"; "Martinez Jo." -> "Martinez"; "Tiago Gabriel" ->
    "Tiago Gabriel" (two full words, keep both — matching just "Gabriel"
    would false-positive on every Gabriel in the league).
    """
    parts = [p for p in nome.split() if not re.fullmatch(r"[A-Z][a-z]?\.", p)]
    return " ".join(parts) or nome


def _matcher(roster: list[dict]) -> list[tuple[dict, re.Pattern]]:
    out = []
    for p in roster:
        sur = _surname(p["nome"])
        out.append((p, re.compile(r"\b" + re.escape(sur) + r"\b", re.I)))
    return out


def _parse_feed(xml_text: str, source: str) -> list[dict]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    items = []
    for it in root.iter("item"):
        title = html_mod.unescape(
            re.sub(r"<[^>]+>", " ", it.findtext("title") or "")).strip()
        desc = html_mod.unescape(
            re.sub(r"<[^>]+>", " ", it.findtext("description") or "").strip())
        link = (it.findtext("link") or "").strip()
        pub = it.findtext("pubDate")
        try:
            ts = parsedate_to_datetime(pub).astimezone(UTC).isoformat() if pub else None
        except (TypeError, ValueError):
            ts = None
        if title and link:
            items.append({"source": source, "title": title, "desc": desc[:280],
                          "link": link, "published": ts})
    return items


def fetch_news(roster: list[dict], refresh: bool = True) -> dict:
    """Fetch all feeds, match to roster, merge with cached items, persist.

    A feed that fails is skipped (the others still land); the cache is the
    accumulator, so transient feed outages only delay headlines.
    """
    try:
        cached = json.loads(CACHE.read_text()).get("items", [])
    except (OSError, ValueError):
        cached = []
    by_link = {c["link"]: c for c in cached}

    if refresh:
        from curl_cffi import requests as rq
        matchers = _matcher(roster)
        for source, url in FEEDS.items():
            try:
                r = rq.get(url, impersonate="chrome124", timeout=25)
                if r.status_code != 200:
                    continue
                for item in _parse_feed(r.text, source):
                    hay = f"{item['title']} {item['desc']}"
                    players = [p["nome"] for p, rx in matchers if rx.search(hay)]
                    if players:
                        item["players"] = sorted(set(players))
                        by_link[item["link"]] = item
            except Exception:
                continue

    cutoff = (datetime.now(UTC) - timedelta(days=RETENTION_DAYS)).isoformat()
    items = [i for i in by_link.values()
             if (i.get("published") or i.get("seen_at") or "9999") >= cutoff
             or not i.get("published")]
    for i in items:
        i.setdefault("seen_at", datetime.now(UTC).isoformat())
    items.sort(key=lambda i: i.get("published") or i["seen_at"], reverse=True)
    items = items[:MAX_ITEMS]
    out = {"generated_at": datetime.now(UTC).isoformat(), "items": items}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


# Escalation lexicon: a headline matched to a rostered player that also hits
# one of these patterns is a ROSTER RISK — pushed once per (player, category)
# by the tracker job, never silently scrolled away. Tight on purpose: a false
# push costs trust in the channel (2026-08-27 lesson), a missed one costs a
# headline that is still on the page.
_RISK = {
    "infortunio": re.compile(
        r"infortun|lesion|si ferma|frattura|crociat|operazi|stiramento"
        r"|risonanza|stop di \d|out (?:per )?\d|salta (?:il|la|le|due|tre)",
        re.I),
    "mercato-out": re.compile(
        r"cessione|addio|rescission|saudit|al[- ](?:hilal|ittihad|nassr|ahli"
        r"|sadd)|ufficiale il trasferimento|lascia (?:la|il|l')", re.I),
    "squalifica": re.compile(r"squalificat|giudice sportivo", re.I),
}


def classify_risk(item: dict) -> str | None:
    """First risk category the headline+description hits, else None."""
    hay = f"{item.get('title', '')} {item.get('desc', '')}"
    for cat, rx in _RISK.items():
        if rx.search(hay):
            return cat
    return None


def risk_hits(item: dict) -> list[tuple[str, str]]:
    """(player, category) pairs worth alerting for one news item.

    The player's surname must be in the TITLE — a body-only mention is
    usually somebody else's story ("David to Atletico" name-dropping coach
    Simeone tagged MY Simeone; pushed once 2026-09-02 before this guard).
    """
    cat = classify_risk(item)
    if not cat:
        return []
    title = item.get("title", "")
    return [(nome, cat) for nome in item.get("players", [])
            if re.search(r"\b" + re.escape(_surname(nome)) + r"\b",
                         title, re.I)]


def roster_for_news() -> list[dict]:
    """The saved squad joined to board names — the shape fetch_news wants."""
    board = json.loads((ROOT / "data" / "fantacalcio" / "auction_board.json").read_text())
    by_id = {int(p["id"]): p for p in board["players"]}
    team = json.loads((ROOT / "data" / "fantacalcio" / "my_team.json").read_text())
    out = []
    for r in team.get("roster", []):
        p = by_id.get(int(r["id"]))
        if p:
            out.append({"id": int(p["id"]), "nome": p["nome"], "R": p["R"],
                        "team": p["team"]})
    return out


if __name__ == "__main__":
    res = fetch_news(roster_for_news())
    print(f"{len(res['items'])} matched items")
    for i in res["items"][:10]:
        print(f"  [{i['source']}] {', '.join(i['players'])}: {i['title'][:80]}")
