"""Sofascore fetchers for WC2026 via the www.sofascore.com/api/v1 proxy.

Discovery (2026-06-10): api.sofascore.com is TCP-unreachable from this
network, but www.sofascore.com/api/v1 serves the identical API with no
Cloudflare challenge — including upcoming-event ODDS and lineups. This module
is the production home for that access pattern (the original bulk scrape was
a one-shot per repo policy; this stays because the tournament needs daily
refreshes: odds, confirmed lineups, new played matches).

CLI:
  python3 -m scripts.worldcup.sofascore_fetch --odds
      Resolve all known fixtures to Sofascore event ids and write
      data/worldcup/market_odds.json (1X2 decimal odds + de-vigged implied
      probabilities). Display-only market comparison — NOT blended into the
      model (no validated blend weight exists yet).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any

from scripts.worldcup.engine import atomic_write_json, canon_team, read_json_safe


def sofa_canon(name: str) -> str:
    """canon_team with Sofascore's spelling quirks pre-normalized:
    'Bosnia & Herzegovina' (ampersand) and curly apostrophes (’)."""
    return canon_team(name.replace("’", "'").replace(" & ", " and "))

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "worldcup"
FIXTURES_JSON = DATA_DIR / "fixtures.json"
MARKET_ODDS_JSON = DATA_DIR / "market_odds.json"
SOFA_TEAM_DIR = DATA_DIR / "sofascore_intl"

BASE = "https://www.sofascore.com/api/v1"
DELAY_RANGE = (2.0, 4.5)


def _session() -> Any:
    from curl_cffi import requests as creq

    return creq.Session(impersonate="chrome")


_CONSEC_FAILURES = 0
_BREAKER_LIMIT = 8


class SourceDownError(RuntimeError):
    """Raised when the host has failed _BREAKER_LIMIT consecutive requests —
    callers save partials and stop instead of crawling through a dead network."""


def get_json(session: Any, path: str) -> dict[str, Any] | None:
    """Polite GET with one retry; None on failure (caller decides severity).
    Trips SourceDownError after _BREAKER_LIMIT consecutive failures."""
    global _CONSEC_FAILURES
    if _CONSEC_FAILURES >= _BREAKER_LIMIT:
        raise SourceDownError(
            f"{_BREAKER_LIMIT} consecutive Sofascore failures — breaker open"
        )
    for attempt in (1, 2):
        try:
            r = session.get(f"{BASE}/{path}", timeout=20)
            if r.status_code == 200:
                data = dict(r.json())  # parse BEFORE resetting: poisoned-200
                _CONSEC_FAILURES = 0   # challenge pages must count as failures
                return data
            if r.status_code == 404:
                # resource not published yet (pre-kickoff lineups) — the host
                # is healthy; never count toward the breaker (sibling scraper
                # learned this lesson: sofascore_events.py)
                _CONSEC_FAILURES = 0
                return None
            if r.status_code in (403, 429):
                time.sleep(10.0 * attempt)
                continue
            _CONSEC_FAILURES += 1
            return None
        except Exception:  # noqa: BLE001 — transport errors vary by backend; retry-once then None
            time.sleep(3.0 * attempt)
    _CONSEC_FAILURES += 1
    return None


def _sleep() -> None:
    # politeness jitter, not crypto
    time.sleep(random.uniform(*DELAY_RANGE))  # noqa: S311


def team_ids_from_scrape() -> dict[str, int]:
    """Team display name -> Sofascore team id, from the per-team scrape JSONs."""
    out: dict[str, int] = {}
    fixtures = json.loads(FIXTURES_JSON.read_text())
    display_names = {
        str(t)
        for f in fixtures
        if f["stage"] == "group"
        for t in (f["home"], f["away"])
    }
    by_ascii = {p.stem: p for p in SOFA_TEAM_DIR.glob("*.json")}
    for name in display_names:
        ascii_key = (
            name.replace("ô", "o").replace("ç", "c").replace("ü", "u")
            .replace("'", "_").replace(" ", "_").replace("é", "e")
        )
        path = by_ascii.get(ascii_key) or by_ascii.get(name.replace(" ", "_"))
        if path is None:
            continue
        blob = json.loads(path.read_text())
        tid = blob.get("team_id") or blob.get("meta", {}).get("team_id")
        if tid:
            out[name] = int(tid)
    return out


def fractional_to_decimal(value: str) -> float:
    """Sofascore fractionalValue ('21/50') -> decimal odds (1.42)."""
    return float(1 + Fraction(value))


def parse_1x2(odds_payload: dict[str, Any]) -> dict[str, float] | None:
    """Extract the full-time 1X2 market as decimal odds {home, draw, away}."""
    for market in odds_payload.get("markets", []):
        if market.get("marketName") == "Full time" and not market.get("isLive"):
            decs: dict[str, float] = {}
            key_map = {"1": "home", "X": "draw", "2": "away"}
            for choice in market.get("choices", []):
                side = key_map.get(str(choice.get("name", "")))
                if side and choice.get("fractionalValue"):
                    decs[side] = round(
                        fractional_to_decimal(choice["fractionalValue"]), 3
                    )
            if len(decs) == 3:
                return decs
    return None


def devig(decimal_odds: dict[str, float]) -> dict[str, float]:
    """Implied probabilities, overround removed proportionally."""
    raw = {k: 1.0 / v for k, v in decimal_odds.items()}
    total = sum(raw.values())
    return {k: round(v / total, 4) for k, v in raw.items()}


def fetch_market_odds() -> dict[str, Any]:
    """Resolve fixtures -> Sofascore events -> 1X2 odds. Incremental-merge:
    keeps previously fetched matches when a refresh run misses them."""
    fixtures = json.loads(FIXTURES_JSON.read_text())
    team_ids = team_ids_from_scrape()
    session = _session()

    # fixture lookup: (canon_home, canon_away) -> (match_number, date).
    # Group-stage pairings are unique, so the date is a sanity check (±1 day
    # tolerated — late local kickoffs cross UTC midnight between sources),
    # not part of the join key.
    fix_key: dict[tuple[str, str], tuple[int, str]] = {}
    for f in fixtures:
        fix_key[(canon_team(str(f["home"])), canon_team(str(f["away"])))] = (
            int(f["match_number"]),
            str(f["date_utc"])[:10],
        )

    events = _collect_wc_events(session, team_ids)

    out: dict[str, Any] = dict(read_json_safe(MARKET_ODDS_JSON, {}))  # type: ignore[arg-type]
    matched = 0
    breaker_tripped = False
    for eid, e in sorted(events.items()):
        date = datetime.fromtimestamp(int(e["startTimestamp"]), tz=UTC)
        key = (
            sofa_canon(str(e["homeTeam"]["name"])),
            sofa_canon(str(e["awayTeam"]["name"])),
        )
        hit = fix_key.get(key)
        if hit is None:
            continue
        match_number, fix_date = hit
        gap_days = abs(
            (date.date() - datetime.strptime(fix_date, "%Y-%m-%d").date()).days
        )
        if gap_days > 1:
            continue  # same pairing, wrong occasion — not a group fixture
        try:
            payload = get_json(session, f"event/{eid}/odds/1/all")
        except SourceDownError as exc:
            print(f"breaker: {exc} — saving {matched} odds fetched so far")
            breaker_tripped = True
            break
        _sleep()
        if payload is None:
            continue
        decs = parse_1x2(payload)
        if decs is None:
            continue
        out[str(match_number)] = {
            "event_id": eid,
            "odds": decs,
            "implied": devig(decs),
            "fetched_at": datetime.now(UTC).isoformat(),
        }
        matched += 1

    atomic_write_json(MARKET_ODDS_JSON, out)
    print(
        f"market odds: {matched} fetched this run, {len(out)} total "
        f"-> {MARKET_ODDS_JSON}"
    )
    if breaker_tripped and matched == 0:
        raise SystemExit(1)  # refresh.py keys last_odds_fetch off the rc
    return out


CONFIRMED_LINEUPS_JSON = DATA_DIR / "confirmed_lineups.json"

# Sofascore missing-player reason codes (same map the Serie A
# lineup_predictor uses); type "doubtful" arrives alongside "missing".
MISSING_REASON_MAP = {1: "Injury", 2: "Suspension", 3: "National team",
                      4: "Personal", 5: "Other", 13: "Suspension"}

SOFA_PARQUET = DATA_DIR / "sofascore_intl_player_stats.parquet"

# Historical closing odds for the backtest tournaments (blend validation).
HISTORICAL_ODDS_JSON = DATA_DIR / "historical_odds.json"


def _player_stat_rows(
    e: dict[str, Any],
    lineups: dict[str, Any],
    side: str,
    team: str,
    tid: int,
    now_iso: str,
    source_url: str,
) -> list[dict[str, Any]]:
    """One side of an event-lineups payload -> player-stats parquet rows.
    Shared by the API route and the match-page __NEXT_DATA__ route."""
    is_home = side == "home"
    date = datetime.fromtimestamp(int(e["startTimestamp"]), tz=UTC)
    our_score = (e.get("homeScore" if is_home else "awayScore") or {}).get("current")
    opp_score = (e.get("awayScore" if is_home else "homeScore") or {}).get("current")
    rows: list[dict[str, Any]] = []
    for p in (lineups.get(side) or {}).get("players", []):
        stats = p.get("statistics") or {}
        minutes = stats.get("minutesPlayed", 0) or 0
        rows.append(
            {
                "team": team,
                "team_id": tid,
                "match_id": int(e["id"]),
                "date": date.isoformat(),  # '+00:00' form, matches existing rows
                "tournament": str((e.get("tournament") or {}).get("name", "")),
                "opponent": str(
                    (e.get("awayTeam" if is_home else "homeTeam") or {}).get(
                        "name", ""
                    )
                ),
                "home_or_away": side,
                "our_score": our_score,
                "opp_score": opp_score,
                "player_name": str(p["player"]["name"]),
                "norm_name": normalize_simple(str(p["player"]["name"])),
                "norm_name_sorted": norm_sorted_simple(str(p["player"]["name"])),
                "player_id": int(p["player"].get("id", 0)),
                "position": str(p["player"].get("position", "")),
                "started": (not p.get("substitute", True)) and minutes > 0,
                "minutes": minutes,
                "shots": (stats.get("totalShots", 0) or 0)
                if minutes > 0 else None,
                "shots_on_target": (
                    stats.get("onTargetScoringAttempt", 0) or 0
                ) if minutes > 0 else None,
                "goals": (stats.get("goals", 0) or 0) if minutes > 0 else None,
                "rating": stats.get("rating"),
                "scraped_at": now_iso,
                "source_url": source_url,
            }
        )
    return rows


def fetch_played_stats() -> int:
    """Append per-player stats for newly PLAYED matches (WC + any other
    internationals) to the scrape parquet. This is what feeds (a) the
    mid-tournament re-gating of the shots-floor market (3+ real WC matches
    per team) and (b) starter-dampening freshness. Idempotent: existing
    (team, match_id) pairs are skipped. Returns number of rows appended.

    When the API tier is 403-banned, falls back to finished WC matches on
    the daily-schedule HTML pages and their match-page __NEXT_DATA__
    lineups (statistics ride on the same embedded payload). The fallback
    covers WC fixture dates only — other internationals resume when the
    API does."""
    import pandas as pd

    df = pd.read_parquet(SOFA_PARQUET)
    seen = set(zip(df["team"], df["match_id"], strict=True))
    team_ids = team_ids_from_scrape()
    session = _session()

    new_rows: list[dict[str, Any]] = []
    now_iso = datetime.now(UTC).isoformat()
    breaker_open = False
    api_alive = False
    for team, tid in sorted(team_ids.items()):
        if breaker_open:
            break
        try:
            payload = get_json(session, f"team/{tid}/events/last/0")
        except SourceDownError as exc:
            print(f"breaker: {exc} — appending {len(new_rows)} rows collected so far")
            break
        _sleep()
        if payload is None:
            continue
        api_alive = True
        for e in payload.get("events", []):
            eid = int(e["id"])
            if (team, eid) in seen:
                continue
            if str(e.get("status", {}).get("type")) != "finished":
                continue
            is_home = int(e["homeTeam"].get("id", -1)) == tid
            try:
                lineups = get_json(session, f"event/{eid}/lineups")
            except SourceDownError:
                breaker_open = True
                print(f"breaker open — appending {len(new_rows)} rows collected so far")
                break
            _sleep()
            if lineups is None:
                continue
            side = "home" if is_home else "away"
            new_rows.extend(
                _player_stat_rows(
                    e, lineups, side, team, tid, now_iso,
                    f"{BASE}/event/{eid}/lineups",
                )
            )
            seen.add((team, eid))

    if not api_alive:
        # API dark — finished WC matches off the page tier; statistics ride
        # on the same embedded lineups payload the match pages carry.
        id_to_name = {tid_: name for name, tid_ in team_ids.items()}
        fixtures = json.loads(FIXTURES_JSON.read_text())
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        cutoff = (datetime.now(UTC) - timedelta(days=3)).strftime("%Y-%m-%d")
        dates = sorted({
            d for f in fixtures
            if cutoff <= (d := str(f.get("date_utc", ""))[:10]) <= today
        })
        for eid, e in sorted(_day_page_wc_events(session, dates).items()):
            if str((e.get("status") or {}).get("type")) != "finished":
                continue
            url = _match_page_url(e)
            if url is None:
                continue
            lineups = _lineups_from_next_data(_get_html(session, url))
            if lineups is None:
                continue
            has_stats = any(
                (p.get("statistics") or {}).get("minutesPlayed") is not None
                for s in ("home", "away")
                for p in (lineups.get(s) or {}).get("players", [])
            )
            if not has_stats:
                continue  # pre-match page — no real statistics yet
            for side, tkey in (("home", "homeTeam"), ("away", "awayTeam")):
                team = id_to_name.get(int((e.get(tkey) or {}).get("id", 0) or 0))
                if team is None or (team, eid) in seen:
                    continue
                new_rows.extend(
                    _player_stat_rows(
                        e, lineups, side, team, team_ids[team], now_iso, url
                    )
                )
                seen.add((team, eid))

    if new_rows:
        import pandas as pd

        updated = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        updated.to_parquet(SOFA_PARQUET, index=False)
    print(f"played-stats: {len(new_rows)} player-match rows appended")
    return len(new_rows)


SOFA_RESULTS_JSON = DATA_DIR / "sofascore_results.json"


def _event_to_result(
    e: dict[str, Any], id_to_name: dict[int, str], wc_start: str
) -> dict[str, Any] | None:
    """One Sofascore event object -> a stored result record, or None.

    Accepts only: this World Cup (tournament name + on/after the opening
    day), status=finished (never a live score), both team ids mapped to
    fixture display names, both scores present. Shared by the API and the
    __NEXT_DATA__ HTML routes — one parser, two transports.
    """
    if "World Cup" not in str((e.get("tournament") or {}).get("name", "")):
        return None
    if str((e.get("status") or {}).get("type", "")) != "finished":
        return None
    ts = int(e.get("startTimestamp", 0) or 0)
    date = datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d")
    if date < wc_start:
        return None  # qualifiers/friendlies also carry "World Cup"
    home = id_to_name.get(int((e.get("homeTeam") or {}).get("id", 0) or 0))
    away = id_to_name.get(int((e.get("awayTeam") or {}).get("id", 0) or 0))
    hs = e.get("homeScore") or {}
    as_ = e.get("awayScore") or {}
    if not home or not away or hs.get("current") is None or as_.get("current") is None:
        return None
    winner_code = int(e.get("winnerCode", 0) or 0)
    pens = hs.get("penalties") is not None or as_.get("penalties") is not None
    return {
        "event_id": int(e["id"]),
        "date": date,
        "home": home,
        "away": away,
        # ET-inclusive, like results.csv knockout scores
        "home_score": int(hs["current"]),
        "away_score": int(as_["current"]),
        "winner": home if winner_code == 1 else away if winner_code == 2 else None,
        "decided_by": "PEN" if pens else "FT",
        "penalties": [hs.get("penalties"), as_.get("penalties")] if pens else None,
    }


_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.DOTALL
)


def _next_data_blob(html: str) -> Any | None:
    """Parsed __NEXT_DATA__ JSON of a sofascore page, or None."""
    m = _NEXT_DATA_RE.search(html)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _events_from_next_data(html: str) -> list[dict[str, Any]]:
    """All event-shaped objects in a sofascore page's __NEXT_DATA__ blob,
    found by SHAPE, not by path — the pageProps nesting moved in 2026-06
    already (see the Serie A parsers); the event object shape hasn't."""
    blob = _next_data_blob(html)
    if blob is None:
        return []
    events: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if (
                "homeTeam" in node and "awayTeam" in node
                and "status" in node and "id" in node and "homeScore" in node
            ):
                events.append(node)
                return  # event objects don't nest further events
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(blob)
    return events


def _lineups_from_next_data(html: str) -> dict[str, Any] | None:
    """The api/v1 event-lineups object embedded in a match page — found by
    SHAPE: a {home, away} pair whose sides both carry a players list of
    {player: {...}} rows. Statistics period objects also use home/away keys
    but hold numbers, not player lists, so they can't false-positive."""
    blob = _next_data_blob(html)
    if blob is None:
        return None
    hits: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            h, a = node.get("home"), node.get("away")
            if (
                isinstance(h, dict) and isinstance(a, dict)
                and isinstance(h.get("players"), list)
                and isinstance(a.get("players"), list)
            ):
                hits.append(node)
                return
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(blob)
    for cand in hits:
        if all(
            isinstance(p, dict) and isinstance(p.get("player"), dict)
            for side in ("home", "away")
            for p in cand[side]["players"][:5]
        ):
            return cand
    return None


def _get_html(session: Any, url: str) -> str:
    """Polite page GET — empty string on any failure (page tier has no
    breaker; a missing page is just skipped)."""
    try:
        r = session.get(url, timeout=25)
        html = str(r.text) if r.status_code == 200 else ""
    except Exception:  # noqa: BLE001 — transport varies; page skip is safe
        html = ""
    _sleep()
    return html


def _match_page_url(e: dict[str, Any]) -> str | None:
    """Event object → www.sofascore.com match-page URL (slug + customId
    ride on every event object the day pages embed)."""
    slug, cid = e.get("slug"), e.get("customId")
    if not slug or not cid:
        return None
    return f"https://www.sofascore.com/football/match/{slug}/{cid}"


def _day_page_wc_events(
    session: Any, dates: list[str]
) -> dict[int, dict[str, Any]]:
    """World Cup event objects from the daily-schedule HTML pages — the
    transport that survives API-tier 403 bans. Deduped by event id."""
    events: dict[int, dict[str, Any]] = {}
    for date in dates:
        html = _get_html(session, f"https://www.sofascore.com/football/{date}")
        if not html:
            print(f"html route: {date} page unavailable")
            continue
        page_events = _events_from_next_data(html)
        n = 0
        for e in page_events:
            if "World Cup" not in str((e.get("tournament") or {}).get("name", "")):
                continue
            try:
                eid = int(e["id"])
            except (KeyError, TypeError, ValueError):
                continue
            events[eid] = e
            n += 1
        print(f"html route: {date} — {len(page_events)} events on page, {n} WC")
    return events


def fetch_results() -> dict[str, Any]:
    """Final scores of played WC matches — the same-night fallback for
    results.csv (martj42 lags by hours-to-a-day; Sofascore is live).

    Two routes, same parser: the API team event lists when the API is
    healthy, else the www.sofascore.com daily-schedule HTML pages
    (__NEXT_DATA__ blob) — the canonical 403 playbook from the Serie A
    scrapers; the page tier survives the API-tier Cloudflare bans.

    Only status=finished events are stored, joined by TEAM ID (never by
    name) and oriented exactly as Sofascore reports home/away. Knockout
    'current' scores are ET-inclusive with penalties recorded separately —
    the same semantics as results.csv + shootouts.csv, so the merged lookup
    in scripts.worldcup.knockout can treat the two sources interchangeably,
    CSV always first. The store is union-merged by event id: a breaker trip
    mid-run keeps everything already collected.
    """
    team_ids = team_ids_from_scrape()
    id_to_name = {tid: name for name, tid in team_ids.items()}
    fixtures = json.loads(FIXTURES_JSON.read_text())
    wc_start = min(str(f.get("date_utc", "9999"))[:10] for f in fixtures)

    prior = read_json_safe(SOFA_RESULTS_JSON, {})
    by_event: dict[int, dict[str, Any]] = {
        int(r["event_id"]): r for r in prior.get("results", [])
    }

    session = _session()
    n_new = 0
    api_alive = False
    try:
        for _name, tid in sorted(team_ids.items()):
            payload = get_json(session, f"team/{tid}/events/last/0")
            _sleep()
            if payload is None:
                continue
            api_alive = True
            for e in payload.get("events", []):
                rec = _event_to_result(e, id_to_name, wc_start)
                if rec is not None and rec["event_id"] not in by_event:
                    by_event[rec["event_id"]] = rec
                    n_new += 1
    except SourceDownError as exc:
        print(f"breaker: {exc} — trying the HTML route")

    if not api_alive:
        # API dark (the known Cloudflare API-tier ban) — daily schedule
        # pages still serve 200 with the same event objects embedded.
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        cutoff = (datetime.now(UTC) - timedelta(days=3)).strftime("%Y-%m-%d")
        dates = sorted({
            d for f in fixtures
            if cutoff <= (d := str(f.get("date_utc", ""))[:10]) <= today
        })
        for e in _day_page_wc_events(session, dates).values():
            rec = _event_to_result(e, id_to_name, wc_start)
            if rec is not None and rec["event_id"] not in by_event:
                by_event[rec["event_id"]] = rec
                n_new += 1

    out = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "results": sorted(
            by_event.values(), key=lambda r: (r["date"], r["event_id"])
        ),
    }
    atomic_write_json(SOFA_RESULTS_JSON, out)
    print(
        f"sofascore results: {len(by_event)} stored ({n_new} new) "
        f"-> {SOFA_RESULTS_JSON}"
    )
    return out


def normalize_simple(name: str) -> str:
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", name.strip().lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def norm_sorted_simple(name: str) -> str:
    return " ".join(sorted(normalize_simple(name).split()))


def _collect_wc_events(
    session: Any, team_ids: dict[str, int]
) -> dict[int, dict[str, Any]]:
    """Upcoming WC events across all teams (each fixture found twice; deduped).
    Returns whatever was collected before a breaker trip."""
    events: dict[int, dict[str, Any]] = {}
    try:
        for _name, tid in sorted(team_ids.items()):
            payload = get_json(session, f"team/{tid}/events/next/0")
            _sleep()
            if payload is None:
                continue
            for e in payload.get("events", []):
                if "World Cup" in str(e.get("tournament", {}).get("name", "")):
                    events[int(e["id"])] = e
    except SourceDownError as exc:
        print(f"breaker: {exc} — proceeding with {len(events)} events collected")
    return events


def _lineup_entry(payload: dict[str, Any], eid: int) -> dict[str, Any]:
    """event/{id}/lineups payload -> confirmed_lineups.json entry.
    Shared by the API route and the match-page __NEXT_DATA__ route — the
    page embeds the API object verbatim, so one builder serves both."""
    entry: dict[str, Any] = {
        "event_id": eid,
        "confirmed": bool(payload.get("confirmed", False)),
        "fetched_at": datetime.now(UTC).isoformat(),
    }
    for side in ("home", "away"):
        side_blob = payload.get(side, {}) or {}
        entry[side] = {
            "starters": [
                str(p["player"]["name"])
                for p in side_blob.get("players", [])
                if not p.get("substitute", True)
            ],
            "bench": [
                str(p["player"]["name"])
                for p in side_blob.get("players", [])
                if p.get("substitute", False)
            ],
            "missing": [
                {
                    "name": str(m["player"]["name"]),
                    "position": str(m["player"].get("position", "")),
                    "type": str(m.get("type", "missing")),
                    "reason_code": int(m.get("reason", 0) or 0),
                    "reason": MISSING_REASON_MAP.get(
                        int(m.get("reason", 0) or 0), "Other"
                    ),
                }
                for m in side_blob.get("missingPlayers", [])
                if m.get("player")
            ],
        }
    return entry


def fetch_confirmed_lineups(horizon_hours: float = 48.0) -> dict[str, Any]:
    """Confirmed starting XIs for fixtures kicking off within the horizon.
    Sofascore publishes lineups ~1h pre-kickoff (confirmed=true); run this on
    match days and regenerate the player predictions afterwards.

    When the API tier is 403-banned, falls back to the match pages'
    __NEXT_DATA__ (same payload, page transport). A confirmed entry never
    regresses to an unconfirmed one, and an empty shell never overwrites
    real data."""
    fixtures = json.loads(FIXTURES_JSON.read_text())
    team_ids = team_ids_from_scrape()
    session = _session()
    fix_key: dict[tuple[str, str], int] = {
        (canon_team(str(f["home"])), canon_team(str(f["away"]))): int(
            f["match_number"]
        )
        for f in fixtures
    }

    out: dict[str, Any] = dict(read_json_safe(CONFIRMED_LINEUPS_JSON, {}))  # type: ignore[arg-type]

    now = datetime.now(UTC).timestamp()
    confirmed_count = 0
    api_events = _collect_wc_events(session, team_ids)
    for eid, e in sorted(api_events.items()):
        ts = int(e["startTimestamp"])
        if not (now - 3 * 3600 <= ts <= now + horizon_hours * 3600):
            continue
        key = (
            sofa_canon(str(e["homeTeam"]["name"])),
            sofa_canon(str(e["awayTeam"]["name"])),
        )
        match_number = fix_key.get(key)
        if match_number is None:
            continue
        try:
            payload = get_json(session, f"event/{eid}/lineups")
        except SourceDownError as exc:
            print(f"breaker: {exc} — saving partial lineups")
            break
        _sleep()
        if payload is None:
            continue
        entry = _lineup_entry(payload, eid)
        out[str(match_number)] = entry
        if entry["confirmed"]:
            confirmed_count += 1

    if not api_events:
        # API dark — same events + lineups off the page tier. Window starts
        # a day back so a just-kicked-off match near midnight still counts.
        start_dt = datetime.now(UTC) - timedelta(hours=3)
        end_dt = datetime.now(UTC) + timedelta(hours=horizon_hours)
        dates = sorted({
            d for f in fixtures
            if start_dt.strftime("%Y-%m-%d")
            <= (d := str(f.get("date_utc", ""))[:10])
            <= end_dt.strftime("%Y-%m-%d")
        })
        for eid, e in sorted(_day_page_wc_events(session, dates).items()):
            ts = int(e.get("startTimestamp", 0) or 0)
            if not (now - 3 * 3600 <= ts <= now + horizon_hours * 3600):
                continue
            key = (
                sofa_canon(str((e.get("homeTeam") or {}).get("name", ""))),
                sofa_canon(str((e.get("awayTeam") or {}).get("name", ""))),
            )
            match_number = fix_key.get(key)
            if match_number is None:
                continue
            url = _match_page_url(e)
            if url is None:
                continue
            payload = _lineups_from_next_data(_get_html(session, url))
            if payload is None:
                print(f"html route: no embedded lineups for match {match_number}")
                continue
            entry = _lineup_entry(payload, eid)
            if not any(
                entry[s]["starters"] or entry[s]["missing"]
                for s in ("home", "away")
            ):
                continue  # empty shell — never store nothing
            prev = dict(out.get(str(match_number), {}) or {})
            if prev.get("confirmed") and not entry["confirmed"]:
                continue  # a confirmed XI never regresses to a projection
            out[str(match_number)] = entry
            if entry["confirmed"]:
                confirmed_count += 1

    atomic_write_json(CONFIRMED_LINEUPS_JSON, out)
    print(
        f"lineups: {len(out)} fixtures tracked, {confirmed_count} confirmed "
        f"-> {CONFIRMED_LINEUPS_JSON}"
    )
    return out

# (label, unique-tournament id, season year string, group window [start, end])
HISTORICAL_TOURNAMENTS = [
    ("World Cup 2018", 16, "2018", "2018-06-14", "2018-06-28"),
    ("Euro 2020", 1, "2020", "2021-06-11", "2021-06-23"),
    ("World Cup 2022", 16, "2022", "2022-11-20", "2022-12-02"),
    ("Euro 2024", 1, "2024", "2024-06-14", "2024-06-26"),
    ("Copa América 2024", 133, "2024", "2024-06-20", "2024-07-02"),
]


def fetch_historical_odds() -> dict[str, Any]:
    """Closing 1X2 odds for the group-stage windows of the five backtest
    tournaments. Output rows keyed for the backtest join:
    (canon_home, canon_away, date). Incremental: completed tournaments skip."""
    session = _session()
    out: dict[str, Any] = {}
    if HISTORICAL_ODDS_JSON.exists():
        out = json.loads(HISTORICAL_ODDS_JSON.read_text())

    for label, ut_id, year, start, end in HISTORICAL_TOURNAMENTS:
        if label in out and out[label].get("complete"):
            continue
        try:
            seasons = get_json(session, f"unique-tournament/{ut_id}/seasons")
        except SourceDownError as exc:
            print(f"breaker: {exc} — stopping historical fetch")
            break
        _sleep()
        if seasons is None:
            continue
        season_id = None
        for x in seasons.get("seasons", []):
            if str(x.get("year")) in (year, str(int(year) + 1)):
                season_id = int(x["id"])
                break
        if season_id is None:
            out[label] = {"complete": False, "error": "season not found", "rows": []}
            continue

        # Page through the season's events; collect those in the group window.
        events: list[dict[str, Any]] = []
        for page in range(6):
            try:
                payload = get_json(
                    session,
                    f"unique-tournament/{ut_id}/season/{season_id}/events/last/{page}",
                )
            except SourceDownError as exc:
                print(f"breaker: {exc} — stopping pagination for {label}")
                break
            _sleep()
            if payload is None or not payload.get("events"):
                break
            events.extend(payload["events"])
            if not payload.get("hasNextPage"):
                break

        rows: list[dict[str, Any]] = []
        misses = 0
        for e in events:
            date = datetime.fromtimestamp(int(e["startTimestamp"]), tz=UTC)
            day = date.strftime("%Y-%m-%d")
            if not (start <= day <= end):
                continue
            try:
                payload = get_json(session, f"event/{int(e['id'])}/odds/1/all")
            except SourceDownError as exc:
                print(f"breaker: {exc} — saving partial rows for {label}")
                break
            _sleep()
            decs = parse_1x2(payload) if payload else None
            if decs is None:
                misses += 1
                continue
            rows.append(
                {
                    "home": sofa_canon(str(e["homeTeam"]["name"])),
                    "away": sofa_canon(str(e["awayTeam"]["name"])),
                    "date": day,
                    "odds": decs,
                    "implied": devig(decs),
                }
            )
        out[label] = {
            "complete": True,
            "fetched_at": datetime.now(UTC).isoformat(),
            "n_rows": len(rows),
            "n_missing_odds": misses,
            "rows": rows,
        }
        HISTORICAL_ODDS_JSON.write_text(json.dumps(out, indent=2))
        print(f"{label}: {len(rows)} matches with odds, {misses} without")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odds", action="store_true")
    parser.add_argument("--historical-odds", action="store_true")
    parser.add_argument("--lineups", action="store_true")
    parser.add_argument("--refresh-stats", action="store_true")
    parser.add_argument("--results", action="store_true")
    args = parser.parse_args()
    if args.odds:
        fetch_market_odds()
    elif args.historical_odds:
        fetch_historical_odds()
    elif args.lineups:
        fetch_confirmed_lineups()
    elif args.refresh_stats:
        fetch_played_stats()
    elif args.results:
        fetch_results()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
