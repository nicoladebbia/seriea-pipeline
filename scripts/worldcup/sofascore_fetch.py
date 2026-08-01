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

    API-only by necessity: per-player statistics live in the event-lineups
    API payload, and match pages are client-rendered shells that embed no
    such payload (measured 2026-06-11 — see WC_TOURNAMENT_PAGE comment).
    During API bans nothing is lost, only delayed: events/last re-serves
    history, so the parquet catches up on the first healthy run."""
    import pandas as pd

    df = pd.read_parquet(SOFA_PARQUET)
    seen = set(zip(df["team"], df["match_id"], strict=True))
    team_ids = team_ids_from_scrape()
    session = _session()

    # ── Fail-fast health probe ───────────────────────────────────────────────
    # Under a full IP-ban the detail endpoints don't 403 cleanly — get_json
    # retries+backs-off on each of ~48 teams, so a fully-banned run HANGS for
    # many minutes and only THEN prints its status (measured 2026-07-14). That
    # both wastes the run and risks overlapping the next cron. So probe ONE team
    # endpoint first: if it's down, we're banned — say so loudly and return now,
    # instead of grinding the whole roster to reach the same conclusion slowly.
    if team_ids:
        probe_team, probe_tid = sorted(team_ids.items())[0]
        try:
            probe = get_json(session, f"team/{probe_tid}/events/last/0")
        except SourceDownError:
            probe = None
        if probe is None:
            print(
                f"played-stats: SOURCE DOWN — probe team/{probe_tid}/events/last "
                f"({probe_team}) returned nothing; Sofascore detail endpoints appear "
                f"IP-banned. 0 rows appended, gap PERSISTS and self-heals on the first "
                f"healthy run. Do NOT treat rc=0 as success. (Skipped the full-roster "
                f"sweep to avoid a multi-minute hang against a banned source.)"
            )
            return 0
        _sleep()

    new_rows: list[dict[str, Any]] = []
    now_iso = datetime.now(UTC).isoformat()
    breaker_open = False
    # ── Health accounting so a "0 rows" outcome is never silent ──────────────
    # A run can append 0 rows for two OPPOSITE reasons, and the old log
    # ("0 player-match rows appended") couldn't tell them apart:
    #   • source DOWN  — the endpoints are banned, the gap persists, retrying
    #     won't help; this is the state to flag loudly.
    #   • UP-TO-DATE   — every finished match is already captured; all good.
    # We track: how many teams answered at all, how many finished matches we
    # found that are NOT yet in the parquet (the real pending gap), and how
    # many of those we couldn't fetch because the source went down mid-run.
    teams_reachable = 0          # team/events/last calls that returned data
    teams_down = 0               # team/events/last calls that returned None / raised
    pending_uncaptured = 0       # finished, not-yet-in-parquet matches we saw
    blocked_by_source = 0        # of those pending, ones we couldn't fetch (source down)
    for team, tid in sorted(team_ids.items()):
        if breaker_open:
            break
        try:
            payload = get_json(session, f"team/{tid}/events/last/0")
        except SourceDownError as exc:
            teams_down += 1
            print(f"breaker: {exc} — appending {len(new_rows)} rows collected so far")
            break
        _sleep()
        if payload is None:
            teams_down += 1
            continue
        teams_reachable += 1
        for e in payload.get("events", []):
            eid = int(e["id"])
            if (team, eid) in seen:
                continue
            if str(e.get("status", {}).get("type")) != "finished":
                continue
            pending_uncaptured += 1
            is_home = int(e["homeTeam"].get("id", -1)) == tid
            try:
                lineups = get_json(session, f"event/{eid}/lineups")
            except SourceDownError:
                blocked_by_source += 1
                breaker_open = True
                print(f"breaker open — appending {len(new_rows)} rows collected so far")
                break
            _sleep()
            if lineups is None:
                blocked_by_source += 1  # source served nothing for this event
                continue
            side = "home" if is_home else "away"
            new_rows.extend(
                _player_stat_rows(
                    e, lineups, side, team, tid, now_iso,
                    f"{BASE}/event/{eid}/lineups",
                )
            )
            seen.add((team, eid))

    if new_rows:
        import pandas as pd

        updated = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        updated.to_parquet(SOFA_PARQUET, index=False)

    # ── Loud, self-explaining status line (the whole point of this fix) ──────
    total_teams = len(team_ids)
    if teams_reachable == 0 and total_teams:
        # Not one team endpoint answered — the detail endpoints are banned.
        status = (
            f"played-stats: SOURCE DOWN — 0/{total_teams} team endpoints reachable, "
            f"0 rows appended. Sofascore detail endpoints appear IP-banned; the gap "
            f"PERSISTS and will self-heal on the first healthy run. Do NOT treat rc=0 "
            f"as success."
        )
    elif len(new_rows) == 0 and pending_uncaptured == 0:
        status = (
            f"played-stats: UP-TO-DATE — {teams_reachable}/{total_teams} teams "
            f"reachable, 0 uncaptured finished matches, 0 rows appended (nothing to do)."
        )
    else:
        status = (
            f"played-stats: {len(new_rows)} rows appended | "
            f"{teams_reachable}/{total_teams} teams reachable, {teams_down} down | "
            f"{pending_uncaptured} uncaptured finished matches seen, "
            f"{blocked_by_source} still blocked by source"
            + (" (PARTIAL — source flapped mid-run, gap not fully drained)"
               if blocked_by_source else "")
        )
    print(status)
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
    rec = {
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
    # Half-time scores for half-based parlay legs (1T/2T markets). DEFENSIVE +
    # UNVERIFIED: the Sofascore API was 403-banned when this was written, so
    # `period1` could not be confirmed against a live specimen. We store it
    # only when present; a finished match WITHOUT it prints a one-line warning
    # so the schema assumption is falsifiable the moment the ban clears. The
    # parlay engine treats missing HT as "settle manually", never a guess.
    ht_h, ht_a = hs.get("period1"), as_.get("period1")
    if ht_h is not None and ht_a is not None:
        rec["ht_home"] = int(ht_h)
        rec["ht_away"] = int(ht_a)
    else:
        print(f"HT-fetch: finished match {home}-{away} has no period1 "
              "(half-time legs will fall back to manual settle)")
    return rec


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


# The tournament hub page is ISR-rendered with FRESH event data; the
# daily-schedule pages are stale prerenders (measured 2026-06-11: the
# opener showed 'notstarted' on the day page 75 minutes after kickoff,
# 'inprogress 1-0' here). Match pages are client-rendered shells — their
# __NEXT_DATA__ holds i18n strings only, NO lineups/event payloads, so an
# HTML fallback for lineups/player-stats is INFEASIBLE; don't re-attempt.
WC_TOURNAMENT_PAGE = "https://www.sofascore.com/tournament/football/world/world-cup/16"


def _page_wc_events(session: Any, urls: list[str]) -> dict[int, dict[str, Any]]:
    """World Cup event objects from www.sofascore.com HTML pages — the
    transport that survives API-tier 403 bans. Deduped by event id."""
    events: dict[int, dict[str, Any]] = {}
    for url in urls:
        html = _get_html(session, url)
        label = url.rsplit("/", 1)[-1]
        if not html:
            print(f"html route: {label} page unavailable")
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
        print(f"html route: {label} — {len(page_events)} events on page, {n} WC")
    return events


# ESPN team-name → our canonical fixture names (only the few that differ).
# ESPN's hidden scoreboard JSON is key-free and NOT behind Cloudflare, so it
# survives the Sofascore/FBref bans. Verified 2026-06-15: matched every known
# stored result and reported live state correctly (not a stale prerender).
_ESPN_NAME_FIX = {
    "ivory coast": "Côte d'Ivoire",
    "cape verde": "Cabo Verde",
    "south korea": "Korea Republic",
    # fixtures.json canonical is "USA" (not "United States") — map ESPN's
    # display form onto the actual fixture name or the join silently drops it.
    "usa": "USA",
    "united states": "USA",
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
}


def _espn_results(fixtures: list, wc_start: str) -> list[dict[str, Any]]:
    """Final WC scores from ESPN's key-free JSON scoreboard — the ban-proof
    fallback when both Sofascore tiers (API + HTML) are dark.

    Joins to our fixtures BY NORMALIZED NAME (ESPN carries no Sofascore event
    ids), honouring ESPN's explicit home/away. Only STATUS_FULL_TIME events
    on/after the opening day are returned. event_id is synthesised as a stable
    negative int from the match_number so it never collides with Sofascore ids
    and is idempotent across runs.
    """
    import urllib.request

    # canonical names from fixtures, keyed by normalized form for the join
    canon_by_norm: dict[str, str] = {}
    mn_by_pair: dict[frozenset, int] = {}
    for f in fixtures:
        h, a = f.get("home"), f.get("away")
        if h:
            canon_by_norm[normalize_simple(h)] = h
        if a:
            canon_by_norm[normalize_simple(a)] = a
        if h and a and f.get("match_number"):
            mn_by_pair[frozenset((normalize_simple(h), normalize_simple(a)))] = int(
                f["match_number"]
            )

    def to_canon(espn_name: str) -> str | None:
        fixed = _ESPN_NAME_FIX.get(espn_name.strip().lower())
        if fixed:
            return fixed
        return canon_by_norm.get(normalize_simple(espn_name))

    today = datetime.now(UTC).date()
    start = datetime.fromisoformat(wc_start).date() if wc_start[:4].isdigit() else today
    out: list[dict[str, Any]] = []
    seen_pairs: set[frozenset] = set()
    d = start
    while d <= today:
        url = (
            "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/"
            f"scoreboard?dates={d.strftime('%Y%m%d')}"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})  # noqa: S310 — hardcoded https ESPN host, only path/query interpolated
            with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310 — hardcoded https ESPN host, only path/query interpolated
                payload = json.loads(resp.read())
        except Exception as exc:  # noqa: BLE001 — a bad day must not kill the fallback
            print(f"espn: {d} fetch failed ({type(exc).__name__})")
            d += timedelta(days=1)
            continue
        for e in payload.get("events", []):
            st = (e.get("status") or {}).get("type", {}).get("name", "")
            if st != "STATUS_FULL_TIME":
                continue
            comp = (e.get("competitions") or [{}])[0]
            cs = comp.get("competitors", [])
            hs = next((x for x in cs if x.get("homeAway") == "home"), None)
            aw = next((x for x in cs if x.get("homeAway") == "away"), None)
            if not hs or not aw:
                continue
            home = to_canon(hs.get("team", {}).get("displayName", ""))
            away = to_canon(aw.get("team", {}).get("displayName", ""))
            if not home or not away:
                print(f"espn: unmapped teams "
                      f"{hs.get('team',{}).get('displayName')} / "
                      f"{aw.get('team',{}).get('displayName')} — skipped")
                continue
            try:
                h_score, a_score = int(hs.get("score")), int(aw.get("score"))
            except (TypeError, ValueError):
                continue
            pair = frozenset((normalize_simple(home), normalize_simple(away)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            mn = mn_by_pair.get(pair)
            out.append({
                "event_id": -(mn) if mn else -(10000 + len(out)),  # stable, non-colliding
                "date": d.strftime("%Y-%m-%d"),
                "home": home,
                "away": away,
                "home_score": h_score,
                "away_score": a_score,
                "winner": home if h_score > a_score else away if a_score > h_score else None,
                "decided_by": "FT",
                "penalties": None,
                "source": "espn",
            })
        d += timedelta(days=1)
    return out


def _espn_summary(event_id: str | int) -> dict[str, Any] | None:
    """One ESPN event summary (key-free, ban-proof). None on any failure."""
    import urllib.request

    url = (
        "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/"
        f"summary?event={event_id}"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})  # noqa: S310 — hardcoded https ESPN host, only path/query interpolated
        with urllib.request.urlopen(req, timeout=18) as resp:  # noqa: S310 — hardcoded https ESPN host, only path/query interpolated
            return json.loads(resp.read())  # type: ignore[no-any-return]
    except Exception as exc:  # noqa: BLE001 — a bad summary must not kill the run
        print(f"espn summary {event_id} failed ({type(exc).__name__})")
        return None


def _espn_roster_xi(summary: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    """Extract {espn_display_name -> {starters:[...], bench:[...]}} from an
    ESPN summary's rosters block. Empty roster (scheduled, XI not yet released)
    yields an empty starters list, which the caller treats as 'no XI yet'."""
    out: dict[str, dict[str, list[str]]] = {}
    for t in summary.get("rosters") or []:
        team_name = str(t.get("team", {}).get("displayName", ""))
        starters, bench = [], []
        for p in t.get("roster", []):
            nm = str(p.get("athlete", {}).get("displayName", "")).strip()
            if not nm:
                continue
            (starters if p.get("starter") else bench).append(nm)
        out[team_name] = {"starters": starters, "bench": bench}
    return out


def _espn_last_xi(team_ref: dict[str, Any]) -> dict[str, list[str]] | None:
    """A team's most-recent completed-match XI from ESPN — the PROJECTED
    lineup used before the confirmed XI is released. team_ref is an ESPN
    competitor's 'team' object (carries an id we resolve a schedule from)."""
    import urllib.request

    tid = team_ref.get("id")
    if not tid:
        return None
    url = (
        "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/"
        f"teams/{tid}/schedule"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})  # noqa: S310 — hardcoded https ESPN host, only path/query interpolated
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310 — hardcoded https ESPN host, only path/query interpolated
            sched = json.loads(resp.read())
    except Exception:  # noqa: BLE001
        return None
    played = [
        ev for ev in sched.get("events", [])
        if (ev.get("competitions") or [{}])[0].get("status", {})
        .get("type", {}).get("completed")
    ]
    if not played:
        return None
    # sort by date explicitly rather than trusting ESPN's event order
    last = sorted(played, key=lambda ev: str(ev.get("date", "")))[-1]
    summary = _espn_summary(last.get("id"))
    if not summary:
        return None
    rosters = _espn_roster_xi(summary)
    want = str(team_ref.get("displayName", ""))
    entry = rosters.get(want)
    if entry and entry["starters"]:
        return entry
    # displayName may differ across endpoints — take whichever roster is this team
    for name, e in rosters.items():
        if normalize_simple(name) == normalize_simple(want) and e["starters"]:
            return e
    return None


def _espn_lineups(fixtures: list, horizon_hours: float) -> dict[str, dict[str, Any]]:
    """ESPN confirmed/projected XIs for fixtures kicking off within the horizon
    — the ban-proof lineup fallback (Sofascore API is the only confirmed-XI
    source and it's IP-banned; ESPN's key-free JSON is not).

    Returns {match_number(str) -> entry}. Each entry is _lineup_entry-shaped
    (starters/bench per side) plus:
      confirmed=True,  source='espn'            when ESPN has released the XI
      confirmed=False, source='espn_projected'  team's last-match XI (a guess)
    ESPN populates a scheduled event's roster only when the official XI drops
    (~1h pre-kickoff, verified 2026-07-13); before that we fall back per team
    to the last completed match's XI so the UI shows a labelled projection."""
    from datetime import datetime as _dt

    def _canon(espn_name: str) -> str | None:
        fixed = _ESPN_NAME_FIX.get(espn_name.strip().lower())
        if fixed:
            return fixed
        for f in fixtures:
            for side in ("home", "away"):
                nm = f.get(side)
                if nm and normalize_simple(nm) == normalize_simple(espn_name):
                    return str(nm)
        return None

    mn_by_pair: dict[frozenset, int] = {}
    for f in fixtures:
        h, a = f.get("home"), f.get("away")
        if h and a and f.get("match_number"):
            mn_by_pair[frozenset((normalize_simple(h), normalize_simple(a)))] = int(
                f["match_number"]
            )

    now = datetime.now(UTC)
    # dates to scan: any fixture whose kickoff is within [-3h, +horizon]
    scan_dates: set[str] = set()
    for f in fixtures:
        ku = str(f.get("date_utc", ""))
        if not ku[:4].isdigit():
            continue
        try:
            kt = _dt.fromisoformat(ku.replace("Z", "+00:00"))
        except ValueError:
            continue
        if -3 * 3600 <= (kt - now).total_seconds() <= horizon_hours * 3600:
            scan_dates.add(kt.strftime("%Y%m%d"))

    out: dict[str, dict[str, Any]] = {}
    import urllib.request
    for d in sorted(scan_dates):
        url = (
            "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/"
            f"scoreboard?dates={d}"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})  # noqa: S310 — hardcoded https ESPN host, only path/query interpolated
            with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310 — hardcoded https ESPN host, only path/query interpolated
                board = json.loads(resp.read())
        except Exception as exc:  # noqa: BLE001
            print(f"espn lineups: {d} scoreboard failed ({type(exc).__name__})")
            continue
        for e in board.get("events", []):
            comp = (e.get("competitions") or [{}])[0]
            cs = comp.get("competitors", [])
            hs = next((x for x in cs if x.get("homeAway") == "home"), None)
            aw = next((x for x in cs if x.get("homeAway") == "away"), None)
            if not hs or not aw:
                continue
            home = _canon(hs.get("team", {}).get("displayName", ""))
            away = _canon(aw.get("team", {}).get("displayName", ""))
            if not home or not away:
                print(f"espn lineups: unmapped "
                      f"{hs.get('team',{}).get('displayName')} / "
                      f"{aw.get('team',{}).get('displayName')} — skipped")
                continue
            mn = mn_by_pair.get(
                frozenset((normalize_simple(home), normalize_simple(away)))
            )
            if mn is None:
                continue
            summary = _espn_summary(e.get("id"))
            if not summary:
                continue
            rosters = _espn_roster_xi(summary)
            entry: dict[str, Any] = {
                "event_id": -(int(mn)),  # negative: never collides with Sofascore ids
                "fetched_at": datetime.now(UTC).isoformat(),
            }
            confirmed_sides = 0
            for side, comp_obj in (("home", hs), ("away", aw)):
                espn_name = comp_obj.get("team", {}).get("displayName", "")
                xi = rosters.get(espn_name)
                if xi and xi["starters"]:
                    entry[side] = {"starters": xi["starters"], "bench": xi["bench"],
                                   "missing": [], "projected": False}
                    confirmed_sides += 1
                else:
                    proj = _espn_last_xi(comp_obj.get("team", {}))
                    entry[side] = {
                        "starters": proj["starters"] if proj else [],
                        "bench": proj["bench"] if proj else [],
                        "missing": [],
                        "projected": True,
                    }
            entry["confirmed"] = confirmed_sides == 2
            entry["source"] = "espn" if entry["confirmed"] else "espn_projected"
            out[str(mn)] = entry
    return out


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
        # API dark (the known Cloudflare API-tier ban) — the tournament hub
        # page is ISR-rendered with fresh event data. Day pages only as a
        # last resort: they're stale prerenders (see WC_TOURNAMENT_PAGE).
        events = _page_wc_events(session, [WC_TOURNAMENT_PAGE])
        if not events:
            today = datetime.now(UTC).strftime("%Y-%m-%d")
            cutoff = (datetime.now(UTC) - timedelta(days=3)).strftime("%Y-%m-%d")
            dates = sorted({
                d for f in fixtures
                if cutoff <= (d := str(f.get("date_utc", ""))[:10]) <= today
            })
            events = _page_wc_events(
                session,
                [f"https://www.sofascore.com/football/{d}" for d in dates],
            )
        for e in events.values():
            rec = _event_to_result(e, id_to_name, wc_start)
            if rec is not None and rec["event_id"] not in by_event:
                by_event[rec["event_id"]] = rec
                n_new += 1

    # ESPN fallback — fires when both Sofascore tiers are dark / dry. ESPN is
    # key-free and not behind Cloudflare, so it backfills the games the ban
    # hides. Dedup is BY NORMALISED TEAM-PAIR (ESPN ids differ from Sofascore),
    # so a match already stored from Sofascore is never duplicated; ESPN only
    # ADDS genuinely-missing games.
    if n_new == 0:
        existing_pairs = {
            frozenset((normalize_simple(r["home"]), normalize_simple(r["away"])))
            for r in by_event.values()
        }
        n_espn = 0
        for rec in _espn_results(fixtures, wc_start):
            pair = frozenset(
                (normalize_simple(rec["home"]), normalize_simple(rec["away"]))
            )
            if pair in existing_pairs:
                continue
            existing_pairs.add(pair)
            by_event[rec["event_id"]] = rec
            n_espn += 1
        if n_espn:
            print(f"espn fallback: +{n_espn} results the Sofascore ban hid")
            n_new += n_espn

    out = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "results": sorted(
            by_event.values(), key=lambda r: (r["date"], str(r["event_id"]))
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

    Two-source: Sofascore API first (confirmed XIs, but IP-bannable). When it
    yields no confirmed XI for an in-horizon fixture, fall back to ESPN's
    key-free JSON (ban-proof) — confirmed once ESPN releases the official XI,
    else the team's last-match XI as a labelled PROJECTION (projected=true,
    source=espn_projected). This replaces the old 'no fallback' behaviour where
    a Sofascore ban left every knockout fixture on caps-based XIs."""
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

    # ESPN fallback — ban-proof. Fills any in-horizon fixture Sofascore didn't
    # confirm (banned, or XI not yet up). ESPN-confirmed replaces a stale entry;
    # ESPN-projected only fills where we have no confirmed XI, and never
    # downgrades an existing confirmed one.
    espn_confirmed = espn_projected = 0
    for mn, entry in _espn_lineups(fixtures, horizon_hours).items():
        prior = out.get(mn)
        prior_confirmed = bool(prior and prior.get("confirmed"))
        if entry["confirmed"]:
            out[mn] = entry
            confirmed_count += 1
            espn_confirmed += 1
        elif not prior_confirmed and any(
            entry.get(s, {}).get("starters") for s in ("home", "away")
        ):
            out[mn] = entry  # labelled projection; won't override a confirmed XI
            espn_projected += 1

    atomic_write_json(CONFIRMED_LINEUPS_JSON, out)
    print(
        f"lineups: {len(out)} fixtures tracked, {confirmed_count} confirmed "
        f"(espn +{espn_confirmed} confirmed, +{espn_projected} projected) "
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
