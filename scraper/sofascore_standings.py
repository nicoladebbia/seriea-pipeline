"""Live standings from Sofascore tournament HTML pages — the API-block fallback.

Extracted verbatim from ``web/app.py`` (2026-07-16). It lived there because the
dashboard was its only caller; ``scripts.data.sofascore_watcher`` needs the same
scraper, and **importing ``web.app`` starts the auto-settle scheduler thread**
(measured: logs "Auto-settle scheduler started"). That thread sleeps 300s before
its first pass, which is harmless inside a 1–6s dashboard request but *not*
inside a watcher tick that can exceed 300s on a post-matchday refresh — it would
fire the ledger settler and spend Odds API credits from a scrape job.

Importing **this** module starts no threads and touches no Flask state.

The alternative was a second copy of the scraper. That would mean two sentinel
lists and two breakers, and the sentinel is the only thing standing between a
Sofascore schema change and silently-wrong standings — so it is shared, not
duplicated.

Why the HTML path exists at all: ``api.sofascore.com`` returns 403 under
Cloudflare IP bans (see the project CLAUDE.md), while the tournament *hub* pages
stay reachable and ISR-fresh. This parses the ``__NEXT_DATA__`` blob they embed.

⚠️ **Two known gaps, pre-existing and deliberately preserved by the extraction —
neither is introduced here, and neither is fixed here:**

1. **The season is stamped from the clock, not the page.** ``season`` comes from
   ``get_current_season()`` (boundary: Aug 1), but the tournament page serves
   whatever season Sofascore considers current. Measured 2026-07-16: both pages
   already served **26/27** tables (20 rows, all ``played=0``) while
   ``get_current_season()`` still returned ``'2025-2026'``. The dashboard is
   shielded by its own freshness guard (it prefers HTML only when
   ``html_max_played >= parquet_max_played``, and 0 >= 36 is false), and after
   Aug 1 the clock and the page agree again — so this only misfires in the
   June–July dead window. The page states the truth (``standings[0].name`` ==
   "Serie A 26/27"); nothing reads it yet.
2. **The sentinel cannot catch a partial table.** It asserts one known team is
   present, so a table that parses with *some* rows missing passes. The stored
   ``standings_premier_league.json`` written by the original watcher on
   2026-06-01 holds **14 of 20** teams with the sentinel (Arsenal) intact.
   A row-count floor would catch it; that is a behaviour change, not a move.
"""

from __future__ import annotations

import json
import logging
import time as _time
from typing import Any

from config.settings import get_current_season

log = logging.getLogger(__name__)

# Sofascore tournament SEO slugs (used for HTML scraping fallback when API blocked)
LEAGUE_SOFASCORE_PAGE = {
    "serie_a": ("italy/serie-a", 23),
    "premier_league": ("england/premier-league", 17),
}

# Live HTML standings cache (5min TTL on success; 30s on failure for fast retry)
_html_standings_cache: dict[str, tuple[float, dict]] = {}
HTML_STANDINGS_TTL = 300
HTML_FAILURE_TTL = 30  # negative-cache TTL: empty payload = recent failure marker
HTML_BAN_TTL = 900  # negative-cache TTL once the breaker is open (see _failure_ttl)

# Sentinel teams — the league is broken if these don't appear
HTML_SENTINEL_TEAM = {
    "serie_a": "Inter",
    "premier_league": "Arsenal",
}

# Per-league HTML scrape health.
# {league: {last_success_at, last_attempt_at, consecutive_failures, last_error}}
_html_health: dict[str, dict] = {}
HTML_FAILURE_THRESHOLD = 3  # consecutive failures before "broken"


def html_health_now(league: str) -> dict:
    """Return health entry for a league, creating it if missing."""
    return _html_health.setdefault(league, {
        "last_success_at": 0.0,
        "last_attempt_at": 0.0,
        "consecutive_failures": 0,
        "last_error": "",
        "schema_break": False,
    })


def _failure_ttl(h: dict) -> int:
    """How long to sit on a failure before touching Sofascore again.

    A blip deserves 30s. A ban does not. Sofascore 403s every tier for
    hours-to-days at a time (CLAUDE.md "Sofascore API blocks"), and the breaker
    already knows the difference — it just was not being used to back off, so a
    banned endpoint kept getting re-probed on the 30s blip schedule: ~120
    requests an hour per league, none of which could succeed.
    """
    if h["consecutive_failures"] >= HTML_FAILURE_THRESHOLD or h["schema_break"]:
        return HTML_BAN_TTL
    return HTML_FAILURE_TTL


def html_is_broken(league: str) -> bool:
    """True if HTML scrape has been failing repeatedly OR schema-broke."""
    h = html_health_now(league)
    return h["consecutive_failures"] >= HTML_FAILURE_THRESHOLD or h["schema_break"]


def sofascore_get_retry(s: Any, url: str, timeout: int = 10, attempts: int = 3) -> tuple[Any, str]:
    """GET with in-call retry on transient transport errors.

    Retries connect failures, timeouts, and Cloudflare-mood statuses (403/429/5xx)
    with 1s/2s backoff so a single blip never reaches the failure breaker.
    Returns (response, "") on success or hard status (404 etc.); (None, err) when
    all attempts were transient failures.
    """
    err = ""
    for attempt in range(1, attempts + 1):
        try:
            r = s.get(url, timeout=timeout)
        except Exception as e:  # noqa: BLE001 — transport errors vary by curl_cffi backend
            err = f"{type(e).__name__}: {str(e)[:120]}"
        else:
            if r.status_code in (403, 429) or r.status_code >= 500:
                err = f"HTTP {r.status_code}"
            else:
                return r, ""
        if attempt < attempts:
            _time.sleep(attempt)
    return None, f"transport ({attempts} attempts): {err}"


def live_standings_via_html(league: str) -> dict:
    """Scrape live standings from Sofascore tournament HTML page.

    Self-instrumenting: tracks success/failure per league. Sentinel-checks the
    parsed payload (must contain a known team) so silent schema breaks trip
    the breaker. Returns {} on any failure — caller falls back to parquet.
    """
    import time as _t
    now = _t.time()
    h = html_health_now(league)

    # Cache hit — success payloads live 5min; failure markers ({}) 30s, so a
    # flaky Sofascore isn't re-hammered by every caller
    if league in _html_standings_cache:
        cached_at, payload = _html_standings_cache[league]
        ttl = HTML_STANDINGS_TTL if payload else _failure_ttl(h)
        if (now - cached_at) < ttl:
            import copy
            return copy.deepcopy(payload)

    def _fail(err: str, *, schema: bool = False) -> dict:
        (log.error if schema else log.warning)("HTML standings %s: %s", league, err)
        h["last_error"] = err
        if schema:
            h["schema_break"] = True
        h["consecutive_failures"] += 1
        _html_standings_cache[league] = (now, {})
        return {}

    h["last_attempt_at"] = now
    slug_info = LEAGUE_SOFASCORE_PAGE.get(league)
    if not slug_info:
        h["last_error"] = "league not configured"
        h["consecutive_failures"] += 1
        return {}
    slug, _tid = slug_info
    url = f"https://www.sofascore.com/tournament/football/{slug}/{_tid}"

    try:
        from curl_cffi import requests as cffi  # type: ignore
        s = cffi.Session(impersonate="chrome120")
        s.headers.update({
            "Referer": "https://www.google.com/",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        })
        r, terr = sofascore_get_retry(s, url, timeout=10)
        if r is None:
            return _fail(terr)
        if r.status_code != 200:
            return _fail(f"HTTP {r.status_code}")

        import re as _re
        m = _re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', r.text, _re.DOTALL)
        if not m:
            return _fail("NEXT_DATA script not found", schema=True)

        data = json.loads(m.group(1))
        pp = data.get("props", {}).get("pageProps", {})
        # 2026-06: Sofascore hoisted initialProps.* directly onto pageProps — support both
        st_list = pp.get("standings") or (pp.get("initialProps") or {}).get("standings", [])
        if not st_list or not isinstance(st_list, list):
            return _fail("standings list missing in NEXT_DATA", schema=True)

        rows = st_list[0].get("rows", [])
        if not rows:
            return _fail("rows array empty", schema=True)

        teams: dict[str, dict] = {}
        for row in rows:
            t = row.get("team", {})
            tname = t.get("name", "")
            if not tname:
                continue
            played = int(row.get("matches", 0) or 0)
            wins = int(row.get("wins", 0) or 0)
            draws = int(row.get("draws", 0) or 0)
            losses = int(row.get("losses", 0) or 0)
            gf = int(row.get("scoresFor", 0) or 0)
            ga = int(row.get("scoresAgainst", 0) or 0)
            teams[tname] = {
                "team": tname,
                "position": int(row.get("position", 0) or 0),
                "played": played,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "gf": gf,
                "ga": ga,
                "gd": gf - ga,
                "points": int(row.get("points", 0) or 0),
                "form_last5": "",
                "home": {"played": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0, "ppg": 0.0},
                "away": {"played": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0, "ppg": 0.0},
                "league": league,
            }

        # Sentinel: known team must be in the parsed standings.
        sentinel = HTML_SENTINEL_TEAM.get(league)
        if sentinel and sentinel not in teams:
            log.error("HTML standings %s: sentinel %r missing. Found teams: %s",
                      league, sentinel, sorted(teams.keys())[:5])
            return _fail(f"sentinel team {sentinel!r} missing — schema break", schema=True)

        max_played = max((t["played"] for t in teams.values()), default=0)
        payload = {
            "standings": teams,
            "current_matchweek": max_played,
            "season": get_current_season(),
            "league": league,
            "_source": "sofascore_html",
            "_scraped_at": now,
        }
        _html_standings_cache[league] = (now, payload)

        # Success — reset breaker
        h["last_success_at"] = now
        h["consecutive_failures"] = 0
        h["schema_break"] = False
        h["last_error"] = ""

        log.info("HTML standings %s: %d teams, MW%d", league, len(teams), max_played)
        import copy
        return copy.deepcopy(payload)
    except Exception as e:
        return _fail(f"{type(e).__name__}: {str(e)[:120]}")
