"""ESPN confirmed lineups for club leagues (backup lineup source, no key).

Second link in the lineup chain (scraper/lineup_fetcher.py): Sofascore first,
then ESPN, then football-data.org, then API-Football. Built 2026-09-05 when the
first link answered 403 "challenge" on every endpoint from this network and the
two paid-tier backups turned out structurally dead (football-data.org's free
tier carries no lineup field at all; API-Football's free plan refuses the
season). The same ESPN JSON already backs the World Cup page
(scripts/worldcup/sofascore_fetch._espn_lineups): `summary?event=<id>` carries
`rosters[*].roster[*]` with `starter` flags, `formation`, and the bench, and
ESPN fills it only once the official XI drops (~1h pre-kickoff, verified
2026-07-13 on the WC and 2026-09-05 on Roma–Atalanta at half-time: Pašalić on
the bench while our predicted XI had him starting).

Output matches confirmed_lineups.json exactly (home_lineup / home_bench /
formation / lineup_source / source_api), keyed "Home vs Away" in the repo's
canonical team names.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from config.team_names import normalize_team

log = logging.getLogger(__name__)

_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
ESPN_LEAGUE = {"serie_a": "ita.1", "premier_league": "eng.1"}
MIN_XI = 7  # fewer starters than this is a partial roster, not a team sheet
# Measured 2026-09-05 from this network: the DEFAULT python-requests agent gets 200,
# "Mozilla/5.0" (what the WC fetcher sends) and a full Chrome UA both get 403.
# Try the default first, the browser string only as a fallback.
_HEADER_LADDER = ({}, {"User-Agent": "Mozilla/5.0"})


def _get(url: str, params: dict, timeout: float = 15.0) -> dict | None:
    import requests

    status = None
    for headers in _HEADER_LADDER:
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
        except Exception as e:  # noqa: BLE001 - one source of several, never raise
            log.warning("ESPN request failed: %s", e)
            return None
        status = r.status_code
        if status == 200:
            return r.json()
        if status != 403:
            break
    # third rung: a browser TLS fingerprint (the repo's Sofascore transport)
    try:
        from curl_cffi import requests as cffi
        r = cffi.get(url, params=params, impersonate="chrome124", timeout=timeout)
        if r.status_code == 200:
            return r.json()
        status = r.status_code
    except Exception as e:  # noqa: BLE001
        log.debug("ESPN curl_cffi rung failed: %s", e)
    log.warning("ESPN %s -> HTTP %s", url.rsplit("/", 1)[-1], status)
    return None


def parse_summary_rosters(summary: dict) -> dict[str, dict[str, Any]]:
    """{canonical team -> {xi, bench, formation}} from a summary payload;
    a side with fewer than MIN_XI starters is dropped (roster not released)."""
    out: dict[str, dict[str, Any]] = {}
    for r in summary.get("rosters") or []:
        team = normalize_team((r.get("team") or {}).get("displayName") or "")
        players = r.get("roster") or []
        xi = [p["athlete"]["displayName"] for p in players if p.get("starter") and p.get("athlete")]
        bench = [p["athlete"]["displayName"] for p in players if not p.get("starter") and p.get("athlete")]
        if team and len(xi) >= MIN_XI:
            out[team] = {"xi": xi, "bench": bench, "formation": r.get("formation") or ""}
    return out


def fetch_lineups_espn(odds_data: dict | None, league: str = "serie_a", now: datetime | None = None,
                       deadline_sec: float = 0.0) -> dict[str, dict]:
    """Confirmed lineups for every imminent/approaching match in odds_data.
    One scoreboard call per kickoff date, one summary call per match found."""
    from scripts.utils.match_timing import classify_match_window

    code = ESPN_LEAGUE.get(league)
    if not code or not odds_data:
        return {}
    now = now or datetime.now(UTC)
    wanted: dict[str, str] = {}   # match_key -> kickoff date YYYYMMDD
    for mk, info in odds_data.items():
        ct = (info or {}).get("commence_time") or ""
        if classify_match_window(ct, now=now) in ("imminent", "approaching"):
            try:
                wanted[mk] = datetime.fromisoformat(ct.replace("Z", "+00:00")).strftime("%Y%m%d")
            except ValueError:
                continue
    if not wanted:
        return {}
    t0 = datetime.now(UTC)
    confirmed: dict[str, dict] = {}
    for d in sorted(set(wanted.values())):
        board = _get(f"{_BASE}/{code}/scoreboard", {"dates": d})
        for ev in (board or {}).get("events") or []:
            if deadline_sec and (datetime.now(UTC) - t0).total_seconds() > deadline_sec:
                log.warning("ESPN lineup deadline (%.0fs) reached — returning %d", deadline_sec, len(confirmed))
                return confirmed
            comp = (ev.get("competitions") or [{}])[0]
            sides = {c.get("homeAway"): normalize_team((c.get("team") or {}).get("displayName") or "")
                     for c in comp.get("competitors") or []}
            mk = f"{sides.get('home')} vs {sides.get('away')}"
            if mk not in wanted:
                continue
            summary = _get(f"{_BASE}/{code}/summary", {"event": ev.get("id")})
            rosters = parse_summary_rosters(summary or {})
            home, away = rosters.get(sides["home"]), rosters.get(sides["away"])
            if not home or not away:
                log.info("ESPN: XI not released yet for %s", mk)
                continue
            confirmed[mk] = {
                "home_team": sides["home"], "away_team": sides["away"],
                "home_lineup": home["xi"], "away_lineup": away["xi"],
                "home_bench": home["bench"], "away_bench": away["bench"],
                "home_formation": home["formation"], "away_formation": away["formation"],
                "lineup_source": "confirmed", "source_api": "espn", "espn_event_id": ev.get("id"),
            }
            log.info("Confirmed lineup (ESPN): %s (%s vs %s)", mk, home["formation"], away["formation"])
    return confirmed
