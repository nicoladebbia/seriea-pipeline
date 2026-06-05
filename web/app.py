"""Flask web application for Serie A betting intelligence dashboard."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time as _time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env for API keys
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

from flask import Flask, render_template, jsonify, request as flask_request
from config.settings import (
    DATA_DIR, get_current_season, UPCOMING_DIR, BETTING_DIR, BANKROLL_DIR, LIVE_DIR,
)
from config.leagues import LEAGUE_REGISTRY
from config.team_names import strip_accents
from scripts.utils.parsing import extract_line

_BASE = Path(__file__).parent.parent  # project root
FEEDBACK_DIR = DATA_DIR / "feedback"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Multi-league helpers
# ---------------------------------------------------------------------------

# Import from central config — single source of truth for active leagues
from config.leagues import ACTIVE_LEAGUES

# League key aliases for flexible query params
_LEAGUE_ALIASES: dict[str, str] = {
    "epl": "premier_league",
    "pl": "premier_league",
    "eng": "premier_league",
    "serie_a": "serie_a",
    "seriea": "serie_a",
    "ita": "serie_a",
    "la_liga": "la_liga",
    "laliga": "la_liga",
    "bundesliga": "bundesliga",
    "ligue_1": "ligue_1",
    "ligue1": "ligue_1",
}
# Add all registry keys as self-aliases
for _k in LEAGUE_REGISTRY:
    _LEAGUE_ALIASES.setdefault(_k, _k)


def _resolve_league(raw: str | None) -> str | None:
    """Resolve a league query param to a canonical LEAGUE_REGISTRY key.

    Returns None if raw is None/empty (meaning 'all leagues').
    Raises ValueError if the value is non-empty but unrecognized.
    """
    if not raw:
        return None
    key = raw.strip().lower().replace("-", "_")
    resolved = _LEAGUE_ALIASES.get(key)
    if resolved is None:
        raise ValueError(f"Unknown league '{raw}'. Valid: {', '.join(sorted(LEAGUE_REGISTRY))}")
    return resolved


def _load_all_leagues_json(base_name: str, key: str | None = None) -> dict | list:
    """Load a JSON data file merged across all active leagues.

    For dict-type data (standings, odds, form, h2h):
      base_name="standings.json", key="standings"
      → loads standings.json + standings_premier_league.json + ...
      → merges the dicts under the given key

    For list-type data (predictions):
      base_name="predictions.json", key="predictions"
      → loads predictions.json + predictions_premier_league.json + ...
      → concatenates the lists, tagging each with league

    Args:
        base_name: Base filename (e.g., "odds_full.json")
        key: Key to extract from each file (e.g., "matches", "standings")
             If None, returns the raw merged dict.
    """
    stem = base_name.replace(".json", "")
    merged = _load_json(UPCOMING_DIR / base_name, {})

    for league in ACTIVE_LEAGUES:
        if league == "serie_a":
            continue
        league_file = UPCOMING_DIR / f"{stem}_{league}.json"
        extra = _load_json(league_file, {})
        if not extra or not isinstance(extra, dict):
            continue
        if key:
            src = extra.get(key, {})
            if isinstance(src, dict):
                merged.setdefault(key, {})
                if isinstance(merged.get(key), dict):
                    merged[key].update(src)
            elif isinstance(src, list):
                merged.setdefault(key, [])
                if isinstance(merged.get(key), list):
                    # Tag each item with league
                    for item in src:
                        if isinstance(item, dict):
                            item.setdefault("league", league)
                    merged[key].extend(src)
        else:
            if isinstance(extra.get("matches"), dict) and isinstance(merged.get("matches"), dict):
                merged["matches"].update(extra["matches"])

    return merged


def _get_league_filter() -> str | None:
    """Extract and resolve the 'league' query param from the current request.

    Returns a canonical league key or None (all leagues).
    Returns None on invalid league (logs warning instead of crashing).
    """
    raw = flask_request.args.get("league", "").strip()
    if not raw:
        return None
    try:
        return _resolve_league(raw)
    except ValueError:
        log.warning("Invalid league param '%s', treating as all leagues", raw)
        return None


def _team_belongs_to_league(team_name: str, league_key: str) -> bool:
    """Check if a team name belongs to a specific league.

    Uses the canonical team name maps from config.team_names.
    """
    from config.team_names import normalize_team, SERIE_A_NAMES, PREMIER_LEAGUE_NAMES

    # Map league key -> set of canonical names for that league
    _LEAGUE_TEAMS: dict[str, dict] = {
        "serie_a": SERIE_A_NAMES,
        "premier_league": PREMIER_LEAGUE_NAMES,
    }
    league_names = _LEAGUE_TEAMS.get(league_key)
    if league_names is None:
        return True  # Unknown league config -> don't filter
    canonical = normalize_team(team_name)
    # canonical name is one of the *values* in the mapping dict
    return canonical in league_names.values() or canonical in league_names


def _match_belongs_to_league(match_dict: dict, league_key: str) -> bool:
    """Check if a match/bet/prediction dict belongs to a league.

    1. If the dict has an explicit 'league' field, use it directly.
    2. Otherwise, resolve by checking the home_team against known league teams.
    """
    explicit = match_dict.get("league", "")
    if explicit:
        return _resolve_league(explicit) == league_key
    home = match_dict.get("home_team", "")
    if not home:
        # Try to extract from 'match' key: "Home vs Away"
        mk = match_dict.get("match", "")
        if " vs " in mk:
            home = mk.split(" vs ", 1)[0].strip()
    if not home:
        return True  # Can't determine -> include
    return _team_belongs_to_league(home, league_key)


def _filter_by_league(items: list, league_key: str | None) -> list:
    """Filter a list of match/bet dicts by league. Returns all if league_key is None."""
    if league_key is None:
        return items
    return [item for item in items if _match_belongs_to_league(item, league_key)]


def _filter_dict_by_league(items: dict, league_key: str | None) -> dict:
    """Filter a match-keyed dict by league. Returns all if league_key is None."""
    if league_key is None:
        return items
    return {
        k: v for k, v in items.items()
        if _match_belongs_to_league({"match": k, **(v if isinstance(v, dict) else {})}, league_key)
    }


def _active_leagues_info() -> list[dict]:
    """Return info dicts for all active leagues."""
    result = []
    for key in ACTIVE_LEAGUES:
        cfg = LEAGUE_REGISTRY.get(key)
        if cfg:
            result.append({
                "key": key,
                "name": cfg.name,
                "country": cfg.country,
                "timezone": cfg.timezone,
            })
    return result

app = Flask(__name__, template_folder="templates", static_folder="static")

# Gzip compression for all responses
try:
    from flask_compress import Compress
    Compress(app)
except ImportError:
    pass

# AI Advisor Blueprint
from web.advisor import advisor_bp
app.register_blueprint(advisor_bp)


@app.after_request
def add_security_and_cache_headers(response):
    """Add security headers and prevent browsers from caching API responses."""
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
        "img-src 'self' data: https:; "
        "font-src 'self' https://cdn.jsdelivr.net https://fonts.gstatic.com; "
        "connect-src 'self'"
    )
    # Cache control for API responses
    if response.content_type and "application/json" in response.content_type:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_json_cache: dict[str, tuple[float, float, object]] = {}  # path → (mtime, cached_at, data)
_JSON_CACHE_TTL = 60  # seconds

_parquet_cache: dict[str, tuple[float, float, object]] = {}  # path → (mtime, cached_at, df)
_PARQUET_CACHE_TTL = 300  # 5 minutes


def _read_parquet_cached(path, columns=None):
    """Read parquet with 5-minute mtime-aware cache."""
    import pandas as pd
    key = f"{path}|{columns}"
    now = _time.time()
    if key in _parquet_cache:
        cached_mtime, cached_at, cached_df = _parquet_cache[key]
        if (now - cached_at) < _PARQUET_CACHE_TTL:
            return cached_df
    try:
        p = Path(path) if not isinstance(path, Path) else path
        if not p.exists():
            return pd.DataFrame()
        mtime = p.stat().st_mtime
        if key in _parquet_cache and _parquet_cache[key][0] == mtime:
            _parquet_cache[key] = (mtime, now, _parquet_cache[key][2])
            return _parquet_cache[key][2]
        df = pd.read_parquet(p, columns=columns) if columns else pd.read_parquet(p)
        _parquet_cache[key] = (mtime, now, df)
        return df
    except Exception as e:
        log.warning("Failed to read parquet %s: %s", path, e)
        return pd.DataFrame()

def _load_json(path: Path, default=None):
    """Safely load a JSON file with 60-second mtime-aware cache.

    Returns a deep copy of cached data so callers can mutate freely without
    poisoning the cache. The previous behaviour returned a shared reference,
    which let one endpoint's mutation (e.g. merging EPL standings into SA)
    leak into every subsequent caller.
    """
    import copy
    if default is None:
        default = {}
    key = str(path)
    now = _time.time()
    # Check cache: valid if within TTL and file mtime hasn't changed
    if key in _json_cache:
        cached_mtime, cached_at, cached_data = _json_cache[key]
        if (now - cached_at) < _JSON_CACHE_TTL:
            return copy.deepcopy(cached_data)
    try:
        if path.exists():
            mtime = path.stat().st_mtime
            # If file mtime matches cached mtime, refresh TTL without re-reading
            if key in _json_cache and _json_cache[key][0] == mtime:
                _json_cache[key] = (mtime, now, _json_cache[key][2])
                return copy.deepcopy(_json_cache[key][2])
            with open(path) as f:
                data = json.load(f)
            _json_cache[key] = (mtime, now, data)
            return copy.deepcopy(data)
    except Exception as e:
        log.warning(f"Failed to load {path.name}: {e}")
    return default


# ── Standings derived from Sofascore parquet (single source of truth) ──
_standings_cache: dict[str, tuple[float, dict]] = {}  # league → (cached_at, payload)
_STANDINGS_TTL = 60  # 60s — recomputes from parquet on each tab refresh

_LEAGUE_PARQUET = {
    "serie_a": "data/external/sofascore/player_match_stats.parquet",
    "premier_league": "data/external/sofascore/player_match_stats_premier_league.parquet",
}

_LEAGUE_FIXTURES_FILE = {
    "serie_a": "data/external/sofascore/fixtures_2025_2026.json",
    "premier_league": "data/external/sofascore/fixtures_2025_2026_premier_league.json",
}

# Sofascore tournament SEO slugs (used for HTML scraping fallback when API blocked)
_LEAGUE_SOFASCORE_PAGE = {
    "serie_a": ("italy/serie-a", 23),
    "premier_league": ("england/premier-league", 17),
}

# Live HTML standings cache (5min TTL on success; 30s on failure for fast retry)
_html_standings_cache: dict[str, tuple[float, dict]] = {}
_HTML_STANDINGS_TTL = 300

# Sentinel teams — the league is broken if these don't appear
_HTML_SENTINEL_TEAM = {
    "serie_a": "Inter",
    "premier_league": "Arsenal",
}

# Per-league HTML scrape health.
# {league: {last_success_at, last_attempt_at, consecutive_failures, last_error}}
_html_health: dict[str, dict] = {}
_HTML_FAILURE_THRESHOLD = 3  # consecutive failures before "broken"


def _html_health_now(league: str) -> dict:
    """Return health entry for a league, creating it if missing."""
    return _html_health.setdefault(league, {
        "last_success_at": 0.0,
        "last_attempt_at": 0.0,
        "consecutive_failures": 0,
        "last_error": "",
        "schema_break": False,
    })


def _html_is_broken(league: str) -> bool:
    """True if HTML scrape has been failing repeatedly OR schema-broke."""
    h = _html_health_now(league)
    return h["consecutive_failures"] >= _HTML_FAILURE_THRESHOLD or h["schema_break"]


def _live_standings_via_html(league: str) -> dict:
    """Scrape live standings from Sofascore tournament HTML page.

    Self-instrumenting: tracks success/failure per league. Sentinel-checks the
    parsed payload (must contain a known team) so silent schema breaks trip
    the breaker. Returns {} on any failure — caller falls back to parquet.
    """
    import time as _t
    now = _t.time()
    h = _html_health_now(league)

    # Cache hit on a recent successful payload — short-circuit
    if league in _html_standings_cache:
        cached_at, payload = _html_standings_cache[league]
        if (now - cached_at) < _HTML_STANDINGS_TTL:
            import copy
            return copy.deepcopy(payload)

    h["last_attempt_at"] = now
    slug_info = _LEAGUE_SOFASCORE_PAGE.get(league)
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
        r = s.get(url, timeout=8)
        if r.status_code != 200:
            err = f"HTTP {r.status_code}"
            log.warning("HTML standings %s: %s", league, err)
            h["last_error"] = err
            h["consecutive_failures"] += 1
            return {}

        import re as _re
        m = _re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', r.text, _re.DOTALL)
        if not m:
            err = "NEXT_DATA script not found"
            log.error("HTML standings %s schema break: %s", league, err)
            h["last_error"] = err
            h["schema_break"] = True
            h["consecutive_failures"] += 1
            return {}

        data = json.loads(m.group(1))
        st_list = data.get("props", {}).get("pageProps", {}).get("initialProps", {}).get("standings", [])
        if not st_list or not isinstance(st_list, list):
            err = "standings list missing in NEXT_DATA"
            log.error("HTML standings %s schema break: %s", league, err)
            h["last_error"] = err
            h["schema_break"] = True
            h["consecutive_failures"] += 1
            return {}

        rows = st_list[0].get("rows", [])
        if not rows:
            err = "rows array empty"
            log.error("HTML standings %s schema break: %s", league, err)
            h["last_error"] = err
            h["schema_break"] = True
            h["consecutive_failures"] += 1
            return {}

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
        sentinel = _HTML_SENTINEL_TEAM.get(league)
        if sentinel and sentinel not in teams:
            err = f"sentinel team {sentinel!r} missing — schema break"
            log.error("HTML standings %s: %s. Found teams: %s",
                      league, err, sorted(teams.keys())[:5])
            h["last_error"] = err
            h["schema_break"] = True
            h["consecutive_failures"] += 1
            return {}

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
        err = f"{type(e).__name__}: {str(e)[:120]}"
        log.warning("HTML standings %s failed: %s", league, err)
        h["last_error"] = err
        h["consecutive_failures"] += 1
        return {}


def _next_fixture_for_team(league: str, team: str) -> dict | None:
    """Find the next not-yet-played fixture for a team from Sofascore fixtures.

    Falls back gracefully when predictions.json is stale (Odds API outage).
    Returns a normalized dict matching the predictions.json shape so consumers
    don't need branching, or None if no upcoming fixture exists.
    """
    import json
    from datetime import datetime, timezone
    from config.settings import PROJECT_ROOT
    from config.team_names import normalize_team

    rel = _LEAGUE_FIXTURES_FILE.get(league)
    if not rel:
        return None
    path = PROJECT_ROOT / rel
    if not path.exists():
        return None
    try:
        with open(path) as f:
            fixtures = json.load(f)
    except Exception:
        return None
    if not isinstance(fixtures, list):
        return None

    team_norm = normalize_team(team)
    now_ts = datetime.now(timezone.utc).timestamp()

    candidates = []
    for fx in fixtures:
        if not isinstance(fx, dict):
            continue
        st = fx.get("status", {}).get("type", "")
        if st in ("finished", "canceled", "postponed"):
            continue
        ts = fx.get("startTimestamp")
        if not ts or ts <= now_ts:
            continue  # already kicked off or no time
        ht = fx.get("homeTeam", {}).get("name", "")
        at = fx.get("awayTeam", {}).get("name", "")
        if normalize_team(ht) == team_norm or normalize_team(at) == team_norm:
            candidates.append((ts, fx, ht, at))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    ts, fx, ht, at = candidates[0]
    kickoff = datetime.fromtimestamp(ts, tz=timezone.utc)
    is_home = normalize_team(ht) == team_norm
    return {
        "match": f"{ht} vs {at}",
        "home_team": ht,
        "away_team": at,
        "date": kickoff.strftime("%Y-%m-%d"),
        "commence_time": kickoff.isoformat(),
        "venue": _HOME_VENUES.get(ht, "") if "_HOME_VENUES" in globals() else "",
        "is_home": is_home,
        "matchweek": fx.get("roundInfo", {}).get("round"),
        "predicted_outcome": "",
        "confidence": "",
        "probabilities": {},
        "betting_probabilities": {},
        "source": "sofascore_fixtures",  # marker so UI can flag "no model output yet"
    }


def _compute_standings(league: str) -> dict:
    """Compute live standings from the Sofascore parquet for a league.

    Returns the same shape as standings.json:
        {
            "standings": { team_name: { position, team, played, wins, draws,
                                        losses, gf, ga, gd, points, form_last5,
                                        home: {...}, away: {...}, league } },
            "current_matchweek": int,
            "season": "2025-2026",
            "league": "serie_a",
        }

    Cached for 60 seconds. Source files are the same parquets the Results page
    reads, so standings can never disagree with results.
    """
    import pandas as pd
    from config.settings import PROJECT_ROOT

    now = _time.time()
    if league in _standings_cache:
        cached_at, payload = _standings_cache[league]
        if (now - cached_at) < _STANDINGS_TTL:
            import copy
            return copy.deepcopy(payload)

    rel = _LEAGUE_PARQUET.get(league)
    if not rel:
        return {"standings": {}, "current_matchweek": 0, "season": "", "league": league}

    parquet_path = PROJECT_ROOT / rel
    df = _read_parquet_cached(parquet_path)
    if df is None or len(df) == 0:
        return {"standings": {}, "current_matchweek": 0, "season": "", "league": league}

    season = get_current_season()
    df = df[df["season"] == season]
    if len(df) == 0:
        return {"standings": {}, "current_matchweek": 0, "season": season, "league": league}

    # One row per match (parquet has one row per player; collapse to match-level)
    matches = df[
        ["match_id", "date", "round", "home_team", "away_team", "home_score", "away_score"]
    ].drop_duplicates(subset=["match_id"])
    matches = matches.dropna(subset=["home_score", "away_score"])
    if len(matches) == 0:
        return {"standings": {}, "current_matchweek": 0, "season": season, "league": league}

    teams: dict[str, dict] = {}

    def _row(t: str) -> dict:
        if t not in teams:
            teams[t] = {
                "team": t,
                "played": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "gf": 0,
                "ga": 0,
                "gd": 0,
                "points": 0,
                "_form": [],  # list of (date, result) for form_last5
                "home": {"played": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0, "ppg": 0.0},
                "away": {"played": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0, "ppg": 0.0},
                "league": league,
            }
        return teams[t]

    matches_sorted = matches.sort_values("date")
    for _, m in matches_sorted.iterrows():
        ht, at = m["home_team"], m["away_team"]
        hs, ascore = int(m["home_score"]), int(m["away_score"])
        h, a = _row(ht), _row(at)

        h["played"] += 1; a["played"] += 1
        h["gf"] += hs; h["ga"] += ascore
        a["gf"] += ascore; a["ga"] += hs
        h["home"]["played"] += 1; a["away"]["played"] += 1
        h["home"]["gf"] += hs; h["home"]["ga"] += ascore
        a["away"]["gf"] += ascore; a["away"]["ga"] += hs

        if hs > ascore:
            h["wins"] += 1; h["points"] += 3; h["home"]["wins"] += 1
            a["losses"] += 1; a["away"]["losses"] += 1
            hr, ar = "W", "L"
        elif hs < ascore:
            a["wins"] += 1; a["points"] += 3; a["away"]["wins"] += 1
            h["losses"] += 1; h["home"]["losses"] += 1
            hr, ar = "L", "W"
        else:
            h["draws"] += 1; h["points"] += 1; h["home"]["draws"] += 1
            a["draws"] += 1; a["points"] += 1; a["away"]["draws"] += 1
            hr, ar = "D", "D"
        h["_form"].append((m["date"], hr))
        a["_form"].append((m["date"], ar))

    # Finalize per-team derived fields
    for t in teams.values():
        t["gd"] = t["gf"] - t["ga"]
        # form_last5: last 5 results in chronological order, oldest→newest
        t["_form"].sort(key=lambda x: x[0])
        t["form_last5"] = "".join(r for _, r in t["_form"][-5:])
        del t["_form"]
        # PPG home/away
        hp = t["home"]["played"]
        ap = t["away"]["played"]
        if hp:
            t["home"]["ppg"] = round((t["home"]["wins"] * 3 + t["home"]["draws"]) / hp, 2)
        if ap:
            t["away"]["ppg"] = round((t["away"]["wins"] * 3 + t["away"]["draws"]) / ap, 2)

    # Rank by points (desc), then GD (desc), then GF (desc) — standard tie-break
    ranked = sorted(teams.values(), key=lambda r: (-r["points"], -r["gd"], -r["gf"]))
    for i, t in enumerate(ranked, 1):
        t["position"] = i

    standings_dict = {t["team"]: t for t in ranked}

    payload = {
        "standings": standings_dict,
        "current_matchweek": int(matches["round"].max()),
        "season": season,
        "league": league,
    }
    _standings_cache[league] = (now, payload)
    import copy
    return copy.deepcopy(payload)


def _get_standings(league: str) -> dict:
    """Public accessor: live standings for a league.

    Order of preference:
      1. Sofascore HTML scrape (live, current; bypasses the API CF block)
      2. Parquet-derived (fast but can lag if scraper hasn't run)
      3. legacy standings.json on disk

    HTML is preferred whenever it returns more `played` than the parquet —
    that's the signal the parquet hasn't ingested the latest matchweek.
    """
    parquet_payload = _compute_standings(league)
    parquet_max_played = 0
    if parquet_payload.get("standings"):
        parquet_max_played = max(
            (t.get("played", 0) for t in parquet_payload["standings"].values()),
            default=0,
        )

    html_payload = _live_standings_via_html(league)
    html_max_played = 0
    if html_payload.get("standings"):
        html_max_played = max(
            (t.get("played", 0) for t in html_payload["standings"].values()),
            default=0,
        )

    # Prefer HTML when it's at least as fresh
    if html_payload.get("standings") and html_max_played >= parquet_max_played:
        # Splice form/home/away splits from parquet (HTML doesn't expose these).
        # Sofascore HTML uses long names ("Manchester City") while the parquet
        # often uses Sofascore short names ("Man City"). Normalize before lookup.
        if parquet_payload.get("standings"):
            try:
                from config.team_names import normalize_team
                pq_by_norm = {
                    normalize_team(str(k)): v
                    for k, v in parquet_payload["standings"].items()
                }
            except Exception:
                pq_by_norm = parquet_payload["standings"]
            for team_name, html_row in html_payload["standings"].items():
                try:
                    norm_key = normalize_team(team_name)
                except Exception:
                    norm_key = team_name
                pq_row = pq_by_norm.get(norm_key) or parquet_payload["standings"].get(team_name)
                if pq_row:
                    # Form letters: always splice if available, regardless of MW gap
                    if pq_row.get("form_last5"):
                        html_row["form_last5"] = pq_row["form_last5"]
                    # Home/away splits: only splice if MWs match (else stale splits would lie)
                    if parquet_max_played == html_max_played:
                        if pq_row.get("home"):
                            html_row["home"] = pq_row["home"]
                        if pq_row.get("away"):
                            html_row["away"] = pq_row["away"]
        return html_payload

    if parquet_payload.get("standings"):
        return parquet_payload

    # Last resort: legacy standings.json
    fname = "standings.json" if league == "serie_a" else f"standings_{league}.json"
    raw = _load_json(UPCOMING_DIR / fname, {})
    if isinstance(raw, dict) and raw.get("standings"):
        return raw
    return parquet_payload



def _get_betting_stats(league_filter: str = None):
    """Aggregate wins/losses/pushes/ROI from history.json.

    Args:
        league_filter: If set, only count bets belonging to this league.
    """
    history_raw = _load_json(BETTING_DIR / "history.json")
    settled_bets = []
    history_totals = {}
    if isinstance(history_raw, dict):
        settled_bets = history_raw.get("settled_bets", [])
        history_totals = history_raw.get("totals", {})
    elif isinstance(history_raw, list):
        settled_bets = history_raw

    for bet in settled_bets:
        outcome = bet.get("outcome", bet.get("status", "")).upper()
        bet["status"] = outcome.lower()
        bet["profit"] = bet.get("profit_loss", bet.get("profit", 0))

    # Apply league filter if specified
    if league_filter:
        settled_bets = [b for b in settled_bets if _match_belongs_to_league(b, league_filter)]

    settled_only = [b for b in settled_bets if b.get("status") in ("won", "lost", "push")]
    wins = sum(1 for b in settled_only if b.get("status") == "won")
    losses = sum(1 for b in settled_only if b.get("status") == "lost")
    pushes = sum(1 for b in settled_only if b.get("status") == "push")
    total_stake = sum(b.get("stake", 0) for b in settled_only)
    total_profit = sum(b.get("profit", 0) for b in settled_only)

    # Only use pre-aggregated totals when NOT league-filtering (they're global)
    if history_totals and not league_filter:
        total_stake = history_totals.get("total_staked", total_stake)
        total_profit = history_totals.get("net_profit", total_profit)
        wins = history_totals.get("wins", wins)
        losses = history_totals.get("losses", losses)
        pushes = history_totals.get("pushes", pushes)

    return {
        "wins": wins, "losses": losses, "pushes": pushes,
        "settled_bets": wins + losses + pushes,
        "total_stake": round(total_stake, 2), "total_profit": round(total_profit, 2),
        "win_rate": round(wins / (wins + losses + pushes) * 100, 1) if (wins + losses + pushes) > 0 else 0,
        "roi": round(total_profit / total_stake * 100, 1) if total_stake > 0 else 0,
    }


_SQUAD_CACHE: dict = {}
_SQUAD_CACHE_TTL = 300  # 5 minutes


_EPL_TEAM_NAME_MAP = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Brighton and Hove Albion": "Brighton",
    "Wolverhampton Wanderers": "Wolves",
    "West Ham United": "West Ham",
    "Tottenham Hotspur": "Tottenham",
    "Leeds United": "Leeds",
    "Ipswich Town": "Ipswich",
    "Leicester City": "Leicester",
    "Nott'm Forest": "Nottingham Forest",
}


def _load_team_squad_roster(team: str, current_season: str = "2025-2026", limit: int = 18) -> list:
    """Aggregate per-player season stats for a team. Returns top N by minutes.

    Reads the league-appropriate Sofascore parquet (SA or EPL). Computes per
    player: minutes total, matches, goals, assists, xg, xa, average rating,
    shots/match, key_passes/match, position. Used by /api/team/<name> to
    populate squad.players.
    """
    cache_key = (team, current_season, limit)
    now = _time.time()
    if cache_key in _SQUAD_CACHE:
        cached_at, cached_data = _SQUAD_CACHE[cache_key]
        if (now - cached_at) < _SQUAD_CACHE_TTL:
            return cached_data

    try:
        from config.leagues import infer_league
        league = infer_league(team)
    except Exception:
        league = "serie_a"

    if league == "premier_league":
        path = DATA_DIR / "external" / "sofascore" / "player_match_stats_premier_league.parquet"
        normalized_team = _EPL_TEAM_NAME_MAP.get(team, team)
    else:
        path = DATA_DIR / "external" / "sofascore" / "player_match_stats.parquet"
        normalized_team = "Hellas Verona" if team == "Verona" else team

    if not path.exists():
        _SQUAD_CACHE[cache_key] = (now, [])
        return []

    try:
        import pandas as pd
        df = pd.read_parquet(path)
        df = df[df["season"] == current_season]
        if df.empty:
            _SQUAD_CACHE[cache_key] = (now, [])
            return []

        # Try the normalized name first; fall back to team_name as-is.
        team_data = df[df["team"] == normalized_team]
        if team_data.empty and normalized_team != team:
            team_data = df[df["team"] == team]
        if team_data.empty:
            _SQUAD_CACHE[cache_key] = (now, [])
            return []

        # Per-player aggregation
        roster = []
        for player_name, g in team_data.groupby("player_name"):
            mins = int(g["minutes"].sum() or 0)
            if mins < 90:  # less than one full match — exclude noise
                continue
            n_matches = int(len(g))
            goals = int(g["goals"].sum() or 0)
            assists = int(g["assists"].sum() or 0)
            xg_total = float(g["xg"].sum() or 0)
            xa_total = float(g["xa"].sum() or 0)
            avg_rating = float(g["rating"].mean() or 0)
            shots_per_match = float(g["total_shots"].mean() or 0)
            kp_per_match = float(g["key_passes"].mean() or 0)
            # Position: take the most common
            try:
                pos = g["position"].mode().iloc[0] if not g["position"].mode().empty else ""
            except Exception:
                pos = ""

            roster.append({
                "name": player_name,
                "position": str(pos),
                "minutes": mins,
                "matches": n_matches,
                "goals": goals,
                "assists": assists,
                "xg": round(xg_total, 2),
                "xa": round(xa_total, 2),
                "rating": round(avg_rating, 2),
                "shots_per_match": round(shots_per_match, 1),
                "key_passes_per_match": round(kp_per_match, 1),
                "goals_per_90": round(goals / (mins / 90), 2) if mins > 0 else 0,
                "assists_per_90": round(assists / (mins / 90), 2) if mins > 0 else 0,
            })

        # Top N by minutes
        roster.sort(key=lambda p: p["minutes"], reverse=True)
        roster = roster[:limit]

        _SQUAD_CACHE[cache_key] = (now, roster)
        return roster
    except Exception as e:
        log.warning("Failed to load squad roster for %s: %s", team, e)
        _SQUAD_CACHE[cache_key] = (now, [])
        return []


def _index_list_by_match(items: list, home_key="home_team", away_key="away_team") -> dict:
    """Convert a list of match dicts into a dict keyed by 'Home vs Away'."""
    result = {}
    for item in items:
        if isinstance(item, dict):
            h = item.get(home_key, "")
            a = item.get(away_key, "")
            key = item.get("match", "")
            if not key and h and a:
                key = f"{h} vs {a}"
            if key:
                if key in result:
                    log.debug("Duplicate match key '%s' in index — keeping latest entry", key)
                result[key] = item
    return result


def _compute_market_edge(pred: dict, odds_data: dict) -> float:
    """Compute market edge from model probabilities vs implied odds probability.

    Returns the edge for the predicted outcome (model_prob - implied_prob).
    Returns 0 if odds data is missing.
    """
    probs = pred.get("probabilities", {})
    h2h = odds_data.get("h2h", {})
    if not h2h or not probs:
        return 0

    predicted = pred.get("predicted_outcome", "HOME")
    home_odds = h2h.get("home", 0)
    draw_odds = h2h.get("draw", 0)
    away_odds = h2h.get("away", 0)

    if not all(o > 1 for o in [home_odds, draw_odds, away_odds]):
        return 0

    # Remove overround to get fair implied probabilities
    overround = (1/home_odds) + (1/draw_odds) + (1/away_odds)
    if overround <= 0:
        return 0

    implied = {
        "HOME": (1/home_odds) / overround,
        "DRAW": (1/draw_odds) / overround,
        "AWAY": (1/away_odds) / overround,
    }

    model_prob = probs.get(predicted.lower(), probs.get("home", 0))
    implied_prob = implied.get(predicted, 0.33)

    return round(model_prob - implied_prob, 4)


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@app.route("/")
def dashboard():
    return render_template("dashboard.html", active_page="dashboard")


@app.route("/betting")
def betting():
    return render_template("betting.html", active_page="betting")


@app.route("/analytics")
def analytics():
    return render_template("analytics.html", active_page="analytics")


@app.route("/performance")
def performance_page():
    return render_template("performance.html", active_page="performance")


@app.route("/api/performance")
def api_performance():
    try:
        from scripts.betting.benchmark_tracker import get_benchmark_report
        return jsonify(get_benchmark_report())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/quota")
def api_quota():
    """The Odds API quota snapshot: remaining credits, days to reset, undercount warnings.

    Surfaces the authoritative header-derived numbers from odds_fetcher so the
    dashboard never silently over-burns again. Frontend can poll this.
    """
    try:
        from scripts.data.odds_fetcher import get_usage_summary
        s = get_usage_summary()
        # Derive a simple status for the UI card
        remaining = s.get("api_remaining")
        if remaining is None:
            status = "unknown"
        elif remaining < 500:
            status = "hard_stop"
        elif remaining < 2000:
            status = "soft_stop"
        else:
            status = "ok"
        s["status"] = status
        return jsonify(s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/system")
def system():
    return render_template("system.html", active_page="system")


@app.route("/live")
def live_page():
    return render_template("live.html", active_page="live")


@app.route("/matches")
def matches_page():
    return render_template("matches.html", active_page="matches")


@app.route("/projections")
@app.route("/proj")
@app.route("/score-projections")
def projections_page():
    # no-cache so the browser never serves a stale version of this evolving page
    resp = app.make_response(render_template("projections.html", active_page="projections"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


_COMPARATIVE_CACHE: dict = {"df": None, "mtime": 0.0}
_PLAYER_ENGINE_CACHE: dict = {"pms": None, "base_rates": None, "mtime": 0.0}

# Markets shown on the page, in display order. Goalscorer last (weakest, +6.9%).
_FLOOR_DISPLAY_MARKETS = [
    "shots_o15", "sot_o05", "shots_o05", "shots_o25",
    "sot_o15", "fouls_o05", "fouled_o05", "goalscorer",
]


def _player_engine():
    """Cached (pms-with-priors, base_rates) for the player floor engine.

    Rebuilds only when the underlying parquet changes. Building priors over
    100k rows is ~1s, so we never want to do it per request.
    """
    path = DATA_DIR / "external" / "sofascore" / "player_match_stats.parquet"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None, None
    if _PLAYER_ENGINE_CACHE["pms"] is None or _PLAYER_ENGINE_CACHE["mtime"] != mtime:
        from scripts.betting.player_predictions import (
            load_player_data, build_player_features, compute_position_base_rates,
        )
        pms = build_player_features(load_player_data("serie_a"))
        _PLAYER_ENGINE_CACHE["pms"] = pms
        _PLAYER_ENGINE_CACHE["base_rates"] = compute_position_base_rates(pms)
        _PLAYER_ENGINE_CACHE["mtime"] = mtime
    return _PLAYER_ENGINE_CACHE["pms"], _PLAYER_ENGINE_CACHE["base_rates"]


def _lineup_entries(side_lineup: dict) -> list:
    """Flatten a lineup_predictions side into [{player_name, player_id, position,
    proj_minutes, is_starter}], using start_pct * avg_minutes for projected mins.
    """
    out = []
    for grp, starter in (("predicted_xi", True), ("bench", False)):
        for p in side_lineup.get(grp, []) or []:
            avg_min = float(p.get("avg_minutes") or (82.0 if starter else 20.0))
            start_pct = float(p.get("start_pct") or (100.0 if starter else 0.0))
            proj_min = avg_min if starter else avg_min * (start_pct / 100.0)
            out.append({
                "player_name": p.get("name") or p.get("player_name", ""),
                "player_id": p.get("player_id"),
                "position": p.get("position", "M"),
                "proj_minutes": proj_min,
                "is_starter": starter,
            })
    return out


def _attach_player_floors(proj_by_match: dict) -> None:
    """Attach top player floor markets to each projection (display only).

    For each match with a predicted lineup, predicts every player's floor
    markets and keeps, per side, the players with the strongest single floor —
    so the page shows "who's most likely to shoot / be fouled / score".
    """
    pms, base_rates = _player_engine()
    if pms is None:
        return
    from scripts.betting.player_predictions import predict_match_players

    lineups = _load_json(UPCOMING_DIR / "lineup_predictions.json", {})
    lp_matches = lineups.get("matches", {}) if isinstance(lineups, dict) else {}

    def _recent_xi(team):
        """Fallback likely-XI: the team's most-recent match starters from pms.

        Used when lineup_predictions has no matching entry (off-season / stale
        lineup file). Names + ids come straight from sofascore so they resolve.
        """
        tdf = pms[pms["team"] == team]
        if not len(tdf):
            return []
        last_date = tdf["date"].max()
        last = tdf[(tdf["date"] == last_date) & (tdf["minutes"] >= 60)]
        return [{
            "player_name": r["player_name"], "player_id": r["player_id"],
            "position": r["position"], "is_starter": True,
            "proj_minutes": None,  # engine uses the player's leak-free min_prior
        } for _, r in last.iterrows()]

    from scripts.betting.player_predictions import TARGETS as _FLOOR_TARGETS
    # Only grade HIT/MISS above this confidence — a 17% goal call that doesn't
    # hit is the model being RIGHT, not a miss (advisor). Below threshold we
    # show the neutral outcome (occurred / didn't), never a red ❌.
    _GRADE_MIN_PROB = 0.55

    def _actual_xi(home, away):
        """For a PAST match: the actual starters + their actual stats, joined on
        date+teams (not teams alone — double fixtures would mis-grade). Returns
        ({name: actual_row}, match_date) or ({}, None) if the match hasn't been
        played / isn't in pms.
        """
        mrows = pms[((pms["team"] == home) & (pms["opponent"] == away))
                    | ((pms["team"] == away) & (pms["opponent"] == home))]
        if not len(mrows):
            return {}, None
        last_date = mrows["date"].max()
        played = mrows[(mrows["date"] == last_date) & (mrows["is_starter"] == True)]  # noqa: E712
        return {r["player_name"]: r for _, r in played.iterrows()}, last_date

    for match_key, proj in proj_by_match.items():
        lp = lp_matches.get(match_key)
        # For a PAST match, prefer the ACTUAL starters (so display + grading show
        # the same real XI). Detect "past" = we have the played row in pms.
        actual_by_name, _ = _actual_xi(proj.get("home_team", ""), proj.get("away_team", ""))
        is_past = bool(actual_by_name)
        if lp and not is_past:
            home_xi = _lineup_entries(lp.get("home_lineup", {}))
            away_xi = _lineup_entries(lp.get("away_lineup", {}))
        else:
            home_xi = _recent_xi(proj.get("home_team", ""))
            away_xi = _recent_xi(proj.get("away_team", ""))
        if not home_xi and not away_xi:
            continue
        result = predict_match_players(
            proj.get("home_team", ""), proj.get("away_team", ""),
            home_xi, away_xi, pms=pms, base_rates=base_rates,
        )

        def _grade(name, market_key, prob):
            """Grade one player market vs actual. Returns dict or None.
            hit: True/False only when prob>=threshold; else outcome shown neutral."""
            row = actual_by_name.get(name)
            if row is None:
                return None
            cfg = _FLOOR_TARGETS.get(market_key)
            if not cfg:
                return None
            occurred = bool(row[cfg["col"]] >= cfg["line"] + 1)
            graded = prob >= _GRADE_MIN_PROB
            return {
                "occurred": occurred,
                "actual": int(row[cfg["col"]]),
                "hit": occurred if graded else None,  # None = neutral (sub-threshold)
            }

        def _trim(players):
            rows = []
            for pl in players:
                mk = pl.get("markets", {})
                # headline = the player's most-confident displayable floor
                best = max(
                    (mk[k] for k in _FLOOR_DISPLAY_MARKETS if k in mk),
                    key=lambda m: m["prob"], default=None,
                )
                if not best or best["prob"] < 0.30:
                    continue
                nm = pl["player_name"]
                # grade each displayed market vs actual (past matches only)
                markets = {}
                for k in _FLOOR_DISPLAY_MARKETS:
                    if k not in mk:
                        continue
                    cell = {"label": mk[k]["label"], "prob": mk[k]["prob"],
                            "calibrated": mk[k].get("calibrated", False)}
                    g = _grade(nm, k, mk[k]["prob"]) if is_past else None
                    if g:
                        cell.update(g)
                    markets[k] = cell
                # headline carries the best market's grade for the collapsed row
                best_key = next((k for k in _FLOOR_DISPLAY_MARKETS
                                 if k in mk and mk[k] is best), None)
                hg = _grade(nm, best_key, best["prob"]) if (is_past and best_key) else None
                rows.append({
                    "name": nm,
                    "position": pl["position"],
                    "proj_minutes": pl["proj_minutes"],
                    "prior_matches": pl.get("prior_matches", 0),
                    "headline": {"label": best["label"], "prob": best["prob"],
                                 **({"hit": hg["hit"], "actual": hg["actual"]} if hg else {})},
                    "markets": markets,
                })
            rows.sort(key=lambda r: r["headline"]["prob"], reverse=True)
            return rows[:6]

        src = "actual XI (graded vs result)" if is_past else (
            "predicted XI" if lp else "last XI (lineup TBD)")
        proj["player_floors"] = {
            "home": _trim(result.get("home_players", [])),
            "away": _trim(result.get("away_players", [])),
            "lineup_source": src,
            "is_past": is_past,
            "note": (f"Leak-free predictions vs ACTUAL result · {src}. "
                     "HIT/MISS only on ≥55% calls (a 17% call not hitting is correct, not a miss)."
                     if is_past else
                     f"Validated leak-free model · {src}. Display only — betting gated."),
        }


def _comparative_matches_df():
    """Cached read of matches.parquet for comparative markets (re-reads if file changes)."""
    import pandas as pd
    path = DATA_DIR / "parsed" / "matches.parquet"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    if _COMPARATIVE_CACHE["df"] is None or _COMPARATIVE_CACHE["mtime"] != mtime:
        cols = ["match_date", "home_team", "away_team", "referee",
                "home_score", "away_score",
                "home_corners", "away_corners", "home_fouls", "away_fouls",
                "home_yellow_cards", "away_yellow_cards",
                "home_shots_total", "away_shots_total",
                "home_shots_on_target_count", "away_shots_on_target_count"]
        # read only columns that exist (tolerant of schema changes)
        import pyarrow.parquet as _pq
        present = set(_pq.ParquetFile(path).schema.names)
        df = pd.read_parquet(path, columns=[c for c in cols if c in present])
        _COMPARATIVE_CACHE["df"] = df
        _COMPARATIVE_CACHE["mtime"] = mtime
    return _COMPARATIVE_CACHE["df"]


def _grade_pick(pick: dict, actual: dict) -> bool | None:
    """Did a pick hit, given the actual result? None if not gradable."""
    key = pick.get("key", "")
    sel = (pick.get("pick") or "").lower()
    res = actual["result"]
    if key == "1x2_home":
        return res == "home"
    if key == "1x2_away":
        return res == "away"
    if key == "1x2_draw":
        return res == "draw"
    if key == "double_chance":
        if "1x" in sel or "or draw" in sel:
            return res in ("home", "draw")
        if "x2" in sel:
            return res in ("draw", "away")
        if "12" in sel:
            return res in ("home", "away")
    if key == "ou_2.5":
        return (actual["total_goals"] > 2.5) if "over" in sel else (actual["total_goals"] < 2.5)
    if key == "btts":
        return actual["btts"] if ("goal" in sel or "yes" in sel) else not actual["btts"]
    if key.startswith("ref_cards"):
        ln = float(key.split("_o")[-1])
        return (actual["total_cards"] > ln) if "over" in sel else (actual["total_cards"] < ln)
    if key.startswith("ref_fouls"):
        ln = float(key.split("_o")[-1])
        return (actual["total_fouls"] > ln) if "over" in sel else (actual["total_fouls"] < ln)
    if key == "corners_more":
        # graded only if we have corner data on the actual row (handled by caller)
        return None
    return None


def _build_score_range_projection(pred: dict) -> dict | None:
    """Derive the full score-range market family for one match from its xG.

    Everything here rides the existing calibrated goal/xG Poisson model — the
    only inputs are home_xg / away_xg (already in predictions.json) plus the
    1X2 probabilities. No new model; we read a model that works. All math lives
    in scripts.betting.extended_markets (tested), this just wires it per-match.
    """
    from scripts.betting import extended_markets as em

    hxg = pred.get("home_xg")
    axg = pred.get("away_xg")
    if hxg is None or axg is None:
        return None
    try:
        hxg = float(hxg)
        axg = float(axg)
    except (TypeError, ValueError):
        return None
    if hxg <= 0 or axg <= 0:
        return None

    probs = pred.get("probabilities") or {}
    # normalise 1X2 prob keys (predictions.json uses home/draw/away)
    p1x2 = {
        "home": probs.get("home", probs.get("H", 0.0)),
        "draw": probs.get("draw", probs.get("D", 0.0)),
        "away": probs.get("away", probs.get("A", 0.0)),
    }

    # BTTS (Goal/No Goal) — Poisson: P(both score) = (1-P(home=0))*(1-P(away=0)).
    # Independence is the standard approximation; the rho-correlated matrix is
    # used for score-grid markets above where it matters more.
    import math as _math
    btts_yes = (1 - _math.exp(-hxg)) * (1 - _math.exp(-axg))
    btts_yes = max(0.0, min(1.0, btts_yes))
    btts = {
        "yes": {"prob": round(btts_yes, 4),
                "fair_odds": round(1 / btts_yes, 2) if btts_yes > 0.001 else 99},
        "no": {"prob": round(1 - btts_yes, 4),
               "fair_odds": round(1 / (1 - btts_yes), 2) if (1 - btts_yes) > 0.001 else 99},
    }

    return {
        "match": pred.get("match"),
        "home_team": pred.get("home_team"),
        "away_team": pred.get("away_team"),
        "date": pred.get("date", ""),
        "time": pred.get("time", ""),
        "home_xg": round(hxg, 2),
        "away_xg": round(axg, 2),
        "predicted_outcome": pred.get("predicted_outcome"),
        "probabilities": p1x2,
        # The full tipster market menu, all Poisson-derived:
        "btts": btts,
        "double_chance": em.compute_double_chance(p1x2),
        "multi_goal": em.compute_multi_goal(hxg, axg),
        "exact_score": em.compute_exact_score_top10(hxg, axg),
        "htft": em.compute_htft(hxg, axg),
        "first_half": em.compute_first_half(hxg, axg),
        "second_half": em.compute_second_half_result(hxg, axg),
        "team_totals": em.compute_team_totals(hxg, axg),
        "winning_margin": em.compute_winning_margin(hxg, axg),
        "odd_even": em.compute_odd_even(hxg, axg),
        "goal_both_halves": em.compute_goal_in_both_halves(hxg, axg),
        "team_to_score_first": em.compute_team_to_score_first(hxg, axg),
        "win_to_nil": em.compute_win_to_nil(hxg, axg, p1x2),
        "european_handicap": em.compute_european_handicap(hxg, axg),
        "somma_goal": em.compute_somma_goal(hxg, axg),
        "team_odd_even": em.compute_team_odd_even(hxg, axg),
        "ribaltone": em.compute_ribaltone(hxg, axg),
        "combos": em.compute_combos(hxg, axg),
        "team_win_to_nil": em.compute_team_win_to_nil(hxg, axg),
        "half_goals": em.compute_half_goals_ou(hxg, axg),
    }


@app.route("/api/projections")
def api_projections():
    """Score-range projection family for every upcoming match.

    Display-only insight (NOT betting) — derived from the calibrated goal model.
    Betting on these markets is gated on the Betfair odds layer + per-market
    validation (see .plans/prop-projection-product-plan.md). These are honest
    model probabilities, not edges.
    """
    predictions_raw = _load_json(UPCOMING_DIR / "predictions.json")
    if isinstance(predictions_raw, dict):
        preds = predictions_raw.get("predictions", [])
        generated_at = predictions_raw.get("generated_at", "")
    else:
        preds = predictions_raw or []
        generated_at = ""

    projections = []
    for p in preds:
        proj = _build_score_range_projection(p)
        if proj is not None:
            projections.append(proj)

    # Attach comparative team-stat markets (who makes more corners/fouls/cards).
    # Opponent-adjusted; fouls has real signal, corners/cards fall back to base rate.
    try:
        from scripts.prediction.comparative_markets import (
            compute_expected_counts, all_comparative_markets, compute_base_rates,
            total_cards_over_under, ref_card_avg, team_card_rate,
            total_fouls_over_under, ref_stat_avg, team_stat_rate)
        _matches = _comparative_matches_df()   # cached read (see helper)
        _base_rates = compute_base_rates(_matches) if _matches is not None else {}
        # predictions.json carries the assigned referee per match
        ref_by_match = {p.get("match"): p.get("referee")
                        for p in (preds if isinstance(preds, list) else [])}
        if _matches is not None:
            for proj in projections:
                exp = compute_expected_counts(proj.get("home_team"), proj.get("away_team"), _matches)
                if exp:
                    proj["comparative"] = all_comparative_markets(exp, base_rates=_base_rates)
                # referee-aware total-cards O/U (the validated signal market)
                ref = ref_by_match.get(proj.get("match"))
                ra = ref_card_avg(ref, _matches) if ref else None
                hcr = team_card_rate(proj.get("home_team"), _matches)
                acr = team_card_rate(proj.get("away_team"), _matches)
                tc = total_cards_over_under(ra, hcr, acr)
                if tc:
                    tc["referee"] = ref
                    proj["total_cards"] = tc
                # referee-aware total-fouls O/U (second validated signal market)
                rfa = ref_stat_avg(ref, _matches, "fouls") if ref else None
                hfr = team_stat_rate(proj.get("home_team"), _matches, "fouls")
                afr = team_stat_rate(proj.get("away_team"), _matches, "fouls")
                tf = total_fouls_over_under(rfa, hfr, afr)
                if tf:
                    tf["referee"] = ref
                    proj["total_fouls"] = tf
                # shot markets (1X2 tiri + U/O shots / shots-on-target) — display
                from scripts.prediction.comparative_markets import shot_markets
                sm = shot_markets(proj.get("home_team"), proj.get("away_team"), _matches)
                if sm:
                    proj["shots"] = sm
    except Exception as e:
        log.warning("comparative markets skipped: %s", e)

    # Player floor markets (shots / SoT / fouls O-U per likely starter).
    # Validated leak-free engine (see .plans/player-props-deep-plan.md): every
    # market beats base rate on walk-forward Brier. DISPLAY only — betting gated.
    try:
        _attach_player_floors({p.get("match"): p for p in projections})
    except Exception as e:
        log.warning("player floors skipped: %s", e)

    # Calibrate the overclaiming markets so the displayed % = real hit rate
    # (live isotonic maps from 2017+ history; away-win/O-U overclaim otherwise).
    try:
        from scripts.prediction.market_calibration import calibrate
        _mp = str(DATA_DIR / "features" / "features_serie_a.parquet")
        for proj in projections:
            pr = proj.get("probabilities") or {}
            for side, key in (("home", "1x2_home"), ("draw", "1x2_draw"), ("away", "1x2_away")):
                if pr.get(side) is not None:
                    pr[side] = calibrate(key, pr[side], _mp)
            b = proj.get("btts") or {}
            if b.get("yes", {}).get("prob") is not None:
                cy = calibrate("btts", b["yes"]["prob"], _mp)
                b["yes"]["prob"] = cy
                if "no" in b:
                    b["no"]["prob"] = round(1 - cy, 4)
            # double chance — backtest showed ECE 0.075 (miscalibrated); calibrate each leg
            dc = proj.get("double_chance") or {}
            for leg, key in (("1X", "dc_1x"), ("X2", "dc_x2"), ("12", "dc_12")):
                cell = dc.get(leg)
                if isinstance(cell, dict) and cell.get("prob") is not None:
                    cp = calibrate(key, cell["prob"], _mp)
                    cell["prob"] = round(cp, 4)
                    cell["fair_odds"] = round(1.0 / max(cp, 0.01), 2)
            # team totals over-lines — ECE sweep found 0.066-0.073, held-out validated
            tt = proj.get("team_totals") or {}
            for side in ("home", "away"):
                for line, key in (("over_1.5", f"tt_{side}_o1.5"), ("over_2.5", f"tt_{side}_o2.5")):
                    cell = (tt.get(side) or {}).get(line)
                    if isinstance(cell, dict) and cell.get("prob") is not None:
                        cp = calibrate(key, cell["prob"], _mp)
                        cell["prob"] = round(cp, 4)
                        cell["fair_odds"] = round(1.0 / max(cp, 0.01), 2)
                        # keep under = 1 - over coherent
                        u = (tt.get(side) or {}).get(line.replace("over", "under"))
                        if isinstance(u, dict):
                            u["prob"] = round(1 - cp, 4)
                            u["fair_odds"] = round(1.0 / max(1 - cp, 0.01), 2)
            # multigol 2-3 — calibrated (held-out ECE 0.040 -> 0.006)
            for cell in (proj.get("multi_goal") or []):
                if isinstance(cell, dict) and cell.get("range") == "2-3" and cell.get("prob") is not None:
                    cp = calibrate("multigol_2_3", cell["prob"], _mp)
                    cell["prob"] = round(cp, 4)
                    cell["fair_odds"] = round(1.0 / max(cp, 0.01), 2)
            # european handicap — home outcome at +1/+2 was miscalibrated (held-out
            # 0.075 -> 0.034); calibrate the home leg + keep draw/away renormalized.
            eh = proj.get("european_handicap") or {}
            for lk, key in (("home_+1", "eh_home_+1"), ("home_+2", "eh_home_+2")):
                line = eh.get(lk)
                if isinstance(line, dict) and (line.get("home") or {}).get("prob") is not None:
                    cp = calibrate(key, line["home"]["prob"], _mp)
                    old = line["home"]["prob"]
                    line["home"]["prob"] = round(cp, 4)
                    line["home"]["fair_odds"] = round(1.0 / max(cp, 0.01), 2)
                    # renormalize draw/away to keep the 3 outcomes summing to 1
                    rem = max(0.0, 1.0 - cp)
                    dr, aw = line.get("draw") or {}, line.get("away") or {}
                    base = (dr.get("prob", 0) + aw.get("prob", 0)) or 1.0
                    for cell in (dr, aw):
                        if cell.get("prob") is not None:
                            cell["prob"] = round(rem * cell["prob"] / base, 4)
                            cell["fair_odds"] = round(1.0 / max(cell["prob"], 0.01), 2)
            proj["calibrated"] = True
    except Exception as e:
        log.warning("calibration skipped: %s", e)

    # Best-bets: per match, the strongest plays from backtest-trusted markets only.
    try:
        from scripts.prediction.best_bets import rank_bets
        for proj in projections:
            proj["best_bets"] = rank_bets(
                proj, proj.get("comparative"),
                proj.get("total_cards"), proj.get("total_fouls"))
    except Exception as e:
        log.warning("best-bets skipped: %s", e)

    # Results grading: for COMPLETED matches, attach the actual result and grade
    # whether the ⭐ best-bet hit. Lets the user see if the predictions came true.
    try:
        import pandas as pd
        _m = _comparative_matches_df()
        if _m is not None and "home_score" in _m.columns:
            for proj in projections:
                row = _m[(_m["home_team"] == proj.get("home_team")) &
                         (_m["away_team"] == proj.get("away_team"))].sort_values("match_date").tail(1)
                if not len(row) or pd.isna(row.iloc[0].get("home_score")):
                    continue
                r = row.iloc[0]
                hs, as_ = int(r["home_score"]), int(r["away_score"])
                actual = {
                    "score": f"{hs}-{as_}",
                    "result": "home" if hs > as_ else ("away" if as_ > hs else "draw"),
                    "total_goals": hs + as_,
                    "total_cards": int((r.get("home_yellow_cards") or 0) + (r.get("away_yellow_cards") or 0)),
                    "total_fouls": int((r.get("home_fouls") or 0) + (r.get("away_fouls") or 0)),
                    "btts": hs > 0 and as_ > 0,
                }
                # add corners to actual for grading the corners-more market
                actual["home_corners"] = int(r.get("home_corners") or 0)
                actual["away_corners"] = int(r.get("away_corners") or 0)
                proj["actual"] = actual

                # grade EVERY ranked candidate (all markets the engine considered)
                ranked = (proj.get("best_bets") or {}).get("all_ranked", [])
                graded = []
                for c in ranked:
                    hit = _grade_pick(c, actual)
                    if c.get("key") == "corners_more" and actual.get("home_corners") is not None:
                        hc, ac = actual["home_corners"], actual["away_corners"]
                        sel = (c.get("pick") or "")
                        if "+" in sel and proj.get("home_team", "") in sel:
                            hit = hc > ac
                        elif "+" in sel and proj.get("away_team", "") in sel:
                            hit = ac > hc
                        elif "pari" in sel.lower():
                            hit = hc == ac
                    if hit is None:
                        continue
                    graded.append({**c, "hit": bool(hit)})

                # insights: what we recommended vs what actually won
                rec_hit = [g for g in graded if g["recommended"] and g["hit"]]
                rec_miss = [g for g in graded if g["recommended"] and not g["hit"]]
                # MISSED OPPORTUNITY: high-confidence, hit, but NOT recommended (trust gate kept it off)
                missed_opp = [g for g in graded if not g["recommended"] and g["hit"]
                              and g["prob"] >= 0.60 and g["lift"] >= 0.08]
                # GOOD SKIP: not recommended AND correctly didn't hit
                good_skip = [g for g in graded if not g["recommended"] and not g["hit"]
                             and g["prob"] >= 0.55]
                bs = (proj.get("best_bets") or {}).get("best_single")
                if bs:
                    proj["best_bets"]["best_single"]["hit"] = _grade_pick(bs, actual)
                proj["grading"] = {
                    "n_recommended_hit": len(rec_hit),
                    "n_recommended_miss": len(rec_miss),
                    "recommended": [{"market": g["market"], "pick": g["pick"],
                                     "prob": g["prob"], "hit": g["hit"]} for g in (rec_hit + rec_miss)],
                    "missed_opportunities": [{"market": g["market"], "pick": g["pick"],
                                              "prob": g["prob"]} for g in missed_opp[:3]],
                    "good_skips": [{"market": g["market"], "pick": g["pick"]} for g in good_skip[:3]],
                }
    except Exception as e:
        log.warning("results grading skipped: %s", e)

    # Attach odds-edge comparison when a book's odds are available.
    # comparison_odds.json is written by the Betfair client (or a sample);
    # format {match: {market: {outcome: price}}}. The engine is source-agnostic.
    odds_raw = _load_json(UPCOMING_DIR / "comparison_odds.json", {})
    book_name = odds_raw.get("book", "") if isinstance(odds_raw, dict) else ""
    book_odds_by_match = odds_raw.get("odds", {}) if isinstance(odds_raw, dict) else {}
    if book_odds_by_match:
        from scripts.betting.odds_comparison import compare_match, best_value_bets
        for proj in projections:
            mo = book_odds_by_match.get(proj.get("match"))
            if not mo:
                continue
            results = compare_match(proj, mo, book=book_name)
            proj["odds_book"] = book_name
            proj["edges"] = [r.to_dict() for r in results]
            proj["value_bets"] = [r.to_dict() for r in best_value_bets(results)]

    return jsonify({
        "generated_at": generated_at,
        "count": len(projections),
        "odds_source": book_name,
        "projections": projections,
    })


@app.route("/value-bets")
@app.route("/value")
def value_bets_page():
    resp = app.make_response(render_template("value_bets.html", active_page="value_bets"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp


@app.route("/api/value-bets")
def api_value_bets():
    """All value bets across all matches, sorted by edge — the bet-slip view.

    Reads predictions.json + comparison_odds.json, runs the comparison engine,
    flattens every flagged value bet with its match context. Separates 'skill'
    (trusted) from 'noise' (model-unreliable) so the UI can warn appropriately.
    """
    predictions_raw = _load_json(UPCOMING_DIR / "predictions.json")
    preds = predictions_raw.get("predictions", []) if isinstance(predictions_raw, dict) else (predictions_raw or [])
    odds_raw = _load_json(UPCOMING_DIR / "comparison_odds.json", {})
    book_name = odds_raw.get("book", "") if isinstance(odds_raw, dict) else ""
    book_odds_by_match = odds_raw.get("odds", {}) if isinstance(odds_raw, dict) else {}

    if not book_odds_by_match:
        return jsonify({"odds_source": "", "value_bets": [], "count": 0,
                        "message": "No book odds loaded — wire Betfair/Sisal odds to comparison_odds.json"})

    from scripts.betting.odds_comparison import compare_match, best_value_bets

    all_bets = []
    for p in preds:
        proj = _build_score_range_projection(p)
        if proj is None:
            continue
        mo = book_odds_by_match.get(proj.get("match"))
        if not mo:
            continue
        for r in best_value_bets(compare_match(proj, mo, book=book_name), top_n=99):
            bet = r.to_dict()
            bet["match"] = proj.get("match")
            bet["home_team"] = proj.get("home_team")
            bet["away_team"] = proj.get("away_team")
            bet["date"] = proj.get("date", "")
            bet["time"] = proj.get("time", "")
            all_bets.append(bet)

    all_bets.sort(key=lambda b: -b["edge_pct"])
    return jsonify({
        "odds_source": book_name,
        "count": len(all_bets),
        "trusted_count": sum(1 for b in all_bets if b["trust"] == "skill"),
        "value_bets": all_bets,
    })


@app.route("/api/data-freshness")
def api_data_freshness():
    """Sofascore refresh freshness — used by the global staleness banner.

    Reads data/external/sofascore/.last_refresh.json and compares to wall clock.
    Returns:
        {
          "ok": bool,                 # green/red flag
          "stale_hours": float,       # how long since last successful refresh
          "last_refresh": iso str,
          "last_success": iso str | null,
          "any_failure": bool,
          "leagues": {...}            # passthrough of heartbeat
        }
    """
    from datetime import datetime, timezone
    import json
    from config.settings import DATA_DIR

    hb_path = DATA_DIR / "external" / "sofascore" / ".last_refresh.json"
    out: dict = {
        "ok": False,
        "stale_hours": 9999,
        "last_refresh": None,
        "last_success": None,
        "any_failure": True,
        "leagues": {},
        "message": "",
    }
    if not hb_path.exists():
        out["message"] = "No refresh has ever run. Cron may not be installed."
        return jsonify(out)

    try:
        with open(hb_path) as f:
            hb = json.load(f)
    except Exception as e:
        out["message"] = f"Heartbeat unreadable: {e}"
        return jsonify(out)

    out["last_refresh"] = hb.get("completed_at") or hb.get("started_at")
    out["any_failure"] = bool(hb.get("any_failure"))
    out["leagues"] = hb.get("leagues", {})

    # Compute stale_hours from heartbeat completed_at
    last = out["last_refresh"]
    if last:
        try:
            t = datetime.fromisoformat(last.replace("Z", "+00:00"))
            delta = datetime.now(timezone.utc) - t
            out["stale_hours"] = round(delta.total_seconds() / 3600, 1)
        except Exception:
            pass

    # Also scan each fixtures file mtime — that's the real "data age" signal
    fixtures_paths = {
        "serie_a": DATA_DIR / "external" / "sofascore" / "fixtures_2025_2026.json",
        "premier_league": DATA_DIR / "external" / "sofascore" / "fixtures_2025_2026_premier_league.json",
    }
    file_age = {}
    max_file_hours = 0
    for league, p in fixtures_paths.items():
        if p.exists():
            mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            age_h = round((datetime.now(timezone.utc) - mtime).total_seconds() / 3600, 1)
            file_age[league] = age_h
            max_file_hours = max(max_file_hours, age_h)
        else:
            file_age[league] = None
            max_file_hours = 9999
    out["fixtures_age_hours"] = file_age
    out["worst_fixture_age_hours"] = max_file_hours

    # Per-league probe: live HTML scrape + parquet age
    parquet_paths = {
        "serie_a": DATA_DIR / "external" / "sofascore" / "player_match_stats.parquet",
        "premier_league": DATA_DIR / "external" / "sofascore" / "player_match_stats_premier_league.parquet",
    }
    league_health: dict = {}
    any_html_ok = False
    any_hard_fail = False
    schema_break_seen = False

    for lg in ("serie_a", "premier_league"):
        # Trigger scrape (cached) to refresh health entry
        html_payload = _live_standings_via_html(lg)
        h = _html_health_now(lg)
        broken = _html_is_broken(lg)
        html_ok = bool(html_payload.get("standings")) and not broken

        pq_path = parquet_paths.get(lg)
        pq_age_h = None
        if pq_path and pq_path.exists():
            mt = datetime.fromtimestamp(pq_path.stat().st_mtime, tz=timezone.utc)
            pq_age_h = round((datetime.now(timezone.utc) - mt).total_seconds() / 3600, 1)

        last_success = h.get("last_success_at", 0)
        last_success_iso = (
            datetime.fromtimestamp(last_success, tz=timezone.utc).isoformat()
            if last_success else None
        )

        league_health[lg] = {
            "html_ok": html_ok,
            "html_broken": broken,
            "html_consecutive_failures": h.get("consecutive_failures", 0),
            "html_schema_break": h.get("schema_break", False),
            "html_last_error": h.get("last_error", ""),
            "html_last_success": last_success_iso,
            "parquet_age_hours": pq_age_h,
            "parquet_too_old": (pq_age_h is not None and pq_age_h > 24 * 7),
        }
        if html_ok:
            any_html_ok = True
        if h.get("schema_break"):
            schema_break_seen = True
        # Hard fail: this league has no fresh source at all
        league_hard = (not html_ok) and (
            league_health[lg]["parquet_too_old"] or pq_age_h is None
        )
        if league_hard:
            any_hard_fail = True

    out["leagues_health"] = league_health
    out["live_standings_ok"] = any_html_ok
    out["schema_break"] = schema_break_seen

    # Decision tree
    if schema_break_seen:
        out["severity"] = "schema_break"
        out["ok"] = False
        out["message"] = (
            "Sofascore HTML schema changed — scrape is parsing but a sentinel "
            "team is missing. Check _live_standings_via_html parsers and update "
            "_HTML_SENTINEL_TEAM / NEXT_DATA paths."
        )
    elif any_hard_fail:
        out["severity"] = "hard_fail"
        out["ok"] = False
        out["message"] = (
            "BOTH live HTML scrape AND cached parquet are stale. User-facing "
            "data is unreliable. Fix Sofascore access or restore parquet."
        )
    elif not any_html_ok:
        # No HTML, but parquet within 7d — degraded but tolerable
        out["severity"] = "degraded_parquet_only"
        out["ok"] = False  # still surface the banner — we're not live
        out["message"] = (
            "Live HTML scrape failing — serving cached parquet data. "
            f"Parquet ages: {league_health}"
        )
    elif max_file_hours >= 36:
        out["severity"] = "fixtures_stale_html_ok"
        out["ok"] = True  # standings are live, fixtures only used for next-match
        out["message"] = (
            f"Fixtures file {max_file_hours:.0f}h stale (Sofascore API blocked); "
            "standings are live via HTML scrape"
        )
    else:
        out["severity"] = "ok"
        out["ok"] = True
        out["message"] = "ok"

    return jsonify(out)


@app.route("/standings")
def standings_page():
    """Unified standings page (Serie A / Premier League / future leagues).

    Reuses the teams.html template. League is chosen via ?league= query param
    or the dropdown at the top of the page.
    """
    return render_template("teams.html", active_page="standings")


@app.route("/api/standings/<league>")
def api_standings(league):
    """Live standings for a league, derived from Sofascore parquet."""
    payload = _get_standings(league)
    inner = payload.get("standings", {})
    items = list(inner.values()) if isinstance(inner, dict) else (inner if isinstance(inner, list) else [])
    items.sort(key=lambda r: r.get("position", 99))
    return jsonify({
        "league": league,
        "season": payload.get("season", ""),
        "current_matchweek": payload.get("current_matchweek", 0),
        "standings": items,
    })


@app.route("/predictions")
def predictions_page():
    return render_template("predictions.html", active_page="predictions")


@app.route("/prediction/<match_slug>")
def prediction_detail_page(match_slug):
    """Individual match prediction page.

    Slug format: 'sassuolo-vs-verona' (lowercase, hyphens).
    """
    return render_template("prediction_detail.html",
                           active_page="predictions",
                           match_slug=match_slug)


# ---------------------------------------------------------------------------
# API: Predictions context — H2H, standings, calibration
# ---------------------------------------------------------------------------

@app.route("/api/predictions/context")
def api_predictions_context():
    """Supplementary intelligence for predictions page.

    Returns H2H history per match, league standings, and model calibration
    metrics. Consumed by the predictions page alongside /api/dashboard.
    """
    h2h_raw = _load_json(UPCOMING_DIR / "h2h_upcoming.json")
    analysis = _load_json(FEEDBACK_DIR / "analysis.json")

    # Build standings lookup keyed by team name. Live derivation from Sofascore
    # parquet via _get_standings — same source as /matches, so they cannot drift.
    standings_by_team: dict = {}
    sa_payload = _get_standings("serie_a")
    sa_dict = sa_payload.get("standings", {})
    if isinstance(sa_dict, dict):
        standings_by_team.update(sa_dict)

    # Load EPL standings (live)
    _epl_full_to_short = {
        "Brighton and Hove Albion": "Brighton", "Manchester City": "Man City",
        "Manchester United": "Man United", "Newcastle United": "Newcastle",
        "Wolverhampton Wanderers": "Wolves", "West Ham United": "West Ham",
        "Tottenham Hotspur": "Tottenham", "Leeds United": "Leeds",
    }
    epl_payload = _get_standings("premier_league")
    epl_dict = epl_payload.get("standings", {})
    if isinstance(epl_dict, dict):
        standings_by_team.update(epl_dict)
        # Also add entries keyed by full Odds API names so JS lookups work
        _short_to_full = {v: k for k, v in _epl_full_to_short.items()}
        for short_name, entry in epl_dict.items():
            full_name = _short_to_full.get(short_name)
            if full_name and full_name not in standings_by_team:
                standings_by_team[full_name] = entry

    # Compute H2H for EPL from matches.parquet if not in static file
    all_h2h = h2h_raw.get("h2h", {})
    try:
        epl_h2h = _compute_epl_h2h()
        all_h2h.update(epl_h2h)
    except Exception:
        pass

    # Calibration: by-confidence accuracy + method rankings
    overall = analysis.get("overall", {})
    calibration = {
        "by_confidence": overall.get("by_confidence", {}),
        "method_scores": analysis.get("method_brier_scores", {}),
        "n_settled": analysis.get("n_settled", overall.get("total_predictions", 0)),
    }

    return jsonify({
        "h2h": all_h2h,
        "standings": standings_by_team,
        "calibration": calibration,
    })



@app.route("/api/match-intel/<match_slug>")
def api_match_intel(match_slug):
    """Per-match intelligence bundle for the prediction-detail page.

    Returns scorer suitability + AI reasoning. The corners/cards/yc-eagerness/
    pass-probability fields were removed 2026-05-04 — held-out 2024-25 SA
    backtest (380 matches) showed corners/cards walkforward models had
    skill score ≤ 0 (worse than always-predict-base-rate) and AUC ≈
    0.51-0.60. They weren't worth showing.

    Reads from scorers_predictions.json and match_reasoning.json. Anything
    not produced by an existing model is returned with `_unavailable=True`.
    """
    def _to_slug(s: str) -> str:
        import re
        return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")

    scorers_raw = _load_json(UPCOMING_DIR / "scorers_predictions.json", default=[])
    reasoning_raw = _load_json(UPCOMING_DIR / "match_reasoning.json", default=[])

    def _list(d):
        if isinstance(d, list):
            return d
        if isinstance(d, dict):
            return d.get("predictions", []) or []
        return []

    def _find_by_slug(items):
        for p in _list(items):
            mk = p.get("match", "")
            if _to_slug(mk) == match_slug:
                return p
        return None

    scorers = _find_by_slug(scorers_raw)
    reasoning = _find_by_slug(reasoning_raw)

    out = {
        "match_slug": match_slug,
        "scorer_suitability": {
            "home_top": (scorers or {}).get("home_top_scorers", []),
            "away_top": (scorers or {}).get("away_top_scorers", []),
            "source": (scorers or {}).get("source"),
            "_unavailable": scorers is None,
        },
        "ai_reason": {
            "sentence": (reasoning or {}).get("reasoning"),
            "_unavailable": reasoning is None,
        },
    }
    return jsonify(out)


def _compute_epl_h2h() -> dict:
    """Compute H2H stats for upcoming EPL matches from matches.parquet."""
    import pandas as pd
    from config.team_names import normalize_team

    preds_path = UPCOMING_DIR / "predictions_premier_league.json"
    if not preds_path.exists():
        return {}

    with open(preds_path) as f:
        preds = json.load(f).get("predictions", [])

    matches_path = DATA_DIR / "parsed" / "matches.parquet"
    if not matches_path.exists():
        return {}

    df = pd.read_parquet(matches_path)
    epl = df[df["league"] == "premier_league"]

    h2h = {}
    for pred in preds:
        home = normalize_team(pred.get("home_team", ""))
        away = normalize_team(pred.get("away_team", ""))
        match_key = f"{pred.get('home_team', '')} vs {pred.get('away_team', '')}"

        past = epl[
            ((epl["home_team"] == home) & (epl["away_team"] == away)) |
            ((epl["home_team"] == away) & (epl["away_team"] == home))
        ].sort_values("match_date", ascending=False)

        if len(past) == 0:
            continue

        home_wins = ((past["home_team"] == home) & (past["result"] == "H")).sum() + \
                    ((past["away_team"] == home) & (past["result"] == "A")).sum()
        away_wins = ((past["home_team"] == away) & (past["result"] == "H")).sum() + \
                    ((past["away_team"] == away) & (past["result"] == "A")).sum()
        draws = (past["result"] == "D").sum()
        total = len(past)

        last = past.iloc[0]
        last_score = f"{int(last['home_score'])}-{int(last['away_score'])}"
        last_result = "draw" if last["result"] == "D" else (
            "home" if (last["home_team"] == home and last["result"] == "H") or
                      (last["away_team"] == home and last["result"] == "A")
            else "away"
        )

        home_goals = past[past["home_team"] == home]["home_score"].sum() + \
                     past[past["away_team"] == home]["away_score"].sum()
        away_goals = past[past["home_team"] == away]["home_score"].sum() + \
                     past[past["away_team"] == away]["away_score"].sum()

        h2h[match_key] = {
            "total_meetings": int(total),
            "home_wins": int(home_wins),
            "away_wins": int(away_wins),
            "draws": int(draws),
            "home_goals_avg": round(float(home_goals / total), 1),
            "away_goals_avg": round(float(away_goals / total), 1),
            "last_result": last_result,
            "last_score": last_score,
        }

    return h2h


# ---------------------------------------------------------------------------
# API: Matches - all played matches grouped by matchweek
# ---------------------------------------------------------------------------

@app.route("/api/matches")
def api_matches():
    """All played matches for a season, grouped by matchweek.

    Query params:
        season: e.g. "2025-2026" (default: current season)
        source: "sofascore" (default) or "features" (for pre-2022 seasons)

    Returns match score, xG, shots, top scorers per match — everything
    needed for a rich match listing page.
    """
    import pandas as pd
    from config.team_names import normalize_team

    season = flask_request.args.get("season", get_current_season())

    # For seasons covered by player_match_stats (2022-2023+), use Sofascore
    # For older seasons, fall back to features.parquet + fixtures JSON
    current = get_current_season()
    sofascore_seasons = {"2022-2023", "2023-2024", "2024-2025", current}

    if season in sofascore_seasons:
        return _api_matches_sofascore(season)
    else:
        return _api_matches_features(season)


@app.route("/api/matches/seasons")
def api_matches_seasons():
    """List available seasons."""
    import pandas as pd
    try:
        df = pd.read_parquet(
            DATA_DIR / "features" / "features.parquet",
            columns=["season"]
        )
        seasons = sorted(df["season"].unique(), reverse=True)
        return jsonify({"seasons": list(seasons)})
    except Exception:
        return jsonify({"seasons": [get_current_season()]})


_matches_response_cache: dict[tuple, tuple[float, dict]] = {}  # key → (cached_at, payload)
_MATCHES_RESPONSE_TTL = 300  # 5 minutes — short enough to feel live, long enough to cover bursty navigation


def _api_matches_sofascore(season: str):
    """Matches from Sofascore player_match_stats (rich per-player data).

    Loads BOTH leagues' parquets so Results page covers SA + EPL. Earlier
    bug: only SA parquet was read, so /matches never showed EPL fixtures.

    Response cached for up to _MATCHES_RESPONSE_TTL seconds AND auto-invalidated
    on parquet mtime change. This route was the dashboard's slowest endpoint
    (~11.5s cold) due to player-row groupby + iterrows over ~22K rows; the
    finished payload is what actually changes (only after scrapes), so caching
    the JSON response is the right granularity.
    """
    import pandas as pd
    from config.team_names import normalize_team

    parquet_paths = [
        DATA_DIR / "external" / "sofascore" / "player_match_stats.parquet",
        DATA_DIR / "external" / "sofascore" / "player_match_stats_premier_league.parquet",
    ]
    existing = [p for p in parquet_paths if p.exists()]
    if not existing:
        return jsonify({"error": "no sofascore data", "matchweeks": []})

    # Cache key includes parquet mtimes — when scrapers rewrite, key flips
    league_filter = _get_league_filter()
    mtimes = tuple(p.stat().st_mtime for p in existing)
    cache_key = (season, league_filter, mtimes)
    now = _time.time()
    cached = _matches_response_cache.get(cache_key)
    if cached and (now - cached[0]) < _MATCHES_RESPONSE_TTL:
        return jsonify(cached[1])

    try:
        frames = [pd.read_parquet(p) for p in existing]
        pms = pd.concat(frames, ignore_index=True)
    except Exception as e:
        return jsonify({"error": str(e), "matchweeks": []})

    pms = pms[pms["season"].astype(str).str.startswith(season)]
    if pms.empty:
        return jsonify({"matchweeks": []})

    pms["_home"] = pms["home_team"].apply(lambda t: normalize_team(str(t)))
    pms["_away"] = pms["away_team"].apply(lambda t: normalize_team(str(t)))

    # Build per-match aggregates
    match_keys = pms.groupby(["date", "_home", "_away", "home_score", "away_score", "round"])
    matches = []
    for (date, home, away, hs, aws, rnd), grp in match_keys:
        home_players = grp[grp["is_home"] == True]
        away_players = grp[grp["is_home"] == False]

        # xG
        home_xg = round(float(home_players["xg"].sum()), 2) if "xg" in grp.columns else None
        away_xg = round(float(away_players["xg"].sum()), 2) if "xg" in grp.columns else None

        # Shots
        home_shots = int(home_players["total_shots"].sum()) if "total_shots" in grp.columns else 0
        away_shots = int(away_players["total_shots"].sum()) if "total_shots" in grp.columns else 0

        # Top scorers (goals > 0)
        scorers = []
        for _, p in grp[grp["goals"] > 0].iterrows():
            scorers.append({
                "name": str(p["player_name"]),
                "team": normalize_team(str(p["team"])),
                "goals": int(p["goals"]),
                "minute": None,  # not available per-player in this dataset
            })

        # Top rated players
        top_home = home_players.nlargest(1, "rating").iloc[0] if len(home_players) and "rating" in home_players.columns and home_players["rating"].notna().any() else None
        top_away = away_players.nlargest(1, "rating").iloc[0] if len(away_players) and "rating" in away_players.columns and away_players["rating"].notna().any() else None

        matches.append({
            "date": str(date)[:10],
            "matchweek": int(rnd) if pd.notna(rnd) else 0,
            "home_team": home,
            "away_team": away,
            "home_score": int(hs),
            "away_score": int(aws),
            "home_xg": home_xg,
            "away_xg": away_xg,
            "home_shots": home_shots,
            "away_shots": away_shots,
            "scorers": scorers,
            "home_top_rated": {
                "name": str(top_home["player_name"]),
                "rating": round(float(top_home["rating"]), 1),
            } if top_home is not None and pd.notna(top_home.get("rating")) else None,
            "away_top_rated": {
                "name": str(top_away["player_name"]),
                "rating": round(float(top_away["rating"]), 1),
            } if top_away is not None and pd.notna(top_away.get("rating")) else None,
        })

    # league_filter already resolved at function top for cache key
    if league_filter:
        matches = [m for m in matches if _team_belongs_to_league(m.get("home_team", ""), league_filter)]

    # Group by matchweek, sort within each MW by date desc
    from collections import defaultdict
    by_mw = defaultdict(list)
    for m in matches:
        by_mw[m["matchweek"]].append(m)

    # Sort matchweeks by MW number desc (highest first)
    matchweeks = []
    for mw in sorted(by_mw.keys(), reverse=True):
        mw_matches = sorted(by_mw[mw], key=lambda m: m["date"], reverse=True)
        total_goals = sum(m["home_score"] + m["away_score"] for m in mw_matches)
        matchweeks.append({
            "matchweek": mw,
            "date_range": f"{mw_matches[-1]['date']} — {mw_matches[0]['date']}" if len(mw_matches) > 1 else mw_matches[0]["date"],
            "matches": mw_matches,
            "total_goals": total_goals,
            "match_count": len(mw_matches),
        })

    payload = {
        "season": season,
        "total_matches": len(matches),
        "matchweeks": matchweeks,
    }
    _matches_response_cache[cache_key] = (now, payload)
    return jsonify(payload)


def _api_matches_features(season: str):
    """Matches from features.parquet (for older seasons without Sofascore player data)."""
    import pandas as pd
    from config.team_names import normalize_team

    try:
        cols = ["season", "match_date", "home_team", "away_team", "result",
                "matchweek", "home_score", "away_score"]
        df = _read_parquet_cached(DATA_DIR / "features" / "features.parquet", columns=cols)
    except Exception as e:
        return jsonify({"error": str(e), "matchweeks": []})

    df = df[df["season"] == season]
    if df.empty:
        return jsonify({"season": season, "total_matches": 0, "matchweeks": []})

    matches = []
    for _, row in df.iterrows():
        home = str(row["home_team"])
        away = str(row["away_team"])
        date = str(row["match_date"])[:10]
        mw = int(row["matchweek"]) if pd.notna(row.get("matchweek")) else 0
        hs = int(row["home_score"]) if pd.notna(row.get("home_score")) else 0
        aws = int(row["away_score"]) if pd.notna(row.get("away_score")) else 0

        matches.append({
            "date": date,
            "matchweek": mw,
            "home_team": home,
            "away_team": away,
            "home_score": hs,
            "away_score": aws,
            "home_xg": None,
            "away_xg": None,
            "home_shots": 0,
            "away_shots": 0,
            "scorers": [],
            "home_top_rated": None,
            "away_top_rated": None,
        })

    # Group by matchweek, sort within each MW by date desc
    from collections import defaultdict as _defaultdict
    by_mw = _defaultdict(list)
    for m in matches:
        by_mw[m["matchweek"]].append(m)

    # Sort matchweeks by MW number desc (highest first)
    matchweeks = []
    for mw in sorted(by_mw.keys(), reverse=True):
        mw_matches = sorted(by_mw[mw], key=lambda m: m["date"], reverse=True)
        total_goals = sum(m["home_score"] + m["away_score"] for m in mw_matches)
        matchweeks.append({
            "matchweek": mw,
            "date_range": f"{mw_matches[-1]['date']} — {mw_matches[0]['date']}" if len(mw_matches) > 1 else mw_matches[0]["date"],
            "matches": mw_matches,
            "total_goals": total_goals,
            "match_count": len(mw_matches),
        })

    return jsonify({
        "season": season,
        "total_matches": len(matches),
        "matchweeks": matchweeks,
    })


# ---------------------------------------------------------------------------
# API: Dashboard - all match data merged
# ---------------------------------------------------------------------------

@app.route("/api/dashboard")
def api_dashboard():
    league_filter = _get_league_filter()

    # Load all data sources
    predictions_raw = _load_json(UPCOMING_DIR / "predictions.json")
    odds_full = _load_json(UPCOMING_DIR / "odds_full.json")
    # Merge in EPL-specific odds file
    epl_odds = _load_json(UPCOMING_DIR / "odds_full_premier_league.json")
    if epl_odds and isinstance(epl_odds, dict):
        epl_matches = epl_odds.get("matches", {})
        if isinstance(epl_matches, dict):
            odds_full_matches = odds_full.get("matches", {}) if isinstance(odds_full, dict) else {}
            # If odds_full contains wrong league, detect and use EPL file correctly
            odds_full.setdefault("matches", {})
            if isinstance(odds_full["matches"], dict):
                odds_full["matches"].update(epl_matches)
    market_intel = _load_json(UPCOMING_DIR / "market_intelligence.json")
    bookmaker_raw = _load_json(UPCOMING_DIR / "bookmaker_analysis.json")
    cross_market = _load_json(UPCOMING_DIR / "cross_market_signals.json")
    odds_movement = _load_json(UPCOMING_DIR / "odds_movement.json")
    sentiment_raw = _load_json(UPCOMING_DIR / "sentiment_analysis.json")
    player_raw = _load_json(UPCOMING_DIR / "player_analysis.json")
    weather_raw = _load_json(UPCOMING_DIR / "weather.json")
    referees_raw = _load_json(UPCOMING_DIR / "referees.json")
    lineups_raw = _load_json(UPCOMING_DIR / "confirmed_lineups.json")
    lineup_preds_raw = _load_json(UPCOMING_DIR / "lineup_predictions.json")
    epl_injuries_raw = _load_json(UPCOMING_DIR / "injuries_premier_league.json")
    btts_raw = _load_json(UPCOMING_DIR / "btts_predictions.json", default=[])
    btts_list = btts_raw if isinstance(btts_raw, list) else btts_raw.get("predictions", [])
    btts_by_match = {b.get("match", ""): b for b in btts_list if isinstance(b, dict)}

    # Normalize into match-keyed dicts
    # Merge Serie A predictions (default) with extra league prediction files
    predictions_list = predictions_raw.get("predictions", []) if isinstance(predictions_raw, dict) else predictions_raw
    # Tag Serie A predictions with league field
    for p in predictions_list:
        p.setdefault("league", "serie_a")
    # Load extra league predictions (e.g. predictions_premier_league.json)
    for league_key in ACTIVE_LEAGUES:
        if league_key == "serie_a":
            continue
        extra_path = UPCOMING_DIR / f"predictions_{league_key}.json"
        extra_raw = _load_json(extra_path)
        if extra_raw:
            extra_list = extra_raw.get("predictions", []) if isinstance(extra_raw, dict) else extra_raw
            for p in extra_list:
                p.setdefault("league", league_key)
            predictions_list.extend(extra_list)
    predictions_by_match = _index_list_by_match(predictions_list)

    odds_matches = odds_full.get("matches", {}) if isinstance(odds_full, dict) else {}
    # Validate league consistency: warn if predictions and odds have zero overlap
    if predictions_by_match and odds_matches:
        pred_teams = {m.split(" vs ")[0] for m in predictions_by_match if " vs " in m}
        odds_teams = {m.split(" vs ")[0] for m in odds_matches if " vs " in m}
        overlap = pred_teams & odds_teams
        if not overlap and pred_teams and odds_teams:
            log.warning(
                "LEAGUE MISMATCH: predictions have %d teams (%s...) but odds have %d teams (%s...). "
                "Zero overlap — odds may be for the wrong league.",
                len(pred_teams), list(pred_teams)[:3],
                len(odds_teams), list(odds_teams)[:3],
            )
    intel_matches = market_intel.get("matches", {}) if isinstance(market_intel, dict) else {}
    book_matches = bookmaker_raw.get("matches", {}) if isinstance(bookmaker_raw, dict) else {}
    cross_matches = cross_market.get("matches", {}) if isinstance(cross_market, dict) else {}
    move_matches = odds_movement.get("matches", {}) if isinstance(odds_movement, dict) else {}
    weather_matches = weather_raw.get("matches", {}) if isinstance(weather_raw, dict) else {}
    lineup_matches = lineups_raw.get("matches", {}) if isinstance(lineups_raw, dict) else {}
    lineup_pred_matches = lineup_preds_raw.get("matches", {}) if isinstance(lineup_preds_raw, dict) else {}
    epl_injury_matches = epl_injuries_raw.get("matches", {}) if isinstance(epl_injuries_raw, dict) else {}

    sentiment_list = sentiment_raw.get("matches", []) if isinstance(sentiment_raw, dict) else sentiment_raw if isinstance(sentiment_raw, list) else []
    sentiment_by_match = _index_list_by_match(sentiment_list)

    player_list = player_raw.get("matches", []) if isinstance(player_raw, dict) else player_raw if isinstance(player_raw, list) else []
    player_by_match = _index_list_by_match(player_list)

    # referees.json is a flat dict: {"Team vs Team": "Referee Name"}
    referees_map = referees_raw if isinstance(referees_raw, dict) else {}

    # Load settled results to mark completed matches
    results_data = _load_json(UPCOMING_DIR / "results.json")
    settled_results = results_data.get("results", {}) if isinstance(results_data, dict) else {}

    # Build unified match list from predictions (primary source)
    matches = []
    for match_key, pred in predictions_by_match.items():
        home = pred.get("home_team", "")
        away = pred.get("away_team", "")
        if not home and " vs " in match_key:
            parts = match_key.split(" vs ", 1)
            home, away = parts[0].strip(), parts[1].strip()

        odds_data = odds_matches.get(match_key, {})
        # Fallback: build h2h odds from odds_movement if odds_full has no data for this match
        if not odds_data.get("h2h") and match_key in move_matches:
            om = move_matches[match_key]
            odds_data = dict(odds_data)  # copy to avoid mutation
            odds_data["h2h"] = {
                "home": om.get("current_home", 0),
                "draw": om.get("current_draw", 0),
                "away": om.get("current_away", 0),
            }
        ct = odds_data.get("commence_time", "")
        # Fall back to prediction date+time if odds don't have commence_time
        if not ct and pred.get("date") and pred.get("time"):
            ct = f"{pred['date']}T{pred['time']}:00Z"

        # Check if match has been settled (result exists in results.json)
        result_entry = settled_results.get(match_key, {})
        if result_entry.get("completed"):
            match_status = "completed"
        else:
            _status_entry = {"match": match_key, "commence_time": ct, "date": pred.get("date", "")}
            match_status = _match_status(_status_entry)

        match = {
            "match": match_key,
            "home_team": home,
            "away_team": away,
            "league": pred.get("league", "serie_a"),
            "status": match_status,
            "date": pred.get("date", pred.get("match_date", "")),
            "venue": pred.get("venue", ""),
            "commence_time": ct,
            "match_window": odds_data.get("match_window", ""),

            # Core prediction
            "predicted_outcome": pred.get("predicted_outcome", ""),
            "probabilities": pred.get("probabilities", {}),
            "confidence": pred.get("confidence", 0),
            "confidence_level": pred.get("confidence_level", ""),
            "home_xg": pred.get("home_xg") or 0,
            "away_xg": pred.get("away_xg") or 0,
            "market_edge": pred.get("market_edge") or _compute_market_edge(pred, odds_data),
            "betting_recommendation": pred.get("betting_recommendation", ""),
            "lineup_source": pred.get("lineup_source", "predicted"),

            # Component methods
            "component_methods": pred.get("component_predictions", pred.get("component_methods", pred.get("method_predictions", {}))),

            # Draw analysis
            "draw_analysis": pred.get("draw_analysis", {}),
            "formation_analysis": pred.get("formation_analysis", {}),

            # Form & momentum
            "home_form": pred.get("home_form", {}),
            "away_form": pred.get("away_form", {}),
            "momentum_analysis": pred.get("momentum_analysis", {}),

            # Key factors
            "home_factors": pred.get("home_factors", []),
            "away_factors": pred.get("away_factors", []),
            "neutral_factors": pred.get("neutral_factors", []),

            # Ensemble weights
            "weights_applied": pred.get("weights_applied", {}),

            # Injuries — use prediction-embedded data, or EPL injury file as fallback
            "injury_adjustments": pred.get("injury_adjustments") or epl_injury_matches.get(match_key, {}),

            # Sentiment (prediction engine)
            "sentiment_analysis": pred.get("sentiment_analysis", {}),

            # Market implied
            "market_implied": pred.get("market_implied", {}),

            # Odds (h2h, totals, spreads)
            "odds": {
                "h2h": odds_data.get("h2h", {}),
                "totals": odds_data.get("totals", []),
                "spreads": odds_data.get("spreads", []),
            },

            # Market intelligence
            "market_intelligence": intel_matches.get(match_key, {}),

            # Bookmaker analysis
            "bookmaker_analysis": book_matches.get(match_key, {}),

            # Odds movement
            "odds_movement": move_matches.get(match_key, {}),

            # Cross-market signals
            "cross_market": cross_matches.get(match_key, {}),

            # Sentiment
            "sentiment": sentiment_by_match.get(match_key, {}),

            # Player analysis
            "player_analysis": player_by_match.get(match_key, {}),

            # Weather
            "weather": weather_matches.get(match_key, {}),

            # Referee
            "referee": referees_map.get(match_key, pred.get("referee", "")),
            "referee_bias": pred.get("referee_bias", ""),

            # Confirmed lineups
            "confirmed_lineups": lineup_matches.get(match_key, {}),

            # Predicted lineups & formations
            "lineup_prediction": lineup_pred_matches.get(match_key, {}),

            # BTTS probability
            "btts_probability": btts_by_match.get(match_key, {}).get("btts_yes"),

            # Actual result (if settled)
            "actual_result": result_entry.get("result", "") if result_entry else "",
            "actual_score": [result_entry.get("home_score", 0), result_entry.get("away_score", 0)] if result_entry.get("completed") else None,
            "actual_total_goals": result_entry.get("total_goals") if result_entry.get("completed") else None,
            "actual_btts": result_entry.get("btts") if result_entry.get("completed") else None,
        }
        matches.append(match)

    # Sort by commence_time (earliest first), then by date
    def sort_key(m):
        ct = m.get("commence_time", "")
        d = m.get("date", "")
        return ct or d or "9999"
    matches.sort(key=sort_key)

    # Apply league filter
    matches = _filter_by_league(matches, league_filter)

    # Split: upcoming for dashboard display, all for detail page lookups
    upcoming = [m for m in matches if m.get("status") != "completed"]

    # Overlay actual live status from live monitoring data (replaces wall-clock guessing)
    try:
        live_path = LIVE_DIR / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.json"
        live_data = _load_json(live_path, default=None)
        if live_data and isinstance(live_data.get("matches"), dict):
            _LIVE_STATUS_MAP = {
                "pre_match": "upcoming",
                "first_half": "live",
                "half_time": "live",
                "second_half": "live",
                "extra_time": "live",
                "penalties": "live",
                "completed": "completed",
                "finished": "completed",
            }
            for m in matches:
                mk = m.get("match", "")
                live_match = live_data["matches"].get(mk)
                if live_match and isinstance(live_match, dict):
                    raw_status = live_match.get("status", "")
                    mapped = _LIVE_STATUS_MAP.get(raw_status)
                    if mapped:
                        m["status"] = mapped
            # Re-filter upcoming after status overlay (completed matches may have shifted)
            upcoming = [m for m in matches if m.get("status") != "completed"]
    except Exception:
        pass  # Non-critical — fall back to wall-clock status

    # Alerts
    alerts_raw = _load_json(BETTING_DIR / "alerts.json", default=[])
    alerts = [a.get("bet", {}) | {"alert_type": a.get("type", "")} for a in alerts_raw if isinstance(a, dict)]
    alerts = _filter_by_league(alerts, league_filter)

    # Steam moves summary
    steam_moves = [
        {"match": k, **{sk: v.get(sk) for sk in ("direction", "is_steam_move", "snapshots_count", "hours_tracked")}}
        for k, v in move_matches.items()
        if isinstance(v, dict) and v.get("is_steam_move")
    ]

    # Freshness timestamps (match betting page style)
    odds_fetched_at = ""
    if isinstance(odds_full, dict):
        odds_fetched_at = odds_full.get("fetched_at", odds_full.get("generated_at", ""))
    extended_raw = _load_json(UPCOMING_DIR / "extended_markets.json")
    market_intel_at = (market_intel.get("generated_at", "") or market_intel.get("analyzed_at", "")) if isinstance(market_intel, dict) else ""

    # Record fair odds predictions (fire-and-forget, don't block response)
    try:
        from scripts.betting.fair_odds_tracker import record_predictions
        record_predictions(upcoming)
    except Exception:
        pass  # Non-critical — don't block dashboard

    return jsonify({
        "generated_at": predictions_raw.get("generated_at", ""),
        "model_version": predictions_raw.get("model_version", ""),
        "match_count": len(upcoming),
        "matches": upcoming,
        "all_matches": matches,  # includes completed — used by prediction detail page
        "alerts": alerts,
        "steam_moves": steam_moves,
        "odds_fetched_at": odds_fetched_at,
        "predictions_generated_at": predictions_raw.get("generated_at", "") if isinstance(predictions_raw, dict) else "",
        "extended_markets_at": extended_raw.get("generated_at", "") if isinstance(extended_raw, dict) else "",
        "market_intelligence_at": market_intel_at,
        "league_filter": league_filter,
        "active_leagues": _active_leagues_info(),
    })


# ---------------------------------------------------------------------------
# API: Betting - all bets + bankroll
# ---------------------------------------------------------------------------

def _italy_offset(date_str: str) -> timedelta:
    """Return Italy's UTC offset for a date, respecting actual DST transition dates."""
    try:
        from zoneinfo import ZoneInfo
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        italy_tz = ZoneInfo("Europe/Rome")
        aware = dt.replace(hour=12, tzinfo=italy_tz)  # noon avoids midnight ambiguity
        return aware.utcoffset()
    except Exception:
        return timedelta(hours=1)


def _load_commence_times() -> dict:
    """Load commence_time UTC map from odds + predictions files (all leagues).

    Priority: odds_full*.json (authoritative UTC from Odds API), then
    predictions*.json date+time fields (converted to UTC).
    """
    ct_map = {}

    # Load odds commence times from all league files
    for odds_file in [UPCOMING_DIR / "odds_full.json"] + list(UPCOMING_DIR.glob("odds_full_*.json")):
        if not odds_file.exists():
            continue
        odds_full = _load_json(odds_file)
        matches = odds_full.get("matches", {}) if isinstance(odds_full, dict) else {}
        for mk, m in matches.items():
            ct = m.get("commence_time", "")
            if ct:
                ct_map[mk] = ct

    # Fallback: build commence_time from all prediction files
    for pred_file in [UPCOMING_DIR / "predictions.json"] + list(UPCOMING_DIR.glob("predictions_*.json")):
        if not pred_file.exists():
            continue
        predictions = _load_json(pred_file)
        pred_list = predictions if isinstance(predictions, list) else predictions.get("predictions", predictions.get("matches", []))
        for pred in pred_list:
            mk = pred.get("match", "")
            if mk and mk not in ct_map:
                date_str = pred.get("date", "")
                time_str = pred.get("time", "")
                if date_str and time_str:
                    try:
                        local_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
                        local_aware = local_dt.astimezone()
                        utc_dt = local_aware.astimezone(timezone.utc)
                        ct_map[mk] = utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                    except (ValueError, TypeError):
                        pass
    return ct_map


# Cached at module level with file-mtime invalidation
_COMMENCE_TIME_MAP = {}
_COMMENCE_TIME_MTIME = 0.0


def _get_commence_times() -> dict:
    global _COMMENCE_TIME_MAP, _COMMENCE_TIME_MTIME
    # Invalidate cache if any source file changed (all leagues)
    try:
        mtimes = []
        for pattern in ["odds_full*.json", "predictions*.json"]:
            for p in UPCOMING_DIR.glob(pattern):
                mtimes.append(p.stat().st_mtime)
        current_mtime = max(mtimes) if mtimes else 0
    except OSError:
        current_mtime = 0
    if not _COMMENCE_TIME_MAP or current_mtime > _COMMENCE_TIME_MTIME:
        _COMMENCE_TIME_MAP = _load_commence_times()
        _COMMENCE_TIME_MTIME = current_mtime
    return _COMMENCE_TIME_MAP


def _parse_kickoff_utc(entry: dict, pred_lookup: dict = None) -> "datetime | None":
    """Parse kickoff time to UTC datetime.

    Priority:
    1. commence_time on the entry itself (UTC ISO string from Odds API)
    2. commence_time from the global odds_full map (by match key)
    3. commence_time from pred_lookup
    4. date-only fallback: assume end-of-day Italian time (23:59 CET/CEST)
       so the match stays "upcoming" all day

    NOTE: The `time` field in predictions.json is unreliable (stored in
    mixed/unknown timezone), so we intentionally skip date+time fallback
    and use date-only with generous end-of-day assumption.
    """
    ct_map = _get_commence_times()
    match_key = entry.get("match", "")

    # 1. Direct on entry
    ct = entry.get("commence_time", "")
    # 2. From odds_full map
    if not ct:
        ct = ct_map.get(match_key, "")
    # 3. From pred_lookup
    if not ct and pred_lookup:
        pred = pred_lookup.get(match_key, {})
        ct = pred.get("commence_time", "")
    # 4. From pred_lookup via odds_full map key
    if not ct and pred_lookup:
        ct = ct_map.get(match_key, "")

    if ct:
        try:
            return datetime.fromisoformat(ct.replace("Z", "+00:00"))
        except (ValueError, TypeError) as e:
            log.debug(f"Failed to parse commence_time '{ct}': {e}")

    # Date-only fallback: treat as end-of-day Italian time
    d = entry.get("date", "")
    if not d and pred_lookup:
        d = pred_lookup.get(match_key, {}).get("date", "")
    if d:
        try:
            # Assume match hasn't happened yet if it's today — use 23:59 local
            local = datetime.strptime(f"{d} 23:59", "%Y-%m-%d %H:%M")
            return local.replace(tzinfo=timezone(_italy_offset(d))).astimezone(timezone.utc)
        except (ValueError, TypeError) as e:
            log.debug(f"Failed to parse date-only '{d}': {e}")

    return None


def _match_status(entry: dict, pred_lookup: dict = None) -> str:
    """Return 'upcoming', 'live', or 'completed' for a match/bet dict.

    Uses UTC commence_time from odds_full.json as authoritative source.
    A match is 'live' for 2.5 hours after kickoff, then 'completed'.
    """
    kick = _parse_kickoff_utc(entry, pred_lookup)
    if kick is None:
        return "upcoming"  # If we can't determine kickoff, assume upcoming
    now = datetime.now(timezone.utc)
    elapsed = (now - kick).total_seconds()
    if elapsed < 0:
        return "upcoming"
    if elapsed <= 9000:  # 2.5 hours (150 min) to cover extra time + penalties
        return "live"
    return "completed"


def _is_upcoming(entry: dict, pred_lookup: dict = None) -> bool:
    """Return True for upcoming or live matches (exclude only completed)."""
    return _match_status(entry, pred_lookup) != "completed"


@app.route("/api/betting")
def api_betting():
    league_filter = _get_league_filter()

    # ---- Load all data sources ----
    # Primary: unified_bet_slip.json (written by current pipeline)
    # Fallback: unified_report.json (legacy)
    unified = _load_json(UPCOMING_DIR / "unified_bet_slip.json")
    if not unified or not unified.get("selected_bets"):
        unified = _load_json(BETTING_DIR / "unified_report.json")
    # Normalize: unified_bet_slip uses "selected_bets", legacy uses "bets"
    if "selected_bets" in unified and "bets" not in unified:
        unified["bets"] = unified["selected_bets"]
    bankroll_file = _load_json(BETTING_DIR / "bankroll.json")
    bankroll_state = _load_json(BANKROLL_DIR / "state.json")
    predictions_raw = _load_json(UPCOMING_DIR / "predictions.json")
    odds_full = _load_all_leagues_json("odds_full.json", "matches")
    parlay_raw = _load_json(BETTING_DIR / "parlay_report.json")
    standings_raw = _load_all_leagues_json("standings.json", "standings")
    h2h_raw = _load_all_leagues_json("h2h_upcoming.json", "h2h")
    extended_raw = _load_json(UPCOMING_DIR / "extended_markets.json")
    player_props_raw = _load_json(UPCOMING_DIR / "player_props.json")
    player_raw = _load_json(UPCOMING_DIR / "player_analysis.json")

    # Step 2: additional data sources
    sentiment_raw = _load_json(UPCOMING_DIR / "sentiment_analysis.json")
    odds_movement_raw = _load_json(UPCOMING_DIR / "odds_movement.json")
    bookmaker_raw = _load_json(UPCOMING_DIR / "bookmaker_analysis.json")
    market_intel_raw = _load_json(UPCOMING_DIR / "market_intelligence.json")
    cross_market_raw = _load_json(UPCOMING_DIR / "cross_market_signals.json")
    goal_preds_raw = _load_json(UPCOMING_DIR / "goal_predictions.json")
    margin_preds_raw = _load_json(UPCOMING_DIR / "margin_predictions.json")
    weather_raw = _load_json(UPCOMING_DIR / "weather.json")
    form_raw = _load_all_leagues_json("current_form.json", "teams")
    # Also merge matchups key for form
    _form_matchups = _load_all_leagues_json("current_form.json", "matchups")
    form_raw["matchups"] = _form_matchups.get("matchups", {})
    odds_bk_raw = _load_json(UPCOMING_DIR / "odds_bookmakers.json")

    lineup_preds_raw2 = _load_json(UPCOMING_DIR / "lineup_predictions.json")
    epl_injuries_raw2 = _load_json(UPCOMING_DIR / "injuries_premier_league.json")
    risk_state = _load_json(BETTING_DIR / "risk_state.json")
    player_prop_vb = _load_json(UPCOMING_DIR / "player_prop_value_bets.json")
    ultimate_slip = _load_json(UPCOMING_DIR / "ultimate_bet_slip.json")
    odds_fetched_at = odds_full.get("fetched_at", "") if isinstance(odds_full, dict) else ""

    # ---- Normalize into match-keyed dicts ----
    lineup_pred_matches2 = lineup_preds_raw2.get("matches", {}) if isinstance(lineup_preds_raw2, dict) else {}
    epl_injury_matches2 = epl_injuries_raw2.get("matches", {}) if isinstance(epl_injuries_raw2, dict) else {}
    standings = standings_raw.get("standings", {}) if isinstance(standings_raw, dict) else {}
    h2h_map = h2h_raw.get("h2h", {}) if isinstance(h2h_raw, dict) else {}
    ext_matches = extended_raw.get("matches", {}) if isinstance(extended_raw, dict) else {}
    pp_matches = player_props_raw.get("matches", {}) if isinstance(player_props_raw, dict) else {}

    player_list = player_raw.get("matches", []) if isinstance(player_raw, dict) else player_raw if isinstance(player_raw, list) else []
    player_by_match = _index_list_by_match(player_list)

    sentiment_list = sentiment_raw.get("matches", []) if isinstance(sentiment_raw, dict) else []
    sentiment_by_match = _index_list_by_match(sentiment_list)

    odds_move_matches = odds_movement_raw.get("matches", {}) if isinstance(odds_movement_raw, dict) else {}
    book_matches = bookmaker_raw.get("matches", {}) if isinstance(bookmaker_raw, dict) else {}
    intel_matches = market_intel_raw.get("matches", {}) if isinstance(market_intel_raw, dict) else {}
    cross_matches = cross_market_raw.get("matches", {}) if isinstance(cross_market_raw, dict) else {}
    weather_matches = weather_raw.get("matches", {}) if isinstance(weather_raw, dict) else {}
    form_matchups = form_raw.get("matchups", {}) if isinstance(form_raw, dict) else {}
    odds_bk_matches = odds_bk_raw.get("matches", {}) if isinstance(odds_bk_raw, dict) else {}

    goal_pred_list = goal_preds_raw.get("predictions", []) if isinstance(goal_preds_raw, dict) else []
    goal_by_match = _index_list_by_match(goal_pred_list)
    margin_pred_list = margin_preds_raw.get("predictions", []) if isinstance(margin_preds_raw, dict) else []
    margin_by_match = _index_list_by_match(margin_pred_list)

    # ---- Build predictions lookup ----
    predictions_list = predictions_raw.get("predictions", []) if isinstance(predictions_raw, dict) else []
    # Tag Serie A predictions with league field
    for p in predictions_list:
        p.setdefault("league", "serie_a")
    # Load extra league predictions
    for league_key in ACTIVE_LEAGUES:
        if league_key == "serie_a":
            continue
        extra_path = UPCOMING_DIR / f"predictions_{league_key}.json"
        extra_raw = _load_json(extra_path)
        if extra_raw:
            extra_list = extra_raw.get("predictions", []) if isinstance(extra_raw, dict) else extra_raw
            for p in extra_list:
                p.setdefault("league", league_key)
            predictions_list.extend(extra_list)
    pred_by_match = _index_list_by_match(predictions_list)

    # ---- Load AI reasoning cache ----
    ai_cache_dir = DATA_DIR / "ai_reasoning_cache"
    ai_reasoning_by_match = {}
    if ai_cache_dir.exists():
        for f in ai_cache_dir.glob("*.json"):
            try:
                entry = json.loads(f.read_text())
                mk = entry.get("match", "")
                if mk:
                    ai_reasoning_by_match.setdefault(mk, []).append(entry)
            except Exception as e:
                log.debug(f"Failed to load AI reasoning cache entry: {e}")

    # ---- Bets: enrich with time, filter live/completed ----
    ct_map = _get_commence_times()
    raw_bets = unified.get("bets", [])
    # Normalize bet field names (unified_bet_slip vs legacy unified_report)
    for bet in raw_bets:
        if "best_odds" in bet and "odds" not in bet:
            bet["odds"] = bet["best_odds"]
        if "stake_amount" in bet and "stake" not in bet:
            bet["stake"] = bet["stake_amount"]
        if "edge_pct" in bet and "value_pct" not in bet:
            bet["value_pct"] = bet["edge_pct"]
        if "confidence_tier" in bet and "confidence" not in bet:
            bet["confidence"] = bet["confidence_tier"]
        if "model_prob" in bet and "our_probability" not in bet:
            bet["our_probability"] = bet["model_prob"]
        if "best_bookmaker" in bet and not bet.get("bookmaker"):
            bet["bookmaker"] = bet["best_bookmaker"]
    for bet in raw_bets:
        match_key = bet.get("match", "")
        pred = pred_by_match.get(match_key, {})
        if not bet.get("commence_time"):
            bet["commence_time"] = ct_map.get(match_key, pred.get("commence_time", ""))
        if not bet.get("date"):
            bet["date"] = pred.get("date", "")
    main_bets = [b for b in raw_bets if _is_upcoming(b)]
    main_bets = _filter_by_league(main_bets, league_filter)

    # Specialty market bets
    ou_raw = _load_json(UPCOMING_DIR / "over_under_bets.json")
    hc_raw = _load_json(UPCOMING_DIR / "handicap_bets.json")
    cards_raw = _load_json(UPCOMING_DIR / "cards_bets.json")
    btts_raw = _load_json(UPCOMING_DIR / "btts_predictions.json")
    corners_raw = _load_json(UPCOMING_DIR / "corners_predictions.json")
    btts_corners_raw = _load_json(UPCOMING_DIR / "btts_corners_bets.json")

    def _filter_upcoming(items):
        return [b for b in items if _is_upcoming(b, pred_by_match)]

    def _filter_ou_over_only(items):
        """Filter O/U bets to Over only (Under disabled: -1% to -14% ROI in backtest)."""
        return [b for b in items if "OVER" in (b.get("bet", "") or "").upper()]

    ou_rec = _filter_upcoming(ou_raw.get("recommended", []))
    ou_con = _filter_upcoming(ou_raw.get("consider", []))

    specialty_bets = {
        "over_under": {
            "recommended": _filter_ou_over_only(ou_rec),
            "consider": _filter_ou_over_only(ou_con),
        },
        "handicap": {
            # AH DISABLED: -20% to -42% ROI in multi-market backtest (760 matches)
            "recommended": [],
            "consider": [],
        },
        "cards": {
            "recommended": _filter_upcoming(cards_raw.get("recommended", [])),
            "consider": _filter_upcoming(cards_raw.get("consider", [])),
        },
        "btts_corners": {
            "recommended": _filter_upcoming(btts_corners_raw.get("recommended", [])),
            "consider": _filter_upcoming(btts_corners_raw.get("consider", [])),
        },
    }
    btts_predictions = btts_raw if isinstance(btts_raw, list) else btts_raw.get("predictions", [])
    corners_predictions = corners_raw if isinstance(corners_raw, list) else corners_raw.get("predictions", [])

    # Enrich main bets with prediction context
    for bet in main_bets:
        match_key = bet.get("match", "")
        pred = pred_by_match.get(match_key, {})
        bet["home_xg"] = pred.get("home_xg") or None
        bet["away_xg"] = pred.get("away_xg") or None
        bet["predicted_outcome"] = pred.get("predicted_outcome", "")
        bet["match_confidence"] = pred.get("confidence_level", "")

    # ---- Index all bets by match ----
    bets_by_match = defaultdict(list)
    for b in main_bets:
        bets_by_match[b.get("match", "")].append(b)
    for mtype, mdata in specialty_bets.items():
        for b in mdata.get("recommended", []) + mdata.get("consider", []):
            mk = b.get("match", "")
            b["_specialty_type"] = mtype
            # Normalize specialty bets: use fair_odds as odds if no bookmaker odds
            if not b.get("odds") and b.get("fair_odds"):
                b["odds"] = b["fair_odds"]
                b["_advisory"] = True  # No bookmaker odds, advisory only
            if not b.get("selection") and b.get("bet"):
                b["selection"] = b["bet"]
            bets_by_match[mk].append(b)

    # ---- Step 3: Build unified match-centric response ----
    ct_map = _get_commence_times()
    odds_full_matches = odds_full.get("matches", {}) if isinstance(odds_full, dict) else {}
    # Apply league filter to predictions before building match objects
    filtered_pred_by_match = _filter_dict_by_league(pred_by_match, league_filter)
    matches = {}
    for match_key, pred in filtered_pred_by_match.items():
        # Inject commence_time from odds_full for accurate status check
        if not pred.get("commence_time"):
            pred["commence_time"] = ct_map.get(match_key, "")
        # Show upcoming and live matches (hide only completed)
        status = _match_status(pred)
        if status == "completed":
            continue

        home = pred.get("home_team", "")
        away = pred.get("away_team", "")
        home_form = pred.get("home_form", {})
        away_form = pred.get("away_form", {})
        formation = pred.get("formation_analysis", {})
        pa = player_by_match.get(match_key, {})
        h2h = h2h_map.get(match_key, {})
        home_st = standings.get(home, {})
        away_st = standings.get(away, {})

        hf = formation.get("home_formation", "") or pred.get("predicted_home_formation", "")
        af = formation.get("away_formation", "") or pred.get("predicted_away_formation", "")

        # Gather AI reasoning for this match
        ai_entries = ai_reasoning_by_match.get(match_key, [])
        ai_reasoning_text = ""
        if ai_entries:
            best = max(ai_entries, key=lambda e: len(e.get("analysis", "")))
            ai_reasoning_text = best.get("analysis", "")

        matches[match_key] = {
            "status": status,
            "commence_time": pred.get("commence_time", ct_map.get(match_key, "")),
            "date": pred.get("date", ""),
            "venue": pred.get("venue", ""),
            "home_team": home,
            "away_team": away,

            # Prediction core
            "predicted_outcome": pred.get("predicted_outcome", ""),
            "probabilities": pred.get("probabilities", {}),
            "confidence": pred.get("confidence", 0),
            "confidence_level": pred.get("confidence_level", ""),
            "home_xg": pred.get("home_xg") or None,
            "away_xg": pred.get("away_xg") or None,
            "expected_goals": pred.get("expected_goals", {}),
            "over_25": pred.get("over_25", 0),
            "market_edge": pred.get("market_edge", 0),
            "betting_recommendation": pred.get("betting_recommendation", ""),
            "lineup_source": pred.get("lineup_source", "predicted"),

            # Component methods
            "component_predictions": pred.get("component_predictions", {}),
            "market_implied": pred.get("market_implied", {}),

            # Context: standings, form, H2H, Elo
            "home_position": home_st.get("position"),
            "away_position": away_st.get("position"),
            "home_points": home_st.get("points"),
            "away_points": away_st.get("points"),
            "home_form_last5": home_st.get("form_last5", ""),
            "away_form_last5": away_st.get("form_last5", ""),
            "home_ppg": home_form.get("ppg", 0),
            "away_ppg": away_form.get("ppg", 0),
            "home_elo": home_form.get("elo", 0),
            "away_elo": away_form.get("elo", 0),
            "h2h": h2h,

            # Formations
            "home_formation": hf,
            "away_formation": af,
            "home_formation_confidence": pred.get("home_formation_confidence", 0),
            "away_formation_confidence": pred.get("away_formation_confidence", 0),
            "predicted_stats": pred.get("predicted_stats", {}),

            # Injuries — use prediction-embedded data, or EPL injury file as fallback
            "injury_adjustments": pred.get("injury_adjustments") or epl_injury_matches2.get(match_key, {}),

            # Factors
            "home_factors": pred.get("home_factors", []),
            "away_factors": pred.get("away_factors", []),
            "neutral_factors": pred.get("neutral_factors", []),
            "draw_analysis": pred.get("draw_analysis", {}),

            # Player analysis
            "player_analysis": pa,
            "home_strength": pa.get("home_strength", 0),
            "away_strength": pa.get("away_strength", 0),

            # Player props
            "player_props": pp_matches.get(match_key, {}),

            # All bets for this match
            "bets": bets_by_match.get(match_key, []),

            # Odds movement
            "odds_movement": odds_move_matches.get(match_key, {}),

            # Bookmaker analysis
            "bookmaker_analysis": book_matches.get(match_key, {}),

            # Market intelligence
            "market_intelligence": intel_matches.get(match_key, {}),

            # Cross-market signals
            "cross_market": cross_matches.get(match_key, {}),

            # Extended markets
            "extended_markets": ext_matches.get(match_key, {}),

            # Sentiment
            "sentiment": sentiment_by_match.get(match_key, {}),

            # Weather
            "weather": weather_matches.get(match_key, {}),

            # Goal/margin predictions
            "goal_predictions": goal_by_match.get(match_key, {}),
            "margin_predictions": margin_by_match.get(match_key, {}),

            # Per-bookmaker odds
            "odds_bookmakers": odds_bk_matches.get(match_key, {}),

            # Referee
            "referee": pred.get("referee", ""),
            "referee_bias": pred.get("referee_bias", ""),

            # AI reasoning
            "ai_reasoning": ai_reasoning_text,

            # Intelligence adjustments
            "intelligence_adjustments": pred.get("intelligence_adjustments", []),

            # Predicted lineups
            "lineup_prediction": lineup_pred_matches2.get(match_key, {}),

            # League
            "league": pred.get("league", "serie_a"),
        }

    # ---- Best bets: top 10 by composite score ----
    all_value_bets = []
    for mk, m in matches.items():
        for b in m.get("bets", []):
            vp = b.get("value_pct", 0) or 0
            odds = b.get("odds", 0) or b.get("fair_odds", 0) or 0
            # Only include bets that have real value and odds
            if vp <= 0 or odds <= 1:
                continue
            conf_score = b.get("confidence_score", b.get("our_probability", 0)) or 0
            our_prob = b.get("our_probability", 0) or 0
            # Weight edge (value) highest, then confidence, then raw probability
            composite = vp * 0.5 + conf_score * 100 * 0.3 + our_prob * 100 * 0.2
            # Normalize fields for frontend consistency
            enriched = {**b, "_composite": composite, "_match_key": mk}
            if "selection" not in enriched and "bet" in enriched:
                enriched["selection"] = enriched["bet"]
            if "odds" not in enriched and "fair_odds" in enriched:
                enriched["odds"] = enriched["fair_odds"]
            all_value_bets.append(enriched)
    all_value_bets.sort(key=lambda x: x.get("_composite", 0), reverse=True)
    best_bets = all_value_bets[:10]

    # ---- Legacy match_context for backward compat ----
    match_context = {}
    for mk, m in matches.items():
        match_context[mk] = {
            "date": m["date"], "commence_time": m.get("commence_time", ""),
            "home_team": m["home_team"], "away_team": m["away_team"],
            "home_position": m["home_position"], "away_position": m["away_position"],
            "home_points": m["home_points"], "away_points": m["away_points"],
            "home_form_last5": m["home_form_last5"], "away_form_last5": m["away_form_last5"],
            "home_ppg": m["home_ppg"], "away_ppg": m["away_ppg"],
            "home_elo": m["home_elo"], "away_elo": m["away_elo"],
            "h2h": m["h2h"],
            "home_strength": m["home_strength"], "away_strength": m["away_strength"],
            "home_formation": m["home_formation"], "away_formation": m["away_formation"],
            "predicted_stats": m["predicted_stats"],
        }

    # ---- Bankroll (derive from history.json as source of truth) ----
    stats = _get_betting_stats()
    initial_balance = bankroll_file.get("initial_balance", bankroll_state.get("initial_bankroll", 1000))
    # Derive balance and profit from settled bet history (bankroll.json is often stale)
    net_profit = stats["total_profit"]
    current_balance = initial_balance + net_profit
    bankroll = {
        "current_balance": round(current_balance, 2),
        "initial_balance": initial_balance,
        "peak_balance": max(bankroll_file.get("peak_balance", bankroll_state.get("peak_bankroll", current_balance)), current_balance),
        "lowest_balance": bankroll_file.get("lowest_balance", initial_balance),
        "net_profit": round(net_profit, 2),
        "settled_bets": stats["settled_bets"],
        "wins": stats["wins"],
        "losses": stats["losses"],
        "pushes": stats["pushes"],
        "win_rate": stats["win_rate"],
        "roi": stats["roi"],
        "total_stake": stats["total_stake"],
        "total_profit": stats["total_profit"],
        "pending_bets": bankroll_file.get("pending_bets", 0),
        "pending_stakes": bankroll_file.get("pending_stakes", 0),
        "drawdown": risk_state.get("checks", {}).get("drawdown", {}).get("current_drawdown_pct", 0) / 100 if isinstance(risk_state, dict) else 0,
        "current_streak": bankroll_state.get("current_streak", 0),
        "risk_level": risk_state.get("risk_level", "LOW") if isinstance(risk_state, dict) else "LOW",
        "can_bet": risk_state.get("allow_betting", True) if isinstance(risk_state, dict) else True,
        "updated_at": bankroll_file.get("updated_at", bankroll_state.get("last_updated", "")),
    }

    # ---- Compute summary from actual bets (frontend-compatible fields) ----
    raw_summary = unified.get("summary", {})
    all_bets_for_summary = main_bets + [
        b for mdata in specialty_bets.values()
        for b in mdata.get("recommended", [])
    ]
    bet_odds = [b.get("odds", 0) for b in all_bets_for_summary if b.get("odds")]
    bet_values = [b.get("value_pct", 0) for b in all_bets_for_summary if b.get("value_pct")]
    bet_stakes = [b.get("stake", 0) for b in all_bets_for_summary if b.get("stake")]
    total_pot_profit = sum(
        (b.get("stake", 0) or 0) * ((b.get("odds", 0) or 0) - 1)
        for b in all_bets_for_summary if b.get("odds") and b.get("stake")
    )
    # Add potential_profit to each bet for frontend display (use copies to avoid cache mutation)
    all_bets_for_summary = [{**b} for b in all_bets_for_summary]
    for b in all_bets_for_summary:
        if "potential_profit" not in b and b.get("odds") and b.get("stake"):
            b["potential_profit"] = round((b["odds"] - 1) * b["stake"], 2)

    summary = {
        "total_bets": raw_summary.get("total_bets", len(all_bets_for_summary)),
        "total_stake": raw_summary.get("total_stake", round(sum(bet_stakes), 2)),
        "total_potential_profit": raw_summary.get("total_potential_profit", round(total_pot_profit, 2)),
        "expected_profit": raw_summary.get("expected_profit", round(total_pot_profit, 2)),
        "expected_roi_pct": raw_summary.get("expected_roi_pct", 0),
        "average_odds": raw_summary.get("average_odds",
            round(sum(bet_odds) / len(bet_odds), 2) if bet_odds else 0),
        "average_value": raw_summary.get("average_value",
            round(sum(bet_values) / len(bet_values), 1) if bet_values else 0),
        "exposure_pct": raw_summary.get("exposure_pct", 0),
        "matches_covered": raw_summary.get("matches_covered", 0),
        "by_market": raw_summary.get("by_market", {}),
        "by_confidence": raw_summary.get("by_confidence", {}),
    }

    return jsonify({
        "generated_at": unified.get("generated_at", ""),
        "odds_fetched_at": odds_fetched_at,
        "predictions_generated_at": predictions_raw.get("generated_at", "") if isinstance(predictions_raw, dict) else "",
        "extended_markets_at": extended_raw.get("generated_at", "") if isinstance(extended_raw, dict) else "",
        "bankroll": bankroll,
        "summary": summary,
        "match_analyses": unified.get("match_analyses", []),
        "bets": main_bets,
        "specialty_bets": specialty_bets,
        "btts_predictions": btts_predictions,
        "corners_predictions": corners_predictions,
        "match_context": match_context,
        "extended_markets": ext_matches,
        "player_props": pp_matches,
        "parlays": parlay_raw if isinstance(parlay_raw, dict) else {},
        "pipeline_running": _pipeline_running,
        # NEW: match-centric unified data
        "matches": matches,
        "best_bets": best_bets,
        # Placed bets from journal
        "placed_bets": _get_placed_bets(),
        # Market rules (multi-market backtest-calibrated, Feb 17 2026)
        "market_rules": {
            "1X2":       {"enabled": False, "min_edge": 4.0, "backtest_roi": "DISABLED: 5W/13L -17.6% ROI live"},
            "O/U_Over":  {"enabled": True,  "min_edge": 6.0, "backtest_roi": "+25% (blended lineup xG)"},
            "O/U_Under": {"enabled": False, "min_edge": 6.0, "backtest_roi": "-1% to -14%"},
            "AH":        {"enabled": False, "min_edge": 8.0, "backtest_roi": "-20% to -42%"},
            "DC":        {"enabled": True,  "min_edge": 5.0, "backtest_roi": "Unvalidated (no closing odds)"},
            "DNB":       {"enabled": False, "min_edge": 5.0, "backtest_roi": "-37.3% ROI live (0 of 2 won)"},
            "BTTS":      {"enabled": False, "min_edge": 5.0, "backtest_roi": "-7.6% ROI live (no edge visible)"},
            "Alt_OU":    {"enabled": True,  "min_edge": 6.0, "backtest_roi": "Active (alternate lines)"},
        },
        # Risk management state
        "risk_state": risk_state if isinstance(risk_state, dict) else {},
        # Player prop value bets
        "player_prop_value_bets": player_prop_vb.get("bets", []) if isinstance(player_prop_vb, dict) else [],
        # Rejected bets (with reasons)
        "rejected_bets": ultimate_slip.get("rejected_bets", []) if isinstance(ultimate_slip, dict) else [],
        # League filter metadata
        "league_filter": league_filter,
        "active_leagues": _active_leagues_info(),
    })


def _get_placed_bets():
    """Load placed bets from bet journal (pending = actively placed).

    Excludes cancelled/duplicate bets. Includes pipeline_status tag
    so the UI can distinguish current vs stale (old pipeline) bets.
    """
    journal = _load_json(BETTING_DIR / "bet_journal.json", {})
    bets_dict = journal.get("bets", {})
    if not isinstance(bets_dict, dict):
        return []
    placed = []
    for bet_id, bet in bets_dict.items():
        if not isinstance(bet, dict):
            continue
        if bet.get("status") not in ("pending", "settled"):
            continue
        placed.append({
            "bet_id": bet_id,
            "match": bet.get("match", ""),
            "market": bet.get("market", ""),
            "selection": bet.get("selection", ""),
            "odds": bet.get("odds"),
            "stake": bet.get("stake"),
            "status": bet.get("status", "pending"),
            "date": bet.get("date", ""),
            "edge_pct": bet.get("edge_pct"),
            "confidence": bet.get("confidence", ""),
            "profit": bet.get("profit"),
            "result_score": bet.get("result_score"),
            "pipeline_status": bet.get("pipeline_status", "unknown"),
        })
    return placed


# ---------------------------------------------------------------------------
# API: Analytics - P&L, history, performance
# ---------------------------------------------------------------------------

@app.route("/api/analytics")
def api_analytics():
    bankroll_file = _load_json(BETTING_DIR / "bankroll.json")
    bankroll_state = _load_json(BANKROLL_DIR / "state.json")
    history_raw = _load_json(BETTING_DIR / "history.json")
    placed_log_raw = _load_json(BETTING_DIR / "placed_bets_log.json", default=[])

    # --- Bankroll + stats from shared helper (history.json = source of truth) ---
    league_filter = _get_league_filter()
    stats = _get_betting_stats(league_filter=league_filter)
    initial_balance = bankroll_file.get("initial_balance", bankroll_state.get("initial_bankroll", 1000))
    net_profit = stats["total_profit"]
    current_balance = initial_balance + net_profit
    bankroll = {
        "current_balance": round(current_balance, 2),
        "initial_balance": initial_balance,
        "peak_balance": max(bankroll_file.get("peak_balance", bankroll_state.get("peak_bankroll", current_balance)), current_balance),
        "lowest_balance": bankroll_file.get("lowest_balance", initial_balance),
        "net_profit": round(net_profit, 2),
        "settled_bets": stats["settled_bets"],
        "wins": stats["wins"],
        "losses": stats["losses"],
        "pushes": stats["pushes"],
        "win_rate": stats["win_rate"],
        "roi": stats["roi"],
        "total_stake": stats["total_stake"],
        "total_profit": stats["total_profit"],
        "pending_bets": bankroll_file.get("pending_bets", 0),
        "pending_stakes": bankroll_file.get("pending_stakes", 0),
        "updated_at": bankroll_file.get("updated_at", bankroll_state.get("last_updated", "")),
    }

    # --- Parse history.json for bet-level details ---
    settled_bets = []
    pending_bets = []

    if isinstance(history_raw, dict):
        settled_bets = history_raw.get("settled_bets", [])
        pending_bets = history_raw.get("pending_bets", [])
    elif isinstance(history_raw, list):
        settled_bets = history_raw

    # Normalize settled bets: outcome field (uppercase) -> status field
    for bet in settled_bets:
        outcome = bet.get("outcome", bet.get("status", "")).upper()
        bet["status"] = outcome.lower()  # won, lost, push
        bet["profit"] = bet.get("profit_loss", bet.get("profit", 0))

    # Parse placed_bets_log
    if isinstance(placed_log_raw, dict):
        placed_log_raw = placed_log_raw.get("bets", placed_log_raw.get("log", []))
    placed_log = placed_log_raw if isinstance(placed_log_raw, list) else []

    # Combine settled + pending + placed_log (deduped)
    all_bets = []
    seen = set()
    for bet in settled_bets:
        key = f"{bet.get('match', '')}_{bet.get('selection', '')}_{bet.get('date', '')}"
        if key not in seen:
            seen.add(key)
            all_bets.append(bet)
    for bet in pending_bets:
        key = f"{bet.get('match', '')}_{bet.get('selection', '')}_{bet.get('date', '')}"
        if key not in seen:
            seen.add(key)
            bet["status"] = "pending"
            all_bets.append(bet)
    for bet in placed_log:
        key = f"{bet.get('match', '')}_{bet.get('selection', '')}_{bet.get('date', '')}"
        if key not in seen:
            seen.add(key)
            all_bets.append(bet)

    # Sort by date
    all_bets.sort(key=lambda b: b.get("date", b.get("settled_at", "")))

    league_filter = _get_league_filter()
    if league_filter:
        all_bets = [b for b in all_bets if _match_belongs_to_league(b, league_filter)]

    # Use shared stats; compute local vars for breakdowns
    settled_only = [b for b in all_bets if b.get("status") in ("won", "lost", "push")]
    wins = stats["wins"]
    losses = stats["losses"]
    pushes = stats["pushes"]
    pending_count = sum(1 for b in all_bets if b.get("status") == "pending")
    total_stake = stats["total_stake"]
    total_profit = stats["total_profit"]

    # By confidence
    by_confidence = defaultdict(lambda: {"bets": 0, "wins": 0, "losses": 0, "pushes": 0, "stake": 0, "profit": 0})
    for bet in settled_only:
        conf = bet.get("confidence", bet.get("confidence_level", "UNKNOWN"))
        if conf is None:
            conf = "UNKNOWN"
        if isinstance(conf, (int, float)):
            if conf >= 0.58: conf = "VERY HIGH"
            elif conf >= 0.48: conf = "HIGH"
            elif conf >= 0.40: conf = "MEDIUM-HIGH"
            elif conf >= 0.33: conf = "MEDIUM"
            else: conf = "LOW"
        by_confidence[conf]["bets"] += 1
        by_confidence[conf]["stake"] += bet.get("stake", 0)
        by_confidence[conf]["profit"] += bet.get("profit", 0)
        if bet.get("status") == "won":
            by_confidence[conf]["wins"] += 1
        elif bet.get("status") == "lost":
            by_confidence[conf]["losses"] += 1
        else:
            by_confidence[conf]["pushes"] += 1

    # By market
    by_market = defaultdict(lambda: {"bets": 0, "wins": 0, "losses": 0, "pushes": 0, "stake": 0, "profit": 0})
    for bet in settled_only:
        market = bet.get("market", "unknown")
        by_market[market]["bets"] += 1
        by_market[market]["stake"] += bet.get("stake", 0)
        by_market[market]["profit"] += bet.get("profit", 0)
        if bet.get("status") == "won":
            by_market[market]["wins"] += 1
        elif bet.get("status") == "lost":
            by_market[market]["losses"] += 1
        else:
            by_market[market]["pushes"] += 1

    # Cumulative P&L timeline (only settled bets that move P&L)
    cumulative = 0
    timeline = []
    for bet in settled_only:
        profit = bet.get("profit", 0)
        cumulative += profit
        timeline.append({
            "date": bet.get("date", bet.get("settled_at", "")),
            "match": bet.get("match", ""),
            "selection": bet.get("selection", ""),
            "odds": bet.get("odds", 0),
            "stake": bet.get("stake", 0),
            "profit": round(profit, 2),
            "status": bet.get("status", ""),
            "notes": bet.get("notes", ""),
            "cumulative_profit": round(cumulative, 2),
        })

    return jsonify({
        "bankroll": bankroll,
        "summary": {
            "total_bets": len(all_bets),
            "settled_bets": stats["settled_bets"],
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "pending": pending_count,
            "win_rate": stats["win_rate"],
            "total_stake": total_stake,
            "total_profit": total_profit,
            "roi": stats["roi"],
            "pending_stake": round(sum(b.get("stake", 0) for b in all_bets if b.get("status") == "pending"), 2),
        },
        "by_confidence": dict(by_confidence),
        "by_market": dict(by_market),
        "timeline": timeline,
        "all_bets": all_bets,
        # Extra analytics data
        "clv": _load_json(BETTING_DIR / "clv_history.json", default={}),
        "feedback": _load_json(FEEDBACK_DIR / "analysis.json", default={}),
        "calibration": _load_json(FEEDBACK_DIR / "calibration_curve.json", default={}),
        "weights": _load_json(FEEDBACK_DIR / "optimized_weights.json", default={}),
        # Per-market ROI breakdown
        "roi_report": _load_json(FEEDBACK_DIR / "roi_report.json", default={}),
        # Backtest baselines (per-season)
        "backtests": {
            "2023-2024": _load_json(DATA_DIR / "models" / "markets" / "backtest_2023-2024.json", default={}),
            "2024-2025": _load_json(DATA_DIR / "models" / "markets" / "backtest_2024-2025.json", default={}),
        },
    })


# ---------------------------------------------------------------------------
# API: CSV Export
# ---------------------------------------------------------------------------

@app.route("/api/analytics/export")
def api_analytics_export():
    """Export bet history as CSV."""
    import csv, io
    history_raw = _load_json(BETTING_DIR / "history.json")
    placed_log = _load_json(BETTING_DIR / "placed_bets_log.json", default=[])
    if isinstance(placed_log, dict):
        placed_log = placed_log.get("bets", placed_log.get("log", []))

    settled = []
    if isinstance(history_raw, dict):
        settled = history_raw.get("settled_bets", [])
    elif isinstance(history_raw, list):
        settled = history_raw

    all_bets = settled + (placed_log if isinstance(placed_log, list) else [])
    all_bets.sort(key=lambda b: b.get("date", ""))

    output = io.StringIO()
    fields = ["date", "match", "market", "selection", "odds", "stake",
              "profit_loss", "status", "outcome", "confidence", "value_pct", "notes"]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for bet in all_bets:
        row = {k: bet.get(k, "") for k in fields}
        writer.writerow(row)

    from flask import Response
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=bet_history.csv"},
    )


# ---------------------------------------------------------------------------
# API: System Health - pipeline status, prediction tracking, feature drift
# ---------------------------------------------------------------------------

def _live_data_freshness(cached: dict) -> dict:
    """Override cached data_freshness with live file stats.

    The performance_dashboard.json is only regenerated during full pipeline runs.
    Between runs, the cached freshness values go stale even if the underlying
    data was refreshed (e.g., Understat scrape via dashboard button).
    Read actual file mtimes for sources that can be refreshed independently.
    """
    result = dict(cached)

    # Understat PPDA — check parsed parquet (most reliable) then JSON files
    try:
        us_parquet = DATA_DIR / "parsed" / "understat_players.parquet"
        us_dir = DATA_DIR / "external" / "understat"
        # Use the newest of: parsed parquet OR JSON files
        candidates = []
        if us_parquet.exists():
            candidates.append(us_parquet.stat().st_mtime)
        json_files = sorted(us_dir.glob("understat_*.json")) if us_dir.exists() else []
        if json_files:
            candidates.append(max(f.stat().st_mtime for f in json_files))
        if candidates:
            newest = max(candidates)
            mtime = datetime.fromtimestamp(newest)
            age_h = (datetime.now() - mtime).total_seconds() / 3600
            result["understat_ppda"] = {
                "seasons": len(json_files) if json_files else 0,
                "last_updated": mtime.isoformat(),
                "age_hours": round(age_h, 1),
                "fresh": age_h <= 168,  # 7 days
            }
    except Exception:
        pass

    # SofaScore — check newest file across all parquets
    try:
        ss_dir = DATA_DIR / "external" / "sofascore"
        if ss_dir.exists():
            ss_files = [f for f in ss_dir.glob("*.parquet") if f.is_file()]
            if ss_files:
                newest = max(f.stat().st_mtime for f in ss_files)
                mtime = datetime.fromtimestamp(newest)
                age_h = (datetime.now() - mtime).total_seconds() / 3600
                result["sofascore"] = {
                    **(result.get("sofascore") or {}),
                    "last_updated": mtime.isoformat(),
                    "age_hours": round(age_h, 1),
                    "fresh": age_h <= 72,  # 3 days
                }
    except Exception:
        pass

    # Injuries — check newest file in injuries directory
    try:
        inj_dir = DATA_DIR / "external" / "injuries"
        if inj_dir.exists():
            inj_files = [f for f in inj_dir.glob("*.parquet") if f.is_file()]
            if inj_files:
                newest = max(f.stat().st_mtime for f in inj_files)
                mtime = datetime.fromtimestamp(newest)
                age_h = (datetime.now() - mtime).total_seconds() / 3600
                result["injuries"] = {
                    **(result.get("injuries") or {}),
                    "last_updated": mtime.isoformat(),
                    "age_hours": round(age_h, 1),
                    "fresh": age_h <= 72,  # 3 days
                }
    except Exception:
        pass

    # Predictions — check predictions.json mtime
    try:
        pred_path = UPCOMING_DIR / "predictions.json"
        if pred_path.exists():
            mtime = datetime.fromtimestamp(pred_path.stat().st_mtime)
            age_h = (datetime.now() - mtime).total_seconds() / 3600
            result["predictions"] = {
                **(result.get("predictions") or {}),
                "last_updated": mtime.isoformat(),
                "fresh": age_h <= 24,
            }
    except Exception:
        pass

    # Odds — check odds_full.json mtime
    try:
        odds_path = UPCOMING_DIR / "odds_full.json"
        if odds_path.exists():
            mtime = datetime.fromtimestamp(odds_path.stat().st_mtime)
            age_h = (datetime.now() - mtime).total_seconds() / 3600
            result["odds"] = {
                **(result.get("odds") or {}),
                "last_updated": mtime.isoformat(),
                "fresh": age_h <= 4,
            }
    except Exception:
        pass

    return result


@app.route("/api/system")
def api_system():
    dashboard = _load_json(DATA_DIR / "performance_dashboard.json")
    quality = _load_json(DATA_DIR / "quality_report.json")
    archive = _load_json(UPCOMING_DIR / "predictions_archive.json")
    bankroll_state = _load_json(BANKROLL_DIR / "state.json")
    bankroll_live = _load_json(BETTING_DIR / "bankroll.json")
    history_raw = _load_json(BETTING_DIR / "history.json")

    # Prediction archive: convert dict to list
    archive_list = []
    if isinstance(archive, dict):
        for key, pred in archive.items():
            pred["_key"] = key
            archive_list.append(pred)
    archive_list.sort(key=lambda p: p.get("date", ""), reverse=True)

    # Results for matching against archive
    results_raw = _load_json(UPCOMING_DIR / "results.json")
    results_map = {}
    if isinstance(results_raw, dict):
        raw = results_raw.get("results", results_raw)
        if isinstance(raw, dict):
            for k, v in raw.items():
                if isinstance(v, dict):
                    results_map[v.get("match", k)] = v

    # Enrich archive with results
    for pred in archive_list:
        match = pred.get("match", "")
        result = results_map.get(match)
        if result:
            home_score = result.get("home_score", 0)
            away_score = result.get("away_score", 0)
            if home_score > away_score:
                actual = "HOME"
            elif away_score > home_score:
                actual = "AWAY"
            else:
                actual = "DRAW"
            pred["actual_result"] = actual
            pred["score"] = f"{home_score}-{away_score}"
            pred["correct"] = pred.get("predicted_outcome", "").upper() == actual
        else:
            pred["actual_result"] = None
            pred["score"] = None
            pred["correct"] = None

    # Settled bet history
    settled_bets = []
    if isinstance(history_raw, dict):
        settled_bets = history_raw.get("settled_bets", [])

    # Bankroll health — derive from history.json (source of truth) via _get_betting_stats()
    stats = _get_betting_stats()
    initial = bankroll_live.get("initial_balance",
              bankroll_state.get("initial_bankroll", 1000))
    current = initial + stats["total_profit"]
    peak = max(
        bankroll_live.get("peak_balance",
        bankroll_state.get("peak_bankroll", current)),
        current,
    )
    lowest = min(
        bankroll_live.get("lowest_balance", initial),
        current,
    )
    bankroll_health = {
        "current_bankroll": round(current, 2),
        "initial_bankroll": initial,
        "peak_bankroll": round(peak, 2),
        "lowest_bankroll": round(lowest, 2),
        "drawdown": round((peak - current) / peak, 4) if peak > 0 else 0,
        "net_profit": round(stats["total_profit"], 2),
        "total_bets": stats["settled_bets"],
        "total_wins": stats["wins"],
        "wins": stats["wins"],
        "losses": stats["losses"],
        "pushes": stats["pushes"],
        "win_rate": stats["win_rate"],
        "roi": stats["roi"],  # percentage, consistent with other endpoints
        "total_stake": stats["total_stake"],
        "can_bet": True,
        "risk_level": "LOW",
        "updated_at": bankroll_live.get("updated_at", bankroll_state.get("last_updated", "")),
    }

    # Feedback loop data
    fb_analysis = _load_json(FEEDBACK_DIR / "analysis.json")
    fb_weights = _load_json(FEEDBACK_DIR / "optimized_weights.json")
    fb_factors = _load_json(FEEDBACK_DIR / "factor_adjustments.json")
    fb_calibration = _load_json(FEEDBACK_DIR / "calibration_curve.json")

    feedback_loop = {}
    if fb_analysis:
        feedback_loop["overall"] = fb_analysis.get("overall", {})
        feedback_loop["method_brier_scores"] = fb_analysis.get("method_brier_scores", {})
        feedback_loop["intelligence_effectiveness"] = fb_analysis.get("intelligence_effectiveness", {})
        feedback_loop["xg_accuracy"] = fb_analysis.get("xg_accuracy", {})
        feedback_loop["factor_count_accuracy"] = fb_analysis.get("factor_count_accuracy", {})
    if fb_weights:
        feedback_loop["weight_optimization"] = {
            "status": fb_weights.get("status", ""),
            "n_settled": fb_weights.get("n_settled", 0),
            "current_weights": fb_weights.get("current_weights", {}),
            "optimized_weights": fb_weights.get("optimized_weights", {}),
            "changes": fb_weights.get("changes", {}),
        }
    if fb_factors:
        feedback_loop["factor_adjustments"] = {
            "multipliers": fb_factors.get("multipliers", {}),
            "details": fb_factors.get("details", {}),
        }
    if fb_calibration:
        feedback_loop["calibration"] = {
            "status": fb_calibration.get("status", ""),
            "bias": fb_calibration.get("bias", ""),
            "mean_calibration_error": fb_calibration.get("mean_calibration_error", 0),
            "curve": fb_calibration.get("curve", {}),
        }

    # Extra feedback data
    drift_report = _load_json(FEEDBACK_DIR / "drift_report.json", {})
    prediction_audit = _load_json(FEEDBACK_DIR / "prediction_audit.json", {})
    lessons = _load_json(FEEDBACK_DIR / "lessons.json", {})
    market_adj = _load_json(FEEDBACK_DIR / "market_adjustments.json", {})
    # Load per-league deployment states (created by validate_league_deployment.py)
    deployments = {}
    for _lk in ACTIVE_LEAGUES:
        _ds = _load_json(DATA_DIR / "models" / _lk / "deployment_state.json", None)
        if _ds:
            deployments[_lk] = _ds
    if not deployments:
        deployments["universal"] = _load_json(DATA_DIR / "models" / "deployment_state.json", {})
    deployment = deployments

    return jsonify({
        "generated_at": dashboard.get("generated_at", ""),
        "prediction_accuracy": dashboard.get("prediction_accuracy", {}),
        "betting_performance": dashboard.get("betting_performance", {}),
        "bankroll_health": bankroll_health,
        "data_freshness": _live_data_freshness(dashboard.get("data_freshness", quality.get("data_freshness", {}))),
        "feature_drift": dashboard.get("feature_drift", {}),
        "confidence_calibration": dashboard.get("confidence_calibration", {}),
        "season_coverage": quality.get("season_coverage", {}),
        "null_rates": quality.get("null_rates", {}),
        "prediction_archive": archive_list,
        "settled_bets": settled_bets,
        "feedback_loop": feedback_loop,
        "drift_report": drift_report,
        "prediction_audit": prediction_audit,
        "lessons": lessons.get("lessons", []) if isinstance(lessons, dict) else [],
        "market_adjustments": market_adj,
        "deployment_state": deployment,
        "auto_settle": {
            "active": _auto_settle_active,
            "last_run": _auto_settle_last_run,
            "last_result": _auto_settle_last_result,
            "interval_seconds": SETTLE_CHECK_INTERVAL,
        },
        "active_leagues": _active_leagues_info(),
    })


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.route("/api/health")
def api_health():
    data_files = {
        "predictions": UPCOMING_DIR / "predictions.json",
        "odds_full": UPCOMING_DIR / "odds_full.json",
        "market_intelligence": UPCOMING_DIR / "market_intelligence.json",
        "unified_report": BETTING_DIR / "unified_report.json",
        "bankroll": BETTING_DIR / "bankroll.json",
    }

    files = {}
    for name, path in data_files.items():
        if path.exists():
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            age_h = (datetime.now() - mtime).total_seconds() / 3600
            files[name] = {"exists": True, "age_hours": round(age_h, 1), "stale": age_h > 24}
        else:
            files[name] = {"exists": False, "stale": True}

    stale = sum(1 for f in files.values() if f["stale"])
    return jsonify({
        "status": "fresh" if stale == 0 else "stale",
        "files": files,
        "timestamp": datetime.now().isoformat(),
    })


# ---------------------------------------------------------------------------
# API: Log viewer
# ---------------------------------------------------------------------------

_LOG_DIR = _BASE / "logs"

# Available log files with display names and descriptions
_LOG_FILES = {
    # Primary logs — the ones you actually need
    "pipeline":     {"file": "pipeline.log",                        "label": "Pipeline",      "desc": "Main pipeline execution"},
    "errors":       {"file": "errors.log",                          "label": "Errors",        "desc": "Error tracebacks across all modules"},
    "scheduler":    {"file": "scheduler.log",                       "label": "Scheduler",     "desc": "Scheduled jobs (pre-kickoff, settlement, retrain)"},
    "monitor":      {"file": "monitor.log",                         "label": "Health",        "desc": "Health check results"},
    "retrain":      {"file": "retrain.log",                         "label": "Retrain",       "desc": "Model retraining logs"},
    "telegram-bot": {"file": "telegram-bot.log",                    "label": "Telegram",      "desc": "Telegram bot messages & AI calls"},
    # Launchd job logs — combined stdout+stderr per job
    "settlement":   {"file": "launchd-settlement-err.log",          "label": "Settlement",    "desc": "Auto-settlement job output",
                     "also": "launchd-settlement.log"},
    "pre-kickoff":  {"file": "launchd-pre-kickoff-monitor-err.log", "label": "Pre-Kickoff",   "desc": "Lineup fetch & pre-match updates",
                     "also": "launchd-pre-kickoff-monitor.log"},
    "morning":      {"file": "launchd-morning-err.log",             "label": "Morning",       "desc": "Morning pipeline job",
                     "also": "launchd-morning.log"},
    "evening":      {"file": "launchd-evening-err.log",             "label": "Evening",       "desc": "Evening pipeline job",
                     "also": "launchd-evening.log"},
    "health-launchd": {"file": "launchd-health-monitor-err.log",    "label": "Health Job",    "desc": "Launchd health monitor output",
                       "also": "launchd-health-monitor.log"},
}


@app.route("/api/logs")
def api_logs():
    """Return available log files with metadata."""
    log_key = flask_request.args.get("log", "")
    lines = int(flask_request.args.get("lines", 100))
    search = flask_request.args.get("search", "").strip().lower()
    level = flask_request.args.get("level", "").strip().lower()  # error, warning, info

    # If no specific log requested, return the file list
    if not log_key:
        result = []
        # Known log files
        seen_files = set()
        for key, info in _LOG_FILES.items():
            path = _LOG_DIR / info["file"]
            seen_files.add(info["file"])
            also_file = info.get("also", "")
            if also_file:
                seen_files.add(also_file)
            # Combine sizes from primary + also file
            size = 0
            modified = ""
            exists = False
            latest_ts = 0
            for fname in [info["file"]] + ([also_file] if also_file else []):
                fp = _LOG_DIR / fname
                if fp.exists():
                    exists = True
                    size += fp.stat().st_size
                    mt = fp.stat().st_mtime
                    if mt > latest_ts:
                        latest_ts = mt
                        modified = datetime.fromtimestamp(mt).isoformat()
            # Read last line to show latest timestamp in the tab
            last_entry = ""
            if exists and size > 0:
                try:
                    with open(_LOG_DIR / info["file"], "rb") as _f:
                        _f.seek(max(0, _f.seek(0, 2) - 500))
                        tail = _f.read().decode("utf-8", errors="replace").strip()
                        if tail:
                            last_entry = tail.splitlines()[-1][:80]
                except Exception:
                    pass
            result.append({
                "key": key,
                "label": info["label"],
                "desc": info["desc"],
                "file": info["file"],
                "exists": exists,
                "size": size,
                "size_human": f"{size / 1024:.0f}KB" if size < 1048576 else f"{size / 1048576:.1f}MB",
                "modified": modified,
                "last_entry": last_entry,
            })

        # Auto-discover any .log files not in the known list
        if _LOG_DIR.exists():
            for log_file in sorted(_LOG_DIR.glob("*.log")):
                if log_file.name in seen_files:
                    continue
                if log_file.stat().st_size == 0:
                    continue
                auto_key = log_file.stem.replace(".", "-")
                auto_label = log_file.stem.replace("-", " ").replace("_", " ").title()
                result.append({
                    "key": auto_key,
                    "label": auto_label,
                    "desc": f"Auto-discovered: {log_file.name}",
                    "file": log_file.name,
                    "exists": True,
                    "size": log_file.stat().st_size,
                    "size_human": f"{log_file.stat().st_size / 1024:.0f}KB" if log_file.stat().st_size < 1048576 else f"{log_file.stat().st_size / 1048576:.1f}MB",
                    "modified": datetime.fromtimestamp(log_file.stat().st_mtime).isoformat(),
                })

        return jsonify({"logs": result})

    # Return specific log content — validate log_key to prevent path traversal
    import re as _re_logs
    if not _re_logs.match(r'^[a-zA-Z0-9_.-]+$', log_key):
        return jsonify({"error": "Invalid log key"}), 400

    info = _LOG_FILES.get(log_key)
    if not info:
        # Try auto-discovered file
        candidate = _LOG_DIR / f"{log_key.replace('-', '.')}.log"
        if not candidate.exists():
            candidate = _LOG_DIR / f"{log_key}.log"
        if not candidate.exists():
            candidate = _LOG_DIR / f"{log_key.replace('-', '_')}.log"
        # Verify resolved path stays within _LOG_DIR
        if candidate.exists():
            if not str(candidate.resolve()).startswith(str(_LOG_DIR.resolve())):
                return jsonify({"error": "Invalid log path"}), 400
            info = {"file": candidate.name, "label": log_key}
        else:
            return jsonify({"error": f"Unknown log: {log_key}"}), 404

    # Read from primary file + "also" file (merged, sorted by timestamp)
    files_to_read = [_LOG_DIR / info["file"]]
    also_file = info.get("also", "")
    if also_file:
        also_path = _LOG_DIR / also_file
        if also_path.exists():
            files_to_read.append(also_path)

    if not any(f.exists() for f in files_to_read):
        return jsonify({"lines": [], "total": 0, "file": info["file"]})

    try:
        all_raw_lines = []
        fsize = 0
        for fpath in files_to_read:
            if not fpath.exists():
                continue
            with open(fpath, "rb") as f:
                f.seek(0, 2)
                sz = f.tell()
                fsize += sz
                max_bytes = min(sz, max(500_000, lines * 200))
                f.seek(max(0, sz - max_bytes))
                raw = f.read().decode("utf-8", errors="replace")
                all_raw_lines.extend(raw.splitlines())

        # Sort merged lines by timestamp if they have one (YYYY-MM-DD HH:MM:SS)
        import re as _re
        _ts_re = _re.compile(r'^(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2})')
        def _sort_key(line):
            m = _ts_re.match(line)
            return m.group(1) if m else "9999"
        all_raw_lines.sort(key=_sort_key)
        all_lines = all_raw_lines

        # Apply search filter
        if search:
            all_lines = [l for l in all_lines if search in l.lower()]

        # Apply level filter
        if level == "error":
            all_lines = [l for l in all_lines if "error" in l.lower() or "traceback" in l.lower() or "exception" in l.lower()]
        elif level == "warning":
            all_lines = [l for l in all_lines if "warn" in l.lower()]

        # Return last N lines, newest first
        result_lines = all_lines[-lines:]
        result_lines.reverse()

        return jsonify({
            "lines": result_lines,
            "total": len(all_lines),
            "file": info["file"],
            "label": info["label"],
            "size": fsize,
        })
    except Exception as e:
        return jsonify({"error": str(e), "lines": [], "total": 0})


@app.route("/api/logs/clear", methods=["POST"])
def api_logs_clear():
    """Clear one or all log files."""
    target = flask_request.json.get("log", "all") if flask_request.is_json else "all"

    cleared = []
    if target == "all":
        # Clear all known log files
        for key, info in _LOG_FILES.items():
            for fname in [info["file"]] + ([info["also"]] if info.get("also") else []):
                fpath = _LOG_DIR / fname
                if fpath.exists():
                    fpath.write_text("")
                    cleared.append(fname)
        # Also clear rotated files
        for rotated in _LOG_DIR.glob("*.log.1"):
            rotated.write_text("")
            cleared.append(rotated.name)
    else:
        info = _LOG_FILES.get(target)
        if info:
            for fname in [info["file"]] + ([info["also"]] if info.get("also") else []):
                fpath = _LOG_DIR / fname
                if fpath.exists():
                    fpath.write_text("")
                    cleared.append(fname)

    return jsonify({"ok": True, "cleared": cleared})


# ---------------------------------------------------------------------------
# API: Live monitoring data
# ---------------------------------------------------------------------------

def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


@app.route("/api/live")
def api_live():
    """Return today's live monitoring data (scores, odds snapshots, bet tracker)."""
    today = _today_utc()
    path = LIVE_DIR / f"{today}.json"
    data = _load_json(path, default=None)

    # If auto-poll isn't running, only start it when a match is IMMINENT
    # (kicks off within next 30 min) or already live. Just being a "match day"
    # is too lax — burns ~24 credits/hour for hours before kickoff.
    if not _auto_poll_active:
        try:
            from scripts.pipeline.scheduler import get_kickoff_times
            from datetime import datetime as _dt, timezone as _tz
            kickoffs = get_kickoff_times()
            now = _dt.now(_tz.utc)
            imminent = any(
                -180 <= (k["kickoff_utc"] - now).total_seconds() / 60 <= 30
                for k in kickoffs
            )
            if imminent:
                _ensure_auto_poll()
                log.info("Live poll auto-started — match imminent (within 30min) or live")
        except Exception:
            pass
    # Re-read file in case a poll just completed
    if data is None:
        data = _load_json(path, default=None)

    if data is None:
        return jsonify({
            "date": today,
            "polls": 0,
            "api_calls": 0,
            "matches": {},
            "bet_tracking": [],
            "has_live": False,
            "auto_poll_active": _auto_poll_active,
            "auto_poll_interval": _auto_poll_interval,
            "auto_poll_next_at": 0,
        })

    # Filter out matches from previous days (they leak in when monitor runs past midnight)
    filtered_matches = {}
    for mk, md in data.get("matches", {}).items():
        commence = md.get("commence_time", "")
        if commence and commence[:10] < today:
            continue  # skip yesterday's matches
        filtered_matches[mk] = md

    league_filter = _get_league_filter()
    if league_filter:
        filtered_matches = {k: v for k, v in filtered_matches.items()
                           if _team_belongs_to_league(v.get("home_team", v.get("home", "")), league_filter)}

    data["matches"] = filtered_matches

    # Determine if any match is currently live
    has_live = False
    for mk, md in filtered_matches.items():
        st = md.get("status", "")
        if st in ("first_half", "half_time", "second_half"):
            has_live = True
            break

    data["has_live"] = has_live
    data["auto_poll_active"] = _auto_poll_active
    data["auto_poll_interval"] = _auto_poll_interval
    data["auto_poll_next_at"] = _auto_poll_next_at if _auto_poll_active else 0
    return jsonify(data)


@app.route("/api/live/match/<match_slug>")
def api_live_match(match_slug):
    """Rich live data for a single match (events, stats, odds timeline).

    Returns combined data from live monitoring (scores/odds) and Sofascore
    (events/stats). Used by the prediction detail page during live matches.
    """
    today = _today_utc()
    path = LIVE_DIR / f"{today}.json"
    data = _load_json(path, default=None)

    if not data:
        return jsonify({"found": False, "match_slug": match_slug})

    # Find match by slug — try both raw key and normalized team names
    import re
    from config.team_names import normalize_team as _nt

    def _to_slug(s):
        return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

    target = None
    target_key = None
    for mk, md in data.get("matches", {}).items():
        # Try raw key slug
        if _to_slug(mk) == match_slug:
            target = md
            target_key = mk
            break
        # Try normalized team names (Odds API uses "Hellas Verona", we use "Verona")
        home = _nt(md.get("home_team", ""))
        away = _nt(md.get("away_team", ""))
        if _to_slug(f"{home} vs {away}") == match_slug:
            target = md
            target_key = mk
            break

    if not target:
        return jsonify({"found": False, "match_slug": match_slug})

    # Build response with all live data
    snapshots = target.get("snapshots", [])
    last_snap = snapshots[-1] if snapshots else {}

    result = {
        "found": True,
        "match_key": target_key,
        "home_team": target.get("home_team", ""),
        "away_team": target.get("away_team", ""),
        "status": target.get("status", ""),
        "commence_time": target.get("commence_time", ""),
        "score": last_snap.get("score", [0, 0]),
        "minute": last_snap.get("min"),
        "final_score": target.get("final_score"),

        # Pre-match baseline
        "pre_match_odds": target.get("pre_match_odds", {}),

        # Current odds
        "current_odds": last_snap.get("avg_odds", {}),
        "sharp_odds": last_snap.get("sharp", {}),

        # Odds timeline (all snapshots with ts, score, odds)
        "odds_timeline": [{
            "ts": s.get("ts"),
            "min": s.get("min"),
            "score": s.get("score"),
            "odds": s.get("avg_odds", {}),
        } for s in snapshots],

        # Sofascore live data
        "events": target.get("live_events", []),
        "statistics": target.get("live_stats", {}),
        "player_stats": target.get("live_player_stats", {}),
        "sofascore_id": target.get("sofascore_id"),
        "sofascore_fetched_at": target.get("sofascore_fetched_at", ""),

        # Bet tracking for this match
        "bet_tracking": [
            bt for bt in data.get("bet_tracking", [])
            if bt.get("match") == target_key
        ],

        # Polling metadata
        "polls": data.get("polls", 0),
        "last_poll_at": snapshots[-1].get("ts") if snapshots else None,
    }

    return jsonify(result)


@app.route("/api/live/props/<match_slug>")
def api_live_props(match_slug):
    """Live player prop evaluation for a match.

    Cross-references Sofascore per-player live stats against player_prop_value_bets.json
    to determine HIT / OPEN / LOST status for each prop bet.
    """
    import re
    from config.team_names import normalize_team as _nt

    today = _today_utc()
    live_path = LIVE_DIR / f"{today}.json"
    live_data = _load_json(live_path, default=None)

    if not live_data:
        return jsonify({"found": False, "match_slug": match_slug, "props": []})

    def _to_slug(s):
        return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

    # Find match in live data
    target = None
    target_key = None
    for mk, md in live_data.get("matches", {}).items():
        if _to_slug(mk) == match_slug:
            target = md
            target_key = mk
            break
        home = _nt(md.get("home_team", ""))
        away = _nt(md.get("away_team", ""))
        if _to_slug(f"{home} vs {away}") == match_slug:
            target = md
            target_key = mk
            break

    if not target:
        return jsonify({"found": False, "match_slug": match_slug, "props": []})

    # Get live player stats and events
    player_stats = target.get("live_player_stats", {})
    live_events = target.get("live_events", [])
    match_status = target.get("status", "")
    snapshots = target.get("snapshots", [])
    last_snap = snapshots[-1] if snapshots else {}
    is_completed = match_status == "completed"

    # Build event lookup: lowercase player name -> list of events
    stat_lookup = {}
    for side in ("home", "away"):
        for p in player_stats.get(side, []):
            name = p.get("name", "")
            if name:
                stat_lookup[name.lower()] = p
                stat_lookup[strip_accents(name).lower()] = p
                # Also index by short name and surname
                short = p.get("short_name", "")
                if short:
                    stat_lookup[short.lower()] = p
                    stat_lookup[strip_accents(short).lower()] = p
                parts = name.split()
                if len(parts) > 1:
                    stat_lookup[parts[-1].lower()] = p
                    stat_lookup[strip_accents(parts[-1]).lower()] = p

    # Build event lookup: lowercase player name -> list of relevant events
    event_lookup = {}
    for ev in live_events:
        ev_type = ev.get("type", "")
        minute = ev.get("minute", "")
        added = ev.get("added_time", 0)
        min_str = f"{minute}'" if not added else f"{minute}+{added}'"

        if ev_type == "goal":
            scorer = ev.get("player", "")
            assister = ev.get("assist", "")
            goal_type = ev.get("goal_type", "regular")
            if scorer:
                key = strip_accents(scorer).lower()
                event_lookup.setdefault(key, []).append(f"Scored {min_str}" + (f" ({goal_type})" if goal_type != "regular" else ""))
            if assister:
                key = strip_accents(assister).lower()
                event_lookup.setdefault(key, []).append(f"Assist {min_str}")
        elif ev_type == "card":
            player = ev.get("player", "")
            card_type = ev.get("card_type", "yellow")
            if player:
                key = strip_accents(player).lower()
                event_lookup.setdefault(key, []).append(f"{card_type.capitalize()} card {min_str}")

    def _get_live_evidence(player_name, market, pstats_data, result):
        """Build human-readable live evidence string."""
        # Check event lookup for this player (try multiple name forms)
        name_keys = [strip_accents(player_name).lower()]
        parts = player_name.split()
        if len(parts) > 1:
            name_keys.append(strip_accents(parts[-1]).lower())

        player_events = []
        for nk in name_keys:
            player_events.extend(event_lookup.get(nk, []))
        # Deduplicate while preserving order
        seen = set()
        unique_events = []
        for e in player_events:
            if e not in seen:
                seen.add(e)
                unique_events.append(e)

        market_lower = market.lower()
        status = result.get("status", "open") if result else "no_data"
        actual = result.get("actual") if result else None
        line = result.get("line") if result else None

        # For goalscorer props, prioritize event-based evidence
        if "goal" in market_lower:
            goal_evs = [e for e in unique_events if e.startswith("Scored")]
            if goal_evs:
                return "; ".join(goal_evs)
            if actual is not None and actual == 0:
                return "No goals yet" if not is_completed else "No goals"

        # For assist props
        if "assist" in market_lower:
            assist_evs = [e for e in unique_events if e.startswith("Assist")]
            if assist_evs:
                return "; ".join(assist_evs)
            if actual is not None:
                return f"{actual} assist(s)" if actual > 0 else ("No assists yet" if not is_completed else "No assists")

        # For card props
        if "card" in market_lower or "booked" in market_lower:
            card_evs = [e for e in unique_events if "card" in e.lower()]
            if card_evs:
                return "; ".join(card_evs)
            if actual is not None:
                return f"{actual} card(s)" if actual > 0 else ("Not carded yet" if not is_completed else "Not carded")

        # For shot props
        if "shot" in market_lower:
            stat_key = "shots_on_target" if ("sot" in market_lower or "target" in market_lower) else "shots"
            label = "SOT" if stat_key == "shots_on_target" else "shots"
            if pstats_data and actual is not None:
                return f"{actual} {label}" + (f" (line {line})" if line else "")

        # For tackle/foul/cross/key pass props with numeric stats
        if actual is not None and line is not None:
            return f"{actual} (line {line})"

        if not pstats_data:
            return "No live stats available"

        return None

    # Load prop value bets
    prop_vb = _load_json(UPCOMING_DIR / "player_prop_value_bets.json")
    all_bets = prop_vb.get("bets", []) if isinstance(prop_vb, dict) else []

    # Also load base props for richer data
    prop_base = _load_json(UPCOMING_DIR / "player_props.json")
    pp_matches = prop_base.get("matches", {}) if isinstance(prop_base, dict) else {}

    # Filter bets for this match (try normalized key match)
    norm_target = None
    if target_key:
        parts = target_key.split(" vs ", 1)
        if len(parts) == 2:
            norm_target = f"{_nt(parts[0].strip())} vs {_nt(parts[1].strip())}"

    match_bets = []
    for b in all_bets:
        bm = b.get("match", "")
        if bm == target_key or bm == norm_target:
            match_bets.append(b)
        elif norm_target:
            bm_parts = bm.split(" vs ", 1)
            if len(bm_parts) == 2:
                bm_norm = f"{_nt(bm_parts[0].strip())} vs {_nt(bm_parts[1].strip())}"
                if bm_norm == norm_target:
                    match_bets.append(b)

    # Evaluate each prop bet against live stats
    evaluated = []
    for bet in match_bets:
        player_name = bet.get("player", "")
        market = bet.get("market", "")

        # Find player's live stats (with accent-stripped fallback)
        pstats = stat_lookup.get(player_name.lower())
        if not pstats:
            pstats = stat_lookup.get(strip_accents(player_name).lower())
        if not pstats:
            # Try surname match
            parts = player_name.split()
            if len(parts) > 1:
                pstats = stat_lookup.get(parts[-1].lower())
                if not pstats:
                    pstats = stat_lookup.get(strip_accents(parts[-1]).lower())

        result_status = "no_data"
        actual_value = None

        if pstats:
            result_status = "open"
            actual_value = _evaluate_prop(market, pstats, is_completed)
            if actual_value is not None:
                result_status = actual_value.get("status", "open")

        # Build live evidence string from events and stats
        live_evidence = _get_live_evidence(player_name, market, pstats, actual_value)

        evaluated.append({
            "player": player_name,
            "market": market,
            "selection": bet.get("selection", "Yes"),
            "odds": bet.get("best_odds"),
            "status": result_status,
            "live_evidence": live_evidence,
            "actual": actual_value.get("actual") if actual_value else None,
            "line": actual_value.get("line") if actual_value else None,
            "edge_pct": bet.get("edge_pct"),
            "tier": bet.get("tier"),
            "best_bookmaker": bet.get("best_bookmaker"),
        })

    return jsonify({
        "found": True,
        "match": target_key,
        "match_slug": match_slug,
        "match_status": match_status,
        "is_completed": is_completed,
        "minute": last_snap.get("min"),
        "score": last_snap.get("score", [0, 0]),
        "props": evaluated,
        "player_count": sum(len(player_stats.get(s, [])) for s in ("home", "away")),
    })


def _evaluate_prop(market: str, pstats: dict, is_completed: bool) -> dict:
    """Evaluate a single prop bet against live player stats.

    Returns {"status": "hit"|"lost"|"open"|"winning"|"losing", "actual": value, "line": line}
    """
    market_lower = market.lower()

    # Anytime Goalscorer
    if "anytime" in market_lower and "goal" in market_lower:
        goals = pstats.get("goals", 0)
        return {
            "status": "hit" if goals >= 1 else ("lost" if is_completed else "open"),
            "actual": goals,
            "line": 0.5,
        }

    # 2+ Goals
    if ("2+" in market_lower or "two" in market_lower) and "goal" in market_lower:
        goals = pstats.get("goals", 0)
        return {
            "status": "hit" if goals >= 2 else ("lost" if is_completed else ("winning" if goals >= 1 else "open")),
            "actual": goals,
            "line": 1.5,
        }

    # First / Last Goalscorer — can't evaluate precisely, treat as anytime
    if ("first" in market_lower or "last" in market_lower) and "goal" in market_lower:
        goals = pstats.get("goals", 0)
        return {
            "status": "open",  # Can't determine first/last from stats alone
            "actual": goals,
            "line": None,
        }

    # Shots On Target Over X.5 (check before general shots — "sot" must match first)
    if ("sot" in market_lower or ("shot" in market_lower and "target" in market_lower)):
        sot = pstats.get("shots_on_target", 0)
        line = extract_line(market_lower, default=0.5)
        if sot > line:
            return {"status": "hit", "actual": sot, "line": line}
        return {
            "status": "lost" if is_completed else "open",
            "actual": sot,
            "line": line,
        }

    # Shots Over X.5 (general — not "on target")
    if "shot" in market_lower and "target" not in market_lower:
        shots = pstats.get("shots", 0)
        line = extract_line(market_lower, default=0.5)
        if shots > line:
            return {"status": "hit", "actual": shots, "line": line}
        return {
            "status": "lost" if is_completed else "open",
            "actual": shots,
            "line": line,
        }

    # Tackles Over X.5
    if "tackle" in market_lower:
        tackles = pstats.get("tackles", 0)
        line = extract_line(market_lower, default=0.5)
        if tackles > line:
            return {"status": "hit", "actual": tackles, "line": line}
        return {
            "status": "lost" if is_completed else "open",
            "actual": tackles,
            "line": line,
        }

    # Fouls Over X.5
    if "foul" in market_lower:
        fouls = pstats.get("fouls_committed", 0)
        line = extract_line(market_lower, default=0.5)
        if fouls > line:
            return {"status": "hit", "actual": fouls, "line": line}
        return {
            "status": "lost" if is_completed else "open",
            "actual": fouls,
            "line": line,
        }

    # To Be Carded
    if "card" in market_lower or "booked" in market_lower:
        yellows = pstats.get("yellow_cards", 0)
        reds = pstats.get("red_cards", 0)
        cards = yellows + reds
        return {
            "status": "hit" if cards >= 1 else ("lost" if is_completed else "open"),
            "actual": cards,
            "line": 0.5,
        }

    # Assists
    if "assist" in market_lower:
        assists = pstats.get("assists", 0)
        line = extract_line(market_lower, default=0.5)
        if assists > line:
            return {"status": "hit", "actual": assists, "line": line}
        return {
            "status": "lost" if is_completed else "open",
            "actual": assists,
            "line": line,
        }

    # Key Passes
    if "key" in market_lower and "pass" in market_lower:
        kp = pstats.get("key_passes", 0)
        line = extract_line(market_lower, default=0.5)
        if kp > line:
            return {"status": "hit", "actual": kp, "line": line}
        return {
            "status": "lost" if is_completed else "open",
            "actual": kp,
            "line": line,
        }

    # Crosses
    if "cross" in market_lower:
        crosses = pstats.get("crosses", 0)
        line = extract_line(market_lower, default=0.5)
        if crosses > line:
            return {"status": "hit", "actual": crosses, "line": line}
        return {
            "status": "lost" if is_completed else "open",
            "actual": crosses,
            "line": line,
        }

    return {"status": "open", "actual": None, "line": None}



def _ensure_auto_poll():
    """Restart auto-poll if it's not running and there are live matches."""
    global _auto_poll_active, _auto_poll_thread
    if _auto_poll_active and _auto_poll_thread and _auto_poll_thread.is_alive():
        return  # Already running
    _auto_poll_thread = threading.Thread(target=_auto_poll_loop, daemon=True)
    _auto_poll_thread.start()


@app.route("/api/live/trigger", methods=["POST"])
def api_live_trigger():
    """Trigger a single live poll on demand (2 API calls)."""
    try:
        from scripts.data.live_monitor import poll_once
        result = poll_once()
        # If we found live matches, make sure auto-poll is running
        if result.get("has_live_matches"):
            _ensure_auto_poll()
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        log.error(f"Live poll trigger failed: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Background auto-poll thread (enhanced with configurable interval)
# ---------------------------------------------------------------------------

_auto_poll_active = False
_auto_poll_thread = None
_auto_poll_interval = 300  # seconds (5 min default — was 60s which burnt ~2880 Odds API calls/day)
_auto_poll_next_at = 0.0  # timestamp of next poll

def _auto_poll_loop():
    """Background thread: poll at configurable interval while live matches exist."""
    global _auto_poll_active, _auto_poll_next_at
    _auto_poll_active = True
    consecutive_no_live = 0
    consecutive_auth_errors = 0

    log.info(f"Auto-poll thread started (every {_auto_poll_interval}s)")

    while _auto_poll_active:
        try:
            from scripts.data.live_monitor import poll_once
            result = poll_once()

            if result.get("error"):
                log.warning(f"Auto-poll error: {result['error']}")
                break

            if not result.get("has_live_matches"):
                consecutive_no_live += 1
                log.info(f"Auto-poll: no live matches ({consecutive_no_live}/4)")
                # Stop after 4 empty polls (20 min) instead of 12 (60 min) — saves ~16 credits per false start
                if consecutive_no_live >= 4:
                    log.info("Auto-poll: stopping after 4 polls with no live matches")
                    break
            else:
                consecutive_no_live = 0
            consecutive_auth_errors = 0

        except Exception as e:
            msg = str(e)
            if "401" in msg or "OUT_OF_USAGE_CREDITS" in msg or "Unauthorized" in msg:
                consecutive_auth_errors += 1
                if consecutive_auth_errors >= 3:
                    log.warning(
                        "Auto-poll: 3 consecutive auth/quota errors — stopping thread. "
                        "Will restart on next match-day trigger."
                    )
                    break
                log.warning(f"Auto-poll auth error ({consecutive_auth_errors}/3): {e}")
            else:
                log.error(f"Auto-poll error: {e}")

        # Sleep in 5s chunks for responsive stop and interval changes
        _auto_poll_next_at = _time.time() + _auto_poll_interval
        elapsed = 0
        while elapsed < _auto_poll_interval and _auto_poll_active:
            _time.sleep(5)
            elapsed += 5
            # Re-read interval in case it changed mid-sleep
            if elapsed < _auto_poll_interval and _auto_poll_interval < elapsed + 5:
                break  # interval was shortened, break early

    _auto_poll_active = False
    _auto_poll_next_at = 0.0
    log.info("Auto-poll thread stopped")


@app.route("/api/live/auto-poll", methods=["POST"])
def api_live_auto_poll():
    """Start or stop the background auto-poll thread."""
    global _auto_poll_active, _auto_poll_thread
    action = flask_request.json.get("action", "start") if flask_request.is_json else "start"

    if action == "stop":
        _auto_poll_active = False
        return jsonify({"ok": True, "active": False})

    if _auto_poll_active and _auto_poll_thread and _auto_poll_thread.is_alive():
        return jsonify({"ok": True, "active": True, "message": "already running"})

    _auto_poll_thread = threading.Thread(target=_auto_poll_loop, daemon=True)
    _auto_poll_thread.start()
    return jsonify({"ok": True, "active": True})


@app.route("/api/live/config", methods=["POST"])
def api_live_config():
    """Set live poll interval (seconds). Accepts 30-3600."""
    global _auto_poll_interval
    data = flask_request.get_json(silent=True) or {}
    interval = data.get("interval", _auto_poll_interval)
    interval = max(30, min(3600, int(interval)))
    _auto_poll_interval = interval
    log.info(f"Live poll interval set to {interval}s")
    return jsonify({"ok": True, "interval": interval})


# ---------------------------------------------------------------------------
# API: Match Clock — lightweight real-time status for all matches
# ---------------------------------------------------------------------------

def _parse_kickoff(commence_time: str, date_str: str = "", time_str: str = ""):
    """Parse kickoff into UTC datetime.

    Priority:
    1. commence_time (UTC ISO string from Odds API) — authoritative
    2. commence_time from odds_full map (by match key)
    3. Date-only fallback: assume 19:45 UTC (typical prime-time Italian kickoff)

    NOTE: The `time` field in predictions.json is unreliable (wrong timezone).
    We intentionally skip date+time fallback and use date-only with typical
    Italian kickoff time.
    """
    # 1. Try commence_time directly
    if commence_time:
        try:
            return datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        except (ValueError, TypeError) as e:
            log.debug(f"Failed to parse kickoff commence_time: {e}")

    # 2. Try to find commence_time from odds_full map
    ct_map = _get_commence_times()
    # We don't have match_key here, but the caller may pass it via commence_time
    # This fallback is for when we only have date_str
    if date_str:
        try:
            # Assume typical Italian prime-time kickoff: 20:45 CET = 19:45 UTC
            local = datetime.strptime(f"{date_str} 19:45", "%Y-%m-%d %H:%M")
            return local.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError) as e:
            log.debug(f"Failed to parse kickoff date-only: {e}")
    return None


def _format_countdown(secs: float) -> str:
    """Human-readable countdown string from seconds until kickoff."""
    if secs <= 0:
        return ""
    secs = int(secs)
    d = secs // 86400
    h = (secs % 86400) // 3600
    m = (secs % 3600) // 60
    if d >= 2:
        return f"in {d}d {h}h"
    if d == 1:
        return f"in 1d {h}h"
    if h > 0:
        return f"in {h}h {m}m"
    if m > 0:
        return f"in {m}m"
    return f"in {secs}s"


@app.route("/api/match-clock")
def api_match_clock():
    """Real-time match status for all upcoming/live/completed-today matches.
    Reads local files only — no external API calls. Designed for 30s polling."""
    now = datetime.now(timezone.utc)
    today_str = now.strftime("%Y-%m-%d")

    # Load data sources (all leagues)
    predictions_raw = _load_json(UPCOMING_DIR / "predictions.json")
    odds_full = _load_json(UPCOMING_DIR / "odds_full.json")
    live_data = _load_json(LIVE_DIR / f"{today_str}.json", default=None)

    predictions_list = predictions_raw.get("predictions", []) if isinstance(predictions_raw, dict) else []
    for p in predictions_list:
        p.setdefault("league", "serie_a")
    # Merge EPL predictions + odds
    for _lk in ACTIVE_LEAGUES:
        if _lk == "serie_a":
            continue
        _ep = _load_json(UPCOMING_DIR / f"predictions_{_lk}.json")
        if _ep:
            _el = _ep.get("predictions", []) if isinstance(_ep, dict) else []
            for p in _el:
                p.setdefault("league", _lk)
            predictions_list.extend(_el)
    odds_matches = odds_full.get("matches", {}) if isinstance(odds_full, dict) else {}
    _epl_of = _load_json(UPCOMING_DIR / "odds_full_premier_league.json")
    if _epl_of and isinstance(_epl_of.get("matches"), dict):
        odds_matches.update(_epl_of["matches"])
    live_matches = live_data.get("matches", {}) if live_data else {}
    live_bet_tracking = live_data.get("bet_tracking", []) if live_data else []

    # Build normalized lookup for live data (Odds API uses "Hellas Verona", pipeline uses "Verona")
    _live_normalized = {}
    try:
        from config.team_names import normalize_team as _nt
        for lmk, lmd in live_matches.items():
            parts = lmk.split(" vs ", 1)
            if len(parts) == 2:
                norm_key = f"{_nt(parts[0].strip())} vs {_nt(parts[1].strip())}"
                _live_normalized[norm_key] = lmk
    except ImportError:
        pass

    def _find_live_entry(mk):
        """Find live data entry by exact key or normalized team names."""
        if mk in live_matches:
            return live_matches[mk]
        if mk in _live_normalized:
            return live_matches[_live_normalized[mk]]
        return {}

    matches = {}
    for pred in predictions_list:
        home = pred.get("home_team", "")
        away = pred.get("away_team", "")
        mk = pred.get("match", "") or (f"{home} vs {away}" if home and away else "")
        if not mk:
            continue

        odds_data = odds_matches.get(mk, {})
        commence = odds_data.get("commence_time", "")

        # Parse kickoff
        kickoff = _parse_kickoff(commence, pred.get("date", ""))
        if kickoff is None:
            continue

        secs_until = (kickoff - now).total_seconds()

        # Check live monitor data (with normalized key fallback)
        live_entry = _find_live_entry(mk)
        live_status = live_entry.get("status", "")
        live_snapshots = live_entry.get("snapshots", [])
        last_snap = live_snapshots[-1] if live_snapshots else None

        # Determine status and clock info
        if live_status == "completed" or live_entry.get("final_score"):
            # Completed — use final score from live monitor
            final = live_entry.get("final_score") or (last_snap.get("score") if last_snap else None)
            matches[mk] = {
                "status": "completed",
                "clock": "FT",
                "minute": 90,
                "period": "FT",
                "score": final,
                "secs_until": None,
                "kickoff_utc": kickoff.isoformat(),
                "live_odds": last_snap.get("avg_odds") if last_snap else None,
                "pre_match_odds": live_entry.get("pre_match_odds") or (live_snapshots[0].get("avg_odds") if live_snapshots else None),
                "bet_tracking": [bt for bt in live_bet_tracking if bt.get("match") == mk],
            }
        elif live_status in ("first_half", "half_time", "second_half"):
            # Live — use live monitor data
            score = last_snap.get("score") if last_snap else None
            minute = last_snap.get("min")
            if live_status == "half_time":
                period = "HT"
                clock = "HT"
            elif live_status == "first_half":
                period = "1H"
                clock = f"{minute}'" if minute else "1H"
            else:
                period = "2H"
                clock = f"{minute}'" if minute else "2H"
            matches[mk] = {
                "status": "half_time" if live_status == "half_time" else "live",
                "clock": clock,
                "minute": minute,
                "period": period,
                "score": score,
                "secs_until": None,
                "kickoff_utc": kickoff.isoformat(),
                "live_odds": last_snap.get("avg_odds") if last_snap else None,
                "pre_match_odds": live_entry.get("pre_match_odds") or (live_snapshots[0].get("avg_odds") if live_snapshots else None),
                "bet_tracking": [bt for bt in live_bet_tracking if bt.get("match") == mk],
            }
        elif secs_until <= -7200:
            # More than 2h past kickoff, no live data — probably completed
            continue
        elif secs_until <= 0:
            # Past kickoff but no live monitor data — estimate from clock
            elapsed_min = abs(secs_until) / 60
            if elapsed_min <= 47:
                minute = min(int(elapsed_min), 45)
                period = "1H"
                clock = f"{minute}'"
            elif elapsed_min <= 62:
                minute = 45
                period = "HT"
                clock = "HT"
            elif elapsed_min <= 110:
                minute = min(45 + int(elapsed_min - 62), 90)
                period = "2H"
                clock = f"{minute}'"
            else:
                # Likely finished
                period = "FT"
                minute = 90
                clock = "FT"
            status = "half_time" if period == "HT" else ("completed" if period == "FT" else "live")
            matches[mk] = {
                "status": status,
                "clock": clock,
                "minute": minute,
                "period": period,
                "score": None,
                "secs_until": None,
                "kickoff_utc": kickoff.isoformat(),
                "live_odds": None,
                "pre_match_odds": None,
                "bet_tracking": [bt for bt in live_bet_tracking if bt.get("match") == mk],
            }
        else:
            # Upcoming
            matches[mk] = {
                "status": "upcoming",
                "clock": _format_countdown(secs_until),
                "minute": None,
                "period": None,
                "score": None,
                "secs_until": secs_until,
                "kickoff_utc": kickoff.isoformat(),
                "live_odds": None,
                "pre_match_odds": None,
                "bet_tracking": [],
            }

    return jsonify({
        "ts": now.isoformat(),
        "auto_poll_active": _auto_poll_active,
        "auto_poll_interval": _auto_poll_interval,
        "matches": matches,
    })


# ---------------------------------------------------------------------------
# Pipeline Refresh (run full pipeline from web UI)
# ---------------------------------------------------------------------------

_pipeline_running = False
_pipeline_progress = {"step": 0, "total": 28, "message": "", "started_at": "", "error": ""}


def _run_pipeline_background():
    """Run the full pipeline in a background thread."""
    global _pipeline_running, _pipeline_progress
    _pipeline_running = True
    _pipeline_progress = {
        "step": 0, "total": 28, "message": "Starting pipeline...",
        "started_at": datetime.now().isoformat(), "error": "",
    }

    try:
        from scripts.pipeline.run_full_pipeline import run_pipeline
        run_pipeline(quick=False)
        _pipeline_progress["message"] = "Pipeline complete!"
        _pipeline_progress["step"] = 28
    except Exception as e:
        log.error(f"Pipeline refresh failed: {e}")
        _pipeline_progress["error"] = str(e)
        _pipeline_progress["message"] = f"Failed: {e}"
    finally:
        _pipeline_running = False


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    """Trigger a full pipeline re-run in background."""
    global _pipeline_running
    if _pipeline_running:
        return jsonify({"ok": False, "message": "Pipeline already running", "progress": _pipeline_progress})

    t = threading.Thread(target=_run_pipeline_background, daemon=True)
    t.start()
    return jsonify({"ok": True, "message": "Pipeline started", "progress": _pipeline_progress})


@app.route("/api/refresh/status")
def api_refresh_status():
    """Check pipeline progress."""
    # Estimate progress from file modification times
    progress = dict(_pipeline_progress)
    progress["running"] = _pipeline_running

    if _pipeline_running:
        # Check which files have been updated since pipeline started
        started = _pipeline_progress.get("started_at", "")
        if started:
            try:
                start_dt = datetime.fromisoformat(started)
                check_files = [
                    (1, UPCOMING_DIR / "odds_full.json", "Fetching odds..."),
                    (3, UPCOMING_DIR / "odds_bookmakers.json", "Bookmaker analysis..."),
                    (5, UPCOMING_DIR / "odds_movement.json", "Odds movement..."),
                    (7, UPCOMING_DIR / "market_intelligence.json", "Market intelligence..."),
                    (9, UPCOMING_DIR / "predictions.json", "Running predictions..."),
                    (12, UPCOMING_DIR / "standings.json", "Generating standings..."),
                    (14, UPCOMING_DIR / "extended_markets.json", "Extended markets..."),
                    (18, UPCOMING_DIR / "over_under_bets.json", "Over/Under model..."),
                    (22, BETTING_DIR / "unified_report.json", "Betting engine..."),
                    (27, DATA_DIR / "performance_dashboard.json", "Dashboard..."),
                ]
                for step_num, fpath, msg in reversed(check_files):
                    if fpath.exists():
                        mtime = datetime.fromtimestamp(fpath.stat().st_mtime)
                        if mtime > start_dt:
                            progress["step"] = step_num
                            progress["message"] = msg
                            break
            except Exception as e:
                log.debug(f"Failed to check pipeline progress file timestamps: {e}")

    return jsonify(progress)


# ---------------------------------------------------------------------------
# API: Snapshot-only pipeline (odds refresh, no full pipeline)
# ---------------------------------------------------------------------------

_snapshot_running = False

@app.route("/api/refresh/snapshot", methods=["POST"])
def api_refresh_snapshot():
    """Trigger a snapshot-only refresh: bulk odds + movement analysis."""
    global _snapshot_running
    if _snapshot_running:
        return jsonify({"ok": False, "message": "Snapshot already running"})

    def _run():
        global _snapshot_running
        _snapshot_running = True
        try:
            from scripts.data.odds_tracker import run_single_snapshot
            result = run_single_snapshot()
            log.info(f"Snapshot complete: {result}")

            # Notify odds snapshot complete
            try:
                from scripts.pipeline.notify import notify_odds_snapshot
                notify_odds_snapshot(
                    n_matches=result.get("matches", 0) if isinstance(result, dict) else 0,
                    n_bookmakers=result.get("bookmakers", 0) if isinstance(result, dict) else 0,
                )
            except Exception:
                pass
        except Exception as e:
            log.error(f"Snapshot failed: {e}")
        finally:
            _snapshot_running = False

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True, "message": "Snapshot started"})


@app.route("/api/refresh/snapshot/status")
def api_refresh_snapshot_status():
    return jsonify({"running": _snapshot_running})


# ---------------------------------------------------------------------------
# API: Extra markets (per-event) refresh
# ---------------------------------------------------------------------------

_extra_markets_running = False

@app.route("/api/refresh/extra-markets", methods=["POST"])
def api_refresh_extra_markets():
    """Trigger per-event extra markets fetch (btts, dnb, double chance, etc.)."""
    global _extra_markets_running
    if _extra_markets_running:
        return jsonify({"ok": False, "message": "Extra markets already running"})

    def _run():
        global _extra_markets_running
        _extra_markets_running = True
        try:
            from scripts.data.odds_fetcher import fetch_extra_markets_per_event, process_extra_markets, save_extra_markets
            raw = fetch_extra_markets_per_event(use_cache=False)
            if raw:
                processed = process_extra_markets(raw)
                save_extra_markets(processed)
                log.info(f"Extra markets complete: {len(processed)} matches")
        except Exception as e:
            log.error(f"Extra markets failed: {e}")
        finally:
            _extra_markets_running = False

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True, "message": "Extra markets fetch started"})


# ---------------------------------------------------------------------------
# API: SofaScore scrape
# ---------------------------------------------------------------------------

_sofascore_running = False
_sofascore_status = {"message": "", "started_at": ""}

@app.route("/api/sofascore/scrape", methods=["POST"])
def api_sofascore_scrape():
    """Start SofaScore scrape in a subprocess (needs its own event loop for Playwright)."""
    global _sofascore_running, _sofascore_status
    if _sofascore_running:
        return jsonify({"ok": False, "message": "SofaScore scrape already running"})

    def _run():
        global _sofascore_running, _sofascore_status
        _sofascore_running = True
        _sofascore_status = {"message": "Running...", "started_at": datetime.now().isoformat()}
        try:
            import subprocess
            result = subprocess.run(
                ["python3", "-m", "scripts.data.scrape_sofascore", "--season", get_current_season()],
                capture_output=True, text=True, timeout=900,
                cwd=str(Path(__file__).parent.parent),
            )
            if result.returncode == 0:
                _sofascore_status["message"] = "Complete"
            else:
                _sofascore_status["message"] = f"Error: {result.stderr[-200:]}" if result.stderr else "Unknown error"
        except subprocess.TimeoutExpired:
            _sofascore_status["message"] = "Timeout (15 min)"
        except Exception as e:
            _sofascore_status["message"] = f"Failed: {e}"
        finally:
            _sofascore_running = False

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True, "message": "SofaScore scrape started"})


@app.route("/api/sofascore/status")
def api_sofascore_status():
    """Get SofaScore scrape status and data stats."""
    parquet_path = DATA_DIR / "external" / "sofascore" / "player_match_stats.parquet"
    stats = {}
    if parquet_path.exists():
        mtime = datetime.fromtimestamp(parquet_path.stat().st_mtime)
        age_h = (datetime.now() - mtime).total_seconds() / 3600
        stats = {
            "exists": True,
            "last_modified": mtime.isoformat(),
            "age_hours": round(age_h, 1),
            "size_mb": round(parquet_path.stat().st_size / 1048576, 1),
        }
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            stats["total_records"] = len(df)
            stats["total_matches"] = df["match_id"].nunique() if "match_id" in df.columns else 0
            stats["seasons"] = sorted(df["season"].unique().tolist()) if "season" in df.columns else []
        except Exception as e:
            log.warning(f"Failed to read Sofascore parquet stats: {e}")
    else:
        stats = {"exists": False}

    return jsonify({
        "running": _sofascore_running,
        "status": _sofascore_status,
        "data": stats,
    })


# ---------------------------------------------------------------------------
# API: Understat PPDA scrape
# ---------------------------------------------------------------------------

_understat_running = False
_understat_status = {"message": "", "started_at": ""}


@app.route("/api/understat/scrape", methods=["POST"])
def api_understat_scrape():
    """Start Understat PPDA scrape in background."""
    global _understat_running, _understat_status
    if _understat_running:
        return jsonify({"ok": False, "message": "Understat scrape already running"})

    def _run():
        global _understat_running, _understat_status
        _understat_running = True
        _understat_status = {"message": "Running...", "started_at": datetime.now().isoformat()}
        try:
            import subprocess
            result = subprocess.run(
                ["python3", "-c",
                 "from scraper.understat_scraper import scrape_understat_xg; "
                 "from config.settings import DATA_DIR; "
                 "scrape_understat_xg(seasons=['2025-2026'], "
                 "output_path=DATA_DIR / 'external' / 'understat' / 'matches_xg.parquet')"],
                capture_output=True, text=True, timeout=300,
                cwd=str(Path(__file__).parent.parent),
            )
            if result.returncode == 0:
                _understat_status["message"] = "Complete"
                log.info("Understat scrape complete: %s", result.stdout[-200:] if result.stdout else "ok")
            else:
                err = result.stderr[-300:] if result.stderr else "Unknown error"
                _understat_status["message"] = f"Error: {err}"
                log.warning("Understat scrape failed: %s", err)
        except subprocess.TimeoutExpired:
            _understat_status["message"] = "Timeout (5 min)"
        except Exception as e:
            _understat_status["message"] = f"Failed: {e}"
        finally:
            _understat_running = False

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True, "message": "Understat PPDA scrape started"})


@app.route("/api/understat/status")
def api_understat_status():
    """Get Understat scrape status."""
    understat_dir = DATA_DIR / "external" / "understat"
    stats = {}
    json_files = sorted(understat_dir.glob("understat_*.json")) if understat_dir.exists() else []
    if json_files:
        latest = json_files[-1]
        mtime = datetime.fromtimestamp(latest.stat().st_mtime)
        age_h = (datetime.now() - mtime).total_seconds() / 3600
        stats = {
            "exists": True,
            "last_modified": mtime.isoformat(),
            "age_hours": round(age_h, 1),
            "seasons": len(json_files),
            "latest_file": latest.name,
        }
    else:
        stats = {"exists": False}

    return jsonify({
        "running": _understat_running,
        "status": _understat_status,
        "data": stats,
    })


# ---------------------------------------------------------------------------
# API: Credits usage
# ---------------------------------------------------------------------------

@app.route("/api/credits")
def api_credits():
    """Get API credit usage summary."""
    try:
        from scripts.data.odds_fetcher import get_usage_summary, _load_usage
        summary = get_usage_summary()
        usage = _load_usage()

        # Build per-day breakdown for the current month
        month = datetime.now().strftime("%Y-%m")
        daily = {k: v for k, v in usage.get("daily_calls", {}).items() if k.startswith(month)}

        return jsonify({
            "today": summary["daily_used"],
            "daily_limit": summary["daily_limit"],
            "monthly": summary["monthly_used"],
            "monthly_limit": 100000,
            "api_remaining": summary.get("api_remaining"),
            "last_call": summary.get("last_call"),
            "daily_breakdown": daily,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Auto-Scheduler
# ---------------------------------------------------------------------------

_scheduler_active = False
_scheduler_thread = None
_SCHEDULER_CONFIG_PATH = DATA_DIR / "scheduler_config.json"
_scheduler_config = {
    "snapshot_interval_min": 15,
    "extra_markets_times": ["08:00", "16:00"],
    "full_pipeline_times": ["07:00", "19:00"],
    "results_interval_min": 30,
    "matchday_live_poll_min": 2,
    "auto_live_on_matchday": True,
}
_scheduler_log = []  # last 50 events


def _load_scheduler_config():
    """Load scheduler config from disk (survives app restarts)."""
    global _scheduler_config
    if _SCHEDULER_CONFIG_PATH.exists():
        try:
            with open(_SCHEDULER_CONFIG_PATH) as f:
                saved = json.load(f)
            _scheduler_config.update(saved.get("config", saved))
            # Restore log
            for entry in saved.get("log", [])[-30:]:
                _scheduler_log.append(entry)
            log.info("Loaded scheduler config from disk")
        except Exception as e:
            log.warning("Failed to load scheduler config: %s", e)


def _save_scheduler_config():
    """Persist scheduler config + recent log to disk."""
    try:
        _SCHEDULER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_SCHEDULER_CONFIG_PATH, "w") as f:
            json.dump({
                "config": _scheduler_config,
                "log": _scheduler_log[-30:],
                "saved_at": datetime.now().isoformat(),
            }, f, indent=2)
    except Exception as e:
        log.warning("Failed to save scheduler config: %s", e)


# Load config on startup
_load_scheduler_config()


def _scheduler_add_log(action: str, detail: str = ""):
    _scheduler_log.append({
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "detail": detail,
    })
    if len(_scheduler_log) > 50:
        _scheduler_log.pop(0)
    _save_scheduler_config()  # persist after every log entry


def _scheduler_loop():
    """Background scheduler: runs actions at configured intervals/times."""
    global _scheduler_active
    _scheduler_active = True

    last_snapshot = 0.0
    last_results = 0.0
    fired_today = set()  # track which timed actions fired today
    current_day = ""

    log.info("Scheduler started")
    _scheduler_add_log("started", "Scheduler thread started")

    while _scheduler_active:
        try:
            now = datetime.now()
            now_ts = _time.time()
            today_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M")

            # Reset fired set at midnight
            if today_str != current_day:
                fired_today = set()
                current_day = today_str

            cfg = _scheduler_config

            # 1. Snapshot at interval
            snapshot_interval_s = cfg["snapshot_interval_min"] * 60
            if now_ts - last_snapshot >= snapshot_interval_s:
                _scheduler_add_log("snapshot", f"Running odds snapshot")
                try:
                    from scripts.data.odds_tracker import run_single_snapshot
                    result = run_single_snapshot()
                    _scheduler_add_log("snapshot_done", f"{result.get('matches', 0)} matches, {result.get('steam_moves', 0)} steam")

                    # Notify odds snapshot
                    try:
                        from scripts.pipeline.notify import notify_odds_snapshot
                        notify_odds_snapshot(
                            n_matches=result.get("matches", 0) if isinstance(result, dict) else 0,
                            n_bookmakers=result.get("bookmakers", 0) if isinstance(result, dict) else 0,
                        )
                    except Exception:
                        pass
                except Exception as e:
                    _scheduler_add_log("snapshot_error", str(e))
                last_snapshot = now_ts

            # 2. Extra markets at scheduled times
            for em_time in cfg.get("extra_markets_times", []):
                action_key = f"extra_{em_time}"
                if time_str == em_time and action_key not in fired_today:
                    fired_today.add(action_key)
                    _scheduler_add_log("extra_markets", f"Fetching extra markets ({em_time})")
                    try:
                        from scripts.data.odds_fetcher import fetch_extra_markets_per_event, process_extra_markets, save_extra_markets
                        raw = fetch_extra_markets_per_event(use_cache=False)
                        if raw:
                            processed = process_extra_markets(raw)
                            save_extra_markets(processed)
                            _scheduler_add_log("extra_markets_done", f"{len(processed)} matches")
                    except Exception as e:
                        _scheduler_add_log("extra_markets_error", str(e))

            # 3. Full pipeline at scheduled times
            for fp_time in cfg.get("full_pipeline_times", []):
                action_key = f"pipeline_{fp_time}"
                if time_str == fp_time and action_key not in fired_today:
                    fired_today.add(action_key)
                    if not _pipeline_running:
                        _scheduler_add_log("pipeline", f"Running full pipeline ({fp_time})")
                        threading.Thread(target=_run_pipeline_background, daemon=True).start()
                    else:
                        _scheduler_add_log("pipeline_skip", "Already running")

            # 4. Results check at interval — also update standings + features
            results_interval_s = cfg.get("results_interval_min", 30) * 60
            if now_ts - last_results >= results_interval_s:
                try:
                    from scripts.data.results_fetcher import fetch_and_settle
                    settle_result = fetch_and_settle()
                    settled_count = settle_result.get("settled", 0) if isinstance(settle_result, dict) else 0
                    _scheduler_add_log("results", f"Results checked, {settled_count} settled")

                    # If new results were settled, ingest match data and regenerate standings
                    if settled_count > 0:
                        try:
                            from scripts.data.matchday_updater import run_matchday_update
                            update_result = run_matchday_update(
                                rebuild_features=True,
                                regenerate_standings=True,
                            )
                            fetched = update_result.get("matches_fetched", 0)
                            _scheduler_add_log("matchday_update", f"Ingested {fetched} matches, standings rebuilt")
                        except Exception as e:
                            _scheduler_add_log("matchday_update_warn", str(e))
                except Exception as e:
                    _scheduler_add_log("results_error", str(e))
                last_results = now_ts

            # 5. Auto-start live polling on matchday
            if cfg.get("auto_live_on_matchday") and not _auto_poll_active:
                try:
                    manual_path = DATA_DIR / "upcoming" / "manual_matches.json"
                    if manual_path.exists():
                        with open(manual_path) as f:
                            manual = json.load(f)
                        for m in manual.get("matches", []):
                            if m.get("date") == today_str:
                                # Match today — start auto poll
                                global _auto_poll_thread, _auto_poll_interval
                                _auto_poll_interval = cfg.get("matchday_live_poll_min", 2) * 60
                                _auto_poll_thread = threading.Thread(target=_auto_poll_loop, daemon=True)
                                _auto_poll_thread.start()
                                _scheduler_add_log("auto_live", f"Started live polling ({cfg.get('matchday_live_poll_min', 2)} min)")
                                break
                except Exception as e:
                    log.debug(f"Failed to check match schedule for live polling: {e}")

        except Exception as e:
            _scheduler_add_log("error", str(e))

        # Check every 60 seconds
        for _ in range(12):
            if not _scheduler_active:
                break
            _time.sleep(5)

    _scheduler_active = False
    _scheduler_add_log("stopped", "Scheduler thread stopped")
    log.info("Scheduler stopped")


@app.route("/api/scheduler/toggle", methods=["POST"])
def api_scheduler_toggle():
    """Start or stop the auto-scheduler."""
    global _scheduler_active, _scheduler_thread
    data = flask_request.get_json(silent=True) or {}
    action = data.get("action", "toggle")

    if action == "stop" or (_scheduler_active and action == "toggle"):
        _scheduler_active = False
        return jsonify({"ok": True, "active": False})

    if _scheduler_active and _scheduler_thread and _scheduler_thread.is_alive():
        return jsonify({"ok": True, "active": True, "message": "already running"})

    _scheduler_thread = threading.Thread(target=_scheduler_loop, daemon=True)
    _scheduler_thread.start()
    return jsonify({"ok": True, "active": True})


@app.route("/api/scheduler/status")
def api_scheduler_status():
    """Get scheduler status, config, launchd jobs, upcoming match windows, and log."""
    now = datetime.now()
    time_str = now.strftime("%H:%M")
    today_str = now.strftime("%Y-%m-%d")
    cfg = _scheduler_config

    # ── Next scheduled runs (from in-app scheduler) ──
    next_runs = {}
    for fp_time in cfg.get("full_pipeline_times", []):
        if fp_time > time_str:
            next_runs["full_pipeline"] = f"Today {fp_time}"
            break
    if "full_pipeline" not in next_runs and cfg.get("full_pipeline_times"):
        next_runs["full_pipeline"] = f"Tomorrow {cfg['full_pipeline_times'][0]}"

    for em_time in cfg.get("extra_markets_times", []):
        if em_time > time_str:
            next_runs["extra_markets"] = f"Today {em_time}"
            break
    if "extra_markets" not in next_runs and cfg.get("extra_markets_times"):
        next_runs["extra_markets"] = f"Tomorrow {cfg['extra_markets_times'][0]}"

    # ── Real launchd jobs status ──
    launchd_jobs = []
    try:
        import subprocess as _sub
        # Get all loaded seriea jobs in one call
        ls_result = _sub.run(["launchctl", "list"],
                             capture_output=True, text=True, timeout=5)
        loaded_labels = set()
        for line in ls_result.stdout.splitlines():
            if "com.seriea-pipeline." in line:
                parts = line.split()
                if len(parts) >= 3:
                    loaded_labels.add(parts[-1])

        plist_dir = Path.home() / "Library" / "LaunchAgents"
        for plist in sorted(plist_dir.glob("com.seriea-pipeline.*.plist")):
            label = plist.stem
            short_name = label.replace("com.seriea-pipeline.", "")
            loaded = label in loaded_labels

            # Get last run from log
            log_path = _BASE / "logs" / f"launchd-{short_name}.log"
            last_run = ""
            log_size = 0
            if log_path.exists():
                log_size = log_path.stat().st_size
                mtime = log_path.stat().st_mtime
                last_run = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")

            launchd_jobs.append({
                "name": short_name,
                "label": label,
                "loaded": loaded,
                "last_run": last_run,
                "log_size": log_size,
            })
    except Exception as e:
        log.warning("Failed to check launchd jobs: %s", e)

    # ── Upcoming match windows (T-60, T-30, T+120) ──
    match_windows = []
    try:
        manual_path = DATA_DIR / "upcoming" / "manual_matches.json"
        preds_path = DATA_DIR / "upcoming" / "predictions.json"
        matches_list = []

        if manual_path.exists():
            with open(manual_path) as f:
                manual = json.load(f)
            matches_list = manual.get("matches", [])
        elif preds_path.exists():
            with open(preds_path) as f:
                preds = json.load(f)
            matches_list = preds.get("predictions", [])

        for m in matches_list:
            match_date = m.get("date", "")
            match_time = m.get("time", "")
            if not match_date or not match_time:
                continue
            # Only show next 7 days
            try:
                from datetime import timedelta
                kickoff = datetime.strptime(f"{match_date} {match_time}", "%Y-%m-%d %H:%M")
                if kickoff < now - timedelta(hours=3) or kickoff > now + timedelta(days=7):
                    continue
                t60 = kickoff - timedelta(minutes=60)
                t30 = kickoff - timedelta(minutes=30)
                t120 = kickoff + timedelta(minutes=120)

                match_windows.append({
                    "match": f"{m.get('home_team', '?')} vs {m.get('away_team', '?')}",
                    "date": match_date,
                    "kickoff": match_time,
                    "t60_lineups": t60.strftime("%H:%M"),
                    "t30_predict": t30.strftime("%H:%M"),
                    "t120_settle": t120.strftime("%H:%M"),
                    "status": "live" if now >= kickoff and now < t120 else
                              "upcoming" if now < kickoff else "settling",
                })
            except Exception:
                continue

        match_windows.sort(key=lambda x: (x["date"], x["kickoff"]))
    except Exception as e:
        log.debug("Failed to build match windows: %s", e)

    return jsonify({
        "active": _scheduler_active,
        "config": _scheduler_config,
        "next_runs": next_runs,
        "launchd_jobs": launchd_jobs,
        "match_windows": match_windows[:20],
        "log": _scheduler_log[-20:],
        "pipeline_running": _pipeline_running,
        "snapshot_running": _snapshot_running,
        "auto_poll_active": _auto_poll_active,
        "auto_poll_interval": _auto_poll_interval,
    })


@app.route("/api/scheduler/config", methods=["POST"])
def api_scheduler_config():
    """Update scheduler configuration."""
    data = flask_request.get_json(silent=True) or {}
    cfg = _scheduler_config

    if "snapshot_interval_min" in data:
        cfg["snapshot_interval_min"] = max(5, min(120, int(data["snapshot_interval_min"])))
    if "extra_markets_times" in data:
        cfg["extra_markets_times"] = data["extra_markets_times"]
    if "full_pipeline_times" in data:
        cfg["full_pipeline_times"] = data["full_pipeline_times"]
    if "results_interval_min" in data:
        cfg["results_interval_min"] = max(10, min(120, int(data["results_interval_min"])))
    if "matchday_live_poll_min" in data:
        cfg["matchday_live_poll_min"] = max(1, min(30, int(data["matchday_live_poll_min"])))
    if "auto_live_on_matchday" in data:
        cfg["auto_live_on_matchday"] = bool(data["auto_live_on_matchday"])

    _scheduler_add_log("config_update", json.dumps(data))
    _save_scheduler_config()
    return jsonify({"ok": True, "config": cfg})


# ---------------------------------------------------------------------------
# API: Smart Incremental Refresh
# ---------------------------------------------------------------------------

_smart_refresh_running = False
_smart_refresh_result = {}
_smart_refresh_progress = {"step": 0, "total": 6, "message": "Idle", "steps_done": []}
_settle_running = False
_settle_result = {}
_settle_progress = {"step": 0, "total": 3, "message": "Idle", "steps_done": []}

@app.route("/api/refresh/smart", methods=["POST"])
def api_refresh_smart():
    """Trigger an incremental pipeline refresh in background.

    Fetches new results, refreshes odds if stale (>4h), runs market analysis
    chain (bookmaker, cross-market, intelligence), refreshes player/lineup data
    if stale, and predicts unseen fixtures. ~4 API credits + local analysis.
    """
    global _smart_refresh_running, _smart_refresh_result
    if _smart_refresh_running:
        return jsonify({"ok": False, "message": "Smart refresh already running"})
    if _pipeline_running:
        return jsonify({"ok": False, "message": "Full pipeline already running"})

    _smart_refresh_result = {
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "message": "Starting incremental refresh...",
    }
    _smart_refresh_progress["step"] = 0
    _smart_refresh_progress["message"] = "Starting..."
    _smart_refresh_progress["steps_done"] = []

    def _run():
        global _smart_refresh_running, _smart_refresh_result, _smart_refresh_progress
        _smart_refresh_running = True
        try:
            # Step-by-step progress tracking
            def _step(n, total, msg):
                _smart_refresh_progress["step"] = n
                _smart_refresh_progress["total"] = total
                _smart_refresh_progress["message"] = msg
                _smart_refresh_progress["steps_done"].append(msg)

            _step(1, 6, "Fetching results & settling bets...")
            from scripts.pipeline.run_full_pipeline import run_incremental
            # We hook into the summary dict to track progress
            import scripts.pipeline.run_full_pipeline as _pipeline_mod
            _orig_print = _pipeline_mod.print if hasattr(_pipeline_mod, 'print') else print

            # Override print to capture step messages
            import builtins
            _real_print = builtins.print
            def _progress_print(*args, **kwargs):
                msg = " ".join(str(a) for a in args)
                # Detect step markers like [1/6], [2/6] etc
                import re
                m = re.match(r'\[(\d+)/(\d+)\]\s*(.*)', msg.strip())
                if m:
                    step_n, step_total, step_msg = int(m.group(1)), int(m.group(2)), m.group(3)
                    _smart_refresh_progress["step"] = step_n
                    _smart_refresh_progress["total"] = step_total
                    _smart_refresh_progress["message"] = step_msg.strip()
                    _smart_refresh_progress["steps_done"].append(step_msg.strip())
                _real_print(*args, **kwargs)

            builtins.print = _progress_print
            try:
                result = run_incremental()
            finally:
                builtins.print = _real_print

            _smart_refresh_result = result
            _smart_refresh_result["finished_at"] = datetime.now().isoformat()
            _smart_refresh_progress["message"] = "Complete"

            # Notify if new value bets were found (coaching style)
            try:
                from scripts.pipeline.notify import notify_value_bets
                # Load the full bet slip for rich narratives
                slip_path = DATA_DIR / "upcoming" / "unified_bet_slip.json"
                if slip_path.exists():
                    slip_data = _load_json(slip_path)
                    vb_list = slip_data.get("selected_bets", [])
                    if vb_list:
                        notify_value_bets(vb_list)
            except Exception as e:
                log.debug(f"Value bet notification failed: {e}")
        except Exception as e:
            log.error(f"Smart refresh failed: {e}")
            _smart_refresh_result = {
                "status": "error",
                "error": str(e),
                "finished_at": datetime.now().isoformat(),
            }
            _smart_refresh_progress["message"] = f"Error: {e}"
        finally:
            _smart_refresh_running = False

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True, "message": "Smart refresh started"})


@app.route("/api/refresh/smart/status")
def api_refresh_smart_status():
    """Check smart refresh progress and results."""
    result = dict(_smart_refresh_result)
    result["running"] = _smart_refresh_running
    result["progress"] = dict(_smart_refresh_progress)

    # Also include pipeline state summary
    try:
        from scripts.pipeline.pipeline_state import load_state, get_state_summary
        state = load_state()
        result["pipeline_state"] = get_state_summary(state)
    except Exception:
        pass

    return jsonify(result)


# ---------------------------------------------------------------------------
# API: Auto-Settle (triggered when matches complete)
# ---------------------------------------------------------------------------

@app.route("/api/settle", methods=["POST"])
def api_settle():
    """Fetch latest results and settle completed bets.

    Lightweight alternative to smart-refresh: only fetches scores and settles
    bets, doesn't re-run predictions or refresh odds. Costs 2 API credits.
    Designed to be auto-triggered by the frontend when a match completes.
    """
    global _settle_running, _settle_result
    if _settle_running:
        return jsonify({"ok": False, "message": "Settlement already running"})
    if _smart_refresh_running:
        return jsonify({"ok": False, "message": "Smart refresh running, settlement included"})

    _settle_result = {
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "message": "Fetching results and settling bets...",
    }
    _settle_progress["step"] = 0
    _settle_progress["message"] = "Starting..."
    _settle_progress["steps_done"] = []

    def _run_settle():
        global _settle_running, _settle_result, _settle_progress
        _settle_running = True
        try:
            _settle_progress.update({"step": 1, "total": 4, "message": "Fetching scores from Odds API..."})
            _settle_progress["steps_done"].append("Fetching scores...")
            from scripts.data.results_fetcher import fetch_and_settle
            summary = fetch_and_settle()
            _settle_progress.update({"step": 2, "message": f"Settled {summary.get('settled', 0)} bets"})
            _settle_progress["steps_done"].append(f"Settled {summary.get('settled', 0)} bets")

            # Also settle player props
            _settle_progress.update({"step": 3, "message": "Settling player props..."})
            _settle_progress["steps_done"].append("Settling player props...")
            prop_summary = {}
            try:
                from scripts.betting.prop_tracker import settle_props
                prop_summary = settle_props()
                if prop_summary.get("total_settled", 0) > 0:
                    log.info("Settled %d player props (%.1f%% hit rate, %.1f%% ROI)",
                             prop_summary["total_settled"],
                             prop_summary.get("hit_rate", 0),
                             prop_summary.get("roi_pct", 0))
            except Exception as e:
                log.warning(f"Prop settlement failed: {e}")
                prop_summary = {"error": str(e)}

            # Persist substitution data for future predictions
            try:
                from scripts.data.substitution_tracker import extract_substitutions
                sub_result = extract_substitutions()
                if sub_result.get("extracted", 0) > 0:
                    log.info("Persisted %d substitution records", sub_result["extracted"])
            except Exception as e:
                log.warning(f"Substitution extraction failed: {e}")

            # Settle fair odds predictions
            try:
                from scripts.betting.fair_odds_tracker import settle_predictions
                fo_result = settle_predictions()
                if fo_result.get("settled", 0) > 0:
                    log.info("Settled %d fair odds records", fo_result["settled"])
            except Exception as e:
                log.warning(f"Fair odds settlement failed: {e}")

            # Mark bets as settled (advisor can pick this up immediately)
            _settle_result = {
                "status": "done",
                **summary,
                "prop_settlement": prop_summary,
                "finished_at": datetime.now().isoformat(),
                "matchday_update": {"status": "starting"},
            }

            # Ingest new match data and regenerate standings + features
            _settle_progress.update({"step": 4, "message": "Rebuilding standings & features..."})
            _settle_progress["steps_done"].append("Rebuilding standings & features...")
            matchday_summary = {}
            try:
                from scripts.data.matchday_updater import run_matchday_update
                matchday_summary = run_matchday_update(
                    rebuild_features=True,
                    regenerate_standings=True,
                )
                fetched = matchday_summary.get("matches_fetched", 0)
                if fetched:
                    log.info("Post-settle: ingested %d matches, standings + features rebuilt", fetched)
            except Exception as e:
                log.warning(f"Post-settle matchday update failed: {e}")
                matchday_summary = {"error": str(e)}

            _settle_result = {
                "status": "done",
                **summary,
                "prop_settlement": prop_summary,
                "matchday_update": matchday_summary,
                "finished_at": datetime.now().isoformat(),
            }

            # Send coaching-style settlement notification
            try:
                from scripts.pipeline.notify import notify_settlement
                n_settled = summary.get("settled", 0)
                won = summary.get("won", 0)
                lost = summary.get("lost", 0)
                push = summary.get("push", 0)
                profit = summary.get("profit", summary.get("net_profit", 0)) or 0
                balance = summary.get("balance", summary.get("bankroll", 0)) or 0
                if n_settled > 0:
                    notify_settlement(
                        settled=n_settled, won=won, lost=lost, push=push,
                        profit=profit, balance=balance,
                    )
            except Exception as e:
                log.debug(f"Settlement notification failed: {e}")

            # Bankroll milestone and drawdown notifications
            try:
                from scripts.pipeline.notify import notify_bankroll_milestone, notify_drawdown
                old_balance = summary.get("old_balance", summary.get("previous_balance", 0)) or 0
                new_balance = summary.get("balance", summary.get("bankroll", 0)) or 0
                if old_balance and new_balance:
                    notify_bankroll_milestone(old_balance, new_balance)

                # Drawdown check: if drawdown from peak > 15%
                peak = summary.get("peak_balance", 0) or 0
                if not peak:
                    # Try to load peak from bankroll files
                    try:
                        _br = _load_json(BETTING_DIR / "bankroll.json")
                        _bs = _load_json(BANKROLL_DIR / "state.json")
                        peak = max(
                            _br.get("peak_balance", 0),
                            _bs.get("peak_bankroll", 0),
                            new_balance,
                        )
                    except Exception:
                        peak = new_balance
                if peak > 0 and new_balance > 0:
                    dd_pct = (peak - new_balance) / peak * 100
                    if dd_pct > 15:
                        notify_drawdown(new_balance, peak, dd_pct)
            except Exception as e:
                log.debug(f"Bankroll milestone/drawdown notification failed: {e}")
        except Exception as e:
            log.error(f"Auto-settle failed: {e}")
            _settle_result = {
                "status": "error",
                "error": str(e),
                "finished_at": datetime.now().isoformat(),
            }
        finally:
            _settle_running = False

    threading.Thread(target=_run_settle, daemon=True).start()
    return jsonify({"ok": True, "message": "Settlement started"})


@app.route("/api/settle/status")
def api_settle_status():
    """Check settlement progress and results."""
    result = dict(_settle_result)
    result["running"] = _settle_running
    result["progress"] = dict(_settle_progress)
    return jsonify(result)


@app.route("/api/props/performance")
def api_props_performance():
    """Return prop betting performance stats (hit rates, ROI by market/tier/edge)."""
    perf_path = DATA_DIR / "betting" / "prop_performance.json"
    ledger_path = DATA_DIR / "betting" / "prop_ledger.json"

    perf = _load_json(perf_path, default={})
    ledger = _load_json(ledger_path, default=[])

    # Add recent entries for display
    recent = sorted(ledger, key=lambda e: e.get("settled_at", ""), reverse=True)[:50]

    return jsonify({
        "performance": perf,
        "recent_entries": recent,
        "total_entries": len(ledger),
    })


@app.route("/api/fair-odds/summary")
def api_fair_odds_summary():
    """Fair odds historical tracking summary — model accuracy over time."""
    try:
        from scripts.betting.fair_odds_tracker import get_summary
        summary = get_summary()
        return jsonify(summary)
    except Exception as e:
        log.warning(f"Fair odds summary failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/props/kelly-pnl")
def api_props_kelly_pnl():
    """Hypothetical Kelly P&L: what if we'd bet quarter-Kelly on all value props."""
    try:
        from scripts.betting.prop_tracker import get_hypothetical_kelly_pnl
        bankroll = float(flask_request.args.get("bankroll", 1000))
        result = get_hypothetical_kelly_pnl(bankroll=bankroll)
        return jsonify(result)
    except Exception as e:
        log.warning(f"Kelly P&L calculation failed: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Background Auto-Settlement Scheduler
# ---------------------------------------------------------------------------
# Runs every 5 minutes while the Flask app is running.  Checks if there are
# unsettled bets whose match dates have passed, and if so fetches results
# from the Odds API and settles them.  Costs 2 API credits per check
# (only when there are bets to settle).
# ---------------------------------------------------------------------------

_auto_settle_thread = None
_auto_settle_active = False
_auto_settle_last_run = ""
_auto_settle_last_result = {}

SETTLE_CHECK_INTERVAL = 300   # 5 minutes between checks
SETTLE_QUIET_HOURS = (2, 10)  # Don't check between 2 AM and 10 AM (no matches)


def _auto_settle_loop():
    """Background loop: check for unsettled bets and settle when matches complete."""
    global _auto_settle_active, _auto_settle_last_run, _auto_settle_last_result
    import time as _time
    _auto_settle_active = True
    log.info("Auto-settle scheduler started (every %ds)", SETTLE_CHECK_INTERVAL)

    while _auto_settle_active:
        try:
            _time.sleep(SETTLE_CHECK_INTERVAL)

            # Skip quiet hours (no matches finishing at 3 AM)
            hour = datetime.now().hour
            if SETTLE_QUIET_HOURS[0] <= hour < SETTLE_QUIET_HOURS[1]:
                continue

            # Skip if another settlement or refresh is already running
            if _settle_running or _smart_refresh_running or _pipeline_running:
                continue

            # Check if there are unsettled bets with past match dates
            has_unsettled = _has_unsettled_past_bets()
            if not has_unsettled:
                continue

            # Run full settlement (same flow as /api/settle)
            log.info("Auto-settle: found unsettled past bets, fetching results...")
            _auto_settle_last_run = datetime.now().isoformat()
            try:
                from scripts.data.results_fetcher import fetch_and_settle
                summary = fetch_and_settle()
                settled = summary.get("settled", 0)

                # Also settle player props
                prop_summary = {}
                try:
                    from scripts.betting.prop_tracker import settle_props
                    prop_summary = settle_props()
                    if prop_summary.get("total_settled", 0) > 0:
                        log.info("Auto-settle: settled %d player props",
                                 prop_summary["total_settled"])
                except Exception as e:
                    log.warning("Auto-settle prop settlement failed: %s", e)

                # Also settle fair odds tracker
                try:
                    from scripts.betting.fair_odds_tracker import settle_predictions
                    fo_result = settle_predictions()
                    if fo_result.get("settled", 0) > 0:
                        log.info("Auto-settle: settled %d fair odds records", fo_result["settled"])
                except Exception as e:
                    log.warning("Auto-settle fair odds settlement failed: %s", e)

                # Post-settle matchday update removed — module scripts.pipeline.matchday_update
                # does not exist. Settlement proceeds without post-settle data ingest.

                _auto_settle_last_result = {
                    "settled": settled,
                    "summary": summary,
                    "prop_settlement": prop_summary,
                }

                if settled > 0:
                    log.info("Auto-settle: settled %d bets | P&L: %+.2f | Balance: %.2f",
                             settled, summary.get("profit", 0), summary.get("new_balance", 0))

                    # Settlement, bankroll milestone, and drawdown notifications
                    try:
                        from scripts.pipeline.notify import (
                            notify_settlement, notify_bankroll_milestone, notify_drawdown,
                        )
                        won = summary.get("won", 0)
                        lost = summary.get("lost", 0)
                        push = summary.get("push", 0)
                        profit = summary.get("profit", summary.get("net_profit", 0)) or 0
                        new_balance = summary.get("new_balance", summary.get("balance", 0)) or 0
                        notify_settlement(
                            settled=settled, won=won, lost=lost, push=push,
                            profit=profit, balance=new_balance,
                        )
                        old_balance = summary.get("old_balance", summary.get("previous_balance", 0)) or 0
                        if old_balance and new_balance:
                            notify_bankroll_milestone(old_balance, new_balance)
                        # Drawdown check
                        peak = summary.get("peak_balance", 0) or 0
                        if not peak:
                            try:
                                _br = _load_json(BETTING_DIR / "bankroll.json")
                                _bs = _load_json(BANKROLL_DIR / "state.json")
                                peak = max(_br.get("peak_balance", 0), _bs.get("peak_bankroll", 0), new_balance)
                            except Exception:
                                peak = new_balance
                        if peak > 0 and new_balance > 0:
                            dd_pct = (peak - new_balance) / peak * 100
                            if dd_pct > 15:
                                notify_drawdown(new_balance, peak, dd_pct)
                    except Exception as e:
                        log.debug("Auto-settle notification failed: %s", e)
                else:
                    log.debug("Auto-settle: no new results to settle")
            except Exception as e:
                log.error("Auto-settle failed: %s", e)
                _auto_settle_last_result = {"error": str(e)}

        except Exception as e:
            log.error("Auto-settle loop error: %s", e)

    log.info("Auto-settle scheduler stopped")


def _has_unsettled_past_bets() -> bool:
    """Check if there are pending/superseded bets whose match date has passed or is today.

    Same-day matches can finish within ~2 hours, so we check today's matches
    too (the settle flow will skip any that haven't completed yet).
    """
    try:
        journal_path = BETTING_DIR / "bet_journal.json"
        if not journal_path.exists():
            return False
        with open(journal_path) as f:
            journal = json.load(f)
        bets = journal.get("bets", {})
        today = datetime.now().strftime("%Y-%m-%d")
        for bet in bets.values():
            status = bet.get("status", "")
            if status not in ("pending", "superseded"):
                continue
            match_date = bet.get("date", "9999-99-99")
            if match_date <= today:  # <= includes same-day matches
                return True
        return False
    except Exception:
        return False


def start_auto_settle():
    """Start the background auto-settlement thread."""
    global _auto_settle_thread, _auto_settle_active
    if _auto_settle_thread and _auto_settle_thread.is_alive():
        return  # Already running
    _auto_settle_active = True
    _auto_settle_thread = threading.Thread(target=_auto_settle_loop, daemon=True)
    _auto_settle_thread.start()


def stop_auto_settle():
    """Stop the background auto-settlement thread."""
    global _auto_settle_active
    _auto_settle_active = False


@app.route("/api/settle/scheduler")
def api_settle_scheduler():
    """Check auto-settle scheduler status."""
    return jsonify({
        "active": _auto_settle_active,
        "last_run": _auto_settle_last_run,
        "last_result": _auto_settle_last_result,
        "interval_seconds": SETTLE_CHECK_INTERVAL,
        "quiet_hours": list(SETTLE_QUIET_HOURS),
    })


# ---------------------------------------------------------------------------
# API: Value Bet Alerts
# ---------------------------------------------------------------------------
# Tracks changes to the unified bet slip and returns new/improved bets.
# Frontend polls this every 5 minutes and shows browser notifications.

_alerts_last_seen = {}  # {bet_key: {odds, edge_pct, generated_at}}


@app.route("/api/alerts/check")
def api_alerts_check():
    """Check for new or improved value bets since last check.

    Returns new bets (not seen before) and improved bets (odds/edge improved).
    Client polls this every 5 minutes.
    """
    global _alerts_last_seen

    slip = _load_json(UPCOMING_DIR / "unified_bet_slip.json", default={})
    bets = slip.get("selected_bets", [])
    generated_at = slip.get("generated_at", "")

    new_bets = []
    improved_bets = []
    current_keys = set()

    for b in bets:
        key = f"{b.get('date')}_{b.get('match')}_{b.get('market')}_{b.get('selection')}"
        current_keys.add(key)

        odds = b.get("best_odds", 0)
        edge = b.get("edge_pct", 0)

        if key not in _alerts_last_seen:
            # Brand new bet
            new_bets.append({
                "match": b.get("match"),
                "date": b.get("date"),
                "market": b.get("market"),
                "selection": b.get("selection"),
                "odds": odds,
                "edge_pct": edge,
                "bookmaker": b.get("best_bookmaker"),
                "stake": b.get("stake_amount"),
                "confidence": b.get("confidence_tier"),
            })
        else:
            prev = _alerts_last_seen[key]
            # Odds improved (higher odds = better value)
            if odds > prev.get("odds", 0) + 0.02:
                improved_bets.append({
                    "match": b.get("match"),
                    "market": b.get("market"),
                    "selection": b.get("selection"),
                    "old_odds": prev.get("odds"),
                    "new_odds": odds,
                    "edge_pct": edge,
                    "bookmaker": b.get("best_bookmaker"),
                })

        _alerts_last_seen[key] = {"odds": odds, "edge_pct": edge, "generated_at": generated_at}

    # Remove bets that are no longer in the slip (expired/removed)
    removed = [k for k in _alerts_last_seen if k not in current_keys]
    for k in removed:
        del _alerts_last_seen[k]

    has_alerts = len(new_bets) > 0 or len(improved_bets) > 0

    return jsonify({
        "has_alerts": has_alerts,
        "new_bets": new_bets,
        "improved_bets": improved_bets,
        "total_value_bets": len(bets),
        "slip_generated_at": generated_at,
        "removed_count": len(removed),
    })


# Start auto-settle on app startup (outside of debug reloader)
if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
    start_auto_settle()

    # Auto-start live poll if there are matches today
    def _maybe_start_live_poll():
        """Start live polling if matches are happening today."""
        import time as _t
        _t.sleep(10)  # Let app fully initialize
        try:
            # Check if there are matches today
            from scripts.pipeline.scheduler import is_match_day
            if is_match_day():
                global _auto_poll_active, _auto_poll_thread
                if not _auto_poll_active:
                    log.info("Match day detected — auto-starting live poll")
                    _auto_poll_thread = threading.Thread(target=_auto_poll_loop, daemon=True)
                    _auto_poll_thread.start()
        except Exception as e:
            log.debug("Auto-start live poll check failed: %s", e)

    threading.Thread(target=_maybe_start_live_poll, daemon=True).start()


# ---------------------------------------------------------------------------
# API: Team Research
# ---------------------------------------------------------------------------

@app.route("/teams")
def teams_page():
    """Render teams overview page with league table, grid, and comparison mode."""
    return render_template("teams.html", active_page="teams")


@app.route("/team/<team_name>")
def team_page(team_name):
    """Render team research page."""
    return render_template("team.html", team_name=team_name, active_page="teams")


@app.route("/api/teams")
def api_teams():
    """List all teams for search/autocomplete."""
    teams = set()

    # From standings (all leagues)
    for _sf in [UPCOMING_DIR / "standings.json", UPCOMING_DIR / "standings_premier_league.json"]:
        standings = _load_json(_sf, {})
        if isinstance(standings, list):
            for entry in standings:
                if isinstance(entry, dict) and entry.get("team"):
                    teams.add(entry["team"])
        elif isinstance(standings, dict):
            inner = standings.get("standings", standings.get("table", {}))
            if isinstance(inner, dict):
                for team_name, entry in inner.items():
                    if isinstance(entry, dict) and entry.get("team"):
                        teams.add(entry["team"])
                    else:
                        teams.add(team_name)
            elif isinstance(inner, list):
                for entry in inner:
                    if isinstance(entry, dict) and entry.get("team"):
                        teams.add(entry["team"])

    # From predictions (all leagues)
    for _pf in [UPCOMING_DIR / "predictions.json"] + list(UPCOMING_DIR.glob("predictions_*.json")):
        if not _pf.exists() or _pf.name == "predictions_archive.json":
            continue
        preds = _load_json(_pf, [])
        pred_list = preds if isinstance(preds, list) else preds.get("predictions", [])
        for p in pred_list:
            if isinstance(p, dict):
                if p.get("home_team"):
                    teams.add(p["home_team"])
                if p.get("away_team"):
                    teams.add(p["away_team"])

    # From features.parquet if available
    try:
        import pandas as pd
        fp = DATA_DIR / "features" / "features.parquet"
        if fp.exists():
            df = pd.read_parquet(fp, columns=["team"])
            teams.update(df["team"].dropna().unique().tolist())
    except Exception:
        pass

    league_filter = _get_league_filter()
    if league_filter:
        teams = {t for t in teams if _team_belongs_to_league(t, league_filter)}

    return jsonify({"teams": sorted(teams)})


@app.route("/api/teams/overview")
def api_teams_overview():
    """Full league overview with rich per-team stats for the Teams page.

    Returns all 20 teams with: standings, form, xG, Elo, home/away splits,
    over/underperformance (xPts vs actual), upcoming match, and squad strength.

    Query params:
        league: "serie_a" (default), "premier_league", etc.
    """
    import pandas as pd

    league_filter = _get_league_filter()
    if league_filter is None:
        league_filter = "serie_a"  # Default to Serie A for overview (needs standings data)

    # 1. Standings — live derivation from Sofascore parquet (single source of truth)
    standings_payload = _get_standings(league_filter)
    standings_map = {}
    standings_list = []
    inner = standings_payload.get("standings", {})
    if isinstance(inner, dict):
        for team_name, entry in inner.items():
            if isinstance(entry, dict):
                entry["team"] = team_name
                standings_list.append(entry)
    elif isinstance(inner, list):
        standings_list = inner
    current_matchweek = standings_payload.get("current_matchweek", 0)

    for entry in standings_list:
        if isinstance(entry, dict) and entry.get("team"):
            team = entry["team"]
            standings_map[team] = {
                "position": entry.get("position", entry.get("rank", 0)),
                "played": entry.get("played", entry.get("mp", 0)),
                "won": entry.get("wins", entry.get("won", entry.get("w", 0))),
                "drawn": entry.get("draws", entry.get("drawn", entry.get("d", 0))),
                "lost": entry.get("losses", entry.get("lost", entry.get("l", 0))),
                "gf": entry.get("gf", entry.get("goals_for", 0)),
                "ga": entry.get("ga", entry.get("goals_against", 0)),
                "gd": entry.get("gd", entry.get("goal_difference", 0)),
                "points": entry.get("points", entry.get("pts", 0)),
                "form_last5": entry.get("form_last5", ""),
                "home": entry.get("home", {}),
                "away": entry.get("away", {}),
            }

    # 2. Per-team stats from features.parquet (Elo, home/away records, xG rolling, xPts)
    team_stats = {}
    try:
        fp = DATA_DIR / "features" / "features.parquet"
        if fp.exists():
            desired_cols = [
                "home_team", "away_team", "season", "match_date",
                "home_score", "away_score", "home_elo", "away_elo",
                "home_ss_roll_xg", "away_ss_roll_xg",
            ]
            import pyarrow.parquet as pq
            schema_cols = [f.name for f in pq.read_schema(fp)]
            valid_cols = [c for c in desired_cols if c in schema_cols]
            df = pd.read_parquet(fp, columns=valid_cols)

            # Current season only
            if "season" in df.columns:
                current_season = df["season"].max()
                df = df[df["season"] == current_season]

            has_elo = "home_elo" in df.columns
            has_xg = "home_ss_roll_xg" in df.columns

            for team in standings_map:
                home_mask = df["home_team"].str.lower() == team.lower()
                away_mask = df["away_team"].str.lower() == team.lower()
                home_matches = df[home_mask].sort_values("match_date") if "match_date" in df.columns else df[home_mask]
                away_matches = df[away_mask].sort_values("match_date") if "match_date" in df.columns else df[away_mask]

                # Elo (latest match, whether home or away)
                elo = None
                if has_elo:
                    parts = []
                    if not home_matches.empty:
                        parts.append((home_matches.iloc[-1].get("match_date", ""), float(home_matches.iloc[-1]["home_elo"])))
                    if not away_matches.empty:
                        parts.append((away_matches.iloc[-1].get("match_date", ""), float(away_matches.iloc[-1]["away_elo"])))
                    if parts:
                        elo = max(parts, key=lambda x: str(x[0]))[1]

                # xG: season average of Sofascore rolling xG across all matches
                avg_xg = avg_xga = 0.0
                if has_xg:
                    xg_for_vals = []
                    xg_against_vals = []
                    for _, row in home_matches.iterrows():
                        v = row.get("home_ss_roll_xg")
                        if pd.notna(v): xg_for_vals.append(float(v))
                        v = row.get("away_ss_roll_xg")
                        if pd.notna(v): xg_against_vals.append(float(v))
                    for _, row in away_matches.iterrows():
                        v = row.get("away_ss_roll_xg")
                        if pd.notna(v): xg_for_vals.append(float(v))
                        v = row.get("home_ss_roll_xg")
                        if pd.notna(v): xg_against_vals.append(float(v))
                    if xg_for_vals:
                        avg_xg = sum(xg_for_vals) / len(xg_for_vals)
                    if xg_against_vals:
                        avg_xga = sum(xg_against_vals) / len(xg_against_vals)

                # Home/away records from standings.json (single source of truth)
                s_entry = standings_map.get(team, {})
                s_home = s_entry.get("home", {})
                s_away = s_entry.get("away", {})
                home_rec = {
                    "w": s_home.get("wins", 0), "d": s_home.get("draws", 0),
                    "l": s_home.get("losses", 0), "played": s_home.get("played", 0),
                    "ppg": s_home.get("ppg", 0),
                }
                away_rec = {
                    "w": s_away.get("wins", 0), "d": s_away.get("draws", 0),
                    "l": s_away.get("losses", 0), "played": s_away.get("played", 0),
                    "ppg": s_away.get("ppg", 0),
                }

                # xPts via Pythagorean expectation (GF^1.6 / (GF^1.6 + GA^1.6))
                s = standings_map.get(team, {})
                gf = max(s.get("gf", 1), 1)
                ga = max(s.get("ga", 1), 1)
                played = max(s.get("played", 1), 1)
                pyth_pct = (gf ** 1.6) / ((gf ** 1.6) + (ga ** 1.6))
                xpts = round(played * (pyth_pct * 0.75 * 3 + 0.25), 1)

                team_stats[team] = {
                    "avg_xg": round(avg_xg, 2),
                    "avg_xga": round(avg_xga, 2),
                    "xg_diff": round(avg_xg - avg_xga, 2),
                    "elo": round(elo, 0) if elo else None,
                    "home": home_rec,
                    "away": away_rec,
                    "xpts": xpts,
                }
    except Exception as e:
        log.warning("teams/overview stats error: %s", e)

    # 3. Upcoming matches per team — load from all league prediction files
    preds = _load_json(UPCOMING_DIR / "predictions.json", [])
    pred_list = preds if isinstance(preds, list) else preds.get("predictions", [])
    # Also load league-specific prediction files
    for lk in ACTIVE_LEAGUES:
        if lk == "serie_a":
            continue
        extra_path = UPCOMING_DIR / f"predictions_{lk}.json"
        extra_raw = _load_json(extra_path)
        if extra_raw:
            extra_list = extra_raw.get("predictions", []) if isinstance(extra_raw, dict) else extra_raw
            pred_list.extend(extra_list)
    ct_map = _get_commence_times()
    upcoming_by_team = {}
    for pred in pred_list:
        if not isinstance(pred, dict):
            continue
        mk = pred.get("match", "")
        ct = ct_map.get(mk, "")
        for side in ("home_team", "away_team"):
            t = pred.get(side, "")
            if t and t not in upcoming_by_team:
                upcoming_by_team[t] = {
                    "match": mk,
                    "opponent": pred.get("away_team" if side == "home_team" else "home_team", ""),
                    "is_home": side == "home_team",
                    "date": pred.get("date", ""),
                    "commence_time": ct,
                    "predicted_outcome": pred.get("predicted_outcome", ""),
                }

    # 4. Build response
    teams = []
    for team, standing in sorted(standings_map.items(), key=lambda x: x[1].get("position", 99)):
        stats = team_stats.get(team, {})
        upcoming = upcoming_by_team.get(team)
        actual_pts = standing.get("points", 0)
        xpts = stats.get("xpts", 0)
        teams.append({
            "team": team,
            "standing": standing,
            "stats": stats,
            "upcoming": upcoming,
            "pts_diff": round(actual_pts - xpts, 1) if xpts else 0,
        })

    return jsonify({
        "teams": teams,
        "season": standings_payload.get("season", ""),
        "current_matchweek": current_matchweek,
        "league": league_filter,
        "generated_at": "",
    })


@app.route("/api/teams/compare")
def api_teams_compare():
    """Compare two teams head-to-head."""
    import pandas as pd
    team_a = flask_request.args.get("a", "")
    team_b = flask_request.args.get("b", "")
    if not team_a or not team_b:
        return jsonify({"error": "Both ?a= and ?b= required"}), 400

    result = {"team_a": team_a, "team_b": team_b, "h2h": [], "prediction": None}

    # H2H from features.parquet
    try:
        fp = DATA_DIR / "features" / "features.parquet"
        if fp.exists():
            import pyarrow.parquet as pq
            schema_cols = [f.name for f in pq.read_schema(fp)]
            h2h_cols = ["home_team", "away_team", "match_date", "season",
                        "home_score", "away_score"]
            # Try Sofascore rolling xG for H2H context
            for xc in ["home_ss_roll_xg", "away_ss_roll_xg"]:
                if xc in schema_cols:
                    h2h_cols.append(xc)
            df = pd.read_parquet(fp, columns=h2h_cols)
            mask = (
                (df["home_team"].str.lower() == team_a.lower()) & (df["away_team"].str.lower() == team_b.lower())
            ) | (
                (df["home_team"].str.lower() == team_b.lower()) & (df["away_team"].str.lower() == team_a.lower())
            )
            h2h = df[mask].sort_values("match_date", ascending=False)
            for _, row in h2h.head(10).iterrows():
                result["h2h"].append({
                    "date": str(row["match_date"])[:10],
                    "season": str(row.get("season", "")),
                    "home": row["home_team"],
                    "away": row["away_team"],
                    "score": f"{int(row['home_score'])}-{int(row['away_score'])}" if pd.notna(row.get("home_score")) else None,
                    "home_xg": round(float(row["home_ss_roll_xg"]), 2) if pd.notna(row.get("home_ss_roll_xg")) else None,
                    "away_xg": round(float(row["away_ss_roll_xg"]), 2) if pd.notna(row.get("away_ss_roll_xg")) else None,
                })

            # Summary
            a_wins = sum(1 for m in result["h2h"] if m["score"] and (
                (m["home"].lower() == team_a.lower() and int(m["score"].split("-")[0]) > int(m["score"].split("-")[1])) or
                (m["away"].lower() == team_a.lower() and int(m["score"].split("-")[1]) > int(m["score"].split("-")[0]))
            ))
            draws = sum(1 for m in result["h2h"] if m["score"] and m["score"].split("-")[0] == m["score"].split("-")[1])
            result["summary"] = {
                "total": len(result["h2h"]),
                "a_wins": a_wins,
                "draws": draws,
                "b_wins": len(result["h2h"]) - a_wins - draws,
            }
    except Exception as e:
        log.warning("teams/compare h2h error: %s", e)

    # Check if they have an upcoming match against each other (all leagues)
    pred_list = []
    for _pf in [UPCOMING_DIR / "predictions.json"] + list(UPCOMING_DIR.glob("predictions_*.json")):
        if not _pf.exists() or _pf.name == "predictions_archive.json":
            continue
        _pr = _load_json(_pf, [])
        _pl = _pr if isinstance(_pr, list) else _pr.get("predictions", [])
        pred_list.extend(_pl)
    ct_map = _get_commence_times()
    for pred in pred_list:
        if not isinstance(pred, dict):
            continue
        h = pred.get("home_team", "").lower()
        a = pred.get("away_team", "").lower()
        if (h == team_a.lower() and a == team_b.lower()) or (h == team_b.lower() and a == team_a.lower()):
            mk = pred.get("match", "")
            result["prediction"] = {
                "match": mk,
                "home_team": pred.get("home_team"),
                "away_team": pred.get("away_team"),
                "predicted_outcome": pred.get("predicted_outcome"),
                "probabilities": pred.get("probabilities", {}),
                "confidence_level": pred.get("confidence_level"),
                "home_xg": pred.get("home_xg"),
                "away_xg": pred.get("away_xg"),
                "commence_time": ct_map.get(mk, ""),
            }
            break

    return jsonify(result)


@app.route("/api/team/<team_name>")
def api_team(team_name):
    """Get comprehensive team research data."""
    from urllib.parse import unquote
    team = unquote(team_name)

    data = {
        "team": team,
        "overview": {},
        "form": [],
        "match_history": [],
        "stats": {},
        "upcoming": {},
        "squad": {},
    }

    # ── 1. Standings & position ──
    # Live derivation from Sofascore parquet (single source of truth).
    standings_entries = []
    for league_tag in ("serie_a", "premier_league"):
        payload = _get_standings(league_tag)
        inner = payload.get("standings", {})
        items = list(inner.values()) if isinstance(inner, dict) else (inner if isinstance(inner, list) else [])
        for it in items:
            if isinstance(it, dict):
                it.setdefault("league", league_tag)
        standings_entries.extend(items)

    # Normalize team name for lookup (handles "Manchester City" → "Man City" etc.)
    from config.team_names import normalize_team
    team_norm = normalize_team(team)
    for entry in standings_entries:
        entry_team = entry.get("team", "") if isinstance(entry, dict) else ""
        if entry_team.lower() == team.lower() or normalize_team(entry_team) == team_norm:
            data["overview"] = {
                "position": entry.get("position", entry.get("rank", 0)),
                "played": entry.get("played", entry.get("mp", 0)),
                "won": entry.get("wins", entry.get("won", entry.get("w", 0))),
                "drawn": entry.get("draws", entry.get("drawn", entry.get("d", 0))),
                "lost": entry.get("losses", entry.get("lost", entry.get("l", 0))),
                "goals_for": entry.get("gf", entry.get("goals_for", 0)),
                "goals_against": entry.get("ga", entry.get("goals_against", 0)),
                "goal_difference": entry.get("gd", entry.get("goal_difference", 0)),
                "points": entry.get("points", entry.get("pts", 0)),
                "home": entry.get("home", {}),
                "away": entry.get("away", {}),
            }
            data["league"] = entry.get("league", "serie_a")
            break

    # ── 2. Current form (all leagues) ──
    form_data = _load_json(UPCOMING_DIR / "current_form.json", {})
    _epl_form = _load_json(UPCOMING_DIR / "current_form_premier_league.json", {})
    if isinstance(_epl_form, dict):
        for _fk in ("teams", "matchups"):
            if isinstance(_epl_form.get(_fk), dict):
                form_data.setdefault(_fk, {})
                if isinstance(form_data.get(_fk), dict):
                    form_data[_fk].update(_epl_form[_fk])
    if isinstance(form_data, dict):
        team_form = form_data.get("teams", {}).get(team, form_data.get(team, {}))
        if isinstance(team_form, dict):
            data["form"] = team_form.get("results", team_form.get("form", []))
            data["form_stats"] = {
                "ppg": team_form.get("ppg"),
                "wins": team_form.get("wins"),
                "draws": team_form.get("draws"),
                "losses": team_form.get("losses"),
                "goals_scored": team_form.get("goals_scored"),
                "goals_conceded": team_form.get("goals_conceded"),
                "form_status": team_form.get("form_status"),
                "home_form_status": team_form.get("home_form_status"),
                "away_form_status": team_form.get("away_form_status"),
            }
        elif isinstance(team_form, list):
            data["form"] = team_form

    # ── 3. Match history from features.parquet (wide-format: one row per match) ──
    try:
        import pandas as pd
        fp = DATA_DIR / "features" / "features.parquet"
        if fp.exists():
            try:
                df = pd.read_parquet(fp)

                # Find matches where team was home or away
                # Try both original name and normalized name (handles "Manchester City" → "Man City")
                home_mask = (df["home_team"].str.lower() == team.lower()) | (df["home_team"].str.lower() == team_norm.lower())
                away_mask = (df["away_team"].str.lower() == team.lower()) | (df["away_team"].str.lower() == team_norm.lower())
                team_matches = df[home_mask | away_mask].copy()

                if not team_matches.empty and "match_date" in team_matches.columns:
                    team_matches = team_matches.sort_values("match_date", ascending=False)

                    # Helper to safely extract a float from a row
                    def _val(row, col):
                        if col in row.index:
                            v = row[col]
                            if pd.notna(v):
                                return round(float(v), 2)
                        return None

                    # Build match history — ALL matches, grouped by season
                    all_history = []
                    for _, row in team_matches.iterrows():
                        is_home = str(row.get("home_team", "")).lower() == team.lower()
                        p = "home" if is_home else "away"
                        op = "away" if is_home else "home"

                        gs = row.get(f"{p}_score")
                        gc = row.get(f"{op}_score")
                        gs = float(gs) if pd.notna(gs) else None
                        gc = float(gc) if pd.notna(gc) else None

                        result = ""
                        if gs is not None and gc is not None:
                            result = "W" if gs > gc else ("D" if gs == gc else "L")

                        entry = {
                            "match_date": str(row.get("match_date", ""))[:10],
                            "season": str(row.get("season", "")) if pd.notna(row.get("season")) else "",
                            "opponent": row.get(f"{op}_team", ""),
                            "is_home": is_home,
                            "goals_scored": int(gs) if gs is not None else None,
                            "goals_conceded": int(gc) if gc is not None else None,
                            "result": result,
                        }

                        # Elo
                        entry["elo"] = _val(row, f"{p}_elo")

                        # xG: prefer SofaScore rolling (100% coverage), fallback to venue_roll
                        xg = _val(row, f"{p}_ss_roll_xg")
                        if xg is None:
                            xg = _val(row, f"{p}_venue_roll_5_xg_for")
                        entry["xg"] = xg

                        xga = _val(row, f"{op}_ss_roll_xg")
                        if xga is None:
                            xga = _val(row, f"{p}_venue_roll_5_xg_against")
                        entry["xg_against"] = xga

                        # Rich stats from SofaScore
                        entry["shots"] = _val(row, f"{p}_ss_roll_total_shots")
                        entry["shots_on_target"] = _val(row, f"{p}_ss_roll_shots_on_target")
                        entry["big_chances"] = _val(row, f"{p}_ss_roll_big_chances_created")
                        entry["key_passes"] = _val(row, f"{p}_ss_roll_key_passes")
                        entry["possession"] = _val(row, f"{p}_ss_roll_territory_ratio")
                        if entry["possession"] is not None:
                            entry["possession"] = round(entry["possession"] * 100, 1)
                        entry["pass_accuracy"] = _val(row, f"{p}_ss_roll_pass_accuracy")
                        if entry["pass_accuracy"] is not None:
                            entry["pass_accuracy"] = round(entry["pass_accuracy"] * 100, 1)
                        entry["tackles_won"] = _val(row, f"{p}_ss_roll_tackles_won")
                        entry["interceptions"] = _val(row, f"{p}_ss_roll_interceptions")
                        entry["gk_saves"] = _val(row, f"{p}_ss_roll_gk_saves")
                        entry["gk_rating"] = _val(row, f"{p}_ss_roll_gk_rating")
                        entry["rating"] = _val(row, f"{p}_ss_roll_rating")
                        entry["ppda"] = _val(row, f"{p}_ppda")
                        entry["fouls"] = _val(row, f"{p}_ss_roll_fouls")

                        # Matchweek
                        mw = row.get("matchweek")
                        entry["matchweek"] = int(mw) if pd.notna(mw) else None

                        all_history.append(entry)

                    # ── Merge recent results from results.json (fills gap between parquet rebuilds) ──
                    latest_parquet_date = all_history[0]["match_date"] if all_history else "2000-01-01"
                    results_cache = _load_json(UPCOMING_DIR / "results.json", {})
                    cached_results = results_cache.get("results", {})
                    try:
                        from config.team_names import normalize_team as _norm_team
                    except ImportError:
                        _norm_team = lambda x: x
                    for match_key, res in cached_results.items():
                        if not res.get("completed"):
                            continue
                        ct = res.get("commence_time", "")
                        match_date = ct[:10] if ct else ""
                        if not match_date or match_date <= latest_parquet_date:
                            continue  # Already in parquet
                        home = _norm_team(res.get("home_team", ""))
                        away = _norm_team(res.get("away_team", ""))
                        if home.lower() != team.lower() and away.lower() != team.lower():
                            continue
                        is_home = home.lower() == team.lower()
                        hs = res.get("home_score")
                        as_ = res.get("away_score")
                        gs = hs if is_home else as_
                        gc = as_ if is_home else hs
                        result = ""
                        if gs is not None and gc is not None:
                            result = "W" if gs > gc else ("D" if gs == gc else "L")
                        # Estimate current season from date
                        yr = int(match_date[:4])
                        mo = int(match_date[5:7])
                        season = f"{yr-1}-{yr}" if mo < 7 else f"{yr}-{yr+1}"
                        recent_entry = {
                            "match_date": match_date,
                            "season": season,
                            "opponent": away if is_home else home,
                            "is_home": is_home,
                            "goals_scored": gs,
                            "goals_conceded": gc,
                            "result": result,
                            "elo": None, "xg": None, "xg_against": None,
                            "shots": None, "shots_on_target": None,
                            "big_chances": None, "key_passes": None,
                            "possession": None, "pass_accuracy": None,
                            "tackles_won": None, "interceptions": None,
                            "gk_saves": None, "gk_rating": None,
                            "rating": None, "ppda": None, "fouls": None,
                            "matchweek": None,
                            "partial": True,  # Flag: no detailed stats yet
                        }
                        all_history.insert(0, recent_entry)

                    # Re-sort after merge
                    all_history.sort(key=lambda m: m.get("match_date", ""), reverse=True)

                    # Group by season
                    seasons_dict = {}
                    for m in all_history:
                        s = m.get("season", "Unknown")
                        if s not in seasons_dict:
                            seasons_dict[s] = []
                        seasons_dict[s].append(m)

                    data["match_history"] = all_history[:20]  # Last 20 for default view
                    data["seasons"] = {}
                    for season, matches in sorted(seasons_dict.items(), reverse=True):
                        data["seasons"][season] = matches

                    # Aggregate stats from last 10 matches
                    stats = {}
                    history = all_history[:10]

                    if history:
                        def _avg(key):
                            vals = [m[key] for m in history if m.get(key) is not None]
                            return round(sum(vals) / len(vals), 2) if vals else None

                        stats["avg_goals_scored"] = _avg("goals_scored")
                        stats["avg_goals_conceded"] = _avg("goals_conceded")
                        stats["avg_xg"] = _avg("xg")
                        stats["avg_xg_against"] = _avg("xg_against")
                        stats["avg_shots"] = _avg("shots")
                        stats["avg_shots_on_target"] = _avg("shots_on_target")
                        stats["avg_possession"] = _avg("possession")
                        stats["avg_pass_accuracy"] = _avg("pass_accuracy")
                        stats["avg_big_chances"] = _avg("big_chances")
                        stats["avg_key_passes"] = _avg("key_passes")
                        stats["avg_tackles_won"] = _avg("tackles_won")
                        stats["avg_interceptions"] = _avg("interceptions")
                        stats["avg_rating"] = _avg("rating")

                        goals_scored = [m["goals_scored"] for m in history if m.get("goals_scored") is not None]
                        goals_conceded = [m["goals_conceded"] for m in history if m.get("goals_conceded") is not None]
                        results_list = [m["result"] for m in history if m.get("result")]

                        if goals_scored and goals_conceded:
                            n = min(len(goals_scored), len(goals_conceded))
                            stats["clean_sheet_pct"] = round(sum(1 for g in goals_conceded[:n] if g == 0) / n * 100, 1)
                            stats["btts_pct"] = round(sum(1 for i in range(n) if goals_scored[i] > 0 and goals_conceded[i] > 0) / n * 100, 1)
                        if results_list:
                            stats["win_pct"] = round(results_list.count("W") / len(results_list) * 100, 1)
                            stats["draw_pct"] = round(results_list.count("D") / len(results_list) * 100, 1)
                            stats["loss_pct"] = round(results_list.count("L") / len(results_list) * 100, 1)

                    # Elo from most recent match
                    elos = [m.get("elo") for m in all_history if m.get("elo") is not None]
                    if elos:
                        stats["current_elo"] = elos[0]
                        if len(elos) > 1:
                            stats["elo_trend"] = round(elos[0] - elos[-1], 1)

                    # Elo history for chart (chronological, last 30 matches)
                    elo_points = [{"date": m["match_date"], "elo": m["elo"]}
                                  for m in reversed(all_history[:30]) if m.get("elo") is not None]
                    if elo_points:
                        stats["elo_history"] = elo_points

                    # xG history for chart (chronological, last 30 matches)
                    xg_points = [{"date": m["match_date"], "xg_for": m["xg"], "xg_against": m["xg_against"]}
                                 for m in reversed(all_history[:30])
                                 if m.get("xg") is not None and m.get("xg_against") is not None]
                    if xg_points:
                        stats["xg_history"] = xg_points

                    # Home vs Away split (current season only)
                    current_season_matches = seasons_dict.get(
                        sorted(seasons_dict.keys(), reverse=True)[0], []
                    ) if seasons_dict else []
                    home_matches = [m for m in current_season_matches if m.get("is_home")]
                    away_matches = [m for m in current_season_matches if not m.get("is_home")]
                    def _split_stats(matches):
                        results = [m["result"] for m in matches if m.get("result")]
                        gs = [m["goals_scored"] for m in matches if m.get("goals_scored") is not None]
                        gc = [m["goals_conceded"] for m in matches if m.get("goals_conceded") is not None]
                        n = len(results)
                        return {
                            "played": n,
                            "won": results.count("W"), "drawn": results.count("D"), "lost": results.count("L"),
                            "gf": sum(gs) if gs else 0, "ga": sum(gc) if gc else 0,
                            "win_pct": round(results.count("W") / n * 100, 1) if n > 0 else 0,
                            "avg_gf": round(sum(gs) / len(gs), 2) if gs else 0,
                            "avg_ga": round(sum(gc) / len(gc), 2) if gc else 0,
                        }
                    stats["home_record"] = _split_stats(home_matches)
                    stats["away_record"] = _split_stats(away_matches)

                    # Override W/D/L totals from standings.json (single source of truth)
                    s_home = data["overview"].get("home", {})
                    s_away = data["overview"].get("away", {})
                    if s_home:
                        stats["home_record"].update({
                            "played": s_home.get("played", stats["home_record"]["played"]),
                            "won": s_home.get("wins", stats["home_record"]["won"]),
                            "drawn": s_home.get("draws", stats["home_record"]["drawn"]),
                            "lost": s_home.get("losses", stats["home_record"]["lost"]),
                        })
                    if s_away:
                        stats["away_record"].update({
                            "played": s_away.get("played", stats["away_record"]["played"]),
                            "won": s_away.get("wins", stats["away_record"]["won"]),
                            "drawn": s_away.get("draws", stats["away_record"]["drawn"]),
                            "lost": s_away.get("losses", stats["away_record"]["lost"]),
                        })

                    # Goals distribution (how many 0-goal, 1-goal, 2-goal, 3+goal games)
                    gs_all = [m["goals_scored"] for m in current_season_matches if m.get("goals_scored") is not None]
                    gc_all = [m["goals_conceded"] for m in current_season_matches if m.get("goals_conceded") is not None]
                    if gs_all:
                        stats["goals_scored_dist"] = {
                            "0": sum(1 for g in gs_all if g == 0),
                            "1": sum(1 for g in gs_all if g == 1),
                            "2": sum(1 for g in gs_all if g == 2),
                            "3+": sum(1 for g in gs_all if g >= 3),
                        }
                    if gc_all:
                        stats["goals_conceded_dist"] = {
                            "0": sum(1 for g in gc_all if g == 0),
                            "1": sum(1 for g in gc_all if g == 1),
                            "2": sum(1 for g in gc_all if g == 2),
                            "3+": sum(1 for g in gc_all if g >= 3),
                        }

                    # Form trend (rolling 5-match PPG, last 20 matches, chronological)
                    recent_20 = list(reversed(all_history[:20]))
                    form_trend = []
                    for i in range(4, len(recent_20)):
                        window = recent_20[max(0, i-4):i+1]
                        pts = sum(3 if m.get("result") == "W" else 1 if m.get("result") == "D" else 0 for m in window)
                        form_trend.append({"date": recent_20[i]["match_date"], "ppg": round(pts / len(window), 2)})
                    if form_trend:
                        stats["form_trend"] = form_trend

                    # Per-season summary stats
                    season_summaries = {}
                    for season, matches in seasons_dict.items():
                        s_results = [m["result"] for m in matches if m.get("result")]
                        s_gs = [m["goals_scored"] for m in matches if m.get("goals_scored") is not None]
                        s_gc = [m["goals_conceded"] for m in matches if m.get("goals_conceded") is not None]
                        n = len(s_results)
                        if n > 0:
                            season_summaries[season] = {
                                "played": n,
                                "won": s_results.count("W"),
                                "drawn": s_results.count("D"),
                                "lost": s_results.count("L"),
                                "gf": sum(s_gs) if s_gs else 0,
                                "ga": sum(s_gc) if s_gc else 0,
                                "win_pct": round(s_results.count("W") / n * 100, 1),
                            }
                    data["season_summaries"] = season_summaries

                    data["stats"] = stats
            except Exception as e:
                log.warning(f"Failed to read features.parquet for team {team}: {e}")
    except ImportError:
        pass

    # ── 4. Player analysis / squad ──
    player_data = _load_json(UPCOMING_DIR / "player_analysis.json", {})
    match_analyses = []
    if isinstance(player_data, dict):
        match_analyses = player_data.get("matches", [])
    elif isinstance(player_data, list):
        match_analyses = player_data
    for analysis in match_analyses:
        if not isinstance(analysis, dict):
            continue
        home_t = analysis.get("home_team", "")
        away_t = analysis.get("away_team", "")
        if team.lower() not in (home_t.lower(), away_t.lower()):
            continue
        # Determine which side this team is on
        if team.lower() == home_t.lower():
            squad_data = analysis.get("home_squad", {})
        else:
            squad_data = analysis.get("away_squad", {})
        if isinstance(squad_data, dict):
            data["squad"] = {
                "rating": squad_data.get("squad_rating", analysis.get("home_strength" if team.lower() == home_t.lower() else "away_strength", 0)),
                "key_players": squad_data.get("key_players", {}),
                "total_xg": squad_data.get("total_xg", 0),
                "squad_depth": squad_data.get("squad_depth", 0),
                "strengths": squad_data.get("strengths", []),
                "weaknesses": squad_data.get("weaknesses", []),
                "players": [],  # filled below from league parquet
            }
            break

    # Populate squad.players from the league-appropriate Sofascore parquet.
    # Works for both leagues; player_analyzer.py doesn't serialize the player
    # list itself, so we build it here from primary data.
    _roster = _load_team_squad_roster(team)
    if _roster:
        if "squad" not in data or not isinstance(data.get("squad"), dict):
            data["squad"] = {
                "rating": 0, "key_players": {}, "total_xg": 0, "squad_depth": 0,
                "strengths": [], "weaknesses": [], "players": [],
            }
        data["squad"]["players"] = _roster
        data["squad"]["top_scorers"] = sorted(
            [p for p in _roster if p.get("goals", 0) > 0],
            key=lambda p: (p.get("goals", 0), p.get("assists", 0)),
            reverse=True,
        )[:5]
        data["squad"]["most_played"] = sorted(
            _roster, key=lambda p: p.get("minutes", 0), reverse=True
        )[:5]
        if not data["squad"].get("total_xg"):
            data["squad"]["total_xg"] = round(sum(p.get("xg", 0) for p in _roster), 1)

    # ── 5. Upcoming prediction (all leagues) ──
    archive = _load_json(UPCOMING_DIR / "predictions_archive.json", {})
    pred_list = []
    for _pf in [UPCOMING_DIR / "predictions.json"] + list(UPCOMING_DIR.glob("predictions_*.json")):
        if not _pf.exists() or _pf.name == "predictions_archive.json":
            continue
        _pr = _load_json(_pf, [])
        _pl = _pr if isinstance(_pr, list) else _pr.get("predictions", [])
        pred_list.extend(_pl)

    ct_map = _get_commence_times()
    from datetime import datetime, timezone
    _now_utc = datetime.now(timezone.utc)
    _today_str = _now_utc.strftime("%Y-%m-%d")
    for pred in pred_list:
        if isinstance(pred, dict):
            if (pred.get("home_team", "").lower() == team.lower() or
                    pred.get("away_team", "").lower() == team.lower()):
                match_key = pred.get("match", "")
                ct = ct_map.get(match_key, pred.get("commence_time", ""))
                # Skip predictions for matches that have already kicked off
                # (predictions.json gets stale when Odds API quota stops feeding it)
                _pred_date = pred.get("date", "")
                _is_past = False
                if ct:
                    try:
                        _kt = datetime.fromisoformat(str(ct).replace("Z", "+00:00"))
                        if _kt.tzinfo is None:
                            _kt = _kt.replace(tzinfo=timezone.utc)
                        _is_past = _kt < _now_utc
                    except Exception:
                        pass
                elif _pred_date and _pred_date < _today_str:
                    _is_past = True
                if _is_past:
                    continue
                data["upcoming"] = {
                    "match": match_key,
                    "date": pred.get("date", ""),
                    "commence_time": ct,
                    "venue": pred.get("venue", ""),
                    "predicted_outcome": pred.get("predicted_outcome", ""),
                    "confidence": pred.get("confidence_level", ""),
                    "probabilities": pred.get("probabilities", {}),
                    "betting_probabilities": pred.get("betting_probabilities", {}),
                    "is_home": pred.get("home_team", "").lower() == team.lower(),
                    "referee": pred.get("referee", ""),
                    "referee_bias": pred.get("referee_bias", ""),
                    "expected_goals": pred.get("expected_goals"),
                    "home_xg": pred.get("home_xg"),
                    "away_xg": pred.get("away_xg"),
                    "over_25": pred.get("over_25"),
                    "betting_recommendation": pred.get("betting_recommendation", ""),
                    "market_edge": pred.get("market_edge"),
                    "market_implied": pred.get("market_implied", {}),
                    "home_factors": pred.get("home_factors", []),
                    "away_factors": pred.get("away_factors", []),
                    "home_form": pred.get("home_form", {}),
                    "away_form": pred.get("away_form", {}),
                    "component_predictions": pred.get("component_predictions", {}),
                    "draw_analysis": pred.get("draw_analysis", {}),
                    "methods_used": pred.get("methods_used", []),
                    "formation_analysis": pred.get("formation_analysis", {}),
                }
                break

    # ── 6. H2H data (all leagues) ──
    h2h_all = {}
    for _hf in [UPCOMING_DIR / "h2h_upcoming.json", UPCOMING_DIR / "h2h_upcoming_premier_league.json"]:
        _hr = _load_json(_hf, {})
        if isinstance(_hr, dict):
            h2h_all.update(_hr.get("h2h", _hr))
    for match_key, h2h_data in h2h_all.items():
        if team.lower() in match_key.lower():
            data["h2h"] = h2h_data
            break

    # ── 7. Lineup prediction ──
    lineup_preds = _load_json(UPCOMING_DIR / "lineup_predictions.json", {})
    team_lineup = {}
    if isinstance(lineup_preds, dict):
        teams_data = lineup_preds.get("teams", {})
        team_lineup = teams_data.get(team, {})
        # Try case-insensitive match
        if not team_lineup:
            for t, tdata in teams_data.items():
                if t.lower() == team.lower():
                    team_lineup = tdata
                    break
    if team_lineup:
        data["lineup_prediction"] = team_lineup

    if not data.get("league"):
        try:
            from config.leagues import infer_league
            data["league"] = infer_league(team, None)
        except Exception:
            data["league"] = "serie_a"

    # Fallback: if predictions.json had nothing in the future for this team
    # (Odds API outage), look up the next fixture from Sofascore fixtures file.
    if not data.get("upcoming") or not data["upcoming"].get("match"):
        next_fx = _next_fixture_for_team(data.get("league", "serie_a"), team)
        if next_fx:
            data["upcoming"] = next_fx

    return jsonify(data)


# ---------------------------------------------------------------------------
# Lineup Predictions API
# ---------------------------------------------------------------------------

@app.route("/api/lineup-predictions")
def api_lineup_predictions():
    """Get all lineup predictions for upcoming matches."""
    data = _load_json(UPCOMING_DIR / "lineup_predictions.json", {})
    return jsonify(data)


@app.route("/api/lineup-prediction/<team_name>")
def api_lineup_prediction(team_name):
    """Get lineup prediction for a specific team."""
    from urllib.parse import unquote
    team = unquote(team_name)
    data = _load_json(UPCOMING_DIR / "lineup_predictions.json", {})
    teams = data.get("teams", {})
    result = teams.get(team, {})
    if not result:
        for t, tdata in teams.items():
            if t.lower() == team.lower():
                result = tdata
                break
    return jsonify(result)


@app.route("/api/lineup-assessment")
def api_lineup_assessment():
    """Compare confirmed vs predicted lineups and return impact assessment.

    Only returns data for matches that have confirmed lineups available.
    Includes squad strength delta, key changes, and a verdict on whether
    the original match prediction should still be trusted.
    """
    from scripts.prediction.lineup_predictor import assess_lineup_impact

    confirmed_raw = _load_json(UPCOMING_DIR / "confirmed_lineups.json", {})
    predicted_raw = _load_json(UPCOMING_DIR / "lineup_predictions.json", {})

    confirmed_matches = confirmed_raw.get("matches", {}) if isinstance(confirmed_raw, dict) else {}
    predicted_matches = predicted_raw.get("matches", {}) if isinstance(predicted_raw, dict) else {}

    if not confirmed_matches:
        return jsonify({"matches": {}, "message": "No confirmed lineups available yet"})

    # Build player xG map from Sofascore stats (per-90 rates) — both leagues
    player_xg_map = {}
    try:
        import pandas as pd
        frames = []
        for fname in ("player_match_stats.parquet",
                      "player_match_stats_premier_league.parquet"):
            p = DATA_DIR / "external" / "sofascore" / fname
            if p.exists():
                frames.append(pd.read_parquet(p, columns=["player_name", "xg", "minutes"]))
        if frames:
            pms = pd.concat(frames, ignore_index=True)
            agg = pms.groupby("player_name").agg(
                total_xg=("xg", "sum"), total_mins=("minutes", "sum"),
            )
            for name, row in agg.iterrows():
                if row["total_mins"] > 0:
                    player_xg_map[name] = row["total_xg"] / row["total_mins"] * 90
    except Exception:
        pass  # Assessment will work without xG data (overlap only)

    assessments = {}
    for match_key, conf_data in confirmed_matches.items():
        pred_data = predicted_matches.get(match_key, {})
        if not pred_data:
            continue

        match_assessment = {}
        for side in ("home", "away"):
            conf_names = conf_data.get(f"{side}_lineup", [])
            pred_lineup = pred_data.get(f"{side}_lineup", {})
            pred_xi = pred_lineup.get("predicted_xi", [])
            pred_names = [p["name"] for p in pred_xi]
            team = conf_data.get(f"{side}_team", pred_data.get(f"{side}_team", ""))

            if not conf_names or not pred_names:
                continue

            match_assessment[side] = assess_lineup_impact(
                predicted_names=pred_names,
                confirmed_names=conf_names,
                team=team,
                player_xg_map=player_xg_map,
            )
            match_assessment[side]["confirmed_formation"] = conf_data.get(f"{side}_formation", "")
            match_assessment[side]["predicted_formation"] = (
                pred_lineup.get("formation", {}).get("predicted", "")
            )

        if match_assessment:
            assessments[match_key] = match_assessment

    return jsonify({
        "matches": assessments,
        "confirmed_at": confirmed_raw.get("fetched_at", ""),
        "match_count": len(assessments),
    })


@app.route("/api/lineup-accuracy")
def api_lineup_accuracy():
    """Get historical lineup prediction accuracy from past evaluations.

    Reads data/lineup_history/evaluation.json which is populated by
    running: python3 -m scripts.prediction.lineup_predictor --evaluate
    """
    eval_path = DATA_DIR / "lineup_history" / "evaluation.json"
    data = _load_json(eval_path, {})
    if not data:
        return jsonify({
            "message": "No evaluation data yet. Run: python3 -m scripts.prediction.lineup_predictor --evaluate",
            "overall_avg": 0,
            "total_evaluations": 0,
        })
    return jsonify({
        "overall_avg": data.get("overall_avg_overlap", 0),
        "total_evaluations": data.get("total_evaluations", 0),
        "team_averages": data.get("team_averages", {}),
        "last_evaluated": data.get("last_evaluated", ""),
        "recent_results": data.get("results", [])[-20:],  # last 20
    })


# ---------------------------------------------------------------------------
# Match Events / History API
# ---------------------------------------------------------------------------

_sofascore_match_lookup = None  # cached (date, home, away) -> sofascore_match_id


def _get_sofascore_lookup():
    """Build (date, normalized_home, normalized_away) -> sofascore_match_id lookup.

    Uses two sources:
    1. player_match_stats.parquet — covers 2022-2023 onwards (precise dates)
    2. Sofascore fixtures JSON files — covers 2017-2018 onwards (all seasons)
    """
    global _sofascore_match_lookup
    if _sofascore_match_lookup is not None:
        return _sofascore_match_lookup
    try:
        from config.team_names import normalize_team
        from datetime import datetime
        lookup = {}

        # Source 1: fixtures JSON files (covers all seasons 2017+)
        fixtures_dir = DATA_DIR / "external" / "sofascore"
        for fx_file in sorted(fixtures_dir.glob("fixtures_*.json")):
            try:
                with open(fx_file) as f:
                    fixtures = json.load(f)
                for fix in fixtures:
                    fid = fix.get("id")
                    ts = fix.get("startTimestamp", 0)
                    if not fid or not ts:
                        continue
                    d = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
                    h = normalize_team(fix.get("homeTeam", {}).get("name", ""))
                    a = normalize_team(fix.get("awayTeam", {}).get("name", ""))
                    if h and a:
                        lookup[(d, h, a)] = int(fid)
            except Exception:
                continue

        # Source 2: player_match_stats.parquet — both leagues (overwrites fixture
        # estimates with precise dates). Must include EPL parquet too, otherwise
        # _get_sofascore_lookup misses EPL match IDs and downstream features
        # (match-events, match-detail) silently 404 for EPL queries.
        try:
            import pandas as pd
            for fname in ("player_match_stats.parquet",
                          "player_match_stats_premier_league.parquet"):
                p = DATA_DIR / "external" / "sofascore" / fname
                if not p.exists():
                    continue
                pms = pd.read_parquet(
                    p, columns=["match_id", "date", "home_team", "away_team"]
                ).drop_duplicates(subset="match_id")
                for _, row in pms.iterrows():
                    d = str(row["date"])[:10]
                    h = normalize_team(str(row["home_team"]))
                    a = normalize_team(str(row["away_team"]))
                    lookup[(d, h, a)] = int(row["match_id"])
        except Exception:
            pass

        _sofascore_match_lookup = lookup
        log.info(f"Sofascore lookup: {len(lookup)} matches indexed")
        return lookup
    except Exception as e:
        log.warning(f"Failed to build Sofascore lookup: {e}")
        return {}


def _load_subbed_off_map(ss_match_id: int) -> dict:
    """Build {(is_home, minute) -> [player_name]} from match JSON starters' minutesPlayed.

    Used to resolve the 'player out' in substitutions, since match_incidents
    only stores the player coming IN.
    """
    result = {}  # (is_home, minute) -> [player_names]
    try:
        matches_dir = DATA_DIR / "external" / "sofascore" / "matches"
        for season_dir in sorted(matches_dir.glob("*"), reverse=True):
            match_file = season_dir / f"{ss_match_id}.json"
            if match_file.exists():
                with open(match_file) as f:
                    mdata = json.load(f)
                for side, is_home in [("home", True), ("away", False)]:
                    lineup = mdata.get(f"{side}_lineup", {})
                    for p in lineup.get("starters", []):
                        player = p.get("player", {})
                        stats = p.get("statistics", {})
                        mins = stats.get("minutesPlayed", 90)
                        if mins < 88:  # Was subbed off
                            key = (is_home, mins)
                            result.setdefault(key, []).append(player.get("name", ""))
                break
    except Exception:
        pass
    return result


# ── Venue lookup (home stadiums for SA + EPL) ──────────────────────────
_HOME_VENUES: dict[str, str] = {
    # Serie A 2025-26
    "Inter": "San Siro, Milan",
    "Milan": "San Siro, Milan",
    "Juventus": "Allianz Stadium, Turin",
    "Napoli": "Diego Armando Maradona, Naples",
    "Roma": "Stadio Olimpico, Rome",
    "Lazio": "Stadio Olimpico, Rome",
    "Atalanta": "Gewiss Stadium, Bergamo",
    "Fiorentina": "Artemio Franchi, Florence",
    "Bologna": "Renato Dall'Ara, Bologna",
    "Torino": "Stadio Olimpico Grande Torino, Turin",
    "Como": "Giuseppe Sinigaglia, Como",
    "Genoa": "Luigi Ferraris, Genoa",
    "Udinese": "Bluenergy Stadium, Udine",
    "Cagliari": "Unipol Domus, Cagliari",
    "Verona": "Marc'Antonio Bentegodi, Verona",
    "Lecce": "Via del Mare, Lecce",
    "Parma": "Ennio Tardini, Parma",
    "Sassuolo": "Mapei Stadium, Reggio Emilia",
    "Cremonese": "Giovanni Zini, Cremona",
    "Pisa": "Arena Garibaldi, Pisa",
    # Premier League 2025-26
    "Liverpool": "Anfield, Liverpool",
    "Arsenal": "Emirates Stadium, London",
    "Man City": "Etihad Stadium, Manchester",
    "Manchester City": "Etihad Stadium, Manchester",
    "Man United": "Old Trafford, Manchester",
    "Manchester United": "Old Trafford, Manchester",
    "Chelsea": "Stamford Bridge, London",
    "Tottenham": "Tottenham Hotspur Stadium, London",
    "Tottenham Hotspur": "Tottenham Hotspur Stadium, London",
    "Newcastle": "St James' Park, Newcastle",
    "Newcastle United": "St James' Park, Newcastle",
    "Aston Villa": "Villa Park, Birmingham",
    "Brighton": "Amex Stadium, Brighton",
    "Brighton and Hove Albion": "Amex Stadium, Brighton",
    "West Ham": "London Stadium, London",
    "West Ham United": "London Stadium, London",
    "Crystal Palace": "Selhurst Park, London",
    "Brentford": "Gtech Community Stadium, London",
    "Fulham": "Craven Cottage, London",
    "Wolves": "Molineux, Wolverhampton",
    "Wolverhampton Wanderers": "Molineux, Wolverhampton",
    "Bournemouth": "Vitality Stadium, Bournemouth",
    "Everton": "Hill Dickinson Stadium, Liverpool",
    "Nottingham Forest": "City Ground, Nottingham",
    "Burnley": "Turf Moor, Burnley",
    "Leeds": "Elland Road, Leeds",
    "Leeds United": "Elland Road, Leeds",
    "Sunderland": "Stadium of Light, Sunderland",
}


def _venue_for(home_team: str) -> str:
    """Return home stadium name for a team, or empty string if unknown."""
    return _HOME_VENUES.get(home_team, "")


def _standings_as_of(league: str, cutoff_date: str) -> dict[str, dict]:
    """Compute standings as they were BEFORE matches on `cutoff_date`.

    Returns {team_name: {position, played, points, ...}} reflecting only
    matches strictly before cutoff_date. Used for "table position at kickoff".
    """
    import pandas as pd
    from config.settings import PROJECT_ROOT

    rel = _LEAGUE_PARQUET.get(league)
    if not rel:
        return {}
    parquet_path = PROJECT_ROOT / rel
    df = _read_parquet_cached(parquet_path)
    if df is None or len(df) == 0:
        return {}

    season = get_current_season()
    df = df[df["season"] == season]
    matches = df[
        ["match_id", "date", "home_team", "away_team", "home_score", "away_score"]
    ].drop_duplicates(subset=["match_id"]).dropna(subset=["home_score", "away_score"])
    # Only matches strictly before cutoff
    matches = matches[matches["date"].astype(str) < str(cutoff_date)]
    if len(matches) == 0:
        return {}

    teams: dict[str, dict] = {}
    def _row(t: str) -> dict:
        if t not in teams:
            teams[t] = {"team": t, "played": 0, "wins": 0, "draws": 0, "losses": 0,
                        "gf": 0, "ga": 0, "gd": 0, "points": 0}
        return teams[t]

    for _, m in matches.iterrows():
        ht, at = m["home_team"], m["away_team"]
        hs, ascore = int(m["home_score"]), int(m["away_score"])
        h, a = _row(ht), _row(at)
        h["played"] += 1; a["played"] += 1
        h["gf"] += hs; h["ga"] += ascore
        a["gf"] += ascore; a["ga"] += hs
        if hs > ascore:
            h["wins"] += 1; h["points"] += 3
            a["losses"] += 1
        elif hs < ascore:
            a["wins"] += 1; a["points"] += 3
            h["losses"] += 1
        else:
            h["draws"] += 1; h["points"] += 1
            a["draws"] += 1; a["points"] += 1

    for t in teams.values():
        t["gd"] = t["gf"] - t["ga"]

    ranked = sorted(teams.values(), key=lambda r: (-r["points"], -r["gd"], -r["gf"]))
    for i, t in enumerate(ranked, 1):
        t["position"] = i
    return {t["team"]: t for t in ranked}


def _h2h_last_n(league: str, team_a: str, team_b: str, before_date: str, n: int = 5) -> list[dict]:
    """Last N head-to-head meetings between two teams strictly before a date.

    Source: Sofascore parquet (one row per match). Returns oldest→newest.
    """
    import pandas as pd
    from config.settings import PROJECT_ROOT
    from config.team_names import normalize_team

    rel = _LEAGUE_PARQUET.get(league)
    if not rel:
        return []
    df = _read_parquet_cached(PROJECT_ROOT / rel)
    if df is None or len(df) == 0:
        return []

    a_norm = normalize_team(team_a)
    b_norm = normalize_team(team_b)
    matches = df[["match_id", "date", "round", "season", "home_team", "away_team",
                  "home_score", "away_score"]].drop_duplicates(subset=["match_id"])
    matches = matches.dropna(subset=["home_score", "away_score"])
    # Filter to fixtures involving both teams
    pairs = matches[
        (
            (matches["home_team"].apply(lambda t: normalize_team(str(t))) == a_norm) &
            (matches["away_team"].apply(lambda t: normalize_team(str(t))) == b_norm)
        ) | (
            (matches["home_team"].apply(lambda t: normalize_team(str(t))) == b_norm) &
            (matches["away_team"].apply(lambda t: normalize_team(str(t))) == a_norm)
        )
    ]
    pairs = pairs[pairs["date"].astype(str) < str(before_date)]
    pairs = pairs.sort_values("date", ascending=False).head(n)
    pairs = pairs.sort_values("date")  # oldest → newest

    out = []
    for _, m in pairs.iterrows():
        out.append({
            "date": str(m["date"])[:10],
            "season": str(m.get("season", "")),
            "matchweek": int(m["round"]) if pd.notna(m.get("round")) else None,
            "home_team": str(m["home_team"]),
            "away_team": str(m["away_team"]),
            "home_score": int(m["home_score"]),
            "away_score": int(m["away_score"]),
        })
    return out


def _weather_for_match(match_id: int, date: str = "", home: str = "", away: str = "") -> dict:
    """Look up weather for a match.

    weather.parquet has two ID formats living together:
      - Legacy: '{date}_{home}_{away}' (used for SA + new EPL backfill)
      - Sofascore numeric: e.g. 13980106
    Try both — fall back to date+home+away derivation if numeric lookup misses.
    """
    import pandas as pd
    from config.settings import PROJECT_ROOT
    try:
        w = _read_parquet_cached(PROJECT_ROOT / "data" / "external" / "weather.parquet")
        if w is None or len(w) == 0:
            return {}
        # Try numeric match_id first
        rows = w[w["match_id"].astype(str) == str(match_id)]
        # Fallback: legacy "{date}_{home}_{away}" key (with both raw and underscore-sanitized variants)
        if len(rows) == 0 and date and home and away:
            for h_norm, a_norm in [(home, away), (home.replace(" ", "_"), away.replace(" ", "_"))]:
                key = f"{str(date)[:10]}_{h_norm}_{a_norm}"
                rows = w[w["match_id"].astype(str) == key]
                if len(rows):
                    break
        if len(rows) == 0:
            return {}
        r = rows.iloc[0]
        return {
            "temp_mean": float(r["weather_temperature_2m_mean"]) if pd.notna(r.get("weather_temperature_2m_mean")) else None,
            "temp_min": float(r["weather_temperature_2m_min"]) if pd.notna(r.get("weather_temperature_2m_min")) else None,
            "temp_max": float(r["weather_temperature_2m_max"]) if pd.notna(r.get("weather_temperature_2m_max")) else None,
            "precipitation_mm": float(r["weather_precipitation_sum"]) if pd.notna(r.get("weather_precipitation_sum")) else None,
            "rain_mm": float(r["weather_rain_sum"]) if pd.notna(r.get("weather_rain_sum")) else None,
            "wind_speed_kmh": float(r["weather_wind_speed_10m_max"]) if pd.notna(r.get("weather_wind_speed_10m_max")) else None,
            "humidity_pct": float(r["weather_relative_humidity_2m_mean"]) if pd.notna(r.get("weather_relative_humidity_2m_mean")) else None,
            "snowfall_cm": float(r["weather_snowfall_sum"]) if pd.notna(r.get("weather_snowfall_sum")) else None,
        }
    except Exception as e:
        log.warning(f"weather lookup failed for {match_id}: {e}")
        return {}


def _referee_for_match(date: str, home: str, away: str) -> str:
    """Resolve referee name from matches.parquet (or upcoming/referees.json)."""
    import pandas as pd
    from config.settings import PROJECT_ROOT
    from config.team_names import normalize_team

    try:
        mp = _read_parquet_cached(PROJECT_ROOT / "data" / "parsed" / "matches.parquet")
        if mp is not None and len(mp):
            # match_date is datetime; cast to str for compare
            d = str(date)[:10]
            home_n = normalize_team(home)
            away_n = normalize_team(away)
            rows = mp[
                (mp["match_date"].astype(str).str[:10] == d) &
                (mp["home_team"].apply(lambda t: normalize_team(str(t))) == home_n) &
                (mp["away_team"].apply(lambda t: normalize_team(str(t))) == away_n)
            ]
            if len(rows):
                ref = str(rows.iloc[0].get("referee", "")).strip()
                if ref and ref.lower() != "nan":
                    return ref
    except Exception:
        pass

    # Fallback: upcoming/referees.json (for scheduled matches)
    refs = _load_json(UPCOMING_DIR / "referees.json", {})
    if isinstance(refs, dict):
        # Common formats: {date_home_away: {referee:...}} or {match: {...}}
        key1 = f"{date}_{home}_{away}"
        for k in (key1, f"{home} vs {away}"):
            if k in refs and isinstance(refs[k], dict):
                return refs[k].get("referee", refs[k].get("name", ""))
    return ""


def _build_match_context(match_id: int | None, date: str, home: str, away: str, league: str) -> dict:
    """Assemble the match-context block: referee, venue, weather, table positions, h2h.

    Live HTML scrape augments parquet/static lookups when those are missing
    or stale (e.g. recent matchweeks not yet ingested by FBref).
    """
    referee = _referee_for_match(date, home, away)
    venue = _venue_for_home(league, home)
    stoppage_time: dict = {}
    had_extra_time = False
    attendance = None
    # Augment from live match-page HTML when match_id is known. Always pull
    # stoppage_time / extra_time / attendance — those don't exist in any
    # local source. Only fetch HTML once (it's cached).
    if match_id:
        html_data = _scrape_match_html(match_id)
        if not referee:
            referee = html_data.get("referee", "")
        if not venue:
            venue = html_data.get("venue", venue)
        stoppage_time = html_data.get("stoppage_time", {}) or {}
        had_extra_time = bool(html_data.get("had_extra_time"))
        attendance = html_data.get("attendance")
    ctx: dict = {
        "referee": referee,
        "venue": venue,
        "weather": _weather_for_match(match_id, date, home, away) if match_id else
                   _weather_for_match(0, date, home, away) if (date and home and away) else {},
        "league": league,
        "stoppage_time": stoppage_time,
        "had_extra_time": had_extra_time,
        "attendance": attendance,
    }
    # Table positions as of the day before kickoff
    standings = _standings_as_of(league, date)
    home_row = standings.get(home, {})
    away_row = standings.get(away, {})
    if home_row:
        ctx["home_position"] = {
            "position": home_row.get("position"),
            "played": home_row.get("played", 0),
            "points": home_row.get("points", 0),
            "gd": home_row.get("gd", 0),
        }
    if away_row:
        ctx["away_position"] = {
            "position": away_row.get("position"),
            "played": away_row.get("played", 0),
            "points": away_row.get("points", 0),
            "gd": away_row.get("gd", 0),
        }
    # H2H last 5
    ctx["h2h_last5"] = _h2h_last_n(league, home, away, date, n=5)
    return ctx


def _venue_for_home(league: str, home_team: str) -> str:
    """Wrapper around _venue_for that respects league when names collide."""
    return _HOME_VENUES.get(home_team, "")


# ── Live Sofascore match-page HTML scraper ──────────────────────────────
# Cloudflare blocks api.sofascore.com but allows www.sofascore.com HTML.
# We use the embedded NEXT_DATA blob to recover incidents/venue/referee
# for matches that are missing from the parquet OR that have empty
# substitution OFF fields.
_match_html_cache: dict[int, tuple[float, dict]] = {}
_MATCH_HTML_TTL = 600  # 10min — match pages are heavy


def _scrape_match_html(ss_match_id: int) -> dict:
    """Fetch a Sofascore match page and parse its NEXT_DATA blob.

    Returns:
        {
          "incidents": [...],   # raw list with 'incidentType', 'playerOut', etc.
          "venue": str,         # "San Siro/Giuseppe Meazza, Milan"
          "referee": str,       # "Simone Sozza"
          "manager_home": str,
          "manager_away": str,
        }
    Returns {} on failure (CF block, parse error, etc.).
    """
    import time as _t
    now = _t.time()
    if ss_match_id in _match_html_cache:
        cached_at, payload = _match_html_cache[ss_match_id]
        if (now - cached_at) < _MATCH_HTML_TTL:
            import copy
            return copy.deepcopy(payload)

    try:
        from curl_cffi import requests as cffi  # type: ignore
        s = cffi.Session(impersonate="chrome120")
        s.headers.update({
            "Referer": "https://www.google.com/",
            "Accept": "text/html",
            "Accept-Language": "en-US,en;q=0.9",
        })
        # /event/{id} redirects to the canonical /football/match/{slug}/{token}
        r = s.get(f"https://www.sofascore.com/event/{ss_match_id}", timeout=5)
        if r.status_code != 200:
            log.warning("match HTML %d: HTTP %d", ss_match_id, r.status_code)
            return {}

        import re as _re
        m = _re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', r.text, _re.DOTALL)
        if not m:
            return {}
        data = json.loads(m.group(1))
        ip = data.get("props", {}).get("pageProps", {}).get("initialProps", {})
        if not ip:
            return {}
        ev = ip.get("event", {}) or {}
        venue_obj = ev.get("venue", {}) or {}
        venue_name = venue_obj.get("name", "") or venue_obj.get("stadium", {}).get("name", "")
        venue_city = (venue_obj.get("city", {}) or {}).get("name", "")
        venue_str = f"{venue_name}, {venue_city}".strip(", ").strip() if venue_name or venue_city else ""

        ref_obj = ev.get("referee", {}) or {}
        ref_name = ref_obj.get("name", "")

        home_mgr = ((ev.get("homeTeam") or {}).get("manager") or {}).get("name", "") if isinstance((ev.get("homeTeam") or {}).get("manager"), dict) else ""
        away_mgr = ((ev.get("awayTeam") or {}).get("manager") or {}).get("name", "") if isinstance((ev.get("awayTeam") or {}).get("manager"), dict) else ""

        # Stoppage / extra time per period.
        # Sofascore exposes injuryTime1..4 on event.time:
        #   injuryTime1 = added minutes at end of 1st half
        #   injuryTime2 = added minutes at end of 2nd half
        #   injuryTime3 = added minutes at end of 1st extra-time half
        #   injuryTime4 = added minutes at end of 2nd extra-time half
        # All optional. None means "didn't happen / not applicable".
        et = ev.get("time", {}) or {}
        def _injury(key):
            v = et.get(key)
            try:
                v = int(v)
                return v if v > 0 else None
            except (TypeError, ValueError):
                return None
        stoppage_time = {
            "first_half": _injury("injuryTime1"),
            "second_half": _injury("injuryTime2"),
            "extra_time_first_half": _injury("injuryTime3"),
            "extra_time_second_half": _injury("injuryTime4"),
        }
        had_extra_time = stoppage_time["extra_time_first_half"] is not None or \
                         stoppage_time["extra_time_second_half"] is not None

        # Attendance, if reported
        attendance = ev.get("attendance")

        payload = {
            "incidents": ip.get("incidents", []) or [],
            "venue": venue_str,
            "referee": ref_name,
            "manager_home": home_mgr,
            "manager_away": away_mgr,
            "stoppage_time": stoppage_time,
            "had_extra_time": had_extra_time,
            "attendance": attendance,
        }
        _match_html_cache[ss_match_id] = (now, payload)
        return payload
    except Exception as e:
        log.warning("match HTML %d failed: %s", ss_match_id, e)
        return {}


def _events_from_html_incidents(incidents: list) -> dict:
    """Convert Sofascore raw incidents (from HTML) to our normalized shape."""
    goals = []
    cards = []
    subs = []
    for inc in incidents:
        if not isinstance(inc, dict):
            continue
        itype = inc.get("incidentType", "")
        minute = int(inc.get("time", 0) or 0)
        added = int(inc.get("addedTime", 0) or 0)
        is_home = bool(inc.get("isHome", False))
        if itype == "goal":
            player = (inc.get("player") or {}).get("name", "")
            assist = (inc.get("assist1") or {}).get("name", "")
            goals.append({
                "player": player, "minute": minute, "added_time": added,
                "is_home": is_home, "type": inc.get("incidentClass", "regular"),
                "assist": assist,
            })
        elif itype == "card":
            player = (inc.get("player") or {}).get("name", "")
            cards.append({
                "player": player, "minute": minute, "added_time": added,
                "is_home": is_home, "card_type": inc.get("incidentClass", "yellow"),
            })
        elif itype == "substitution":
            player_out = (inc.get("playerOut") or {}).get("name", "")
            player_in = (inc.get("playerIn") or {}).get("name", "")
            subs.append({
                "player_out": player_out, "player_in": player_in,
                "minute": minute, "added_time": added, "is_home": is_home,
            })
    return {
        "goals": sorted(goals, key=lambda x: x["minute"]),
        "cards": sorted(cards, key=lambda x: x["minute"]),
        "substitutions": sorted(subs, key=lambda x: x["minute"]),
    }


def _load_match_team_stats(ss_match_id: int) -> dict:
    """Load match team stats (possession, shots, xG, fouls, corners, ...).

    Returns: {
        "home": {ALL: {...}, "1ST": {...}, "2ND": {...}},
        "away": {ALL: {...}, "1ST": {...}, "2ND": {...}}
    }

    Empty dicts for missing periods/teams.
    """
    out = {"home": {}, "away": {}}
    try:
        import pandas as pd
        # Try SA file first, then EPL file. The match_id is unique across leagues
        # so only one will have rows for any given ss_match_id.
        df = None
        for fname in ("match_team_stats.parquet",
                      "match_team_stats_premier_league.parquet"):
            p = DATA_DIR / "external" / "sofascore" / fname
            if not p.exists():
                continue
            cand = _read_parquet_cached(p)
            cand = cand[cand["match_id"] == ss_match_id]
            if not cand.empty:
                df = cand
                break
        if df is None or df.empty:
            return out
        # Pick interesting columns; ignore rare/missing ones gracefully
        keys = ("possession", "total_shots", "shots_on_target",
                "shots_inside_box", "corners", "fouls", "offsides",
                "xg", "accurate_passes", "total_passes",
                "total_tackles", "interceptions", "clearances",
                "duel_won_pct", "big_chances_scored", "big_chances_missed",
                "gk_saves")
        for _, row in df.iterrows():
            side = "home" if bool(row.get("is_home", False)) else "away"
            period = str(row.get("period", "ALL"))
            blk = {}
            for k in keys:
                v = row.get(k)
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    continue
                blk[k] = float(v) if isinstance(v, (float, int)) else v
            if blk:
                out[side][period] = blk
    except Exception as e:
        log.warning("team stats load failed for %d: %s", ss_match_id, e)
    return out


def _load_match_events(ss_match_id: int) -> dict:
    """Load events (goals, cards, subs) for a Sofascore match_id.

    Strategy:
      1. Try parquet first (fast).
      2. If parquet missing OR substitution OFF fields all empty (known scraper
         bug), augment by scraping the live match-page HTML.
    """
    try:
        import pandas as pd
        inc_path = DATA_DIR / "external" / "sofascore" / "match_incidents.parquet"
        df = None
        if inc_path.exists():
            df = pd.read_parquet(inc_path)
            df = df[df["match_id"] == ss_match_id]

        # Decide: do we need to fall back to HTML?
        # — parquet has no rows for this match
        # — OR all substitutions in parquet have empty player_name (the OFF bug)
        needs_html = (df is None or df.empty)
        if not needs_html:
            sub_rows = df[df["incident_type"] == "substitution"]
            if len(sub_rows) and (sub_rows["player_name"].fillna("").str.strip() == "").all():
                needs_html = True

        if needs_html:
            html_data = _scrape_match_html(ss_match_id)
            incidents = html_data.get("incidents") or []
            if incidents:
                return _events_from_html_incidents(incidents)
            if df is None or df.empty:
                return {"goals": [], "cards": [], "substitutions": []}

        # Parquet path (existing logic below)
        if df.empty:
            return {"goals": [], "cards": [], "substitutions": []}

        # Pre-load subbed-off map for resolving player_out
        subbed_off = _load_subbed_off_map(ss_match_id)
        # Track which subbed-off players have been assigned
        used_off = set()

        goals = []
        cards = []
        subs = []
        for _, row in df.iterrows():
            itype = str(row.get("incident_type", ""))
            minute = int(row.get("minute", 0))
            added = int(row.get("added_time", 0)) if pd.notna(row.get("added_time")) else 0
            is_home = bool(row.get("is_home", False))

            if itype == "goal":
                goals.append({
                    "player": str(row.get("player_name", "")),
                    "minute": minute,
                    "added_time": added,
                    "is_home": is_home,
                    "type": str(row.get("goal_type", "regular")),
                    "assist": str(row.get("assist_player", "")) if pd.notna(row.get("assist_player")) else "",
                })
            elif itype == "card":
                cards.append({
                    "player": str(row.get("player_name", "")),
                    "minute": minute,
                    "added_time": added,
                    "is_home": is_home,
                    "card_type": str(row.get("card_type", row.get("incident_class", "yellow"))),
                })
            elif itype == "substitution":
                player_out = str(row.get("player_name", ""))
                player_in = str(row.get("player_in_name", ""))
                # If player_out is empty, resolve from starters' minutes
                if not player_out:
                    candidates = subbed_off.get((is_home, minute), [])
                    for c in candidates:
                        if c not in used_off:
                            player_out = c
                            used_off.add(c)
                            break
                subs.append({
                    "player_out": player_out,
                    "player_in": player_in,
                    "minute": minute,
                    "added_time": added,
                    "is_home": is_home,
                })
        return {
            "goals": sorted(goals, key=lambda x: x["minute"]),
            "cards": sorted(cards, key=lambda x: x["minute"]),
            "substitutions": sorted(subs, key=lambda x: x["minute"]),
        }
    except Exception as e:
        log.warning(f"Error loading match events: {e}")
        return {"goals": [], "cards": [], "substitutions": []}


def _extract_player_stats_from_lineup_json(ss_match_id: int) -> dict:
    """Extract per-player stats from lineup JSON data (for pre-2022 matches).

    The Sofascore match JSONs store per-player statistics inside
    home_lineup/away_lineup starters/substitutes objects, even for
    old seasons that don't have player_match_stats.parquet data.
    """
    result = {"home": [], "away": []}
    # Sofascore camelCase -> our API format
    stat_map = {
        "rating": "rating",
        "minutesPlayed": "minutes",
        "goals": "goals",
        "totalShots": "total_shots",
        "onTargetScoringAttempt": "shots_on_target",
        "keyPass": "key_passes",
        "totalPass": "total_passes",
        "accuratePass": "accurate_passes",
        "totalTackle": "tackles",
        "wonTackle": "tackles_won",
        "interceptionWon": "interceptions",
        "totalClearance": "clearances",
        "duelWon": "duels_won",
        "duelLost": "duels_lost",
        "aerialWon": "aerial_won",
        "aerialLost": "aerial_lost",
        "fouls": "fouls",
        "wasFouled": "was_fouled",
        "ballRecovery": "ball_recoveries",
        "saves": "saves",
        "bigChanceCreated": "big_chances_created",
        "bigChanceMissed": "big_chances_missed",
        "touches": "touches",
        "totalContest": "dribbles_total",
        "wonContest": "dribbles_won",
        "dispossessed": "dispossessed",
        "totalCross": "total_crosses",
        "accurateCross": "accurate_crosses",
        "totalLongBalls": "total_long_balls",
        "accurateLongBalls": "accurate_long_balls",
        "outfielderBlock": "blocks",
        "totalOffside": "offsides",
    }
    try:
        matches_dir = DATA_DIR / "external" / "sofascore" / "matches"
        for season_dir in sorted(matches_dir.glob("*"), reverse=True):
            match_file = season_dir / f"{ss_match_id}.json"
            if match_file.exists():
                with open(match_file) as f:
                    mdata = json.load(f)
                for side, key in [("home", "home_lineup"), ("away", "away_lineup")]:
                    lineup = mdata.get(key, {})
                    pos_map_inv = {"G": "G", "D": "D", "M": "M", "F": "F"}
                    for p_list, is_starter in [(lineup.get("starters", []), True),
                                                (lineup.get("substitutes", []), False)]:
                        for p in p_list:
                            player = p.get("player", {})
                            stats = p.get("statistics", {})
                            if not player.get("name"):
                                continue
                            entry = {
                                "name": player.get("name", ""),
                                "player_id": player.get("id"),
                                "position": p.get("position", ""),
                                "shirt_number": player.get("jerseyNumber") or p.get("shirtNumber"),
                                "is_starter": is_starter,
                                "minutes": 0,
                                "rating": None,
                                "goals": 0,
                                "assists": 0,
                                "xg": None,
                                "xa": None,
                                "xgot": None,
                                "total_shots": 0,
                                "shots_on_target": 0,
                                "key_passes": 0,
                                "accurate_passes": 0,
                                "total_passes": 0,
                                "tackles": 0,
                                "interceptions": 0,
                                "clearances": 0,
                                "duels_won": 0,
                                "duels_lost": 0,
                                "aerial_won": 0,
                                "aerial_lost": 0,
                                "fouls": 0,
                                "was_fouled": 0,
                                "ball_recoveries": 0,
                                "saves": 0,
                                "big_chances_created": 0,
                                "big_chances_missed": 0,
                                "touches": 0,
                                "blocks": 0,
                            }
                            for ss_key, our_key in stat_map.items():
                                val = stats.get(ss_key)
                                if val is not None:
                                    if our_key == "rating":
                                        entry["rating"] = round(float(val), 1)
                                    elif our_key == "minutes":
                                        entry["minutes"] = int(val)
                                    else:
                                        entry[our_key] = int(val) if isinstance(val, (int, float)) else 0
                            result[side].append(entry)
                # Sort: starters first, then by position
                for side in ["home", "away"]:
                    pos_order = {"G": 0, "D": 1, "M": 2, "F": 3}
                    result[side].sort(
                        key=lambda x: (0 if x["is_starter"] else 1,
                                       pos_order.get(x.get("position", ""), 9))
                    )
                break
    except Exception as e:
        log.warning(f"Error extracting player stats from lineup JSON: {e}")
    return result


def _load_match_lineup(ss_match_id: int) -> dict:
    """Load formation + starting XI for a Sofascore match_id."""
    import glob as _glob
    result = {"home": {"formation": "", "starters": [], "bench": []},
              "away": {"formation": "", "starters": [], "bench": []}}
    try:
        # Try match JSON files. EPL JSONs live in matches_premier_league/ alongside SA's matches/.
        match_file = None
        for league_dir_name in ("matches", "matches_premier_league"):
            matches_dir = DATA_DIR / "external" / "sofascore" / league_dir_name
            if not matches_dir.exists():
                continue
            for season_dir in sorted(matches_dir.glob("*"), reverse=True):
                cand = season_dir / f"{ss_match_id}.json"
                if cand.exists():
                    match_file = cand
                    break
            if match_file:
                break
        if match_file:
            with open(match_file) as f:
                mdata = json.load(f)
            for side in ["home", "away"]:
                lineup = mdata.get(f"{side}_lineup", {})
                result[side]["formation"] = lineup.get("formation", "")
                for p in lineup.get("starters", []):
                    player = p.get("player", {})
                    result[side]["starters"].append({
                        "name": player.get("name", ""),
                        "position": p.get("position", ""),
                        "shirt_number": p.get("shirtNumber", 0),
                    })
                for p in lineup.get("substitutes", []):
                    player = p.get("player", {})
                    result[side]["bench"].append({
                        "name": player.get("name", ""),
                        "position": p.get("position", ""),
                        "shirt_number": p.get("shirtNumber", 0),
                    })
    except Exception as e:
        log.warning(f"Error loading lineup for {ss_match_id}: {e}")

    # Fallback to player_match_stats if no JSON
    if not result["home"]["starters"]:
        try:
            import pandas as pd
            pms = pd.read_parquet(
                DATA_DIR / "external" / "sofascore" / "player_match_stats.parquet"
            )
            match_df = pms[pms["match_id"] == ss_match_id]
            for side, is_home_val in [("home", True), ("away", False)]:
                side_df = match_df[match_df["is_home"] == is_home_val]
                starters = side_df[side_df.get("is_starter", pd.Series([True]*len(side_df))) == True]
                bench = side_df[side_df.get("is_starter", pd.Series([True]*len(side_df))) == False]
                for _, row in starters.iterrows():
                    result[side]["starters"].append({
                        "name": str(row.get("player_name", "")),
                        "position": str(row.get("position", "")),
                        "shirt_number": int(row.get("shirt_number", 0)) if pd.notna(row.get("shirt_number")) else 0,
                    })
                for _, row in bench.iterrows():
                    result[side]["bench"].append({
                        "name": str(row.get("player_name", "")),
                        "position": str(row.get("position", "")),
                        "shirt_number": int(row.get("shirt_number", 0)) if pd.notna(row.get("shirt_number")) else 0,
                    })
        except Exception:
            pass
    return result


@app.route("/api/match-events/<date>/<home_team>/<away_team>")
def api_match_events(date, home_team, away_team):
    """Get detailed match events: formation, lineup, goals, cards, subs.

    URL: /api/match-events/2026-02-09/Roma/Cagliari
    """
    from urllib.parse import unquote
    from config.team_names import normalize_team
    home = normalize_team(unquote(home_team))
    away = normalize_team(unquote(away_team))

    lookup = _get_sofascore_lookup()
    ss_id = lookup.get((date, home, away))

    # League inference (match parquet → fallback to team-name lookup)
    try:
        from config.leagues import infer_league
        league_inferred = infer_league(home, away)
    except Exception:
        league_inferred = "serie_a"

    context = _build_match_context(ss_id, date, home, away, league_inferred)

    if not ss_id:
        # Return empty events gracefully instead of 404
        # (match exists in features.parquet but not in Sofascore data)
        return jsonify({
            "match_id": None,
            "date": date,
            "home_team": home,
            "away_team": away,
            "home_lineup": {"formation": "", "starters": [], "bench": []},
            "away_lineup": {"formation": "", "starters": [], "bench": []},
            "goals": [],
            "cards": [],
            "substitutions": [],
            "context": context,
            "no_detail": True,
        })

    events = _load_match_events(ss_id)
    lineup = _load_match_lineup(ss_id)
    team_stats = _load_match_team_stats(ss_id)

    return jsonify({
        "match_id": ss_id,
        "date": date,
        "home_team": home,
        "away_team": away,
        "home_lineup": lineup["home"],
        "away_lineup": lineup["away"],
        "goals": events["goals"],
        "cards": events["cards"],
        "substitutions": events["substitutions"],
        "team_stats": team_stats,
        "context": context,
    })


@app.route("/api/team/<team_name>/match-history")
def api_team_match_history(team_name):
    """Get enriched match history with events for a team.

    Returns last N matches with formation, goals, cards, subs.
    Query param: ?limit=10 (default 10, max 40)
    """
    from urllib.parse import unquote
    from config.team_names import normalize_team
    team = normalize_team(unquote(team_name))
    limit = min(int(flask_request.args.get("limit", 10)), 40)

    try:
        import pandas as pd
        # Load both leagues' player_match_stats — historic bug was loading only SA
        cols = ["match_id", "date", "home_team", "away_team",
                "home_score", "away_score", "season", "round",
                "is_home", "team"]
        frames = []
        for fname in ("player_match_stats.parquet",
                      "player_match_stats_premier_league.parquet"):
            p = DATA_DIR / "external" / "sofascore" / fname
            if p.exists():
                frames.append(pd.read_parquet(p, columns=cols))
        if not frames:
            return jsonify({"team": team, "matches": []})
        pms = pd.concat(frames, ignore_index=True)
        team_matches = pms[pms["team"].apply(
            lambda t: normalize_team(str(t)) == team
        )].drop_duplicates(subset="match_id")
        team_matches = team_matches.sort_values("date", ascending=False).head(limit)
    except Exception as e:
        return jsonify({"error": f"Failed to load match history: {e}"}), 500

    results = []
    for _, row in team_matches.iterrows():
        ss_id = int(row["match_id"])
        h = normalize_team(str(row["home_team"]))
        a = normalize_team(str(row["away_team"]))
        is_home = bool(row.get("is_home", True))
        events = _load_match_events(ss_id)
        lineup = _load_match_lineup(ss_id)

        side = "home" if is_home else "away"
        results.append({
            "date": str(row["date"])[:10],
            "home_team": h,
            "away_team": a,
            "home_score": int(row.get("home_score", 0)),
            "away_score": int(row.get("away_score", 0)),
            "is_home": is_home,
            "formation": lineup[side].get("formation", ""),
            "starters": lineup[side].get("starters", []),
            "goals": events["goals"],
            "cards": events["cards"],
            "substitutions": events["substitutions"],
            "matchweek": int(row.get("round", 0)) if pd.notna(row.get("round")) else 0,
            "season": str(row.get("season", "")),
        })
    return jsonify({"team": team, "matches": results})


# ---------------------------------------------------------------------------
# Match Detail Page
# ---------------------------------------------------------------------------

@app.route("/match/<date>/<home_team>/<away_team>")
def match_page(date, home_team, away_team):
    """Render match detail page."""
    from urllib.parse import unquote
    return render_template("match.html", active_page="matches",
                           date=date, home_team=unquote(home_team),
                           away_team=unquote(away_team))


@app.route("/api/match/<date>/<home_team>/<away_team>")
def api_match_detail(date, home_team, away_team):
    """Comprehensive match detail: stats, players, shots, events, lineups."""
    import pandas as pd
    from urllib.parse import unquote
    from config.team_names import normalize_team

    home = normalize_team(unquote(home_team))
    away = normalize_team(unquote(away_team))

    result = {
        "match": {"date": date, "home_team": home, "away_team": away},
        "team_stats": {"home": {}, "away": {}},
        "player_stats": {"home": [], "away": []},
        "events": {"goals": [], "cards": [], "substitutions": []},
        "lineups": {"home": {}, "away": {}},
        "shots": [],
        "pre_match": {},
    }

    # --- Sofascore match_id lookup ---
    lookup = _get_sofascore_lookup()
    ss_id = lookup.get((date, home, away))
    swapped = False
    if not ss_id:
        # Try reversed home/away (URL might have teams swapped)
        ss_id = lookup.get((date, away, home))
        if ss_id:
            swapped = True
            home, away = away, home
            result["match"]["home_team"] = home
            result["match"]["away_team"] = away

    # --- Match JSON (team stats, lineups, shotmap) ---
    if ss_id:
        matches_dir = DATA_DIR / "external" / "sofascore" / "matches"
        for season_dir in sorted(matches_dir.glob("*"), reverse=True):
            match_file = season_dir / f"{ss_id}.json"
            if match_file.exists():
                try:
                    with open(match_file) as f:
                        mdata = json.load(f)

                    # Team stats
                    stats_raw = mdata.get("team_stats", {}).get("statistics", [])
                    for period_block in stats_raw:
                        if period_block.get("period") != "ALL":
                            continue
                        for group in period_block.get("groups", []):
                            gname = group.get("groupName", "")
                            for item in group.get("statisticsItems", []):
                                key = item.get("key", item.get("name", ""))
                                result["team_stats"]["home"][key] = {
                                    "value": item.get("homeValue", item.get("home", "")),
                                    "display": item.get("home", ""),
                                    "group": gname,
                                }
                                result["team_stats"]["away"][key] = {
                                    "value": item.get("awayValue", item.get("away", "")),
                                    "display": item.get("away", ""),
                                    "group": gname,
                                }

                    # Lineups from JSON
                    for side in ["home", "away"]:
                        lineup = mdata.get(f"{side}_lineup", {})
                        result["lineups"][side] = {
                            "formation": lineup.get("formation", ""),
                            "starters": [],
                            "bench": [],
                        }
                        for p in lineup.get("starters", []):
                            player = p.get("player", {})
                            stats = p.get("statistics", {})
                            result["lineups"][side]["starters"].append({
                                "name": player.get("name", ""),
                                "short_name": player.get("shortName", ""),
                                "position": p.get("position", ""),
                                "shirt_number": player.get("jerseyNumber", ""),
                                "rating": round(stats.get("rating", 0), 1) if stats.get("rating") else None,
                                "minutes": stats.get("minutesPlayed", 90),
                                "player_id": player.get("id"),
                            })
                        for p in lineup.get("substitutes", []):
                            player = p.get("player", {})
                            stats = p.get("statistics", {})
                            result["lineups"][side]["bench"].append({
                                "name": player.get("name", ""),
                                "short_name": player.get("shortName", ""),
                                "position": p.get("position", ""),
                                "shirt_number": player.get("jerseyNumber", ""),
                                "rating": round(stats.get("rating", 0), 1) if stats.get("rating") else None,
                                "minutes": stats.get("minutesPlayed", 0),
                                "player_id": player.get("id"),
                            })

                    # Shotmap from JSON
                    shotmap = mdata.get("shotmap", {}).get("shotmap", [])
                    for s in shotmap:
                        player = s.get("player", {})
                        coords = s.get("playerCoordinates") or {}
                        gmc = s.get("goalMouthCoordinates") or {}
                        result["shots"].append({
                            "player_name": player.get("name", ""),
                            "player_id": player.get("id"),
                            "is_home": s.get("isHome", False),
                            "shot_x": coords.get("x") if isinstance(coords, dict) else None,
                            "shot_y": coords.get("y") if isinstance(coords, dict) else None,
                            "goal_x": gmc.get("x") if isinstance(gmc, dict) else None,
                            "goal_y": gmc.get("y") if isinstance(gmc, dict) else None,
                            "goal_z": gmc.get("z") if isinstance(gmc, dict) else None,
                            "xg": s.get("xg"),
                            "xgot": s.get("xgot"),
                            "body_part": s.get("bodyPart"),
                            "situation": s.get("situation"),
                            "shot_type": s.get("shotType"),
                            "is_goal": s.get("shotType") == "goal",
                            "time": s.get("time"),
                            "added_time": s.get("addedTime"),
                        })
                except Exception as e:
                    log.warning(f"Error loading match JSON {ss_id}: {e}")
                break

        # Events from incidents parquet
        result["events"] = _load_match_events(ss_id)

    # --- Player match stats (Sofascore) ---
    if ss_id:
        try:
            pms = pd.read_parquet(
                DATA_DIR / "external" / "sofascore" / "player_match_stats.parquet"
            )
            match_pms = pms[pms["match_id"] == ss_id]
            for _, row in match_pms.iterrows():
                side = "home" if row.get("is_home") else "away"
                p = {
                    "name": str(row.get("player_name", "")),
                    "player_id": int(row["player_id"]) if pd.notna(row.get("player_id")) else None,
                    "position": str(row.get("position", "")),
                    "shirt_number": int(row["shirt_number"]) if pd.notna(row.get("shirt_number")) else None,
                    "is_starter": bool(row.get("is_starter", False)),
                    "minutes": int(row["minutes"]) if pd.notna(row.get("minutes")) else 0,
                    "rating": round(float(row["rating"]), 1) if pd.notna(row.get("rating")) else None,
                    "goals": int(row["goals"]) if pd.notna(row.get("goals")) else 0,
                    "assists": int(row["assists"]) if pd.notna(row.get("assists")) else 0,
                    "xg": round(float(row["xg"]), 2) if pd.notna(row.get("xg")) else None,
                    "xa": round(float(row["xa"]), 2) if pd.notna(row.get("xa")) else None,
                    "xgot": round(float(row["xgot"]), 2) if pd.notna(row.get("xgot")) else None,
                    "total_shots": int(row["total_shots"]) if pd.notna(row.get("total_shots")) else 0,
                    "shots_on_target": int(row["shots_on_target"]) if pd.notna(row.get("shots_on_target")) else 0,
                    "key_passes": int(row["key_passes"]) if pd.notna(row.get("key_passes")) else 0,
                    "accurate_passes": int(row["accurate_passes"]) if pd.notna(row.get("accurate_passes")) else 0,
                    "total_passes": int(row["total_passes"]) if pd.notna(row.get("total_passes")) else 0,
                    "tackles": int(row["tackles"]) if pd.notna(row.get("tackles")) else 0,
                    "interceptions": int(row["interceptions"]) if pd.notna(row.get("interceptions")) else 0,
                    "clearances": int(row["clearances"]) if pd.notna(row.get("clearances")) else 0,
                    "duels_won": int(row["duels_won"]) if pd.notna(row.get("duels_won")) else 0,
                    "duels_lost": int(row["duels_lost"]) if pd.notna(row.get("duels_lost")) else 0,
                    "aerial_won": int(row["aerial_won"]) if pd.notna(row.get("aerial_won")) else 0,
                    "aerial_lost": int(row["aerial_lost"]) if pd.notna(row.get("aerial_lost")) else 0,
                    "fouls": int(row["fouls"]) if pd.notna(row.get("fouls")) else 0,
                    "was_fouled": int(row["was_fouled"]) if pd.notna(row.get("was_fouled")) else 0,
                    "ball_recoveries": int(row["ball_recoveries"]) if pd.notna(row.get("ball_recoveries")) else 0,
                    "saves": int(row["saves"]) if pd.notna(row.get("saves")) else 0,
                    "big_chances_created": int(row["big_chances_created"]) if pd.notna(row.get("big_chances_created")) else 0,
                    "big_chances_missed": int(row["big_chances_missed"]) if pd.notna(row.get("big_chances_missed")) else 0,
                }
                result["player_stats"][side].append(p)

            # Sort: starters first (by position), then subs
            for side in ["home", "away"]:
                pos_order = {"G": 0, "D": 1, "M": 2, "F": 3}
                result["player_stats"][side].sort(
                    key=lambda x: (0 if x["is_starter"] else 1,
                                   pos_order.get(x.get("position", ""), 9))
                )
        except Exception as e:
            log.warning(f"Error loading player match stats: {e}")

    # --- Fallback: extract player stats from lineup JSON (for pre-2022 matches) ---
    if not result["player_stats"]["home"] and not result["player_stats"]["away"]:
        for side, lineup_key in [("home", "home"), ("away", "away")]:
            lineup = result["lineups"].get(side, {})
            all_players = lineup.get("starters", []) + lineup.get("bench", [])
            for p in all_players:
                if not p.get("name"):
                    continue
                # Re-read from JSON lineup (which has statistics embedded)
                result["player_stats"][side].append(p)
        # If lineups have stats from JSON, re-extract with proper mapping
        if result["lineups"]["home"].get("starters"):
            result["player_stats"] = _extract_player_stats_from_lineup_json(ss_id)

    # --- Fallback: build team stats from player stats ---
    # Keys must match the renderStats() template in match.html
    if not result["team_stats"]["home"] and result["player_stats"]["home"]:
        for side in ["home", "away"]:
            players = result["player_stats"][side]
            if not players:
                continue
            total_shots = sum(p.get("total_shots", 0) for p in players)
            shots_on = sum(p.get("shots_on_target", 0) for p in players)
            shots_off = total_shots - shots_on
            total_passes = sum(p.get("total_passes", 0) for p in players)
            acc_passes = sum(p.get("accurate_passes", 0) for p in players)
            tackles = sum(p.get("tackles", 0) for p in players)
            clearances = sum(p.get("clearances", 0) for p in players)
            interceptions = sum(p.get("interceptions", 0) for p in players)
            fouls = sum(p.get("fouls", 0) for p in players)
            saves = sum(p.get("saves", 0) for p in players)
            duels_won = sum(p.get("duels_won", 0) for p in players)
            duels_lost = sum(p.get("duels_lost", 0) for p in players)
            aerial_won = sum(p.get("aerial_won", 0) for p in players)
            aerial_lost = sum(p.get("aerial_lost", 0) for p in players)
            ball_rec = sum(p.get("ball_recoveries", 0) for p in players)
            key_passes = sum(p.get("key_passes", 0) for p in players)
            touches = sum(p.get("touches", 0) for p in players)

            pass_pct = round(acc_passes * 100 / total_passes) if total_passes else 0
            duels_total = duels_won + duels_lost
            duels_pct = round(duels_won * 100 / duels_total) if duels_total else 0
            aerial_total = aerial_won + aerial_lost
            aerial_pct = round(aerial_won * 100 / aerial_total) if aerial_total else 0

            stats = {}
            if total_shots:
                stats["totalShotsOnGoal"] = {"value": total_shots, "display": str(total_shots), "group": "Shots"}
                stats["shotsOnGoal"] = {"value": shots_on, "display": str(shots_on), "group": "Shots"}
                if shots_off:
                    stats["shotsOffGoal"] = {"value": shots_off, "display": str(shots_off), "group": "Shots"}
            if total_passes:
                stats["passes"] = {"value": total_passes, "display": str(total_passes), "group": "Passes"}
                stats["accuratePasses"] = {"value": acc_passes, "display": f"{acc_passes}/{total_passes} ({pass_pct}%)", "group": "Passes"}
            if tackles:
                stats["totalTackle"] = {"value": tackles, "display": str(tackles), "group": "Defence"}
            if clearances:
                stats["totalClearance"] = {"value": clearances, "display": str(clearances), "group": "Defence"}
            if interceptions:
                stats["interceptionWon"] = {"value": interceptions, "display": str(interceptions), "group": "Defence"}
            if ball_rec:
                stats["ballRecovery"] = {"value": ball_rec, "display": str(ball_rec), "group": "Defence"}
            if fouls:
                stats["fouls"] = {"value": fouls, "display": str(fouls), "group": "Duels"}
            if duels_total:
                stats["duelWonPercent"] = {"value": f"{duels_pct}%", "display": f"{duels_pct}%", "group": "Duels"}
            if aerial_total:
                stats["aerialDuelsPercentage"] = {"value": f"{aerial_won}/{aerial_total} ({aerial_pct}%)", "display": f"{aerial_won}/{aerial_total} ({aerial_pct}%)", "group": "Duels"}
            if saves:
                stats["goalkeeperSaves"] = {"value": saves, "display": str(saves), "group": "Goalkeeping"}
            if touches:
                stats["touches"] = {"value": touches, "display": str(touches), "group": "Passes"}
            result["team_stats"][side] = stats

    # --- Match metadata from features.parquet ---
    try:
        fp_path = DATA_DIR / "features" / "features.parquet"
        if fp_path.exists():
            import pyarrow.parquet as pq
            schema = pq.read_schema(fp_path)
            col_names = set(schema.names)

            want = ["match_id", "match_date", "season", "matchweek", "venue",
                    "referee", "home_score", "away_score",
                    "odds_PSH", "odds_PSD", "odds_PSA",
                    "odds_AvgH", "odds_AvgD", "odds_AvgA",
                    "weather_temperature_2m_max", "weather_wind_speed_10m_max",
                    "weather_precipitation_sum",
                    "h2h_matches_played", "h2h_home_wins", "h2h_away_wins",
                    "h2h_btts_rate", "h2h_over25_rate",
                    "home_league_pos", "away_league_pos",
                    "is_derby", "home_elo", "away_elo",
                    "ref_avg_yellows", "ref_avg_fouls",
                    "attendance", "kickoff_time",
                    # Form: rolling 5-match averages
                    "home_roll_5_win_rate", "away_roll_5_win_rate",
                    "home_roll_5_goals_scored", "away_roll_5_goals_scored",
                    "home_roll_5_goals_conceded", "away_roll_5_goals_conceded",
                    "home_roll_5_clean_sheet", "away_roll_5_clean_sheet",
                    "home_roll_5_shots_on_target", "away_roll_5_shots_on_target",
                    "home_roll_5_corners", "away_roll_5_corners",
                    "home_roll_5_yellow_cards", "away_roll_5_yellow_cards",
                    "home_roll_5_points", "away_roll_5_points",
                    # Form: points last 3 and 5
                    "home_form_points_3", "away_form_points_3",
                    "home_form_points_5", "away_form_points_5",
                    # Streaks
                    "home_win_streak", "away_win_streak",
                    "home_loss_streak", "away_loss_streak",
                    "home_unbeaten_run", "away_unbeaten_run",
                    "home_winless_run", "away_winless_run",
                    "home_scoring_streak", "away_scoring_streak",
                    "home_clean_sheet_streak", "away_clean_sheet_streak",
                    # Goal difference
                    "home_gd_roll_5", "away_gd_roll_5",
                    # xG form
                    "home_ss_roll_xg", "away_ss_roll_xg",
                    "home_ss_roll_goals", "away_ss_roll_goals",
                    "home_ss_roll_rating", "away_ss_roll_rating",
                    "home_ss_roll_total_shots", "away_ss_roll_total_shots",
                    "home_ss_roll_shots_on_target", "away_ss_roll_shots_on_target",
                    "home_ss_roll_pass_accuracy", "away_ss_roll_pass_accuracy",
                    "home_ss_roll_territory_ratio", "away_ss_roll_territory_ratio",
                    "home_xg_overperformance_roll_5", "away_xg_overperformance_roll_5",
                    # Venue-specific form
                    "home_venue_roll_5_goals_scored", "away_venue_roll_5_goals_scored",
                    "home_venue_roll_5_goals_conceded", "away_venue_roll_5_goals_conceded",
                    "home_venue_roll_5_points", "away_venue_roll_5_points",
                    "home_venue_roll_5_xg_for", "away_venue_roll_5_xg_for",
                    "home_venue_roll_5_xg_against", "away_venue_roll_5_xg_against",
                    # Formation
                    "home_formation", "away_formation",
                    "home_formation_flexibility", "away_formation_flexibility",
                    "formation_matchup_home_rate", "formation_matchup_draw_rate"]
            avail = [c for c in want if c in col_names]
            df = pd.read_parquet(fp_path, columns=avail)

            # Try to find the match row
            match_row = None
            if "match_date" in df.columns:
                df["_date_str"] = df["match_date"].astype(str).str[:10]
                candidates = df[df["_date_str"] == date]
                for _, row in candidates.iterrows():
                    mid = str(row.get("match_id", ""))
                    if home.lower() in mid.lower() and away.lower() in mid.lower():
                        match_row = row
                        break
                    # Also try match_id patterns
                    if f"{home}" in mid or f"{away}" in mid:
                        match_row = row
                        break

            if match_row is not None:
                pre = {}
                for col in avail:
                    val = match_row.get(col)
                    if pd.notna(val):
                        if isinstance(val, (int, float)):
                            pre[col] = round(float(val), 3) if isinstance(val, float) else int(val)
                        else:
                            pre[col] = str(val)
                result["pre_match"] = pre
                # Enrich match info
                if "home_score" in pre:
                    result["match"]["home_score"] = int(float(pre["home_score"]))
                if "away_score" in pre:
                    result["match"]["away_score"] = int(float(pre["away_score"]))
                if "matchweek" in pre:
                    result["match"]["matchweek"] = pre["matchweek"]
                if "season" in pre:
                    result["match"]["season"] = pre["season"]
                if "venue" in pre:
                    result["match"]["venue"] = pre["venue"]
                if "referee" in pre:
                    result["match"]["referee"] = pre["referee"]
                if "attendance" in pre:
                    result["match"]["attendance"] = pre["attendance"]
    except Exception as e:
        log.warning(f"Error loading pre-match features: {e}")

    # --- Score from Sofascore team stats if not from features ---
    if ss_id and "home_score" not in result["match"]:
        try:
            # Try SA file first, then EPL — match_id is unique across leagues,
            # so only one parquet has rows for any given ss_id. (Mirrors the
            # dual-league idiom in _load_match_team_stats; EPL match-detail
            # pages showed no score before this.)
            row = None
            for _fname in ("match_team_stats.parquet",
                           "match_team_stats_premier_league.parquet"):
                _p = DATA_DIR / "external" / "sofascore" / _fname
                if not _p.exists():
                    continue
                mts = _read_parquet_cached(_p)
                cand = mts[(mts["match_id"] == ss_id) & (mts["is_home"] == True)]
                if not cand.empty:
                    row = cand
                    break
            if row is not None and not row.empty:
                r = row.iloc[0]
                result["match"]["home_score"] = int(r["home_score"]) if pd.notna(r.get("home_score")) else None
                result["match"]["away_score"] = int(r["away_score"]) if pd.notna(r.get("away_score")) else None
                result["match"]["matchweek"] = int(r["round"]) if pd.notna(r.get("round")) else None
                result["match"]["season"] = str(r["season"]) if pd.notna(r.get("season")) else None
        except Exception:
            pass

    return jsonify(result)


# ---------------------------------------------------------------------------
# Players Pages
# ---------------------------------------------------------------------------

@app.route("/players")
def players_page():
    """Render players overview page."""
    return render_template("players.html", active_page="players")


@app.route("/player/<team_name>/<player_name>")
def player_page(team_name, player_name):
    """Render individual player detail page."""
    from urllib.parse import unquote
    return render_template("player.html", active_page="players",
                           team=unquote(team_name), player=unquote(player_name))


@app.route("/api/players")
def api_players():
    """List all players grouped by team with basic stats.

    Uses Sofascore player_match_stats as PRIMARY source (full names, real stats),
    enriched with Transfermarkt market values and xG profiles.
    """
    import pandas as pd
    from config.team_names import normalize_team
    from scraper.lineup_fetcher import normalize_player_name

    teams_data = []

    # --- Primary: Sofascore player match stats (current season, both leagues) ---
    try:
        frames = []
        for fname in ("player_match_stats.parquet",
                      "player_match_stats_premier_league.parquet"):
            p = DATA_DIR / "external" / "sofascore" / fname
            if p.exists():
                frames.append(pd.read_parquet(p))
        if not frames:
            return jsonify({"teams": []})
        pms = pd.concat(frames, ignore_index=True)
        if "season" in pms.columns:
            pms = pms[pms["season"].astype(str).str.startswith(get_current_season())]
    except Exception:
        return jsonify({"teams": []})

    pms["_team"] = pms["team"].apply(lambda t: normalize_team(str(t)))

    # --- Player metadata (age, nationality, height, market value from Sofascore JSONs) ---
    # Keyed by player_id — perfect match with player_match_stats
    player_metadata = _load_json(DATA_DIR / "features" / "player_metadata.json", {})

    # --- xG profiles ---
    xg_profiles = _load_json(DATA_DIR / "features" / "player_xg_profiles.json", {})

    # --- FBref stats (shooting, passing, GCA) ---
    fbref_shooting = {}
    fbref_passing = {}
    fbref_gca = {}
    try:
        for stat_type, target_dict, col_map in [
            ("fbref_stats_shooting", fbref_shooting, {
                "Standard_Sh": "shots", "Standard_SoT": "shots_on_target",
                "Standard_SoT%": "shot_accuracy", "Standard_G/Sh": "goals_per_shot",
                "Unnamed: 7_level_0_90s": "nineties",
            }),
            ("fbref_stats_passing", fbref_passing, {
                "Total_Cmp": "passes_completed", "Total_Att": "passes_attempted",
                "Total_Cmp%": "pass_pct",
            }),
            ("fbref_stats_gca", fbref_gca, {
                "SCA_SCA": "sca", "SCA_SCA90": "sca_per_90",
                "GCA_GCA": "gca", "GCA_GCA90": "gca_per_90",
            }),
        ]:
            df = pd.read_parquet(DATA_DIR / "parsed" / f"{stat_type}.parquet")
            if "season" in df.columns:
                df = df[df["season"].astype(str).str.startswith(get_current_season())]
            for _, row in df.iterrows():
                pname = str(row.get("Unnamed: 1_level_0_Player", ""))
                squad = str(row.get("Unnamed: 4_level_0_Squad", ""))
                key = f"{pname}|{normalize_team(squad)}"
                entry = {}
                for src_col, dst_col in col_map.items():
                    val = row.get(src_col)
                    if pd.notna(val):
                        try:
                            entry[dst_col] = round(float(val), 2)
                        except (ValueError, TypeError):
                            pass
                if entry:
                    target_dict[key] = entry
    except Exception:
        pass  # FBref data is optional enrichment

    # --- Aggregate per player per team from Sofascore ---
    for team_name in sorted(pms["_team"].unique()):
        team_pms = pms[pms["_team"] == team_name]
        enriched = []

        for player_name in team_pms["player_name"].unique():
            player_rows = team_pms[team_pms["player_name"] == player_name]
            pnorm = normalize_player_name(str(player_name))

            entry = {
                "name": str(player_name),
                "position": "",
                "number": None,
                "age": None,
                "nationality": None,
                "market_value": None,
                "matches": len(player_rows),
                "starts": int(player_rows["is_starter"].sum()) if "is_starter" in player_rows else 0,
                "goals": int(player_rows["goals"].sum()) if "goals" in player_rows else 0,
                "assists": int(player_rows["assists"].sum()) if "assists" in player_rows else 0,
                "minutes": int(player_rows["minutes"].sum()) if "minutes" in player_rows else 0,
                "rating": None,
                "xg_per_90": None,
                "xa_per_90": None,
            }

            # Position from Sofascore (most common)
            if "position" in player_rows.columns:
                pos = player_rows["position"].mode()
                if len(pos) and pd.notna(pos.iloc[0]):
                    entry["position"] = str(pos.iloc[0])

            # Shirt number from Sofascore (most common)
            if "shirt_number" in player_rows.columns:
                sn = player_rows["shirt_number"].mode()
                if len(sn) and pd.notna(sn.iloc[0]):
                    entry["number"] = int(sn.iloc[0])

            # Average rating
            if "rating" in player_rows.columns:
                valid_r = player_rows["rating"].dropna()
                if len(valid_r):
                    entry["rating"] = round(float(valid_r.mean()), 2)

            # --- Enrich from player_metadata (age, nationality, height, market value) ---
            # player_id from Sofascore → lookup in player_metadata.json
            if "player_id" in player_rows.columns:
                pid = str(int(player_rows["player_id"].iloc[0]))
                pm = player_metadata.get(pid, {})
                if pm:
                    if not entry["age"] and pm.get("age"):
                        entry["age"] = pm["age"]
                    if not entry["nationality"] and pm.get("nationality"):
                        entry["nationality"] = pm["nationality"]
                    if not entry["market_value"] and pm.get("market_value"):
                        entry["market_value"] = pm["market_value"]
                    if not entry["position"] and pm.get("position"):
                        entry["position"] = pm["position"]
                    if pm.get("height"):
                        entry["height"] = pm["height"]

            # --- xG profile ---
            profile = xg_profiles.get(f"{player_name}|{team_name}", {})
            if profile:
                entry["xg_per_90"] = round(profile.get("xg_per_90", 0), 3) if profile.get("xg_per_90") else None
                entry["xa_per_90"] = round(profile.get("xa_per_90", 0), 3) if profile.get("xa_per_90") else None

            # --- FBref enrichment (shooting, passing, GCA) ---
            fbref_key = f"{player_name}|{team_name}"
            fb_sh = fbref_shooting.get(fbref_key, {})
            fb_pa = fbref_passing.get(fbref_key, {})
            fb_gc = fbref_gca.get(fbref_key, {})
            if fb_sh or fb_pa or fb_gc:
                entry["fbref"] = {**fb_sh, **fb_pa, **fb_gc}

            enriched.append(entry)

        # Sort: starters by minutes desc, then subs
        enriched.sort(key=lambda x: -(x.get("minutes") or 0))

        teams_data.append({
            "team": team_name,
            "player_count": len(enriched),
            "players": enriched,
        })

    league_filter = _get_league_filter()
    if league_filter:
        teams_data = [t for t in teams_data if _team_belongs_to_league(t.get("team", ""), league_filter)]

    return jsonify({"teams": teams_data})


@app.route("/api/player/<team_name>/<player_name>")
def api_player_detail(team_name, player_name):
    """Detailed player stats: profile, season stats, match log.

    Accepts full Sofascore names (primary) and falls back to fuzzy matching
    for abbreviated squad names.
    """
    import pandas as pd
    from urllib.parse import unquote
    from config.team_names import normalize_team
    from scraper.lineup_fetcher import normalize_player_name

    team = normalize_team(unquote(team_name))
    pname = unquote(player_name)
    pnorm = normalize_player_name(pname)

    result = {
        "player": {"name": pname, "team": team},
        "season_stats": {},
        "match_log": [],
        "career": {},
        "market_value_history": [],
    }

    # --- Load Sofascore data (primary source) — both leagues ---
    player_rows = pd.DataFrame()
    try:
        frames = []
        for fname in ("player_match_stats.parquet",
                      "player_match_stats_premier_league.parquet"):
            p = DATA_DIR / "external" / "sofascore" / fname
            if p.exists():
                frames.append(pd.read_parquet(p))
        if not frames:
            raise FileNotFoundError("no player_match_stats parquets")
        pms = pd.concat(frames, ignore_index=True)
        team_pms = pms[pms["team"].apply(lambda t: normalize_team(str(t)) == team)]

        # Try exact normalized match first
        player_rows = team_pms[team_pms["player_name"].apply(
            lambda n: normalize_player_name(str(n)) == pnorm
        )]

        # If no match, try surname-based fuzzy (handles abbreviated names from old links)
        if player_rows.empty:
            parts = pnorm.split()
            if len(parts) >= 2:
                surname = " ".join(parts[1:])
                initial = parts[0][:1]
                for ss_name in team_pms["player_name"].unique():
                    ss_norm = normalize_player_name(str(ss_name))
                    ss_parts = ss_norm.split()
                    if len(ss_parts) >= 2:
                        ss_surname = " ".join(ss_parts[1:])
                        if ss_surname == surname and ss_parts[0][:1] == initial:
                            player_rows = team_pms[team_pms["player_name"] == ss_name]
                            # Update display name to full Sofascore name
                            result["player"]["name"] = str(ss_name)
                            pname = str(ss_name)
                            pnorm = normalize_player_name(pname)
                            break

        if not player_rows.empty:
            player_rows = player_rows.sort_values("date", ascending=False)
            latest = player_rows.iloc[0]

            # Profile from Sofascore
            if pd.notna(latest.get("shirt_number")):
                result["player"]["number"] = int(latest["shirt_number"])
            if pd.notna(latest.get("position")):
                result["player"]["position"] = str(latest["position"])

            # Season stats (current season)
            current = player_rows[
                player_rows["season"].astype(str).str.startswith(get_current_season())
            ] if "season" in player_rows.columns else player_rows.head(30)
            if not current.empty:
                result["season_stats"] = {
                    "matches": len(current),
                    "starts": int(current["is_starter"].sum()) if "is_starter" in current else 0,
                    "minutes": int(current["minutes"].sum()) if "minutes" in current else 0,
                    "goals": int(current["goals"].sum()) if "goals" in current else 0,
                    "assists": int(current["assists"].sum()) if "assists" in current else 0,
                    "xg": round(float(current["xg"].sum()), 2) if "xg" in current and current["xg"].notna().any() else None,
                    "xa": round(float(current["xa"].sum()), 2) if "xa" in current and current["xa"].notna().any() else None,
                    "avg_rating": round(float(current["rating"].dropna().mean()), 2) if "rating" in current and current["rating"].notna().any() else None,
                    "total_shots": int(current["total_shots"].sum()) if "total_shots" in current else 0,
                    "shots_on_target": int(current["shots_on_target"].sum()) if "shots_on_target" in current else 0,
                    "key_passes": int(current["key_passes"].sum()) if "key_passes" in current else 0,
                    "big_chances_created": int(current["big_chances_created"].sum()) if "big_chances_created" in current else 0,
                    "tackles": int(current["tackles"].sum()) if "tackles" in current else 0,
                    "interceptions": int(current["interceptions"].sum()) if "interceptions" in current else 0,
                    "ball_recoveries": int(current["ball_recoveries"].sum()) if "ball_recoveries" in current else 0,
                    "duels_won": int(current["duels_won"].sum()) if "duels_won" in current else 0,
                    "duels_lost": int(current["duels_lost"].sum()) if "duels_lost" in current else 0,
                    "fouls": int(current["fouls"].sum()) if "fouls" in current else 0,
                    "was_fouled": int(current["was_fouled"].sum()) if "was_fouled" in current else 0,
                    "accurate_passes": int(current["accurate_passes"].sum()) if "accurate_passes" in current else 0,
                    "total_passes": int(current["total_passes"].sum()) if "total_passes" in current else 0,
                    "saves": int(current["saves"].sum()) if "saves" in current else 0,
                    "big_chances_missed": int(current["big_chances_missed"].sum()) if "big_chances_missed" in current else 0,
                    "dribbles_won": int(current["contest_won"].sum()) if "contest_won" in current else 0,
                    "dribbles_total": int(current["contest_total"].sum()) if "contest_total" in current else 0,
                    "progressive_carries": int(current["progressive_carries"].sum()) if "progressive_carries" in current else 0,
                    "aerial_won": int(current["aerial_won"].sum()) if "aerial_won" in current else 0,
                    "aerial_lost": int(current["aerial_lost"].sum()) if "aerial_lost" in current else 0,
                    "touches": int(current["touches"].sum()) if "touches" in current else 0,
                    "clearances": int(current["clearances"].sum()) if "clearances" in current else 0,
                    "blocks": int(current["blocks"].sum()) if "blocks" in current else 0,
                    "offsides": int(current["offsides"].sum()) if "offsides" in current else 0,
                }

            # Match log (last 40 matches)
            for _, row in player_rows.head(40).iterrows():
                entry = {
                    "date": str(row.get("date", ""))[:10],
                    "season": str(row.get("season", "")),
                    "opponent": str(row.get("opponent", "")),
                    "is_home": bool(row.get("is_home", False)),
                    "is_starter": bool(row.get("is_starter", False)),
                    "minutes": int(row["minutes"]) if pd.notna(row.get("minutes")) else 0,
                    "rating": round(float(row["rating"]), 1) if pd.notna(row.get("rating")) else None,
                    "goals": int(row["goals"]) if pd.notna(row.get("goals")) else 0,
                    "assists": int(row["assists"]) if pd.notna(row.get("assists")) else 0,
                    "xg": round(float(row["xg"]), 2) if pd.notna(row.get("xg")) else None,
                    "xa": round(float(row["xa"]), 2) if pd.notna(row.get("xa")) else None,
                    "total_shots": int(row["total_shots"]) if pd.notna(row.get("total_shots")) else 0,
                    "key_passes": int(row["key_passes"]) if pd.notna(row.get("key_passes")) else 0,
                    "tackles": int(row["tackles"]) if pd.notna(row.get("tackles")) else 0,
                    "interceptions": int(row["interceptions"]) if pd.notna(row.get("interceptions")) else 0,
                    "accurate_passes": int(row["accurate_passes"]) if pd.notna(row.get("accurate_passes")) else 0,
                    "total_passes": int(row["total_passes"]) if pd.notna(row.get("total_passes")) else 0,
                    "home_team": str(row.get("home_team", "")),
                    "away_team": str(row.get("away_team", "")),
                    "home_score": int(row["home_score"]) if pd.notna(row.get("home_score")) else None,
                    "away_score": int(row["away_score"]) if pd.notna(row.get("away_score")) else None,
                }
                result["match_log"].append(entry)
    except Exception as e:
        log.warning(f"Error loading player Sofascore stats: {e}")

    # --- Enrich profile from player_metadata.json (age, nationality, height, market value) ---
    player_metadata = _load_json(DATA_DIR / "features" / "player_metadata.json", {})
    if not player_rows.empty and "player_id" in player_rows.columns:
        pid = str(int(player_rows["player_id"].iloc[0]))
        pm = player_metadata.get(pid, {})
        if pm:
            if not result["player"].get("age") and pm.get("age"):
                result["player"]["age"] = pm["age"]
            if not result["player"].get("nationality") and pm.get("nationality"):
                result["player"]["nationality"] = pm["nationality"]
            if pm.get("height"):
                result["player"]["height"] = pm["height"]
            if pm.get("market_value"):
                result["player"]["market_value"] = pm["market_value"]

    # --- Career xG profile ---
    xg_profiles = _load_json(DATA_DIR / "features" / "player_xg_profiles.json", {})
    profile = xg_profiles.get(f"{pname}|{team}", {})
    if profile:
        result["career"] = {
            "total_matches": profile.get("matches_played", 0),
            "total_minutes": profile.get("total_minutes", 0),
            "total_xg": round(profile.get("total_xg", 0), 2),
            "total_xa": round(profile.get("total_xa", 0), 2),
            "total_goals": profile.get("total_goals", 0),
            "total_assists": profile.get("total_assists", 0),
            "xg_per_90": round(profile.get("xg_per_90", 0), 3),
            "xa_per_90": round(profile.get("xa_per_90", 0), 3),
            "recent_form_xg": round(profile.get("recent_form_xg", 0), 3),
            "recent_form_xa": round(profile.get("recent_form_xa", 0), 3),
            "position": profile.get("position"),
        }

    # --- Full career history (all teams played for) ---
    try:
        from scripts.analysis.player_history import build_player_history, get_player_profile
        history = build_player_history()
        profile = get_player_profile(pname, history)
        if profile:
            result["career_history"] = profile.get("career", [])
            result["career_gaps"] = profile.get("career_gaps", [])
            if profile.get("nationality") and not result["player"].get("nationality"):
                result["player"]["nationality"] = profile["nationality"]
            if profile.get("market_value_eur"):
                result["player"]["market_value_eur"] = profile["market_value_eur"]
            if profile.get("transfer_fee_eur"):
                result["player"]["transfer_fee_eur"] = profile["transfer_fee_eur"]
        else:
            # Try fuzzy match by last name
            query = pname.split()[-1].lower() if pname else ""
            for name, teams in history.items():
                if query in name.lower():
                    prof = get_player_profile(name, history)
                    if prof:
                        result["career_history"] = prof.get("career", [])
                        if prof.get("nationality") and not result["player"].get("nationality"):
                            result["player"]["nationality"] = prof["nationality"]
                        if prof.get("market_value_eur"):
                            result["player"]["market_value_eur"] = prof["market_value_eur"]
                        break
    except Exception as e:
        log.debug("Career history lookup failed: %s", e)

    # --- Market value history (across all teams, all seasons) ---
    try:
        mv_dir = DATA_DIR / "external" / "transfermarkt"
        if mv_dir.exists():
            for mv_file in sorted(mv_dir.glob("market_values_*.parquet")):
                if "backup" in mv_file.stem:
                    continue
                season = mv_file.stem.replace("market_values_", "").replace("_", "-")
                mvdf = pd.read_parquet(mv_file)
                # Search across ALL teams (not just current) for this player
                pmv = mvdf[mvdf["player_name"].apply(
                    lambda n: normalize_player_name(str(n)) == pnorm
                )]
                if pmv.empty:
                    # Fuzzy by surname
                    surname = pnorm.split()[-1] if pnorm else ""
                    pmv = mvdf[mvdf["player_name"].apply(
                        lambda n: surname in normalize_player_name(str(n))
                    )]
                if not pmv.empty:
                    row = pmv.iloc[0]
                    val = row.get("market_value_eur")
                    if pd.notna(val):
                        result["market_value_history"].append({
                            "season": season,
                            "value": int(val),
                            "team": str(row.get("team", "")),
                        })
    except Exception:
        pass

    return jsonify(result)


# ---------------------------------------------------------------------------
# API: Notification Test
# ---------------------------------------------------------------------------

@app.route("/api/notifications/test", methods=["POST"])
def api_notifications_test():
    """Send a test notification on all configured channels."""
    try:
        from scripts.pipeline.notify import notify, notify_status

        data = flask_request.get_json(silent=True) or {}
        message = data.get("message", "Test notification from SerieAI dashboard")

        status = notify_status()
        results = notify(message, title="SerieAI Test", level="info")

        return jsonify({
            "ok": True,
            "channels": {
                ch: {"configured": status[ch]["configured"], "sent": results.get(ch, False)}
                for ch in status
            },
        })
    except Exception as e:
        log.error("Notification test failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/notifications/status")
def api_notifications_status():
    """Return configuration status for each notification channel."""
    try:
        from scripts.pipeline.notify import notify_status
        return jsonify({"ok": True, "channels": notify_status()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/notifications/preferences")
def api_notifications_preferences_get():
    """Return current notification preferences."""
    try:
        from scripts.pipeline.notify import load_preferences
        return jsonify({"ok": True, "preferences": load_preferences()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/notifications/preferences", methods=["POST"])
def api_notifications_preferences_set():
    """Update notification preferences."""
    try:
        from scripts.pipeline.notify import load_preferences, save_preferences
        data = flask_request.get_json(silent=True) or {}
        # Merge incoming data with existing preferences
        prefs = load_preferences()
        if "channels" in data:
            prefs["channels"].update(data["channels"])
        if "categories" in data:
            for cat, ch_map in data["categories"].items():
                if cat in prefs["categories"]:
                    prefs["categories"][cat].update(ch_map)
                else:
                    prefs["categories"][cat] = ch_map
        if "quiet_hours" in data:
            prefs.setdefault("quiet_hours", {}).update(data["quiet_hours"])
        if "mute_all" in data:
            prefs["mute_all"] = bool(data["mute_all"])
        if "sound" in data:
            prefs["sound"] = data["sound"]
        ok = save_preferences(prefs)
        return jsonify({"ok": ok, "preferences": prefs})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/notifications/history")
def api_notifications_history():
    """Return recent notification history."""
    try:
        from scripts.pipeline.notify import get_notification_history
        limit = int(flask_request.args.get("limit", 50))
        limit = min(limit, 200)
        history = get_notification_history(limit=limit)
        return jsonify({"ok": True, "history": history, "count": len(history)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/notifications/history", methods=["DELETE"])
def api_notifications_history_clear():
    """Clear all notification history."""
    try:
        from scripts.pipeline.notify import clear_notification_history
        ok = clear_notification_history()
        return jsonify({"ok": ok})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/notifications/stats")
def api_notifications_stats():
    """Return notification statistics."""
    try:
        from scripts.pipeline.notify import get_notification_stats
        stats = get_notification_stats()
        return jsonify({"ok": True, "stats": stats})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/notifications/digest", methods=["POST"])
def api_notifications_digest():
    """Send a daily digest notification now."""
    try:
        from scripts.pipeline.notify import notify, get_notification_stats
        stats = get_notification_stats()
        today = stats.get("today", 0)
        week = stats.get("this_week", 0)
        by_cat = stats.get("by_category", {})

        lines = [f"Today: {today} notifications, This week: {week}"]
        if by_cat:
            breakdown = ", ".join(f"{k}: {v}" for k, v in sorted(by_cat.items()))
            lines.append(f"By category: {breakdown}")

        message = "\n".join(lines)
        results = notify(message, title="Daily Digest", level="info", category="system")
        return jsonify({"ok": True, "results": results})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Medium-Impact Betting APIs (Mar 24 2026)
# ---------------------------------------------------------------------------

@app.route("/api/best-bets")
def api_best_bets():
    """Top bets right now — the 5-second answer to 'what should I bet?'

    Returns the best value bets ranked by confidence × edge, with
    clear action: match, selection, odds, bookmaker, stake, edge.
    Designed for a quick-glance widget on the dashboard.

    Query params:
        league: optional league filter (e.g. 'premier_league', 'epl', 'serie_a')
    """
    league_filter = _get_league_filter()

    try:
        # Load latest bets from unified report
        report = _load_json(BETTING_DIR / "unified_report.json")
        bets = _filter_by_league(report.get("bets", []), league_filter)

        # Also check edge scan for newer discoveries
        scan = _load_json(BETTING_DIR / "edge_scan_latest.json")
        scan_bets = _filter_by_league(scan.get("value_bets", []) if scan else [], league_filter)

        # Build a unified best-bets list
        best = []
        seen = set()

        # Score each bet: confidence (0-100) × edge (%)
        for b in bets:
            match = b.get("match", "")
            sel = b.get("selection", "")
            key = f"{match}_{sel}"
            if key in seen:
                continue
            seen.add(key)

            edge = b.get("edge_pct", 0)
            odds = b.get("best_odds", b.get("odds", 0))
            conf = b.get("confidence_tier", "")
            conf_score = {"ELITE": 90, "STRONG": 70, "STANDARD": 50}.get(conf, 40)
            score = conf_score * (1 + edge / 100)

            best.append({
                "match": match,
                "date": b.get("date", ""),
                "selection": sel,
                "market": b.get("market", ""),
                "odds": round(odds, 2),
                "bookmaker": b.get("best_bookmaker", b.get("bookmaker", "")),
                "edge_pct": round(edge, 1),
                "model_prob": round(b.get("model_prob", 0) * 100, 1),
                "stake": round(b.get("stake_amount", b.get("stake", 0)), 2),
                "confidence": conf,
                "score": round(score, 1),
                "source": "pipeline",
            })

        # Add edge scan discoveries not already in pipeline bets
        for b in scan_bets:
            match = b.get("match", "")
            sel = b.get("selection", "")
            key = f"{match}_{sel}"
            if key in seen:
                continue
            seen.add(key)

            best.append({
                "match": match,
                "selection": sel,
                "market": b.get("market", ""),
                "odds": round(b.get("best_odds", 0), 2),
                "bookmaker": b.get("best_bookmaker", ""),
                "edge_pct": round(b.get("edge_pct", 0), 1),
                "model_prob": round(b.get("model_prob", 0) * 100, 1),
                "stake": 0,  # Not sized yet
                "confidence": "",
                "score": round(b.get("edge_pct", 0) * 10, 1),
                "source": "edge_scan",
            })

        # Sort by score (best first) and return top 5
        best.sort(key=lambda x: x["score"], reverse=True)

        return jsonify({
            "best_bets": best[:5],
            "total_value_bets": len(best),
            "last_scan": scan.get("scan_time") if scan else None,
            "summary": f"{len(best)} value bet{'s' if len(best) != 1 else ''} available"
                       if best else "No value bets right now — market is efficient",
        })
    except Exception as e:
        return jsonify({"best_bets": [], "error": str(e)}), 200


@app.route("/api/player-history/<match_key>")
def api_player_history(match_key):
    """Player team history for a match — who's facing their former team."""
    try:
        from scripts.analysis.player_history import get_match_context, build_player_history, find_ex_players

        # Try confirmed lineups first
        context = get_match_context(match_key)
        if "error" not in context:
            return jsonify(context)

        # Fallback: check predictions for team names and try Sofascore lineups
        return jsonify({"match": match_key, "home_vs_former": [], "away_vs_former": [],
                       "note": "No confirmed lineups yet — check closer to kickoff"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/odds-timeline/<match_key>")
def api_odds_timeline(match_key):
    """Odds movement timeline for a specific match.

    Returns historical odds snapshots showing how odds moved from
    opening to current. Enables visual charting on the frontend.
    """
    try:
        from scripts.betting.entry_timer import EntryTimingAnalyzer
        analyzer = EntryTimingAnalyzer()
        analyzer.load_snapshots()

        tl = analyzer.timelines.get(match_key)
        if not tl:
            # Try fuzzy match
            for mk in analyzer.timelines:
                if match_key.replace("-", " ").lower() in mk.lower():
                    tl = analyzer.timelines[mk]
                    break

        if not tl or tl.n_snapshots < 2:
            return jsonify({"error": f"No timeline for {match_key}", "match": match_key})

        # Build timeline for all outcomes
        timeline = {"match": match_key, "snapshots": tl.n_snapshots, "outcomes": {}}
        for outcome in ["home", "draw", "away"]:
            velocity_data = tl.compute_velocity(outcome)
            timeline["outcomes"][outcome] = [{
                "ts": v["ts"],
                "sharp_prob": v["sharp_prob"],
                "market_prob": v["market_prob"],
                "best_odds": v["best_odds"],
                "best_bookmaker": v["best_bookmaker"],
                "divergence": v["divergence"],
                "velocity": v["velocity"],
                "hours_to_kick": v["hours_to_kick"],
            } for v in velocity_data]

        # Add summary
        summary = analyzer.get_match_summary(match_key)
        timeline["summary"] = summary.get("outcomes", {})

        return jsonify(timeline)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/edge-scan")
def api_edge_scan():
    """Run lightweight edge scan — returns current value bets, watchlist, live value.

    Uses cached odds (no API cost). For fresh odds, run the full pipeline.
    """
    try:
        from scripts.betting.odds_edge_monitor import run_scan
        result = run_scan(fetch_fresh=False)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/bet/place", methods=["POST"])
def api_place_bet():
    """Mark a bet as placed — records execution details in journal.

    Expects JSON body:
    {
        "bet_id": "2026-03-30_Milan_vs_Roma_h2h_DRAW",
        "bookmaker": "Pinnacle",
        "execution_odds": 3.45,
        "stake": 25.00
    }
    """
    try:
        data = flask_request.get_json(silent=True) or {}
        if not data:
            return jsonify({"ok": False, "error": "No JSON body"}), 400

        bet_id = data.get("bet_id", "")
        bookmaker = data.get("bookmaker", "")
        exec_odds = data.get("execution_odds", 0)
        stake = data.get("stake", 0)

        if not bet_id:
            return jsonify({"ok": False, "error": "bet_id required"}), 400

        # Validate stake
        import re as _re_bet
        try:
            stake = float(stake)
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "stake must be a number"}), 400
        if stake <= 0:
            return jsonify({"ok": False, "error": "stake must be positive"}), 400
        if stake > 10000:
            return jsonify({"ok": False, "error": "stake exceeds maximum (10000)"}), 400

        # Validate odds
        try:
            exec_odds = float(exec_odds)
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "execution_odds must be a number"}), 400
        if exec_odds < 1.01 or exec_odds > 50.0:
            return jsonify({"ok": False, "error": "odds must be between 1.01 and 50.0"}), 400

        # Validate bookmaker
        if bookmaker and not _re_bet.match(r'^[a-zA-Z0-9_ -]+$', str(bookmaker)):
            return jsonify({"ok": False, "error": "Invalid bookmaker name"}), 400

        # Record in journal
        from scripts.betting.bet_journal import add_bet
        bet_data = {
            "bet_id": bet_id,
            "match": data.get("match", bet_id.rsplit("_", 2)[0] if "_" in bet_id else bet_id),
            "date": data.get("date", ""),
            "market": data.get("market", ""),
            "selection": data.get("selection", ""),
            "bookmaker": bookmaker,
            "odds": exec_odds,
            "stake": stake,
            "model_prob": data.get("model_prob", 0),
            "edge_pct": data.get("edge_pct", 0),
            "placed_at": datetime.now().isoformat(),
        }
        result = add_bet(bet_data)

        # Notify
        try:
            from scripts.pipeline.notify import notify
            notify(
                message=f"Bet placed: {bet_id} @{exec_odds:.2f} ({bookmaker}) EUR {stake:.2f}",
                title="Bet Placed",
                level="info",
                category="betting",
            )
        except Exception:
            pass

        return jsonify({"ok": True, "bet_id": bet_id, "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/odds-improved")
def api_odds_improved():
    """Check which recommended bets now have better odds than when first discovered.

    Compares current odds vs last scan's odds from edge_monitor_state.json.
    """
    try:
        state_path = BETTING_DIR / "edge_monitor_state.json"
        if not state_path.exists():
            return jsonify({"improved": [], "message": "No edge monitor state yet"})

        state = _load_json(state_path)
        alerted = state.get("alerted", {})

        # Load current odds
        odds_data = _load_json(UPCOMING_DIR / "odds_full.json")
        matches = odds_data.get("matches", {})

        improved = []
        for key, prev_bet in alerted.items():
            match = prev_bet.get("match", "")
            selection = prev_bet.get("selection", "").lower()
            prev_odds = prev_bet.get("best_odds", 0)

            mo = matches.get(match, {})
            h2h = mo.get("h2h", {})
            current_odds = 0
            if isinstance(h2h, dict):
                current_odds = h2h.get(f"best_{selection}", h2h.get(selection, 0))

            if current_odds > prev_odds + 0.05:
                improved.append({
                    "match": match,
                    "selection": prev_bet.get("selection"),
                    "prev_odds": round(prev_odds, 3),
                    "current_odds": round(current_odds, 3),
                    "improvement": round(current_odds - prev_odds, 3),
                    "edge_pct": prev_bet.get("edge_pct", 0),
                })

        return jsonify({
            "improved": sorted(improved, key=lambda x: x["improvement"], reverse=True),
            "total_tracked": len(alerted),
            "last_scan": state.get("last_poll"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    port = int(os.environ.get("PORT", 5001))
    log.info(f"Starting server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
