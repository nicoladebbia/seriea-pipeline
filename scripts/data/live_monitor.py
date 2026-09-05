#!/usr/bin/env python3
"""LIVE MATCH MONITOR — Real-time score + odds tracking during matchday.

Polls the Odds API every 15 minutes (2 API calls per poll):
  1. /scores — live scores for all Serie A matches
  2. /odds (h2h,totals,spreads combined in one call) — live odds from all bookmakers

Builds a minute-by-minute timeline of how odds shift during live play,
detects early bet settlement, and archives everything for analysis.

Cost: ~3 credits per poll × ~8 polls per matchday = ~24 credits/matchday.

Usage:
    python scripts/live_monitor.py --once          # Single poll
    python scripts/live_monitor.py --watch          # Loop every 15 min until no live matches
    python scripts/live_monitor.py --status         # Show current live state
    python scripts/live_monitor.py --history 2026-02-07  # Show archived matchday

Data:
    data/live/YYYY-MM-DD.json   — live snapshots for each matchday
    data/live/bet_tracker.json  — active bet tracking with early settlement
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR
from config.team_names import normalize_team

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "soccer_italy_serie_a"  # legacy default; prefer SPORT_KEYS_BY_LEAGUE
POLL_INTERVAL_SECONDS = 15 * 60  # 15 minutes

# Multi-league sport keys (all leagues we actively monitor for live events)
SPORT_KEYS_BY_LEAGUE = {
    "serie_a": "soccer_italy_serie_a",
    "premier_league": "soccer_epl",
}

LIVE_DIR = DATA_DIR / "live"

# Sharp bookmakers we want to track individually
SHARP_BOOKS = {"pinnacle", "betfair_ex_eu", "matchbook", "betcris"}

# ─── API Layer ────────────────────────────────────────────────────────────────

from config.api_keys import get_odds_api_key


def _track_credits(response, endpoint: str = "live_monitor"):
    """Track API credits from response headers (delta-based, see odds_fetcher.track_api_call)."""
    try:
        from scripts.data.odds_fetcher import REGIONS, track_api_call
        remaining_hdr = response.headers.get("x-requests-remaining")
        remaining = int(remaining_hdr) if remaining_hdr is not None else None
        # Estimate: regions × 1 market (live_monitor fetches one market at a time)
        est = len(REGIONS.split(","))
        track_api_call(credits_remaining=remaining, estimated_cost=est, endpoint=endpoint)
    except Exception as e:
        log.debug(f"Failed to track API credits: {e}")


def fetch_live_scores(api_key: str, sport_key: str = SPORT_KEY) -> List[Dict]:
    """Call /scores endpoint. Returns list of events with scores.

    Cost: 2 credits (with daysFrom). Per league.
    """
    url = f"{API_BASE}/sports/{sport_key}/scores/"
    params = {"apiKey": api_key, "daysFrom": 1}

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    _track_credits(resp, endpoint=f"live_monitor:{sport_key}:scores")

    return resp.json()


def fetch_live_odds(api_key: str, sport_key: str = SPORT_KEY) -> List[Dict]:
    """Call /odds endpoint for h2h,totals,spreads markets, eu region.

    Combines all three markets in a single API call for efficiency.
    Cost: 1 credit. Per league.
    """
    url = f"{API_BASE}/sports/{sport_key}/odds/"
    params = {
        "apiKey": api_key,
        "regions": "eu",
        "markets": "h2h,totals,spreads",
        "oddsFormat": "decimal",
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    _track_credits(resp, endpoint=f"live_monitor:{sport_key}:odds")

    return resp.json()


def _leagues_with_active_matches(window_min: int = 180) -> set[str]:
    """Return set of leagues with a match in (kickoff - 30min, kickoff + 180min).

    Used to skip API calls for leagues that have no relevant matches.
    Reads fixtures_*.json (cheap, local).
    """
    import json as _json
    from datetime import datetime, timezone

    # Season is DERIVED via the one existing helper — a "fixtures_2025_2026"
    # literal here is exactly the annual-fuse trap CLAUDE.md documents (this
    # was its 4th instance: frozen last-season files meant "active leagues:
    # none" on every matchday since the 2026-08-01 rollover, so live scores
    # were silently dead all season).
    from scripts.utils.match_timing import _sofascore_fixture_files
    active = set()
    now_ts = datetime.now(timezone.utc).timestamp()
    for p, league in _sofascore_fixture_files():
        if league not in SPORT_KEYS_BY_LEAGUE or not p.exists():
            continue
        try:
            fixtures = _json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(fixtures, list):
            continue
        for fx in fixtures:
            ts = fx.get("startTimestamp")
            if not ts:
                continue
            elapsed_min = (now_ts - ts) / 60
            if -30 <= elapsed_min <= window_min:
                active.add(league)
                break
    return active


def fetch_live_scores_all_leagues(api_key: str, leagues: set[str] | None = None) -> List[Dict]:
    """Pull /scores for the active leagues. If `leagues` is None, only call
    leagues that actually have a match in the live window — saves credits when
    one league has no matches today.

    Cost: 2 credits per league called.
    """
    target = leagues if leagues is not None else _leagues_with_active_matches()
    if not target:
        # No live matches anywhere — skip entirely
        return []
    out: List[Dict] = []
    for league in target:
        sport_key = SPORT_KEYS_BY_LEAGUE.get(league)
        if not sport_key:
            continue
        try:
            league_scores = fetch_live_scores(api_key, sport_key=sport_key)
            for ev in league_scores:
                ev["_league"] = league
                ev["_sport_key"] = sport_key
            out.extend(league_scores)
        except Exception as e:
            log.warning("fetch_live_scores(%s) failed: %s", league, e)
    return out


def fetch_live_odds_all_leagues(api_key: str, leagues: set[str] | None = None) -> List[Dict]:
    """Pull /odds for active leagues only. Cost: 1 credit per league called."""
    target = leagues if leagues is not None else _leagues_with_active_matches()
    if not target:
        return []
    out: List[Dict] = []
    for league in target:
        sport_key = SPORT_KEYS_BY_LEAGUE.get(league)
        if not sport_key:
            continue
        try:
            league_odds = fetch_live_odds(api_key, sport_key=sport_key)
            for ev in league_odds:
                ev["_league"] = league
                ev["_sport_key"] = sport_key
            out.extend(league_odds)
        except Exception as e:
            log.warning("fetch_live_odds(%s) failed: %s", league, e)
    return out


# ─── Match State Logic ────────────────────────────────────────────────────────

def estimate_match_minute(commence_iso: str, now: datetime = None) -> Optional[int]:
    """Estimate current match minute from commence_time.

    Returns None if match hasn't started. Accounts for ~15 min half-time.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    try:
        ct = datetime.fromisoformat(commence_iso.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None

    elapsed_seconds = (now - ct).total_seconds()
    if elapsed_seconds < 0:
        return None

    elapsed_minutes = elapsed_seconds / 60

    # Simple model: 0-45 = first half, 45-60 = half-time, 60-105 = second half
    if elapsed_minutes <= 47:
        return min(int(elapsed_minutes), 45)
    elif elapsed_minutes <= 62:
        return 45  # half-time
    else:
        second_half_min = elapsed_minutes - 62  # ~15 min HT gap
        return min(45 + int(second_half_min), 90)


def classify_match_status(commence_iso: str, completed: bool, scores) -> str:
    """Classify: pre_match, first_half, half_time, second_half, completed."""
    if completed:
        return "completed"

    minute = estimate_match_minute(commence_iso)
    if minute is None:
        return "pre_match"
    if minute <= 0:
        return "pre_match"
    if minute < 45:
        return "first_half"
    if minute == 45:
        # Could be end of first half or start of second
        now = datetime.now(timezone.utc)
        try:
            ct = datetime.fromisoformat(commence_iso.replace("Z", "+00:00"))
            elapsed = (now - ct).total_seconds() / 60
            if elapsed < 50:
                return "first_half"
            elif elapsed < 62:
                return "half_time"
            else:
                return "second_half"
        except Exception:
            return "half_time"
    return "second_half"


# ─── Early Settlement Detection ───────────────────────────────────────────────

def check_bet_settlement(
    bet: Dict,
    home_score: int,
    away_score: int,
    minute: Optional[int],
    completed: bool,
) -> Optional[str]:
    """Check if a bet outcome is decided.

    Understands the REAL journal shapes (market 'O/U 2.5' / '1X2' / 'DC' /
    'DNB' / 'BTTS' / 'AH 0.25' / 'spreads'; selections 'Over 2.5', 'Home',
    '1X (Home or Draw)', 'BTTS No', 'DNB Home', 'Away -0.2', ...).

    Returns: won / lost / push / half_won / half_lost /
    virtually_won / virtually_lost, or None if still open.
    """
    try:
        from scripts.betting.live_bet_context import (
            _normalize_market,
            _sel_side,
            resolve_bet_line,
        )
    except ImportError:
        return None

    market = bet.get("market", "")
    selection = bet.get("selection", "")
    norm = _normalize_market(market)
    sel_l = selection.lower()
    total_goals = home_score + away_score

    if norm == "totals":
        line = resolve_bet_line(market, selection)
        if line is None:
            return None
        floor_line = int(line)
        frac = int(round((line - floor_line) * 100))  # 0 / 25 / 50 / 75

        if "over" in sel_l:
            if frac == 75 and total_goals == floor_line + 1:
                return "half_won" if completed else None
            if total_goals > line:
                return "won" if completed else "virtually_won"
            if completed:
                if frac == 0 and total_goals == floor_line:
                    return "push"
                if frac == 25 and total_goals == floor_line:
                    return "half_lost"
                return "lost"
            # Can't lose OVER until FT — but ≥85' needing 3+ is virtually lost
            if minute and minute >= 85 and (floor_line + 1 - total_goals) >= 3:
                return "virtually_lost"
            return None

        if "under" in sel_l:
            if frac == 75 and total_goals == floor_line + 1:
                return "half_lost" if completed else None
            if total_goals > line:
                return "lost" if completed else "virtually_lost"
            if completed:
                if frac == 0 and total_goals == floor_line:
                    return "push"
                if frac == 25 and total_goals == floor_line:
                    return "half_won"
                return "won"
            return None
        return None

    if norm == "h2h":
        side = _sel_side(selection)
        if side == "home":
            if completed:
                return "won" if home_score > away_score else "lost"
            if minute and minute >= 80 and home_score > away_score + 1:
                return "virtually_won"
            if minute and minute >= 80 and away_score > home_score + 1:
                return "virtually_lost"
            return None
        if side == "away":
            if completed:
                return "won" if away_score > home_score else "lost"
            if minute and minute >= 80 and away_score > home_score + 1:
                return "virtually_won"
            if minute and minute >= 80 and home_score > away_score + 1:
                return "virtually_lost"
            return None
        if side == "draw":
            if completed:
                return "won" if home_score == away_score else "lost"
            return None
        return None

    if norm == "double_chance":
        tok = next((t for t in re.findall(r"[A-Z0-9]+", selection.upper())
                    if t in ("1X", "X2", "12")), "")
        if not tok or not completed:
            return None
        if tok == "1X":
            return "won" if home_score >= away_score else "lost"
        if tok == "X2":
            return "won" if away_score >= home_score else "lost"
        return "won" if home_score != away_score else "lost"

    if norm == "draw_no_bet":
        side = _sel_side(selection)
        if side not in ("home", "away") or not completed:
            return None
        lead = (home_score - away_score) if side == "home" else (away_score - home_score)
        if lead > 0:
            return "won"
        if lead == 0:
            return "push"
        return "lost"

    if norm == "btts":
        toks = re.findall(r"[a-z]+", sel_l)
        both_scored = home_score > 0 and away_score > 0
        if "yes" in toks:
            if both_scored:
                return "won" if completed else "virtually_won"
            return "lost" if completed else None
        if "no" in toks:
            if both_scored:
                return "lost" if completed else "virtually_lost"
            return "won" if completed else None
        return None

    if norm == "spreads":
        line = resolve_bet_line(market, selection)
        if line is None or not completed:
            return None
        side = _sel_side(selection) or "home"
        if side == "home":
            adjusted = home_score + line - away_score
        else:
            adjusted = away_score + line - home_score
        adjusted = round(adjusted, 2)
        if adjusted >= 0.5:
            return "won"
        if abs(adjusted - 0.25) < 0.01:
            return "half_won"
        if adjusted == 0:
            return "push"
        if abs(adjusted + 0.25) < 0.01:
            return "half_lost"
        return "lost"

    return None


# ─── Data Persistence ─────────────────────────────────────────────────────────

def _matchday_path(date_str: str = None) -> Path:
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return LIVE_DIR / f"{date_str}.json"


def load_matchday(date_str: str = None) -> Dict:
    path = _matchday_path(date_str)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {
        "date": date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "polls": 0,
        "api_calls": 0,
        "matches": {},
        "bet_tracking": [],
    }


def save_matchday(data: Dict):
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    path = _matchday_path(data["date"])
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_active_bets() -> List[Dict]:
    """Load pending bets from the bet journal (single source of truth)."""
    try:
        from scripts.betting.bet_journal import get_pending_bets
        return get_pending_bets()
    except Exception as e:
        log.warning("Failed to load pending bets from journal: %s", e)
        return []


def _pre_match_key(home: str, away: str) -> str:
    """odds_full.json and the odds snapshots key matches by NORMALISED names
    ("Roma vs Atalanta"); the live monitor keys by the raw Odds API names
    ("AS Roma vs Atalanta BC"). Until 2026-09-05 the lookup used the raw key
    and never matched, so pre_match_odds was {} on every match ever tracked."""
    return f"{normalize_team(home)} vs {normalize_team(away)}"


def _closing_line_from_snapshots(home: str, away: str, commence: str) -> Dict:
    """The last h2h price written to data/odds_snapshots/odds_*.json BEFORE
    kickoff. odds_full.json rolls over to the next fixtures as soon as a match
    starts, so a match first seen in-play has no pre-match row there; the
    snapshot store still has its closing line. Snapshot timestamps are naive
    LOCAL time; commence_time is UTC."""
    try:
        kick = datetime.fromisoformat(commence.replace("Z", "+00:00"))
        if kick.tzinfo is None:
            kick = kick.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError, AttributeError):
        return {}
    key = _pre_match_key(home, away)
    snap_dir = DATA_DIR / "odds_snapshots"
    if not snap_dir.exists():
        return {}
    days = {(kick - timedelta(days=d)).astimezone().strftime("%Y%m%d") for d in (0, 1)}
    best: Optional[Tuple[datetime, Dict, str]] = None
    for path in snap_dir.glob("odds_*.json"):
        if path.stem.split("_")[1] not in days:
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            ts = datetime.fromisoformat(data.get("timestamp", "")).astimezone()
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if ts >= kick:
            continue
        row = (data.get("matches") or {}).get(key)
        if not row or row.get("home") is None:
            continue
        if best is None or ts > best[0]:
            best = (ts, row, path.name)
    if best is None:
        return {}
    ts, row, name = best
    return {"home": row.get("home"), "draw": row.get("draw"), "away": row.get("away"),
            "source": "closing_snapshot", "snapshot": name, "snapshot_at": ts.isoformat()}


def _pre_match_odds_for(pre_map: Dict[str, Dict], home: str, away: str, commence: str) -> Dict:
    """odds_full row (normalised key) else the closing snapshot line, else {}."""
    entry = pre_map.get(_pre_match_key(home, away)) or pre_map.get(f"{home} vs {away}")
    if entry and entry.get("home") is not None:
        return entry
    closing = _closing_line_from_snapshots(home, away, commence)
    if closing and entry:
        return {**entry, **closing}
    return closing or entry or {}


def _load_pre_match_odds() -> Dict[str, Dict]:
    """Load pre-match odds from odds_full.json for baseline comparison.

    Returns dict keyed by the NORMALISED match_key ("Roma vs Atalanta") with
    h2h, totals, and spreads baselines. Resolve through _pre_match_odds_for.
    """
    # Both leagues: the EPL file is the same shape under its own name.
    matches: Dict[str, Dict] = {}
    for fname in ("odds_full.json", "odds_full_premier_league.json"):
        path = DATA_DIR / "upcoming" / fname
        if not path.exists():
            continue
        try:
            with open(path) as f:
                matches.update(json.load(f).get("matches", {}))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("pre-match odds: could not read %s: %s", fname, exc)
    if not matches:
        return {}
    try:
        result = {}
        for mk, md in matches.items():
            entry = {}
            # H2H baseline
            h2h = md.get("h2h", {})
            if h2h:
                entry["home"] = h2h.get("home")
                entry["draw"] = h2h.get("draw")
                entry["away"] = h2h.get("away")

            # Totals baseline — find the 2.5 line (most common)
            totals_list = md.get("totals", [])
            if totals_list:
                for t in totals_list:
                    line = t.get("line")
                    if line == 2.5 or str(line) == "2.5":
                        entry["totals"] = {
                            "line": 2.5,
                            "over": t.get("avg_over") or t.get("over"),
                            "under": t.get("avg_under") or t.get("under"),
                        }
                        break
                # Fallback: use the first available line
                if "totals" not in entry and totals_list:
                    t = totals_list[0]
                    entry["totals"] = {
                        "line": t.get("line"),
                        "over": t.get("avg_over") or t.get("over"),
                        "under": t.get("avg_under") or t.get("under"),
                    }

            # Spreads baseline — use the primary (first) line
            spreads_list = md.get("spreads", [])
            if spreads_list:
                s = spreads_list[0]
                entry["spreads"] = {
                    "home_line": s.get("home_point") or s.get("line"),
                    "home_price": s.get("avg_home") or s.get("home"),
                    "away_line": s.get("away_point") or (-(s.get("home_point") or s.get("line", 0)) if s.get("home_point") or s.get("line") else None),
                    "away_price": s.get("avg_away") or s.get("away"),
                }

            if entry:
                result[mk] = entry
        return result
    except Exception:
        return {}


# --- Football-Data.org Backup Score Source -----------------------------------

# Import canonical football-data.org team normalization
from scraper.footballdata_lineups import _normalize_fd_team  # noqa: E402


def _get_footballdata_key() -> str:
    """Read FOOTBALLDATA_KEY from env or .env file."""
    key = os.environ.get("FOOTBALLDATA_KEY", "")
    if not key:
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("FOOTBALLDATA_KEY="):
                    key = line.split("=", 1)[1].strip()
    return key


def _fetch_footballdata_scores() -> Dict[str, Dict]:
    """Backup score source from Football-Data.org (free, 10 req/min).

    Returns dict keyed by 'HomeTeam vs AwayTeam' (canonical names) with:
      {home_score: int, away_score: int, status: str, minute: int|None,
       home_team: str, away_team: str}
    """
    fd_key = _get_footballdata_key()
    if not fd_key:
        return {}

    if not HAS_REQUESTS:
        log.warning("football-data.org backup: requests library not available")
        return {}

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    url = "https://api.football-data.org/v4/competitions/SA/matches"
    params = {
        "status": "LIVE,IN_PLAY,PAUSED,FINISHED",
        "dateFrom": today,
        "dateTo": today,
    }
    headers = {"X-Auth-Token": fd_key}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code == 429:
            log.warning("football-data.org: rate limit hit")
            return {}
        if resp.status_code == 403:
            log.warning("football-data.org: forbidden (check API key)")
            return {}
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.SSLError as e:
        log.warning("football-data.org: SSL error: %s", e)
        return {}
    except Exception as e:
        log.warning("football-data.org: request failed: %s", e)
        return {}

    results = {}
    for match in data.get("matches", []):
        home_raw = match.get("homeTeam", {}).get("name", "")
        away_raw = match.get("awayTeam", {}).get("name", "")
        home = _normalize_fd_team(home_raw)
        away = _normalize_fd_team(away_raw)
        if not home or not away:
            continue

        score_data = match.get("score", {})
        full_time = score_data.get("fullTime", {})
        home_score = full_time.get("home")
        away_score = full_time.get("away")

        # Fallback: halftime scores if fulltime not populated yet
        if home_score is None:
            half_time = score_data.get("halfTime", {})
            home_score = half_time.get("home", 0)
            away_score = half_time.get("away", 0)

        home_score = int(home_score) if home_score is not None else 0
        away_score = int(away_score) if away_score is not None else 0

        # Map Football-Data status to our status format
        fd_status = match.get("status", "")
        minute = match.get("minute")  # Football-Data provides this for live matches

        if fd_status == "IN_PLAY":
            if minute is not None and int(minute) > 45:
                status = "second_half"
            else:
                status = "first_half"
        elif fd_status == "PAUSED":
            status = "half_time"
        elif fd_status == "FINISHED":
            status = "completed"
        elif fd_status in ("TIMED", "SCHEDULED"):
            status = "pre_match"
        else:
            status = "first_half"  # safe default for unknown live states

        mk = f"{home} vs {away}"
        results[mk] = {
            "home_score": home_score,
            "away_score": away_score,
            "status": status,
            "minute": int(minute) if minute is not None else None,
            "home_team": home,
            "away_team": away,
        }

    return results


# ─── Live Event Notifications ────────────────────────────────────────────────

def _make_event_key(event: Dict) -> str:
    """Create a unique key for a live event to deduplicate notifications."""
    etype = event.get("type", "")
    minute = event.get("minute", 0)
    player = event.get("player", event.get("player_out", ""))
    return f"{etype}:{minute}:{player}"


def _get_bet_context(match_key: str, home_score: int = 0, away_score: int = 0,
                     minute: int = None,
                     player_stats: dict = None) -> Optional[Dict]:
    """Get bet context for a match, or None if unavailable."""
    try:
        from scripts.betting.live_bet_context import get_match_bet_context
        return get_match_bet_context(match_key, home_score, away_score, minute,
                                     player_stats=player_stats)
    except Exception as e:
        log.debug("Failed to get bet context for %s: %s", match_key, e)
        return None


GOAL_PING_MODES = ("all", "bets")
GOAL_PING_STATE_KEY = "live_goal_pings"


def _goal_ping_mode() -> str:
    """'all' or 'bets', from pipeline_state.json (set on /live or via /api/live/config)."""
    try:
        from scripts.pipeline.pipeline_state import load_state
        mode = str(load_state().get(GOAL_PING_STATE_KEY) or "all").lower()
    except Exception:  # noqa: BLE001 - state trouble must not silence goals
        return "all"
    return mode if mode in GOAL_PING_MODES else "all"


def _send_live_event_notifications(match_key: str, match_data: Dict,
                                    old_events: List, new_events: List):
    """Compare old vs new Sofascore events and send coaching-style notifications."""
    try:
        from scripts.pipeline.notify import notify, notify_goal
    except Exception:
        return

    # Build set of already-seen event keys
    seen_keys = {_make_event_key(e) for e in old_events}

    home = match_data.get("home_team", match_key.split(" vs ")[0] if " vs " in match_key else match_key)
    away = match_data.get("away_team", match_key.split(" vs ")[1] if " vs " in match_key else "")

    # Derive score from ALL goal events (not from snapshots, which lag behind).
    # The Odds API score can be minutes behind SofaScore events.
    all_events = new_events  # new_events is the full current event list
    h_score = sum(1 for e in all_events if e.get("type") == "goal"
                  and e.get("is_home") is True and e.get("goal_type") != "ownGoal")
    h_score += sum(1 for e in all_events if e.get("type") == "goal"
                   and e.get("is_home") is False and e.get("goal_type") == "ownGoal")
    a_score = sum(1 for e in all_events if e.get("type") == "goal"
                  and e.get("is_home") is False and e.get("goal_type") != "ownGoal")
    a_score += sum(1 for e in all_events if e.get("type") == "goal"
                   and e.get("is_home") is True and e.get("goal_type") == "ownGoal")

    # Fallback to snapshot if no goal events (shouldn't happen, but safe)
    if h_score == 0 and a_score == 0:
        snapshots = match_data.get("snapshots", [])
        if snapshots:
            last_score = snapshots[-1].get("score", [0, 0])
            h_score, a_score = last_score[0], last_score[1]

    # Find new events (not seen before)
    new_goal_events = []
    new_other_events = []
    for event in new_events:
        key = _make_event_key(event)
        if key in seen_keys:
            continue
        if event.get("type") == "goal":
            new_goal_events.append(event)
        else:
            new_other_events.append(event)

    # Get bet context once for all events in this match
    latest_minute = None
    if new_goal_events:
        latest_minute = max(e.get("minute", 0) for e in new_goal_events)
    elif new_other_events:
        latest_minute = max(e.get("minute", 0) for e in new_other_events)
    bet_ctx = _get_bet_context(match_key, h_score, a_score, latest_minute,
                               player_stats=match_data.get("live_player_stats"))

    # Goal/red-card pings: "all" (every tracked match — Nicola's 2026-09-05
    # ask, now that the events arrive via ESPN within seconds) or "bets"
    # (the 2026-08-31 setting: only matches carrying a journal bet).
    if _goal_ping_mode() == "bets" and not (bet_ctx and bet_ctx.get("has_bets")):
        if new_goal_events or new_other_events:
            log.info("Live events on %s suppressed — no bets on this match (goal pings: bets)", match_key)
        return

    # Send goal notifications (batch multiple into one if needed)
    if len(new_goal_events) > 1 and bet_ctx and bet_ctx.get("has_bets"):
        # Multi-goal batch: combine into single message
        goal_lines = []
        for event in new_goal_events:
            minute = event.get("minute", 0)
            player = event.get("player", "Unknown")
            is_home = event.get("is_home", True)
            team = home if is_home else away
            goal_type = event.get("goal_type", "regular")
            scorer = player
            if goal_type == "ownGoal":
                scorer = f"{player} (OG)"
            elif goal_type == "penalty":
                scorer = f"{player} (pen)"
            goal_lines.append(f"{scorer} ({team}) {minute}'")
        try:
            msg = f"GOALS! {match_key.replace(' vs ', ' ')} {h_score}-{a_score}\n"
            msg += "\n".join(f"  {gl}" for gl in goal_lines)
            if bet_ctx and bet_ctx.get("has_bets"):
                msg += "\n"
                for b in bet_ctx["bets"]:
                    msg += f"\n  \u00b7 {b['selection']}: {b['commentary']}"
            notify(msg, title=f"GOALS {h_score}-{a_score}", level="info", category="live")
        except Exception as e:
            log.debug("Batch goal notification failed: %s", e)
    else:
        for event in new_goal_events:
            minute = event.get("minute", 0)
            player = event.get("player", "Unknown")
            is_home = event.get("is_home", True)
            team = home if is_home else away
            goal_type = event.get("goal_type", "regular")
            scorer = player
            if goal_type == "ownGoal":
                scorer = f"{player} (OG)"
            elif goal_type == "penalty":
                scorer = f"{player} (pen)"
            try:
                notify_goal(
                    match_key=match_key,
                    scorer=scorer,
                    team=team,
                    home_score=h_score,
                    away_score=a_score,
                    minute=minute,
                    is_home=is_home,
                    bet_context=bet_ctx,
                )
            except Exception as e:
                log.debug("Goal notification failed: %s", e)

    # Other events (red cards, etc.)
    for event in new_other_events:
        etype = event.get("type", "")
        minute = event.get("minute", 0)
        player = event.get("player", "Unknown")
        is_home = event.get("is_home", True)
        team = home if is_home else away

        if etype == "card":
            card_type = event.get("card_type", "")
            if card_type in ("red", "yellowRed"):
                # Only notify if user has bets on this match
                if bet_ctx and bet_ctx.get("has_bets"):
                    card_label = "Red card" if card_type == "red" else "Second yellow"
                    msg = f"{card_label}: {player} ({team}) {minute}'"
                    msg += f"\nYou have {len(bet_ctx['bets'])} bet(s) on this match \u2014 could shift the game."
                    try:
                        notify(msg, title=f"RED: {match_key}", level="warning", category="live")
                    except Exception as e:
                        log.debug("Red card notification failed: %s", e)


# ─── Core Poll Logic ──────────────────────────────────────────────────────────

LIVE_STATUSES = ("first_half", "half_time", "second_half")
# How long an ESPN fast-loop write counts as "fresh": while it is, the slow
# Sofascore cycle leaves events/stats alone (no source flicker every minute)
# and only contributes what ESPN cannot — per-player stats.
FAST_FRESH_S = 60


def _stamp_age_s(stamp: str | None) -> float:
    if not stamp:
        return float("inf")
    try:
        return (datetime.now(timezone.utc) - datetime.fromisoformat(stamp)).total_seconds()
    except (TypeError, ValueError):
        return float("inf")


def _fast_is_fresh(entry: Dict, now: datetime | None = None) -> bool:
    stamp = entry.get("live_fast_at")
    if not stamp:
        return False
    try:
        age = ((now or datetime.now(timezone.utc)) - datetime.fromisoformat(stamp)).total_seconds()
    except (TypeError, ValueError):
        return False
    return 0 <= age < FAST_FRESH_S


def _apply_live_data(mk: str, entry: Dict, live_data: Dict, fast: bool = False) -> None:
    """Merge one source's payload into a matchday entry and fire event pings.

    A field is overwritten only when its source ANSWERED this cycle (legacy
    payloads without flags count as answered): a 403 on one endpoint must not
    blank last good data. On the slow cycle, a Sofascore payload does not
    overwrite events/stats that the fast ESPN loop refreshed within
    FAST_FRESH_S — it only adds player stats.
    """
    flags = dict(live_data.get("fetched") or {"events": True, "statistics": True, "player_stats": True})
    source = live_data.get("source", "sofascore")
    if not fast and source != "espn" and _fast_is_fresh(entry):
        flags["events"] = False
        flags["statistics"] = False
    old_events = entry.get("live_events", [])
    new_events = live_data.get("events", []) if flags.get("events") else old_events
    if flags.get("events"):
        entry["live_events"] = new_events
    if flags.get("statistics"):
        entry["live_stats"] = live_data.get("statistics", {})
    if flags.get("player_stats"):
        # Sofascore's per-player stats are richer (minutes, passes, tackles,
        # rating); while a Sofascore read is recent the fast ESPN roster does
        # not replace it. When Sofascore is blocked, ESPN's roster is the data.
        sofa_fresh = (entry.get("live_player_source") == "sofascore"
                      and _stamp_age_s(entry.get("sofascore_fetched_at")) < 180)
        if not (fast and source == "espn" and sofa_fresh):
            entry["live_player_stats"] = live_data.get("player_stats", {})
            entry["live_player_source"] = source
    if live_data.get("sofascore_id"):
        entry["sofascore_id"] = live_data["sofascore_id"]
    if fast:
        entry["live_fast_at"] = live_data.get("fetched_at", "")
        if live_data.get("score") is not None:
            entry["live_score"] = live_data["score"]
        if live_data.get("clock"):
            entry["live_clock"] = live_data["clock"]
        entry["live_source"] = source
        # The whistle comes from the fast feed too: ESPN says post → the match
        # is completed NOW, not at the next Odds API poll (which only knows the
        # kickoff time and can sit on "2nd half 90'" for ten minutes).
        if live_data.get("state") == "post" and entry.get("status") != "completed":
            entry["status"] = "completed"
            if live_data.get("score") is not None:
                entry["final_score"] = live_data["score"]
            entry["completed_by"] = source
        elif live_data.get("clock") == "HT" and entry.get("status") in ("first_half", "second_half"):
            entry["status"] = "half_time"
    elif flags.get("events") or flags.get("statistics"):
        entry["sofascore_fetched_at"] = live_data.get("fetched_at", "")
        entry["live_source"] = source
    entry.pop("live_fetch_error", None)

    # ── Live event notifications (goals, red cards) ──
    _send_live_event_notifications(mk, entry, old_events, new_events)


PLAYER_BACKFILL_TRIES = 3


def backfill_completed_players(matchday: Dict) -> int:
    """One ESPN roster read for every completed match that has no per-player
    stats (a match that finished before the fast tick could read it, or with
    Sofascore blocked all evening). Writes player stats ONLY — never events,
    so no goal ping can re-fire for a finished match. Returns entries filled.
    """
    from scripts.data import live_espn

    filled = 0
    for mk, entry in matchday.get("matches", {}).items():
        if entry.get("status") != "completed" or entry.get("live_player_stats"):
            continue
        tries = int(entry.get("_players_backfill_tries") or 0)
        if tries >= PLAYER_BACKFILL_TRIES:
            continue
        entry["_players_backfill_tries"] = tries + 1
        home = entry.get("home_team") or mk.split(" vs ")[0]
        away = entry.get("away_team") or (mk.split(" vs ")[1] if " vs " in mk else "")
        # A Friday match sits in Saturday's file: ask ESPN for the kickoff day.
        date = (entry.get("commence_time") or "")[:10].replace("-", "") or None
        try:
            data = live_espn.fetch_live_data_for_match(home, away, date=date)
        except Exception as exc:  # noqa: BLE001 - one match must not sink the pass
            log.warning("player backfill failed for %s: %s", mk, exc)
            continue
        if not data or not (data.get("fetched") or {}).get("player_stats"):
            continue
        entry["live_player_stats"] = data.get("player_stats", {})
        entry["live_player_source"] = data.get("source", "espn")
        entry["live_players_backfilled_at"] = data.get("fetched_at", "")
        filled += 1
    return filled


def _status_after_poll(prev_status: str, polled_status: str) -> str:
    """The Odds API cannot un-finish a match the live feed already finished."""
    if prev_status == "completed" and polled_status != "completed":
        return "completed"
    return polled_status


def refresh_live_fast() -> Dict:
    """The fast live tick: ESPN events + stats + score for every match on the pitch.

    Costs no Odds API credits and never touches Sofascore (a 5s loop there is
    how the June ban was earned). Runs on the auto-poll thread between two
    Odds API polls, so it never races poll_once on the matchday file.
    """
    from scripts.data import live_espn

    now = datetime.now(timezone.utc)
    matchday = load_matchday(now.strftime("%Y-%m-%d"))
    live = {mk: e for mk, e in matchday.get("matches", {}).items() if e.get("status") in LIVE_STATUSES}
    if not live:
        backfilled = backfill_completed_players(matchday)
        if backfilled:
            save_matchday(matchday)
        return {"has_live_matches": False, "refreshed": 0, "players_backfilled": backfilled}
    refreshed = 0
    for mk, entry in live.items():
        home = entry.get("home_team") or mk.split(" vs ")[0]
        away = entry.get("away_team") or (mk.split(" vs ")[1] if " vs " in mk else "")
        try:
            data = live_espn.fetch_live_data_for_match(home, away)
        except Exception as exc:  # noqa: BLE001 - one match must not sink the tick
            log.warning("fast live refresh failed for %s: %s", mk, exc)
            continue
        if not data:
            continue
        _apply_live_data(mk, entry, data, fast=True)
        refreshed += 1
    if refreshed:
        save_matchday(matchday)
    return {"has_live_matches": True, "refreshed": refreshed, "live": len(live)}


def poll_once() -> Dict:
    """Execute one poll cycle: fetch scores + odds + Sofascore events/stats.

    Makes 2 Odds API calls + 2 Sofascore calls per live match.
    Returns summary dict.
    """
    api_key = get_odds_api_key()
    if not api_key:
        log.error("ODDS_API_KEY not set")
        return {"error": "no_api_key"}

    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    matchday = load_matchday(today)

    # ── API Call 1: Scores (Odds API — primary) for ALL active leagues ──
    log.info("Fetching live scores (all leagues)...")
    scores_data = []
    odds_api_scores_failed = False
    try:
        scores_data = fetch_live_scores_all_leagues(api_key)
    except Exception as e:
        log.warning("Odds API scores fetch failed: %s", e)
        odds_api_scores_failed = True

    # ── Backup: Football-Data.org scores ──
    fd_scores = {}
    try:
        fd_scores = _fetch_footballdata_scores()
        if fd_scores:
            log.info("Football-Data.org: %d match score(s) fetched (backup)", len(fd_scores))
    except Exception as e:
        log.warning("Football-Data.org backup fetch failed: %s", e)

    # If Odds API failed entirely, synthesize scores_data from Football-Data
    if odds_api_scores_failed and fd_scores:
        log.info("Using Football-Data.org as primary score source (Odds API failed)")
        for fd_mk, fd in fd_scores.items():
            scores_data.append({
                "home_team": fd["home_team"],
                "away_team": fd["away_team"],
                "commence_time": "",
                "completed": fd["status"] == "completed",
                "scores": [
                    {"name": fd["home_team"], "score": str(fd["home_score"])},
                    {"name": fd["away_team"], "score": str(fd["away_score"])},
                ],
                "_fd_status": fd["status"],
                "_fd_minute": fd["minute"],
                "_score_source": "football_data",
            })

    # ── API Call 2: Odds (h2h + totals + spreads combined) for active leagues only ──
    active = _leagues_with_active_matches()
    log.info("Fetching live odds (h2h,totals,spreads — eu); active leagues: %s",
             sorted(active) or "none")
    odds_data = fetch_live_odds_all_leagues(api_key, leagues=active)

    matchday["polls"] += 1
    # 2 calls per league with an active match (scores + odds = 2 calls)
    matchday["api_calls"] += 2 * len(active)

    # Build odds lookup: match_key → bookmaker odds (h2h + totals + spreads)
    odds_lookup: Dict[str, Dict] = {}
    for event in odds_data:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        mk = f"{home} vs {away}"

        # ── H2H parsing ──
        avg_odds = {}
        sharp_odds = {}
        all_bookmaker_h2h = {}

        # ── Totals parsing ──  {line: {"over": [prices], "under": [prices], "sharp_over": [], "sharp_under": []}}
        totals_by_line: Dict[float, Dict[str, list]] = {}

        # ── Spreads parsing ──  {line: {"home": [prices], "away": [prices], "sharp_home": [], "sharp_away": []}}
        spreads_by_line: Dict[float, Dict[str, list]] = {}

        for bm in event.get("bookmakers", []):
            bm_key = bm.get("key", "")
            bm_title = bm.get("title", bm_key)
            is_sharp = bm_key in SHARP_BOOKS

            for market in bm.get("markets", []):
                market_key = market.get("key", "")

                if market_key == "h2h":
                    outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                    h = outcomes.get(home)
                    d = outcomes.get("Draw")
                    a = outcomes.get(away)
                    if h and d and a:
                        all_bookmaker_h2h[bm_title] = {"home": h, "draw": d, "away": a}
                        if is_sharp:
                            sharp_odds[bm_title] = {"home": h, "draw": d, "away": a}

                elif market_key == "totals":
                    for o in market.get("outcomes", []):
                        name = o.get("name", "")  # "Over" or "Under"
                        price = o.get("price")
                        point = o.get("point")  # e.g. 2.5
                        if not price or point is None:
                            continue
                        if point not in totals_by_line:
                            totals_by_line[point] = {"over": [], "under": [], "sharp_over": [], "sharp_under": []}
                        if name == "Over":
                            totals_by_line[point]["over"].append(price)
                            if is_sharp:
                                totals_by_line[point]["sharp_over"].append(price)
                        elif name == "Under":
                            totals_by_line[point]["under"].append(price)
                            if is_sharp:
                                totals_by_line[point]["sharp_under"].append(price)

                elif market_key == "spreads":
                    for o in market.get("outcomes", []):
                        name = o.get("name", "")  # team name
                        price = o.get("price")
                        point = o.get("point")  # e.g. -0.5, +0.5
                        if not price or point is None:
                            continue
                        # Normalize: use absolute value of home point as line key
                        if name == home:
                            line_key = point
                            if line_key not in spreads_by_line:
                                spreads_by_line[line_key] = {"home": [], "away": [], "sharp_home": [], "sharp_away": [], "away_point": -point}
                            spreads_by_line[line_key]["home"].append(price)
                            if is_sharp:
                                spreads_by_line[line_key]["sharp_home"].append(price)
                        elif name == away:
                            # Away point is the inverse; find/create via home line
                            home_line = -point
                            if home_line not in spreads_by_line:
                                spreads_by_line[home_line] = {"home": [], "away": [], "sharp_home": [], "sharp_away": [], "away_point": point}
                            spreads_by_line[home_line]["away"].append(price)
                            if is_sharp:
                                spreads_by_line[home_line]["sharp_away"].append(price)

        # ── Compute H2H medians ──
        def _median(values):
            s = sorted(values)
            n = len(s)
            if n % 2 == 1:
                return s[n // 2]
            return (s[n // 2 - 1] + s[n // 2]) / 2

        if all_bookmaker_h2h:
            avg_odds = {
                "home": round(_median([b["home"] for b in all_bookmaker_h2h.values()]), 2),
                "draw": round(_median([b["draw"] for b in all_bookmaker_h2h.values()]), 2),
                "away": round(_median([b["away"] for b in all_bookmaker_h2h.values()]), 2),
            }

        # ── Compute totals summary (prefer 2.5 line, fallback to most-offered line) ──
        totals_summary = {}
        if totals_by_line:
            # Prefer 2.5 line; fallback to line with most bookmaker prices
            target_line = 2.5 if 2.5 in totals_by_line else max(
                totals_by_line, key=lambda k: len(totals_by_line[k]["over"]) + len(totals_by_line[k]["under"])
            )
            tl = totals_by_line[target_line]
            if tl["over"] and tl["under"]:
                totals_summary = {
                    "line": target_line,
                    "over_avg": round(_median(tl["over"]), 2),
                    "under_avg": round(_median(tl["under"]), 2),
                }
                if tl["sharp_over"]:
                    totals_summary["over_sharp"] = round(_median(tl["sharp_over"]), 2)
                if tl["sharp_under"]:
                    totals_summary["under_sharp"] = round(_median(tl["sharp_under"]), 2)

            # Also include all available lines for completeness
            all_lines = {}
            for line_val, data in sorted(totals_by_line.items()):
                if data["over"] and data["under"]:
                    all_lines[str(line_val)] = {
                        "over": round(_median(data["over"]), 2),
                        "under": round(_median(data["under"]), 2),
                    }
            if all_lines:
                totals_summary["all_lines"] = all_lines

        # ── Compute spreads summary (use primary/first line) ──
        spreads_summary = {}
        if spreads_by_line:
            # Use the line closest to -0.5 (most common soccer handicap)
            primary_line = min(spreads_by_line.keys(), key=lambda k: abs(k + 0.5))
            sl = spreads_by_line[primary_line]
            if sl["home"] and sl["away"]:
                spreads_summary = {
                    "home_line": primary_line,
                    "away_line": sl.get("away_point", -primary_line),
                    "home_avg": round(_median(sl["home"]), 2),
                    "away_avg": round(_median(sl["away"]), 2),
                }
                if sl["sharp_home"]:
                    spreads_summary["home_sharp"] = round(_median(sl["sharp_home"]), 2)
                if sl["sharp_away"]:
                    spreads_summary["away_sharp"] = round(_median(sl["sharp_away"]), 2)

        odds_lookup[mk] = {
            "avg": avg_odds,
            "sharp": sharp_odds,
            "totals": totals_summary,
            "spreads": spreads_summary,
            "bookmaker_count": len(all_bookmaker_h2h),
        }

    # Load pre-match baselines for comparison
    pre_match_odds = _load_pre_match_odds()

    # ── Process each scored event ──
    live_count = 0
    completed_count = 0
    summary_lines = []
    live_match_keys = []  # track which matches are live for Sofascore fetch

    for event in scores_data:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        mk = f"{home} vs {away}"
        commence = event.get("commence_time", "")
        completed = event.get("completed", False)
        raw_scores = event.get("scores")

        # Parse scores
        home_score = 0
        away_score = 0
        if raw_scores:
            for s in raw_scores:
                try:
                    if s["name"] == home:
                        home_score = int(s["score"])
                    elif s["name"] == away:
                        away_score = int(s["score"])
                except (KeyError, ValueError) as e:
                    log.debug(f"Failed to parse live score for {home} vs {away}: {e}")

        # Use Football-Data status/minute if this event came from FD backup
        score_source = event.get("_score_source", "odds_api")
        if score_source == "football_data":
            status = event.get("_fd_status", "first_half")
            minute = event.get("_fd_minute")
        else:
            status = classify_match_status(commence, completed, raw_scores)
            minute = estimate_match_minute(commence)

        # Cross-check: if both sources have data, log discrepancies
        if score_source == "odds_api" and fd_scores:
            fd_match = fd_scores.get(mk)
            if fd_match:
                if fd_match["home_score"] != home_score or fd_match["away_score"] != away_score:
                    log.warning(
                        "Score discrepancy for %s: OddsAPI=%d-%d vs FD=%d-%d",
                        mk, home_score, away_score,
                        fd_match["home_score"], fd_match["away_score"],
                    )

        # Only track matches that are live or just completed
        if status == "pre_match":
            continue

        if status == "completed":
            completed_count += 1
        else:
            live_count += 1
            live_match_keys.append(mk)

        # ── Build snapshot ──
        match_odds = odds_lookup.get(mk, {})
        snapshot = {
            "ts": now.isoformat(),
            "min": minute,
            "score": [home_score, away_score],
            "status": status,
            "score_source": score_source,
            "avg_odds": match_odds.get("avg", {}),
            "sharp": match_odds.get("sharp", {}),
            "bookmaker_count": match_odds.get("bookmaker_count", 0),
        }
        # Add totals and spreads if available (backward-compatible: old snapshots won't have these)
        totals_data = match_odds.get("totals", {})
        if totals_data:
            snapshot["totals"] = totals_data
        spreads_data = match_odds.get("spreads", {})
        if spreads_data:
            snapshot["spreads"] = spreads_data

        # ── Update match in matchday ──
        if mk not in matchday["matches"]:
            pre = _pre_match_odds_for(pre_match_odds, home, away, commence)
            matchday["matches"][mk] = {
                "commence_time": commence,
                "home_team": home,
                "away_team": away,
                "status": status,
                "final_score": None,
                "pre_match_odds": pre,
                "snapshots": [],
            }

        match_entry = matchday["matches"][mk]
        if not match_entry.get("pre_match_odds"):
            # Self-heal: the row may have been absent (or the key mismatched)
            # when the entry was created; the closing snapshot is on disk.
            match_entry["pre_match_odds"] = _pre_match_odds_for(pre_match_odds, home, away, commence)
        prev_status = match_entry.get("status", "pre_match")
        status = _status_after_poll(prev_status, status)
        match_entry["status"] = status
        match_entry["snapshots"].append(snapshot)

        # ── Score-change goal alert (Odds-API-only fallback) ──
        # When Sofascore is blocked we don't know who scored, but we DO know
        # the score changed (Odds API /scores/ endpoint gives home_score and
        # away_score). Send a basic "Pisa score!" notification with team name
        # only — the rich Sofascore-based notification adds player+minute and
        # supersedes this when Sofascore comes back.
        if status in ("first_half", "second_half", "extra_time", "live"):
            prev_h = match_entry.get("_last_home_score")
            prev_a = match_entry.get("_last_away_score")
            if prev_h is None or prev_a is None:
                # First snapshot for this match — record baseline, don't alert
                pass
            else:
                if home_score > prev_h or away_score > prev_a:
                    scoring_team = home if home_score > prev_h else away
                    # Bet-games only (2026-08-31): fires ONLY when a journal
                    # bet rides on this match, and ONLY while Sofascore events
                    # are absent (the rich path owns the ping otherwise).
                    bet_ctx = _get_bet_context(mk, home_score, away_score, minute)
                    if (bet_ctx and bet_ctx.get("has_bets")
                            and not match_entry.get("live_events")):
                        try:
                            from scripts.pipeline.notify import notify
                            lines = [
                                f"⚽ <b>{scoring_team} scored!</b>",
                                f"{home} {home_score}-{away_score} {away}"
                                + (f"  ({minute}')" if minute else ""),
                                "",
                            ]
                            for b in bet_ctx["bets"]:
                                icon = ("✅" if b.get("is_winning") is True
                                        else "❌" if b.get("is_winning") is False
                                        else "⏳")
                                lines.append(
                                    f"{icon} {b['selection']} @ {b.get('odds', 0):.2f}"
                                    f" — {b.get('commentary', '')}")
                            notify(
                                "\n".join(lines),
                                title=f"GOAL {home_score}-{away_score}",
                                level="info",
                                category="live",
                            )
                        except Exception as e:
                            log.debug("Score-change goal alert failed: %s", e)
                    else:
                        log.info("Goal on %s (%d-%d) — no bets or Sofascore owns it, alert suppressed",
                                 mk, home_score, away_score)
        # Update score baseline for next tick
        match_entry["_last_home_score"] = home_score
        match_entry["_last_away_score"] = away_score

        # ── Kickoff notification (transition into first_half) ──
        if status == "first_half" and prev_status in ("pre_match", None) and not match_entry.get("_kickoff_notified"):
            # Kickoff ping removed (2026-08-27): it repeated the T-30 order
            # ticket half an hour later with less detail; the goal thread
            # takes over seconds after the whistle. Flag kept for state flow.
            match_entry["_kickoff_notified"] = True

        # Half-time ping removed (2026-08-27): the goal thread already carries
        # the score at every change; a no-goal heartbeat earned no interrupt.
        if status == "half_time" and prev_status != "half_time" and not match_entry.get("_ht_notified"):
            match_entry["_ht_notified"] = True

        if completed and not match_entry.get("final_score"):
            match_entry["final_score"] = [home_score, away_score]

        # ── Format summary ──
        odds_str = ""
        avg = match_odds.get("avg", {})
        if avg:
            odds_str = f"H={avg['home']:.2f} D={avg['draw']:.2f} A={avg['away']:.2f}"

        # Totals summary
        totals_str = ""
        live_totals = match_odds.get("totals", {})
        if live_totals and live_totals.get("line") is not None:
            o = live_totals.get("over_avg", 0)
            u = live_totals.get("under_avg", 0)
            if o and u:
                totals_str = f" | O/U {live_totals['line']}: {o:.2f}/{u:.2f}"

        # Spreads summary
        spreads_str = ""
        live_spreads = match_odds.get("spreads", {})
        if live_spreads and live_spreads.get("home_line") is not None:
            hl = live_spreads["home_line"]
            hp = live_spreads.get("home_avg", 0)
            ap = live_spreads.get("away_avg", 0)
            if hp and ap:
                spreads_str = f" | AH {hl:+.1f}: {hp:.2f}/{ap:.2f}"

        # Show shift from pre-match
        pre = pre_match_odds.get(mk, {})
        shift_str = ""
        if pre and avg:
            dh = 1/avg.get("home", 99)*100 - 1/pre.get("home", 99)*100
            dd = 1/avg.get("draw", 99)*100 - 1/pre.get("draw", 99)*100
            da = 1/avg.get("away", 99)*100 - 1/pre.get("away", 99)*100
            shift_str = f" [shift: H{dh:+.1f}pp D{dd:+.1f}pp A{da:+.1f}pp]"

        status_icon = {
            "first_half": "1H",
            "half_time": "HT",
            "second_half": "2H",
            "completed": "FT",
        }.get(status, "??")

        min_str = f"{minute}'" if minute else ""
        summary_lines.append(
            f"  [{status_icon} {min_str:>4}] {mk}: {home_score}-{away_score}  {odds_str}{totals_str}{spreads_str}{shift_str}"
        )

    # ── Sofascore: Fetch rich live data (events, stats) for live matches ──
    if live_match_keys:
        try:
            from scripts.data.live_sofascore import LAST_ERRORS, fetch_live_data_for_matches
            ss_data = fetch_live_data_for_matches(live_match_keys)
            for mk, live_data in ss_data.items():
                if mk in matchday["matches"]:
                    _apply_live_data(mk, matchday["matches"][mk], live_data)
            for mk, why in LAST_ERRORS.items():
                if mk in matchday["matches"]:
                    matchday["matches"][mk]["live_fetch_error"] = why
            if ss_data:
                sources = sorted({d.get("source", "sofascore") for d in ss_data.values()})
                log.info("Live data for %d/%d match(es) via %s", len(ss_data), len(live_match_keys), "+".join(sources))
        except Exception as e:
            log.warning("Sofascore live data fetch failed: %s", e)

    # ── Full-time notifications for completed matches ──
    for mk, mdata in matchday.get("matches", {}).items():
        if mdata.get("status") == "completed" and not mdata.get("_ft_notified"):
            fs = mdata.get("final_score") or (mdata["snapshots"][-1]["score"] if mdata.get("snapshots") else None)
            if fs:
                # Get full bet context with final score for P&L
                bet_ctx = _get_bet_context(mk, fs[0], fs[1], 90)
                try:
                    from scripts.pipeline.notify import notify_full_time
                    notify_full_time(
                        match_key=mk,
                        home_score=fs[0],
                        away_score=fs[1],
                        bet_context=bet_ctx,
                    )
                    mdata["_ft_notified"] = True
                except Exception as e:
                    log.debug("FT notification failed: %s", e)

    # ── Reconciliation: cross-check scores across sources ──
    reconciliation_discrepancies = 0
    try:
        from scripts.data.live_reconciliation import reconcile_all_matches
        reconciliation_discrepancies = reconcile_all_matches(matchday)
        if reconciliation_discrepancies > 0:
            log.warning("Reconciliation: %d discrepancy(ies) found", reconciliation_discrepancies)
        else:
            log.info("Reconciliation: all sources agree")
    except Exception as e:
        log.warning("Reconciliation check failed (non-blocking): %s", e)

    # ── Track bets — continuous (2026-08-31): every journal bet on a
    #    tracked match gets an entry updated EVERY poll (status, score,
    #    live commentary), not only when decided. Feeds /live "Your Bets".
    active_bets = _load_active_bets()
    bet_updates = []
    norm_map = None
    try:
        from scripts.betting.live_bet_context import (
            _check_winning,
            _fuzzy_match_key,
            _generate_commentary,
        )
        norm_map = {_fuzzy_match_key(k): k for k in matchday["matches"]}
    except ImportError as e:
        log.warning("live_bet_context unavailable (%s) — bet tracking skipped", e)

    for bet in (active_bets if norm_map is not None else []):
        match_name = bet.get("match", "")
        mk_resolved = norm_map.get(_fuzzy_match_key(match_name))
        if not mk_resolved:
            continue
        match_entry = matchday["matches"][mk_resolved]

        last_snap = match_entry["snapshots"][-1] if match_entry["snapshots"] else None
        if not last_snap:
            continue

        hs, aws = last_snap["score"]
        minute = last_snap.get("min")
        completed = match_entry["status"] == "completed"

        new_status = check_bet_settlement(bet, hs, aws, minute, completed) or "open"
        market = bet.get("market", "")
        selection = bet.get("selection", "")
        commentary = _generate_commentary(
            market, selection, hs, aws, minute,
            player_stats=match_entry.get("live_player_stats"))
        is_winning = _check_winning(market, selection, hs, aws)

        placed_odds = bet.get("odds", 0)
        placed_stake = bet.get("stake", 0)
        if new_status == "half_won":
            profit = round(placed_stake / 2 * (placed_odds - 1), 2)
        elif new_status == "half_lost":
            profit = round(-placed_stake / 2, 2)
        elif "won" in new_status:
            profit = round(placed_stake * (placed_odds - 1), 2)
        elif "lost" in new_status:
            profit = -placed_stake
        else:  # open / push
            profit = 0.0

        existing = None
        for bt in matchday["bet_tracking"]:
            if (bt.get("match") in (mk_resolved, match_name) and
                    bt.get("market") == market and
                    bt.get("selection") == selection):
                existing = bt
                break
        if existing is None:
            existing = {
                "match": mk_resolved,
                "market": market,
                "selection": selection,
                "placed_odds": placed_odds,
                "placed_stake": placed_stake,
                "status": "open",
            }
            matchday["bet_tracking"].append(existing)

        prev_status = existing.get("status", "open")
        existing.update({
            "match": mk_resolved,
            "status": new_status,
            "score": [hs, aws],
            "minute": minute,
            "commentary": commentary,
            "is_winning": is_winning,
            "potential_profit": profit,
            "updated_at": now.isoformat(),
        })
        if new_status != "open" and prev_status != new_status:
            existing.setdefault("decided_at_minute", minute)
            existing.setdefault("decided_at_ts", now.isoformat())
            existing["score_when_decided"] = [hs, aws]
            bet_updates.append((mk_resolved, selection, new_status))

    # ── Save ──
    save_matchday(matchday)

    # ── Print Summary ──
    print(f"\n{'='*70}")
    print(f" LIVE MONITOR — {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f" Poll #{matchday['polls']} | API calls today: {matchday['api_calls']}")
    print(f"{'='*70}")

    if summary_lines:
        print(f"\n  Live: {live_count} | Completed: {completed_count}")
        print()
        for line in summary_lines:
            print(line)
    else:
        print("\n  No live or recently completed matches")

    if bet_updates:
        print(f"\n  BET UPDATES:")
        for match, sel, status in bet_updates:
            icon = "✓" if "won" in status else "✗" if "lost" in status else "~"
            print(f"    {icon} {match} | {sel} → {status}")

    tracked_bets = matchday.get("bet_tracking", [])
    if tracked_bets:
        print(f"\n  BET TRACKER ({len(tracked_bets)} bets):")
        total_pnl = 0
        for bt in tracked_bets:
            status = bt["status"]
            icon = "✓" if "won" in status else "✗" if "lost" in status else "○"
            pnl = bt.get("potential_profit", 0)
            total_pnl += pnl
            print(f"    {icon} {bt['match']} | {bt['selection']} @{bt['placed_odds']} "
                  f"→ {status} (min {bt.get('decided_at_minute', '?')}) "
                  f"{'+'if pnl>0 else ''}${pnl:.2f}")
        print(f"    {'─'*50}")
        print(f"    Net P&L: {'+'if total_pnl>0 else ''}${total_pnl:.2f}")

    # ── Reconciliation Summary ──
    if reconciliation_discrepancies > 0:
        print(f"\n  RECONCILIATION: {reconciliation_discrepancies} discrepancy(ies)")
        for mk, md in matchday.get("matches", {}).items():
            recon = md.get("reconciliation", {})
            discs = recon.get("discrepancies", [])
            if discs:
                severity = recon.get("severity", "info").upper()
                print(f"    [{severity}] {mk}:")
                for d in discs:
                    print(f"      - {d.get('message', d.get('type', '?'))}")
    elif live_count > 0 or completed_count > 0:
        print(f"\n  RECONCILIATION: all sources agree")

    print(f"\n  Data saved to: {_matchday_path(today)}")
    print(f"{'='*70}\n")

    return {
        "live": live_count,
        "completed": completed_count,
        "polls": matchday["polls"],
        "bet_updates": len(bet_updates),
        "has_live_matches": live_count > 0,
        "reconciliation_discrepancies": reconciliation_discrepancies,
    }


def show_status():
    """Display current live monitoring status."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    matchday = load_matchday(today)

    print(f"\n{'='*70}")
    print(f" LIVE MONITOR STATUS — {today}")
    print(f"{'='*70}")
    print(f"  Polls: {matchday['polls']} | API calls: {matchday['api_calls']}")

    matches = matchday.get("matches", {})
    if not matches:
        print("\n  No matches tracked today")
    else:
        print(f"\n  Matches tracked: {len(matches)}")
        for mk, md in matches.items():
            status = md.get("status", "?")
            n_snaps = len(md.get("snapshots", []))
            final = md.get("final_score")
            last_snap = md["snapshots"][-1] if md.get("snapshots") else None

            if final:
                score_str = f"{final[0]}-{final[1]} (FT)"
            elif last_snap:
                s = last_snap["score"]
                score_str = f"{s[0]}-{s[1]} ({last_snap.get('min', '?')}')"
            else:
                score_str = "no data"

            # Show odds timeline summary
            odds_timeline = ""
            if n_snaps >= 2:
                first = md["snapshots"][0].get("avg_odds", {})
                last = md["snapshots"][-1].get("avg_odds", {})
                if first and last:
                    d_first = first.get("draw", 0)
                    d_last = last.get("draw", 0)
                    if d_first and d_last:
                        odds_timeline = f" | Draw: {d_first:.2f}→{d_last:.2f}"

                # Show totals movement
                first_totals = md["snapshots"][0].get("totals", {})
                last_totals = md["snapshots"][-1].get("totals", {})
                if first_totals and last_totals:
                    o_first = first_totals.get("over_avg", 0)
                    o_last = last_totals.get("over_avg", 0)
                    line = last_totals.get("line", "?")
                    if o_first and o_last:
                        odds_timeline += f" | O{line}: {o_first:.2f}→{o_last:.2f}"

            print(f"    {mk}: {score_str} [{status}] ({n_snaps} snapshots){odds_timeline}")

    bets = matchday.get("bet_tracking", [])
    if bets:
        print(f"\n  Bets tracked: {len(bets)}")
        total_pnl = 0
        for bt in bets:
            pnl = bt.get("potential_profit", 0)
            total_pnl += pnl
            icon = "✓" if "won" in bt["status"] else "✗" if "lost" in bt["status"] else "○"
            print(f"    {icon} {bt['match']} | {bt['selection']} @{bt['placed_odds']} → {bt['status']} (${pnl:+.2f})")
        print(f"    Net P&L: ${total_pnl:+.2f}")

    print()


def show_history(date_str: str):
    """Display archived matchday data with full odds timeline."""
    matchday = load_matchday(date_str)

    print(f"\n{'='*70}")
    print(f" MATCHDAY ARCHIVE — {date_str}")
    print(f"{'='*70}")
    print(f"  Polls: {matchday['polls']} | API calls: {matchday['api_calls']}")

    for mk, md in matchday.get("matches", {}).items():
        final = md.get("final_score")
        pre = md.get("pre_match_odds", {})
        snaps = md.get("snapshots", [])

        score_str = f"{final[0]}-{final[1]}" if final else "?"
        print(f"\n  {'─'*60}")
        print(f"  {mk} — Final: {score_str}")

        if pre:
            pre_h2h_str = ""
            if pre.get("home"):
                pre_h2h_str = f"H={pre.get('home','?'):.2f} D={pre.get('draw','?'):.2f} A={pre.get('away','?'):.2f}"
            pre_totals_str = ""
            pre_totals = pre.get("totals", {})
            if pre_totals and pre_totals.get("over"):
                pre_totals_str = f" | O/U {pre_totals.get('line', '?')}: {pre_totals['over']:.2f}/{pre_totals.get('under', 0):.2f}"
            pre_spreads_str = ""
            pre_spreads = pre.get("spreads", {})
            if pre_spreads and pre_spreads.get("home_price"):
                pre_spreads_str = f" | AH {pre_spreads.get('home_line', 0):+.1f}: {pre_spreads['home_price']:.2f}/{pre_spreads.get('away_price', 0):.2f}"
            print(f"    Pre-match: {pre_h2h_str}{pre_totals_str}{pre_spreads_str}")

        if snaps:
            print(f"    Snapshots ({len(snaps)}):")
            for snap in snaps:
                min_str = f"{snap.get('min', '?'):>3}'"
                s = snap.get("score", [0, 0])
                avg = snap.get("avg_odds", {})
                odds_str = ""
                if avg:
                    odds_str = f"H={avg['home']:.2f} D={avg['draw']:.2f} A={avg['away']:.2f}"

                # Totals in snapshot (backward-compatible)
                snap_totals = snap.get("totals", {})
                totals_str = ""
                if snap_totals and snap_totals.get("line") is not None:
                    o = snap_totals.get("over_avg", 0)
                    u = snap_totals.get("under_avg", 0)
                    if o and u:
                        totals_str = f" | O/U {snap_totals['line']}: {o:.2f}/{u:.2f}"

                # Spreads in snapshot
                snap_spreads = snap.get("spreads", {})
                spreads_str = ""
                if snap_spreads and snap_spreads.get("home_line") is not None:
                    hp = snap_spreads.get("home_avg", 0)
                    ap = snap_spreads.get("away_avg", 0)
                    if hp and ap:
                        spreads_str = f" | AH {snap_spreads['home_line']:+.1f}: {hp:.2f}/{ap:.2f}"

                sharp_str = ""
                sharp = snap.get("sharp", {})
                if sharp:
                    for name in ("Pinnacle", "pinnacle"):
                        if name in sharp:
                            p = sharp[name]
                            sharp_str = f" [PIN: H={p['home']:.2f} D={p['draw']:.2f} A={p['away']:.2f}]"
                            break

                print(f"      {min_str} {s[0]}-{s[1]}  {odds_str}{totals_str}{spreads_str}{sharp_str}")

    bets = matchday.get("bet_tracking", [])
    if bets:
        print(f"\n  {'─'*60}")
        print(f"  Bets: {len(bets)}")
        total_pnl = 0
        for bt in bets:
            pnl = bt.get("potential_profit", 0)
            total_pnl += pnl
            icon = "✓" if "won" in bt["status"] else "✗" if "lost" in bt["status"] else "○"
            print(f"    {icon} {bt['match']} | {bt['selection']} @{bt['placed_odds']} → {bt['status']} "
                  f"(min {bt.get('decided_at_minute', '?')}, score {bt.get('score_when_decided', '?')}) ${pnl:+.2f}")
        print(f"    Net P&L: ${total_pnl:+.2f}")

    print()


def watch_loop():
    """Continuously poll every 15 minutes until no live matches remain."""
    print(f"\n  Starting live monitor loop (poll every {POLL_INTERVAL_SECONDS // 60} min)")
    print(f"  Press Ctrl+C to stop\n")

    consecutive_no_live = 0
    MAX_NO_LIVE = 3  # Stop after 3 consecutive polls with no live matches

    while True:
        try:
            result = poll_once()

            if result.get("error"):
                print(f"  Error: {result['error']}")
                break

            if not result.get("has_live_matches"):
                consecutive_no_live += 1
                print(f"  No live matches ({consecutive_no_live}/{MAX_NO_LIVE} before auto-stop)")
                if consecutive_no_live >= MAX_NO_LIVE:
                    print(f"\n  No live matches for {MAX_NO_LIVE} consecutive polls. Stopping.")
                    break
            else:
                consecutive_no_live = 0

            # Wait for next poll
            next_poll = datetime.now(timezone.utc) + timedelta(seconds=POLL_INTERVAL_SECONDS)
            print(f"  Next poll at {next_poll.strftime('%H:%M:%S')} UTC")
            time.sleep(POLL_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            print(f"\n  Monitor stopped by user.")
            break
        except Exception as e:
            log.error(f"Poll error: {e}")
            time.sleep(60)  # Wait 1 min on error, then retry


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Live Match Monitor")
    parser.add_argument("--once", action="store_true",
                       help="Single poll (2 API calls)")
    parser.add_argument("--watch", action="store_true",
                       help="Continuous polling every 15 min")
    parser.add_argument("--status", action="store_true",
                       help="Show current live monitoring status")
    parser.add_argument("--history", type=str, metavar="YYYY-MM-DD",
                       help="Show archived matchday data")
    args = parser.parse_args()

    if args.history:
        show_history(args.history)
    elif args.status:
        show_status()
    elif args.watch:
        watch_loop()
    elif args.once:
        poll_once()
    else:
        # Default: single poll
        poll_once()


if __name__ == "__main__":
    main()
