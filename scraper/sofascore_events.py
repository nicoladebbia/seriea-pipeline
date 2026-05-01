"""Sofascore match events scraper — incidents (cards, goals, subs) + captain data.

Fetches from Sofascore's undocumented API:
  - /api/v1/event/{matchId}           → match incidents (goals, cards, subs with minute)
  - /api/v1/event/{matchId}/lineups   → lineup with captain flag

Outputs:
  data/external/sofascore/match_incidents.parquet
  data/external/sofascore/captains.parquet

Rate-limited to 1 request per 3 seconds to avoid blocks.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

try:
    from curl_cffi import requests as cffi_requests
    _HAS_CFFI = True
except ImportError:
    import requests as _fallback_requests
    _HAS_CFFI = False

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SOFASCORE_DIR = _PROJECT_ROOT / "data" / "external" / "sofascore"
_INCIDENTS_PATH = _SOFASCORE_DIR / "match_incidents.parquet"
_CAPTAINS_PATH = _SOFASCORE_DIR / "captains.parquet"

_BASE_URL = "https://api.sofascore.com/api/v1"
_DELAY = 2.5  # seconds between requests (session reuse makes lower delay safe)

_HEADERS = {
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.sofascore.com/",
}

# Impersonation options to rotate through on failures
_IMPERSONATE_OPTIONS = ["chrome", "chrome110", "chrome120", "safari", "safari15_5"]

_consecutive_failures = 0  # track API outages
_current_impersonate_idx = 0
_session = None  # persistent session for connection reuse


def _get_session() -> "cffi_requests.Session":
    """Get or create a persistent curl_cffi Session.

    Using a session with connection reuse is critical — Cloudflare/Sofascore
    intermittently blocks NEW TCP connections (~40% failure rate), but an
    established session with keep-alive reuses the same connection (100% success).
    """
    global _session, _current_impersonate_idx
    if _session is None:
        imp = _IMPERSONATE_OPTIONS[_current_impersonate_idx % len(_IMPERSONATE_OPTIONS)]
        if _HAS_CFFI:
            _session = cffi_requests.Session(impersonate=imp)
            _session.headers.update(_HEADERS)
        else:
            import requests as _req
            _session = _req.Session()
            _session.headers.update(_HEADERS)
        log.info("Created new Sofascore session (impersonate=%s)", imp)
    return _session


def _reset_session() -> None:
    """Close and recreate the session (e.g. after repeated failures)."""
    global _session, _current_impersonate_idx
    if _session is not None:
        try:
            _session.close()
        except Exception as e:
            log.debug(f"Failed to close session cleanly: {e}")
    _session = None
    _current_impersonate_idx += 1
    log.info("Session reset, will use impersonate=%s",
             _IMPERSONATE_OPTIONS[_current_impersonate_idx % len(_IMPERSONATE_OPTIONS)])


def _jitter_delay(base: float = _DELAY) -> None:
    """Sleep for base ± 30% random jitter to avoid pattern detection."""
    import random
    jitter = base * 0.3 * (2 * random.random() - 1)
    time.sleep(max(1.0, base + jitter))


def _get_json(url: str, session=None) -> dict | None:
    """Fetch JSON from Sofascore API using a persistent curl_cffi Session.

    Key reliability features:
    - Persistent session with TCP connection reuse (100% vs 60% success rate)
    - Session recreation after 5 consecutive failures
    - Exponential backoff up to 60s per attempt
    - Pauses 3 minutes after 8 consecutive failures
    """
    global _consecutive_failures

    # If API has been consistently failing, pause and reset session
    if _consecutive_failures >= 8:
        pause = 180
        log.warning("API rate-limited (%d consecutive failures). Pausing %ds + resetting session...",
                     _consecutive_failures, pause)
        _reset_session()
        time.sleep(pause)
        _consecutive_failures = 3

    sess = session or _get_session()

    for attempt in range(4):
        try:
            resp = sess.get(url, timeout=20)

            if resp.status_code == 200:
                _consecutive_failures = 0
                return resp.json()
            if resp.status_code == 404:
                _consecutive_failures = 0
                return None
            if resp.status_code in (403, 429, 503):
                wait = min(60, _DELAY * (2 ** attempt))
                log.warning("HTTP %d, waiting %.0fs (attempt %d)",
                            resp.status_code, wait, attempt + 1)
                if attempt >= 2:
                    _reset_session()
                    sess = _get_session()
                time.sleep(wait)
                continue
            log.warning("HTTP %d for %s", resp.status_code, url)
            _consecutive_failures = 0
            return None
        except Exception as exc:
            _consecutive_failures += 1
            wait = min(60, _DELAY * (2 ** attempt))
            log.warning("Request error (attempt %d/4, consec=%d): %s",
                        attempt + 1, _consecutive_failures, str(exc)[:80])
            # Recreate session after connection errors
            if _consecutive_failures >= 5 or attempt >= 1:
                _reset_session()
                sess = _get_session()
            time.sleep(wait)
    return None


def _parse_incidents(data: dict, match_id: int) -> list[dict]:
    """Parse incidents from Sofascore /incidents endpoint.

    API structure per incident:
      incidentType: 'card' | 'goal' | 'substitution' | 'period' | ...
      incidentClass: 'yellow' | 'red' | 'yellowRed' | 'regular' | 'penalty' | 'ownGoal' | 'injury' | ...
      time: int (minute)
      addedTime: int (stoppage time, optional)
      isHome: bool
      player: {name, id, ...}                (cards/goals: the subject)
      playerOut: {name, id, ...}             (substitutions: player coming off)
      playerIn:  {name, id, ...}             (substitutions: player coming on)

    Schema convention: row['player_name'] always holds the "subject" of the
    incident — for cards/goals that's `inc.player`; for substitutions that's
    `inc.playerOut` (the player being replaced). row['player_in_name'] is only
    populated for substitutions.
    """
    rows = []
    incidents = data.get("incidents", [])

    for inc in incidents:
        inc_type = inc.get("incidentType", "")

        # Skip non-event types (period markers, var decisions, etc.)
        if inc_type not in ("card", "goal", "substitution"):
            continue

        minute = inc.get("time", 0) or 0
        added_time = inc.get("addedTime", 0) or 0
        is_home = inc.get("isHome")
        inc_class = inc.get("incidentClass", "")

        # Substitutions use playerOut for the subject; everything else uses player.
        if inc_type == "substitution":
            subject = inc.get("playerOut", {}) or {}
        else:
            subject = inc.get("player", {}) or {}
        player_name = subject.get("name", "") or subject.get("shortName", "")
        player_id = subject.get("id", "")

        row = {
            "match_id": match_id,
            "incident_type": inc_type,
            "incident_class": inc_class,
            "minute": minute,
            "added_time": added_time,
            "player_name": player_name,
            "player_id": str(player_id),
            "is_home": is_home,
        }

        if inc_type == "card":
            row["card_type"] = inc_class  # yellow, red, yellowRed
        elif inc_type == "goal":
            row["goal_type"] = inc_class  # regular, penalty, ownGoal
            assist = inc.get("assist1", {}) or {}
            row["assist_player"] = assist.get("name", "")
        elif inc_type == "substitution":
            player_in = inc.get("playerIn", {}) or {}
            row["player_in_name"] = player_in.get("name", "")
            row["player_in_id"] = str(player_in.get("id", ""))

        rows.append(row)

    return rows


def _parse_captains(data: dict, match_id: int) -> list[dict]:
    """Parse captain data from Sofascore lineups response."""
    rows = []

    for side, is_home in [("home", True), ("away", False)]:
        side_data = data.get(side, {})
        players = side_data.get("players", [])

        for p in players:
            player_data = p.get("player", p)
            if p.get("captain", False):
                rows.append({
                    "match_id": match_id,
                    "is_home": is_home,
                    "captain_name": player_data.get("name", ""),
                    "captain_id": player_data.get("id", ""),
                    "position": player_data.get("position", ""),
                })

    return rows


def get_match_ids() -> list[int]:
    """Get all match IDs from existing Sofascore data across all leagues.

    Pulls from BOTH player_match_stats.parquet (Serie A) and
    player_match_stats_premier_league.parquet (EPL) so incidents/captains/etc.
    cover the full multi-league scope. Pre-EPL versions of this function only
    looked at the SA parquet, leaving 309 EPL matches unscraped.
    """
    ids = set()
    for fname in ("player_match_stats.parquet",
                  "player_match_stats_premier_league.parquet"):
        p = _SOFASCORE_DIR / fname
        if p.exists():
            try:
                df = pd.read_parquet(p, columns=["match_id"])
                ids.update(df["match_id"].unique().tolist())
            except Exception as e:
                log.warning(f"get_match_ids: failed to read {fname}: {e}")
    if ids:
        return sorted(ids)

    # Fallback: from match JSONs
    match_dir = _SOFASCORE_DIR / "matches"
    if match_dir.exists():
        for season_dir in match_dir.iterdir():
            for f in season_dir.glob("*.json"):
                try:
                    ids.add(int(f.stem))
                except ValueError as e:
                    log.debug(f"Skipping non-numeric match file '{f.stem}': {e}")
    return sorted(ids)


def scrape_incidents(
    match_ids: list[int] | None = None,
    force: bool = False,
    max_matches: int | None = None,
) -> pd.DataFrame:
    """Scrape match incidents (goals, cards, subs) from Sofascore API.

    Args:
        match_ids: Specific match IDs to scrape. None = all available.
        force: Re-scrape even if data exists.
        max_matches: Limit number of matches to scrape.

    Returns:
        DataFrame with all incidents.
    """
    if match_ids is None:
        match_ids = get_match_ids()

    # Load existing data to skip already-scraped matches
    existing_ids = set()
    if _INCIDENTS_PATH.exists() and not force:
        existing = pd.read_parquet(_INCIDENTS_PATH)
        existing_ids = set(existing["match_id"].unique())
        log.info("Found %d matches already scraped for incidents", len(existing_ids))

    to_scrape = [mid for mid in match_ids if mid not in existing_ids]
    if max_matches:
        to_scrape = to_scrape[:max_matches]

    if not to_scrape:
        log.info("All incidents already scraped")
        if _INCIDENTS_PATH.exists():
            return pd.read_parquet(_INCIDENTS_PATH)
        return pd.DataFrame()

    total = len(to_scrape)
    log.info("Scraping incidents for %d matches (already have %d)...", total, len(existing_ids))

    all_rows = []
    scraped = 0
    failed = 0
    failed_ids = []
    start_time = time.time()

    for i, mid in enumerate(to_scrape):
        url = f"{_BASE_URL}/event/{mid}/incidents"
        _jitter_delay()

        data = _get_json(url)
        if data:
            rows = _parse_incidents(data, mid)
            all_rows.extend(rows)
            scraped += 1
        else:
            failed += 1
            failed_ids.append(mid)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60  # matches per minute
            remaining = (total - i - 1) / rate if rate > 0 else 0
            log.info("  Incidents: %d/%d done (%d ok, %d fail), %d events | %.0f/min, ~%.0f min left",
                     i + 1, total, scraped, failed, len(all_rows), rate, remaining)

        # Save checkpoint every 100 matches
        if (i + 1) % 100 == 0 and all_rows:
            _save_incidents(all_rows, existing_ids)
            all_rows = []  # Clear buffer after checkpoint
            # Reload existing_ids to include newly saved
            if _INCIDENTS_PATH.exists():
                existing_ids = set(pd.read_parquet(_INCIDENTS_PATH, columns=["match_id"])["match_id"].unique())

    # Final save
    if all_rows:
        _save_incidents(all_rows, existing_ids)

    # Retry failed IDs once
    if failed_ids:
        log.info("Retrying %d failed matches...", len(failed_ids))
        retry_rows = []
        for mid in failed_ids:
            _jitter_delay(6.0)  # Longer delay for retries
            data = _get_json(f"{_BASE_URL}/event/{mid}/incidents")
            if data:
                retry_rows.extend(_parse_incidents(data, mid))
        if retry_rows:
            if _INCIDENTS_PATH.exists():
                existing_ids = set(pd.read_parquet(_INCIDENTS_PATH, columns=["match_id"])["match_id"].unique())
            _save_incidents(retry_rows, existing_ids)

    if _INCIDENTS_PATH.exists():
        final = pd.read_parquet(_INCIDENTS_PATH)
        log.info("Incidents complete: %d total matches, %d total events",
                 final["match_id"].nunique(), len(final))
        return final
    return pd.DataFrame(all_rows)


def _save_incidents(new_rows: list[dict], existing_ids: set) -> pd.DataFrame | None:
    """Save incidents, merging with existing data."""
    if not new_rows:
        return None

    new_df = pd.DataFrame(new_rows)

    if _INCIDENTS_PATH.exists():
        existing = pd.read_parquet(_INCIDENTS_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["match_id", "incident_type", "minute", "player_name"],
            keep="last",
        )
    else:
        combined = new_df

    _SOFASCORE_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(_INCIDENTS_PATH, index=False)
    log.info("Saved %d incidents to %s", len(combined), _INCIDENTS_PATH)
    return combined


def _save_captains(new_rows: list[dict], existing_ids: set) -> pd.DataFrame | None:
    """Save captains, merging with existing data."""
    if not new_rows:
        return None

    new_df = pd.DataFrame(new_rows)

    if _CAPTAINS_PATH.exists():
        existing = pd.read_parquet(_CAPTAINS_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["match_id", "is_home"], keep="last",
        )
    else:
        combined = new_df

    _SOFASCORE_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(_CAPTAINS_PATH, index=False)
    log.info("Saved %d captain records to %s", len(combined), _CAPTAINS_PATH)
    return combined


def scrape_captains(
    match_ids: list[int] | None = None,
    force: bool = False,
    max_matches: int | None = None,
) -> pd.DataFrame:
    """Scrape captain data from Sofascore lineups API.

    Args:
        match_ids: Specific match IDs to scrape. None = all available.
        force: Re-scrape even if data exists.
        max_matches: Limit number of matches to scrape.

    Returns:
        DataFrame with captain info per match.
    """
    if match_ids is None:
        match_ids = get_match_ids()

    existing_ids = set()
    if _CAPTAINS_PATH.exists() and not force:
        existing = pd.read_parquet(_CAPTAINS_PATH)
        existing_ids = set(existing["match_id"].unique())
        log.info("Found %d matches already scraped for captains", len(existing_ids))

    to_scrape = [mid for mid in match_ids if mid not in existing_ids]
    if max_matches:
        to_scrape = to_scrape[:max_matches]

    if not to_scrape:
        log.info("All captain data already scraped")
        if _CAPTAINS_PATH.exists():
            return pd.read_parquet(_CAPTAINS_PATH)
        return pd.DataFrame()

    total = len(to_scrape)
    log.info("Scraping captain data for %d matches (already have %d)...", total, len(existing_ids))

    all_rows = []
    scraped = 0
    failed = 0
    no_captain_count = 0
    failed_ids = []
    start_time = time.time()

    for i, mid in enumerate(to_scrape):
        url = f"{_BASE_URL}/event/{mid}/lineups"
        _jitter_delay()

        data = _get_json(url)
        if data:
            rows = _parse_captains(data, mid)
            if rows:
                all_rows.extend(rows)
            else:
                no_captain_count += 1
            scraped += 1
        else:
            failed += 1
            failed_ids.append(mid)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60
            remaining = (total - i - 1) / rate if rate > 0 else 0
            log.info("  Captains: %d/%d done (%d ok, %d fail, %d no-captain), %d records | %.0f/min, ~%.0f min left",
                     i + 1, total, scraped, failed, no_captain_count, len(all_rows), rate, remaining)

        # Save checkpoint every 100 matches
        if (i + 1) % 100 == 0 and all_rows:
            _save_captains(all_rows, existing_ids)
            all_rows = []
            if _CAPTAINS_PATH.exists():
                existing_ids = set(pd.read_parquet(_CAPTAINS_PATH, columns=["match_id"])["match_id"].unique())

    # Final save
    if all_rows:
        _save_captains(all_rows, existing_ids)

    # Retry failed IDs once
    if failed_ids:
        log.info("Retrying %d failed captain matches...", len(failed_ids))
        retry_rows = []
        for mid in failed_ids:
            _jitter_delay(6.0)
            data = _get_json(f"{_BASE_URL}/event/{mid}/lineups")
            if data:
                rows = _parse_captains(data, mid)
                if rows:
                    retry_rows.extend(rows)
        if retry_rows:
            if _CAPTAINS_PATH.exists():
                existing_ids = set(pd.read_parquet(_CAPTAINS_PATH, columns=["match_id"])["match_id"].unique())
            _save_captains(retry_rows, existing_ids)

    if _CAPTAINS_PATH.exists():
        final = pd.read_parquet(_CAPTAINS_PATH)
        log.info("Captains complete: %d matches with captain data, %d total records",
                 final["match_id"].nunique(), len(final))
        return final
    return pd.DataFrame(all_rows)


def scrape_all() -> dict:
    """Run full scrape: incidents first, then captains (sequentially, never concurrent).

    Returns dict with summary stats.
    """
    log.info("Starting full Sofascore scrape (incidents + captains)...")
    t0 = time.time()

    incidents = scrape_incidents()
    n_inc = incidents["match_id"].nunique() if not incidents.empty else 0

    captains = scrape_captains()
    n_cap = captains["match_id"].nunique() if not captains.empty else 0

    elapsed = (time.time() - t0) / 60
    log.info("Full scrape done in %.1f min: %d matches w/ incidents, %d w/ captains",
             elapsed, n_inc, n_cap)
    return {"incidents_matches": n_inc, "captain_matches": n_cap, "elapsed_min": elapsed}


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Sofascore events scraper")
    parser.add_argument("--test", action="store_true", help="Test with 5 matches only")
    parser.add_argument("--incidents", action="store_true", help="Scrape incidents only")
    parser.add_argument("--captains", action="store_true", help="Scrape captains only")
    parser.add_argument("--limit", type=int, default=None, help="Max matches to scrape")
    parser.add_argument("--force", action="store_true", help="Re-scrape matches even if already in parquet (used for backfill after schema fix)")
    args = parser.parse_args()

    ids = get_match_ids()
    print(f"Total match IDs available: {len(ids)}")

    if args.test:
        print(f"\nTesting with 5 matches...")
        incidents = scrape_incidents(match_ids=ids[-5:], force=True)
        print(f"Incidents scraped: {len(incidents)}")
        if not incidents.empty:
            print(f"Types: {incidents['incident_type'].value_counts().to_dict()}")
        captains = scrape_captains(match_ids=ids[-5:], force=True)
        print(f"Captains found: {len(captains)}")
        if not captains.empty:
            print(captains.to_string())
    elif args.incidents:
        scrape_incidents(max_matches=args.limit, force=args.force)
    elif args.captains:
        scrape_captains(max_matches=args.limit, force=args.force)
    else:
        scrape_all()
