"""Extra-competition scraper: Coppa Italia + Champions League + Europa League.

Scrapes match fixtures/results from FotMob API for use as congestion features
in Serie A predictions. Only dates and team names are needed (not detailed stats).

Usage:
    python -m scraper.xcomp_scraper          # Scrape all competitions, all seasons
    python -m scraper.xcomp_scraper --current # Only current season (fast update)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import requests

from config.settings import DATA_DIR

log = logging.getLogger(__name__)

PARSED_DIR = DATA_DIR / "parsed"

# FotMob competition IDs
COMPETITIONS = {
    141: "Coppa Italia",
    42: "Champions League",
    73: "Europa League",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}


def scrape_competition(comp_id: int, comp_name: str, seasons: List[str] = None) -> List[Dict]:
    """Scrape all matches for a competition from FotMob.

    Args:
        comp_id: FotMob unique tournament ID.
        comp_name: Human-readable competition name.
        seasons: List of season strings (e.g. ["2024/2025"]). If None, scrapes all.

    Returns:
        List of match dicts with date, teams, score, etc.
    """
    if seasons is None:
        r = requests.get(
            f"https://www.fotmob.com/api/leagues?id={comp_id}",
            headers=HEADERS, timeout=15,
        )
        r.raise_for_status()
        seasons = r.json().get("allAvailableSeasons", [])
        time.sleep(0.5)

    all_matches = []
    for season in seasons:
        try:
            r = requests.get(
                f"https://www.fotmob.com/api/leagues?id={comp_id}&season={season}&tab=fixtures",
                headers=HEADERS, timeout=15,
            )
            if r.status_code != 200:
                log.warning("  %s %s: HTTP %d", comp_name, season, r.status_code)
                continue

            matches = r.json().get("fixtures", {}).get("allMatches", [])
            count = 0
            for m in matches:
                status = m.get("status", {})
                score_str = status.get("scoreStr", "")
                hs, as_ = None, None
                if score_str:
                    parts = score_str.split(" - ")
                    if len(parts) == 2:
                        try:
                            hs = int(parts[0].strip())
                            as_ = int(parts[1].strip())
                        except ValueError as e:
                            log.debug(f"Failed to parse score '{score_str}': {e}")

                reason = status.get("reason", {}).get("short", "")
                utc = status.get("utcTime", "")

                all_matches.append({
                    "competition": comp_name,
                    "season": season,
                    "date": utc[:10] if utc else None,
                    "home_team": m.get("home", {}).get("name", "?"),
                    "away_team": m.get("away", {}).get("name", "?"),
                    "home_score": hs,
                    "away_score": as_,
                    "finished": status.get("finished", False),
                    "extra_time": reason in ("AET", "AP"),
                })
                if hs is not None:
                    count += 1

            log.info("  %s %s: %d matches, %d with scores", comp_name, season, len(matches), count)
        except Exception as e:
            log.warning("  %s %s: %s", comp_name, season, e)

        time.sleep(0.4)

    return all_matches


def scrape_all(current_only: bool = False) -> Dict[str, List[Dict]]:
    """Scrape all extra competitions and save to JSON files.

    Args:
        current_only: If True, only scrape the current season (2025/2026).

    Returns:
        Dict mapping output filename to list of match dicts.
    """
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    # Coppa Italia
    log.info("Scraping Coppa Italia...")
    seasons = ["2025/2026"] if current_only else None
    coppa = scrape_competition(141, "Coppa Italia", seasons)

    if current_only:
        # Merge with existing data
        coppa_path = PARSED_DIR / "coppa_italia_matches.json"
        if coppa_path.exists():
            with open(coppa_path) as f:
                existing = json.load(f)
            # Remove current season from existing, add fresh
            existing = [m for m in existing if m.get("season") != "2025/2026"]
            coppa = existing + coppa
    
    with open(PARSED_DIR / "coppa_italia_matches.json", "w") as f:
        json.dump(coppa, f, indent=2)
    log.info("Saved %d Coppa Italia matches", len(coppa))
    results["coppa_italia_matches.json"] = coppa

    # Champions League + Europa League
    log.info("Scraping European competitions...")
    euro = []
    for comp_id, comp_name in [(42, "Champions League"), (73, "Europa League")]:
        seasons = ["2025/2026"] if current_only else None
        euro.extend(scrape_competition(comp_id, comp_name, seasons))

    if current_only:
        euro_path = PARSED_DIR / "european_matches.json"
        if euro_path.exists():
            with open(euro_path) as f:
                existing = json.load(f)
            existing = [m for m in existing if m.get("season") != "2025/2026"]
            euro = existing + euro

    with open(PARSED_DIR / "european_matches.json", "w") as f:
        json.dump(euro, f, indent=2)
    log.info("Saved %d European matches", len(euro))
    results["european_matches.json"] = euro

    return results


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Scrape extra-competition fixtures")
    parser.add_argument("--current", action="store_true", help="Only scrape current season")
    args = parser.parse_args()

    results = scrape_all(current_only=args.current)
    total = sum(len(v) for v in results.values())
    print(f"\nTotal: {total} matches across {len(results)} files")
