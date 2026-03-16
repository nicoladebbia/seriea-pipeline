#!/usr/bin/env python3
"""Generate FBref match report URLs for manual download.

Since Cloudflare blocks automated scraping from this IP, this script
generates the URLs you need to download manually via your browser.

Usage:
    python3 generate_fbref_urls.py --season 2023-2024

Then download each HTML file manually and save to:
    data/raw/html/{season}/match_{match_id}.html
"""

import argparse
import json
from pathlib import Path

# FBref URL patterns
FBREF_BASE = "https://fbref.com"
FIXTURES_URL = "{base}/en/comps/11/{season}/schedule/{season}-Serie-A-Scores-and-Fixtures"

# Project paths
PROJECT_ROOT = Path(__file__).parent
REGISTRY_PATH = PROJECT_ROOT / "data" / "registry.json"
HTML_DIR = PROJECT_ROOT / "data" / "raw" / "html"


def load_registry():
    """Load the match registry."""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {"matches": {}}


def get_missing_matches(season: str) -> list[dict]:
    """Get matches that need to be downloaded for a season."""
    registry = load_registry()
    matches = registry.get("matches", {})

    missing = []
    for match_id, info in matches.items():
        if info.get("season") != season:
            continue

        # Check if HTML file exists
        html_path = HTML_DIR / season / f"match_{match_id}.html"
        if html_path.exists():
            continue

        # Get match URL
        match_url = info.get("match_url", "")
        if not match_url:
            # Construct from match_id if URL not stored
            # Format: e.g., "a1b2c3d4" -> "/en/matches/a1b2c3d4/..."
            match_url = f"/en/matches/{match_id}/"

        missing.append({
            "match_id": match_id,
            "url": FBREF_BASE + match_url if not match_url.startswith("http") else match_url,
            "home": info.get("home_team", "Unknown"),
            "away": info.get("away_team", "Unknown"),
            "date": info.get("date", "Unknown"),
        })

    return missing


def main():
    parser = argparse.ArgumentParser(description="Generate FBref URLs for manual download")
    parser.add_argument("--season", required=True, help="Season to generate URLs for (e.g., 2023-2024)")
    parser.add_argument("--output", default="fbref_urls.txt", help="Output file for URLs")
    args = parser.parse_args()

    print(f"Checking for missing match reports in season {args.season}...")

    missing = get_missing_matches(args.season)

    if not missing:
        print(f"All match reports for {args.season} are already downloaded!")
        return

    print(f"Found {len(missing)} missing match reports")

    # Write URLs to file
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        f.write(f"# FBref Match Report URLs for {args.season}\n")
        f.write(f"# Download each HTML file and save to: data/raw/html/{args.season}/match_{{match_id}}.html\n")
        f.write(f"# Total: {len(missing)} matches\n\n")

        for match in missing:
            f.write(f"# {match['date']}: {match['home']} vs {match['away']}\n")
            f.write(f"# Save as: data/raw/html/{args.season}/match_{match['match_id']}.html\n")
            f.write(f"{match['url']}\n\n")

    print(f"URLs written to {output_path}")
    print(f"\nTo download, open each URL in your browser and save the page as HTML.")
    print(f"Save files to: {HTML_DIR}/{args.season}/")

    # Also print fixtures URL
    fixtures_url = FIXTURES_URL.format(base=FBREF_BASE, season=args.season)
    print(f"\nFixtures page (to find match URLs): {fixtures_url}")


if __name__ == "__main__":
    main()
