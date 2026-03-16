#!/usr/bin/env python3
"""Generate FBref download URLs for manual browser download.

This script generates all the URLs you need to download from FBref.
You can then download each page manually in your browser (which handles
Cloudflare automatically) and save the HTML files.

Usage:
    python tools/generate_download_urls.py --seasons 2017-2024

    Then open each URL in your browser and save as HTML to:
    data/raw/html/{season}/

The parser will then extract player stats and shots from these files.
"""

import argparse
from pathlib import Path

# FBref URL templates
FBREF_BASE = "https://fbref.com"
SERIE_A_COMP_ID = 11

# URLs we need for each season
URL_TEMPLATES = {
    "fixtures": "/en/comps/{comp_id}/{season}/schedule/{season}-Serie-A-Scores-and-Fixtures",
    "stats_standard": "/en/comps/{comp_id}/{season}/stats/{season}-Serie-A-Stats",
    "stats_shooting": "/en/comps/{comp_id}/{season}/shooting/{season}-Serie-A-Stats",
    "stats_passing": "/en/comps/{comp_id}/{season}/passing/{season}-Serie-A-Stats",
    "stats_passing_types": "/en/comps/{comp_id}/{season}/passing_types/{season}-Serie-A-Stats",
    "stats_gca": "/en/comps/{comp_id}/{season}/gca/{season}-Serie-A-Stats",
    "stats_defense": "/en/comps/{comp_id}/{season}/defense/{season}-Serie-A-Stats",
    "stats_possession": "/en/comps/{comp_id}/{season}/possession/{season}-Serie-A-Stats",
    "stats_misc": "/en/comps/{comp_id}/{season}/misc/{season}-Serie-A-Stats",
    "stats_keepers": "/en/comps/{comp_id}/{season}/keepers/{season}-Serie-A-Stats",
    "stats_keepers_adv": "/en/comps/{comp_id}/{season}/keepersadv/{season}-Serie-A-Stats",
}

# Seasons available on FBref for Serie A
AVAILABLE_SEASONS = [
    "2017-2018",
    "2018-2019",
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
]


def generate_season_urls(season: str) -> dict[str, str]:
    """Generate all URLs for a season."""
    urls = {}
    for name, template in URL_TEMPLATES.items():
        path = template.format(comp_id=SERIE_A_COMP_ID, season=season)
        urls[name] = FBREF_BASE + path
    return urls


def generate_download_instructions(
    seasons: list[str],
    output_dir: Path,
) -> str:
    """Generate download instructions with URLs."""
    lines = [
        "=" * 70,
        "FBREF MANUAL DOWNLOAD INSTRUCTIONS",
        "=" * 70,
        "",
        "For each URL below:",
        "1. Open the URL in your browser (Chrome/Firefox/Safari)",
        "2. Wait for the page to fully load (Cloudflare will pass automatically)",
        "3. Press Ctrl+S (or Cmd+S on Mac) to save the page",
        "4. Save as 'Webpage, HTML Only' (not 'Complete')",
        "5. Save to the directory shown for each file",
        "",
        "TIP: You can open multiple tabs and download in parallel!",
        "",
        "-" * 70,
        "",
    ]

    total_files = 0
    for season in seasons:
        season_dir = output_dir / season.replace("-", "_")
        lines.append(f"SEASON: {season}")
        lines.append(f"Save to: {season_dir}/")
        lines.append("")

        urls = generate_season_urls(season)
        for name, url in urls.items():
            filename = f"{name}.html"
            lines.append(f"  {filename}")
            lines.append(f"  {url}")
            lines.append("")
            total_files += 1

        lines.append("-" * 70)
        lines.append("")

    lines.append(f"TOTAL FILES TO DOWNLOAD: {total_files}")
    lines.append("")
    lines.append("After downloading, run:")
    lines.append("  python cli.py parse-html")
    lines.append("")

    return "\n".join(lines)


def generate_url_list(seasons: list[str]) -> str:
    """Generate simple URL list for easy copying."""
    urls = []
    for season in seasons:
        season_urls = generate_season_urls(season)
        for name, url in season_urls.items():
            urls.append(url)
    return "\n".join(urls)


def check_existing_downloads(seasons: list[str], html_dir: Path) -> dict[str, list[str]]:
    """Check which files are already downloaded."""
    status = {"downloaded": [], "missing": []}

    for season in seasons:
        season_dir = html_dir / season.replace("-", "_")
        for name in URL_TEMPLATES.keys():
            filepath = season_dir / f"{name}.html"
            if filepath.exists() and filepath.stat().st_size > 1000:
                status["downloaded"].append(f"{season}/{name}.html")
            else:
                status["missing"].append(f"{season}/{name}.html")

    return status


def main():
    parser = argparse.ArgumentParser(description="Generate FBref download URLs")
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=AVAILABLE_SEASONS,
        help="Seasons to generate URLs for",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/html"),
        help="Directory where HTML files should be saved",
    )
    parser.add_argument(
        "--urls-only",
        action="store_true",
        help="Output only URLs (for easy copying)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check which files are already downloaded",
    )
    args = parser.parse_args()

    if args.check:
        status = check_existing_downloads(args.seasons, args.output_dir)
        print(f"Downloaded: {len(status['downloaded'])} files")
        print(f"Missing: {len(status['missing'])} files")
        if status["missing"]:
            print("\nMissing files:")
            for f in status["missing"][:20]:
                print(f"  - {f}")
            if len(status["missing"]) > 20:
                print(f"  ... and {len(status['missing']) - 20} more")
        return

    if args.urls_only:
        print(generate_url_list(args.seasons))
    else:
        # Create output directories
        for season in args.seasons:
            season_dir = args.output_dir / season.replace("-", "_")
            season_dir.mkdir(parents=True, exist_ok=True)

        print(generate_download_instructions(args.seasons, args.output_dir))


if __name__ == "__main__":
    main()
