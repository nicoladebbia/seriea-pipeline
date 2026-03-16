#!/usr/bin/env python3
"""Open FBref URLs in browser tabs with automatic delays.

This script opens URLs in your default browser with delays between each.
You then just need to save each page (Cmd+S).

Usage:
    python tools/open_urls_in_browser.py --season 2023-2024

This will open 11 tabs for that season, with 5-second delays between each.
Save each page as you go, or use a browser extension to batch save.
"""

import argparse
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# Import from generate_download_urls
from generate_download_urls import (
    AVAILABLE_SEASONS,
    generate_season_urls,
)


def open_url_in_browser(url: str):
    """Open URL in default browser."""
    # Use 'open' on macOS, 'xdg-open' on Linux, 'start' on Windows
    if sys.platform == "darwin":
        subprocess.run(["open", url], check=False)
    elif sys.platform == "linux":
        subprocess.run(["xdg-open", url], check=False)
    else:
        webbrowser.open(url)


def main():
    parser = argparse.ArgumentParser(description="Open FBref URLs in browser")
    parser.add_argument(
        "--season",
        required=True,
        help="Season to download (e.g., 2023-2024)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=5,
        help="Seconds between opening tabs (default: 5)",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only open stats pages (skip fixtures)",
    )
    args = parser.parse_args()

    season = args.season
    if season not in AVAILABLE_SEASONS:
        print(f"Warning: {season} may not be available on FBref")

    urls = generate_season_urls(season)

    if args.stats_only:
        urls = {k: v for k, v in urls.items() if k.startswith("stats_")}

    # Create save directory
    save_dir = Path("data/raw/html") / season.replace("-", "_")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening {len(urls)} URLs for season {season}")
    print(f"Save files to: {save_dir.absolute()}")
    print()
    print("Instructions:")
    print("  1. Wait for each page to fully load")
    print("  2. Press Cmd+S (Mac) or Ctrl+S (Windows/Linux)")
    print("  3. Save as 'Webpage, HTML Only'")
    print("  4. Use the filename shown below")
    print()

    for name, url in urls.items():
        filename = f"{name}.html"
        print(f"Opening: {filename}")
        print(f"  URL: {url}")
        print(f"  Save as: {save_dir / filename}")

        open_url_in_browser(url)

        if args.delay > 0:
            print(f"  Waiting {args.delay} seconds...")
            time.sleep(args.delay)
        print()

    print("Done! All URLs opened.")
    print(f"After saving all files, run: python cli.py parse-html --season {season}")


if __name__ == "__main__":
    main()
