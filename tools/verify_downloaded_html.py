#!/usr/bin/env python3
"""Verify downloaded FBref HTML files are valid and contain expected data.

This script checks:
1. All expected files exist
2. Files are not Cloudflare challenge pages
3. Files contain the expected tables
4. Data can be parsed correctly
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


def check_file_exists(filepath: Path) -> tuple[bool, str]:
    """Check if file exists and has reasonable size."""
    if not filepath.exists():
        return False, "File not found"

    size = filepath.stat().st_size
    if size < 5000:
        return False, f"File too small ({size} bytes) - likely incomplete"

    return True, f"OK ({size:,} bytes)"


def check_not_cloudflare(filepath: Path) -> tuple[bool, str]:
    """Check file is not a Cloudflare challenge page."""
    content = filepath.read_text(encoding="utf-8", errors="ignore")

    if "Just a moment" in content and "Cloudflare" in content:
        return False, "CLOUDFLARE CHALLENGE PAGE - not real content!"

    if "Enable JavaScript and cookies" in content:
        return False, "Cloudflare JS challenge page"

    return True, "Not Cloudflare"


def check_has_fbref_content(filepath: Path) -> tuple[bool, str]:
    """Check file has FBref-specific content."""
    content = filepath.read_text(encoding="utf-8", errors="ignore")

    indicators = [
        ("body.fb" in content or 'class="fb"' in content, "FBref body class"),
        ("Serie A" in content, "Serie A mention"),
        ("<table" in content.lower(), "Has tables"),
    ]

    passed = [name for ok, name in indicators if ok]
    failed = [name for ok, name in indicators if not ok]

    if len(passed) >= 2:
        return True, f"Valid FBref content ({', '.join(passed)})"
    else:
        return False, f"Missing: {', '.join(failed)}"


def check_can_parse_tables(filepath: Path, expected_table_type: str) -> tuple[bool, str]:
    """Check if we can parse the expected tables from the file."""
    content = filepath.read_text(encoding="utf-8", errors="ignore")

    # Remove HTML comments to reveal hidden tables
    content_clean = content.replace("<!--", "").replace("-->", "")

    try:
        tables = pd.read_html(content_clean)

        if not tables:
            return False, "No tables found"

        # Check for expected content based on file type
        if expected_table_type == "fixtures":
            # Should have match data
            for t in tables:
                if len(t) > 100 and any("Home" in str(c) or "Score" in str(c) for c in t.columns):
                    return True, f"Found fixtures table ({len(t)} rows)"
            return False, f"Found {len(tables)} tables but no fixtures table"

        elif expected_table_type.startswith("stats_"):
            # Should have player stats
            for t in tables:
                cols_str = str(t.columns)
                if len(t) > 50 and ("Player" in cols_str or "Squad" in cols_str):
                    return True, f"Found player stats ({len(t)} rows, {len(t.columns)} cols)"
            return False, f"Found {len(tables)} tables but no player stats"

        else:
            return True, f"Found {len(tables)} tables"

    except Exception as e:
        return False, f"Parse error: {e}"


def verify_season(season: str, html_dir: Path) -> dict:
    """Verify all files for a season."""
    season_dir = html_dir / season.replace("-", "_")

    expected_files = [
        "fixtures",
        "stats_standard",
        "stats_shooting",
        "stats_passing",
        "stats_passing_types",
        "stats_gca",
        "stats_defense",
        "stats_possession",
        "stats_misc",
        "stats_keepers",
        "stats_keepers_adv",
    ]

    results = {}

    print(f"\n{'='*60}")
    print(f"VERIFYING SEASON: {season}")
    print(f"Directory: {season_dir}")
    print(f"{'='*60}\n")

    all_ok = True

    for file_type in expected_files:
        filepath = season_dir / f"{file_type}.html"
        print(f"Checking {file_type}.html...")

        checks = []

        # Check 1: File exists
        ok, msg = check_file_exists(filepath)
        checks.append(("Exists", ok, msg))

        if ok:
            # Check 2: Not Cloudflare
            ok2, msg2 = check_not_cloudflare(filepath)
            checks.append(("Content", ok2, msg2))

            if ok2:
                # Check 3: Has FBref content
                ok3, msg3 = check_has_fbref_content(filepath)
                checks.append(("FBref", ok3, msg3))

                # Check 4: Can parse tables
                ok4, msg4 = check_can_parse_tables(filepath, file_type)
                checks.append(("Parse", ok4, msg4))

        # Print results
        file_ok = all(c[1] for c in checks)
        status = "✅" if file_ok else "❌"
        print(f"  {status} {file_type}.html")

        for check_name, check_ok, check_msg in checks:
            icon = "  ✓" if check_ok else "  ✗"
            print(f"     {icon} {check_name}: {check_msg}")

        results[file_type] = {
            "ok": file_ok,
            "checks": checks,
        }

        if not file_ok:
            all_ok = False

        print()

    # Summary
    ok_count = sum(1 for r in results.values() if r["ok"])
    total_count = len(results)

    print(f"{'='*60}")
    print(f"SUMMARY: {ok_count}/{total_count} files OK")

    if all_ok:
        print("✅ All files verified successfully!")
    else:
        failed = [f for f, r in results.items() if not r["ok"]]
        print(f"❌ Problems with: {', '.join(failed)}")
        print("\nTo fix:")
        print("  1. Re-open the URLs for failed files in your browser")
        print("  2. Make sure page fully loads (wait for tables)")
        print("  3. Save as 'Webpage, HTML Only'")

    print(f"{'='*60}\n")

    return {"season": season, "all_ok": all_ok, "results": results}


def main():
    parser = argparse.ArgumentParser(description="Verify downloaded FBref HTML files")
    parser.add_argument(
        "--season",
        required=True,
        help="Season to verify (e.g., 2023-2024)",
    )
    parser.add_argument(
        "--html-dir",
        type=Path,
        default=Path("data/raw/html"),
        help="Directory containing HTML files",
    )
    args = parser.parse_args()

    result = verify_season(args.season, args.html_dir)

    # Exit with error code if verification failed
    sys.exit(0 if result["all_ok"] else 1)


if __name__ == "__main__":
    main()
