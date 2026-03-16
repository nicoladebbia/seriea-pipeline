"""Parse downloaded FBref HTML files to extract player stats and shots.

This parser reads HTML files that were manually downloaded from FBref
and extracts structured data (player stats, team stats, shots).

The extracted data is saved to parquet files that integrate with
the existing feature pipeline.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)


def parse_fbref_table(html: str, table_id: str | None = None) -> pd.DataFrame:
    """Parse an FBref HTML table, handling the hidden comment tables.

    FBref hides some tables inside HTML comments. This function handles that.
    """
    # Remove HTML comments to reveal hidden tables
    html_clean = html.replace("<!--", "").replace("-->", "")

    try:
        tables = pd.read_html(html_clean, attrs={"id": table_id} if table_id else None)
        if tables:
            return tables[0]
    except Exception as e:
        log.warning(f"Failed to parse table {table_id}: {e}")

    return pd.DataFrame()


def parse_player_stats_page(html_path: Path, stat_type: str) -> pd.DataFrame:
    """Parse a player stats page (standard, shooting, passing, etc.).

    Args:
        html_path: Path to the HTML file
        stat_type: Type of stats (standard, shooting, passing, etc.)

    Returns:
        DataFrame with player stats
    """
    if not html_path.exists():
        log.warning(f"File not found: {html_path}")
        return pd.DataFrame()

    html = html_path.read_text(encoding="utf-8", errors="ignore")

    # Table IDs for different stat types
    table_ids = {
        "stats_standard": "stats_standard",
        "stats_shooting": "stats_shooting",
        "stats_passing": "stats_passing",
        "stats_passing_types": "stats_passing_types",
        "stats_gca": "stats_gca",
        "stats_defense": "stats_defense",
        "stats_possession": "stats_possession",
        "stats_misc": "stats_misc",
        "stats_keepers": "stats_keeper",
        "stats_keepers_adv": "stats_keeper_adv",
    }

    table_id = table_ids.get(stat_type, f"stats_{stat_type}")

    # Parse the table
    df = parse_fbref_table(html, table_id)

    if df.empty:
        # Try without specific ID
        html_clean = html.replace("<!--", "").replace("-->", "")
        try:
            tables = pd.read_html(html_clean)
            # Find the largest table (usually the main stats table)
            if tables:
                df = max(tables, key=lambda t: len(t))
        except Exception as e:
            log.warning(f"Failed to parse {html_path}: {e}")
            return pd.DataFrame()

    if df.empty:
        return df

    # Clean up multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten multi-level columns
        df.columns = ["_".join(str(c) for c in col if str(c) != "Unnamed: 0_level_0")
                      .strip("_") for col in df.columns]

    # Remove header rows that got included as data
    if "Player" in df.columns:
        df = df[df["Player"] != "Player"].copy()

    # Add metadata
    df["stat_type"] = stat_type
    df["source_file"] = html_path.name

    log.info(f"Parsed {len(df)} rows from {html_path.name}")
    return df


def parse_fixtures_page(html_path: Path) -> pd.DataFrame:
    """Parse a fixtures/schedule page to get match URLs.

    Args:
        html_path: Path to the fixtures HTML file

    Returns:
        DataFrame with match info and URLs
    """
    if not html_path.exists():
        log.warning(f"File not found: {html_path}")
        return pd.DataFrame()

    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    matches = []

    # Find the fixtures table
    table = soup.find("table", {"id": lambda x: x and "sched" in str(x)})
    if not table:
        log.warning(f"No fixtures table found in {html_path}")
        return pd.DataFrame()

    tbody = table.find("tbody")
    if not tbody:
        return pd.DataFrame()

    for row in tbody.find_all("tr"):
        if "thead" in row.get("class", []):
            continue

        match = {}

        # Extract each cell
        for cell in row.find_all(["td", "th"]):
            stat = cell.get("data-stat", "")
            if not stat:
                continue

            # Get text, preferring link text
            link = cell.find("a")
            text = link.text.strip() if link else cell.text.strip()
            match[stat] = text

            # Get match URL from score cell
            if stat == "score" and link:
                href = link.get("href", "")
                if href:
                    match["match_url"] = "https://fbref.com" + href
                    # Extract match ID from URL
                    match_id = href.split("/")[-2] if "/" in href else ""
                    match["match_id"] = match_id

        if match.get("home_team") and match.get("away_team"):
            matches.append(match)

    df = pd.DataFrame(matches)
    log.info(f"Parsed {len(df)} matches from {html_path.name}")
    return df


def parse_match_report(html_path: Path) -> dict:
    """Parse a single match report page.

    Args:
        html_path: Path to the match report HTML file

    Returns:
        Dict with match data, player stats, and shots
    """
    if not html_path.exists():
        log.warning(f"File not found: {html_path}")
        return {}

    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    result = {
        "match_id": html_path.stem.replace("match_", ""),
        "player_stats": [],
        "shots": [],
    }

    # Extract team names and scores from scorebox
    scorebox = soup.find("div", {"class": "scorebox"})
    if scorebox:
        teams = scorebox.find_all("div", {"itemprop": "performer"})
        if len(teams) >= 2:
            home_name = teams[0].find("a", {"itemprop": "name"})
            away_name = teams[1].find("a", {"itemprop": "name"})
            result["home_team"] = home_name.text.strip() if home_name else ""
            result["away_team"] = away_name.text.strip() if away_name else ""

        scores = scorebox.find_all("div", {"class": "score"})
        if len(scores) >= 2:
            try:
                result["home_score"] = int(scores[0].text.strip())
                result["away_score"] = int(scores[1].text.strip())
            except ValueError as e:
                log.debug(f"Failed to parse match scores from HTML: {e}")

    # Extract xG
    for div in soup.find_all("div", {"class": "score_xg"}):
        try:
            xg = float(div.text.strip())
            if "home_xg" not in result:
                result["home_xg"] = xg
            else:
                result["away_xg"] = xg
        except ValueError as e:
            log.debug(f"Failed to parse xG value from HTML: {e}")

    # Extract player stats tables
    html_clean = html.replace("<!--", "").replace("-->", "")
    try:
        tables = pd.read_html(html_clean)
        for i, df in enumerate(tables):
            if len(df) > 5 and any("Player" in str(c) for c in df.columns):
                # This is likely a player stats table
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ["_".join(str(c) for c in col).strip("_")
                                  for col in df.columns]
                df["table_index"] = i
                result["player_stats"].append(df.to_dict("records"))
    except Exception as e:
        log.warning(f"Failed to parse player stats from {html_path}: {e}")

    # Extract shots table
    shots_table = soup.find("table", {"id": "shots_all"})
    if shots_table:
        try:
            shots_df = pd.read_html(str(shots_table))[0]
            if isinstance(shots_df.columns, pd.MultiIndex):
                shots_df.columns = ["_".join(str(c) for c in col).strip("_")
                                    for col in shots_df.columns]
            result["shots"] = shots_df.to_dict("records")
        except Exception as e:
            log.warning(f"Failed to parse shots from {html_path}: {e}")

    return result


def parse_season_html(
    season_dir: Path,
    season: str,
) -> dict[str, pd.DataFrame]:
    """Parse all HTML files for a season.

    Args:
        season_dir: Directory containing HTML files for the season
        season: Season string (e.g., "2023-2024")

    Returns:
        Dict with DataFrames for each data type
    """
    result = {
        "fixtures": pd.DataFrame(),
        "player_stats": [],
    }

    # Parse fixtures
    fixtures_path = season_dir / "fixtures.html"
    if fixtures_path.exists():
        result["fixtures"] = parse_fixtures_page(fixtures_path)
        result["fixtures"]["season"] = season

    # Parse player stats
    stat_types = [
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

    for stat_type in stat_types:
        html_path = season_dir / f"{stat_type}.html"
        if html_path.exists():
            df = parse_player_stats_page(html_path, stat_type)
            if not df.empty:
                df["season"] = season
                result["player_stats"].append(df)

    # Keep player stats as list of DataFrames (one per stat type)
    # This avoids expensive merge operations
    return result


def parse_all_seasons(
    html_dir: Path,
    output_dir: Path,
    seasons: list[str] | None = None,
) -> None:
    """Parse all downloaded HTML files and save to parquet.

    Args:
        html_dir: Directory containing season subdirectories with HTML files
        output_dir: Directory to save parsed parquet files
        seasons: List of seasons to parse (default: all found)
    """
    if seasons is None:
        # Find all season directories
        seasons = [
            d.name.replace("_", "-")
            for d in html_dir.iterdir()
            if d.is_dir() and re.match(r"\d{4}_\d{4}", d.name)
        ]
        seasons.sort()

    all_fixtures = []
    stats_by_type = {}  # Group stats by type for separate files

    for season in seasons:
        season_dir = html_dir / season.replace("-", "_")
        if not season_dir.exists():
            log.warning(f"Season directory not found: {season_dir}")
            continue

        log.info(f"Parsing season {season}...")
        data = parse_season_html(season_dir, season)

        if not data["fixtures"].empty:
            all_fixtures.append(data["fixtures"])

        # Group player stats by stat_type
        for df in data["player_stats"]:
            if df.empty:
                continue
            stat_type = df["stat_type"].iloc[0] if "stat_type" in df.columns else "unknown"
            if stat_type not in stats_by_type:
                stats_by_type[stat_type] = []
            stats_by_type[stat_type].append(df)

    # Save combined data
    output_dir.mkdir(parents=True, exist_ok=True)

    if all_fixtures:
        fixtures_df = pd.concat(all_fixtures, ignore_index=True)
        fixtures_path = output_dir / "fbref_fixtures.parquet"
        fixtures_df.to_parquet(fixtures_path, index=False)
        log.info(f"Saved {len(fixtures_df)} fixtures to {fixtures_path}")

    # Save each stat type separately
    for stat_type, dfs in stats_by_type.items():
        combined = pd.concat(dfs, ignore_index=True)
        stat_path = output_dir / f"fbref_{stat_type}.parquet"
        combined.to_parquet(stat_path, index=False)
        log.info(f"Saved {len(combined)} rows to {stat_path}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    html_dir = Path("data/raw/html")
    output_dir = Path("data/parsed")

    seasons = sys.argv[1:] if len(sys.argv) > 1 else None
    parse_all_seasons(html_dir, output_dir, seasons)
