"""Orchestrator: parse a complete FBref match report HTML page."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from models.schemas import MatchData
from parser.events import parse_events
from parser.goalkeeper_stats import parse_goalkeeper_stats
from parser.html_utils import get_soup
from parser.lineups import parse_lineups
from parser.player_stats import parse_player_stats
from parser.scorebox import parse_scorebox
from parser.shots import parse_shots
from parser.team_stats import parse_team_stats
from scraper.registry import Registry
from storage.paths import html_match_path

log = logging.getLogger(__name__)


def _discover_team_hashes(soup) -> list[str]:
    """Discover team hash codes from stat table IDs.

    FBref uses table ids like `stats_{hash}_summary`.
    The first hash = home team, the second = away team.
    """
    hashes: list[str] = []
    seen: set[str] = set()

    for table in soup.find_all("table", id=re.compile(r"^stats_[a-f0-9]+_summary$")):
        m = re.match(r"stats_([a-f0-9]+)_summary", table.get("id", ""))
        if m and m.group(1) not in seen:
            hashes.append(m.group(1))
            seen.add(m.group(1))

    if len(hashes) < 2:
        # Fallback: try any stats_*_* pattern
        for table in soup.find_all("table", id=re.compile(r"^stats_[a-f0-9]+")):
            m = re.match(r"stats_([a-f0-9]+)_", table.get("id", ""))
            if m and m.group(1) not in seen:
                hashes.append(m.group(1))
                seen.add(m.group(1))
            if len(hashes) >= 2:
                break

    return hashes


def parse_match(html_path: Path, match_id: str, season: str) -> MatchData:
    """Parse a complete match HTML file into structured data."""
    html = html_path.read_text(encoding="utf-8")
    soup = get_soup(html)

    # 1. Metadata
    metadata = parse_scorebox(soup, match_id, season)

    # 2. Team stats
    team_stats = parse_team_stats(soup)

    # 3. Discover team hashes for player/GK tables
    hashes = _discover_team_hashes(soup)
    home_hash = hashes[0] if len(hashes) > 0 else ""
    away_hash = hashes[1] if len(hashes) > 1 else ""

    # 4. Player stats
    home_players: list[dict] = []
    away_players: list[dict] = []
    if home_hash:
        home_players = parse_player_stats(
            soup, home_hash, metadata.home_team, True, match_id
        )
    if away_hash:
        away_players = parse_player_stats(
            soup, away_hash, metadata.away_team, False, match_id
        )

    # 5. Goalkeeper stats
    home_gk: list[dict] = []
    away_gk: list[dict] = []
    if home_hash:
        home_gk = parse_goalkeeper_stats(
            soup, home_hash, metadata.home_team, True, match_id
        )
    if away_hash:
        away_gk = parse_goalkeeper_stats(
            soup, away_hash, metadata.away_team, False, match_id
        )

    # 6. Shots
    shots = parse_shots(soup, match_id, metadata.home_team, metadata.away_team)

    # 7. Lineups
    home_lineup, away_lineup = parse_lineups(soup)

    # 8. Events
    events = parse_events(soup, metadata.home_team, metadata.away_team)

    # Collect warnings
    warnings: list[str] = []
    if not home_hash:
        warnings.append("Could not discover home team hash")
    if not away_hash:
        warnings.append("Could not discover away team hash")
    if not home_players:
        warnings.append("No home player stats parsed")
    if not away_players:
        warnings.append("No away player stats parsed")

    if warnings:
        log.warning("Match %s warnings: %s", match_id, "; ".join(warnings))

    return MatchData(
        metadata=metadata,
        team_stats=team_stats,
        home_players=home_players,
        away_players=away_players,
        home_gk=home_gk,
        away_gk=away_gk,
        shots=shots,
        home_lineup=home_lineup,
        away_lineup=away_lineup,
        events=events,
        parse_warnings=warnings,
    )


def parse_all_matches(
    season: str | None = None,
    reparse: bool = False,
) -> list[MatchData]:
    """Parse all (or season-filtered) downloaded matches.

    Uses the registry to determine what needs parsing.
    Returns a list of MatchData objects.
    """
    registry = Registry()
    entries = registry.get_by_season(season) if season else registry.get_all()

    results: list[MatchData] = []

    for entry in entries:
        if not reparse and registry.is_parsed(entry.match_id):
            continue

        html_path = Path(entry.html_path)
        if not html_path.exists():
            # Try reconstructing path
            html_path = html_match_path(entry.season, entry.match_id)

        if not html_path.exists():
            log.warning("HTML file not found for %s: %s", entry.match_id, html_path)
            continue

        try:
            match_data = parse_match(html_path, entry.match_id, entry.season)
            results.append(match_data)
            registry.mark_parsed(entry.match_id)
            log.info(
                "Parsed %s: %s %d-%d %s | %d home players, %d away players, %d shots",
                entry.match_id,
                match_data.metadata.home_team,
                match_data.metadata.home_score or 0,
                match_data.metadata.away_score or 0,
                match_data.metadata.away_team,
                len(match_data.home_players),
                len(match_data.away_players),
                len(match_data.shots),
            )
        except Exception as exc:
            log.error("Failed to parse %s: %s", entry.match_id, exc, exc_info=True)

    return results
