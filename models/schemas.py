"""Data models for the Serie A pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class MatchMetadata:
    """Core match information extracted from the scorebox."""

    match_id: str
    season: str
    match_date: date
    kickoff_time: str
    matchweek: Optional[int]
    home_team: str
    away_team: str
    home_score: Optional[int]
    away_score: Optional[int]
    home_xg: Optional[float]
    away_xg: Optional[float]
    venue: str
    attendance: Optional[int]
    referee: str
    home_manager: str = ""
    away_manager: str = ""
    home_captain: str = ""
    away_captain: str = ""


@dataclass
class TeamStats:
    """Aggregate team-level stats from div#team_stats and div#team_stats_extra."""

    stats: dict[str, object]  # flat dict with home_* and away_* keys


@dataclass
class LineupInfo:
    """Lineup data for one team."""

    formation: Optional[str]
    starters: list[dict[str, str]]  # [{name, shirt_number}]
    bench: list[dict[str, str]]


@dataclass
class MatchEvent:
    """A single match event (goal, card, substitution)."""

    minute: str
    event_type: str  # goal, yellow_card, red_card, second_yellow, substitution, own_goal, penalty
    team: str
    player: str
    detail: str  # assist name, sub in/out, etc.


@dataclass
class MatchData:
    """Complete parsed data from a single match HTML page."""

    metadata: MatchMetadata
    team_stats: dict[str, object]
    home_players: list[dict]  # each dict has ~120 stat columns
    away_players: list[dict]
    home_gk: list[dict]
    away_gk: list[dict]
    shots: list[dict]
    home_lineup: Optional[LineupInfo]
    away_lineup: Optional[LineupInfo]
    events: list[MatchEvent]
    parse_warnings: list[str] = field(default_factory=list)
