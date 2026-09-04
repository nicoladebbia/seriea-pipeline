"""Data models for the FBref leaf parsers (parser/lineups.py, parser/events.py).

The MatchMetadata / TeamStats / MatchData containers that fed the original
scrape -> parse -> store spine were removed with it on 2026-09-04.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


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
