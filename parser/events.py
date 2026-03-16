"""Parse match events (goals, cards, substitutions) from a FBref match report."""

from __future__ import annotations

import logging
import re

from bs4 import BeautifulSoup, Tag

from config.team_names import normalize_team
from models.schemas import MatchEvent

log = logging.getLogger(__name__)


def parse_events(soup: BeautifulSoup, home_team: str, away_team: str) -> list[MatchEvent]:
    """Parse all match events from the scorebox.

    FBref scorebox has 5 direct children:
      [0] home team, [1] away team, [2] meta, [3] home events (id="a"), [4] away events (id="b")

    Each event container has sub-divs, one per event, with structure:
      <div><a href="/en/players/...">Player Name</a> · 3' <div class="event_icon goal"></div></div>
    """
    events: list[MatchEvent] = []
    scorebox = soup.find("div", class_="scorebox")

    if scorebox is None:
        return events

    # Find event containers by id or by class
    home_events_div = scorebox.find("div", id="a") or scorebox.find("div", class_="event")
    away_events_div = scorebox.find("div", id="b")

    # Fallback: get by position
    if home_events_div is None:
        direct_children = scorebox.find_all("div", recursive=False)
        if len(direct_children) >= 4:
            home_events_div = direct_children[3]
        if len(direct_children) >= 5:
            away_events_div = direct_children[4]

    for container, team in [(home_events_div, home_team), (away_events_div, away_team)]:
        if container is None:
            continue
        # Each sub-div is an individual event
        for event_div in container.find_all("div", recursive=False):
            parsed = _parse_single_event(event_div, team)
            if parsed:
                events.append(parsed)

    events.sort(key=lambda e: _minute_sort_key(e.minute))
    return events


def _parse_single_event(div: Tag, team: str) -> MatchEvent | None:
    """Parse a single event div.

    Typical structure:
      <a href="/en/players/...">Mateo Retegui</a> · 3' <div class="event_icon goal"></div>
    Or for substitutions:
      <a>Player Out</a> <div class="event_icon substitute_out"></div> 61'
      <a>Player In</a> <div class="event_icon substitute_in"></div>
    """
    text = div.get_text(separator=" ", strip=True)
    if not text:
        return None

    # Extract minute
    minute = ""
    m = re.search(r"(\d+(?:\+\d+)?)['\u2032\u2019\u0027]", text)
    if m:
        minute = m.group(1)

    if not minute:
        # No minute means not a real event
        return None

    # Determine event type from event_icon class
    event_type = "unknown"
    icons = div.find_all("div", class_=re.compile(r"event_icon"))
    icon_classes = []
    for icon in icons:
        icon_classes.extend(icon.get("class", []))
    icon_str = " ".join(icon_classes).lower()

    if "goal" in icon_str and "own_goal" in icon_str:
        event_type = "own_goal"
    elif "penalty_made" in icon_str or ("goal" in icon_str and "penalty" in text.lower()):
        event_type = "penalty"
    elif "goal" in icon_str:
        event_type = "goal"
    elif "yellow_card" in icon_str or "yellowcard" in icon_str:
        event_type = "yellow_card"
    elif "red_card" in icon_str or "redcard" in icon_str:
        event_type = "red_card"
    elif "second_yellow" in icon_str or "yellow_red" in icon_str:
        event_type = "second_yellow"
    elif "substitute" in icon_str:
        event_type = "substitution"

    # Extract player names from <a> tags
    player_links = [a for a in div.find_all("a") if "/players/" in a.get("href", "")]
    player_names = [a.get_text(strip=True) for a in player_links]

    player = ""
    detail = ""

    if player_names:
        player = player_names[0]
        if len(player_names) > 1:
            if event_type == "substitution":
                # First player is typically the one going off, second going on
                # But depends on FBref layout - use icon context
                detail = f"out:{player_names[0]} in:{player_names[1]}"
            elif event_type in ("goal", "penalty"):
                detail = f"assist:{player_names[1]}"

    return MatchEvent(
        minute=minute,
        event_type=event_type,
        team=team,
        player=player,
        detail=detail,
    )


def events_to_records(events: list[MatchEvent], match_id: str, season: str) -> list[dict]:
    """Convert MatchEvent list to flat dicts for storage."""
    return [
        {
            "match_id": match_id,
            "season": season,
            "minute": e.minute,
            "event_type": e.event_type,
            "team": e.team,
            "player": e.player,
            "detail": e.detail,
        }
        for e in events
    ]


def _minute_sort_key(minute: str) -> float:
    """Convert minute strings like '45+2' to sortable floats."""
    if not minute:
        return 999.0
    m = re.match(r"(\d+)(?:\+(\d+))?", minute)
    if m:
        base = int(m.group(1))
        extra = int(m.group(2)) if m.group(2) else 0
        return base + extra * 0.01
    return 999.0
