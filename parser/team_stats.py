"""Parse team-level aggregate stats from div#team_stats and div#team_stats_extra."""

from __future__ import annotations

import logging
import re

from bs4 import BeautifulSoup, Tag

from parser.html_utils import safe_float, safe_int

log = logging.getLogger(__name__)


def parse_team_stats(soup: BeautifulSoup) -> dict[str, object]:
    """Parse div#team_stats and div#team_stats_extra.

    Returns a flat dict with home_* and away_* prefixed keys.
    """
    result: dict[str, object] = {}

    # --- div#team_stats ---
    # Structure: <table> with alternating rows:
    #   <tr><th colspan="2">Possession</th></tr>
    #   <tr><td>...<strong>41%</strong>...</td><td>...<strong>59%</strong>...</td></tr>
    team_stats_div = soup.find("div", id="team_stats")
    if team_stats_div:
        _parse_main_stats(team_stats_div, result)

    # --- div#team_stats_extra ---
    # Structure: groups of <div> children with pattern:
    #   <div class="th">Team A</div><div class="th"> </div><div class="th">Team B</div>
    #   <div>17</div><div>Fouls</div><div>12</div>
    extra_div = soup.find("div", id="team_stats_extra")
    if extra_div:
        _parse_extra_stats(extra_div, result)

    return result


def _parse_main_stats(div: Tag, result: dict) -> None:
    """Parse the main team_stats table.

    Iterate rows: when we find a <th colspan="2"> that's a label,
    the next row's two <td> cells contain home and away values inside <strong>.
    """
    rows = div.find_all("tr")
    current_label = ""

    for tr in rows:
        # Check for label row: <th colspan="2">
        th = tr.find("th", colspan=True)
        if th:
            current_label = th.get_text(strip=True)
            continue

        # Value row: two <td> cells
        tds = tr.find_all("td")
        if len(tds) == 2 and current_label:
            key = _normalize_stat_key(current_label)

            # Special handling for Cards (uses icon spans, not text)
            if key == "cards":
                for i, prefix in enumerate(("home", "away")):
                    yellows = len(tds[i].find_all("span", class_="yellow_card"))
                    reds = len(tds[i].find_all("span", class_="red_card"))
                    result[f"{prefix}_yellow_cards"] = yellows
                    result[f"{prefix}_red_cards"] = reds
                current_label = ""
                continue

            # Extract both the percentage (from <strong>) and raw counts
            # e.g. "2 of 4 — 50%" → pct=50.0, count=2, total=4
            for i, prefix in enumerate(("home", "away")):
                full_text = tds[i].get_text(strip=True)
                strong_text = _extract_strong_value(tds[i])

                # Store the percentage / primary value
                result[f"{prefix}_{key}"] = _parse_stat_value(strong_text)

                # For fraction-based stats, also extract the raw counts
                frac = re.search(r"(\d+)\s+of\s+(\d+)", full_text)
                if frac:
                    result[f"{prefix}_{key}_count"] = int(frac.group(1))
                    result[f"{prefix}_{key}_total"] = int(frac.group(2))

            current_label = ""  # consumed


def _extract_strong_value(td: Tag) -> str:
    """Extract the <strong> text from a team_stats td cell.

    FBref wraps the key value in <strong>, e.g.:
      <td>250 of 352 — <strong>71%</strong></td>
    """
    strong = td.find("strong")
    if strong:
        return strong.get_text(strip=True)
    return td.get_text(strip=True)


def _parse_extra_stats(div: Tag, result: dict) -> None:
    """Parse div#team_stats_extra.

    Layout: groups of <div> children with pattern:
      <div class="th">Team A</div><div class="th"> </div><div class="th">Team B</div>
      <div>17</div><div>Fouls</div><div>12</div>
      <div>4</div><div>Corners</div><div>5</div>
    Each group is a direct child <div> of team_stats_extra.
    """
    for group in div.find_all("div", recursive=False):
        children = group.find_all("div", recursive=False)
        if len(children) < 3:
            continue

        # Process in groups of 3: home_value, label, away_value
        # Skip header rows (class="th")
        i = 0
        while i + 2 < len(children):
            child = children[i]
            if "th" in (child.get("class") or []):
                i += 1
                continue

            home_text = children[i].get_text(strip=True)
            label_text = children[i + 1].get_text(strip=True)
            away_text = children[i + 2].get_text(strip=True)

            # Validate: label should be non-numeric text
            if label_text and not re.match(r"^\d", label_text):
                key = _normalize_stat_key(label_text)
                result[f"home_{key}"] = _parse_stat_value(home_text)
                result[f"away_{key}"] = _parse_stat_value(away_text)

            i += 3


def _normalize_stat_key(label: str) -> str:
    """Convert a label like 'Passing Accuracy' to 'passing_accuracy'."""
    key = label.lower().strip()
    key = re.sub(r"[^a-z0-9]+", "_", key)
    key = key.strip("_")
    return key


def _parse_stat_value(text: str) -> float | int | None:
    """Parse a stat value: '65%' -> 65.0, '12' -> 12, etc."""
    text = text.strip()
    if not text or text == "—":
        return None

    # Percentage
    if text.endswith("%"):
        return safe_float(text.rstrip("%"))

    # Fraction like "14 of 18 — 78%"
    m = re.match(r"(\d+)\s+of\s+(\d+)", text)
    if m:
        return safe_int(m.group(1))

    # Plain number
    return safe_float(text) if "." in text else safe_int(text)
