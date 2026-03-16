"""Shared BeautifulSoup utilities for parsing FBref HTML pages."""

from __future__ import annotations

import re
import logging
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup, Comment, Tag

log = logging.getLogger(__name__)


def get_soup(html: str) -> BeautifulSoup:
    """Parse HTML into a BeautifulSoup tree and uncomment hidden tables."""
    soup = BeautifulSoup(html, "lxml")
    _uncomment_tables(soup)
    return soup


def _uncomment_tables(soup: BeautifulSoup) -> None:
    """FBref hides some stat tables inside HTML comments. Unwrap them in-place."""
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        text = str(comment)
        if "<table" in text:
            fragment = BeautifulSoup(text, "lxml")
            for table in fragment.find_all("table"):
                comment.insert_before(table)
            comment.extract()


def find_table(soup: BeautifulSoup, table_id: str) -> Optional[Tag]:
    """Find a table by exact id."""
    return soup.find("table", id=table_id)


def find_tables_by_pattern(soup: BeautifulSoup, pattern: str) -> list[Tag]:
    """Find all tables whose id matches a regex pattern."""
    return soup.find_all("table", id=re.compile(pattern))


def table_to_dataframe(table: Tag) -> pd.DataFrame:
    """Convert a FBref stat table to a DataFrame using data-stat attributes.

    FBref tables use <th data-stat="..."> and <td data-stat="..."> on each cell,
    which provides stable, locale-independent column names.
    """
    if table is None:
        return pd.DataFrame()

    # Extract column names from thead
    thead = table.find("thead")
    columns: list[str] = []
    if thead:
        # Use the last header row (FBref often has a multi-level header)
        header_rows = thead.find_all("tr")
        last_header = header_rows[-1] if header_rows else None
        if last_header:
            for th in last_header.find_all(["th", "td"]):
                stat = th.get("data-stat", "")
                columns.append(stat)

    # Extract body rows
    tbody = table.find("tbody")
    if tbody is None:
        return pd.DataFrame(columns=columns)

    # Columns where csk is a sort key, NOT the display value.
    # For these, always use get_text() instead.
    _CSK_SKIP = {"minute", "player", "team", "nationality", "squad"}

    rows: list[dict[str, str]] = []
    for tr in tbody.find_all("tr"):
        # Skip spacer, total, and section-header rows
        css = tr.get("class", [])
        if any(c in css for c in ("spacer", "thead", "partial_table")):
            continue
        if tr.find("th", {"colspan": True}):
            continue

        row: dict[str, str] = {}
        for cell in tr.find_all(["th", "td"]):
            stat = cell.get("data-stat")
            if not stat:
                continue
            # For most columns, csk provides clean numeric sort values.
            # But for text/minute columns, csk is a sort key that
            # doesn't match the display value (e.g. minute 3 → csk=300).
            csk = cell.get("csk")
            if csk is not None and stat not in _CSK_SKIP:
                row[stat] = csk
            else:
                row[stat] = cell.get_text(strip=True)

        if row:
            rows.append(row)

    df = pd.DataFrame(rows)

    # If columns were found from thead, reorder/filter
    if columns:
        # Keep only columns that exist in both
        valid = [c for c in columns if c in df.columns]
        extra = [c for c in df.columns if c not in columns]
        df = df[valid + extra]

    return df


def safe_text(element: Optional[Tag], default: str = "") -> str:
    """Safely extract text from an element."""
    if element is None:
        return default
    return element.get_text(strip=True)


def safe_int(text: str | None) -> int | None:
    if text is None:
        return None
    text = text.replace(",", "").strip()
    try:
        return int(text)
    except (ValueError, TypeError):
        return None


def safe_float(text: str | None) -> float | None:
    if text is None:
        return None
    text = text.strip()
    try:
        return float(text)
    except (ValueError, TypeError):
        return None
