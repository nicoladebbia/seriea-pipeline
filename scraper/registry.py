"""
Module: scraper/registry.py
Purpose: JSON-backed manifest tracking scraping and parsing state per match for resumable downloads
Inputs:  data/registry.json (existing state), match_id and metadata from scrapers
Outputs: data/registry.json (updated with downloaded_at, parsed, parsed_at timestamps)
Called by: scripts/backfill_match_reports.py (mark downloads), parsers (mark parsed), query scripts (check state)
Depends on: storage.paths (registry_path)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from storage.paths import registry_path

log = logging.getLogger(__name__)


@dataclass
class RegistryEntry:
    match_id: str
    season: str
    match_url: str
    html_path: str  # relative to data dir
    downloaded_at: str  # ISO timestamp
    parsed: bool = False
    parsed_at: Optional[str] = None


class Registry:
    """Persistent JSON manifest at data/registry.json."""

    def __init__(self):
        self._path = registry_path()
        self._entries: dict[str, RegistryEntry] = {}
        self._load()

    # -- queries --

    def is_downloaded(self, match_id: str) -> bool:
        return match_id in self._entries

    def is_parsed(self, match_id: str) -> bool:
        entry = self._entries.get(match_id)
        return entry is not None and entry.parsed

    def get_unparsed(self) -> list[RegistryEntry]:
        return [e for e in self._entries.values() if not e.parsed]

    def get_all(self) -> list[RegistryEntry]:
        return list(self._entries.values())

    def get_by_season(self, season: str) -> list[RegistryEntry]:
        return [e for e in self._entries.values() if e.season == season]

    def count_total(self) -> int:
        return len(self._entries)

    def count_downloaded(self) -> int:
        return len(self._entries)

    def count_parsed(self) -> int:
        return sum(1 for e in self._entries.values() if e.parsed)

    # -- mutations --

    def mark_downloaded(
        self,
        match_id: str,
        season: str,
        match_url: str,
        html_path: str,
    ) -> None:
        self._entries[match_id] = RegistryEntry(
            match_id=match_id,
            season=season,
            match_url=match_url,
            html_path=html_path,
            downloaded_at=datetime.now(timezone.utc).isoformat(),
        )
        self._save()

    def mark_parsed(self, match_id: str) -> None:
        entry = self._entries.get(match_id)
        if entry:
            entry.parsed = True
            entry.parsed_at = datetime.now(timezone.utc).isoformat()
            self._save()

    # -- persistence --

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                for mid, raw in data.items():
                    self._entries[mid] = RegistryEntry(**raw)
                log.debug("Loaded registry with %d entries", len(self._entries))
            except (json.JSONDecodeError, TypeError) as exc:
                log.warning("Corrupted registry, starting fresh: %s", exc)
                self._entries = {}

    def _save(self) -> None:
        data = {mid: asdict(entry) for mid, entry in self._entries.items()}
        self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
