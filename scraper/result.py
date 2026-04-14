"""Structured result type for scrapers — distinguishes failure from empty data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScraperResult:
    """Wraps scraper output with status metadata.

    Allows callers to distinguish:
    - Source down (status="failed", data=None)
    - Source returned no data (status="success", data=[] or empty DF)
    - Partial data (status="partial", completeness < 1.0)
    - Degraded fallback (status="degraded", source="fallback")
    """

    data: Any = None
    status: str = "success"        # "success", "partial", "failed", "degraded"
    source: str = ""               # Which source returned data
    completeness: float = 1.0      # 0.0–1.0
    error: str = ""                # Error message if failed

    @property
    def ok(self) -> bool:
        return self.status in ("success", "partial")

    @classmethod
    def success(cls, data: Any, source: str = "") -> ScraperResult:
        return cls(data=data, status="success", source=source, completeness=1.0)

    @classmethod
    def partial(cls, data: Any, source: str = "", completeness: float = 0.5) -> ScraperResult:
        return cls(data=data, status="partial", source=source, completeness=completeness)

    @classmethod
    def failed(cls, source: str = "", error: str = "") -> ScraperResult:
        return cls(data=None, status="failed", source=source, completeness=0.0, error=error)

    @classmethod
    def degraded(cls, data: Any, source: str = "", error: str = "") -> ScraperResult:
        return cls(data=data, status="degraded", source=source, completeness=0.5, error=error)
