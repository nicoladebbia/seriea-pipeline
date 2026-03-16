"""Read and write raw HTML files."""

from pathlib import Path

from storage.paths import html_match_path


def save_html(season: str, match_id: str, html: str) -> Path:
    """Save raw HTML to disk and return the file path."""
    path = html_match_path(season, match_id)
    path.write_text(html, encoding="utf-8")
    return path


def load_html(season: str, match_id: str) -> str:
    """Load raw HTML from disk."""
    path = html_match_path(season, match_id)
    return path.read_text(encoding="utf-8")


def html_exists(season: str, match_id: str) -> bool:
    return html_match_path(season, match_id).exists()
