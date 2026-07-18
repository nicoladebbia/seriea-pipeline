"""The FBref match-report parsers must key rows by the canonical
{date}_{home}_{away} match_id built from each report's own date+teams — NOT
html_path.stem.

Why this test exists: FBref 2025-26 match reports are saved as {hash}.html, so
keying by the filename stem emitted 8-hex ids that never joined matches.parquet.
Every canonical join then silently dropped the whole current season —
player_impact / team_aggregates / advanced_player (player_stats) and gk_quality
(goalkeeper_stats) went null for 2025-26, and promoted teams (no prior history)
went null for every match. Fixed 2026-07-17; this pins the id logic for both
parsers. See DATA_CATALOG player_stats/goalkeeper_stats rows.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from config.team_names import normalize_team

# (module, name of the per-team parse function to stub)
PARSERS = [
    ("scripts.data.parse_all_player_stats", "parse_player_stats"),
    ("scripts.data.parse_all_goalkeeper_stats", "parse_goalkeeper_stats"),
]


def _install_stubs(monkeypatch, mod, parse_attr, teams, match_date):
    """Stub the HTML extractors so the test pins only the id-building logic."""
    monkeypatch.setattr(mod, "get_soup", lambda html: object())
    monkeypatch.setattr(mod, "extract_team_info_from_html", lambda soup: teams)
    monkeypatch.setattr(mod, "extract_match_date", lambda soup: match_date)
    # echo the match_id the writer passes in, so we can assert on it
    monkeypatch.setattr(
        mod,
        parse_attr,
        lambda soup, team_hash, team_name, is_home, match_id: [
            {"player": f"p_{team_name}", "team": team_name, "match_id": match_id}
        ],
    )


def _report(tmp_path: Path, stem: str) -> Path:
    p = tmp_path / f"{stem}.html"
    p.write_text("<html></html>", encoding="utf-8")
    return p


@pytest.mark.parametrize("modname,parse_attr", PARSERS)
def test_builds_canonical_id_from_date_and_teams(tmp_path, monkeypatch, modname, parse_attr):
    mod = importlib.import_module(modname)
    teams = [
        {"name": "Sassuolo", "is_home": True, "hash": "h1"},
        {"name": "Napoli", "is_home": False, "hash": "h2"},
    ]
    _install_stubs(monkeypatch, mod, parse_attr, teams, "2025-08-23")
    # stem is the FBref hash — the bug used this as the key
    rows = mod.parse_match_html(_report(tmp_path, "07e76b58"), "2025-2026", "07e76b58")

    assert rows, "expected records"
    assert {r["match_id"] for r in rows} == {"2025-08-23_Sassuolo_Napoli"}
    assert all(r["match_id"] != "07e76b58" for r in rows)


@pytest.mark.parametrize("modname,parse_attr", PARSERS)
def test_normalizes_team_names_in_id(tmp_path, monkeypatch, modname, parse_attr):
    mod = importlib.import_module(modname)
    teams = [
        {"name": "Hellas Verona", "is_home": True, "hash": "h1"},
        {"name": "Internazionale", "is_home": False, "hash": "h2"},
    ]
    _install_stubs(monkeypatch, mod, parse_attr, teams, "2025-09-14")
    rows = mod.parse_match_html(_report(tmp_path, "deadbeef"), "2025-2026", "deadbeef")

    expected = f"2025-09-14_{normalize_team('Hellas Verona')}_{normalize_team('Internazionale')}"
    assert {r["match_id"] for r in rows} == {expected}


@pytest.mark.parametrize("modname,parse_attr", PARSERS)
def test_falls_back_to_stem_when_date_unreadable(tmp_path, monkeypatch, modname, parse_attr):
    mod = importlib.import_module(modname)
    teams = [
        {"name": "Milan", "is_home": True, "hash": "h1"},
        {"name": "Cremonese", "is_home": False, "hash": "h2"},
    ]
    _install_stubs(monkeypatch, mod, parse_attr, teams, None)  # date can't be read
    rows = mod.parse_match_html(_report(tmp_path, "f12e0a33"), "2025-2026", "f12e0a33")

    # only when the date is missing does it fall back to the stem
    assert {r["match_id"] for r in rows} == {"f12e0a33"}


# parse_all_lineups has a different parse surface (parse_lineups + lineups_to_records
# instead of a per-team parse fn), so it gets its own stubs. Its keying bug was the
# nastiest: correct-team rows were hash-keyed while an empty-team canonical copy
# masked them from player_impact.
def test_lineups_builds_canonical_id_from_date_and_teams(tmp_path, monkeypatch):
    import scripts.data.parse_all_lineups as lu

    teams = [
        {"name": "Inter", "is_home": True, "hash": "h1"},
        {"name": "Napoli", "is_home": False, "hash": "h2"},
    ]
    monkeypatch.setattr(lu, "get_soup", lambda html: object())
    monkeypatch.setattr(lu, "extract_team_info_from_html", lambda soup: teams)
    monkeypatch.setattr(lu, "extract_match_date", lambda soup: "2026-01-11")
    monkeypatch.setattr(lu, "parse_lineups", lambda soup: ([], []))
    monkeypatch.setattr(
        lu, "lineups_to_records",
        lambda home_lineup, away_lineup, match_id, home_team, away_team: [
            {"match_id": match_id, "team": home_team},
            {"match_id": match_id, "team": away_team},
        ],
    )
    p = tmp_path / "0078428e.html"
    p.write_text("<html></html>", encoding="utf-8")
    rows = lu.parse_match_html(p, "2025-2026", "0078428e")

    assert {r["match_id"] for r in rows} == {"2026-01-11_Inter_Napoli"}
    assert all(r["match_id"] != "0078428e" for r in rows)
    # teams are preserved (the empty-team bug was the real-world failure)
    assert {r["team"] for r in rows} == {"Inter", "Napoli"}
