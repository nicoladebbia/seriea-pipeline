"""Tests for the Capology salary scraper (scraper/capology_salaries).

These run against a SAVED HTML specimen (tests/fixtures/capology_juventus_specimen.html)
— never the live network — so they're a durable regression guard for the day
Capology changes its markup. What they lock:

  1. Per-object parsing pairs each name with ITS OWN salary (the checksum against
     Capology's own stated total is the proof no row drifted or was dropped).
  2. Every field the /rosters page needs survives the object window:
     annual/monthly/weekly gross, verified flag, age, contract expiry.
  3. Derived monthly/weekly = annual/12, annual/52 exactly.
  4. Salaries land in the sane band (no shirt-numbers-as-salary style bug).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scraper import capology_salaries as cap

_SPECIMEN = Path(__file__).parent / "fixtures" / "capology_juventus_specimen.html"

# Capology's own stated annual-fixed wage bill for this Juventus specimen.
_STATED_TOTAL = 135_210_000


@pytest.fixture(scope="module")
def html() -> str:
    return _SPECIMEN.read_text()


@pytest.fixture(scope="module")
def rows(html):
    return [r for r in cap._parse_club_html(html) if r["annual_gross_eur"]]


def test_specimen_exists(html):
    assert "var data" in html
    assert "annual-fixed" in html


def test_checksum_matches_stated_total(rows, html):
    """The load-bearing integrity check: sum of parsed fixed salaries must equal
    Capology's own stated total to the euro. A drift or dropped row breaks this."""
    assert cap._stated_fixed_total(html) == _STATED_TOTAL
    assert sum(r["annual_gross_eur"] for r in rows) == _STATED_TOTAL


def test_name_and_salary_stay_paired(rows):
    """Yıldız & David top the list at €11.11m — the per-object parse must keep the
    name attached to its own salary, not the next row's."""
    top = max(rows, key=lambda r: r["annual_gross_eur"])
    assert top["annual_gross_eur"] == 11_110_000
    assert "Yıldız" in top["player_name"] or "David" in top["player_name"]


def test_all_required_fields_present(rows):
    """Every field /rosters renders must survive the object window for every row."""
    for r in rows:
        assert r["annual_gross_eur"] and r["annual_gross_eur"] > 0
        assert r["monthly_gross_eur"] == r["annual_gross_eur"] // 12
        assert r["weekly_gross_eur"] == r["annual_gross_eur"] // 52
        assert isinstance(r["verified"], bool)
        assert r["capology_age"] is not None and 15 <= r["capology_age"] <= 45
        assert r["contract_expiration"] and r["contract_expiration"].count("-") == 2


def test_salaries_in_sane_band(rows):
    for r in rows:
        assert cap._MIN_SALARY_EUR <= r["annual_gross_eur"] <= cap._MAX_SALARY_EUR


def test_at_least_some_verified(rows):
    # Juventus specimen is a top club — most salaries are Capology-verified.
    assert sum(1 for r in rows if r["verified"]) >= len(rows) // 2


def test_club_slug_map_covers_20_teams():
    """The slug map must cover exactly the 20 Serie A clubs (keys match the
    market_values team column). A missing club = a silent salary gap."""
    assert len(cap.CLUB_SLUGS) == 20
    # a club returning nothing must be a loud failure — the scraper raises; here
    # we just assert every mapped slug is a non-empty string.
    assert all(isinstance(s, str) and s for s in cap.CLUB_SLUGS.values())
