"""What `scripts/data/scrape_fbref_missing.py` must get right.

The specimen is real: `tests/fixtures/fbref/fixtures_serie_a_trimmed.html` is the
live FBref schedule page trimmed to its header, three real match rows, and one
real foreign-league nav link — the Crystal Palace v West Ham href that an
unscoped regex over the full page happily returns alongside Serie A's.

Nothing here touches the network. The fetch itself is verified by measurement,
recorded in the module docstring, and by the fetch-1-parse-1 check that put a
botasaurus-fetched report through all five consumers.
"""
from __future__ import annotations

import pathlib

import pytest

from scripts.data import scrape_fbref_missing as s

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "fbref"
SPECIMEN = (FIXTURES / "fixtures_serie_a_trimmed.html").read_text()


# --------------------------------------------------------------------------
# The two directory landmines
# --------------------------------------------------------------------------


def test_reports_go_to_the_hyphen_dir_not_the_underscore_one():
    """The parser's own comment: the wrong dir "yields an empty parse that
    still exits 0". That silent success is why this is pinned.
    """
    assert s.reports_dir("2025-2026").name == "2025-2026"


def test_fixtures_are_read_from_the_underscore_dir():
    """_refresh_fbref_fixtures.py writes season.replace("-", "_")."""
    p = s.fixtures_path("2025-2026")
    assert p.parent.name == "2025_2026"
    assert p.name == "fixtures.html"


def test_the_two_dirs_are_not_the_same_place():
    assert s.fixtures_path("2025-2026").parent != s.reports_dir("2025-2026")


def test_the_fixtures_writer_and_this_reader_agree_on_the_path():
    """_refresh_fbref_fixtures.py writes the file this module reads.

    They derive the location independently — the writer from its own
    PROJECT/data/raw/html, this module from config.settings.RAW_HTML_DIR — so
    nothing but this test stops them drifting apart. A drift would be silent in
    the worst way: the writer writes, this module finds no fixtures.html, and
    the gap reads as "nothing missing" rather than as an error. That failure
    shape already cost three months here, when a stale fixtures.html quietly
    shrank the gap from 119 to 69 instead of failing.
    """
    from scripts.pipeline import _refresh_fbref_fixtures as writer

    season = "2025-2026"
    written = writer.HTML_DIR / season.replace("-", "_") / "fixtures.html"
    assert written == s.fixtures_path(season)


# --------------------------------------------------------------------------
# Extraction — the league-poisoning guard
# --------------------------------------------------------------------------


def test_only_serie_a_matches_are_returned():
    out = s.parse_fixtures(SPECIMEN)
    assert out, "the specimen has three real match rows"
    assert all("Serie-A" in url for url in out.values())


def test_the_foreign_nav_link_is_excluded():
    """The specimen carries a real Premier League href outside any table.

    An unscoped regex over the full page returns 364 ids across Serie A, the
    Premier League and Argentina's Liga Profesional. A foreign report parsed
    into the Serie A parquet is silent poison — it would look like data.
    """
    out = s.parse_fixtures(SPECIMEN)
    assert "1b93b08a" not in out  # Crystal Palace v West Ham, present in the specimen
    assert "Crystal-Palace" not in "".join(out.values())


def test_a_foreign_link_inside_a_sched_table_is_still_excluded():
    """Belt two. Scoping to the table is not enough on its own — if FBref ever
    nests another competition in a sched_ table, the slug check still holds.
    """
    poisoned = (
        '<table id="sched_2025-2026_11_1"><tbody><tr>'
        '<td data-stat="match_report">'
        '<a href="/en/matches/deadbeef/Arsenal-Chelsea-May-1-2026-Premier-League">Match Report</a>'
        "</td></tr></tbody></table>"
    )
    assert s.parse_fixtures(poisoned) == {}


def test_links_are_keyed_by_the_8_hex_hash():
    """The key becomes the filename stem, and the stem becomes the match_id."""
    out = s.parse_fixtures(SPECIMEN)
    assert all(len(h) == 8 and all(c in "0123456789abcdef" for c in h) for h in out)


def test_urls_are_absolute():
    out = s.parse_fixtures(SPECIMEN)
    assert all(u.startswith("https://fbref.com/en/matches/") for u in out.values())


def test_a_page_with_no_schedule_table_yields_nothing_rather_than_raising():
    assert s.parse_fixtures("<html><body><p>nope</p></body></html>") == {}


# --------------------------------------------------------------------------
# Telling a real report from the wall
# --------------------------------------------------------------------------


def test_the_turnstile_wall_is_not_mistaken_for_a_report():
    """Measured 2026-07-16: the headless wall is ~27 KB and says "Just a moment".

    Both halves of the check matter — a wall page that somehow grew past the
    size floor must still be rejected on the marker.
    """
    assert not s.is_real_report("Just a moment..." + "x" * 60_000)


def test_a_short_page_is_not_a_report():
    assert not s.is_real_report("scorebox" + "x" * 100)


def test_an_empty_response_is_not_a_report():
    assert not s.is_real_report("")


def test_a_real_report_is_accepted():
    assert s.is_real_report("<div class='scorebox'>...</div>" + "x" * 60_000)


# --------------------------------------------------------------------------
# The gap
# --------------------------------------------------------------------------


@pytest.fixture
def season_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "RAW_HTML_DIR", tmp_path)
    (tmp_path / "2025_2026").mkdir()
    (tmp_path / "2025-2026").mkdir()
    (tmp_path / "2025_2026" / "fixtures.html").write_text(SPECIMEN)
    return tmp_path


def test_missing_is_published_minus_what_is_on_disk(season_dirs):
    published = set(s.parse_fixtures(SPECIMEN))
    already = sorted(published)[0]
    (season_dirs / "2025-2026" / f"{already}.html").write_text("cached")

    missing = s.find_missing("2025-2026")
    assert already not in missing
    assert set(missing) == published - {already}


def test_season_level_pages_are_not_mistaken_for_reports(season_dirs):
    """fixtures.html / stats_*.html share the reports dir in some seasons.

    Counting them as "already downloaded" would be harmless; counting them as
    match_ids would not — the parser skips them by the same prefixes.
    """
    (season_dirs / "2025-2026" / "fixtures.html").write_text("x")
    (season_dirs / "2025-2026" / "stats_defense.html").write_text("x")
    assert set(s.find_missing("2025-2026")) == set(s.parse_fixtures(SPECIMEN))


def test_no_fixtures_file_reports_nothing_rather_than_guessing(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "RAW_HTML_DIR", tmp_path)
    assert s.find_missing("2025-2026") == {}


def test_a_fixtures_file_with_no_serie_a_links_is_a_schema_break_not_an_empty_gap(
    season_dirs,
):
    """An empty parse and a complete cache look identical from the outside.

    They are not: one means "nothing to do", the other means FBref changed the
    page. Both return {}, but the schema break logs an error — this pins that
    the empty page does not silently read as "all done".
    """
    (season_dirs / "2025_2026" / "fixtures.html").write_text("<html><body>x</body></html>")
    assert s.find_missing("2025-2026") == {}


# --------------------------------------------------------------------------
# The caller's contract
# --------------------------------------------------------------------------


def test_dry_run_fetches_nothing_and_exits_zero(season_dirs, monkeypatch):
    def explode(*a, **k):
        raise AssertionError("--dry-run must not open a browser")

    monkeypatch.setattr(s, "download", explode)
    assert s.main(["--season", "2025-2026", "--dry-run"]) == 0


def test_nothing_missing_exits_zero_without_a_browser(season_dirs, monkeypatch):
    for h in s.parse_fixtures(SPECIMEN):
        (season_dirs / "2025-2026" / f"{h}.html").write_text("cached")

    def explode(*a, **k):
        raise AssertionError("must not open a browser with nothing to fetch")

    monkeypatch.setattr(s, "download", explode)
    assert s.main(["--season", "2025-2026"]) == 0


def test_a_cloudflare_wall_exits_zero(season_dirs, monkeypatch):
    """refresh_weekly_data.py reads the exit code. A wall is FBref's posture,
    not a broken job, and the caller's own comment says Sofascore covers the
    results — so a red weekly job here would only train everyone to ignore it.
    """
    monkeypatch.setattr(
        s, "download",
        lambda *a, **k: {"downloaded": 0, "failed": 1, "status": "walled"},
    )
    assert s.main(["--season", "2025-2026", "--headless"]) == 0


def test_a_walled_first_page_stops_the_batch_instead_of_grinding(season_dirs, monkeypatch):
    """Turnstile gates the SESSION, so a wall on page 1 means a wall on all 69.

    This failed for real on the first run: fetch() discarded the wall page and
    returned "", which made a Cloudflare block look like an empty response, so
    the wall check never fired and the caller's 300s budget was spent proving
    the same point three times over. The fetch must hand back what it saw.
    """
    calls = []

    def one_walled_page(driver_url):
        calls.append(driver_url)
        return "Just a moment..." + "x" * 20_000

    counts = s.download(
        "2025-2026",
        {h: f"https://fbref.com/en/matches/{h}/x-Serie-A" for h in ("aaaaaaaa", "bbbbbbbb", "cccccccc")},
        headless=True,
        limit=None,
        _fetch=one_walled_page,
    )
    assert counts["status"] == "walled"
    assert len(calls) == 1, f"walled on page 1 but still fetched {len(calls)} pages"


def test_the_headless_flag_reaches_the_browser(season_dirs, monkeypatch):
    """--headless is measured-useless, but it is what the weekly caller passes.

    It is honoured rather than overridden: the module probes and reports what
    it finds. If Cloudflare ever relaxes, the weekly job starts working with no
    code change — a constant asserting "headless is blocked" could not.
    """
    seen = {}

    def spy(season, missing, headless, limit):
        seen["headless"] = headless
        return {"downloaded": 0, "failed": 0, "status": "nothing_to_do"}

    monkeypatch.setattr(s, "download", spy)
    s.main(["--season", "2025-2026", "--headless"])
    assert seen["headless"] is True

    s.main(["--season", "2025-2026"])
    assert seen["headless"] is False


def test_importing_the_module_opens_no_browser():
    """botasaurus is imported inside download(), not at module scope.

    --dry-run and the tests must never pay for a Chrome launch, and importing
    web.app-style side effects at import time is how this project got two
    stray daemon threads once already.
    """
    import subprocess
    import sys

    r = subprocess.run(  # noqa: S603 — sys.executable and a literal, no user input
        [sys.executable, "-c",
         "import scripts.data.scrape_fbref_missing, sys;"
         " assert 'botasaurus.browser' not in sys.modules"],
        capture_output=True, cwd=pathlib.Path(__file__).parent.parent,
    )
    assert r.returncode == 0, r.stderr.decode()
