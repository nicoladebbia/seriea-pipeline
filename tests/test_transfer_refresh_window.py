"""The refresh skipped wholesale outside the window, and its dates were literals.

Two defects in `scripts/data/refresh_transfers`:

1. `WINDOW_RANGES` was written as ("2026-06-01", "2026-09-05") /
   ("2027-01-01", "2027-02-05") — correct for exactly one season and then
   silently wrong forever, the same annual fuse CLAUDE.md documents for
   season-stamped filenames. The failure mode is the worst kind: a job that
   skips every run while logging that nothing is wrong.

2. Outside the window the whole run exited. Squad MEMBERSHIP is frozen then --
   that is what a closed window means -- but market values and contract dates
   keep moving, and `detect_changes` reports both. Observed 2026-09-07:
   /rosters' "Recent Changes" newest entry was 2026-09-02 (deadline day) while
   the job ran twice a day saying "outside transfer window — skipping".
"""

from __future__ import annotations

from datetime import date

import pytest

from scripts.data import refresh_transfers as rt


class TestWindowIsDerivedNotLiteral:
    def test_summer_and_winter_land_in_the_seasons_own_years(self):
        assert rt._window_ranges("2026-2027") == [
            (date(2026, 6, 1), date(2026, 9, 5)),
            (date(2027, 1, 1), date(2027, 2, 5)),
        ]

    def test_a_later_season_moves_with_it(self):
        """The literal version answered False for every date of this season."""
        assert rt._in_window(date(2030, 7, 15), "2030-2031") is True
        assert rt._in_window(date(2031, 1, 20), "2030-2031") is True
        assert rt._in_window(date(2030, 11, 1), "2030-2031") is False

    def test_deadline_day_slack_is_kept(self):
        assert rt._in_window(date(2026, 9, 5), "2026-2027") is True
        assert rt._in_window(date(2026, 9, 6), "2026-2027") is False


class TestOffWindowStillTracksSquadChanges:
    @pytest.fixture
    def spy(self, monkeypatch):
        calls: list[str] = []
        monkeypatch.setattr(rt, "_run_change_detection", lambda a: calls.append("changes"))

        import scraper.transfermarkt as tm
        monkeypatch.setattr(tm, "current_league_teams", lambda s, l: {"Inter"})
        monkeypatch.setattr(tm, "scrape_squad_market_values",
                            lambda **k: calls.append("squad") or __import__("pandas").DataFrame())
        for name in ("scrape_transfers", "scrape_rumors"):
            monkeypatch.setattr(tm, name,
                                lambda *a, **k: calls.append("WINDOW_ONLY") or __import__("pandas").DataFrame())
        return calls

    def _run(self, monkeypatch, today):
        monkeypatch.setattr(rt.sys, "argv", ["refresh_transfers", "--season", "2026-2027"])

        # Freeze EVERY clock the module reads, not just the one the gate uses:
        # _log() calls datetime.now(UTC).isoformat() on the same name, so a stub
        # that only answers .date() blows up inside the logging, not the branch
        # under test. Return a real datetime and both callers are satisfied.
        import datetime as _dt
        frozen = _dt.datetime(today.year, today.month, today.day, 12, 0,
                              tzinfo=_dt.timezone.utc)

        class _D(_dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return frozen
        monkeypatch.setattr(rt, "datetime", _D)
        return rt.main()

    def test_outside_the_window_the_squad_and_changes_still_run(self, spy, monkeypatch):
        assert self._run(monkeypatch, date(2026, 11, 15)) == 0
        assert "squad" in spy, "squad values froze for four months"
        assert "changes" in spy, "Recent Changes froze at deadline day"
        assert "WINDOW_ONLY" not in spy, "window-only sources must not be scraped off-window"

    def test_inside_the_window_everything_runs(self, spy, monkeypatch):
        assert self._run(monkeypatch, date(2026, 7, 1)) == 0
        assert "squad" in spy and "changes" in spy
        assert "WINDOW_ONLY" in spy, "the full run must still fetch transfers + rumors"
