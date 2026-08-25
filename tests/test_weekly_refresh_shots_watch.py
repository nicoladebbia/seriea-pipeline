"""The weekly refresh must not exit 1 forever on a dead FBref upstream.

FBref removed the ``shots_all`` table from its match reports starting with
2025-26. Measured 2026-07-16 (absent from all 380 cached 2025-26 reports,
present in 2024-25) and re-measured 2026-08-25 against all 8 cached 2026-27
reports — still absent. The parser is healthy: ``--season 2024-2025`` exits 0,
``--season 2026-2027`` exits 1, on the same code path.

``parse_all_shots`` nevertheless sat in the required-parser loop, so
``refresh_weekly_data`` returned 1 EVERY week and the scheduler notification
always read "1 step(s) failed". A permanently red signal is worse than no
signal — it trains you to ignore the week a real step breaks, because the
failure looks identical.

The tests are DIFFERENTIAL: each runs main() twice against an identical
sandbox, changing only one step's exit, and asserts whether the job's exit
code moves. That isolates the property under test from whatever else does or
does not succeed inside a temp directory.
"""

import pandas as pd
import pytest

import scripts.pipeline.refresh_weekly_data as rwd


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Redirect every PROJECT-relative write and stub the networked steps.

    Step 6 calls ``cache.unlink()`` on the real referee parquets, so main()
    must never run against the real project root.
    """
    monkeypatch.setattr(rwd, "PROJECT", tmp_path)
    for sub in ("data/parsed", "data/external/referee", "data/external/understat"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)

    # The baseline run MUST reach exit 0, or every differential below compares
    # 1 to 1 and passes without testing anything.
    from config.leagues import ACTIVE_LEAGUES
    pd.DataFrame({
        "match_id": ["m1"],
        "league": [ACTIVE_LEAGUES[0]],
        "season": [rwd.CURRENT_SEASON],
    }).to_parquet(tmp_path / "data" / "parsed" / "matches.parquet")

    monkeypatch.setattr(rwd, "step_refresh_fbref_fixtures", lambda: True)

    import scraper.referee as ref
    import scraper.understat_scraper as us
    import scraper.weather as wx
    monkeypatch.setattr(us, "scrape_understat_xg", lambda *a, **k: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(ref, "scrape_all_referee_assignments",
                        lambda *a, **k: pd.DataFrame({"season": [rwd.CURRENT_SEASON]}))
    monkeypatch.setattr(wx, "fetch_weather_for_matches",
                        lambda *a, **k: pd.DataFrame({"match_id": ["m1"]}))

    import scripts.pipeline.notify as notify
    monkeypatch.setattr(notify, "notify_scheduler_run", lambda *a, **k: None)
    return tmp_path


def _run_main(monkeypatch, *, failing_module=None):
    """Run main() with a fake `run`; optionally fail exactly one module."""
    invoked = []

    def fake_run(cmd, step, timeout=3600):
        invoked.append(" ".join(cmd))
        if failing_module and failing_module in cmd:
            return False
        return True

    monkeypatch.setattr(rwd, "run", fake_run)
    return rwd.main(), invoked


def test_the_sandbox_baseline_is_green(sandbox, monkeypatch):
    """Precondition for every differential below. Asserted, not assumed."""
    code, _ = _run_main(monkeypatch)
    assert code == 0, (
        f"baseline exit {code}: something in the sandbox fails independently, so "
        "the differential tests would compare 1 to 1 and pass vacuously"
    )


def test_a_dead_shots_upstream_does_not_change_the_jobs_exit_code(sandbox, monkeypatch):
    """The whole point: shots failing must not turn the weekly job red."""
    healthy, _ = _run_main(monkeypatch)
    assert healthy == 0, "baseline not green — see test_the_sandbox_baseline_is_green"
    shots_dead, _ = _run_main(monkeypatch, failing_module="scripts.data.parse_all_shots")
    assert shots_dead == healthy, (
        f"shots failing moved the exit code {healthy} -> {shots_dead}; it is a "
        "dead upstream the job cannot act on and must not gate on"
    )


def test_a_required_parser_failing_still_changes_the_exit_code(sandbox, monkeypatch):
    """True positive: proves the test above has power and the job can still fail.

    Without this, making main() return a constant would satisfy every other
    assertion in this file.
    """
    healthy, _ = _run_main(monkeypatch)
    assert healthy == 0, "baseline not green — see test_the_sandbox_baseline_is_green"
    lineups_dead, _ = _run_main(monkeypatch, failing_module="scripts.data.parse_all_lineups")
    assert lineups_dead != healthy, (
        "a genuinely required parser failing left the exit code unchanged — "
        "the job can no longer report failure at all"
    )


def test_the_shots_parser_is_still_actually_run(sandbox, monkeypatch):
    """Excluded from the exit code, NOT deleted — it is the only recovery signal."""
    _, invoked = _run_main(monkeypatch)
    assert any("scripts.data.parse_all_shots" in c for c in invoked), (
        "parse_all_shots is no longer invoked at all; nothing would ever notice "
        "if FBref restored shots_all"
    )


def test_the_shots_parser_runs_against_the_current_season(sandbox, monkeypatch):
    _, invoked = _run_main(monkeypatch)
    shots = [c for c in invoked if "scripts.data.parse_all_shots" in c]
    assert shots, "parse_all_shots not invoked"
    assert rwd.CURRENT_SEASON in shots[0]


def test_the_watch_result_never_reaches_the_exit_code_arithmetic(sandbox, monkeypatch):
    """Whether shots_all is back or still gone, the exit code is the same.

    Covers the good-news direction too: a firing watch is not a failure.
    """
    back, _ = _run_main(monkeypatch)
    gone, _ = _run_main(monkeypatch, failing_module="scripts.data.parse_all_shots")
    assert back == gone == 0


def test_shots_is_absent_from_the_required_parser_loop(sandbox, monkeypatch):
    """Pins WHERE the exclusion lives, so it cannot be reintroduced by edit."""
    import inspect
    src = inspect.getsource(rwd.main)
    # Scope to the list literal itself — a prose comment naming the parser is
    # exactly what this exclusion needs to keep saying.
    literal = src[src.index("for parser_name, label in ["):]
    literal = literal[:literal.index("]")]
    assert "parse_all_shots" not in literal, (
        "parse_all_shots is back inside the required-parser loop"
    )
