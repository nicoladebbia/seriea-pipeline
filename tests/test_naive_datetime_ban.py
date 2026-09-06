"""No naive `datetime.now()` / `datetime.utcnow()` anywhere in non-legacy code.

The CLAUDE.md rule ("all timestamps UTC-aware") existed since 2026-05-01 and
304 of 416 call sites ignored it on 2026-09-06. Prose is not enforcement; ruff
DTZ003/DTZ005 in pyproject is, and this test is what actually runs it.
Helpers: scripts.utils.match_timing now_utc() / now_local() / to_utc().
"""
import shutil
import subprocess
from pathlib import Path

import pytest
from datetime import UTC

ROOT = Path(__file__).resolve().parent.parent


def test_pyproject_enforces_dtz():
    text = (ROOT / "pyproject.toml").read_text()
    assert '"DTZ005"' in text and '"DTZ003"' in text


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not on PATH")
def test_no_naive_now_in_repo():
    res = subprocess.run(
        [shutil.which("ruff"), "check", "--select", "DTZ003,DTZ005", "--output-format", "concise", "--quiet", "."],
        cwd=ROOT, capture_output=True, text=True,
    )
    lines = [ln for ln in res.stdout.splitlines() if ": DTZ" in ln]
    assert not lines, f"{len(lines)} naive clock reads:\n  " + "\n  ".join(lines[:40])


def test_clock_helpers_are_aware_and_normalise_naive_as_utc():
    from datetime import datetime, timezone

    from scripts.utils.match_timing import now_local, now_utc, to_utc
    assert now_utc().tzinfo is UTC and now_local().utcoffset() is not None
    assert to_utc("2026-05-01T04:30") == datetime(2026, 5, 1, 4, 30, tzinfo=UTC)
    assert to_utc("2026-05-01T04:30:00Z") == to_utc("2026-05-01T04:30+00:00")
    assert to_utc("2026-05-01T06:30+02:00") == datetime(2026, 5, 1, 4, 30, tzinfo=UTC)
    assert to_utc(None) is None and to_utc("") is None and to_utc("nope") is None
    # the failure this pins: aware now minus a parsed naive stamp must not TypeError
    assert (now_utc() - to_utc("2026-05-01T04:30+00:00")).total_seconds() > 0
