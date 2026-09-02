"""Blind-spots batch: pure-logic tests for the ops guards (backup rotation,
missed-commit detection). Fantacalcio-side tests live in test_fanta_tracker."""
from __future__ import annotations

import tarfile

from scripts.utils.state_backup import run as backup_run


def test_backup_creates_archive_prunes_and_stamps(tmp_path, monkeypatch):
    monkeypatch.delenv("SERIEA_BACKUP_DEST", raising=False)
    root = tmp_path / "root"
    (root / "data" / "betting").mkdir(parents=True)
    (root / "data" / "betting" / "bet_journal.json").write_text('{"bets": {}}')
    (root / "data" / "pipeline_state.json").write_text("{}")
    dest = tmp_path / "dest"
    dest.mkdir()
    # pre-existing archives: rotation must prune to keep-1 + the new one
    for i in range(5):
        (dest / f"state_2026-01-0{i + 1}_000000.tar.gz").write_bytes(b"x")
    hb = tmp_path / "hb.json"
    out = backup_run(dest=dest, keep=3,
                     targets=("data/betting", "data/pipeline_state.json",
                              "data/missing_is_fine"),
                     root=root, heartbeat=hb)
    assert out.exists() and out.name.startswith("state_")
    with tarfile.open(out) as tf:
        names = tf.getnames()
    assert "data/betting/bet_journal.json" in names
    assert "data/pipeline_state.json" in names
    kept = sorted(dest.glob("state_*.tar.gz"))
    assert len(kept) == 3 and kept[-1] == out
    assert (dest / "state_2026-01-01_000000.tar.gz").exists() is False
    import json
    stamp = json.loads(hb.read_text())
    assert stamp["dest"] == str(out) and stamp["bytes"] > 0


def test_backup_fails_loudly_when_dest_parent_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("SERIEA_BACKUP_DEST", raising=False)
    import pytest
    with pytest.raises(SystemExit):
        backup_run(dest=tmp_path / "no" / "such" / "parent",
                   root=tmp_path, heartbeat=tmp_path / "hb.json")


def _ko(match, mins_from_now, now):
    from datetime import timedelta
    return {"match": match, "kickoff_utc": now + timedelta(minutes=mins_from_now)}


def test_missed_commits_detects_slept_through_window():
    from datetime import UTC, datetime

    from scripts.pipeline.scheduler import _missed_commits
    now = datetime(2026, 9, 5, 16, 0, tzinfo=UTC)
    kicks = [
        _ko("Inter vs Milan", -90, now),       # kicked off, stage fired -> fine
        _ko("Lazio vs Roma", -60, now),        # kicked off, NO stage -> missed
        _ko("Como vs Parma", -13 * 60, now),   # too old (>12h) -> ignored
        _ko("Bologna vs Pisa", 45, now),       # future -> ignored
        _ko("Genoa vs Torino", -30, now),      # missed but already alerted
    ]
    processed = {"Inter vs Milan": {"stages": {"prediction_update": {}}},
                 "Lazio vs Roma": {"stages": {"lineup_fetch": {}}}}
    alerted = {"Genoa vs Torino": "2026-09-05T15:45:00+00:00"}
    assert _missed_commits(kicks, processed, alerted, now) == ["Lazio vs Roma"]


def test_missed_commits_survives_source_name_flips():
    """During a Sofascore ban the kickoff source flips and the same match
    reappears under the other source's spelling ('AC Milan' vs 'Milan') —
    a commit recorded under one spelling must not false-alarm under the
    other. Same for an already-sent alert."""
    from datetime import UTC, datetime

    from scripts.pipeline.scheduler import _missed_commits
    now = datetime(2026, 9, 5, 16, 0, tzinfo=UTC)
    kicks = [_ko("AC Milan vs Internazionale", -60, now),
             _ko("Hellas Verona vs Como", -30, now)]
    processed = {"Milan vs Inter": {"stages": {"prediction_update": {}}}}
    alerted = {"Verona vs Como": "2026-09-05T15:45:00+00:00"}
    assert _missed_commits(kicks, processed, alerted, now) == []


def test_awake_hold_singleton_and_duration(tmp_path, monkeypatch):
    import os
    from datetime import UTC, datetime

    import scripts.pipeline.scheduler as sched
    now = datetime(2026, 9, 5, 14, 0, tzinfo=UTC)
    pidfile = tmp_path / "caffeinate.pid"
    spawned = []
    # the alive-check requires the pid to still BE caffeinate; stand in for it
    monkeypatch.setattr(sched, "_caffeinate_alive",
                        lambda pid: pid == os.getpid())

    def fake_spawn(secs):
        spawned.append(secs)
        return os.getpid()

    # no kickoff within 2h -> no hold (even with one at 5h)
    assert sched._ensure_awake_hold([_ko("A vs B", 300, now)], now,
                                    spawn=fake_spawn, pidfile=pidfile) is False
    assert spawned == []
    # two kickoffs inside 2h -> hold through the LAST + 45min
    kicks = [_ko("A vs B", 60, now), _ko("C vs D", 100, now)]
    assert sched._ensure_awake_hold(kicks, now, spawn=fake_spawn,
                                    pidfile=pidfile)
    assert spawned == [100 * 60 + 45 * 60]
    # second call: pidfile alive -> no new spawn
    assert sched._ensure_awake_hold(kicks, now, spawn=fake_spawn,
                                    pidfile=pidfile)
    assert spawned == [100 * 60 + 45 * 60]
    # dead pid -> respawns
    pidfile.write_text("999999")
    assert sched._ensure_awake_hold(kicks, now, spawn=fake_spawn,
                                    pidfile=pidfile)
    assert len(spawned) == 2


def test_awake_hold_covers_the_whole_matchday_slate(tmp_path, monkeypatch):
    """Sunday slots are ~2.5h apart: a hold ending 45min after kickoff A
    would expire before kickoff B enters the 2h trigger window, leaving an
    unheld seam. The duration must therefore run through the LAST kickoff
    of the next 6h, not just the trigger window."""
    from datetime import UTC, datetime

    import scripts.pipeline.scheduler as sched
    now = datetime(2026, 9, 6, 13, 30, tzinfo=UTC)
    pidfile = tmp_path / "caffeinate.pid"
    monkeypatch.setattr(sched, "_caffeinate_alive", lambda pid: False)
    spawned = []
    kicks = [_ko("A vs B", 90, now),      # 15:00 — inside trigger window
             _ko("C vs D", 240, now),     # 17:30 — outside 2h, inside 6h
             _ko("E vs F", 420, now)]     # 20:30 — outside 6h horizon
    assert sched._ensure_awake_hold(kicks, now, spawn=lambda s: (
        spawned.append(s) or 12345), pidfile=pidfile)
    assert spawned == [240 * 60 + 45 * 60]
