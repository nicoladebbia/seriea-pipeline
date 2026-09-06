"""notify() must never reach a real channel from a test.

2026-09-06: every pytest run posted eleven real "Matchweek 3" cards to Telegram
(tests/test_weekly_retrain.py -> auto_retrain -> notify_matchweek_summary, imported
inside the function body so the harness stub of weekly_retrain._notify missed it).
The guard is _sending_suppressed(): PYTEST_CURRENT_TEST or NOTIFY_DISABLED=1.
"""
from __future__ import annotations

import os

import pytest

from scripts.pipeline import notify as N


@pytest.fixture
def transports_spied(monkeypatch):
    calls: list = []
    monkeypatch.setattr(N, "_notify_telegram", lambda *a, **k: calls.append(("tg", a, k)) or True)
    monkeypatch.setattr(N, "_notify_macos", lambda *a, **k: calls.append(("mac", a, k)) or True)
    monkeypatch.setattr(N, "load_preferences", lambda: {})
    return calls


def test_pytest_alone_suppresses_every_channel_and_the_history_row(monkeypatch, tmp_path, transports_spied):
    monkeypatch.delenv("NOTIFY_DISABLED", raising=False)
    assert "PYTEST_CURRENT_TEST" in os.environ  # pytest sets it; the guard reads it
    monkeypatch.setattr(N, "_HISTORY_PATH", tmp_path / "history.jsonl")
    monkeypatch.setattr(N, "PROJECT_ROOT", tmp_path)  # critical + all-channels-down writes logs/emergency_alerts.log
    res = N.notify("body", title="t", level="critical", category="alert", tg_html="<b>x</b>")
    assert res == {"macos": False, "telegram": False, "suppressed": True}
    assert transports_spied == []
    assert not (tmp_path / "history.jsonl").exists()
    assert not (tmp_path / "logs").exists()


def test_notify_disabled_flag_suppresses_outside_pytest(monkeypatch, transports_spied):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("NOTIFY_DISABLED", "1")
    assert N._sending_suppressed()
    res = N.notify("body", title="t", level="info", category="betting")
    assert res.get("suppressed") is True
    assert transports_spied == []


def test_the_guard_is_what_blocks_it(monkeypatch, tmp_path, transports_spied):
    """True positive: with both signals removed the same call reaches the transports."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("NOTIFY_DISABLED", raising=False)
    monkeypatch.setattr(N, "_HISTORY_PATH", tmp_path / "history.jsonl")
    assert not N._sending_suppressed()
    res = N.notify("body", title="t", level="info", category="betting", tg_html="<b>x</b>")
    assert res["telegram"] is True and res["macos"] is True
    assert [c[0] for c in transports_spied] == ["mac", "tg"]
    assert (tmp_path / "history.jsonl").exists()


def test_transports_refuse_on_their_own(monkeypatch, _no_real_notifications):
    """Second layer: a caller that bypasses notify() still cannot send from a test."""
    real_telegram, real_macos = _no_real_notifications  # the unpatched transports
    monkeypatch.delenv("NOTIFY_DISABLED", raising=False)
    monkeypatch.setattr(N, "_load_env_key", lambda name: "x")  # credentials present
    called = []
    monkeypatch.setattr(N.urllib.request, "urlopen", lambda *a, **k: called.append(a))
    monkeypatch.setattr(N.subprocess, "run", lambda *a, **k: called.append(a))
    assert real_telegram("m", "t") is False
    assert real_macos("m", "t") is False
    assert called == []


def test_matchweek_summary_never_sends_under_pytest(monkeypatch, transports_spied):
    """The exact path that spammed: the real journal is read, nothing is posted."""
    monkeypatch.delenv("NOTIFY_DISABLED", raising=False)
    N.notify_matchweek_summary(matchweek=3)
    assert transports_spied == []
