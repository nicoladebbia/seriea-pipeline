"""The live loop process at a frozen clock (reliability phase 8, 2026-09-06):
arms on the kickoff window or a web request, polls on its interval, fast-ticks
between polls, stops after four empty polls and does not re-arm for a finished
match, and announces itself through a heartbeat the web app and health check read."""
from __future__ import annotations

import json

import pytest

import scripts.data.live_monitor as lm


@pytest.fixture
def state(monkeypatch):
    store: dict = {}
    import scripts.pipeline.pipeline_state as ps
    monkeypatch.setattr(ps, "load_state", lambda: dict(store))

    def save_state(st):
        store.clear()
        store.update(st)
        return "fake"

    monkeypatch.setattr(ps, "save_state", save_state)
    return store


@pytest.fixture
def status_file(tmp_path, monkeypatch):
    p = tmp_path / "live_loop_status.json"
    monkeypatch.setattr(lm, "LIVE_LOOP_STATUS_FILE", p)
    return p


class _Poll:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        r = self.results.pop(0) if self.results else {"has_live_matches": True}
        if isinstance(r, Exception):
            raise r
        return r


def _win(open_: bool):
    seen = []

    def window(stopped_at=0.0):
        seen.append(stopped_at)
        return open_

    window.seen = seen
    return window


def test_window_closed_means_no_poll_and_no_arming(state):
    st = lm.LiveLoopState(interval=60, fast_interval=5)
    poll = _Poll()
    lm.live_loop_step(st, 1000.0, poll=poll, fast=_Poll(), window=_win(False))
    assert st.active is False and poll.calls == 0


def test_window_open_arms_polls_now_and_fast_ticks_between_polls(state):
    st = lm.LiveLoopState(interval=60, fast_interval=5)
    poll, fast = _Poll({"has_live_matches": True}), _Poll({"has_live_matches": True, "refreshed": 2})
    lm.live_loop_step(st, 1000.0, poll=poll, fast=fast, window=_win(True))
    assert st.active and poll.calls == 1 and st.has_live
    assert st.next_poll_at == 1060.0 and fast.calls == 0
    lm.live_loop_step(st, 1004.0, poll=poll, fast=fast, window=_win(True))
    assert fast.calls == 0, "fast tick is not due before fast_interval"
    lm.live_loop_step(st, 1005.0, poll=poll, fast=fast, window=_win(True))
    assert fast.calls == 1 and st.fast_refreshed == 2 and st.fast_at == 1005.0
    assert poll.calls == 1, "the Odds API poll is not repeated before its interval"
    lm.live_loop_step(st, 1060.0, poll=poll, fast=fast, window=_win(True))
    assert poll.calls == 2


def test_four_empty_polls_stop_the_loop_and_the_window_learns_the_stop_time(state):
    st = lm.LiveLoopState(interval=60, fast_interval=5)
    poll = _Poll(*[{"has_live_matches": False}] * 4)
    w = _win(True)
    now = 1000.0
    for _ in range(4):
        lm.live_loop_step(st, now, poll=poll, fast=_Poll(), window=w)
        now += 60.0
    assert poll.calls == 4 and st.active is False and st.stopped_at == 1180.0
    # the next arm check hands the stop time to the gate, which decides on it
    w2 = _win(False)
    lm.live_loop_step(st, now + 60.0, poll=poll, fast=_Poll(), window=w2)
    assert w2.seen == [1180.0] and st.active is False and poll.calls == 4


def test_a_web_stop_request_is_consumed_once_and_a_start_request_arms_off_window(state):
    st = lm.LiveLoopState(interval=60, fast_interval=5)
    poll = _Poll()
    lm.live_loop_step(st, 1000.0, poll=poll, fast=_Poll(), window=_win(True))
    assert st.active
    lm.request_live_loop("stop")
    lm.live_loop_step(st, 1001.0, poll=poll, fast=_Poll(), window=_win(False))
    assert st.active is False
    # the same stop request must not fight a later re-arm
    st.next_arm_check = 0.0
    lm.live_loop_step(st, 1100.0, poll=poll, fast=_Poll(), window=_win(True))
    assert st.active is True
    lm.request_live_loop("stop")
    lm.live_loop_step(st, 1101.0, poll=poll, fast=_Poll(), window=_win(False))
    assert st.active is False
    lm.request_live_loop("start")
    lm.live_loop_step(st, 1102.0, poll=poll, fast=_Poll(), window=_win(False))
    assert st.active is True, "an explicit start request arms even with the window closed"


def test_an_interval_chosen_on_the_page_applies_at_the_next_poll(state):
    st = lm.LiveLoopState(interval=300, fast_interval=5)
    poll = _Poll()
    lm.live_loop_step(st, 1000.0, poll=poll, fast=_Poll(), window=_win(True))
    assert st.next_poll_at == 1300.0
    lm.save_live_poll_interval(60)
    lm.live_loop_step(st, 1300.0, poll=poll, fast=_Poll(), window=_win(True))
    assert st.interval == 60 and st.next_poll_at == 1360.0


def test_three_auth_errors_stop_the_loop_but_one_generic_error_does_not(state):
    st = lm.LiveLoopState(interval=60, fast_interval=5)
    poll = _Poll(RuntimeError("boom"), RuntimeError("401 Unauthorized"), RuntimeError("401 Unauthorized"),
                 RuntimeError("OUT_OF_USAGE_CREDITS"))
    now = 1000.0
    lm.live_loop_step(st, now, poll=poll, fast=_Poll(), window=_win(True))
    assert st.active, "a generic error keeps the loop alive"
    for _ in range(3):
        now += 60.0
        lm.live_loop_step(st, now, poll=poll, fast=_Poll(), window=_win(True))
    assert st.active is False and st.stopped_at == 0.0, "an auth stop is not a finished match"


def test_the_process_writes_a_heartbeat_the_web_app_reads_and_a_stale_one_reads_dead(state, status_file, monkeypatch):
    monkeypatch.setattr(lm, "poll_once", _Poll({"has_live_matches": True}))
    monkeypatch.setattr(lm, "refresh_live_fast", _Poll({"has_live_matches": True, "refreshed": 1}))
    monkeypatch.setattr(lm, "live_window_open", _win(True))
    clock = iter([1000.0, 1001.0, 1002.0, 1003.0])
    st = lm.run_live_loop(sleep=lambda s: None, clock=lambda: next(clock), max_ticks=3)
    assert st.active and status_file.exists()
    raw = json.loads(status_file.read_text())
    # written on the state change at tick 1 (armed + polled); ticks 2-3 changed nothing
    # and are inside the 15s heartbeat cadence, so the file still says 1000.0
    assert raw["active"] is True and raw["pid"] and raw["heartbeat"] == 1000.0 and raw["polls"] == 1
    fresh = lm.read_live_loop_status(now=1010.0)
    assert fresh["alive"] is True and fresh["active"] is True and fresh["heartbeat_age_s"] == 10
    stale = lm.read_live_loop_status(now=1000.0 + lm.LIVE_LOOP_STALE_SEC + 1)
    assert stale["alive"] is False and stale["active"] is False, "a dead loop is not polling, whatever its file said"
    assert lm.read_live_loop_status(now=1.0)["alive"] is True  # a heartbeat in the future is still a heartbeat


def test_no_status_file_reads_as_never_ran(status_file):
    st = lm.read_live_loop_status(now=1000.0)
    assert st["alive"] is False and st["heartbeat_age_s"] is None and st["active"] is False


def test_health_check_reads_the_heartbeat_and_escalates_inside_the_window():
    from scripts.pipeline.health_check import check_live_loop
    alive = {"alive": True, "active": True, "pid": 42, "heartbeat_age_s": 3}
    assert check_live_loop(alive, window_open=True)["status"] == "OK"
    dead = {"alive": False, "active": False, "pid": 42, "heartbeat_age_s": 900}
    assert check_live_loop(dead, window_open=False)["status"] == "WARNING"
    crit = check_live_loop(dead, window_open=True)
    assert crit["status"] == "CRITICAL" and "900s old" in crit["detail"]
    never = check_live_loop({"alive": False, "heartbeat_age_s": None}, window_open=True)
    assert never["status"] == "CRITICAL" and "never" in never["detail"]
