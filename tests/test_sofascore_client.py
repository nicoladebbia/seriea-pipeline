"""The one Sofascore client: a denial parks the tier on the FIRST hit, transient
errors are retried, 404 is an answer, and the two tiers are independent
(reliability phase 10, 2026-09-06)."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from scraper import sofascore_client as sc


class _Resp:
    def __init__(self, status, body=None):
        self.status_code = status
        self._body = body

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class _Session:
    """Scripted responses; a response that is an Exception is raised by get()."""
    headers: dict = {}

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url, timeout=20):
        self.calls.append(url)
        if not self.responses:
            raise AssertionError(f"unexpected request #{len(self.calls)} to {url}")
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(sc, "_sleep", slept.append)
    return slept


API = f"{sc.API_BASE}/event/1/incidents"
WWW = f"{sc.WWW_BASE}/api/v1/event/1/incidents"


def test_a_403_is_parked_on_the_first_hit_and_the_next_call_never_leaves_the_process():
    s = _Session(_Resp(403))
    assert sc.get_json(API, session=s) is None
    assert s.calls == [API], "a denial must not be retried"
    assert sc.last_failure_status() == 403
    assert 0 < sc.cooldown_remaining("api") <= sc.COOLDOWN_MINUTES * 60
    assert sc.cooldown_remaining("www") == 0, "the www tier is a different denial"
    # parked: no request at all, and the caller can still tell it was a block
    s2 = _Session()
    assert sc.get_json(API, session=s2) is None
    assert s2.calls == []
    assert sc.last_failure_status() == 403


def test_a_denial_shaped_transport_error_parks_the_tier_too():
    s = _Session(RuntimeError("HTTP 403: challenge"))
    assert sc.get_json(API, session=s) is None
    assert s.calls == [API]
    assert sc.cooldown_remaining("api") > 0


def test_transient_5xx_and_connection_errors_are_retried_with_backoff(_no_sleep):
    s = _Session(_Resp(503), ConnectionError("curl (7) failed to connect"), _Resp(200, {"ok": 1}))
    assert sc.get_json(API, session=s) == {"ok": 1}
    assert len(s.calls) == 3
    assert _no_sleep == [sc.DELAY, sc.DELAY * 2]
    assert sc.cooldown_remaining("api") == 0
    assert sc.last_failure_status() is None


def test_exhausted_transient_retries_report_the_status_and_park_nothing():
    s = _Session(*[_Resp(503)] * 4)
    assert sc.get_json(API, session=s) is None
    assert len(s.calls) == 4
    assert sc.last_failure_status() == 503
    assert sc.cooldown_remaining("api") == 0


def test_404_is_an_answer_not_a_failure():
    s = _Session(_Resp(404))
    assert sc.get_json(API, session=s) is None
    assert sc.last_failure_status() is None
    assert sc.cooldown_remaining("api") == 0


def test_a_200_that_is_not_json_is_a_failure_not_a_crash():
    s = _Session(_Resp(200, ValueError("challenge html")))
    assert sc.get_json(API, session=s) is None
    assert sc.last_failure_status() == 200


def test_the_two_tiers_park_independently_and_expire_on_the_clock():
    sc.set_cooldown("www ban", tier="www", minutes=30)
    assert sc.cooldown_remaining("www") > 0 and sc.cooldown_remaining("api") == 0
    s = _Session(_Resp(200, {"api": "fine"}))
    assert sc.get_json(API, session=s) == {"api": "fine"}
    s2 = _Session()
    assert sc.get_json(WWW, session=s2) is None and s2.calls == []
    later = datetime.now(UTC).timestamp() + 31 * 60
    assert sc.cooldown_remaining("www", now=later) == 0
    state = sc.cooldown_state()
    assert set(state) == {"www"} and state["www"]["reason"] == "www ban"


def test_a_second_denial_while_parked_does_not_extend_the_cooldown():
    sc.set_cooldown("first", tier="api", minutes=10)
    first = sc.cooldown_remaining("api")
    sc.set_cooldown("second", tier="api", minutes=60)
    assert sc.cooldown_remaining("api") <= first
    assert sc.cooldown_state()["api"]["reason"] == "first"


def test_the_events_module_mirrors_the_client_status_for_its_readers():
    """lineup chain / live monitor read sofascore_events._LAST_FAILURE_STATUS."""
    from scraper import sofascore_events as se
    s = _Session(_Resp(403))
    assert se._get_json(API, session=s) is None
    assert se._LAST_FAILURE_STATUS == 403
    s = _Session(_Resp(404))
    # parked now -> no request, still reported as a block
    assert se._get_json(API, session=s) is None and s.calls == []
    assert se._LAST_FAILURE_STATUS == 403


def test_looks_denied_reads_the_catalogued_shapes():
    assert sc.looks_denied(RuntimeError("HTTP 403: challenge"))
    assert sc.looks_denied(RuntimeError("429 Too Many Requests"))
    assert not sc.looks_denied(RuntimeError("curl (7) Failed to connect"))
