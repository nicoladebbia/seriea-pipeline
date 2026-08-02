"""API keys must never reach a log file.

Found 2026-08-02: the live Odds API key was sitting in plaintext across TEN log
files — settlement, scheduler, morning, evening, pipeline, errors, the web
dashboard. Nobody logged it on purpose. `requests` embeds the FULL request URL,
query string included, in the string form of an HTTPError, so every module that
logged a failed Odds API call wrote `apiKey=<the real key>` to disk.

Two design decisions are pinned here because both were WRONG on the first
attempt and only measurement caught them:

  1. A handler filter is not enough. Modules call `logging.basicConfig()` AFTER
     importing, adding a handler the filter was never attached to. Only the
     process-wide LogRecord factory has no ordering hazard.
  2. Scrubbing `record.msg` is not enough. `exc_info` is rendered by the
     FORMATTER much later, so the exception's own str() still leaked.

The tests below are written against the real leak shapes, not a tidy one.
"""
from __future__ import annotations

import io
import logging

import pytest

from config.settings import SecretRedactingFilter, install_secret_redaction

SECRET = "sekret1234567890abcdef"


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", SECRET)
    install_secret_redaction()


def _capture(emit) -> str:
    buf = io.StringIO()
    logging.basicConfig(level=logging.INFO, stream=buf, force=True)
    emit(logging.getLogger("scripts.data.odds_fetcher"))
    return buf.getvalue()


# --------------------------------------------------------------------------
# the leak shapes
# --------------------------------------------------------------------------
def test_lazy_percent_args_do_not_leak():
    """`log.error("...%s", key)` — args are formatted at HANDLE time, so a naive
    scrub of record.msg alone would miss this."""
    out = _capture(lambda log: log.error("401 for url ?apiKey=%s", SECRET))
    assert SECRET not in out
    assert "<redacted>" in out


def test_fstring_message_does_not_leak():
    out = _capture(lambda log: log.error(f"401 for url ?apiKey={SECRET}"))
    assert SECRET not in out


def test_basicconfig_called_after_import_is_still_covered():
    """THE ordering test. A handler added after import must still be covered —
    this is precisely the case a handler-attached filter missed."""
    logging.basicConfig(level=logging.INFO, stream=io.StringIO(), force=True)
    out = _capture(lambda log: log.error(f"?apiKey={SECRET}"))
    assert SECRET not in out


def test_a_raised_exception_does_not_leak_and_keeps_its_traceback():
    """exc_info is rendered by the formatter, long after the record is built."""
    def emit(log):
        try:
            raise RuntimeError(
                f"401 Client Error: Unauthorized for url: https://x/events?apiKey={SECRET}"
            )
        except RuntimeError:
            log.exception("fetch failed")

    out = _capture(emit)
    assert SECRET not in out, "the exception string leaked the key"
    assert "<redacted>" in out
    assert "RuntimeError" in out and "Traceback" in out, "traceback must survive"


def test_a_child_logger_propagating_to_root_is_covered():
    out = _capture(lambda _: logging.getLogger("a.b.c").error(f"?apiKey={SECRET}"))
    assert SECRET not in out


def test_a_key_this_filter_was_never_told_about_is_still_stripped():
    """Covers a rotated key: the env var no longer matches, but the query
    parameter shape still does."""
    out = _capture(lambda log: log.error("?apiKey=aBrandNewKeyNobodyRegistered"))
    assert "aBrandNewKeyNobodyRegistered" not in out


# --------------------------------------------------------------------------
# scrub() contract
# --------------------------------------------------------------------------
@pytest.mark.parametrize("param", ["apiKey", "api_key", "token", "access_token", "auth"])
def test_common_credential_params_are_stripped(param):
    got = SecretRedactingFilter.scrub(f"https://x/y?{param}=abcdef123456&regions=eu")
    assert "abcdef123456" not in got
    assert "regions=eu" in got, "non-secret params must survive"


def test_scrub_is_case_insensitive_on_the_param_name():
    assert "abcdef123456" not in SecretRedactingFilter.scrub("?APIKEY=abcdef123456")


def test_scrub_leaves_ordinary_text_alone():
    msg = "Fetched 380 events for serie_a in 1.2s"
    assert SecretRedactingFilter.scrub(msg) == msg


def test_a_short_env_value_is_not_treated_as_a_secret(monkeypatch):
    """Redacting a 2-character value would turn every log line into soup."""
    monkeypatch.setenv("ODDS_API_KEY", "ab")
    assert SecretRedactingFilter.scrub("a rabbit ate a cabbage") == "a rabbit ate a cabbage"


def test_installation_is_idempotent():
    """It is called at import time AND from setup_logging; stacking factories
    would scrub repeatedly and could recurse."""
    for _ in range(5):
        install_secret_redaction()
    out = _capture(lambda log: log.error(f"?apiKey={SECRET}"))
    assert SECRET not in out
    assert out.count("<redacted>") == 1


def test_redaction_never_breaks_logging_on_a_weird_record():
    """A filter that raises would take down the pipeline it protects."""
    out = _capture(lambda log: log.error("plain message with no args"))
    assert "plain message with no args" in out
