"""One Sofascore HTTP client — reliability phase 10 (2026-09-06).

Until this module existed, six modules talked to Sofascore with five different
breakers: sofascore_events retried a 403 four times with backoff (the retry
storm that got the IP blanket-denied on 2026-09-06 — 310 logged 403s in two
days), live_sofascore kept its own 10-minute breaker in pipeline_state.json,
matchday_updater wrote a 60-minute cooldown file, sofascore_standings backed
off 1s/2s on 403 with its own health dict, and xi_advisor / worldcup counted
failures locally. None of them told the others a denial had happened.

Rules, all here and nowhere else:

- **A denial (401/403/429, or a transport error whose text says so) parks the
  TIER for COOLDOWN_MINUTES, on the first hit, with no retry.** Every retry is
  a vote for a longer ban. The cooldown is a file, so every process (scheduler
  children, the web app, the live loop) sees the same one.
- **Two tiers, two cooldowns.** ``api.sofascore.com`` (JSON API) and
  ``www.sofascore.com`` (HTML pages and the ``/api/v1`` proxy) are denied
  independently — the catalogue's third 403 shape is an API-only challenge
  while ``www`` still answers 200.
- **Transient statuses (5xx) and connection errors are retried with backoff;
  404 is an answer, not a failure.** A 404 on ``/lineups`` means "not published
  yet", so it never counts toward anything.
- ``last_failure_status()`` says why the last call returned None (403 while
  cooling down, 503 after retries, None on 404/success), so callers can tell
  "not published" from "blocked" — the lineup chain and live monitor read it.

Tests: tests/test_sofascore_client.py. The cooldown directory is redirected to
a temp dir by an autouse fixture in tests/conftest.py, so no test can park the
real ingest.
"""
from __future__ import annotations

import json
import logging
import random
import time
from datetime import UTC, datetime
from typing import Any

try:
    from curl_cffi import requests as cffi_requests
    _HAS_CFFI = True
except ImportError:  # pragma: no cover - curl_cffi is installed in every environment that scrapes
    _HAS_CFFI = False

from config.settings import DATA_DIR, atomic_write_json

log = logging.getLogger(__name__)

API_BASE = "https://api.sofascore.com/api/v1"
WWW_BASE = "https://www.sofascore.com"
COOLDOWN_DIR = DATA_DIR / "monitoring"
COOLDOWN_FILES = {"api": "sofascore_cooldown.json", "www": "sofascore_cooldown_www.json"}
COOLDOWN_MINUTES = 60
DENIED_STATUSES = (401, 403, 429)
TRANSIENT_STATUSES = (500, 502, 503, 504)
DENIED_MARKERS = ("403", "forbidden", "challenge", "429", "too many requests")
DELAY = 2.5  # seconds between polite requests (session reuse makes a low delay safe)
MAX_BACKOFF = 60.0
HEADERS = {
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.sofascore.com/",
}
IMPERSONATE_OPTIONS = ["chrome", "chrome110", "chrome120", "safari", "safari15_5"]

_sessions: dict[str, Any] = {}
_impersonate_idx: dict[str, int] = {"api": 0, "www": 0}
_LAST_STATUS: int | None = None
_consecutive_failures = 0
_sleep = time.sleep  # patched by tests


# ---------------------------------------------------------------------------
# Cooldown (the one breaker)
# ---------------------------------------------------------------------------
def tier_of(url: str) -> str:
    return "www" if "www.sofascore.com" in url else "api"


def cooldown_path(tier: str = "api"):
    return COOLDOWN_DIR / COOLDOWN_FILES[tier]


def cooldown_remaining(tier: str = "api", now: float | None = None) -> float:
    """Seconds left on the tier's cooldown, 0 when none."""
    try:
        d = json.loads(cooldown_path(tier).read_text())
        until = float(d.get("until_ts") or 0)
    except (OSError, ValueError, TypeError):
        return 0.0
    now = now if now is not None else datetime.now(UTC).timestamp()
    return max(0.0, until - now)


def set_cooldown(reason: str, *, tier: str = "api", minutes: int = COOLDOWN_MINUTES) -> None:
    """Park the tier. Idempotent: a denial while already parked extends nothing."""
    if cooldown_remaining(tier) > 0:
        return
    now = datetime.now(UTC)
    payload = {"tier": tier, "set_at": now.isoformat(), "until_ts": now.timestamp() + minutes * 60,
               "minutes": minutes, "reason": reason[:200]}
    atomic_write_json(cooldown_path(tier), payload, indent=None)
    log.warning("Sofascore %s tier parked for %d min: %s", tier, minutes, reason[:120])


def clear_cooldown(tier: str = "api") -> None:
    try:
        cooldown_path(tier).unlink()
    except OSError:
        pass


def cooldown_state(now: float | None = None) -> dict[str, dict]:
    """{tier: {remaining_s, reason, set_at}} for status cards and health checks."""
    out: dict[str, dict] = {}
    for tier in COOLDOWN_FILES:
        rem = cooldown_remaining(tier, now)
        if rem <= 0:
            continue
        try:
            d = json.loads(cooldown_path(tier).read_text())
        except (OSError, ValueError):
            d = {}
        out[tier] = {"remaining_s": int(rem), "reason": d.get("reason"), "set_at": d.get("set_at")}
    return out


def looks_denied(exc: BaseException) -> bool:
    s = str(exc).lower()
    return any(m in s for m in DENIED_MARKERS)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
def get_session(tier: str = "api") -> Any:
    """A persistent session per tier. Cloudflare intermittently blocks NEW TCP
    connections (~40% failure rate measured) while an established keep-alive
    connection keeps answering — reuse is the reliability feature."""
    sess = _sessions.get(tier)
    if sess is None:
        imp = IMPERSONATE_OPTIONS[_impersonate_idx[tier] % len(IMPERSONATE_OPTIONS)]
        if _HAS_CFFI:
            sess = cffi_requests.Session(impersonate=imp)
        else:  # pragma: no cover
            import requests as _req
            sess = _req.Session()
        sess.headers.update(HEADERS)
        _sessions[tier] = sess
        log.info("Created Sofascore %s session (impersonate=%s)", tier, imp)
    return sess


def reset_session(tier: str = "api") -> None:
    sess = _sessions.pop(tier, None)
    if sess is not None:
        try:
            sess.close()
        except Exception as e:  # noqa: BLE001
            log.debug("session close failed: %s", e)
    _impersonate_idx[tier] += 1


def last_failure_status() -> int | None:
    """Why the last get_json returned None: a status code, or None (success / 404)."""
    return _LAST_STATUS


def jitter_delay(base: float = DELAY) -> None:
    """Sleep base ± 30% (politeness between two requests, not a retry)."""
    jitter = base * 0.3 * (2 * random.random() - 1)  # noqa: S311 - not crypto
    _sleep(max(1.0, base + jitter))


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------
def get_json(url: str, *, session: Any = None, timeout: int = 20, attempts: int = 4) -> dict | None:
    """GET a Sofascore JSON endpoint under the rules in the module docstring."""
    global _LAST_STATUS, _consecutive_failures
    tier = tier_of(url)
    _LAST_STATUS = None
    remaining = cooldown_remaining(tier)
    if remaining > 0:
        _LAST_STATUS = 403
        log.debug("Sofascore %s tier cooling down (%ds left) — not requesting %s", tier, remaining, url)
        return None
    sess = session or get_session(tier)
    for attempt in range(attempts):
        try:
            resp = sess.get(url, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - transport errors vary by backend
            _consecutive_failures += 1
            if looks_denied(exc):
                _LAST_STATUS = 403
                set_cooldown(f"{type(exc).__name__}: {str(exc)[:120]} on {url}", tier=tier)
                reset_session(tier)
                return None
            wait = min(MAX_BACKOFF, DELAY * (2 ** attempt))
            log.warning("Sofascore request error (attempt %d/%d, consec=%d): %s",
                        attempt + 1, attempts, _consecutive_failures, str(exc)[:80])
            if session is None and attempt >= 1:
                reset_session(tier)
                sess = get_session(tier)
            if attempt < attempts - 1:
                _sleep(wait)
            continue
        code = resp.status_code
        if code == 200:
            _consecutive_failures = 0
            _LAST_STATUS = None
            try:
                return resp.json()
            except Exception as exc:  # noqa: BLE001 - a challenge page can come back as 200 HTML
                _LAST_STATUS = 200
                log.warning("Sofascore 200 with non-JSON body for %s: %s", url, str(exc)[:80])
                return None
        if code == 404:
            _consecutive_failures = 0
            _LAST_STATUS = None
            return None
        if code in DENIED_STATUSES:
            _LAST_STATUS = code
            set_cooldown(f"HTTP {code} on {url}", tier=tier)
            reset_session(tier)
            return None
        if code in TRANSIENT_STATUSES:
            _LAST_STATUS = code
            _consecutive_failures += 1
            wait = min(MAX_BACKOFF, DELAY * (2 ** attempt))
            log.warning("HTTP %d for %s, waiting %.0fs (attempt %d/%d)", code, url, wait, attempt + 1, attempts)
            if attempt < attempts - 1:
                _sleep(wait)
            continue
        _LAST_STATUS = code
        log.warning("HTTP %d for %s", code, url)
        return None
    return None
