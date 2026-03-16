"""
Module: scraper/client.py
Purpose: Rate-limited HTTP client with Cloudflare bypass via curl_cffi Chrome TLS impersonation and exponential backoff retries
Inputs:  URLs to fetch, optional Cloudflare cookies, optional user agent
Outputs: HTTP response objects with retry logic and rate limiting enforced
Called by: scripts/backfill_match_reports.py (bulk scraping), other scrapers needing Cloudflare bypass
Depends on: config.settings (REQUEST_DELAY_SECONDS, MAX_RETRIES, BACKOFF_FACTOR, HEADERS)
"""

import logging
import time
from typing import Optional

from config.settings import (
    BACKOFF_FACTOR,
    HEADERS,
    MAX_BACKOFF,
    MAX_RETRIES,
    REQUEST_DELAY_SECONDS,
    REQUEST_TIMEOUT,
)

log = logging.getLogger(__name__)


class FBrefClient:
    """HTTP client that enforces minimum delay between requests and retries on failure.

    When cookies are provided (from a Cloudflare solve), automatically uses
    curl_cffi with Chrome TLS impersonation to bypass Cloudflare's fingerprint
    detection. Falls back to standard requests if curl_cffi is not installed.
    """

    def __init__(
        self,
        delay: float = REQUEST_DELAY_SECONDS,
        cookies: Optional[dict] = None,
        user_agent: Optional[str] = None,
    ):
        self.delay = delay
        self._last_request_time: float = 0
        self._cookies = cookies or {}
        self._user_agent = user_agent

        # Try curl_cffi if we have cookies (needed for Cloudflare bypass)
        self._use_cffi = False
        if cookies:
            try:
                from curl_cffi import requests as cffi_requests
                self._cffi_requests = cffi_requests
                self._use_cffi = True
                log.info("Using curl_cffi with Chrome TLS impersonation")
            except ImportError:
                log.warning(
                    "curl_cffi not installed — falling back to standard requests. "
                    "Cloudflare cookies may not work. Install with: pip install curl_cffi"
                )

        if not self._use_cffi:
            import requests
            self._session = requests.Session()
            headers = dict(HEADERS)
            if user_agent:
                headers["User-Agent"] = user_agent
            self._session.headers.update(headers)
            if cookies:
                self._session.cookies.update(cookies)

    # Server errors that are transient and should be retried
    RETRYABLE_STATUS = {429, 500, 502, 503, 504}

    def get(self, url: str):
        """Fetch *url* with rate limiting, retries, and exponential backoff."""
        for attempt in range(1, MAX_RETRIES + 1):
            self._wait()
            try:
                log.debug("GET %s (attempt %d)", url, attempt)

                if self._use_cffi:
                    resp = self._cffi_get(url)
                else:
                    resp = self._requests_get(url)

                self._last_request_time = time.monotonic()

                if resp.status_code in self.RETRYABLE_STATUS:
                    wait = min(self.delay * BACKOFF_FACTOR ** attempt, MAX_BACKOFF)
                    log.warning(
                        "Server error (%d) for %s. Retrying in %.1fs (attempt %d/%d)",
                        resp.status_code, url, wait, attempt, MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue

                if resp.status_code >= 400:
                    raise _HttpError(resp.status_code, url)

                return resp

            except _HttpError:
                raise
            except Exception as exc:
                if attempt == MAX_RETRIES:
                    log.error("Failed after %d attempts: %s — %s", MAX_RETRIES, url, exc)
                    raise
                wait = min(self.delay * BACKOFF_FACTOR ** attempt, MAX_BACKOFF)
                log.warning("Request error (attempt %d): %s. Retrying in %.1fs", attempt, exc, wait)
                time.sleep(wait)

        # All retries exhausted on server errors — raise so caller can track
        raise _HttpError(0, url)

    def _cffi_get(self, url: str):
        """Fetch using curl_cffi with Chrome impersonation."""
        headers = dict(HEADERS)
        if self._user_agent:
            headers["User-Agent"] = self._user_agent
        return self._cffi_requests.get(
            url,
            impersonate="chrome",
            cookies=self._cookies,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )

    def _requests_get(self, url: str):
        """Fetch using standard requests."""
        return self._session.get(url, timeout=REQUEST_TIMEOUT)

    def _wait(self):
        """Enforce minimum delay between requests."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)


class _HttpError(Exception):
    """HTTP error with status code."""

    def __init__(self, status_code: int, url: str):
        self.status_code = status_code
        self.url = url
        super().__init__(f"{status_code} Client Error for url: {url}")
