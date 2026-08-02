"""
Module: config/settings.py
Purpose: Global configuration constants for FBref scraping, HTTP requests, seasons, paths, and league parameters
Inputs:  None (static configuration)
Outputs: Constants for FBREF_BASE_URL, DATA_DIR, SEASONS, REQUEST_DELAY_SECONDS, HEADERS, etc.
Called by: All modules needing path resolution, HTTP config, or season lists
Depends on: stdlib only (base layer — nothing in this repo may be imported here)
"""

import logging
import os
import re
from datetime import date
from pathlib import Path


def get_current_season() -> str:
    """Return the current season string (e.g. '2025-2026').

    Season boundary: August 1. Before Aug 1 → previous season is current.

    ⚠️ This is the season to SCRAPE, not the season to ANALYSE. Between the
    August 1 rollover and the first matchweek (~3 weeks) it names a season with
    ZERO played matches. Any code that filters a dataframe to "the current
    season" and then computes something from those rows wants
    `latest_season_with_results()` below, not this.
    """
    today = date.today()
    if today.month >= 8:
        return f"{today.year}-{today.year + 1}"
    return f"{today.year - 1}-{today.year}"


def latest_season_with_results(df, season_col: str = "season",
                               result_col: str = "home_score"):
    """Return the latest season that actually has PLAYED matches, or None.

    The counterpart to `get_current_season()`, and the correct choice for every
    consumer that READS match data. Three seasons are routinely distinct:

        get_current_season()          the season we scrape into  (calendar)
        df[season_col].max()          the latest season PRESENT  (incl. fixtures)
        latest_season_with_results()  the latest season PLAYED   (this)

    A plain `.max()` is the tempting version and it is wrong here: fixture rows
    are written before kickoff with a null score, so from the moment next
    season's schedule is ingested `.max()` names a season with no results and
    every downstream mean/count silently reads an empty frame. Measured
    2026-08-02: matches.parquet already held 13 such rows.

    Returns None when nothing has been played at all, which callers must handle
    — the historical failure mode here is a soft empty return that looks like a
    successful run.
    """
    if df is None or len(df) == 0 or season_col not in df:
        return None
    played = df[df[result_col].notna()] if result_col in df else df
    if len(played) == 0:
        return None
    seasons = played[season_col].dropna()
    return str(seasons.max()) if len(seasons) else None


class SecretRedactingFilter(logging.Filter):
    """Strip API keys out of every log record before it is written.

    Found 2026-08-02: the live Odds API key was sitting in plaintext in TEN log
    files — settlement, scheduler, morning, evening, pipeline, errors, the web
    dashboard. Nobody logged it on purpose. `requests` embeds the FULL request
    URL, query string included, in the string form of an HTTPError, so any
    module that logs a failed Odds API call writes `apiKey=<the real key>`.

    That makes per-call-site fixes the wrong shape: the leak is wherever an
    exception is logged, which is everywhere. The primary hook is therefore the
    process-wide LogRecord factory (see install_secret_redaction); this Filter
    class carries the scrubbing logic and is additionally attached to handlers we
    own, as defence in depth.

    It lives in config/settings.py rather than scripts/utils/logging_config.py
    because 163 modules import this one and only ONE imports that one — a
    security control nobody imports protects nobody.

    It redacts two ways, because either alone is insufficient:
      * the literal values of known secret env vars — catches a key that appears
        without its query parameter
      * `apiKey=` / `api_key=` / `token=` / `key=` query params — catches a key
        that has been rotated since the process started, or one this filter was
        never told about
    """

    _PARAM_RE = re.compile(
        r"(?i)\b(apikey|api_key|token|access_token|auth|key)=([^&\s\"']+)"
    )
    _SECRET_ENV_VARS = (
        "ODDS_API_KEY", "OPENAI_API_KEY", "PERPLEXITY_API_KEY",
        "APIFOOTBALL_KEY", "FOOTBALLDATA_KEY", "ANTHROPIC_API_KEY",
        "TELEGRAM_BOT_TOKEN", "GOOGLE_GEMINI_KEY", "GROQ_API_KEY",
        "FLASK_SECRET_KEY",
    )
    # Short values would turn every log line into <redacted> soup.
    _MIN_SECRET_LEN = 12

    @classmethod
    def _literals(cls) -> list[str]:
        out = []
        for var in cls._SECRET_ENV_VARS:
            val = os.environ.get(var)
            if val and len(val) >= cls._MIN_SECRET_LEN:
                out.append(val)
        return out

    @classmethod
    def scrub(cls, text: str) -> str:
        for secret in cls._literals():
            text = text.replace(secret, "<redacted>")
        return cls._PARAM_RE.sub(r"\1=<redacted>", text)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            # Render args in NOW, so lazy %-formatting cannot smuggle a secret
            # past us at format time.
            msg = record.getMessage()
            scrubbed = self.scrub(msg)
            if scrubbed != msg:
                record.msg = scrubbed
                record.args = ()
            if record.exc_info:
                # The traceback text is rendered later by the formatter; the
                # exception's own str() is the part that carries the URL.
                exc = record.exc_info[1]
                if exc is not None and self.scrub(str(exc)) != str(exc):
                    record.exc_info = None
                    record.msg = f"{record.msg} | {self.scrub(str(exc))}"
                    record.args = ()
        except Exception:  # noqa: BLE001 — a logging filter must never raise
            return True
        return True


_REDACTION_INSTALLED = False


def install_secret_redaction(logger: "logging.Logger | None" = None) -> None:
    """Turn on redaction process-wide. Idempotent.

    Implemented with `logging.setLogRecordFactory`, NOT with handler filters.
    Filters were the obvious choice and are not sufficient — measured:

        import logging_config          # installs filters on root + its handlers
        logging.basicConfig(...)       # a module adds its OWN handler afterwards
        log.error("...apiKey=%s", key) # -> LEAKED

    A handler filter can only protect handlers that exist when it is attached,
    and the modules that leaked call basicConfig() after importing anything.
    Every LogRecord in the process goes through the record factory, whenever its
    handler was created, so that is the only hook with no ordering hazard.

    Handler filters are still attached where we own the handlers (setup_logging),
    as defence in depth for records built before this module is imported.
    """
    global _REDACTION_INSTALLED
    if not _REDACTION_INSTALLED:
        previous = logging.getLogRecordFactory()

        def _redacting_factory(*args, **kwargs):
            record = previous(*args, **kwargs)
            try:
                msg = record.getMessage()
                scrubbed = SecretRedactingFilter.scrub(msg)
                if scrubbed != msg:
                    record.msg = scrubbed
                    record.args = ()

                # exc_info is rendered by the FORMATTER, long after this factory
                # runs, so scrubbing the message alone still leaks: an HTTPError
                # carries the full URL in its str(). Measured — this was the one
                # path of five that survived the first version of this fix.
                # Render the traceback here, scrub it, and carry it as text so
                # nothing is lost except the secret.
                if record.exc_info:
                    import traceback as _tb
                    ei = record.exc_info
                    if isinstance(ei, BaseException):
                        ei = (type(ei), ei, ei.__traceback__)
                    if ei and ei[1] is not None:
                        rendered = "".join(_tb.format_exception(*ei))
                        clean = SecretRedactingFilter.scrub(rendered)
                        if clean != rendered:
                            record.exc_info = None
                            record.exc_text = None
                            record.msg = f"{record.getMessage()}\n{clean.rstrip()}"
                            record.args = ()
            except Exception:  # noqa: BLE001,S110 — logging must never raise
                pass  # nosec: a failure to redact must not break logging itself
            return record

        logging.setLogRecordFactory(_redacting_factory)
        _REDACTION_INSTALLED = True

    target = logger or logging.getLogger()
    if not any(isinstance(f, SecretRedactingFilter) for f in target.filters):
        target.addFilter(SecretRedactingFilter())
    for handler in target.handlers:
        if not any(isinstance(f, SecretRedactingFilter) for f in handler.filters):
            handler.addFilter(SecretRedactingFilter())


# FBref
FBREF_BASE_URL = "https://fbref.com"
SERIE_A_COMP_ID = 11

# HTTP
REQUEST_DELAY_SECONDS = 12  # Conservative for historical backfill (revert to 6 after)
REQUEST_TIMEOUT = 30
MAX_RETRIES = 5
BACKOFF_FACTOR = 2  # exponential: 4s, 8s, 16s, 32s, 64s
MAX_BACKOFF = 60
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

# Seasons to process (20 seasons of data for better training)
SEASONS = [
    "2005-2006",
    "2006-2007",
    "2007-2008",
    "2008-2009",
    "2009-2010",
    "2010-2011",
    "2011-2012",
    "2012-2013",
    "2013-2014",
    "2014-2015",
    "2015-2016",
    "2016-2017",
    "2017-2018",
    "2018-2019",
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
    "2025-2026",
]

# League codes for football-data.co.uk
# Maps league name -> (code, display_name)
LEAGUES = {
    "serie_a": ("I1", "Serie A"),
    "premier_league": ("E0", "Premier League"),
    "la_liga": ("SP1", "La Liga"),
    "bundesliga": ("D1", "Bundesliga"),
    "ligue_1": ("F1", "Ligue 1"),
}

# Default league
DEFAULT_LEAGUE = "serie_a"

# Feature engineering
ROLLING_WINDOWS = [3, 5, 10]
ROLLING_WINDOW = 5  # Default single rolling window for feature engineering

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_HTML_DIR = DATA_DIR / "raw" / "html"
RAW_FIXTURES_DIR = DATA_DIR / "raw" / "fixtures"
PARSED_DIR = DATA_DIR / "parsed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"
UPCOMING_DIR = DATA_DIR / "upcoming"
BETTING_DIR = DATA_DIR / "betting"
BANKROLL_DIR = DATA_DIR / "bankroll"
LIVE_DIR = DATA_DIR / "live"
REGISTRY_PATH = DATA_DIR / "registry.json"

# Existing data (from old project)
OLD_PROJECT_ROOT = PROJECT_ROOT.parent
OLD_HTML_DIR = OLD_PROJECT_ROOT / "season_2024_2025"


# ---------------------------------------------------------------------------
# Atomic file writes — prevent corruption from concurrent/interrupted writes
# ---------------------------------------------------------------------------

import json as _json
import tempfile as _tempfile


def atomic_write_json(path: Path, data, indent: int = 2, cls=None):
    """Write JSON atomically: write to temp file, then rename.

    When cls is provided, it takes full control of serialization (json.dump
    ignores `default` when `cls` is set). When cls is None, `default=str`
    handles datetime/Path/etc.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = _tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with open(fd, "w") as f:
            if cls is not None:
                _json.dump(data, f, indent=indent, cls=cls)
            else:
                _json.dump(data, f, indent=indent, default=str)
        Path(tmp).rename(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_parquet(path: Path, df, **kwargs):
    """Write parquet atomically: write to temp file, then rename."""
    import pandas as _pd

    # Safety net: normalize match_date to datetime before every write
    if hasattr(df, "columns") and "match_date" in df.columns:
        df["match_date"] = _pd.to_datetime(df["match_date"], errors="coerce")
    if hasattr(df, "columns") and "matchweek" in df.columns:
        df["matchweek"] = _pd.to_numeric(df["matchweek"], errors="coerce")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    try:
        df.to_parquet(tmp, **kwargs)
        tmp.rename(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# Installed at import time, deliberately: this module is imported by ~163
# others, including every module observed leaking the Odds API key into logs/.
# A redaction hook that requires opting in does not protect anything.
install_secret_redaction()
