"""Append-only transfer-rumor lifecycle store.

WHY THIS EXISTS
---------------
``rumors_<season>.parquet`` is **overwritten** on every refresh (see
``scraper.transfermarkt.scrape_rumors``).  That is correct for the dashboard --
it wants today's live rumors -- but it makes the file useless for any
retrospective question, because you only ever see the rumors that happened to
still be alive on the day you looked.  A rumor that appeared on the 3rd and
died on the 5th leaves no trace at all.  Every retrospective study built on
that file is survivorship-biased, and the history is unrecoverable once the day
passes.

This module accumulates the same rows into ``rumor_history.parquet``, keyed by
content, never overwriting: each rumor gains ``first_seen`` / ``last_seen`` /
``times_seen`` and therefore a measurable LIFETIME.

THE COVERAGE PROBLEM (the part that is easy to get wrong)
---------------------------------------------------------
``last_seen`` alone **cannot** distinguish "the rumor was dropped" from "the
scraper was down".  ``scrape_rumors`` skips a club on any request failure and
``refresh_transfers`` swallows the whole step's exception, so a rumor can go
stale purely because Transfermarkt 403'd for four days.  Those two situations
are OPPOSITE labels for the supervised question "do rumors predict transfers",
and a store that conflates them silently poisons it.

So every run also records, per club, whether it was actually fetched
(``rumor_scrape_log.parquet``), and every history row carries
``last_covered_at``: the last time a SUCCESSFUL run looked at that club.  A
rumor is genuinely dead only when ``last_covered_at > last_seen``.  Use
:func:`annotate_status`, never a bare ``last_seen`` comparison.

FILES WRITTEN
-------------
``data/external/transfermarkt/rumor_history.parquet``
    One row per distinct rumor, ever.  Key ``(league, season, team,
    player_name, current_club)``.
``data/external/transfermarkt/rumor_scrape_log.parquet``
    One row per run: which clubs were covered, how many rows, ok/partial/failed.

NEVER A MODEL FEATURE.  Rumors are speculation; ``compute_net_squad_delta``
reads ``transfers_*``.  This store exists to make rumors *studyable*, which is
a precondition for ever deciding whether they deserve to be a feature.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
TM_DIR = _PROJECT_ROOT / "data" / "external" / "transfermarkt"

HISTORY_NAME = "rumor_history.parquet"
SCRAPE_LOG_NAME = "rumor_scrape_log.parquet"

#: Identity of a rumor.  Content-owned, never positional and never including
#: ``source_url``: that URL embeds a forum ``post_id``, so a fresh post about
#: the SAME rumor would mint a new row and reset ``first_seen`` -- destroying
#: the lifetime measurement this store exists to provide.  The URL is kept as a
#: latest-wins attribute instead.
KEY = ["league", "season", "team", "player_name", "current_club"]

#: Latest-wins attributes: the newest observation replaces the stored one.
_ATTRS = ["age", "market_value_eur", "market_value_text", "source_date", "source_url"]

_LIFECYCLE = ["first_seen", "last_seen", "last_covered_at", "times_seen",
              "first_run_id", "last_run_id"]

#: Per-club fetch outcomes recorded by the scraper.  Only ``ok`` counts as
#: coverage -- a club we failed to fetch tells us nothing about its rumors.
COVERED = "ok"


def _history_path(tm_dir: Path | None = None) -> Path:
    return (tm_dir or TM_DIR) / HISTORY_NAME


def _log_path(tm_dir: Path | None = None) -> Path:
    return (tm_dir or TM_DIR) / SCRAPE_LOG_NAME


def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    """tmp + replace.  A daily job must never leave a half-written parquet
    behind for the morning pipeline to read (paid lesson: the friendlies
    writer, same repo)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def load_history(tm_dir: Path | None = None) -> pd.DataFrame:
    """Every rumor ever observed.  Empty frame (with schema) if none yet."""
    path = _history_path(tm_dir)
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=KEY + _ATTRS + _LIFECYCLE)


def load_scrape_log(tm_dir: Path | None = None) -> pd.DataFrame:
    path = _log_path(tm_dir)
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["run_id", "league", "season", "status",
                                 "teams_expected", "teams_covered", "n_rows",
                                 "covered_teams", "failed_teams"])


def record_run(
    rumors: pd.DataFrame,
    coverage: dict[str, str],
    season: str = "2026-2027",
    league: str = "serie_a",
    now: datetime | None = None,
    tm_dir: Path | None = None,
) -> dict[str, Any]:
    """Fold one scrape into the history and append a run to the scrape log.

    Args:
        rumors: today's live rumors, exactly as ``scrape_rumors`` returned them.
        coverage: ``{club_name: outcome}``.  Only ``"ok"`` clubs count as
            covered; anything else (``fetch_failed``, ...) means we learned
            nothing about that club today and its stored rumors must NOT be
            treated as dropped.
        now: injectable clock for tests.

    Returns a summary dict (also useful for the refresh log line).
    """
    ts = (now or datetime.now(UTC))
    run_id = ts.isoformat()
    covered = {t for t, s in (coverage or {}).items() if s == COVERED}

    hist = load_history(tm_dir)

    fresh = rumors.copy() if rumors is not None else pd.DataFrame()
    if not fresh.empty:
        fresh["league"] = league
        fresh["season"] = season
        for col in KEY + _ATTRS:
            if col not in fresh.columns:
                fresh[col] = None
        # A club fetched OK but whose rows we somehow lack is still covered;
        # conversely never trust rows from a club the caller marked failed.
        if coverage:
            fresh = fresh[fresh["team"].isin(covered)]
        fresh = fresh.drop_duplicates(subset=KEY, keep="last")

    n_new = n_updated = 0
    if hist.empty:
        merged = fresh.copy()
        if not merged.empty:
            merged["first_seen"] = run_id
            merged["last_seen"] = run_id
            merged["last_covered_at"] = run_id
            merged["times_seen"] = 1
            merged["first_run_id"] = run_id
            merged["last_run_id"] = run_id
            n_new = len(merged)
    else:
        merged = hist.copy()
        for col in _LIFECYCLE:
            if col not in merged.columns:
                merged[col] = None
        # 1. Every stored rumor whose club we successfully looked at today has
        #    been *observed*, whether or not it is still listed.  This is what
        #    makes a later absence interpretable.
        if covered:
            same_scope = (merged["league"] == league) & (merged["season"] == season)
            merged.loc[same_scope & merged["team"].isin(covered), "last_covered_at"] = run_id

        # 2. Fold in today's rows: refresh attributes, bump the counters.
        idx = merged.set_index(KEY).index
        for row in fresh.to_dict("records"):
            key = tuple(row[k] for k in KEY)
            hit = idx == key
            if hit.any():
                pos = merged.index[hit]
                for col in _ATTRS:
                    merged.loc[pos, col] = row.get(col)
                merged.loc[pos, "last_seen"] = run_id
                merged.loc[pos, "last_run_id"] = run_id
                merged.loc[pos, "last_covered_at"] = run_id
                merged.loc[pos, "times_seen"] = (
                    pd.to_numeric(merged.loc[pos, "times_seen"], errors="coerce")
                    .fillna(0).astype(int) + 1
                )
                n_updated += 1
            else:
                new = {**{k: row.get(k) for k in KEY + _ATTRS},
                       "first_seen": run_id, "last_seen": run_id,
                       "last_covered_at": run_id, "times_seen": 1,
                       "first_run_id": run_id, "last_run_id": run_id}
                merged = pd.concat([merged, pd.DataFrame([new])], ignore_index=True)
                idx = merged.set_index(KEY).index
                n_new += 1

    if not merged.empty:
        merged["times_seen"] = (pd.to_numeric(merged["times_seen"], errors="coerce")
                                .fillna(1).astype(int))
        _atomic_write(merged[KEY + _ATTRS + _LIFECYCLE], _history_path(tm_dir))

    expected = len(coverage) if coverage else 0
    failed = sorted(set(coverage or {}) - covered)
    status = ("failed" if expected and not covered
              else "partial" if failed else "ok")
    entry = {"run_id": run_id, "league": league, "season": season,
             "status": status, "teams_expected": expected,
             "teams_covered": len(covered), "n_rows": int(len(fresh)),
             "covered_teams": ",".join(sorted(covered)),
             "failed_teams": ",".join(failed)}
    prior = load_scrape_log(tm_dir)
    _atomic_write(pd.concat([prior, pd.DataFrame([entry])], ignore_index=True),
                  _log_path(tm_dir))

    summary = {**entry, "new_rumors": n_new, "updated_rumors": n_updated,
               "total_tracked": int(len(merged))}
    log.info("rumor history: +%d new, %d refreshed, %d tracked (%s)",
             n_new, n_updated, len(merged), status)
    return summary


def annotate_status(hist: pd.DataFrame | None = None,
                    tm_dir: Path | None = None) -> pd.DataFrame:
    """Add derived lifecycle columns.  **Use this, not a raw ``last_seen``.**

    Adds:
        ``days_alive``   -- last_seen minus first_seen, in days.
        ``is_dropped``   -- a successful run looked at the club AFTER the rumor
                            was last listed, so it genuinely disappeared.
        ``is_live``      -- still listed as of the most recent covering run.
        ``days_dark``    -- days since a run last covered this club.  Large
                            values mean the scraper is blind here; treat both
                            ``is_dropped`` and ``is_live`` as unreliable.
    """
    df = (load_history(tm_dir) if hist is None else hist).copy()
    if df.empty:
        for col in ("days_alive", "is_dropped", "is_live", "days_dark"):
            df[col] = pd.Series(dtype="float64" if "days" in col else "bool")
        return df
    first = pd.to_datetime(df["first_seen"], utc=True, errors="coerce")
    last = pd.to_datetime(df["last_seen"], utc=True, errors="coerce")
    cov = pd.to_datetime(df["last_covered_at"], utc=True, errors="coerce")
    df["days_alive"] = (last - first).dt.total_seconds() / 86400.0
    df["is_dropped"] = (cov > last).fillna(False)
    df["is_live"] = ~df["is_dropped"]
    newest = cov.max()
    df["days_dark"] = (newest - cov).dt.total_seconds() / 86400.0
    return df
