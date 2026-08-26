#!/usr/bin/env python3
"""Rebuild ``data/features/player_metadata.json`` from the Sofascore match JSONs.

Feeds the bio block of the player detail page (``/player/<team>/<name>`` ->
``web/app.py:api_player_detail`` -> ``player.html``): height, nationality,
market value, age.

Why this exists
---------------
The file it writes previously had **no writer at all**. It was a one-shot
artifact dated 2026-02-17 that ``web/app.py`` read and nothing maintained, so
every bio on the player page was frozen at February: ages a birthday behind,
market values six months out of date, and any player who arrived after that
date rendered with a blank bio (46 of 671 current-era players, 6.9%).

The bio fields are already present in every match JSON we scrape
(``player.height`` 99.1%, ``dateOfBirthTimestamp`` 100%, ``country`` 100%,
``proposedMarketValueRaw`` 98.9%), so this needs no new network access.

Design notes
------------
* **Date of birth is stored, not age.** Storing a computed ``age`` int is what
  made the old file rot: it is correct only on the day it is written. Readers
  derive age from ``date_of_birth`` at request time. ``age`` is still emitted
  for backward compatibility with any consumer that has not moved over, but it
  is a convenience copy, never the source of truth.
* **Newest match wins** for the fields that genuinely vary (name, position,
  market value); static fields (dob, height, country) take the first non-null.
* **Market value is display-only and stamped.** ``proposedMarketValueRaw`` is
  whatever Sofascore served when the JSON was fetched, not the value as of that
  kickoff — match JSONs are rewritten after the fact. ``market_value_as_of``
  records the match date it was taken from so the UI can be honest about it.
  It must never become a training feature: a backfilled 2019 match would carry
  today's valuation, which is a leak.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from storage.paths import DATA_DIR  # noqa: E402

log = logging.getLogger(__name__)

OUT_PATH = DATA_DIR / "features" / "player_metadata.json"
MATCH_DIRS = ("matches", "matches_premier_league")
STAT_FILES = ("player_match_stats.parquet", "player_match_stats_premier_league.parquet")


def _match_dates() -> dict[str, str]:
    """match_id -> ISO date, from the per-player stat parquets.

    The match JSONs carry no kickoff date of their own, and file mtime is not a
    substitute: Sofascore rewrites a match JSON after kickoff, so mtime tracks
    the last scrape, not the match. The stat parquets already hold the real date
    per match_id for both leagues.
    """
    out: dict[str, str] = {}
    for fname in STAT_FILES:
        p = DATA_DIR / "external" / "sofascore" / fname
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p, columns=["match_id", "date"])
        except (OSError, ValueError, KeyError) as exc:
            log.warning("Could not read %s for match dates: %s", fname, exc)
            continue
        for mid, d in df.drop_duplicates("match_id").itertuples(index=False):
            if pd.notna(d):
                out[str(mid)] = str(d)[:10]
    return out


def _iter_player_entries(dates: dict[str, str]):
    """Yield (match_date, player_dict) for every player, OLDEST MATCH FIRST.

    Ordering by real match date (not glob order, and not season-directory name)
    is what lets the caller apply a plain last-non-null-wins rule: Sofascore
    corrects player records over time — a height typo, a nationality that
    changes when a dual national commits to a federation — and the newest match
    JSON carries the corrected value. Reading in arbitrary order silently
    preserved whichever record happened to be walked first.
    """
    entries: list[tuple[str, Path]] = []
    for sub in MATCH_DIRS:
        root = DATA_DIR / "external" / "sofascore" / sub
        if not root.exists():
            continue
        for fp in root.glob("*/*.json"):
            mid = fp.stem
            # Fall back to the season directory so a match with no date row
            # still sorts roughly right, and never ahead of a dated match.
            entries.append((dates.get(mid) or fp.parent.name, fp))

    for when, fp in sorted(entries):
        try:
            with open(fp) as fh:
                j = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        for side in ("home_lineup", "away_lineup"):
            lineup = j.get(side)
            if not isinstance(lineup, dict):
                continue
            for grp in ("starters", "substitutes"):
                for entry in lineup.get(grp) or []:
                    if isinstance(entry, dict) and isinstance(entry.get("player"), dict):
                        yield when, entry["player"]


def _age_from_dob(dob_iso: str, on: date | None = None) -> int | None:
    try:
        born = datetime.strptime(dob_iso, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None
    on = on or datetime.now(timezone.utc).date()
    return on.year - born.year - ((on.month, on.day) < (born.month, born.day))


def build() -> dict[str, dict]:
    """Collapse every match appearance into one record per player.

    Last non-null wins, walking oldest match first: a later match that simply
    omits a field must never erase it (Sofascore serves thin player objects for
    some fixtures), but a later match that *carries* a field is the more current
    truth and should replace what came before.
    """
    dates = _match_dates()
    meta: dict[str, dict] = {}

    for when, p in _iter_player_entries(dates):
        pid = p.get("id")
        if pid is None:
            continue
        rec = meta.setdefault(str(int(pid)), {})

        if p.get("dateOfBirthTimestamp"):
            rec["date_of_birth"] = datetime.fromtimestamp(
                int(p["dateOfBirthTimestamp"]), tz=timezone.utc
            ).date().isoformat()
        if p.get("height"):
            rec["height"] = int(p["height"])
        country = p.get("country") or {}
        if country.get("name"):
            rec["nationality"] = country["name"]
            rec["country_code"] = country.get("alpha2") or ""
        if p.get("name"):
            rec["name"] = p["name"]
        if p.get("position"):
            rec["position"] = p["position"]
        mv = (p.get("proposedMarketValueRaw") or {}).get("value")
        if mv:
            rec["market_value"] = int(mv)
            rec["market_value_currency"] = (p.get("proposedMarketValueRaw") or {}).get("currency") or "EUR"
            rec["market_value_as_of"] = when or None

    # Convenience copy only — readers should derive from date_of_birth.
    for rec in meta.values():
        if rec.get("date_of_birth"):
            age = _age_from_dob(rec["date_of_birth"])
            if age is not None:
                rec["age"] = age

    return meta


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    meta = build()
    if not meta:
        log.error("No players extracted — refusing to overwrite %s", OUT_PATH)
        return 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(meta, fh, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, OUT_PATH)

    filled = lambda k: sum(1 for r in meta.values() if r.get(k))  # noqa: E731
    log.info(
        "Wrote %d players -> %s (dob %.1f%%, height %.1f%%, nationality %.1f%%, value %.1f%%)",
        len(meta), OUT_PATH,
        100 * filled("date_of_birth") / len(meta), 100 * filled("height") / len(meta),
        100 * filled("nationality") / len(meta), 100 * filled("market_value") / len(meta),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
