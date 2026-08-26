#!/usr/bin/env python3
"""Re-key legacy FBref-hash ``match_id``s to the canonical ``{date}_{home}_{away}``.

Why this exists
---------------
``parse_all_player_stats.py`` used to key rows by ``html_path.stem``. FBref saves
match reports as ``{hash}.html``, so every row it wrote carried an 8-hex id that
does not exist in ``matches.parquet``. The writer was fixed 2026-07-17, but the
fix only re-keyed the season that was broken *at the time* (2025-26). Seasons
2017-18 through 2023-24 were left hash-keyed and nobody noticed, because a
hash-vs-canonical mismatch does not raise — a ``merge(on="match_id")`` simply
returns no rows and the feature columns come out silently NaN.

Measured 2026-08-26 on Serie A: three feature families merge per-match on
``match_id`` and were therefore 0% filled for all seven seasons —
``adv_roll5_*`` (76 cols), ``tagg_roll5_*`` (52) and player_impact's key-player
block (8). The families that survived (``fb_roll_*``, ``lineup_*``,
suspensions) all aggregate to (season, team) and never join on match_id, which
is exactly why the defect stayed invisible for so long.

The id is rebuilt from each file's own ``match_date`` + ``team`` + ``is_home``,
the same three fields the fixed writer uses, so a re-keyed row is
byte-identical to what the writer would emit today. ``team`` is already
normalised on write, which is why the reconstruction lands at 99.2%.

Safe to re-run: rows already canonical are left alone.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from storage.paths import parsed_path  # noqa: E402

log = logging.getLogger(__name__)

# An FBref report hash: 8+ hex chars and nothing else. A canonical id always
# carries underscores and a leading date, so the two can never be confused.
HASH_RE = re.compile(r"^[0-9a-f]{8,}$")

# Files that the old stem-keyed writer fed, in dependency order. Both are Serie
# A; the EPL variant was written by a different script that always built the
# canonical id. ``goalkeeper_stats`` carries no ``match_date`` of its own, so it
# borrows the mapping player_stats derives — the two were parsed from the same
# reports and share all 2,640 hash ids exactly (verified 2026-08-26).
TARGETS = ("player_stats", "goalkeeper_stats")

# Below this we refuse to write. A healthy run reconstructs ~99%; a sudden drop
# means the schema moved and the rebuilt ids would be garbage.
MIN_RECONSTRUCTION = 0.90


def is_hash_keyed(match_ids: pd.Series) -> pd.Series:
    """Which rows still carry a legacy FBref hash id."""
    return match_ids.astype(str).str.fullmatch(HASH_RE).fillna(False)


def build_hash_to_canonical(df: pd.DataFrame) -> dict[str, str]:
    """Map every legacy hash id in ``df`` to its canonical ``{date}_{home}_{away}``.

    Uses an explicit home/away merge rather than a groupby-apply so the join is
    the thing under test and a duplicate-key collision surfaces as a duplicated
    row instead of being silently resolved by ``.iloc[0]``.
    """
    # A frame without date/side columns cannot derive anything — that is not an
    # error, it is the goalkeeper_stats case, which is served by a donor map.
    if not {"match_date", "is_home", "team"} <= set(df.columns):
        return {}

    hashed = df[is_hash_keyed(df["match_id"])].copy()
    if hashed.empty:
        return {}

    hashed["_date"] = hashed["match_date"].astype(str).str[:10]

    home = (
        hashed[hashed["is_home"]][["match_id", "_date", "team"]]
        .drop_duplicates("match_id")
        .rename(columns={"team": "_home"})
    )
    away = (
        hashed[~hashed["is_home"]][["match_id", "team"]]
        .drop_duplicates("match_id")
        .rename(columns={"team": "_away"})
    )
    pairs = home.merge(away, on="match_id", how="inner")

    # A blank date or team would mint an id like "_Napoli_" that joins to
    # nothing and looks canonical forever after. Drop rather than emit one.
    ok = (
        pairs["_date"].str.len().eq(10)
        & pairs["_home"].astype(str).str.len().gt(0)
        & pairs["_away"].astype(str).str.len().gt(0)
    )
    pairs = pairs[ok]

    canon = pairs["_date"] + "_" + pairs["_home"] + "_" + pairs["_away"]
    return dict(zip(pairs["match_id"], canon, strict=True))


def rekey_frame(
    df: pd.DataFrame, donor: dict[str, str] | None = None
) -> tuple[pd.DataFrame, dict]:
    """Return ``df`` with legacy ids replaced, plus a stats dict.

    ``donor`` supplies hash->canonical pairs for a frame that cannot derive them
    itself (no ``match_date`` column). Pairs the frame CAN derive always win, so
    a donor can only ever add coverage, never overwrite a locally-known id.
    """
    hashed_mask = is_hash_keyed(df["match_id"])
    n_hash_ids = df.loc[hashed_mask, "match_id"].nunique()
    mapping = {**(donor or {}), **build_hash_to_canonical(df)}

    out = df.copy()
    out["match_id"] = out["match_id"].map(lambda m: mapping.get(m, m))

    stats = {
        "rows": len(df),
        "hash_rows": int(hashed_mask.sum()),
        "hash_ids": int(n_hash_ids),
        "rebuilt_ids": len(mapping),
        "reconstruction": (len(mapping) / n_hash_ids) if n_hash_ids else 1.0,
        "rows_still_hashed": int(is_hash_keyed(out["match_id"]).sum()),
    }
    return out, stats


def migrate(name: str, *, apply: bool, donor: dict[str, str] | None = None) -> int:
    path = parsed_path(name)
    if not path.exists():
        log.warning("%s missing — skipping", path.name)
        return 0

    df = pd.read_parquet(path)
    out, s = rekey_frame(df, donor=donor)
    log.info(
        "%s: %d rows, %d hash ids -> rebuilt %d (%.2f%%), %d rows still hashed",
        path.name, s["rows"], s["hash_ids"], s["rebuilt_ids"],
        100 * s["reconstruction"], s["rows_still_hashed"],
    )

    if s["hash_ids"] == 0:
        log.info("%s already canonical — nothing to do", path.name)
        return 0
    if s["reconstruction"] < MIN_RECONSTRUCTION:
        log.error(
            "%s: only %.1f%% of ids could be rebuilt (floor %.0f%%) — refusing to write",
            path.name, 100 * s["reconstruction"], 100 * MIN_RECONSTRUCTION,
        )
        return 1
    if not apply:
        log.info("(dry run — pass --apply to write)")
        return 0

    tmp = path.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    log.info("Wrote %s", path)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write the files (default: dry run)")
    ap.add_argument("--only", choices=TARGETS, help="migrate a single file")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    targets = [args.only] if args.only else list(TARGETS)

    # Derive the mapping once from the file that owns the dates, then lend it.
    donor: dict[str, str] = {}
    donor_path = parsed_path("player_stats")
    if donor_path.exists():
        donor = build_hash_to_canonical(pd.read_parquet(donor_path))

    return max(migrate(t, apply=args.apply, donor=donor) for t in targets)


if __name__ == "__main__":
    sys.exit(main())
