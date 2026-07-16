"""Backfill closing-odds and Asian-handicap columns from football-data.co.uk.

Rebuilt 2026-07-16. The original was never `git add`ed and was swept after
2026-06-01 (AUGUST_RUNBOOK.md §3b). Invoked as a subprocess by
`run_full_pipeline.py:1337` on Mondays:

    python3 -m scripts.data.backfill_historical_odds --league serie_a --since YYYY-MM-DD

## What it does, and how that was established

Division of labour, recovered from the data rather than from the docs:

- `scripts/import_seriea_odds.py` (tracked) maps **16** CSV columns — the
  opening/consensus block (B365/PS/Avg/Max 1X2 + Avg,B365 O/U 2.5).
- `matches.parquet` stores **28** `odds_*` columns.
- The **12** it omits are exactly this module's output: Pinnacle and Bet365
  **closing** prices, B365 closing O/U 2.5, and the **Asian-handicap** block.

Every one of those 12 was verified to come from the football-data.co.uk CSV:
joining the 2024-25 CSV to the stored parquet on normalised (home, away, date)
reproduces all 28 stored columns **exactly** (380 matches, 0 mismatches). The
fill pattern corroborates it — 2023-24 and 2024-25 are 100% filled while
2025-26 sits at 52-68%, which is this job not having run since the pipeline
hibernated.

⚠️ **Two docs are wrong; do not "fix" this module to match them.**
- `run_full_pipeline.py`'s call-site comment says "Odds API since {since}" and
  "only if Odds API quota allows". That is stale: this reads the free CSV and
  needs **no API key and no quota**. (`DATA_CATALOG.md:179` has it right.)
- `DATA_CATALOG.md:172` attributes `odds_PS_close_*` to an "Odds API historical
  backfill". Also wrong: `PSCH/PSCD/PSCA` are columns in the CSV, verified.

## Safety

Writes `matches.parquet` (ground truth). Therefore it **only ever fills NaN
cells** and never overwrites an existing value, which is what makes the merge
idempotent — the call site relies on that ("safe to run daily"). Use
`--dry-run` to report what would change without writing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import DEFAULT_LEAGUE, LEAGUES  # noqa: E402
from config.team_names import normalize_team  # noqa: E402
from scraper.odds import download_odds  # noqa: E402

log = logging.getLogger(__name__)

PARQUET_PATH = ROOT / "data" / "parsed" / "matches.parquet"

# CSV column -> parquet column. Each pair verified against the 2024-25 season:
# all 28 stored odds_* columns reproduce exactly from the CSV (380 matches,
# 0 mismatches). These are the 12 that `import_seriea_odds.py` does NOT map.
CLOSING_AH_MAP = {
    # Pinnacle closing 1X2
    "PSCH": "odds_PS_close_H",
    "PSCD": "odds_PS_close_D",
    "PSCA": "odds_PS_close_A",
    # Bet365 closing 1X2
    "B365CH": "odds_B365_close_H",
    "B365CD": "odds_B365_close_D",
    "B365CA": "odds_B365_close_A",
    # Bet365 closing over/under 2.5
    "B365C>2.5": "odds_B365_close_over25",
    "B365C<2.5": "odds_B365_close_under25",
    # Asian handicap
    "B365AHH": "odds_B365_AH_H",
    "B365AHA": "odds_B365_AH_A",
    "AHh": "odds_AH_line",
    "AHCh": "odds_AH_close_line",
}

_KEY = ["home_team", "away_team", "_date_key"]


def _date_key(series: pd.Series, dayfirst: bool = False) -> pd.Series:
    return pd.to_datetime(
        series, format="mixed", dayfirst=dayfirst, errors="coerce"
    ).dt.date.astype(str)


def _reduce(raw: pd.DataFrame, season: str, league: str, origin: str) -> pd.DataFrame:
    """Normalise one CSV frame down to key + whichever of the 12 columns it has."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam"]).copy()
    raw["home_team"] = raw["HomeTeam"].apply(normalize_team)
    raw["away_team"] = raw["AwayTeam"].apply(normalize_team)
    raw["_date_key"] = _date_key(raw["Date"], dayfirst=True)

    present = [c for c in CLOSING_AH_MAP if c in raw.columns]
    if not present:
        log.warning("%s %s (%s): none of the closing/AH columns present", league, season, origin)
        return pd.DataFrame()

    out = raw[_KEY + present].rename(columns={c: CLOSING_AH_MAP[c] for c in present})
    return out.drop_duplicates(subset=_KEY)


def _fetch_fresh(season: str, league: str) -> pd.DataFrame:
    """Fetch the season CSV bypassing the on-disk cache. Returns empty on failure."""
    league_code, _ = LEAGUES[league]
    parts = season.split("-")
    code = parts[0][-2:] + parts[1][-2:]
    url = f"https://www.football-data.co.uk/mmz4281/{code}/{league_code}.csv"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return pd.read_csv(StringIO(resp.text))
    except Exception as e:  # noqa: BLE001 - offline/blocked is expected, degrade to cache
        log.info("%s %s: live CSV unavailable (%s); using cache only", league, season, e)
        return pd.DataFrame()


def _prepare_csv(season: str, league: str) -> pd.DataFrame:
    """Key + the 12 target columns, unioned from the CACHED and LIVE CSVs.

    Why a union rather than just one of them (measured 2026-07-16, Serie A 2025-26):

      - `scraper.odds.download_odds` is **cache-first and never refreshes** a
        season it has already stored. The cached 2025-26 file was captured
        2026-05-01 with only 346 of 380 matches, so the cache alone permanently
        misses the season's tail.
      - The LIVE file is not a superset either. football-data has since dropped
        ~143 Pinnacle closing prices: cached `PSCH` = 341/346, live = 198/380.
        Re-downloading alone would destroy real data.

    Neither dominates, so take the union. That is only safe because absence is
    absence and not revision: on the overlap, all 12 of these columns agree
    **100%** between the two files (n=198-260 per column). Note this does NOT
    hold repo-wide — `MaxH`/`AvgH` agree only ~78% because football-data
    recomputes market aggregates. Those belong to `import_seriea_odds.py`'s 16
    columns, not to this module's 12, so they are never touched here.
    """
    cached = _reduce(download_odds(season, league=league), season, league, "cache")
    fresh = _reduce(_fetch_fresh(season, league), season, league, "live")

    frames = [f for f in (cached, fresh) if not f.empty]
    if not frames:
        log.warning("No CSV data for %s %s", league, season)
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0]

    # groupby.first() skips NaN, so this takes the first non-null per column --
    # exactly the union. Order is immaterial given the 100% agreement above.
    combined = pd.concat(frames, ignore_index=True)
    out = combined.groupby(_KEY, as_index=False, sort=False).first()
    log.info("%s %s: unioned cache(%d rows) + live(%d rows) -> %d matches",
             league, season, len(cached), len(fresh), len(out))
    return out


def backfill(
    league: str = DEFAULT_LEAGUE,
    since: str | None = None,
    dry_run: bool = False,
    parquet_path: Path | None = None,
) -> int:
    """Fill missing closing/AH odds for `league` on matches on/after `since`.

    Returns the number of cells filled. Never overwrites a non-null value.
    """
    if league not in LEAGUES:
        raise SystemExit(f"unknown league {league!r}; expected one of {sorted(LEAGUES)}")

    path = parquet_path or PARQUET_PATH
    matches = pd.read_parquet(path)
    matches["_date_key"] = _date_key(matches["match_date"])

    scope = matches["league"] == league
    if since:
        scope &= matches["_date_key"] >= since
    target = matches[scope]
    if target.empty:
        log.info("No %s matches on/after %s — nothing to do", league, since or "(any date)")
        matches.drop(columns=["_date_key"], inplace=True)
        return 0

    seasons = sorted(target["season"].dropna().unique())
    log.info("Scope: %d %s matches since %s across seasons %s",
             len(target), league, since or "(start)", seasons)

    csv = pd.concat([_prepare_csv(s, league) for s in seasons], ignore_index=True)
    csv = csv[[c for c in csv.columns if c in _KEY or c.startswith("odds_")]]
    if csv.empty:
        log.warning("No CSV rows for %s %s — nothing to merge", league, seasons)
        matches.drop(columns=["_date_key"], inplace=True)
        return 0
    csv = csv.drop_duplicates(subset=_KEY)

    cols = [c for c in CLOSING_AH_MAP.values() if c in csv.columns]
    lookup = csv.set_index(_KEY)

    filled = 0
    per_col: dict[str, int] = {}
    idx = matches.index[scope]
    keys = pd.MultiIndex.from_frame(matches.loc[idx, _KEY])
    for col in cols:
        if col not in matches.columns:
            matches[col] = pd.NA
        incoming = pd.Series(lookup[col].reindex(keys).to_numpy(), index=idx)
        gap = matches.loc[idx, col].isna() & incoming.notna()
        n = int(gap.sum())
        if n:
            matches.loc[idx[gap], col] = incoming[gap]
            per_col[col] = n
            filled += n

    matches.drop(columns=["_date_key"], inplace=True)

    if per_col:
        for c, n in sorted(per_col.items()):
            log.info("  %-24s +%d", c, n)
    log.info("Filled %d cells across %d columns", filled, len(per_col))

    if dry_run:
        log.info("--dry-run: not writing %s", path)
        return filled

    if filled:
        matches.to_parquet(path, index=False)
        log.info("Wrote %s (%d rows)", path, len(matches))
    else:
        log.info("Nothing to fill; %s left untouched", path)
    return filled


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--league", default=DEFAULT_LEAGUE, choices=sorted(LEAGUES),
                    help="league key (default: %(default)s)")
    ap.add_argument("--since", default=None,
                    help="only backfill matches on/after this ISO date (YYYY-MM-DD)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing the parquet")
    args = ap.parse_args()
    backfill(league=args.league, since=args.since, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
