"""Fill FBref-missing match stats in matches.parquet from cached Sofascore data.

Rebuilt 2026-07-16. The original was never `git add`ed and was swept after
2026-06-01 (AUGUST_RUNBOOK.md §3b). Invoked as a subprocess by
`scripts/pipeline/refresh_weekly_data.py:219`:

    python3 -m scripts.data.fallback_sofascore_to_fbref --season 2025-2026

It exists because FBref scraping fails: every logged weekly run (2026-04-27 →
06-01) shows `✗ fbref_fixtures` and `✗ fbref_htmls`. This is the Plan-B that
keeps match stats flowing. Its inputs are **cached parquets**, so the Sofascore
403 does not block it.

## REIMPLEMENTED, not recovered — read this before trusting it

The stored output is **not** a spec. It covers 94 of the 120 matches in its
window (2026-02-27 → 05-24) with 26 holes *inside* the window, and 260 matches
before it untouched. That pattern is **run history** — a function of what was
cached at each weekly run — and no rule written today reproduces it. Do not try.

What IS recoverable is the *transform*, because the 94 filled rows are a pure
function of surviving inputs. Every mapping below was **derived by searching all
candidate source columns against those 94 rows and keeping only exact
reproductions** — not guessed:

- 46 stat columns: identity-mapped from `match_team_stats.parquet` (period=ALL),
  100% exact on n=94.
- `passing_accuracy` = round(accurate_passes / total_passes * 100, 1) — exact.
- `cards` = yellow_cards + red_cards — exact.
- `ht_score` = count of `match_incidents` goals with minute <= 45 — 94/94 exact.
- `venue`, `manager`, `captain`: the original wrote **empty strings**, not data.
  Reproduced faithfully; Sofascore's team-stats path has no such fields.

## Deliberate deviations from the original (both documented, both intentional)

1. **`match_id` is left alone.** The original *overwrote* it with Sofascore's
   numeric id (e.g. "13981687"), so `matches.parquet` now holds two incompatible
   id formats in one column (94 numeric vs 286 canonical
   `{date}_{home}_{away}`), which breaks joins. Reproducing that would knowingly
   propagate a bug to 286 more rows. Rows written here keep their canonical id.
   Repairing the existing 94 is a separate migration — it must also fix
   `lineups.parquet` (4,577 rows / 99 numeric ids; 86 of the 94 appear there),
   bridged via `match_id_mapping.parquet:sofascore_id`. Tracked in §3b.
2. **`formation` is not filled.** `lineups.parquet` reproduces the stored value
   only 91.5% of the time, which is below the bar used for every other column
   here. Filling it would mean writing values I cannot verify into ground truth.

## Safety

Writes `matches.parquet` (ground truth): fills **NaN cells only**, never
overwrites, so it is idempotent and safe to re-run. `--dry-run` reports without
writing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.team_names import normalize_team  # noqa: E402

log = logging.getLogger(__name__)

MATCHES_PATH = ROOT / "data" / "parsed" / "matches.parquet"
TEAM_STATS_PATH = ROOT / "data" / "external" / "sofascore" / "match_team_stats.parquet"
INCIDENTS_PATH = ROOT / "data" / "external" / "sofascore" / "match_incidents.parquet"

DATA_SOURCE = "sofascore"

# sofascore match_team_stats column -> matches.parquet suffix.
# Every pair below reproduces the 94 stored rows EXACTLY (searched, not assumed).
# The odd-looking ones are faithful to the oracle: `shots_on_target` is stored raw
# (not as a percentage), and `*_total` variants intentionally repeat their source.
IDENTITY_MAP = {
    "total_shots": ["shots_total", "shots_on_target_total"],
    "shots_on_target": ["shots_on_target", "shots_on_target_count"],
    "possession": ["possession"],
    "fouls": ["fouls"],
    "corners": ["corners"],
    "gk_saves": ["saves", "saves_count", "saves_total"],
    "accurate_crosses": ["crosses"],
    "touches_in_opp_box": ["touches"],
    "total_tackles": ["tackles"],
    "interceptions": ["interceptions"],
    "aerial_duels_pct": ["aerials_won"],
    "clearances": ["clearances"],
    "offsides": ["offsides"],
    "goal_kicks": ["goal_kicks"],
    "throw_ins": ["throw_ins"],
    "accurate_long_balls": ["long_balls"],
    "accurate_passes": ["passing_accuracy_count"],
    "total_passes": ["passing_accuracy_total"],
    "xg": ["xg"],
}

# Written as empty strings by the original: Sofascore's team-stats path carries
# no venue/manager/captain. Kept for byte-compatibility with the stored 94.
BLANK_COLS = ["venue", "home_manager", "away_manager", "home_captain", "away_captain"]


def _date_key(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date.astype(str)


def _wide_team_stats(season: str) -> pd.DataFrame:
    """Pivot period=ALL team stats into one row per match with home_/away_ columns."""
    ts = pd.read_parquet(TEAM_STATS_PATH)
    t = ts[(ts["season"] == season) & (ts["period"] == "ALL")].copy()
    if t.empty:
        return pd.DataFrame()
    t["_date_key"] = _date_key(t["date"])
    t["home_team"] = t["home_team"].apply(normalize_team)
    t["away_team"] = t["away_team"].apply(normalize_team)

    key = ["home_team", "away_team", "_date_key"]
    frames = []
    for side, is_home in (("home", True), ("away", False)):
        s = t[t["is_home"] == is_home].drop_duplicates(subset=key).set_index(key)
        cols = {}
        for src, targets in IDENTITY_MAP.items():
            if src not in s.columns:
                continue
            for tgt in targets:
                cols[f"{side}_{tgt}"] = s[src]
        frame = pd.DataFrame(cols)
        # passing_accuracy is derived, and rounded to 1dp -- verified on the oracle.
        if "accurate_passes" in s.columns and "total_passes" in s.columns:
            frame[f"{side}_passing_accuracy"] = (
                s["accurate_passes"] / s["total_passes"] * 100
            ).round(1)
        frames.append(frame)
    wide = frames[0].join(frames[1], how="outer")
    wide["_match_id"] = t.drop_duplicates(subset=key).set_index(key)["match_id"]
    return wide.reset_index()


def _ht_scores(match_ids: pd.Series) -> pd.DataFrame:
    """Half-time score = goals scored at minute <= 45. Verified 94/94."""
    inc = pd.read_parquet(INCIDENTS_PATH)
    g = inc[(inc["incident_type"] == "goal") & (inc["minute"] <= 45)]
    g = g[g["match_id"].isin(set(match_ids))]
    counts = g.groupby(["match_id", "is_home"]).size().unstack(fill_value=0)
    out = pd.DataFrame(index=pd.Index(match_ids.unique(), name="match_id"))
    out["home_ht_score"] = counts.get(True, 0)
    out["away_ht_score"] = counts.get(False, 0)
    return out.fillna(0.0).astype(float)


def fill_from_sofascore(
    season: str,
    league: str = "serie_a",
    dry_run: bool = False,
    parquet_path: Path | None = None,
) -> int:
    """Fill NaN match-stat cells for `season` from cached Sofascore parquets.

    Returns the number of cells filled. Never overwrites an existing value and
    never touches match_id.
    """
    path = parquet_path or MATCHES_PATH
    matches = pd.read_parquet(path)
    matches["_date_key"] = _date_key(matches["match_date"])

    scope = (matches["league"] == league) & (matches["season"] == season)
    if not scope.any():
        log.warning("No %s matches for season %s", league, season)
        matches.drop(columns=["_date_key"], inplace=True)
        return 0

    wide = _wide_team_stats(season)
    if wide.empty:
        log.warning("No cached Sofascore team stats for %s — nothing to fall back on", season)
        matches.drop(columns=["_date_key"], inplace=True)
        return 0

    key = ["home_team", "away_team", "_date_key"]
    lookup = wide.drop_duplicates(subset=key).set_index(key)
    idx = matches.index[scope]
    keys = pd.MultiIndex.from_frame(matches.loc[idx, key])

    ht = _ht_scores(lookup["_match_id"])
    ht_by_key = lookup[["_match_id"]].join(ht, on="_match_id")

    stat_cols = [c for c in lookup.columns if c != "_match_id"]
    filled = 0
    per_col: dict[str, int] = {}

    for col in stat_cols + ["home_ht_score", "away_ht_score"]:
        if col not in matches.columns:
            continue
        source = lookup if col in lookup.columns else ht_by_key
        incoming = pd.Series(source[col].reindex(keys).to_numpy(), index=idx)
        gap = matches.loc[idx, col].isna() & incoming.notna()
        n = int(gap.sum())
        if n:
            matches.loc[idx[gap], col] = incoming[gap]
            per_col[col] = n
            filled += n

    # cards = yellow + red (verified exact on the oracle)
    for side in ("home", "away"):
        tgt, y, r = f"{side}_cards", f"{side}_yellow_cards", f"{side}_red_cards"
        if not {tgt, y, r} <= set(matches.columns):
            continue
        calc = pd.to_numeric(matches.loc[idx, y], errors="coerce") + pd.to_numeric(
            matches.loc[idx, r], errors="coerce"
        )
        gap = matches.loc[idx, tgt].isna() & calc.notna()
        n = int(gap.sum())
        if n:
            matches.loc[idx[gap], tgt] = calc[gap]
            per_col[tgt] = n
            filled += n

    # Rows that received any stat get the provenance marker and the original's
    # empty-string placeholders. match_id is deliberately NOT touched.
    touched = matches.loc[idx, [c for c in stat_cols if c in matches.columns]].notna().any(axis=1)
    newly = idx[touched & matches.loc[idx, "data_source"].isna()]
    for col in BLANK_COLS:
        if col in matches.columns:
            blank = matches.loc[newly, col].isna()
            if blank.any():
                matches.loc[newly[blank], col] = ""
                filled += int(blank.sum())
    matches.loc[newly, "data_source"] = DATA_SOURCE

    matches.drop(columns=["_date_key"], inplace=True)

    for c, n in sorted(per_col.items()):
        log.info("  %-30s +%d", c, n)
    log.info("Filled %d cells across %d columns; marked %d rows data_source=%s",
             filled, len(per_col), len(newly), DATA_SOURCE)

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
    ap = argparse.ArgumentParser(description="Fill FBref-missing match stats from cached Sofascore data")
    ap.add_argument("--season", required=True, help="season to fill, e.g. 2025-2026")
    ap.add_argument("--league", default="serie_a", help="league key (default: %(default)s)")
    ap.add_argument("--dry-run", action="store_true", help="report without writing")
    args = ap.parse_args()
    fill_from_sofascore(season=args.season, league=args.league, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
