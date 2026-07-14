"""One-shot: re-scrape historical Serie A transfers WITH window tags.

Why this exists: the historical transfers_*.parquet files were scraped without a
summer/winter window flag, so compute_net_squad_delta could not exclude January
transfers → training leak (a winter signing inflated that season's August
matches). This re-scrapes each season with window="both", which tags every row
summer/winter so the leak-free filter works.

Bounded to the importance-covered backtest set (2020-21 → 2025-26): older seasons
have no prior-season on-pitch importance data, so their deltas would be noise.

Paced (4s/request already inside scrape_transfers, plus a season gap) and
ban-aware: if a season returns zero rows, TM likely IP-banned us — stop rather
than burn the rest and risk taking down the live cron's IP.

Delete-then-scrape: scrape_transfers short-circuits when all teams are cached, so
the stale untagged file must be removed first or we get the old data back.

Run: python3 -m scripts.data.backfill_transfer_windows
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime

from scraper.transfermarkt import (
    TM_DIR,
    current_league_teams,
    scrape_transfers,
)

# ALL training seasons (2017+). The jan_* features train on the whole window, so
# tagging only 2020+ would leave 2017-2019 with jan_*=0 while 2020+ get real
# winter values → a NEW inconsistency across the training set. Tag everything the
# model trains on. (The gated net_squad_delta backtest still only USES 2020+ —
# a superset of tagged seasons doesn't hurt it.) 2026-2027 included so the live
# season's jan_* are winter-filterable too.
BACKTEST_SEASONS = [
    "2017-2018", "2018-2019", "2019-2020",
    "2020-2021", "2021-2022", "2022-2023",
    "2023-2024", "2024-2025", "2025-2026",
    "2026-2027",
]
LEAGUE = "serie_a"


def _log(msg: str) -> None:
    print(f"[{datetime.now(UTC).isoformat()}] {msg}", flush=True)


def main() -> int:
    _log(f"=== transfer-window backfill start ({len(BACKTEST_SEASONS)} seasons) ===")
    for season in BACKTEST_SEASONS:
        teams = current_league_teams(season, LEAGUE)
        if teams:
            _log(f"{season}: resolved {len(teams)} clubs")
        else:
            _log(f"{season}: could not resolve clubs — using full map")

        # move the stale untagged file aside (NOT delete) so scrape_transfers does
        # not short-circuit — but the original is restorable if the scrape fails.
        # We've hit ReadTimeouts today; a lost historical file is unrecoverable.
        path = TM_DIR / f"transfers_{season.replace('-', '_')}.parquet"
        bak = path.with_suffix(".parquet.bak")
        if path.exists():
            path.rename(bak)
            _log(f"{season}: moved stale untagged file aside → {bak.name}")

        try:
            df = scrape_transfers(
                season=season, league=LEAGUE, only_teams=teams, window="both"
            )
        except Exception as e:  # noqa: BLE001 — one bad season must not lose the rest
            _log(f"{season}: FAILED {type(e).__name__}: {e}")
            if bak.exists():          # restore the original — do not lose it
                bak.rename(path)
                _log(f"{season}: restored original from .bak")
            continue

        n = len(df)
        n_winter = int((df.get("window") == "winter").sum()) if "window" in df.columns else 0
        n_summer = int((df.get("window") == "summer").sum()) if "window" in df.columns else 0

        # ban / thin-scrape check BEFORE committing: if the new scrape is
        # suspiciously small, restore the original and stop rather than overwrite
        # a good historical file with a ban artifact.
        if n < 50:
            _log(f"{season}: SUSPICIOUSLY LOW ({n} rows) — likely IP-ban.")
            if bak.exists():
                if path.exists():
                    path.unlink()     # drop the thin scrape
                bak.rename(path)
                _log(f"{season}: restored original from .bak, stopping.")
            return 1

        _log(f"{season}: {n} rows ({n_summer} summer / {n_winter} winter) — committed")
        if bak.exists():              # good scrape committed — drop the backup
            bak.unlink()

    _log("=== transfer-window backfill done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
