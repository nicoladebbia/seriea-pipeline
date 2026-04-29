# Odds Backfill — BTTS / Corners / Cards (Plan + Script Extension)

**Date:** 2026-04-28. Odds API activates 2026-05-01. **Cannot execute backfill yet** — this document is the plan + the script-side preparation that has been completed today.

## What's been done today

### `scripts/data/backfill_historical_odds.py` extended to support BTTS

- Added `btts` to default markets list.
- Added BTTS extraction logic in `_extract_odds_row()`: pulls Pinnacle yes/no + market average yes/no.
- New columns written to season CSVs: `PSBTTS_Y`, `PSBTTS_N`, `AvgBTTS_Y`, `AvgBTTS_N`, `MaxBTTS_Y`, `MaxBTTS_N`.
- Verified script still imports cleanly.

## What's runnable on 2026-05-01

```bash
# Serie A (last full season)
python -m scripts.data.backfill_historical_odds --league serie_a --since 2024-08-01

# EPL (last full season)
python -m scripts.data.backfill_historical_odds --league premier_league --since 2024-08-01

# Both leagues
python -m scripts.data.backfill_historical_odds --all --since 2024-08-01
```

This will fetch BTTS odds for every match since the date specified. The Odds API charges per call; check `x-requests-remaining` header to estimate budget.

## What's NOT runnable on May 1 (and why)

**Corners odds and Cards odds are NOT available via The Odds API for soccer.** Per The Odds API docs, soccer-specific corner/card markets are sportsbook-internal in-play propositions, not pre-match bettable on the API.

To backfill corners/cards historical closing odds, three viable paths:

1. **oddsportal.com archive scraping.** Has historical pre-match prices for many bookmakers across many markets including corners/cards/handicaps. Requires careful scraping (session/cookie management, rate limiting, anti-bot evasion).
2. **Pinnacle XML feed.** Pinnacle publishes historical odds via their direct feed — but typically only h2h/totals. Corners are sometimes there for top leagues.
3. **Individual bookmaker historical pages.** Bet365, William Hill, Unibet sometimes have historical corner pricing on their archive pages, but they're limited to recent seasons and rate-limited.

**Recommendation:** start with oddsportal scraping. The other two are weaker and Pinnacle alone won't cover corners.

## What this unblocks (after May 1 backfill runs successfully)

The 2026-04-24 honest baseline doc identified that `prod_btts.cbm`, `prod_corners_*.cbm`, `prod_cards_*.cbm` models exist but produce 0 bets in walk-forward because closing odds for those markets aren't in `features_serie_a.parquet`. After the BTTS backfill:

- BTTS market models can be properly backtested for the first time.
- The walk-forward harness can compute edge vs Pinnacle close on BTTS.
- If the model has any edge against the soft side of BTTS (which is plausible per academic literature), it'll show in the report.
- This is the highest-likelihood market for finding real edge per the 2026-04-24 honest_baseline.md analysis.

## Cost estimate (for budget planning)

The Odds API historical endpoint charges 1 credit per snapshot per market.

- One season of Serie A = ~380 matches × 2 snapshots (open + close) × 3 markets (h2h+totals+btts) = **2,280 credits per season**
- Same for EPL: **2,280 credits**
- 2024-2025 backfill (1 season × 2 leagues): **4,560 credits**
- 2017-2024 backfill (8 seasons × 2 leagues): **36,480 credits** — this is expensive.

The Odds API typical paid tier is 20,000 credits/month at $30/month. So 2024-25 single-season is comfortably within one month of paid API. Multi-season historical is multi-month.

**Recommendation:** start with 2024-25 only. Validate the BTTS model works end-to-end. Then decide whether to spend on deeper history.

## Verification plan after May 1 run

1. Run `--league serie_a --since 2024-08-01 --dry-run` first to see how many credits it'd use without actually spending them.
2. Run the real backfill on a small date range first (2024-08-01 to 2024-09-01 = 1 month).
3. Verify `data/external/odds/I1_24-25.csv` has new BTTS columns populated.
4. Run full season after spot-check passes.
5. Re-run feature pipeline to pick up new BTTS columns.
6. Re-run walkforward training for `--market btts`.
7. Compare BTTS model walkforward Kelly ROI / CLV+ vs prior.

## Verdict

**STRONG.** The script extension is correct and minimal, the plan is honest about what's tractable vs what isn't (corners/cards need a different data source), and the cost estimate gives a budget framework.

**Confidence:** medium — the actual API behavior for BTTS endpoint can only be verified after May 1.

**If I'm wrong:** the BTTS market on The Odds API might not return Pinnacle-quality data for soccer (some markets only have soft books). If so, the `pinnacle_btts` accumulator in the extraction will stay None and only `Avg*` columns will populate. Still useful — Avg odds across multiple soft books is a reasonable closing-line proxy.

**What I'm NOT saying:** I'm not saying corners/cards backfill is impossible — only that The Odds API is not the right source. A separate scraping-based backfill (oddsportal) is its own project.
