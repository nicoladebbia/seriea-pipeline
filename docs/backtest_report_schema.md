# Backtest report schema

Every Phase 3b backtest run (`scripts/diagnostics/run_backtest.py`) writes a
JSON file following this exact schema. Future predictors (simulator variants,
new CatBoost tunings, meta-learners) write into the same schema so diffing
two reports answers "which is better per market" with no extra code.

## Top-level

```json
{
  "metadata": { ... },
  "markets": {
    "<market_label>": {
      "<threshold_key>": {
        "<stake_policy>": {
          "<odds_source>": { ...ROIStats }
        }
      }
    }
  }
}
```

## `metadata`

```json
{
  "predictor_name": "prod_baseline",
  "predictor_version": "2026-04-21",
  "seasons": ["2021-2022", "2022-2023", "2023-2024", "2024-2025"],
  "markets": ["O/U 2.5", "1X2"],
  "edge_thresholds_pct": [0.0, 3.0, 5.0, 7.0, 10.0],
  "stake_policies": ["flat", "kelly"],
  "odds_sources_observed": ["pinnacle_close", "b365_close", "market_avg", "ALL"],
  "n_matches_scored": 1523,
  "generated_at": "2026-04-22T03:56:22.140296+00:00",
  "seed_salt": 0,
  "walk_forward_mode": "season_boundary"
}
```

## Nested keys

### `market_label`

One per market registered in `harness.SERIE_A_BINARY_MARKETS` /
`SERIE_A_MULTICLASS_MARKETS`. Examples:

- `"O/U 2.5"`, `"O/U 1.5"`, `"O/U 3.5"`, `"BTTS"`, `"1X2"`

### `threshold_key`

Format: `"thresh_Npct"` where N is the edge threshold percentage.
Examples: `"thresh_0pct"`, `"thresh_3pct"`, `"thresh_5pct"`, `"thresh_7pct"`, `"thresh_10pct"`.

### `stake_policy`

- `"flat"` — €10 per bet
- `"kelly"` — quarter-Kelly with €2 floor / €50 ceiling, €1000 bankroll (not growing)
- `"no_stake"` — shadow-mode log only (rarely used in reports)

### `odds_source`

The source label recorded at bet placement. First tuple in each market's
`odds_chain` is labeled `pinnacle_close`, second is `b365_close` (by
convention of chain priority), third `market_avg`, fourth `b365`, fifth
`opening`. A special aggregate `"ALL"` is always included — it sums across
all sources.

## `ROIStats`

```json
{
  "n_bets": 312,
  "total_stake_eur": 3120.00,
  "total_profit_eur": -306.84,
  "roi_pct_point": -9.83,
  "roi_pct_ci_lower": -13.20,
  "roi_pct_ci_upper": -6.55,
  "sharpe": -1.420,
  "max_drawdown_eur": 410.50,
  "max_drawdown_pct_of_stake": 13.16,
  "longest_losing_streak": 9,
  "max_single_bet_edge_share": 0.023
}
```

Fields:

- `n_bets` — how many bets were placed at this (market, threshold, stake, source).
- `total_stake_eur` — sum of stakes in EUR.
- `total_profit_eur` — realized profit after settlement.
- `roi_pct_point` — `100 * total_profit / total_stake`.
- `roi_pct_ci_lower`, `roi_pct_ci_upper` — 95% bootstrap CI over 1000 resamples
  of the bet vector. Null if n < 5 or total_stake ≤ 0.
- `sharpe` — `mean(profit/stake) / std(profit/stake) × sqrt(n)`. Null if n < 2.
- `max_drawdown_eur` — worst peak-to-trough dip in cumulative profit curve.
- `max_drawdown_pct_of_stake` — drawdown expressed as fraction of total stake.
- `longest_losing_streak` — consecutive losing bets.
- `max_single_bet_edge_share` — fraction of total *positive* profit coming from
  the single biggest winning bet. Used by Phase 4 gate to reject "one lucky
  bet" promotions (threshold: 0.30).

## Promotion gate (Phase 4)

The `should_promote(market)` function in §12.2 of the v3 plan reduces to
one JSON query:

```python
sim_stats = report["markets"][market][optimal_threshold]["flat"]["pinnacle_close"]
baseline_stats = baseline_report["markets"][market][optimal_threshold]["flat"]["pinnacle_close"]

promote = (
    sim_stats["n_bets"] >= 50
    and sim_stats["roi_pct_ci_lower"] > 0
    and sim_stats["roi_pct_point"] >= baseline_stats["roi_pct_point"]
    and sim_stats["max_single_bet_edge_share"] < 0.30
)
```

## Stability

Any schema change is a breaking change. Add fields; don't rename or remove.
Version the schema via a top-level `schema_version` field when the first
breaking change lands. As of 2026-04-21: schema_version = 1 (implicit).
