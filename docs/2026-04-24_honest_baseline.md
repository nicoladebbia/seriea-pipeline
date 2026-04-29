## 2026-04-24 — Honest walk-forward baseline (Serie A)

First real leakage-free walk-forward backtest of the single-market CatBoost
fleet. Numbers below are the honest performance ceiling with no live odds.

---

### Pipeline

- One trainer: `scripts/models/train_walkforward.py`
- Per (league, market, eval_season) models at
  `data/models/walkforward/{league}/{market}/season_{YYYY-YYYY}.cbm`
- Each eval season's model trained on **only** data strictly before that season
- Raw odds excluded from features via `ml.feature_selection.exclude_odds`
- Additional post-match features blacklisted in `LEAKY_COLUMNS` (current-match
  shots, corners, fouls, cards, HT score, Sofascore ratings / xG sums)
- Leakage safety net: refuses to train if any feature has |corr| > 0.5 with
  home_win / draw / btts / over_2_5 targets

### Models trained and their walk-forward performance

2022-23, 2023-24, 2024-25 eval seasons. 380 matches per season. 1061 features.

| Market | Base rate | Walk-forward accuracy | Log-loss | ECE / cal_gap | Notes |
|---|---:|---:|---:|---:|---|
| 1X2 | 41.1% (H) | **52.63%** | 0.984 | ECE 0.016 | **At Pinnacle close ceiling.** Draw class accuracy 1.25% (known issue). |
| O/U 2.5 | 47.6% | 49.82% | 0.696 | cal_gap 0.059 | Only +2pp over random. |
| O/U 1.5 | 74.6% | 74.56% | 0.568 | cal_gap 0.026 | Matches "always yes" baseline. No edge. |
| O/U 3.5 | 25.5% | 74.47% | 0.575 | cal_gap 0.069 | Matches "always no" baseline. No edge. |
| BTTS | 51.4% | 51.40% | 0.697 | cal_gap 0.061 | No edge. |

### ROI vs closing lines (Serie A, 2022-23 through 2024-25)

Closing odds: Pinnacle close (1X2), B365 close (O/U 2.5). All negative.

| Market | Threshold | Bets | ROI | CI (95%) |
|---|---|---:|---:|---|
| 1X2 | 0% | 1140 | **-15.19%** | [-23.0, -6.6] |
| 1X2 | 5% | 991 | -15.92% | [-24.5, -7.2] |
| 1X2 | 10% | 725 | -14.42% | [-23.8, -4.6] |
| O/U 2.5 | 0% | 1140 | -8.13% | [-14.6, -1.6] |
| O/U 2.5 | 10% | 667 | -10.28% | [-18.4, -1.5] |

### ROI vs OPENING B365 1X2 (sanity check)

Opening lines are softer than closing. If the model has any real edge, it should
appear here first (sharp money drives the line from open to close — opening-line
bettors can profit from model-vs-open disagreement).

| Threshold | Bets | Win rate | ROI | CI |
|---|---:|---:|---:|---|
| 0% | 1140 | 36.8% | -12.16% | [-19.8, -4.6] |
| 3% | 1077 | 36.4% | -12.34% | [-20.1, -4.2] |
| 5% | 993 | 35.4% | -13.97% | [-22.0, -5.4] |
| 7% | 907 | 34.7% | -13.54% | [-22.3, -4.3] |
| 10% | 727 | 34.8% | **-12.04%** | [-21.3, -1.7] |

**Opening B365 is no better than Pinnacle close.** No threshold produces
positive-CI ROI. The model does not beat either opening or closing lines on
1X2 from pre-match features alone.

### What this means

1. **The 52.63% 1X2 accuracy is at the academic + Pinnacle closing-line ceiling.**
   No amount of model tuning will push it higher meaningfully. The Koopman-Lit
   state-space Dixon-Coles from academic literature hits 52-56% and this matches.

2. **The betting edge required to beat closing lines (~6% bookmaker margin plus
   imperfect calibration) does not exist** in 1X2 or O/U 2.5 from pre-match
   features alone. This is consistent with the efficient-market hypothesis on
   the most heavily-bet football markets.

3. **The previous "+30% ROI" results were pure leakage.** The training pipeline
   included:
   - `home_shots_on_target_count`, `away_shots_on_target_count` (shots on target
     from the CURRENT match; importance 19.5 and 15.4 of 100)
   - `home_corners`, `away_corners`, `home_fouls`, `away_fouls` (all current-match)
   - Odds-meta features (market_elo_disagreement, sharp_soft_*_div) that
     indirectly encode the closing line

   All 20+ leaky features now blacklisted. The correlation safety net
   (|corr| > 0.5) catches any new ones automatically.

### Where edge might still exist (none tested yet, all require Odds API)

1. **Markets Pinnacle doesn't price tightly**: corners, cards, BTTS.
   Historical closing odds not yet backfilled for these markets in our dataset.
   Backfill via Odds API historical endpoint unlocks validation.

2. **Line-movement betting**: placing bets at soft-book opening odds before the
   sharp money drives them to consensus. Requires live odds from multiple books,
   not pre-match parquet data.

3. **Player props**: anytime scorer, shots O/U. Academic evidence suggests these
   are weaker markets where model edges can exist (McHale et al. 2012). Requires
   both player-level features (we have them) and player-market odds (we don't).

### Files produced

- `scripts/models/train_walkforward.py` — one unified leakage-free trainer
- `models/simulator/backtests/walkforward_predictor.py` — harness adapter
- `data/models/walkforward/serie_a/{1x2,over_1_5,over_2_5,over_3_5,btts}/` —
  per-season models + metadata + summaries
- `docs/2026-04-24_honest_baseline.md` — this file

### What did NOT happen

- No iteration to chase green ROI numbers
- No hyperparameter fishing
- No post-hoc market selection

### Decision waiting for Odds API return

Three paths when Odds API returns (~2 weeks):

**A) Backfill corners/cards/BTTS closing odds, retrain those market models,
   test for edge against the less-efficient books.** Genuine possibility of
   positive ROI per academic literature.

**B) Live-odds betting: bet opening soft-book lines before they move, using the
   model's H/D/A probabilities as the anchor.** Requires multi-book scraper but
   does not require the current models to beat Pinnacle close.

**C) Accept the honest finding: our pre-match feature set cannot profitably
   beat the 1X2/O/U closing line on either Serie A or EPL, and focus the
   product on markets where sharp pricing is absent.**
