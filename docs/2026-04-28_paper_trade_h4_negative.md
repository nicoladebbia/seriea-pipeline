# Paper Trade — H4 Draw Boost Empirically REJECTED

**Date:** 2026-04-28
**Test:** Latest 150 Serie A matches (2025-12-27 → 2026-04-05). Compare paper-trade ROI with vs without `draw_boost=0.30` at production calibration step.
**Outcome:** **The boost destroyed ROI in real betting. Reverted to draw_boost=0.0 in production.**

## The numbers

| | No boost (baseline) | With boost=0.30 (H4) | Δ |
|---|---:|---:|---:|
| Bets placed | 62 | 108 | +46 |
| Hit rate | 51.6% | 28.7% | -23pp |
| Total staked | €1525 | €2174 | +€649 |
| Total P/L | **+€292** | **-€240** | -€532 |
| Flat ROI | **+19.1%** | **-11.0%** | **-30.1pp** |
| €1000 bankroll → | €1291 | €760 | -€531 |
| Draw bets | 14 | 91 | +77 |
| Draw hit rate | 57.1% | 24.2% | -33pp |
| Draw subset ROI | **+86.5%** | **-13.4%** | **-99.9pp** |

## What the H4 boost actually did

The boost multiplies calibrated draw probability by 1.30 before renormalization. Effect:
- Pushes more matches' draw probability above the 5% edge threshold for betting.
- **Number of draw bets went 14 → 91** — the model now bets draws on 60% of all matches.
- **Most of those new draw bets are not actually draws.** 91 bets, only 22 wins (24%). Sportsbook prices were correct on the matches the model ISN'T confident about.

## What the BASELINE accidentally revealed

Without boost, the model picks just 14 draws across 150 matches — only when it's GENUINELY confident. **Those 14 picks won 57% and delivered +86.5% ROI on the draw subset alone.** This is real edge.

The model **already correctly identifies the small number of draws it can profit on.** Boosting indiscriminately made it bet on 77 ADDITIONAL draws that aren't real — wiping out the original edge.

## Why CV said the opposite

The walkforward CV result (multi-seed: +18pp draw recall, ~0pp accuracy cost) measured **classification recall** — does the model identify draws that occur. The paper trade measured **profitability** — are the model's confident draw picks priced wrong by the market.

These metrics diverge here:
- **CV recall improvement was real:** the boost did flag more draws (4% → 22%).
- **But the lift came from low-confidence draws.** Those are exactly the ones sportsbooks price correctly.
- The model only has edge on its TOP-confidence draw picks. Inflating draw probability uniformly converts those high-confidence picks into a sea of low-confidence noise.

**This is a fundamental lesson:** classification metrics on held-out folds don't necessarily predict betting profitability. Especially for under-confident classes — improving recall by lowering the prediction threshold trades quality for quantity, which is the opposite of what betting needs.

## What's happening to the calibration mathematically

CatBoost's raw output for draws on a typical match: ~0.18-0.22 (the model has learned draws happen 27% of the time and softly votes that way).

Isotonic calibration squashes this back toward the empirical distribution: ~0.20-0.30.

**Edge calculation: model_prob - implied_market_prob.** Implied prob from ~3.5 odds (typical draw line) is 0.286.

Without boost: only matches where calibrated D is GENUINELY > 0.34 (= 0.286 + 0.05 edge threshold) get bet. That's 14 matches. Profitable.

With 30% boost: 0.20 → 0.26, 0.25 → 0.325, 0.30 → 0.39. **Suddenly almost every match has a draw probability above the betting threshold.** Half are still wrong because raw confidence wasn't there.

## Decision

**Production: keep `draw_boost=0.0`.** Reverted in `scripts/prediction/ensemble_prediction_engine.py:730`.

**The H4 result in CV was real but unactionable** — boosting calibrated draw prob at inference time is too blunt a tool for the betting decision. A smarter version would only boost when raw uncalibrated draw prob is already in the high-confidence band (e.g. raw > 0.30). That's a future experiment.

## What worked anyway

The paper trade revealed an unrelated good signal:

**Baseline 1X2 model on its own picks: +19.1% flat ROI on 62 bets across 150 latest SA matches.** The Apr 24 `real_edge_found.md` analysis already suggested CLV+ on draw subsets — this paper trade confirms the model has REAL edge on the matches it's most confident about, particularly draws (+86% ROI on 14-bet subset).

**This is a positive finding** — the existing baseline (no boost) has measurable edge on recent matches. Whether it persists at scale is the next question (answer comes from live paper-trade or CLV measurement once Odds API is back).

## Followups

1. **Don't ship `draw_boost` to production.** Keep at 0.0.
2. **Smarter draw-fix design:** boost ONLY when raw P(D) > some threshold (e.g. 0.27, the empirical base rate). Preserves the model's high-confidence picks, doesn't blanket-boost low-confidence ones.
3. **Validate baseline ROI on more data:** the +19% on 62 bets is encouraging but small sample. Run paper-trade on 300+ matches when more data is available.
4. **CLV measurement:** when Odds API returns May 1, we can measure if the bets are getting prices that move toward us (sharp money agrees). The paper-trade ROI is one signal; CLV is the second, more robust signal.

## Verdict

**STRONG NEGATIVE on H4 ship-readiness.** The CV win was a real measurement but didn't translate to betting profit. Real money would have lost €240 on €2174 staked. **Boost reverted to 0.0.**

**Confidence:** high.

**If I'm wrong:** maybe the 150 matches are an unlucky sample for the boosted variant; running on 300+ would tighten the estimate. But the mechanism (boosted = 6.5× more bets, hit rate halves) is internally consistent — sample size won't reverse this direction.

**What I'm NOT saying:** I'm not saying H4 is forever dead. The CV showed the boost has SOMETHING real (better classification recall). The issue is using it for BETTING decisions without filtering by confidence. A targeted "boost only when raw signal is strong" version is worth exploring later.
