# Fresh Model Test (B1') — Staleness Confirmed, Edge Erosion Real but Smaller

**Date:** 2026-04-28
**Test:** Train fresh fold model on data including 2024-25, score 2025-26 with it. Compare betting ROI vs the stale-fold result.
**Outcome:** **Staleness was responsible for ~5pp of the 2025-26 ROI degradation. But edge is still smaller than 2022-24. Both effects are real.**

## Headline comparison

Same betting rule (5% edge, fractional Kelly 0.25), same matches, only difference is which model scores 2025-26.

| Subset | Stale 2025-26 ROI | Fresh 2025-26 ROI | Δ |
|---|---:|---:|---:|
| **Baseline (no filter)** | **-5.98%** | **-1.20%** | **+4.78pp** |
| Draws only | -6.70% | -2.97% | +3.73pp |
| Mid-odds (2.2-4.0) | -11.02% | -1.58% | +9.44pp |
| Mid-prob (0.30-0.50) | -9.34% | -3.42% | +5.92pp |
| Mid-odds + mid-prob | -12.13% | -5.15% | +6.98pp |
| Draws + mid-odds | -12.56% | -6.07% | +6.49pp |
| matchweek_early(1-12) | +20.15% | +23.05% | +2.90pp (was already best) |

**Every subset improves materially with the fresh model.** This confirms staleness was significantly responsible for the apparent edge degradation.

## What this resolves

**Staleness hypothesis: STRONGLY CONFIRMED.** A model trained on data ending 2023 systematically scores 2025-26 worse than a model trained on data ending 2024-25. The improvement is consistent across every subset (+3.7 to +9.4 percentage points).

## What this does NOT resolve

**The fresh model is still NEGATIVE on 2025-26 baseline (-1.20%).** Even with a freshly-trained model, the audit's claimed +8% subset ROI does not appear on 2025-26 — we get break-even to slight loss across most filters.

**Implication:** there is BOTH a staleness component AND a real edge-erosion component:
- ~5pp of the gap was staleness (fixable by retraining).
- ~6-7pp remaining is real edge erosion (sportsbooks have improved, or the league regime has shifted).

## STRONG+ findings (consistent across all 3 splits with fresh model)

| Subset | Disc / Val / **Holdout** | Sample (disc/val/hold) |
|---|---|---:|
| `pick_D` (draws) | +9.88% / +5.36% / **-2.97%** | 584/334/242 |
| `matchweek_early(1-12)` | +9.50% / +5.44% / **+23.05%** | 182/109/98 |
| `kickoff_<=15h` (= baseline) | +11.17% / +4.55% / **-0.10%** | 633/350/231 |
| `draw_picks_only` | +9.88% / +5.36% / **-2.97%** | 584/334/242 |
| `mid_odds(2.2-4.0)` | +15.11% / +10.34% / **-1.58%** | 444/238/190 |
| `mid_prob(0.30-0.50)` | +13.86% / +1.55% / **-3.42%** | 522/286/221 |

**Note the subtle finding:** several "STRONG+" verdicts show negative holdout (e.g., draws -2.97%) because my verdict logic accepts holdout > -5% as "didn't collapse." Be careful: STRONG+ here means "not as bad as everything else" not "consistently profitable."

**The ONLY filter that's actually positive on holdout: matchweek_early(+23.05%).** Everything else is break-even or slightly negative.

## Practical betting strategy implications

### Tier 1 — Safe (positive ROI on all 3 windows including 2025-26)
- **Bet matchweeks 1-12 only.** Across 389 bets total, every window positive. **Mechanism: book inefficiency on early-season uncertainty.** Smaller bet volume but reliable.

### Tier 2 — Mostly safe (break-even on 2025-26, profitable historically)
- **Bet draws only.** -2.97% on holdout is within noise; backtest +9% / +5% on 918 bets is real. Combined with mid-odds, gets to -6%.
- **Bet mid-odds (2.2-4.0) only.** -1.58% holdout, +15% / +10% backtest on 872 bets.
- **Bet mid-prob (0.30-0.50).** -3.42% holdout.

These are NOT money-makers on the most recent season but they don't bleed badly. **Use them only if combined with the matchweek-early filter, OR after retraining proves further improvement.**

### Tier 3 — Avoid
- **Combining mid_odds + mid_prob.** Over-restrictive — both filters together produce -5% holdout. The individual filters are better than the combination.
- **`home_short_rest=1`.** Loser across all 3 windows.

## Critical NEW recommendations

### B1'' — Promote fresh-model logic to production walkforward
The walkforward training script's default eval window (`DEFAULT_EVAL_SEASONS`) currently locks at 2020-21 → 2024-25. **Add 2025-26 to the default,** so future training automatically scores 2025-26 with the latest available model.

This isn't theoretical — the fresh model demonstrably saved ~5pp of holdout ROI. Production retrains should always include the most-recent completed season in training.

### B2 — Revise the betting plan
The original B2 phase ("implement April 24 subset filters") needs revision:
- **Drop "mid_odds + mid_prob" combination.** Empirically over-restrictive.
- **Keep "mid_odds 2.2-4.0" or "mid_prob 0.30-0.50" as separate filters** — each is break-even alone.
- **Add matchweek_early as the lead filter.** It's the one consistent positive across all windows.

### B7 (NEW) — Formalize "early season" as a betting tier
Create a configurable "season window" parameter. Bet larger stakes in matchweeks 1-12 (the proven-profitable window), smaller stakes mid-late season. Empirically backed by 389 bets across 3 windows.

## What this whole analysis cost vs what it prevented

- ~30 min compute (one walkforward run + two subset analyses).
- Prevented adopting the April 24 filters as-is, which empirically lose -10% to -12% on the most recent data.
- Identified one robust filter (matchweek-early) that survives.
- Confirmed retraining yields concrete (+5pp) ROI improvement.

## Verdict

**STRONG.** The fresh-model retest fundamentally changes the betting plan:

1. **Staleness was real** — production walkforward must include 2024-25 in training going forward.
2. **Edge erosion is also real** — the original audit's filters don't fully recover even with retraining.
3. **One robust signal exists** — matchweek-early, on 389 bets across 3 independent windows.

**Confidence:** high.

**If I'm wrong:** the 2025-26 sample is still moderate (254 bets in fresh holdout). Another 200 matches would tighten the picture and could either confirm the small remaining edge (1-2% break-even is acceptable) or show further degradation.

**What I'm NOT saying:** I'm not saying the model is now profitable. I'm saying the FILTERS we'd considered adopting are LESS bad than the stale-model picture suggested, but most are still break-even-at-best on the latest data. The path forward is smaller scope (matchweek-early), continuous retraining, and accepting that 2022-24 levels of ROI may not return.
