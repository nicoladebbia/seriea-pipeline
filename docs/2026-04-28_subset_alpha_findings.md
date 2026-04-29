# Deep Subset Analysis — Edge Degradation Detected

**Date:** 2026-04-28
**Method:** Score 1,478 SA matches across 4 seasons (2022-23 → 2025-26) using walkforward fold models, generate paper-trade bets at 5% edge threshold, slice ROI by 17 candidate dimensions. Splits: discovery (2022-23, 2023-24), validation (2024-25), holdout (2025-26). Adoption rule: consistent direction across all 3 splits.

## Headline finding

**The model's betting ROI is degrading season-by-season.** The April 24 audit's profitable subsets (draws, mid-odds, mid-prob) DO NOT generalize to 2025-26.

| Subset | Discovery (22-24) | Validation (24-25) | **Holdout (25-26)** |
|---|---:|---:|---:|
| Baseline (no filter) | +11.17% | +4.55% | **-5.98%** |
| Draw picks only | +9.88% | +5.36% | **-6.70%** |
| Mid-odds (2.2-4.0) | +15.11% | +10.34% | **-11.02%** |
| Mid-prob (0.30-0.50) | +13.86% | +1.55% | **-9.34%** |
| Mid-odds + mid-prob | +15.95% | +10.72% | **-12.13%** |
| Draws + mid-odds | +12.64% | +11.15% | **-12.56%** |

**Every subset that was profitable in 2022-2024 is unprofitable in 2025-26.** This is the first empirical evidence that the betting system's edge has eroded structurally.

## Hypotheses for the edge degradation

1. **Sportsbooks have improved their pricing.** Most likely. Books continuously refine models; the 2022-24 inefficiencies they had may simply not exist anymore.
2. **The training data drifted.** Model trained on 2017-2023 may not understand 2025-26 football regime (rule changes, tactical evolution, COVID-cohort player profiles aging out).
3. **The 2024-25 model used to score 2025-26 is too stale.** The walkforward fold model trained on data ending 2023 misses 2024 information when scoring 2025-26 — could use a freshly-retrained model.
4. **2025-26 is a small sample (271 bets).** Variance could be making things look worse than they are. But the magnitude (-6% to -12%) is too large to be pure variance with 200+ bets.
5. **Regime change in this specific season.** Promoted teams behaving differently, dominant teams emerging, etc. — could revert in 2026-27.

## STRONG+ finding (the only positive that survived)

**`matchweek_early(1-12)`: +9.5% / +5.4% / +20.2% on n=182/109/103.**

Early-season matches (matchweeks 1-12) deliver positive ROI across ALL THREE windows including 2025-26. This is the only subset that didn't degrade.

**Why it might be real:**
- Early season has more uncertainty (new rosters, fitness, tactics) — sportsbooks can't price as confidently.
- Promoted teams are mispriced (markets default to "they'll be relegated" but variance is high).
- Smaller sample of "form data" available — model trained on long histories has an edge over books still calibrating.

**Caveat:** ~394 bets total is still moderate sample. Direction is consistent but the holdout +20% could be variance.

## Other notable but weaker findings

| Subset | Disc / Val / Hold | Notes |
|---|---|---|
| `is_derby` | -4% / +56% / -11% | High variance, no consistent direction |
| `is_midweek` | +24% / -11% / +3% | Very mixed |
| `away_manager_changed` | +71% / +31% / +10% | All positive but declining; n only 30 total. Possibly worth a future deep look. |
| `home_manager_changed` | +1% / +109% / +85% | All positive but n=31 — could be coincidence |
| `home_in_cl_zone` | +1% / +35% / +8% | All positive — Champions League zone teams' home games |
| `away_short_rest` | +4% / +8% / +20% | All positive, all small n (29/11/17 = 57). Suggestive. |

## What this means for the betting plan

**The B2 phase of the betting improvement plan (implement April 24 subset filters) has its premise EMPIRICALLY UNDERMINED.** The filters that were profitable 2022-24 are NOT profitable 2025-26.

Three possible responses:

### A. Accept the degradation as real → don't adopt April 24 filters
Implementing `mid_odds + mid_prob` filters now would lock in a 2022-24 strategy that LOSES money in 2025-26. **This is the conservative response.**

### B. Suspect model staleness → retrain on most recent data
The walkforward 2024-25 fold model trained on 2005-2023 data. A model trained on 2005-2024 (with 2024-25 included in training) might score 2025-26 better. **Test before adopting filters.**

### C. Pivot to the only consistent finding: matchweek-early filter
Bet ONLY on matches in matchweeks 1-12. Across all three windows, ROI was positive (+9.5 / +5.4 / +20.2%). Smaller bet volume but more reliable.

## Recommendation

**Re-prioritize the betting improvement plan around these findings.**

1. **Phase B2 (April 24 filters): RECONSIDER.** Their backtest performance does not survive holdout. Only adopt if combined with B1' (model retrain on most recent data).

2. **Phase B1' (NEW): retrain walkforward model with 2024-25 INCLUDED in training**, evaluate on 2025-26 only. If ROI on 2025-26 improves materially with the fresh model, the staleness hypothesis is confirmed — old filters might still work after retraining.

3. **Phase B2' (NEW): adopt matchweek-early filter as a SECONDARY bet selection rule.** Even if other filters fail, this one survived holdout — worth running with strict size cap.

4. **Phase B3 (per-class thresholds): defer until B1' clarifies the staleness question.**

5. **The original B5 (bankroll tightening) and B6 (BTTS market) are unaffected** by these findings.

## What this analysis NEEDS

- **A larger 2025-26 holdout.** 271 bets is moderate. As more 2025-26 matches finalize (the season runs through May), re-running this analysis monthly will tighten the verdict.
- **A "freshly retrained" comparison.** Training a model that includes 2024-25 in its training data, then scoring 2025-26 with it, would test the staleness hypothesis directly. Multi-hour compute job.
- **EPL parallel test.** If EPL shows the same degradation pattern, it's a structural sportsbook-improvement story. If not, it's SA-specific (regime change in Italy).

## Key reproducibility info

- Script: `scripts/diagnostics/subset_alpha_search.py`
- Findings JSON: `data/diagnostics/subset_alpha_findings.json`
- Models used: `data/models/walkforward/serie_a/1x2__5season_seed42/`
- Bet selection rule: 5% edge threshold, fractional Kelly 0.25× capped 2%, opening odds = `odds_MaxH/D/A`

## Verdict

**STRONG (the finding is real and important).** Edge degradation across all April 24 subsets is too consistent across many slicing dimensions to be variance. The only subset that survives — matchweek-early — has plausible mechanistic explanation (book inefficiency on early-season uncertainty).

**Confidence:** medium-high. The 271-bet 2025-26 sample is informative but not yet decisive; another 200 matches would tighten the picture significantly.

**If I'm wrong:** the model staleness hypothesis (B1') would be the obvious counter-test. If freshly retraining produces +ROI on 2025-26, the audit's filters might still work in production after a retrain.

**What I'm NOT saying:** I'm not saying the model is broken. I'm saying the betting strategies that worked on 2022-24 don't work on 2025-26 with the current model. The path forward is either (a) accept smaller-edge reality, (b) retrain model freshly, or (c) pivot to the matchweek-early filter that survived.
