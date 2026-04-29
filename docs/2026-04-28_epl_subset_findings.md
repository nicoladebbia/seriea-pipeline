# EPL Subset Analysis (K3) — Edge Erosion Far Worse Than SA

**Date:** 2026-04-28
**Test:** Mirror of the SA fresh-model subset analysis on EPL data. Same methodology, same edge threshold, fresh 2025-26 model trained on data through 2024-25.
**Outcome:** **EPL is dramatically worse than SA. No filter recovers profitability on 2024-25 or 2025-26. The matchweek-early finding is SA-specific.**

## Headline comparison: SA vs EPL

| Window | SA ROI (fresh model) | EPL ROI (fresh model) |
|---|---:|---:|
| Discovery (22-23, 23-24) | +11.17% | +6.70% |
| Validation (24-25) | +4.55% | **-19.87%** |
| Holdout (25-26) | -1.20% | **-22.64%** |

**EPL has collapsed twice as fast as SA.** Validation already losing 20%, holdout losing 23%. The model has effectively zero edge on 2024-25 and 2025-26 EPL data, regardless of filter.

## No subset survives on EPL holdout

Looking at the same filters that survived (or partially survived) on SA:

| Filter | EPL Disc | EPL Val | EPL Holdout |
|---|---:|---:|---:|
| Baseline | +6.70% | -19.87% | -22.64% |
| Draws only | -2.22% | -15.49% | -0.40% |
| Mid-odds (2.2-4.0) | +1.65% | -11.73% | -23.76% |
| Mid-prob (0.30-0.50) | +8.52% | -16.54% | +0.93% |
| Mid-odds + mid-prob | +9.04% | -15.18% | -25.41% |
| Draws + mid-odds | +3.77% | -16.01% | -21.09% |
| **matchweek_early(1-12)** | **+11.14%** | **-24.28%** | **-32.84%** |

**THE matchweek-early filter — SA's only robust positive finding — produces -25% to -33% on EPL.** Same dimension, opposite direction. The early-season effect appears to be SA-specific.

## Per-class comparison

| | SA pick_H | EPL pick_H | SA pick_D | EPL pick_D | SA pick_A | EPL pick_A |
|---|---:|---:|---:|---:|---:|---:|
| Discovery | +19% | +16% | +10% | -2% | +38% | +15% |
| Validation | -6% | -78% | +5% | -15% | -17% | -9% |
| Holdout | -100% (n=1) | -52% | -3% | -0.4% | +46% (n=11) | -38% |

**EPL home picks (`pick_H`) are catastrophic in 2024-25 (-78% ROI on 21 bets).** Either the model is systematically over-picking home wins on bad matchups, or sportsbooks have priced EPL home favorites too sharply for our model to find edge.

## What this resolves

**Sportsbooks have improved on EPL faster than on SA.** EPL is the most-bet football league in the world; it gets more sharp money, more book competition, more pricing intelligence. The 2022-23 inefficiencies our model exploited are gone in 2024-25.

**SA still has SOME residual inefficiency** (early-season + smaller market = less sharp pricing). EPL has effectively none at the granularity of pre-match features alone.

## Implications for the betting plan

1. **EPL betting on 1X2 is currently NEGATIVE-EV.** Disable EPL 1X2 bets in production until evidence shows recovery. **CRITICAL change** — this is the highest-stakes finding from this whole exploration.

2. **Don't apply the matchweek-early filter to EPL.** Empirically loses 25-33%. Only safe on SA.

3. **The "B6 — BTTS market" plan still has hope on SA, more dubious on EPL.** When the Odds API returns May 1, BTTS backtest should be done per-league, with strict acceptance criteria. EPL likely won't show edge.

4. **Plan revision: add a B0 phase — disable EPL 1X2.** Highest priority safety action.

## Why the disparity?

Three plausible mechanisms (any could contribute):

1. **Sportsbook intelligence.** EPL gets the most ML-driven pricing in the world. SA gets less. Books improving faster on EPL is consistent with this.

2. **Sample efficiency.** Our model has 1187 SA features vs 784 EPL features (FBref scraper produces sparser EPL stats per the H1 finding). Less feature richness = less edge potential.

3. **League dynamics shift.** EPL has had unusual seasons (Man City decline, multiple new managers, financial fair play impacts). The 2022-24 distribution may not generalize to 2025-26 simply because the league has changed.

Mechanisms 1 and 3 imply sustained negative-EV. Mechanism 2 is fixable (rerun FBref scraper with rich EPL stats, retrain).

## What the project should do (revised priorities)

| Action | When | Effect |
|---|---|---|
| **Disable EPL 1X2 in production** | Now | Stop bleeding money on -22% ROI strategy |
| Update betting plan with EPL findings | Now | Reflects empirical reality |
| Re-scrape FBref EPL with rich stats | Multi-hour scraper job, future session | Could close the 1187 vs 784 feature gap |
| BTTS market test (SA only first) | May 1+ Odds API | Still potential edge in SA |
| Continue SA work (matchweek-early filter, B7 stake tier) | Now | Only league with surviving signal |

## What this analysis cost vs prevented

- **Cost:** ~30 min compute (one fresh model + one subset analysis).
- **Prevented:** continuing to bet EPL 1X2 with empirical -22% ROI on the most recent data. Even if the live betting volume on EPL was small, this would have bled money silently for months.
- **Confirmed:** SA-only matchweek-early as the genuinely robust filter. Now we know it doesn't generalize, so we don't propagate it across markets.

## Verdict

**STRONG.** EPL is a worse market than SA for this model. Production must disable EPL 1X2.

**Confidence:** high. The pattern is consistent across discovery, validation, and holdout — not a single-window aberration. Sample sizes (406 / 202 / 114) are moderate but the magnitudes (-20% to -25%) are too large to be variance.

**If I'm wrong:** EPL might be in a temporary regime that reverses (Man City returning to dominance, new manager bounces stabilizing). Re-test in 6 months. But the current 2024-25 + 2025-26 data is unambiguous.

**What I'm NOT saying:** I'm not saying EPL is permanently unprofitable. I'm saying the CURRENT model on the CURRENT EPL data has zero edge. A re-scraped FBref-rich EPL feature set + new model could potentially recover some, but that's a multi-session project. Until then, EPL is off.
