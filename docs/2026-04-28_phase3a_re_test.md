# Phase 3a Re-Test — XI Quality Features (NULL RESULT, RECONFIRMED)

**Date:** 2026-04-28
**Tested:** xi_quality features re-evaluated under better conditions than original Phase 3a (2026-04-24).
**Outcome:** Null. The features still don't help — but for a different reason than the original null diagnosis.

## Why we re-tested

The original Phase 3a (`docs/2026-04-24_phase3a_xi_quality.md`) found the xi_quality features did not move walk-forward log-loss or Kelly ROI. Diagnosis at the time was **data sparsity** — SA Sofascore feature fill was 42% (only 2022+ had Sofascore).

On 2026-04-27, we backfilled SA Sofascore to 2017 (Phase A-C of the cleanup session). New fill rate for xi_quality columns: **89-100% across all 5 eval seasons (2020-21 through 2024-25)**.

This re-test asks: with full data coverage, do the features now show signal?

## Methodology

- Walk-forward CV on Serie A 1X2.
- Eval seasons: 2020-21, 2021-22, 2022-23, 2023-24, 2024-25 (the new 5-season default established 2026-04-27).
- 3 seeds (42, 43, 44) per condition. Multi-seed mandate applied.
- Two conditions:
  - **Baseline:** all 1187 features including xi_quality (= F1 result).
  - **Ablation:** 1180 features, the 7 xi_quality columns removed via `--exclude-features`.
- Aggregate metrics: mean and stdev across 3 seeds.

## Results

| | Acc | LL | ECE | Draw recall |
|---|---:|---:|---:|---:|
| **Baseline (with xi_quality)** | 0.5300 ±0.0019 | 1.0164 ±0.0148 | 0.0311 ±0.0048 | 0.0699 ±0.0252 |
| **Ablation (no xi_quality)** | 0.5290 ±0.0046 | 1.0190 ±0.0199 | 0.0195 ±0.0066 | 0.0518 ±0.0127 |
| **Δ (xi_quality effect)** | **+0.0010 (+0.10pp)** | **-0.0026** | **+0.0116** | **+0.0181** |

### Per-seed

| Seed | Baseline acc | Ablation acc | Δ | Baseline ll | Ablation ll | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.5305 | 0.5321 | -0.16pp | 1.0315 | 1.0276 | +0.0039 |
| 43 | 0.5316 | 0.5237 | +0.79pp | 1.0019 | 0.9962 | +0.0057 |
| 44 | 0.5279 | 0.5311 | -0.32pp | 1.0157 | 1.0331 | -0.0174 |

**Per-seed direction is inconsistent.** Seed 42 and 44 say "ablation slightly better" (xi_quality hurts); seed 43 says "baseline better" (xi_quality helps). The original Phase 3a found the same flip-flop on a different seed. **This is the signature of seed noise on a small effect.**

## Adoption thresholds (set in advance)

- Δ accuracy ≤ -0.004 (≥0.4pp improvement)? **NO** (Δ = +0.10pp, wrong direction, below noise threshold).
- Δ log-loss ≤ -0.005 (≥0.005 improvement)? **NO** (Δ = -0.0026, wrong direction, below noise threshold).
- Δ CLV+: not measured in this run (would require deep_backtest_1x2 against Market Max odds — separate test).

**Verdict: NULL.** Do not adopt the xi_quality features as production-promoted. Do not remove them either — they're not hurting enough to justify change risk.

## Why was the original Phase 3a null diagnosis wrong?

The original doc (2026-04-24) hypothesized: "Sample size + seed variance — at n=400 bets per league/seed, Kelly ROI stdev across seeds is ~6pp. A real +3pp improvement is invisible at this sample size — we'd need 1500+ bets per seed to detect it."

That was a partially correct hypothesis, but the actual mechanism is different:

- **Original null hypothesis (2026-04-24):** features have signal, but it's drowned by sample-size noise.
- **Re-test diagnosis (2026-04-28):** features have signal, but it's **redundant with existing Sofascore features** that already fill at 99%. Adding `home_xi_avg_rating_last5` (starter-only rating average) provides no new information once `home_ss_roll_rating` (full-team rating average) is already in the feature set.

Evidence for the redundancy diagnosis:
- We doubled the eval sample size (1140 → 1900) to halve seed noise.
- We multiplied SA fill rate by 2.5× (42% → 99%).
- Both improvements should have surfaced ANY real signal.
- The result is still null.
- Therefore the signal was never there — these features just re-encode what already exists.

## Side findings

**ECE is meaningfully better in the ablation** (0.0195 vs 0.0311). Removing the 7 features tightened calibration by 1.16pp. This suggests xi_quality features were adding **noise** to the calibrator, not just being neutral.

**Draw recall is worse in the ablation** (5.18% vs 6.99%). xi_quality might have been weakly helping draw detection, even if it didn't move overall accuracy. But draw recall is dominated by the broader draw-class problem (model predicts draws ~5-10% of the time; actual base rate is 27%). The 1.81pp gain doesn't change the overall picture.

## What to do next

1. **Keep xi_quality features in the feature set.** Removing them causes change risk for negligible benefit, and the ECE win could be coincidence.
2. **Don't promote them.** Their effect on production accuracy is null.
3. **The real lesson is methodological:** ablations like this should be standard practice for ANY new feature group. The infrastructure now exists (`--exclude-features`). Run it before declaring a Phase X feature group "useful."
4. **Future angle: feature pruning.** The 1187-feature model could likely drop 50-100 low-importance features with no accuracy hit and better calibration. Worth a separate session.

## Verdict

**STRONG NULL.** Confidence: high.

**If I'm wrong:** the seed-43 single-seed positive (Δ -0.79pp acc) might mean the features ARE helping in some training regime we haven't found. To test, run with 6+ seeds instead of 3, see if the seed-43 result reproduces or stays an outlier. Cost: 3 more seeds × 12 min = 36 min. Worth doing only if the ROI question becomes load-bearing.

**What I'm NOT saying:** I'm not saying xi_quality features are useless universally. I'm saying for THIS model on THIS data, with the existing Sofascore features already in place, they don't add signal. A different model architecture or a different feature set might find them useful.
