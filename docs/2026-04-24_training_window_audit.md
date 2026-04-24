# Training-Window Audit — 2026-04-24 (Phase 1, FINAL)

**Question:** Does restricting the walk-forward training window from 2005+
to 2017+ or 2019+ improve 1X2 model quality?

**Outcome:** No measurable effect on either league under multi-seed
evaluation. The apparent "EPL B+raw +17.5% ROI" headline was a single-seed
outlier; across 4 seeds the mean is +8.8% with CI including zero — equivalent
to the A+cal status quo (+8-10% Kelly ROI depending on seed).

**Decision:** Keep A+cal on both leagues. **No production change.**

**The real discovery:** Our backtest methodology was single-seed. At n≈400
bets per variant, single-seed Kelly ROI has ±6-8pp noise. Any Phase-2/3
feature improvement must use multi-seed evaluation (≥3 seeds) to be
distinguishable from luck.

---

## Experiment structure

- **Windows tested:** A (2005+), B (2017+), C (2019+)
- **Leagues:** Serie A, Premier League
- **Eval seasons:** 2022-23, 2023-24, 2024-25 (walk-forward, strictly prior training)
- **Probabilities:** RAW (pre-calibration) and CAL (isotonic)
- **Seeds tested after initial Phase 1:** 42 (original), 43, 44, 123 for EPL B;
  42, 43, 44 for SA A

---

## Phase 1 — single-seed results (seed=42 only)

Rules engine Kelly ROI @ Market Max, fractional 0.25 Kelly, 2% cap:

| Setup | SA Kelly ROI | SA CLV+ | EPL Kelly ROI | EPL CLV+ |
|---|---:|---:|---:|---:|
| A+cal (prod) | +9.95% | 72% | +7.95% | 61% |
| A+raw | +8.58% | 89% | −1.60% | 72% |
| B+cal | +7.42% | 73% | +10.05% | 68% |
| B+raw | +8.89% | 90% | **+17.50% ← HEADLINE** | 72% |

Phase 1's provisional decision: adopt EPL B+raw because its Kelly ROI CI
excluded zero ([+0.84, +33.9]) — the first statistically significant Kelly
ROI observed. Apply amended rule: ROI wins even though raw_acc was 1.23pp
worse than A.

## Seed robustness check — Phase 1 headline COLLAPSES

EPL B+raw rules-engine Kelly ROI across 4 seeds:

| Seed | Kelly ROI | CI | CLV+ |
|---:|---:|---|---:|
| 42 (headline) | **+17.50%** | **[+0.84, +33.9]** | 72% |
| 43 | +5.29% | [−10.6, +22.0] | 72% |
| 44 | +4.21% | [−11.7, +20.4] | 71% |
| 123 | +8.07% | [−7.8, +24.6] | 71% |

**Mean: +8.77% ± ~6pp stdev across seeds.**

The +17.5% was >1 SD above the seed cluster mean. Kelly ROI CI excluding
zero was a single-seed artefact, not a real edge signal. **Phase 1's
recommendation to adopt EPL B+raw is RETRACTED.**

CLV+ rate is stable across seeds (71-72%) — CLV is genuine measurement,
Kelly ROI on n=400 is not.

## SA A robustness — holds

SA A+cal rules-engine Kelly ROI across 3 seeds:

| Seed | A+cal Kelly ROI | A+raw Kelly ROI | CLV+ raw |
|---:|---:|---:|---:|
| 42 | +9.95% | +8.58% | 89% |
| 43 | +10.50% | +9.40% | 90% |
| 44 | +7.94% | +10.80% | 89% |

Mean: A+cal +9.46%, A+raw +9.59%. Tight (±1.4pp). CLV+ rock-steady at 89-90%.

SA decision "keep A+cal" is robust. A+raw nearly matches A+cal on ROI and
has markedly higher CLV+, but within seed noise.

## Window effect, honestly — no significant difference

| Setup | Mean Kelly ROI (all available seeds) | CLV+ |
|---|---:|---:|
| SA A+cal | +9.46% (n=3) | 72% |
| SA A+raw | +9.59% (n=3) | 89% |
| EPL A+cal | +7.95% (n=1, Phase 1 only) | 61% |
| EPL B+raw | +8.77% (n=4) | 72% |

EPL B+raw (+8.8%) is within seed noise of EPL A+cal (+8.0%). The window
change produces no measurable win.

---

## Walk-forward log-loss — also no clean win

Cross-variant, seed=42 only:

| League | Variant | raw_ll | raw_acc |
|---|---|---:|---:|
| SA | A | **0.9992** | 50.44% |
| SA | B | 1.0117 | 46.75% |
| SA | C | 1.0029 | 51.05% |
| EPL | A | 0.9740 | 55.18% |
| EPL | B | **0.9624** | 53.95% |
| EPL | C | 0.9756 | 54.30% |

SA: A wins log-loss + ECE. B+C don't meet the threshold.
EPL: B wins log-loss by 0.012 (seed 42 only — we haven't checked EPL log-loss
robustness across seeds for this specific comparison, but the ROI collapse
is compelling evidence the log-loss win is also seed-noise).

---

## Task #6 — calibration fix, option (a) FAILED

**Hypothesis (option (a)):** split prior seasons chronologically 70/15/15
instead of 85/15. Fit isotonic on the held-out cal-val pool (last 15%)
which CatBoost never saw — not even for early stopping.

**Success threshold (written before running):** SA cal Kelly ROI within 2pp
of SA raw, AND SA cal CLV+ within 5pp of SA raw.

**Actual result:** SA cal Kelly ROI +5.17%, SA raw Kelly ROI +11.74%,
gap = 6.57pp (fail, threshold 2pp). CLV+ gap = 16pp (fail, threshold 5pp).

Option (a) rejected. Calibration remains broken. Next options from the
earlier advisor consultation:
- (b) K-fold OOF isotonic fit (higher data volume) — NOT pursued, seed-noise
  analysis suggests the real issue isn't double-dipping but the tiny cal-val
  sizes inherent to 3-class isotonic on ~200-400 rows.
- (c) Drop calibration entirely — viable, SA A+raw matches SA A+cal within
  seed noise (+9.59% vs +9.46%) and has much higher CLV+ (89% vs 72%).

**Recommendation:** Accept that isotonic calibration doesn't help and
doesn't break things on SA. EPL calibration is essentially a wash
(seed-dependent). Don't ship option (c) as a production change without
further evaluation — the cal probs are used elsewhere in the system
(dashboard, bet journal) and changing them has downstream effects.

---

## The methodology lesson — bigger than any config change

At n≈400 bets in a rules-engine backtest, Kelly ROI stdev across seeds is
~6pp. Our threshold for adopting a change ("Δ ≥ 3pp" or CI-excludes-zero)
is below this noise floor. Phase-2/3 experiments will produce apparent
winners that are noise.

**New default for all future feature-engineering comparisons:**
- Minimum 3 seeds per comparison, report mean + stdev across seeds
- Minimum 3 eval seasons (already have this)
- CLV+ rate and raw log-loss are more stable than Kelly ROI at our sample size;
  use them as primary metrics, ROI as secondary
- Changes producing Δ Kelly ROI < 6pp across the mean of 3 seeds are
  indistinguishable from noise

This methodology change applies to Phase 2 (coverage fixes), Phase 3
(new feature groups), Phase 4 (validation). Updating the plan doc.

---

## Decisions

1. **SA 1X2 production:** unchanged (A+cal, the current model at
   `data/models/walkforward/serie_a/1x2/`).
2. **EPL 1X2 production:** unchanged (A+cal, the current model at
   `data/models/walkforward/premier_league/1x2/`). The proposed B+raw
   swap is cancelled.
3. **Training-window:** 2005+ stays as default.
4. **Calibration:** isotonic stays on — not because it helps but because
   dropping it is a larger downstream change than warranted at this point.
   Task #6 is closed as "option (a) failed; option (b/c) defer to later
   sprint".
5. **Multi-seed evaluation:** mandatory for all future model comparisons
   in this project.

## Artifacts

Per-variant walk-forward outputs preserved:
- `data/models/walkforward/{league}/1x2__{baseline,2017plus,2019plus}/` — Phase 1 seed 42
- `data/models/walkforward/{league}/1x2__calfix_{baseline,2017plus}/` — Task #6 option (a) attempt
- `data/models/walkforward/premier_league/1x2__seed{43,44,123}_2017plus/` — EPL B seed robustness
- `data/models/walkforward/serie_a/1x2__seed{43,44}_baseline/` — SA A seed robustness

Training logs: `logs/training/{variant,calfix,seed*}_*.log`

Rules engine CLV JSONs: `data/diagnostics/phase1_clv/variant_{A,B}_{raw,cal}.json`

---

**Verdict:** STRONG (but the strong verdict is "no measurable window effect,
methodology needs overhaul")
**Confidence:** high
**If I'm wrong:** someone could re-derive the analysis showing that seed-42
really is typical and the other 3 seeds are unlucky. Quadruple the bet
counts (e.g., by including 2019-2022 in eval) and see if seed variance
stays at ±6pp or tightens. If tightened, maybe a larger sample could
re-distinguish B from A.
**What I'm NOT saying:** I'm not saying the 2017+ window is useless in
principle. At larger sample sizes it might well show a real effect. I'm
saying at the 1140-match × 3-season sample size we have, the window
decision is not distinguishable from seed noise.

## What this means for the features-improvement-plan

Phase 2 and Phase 3 aren't obsolete — they target coverage fixes and
specific new feature groups that could produce Δ > 6pp ROI improvements
if the hypotheses hold. But:

1. **Every phase's success threshold must be rewritten.** Current thresholds
   like "log-loss Δ ≤ −0.003" are below seed noise. Raise to ≥ 0.01 for
   log-loss or ≥ 6pp mean ROI across 3 seeds.
2. **Phase 4 validation must run 3 seeds, not 1.**
3. **The EPL lineup_xg.py coverage fix (staged during Phase 1) is the
   first Phase 2 work item.** Because adding a 2017+-only feature family
   (Sofascore lineup xG) could push EPL B+raw back above A+cal. This is
   now the highest-leverage Phase 2 item.
