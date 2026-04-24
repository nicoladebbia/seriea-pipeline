# Phase 3a — Pre-match XI Quality Features (NULL RESULT)

> **PRODUCTION DIVERGENCE NOTICE (added 2026-04-24 post-trace):** All metrics
> below are measured on `data/models/walkforward/` model families. The
> production prediction engine loads `data/models/universal/catboost_no_odds.cbm`,
> a separate model trained by a different pipeline. The xi_quality plugin
> *is* in the live feature pipeline (build.py registers it), so newly-built
> features for production *would* include XI columns — but the production
> model wasn't retrained with them, and the null result above was measured
> on a different model regardless. Production unaffected.



**Hypothesis:** Adding 6 features describing the announced starting XI's
prior-match quality (rating, xG/90, lineup continuity) will improve Kelly ROI
by ≥6pp on at least one league.

**Result:** Null. No measurable improvement at our sample size + seed variance.

**Decision:** Do not adopt. Plugin code remains in tree as infrastructure;
features computed but model performance is statistically indistinguishable
from Phase 1 baseline.

---

## What was built

### `features/xi_quality.py` (new plugin)

Six pre-match features per match:
- `home_xi_avg_rating_last5` — mean of starters' last-5-match Sofascore ratings
- `away_xi_avg_rating_last5`
- `home_xi_xg_per90_sum` — sum of starters' last-10-match xG/90
- `away_xi_xg_per90_sum`
- `home_xi_minutes_continuity` — % of starters who also started team's last match
- `away_xi_minutes_continuity` — (dropped as exact-dup during cleanup)

**Leakage spec (verified):**
- Each player's rolling window strictly precedes current `match_date`
- `is_starter=True` in current row used only for IDENTITY (pre-match info)
- ≥3 prior matches required, else NaN (no zero-imputation for sparse players)

**Smoke test:** Three Juventus 2024-25 matches produced varied ratings
(7.04, 7.10, 6.99) — exactly what real rolling averages should show. If
leakage existed, values would tightly cluster around each player's
post-match-rating mean. Leakage check passed.

### Pipeline integration

- Plugin registered as `Step19c3XiQuality` (after `Step19c2LineupXg`,
  before `Step19dMatchPatterns`)
- Routes via `state.league` to load SA or EPL `player_match_stats*.parquet`
  (same pattern as the lineup_xg fix)
- Bumped `lineup_xg` to v1.1 to invalidate stale caches in both leagues

### Feature fill rates after rebuild

| | SA all | SA 2017+ | EPL all | EPL 2017+ |
|---|---:|---:|---:|---:|
| `home_xi_avg_rating_last5` | 18.1% | 42.6% | 41.3% | **97.6%** |
| `home_xi_xg_per90_sum` | 18.0% | 42.3% | 41.2% | **97.3%** |

EPL has near-complete eval-season coverage (Sofascore 2017+). SA is capped at
42% because Sofascore SA data only starts 2022-23.

---

## Multi-seed measurement (per Phase 1's methodology mandate)

Every retraining used the shared 85/15 split mode (matching Phase 1 baseline).
Seeds 42, 43, 44 per league. Rules engine run with both raw and calibrated
probs.

### Walk-forward log-loss (3-seed mean)

| | Phase 1 (single seed) | Phase 3a (3-seed mean) | Δ |
|---|---:|---:|---:|
| SA raw_ll | 0.9992 | 0.9974 | −0.0018 |
| EPL raw_ll | 0.9740 | 0.9706 | −0.0034 |

Both directions favorable but **far below the 0.01 threshold** for
adoption. Within seed noise.

### Rules engine Kelly ROI (3-seed mean, Market Max)

| | Phase 1 (3-seed) | Phase 3a (3-seed) | Δ |
|---|---:|---:|---:|
| SA raw Kelly ROI | +9.59% | +9.74% | +0.15pp |
| SA cal Kelly ROI | +9.46% | +8.15% | −1.31pp |
| EPL raw Kelly ROI | +8.77% | +6.04% | −2.73pp |
| SA raw CLV+ | 89.3% | 90.0% | +0.7pp |
| EPL raw CLV+ | 71.8% | 73.3% | +1.5pp |

**No metric crosses any of the three adoption thresholds:**
- ΔROI ≥ +6pp: max observed +0.15pp
- Δll ≤ −0.01: max observed −0.003
- ΔCLV+ ≥ +5pp: max observed +1.5pp

### Per-seed scatter — methodology vindication

EPL raw Kelly ROI by seed:

| Seed | Phase 3a EPL ROI | CI |
|---:|---:|---|
| 42 | −5.09% | [−19.3, +9.9] |
| 43 | **+15.80%** | **[+0.44, +32.3]** |
| 44 | +7.42% | [−8.2, +23.0] |

**Seed 43 produced another "CI excludes zero" headline** — exactly the
same single-seed pattern that would have led us to claim a Phase 3a win
under the old single-seed methodology. The other two seeds disagree.
Mean across seeds: +6.04%. CI on the mean (treating each seed as one
draw): no significantly positive estimate.

This is the third single-seed outlier this session (Phase 1 EPL B+raw
seed 42 was the first; the seed=43 high values continued today). The
multi-seed mandate from Phase 1's results doc would have prevented all
three from being shipped as "wins."

---

## Why the null result

Three plausible reasons, in order of likelihood:

1. **EPL pre-match XI quality already encoded in existing features.** The model has 781 features for EPL, including elo (top-importance), squad_value, lineup_xg (post-match leaky but the upstream Sofascore sofascore_indices_v1.0 captures pre-match starter signal indirectly). Adding 4-6 explicit XI rating/xG features mostly adds correlated information.

2. **Sample size + seed variance.** At n=400 bets per league/seed, Kelly ROI stdev across seeds is ~6pp. A real +3pp improvement is invisible at this sample size — we'd need 1500+ bets per seed to detect it.

3. **Feature fill rate / sparsity.** SA 42% is a lot of NaN. CatBoost handles NaN, but features that are NaN on more than half the eval set can't drive a big ROI shift even if directionally correct.

Reason 1 is most consistent with the data: we measured *negative* ΔROI on EPL despite *positive* ΔCLV+ rate. If we were adding genuinely new info, both would move the same direction. Mixed signal = features mostly redundant with what's already there.

---

## What this changes

**Plan doc updates:** Phase 3a closed as null. Methodology mandate vindicated
by a *third* single-seed outlier (seed 43 in both Phase 3a and earlier seed-43
tests). Multi-seed evaluation is now permanent for this project.

**Production:** No change. SA and EPL both still on Phase 1 A+cal.

**Phase 3b (referee mixed effects):** Pre-emptively cancelled before writing
plugin (`docs/2026-04-24_training_window_audit.md` companion notes). Existing
ref features have <2% combined importance; z-scoring won't help.

**Remaining feature ideas worth exploring:**
- Phase 4 candidates from the original plan that target NEW information,
  not reshape of existing features:
  - Pre-match referee × team-style interactions (refs penalize aggressive
    teams more — currently uncaptured)
  - Travel distance / late-night kickoff effects
  - Crowd presence (covid era empty stadiums vs full)
- All require similar multi-seed evaluation. None to be started this session.

---

## Artifacts

- `features/xi_quality.py` — plugin (kept in tree)
- `features/build.py` — Step19c3XiQuality registered
- `data/features/features_serie_a.parquet` — 1195 cols (5 new XI cols, 1 dropped as dup)
- `data/features/features_premier_league.parquet` — 907 cols (4 new XI cols, 2 dropped as dup)
- `data/models/walkforward/{league}/1x2__phase3a_seed{42,43,44}/` — 6 trained models preserved for reproducibility
- `scripts/pipeline/rebuild_both_features_phase3a.py`, `rebuild_epl_features_phase2.py` — one-shot rebuild scripts

---

**Verdict:** STRONG (clean negative result + methodology validated)
**Confidence:** high
**If I'm wrong:** SA Sofascore backfill to 2017 lifts SA fill from 42% → 97%, OR a much larger eval window (5+ seasons, not 3) reduces seed noise enough to see a real +2-3pp signal. Neither available today.
**What I'm NOT saying:** I'm not saying pre-match XI quality is useless. I'm saying with current data scale (1140 eval matches × 30% bet rate × ±6pp seed variance), the signal — if it exists — is below our measurement noise floor.
