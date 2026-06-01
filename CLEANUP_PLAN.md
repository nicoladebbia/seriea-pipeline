# CLEANUP_PLAN.md — Kill-list, Keep-list & Architecture Gaps
> Generated 2026-06-01 from the 22-agent audit. Every deletion was re-verified mechanically: zero Python importers, zero string-invocations, zero launchd-plist references. Companion to ARCHITECTURE_MAP.md.

## 1. Confirmed deletions (true garbage — safe to remove)
All verified `py_refs=0 plist_refs=0`. These are abandoned experiments, not one-shots. Git history preserves them.

| File | Confidence | Reason |
|---|---|---|
| `scripts/models/breakthrough.py` | high | Abandoned model-training experiment. Zero importers, not in any live entry point or plist. Writes only MODELS_DIR/markets/breakthrough_results.json (the ripped-out corners/cards area, dead per CLAUDE.md 2026-05-06). Hangs off the experimental comprehensive_markets.py island. No artifact consumed by live code. |
| `scripts/models/calibrated_ensemble.py` | high | Abandoned experiment off the comprehensive_markets.py dead cluster. Zero importers, not invoked by any plist or live entry point, produces no artifact consumed by the pipeline. |
| `scripts/models/draw_aware_ensemble.py` | high | Abandoned experiment off the comprehensive_markets.py dead cluster. Zero importers, not invoked anywhere. The live draw model is draw_detector.cbm (trained by retrain_draw_detector.py), not this. |
| `scripts/models/draw_oracle.py` | high | Abandoned experiment off the comprehensive_markets.py dead cluster. Zero importers, not invoked. Superseded by live draw_detector path. |
| `scripts/models/optimize_ensemble_calibration.py` | high | Abandoned optimization experiment. Zero importers, not invoked by any plist or live code. Live calibration lives in ml/calibration.py and the walkforward pipeline. |
| `scripts/models/optimize_weights.py` | high | Superseded duplicate. The live weight optimizer is scripts/models/weight_optimizer.py (in reachable_from_live and string_invoked); it defines its own optimize_weights() function. The only optimize_weights symbol referenced anywhere is inside the live file, not this dead script. Confirmed superseded. |
| `scripts/models/push_accuracy.py` | high | Abandoned accuracy-push experiment. The single reference (ensemble_prediction_engine.py:449) is a code comment about feature parity, not an import or invocation. Zero real importers, not in any plist. |
| `scripts/models/stacking_meta.py` | high | Abandoned experiment off the comprehensive_markets.py dead cluster. Zero importers, not invoked. |
| `scripts/models/tune_fast.py` | high | Abandoned tuning experiment off the comprehensive_markets.py dead cluster. Zero importers, not invoked. |
| `scripts/models/unified_model.py` | high | Abandoned experiment off the comprehensive_markets.py dead cluster. Zero importers, not invoked. Live unified training is scripts/models/train_unified.py (in reachable_from_live). |
| `scripts/models/predict_scorer_suitability.py` | medium | Abandoned player-prediction experiment. Zero importers, not invoked by any plist or live entry point. |
| `web/predictor.py` | high | Dead. Zero importers across the repo; web/app.py never imports MatchPredictor and serves predictions from pre-built predictions.json files instead. No runtime wiring exists. Confirmed by grep for MatchPredictor/web.predictor. |

**Total: 12 files.** The `scripts/models/` cluster (10 of them) all hang off the abandoned `comprehensive_markets.py` island — the corners/cards/BTTS work ripped out 2026-05-06.

## 2. Keep-list — one-shot scripts (do NOT delete)
Zero importers, but valid standalone backfills/diagnostics/scrapers. Project rule keeps these; git preserves re-derivation.

- `scraper/cloudflare_solver.py`
- `scraper/fbref_auto_scraper.py`
- `scraper/fbref_fast.py`
- `scraper/fbref_selenium.py`
- `scraper/upcoming.py`
- `scraper/xcomp_scraper.py`
- `scripts/models/generate_match_reasoning.py`
- `scripts/models/predict_walkforward_markets.py`
- `scripts/models/train_draw_specialist_production.py`
- `scripts/models/validate_league_deployment.py`
- `scripts/analysis/backtest_multimarket.py`
- `scripts/analysis/backtest_player_props.py`
- `scripts/analysis/calibration_analysis.py`
- `scripts/analysis/data_quality_report.py`
- `scripts/analysis/feature_importance_analysis.py`
- `scripts/analysis/formation_analyzer.py`
- `scripts/analysis/high_confidence_analyzer.py`
- `scripts/analysis/performance_tracker.py`
- `scripts/analysis/player_data_audit.py`
- `scripts/analysis/train_draw_specialist.py`
- `scripts/analysis/validate_player_backfill.py`
- `scripts/diagnostics/subset_alpha_search.py`
- `parser/html_parser.py`
- `ml/comparison.py`
- `ml/ou_model.py`
- `ml/prediction.py`
- `tools/generate_download_urls.py`
- `tools/open_urls_in_browser.py`
- `tools/verify_downloaded_html.py`

## 3. Architecture gaps — should-be-connected-but-isn't
Validated all 7 narrator-flagged files against .plans/repo-audit/import_graph.json plus precise import-grep cross-checks. Core finding: ZERO of the 7 are genuine "missing wires" — the narrators' should-be-connected premise is wrong for this file class. They split into three real actions. (1) DEDUPE: subset_alpha_fresh.py and subset_alpha_fresh_epl.py differ by exactly 6 lines (all league-param swaps) — collapse to one league-parametrized file. The narrator's separate "subset_alpha_search is an exact duplicate" claim is FALSE; search.py has genuinely different model-selection logic and stays. (2) DELETE dead twins where a live replacement already exists: formation_analyzer.py (live = features/formation_analysis.py + scripts/prediction/formation_predictor.py); plus three orphans the narrators MISSED — ml/ou_model.py (live = scripts/models/over_under_model.py), ml/meta_learner.py (live = inline MetaLearnerCombiner in ensemble_prediction_engine.py:631), and ml/prediction.py + ml/comparison.py (zero importers, no successor). Wiring any of these in would be wrong — they're superseded. (3) FALSE ALARMS, working as designed: backtest_multimarket.py (narrator's "imports player_xg_model without using it" is false — it instantiates LineupXGPredictor at lines 469-476; and it imports backtest_unified rather than being superseded by it), backtest_player_props.py, calibration_analysis.py, data_quality_report.py — all are __main__ diagnostics/backtests; imported_by=[] is the intended architecture, and the live data-quality wire already exists via health_check.py. Ranked dedupe + dead-twin deletes above false alarms. Key caveat: import_graph.json's imported_by is accurate for module imports but cannot see reimplementations (meta_learner) — I verified the four dead modules with direct grep, not graph alone.

| File | Should connect to | Why | Impact |
|---|---|---|---|
| `scripts/diagnostics/subset_alpha_fresh.py + scripts/diagnostics/subset_alpha_fresh_epl.py` | Merge into one league-parametrized scripts/diagnostics/subset_alpha_fresh.py with a --league arg | REAL duplication, confirmed by diff: the two files differ by exactly 6 lines, all of which are league-param swaps (matches["league"]=="serie_a" vs "premier_league", features_serie_a.parquet vs features_premier_league.parquet, models/walkforward/serie_a vs /premier_league). The other ~446 lines are byte-identical. This is the only flagged item that is a genuine maintenance gap. Collapse to one file taking league as a parameter (the codebase already uses ACTIVE_LEAGUES conventions elsewhere). NOTE: the narrator's separate claim that subset_alpha_search.py is an 'exact duplicate' is FALSE — search.py differs from fresh.py in real model-selection logic (fresh maps 2025-26 to a separately-trained 1x2__fresh_2025_seed42 model; search uses the latest-available fold season_2024-2025.cbm). Keep search.py separate. | medium |
| `scripts/analysis/formation_analyzer.py` | DELETE — superseded by the live formation path: features/formation_analysis.py (imported by features/build.py + ensemble_prediction_engine.py) and scripts/prediction/formation_predictor.py (imported by run_full_pipeline.py) | Narrator is RIGHT that it has no consumers (imported_by=[], only referenced by a smoke-import in tests/test_pipeline.py line 62), but WRONG to frame it as a missing wire. Wiring it in would be a mistake: a fully-live formation subsystem already exists and is consumed by both the feature build and the prediction engine. This standalone CLI (--match/--all/--tier2-update) is the older, superseded twin. Action is delete, not connect. | medium |
| `ml/ou_model.py` | DELETE — superseded by the live OU path: scripts/models/over_under_model.py (imported by run_full_pipeline.py) | GAP THE NARRATORS MISSED. Dead module sitting in the live ml/ directory. imported_by=[] in the graph and zero real importers confirmed by grep (the only 'ou_model' hits are self-references and the unrelated scripts/models/over_under_model.py). It defines train_goals_model/predict_ou/predict_btts — an entire parallel OU+BTTS implementation that nothing calls. The production OU market is served by scripts/models/over_under_model.py, which IS wired into run_full_pipeline. Delete the orphan. | medium |
| `ml/meta_learner.py` | DELETE — superseded by the inline MetaLearnerCombiner class in scripts/prediction/ensemble_prediction_engine.py (line 631) | GAP THE NARRATORS MISSED, and a trap I almost fell into. A noisy substring grep suggested the ensemble engine references meta_learner, but the engine actually defines its OWN MetaLearnerCombiner class internally (loads data/models/universal/meta_learner.pkl) and never imports ml/meta_learner.py. Confirmed: zero real importers (grep for 'from ml.meta_learner import' returns nothing). The live meta-learner was reimplemented inside the engine; this module is dead. Delete. | medium |
| `ml/prediction.py + ml/comparison.py` | DELETE — orphans in the live ml/ directory; no live successor needed, no importers | GAP THE NARRATORS MISSED. Both have imported_by=[] in the graph and zero real importers confirmed by precise grep ('from ml.prediction import' / 'from ml.comparison import' both return nothing — earlier apparent hits were substring noise like 'prediction' inside 'predictions'). They sit in ml/ alongside live model code, which makes them look load-bearing during code-reading. Dead weight; delete to stop future readers treating them as part of the model stack. | low |
| `scripts/analysis/backtest_multimarket.py` | No wire needed — FALSE ALARM. Standalone backtest by design. | Both narrator sub-claims are WRONG, verified in the file body. (1) 'imports player_xg_model without using it' is false: lines 469-476 actively instantiate PlayerXGDatabase and LineupXGPredictor and use them for lineup-xG scoring (lines 314-350). (2) 'superseded by backtest_unified' is backwards: backtest_multimarket IMPORTS backtest_unified (it sits on top of unified, not under it). It is a __main__ analysis tool with imported_by=[] — that is the intended architecture for a backtest, not a missing connection. | low |
| `scripts/analysis/backtest_player_props.py, scripts/analysis/calibration_analysis.py, scripts/analysis/data_quality_report.py` | No wire needed — FALSE ALARMS. Standalone diagnostics/backtests by design. | All three are __main__-entrypoint analysis tools with imported_by=[]; not being in the live pipeline is the architecture, not a gap. backtest_player_props correctly sits on scripts/betting/player_predictions.py (which IS live, imported by run_full_pipeline). calibration_analysis correctly imports the live ensemble_prediction_engine to analyze its calibration offline. data_quality_report's checks are duplicated by the LIVE health path (scripts/pipeline/health_check.py:check_data_quality, run by monitor.py + scheduler.py) — so the live data-quality wire already exists; the analysis-tier report is a deeper offline diagnostic, not a missing feedback loop. | low |

## 4. Live/active files graded D or F (0) — edit candidates
These run in production or are kept one-shots but scored low on observable quality. Worth a cleanup pass.

| File | Grade | Signal | Improve |
|---|---|---|---|

## 5. Reconciliation — gap-analyst overrides (verified)

The architecture-gap agent caught 4 dead orphans the file-narrators and the cleanup-adjudicator missed (adjudicator wrongly filed them as one-shots). Re-verified `refs=0` + live successor exists → **added to the delete set**:

| File | Verified | Live successor |
|---|---|---|
| `ml/ou_model.py` | refs=0 | `scripts/models/over_under_model.py` (live, run_full_pipeline) |
| `ml/meta_learner.py` | refs=0 | inline `MetaLearnerCombiner` in `scripts/prediction/ensemble_prediction_engine.py` |
| `ml/prediction.py` | refs=0 | orphan, no successor needed |
| `ml/comparison.py` | refs=0 | orphan, no successor needed |

**Final safe-delete count: 16** (12 confirmed + these 4).

### Flagged for review (NOT auto-deleted)
- `scripts/diagnostics/subset_alpha_fresh.py` + `subset_alpha_fresh_epl.py` — differ by exactly 6 lines (league params). **Merge** into one `--league`-parametrized file. Left for you (it's an edit, not a delete).
- `scripts/analysis/formation_analyzer.py` — superseded by `features/formation_analysis.py`, but `tests/test_pipeline.py` smoke-imports it. Deleting breaks the test. Decide: drop the test stanza + delete, or keep both.
