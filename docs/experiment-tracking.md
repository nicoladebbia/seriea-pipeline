# Experiment Tracking & Deployment

Read this before shipping model changes or logging experiments.

## Experiment Logging
Every model change MUST be logged. Use `scripts/log_experiment.py`:
```bash
python3 scripts/log_experiment.py --show          # Last 10 experiments
python3 scripts/log_experiment.py --compare        # Compare last to baseline
python3 scripts/log_experiment.py --description "..." --result SHIP --accuracy 0.556 --log-loss 0.9494
```
- **Experiment log**: `data/experiments/experiment_log.jsonl` (append-only)
- **Deployment state**: `data/models/deployment_state.json` (current model, metrics, rollback target)
- **Drift reports**: `data/experiments/drift_report.json` (KS-test results)

## Deployment & Rollback
Current deployment state is in `data/models/deployment_state.json`. Before shipping:
1. Read current state → know what you're replacing
2. Run backtest → compare against `rejection_thresholds`
3. On SHIP → `log_experiment.py --result SHIP` auto-updates deployment_state.json
4. On FAIL → `log_experiment.py --result REVERT` logs the failure
5. Rollback model is always available in `deployment_state.json → rollback.rollback_model`

## Data Drift Detection
Run before model training and after shipping:
```bash
# Quick z-score check (4 key features)
# Full KS-test (all numeric features) — see data-validator agent
# Feature staleness check — see data-validator agent
```
**Thresholds**: KS p<0.001 = CRITICAL (block training), p<0.05 = WARNING (flag for review)
