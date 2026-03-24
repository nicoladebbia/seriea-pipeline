# Project Snapshot (2026-03-23 — Post-Tier-2 Deployment)

## Model & Metrics
Model: catboost_no_odds.cbm + 3-model ensemble (XGB+LGB+CB) | 35 features
Training: 2017+ data only (3,340 matches), time-decay 0.85/season, auto draw weights
CV: accuracy=0.6155 | log_loss=0.8589 | brier=0.1695 | F1_Draw=0.328
Production (2025-2026): accuracy=0.693 | ECE=0.031 | log_loss=0.709
Walk-forward backtest: +12.3% ROI | €1000→€4192 | 643 bets

## Betting
Engine: max_edge 8%, steam-move rejection, odds 1.5-2.0 dead zone gating (9% min edge), Kelly min 0.3%
Active markets: DC, O/U Over, Alt_OU (1X2 Draw in backtest only)

## Key Features (35 selected)
xG: us_xg_diff, us_team_xa_per_90, us_xg_rolling_std
Lineup: lineup_rating_mean
Odds velocity: line_vel_ou_over
Squad: squad_value_ratio, squad_total_value, median_player_value
Plus: Elo, form, H2H, injury, weather, contextual

## Architecture
- min_train_season=2017-2018 (drops pre-2017 low-signal data)
- universal_nan_threshold=0.45 (unlocks xG, pressing, lineup, odds velocity)
- Recency-weighted feature selection (base=1.5) + supplementary recent-folds pass
- Time-decay sample weights (0.85/season, Dixon-Coles)
- Auto draw weights (target 38% effective share)

## Knowledge Base
KB: 42 learnings — old 51-52% accuracy ceiling broken by 2017+ data + time-decay
