# Data Dictionary

## Overview

The feature table (`data/features/features.parquet`) contains 7,829 rows (one per match) and 484 columns spanning 21 Serie A seasons (2005-2026). Of these, 449 are ML-safe features (after excluding identifiers, raw stats, and the target).

## Parquet Tables

### `matches.parquet`

One row per match. Primary source: FBref + football-data.co.uk.

| Column | Type | Description |
|--------|------|-------------|
| `match_id` | str | Unique match identifier |
| `season` | str | Season (e.g. "2024-2025") |
| `match_date` | datetime | Kickoff date |
| `home_team` | str | Home team name |
| `away_team` | str | Away team name |
| `home_score` | int | Home goals |
| `away_score` | int | Away goals |
| `result` | str | H/D/A (target variable) |
| `venue` | str | Stadium name |
| `attendance` | int | Attendance figure |
| `referee` | str | Match referee |
| `home_xg` | float | Home expected goals (FBref/Understat) |
| `away_xg` | float | Away expected goals |
| `home_formation` | str | Formation (e.g. "4-3-3") |
| `away_formation` | str | Formation |
| `home_*/away_*` | float | Per-team raw match stats (see below) |

Raw stat columns per team: possession, passing_accuracy, shots_on_target, saves, yellow_cards, red_cards, fouls, corners, crosses, touches, tackles, interceptions, aerials_won, clearances, offsides, goal_kicks, throw_ins, long_balls.

### `player_stats.parquet`

One row per player per match.

| Column | Type | Description |
|--------|------|-------------|
| `match_id` | str | Match reference |
| `player` | str | Player name |
| `team` | str | Team name |
| `minutes` | int | Minutes played |
| `goals` | int | Goals scored |
| `assists` | int | Assists |
| `xg` | float | Expected goals |
| `xa` | float | Expected assists |
| `shots` | int | Total shots |
| `key_passes` | int | Key passes |
| `progressive_passes` | int | Progressive passes |
| `progressive_carries` | int | Progressive carries |
| `tackles_won` | int | Successful tackles |

### `goalkeeper_stats.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `match_id` | str | Match reference |
| `player` | str | GK name |
| `team` | str | Team |
| `saves` | int | Saves made |
| `psxg` | float | Post-shot expected goals |
| `goals_against` | int | Goals conceded |

### `shots.parquet`

One row per shot.

| Column | Type | Description |
|--------|------|-------------|
| `match_id` | str | Match reference |
| `minute` | int | Minute of shot |
| `player` | str | Shooter |
| `team` | str | Team |
| `xg` | float | Shot xG |
| `outcome` | str | Goal/Saved/Blocked/Off Target |
| `body_part` | str | Foot/Head |
| `distance` | float | Distance from goal |

## Feature Categories

### Rolling Features (Steps 1-3)

Format: `{side}_roll_{window}_{stat}`
Example: `home_roll_5_goals_scored`

Windows: 3, 5, 10 games.

| Pattern | Description |
|---------|-------------|
| `*_roll_*_goals_scored` | Rolling average goals scored |
| `*_roll_*_goals_conceded` | Rolling average goals conceded |
| `*_roll_*_xg` | Rolling average xG |
| `*_roll_*_xga` | Rolling average xG against |
| `*_roll_*_shots_on_target` | Rolling shots on target |
| `*_roll_*_possession` | Rolling possession % |
| `*_roll_*_passing_accuracy` | Rolling passing accuracy |
| `*_roll_*_tackles` | Rolling tackles per game |
| `*_roll_*_corners` | Rolling corners per game |
| `*_roll_*_yellow_cards` | Rolling yellows per game |
| `*_roll_*_points` | Rolling points per game (PPG) |
| `*_roll_*_clean_sheets` | Rolling clean sheet rate |

Sides: `home_`, `away_` (team-specific) and `home_home_`, `away_away_` (venue-split).

### xG Trend Features (Step 4)

| Feature | Description |
|---------|-------------|
| `*_xg_trend_5` | 5-game xG slope (improving/declining) |
| `*_xg_overperformance_5` | Goals minus xG (finishing luck) |
| `*_xga_trend_5` | Defensive xG trend |
| `xg_diff` | Home xG trend minus away xG trend |

### Strength Features (Step 5)

| Feature | Description |
|---------|-------------|
| `home_elo` / `away_elo` | Elo ratings (1500 = league mean) |
| `elo_diff` | Home Elo minus away Elo |
| `home_strength` / `away_strength` | Composite strength rating |
| `strength_diff` | Home minus away strength |

### Rest & Momentum (Steps 6-7)

| Feature | Description |
|---------|-------------|
| `home_rest_days` / `away_rest_days` | Days since last match |
| `rest_diff` | Home minus away rest |
| `home_momentum` / `away_momentum` | Momentum score (weighted recent results) |
| `momentum_diff` | Home minus away momentum |

### H2H Features (Step 9)

| Feature | Description |
|---------|-------------|
| `h2h_matches_played` | Total head-to-head meetings |
| `h2h_home_wins` | Home team wins in H2H |
| `h2h_away_wins` | Away team wins in H2H |
| `h2h_draws` | Draws in H2H |
| `h2h_home_win_rate` | Home win rate in H2H |
| `h2h_home_goals_avg` | Home avg goals in H2H |
| `h2h_away_goals_avg` | Away avg goals in H2H |
| `h2h_last_result` | Most recent H2H result |

### Elo Features (Step 10)

| Feature | Description |
|---------|-------------|
| `home_elo` / `away_elo` | Current Elo rating |
| `elo_diff` | Home minus away Elo |
| `elo_expected_home` | Expected home win prob from Elo |

### Player Impact Features (Steps 11, 15-16)

| Feature | Description |
|---------|-------------|
| `*_agg_goals` | Team total player goals |
| `*_agg_assists` | Team total assists |
| `*_agg_xg` | Team aggregated xG |
| `*_agg_tackles` | Team aggregated tackles |
| `adv_roll5_adv_progressive_pass_ratio` | Passing directness |
| `adv_roll5_adv_tackle_success_rate` | Defensive quality |
| `adv_roll5_adv_xg_per_shot` | Shot quality ratio |
| `adv_roll5_adv_goals_gini` | Squad goal dependency |
| `adv_roll5_adv_star_form_xg_xa` | Key player form |

### Referee Features (Step 12)

| Feature | Description |
|---------|-------------|
| `ref_home_win_rate` | Referee's home win tendency |
| `ref_avg_cards` | Referee's average cards per game |
| `ref_avg_fouls` | Referee's average fouls per game |
| `ref_strictness` | Card-to-foul ratio |

### GK Quality Features (Step 14)

| Feature | Description |
|---------|-------------|
| `*_gk_save_pct` | Save percentage |
| `*_gk_psxg_diff` | Goals vs post-shot xG (GK over/underperformance) |
| `*_gk_clean_sheet_rate` | Clean sheet frequency |

### Shot Quality Features (Step 15)

| Feature | Description |
|---------|-------------|
| `advshot_roll5_advshot_big_chance_ratio` | Big chance frequency |
| `advshot_roll5_advshot_conversion_rate` | Shot-to-goal conversion |
| `advshot_roll5_advshot_goals_minus_xg` | Finishing luck |
| `advshot_roll5_advshot_early_threat_ratio` | First-half shot share |
| `advshot_roll5_advshot_late_pressure_ratio` | Late-game intensity |

### League Position Features (Step 21)

| Feature | Description |
|---------|-------------|
| `home_league_position` / `away_league_position` | Current table position |
| `position_diff` | Home minus away position |
| `home_points` / `away_points` | Current season points |

### Manager Features (Step 22)

| Feature | Description |
|---------|-------------|
| `home_manager_tenure` / `away_manager_tenure` | Games managed (current stint) |
| `home_manager_new` | 1 if new manager (<5 games) |
| `home_manager_changed` | 1 if manager changed this season |
| `manager_h2h_*` | Manager vs manager historical record |

### Contextual Features (Steps 23-26)

| Feature | Description |
|---------|-------------|
| `home_congestion_7d` / `away_congestion_7d` | Matches in last 7 days |
| `home_congestion_14d` | Matches in last 14 days |
| `home_suspension_risk` / `away_suspension_risk` | Players near yellow card ban |
| `formation_*` | Formation analysis features |

### External Data Features (Steps 27-31)

| Feature | Description |
|---------|-------------|
| `league_draw_rate` | Season's draw frequency |
| `venue_capacity` | Stadium capacity |
| `is_big_stadium` | Stadium > 50,000 capacity |
| `temperature` | Match-day temperature |
| `precipitation` | Match-day rainfall (mm) |
| `wind_speed` | Match-day wind (km/h) |
| `odds_home` / `odds_draw` / `odds_away` | Historical betting odds |
| `implied_prob_*` | Bookmaker implied probabilities |
| `pinnacle_*` | Pinnacle sharp odds |
| `overround` | Total bookmaker margin |

### Injury & Pressing Features (Steps 32-33)

| Feature | Description |
|---------|-------------|
| `*_injury_impact` | Weighted injury severity |
| `*_injury_count` | Number of injured players |
| `*_ppda` | Passes per defensive action (pressing intensity) |
| `*_pressing_intensity` | PPDA-derived pressing score |

### Transfer Features (Step 35)

| Feature | Description |
|---------|-------------|
| `*_net_spend` | Transfer window net spend |
| `*_squad_value` | Total squad market value |
| `*_transfer_in_quality` | Quality of incoming transfers |

### Interaction Features (Step 36)

16 cross-signal interaction terms (8 original + 8 Phase 4):

| Feature | Description |
|---------|-------------|
| `interact_elo_x_form` | Elo rating * recent form |
| `interact_h2h_x_elo` | H2H dominance * current strength |
| `interact_rest_x_congestion` | Rest advantage * fixture density |
| `interact_xg_x_strength` | xG trend * team strength |

### Availability Flags

Binary indicators added by `DataLoader`:

| Flag | Description |
|------|-------------|
| `_has_player_agg` | 1 if FBref player aggregate data available |
| `_has_gk_data` | 1 if GK quality metrics available |
| `_has_shot_data` | 1 if shot quality data available |
| `_has_adv_player` | 1 if advanced player ratio features available |
| `_has_adv_shots` | 1 if advanced shot features available |
| `_has_odds` | 1 if betting odds data available |

## Target Variable

| Value | Meaning | Frequency |
|-------|---------|-----------|
| `H` | Home win | ~45% |
| `D` | Draw | ~26.5% |
| `A` | Away win | ~28.5% |

## Data Sources

| Source | Seasons | Data |
|--------|---------|------|
| FBref | 2017-2026 | Match reports, player stats, shots, lineups |
| football-data.co.uk | 2005-2026 | Results, betting odds |
| Understat | 2014-2026 | Expected goals (primary xG source for Serie A) |
| worldfootball.net | 2017-2026 | Referee assignments |
| Transfermarkt | 2020-2026 | Market values, transfers |
| Open-Meteo | 2017-2026 | Historical weather data |
| The Odds API | Live | Real-time bookmaker odds |

Note: FBref has **no xG data for Serie A** (unlike Premier League). All xG is sourced from Understat.
