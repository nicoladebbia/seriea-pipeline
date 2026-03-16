# Serie A Future Match Prediction System - Implementation Plan

## Problem Statement
The current system has ~50% accuracy (essentially random) because:
1. **Only showing PAST matches** - not predicting future games
2. **Heavy reliance on betting odds** (~40% of predictive power) - unavailable for future matches
3. **No injury/suspension data** - can't account for missing key players
4. **Underutilized player data** - 12,518 player-match rows with 132 columns not fully leveraged

## Goal
Build a system that predicts **upcoming Serie A matches** with improved accuracy by:
- Scraping upcoming fixtures from FBref
- Adding injury data from Transfermarkt/ESPN
- Tracking suspensions from yellow card accumulation
- Predicting likely lineups based on availability
- Using player-level form features (last 5-20 appearances)

---

## Phase 1: Injury Data Scraper

### New File: `scraper/injuries.py`

**Sources to scrape:**
- Transfermarkt: `https://www.transfermarkt.com/{team}/kader/verein/{id}` (injury section)
- ESPN: `https://www.espn.com/soccer/team/injuries/_/id/{espn_id}/league/ita.1`

**Data to collect:**
```python
@dataclass
class PlayerInjury:
    player_name: str
    team: str
    injury_type: str      # "Muscle", "Knee", "Illness"
    start_date: date
    expected_return: date | None
    is_currently_out: bool
```

**Key functions:**
- `scrape_team_injuries(team: str, season: str) -> list[PlayerInjury]`
- `get_injured_players(team: str, match_date: date) -> list[str]`

**Storage:** `data/external/injuries/{season}_injuries.parquet`

---

## Phase 2: Suspension Tracking

### New File: `features/suspensions.py`

**Use existing data:** `events.parquet` already has yellow/red card events with columns:
- `match_id`, `season`, `minute`, `event_type`, `team`, `player`
- `event_type` in: `['yellow_card', 'red_card', 'second_yellow']`

**Serie A suspension rules:**
- 5 yellow cards = 1 match ban
- 10 yellow cards = 2 match ban
- Red card = 1+ match ban
- Second yellow = 1 match ban

**Key functions:**
```python
def get_yellow_card_count(player: str, team: str, season: str, before_date: date) -> int
def get_suspended_players(team: str, match_date: date) -> list[str]
def compute_suspension_features(match_df: pd.DataFrame) -> pd.DataFrame
```

**Features to generate:**
- `home_suspended_count` / `away_suspended_count`
- `home_key_player_suspended` (is top-5 minutes player out?)
- `home_players_at_4_yellows` (suspension risk)

---

## Phase 3: Lineup Prediction

### New File: `features/lineup_prediction.py`

**Data sources:**
- `lineups.parquet`: 17,697 rows with formation, starters, bench
- `player_stats.parquet`: Minutes played, performance per player
- Injury data (Phase 1)
- Suspension data (Phase 2)

**Algorithm:**
1. Get last 5 lineups for the team
2. Count starter frequency per player
3. Remove injured/suspended players
4. Score remaining players: `frequency * 0.5 + recent_minutes * 0.3 + form * 0.2`
5. Select top 11 by score
6. Infer formation from most common recent formation

**Output:**
```python
@dataclass
class PredictedLineup:
    team: str
    formation: str           # "4-3-3"
    predicted_starters: list[str]
    missing_regulars: list[str]  # Due to injury/suspension
    confidence: float        # 0-1
```

---

## Phase 4: Player Form Features

### New File: `features/player_form.py`

**Use existing:** `player_stats.parquet` (132 columns per player per match)

**Per-player rolling metrics (last 5/10 appearances):**
- xG + xA per 90 minutes
- Goals scored
- Shot conversion rate
- Pass completion rate
- Defensive actions (tackles + interceptions)
- Minutes played (fitness indicator)

**Team-level aggregation:**
```python
def compute_squad_form(
    predicted_lineup: list[str],  # 11 players
    match_date: date
) -> dict:
    return {
        "squad_avg_xg_xa": float,      # Average form of starting 11
        "star_player_form": float,      # Top 3 contributors' form
        "squad_form_variance": float,   # How consistent is team form
    }
```

---

## Phase 5: Upcoming Fixtures Integration

### Modify: `scraper/fixtures.py`

FBref fixtures page includes unplayed matches (empty scores). Add:

```python
def get_upcoming_matches(season: str, days_ahead: int = 7) -> pd.DataFrame:
    """Return matches with no score within next N days."""
    df = scrape_season_fixtures(season)
    today = date.today()
    return df[
        (df["match_date"] >= str(today)) &
        (df["home_score"].isna())
    ]
```

### Modify: `web/predictor.py`

Add function to build features for unplayed matches:

```python
def build_upcoming_match_features(
    home_team: str,
    away_team: str,
    match_date: date,
    injuries: dict,
    suspensions: dict,
) -> pd.DataFrame:
    """
    Build features using only data available BEFORE the match:
    - Team rolling stats (from features.parquet historical)
    - Elo ratings
    - H2H records
    - Predicted lineup quality
    - Injury/suspension impact
    - NO betting odds (not available)
    """
```

---

## Phase 6: Predictive Model (No Odds)

### Modify: `ml/training.py`

Create separate model for future predictions:

```python
def train_predictive_model() -> Dict:
    """
    Train model specifically for predicting future matches.

    Key differences from current model:
    - exclude_odds=True (critical - odds unavailable for future)
    - Include injury impact features
    - Include lineup quality features
    - Include suspension features
    """
    return train_optimized(
        exclude_odds=True,
        # Add new feature categories
    )
```

**Feature priorities (without odds):**
1. Elo ratings & strength ratings
2. Rolling form (goals, xG, points) - last 5/10 matches
3. H2H historical records
4. Player availability (injuries, suspensions)
5. Predicted lineup quality
6. Squad market value
7. Fixture congestion
8. Home advantage

---

## Phase 7: Web Interface for Future Predictions

### New Template: `web/templates/upcoming.html`

Display for each upcoming match:
1. **Match info**: Date, time, teams, venue
2. **Prediction**: H/D/A probabilities with confidence bar
3. **Predicted lineups**: Side-by-side formations
4. **Key absences**: Injured/suspended players highlighted
5. **Key factors**: Top 5 reasons for prediction

### Modify: `web/app.py`

Add routes:
- `GET /upcoming` - Page with upcoming match predictions
- `GET /api/upcoming` - JSON API for upcoming matches with predictions

---

## Implementation Order

### Week 1: Data Collection
1. Create `scraper/injuries.py` - Transfermarkt injury scraper
2. Create `features/suspensions.py` - Yellow card tracking
3. Add `get_upcoming_matches()` to fixtures.py
4. Add CLI commands: `fetch-injuries`, `show-upcoming`

### Week 2: Feature Engineering
1. Create `features/player_form.py` - Per-player rolling stats
2. Create `features/lineup_prediction.py` - Predict starting XI
3. Integrate new features into `features/build.py`

### Week 3: Model & Web
1. Train "predictive" model without odds
2. Create `upcoming.html` template
3. Add `/upcoming` and `/api/upcoming` routes
4. Backtest on recent matches

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `scraper/injuries.py` | CREATE | Scrape injury data from Transfermarkt |
| `features/suspensions.py` | CREATE | Track yellow cards, compute suspensions |
| `features/lineup_prediction.py` | CREATE | Predict starting XI |
| `features/player_form.py` | CREATE | Per-player rolling form metrics |
| `scraper/fixtures.py` | MODIFY | Add `get_upcoming_matches()` |
| `web/predictor.py` | MODIFY | Add `build_upcoming_match_features()` |
| `ml/training.py` | MODIFY | Add `train_predictive_model()` |
| `web/app.py` | MODIFY | Add `/upcoming` routes |
| `web/templates/upcoming.html` | CREATE | Upcoming predictions UI |

---

## Verification

1. **Unit tests** for each new module
2. **Backtest**: Predict 2024-25 matches pretending we don't know results
3. **Accuracy target**: 58-62% without odds (vs 50% current)
4. **Live test**: Run predictions for next matchweek, compare to actual results

---

## Expected Outcome

| Metric | Current | Target |
|--------|---------|--------|
| Future match prediction | Not possible | Working |
| Accuracy (without odds) | ~45% | 58-62% |
| Injury data | None | Full coverage |
| Lineup prediction | None | ~80% starter accuracy |
| Key absence alerts | None | Automated |
