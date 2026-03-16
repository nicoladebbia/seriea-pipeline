# 🏆 Serie A Prediction System - Complete Summary

## Overview

A professional-grade football prediction system validated across **21 seasons** (2005-2026) of Serie A data, achieving up to **95.8% accuracy** when multiple factors align.

---

## 📊 Prediction Tiers

### Tier S+: Near-Perfect (95%+)
| Prediction | Accuracy | Sample Size |
|------------|----------|-------------|
| 5-Factor Perfect Storm | **100%** | 14 matches |
| 3-Factor Best Combo | **95.7%** | 23 matches |
| Over 8.5 Shots on Target | 99.9% | 1749 matches |
| At Least 1 Yellow Card | 98.3% | 1749 matches |

### Tier S: Excellent (85-95%)
| Prediction | Accuracy | Sample Size |
|------------|----------|-------------|
| Big home fav + Home fav ref | **91.5%** | 47 matches |
| Over 0.5 Goals | 93.0% | 1749 matches |
| Home Scores @ 85% conf | 88.8% | 439 matches |
| Over 2.5 Yellows @ 85% conf | 87.0% | 859 matches |

### Tier A: Strong (75-85%)
| Prediction | Accuracy | Sample Size |
|------------|----------|-------------|
| Hot home + Big home fav | **80.8%** | 224 matches |
| Big stadium + Big home fav | **79.9%** | 279 matches |
| 4-Factor Card Storm | 80.0% | 20 matches |
| Perfect Upset (4 factors) | 75.0% | 24 matches |

### Tier B: Good (65-75%)
| Prediction | Accuracy | Sample Size |
|------------|----------|-------------|
| Big stadium + Home favorite | **71.8%** | 967 matches |
| H/D/A @ 65% confidence | 71.8% | 223 matches |
| 3+ Negative Factors → Home | **69.4%** | 863 matches |
| Hot home + Cold away | 68.5% | 314 matches |

---

## 🔑 Validated Factors (21 Seasons)

### HIGH Confidence Factors
| Factor | Effect | Consistency |
|--------|--------|-------------|
| Big Stadium (50k+) | +15% home win | 95% of seasons |
| Home Favorite (Elo +100) | +12% home win | 95% of seasons |
| Big Home Favorite (Elo +200) | +20% home win | 85% of seasons |
| Hot Home Form (10+ pts/5 games) | +8% home win | 86% of seasons |
| Cold Away Form (≤4 pts/5 games) | +6% home win | 86% of seasons |
| Derby Match | +8% draw, +10% cards | 90% of seasons |

### MEDIUM Confidence Factors (8 Seasons)
| Factor | Effect |
|--------|--------|
| Home-Favoring Referee | +12% home win |
| Away-Favoring Referee | -10% home win |
| Strict Referee | +15% high cards |
| Rainy Weather | +3% home win, +0.2 goals |
| Cold Weather (<10°C) | +5% cards |

---

## 📈 Progressive Factor Stacking

| Factors Aligned | Home Win Rate | Lift | Sample |
|-----------------|---------------|------|--------|
| Base | 44.4% | - | 7829 |
| 1+ | 48.8% | +4.4% | 6667 |
| 2+ | 55.0% | +10.6% | 3887 |
| **3+** | **64.7%** | +20.4% | 1350 |
| **4+** | **72.7%** | +28.4% | 264 |
| **5+** | **95.8%** | +51.5% | 24 |

---

## 👤 Referee Analysis

### Home-Favoring Referees
| Referee | Home Win Rate | Sample |
|---------|---------------|--------|
| Federico Dionisi | 58.1% (+16.6%) | 31 |
| Antonio Giua | 56.6% (+15.1%) | 53 |
| Simone Sozza | 54.5% (+13.0%) | 44 |

### Away-Favoring Referees
| Referee | Home Win Rate | Sample |
|---------|---------------|--------|
| Paolo Mazzoleni | 28.6% (-12.9%) | 35 |
| Daniele Doveri | 31.1% (-10.4%) | 119 |
| Marco Di Bello | 35.0% (-6.6%) | 103 |

### Strict Referees (Cards)
| Referee | Avg Cards | High Card Rate |
|---------|-----------|----------------|
| Strict refs | 5.04 | 58.3% |
| Lenient refs | 4.15 | 43.5% |

---

## 🌤️ Weather Impact

| Condition | Effect |
|-----------|--------|
| Rain (>5mm) | +0.20 goals, +3.2% home win |
| Wind (>30 km/h) | -0.22 goals |
| Cold (<10°C) | +5% cards |
| Hot (>25°C) | Minimal impact |

---

## ⏱️ Goal Timing Patterns

| Period | Goals | Percentage |
|--------|-------|------------|
| 0-15 min | 109 | 11.6% |
| 15-30 min | 125 | 13.4% |
| 30-45 min | 156 | 16.7% |
| 45-60 min | 161 | 17.2% |
| **60-75 min** | **168** | **17.9%** (Peak) |
| 75-90 min | 145 | 15.5% |

**Second Half: 50.6%** vs First Half: 41.7%

---

## 🔥 Perfect Storm Scenarios

### Perfect Home Storm
```
Hot home + Cold away + Big stadium + Home favorite + Fav ref
= 100% HOME WIN (14 matches)
= 78.6% DOMINANT WIN
```

### Perfect Card Storm
```
Strict ref + Derby + Cold + Big stadium
= 80% HIGH CARDS (5+)
= 6.5 avg cards (vs 4.5 base)
```

### Perfect Upset Scenario
```
Hot away + Cold home + Away favorite + Away-fav ref
= 75% AWAY WIN (24 matches)
```

---

## 📁 System Files

### Production System (REAL-TIME)
| Script | Purpose |
|--------|---------|
| `scripts/run_predictions.py` | **MAIN ENTRY POINT** - Run full prediction pipeline |
| `scripts/realtime_prediction_engine.py` | Core prediction engine |
| `scripts/fetch_upcoming_matches.py` | Fetch/load upcoming matches |
| `scripts/weather_integration.py` | Real weather data via Open-Meteo API |
| `scripts/current_form_calculator.py` | Calculate current team form |

### Analysis Scripts (Historical)
| Script | Purpose |
|--------|---------|
| `scripts/predict_matchday.py` | Legacy prediction system |
| `scripts/train_ultimate_stacking.py` | Factor stacking analysis |
| `scripts/train_advanced_predictions.py` | Referee/Weather/Timing analysis |
| `scripts/train_advanced_storylines.py` | Football myths validation |
| `scripts/train_high_base_rate.py` | High base rate markets |

---

## 🎯 Usage

### Production Predictions (REAL)
```bash
# Full predictions with all details
python3 scripts/run_predictions.py

# Brief output (high confidence only)
python3 scripts/run_predictions.py --brief

# JSON output for automation
python3 scripts/run_predictions.py --json
```

### Historical Analysis
```bash
# Analyze factor stacking
python3 scripts/train_ultimate_stacking.py

# Run all advanced analysis
python3 scripts/train_advanced_predictions.py
```

---

## 📡 Data Pipeline

1. **Load Matches**: `data/upcoming/manual_matches.json` (or scraped)
2. **Calculate Form**: Uses last 10 matchweeks from `data/features/features.parquet`
3. **Fetch Weather**: Real-time from Open-Meteo API
4. **Apply Factors**: 21-season validated factor stacking
5. **Output**: `data/upcoming/predictions.json`

---

## ⚠️ Important Notes

1. **Robustness Validated**: Core factors tested across 21 seasons (2005-2026)
2. **Referee factors**: 8 seasons only - use with medium confidence
3. **Formation data**: 1 season only - use with low confidence
4. **Sample sizes matter**: High-accuracy combos often have small samples
5. **Always check confidence level**: HIGH = 3+ validated factors

---

## 💰 Betting Strategy

### High-Confidence Bets (Use these!)
1. **5+ factors aligned** → Home win (95.8%)
2. **Big home fav + Home-fav ref** → Home win (91.5%)
3. **Strict ref + Derby** → Over 4.5 cards (63.6%)
4. **4+ card factors** → Over 5.5 cards (80%)

### Volume Plays (More opportunities)
1. **3+ home factors** → Home win 64.7% (1350 matches)
2. **Hot home + Big home fav** → 80.8% (224 matches)
3. **3+ negative factors** → Home win 69.4% (863 matches)

---

*Generated: 2026-02-04*
*Data: 7,829 Serie A matches (2005-2026)*
*Validated across 21 seasons*
