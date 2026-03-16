"""Pre-Match Sentiment Analysis - Phase 4 Implementation.

Analyze pre-match news and social sentiment for prediction adjustment:

1. INJURY NEWS SENTIMENT
   - Key player injury impact
   - Team confidence when missing stars

2. TEAM MORALE INDICATORS
   - Recent controversy/drama
   - Managerial pressure
   - Dressing room issues

3. MOTIVATION SIGNALS
   - Title race tension
   - Relegation battle desperation
   - Cup/Europe focus split

4. PUBLIC PERCEPTION
   - Media expectations
   - Fan confidence
   - Historical sentiment patterns
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

log = logging.getLogger(__name__)


# =============================================================================
# KEYWORD-BASED SENTIMENT SCORING
# =============================================================================

# Positive sentiment keywords (boost team confidence)
POSITIVE_KEYWORDS = {
    "strong": ("high confidence", 0.1),
    "dominant": ("high confidence", 0.12),
    "winning streak": ("momentum", 0.08),
    "fit": ("positive injury news", 0.05),
    "recovered": ("positive injury news", 0.06),
    "returns": ("positive injury news", 0.07),
    "back in training": ("positive injury news", 0.05),
    "confident": ("mental boost", 0.06),
    "focused": ("mental boost", 0.04),
    "motivated": ("mental boost", 0.05),
    "revenge": ("motivation", 0.04),
    "determined": ("motivation", 0.05),
    "top form": ("form", 0.08),
    "in-form": ("form", 0.06),
    "unbeaten": ("momentum", 0.06),
    "title contenders": ("status", 0.05),
    "favorites": ("status", 0.04),
}

# Negative sentiment keywords (reduce team confidence)
NEGATIVE_KEYWORDS = {
    "injury": ("injury concern", -0.08),
    "injured": ("injury concern", -0.08),
    "doubt": ("uncertainty", -0.05),
    "doubtful": ("uncertainty", -0.06),
    "ruled out": ("confirmed absence", -0.10),
    "sidelined": ("confirmed absence", -0.09),
    "miss": ("absence", -0.07),
    "struggling": ("poor form", -0.08),
    "crisis": ("team turmoil", -0.12),
    "pressure": ("managerial pressure", -0.06),
    "sacked": ("managerial change", -0.10),
    "fired": ("managerial change", -0.10),
    "under fire": ("managerial pressure", -0.08),
    "controversy": ("distraction", -0.06),
    "dispute": ("internal issues", -0.05),
    "argument": ("internal issues", -0.05),
    "unhappy": ("morale issues", -0.06),
    "frustrated": ("morale issues", -0.05),
    "poor form": ("form", -0.08),
    "winless": ("momentum loss", -0.07),
    "relegation": ("pressure", -0.05),
    "must win": ("pressure", -0.04),
    "lacking confidence": ("mental", -0.08),
    "tired": ("fitness", -0.05),
    "fatigue": ("fitness", -0.06),
}

# Key player keywords (amplify sentiment if about star players)
KEY_PLAYER_PATTERNS = [
    r"captain",
    r"star\s+player",
    r"top\s+scorer",
    r"key\s+player",
    r"talisman",
    r"playmaker",
]


def analyze_headline_sentiment(
    headline: str,
    team: str,
) -> Dict[str, float]:
    """Analyze a single headline for sentiment.

    Returns:
        Dict with sentiment_score, category, and confidence.
    """
    headline_lower = headline.lower()

    # Check if headline is about the team
    team_variations = [
        team.lower(),
        team.lower().replace(" ", ""),
        team.lower()[:4],  # Abbreviated
    ]

    is_about_team = any(t in headline_lower for t in team_variations)

    if not is_about_team:
        return {
            "sentiment_score": 0.0,
            "category": "irrelevant",
            "confidence": 0.0,
        }

    total_score = 0.0
    categories = []

    # Check for positive keywords
    for keyword, (category, score) in POSITIVE_KEYWORDS.items():
        if keyword in headline_lower:
            total_score += score
            categories.append(category)

    # Check for negative keywords
    for keyword, (category, score) in NEGATIVE_KEYWORDS.items():
        if keyword in headline_lower:
            total_score += score  # Score is already negative
            categories.append(category)

    # Amplify if about key player
    is_key_player = any(re.search(p, headline_lower) for p in KEY_PLAYER_PATTERNS)
    if is_key_player:
        total_score *= 1.5

    # Determine main category
    if not categories:
        main_category = "neutral"
    else:
        main_category = categories[0]  # First matched category

    # Confidence based on keyword matches
    confidence = min(1.0, len(categories) * 0.3 + 0.1)

    return {
        "sentiment_score": max(-0.5, min(0.5, total_score)),
        "category": main_category,
        "confidence": confidence,
        "is_key_player_news": 1.0 if is_key_player else 0.0,
    }


def analyze_multiple_headlines(
    headlines: List[str],
    team: str,
) -> Dict[str, float]:
    """Analyze multiple headlines for aggregate sentiment."""
    if not headlines:
        return {
            "aggregate_sentiment": 0.0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "key_player_news_count": 0,
            "sentiment_confidence": 0.0,
        }

    scores = []
    positive = 0
    negative = 0
    neutral = 0
    key_player_news = 0

    for headline in headlines:
        result = analyze_headline_sentiment(headline, team)

        if result["category"] == "irrelevant":
            continue

        scores.append(result["sentiment_score"])

        if result["sentiment_score"] > 0.02:
            positive += 1
        elif result["sentiment_score"] < -0.02:
            negative += 1
        else:
            neutral += 1

        if result["is_key_player_news"]:
            key_player_news += 1

    if not scores:
        return {
            "aggregate_sentiment": 0.0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "key_player_news_count": 0,
            "sentiment_confidence": 0.0,
        }

    # Aggregate sentiment (average with recency weighting)
    # Recent headlines weighted more
    weights = [0.5 + 0.5 * (i / len(scores)) for i in range(len(scores))]
    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    weighted_avg = weighted_sum / sum(weights)

    # Confidence based on consistency
    if positive > 0 and negative > 0:
        confidence = 0.4  # Mixed signals
    else:
        confidence = 0.7 + 0.1 * min(len(scores), 3)  # More headlines = more confidence

    return {
        "aggregate_sentiment": weighted_avg,
        "positive_count": positive,
        "negative_count": negative,
        "neutral_count": neutral,
        "key_player_news_count": key_player_news,
        "sentiment_confidence": min(1.0, confidence),
    }


# =============================================================================
# TEAM CONTEXT ANALYSIS
# =============================================================================

def compute_motivation_factor(
    team: str,
    league_position: int,
    games_remaining: int = 10,
    is_cup_game: bool = False,
) -> Dict[str, float]:
    """Compute motivation factor based on team context.

    Teams fight harder when:
    - In title race (positions 1-3)
    - In relegation battle (positions 18-20)
    - Chasing European spots (positions 4-7)
    """
    motivation = 0.5  # Base motivation

    # Title race
    if league_position <= 3:
        motivation += 0.15
        if games_remaining <= 10:
            motivation += 0.10  # End of season intensity

    # Champions League spots
    elif league_position in [4, 5, 6]:
        motivation += 0.08
        if games_remaining <= 8:
            motivation += 0.07

    # Europa League spots
    elif league_position in [7, 8]:
        motivation += 0.05

    # Relegation battle
    elif league_position >= 18:
        motivation += 0.12
        if games_remaining <= 6:
            motivation += 0.15  # Survival mode

    # Conference League zone
    elif 15 <= league_position <= 17:
        motivation += 0.06  # Need to avoid drop

    # Mid-table (nothing to play for)
    elif 9 <= league_position <= 14 and games_remaining <= 5:
        motivation -= 0.05  # End of season letdown

    # Cup game modifier
    if is_cup_game:
        motivation += 0.03  # Knockout intensity

    return {
        "motivation_factor": min(1.0, max(0.3, motivation)),
        "is_title_race": 1.0 if league_position <= 3 else 0.0,
        "is_relegation_battle": 1.0 if league_position >= 18 else 0.0,
        "is_european_chase": 1.0 if 4 <= league_position <= 7 else 0.0,
    }


def compute_pressure_factor(
    team: str,
    recent_results: List[str],  # ['W', 'L', 'D', ...]
    manager_tenure_months: Optional[int] = None,
) -> Dict[str, float]:
    """Compute pressure factor affecting team performance.

    Pressure increases with:
    - Poor recent results
    - Short manager tenure after poor results
    - Fan/media expectations
    """
    pressure = 0.5  # Base pressure

    if recent_results:
        # Count recent losses
        recent_5 = recent_results[:5]
        losses = sum(1 for r in recent_5 if r == 'L')
        draws = sum(1 for r in recent_5 if r == 'D')

        # More losses = more pressure
        pressure += losses * 0.06
        pressure += draws * 0.02

        # Losing streak amplifies pressure
        losing_streak = 0
        for r in recent_5:
            if r == 'L':
                losing_streak += 1
            else:
                break
        pressure += losing_streak * 0.05

    # Manager pressure
    if manager_tenure_months is not None:
        # New managers get some leeway, but not much
        if manager_tenure_months < 3:
            pressure -= 0.05  # Honeymoon period
        elif manager_tenure_months < 6 and pressure > 0.6:
            pressure += 0.05  # Early pressure if results bad

    return {
        "pressure_factor": min(1.0, max(0.2, pressure)),
        "is_under_pressure": 1.0 if pressure > 0.7 else 0.0,
    }


# =============================================================================
# SENTIMENT AGGREGATOR
# =============================================================================

class TeamSentimentAnalyzer:
    """Aggregate all sentiment features for a team."""

    def __init__(self, data_dir: Path = None):
        if data_dir is None:
            from config.settings import DATA_DIR
            data_dir = DATA_DIR

        self.data_dir = data_dir
        self.cache_dir = data_dir / "cache" / "sentiment"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.headlines_cache: Dict[str, List[str]] = {}

    def load_cached_headlines(self, team: str) -> List[str]:
        """Load cached headlines for a team."""
        cache_path = self.cache_dir / f"{team.lower().replace(' ', '_')}.json"

        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    data = json.load(f)
                # Only use recent headlines (last 7 days)
                cutoff = (datetime.now() - timedelta(days=7)).isoformat()
                headlines = [
                    h["text"] for h in data.get("headlines", [])
                    if h.get("date", "") > cutoff
                ]
                return headlines
            except Exception as e:
                log.warning(f"Failed to load cached headlines for sentiment analysis: {e}")

        return []

    def analyze_team(
        self,
        team: str,
        league_position: int = 10,
        recent_results: List[str] = None,
        headlines: List[str] = None,
    ) -> Dict[str, float]:
        """Get full sentiment analysis for a team."""
        # Load headlines if not provided
        if headlines is None:
            headlines = self.load_cached_headlines(team)

        # Headline sentiment
        headline_sentiment = analyze_multiple_headlines(headlines, team)

        # Motivation factor
        motivation = compute_motivation_factor(team, league_position)

        # Pressure factor
        pressure = compute_pressure_factor(
            team,
            recent_results or [],
        )

        # Combine all factors
        result = {
            **headline_sentiment,
            **motivation,
            **pressure,
        }

        # Overall sentiment score
        # Combines headline sentiment + motivation - pressure effects
        overall = (
            0.4 * (headline_sentiment["aggregate_sentiment"] + 0.5) +  # Normalize to 0-1
            0.3 * motivation["motivation_factor"] +
            0.3 * (1 - pressure["pressure_factor"])  # Invert: high pressure = low sentiment
        )

        result["overall_sentiment"] = overall
        result["sentiment_boost"] = overall - 0.5  # Deviation from neutral

        return result


def get_match_sentiment_features(
    home_team: str,
    away_team: str,
    home_position: int = 10,
    away_position: int = 10,
    home_results: List[str] = None,
    away_results: List[str] = None,
) -> Dict[str, float]:
    """Get sentiment features for a match.

    Returns features for both teams and differential.
    """
    analyzer = TeamSentimentAnalyzer()

    home_sentiment = analyzer.analyze_team(
        home_team,
        league_position=home_position,
        recent_results=home_results,
    )

    away_sentiment = analyzer.analyze_team(
        away_team,
        league_position=away_position,
        recent_results=away_results,
    )

    # Compute differentials
    result = {}

    # Home team features
    for key, value in home_sentiment.items():
        result[f"home_{key}"] = value

    # Away team features
    for key, value in away_sentiment.items():
        result[f"away_{key}"] = value

    # Differentials
    result["sentiment_diff"] = (
        home_sentiment["overall_sentiment"] -
        away_sentiment["overall_sentiment"]
    )
    result["motivation_diff"] = (
        home_sentiment["motivation_factor"] -
        away_sentiment["motivation_factor"]
    )
    result["pressure_diff"] = (
        home_sentiment["pressure_factor"] -
        away_sentiment["pressure_factor"]
    )

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Sentiment Analysis - Phase 4")
    print("=" * 60)

    # Test headline analysis
    test_headlines = [
        "Inter captain returns to training ahead of derby",
        "Milan struggling with injury crisis, three key players ruled out",
        "Juventus confident of winning title after dominant display",
        "Roma manager under fire after third consecutive defeat",
    ]

    print("\nHeadline Analysis (Inter):")
    for headline in test_headlines:
        result = analyze_headline_sentiment(headline, "Inter")
        print(f"  '{headline[:50]}...'")
        print(f"    Score: {result['sentiment_score']:.2f}, Category: {result['category']}")

    # Test team sentiment
    print("\n" + "=" * 60)
    print("Team Sentiment Analysis:")

    features = get_match_sentiment_features(
        home_team="Inter",
        away_team="Milan",
        home_position=2,
        away_position=5,
        home_results=["W", "W", "D", "W", "L"],
        away_results=["L", "D", "L", "W", "D"],
    )

    print("\nInter vs Milan Sentiment Features:")
    for key, value in sorted(features.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
