#!/usr/bin/env python3
"""WEB SEARCH SENTIMENT ANALYZER - Advanced Market Intelligence

Uses Google Gemini (primary) and Groq Compound (fallback) for real-time
sentiment analysis of Serie A matches via grounded web search.

Features:
- Fan sentiment analysis from social media
- Media hype detection
- Injury impact assessment
- Transfer buzz monitoring
- Contrarian opportunity identification
- Italian-specific sentiment patterns

Sentiment Metrics:
- Fan Confidence: 0-100 (team morale)
- Media Hype: -50 to +50 (media bias)
- Injury Impact: -100 to 0 (negative)
- Transfer Buzz: 0 to +50 (positive)
- Derby Pressure: -25 to +25 (contextual)

Usage:
    python sentiment_analyzer.py --match "Lecce vs Udinese"
    python sentiment_analyzer.py --all  # Analyze all upcoming matches
"""

import os
import sys
import json
import time
import hashlib
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR, PROJECT_ROOT

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from google import genai
    from google.genai import types as genai_types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

# =============================================================================
# CONFIGURATION
# =============================================================================

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass  # dotenv not required if env vars set externally

# Web Search API Configuration (Gemini primary, Groq fallback)
GOOGLE_GEMINI_KEY = os.environ.get("GOOGLE_GEMINI_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
# Legacy — kept for backwards compatibility
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")

HAS_WEB_SEARCH_KEY = bool(GOOGLE_GEMINI_KEY) or bool(GROQ_API_KEY)

if not HAS_WEB_SEARCH_KEY:
    logging.warning("No web search API key set (GOOGLE_GEMINI_KEY / GROQ_API_KEY) - sentiment will use fallback data")

# Cache settings
CACHE_DIR = DATA_DIR / "cache" / "sentiment"
CACHE_DURATION_HOURS = 6  # Sentiment is time-sensitive

# API Rate Limits (free tiers)
# Gemini: 5,000 grounded queries/month free. Reserve 10% buffer.
GEMINI_MONTHLY_LIMIT = 4500
# Groq compound-beta: ~1,000 RPD free tier. Be conservative.
GROQ_DAILY_LIMIT = 800
# Usage tracking file
API_USAGE_FILE = DATA_DIR / "cache" / "api_usage.json"


class APIUsageTracker:
    """Tracks API call counts to prevent hitting free tier limits.

    Persists to disk so usage survives restarts. Auto-resets daily/monthly.
    """

    def __init__(self, usage_file: Path = API_USAGE_FILE):
        self._file = usage_file
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._usage = self._load()

    def _load(self) -> dict:
        try:
            if self._file.exists():
                with open(self._file) as f:
                    return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
        return {}

    def _save(self):
        try:
            with open(self._file, "w") as f:
                json.dump(self._usage, f, indent=2)
        except OSError as e:
            log.warning(f"Failed to save API usage: {e}")

    def _today(self) -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def _month(self) -> str:
        return datetime.now().strftime("%Y-%m")

    def record_call(self, provider: str):
        """Record one API call for a provider."""
        today = self._today()
        month = self._month()

        if provider not in self._usage:
            self._usage[provider] = {}
        p = self._usage[provider]

        # Daily counter (auto-resets)
        if p.get("current_day") != today:
            p["current_day"] = today
            p["daily_calls"] = 0
        p["daily_calls"] = p.get("daily_calls", 0) + 1

        # Monthly counter (auto-resets)
        if p.get("current_month") != month:
            p["current_month"] = month
            p["monthly_calls"] = 0
        p["monthly_calls"] = p.get("monthly_calls", 0) + 1

        self._save()

    def can_call(self, provider: str) -> bool:
        """Check if we're within rate limits for this provider."""
        today = self._today()
        month = self._month()
        p = self._usage.get(provider, {})

        if provider == "gemini":
            monthly = p.get("monthly_calls", 0) if p.get("current_month") == month else 0
            if monthly >= GEMINI_MONTHLY_LIMIT:
                return False
        elif provider == "groq":
            daily = p.get("daily_calls", 0) if p.get("current_day") == today else 0
            if daily >= GROQ_DAILY_LIMIT:
                return False
        return True

    def get_status(self) -> dict:
        """Return usage summary for health monitoring."""
        today = self._today()
        month = self._month()
        status = {}
        for provider in ("gemini", "groq"):
            p = self._usage.get(provider, {})
            if provider == "gemini":
                used = p.get("monthly_calls", 0) if p.get("current_month") == month else 0
                limit = GEMINI_MONTHLY_LIMIT
                status[provider] = {
                    "used": used, "limit": limit, "period": "monthly",
                    "remaining": max(0, limit - used),
                    "pct_used": round(used / limit * 100, 1) if limit else 0,
                }
            elif provider == "groq":
                used = p.get("daily_calls", 0) if p.get("current_day") == today else 0
                limit = GROQ_DAILY_LIMIT
                status[provider] = {
                    "used": used, "limit": limit, "period": "daily",
                    "remaining": max(0, limit - used),
                    "pct_used": round(used / limit * 100, 1) if limit else 0,
                }
        return status

# Sentiment scoring weights
SENTIMENT_WEIGHTS = {
    "fan_confidence": 0.35,
    "media_hype": 0.25,
    "injury_impact": 0.20,
    "transfer_buzz": 0.10,
    "derby_pressure": 0.10,
}

# Italian football context
ITALIAN_RIVALRIES = {
    ("Juventus", "Inter"): "Derby d'Italia",
    ("Inter", "AC Milan"): "Derby della Madonnina",
    ("Roma", "Lazio"): "Derby della Capitale",
    ("Napoli", "Roma"): "Derby del Sole",
    ("Genoa", "Sampdoria"): "Derby della Lanterna",
    ("Torino", "Juventus"): "Derby della Mole",
    ("Fiorentina", "Juventus"): "Historic Rivalry",
}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SentimentScore:
    """Individual sentiment metric."""
    score: float  # -100 to +100
    confidence: float  # 0 to 1
    sources: List[str]
    reasoning: str


@dataclass
class MatchSentiment:
    """Complete sentiment analysis for a match."""
    match: str
    date: str
    home_team: str
    away_team: str

    # Individual metrics
    home_fan_confidence: float  # 0-100
    away_fan_confidence: float  # 0-100
    media_hype_home: float  # -50 to +50
    media_hype_away: float  # -50 to +50
    home_injury_impact: float  # -100 to 0
    away_injury_impact: float  # -100 to 0
    home_transfer_buzz: float  # 0 to +50
    away_transfer_buzz: float  # 0 to +50
    derby_pressure: float  # -25 to +25

    # Composite scores
    home_composite: float  # -100 to +100
    away_composite: float  # -100 to +100
    sentiment_edge: str  # "home", "away", "neutral"

    # Analysis
    key_factors: List[str]
    contrarian_opportunity: bool
    reasoning: str

    # Metadata
    generated_at: str
    sources_used: List[str]


# =============================================================================
# WEB SEARCH API CLIENTS (Gemini primary, Groq fallback)
# =============================================================================

# Shared usage tracker (singleton)
_usage_tracker = APIUsageTracker()


class GeminiClient:
    """Client for Google Gemini API with grounded web search."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or GOOGLE_GEMINI_KEY
        self._client = None
        if HAS_GEMINI and self.api_key:
            self._client = genai.Client(api_key=self.api_key)

    def query(self, prompt: str, system_prompt: str = None) -> Optional[str]:
        """Send query to Gemini API with Google Search grounding."""
        if not self._client:
            log.debug("Gemini client not available")
            return None

        if not _usage_tracker.can_call("gemini"):
            status = _usage_tracker.get_status().get("gemini", {})
            log.warning(
                f"Gemini monthly limit reached ({status.get('used', '?')}/{status.get('limit', '?')}). "
                "Skipping to avoid overage."
            )
            return None

        try:
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response = self._client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=1000,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                    tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
                ),
            )
            _usage_tracker.record_call("gemini")
            return response.text
        except Exception as e:
            log.warning(f"Gemini API error: {e}")
            return None


class GroqClient:
    """Client for Groq Compound Beta API with built-in web search."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or GROQ_API_KEY
        self._client = None
        if HAS_GROQ and self.api_key:
            self._client = Groq(api_key=self.api_key)

    def query(self, prompt: str, system_prompt: str = None) -> Optional[str]:
        """Send query to Groq Compound Beta API."""
        if not self._client:
            log.debug("Groq client not available")
            return None

        if not _usage_tracker.can_call("groq"):
            status = _usage_tracker.get_status().get("groq", {})
            log.warning(
                f"Groq daily limit reached ({status.get('used', '?')}/{status.get('limit', '?')}). "
                "Skipping to avoid overage."
            )
            return None

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model="compound-beta",
                messages=messages,
                max_tokens=1000,
                temperature=0.2,
            )
            _usage_tracker.record_call("groq")
            return response.choices[0].message.content
        except Exception as e:
            log.warning(f"Groq API error: {e}")
            return None


# =============================================================================
# SENTIMENT ANALYSIS
# =============================================================================

class DataDrivenSentiment:
    """Compute sentiment from pipeline data (standings, form, H2H, derbies).

    Used when Perplexity API is unavailable. Produces meaningful per-team
    scores instead of 50/50 defaults.
    """

    def __init__(self):
        self._standings = None
        self._form = None
        self._load_data()

    def _load_data(self):
        """Load standings and form from upstream data files."""
        standings_path = DATA_DIR / "upcoming" / "standings.json"
        form_path = DATA_DIR / "upcoming" / "current_form.json"

        if standings_path.exists():
            with open(standings_path) as f:
                data = json.load(f)
            self._standings = data.get("standings", {})

        if form_path.exists():
            with open(form_path) as f:
                data = json.load(f)
            self._form_teams = data.get("teams", {})
            self._form_matchups = data.get("matchups", {})
        else:
            self._form_teams = {}
            self._form_matchups = {}

    def fan_confidence(self, team: str) -> tuple[float, str]:
        """Derive fan confidence from form and standings.

        Factors: recent form string, ppg, position, form_status.
        Scale: 0-100 where 50=neutral.
        """
        st = self._standings.get(team, {}) if self._standings else {}
        ft = self._form_teams.get(team, {})

        if not st and not ft:
            return 50.0, f"No data available for {team}"

        score = 50.0
        reasons = []

        # Form string (last 5): W=+8, D=+1, L=-6
        form_str = st.get("form_last5", "")
        form_status = ft.get("form_status", "normal")
        if form_str:
            w = form_str.count("W")
            d = form_str.count("D")
            l = form_str.count("L")
            form_pts = w * 8 + d * 1 + l * (-6)
            score += form_pts
            reasons.append(f"Last 5: {form_str} ({w}W {d}D {l}L)")

        # Position bonus/penalty
        pos = st.get("position", 10)
        if pos <= 4:
            score += 10
            reasons.append(f"Champions League zone ({pos}th)")
        elif pos <= 6:
            score += 5
            reasons.append(f"Europa League zone ({pos}th)")
        elif pos >= 18:
            score -= 12
            reasons.append(f"Relegation zone ({pos}th)")
        elif pos >= 15:
            score -= 5
            reasons.append(f"Lower table ({pos}th)")

        # PPG trend
        ppg = ft.get("ppg", 0)
        if ppg >= 2.4:
            score += 8
            reasons.append(f"Excellent run ({ppg:.1f} ppg)")
        elif ppg >= 2.0:
            score += 4
            reasons.append(f"Good run ({ppg:.1f} ppg)")
        elif ppg <= 0.8:
            score -= 8
            reasons.append(f"Poor run ({ppg:.1f} ppg)")
        elif ppg <= 1.2:
            score -= 3
            reasons.append(f"Below average run ({ppg:.1f} ppg)")

        # Form status (hot/normal/cold)
        if form_status == "hot":
            score += 5
            reasons.append("Hot streak")
        elif form_status == "cold":
            score -= 5
            reasons.append("Cold streak")

        # Clamp to 0-100
        score = max(10, min(95, score))
        return score, "; ".join(reasons) if reasons else "Average form"

    def media_hype(self, home: str, away: str) -> tuple[float, float, str]:
        """Derive media hype from position gap and elo difference.

        Top teams get positive media, struggling teams get negative.
        """
        h_st = self._standings.get(home, {}) if self._standings else {}
        a_st = self._standings.get(away, {}) if self._standings else {}
        h_ft = self._form_teams.get(home, {})
        a_ft = self._form_teams.get(away, {})

        h_pos = h_st.get("position", 10)
        a_pos = a_st.get("position", 10)
        h_elo = h_ft.get("elo", 1500)
        a_elo = a_ft.get("elo", 1500)

        # Higher elo / better position = more positive media
        h_hype = (1500 - h_elo) / -30  # Elo diff from average
        a_hype = (1500 - a_elo) / -30

        # Position factor
        h_hype += (10 - h_pos) * 2
        a_hype += (10 - a_pos) * 2

        # Clamp to -50 to +50
        h_hype = max(-40, min(40, h_hype))
        a_hype = max(-40, min(40, a_hype))

        reason = f"{home} ({h_pos}th, Elo {h_elo:.0f}) vs {away} ({a_pos}th, Elo {a_elo:.0f})"
        return round(h_hype, 1), round(a_hype, 1), reason

    def injury_impact(self, team: str) -> tuple[float, str]:
        """Estimate injury impact from goal difference trend.

        Without actual injury data, we use GD and recent form as a proxy
        for squad fitness / depth.
        """
        st = self._standings.get(team, {}) if self._standings else {}
        ft = self._form_teams.get(team, {})

        if not st:
            return -10.0, f"No standings data for {team}"

        gd = st.get("gd", 0)
        played = st.get("played", 1)
        gd_per_game = gd / max(played, 1)
        gc_per_game = st.get("ga", 0) / max(played, 1)

        # Higher goals conceded = worse squad situation
        if gc_per_game >= 2.0:
            impact = -55
            reason = f"Leaking goals ({gc_per_game:.1f}/game)"
        elif gc_per_game >= 1.5:
            impact = -35
            reason = f"Defensive concerns ({gc_per_game:.1f} GA/game)"
        elif gc_per_game <= 0.6:
            impact = -5
            reason = f"Solid defense ({gc_per_game:.1f} GA/game)"
        elif gc_per_game <= 1.0:
            impact = -15
            reason = f"Good defense ({gc_per_game:.1f} GA/game)"
        else:
            impact = -25
            reason = f"Average defense ({gc_per_game:.1f} GA/game)"

        return float(impact), reason

    def transfer_buzz(self, team: str) -> tuple[float, str]:
        """Estimate transfer morale from current position vs expectations.

        Mid-season window closed (Feb) — transfer buzz is based on recent
        signings working out (approximated by form improvement).
        """
        ft = self._form_teams.get(team, {})
        ppg = ft.get("ppg", 1.3)
        form_status = ft.get("form_status", "normal")

        if form_status == "hot" and ppg >= 2.0:
            return 35.0, "Squad gelling well, positive momentum"
        elif form_status == "hot":
            return 28.0, "Good squad chemistry"
        elif form_status == "cold" and ppg <= 1.0:
            return 5.0, "Squad struggling, possible unrest"
        elif form_status == "cold":
            return 10.0, "Mixed signals from squad"
        else:
            return 18.0, "Stable squad situation"


class SentimentAnalyzer:
    """Analyzes sentiment for football matches (Serie A + Premier League).

    Uses Gemini Grounding (primary) or Groq Compound (fallback) for web-search
    sentiment, falls back to data-driven analysis from standings, form, H2H.
    """

    def __init__(self, league: str = "serie_a"):
        # Web search client cascade: Gemini -> Groq
        self._gemini = GeminiClient()
        self._groq = GroqClient()
        if self._gemini._client:
            self.client = self._gemini
        elif self._groq._client:
            self.client = self._groq
        else:
            self.client = GeminiClient()  # Dummy — will return None
        self._use_web_search = HAS_WEB_SEARCH_KEY
        self._data_driven = DataDrivenSentiment()
        self._league = league
        try:
            from config.leagues import LEAGUE_REGISTRY
            self._league_label = LEAGUE_REGISTRY[league].name if league in LEAGUE_REGISTRY else league
        except Exception:
            self._league_label = league
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, match: str, date: str) -> str:
        """Generate cache key for a match."""
        key = f"{match}_{date}"
        return hashlib.md5(key.encode()).hexdigest()

    def _load_from_cache(self, match: str, date: str) -> Optional[MatchSentiment]:
        """Load sentiment from cache if valid."""
        cache_key = self._get_cache_key(match, date)
        cache_file = CACHE_DIR / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file) as f:
                data = json.load(f)

            # Check if cache is still valid
            cached_time = datetime.fromisoformat(data["generated_at"])
            if datetime.now() - cached_time > timedelta(hours=CACHE_DURATION_HOURS):
                return None

            return MatchSentiment(**data)

        except Exception as e:
            log.warning(f"Cache load error: {e}")
            return None

    def _save_to_cache(self, sentiment: MatchSentiment):
        """Save sentiment to cache."""
        cache_key = self._get_cache_key(sentiment.match, sentiment.date)
        cache_file = CACHE_DIR / f"{cache_key}.json"

        try:
            with open(cache_file, "w") as f:
                json.dump(asdict(sentiment), f, indent=2)
        except Exception as e:
            log.warning(f"Cache save error: {e}")

    def analyze_fan_sentiment(self, team: str) -> Tuple[float, str]:
        """Analyze fan sentiment for a team."""
        prompt = f"""Analyze the current fan sentiment for {team} ({self._league_label} football club).

Search for recent social media discussions, fan forums, and supporter reactions.

Provide a sentiment score from 0-100 where:
- 0-30: Very negative (fans frustrated, protests, boycotts)
- 31-50: Negative (disappointed, concerned)
- 51-70: Neutral to slightly positive
- 71-85: Positive (optimistic, confident)
- 86-100: Very positive (euphoric, high expectations)

Format your response as JSON:
{{
    "score": <number 0-100>,
    "reasoning": "<brief explanation of fan sentiment>",
    "key_topics": ["<topic1>", "<topic2>"]
}}"""

        system_prompt = """You are a football sentiment analyst covering the {self._league_label}.
Focus on fan communities, supporter sentiment, and recent news.
Be objective and base your analysis on actual recent discussions and news."""

        response = self.client.query(prompt, system_prompt)
        if not response:
            return 50.0, "Unable to analyze fan sentiment"

        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return float(data.get("score", 50)), data.get("reasoning", "")
        except (json.JSONDecodeError, ValueError) as e:
            log.debug(f"Failed to parse sentiment analysis response: {e}")

        return 50.0, response[:200]

    def analyze_media_hype(self, home_team: str, away_team: str) -> Tuple[float, float, str]:
        """Analyze media hype and bias for both teams."""
        prompt = f"""Analyze Italian sports media coverage of the upcoming {home_team} vs {away_team} match.

Search Gazzetta dello Sport, Corriere dello Sport, Tuttosport, and Sky Sport Italy.

Determine if media is hyping one team over the other.

Provide scores from -50 to +50 where:
- Negative: Media being critical/negative
- 0: Neutral coverage
- Positive: Media hyping/favoring the team

Format your response as JSON:
{{
    "home_hype": <number -50 to +50>,
    "away_hype": <number -50 to +50>,
    "reasoning": "<brief explanation of media coverage>",
    "key_narratives": ["<narrative1>", "<narrative2>"]
}}"""

        system_prompt = """You are an Italian football media analyst.
Focus on major Italian sports newspapers and TV coverage.
Identify media narratives and biases objectively."""

        response = self.client.query(prompt, system_prompt)
        if not response:
            return 0.0, 0.0, "Unable to analyze media hype"

        try:
            import re
            json_match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return (
                    float(data.get("home_hype", 0)),
                    float(data.get("away_hype", 0)),
                    data.get("reasoning", "")
                )
        except (json.JSONDecodeError, ValueError) as e:
            log.debug(f"Failed to parse media hype analysis response: {e}")

        return 0.0, 0.0, response[:200]

    def analyze_injury_impact(self, team: str) -> Tuple[float, str]:
        """Analyze injury situation and its impact."""
        prompt = f"""Analyze the current injury situation for {team} ({self._league_label}).

Search for the latest injury news, player availability, and squad fitness.

Provide an impact score from -100 to 0 where:
- -100 to -70: Severe crisis (multiple key players out)
- -69 to -40: Significant impact (important players missing)
- -39 to -20: Moderate impact (some players doubtful)
- -19 to 0: Minimal impact (nearly full squad available)

Format your response as JSON:
{{
    "impact_score": <number -100 to 0>,
    "injured_players": ["<player1>", "<player2>"],
    "key_absences": ["<key_player>"],
    "reasoning": "<brief explanation>"
}}"""

        system_prompt = """You are a {self._league_label} squad analyst.
Focus on confirmed injuries, doubtful players, and expected return dates.
Assess the importance of missing players to the team."""

        response = self.client.query(prompt, system_prompt)
        if not response:
            return -10.0, "Unable to analyze injury situation"

        try:
            import re
            json_match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return float(data.get("impact_score", -10)), data.get("reasoning", "")
        except (json.JSONDecodeError, ValueError) as e:
            log.debug(f"Failed to parse injury impact analysis response: {e}")

        return -10.0, response[:200]

    def analyze_transfer_buzz(self, team: str) -> Tuple[float, str]:
        """Analyze transfer activity and its impact on morale."""
        prompt = f"""Analyze recent transfer activity for {team} ({self._league_label}).

Search for transfer rumors, completed deals, and their impact on squad morale.

Provide a buzz score from 0 to +50 where:
- 0-10: Negative (key players leaving, failed signings)
- 11-25: Neutral (no significant activity)
- 26-40: Positive (good signings, retained key players)
- 41-50: Very positive (marquee signing, major investment)

Format your response as JSON:
{{
    "buzz_score": <number 0 to 50>,
    "recent_activity": ["<activity1>", "<activity2>"],
    "morale_impact": "<positive/negative/neutral>",
    "reasoning": "<brief explanation>"
}}"""

        system_prompt = """You are a {self._league_label} transfer market analyst.
Focus on recent signings, departures, and ongoing negotiations.
Assess the psychological impact on the squad."""

        response = self.client.query(prompt, system_prompt)
        if not response:
            return 15.0, "Unable to analyze transfer activity"

        try:
            import re
            json_match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return float(data.get("buzz_score", 15)), data.get("reasoning", "")
        except (json.JSONDecodeError, ValueError) as e:
            log.debug(f"Failed to parse transfer buzz analysis response: {e}")

        return 15.0, response[:200]

    def check_derby_pressure(self, home_team: str, away_team: str) -> Tuple[float, str]:
        """Check if this is a derby/rivalry and analyze pressure."""
        # Check known rivalries
        rivalry_name = None
        for (team1, team2), name in ITALIAN_RIVALRIES.items():
            if (home_team in [team1, team2]) and (away_team in [team1, team2]):
                rivalry_name = name
                break

        if not rivalry_name:
            return 0.0, "No significant rivalry"

        prompt = f"""This is the {rivalry_name} between {home_team} and {away_team}.

Analyze the current pressure and stakes for this derby.

Provide a pressure score from -25 to +25 where:
- Negative: More pressure on one team (expectations, recent losses)
- 0: Equal pressure on both teams
- Positive: More pressure on the other team

Also indicate which team faces MORE pressure.

Format your response as JSON:
{{
    "pressure_score": <number -25 to +25>,
    "more_pressure_on": "<home/away>",
    "stakes": "<high/medium/low>",
    "reasoning": "<brief explanation of derby context>"
}}"""

        system_prompt = """You are an Italian football derby expert.
Understand the historical significance and current context of Italian rivalries.
Assess the psychological pressure on each team."""

        response = self.client.query(prompt, system_prompt)
        if not response:
            return 0.0, f"{rivalry_name} - unable to analyze pressure"

        try:
            import re
            json_match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return float(data.get("pressure_score", 0)), data.get("reasoning", "")
        except (json.JSONDecodeError, ValueError) as e:
            log.debug(f"Failed to parse derby pressure analysis response: {e}")

        return 0.0, f"{rivalry_name}: {response[:150]}"

    def calculate_composite_score(
        self,
        fan_confidence: float,
        media_hype: float,
        injury_impact: float,
        transfer_buzz: float,
        derby_pressure: float
    ) -> float:
        """Calculate weighted composite sentiment score."""
        # Normalize scores to -100 to +100 scale
        normalized_fan = (fan_confidence - 50) * 2  # 0-100 -> -100 to +100
        normalized_media = media_hype * 2  # -50 to +50 -> -100 to +100
        normalized_injury = injury_impact  # Already -100 to 0
        normalized_transfer = transfer_buzz * 2  # 0 to +50 -> 0 to +100
        normalized_derby = derby_pressure * 4  # -25 to +25 -> -100 to +100

        composite = (
            normalized_fan * SENTIMENT_WEIGHTS["fan_confidence"] +
            normalized_media * SENTIMENT_WEIGHTS["media_hype"] +
            normalized_injury * SENTIMENT_WEIGHTS["injury_impact"] +
            normalized_transfer * SENTIMENT_WEIGHTS["transfer_buzz"] +
            normalized_derby * SENTIMENT_WEIGHTS["derby_pressure"]
        )

        return round(composite, 1)

    def _analyze_match_web_search(self, home_team: str, away_team: str) -> Optional[dict]:
        """Try web search API for all sentiment components. Returns None on failure.

        Cascade: Gemini Grounding -> Groq Compound -> None (triggers data-driven).
        """
        if not self._use_web_search:
            return None

        try:
            home_fan, home_fan_reason = self.analyze_fan_sentiment(home_team)
            # If web search returned the default fallback, it failed
            if home_fan_reason == "Unable to analyze fan sentiment":
                # Primary failed — try switching to secondary before giving up
                if isinstance(self.client, GeminiClient) and self._groq._client:
                    log.warning("Gemini failed, falling back to Groq")
                    self.client = self._groq
                    home_fan, home_fan_reason = self.analyze_fan_sentiment(home_team)
                    if home_fan_reason == "Unable to analyze fan sentiment":
                        log.warning("Groq also failed, switching to data-driven")
                        self._use_web_search = False
                        return None
                else:
                    log.warning("Web search API non-functional, switching to data-driven")
                    self._use_web_search = False
                    return None

            time.sleep(1)
            away_fan, away_fan_reason = self.analyze_fan_sentiment(away_team)
            time.sleep(1)
            home_media, away_media, media_reason = self.analyze_media_hype(home_team, away_team)
            time.sleep(1)
            home_injury, home_injury_reason = self.analyze_injury_impact(home_team)
            time.sleep(1)
            away_injury, away_injury_reason = self.analyze_injury_impact(away_team)
            time.sleep(1)
            home_transfer, home_transfer_reason = self.analyze_transfer_buzz(home_team)
            time.sleep(1)
            away_transfer, away_transfer_reason = self.analyze_transfer_buzz(away_team)
            time.sleep(1)
            derby_pressure, derby_reason = self.check_derby_pressure(home_team, away_team)

            source = "Gemini Grounding" if isinstance(self.client, GeminiClient) else "Groq Compound"
            return {
                "home_fan": home_fan, "home_fan_reason": home_fan_reason,
                "away_fan": away_fan, "away_fan_reason": away_fan_reason,
                "home_media": home_media, "away_media": away_media,
                "media_reason": media_reason,
                "home_injury": home_injury, "home_injury_reason": home_injury_reason,
                "away_injury": away_injury, "away_injury_reason": away_injury_reason,
                "home_transfer": home_transfer, "home_transfer_reason": home_transfer_reason,
                "away_transfer": away_transfer, "away_transfer_reason": away_transfer_reason,
                "derby_pressure": derby_pressure, "derby_reason": derby_reason,
                "source": source,
            }
        except Exception as e:
            log.warning(f"Web search analysis failed: {e}")
            return None

    def _analyze_match_data_driven(self, home_team: str, away_team: str) -> dict:
        """Data-driven sentiment from standings, form, and elo."""
        dd = self._data_driven

        home_fan, home_fan_reason = dd.fan_confidence(home_team)
        away_fan, away_fan_reason = dd.fan_confidence(away_team)
        home_media, away_media, media_reason = dd.media_hype(home_team, away_team)
        home_injury, home_injury_reason = dd.injury_impact(home_team)
        away_injury, away_injury_reason = dd.injury_impact(away_team)
        home_transfer, home_transfer_reason = dd.transfer_buzz(home_team)
        away_transfer, away_transfer_reason = dd.transfer_buzz(away_team)

        # Derby detection (reuse existing logic — no API needed)
        derby_pressure = 0.0
        derby_reason = "No significant rivalry"
        for (team1, team2), name in ITALIAN_RIVALRIES.items():
            if (home_team in [team1, team2]) and (away_team in [team1, team2]):
                derby_pressure = 15.0
                derby_reason = f"{name} — high-pressure derby"
                break

        return {
            "home_fan": home_fan, "home_fan_reason": home_fan_reason,
            "away_fan": away_fan, "away_fan_reason": away_fan_reason,
            "home_media": home_media, "away_media": away_media,
            "media_reason": media_reason,
            "home_injury": home_injury, "home_injury_reason": home_injury_reason,
            "away_injury": away_injury, "away_injury_reason": away_injury_reason,
            "home_transfer": home_transfer, "home_transfer_reason": home_transfer_reason,
            "away_transfer": away_transfer, "away_transfer_reason": away_transfer_reason,
            "derby_pressure": derby_pressure, "derby_reason": derby_reason,
            "source": "Data-Driven Analysis",
        }

    def analyze_match(self, home_team: str, away_team: str, match_date: str) -> MatchSentiment:
        """Perform complete sentiment analysis for a match.

        Cascade: Gemini Grounding → Groq Compound → data-driven fallback.
        """
        match_name = f"{home_team} vs {away_team}"

        # Check cache first
        cached = self._load_from_cache(match_name, match_date)
        if cached and cached.home_fan_confidence != 50.0:
            # Only use cache if it has real data (not default 50/50)
            log.info(f"Using cached sentiment for {match_name}")
            return cached

        log.info(f"Analyzing sentiment for {match_name}...")

        # Try web search first, fall back to data-driven
        result = self._analyze_match_web_search(home_team, away_team)
        if not result:
            result = self._analyze_match_data_driven(home_team, away_team)

        home_fan = result["home_fan"]
        home_fan_reason = result["home_fan_reason"]
        away_fan = result["away_fan"]
        away_fan_reason = result["away_fan_reason"]
        home_media = result["home_media"]
        away_media = result["away_media"]
        media_reason = result["media_reason"]
        home_injury = result["home_injury"]
        home_injury_reason = result["home_injury_reason"]
        away_injury = result["away_injury"]
        away_injury_reason = result["away_injury_reason"]
        home_transfer = result["home_transfer"]
        home_transfer_reason = result["home_transfer_reason"]
        away_transfer = result["away_transfer"]
        away_transfer_reason = result["away_transfer_reason"]
        derby_pressure = result["derby_pressure"]
        derby_reason = result["derby_reason"]
        source = result["source"]

        # Calculate composite scores
        home_composite = self.calculate_composite_score(
            home_fan, home_media, home_injury, home_transfer, derby_pressure
        )
        away_composite = self.calculate_composite_score(
            away_fan, away_media, away_injury, away_transfer, -derby_pressure
        )

        # Determine sentiment edge
        edge_diff = home_composite - away_composite
        if edge_diff > 15:
            sentiment_edge = "home"
        elif edge_diff < -15:
            sentiment_edge = "away"
        else:
            sentiment_edge = "neutral"

        # Check for contrarian opportunity
        contrarian = abs(home_composite) > 60 or abs(away_composite) > 60

        # Compile key factors
        key_factors = []
        if home_fan > 75:
            key_factors.append("high_home_fan_confidence")
        if away_fan > 75:
            key_factors.append("high_away_fan_confidence")
        if home_fan < 35:
            key_factors.append("low_home_fan_confidence")
        if away_fan < 35:
            key_factors.append("low_away_fan_confidence")
        if abs(home_media) > 30:
            key_factors.append("media_hype_home" if home_media > 0 else "media_criticism_home")
        if abs(away_media) > 30:
            key_factors.append("media_hype_away" if away_media > 0 else "media_criticism_away")
        if home_injury < -50:
            key_factors.append("home_injury_crisis")
        if away_injury < -50:
            key_factors.append("away_injury_crisis")
        if derby_pressure != 0:
            key_factors.append("derby_pressure")
        if contrarian:
            key_factors.append("sentiment_overreaction")

        # Build reasoning
        reasoning = f"""Sentiment Analysis for {match_name} [{source}]:

HOME ({home_team}):
- Fan Confidence: {home_fan:.0f}/100 - {home_fan_reason}
- Media Hype: {home_media:+.0f} - {media_reason}
- Injury Impact: {home_injury:.0f} - {home_injury_reason}
- Transfer Buzz: {home_transfer:.0f} - {home_transfer_reason}
- Composite Score: {home_composite:+.1f}

AWAY ({away_team}):
- Fan Confidence: {away_fan:.0f}/100 - {away_fan_reason}
- Media Hype: {away_media:+.0f}
- Injury Impact: {away_injury:.0f} - {away_injury_reason}
- Transfer Buzz: {away_transfer:.0f} - {away_transfer_reason}
- Composite Score: {away_composite:+.1f}

Derby Factor: {derby_reason}

CONCLUSION: Sentiment edge favors {sentiment_edge.upper()}
{'WARNING: Extreme sentiment detected - potential contrarian opportunity!' if contrarian else ''}"""

        if source == "Data-Driven Analysis":
            sources = ["Standings & Form", "Elo Ratings", "Head-to-Head"]
        elif "Gemini" in source:
            sources = ["Google Gemini", "Google Search", "Italian Sports Media"]
        else:
            sources = ["Groq Compound", "Web Search", "Italian Sports Media"]

        sentiment = MatchSentiment(
            match=match_name,
            date=match_date,
            home_team=home_team,
            away_team=away_team,
            home_fan_confidence=home_fan,
            away_fan_confidence=away_fan,
            media_hype_home=home_media,
            media_hype_away=away_media,
            home_injury_impact=home_injury,
            away_injury_impact=away_injury,
            home_transfer_buzz=home_transfer,
            away_transfer_buzz=away_transfer,
            derby_pressure=derby_pressure,
            home_composite=home_composite,
            away_composite=away_composite,
            sentiment_edge=sentiment_edge,
            key_factors=key_factors,
            contrarian_opportunity=contrarian,
            reasoning=reasoning.strip(),
            generated_at=datetime.now().isoformat(),
            sources_used=sources,
        )

        # Cache result
        self._save_to_cache(sentiment)

        return sentiment


# =============================================================================
# INTEGRATION WITH PREDICTIONS
# =============================================================================

def get_sentiment_factors(sentiment: MatchSentiment) -> Dict[str, any]:
    """Convert sentiment analysis to prediction factors."""
    factors = {}

    # Fan confidence factors
    if sentiment.home_fan_confidence > 75:
        factors["high_home_fan_confidence"] = True
    if sentiment.away_fan_confidence > 75:
        factors["high_away_fan_confidence"] = True
    if sentiment.home_fan_confidence < 35:
        factors["low_home_fan_confidence"] = True
    if sentiment.away_fan_confidence < 35:
        factors["low_away_fan_confidence"] = True

    # Media hype factors
    if sentiment.media_hype_home > 30:
        factors["media_hype_home"] = True
    if sentiment.media_hype_away > 30:
        factors["media_hype_away"] = True
    if sentiment.media_hype_home < -30:
        factors["media_criticism_home"] = True
    if sentiment.media_hype_away < -30:
        factors["media_criticism_away"] = True

    # Injury factors
    if sentiment.home_injury_impact < -60:
        factors["home_injury_crisis"] = True
    if sentiment.away_injury_impact < -60:
        factors["away_injury_crisis"] = True

    # Transfer factors
    if sentiment.home_transfer_buzz > 35:
        factors["home_transfer_boost"] = True
    if sentiment.away_transfer_buzz > 35:
        factors["away_transfer_boost"] = True

    # Derby factor
    if abs(sentiment.derby_pressure) > 10:
        factors["derby_pressure"] = True

    # Contrarian factor
    if sentiment.contrarian_opportunity:
        factors["sentiment_overreaction"] = True

    # Probability adjustments
    edge = sentiment.home_composite - sentiment.away_composite
    if edge > 20:
        factors["sentiment_favors_home"] = round(edge / 100, 2)
    elif edge < -20:
        factors["sentiment_favors_away"] = round(abs(edge) / 100, 2)

    return factors


def analyze_all_upcoming_matches() -> List[MatchSentiment]:
    """Analyze sentiment for all upcoming matches."""
    predictions_path = DATA_DIR / "upcoming" / "predictions.json"

    if not predictions_path.exists():
        log.error("No predictions file found")
        return []

    with open(predictions_path) as f:
        data = json.load(f)

    matches = data.get("predictions", [])
    analyzer = SentimentAnalyzer()
    results = []

    for match in matches:
        # Handle both formats: separate fields or combined "match" field
        home_team = match.get("home_team", "")
        away_team = match.get("away_team", "")

        # If no separate fields, parse from "match" field (e.g., "Roma vs Cagliari")
        if not home_team or not away_team:
            match_str = match.get("match", "")
            if " vs " in match_str:
                parts = match_str.split(" vs ")
                home_team = parts[0].strip()
                away_team = parts[1].strip()

        match_date = match.get("date", "")

        if not home_team or not away_team:
            log.warning(f"Could not parse match: {match}")
            continue

        try:
            sentiment = analyzer.analyze_match(home_team, away_team, match_date)
            results.append(sentiment)
            log.info(f"Analyzed: {home_team} vs {away_team} -> Edge: {sentiment.sentiment_edge}")
        except Exception as e:
            log.error(f"Failed to analyze {home_team} vs {away_team}: {e}")

        # Rate limiting
        time.sleep(2)

    # Save results
    output_path = DATA_DIR / "upcoming" / "sentiment_analysis.json"
    with open(output_path, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "matches": [asdict(s) for s in results],
            "summary": {
                "total_analyzed": len(results),
                "home_edge": sum(1 for s in results if s.sentiment_edge == "home"),
                "away_edge": sum(1 for s in results if s.sentiment_edge == "away"),
                "neutral": sum(1 for s in results if s.sentiment_edge == "neutral"),
                "contrarian_opportunities": sum(1 for s in results if s.contrarian_opportunity)
            }
        }, f, indent=2)

    log.info(f"Saved sentiment analysis to {output_path}")
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Perplexity Sentiment Analyzer")
    parser.add_argument("--match", type=str, help="Specific match to analyze (e.g., 'Lecce vs Udinese')")
    parser.add_argument("--all", action="store_true", help="Analyze all upcoming matches")
    parser.add_argument("--home", type=str, help="Home team name")
    parser.add_argument("--away", type=str, help="Away team name")
    parser.add_argument("--date", type=str, help="Match date (YYYY-MM-DD)")

    args = parser.parse_args()

    if args.all:
        results = analyze_all_upcoming_matches()
        print(f"\nAnalyzed {len(results)} matches")
        for r in results:
            print(f"  {r.match}: {r.sentiment_edge} edge (H:{r.home_composite:+.1f} A:{r.away_composite:+.1f})")

    elif args.home and args.away:
        analyzer = SentimentAnalyzer()
        date = args.date or datetime.now().strftime("%Y-%m-%d")
        sentiment = analyzer.analyze_match(args.home, args.away, date)
        print(sentiment.reasoning)

    elif args.match:
        # Parse match string
        if " vs " in args.match:
            home, away = args.match.split(" vs ")
            analyzer = SentimentAnalyzer()
            date = args.date or datetime.now().strftime("%Y-%m-%d")
            sentiment = analyzer.analyze_match(home.strip(), away.strip(), date)
            print(sentiment.reasoning)
        else:
            print("Invalid match format. Use: 'Home Team vs Away Team'")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
