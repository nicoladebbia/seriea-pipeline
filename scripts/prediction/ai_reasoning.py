#!/usr/bin/env python3
"""AI-Powered Bet Reasoning using OpenAI GPT-4o-mini.

Generates deep analysis and reasoning for betting recommendations
using OpenAI's GPT-4o-mini model - optimized for cost and quality.

Cost: ~$0.10 per bet analysis
Quality: Excellent reasoning for Serie A betting
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import hashlib

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass  # dotenv not required if env vars set externally

# OpenAI API Configuration (GPT-4o-mini - best value)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"  # Best value: $0.15/1M input, $0.60/1M output

if not OPENAI_API_KEY:
    log.warning("OPENAI_API_KEY not set - AI reasoning will be disabled")

# Cache directory
CACHE_DIR = DATA_DIR / "ai_reasoning_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_cache_key(bet_data: Dict) -> str:
    """Generate a cache key for a bet."""
    key_data = f"{bet_data.get('match', '')}-{bet_data.get('selection', '')}-{bet_data.get('market', '')}"
    return hashlib.md5(key_data.encode()).hexdigest()


def load_cached_reasoning(cache_key: str) -> Optional[Dict]:
    """Load cached reasoning if available and fresh."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                cached = json.load(f)
            # Check if cache is less than 24 hours old
            cached_time = datetime.fromisoformat(cached.get("generated_at", "2000-01-01"))
            if (datetime.now() - cached_time).total_seconds() < 86400:
                return cached
        except Exception as e:
            log.debug(f"Failed to load cached AI reasoning: {e}")
    return None


def save_cached_reasoning(cache_key: str, reasoning: Dict):
    """Save reasoning to cache."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    reasoning["generated_at"] = datetime.now().isoformat()
    with open(cache_file, "w") as f:
        json.dump(reasoning, f, indent=2)


def generate_bet_reasoning(bet_data: Dict, match_context: Optional[Dict] = None) -> Dict:
    """Generate AI reasoning for a specific bet using OpenAI GPT-4o-mini.

    Args:
        bet_data: The bet information including match, selection, odds, value
        match_context: Additional context about the match (form, injuries, etc.)

    Returns:
        Dict with reasoning analysis
    """
    cache_key = get_cache_key(bet_data)

    # Check cache first
    cached = load_cached_reasoning(cache_key)
    if cached:
        log.info(f"Using cached reasoning for {bet_data.get('match')}")
        return cached

    if OpenAI is None:
        log.warning("OpenAI package not installed. Using fallback reasoning.")
        return generate_fallback_reasoning(bet_data)

    try:
        # Use OpenAI API (standard endpoint, no base_url needed)
        client = OpenAI(api_key=OPENAI_API_KEY)

        # Build the prompt
        prompt = build_reasoning_prompt(bet_data, match_context)

        log.info(f"Generating AI reasoning for {bet_data.get('match')} - {bet_data.get('selection')} (GPT-4o-mini)")

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert Serie A football betting analyst with deep knowledge of Italian football culture and betting markets.

Your analysis must include:
1. **Value Explanation**: WHY this bet has value - explain the statistical edge clearly
2. **Key Factors**: Specific factors supporting this prediction (form, injuries, tactics, history)
3. **Italian Context**: Serie A specific insights (derby importance, fan pressure, referee tendencies)
4. **Risk Assessment**: Potential risks and concerns to watch
5. **Confidence Level**: Your honest assessment of the bet quality

IMPORTANT GUIDELINES:
- Be specific to this exact match and market
- Use Italian betting market standards (Over/Under 2.5 not 2.75)
- Consider Italian football culture (strong home bias, derby passion)
- Reference specific stats and factors provided
- Keep analysis concise but insightful (400-600 words)
- Format with clear sections using **bold** headers"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.6,
            max_tokens=800,  # Optimized for cost (~$0.10 per request)
        )

        reasoning_text = response.choices[0].message.content

        # Parse the reasoning into structured format
        reasoning = {
            "match": bet_data.get("match"),
            "selection": bet_data.get("selection"),
            "market": bet_data.get("market"),
            "analysis": reasoning_text,
            "key_points": extract_key_points(reasoning_text),
            "confidence_level": determine_confidence(bet_data, reasoning_text),
            "risk_factors": extract_risks(reasoning_text),
            "model": f"OpenAI {OPENAI_MODEL}",
        }

        # Cache the result
        save_cached_reasoning(cache_key, reasoning)

        return reasoning

    except Exception as e:
        log.error(f"Failed to generate AI reasoning: {e}")
        return generate_fallback_reasoning(bet_data)


def build_reasoning_prompt(bet_data: Dict, match_context: Optional[Dict] = None) -> str:
    """Build the prompt for AI reasoning with Italian market context."""

    match = bet_data.get("match", "Unknown")
    market = bet_data.get("market", "unknown")
    selection = bet_data.get("selection", "")
    odds = bet_data.get("odds", 0)
    our_prob = bet_data.get("our_probability", 0)
    value_pct = bet_data.get("value_pct", 0)
    factors = bet_data.get("factors", [])
    confidence = bet_data.get("confidence", "MEDIUM")
    risk_level = bet_data.get("risk_level", "medium")

    # Calculate implied probability from odds
    implied_prob = 1 / odds if odds > 0 else 0
    edge = our_prob - implied_prob

    # Normalize selection for Italian markets
    normalized_selection = normalize_italian_selection(selection)

    prompt = f"""Analyze this Serie A betting recommendation for Italian betting markets:

## Match Details
- **Match:** {match}
- **Market:** {format_market(market)}
- **Selection:** {normalized_selection}
- **Risk Level:** {risk_level.upper()}

## Odds Analysis
- **Bookmaker Odds:** {odds:.2f}
- **Implied Probability:** {implied_prob*100:.1f}%
- **Our Model Probability:** {our_prob*100:.1f}%
- **Value Edge:** +{value_pct:.1f}% ({edge*100:.1f} percentage points)
- **System Confidence:** {confidence}

## Detected Factors
{format_factors(factors)}

"""

    if match_context:
        prompt += f"""## Team Context
- **Home Team Form:** {match_context.get('home_form', 'N/A')}%
- **Away Team Form:** {match_context.get('away_form', 'N/A')}%
- **Home Elo Rating:** {match_context.get('home_elo', 'N/A')}
- **Away Elo Rating:** {match_context.get('away_elo', 'N/A')}
- **Home Injuries:** {match_context.get('home_injuries', 0)} players out
- **Away Injuries:** {match_context.get('away_injuries', 0)} players out
- **Head-to-Head History:** {match_context.get('h2h_matches', 0)} recent matches
"""

    prompt += """
## Your Analysis Task

Provide a detailed analysis covering:

1. **Value Explanation** - Why does our model see value that bookmakers missed?
2. **Key Supporting Factors** - What specific evidence supports this bet?
3. **Italian Football Context** - Any Serie A specific factors (derby, fan pressure, tactical style)?
4. **Risk Assessment** - What could go wrong? Key concerns?
5. **Final Verdict** - Your honest confidence level and any caveats

Focus on THIS specific match. Be concrete and insightful, not generic."""

    return prompt


def normalize_italian_selection(selection: str) -> str:
    """Normalize selection to Italian market standards."""
    # Fix non-standard lines for Italian markets
    replacements = {
        "OVER 2.75": "OVER 2.5/3.0",
        "UNDER 2.75": "UNDER 2.5/3.0",
        "OVER 2.25": "OVER 2.5",
        "UNDER 2.25": "UNDER 2.5",
        "OVER 3.25": "OVER 3.5",
        "UNDER 3.25": "UNDER 3.5",
    }
    return replacements.get(selection.upper(), selection)


def format_market(market: str) -> str:
    """Format market name for display."""
    return {
        "h2h": "Match Result (1X2)",
        "totals": "Over/Under Goals",
        "spreads": "Asian Handicap",
        "cards": "Total Cards",
        "btts": "Both Teams To Score",
        "corners": "Total Corners",
    }.get(market, market)


def format_factors(factors: List[str]) -> str:
    """Format factors list for prompt."""
    factor_descriptions = {
        "big_home_favorite": "Home team is heavily favored (large Elo gap)",
        "big_away_favorite": "Away team is heavily favored (large Elo gap)",
        "home_favorite": "Home team is favored",
        "away_favorite": "Away team is favored",
        "hot_home": "Home team in strong recent form",
        "hot_away": "Away team in strong recent form",
        "cold_home": "Home team in poor recent form",
        "cold_away": "Away team in poor recent form",
        "cold": "Cold weather conditions",
        "strict_ref": "Strict referee assigned (higher cards expected)",
        "away_fav_ref": "Referee historically favors away teams",
        "home_fav_ref": "Referee historically favors home teams",
        "big_stadium": "Large stadium capacity (strong home support)",
    }

    if not factors:
        return "- No specific factors detected"

    return "\n".join([
        f"- {factor_descriptions.get(f, f)}"
        for f in factors
    ])


def extract_key_points(reasoning_text: str) -> List[str]:
    """Extract key points from the reasoning text."""
    # Simple extraction - look for bullet points or numbered items
    points = []
    lines = reasoning_text.split("\n")

    for line in lines:
        line = line.strip()
        if line.startswith(("- ", "* ", "1.", "2.", "3.", "4.", "5.")):
            # Clean up the line
            clean_line = line.lstrip("-*0123456789. ")
            if len(clean_line) > 20 and len(clean_line) < 200:
                points.append(clean_line)

    # If no bullet points found, extract sentences
    if not points:
        sentences = reasoning_text.replace("\n", " ").split(". ")
        for sent in sentences[:5]:
            if len(sent) > 30 and len(sent) < 200:
                points.append(sent.strip() + ".")

    return points[:5]


def extract_risks(reasoning_text: str) -> List[str]:
    """Extract risk factors from reasoning text."""
    risks = []
    text_lower = reasoning_text.lower()

    risk_keywords = ["risk", "concern", "caution", "however", "but", "warning", "downside"]

    sentences = reasoning_text.replace("\n", " ").split(". ")
    for sent in sentences:
        if any(keyword in sent.lower() for keyword in risk_keywords):
            clean_sent = sent.strip()
            if len(clean_sent) > 20 and len(clean_sent) < 200:
                risks.append(clean_sent + ".")

    return risks[:3]


def determine_confidence(bet_data: Dict, reasoning_text: str) -> str:
    """Determine confidence level based on bet data and reasoning."""
    value_pct = bet_data.get("value_pct", 0)
    our_prob = bet_data.get("our_probability", 0)

    # High confidence indicators
    if value_pct > 20 and our_prob > 0.6:
        return "HIGH"
    elif value_pct > 15 and our_prob > 0.5:
        return "MEDIUM-HIGH"
    elif value_pct > 10:
        return "MEDIUM"
    else:
        return "LOW"


def generate_fallback_reasoning(bet_data: Dict) -> Dict:
    """Generate fallback reasoning when AI is unavailable."""
    match = bet_data.get("match", "Unknown")
    market = bet_data.get("market", "unknown")
    selection = bet_data.get("selection", "")
    odds = bet_data.get("odds", 0)
    our_prob = bet_data.get("our_probability", 0)
    value_pct = bet_data.get("value_pct", 0)
    factors = bet_data.get("factors", [])

    implied_prob = 1 / odds if odds > 0 else 0
    edge = our_prob - implied_prob

    # Build analysis based on factors
    analysis_parts = [
        f"**Value Analysis for {match}**\n",
        f"This {format_market(market)} bet on {selection} offers a +{value_pct:.1f}% edge based on our statistical models.\n",
        f"\n**Statistical Edge:**\n",
        f"Our model assigns a {our_prob*100:.1f}% probability to this outcome, ",
        f"while the bookmaker odds ({odds:.2f}) imply only {implied_prob*100:.1f}%. ",
        f"This {edge*100:.1f} percentage point difference represents genuine value.\n",
    ]

    if factors:
        analysis_parts.append(f"\n**Key Factors:**\n")
        for factor in factors:
            factor_desc = {
                "big_home_favorite": "The home team has a significant Elo rating advantage, suggesting dominance",
                "big_away_favorite": "The away team has a significant Elo rating advantage despite playing away",
                "home_favorite": "Home advantage combined with team quality favors the home side",
                "away_favorite": "The away team's quality overcomes home disadvantage",
                "hot_home": "The home team is in excellent recent form with strong results",
                "hot_away": "The away team has built momentum with consecutive good performances",
                "cold_home": "The home team's recent form has been poor, affecting confidence",
                "cold_away": "The away team has struggled recently, indicating potential vulnerability",
                "cold": "Cold weather may affect playing style and favor physical teams",
                "strict_ref": "The assigned referee has a history of showing many cards",
                "away_fav_ref": "Historical data shows this referee's decisions tend to favor away teams",
                "home_fav_ref": "This referee's statistics show a tendency toward home-favorable decisions",
            }.get(factor, f"Factor detected: {factor}")
            analysis_parts.append(f"- {factor_desc}\n")

    analysis_parts.append(f"\n**Risk Assessment:**\n")
    if our_prob < 0.5:
        analysis_parts.append("- This is an underdog selection - higher variance expected\n")
    if value_pct > 25:
        analysis_parts.append("- Very high value edge - verify odds haven't moved\n")
    if "cold" in factors:
        analysis_parts.append("- Weather conditions add unpredictability\n")

    confidence = determine_confidence(bet_data, "")
    analysis_parts.append(f"\n**Confidence Level: {confidence}**")

    return {
        "match": match,
        "selection": selection,
        "market": market,
        "analysis": "".join(analysis_parts),
        "key_points": [
            f"Model probability ({our_prob*100:.1f}%) exceeds implied odds ({implied_prob*100:.1f}%)",
            f"Value edge of +{value_pct:.1f}% detected",
        ] + [format_factors([f]).strip("- ") for f in factors[:3]],
        "confidence_level": confidence,
        "risk_factors": [
            "Market odds may shift before match",
            "Team news and late changes can affect outcome",
        ],
        "model": "statistical_fallback",
        "generated_at": datetime.now().isoformat(),
    }


def generate_all_bet_reasoning(bets: List[Dict], match_contexts: Optional[Dict] = None) -> List[Dict]:
    """Generate reasoning for all bets.

    Args:
        bets: List of bet dictionaries
        match_contexts: Optional dict mapping match names to context data

    Returns:
        List of reasoning dictionaries
    """
    reasonings = []

    for bet in bets:
        match_name = bet.get("match", "")
        context = match_contexts.get(match_name) if match_contexts else None

        reasoning = generate_bet_reasoning(bet, context)
        reasonings.append(reasoning)

    return reasonings


def load_and_analyze_bets() -> Dict:
    """Load current bets and generate AI reasoning for all."""
    report_path = DATA_DIR / "betting" / "unified_report.json"

    if not report_path.exists():
        log.error("No betting report found")
        return {"error": "No betting data available"}

    with open(report_path) as f:
        report = json.load(f)

    bets = report.get("bets", [])

    if not bets:
        return {"error": "No bets in report"}

    # Generate reasoning for each bet
    reasonings = []
    for bet in bets:
        reasoning = generate_bet_reasoning(bet)
        reasonings.append(reasoning)

    return {
        "generated_at": datetime.now().isoformat(),
        "total_bets": len(bets),
        "reasonings": reasonings,
    }


if __name__ == "__main__":
    # Test the reasoning generation
    result = load_and_analyze_bets()

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Generated reasoning for {result['total_bets']} bets")
        for r in result["reasonings"]:
            print(f"\n{'='*60}")
            print(f"Match: {r['match']}")
            print(f"Selection: {r['selection']}")
            print(f"Confidence: {r['confidence_level']}")
            print(f"\nAnalysis:\n{r['analysis'][:500]}...")
