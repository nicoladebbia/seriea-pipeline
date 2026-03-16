#!/usr/bin/env python3
"""INTELLIGENCE INTEGRATOR - Post-Ensemble Probability Adjustments

Applies soft intelligence signals (Perplexity sentiment, player analysis)
to ensemble predictions. All adjustments are bounded and additive.

Design:
1. All adjustments are POST-calibration (ensemble probs are already good)
2. Maximum total shift: 12 percentage points per outcome
3. Missing data = no-op (graceful degradation)
4. Each signal independently capped before application
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Maximum probability adjustment per signal type
MAX_SENTIMENT_SHIFT = 0.08   # 8pp max from Perplexity sentiment
MAX_PLAYER_SHIFT = 0.07      # 7pp max from player analysis
MAX_MARKET_INTEL_SHIFT = 0.05  # 5pp max from market intelligence
MAX_COMBINED_SHIFT = 0.15    # 15pp max total across all signals


@dataclass
class IntelligenceAdjustment:
    """Records what adjustments were applied and why."""
    source: str
    home_shift: float
    draw_shift: float
    away_shift: float
    reasoning: str
    raw_signal: float


def _apply_and_normalize(
    probs: Dict[str, float],
    home_adj: float, draw_adj: float, away_adj: float
) -> Dict[str, float]:
    """Apply adjustments and re-normalize to sum to 1.0."""
    h = max(0.02, probs["prob_H"] + home_adj)
    d = max(0.02, probs["prob_D"] + draw_adj)
    a = max(0.02, probs["prob_A"] + away_adj)

    total = h + d + a
    return {
        "prob_H": h / total,
        "prob_D": d / total,
        "prob_A": a / total,
    }


def apply_sentiment_adjustment(probs, sentiment, max_shift=MAX_SENTIMENT_SHIFT):
    """Apply Perplexity sentiment to prediction probabilities.

    Uses composite scores (home_composite, away_composite) on -100..+100 scale.
    Only applies if edge > 20 composite points.
    Contrarian flag halves the adjustment.
    """
    edge = sentiment.home_composite - sentiment.away_composite
    raw_signal = edge / 100.0

    if sentiment.contrarian_opportunity:
        raw_signal *= 0.5

    if abs(edge) < 20:
        return probs, IntelligenceAdjustment(
            "sentiment", 0, 0, 0, "Edge too small to adjust", raw_signal
        )

    shift = max(-max_shift, min(max_shift, raw_signal * max_shift))

    home_adj = shift
    away_adj = -shift
    draw_adj = 0.0

    # Strong edges slightly reduce draw probability
    if abs(shift) > 0.03:
        draw_reduction = abs(shift) * 0.3
        draw_adj = -draw_reduction
        if shift > 0:
            home_adj += draw_reduction
        else:
            away_adj += draw_reduction

    new_probs = _apply_and_normalize(probs, home_adj, draw_adj, away_adj)

    return new_probs, IntelligenceAdjustment(
        "sentiment", home_adj, draw_adj, away_adj,
        f"Sentiment edge: {edge:+.1f} ({sentiment.sentiment_edge})",
        raw_signal
    )


def apply_player_adjustment(probs, player_analysis, max_shift=MAX_PLAYER_SHIFT):
    """Apply player strength analysis to prediction probabilities.

    Uses home_strength and away_strength on 0-100 scale.
    Only applies if strength diff > 10.
    """
    diff = player_analysis.home_strength - player_analysis.away_strength
    raw_signal = diff / 100.0

    if abs(diff) < 10:
        return probs, IntelligenceAdjustment(
            "player", 0, 0, 0, "Strength difference too small", raw_signal
        )

    shift = max(-max_shift, min(max_shift, raw_signal * max_shift))

    home_adj = shift
    away_adj = -shift
    draw_adj = 0.0

    new_probs = _apply_and_normalize(probs, home_adj, draw_adj, away_adj)

    return new_probs, IntelligenceAdjustment(
        "player", home_adj, draw_adj, away_adj,
        f"Squad strength: H={player_analysis.home_strength:.0f} A={player_analysis.away_strength:.0f}",
        raw_signal
    )


def apply_market_intelligence_adjustment(probs, market_intel, max_shift=MAX_MARKET_INTEL_SHIFT):
    """Apply market intelligence signals to prediction probabilities.

    Uses sharp-soft divergence direction and steam moves to nudge probabilities.
    Only applies if composite_score > 0.2 (enough signal strength).
    """
    composite = market_intel.get("composite_score", 0)
    if composite < 0.2:
        return probs, IntelligenceAdjustment(
            "market_intelligence", 0, 0, 0,
            "Market intelligence signal too weak", composite
        )

    sharp_dir = market_intel.get("sharp_direction", "neutral")
    divergence = market_intel.get("divergence", 0) or 0
    steam = market_intel.get("steam_detected", False)

    # Base shift from sharp direction
    if sharp_dir == "home":
        raw_signal = min(divergence * 5, 1.0)  # Scale divergence to 0-1
    elif sharp_dir == "away":
        raw_signal = -min(divergence * 5, 1.0)
    else:
        raw_signal = 0.0

    # Boost from steam moves
    if steam:
        move_dir = market_intel.get("movement_direction", "stable")
        if "home_strengthening" in str(move_dir):
            raw_signal += 0.3
        elif "away_strengthening" in str(move_dir):
            raw_signal -= 0.3

    shift = max(-max_shift, min(max_shift, raw_signal * max_shift))

    if abs(shift) < 0.005:
        return probs, IntelligenceAdjustment(
            "market_intelligence", 0, 0, 0,
            f"Sharp={sharp_dir} div={divergence:.3f} (shift too small)", raw_signal
        )

    home_adj = shift
    away_adj = -shift
    draw_adj = 0.0

    new_probs = _apply_and_normalize(probs, home_adj, draw_adj, away_adj)

    return new_probs, IntelligenceAdjustment(
        "market_intelligence", home_adj, draw_adj, away_adj,
        f"Sharp={sharp_dir} div={divergence:.3f} steam={steam}",
        raw_signal
    )


def apply_all_intelligence(prediction, sentiment=None, player_analysis=None, market_intel=None):
    """Apply all available intelligence adjustments to a prediction.

    Args:
        prediction: Prediction dict with "probabilities" {home, draw, away}
        sentiment: Optional MatchSentiment from sentiment_analyzer
        player_analysis: Optional MatchPlayerAnalysis from player_analyzer
        market_intel: Optional dict from market_intelligence.json

    Returns:
        Updated prediction dict with adjusted probabilities and metadata.
    """
    probs = {
        "prob_H": prediction["probabilities"]["home"],
        "prob_D": prediction["probabilities"]["draw"],
        "prob_A": prediction["probabilities"]["away"],
    }

    original_probs = probs.copy()
    adjustments = []

    # 1. Sentiment adjustment
    if sentiment:
        try:
            probs, adj = apply_sentiment_adjustment(probs, sentiment)
            adjustments.append(adj)
        except Exception as e:
            log.debug(f"Sentiment adjustment failed: {e}")

    # 2. Player analysis adjustment
    if player_analysis:
        try:
            probs, adj = apply_player_adjustment(probs, player_analysis)
            adjustments.append(adj)
        except Exception as e:
            log.debug(f"Player adjustment failed: {e}")

    # 3. Market intelligence adjustment
    if market_intel:
        try:
            probs, adj = apply_market_intelligence_adjustment(probs, market_intel)
            adjustments.append(adj)
        except Exception as e:
            log.debug(f"Market intelligence adjustment failed: {e}")

    # Cap total shift
    total_home_shift = sum(a.home_shift for a in adjustments)
    if abs(total_home_shift) > MAX_COMBINED_SHIFT:
        scale = MAX_COMBINED_SHIFT / abs(total_home_shift)
        for adj in adjustments:
            adj.home_shift *= scale
            adj.away_shift *= scale
            adj.draw_shift *= scale
        # Recompute from original
        total_h = sum(a.home_shift for a in adjustments)
        total_d = sum(a.draw_shift for a in adjustments)
        total_a = sum(a.away_shift for a in adjustments)
        probs = _apply_and_normalize(original_probs, total_h, total_d, total_a)

    # Update prediction
    prediction["probabilities"] = {
        "home": round(probs["prob_H"], 3),
        "draw": round(probs["prob_D"], 3),
        "away": round(probs["prob_A"], 3),
    }

    # Re-determine predicted outcome
    p = prediction["probabilities"]
    if p["home"] >= p["draw"] and p["home"] >= p["away"]:
        prediction["predicted_outcome"] = "HOME"
    elif p["away"] >= p["draw"]:
        prediction["predicted_outcome"] = "AWAY"
    else:
        prediction["predicted_outcome"] = "DRAW"

    prediction["confidence"] = max(p["home"], p["draw"], p["away"])

    # Add metadata
    active_adjustments = [a for a in adjustments if abs(a.home_shift) > 0.001]
    if active_adjustments:
        prediction["intelligence_adjustments"] = [
            {
                "source": a.source,
                "home_shift": round(a.home_shift, 4),
                "draw_shift": round(a.draw_shift, 4),
                "away_shift": round(a.away_shift, 4),
                "reasoning": a.reasoning,
            }
            for a in active_adjustments
        ]

    return prediction
