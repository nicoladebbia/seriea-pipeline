#!/usr/bin/env python3
"""Italian Market Standards - SISAL-Compatible Betting Lines

Ensures all betting recommendations follow Italian market standards:
- Standard Over/Under lines (0.5, 1.5, 2.5, 3.5, 4.5)
- European-style handicaps (integer values: -1, +1, -2, +2)
- No Asian-style fractional lines (2.75, 2.25, -0.5, etc.)

Italian Betting Culture:
- Popular markets: 1X2 (60%), Over/Under 2.5 (25%), Draw No Bet (10%), Handicaps (5%)
- Strong home team bias
- Derby matches are special
- SISAL is the primary bookmaker
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

log = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR

# =============================================================================
# ITALIAN MARKET STANDARDS
# =============================================================================

# Standard Italian Over/Under lines (SISAL-compatible)
ITALIAN_OVER_UNDER_LINES = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]

# Asian lines to filter out (not used in Italian markets)
ASIAN_LINES_TO_REMOVE = [0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75]

# Standard Italian handicap lines
ITALIAN_HANDICAP_LINES = [-3, -2, -1, 0, 1, 2, 3]

# Asian handicaps to filter out
ASIAN_HANDICAPS_TO_REMOVE = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]

# Italian match result market
ITALIAN_1X2 = ["1", "X", "2"]  # Home, Draw, Away

# Italian popular markets (popularity weighting)
ITALIAN_MARKET_POPULARITY = {
    "1X2": 0.60,          # Most popular
    "over_under_2.5": 0.25,
    "draw_no_bet": 0.10,
    "handicap": 0.05,
}


@dataclass
class ItalianBet:
    """A bet formatted for Italian markets."""
    match: str
    date: str
    market: str
    selection: str
    odds: float
    our_probability: float
    value_pct: float
    stake_pct: float
    confidence: str
    factors: List[str]
    is_standard_italian: bool = True


def is_standard_italian_line(line: float, market_type: str = "totals") -> bool:
    """Check if a betting line is standard for Italian markets.

    Args:
        line: The betting line (e.g., 2.5, 2.75)
        market_type: "totals" or "handicap"

    Returns:
        True if the line is standard for Italian markets
    """
    if market_type == "totals":
        return line in ITALIAN_OVER_UNDER_LINES
    elif market_type == "handicap":
        return line in ITALIAN_HANDICAP_LINES or int(line) == line
    return True


def normalize_line_to_italian(line: float, market_type: str = "totals") -> Optional[float]:
    """Convert an Asian-style line to the nearest Italian standard line.

    Args:
        line: The betting line to normalize
        market_type: "totals" or "handicap"

    Returns:
        The nearest standard Italian line, or None if cannot be normalized
    """
    if market_type == "totals":
        # Map non-standard lines to Italian standard lines
        # For OVER bets: round UP to be conservative (harder to hit = safer)
        # For UNDER bets: the inverse applies but we handle at bet level
        line_mapping = {
            0.75: 1.5,   # Round up to 1.5
            1.0: 1.5,    # Round up to 1.5
            1.25: 1.5,   # Round to 1.5
            1.75: 2.5,   # Round up to 2.5
            2.0: 2.5,    # Round up to 2.5
            2.25: 2.5,   # Round to 2.5
            2.75: 3.5,   # Round up to 3.5 (conservative for overs)
            3.0: 3.5,    # Round up to 3.5 (conservative)
            3.25: 3.5,   # Round to 3.5
            3.75: 4.5,   # Round up to 4.5
            4.0: 4.5,    # Round up to 4.5
            4.25: 4.5,   # Round to 4.5
            4.75: 5.5,   # Round up to 5.5
            5.0: 5.5,    # Round up to 5.5
        }

        if line in ITALIAN_OVER_UNDER_LINES:
            return line
        elif line in line_mapping:
            return line_mapping[line]
        else:
            # Find nearest standard line, preferring to round UP for conservative betting
            import math
            rounded_up = math.ceil(line * 2) / 2  # Round to nearest 0.5
            candidates = [l for l in ITALIAN_OVER_UNDER_LINES if l >= line]
            if candidates:
                return min(candidates)  # Smallest line >= original
            return max(ITALIAN_OVER_UNDER_LINES)  # Fallback to highest

    elif market_type == "handicap":
        # For European handicaps, round TOWARD ZERO (more conservative for the bettor)
        # -1.5 → -1 (easier to cover for home team)
        # +1.5 → +1 (safer for the underdog)
        # This follows the conservative principle for Italian markets
        import math
        if line < 0:
            return math.ceil(line)   # -1.5 → -1, -2.5 → -2
        else:
            return math.floor(line)  # +1.5 → +1, +2.5 → +2

    return line


def format_italian_selection(market: str, selection: str, line: Optional[float] = None) -> str:
    """Format a selection for Italian market display.

    Args:
        market: Market type ("h2h", "totals", "spreads")
        selection: Raw selection
        line: The betting line (if applicable)

    Returns:
        Italian-formatted selection string
    """
    if market == "h2h":
        # 1X2 format for Italian markets
        selection_map = {
            "HOME": "1 (Casa)",
            "DRAW": "X (Pareggio)",
            "AWAY": "2 (Trasferta)",
        }
        return selection_map.get(selection.upper(), selection)

    elif market == "totals":
        if line:
            normalized_line = normalize_line_to_italian(line, "totals")
            if "OVER" in selection.upper():
                return f"OVER {normalized_line} gol"
            else:
                return f"UNDER {normalized_line} gol"
        return selection

    elif market == "spreads":
        if line:
            normalized_line = normalize_line_to_italian(line, "handicap")
            sign = "+" if normalized_line > 0 else ""
            if "HOME" in selection.upper():
                return f"Casa {sign}{int(normalized_line)}"
            else:
                return f"Trasferta {sign}{int(normalized_line)}"
        return selection

    return selection


def filter_italian_standard_bets(bets: List[Dict], convert_instead_of_remove: bool = True) -> List[Dict]:
    """Filter or convert bets to Italian standard lines.

    Args:
        bets: List of bet dictionaries
        convert_instead_of_remove: If True, convert non-standard lines to standard ones.
                                   If False, remove non-standard lines.

    Returns:
        Filtered/converted list
    """
    filtered = []

    for bet in bets:
        bet = bet.copy()  # Don't modify original
        selection = bet.get("selection", bet.get("bet", ""))
        market = bet.get("market", "")
        modified = False

        # Parse line from totals selection
        if market == "totals" or "OVER" in selection.upper() or "UNDER" in selection.upper():
            try:
                parts = selection.split()
                for i, part in enumerate(parts):
                    try:
                        line = float(part)
                        if not is_standard_italian_line(line, "totals"):
                            if convert_instead_of_remove:
                                # Convert to nearest standard line
                                new_line = normalize_line_to_italian(line, "totals")
                                parts[i] = str(new_line)
                                new_selection = " ".join(parts)
                                if "selection" in bet:
                                    bet["selection"] = new_selection
                                elif "bet" in bet:
                                    bet["bet"] = new_selection
                                bet["original_line"] = line
                                bet["converted_to_italian"] = True
                                modified = True
                            else:
                                # Skip non-standard lines
                                continue
                        break
                    except ValueError:
                        continue
            except Exception as e:
                log.debug(f"Failed to parse totals line for Italian standard check: {e}")

        # Parse handicap lines
        if market == "spreads" or ("-" in selection and "HOME" in selection.upper()) or (
            "+" in selection and ("AWAY" in selection.upper() or "HOME" in selection.upper())
        ):
            try:
                # Extract handicap number
                import re
                match = re.search(r'[+-]?\d+\.?\d*', selection)
                if match:
                    line = float(match.group())
                    if not is_standard_italian_line(abs(line), "handicap"):
                        if convert_instead_of_remove:
                            # Convert to nearest integer handicap
                            new_line = round(line)
                            new_selection = re.sub(r'[+-]?\d+\.?\d*', f"{new_line:+d}" if new_line != 0 else "0", selection)
                            if "selection" in bet:
                                bet["selection"] = new_selection
                            elif "bet" in bet:
                                bet["bet"] = new_selection
                            bet["original_line"] = line
                            bet["converted_to_italian"] = True
                            modified = True
                        else:
                            continue
            except Exception as e:
                log.debug(f"Failed to parse handicap line for Italian standard: {e}")

        filtered.append(bet)

    return filtered


def convert_bets_to_italian_standard(bets: List[Dict]) -> List[Dict]:
    """Convert all bets to Italian standard format.

    Args:
        bets: List of bet dictionaries

    Returns:
        List with bets converted to Italian standards
    """
    converted = []

    for bet in bets:
        selection = bet.get("selection", bet.get("bet", ""))
        market = bet.get("market", "")

        # Parse and normalize line
        line = None
        new_selection = selection

        if "OVER" in selection.upper() or "UNDER" in selection.upper():
            try:
                parts = selection.split()
                for i, part in enumerate(parts):
                    try:
                        line = float(part)
                        if not is_standard_italian_line(line, "totals"):
                            normalized = normalize_line_to_italian(line, "totals")
                            parts[i] = str(normalized)
                            new_selection = " ".join(parts)
                        break
                    except ValueError:
                        continue
            except Exception as e:
                log.debug(f"Failed to normalize totals line for conversion: {e}")

        # Create converted bet
        new_bet = bet.copy()
        if "selection" in new_bet:
            new_bet["selection"] = new_selection
        elif "bet" in new_bet:
            new_bet["bet"] = new_selection

        # Add Italian format flag
        new_bet["italian_format"] = True

        converted.append(new_bet)

    return converted


def get_italian_market_summary(bets: List[Dict]) -> Dict:
    """Get a summary of bets by Italian market standards.

    Args:
        bets: List of bet dictionaries

    Returns:
        Summary dictionary with market breakdown
    """
    summary = {
        "total_bets": len(bets),
        "by_market": {},
        "standard_lines": 0,
        "asian_lines_filtered": 0,
        "popular_markets": {},
    }

    for bet in bets:
        market = bet.get("market", "unknown")
        if market not in summary["by_market"]:
            summary["by_market"][market] = 0
        summary["by_market"][market] += 1

        # Check if standard line
        selection = bet.get("selection", bet.get("bet", ""))
        is_standard = True

        if "OVER" in selection.upper() or "UNDER" in selection.upper():
            try:
                for part in selection.split():
                    try:
                        line = float(part)
                        if not is_standard_italian_line(line, "totals"):
                            is_standard = False
                        break
                    except ValueError:
                        continue
            except Exception as e:
                log.debug(f"Failed to check market standard for summary: {e}")

        if is_standard:
            summary["standard_lines"] += 1
        else:
            summary["asian_lines_filtered"] += 1

    return summary


def apply_italian_standards_to_file(input_path: Path, output_path: Path = None) -> Dict:
    """Apply Italian market standards to a bets JSON file.

    Args:
        input_path: Path to input JSON file
        output_path: Path for output (defaults to same file with _italian suffix)

    Returns:
        Summary of changes made
    """
    if not input_path.exists():
        return {"error": f"File not found: {input_path}"}

    with open(input_path) as f:
        data = json.load(f)

    # Process different file structures
    original_count = 0
    filtered_count = 0

    if "recommended" in data:
        original_count += len(data["recommended"])
        data["recommended"] = filter_italian_standard_bets(data["recommended"])
        data["recommended"] = convert_bets_to_italian_standard(data["recommended"])
        filtered_count += len(data["recommended"])

    if "consider" in data:
        original_count += len(data.get("consider", []))
        data["consider"] = filter_italian_standard_bets(data.get("consider", []))
        data["consider"] = convert_bets_to_italian_standard(data.get("consider", []))
        filtered_count += len(data.get("consider", []))

    if "bets" in data:
        original_count += len(data["bets"])
        data["bets"] = filter_italian_standard_bets(data["bets"])
        data["bets"] = convert_bets_to_italian_standard(data["bets"])
        filtered_count += len(data["bets"])

    # Add Italian standards metadata
    data["italian_market_standards"] = {
        "applied": True,
        "applied_at": datetime.now().isoformat(),
        "original_bets": original_count,
        "filtered_bets": filtered_count,
        "removed_asian_lines": original_count - filtered_count,
    }

    # Update summary if present
    if "summary" in data:
        if "total_bets_analyzed" in data["summary"]:
            data["summary"]["italian_standard_bets"] = filtered_count
        if "recommended_bets" in data["summary"]:
            data["summary"]["recommended_bets"] = len(data.get("recommended", []))

    # Save to output
    if output_path is None:
        output_path = input_path

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return {
        "original_bets": original_count,
        "filtered_bets": filtered_count,
        "removed": original_count - filtered_count,
        "output_path": str(output_path),
    }


def apply_all_italian_standards():
    """Apply Italian market standards to all betting data files."""
    files_to_process = [
        DATA_DIR / "upcoming" / "over_under_bets.json",
        DATA_DIR / "upcoming" / "handicap_bets.json",
        # data/betting/unified_report.json removed 2026-08-31: this loop kept
        # re-stamping a February fossil so it always looked fresh
    ]

    results = {}
    for file_path in files_to_process:
        if file_path.exists():
            result = apply_italian_standards_to_file(file_path)
            results[file_path.name] = result
            print(f"  {file_path.name}: {result.get('original_bets', 0)} -> {result.get('filtered_bets', 0)} bets")

    return results


if __name__ == "__main__":
    print("Applying Italian Market Standards...")
    print("=" * 50)

    results = apply_all_italian_standards()

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    total_original = sum(r.get("original_bets", 0) for r in results.values())
    total_filtered = sum(r.get("filtered_bets", 0) for r in results.values())
    total_removed = sum(r.get("removed", 0) for r in results.values())

    print(f"Total original bets: {total_original}")
    print(f"Total after filtering: {total_filtered}")
    print(f"Asian lines removed: {total_removed}")
    print(f"\nItalian market standards applied successfully!")
