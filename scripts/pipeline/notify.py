#!/usr/bin/env python3
"""Unified notification system -- macOS + Telegram with categories & history.

Sends notifications via all configured channels. Telegram is optional:
if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are not set in .env, only
macOS notifications are sent.

Categories control which channels receive each notification:
  - system  — pipeline runs, health checks, API errors
  - betting — bet settlements, P&L updates, new value bets found
  - live    — goal alerts, red cards, match results
  - retrain — model retraining, promotion, rollback
  - alert   — stale data, drawdown warnings, critical errors

Usage as module:
    from scripts.pipeline.notify import notify
    notify("Pipeline complete", title="SerieAI", level="success")
    notify("Goal!", title="GOAL", level="info", category="live")

Usage as CLI (test mode):
    python -m scripts.pipeline.notify --test
    python -m scripts.pipeline.notify --test --message "Custom message"

Coaching narratives:
    Use notify_value_bet(), notify_settlement(), notify_goal(), notify_retrain()
    for rich, coach-style messages instead of raw data dumps.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

log = logging.getLogger("notify")

# Level -> emoji mapping for Telegram
_LEVEL_EMOJI = {
    "info": "\u2139\ufe0f",       # info
    "success": "\u2705",           # check mark
    "warning": "\u26a0\ufe0f",    # warning
    "error": "\u274c",             # cross mark
    "critical": "\u274c",          # cross mark
}

# Valid categories
VALID_CATEGORIES = {"system", "betting", "live", "retrain", "alert"}

# Default preferences — used when data/notification_preferences.json doesn't exist
_DEFAULT_PREFERENCES = {
    "channels": {
        "macos": True,
        "telegram": True,
    },
    "categories": {
        "system":  {"macos": True, "telegram": False},
        "betting": {"macos": True, "telegram": True},
        "live":    {"macos": True, "telegram": True},
        "retrain": {"macos": True, "telegram": True},
        "alert":   {"macos": True, "telegram": True},
    },
}

# Thread lock for history file writes
_history_lock = threading.Lock()

# Max history entries to keep
_MAX_HISTORY = 200

# Preferences path
_PREFS_PATH = DATA_DIR / "notification_preferences.json"
_HISTORY_PATH = DATA_DIR / "notification_history.jsonl"


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _load_env_key(name: str) -> str:
    """Load a key from os.environ or .env file."""
    val = os.environ.get(name)
    if val:
        return val
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    if k.strip() == name:
                        return v.strip()
    return ""


# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------

def load_preferences() -> dict:
    """Load notification preferences from disk. Returns defaults if file missing."""
    try:
        if _PREFS_PATH.exists():
            with open(_PREFS_PATH) as f:
                prefs = json.load(f)
            # Merge with defaults to ensure all categories exist
            merged = json.loads(json.dumps(_DEFAULT_PREFERENCES))
            if "channels" in prefs:
                merged["channels"].update(prefs["channels"])
            if "categories" in prefs:
                for cat, ch_map in prefs["categories"].items():
                    if cat in merged["categories"]:
                        merged["categories"][cat].update(ch_map)
                    else:
                        merged["categories"][cat] = ch_map
            return merged
    except Exception as e:
        log.warning("Failed to load notification preferences: %s", e)
    return json.loads(json.dumps(_DEFAULT_PREFERENCES))


def save_preferences(prefs: dict) -> bool:
    """Save notification preferences to disk. Returns True on success."""
    try:
        _PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_PREFS_PATH, "w") as f:
            json.dump(prefs, f, indent=2)
        return True
    except Exception as e:
        log.warning("Failed to save notification preferences: %s", e)
        return False


def _should_send(channel: str, category: str) -> bool:
    """Check if a notification should be sent to a given channel for a category."""
    prefs = load_preferences()
    # Check global channel toggle
    if not prefs.get("channels", {}).get(channel, True):
        return False
    # Check category-specific toggle
    cat_prefs = prefs.get("categories", {}).get(category, {})
    # Default to True if category not in prefs
    return cat_prefs.get(channel, True)


# ---------------------------------------------------------------------------
# Notification History
# ---------------------------------------------------------------------------

def _record_history(title: str, message: str, level: str, category: str, channels: dict):
    """Append notification to history file (JSONL). Keeps last _MAX_HISTORY entries."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "title": title,
        "message": message,
        "level": level,
        "category": category,
        "channels": channels,
    }
    try:
        _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _history_lock:
            # Read existing lines
            lines = []
            if _HISTORY_PATH.exists():
                with open(_HISTORY_PATH) as f:
                    lines = f.readlines()
            # Append new entry
            lines.append(json.dumps(entry) + "\n")
            # Keep only last _MAX_HISTORY
            if len(lines) > _MAX_HISTORY:
                lines = lines[-_MAX_HISTORY:]
            # Write back
            with open(_HISTORY_PATH, "w") as f:
                f.writelines(lines)
    except Exception as e:
        log.debug("Failed to record notification history: %s", e)


def get_notification_history(limit: int = 50) -> list:
    """Return the last `limit` notification history entries (newest first)."""
    try:
        if not _HISTORY_PATH.exists():
            return []
        with open(_HISTORY_PATH) as f:
            lines = f.readlines()
        entries = []
        for line in reversed(lines):
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            if len(entries) >= limit:
                break
        return entries
    except Exception as e:
        log.warning("Failed to read notification history: %s", e)
        return []


# ---------------------------------------------------------------------------
# macOS notifications
# ---------------------------------------------------------------------------

def _notify_macos(message: str, title: str) -> bool:
    """Send a macOS notification via osascript. Returns True on success."""
    try:
        # Escape double quotes to prevent osascript injection
        safe_msg = message.replace('"', '\\"')[:256]
        safe_title = title.replace('"', '\\"')[:64]
        script = (
            f'display notification "{safe_msg}" '
            f'with title "{safe_title}" sound name "Basso"'
        )
        subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            timeout=5,
        )
        return True
    except Exception as e:
        log.warning("macOS notification failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Telegram notifications
# ---------------------------------------------------------------------------

def _telegram_configured() -> bool:
    """Check if Telegram credentials are configured."""
    return bool(_load_env_key("TELEGRAM_BOT_TOKEN") and _load_env_key("TELEGRAM_CHAT_ID"))


def _notify_telegram(message: str, title: str, level: str = "info") -> bool:
    """Send a Telegram notification. Returns True on success, False otherwise.

    Silently skips if TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID are not set.
    Handles network errors gracefully (timeout, connection refused, etc.).
    """
    token = _load_env_key("TELEGRAM_BOT_TOKEN")
    chat_id = _load_env_key("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        return False

    emoji = _LEVEL_EMOJI.get(level, _LEVEL_EMOJI["info"])
    text = f"{emoji} *{title}*\n{message}"

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                log.debug("Telegram notification sent")
                return True
            log.warning("Telegram returned HTTP %d", resp.status)
            return False
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            pass
        log.warning("Telegram HTTP error %d: %s", e.code, body)
        return False
    except urllib.error.URLError as e:
        log.warning("Telegram connection error: %s", e.reason)
        return False
    except Exception as e:
        log.warning("Telegram notification failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def notify(message: str, title: str = "SerieAI", level: str = "info",
           category: str = "system") -> dict:
    """Send notification via configured channels, respecting category preferences.

    Args:
        message: Notification body text.
        title: Short title / subject line.
        level: One of "info", "success", "warning", "error", "critical".
        category: One of "system", "betting", "live", "retrain", "alert".

    Returns:
        Dict with channel results, e.g. {"macos": True, "telegram": True}.
    """
    # Normalize category
    if category not in VALID_CATEGORIES:
        category = "system"

    results = {}

    # macOS — check preferences
    if _should_send("macos", category):
        results["macos"] = _notify_macos(message, title)
    else:
        results["macos"] = False

    # Telegram — check preferences
    if _should_send("telegram", category):
        results["telegram"] = _notify_telegram(message, title, level)
    else:
        results["telegram"] = False

    # Record to history (non-blocking)
    try:
        _record_history(title, message, level, category, results)
    except Exception:
        pass

    return results


def notify_status() -> dict:
    """Return configuration status for each notification channel."""
    return {
        "macos": {"configured": True, "detail": "Always available on macOS"},
        "telegram": {
            "configured": _telegram_configured(),
            "detail": (
                "Ready" if _telegram_configured()
                else "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Coaching-style narrative generators
# ---------------------------------------------------------------------------
# These functions take raw data and produce human, coach-style messages
# that feel like a sharp friend texting you — not a robot dumping stats.

import random

_VALUE_BET_OPENERS = [
    "I've been scanning the markets and something caught my eye.",
    "Just finished my deep scan of the odds.",
    "The model and the market are disagreeing on this one.",
    "Here's something most people aren't seeing right now.",
    "I found an edge worth looking at.",
    "The numbers are telling me something interesting.",
]

_SETTLEMENT_WIN_OPENERS = [
    "Good call on that one.",
    "That's the kind of pick that builds bankrolls.",
    "Clean execution.",
    "Nailed it.",
    "The model delivered.",
]

_SETTLEMENT_LOSS_OPENERS = [
    "That one didn't go our way.",
    "Can't win them all.",
    "Variance happens. The process was right.",
    "Tough break, but the edge was real.",
]

_GOAL_OPENERS = [
    "GOAL!",
    "It's in!",
    "The net is shaking!",
]

_FT_WIN_OPENERS = [
    "That's the final whistle, and we're smiling.",
    "Full time. Another one in the bag.",
    "Match over. The model called it.",
]

_FT_LOSS_OPENERS = [
    "Full time. Not our day on this one.",
    "That's the whistle. Didn't go as expected.",
    "Match over. The edge was there, the result wasn't.",
]


def notify_value_bets(bets: list[dict]) -> dict:
    """Send a coaching-style notification about new value bets found.

    Args:
        bets: list of bet dicts from unified_bet_slip.json
    """
    if not bets:
        return {}

    # Pick the best bet to lead with
    best = max(bets, key=lambda b: b.get("edge_pct", 0))
    match = best.get("match", "?")
    selection = best.get("selection", "?")
    market = best.get("market", "?")
    edge = best.get("edge_pct", 0)
    odds = best.get("best_odds", "?")
    model_prob = best.get("model_prob", 0)
    confidence = best.get("confidence_tier", "")

    opener = random.choice(_VALUE_BET_OPENERS)

    # Build the why
    reasons = []
    away_factors = best.get("away_factors", [])
    home_factors = best.get("home_factors", [])
    if "cold_home" in (home_factors or []):
        reasons.append("the home side has been cold recently")
    if "big_away_favorite" in (away_factors or []):
        reasons.append("the away team is the clear favorite here")
    if edge > 10:
        reasons.append(f"the market is pricing this at {edge:.0f}% off what the model sees")
    elif edge > 6:
        reasons.append(f"there's a solid {edge:.1f}% gap between our model and the market")
    if model_prob > 0.7:
        reasons.append(f"the model gives this a {model_prob*100:.0f}% probability")

    reason_text = ""
    if reasons:
        reason_text = " " + reasons[0].capitalize()
        if len(reasons) > 1:
            reason_text += f", and {reasons[1]}"
        reason_text += "."

    # Main message
    msg = f"{opener}\n\n"
    msg += f"{match} — {selection} ({market}) at {odds}"
    if confidence:
        msg += f" [{confidence}]"
    msg += f"\nEdge: {edge:.1f}% over the market."
    if reason_text:
        msg += f"\n{reason_text}"

    if len(bets) > 1:
        msg += f"\n\n{len(bets) - 1} more value bet{'s' if len(bets) > 2 else ''} in the slip."

    return notify(msg, title=f"Value: {match}", level="info", category="betting")


def notify_settlement(settled: int, won: int, lost: int, push: int = 0,
                      profit: float = 0, balance: float = 0,
                      best_win: dict | None = None, worst_loss: dict | None = None) -> dict:
    """Send a coaching-style settlement notification.

    Args:
        settled/won/lost/push: bet counts
        profit: total P&L from this settlement
        balance: new bankroll balance
        best_win: dict with match, selection, profit of best winning bet
        worst_loss: dict with match, selection, profit of worst losing bet
    """
    if settled == 0:
        return {}

    is_positive = profit >= 0
    opener = random.choice(_SETTLEMENT_WIN_OPENERS if is_positive else _SETTLEMENT_LOSS_OPENERS)

    msg = f"{opener}\n\n"
    msg += f"{won}W-{lost}L"
    if push:
        msg += f"-{push}P"
    msg += f" | P&L: {'+'  if profit >= 0 else ''}${profit:.2f}"
    if balance:
        msg += f" | Balance: ${balance:,.2f}"

    if best_win and profit > 0:
        msg += f"\n\nBest pick: {best_win.get('match','')} {best_win.get('selection','')} (+${best_win.get('profit',0):.2f})"

    if won > lost:
        msg += f"\n\nYou're {won}-{lost} this round. Keep trusting the process."
    elif lost > won:
        msg += f"\n\nDown {lost}-{won} this round. Don't chase it — the edge is still there long-term."

    level = "success" if is_positive else "warning"
    return notify(msg, title=f"Settled: {'+'  if profit >= 0 else ''}${profit:.2f}", level=level, category="betting")


def notify_goal(match_key: str, scorer: str, team: str,
                home_score: int, away_score: int, minute: int,
                is_home: bool, has_bet: bool = False, bet_selection: str = "") -> dict:
    """Send a coaching-style goal notification.

    Args:
        match_key: "Inter vs Genoa"
        scorer: player name
        team: scoring team
        home_score/away_score: current score after goal
        minute: match minute
        has_bet: whether user has a bet on this match
        bet_selection: what the user's bet is (e.g. "Over 2.5")
    """
    opener = random.choice(_GOAL_OPENERS)

    msg = f"{opener} {scorer} ({team}) {minute}'\n"
    msg += f"{match_key.replace(' vs ', ' ')} {home_score}-{away_score}"

    if has_bet:
        total = home_score + away_score
        msg += f"\n\nYou have {bet_selection} on this match."
        if "over" in bet_selection.lower():
            line = float(bet_selection.lower().replace("over", "").strip()) if "over" in bet_selection.lower() else 0
            if total > line:
                msg += " That's looking good right now."
            else:
                msg += f" Need {line - total + 1:.0f} more goal{'s' if line - total + 1 > 1 else ''} to hit."
        elif "home" in bet_selection.lower() or "1" == bet_selection.strip():
            if home_score > away_score:
                msg += " Your pick is winning."
            else:
                msg += " Still need the turnaround."

    return notify(msg, title=f"GOAL {home_score}-{away_score}", level="info", category="live")


def notify_full_time(match_key: str, home_score: int, away_score: int,
                     had_bet: bool = False, bet_won: bool | None = None,
                     bet_profit: float = 0) -> dict:
    """Send a coaching-style full-time notification."""
    if had_bet and bet_won is not None:
        if bet_won:
            opener = random.choice(_FT_WIN_OPENERS)
            msg = f"{opener}\n\n{match_key}: {home_score}-{away_score}"
            if bet_profit:
                msg += f"\nThat's +${bet_profit:.2f} in the bank."
            level = "success"
        else:
            opener = random.choice(_FT_LOSS_OPENERS)
            msg = f"{opener}\n\n{match_key}: {home_score}-{away_score}"
            level = "warning"
    else:
        msg = f"Full time: {match_key} {home_score}-{away_score}"
        level = "info"

    return notify(msg, title=f"FT {home_score}-{away_score}", level=level, category="live")


def notify_retrain(mode: str, matchweek: int, promoted: bool,
                   old_ll: float = 0, new_ll: float = 0, reason: str = "") -> dict:
    """Send a coaching-style retrain notification."""
    if promoted:
        improved = new_ll < old_ll - 0.005
        if improved:
            msg = f"MW {matchweek} data is in. Retrained the model and it got sharper.\n\n"
            msg += f"Log-loss: {old_ll:.4f} -> {new_ll:.4f} ({reason})\n"
            msg += "Next matchweek's predictions will use the upgraded model."
        else:
            msg = f"MW {matchweek} done. Retrained the model — performance is steady.\n\n"
            msg += f"Log-loss: {new_ll:.4f} ({reason})\n"
            msg += "Model promoted. Predictions are fresh."
        level = "success"
    else:
        msg = f"MW {matchweek} retrain ran but the new model wasn't better.\n\n"
        msg += f"{reason}\n"
        msg += "Keeping the current model. No action needed."
        level = "warning"

    title = f"Retrain: MW {matchweek} {'upgraded' if promoted else 'unchanged'}"
    return notify(msg, title=title, level=level, category="retrain")


def notify_stale_data(source: str, age_hours: float, threshold_hours: float) -> dict:
    """Send a coaching-style stale data alert."""
    if age_hours < threshold_hours * 2:
        msg = f"Heads up — {source} data is {age_hours:.0f}h old (threshold: {threshold_hours:.0f}h).\n"
        msg += "Might want to refresh before placing any bets."
        level = "warning"
    else:
        msg = f"The {source} data is seriously stale ({age_hours:.0f}h old).\n"
        msg += "Don't trust the current predictions until this is refreshed."
        level = "error"
    return notify(msg, title=f"Stale: {source}", level=level, category="alert")


def notify_drawdown(current: float, peak: float, drawdown_pct: float) -> dict:
    """Send a coaching-style drawdown warning."""
    msg = f"Bankroll is ${current:,.2f}, down {drawdown_pct:.0f}% from the peak of ${peak:,.2f}.\n\n"
    if drawdown_pct < 15:
        msg += "Normal variance. Stay disciplined, stick to the model's edges."
    elif drawdown_pct < 25:
        msg += "Getting uncomfortable. Consider reducing stake sizes until momentum turns."
    else:
        msg += "This is a significant drawdown. Pause, review your recent bets, and only take high-confidence plays."
    return notify(msg, title=f"Drawdown: {drawdown_pct:.0f}%", level="warning", category="alert")


# ---------------------------------------------------------------------------
# CLI: test mode
# ---------------------------------------------------------------------------

def _test():
    """Send a test notification on all channels and print results."""
    import argparse

    parser = argparse.ArgumentParser(description="Test notification channels")
    parser.add_argument("--message", "-m", default="Test notification from SerieAI pipeline",
                        help="Custom test message")
    parser.add_argument("--test", action="store_true", help="Run test (default when invoked as script)")
    args = parser.parse_args()

    # Setup minimal logging
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    print("Notification channel status:")
    status = notify_status()
    for channel, info in status.items():
        configured = "YES" if info["configured"] else "NO"
        print(f"  {channel}: configured={configured} -- {info['detail']}")
    print()

    print(f"Sending test notification: \"{args.message}\"")
    results = notify(args.message, title="SerieAI Test", level="info")
    print()
    print("Results:")
    for channel, ok in results.items():
        icon = "OK" if ok else "SKIP"
        print(f"  [{icon}] {channel}")

    if results.get("telegram"):
        print("\nTelegram message sent -- check your Telegram app.")
    elif _telegram_configured():
        print("\nTelegram is configured but the message failed to send.")
        print("Check your TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
    else:
        print("\nTelegram not configured. To enable:")
        print("  1. Create a bot via @BotFather on Telegram")
        print("  2. Get your chat ID via @userinfobot")
        print("  3. Add to .env:")
        print("     TELEGRAM_BOT_TOKEN=your_bot_token")
        print("     TELEGRAM_CHAT_ID=your_chat_id")


if __name__ == "__main__":
    _test()
