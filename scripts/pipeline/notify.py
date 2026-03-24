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

# ---------------------------------------------------------------------------
# Priority / urgency tiers
# ---------------------------------------------------------------------------
# Controls delivery behavior: sound, batching, quiet hours bypass.

PRIORITY_URGENT = "urgent"    # Goal on active bet, drawdown, critical error
PRIORITY_NORMAL = "normal"    # Settlements, value bets, parlays, retrain
PRIORITY_LOW = "low"          # Pipeline status, odds snapshots, system info

# Map (category, level) -> priority. Specific overrides in notification functions.
_DEFAULT_PRIORITY = {
    ("live", "info"): PRIORITY_URGENT,
    ("live", "success"): PRIORITY_URGENT,
    ("live", "warning"): PRIORITY_URGENT,
    ("alert", "error"): PRIORITY_URGENT,
    ("alert", "critical"): PRIORITY_URGENT,
    ("alert", "warning"): PRIORITY_NORMAL,
    ("betting", "success"): PRIORITY_NORMAL,
    ("betting", "info"): PRIORITY_NORMAL,
    ("betting", "warning"): PRIORITY_NORMAL,
    ("retrain", "success"): PRIORITY_NORMAL,
    ("retrain", "warning"): PRIORITY_LOW,
    ("system", "info"): PRIORITY_LOW,
    ("system", "success"): PRIORITY_LOW,
}


def _get_priority(category: str, level: str) -> str:
    return _DEFAULT_PRIORITY.get((category, level), PRIORITY_NORMAL)


# ---------------------------------------------------------------------------
# Telegram HTML message builder
# ---------------------------------------------------------------------------
# Telegram's HTML mode supports: <b>, <i>, <u>, <s>, <code>, <pre>, <a>.
# Much more reliable than Markdown (no issues with underscores in team names).

def _html_escape(text: str) -> str:
    """Escape HTML special characters for Telegram."""
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))


class TgMsg:
    """Fluent builder for structured Telegram HTML messages.

    Usage:
        msg = (TgMsg()
            .title("Parlays Ready")
            .line("3 new picks generated")
            .sep()
            .card("Pick #1", ["Milan DC X2 @3.15", "Roma Away +1 @2.02"])
            .kv("Combined", "6.84x")
            .kv("Hit rate", "30.7%")
            .build())
    """

    def __init__(self):
        self._parts: list[str] = []

    def title(self, text: str, emoji: str = "") -> "TgMsg":
        prefix = f"{emoji} " if emoji else ""
        self._parts.append(f"{prefix}<b>{_html_escape(text)}</b>")
        return self

    def line(self, text: str = "") -> "TgMsg":
        self._parts.append(_html_escape(text) if text else "")
        return self

    def raw(self, html: str) -> "TgMsg":
        """Add pre-formatted HTML (caller must escape user content)."""
        self._parts.append(html)
        return self

    def bold(self, text: str) -> "TgMsg":
        self._parts.append(f"<b>{_html_escape(text)}</b>")
        return self

    def italic(self, text: str) -> "TgMsg":
        self._parts.append(f"<i>{_html_escape(text)}</i>")
        return self

    def sep(self) -> "TgMsg":
        self._parts.append("\u2500" * 20)
        return self

    def mini_sep(self) -> "TgMsg":
        self._parts.append("\u2504" * 16)
        return self

    def blank(self) -> "TgMsg":
        self._parts.append("")
        return self

    def kv(self, key: str, value: str, pad: int = 0) -> "TgMsg":
        """Key-value pair: bold key, normal value."""
        k = _html_escape(key)
        v = _html_escape(str(value))
        self._parts.append(f"<b>{k}:</b> {v}")
        return self

    def leg(self, match: str, selection: str, odds: float,
            market: str = "", status: str = "") -> "TgMsg":
        """Format a single bet leg line."""
        m = _html_escape(match.replace(" vs ", " \u2013 "))
        s = _html_escape(selection)
        mkt_hint = f" {_html_escape(market)}" if market else ""
        status_icon = ""
        if status == "won":
            status_icon = " \u2705"
        elif status == "lost":
            status_icon = " \u274c"
        elif status == "pending":
            status_icon = " \u23f3"
        self._parts.append(f"  \u2022 {m}{mkt_hint} <b>{s}</b> @{odds:.2f}{status_icon}")
        return self

    def stat_row(self, *pairs) -> "TgMsg":
        """Inline stat row: ('Label', 'value'), ('Label2', 'value2'), ..."""
        parts = []
        for label, value in pairs:
            parts.append(f"<b>{_html_escape(label)}:</b> {_html_escape(str(value))}")
        self._parts.append("  ".join(parts))
        return self

    def card(self, header: str, lines: list[str] = None) -> "TgMsg":
        """A card block: bold header + indented lines."""
        self._parts.append(f"\n\u25b8 <b>{_html_escape(header)}</b>")
        if lines:
            for ln in lines:
                self._parts.append(f"  {_html_escape(ln)}")
        return self

    def pnl(self, amount: float, label: str = "P&L") -> "TgMsg":
        """Formatted profit/loss line with color hint."""
        sign = "+" if amount >= 0 else ""
        icon = "\U0001f7e2" if amount >= 0 else "\U0001f534"
        self._parts.append(f"{icon} <b>{_html_escape(label)}:</b> {sign}\u20ac{amount:.2f}")
        return self

    def progress_bar(self, current: float, total: float, width: int = 10) -> "TgMsg":
        """Visual progress bar: [=====-----] 50%."""
        pct = current / total if total > 0 else 0
        filled = round(pct * width)
        bar = "\u2588" * filled + "\u2591" * (width - filled)
        self._parts.append(f"  [{bar}] {pct*100:.0f}%")
        return self

    def build(self) -> str:
        return "\n".join(self._parts)


# ---------------------------------------------------------------------------
# Notification batching
# ---------------------------------------------------------------------------

class _NotificationBatcher:
    """Aggregates rapid-fire notifications within a time window.

    Instead of sending 3 goal alerts in 60 seconds, batches them into one
    combined message. Thread-safe, auto-flushes on timeout.
    """

    def __init__(self, window_sec: float = 45.0):
        self._lock = threading.Lock()
        self._window = window_sec
        self._pending: dict[str, list[dict]] = {}  # category -> list of items
        self._timers: dict[str, threading.Timer] = {}

    def add(self, category: str, item: dict, flush_fn):
        """Add an item to the batch. Starts a timer on first item."""
        with self._lock:
            if category not in self._pending:
                self._pending[category] = []
                # Start flush timer
                timer = threading.Timer(self._window, self._flush, args=(category, flush_fn))
                timer.daemon = True
                timer.start()
                self._timers[category] = timer
            self._pending[category].append(item)

    def _flush(self, category: str, flush_fn):
        """Send batched notification and clear pending."""
        with self._lock:
            items = self._pending.pop(category, [])
            self._timers.pop(category, None)
        if items:
            try:
                flush_fn(items)
            except Exception as e:
                log.warning("Batch flush failed for %s: %s", category, e)

    def flush_now(self, category: str, flush_fn):
        """Immediately flush a category (e.g., at full-time, don't wait)."""
        with self._lock:
            timer = self._timers.pop(category, None)
            if timer:
                timer.cancel()
            items = self._pending.pop(category, [])
        if items:
            try:
                flush_fn(items)
            except Exception as e:
                log.warning("Immediate flush failed for %s: %s", category, e)


# Global batcher instance
_batcher = _NotificationBatcher(window_sec=45.0)

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
    "quiet_hours": {
        "enabled": False,
        "start": "23:00",
        "end": "07:00",
    },
    "mute_all": False,
    "sound": "Basso",
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
            # Merge with defaults to ensure all keys exist
            merged = json.loads(json.dumps(_DEFAULT_PREFERENCES))
            if "channels" in prefs:
                merged["channels"].update(prefs["channels"])
            if "categories" in prefs:
                for cat, ch_map in prefs["categories"].items():
                    if cat in merged["categories"]:
                        merged["categories"][cat].update(ch_map)
                    else:
                        merged["categories"][cat] = ch_map
            if "quiet_hours" in prefs:
                merged["quiet_hours"].update(prefs["quiet_hours"])
            if "mute_all" in prefs:
                merged["mute_all"] = prefs["mute_all"]
            if "sound" in prefs:
                merged["sound"] = prefs["sound"]
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


def _is_quiet_hours(prefs: dict) -> bool:
    """Check if the current local time is within the configured quiet hours."""
    qh = prefs.get("quiet_hours", {})
    if not qh.get("enabled", False):
        return False
    try:
        now = datetime.now()
        start_h, start_m = map(int, qh.get("start", "23:00").split(":"))
        end_h, end_m = map(int, qh.get("end", "07:00").split(":"))
        current_minutes = now.hour * 60 + now.minute
        start_minutes = start_h * 60 + start_m
        end_minutes = end_h * 60 + end_m
        if start_minutes <= end_minutes:
            # Same-day range (e.g., 09:00 - 17:00)
            return start_minutes <= current_minutes < end_minutes
        else:
            # Overnight range (e.g., 23:00 - 07:00)
            return current_minutes >= start_minutes or current_minutes < end_minutes
    except Exception as e:
        log.debug("Quiet hours check failed: %s", e)
        return False


def _should_send(channel: str, category: str) -> bool:
    """Check if a notification should be sent to a given channel for a category."""
    prefs = load_preferences()

    # Global mute — nothing sends
    if prefs.get("mute_all", False):
        return False

    # Check global channel toggle
    if not prefs.get("channels", {}).get(channel, True):
        return False

    # Quiet hours — only alert and live categories bypass for Telegram;
    # macOS always sends regardless of quiet hours.
    if channel == "telegram" and _is_quiet_hours(prefs):
        if category not in ("alert", "live"):
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


def clear_notification_history() -> bool:
    """Delete all notification history. Returns True on success."""
    try:
        with _history_lock:
            if _HISTORY_PATH.exists():
                _HISTORY_PATH.unlink()
        return True
    except Exception as e:
        log.warning("Failed to clear notification history: %s", e)
        return False


def get_notification_stats() -> dict:
    """Return aggregated notification stats: today, this week, by category, by channel."""
    try:
        all_entries = get_notification_history(limit=200)
        now = datetime.now(timezone.utc)

        today_count = 0
        week_count = 0
        by_category: dict[str, int] = {}
        by_channel: dict[str, int] = {}

        for entry in all_entries:
            ts_str = entry.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError):
                continue

            age = now - ts
            if age.days == 0:
                today_count += 1
            if age.days < 7:
                week_count += 1

            cat = entry.get("category", "system")
            by_category[cat] = by_category.get(cat, 0) + 1

            channels = entry.get("channels", {})
            if channels.get("macos"):
                by_channel["macos"] = by_channel.get("macos", 0) + 1
            if channels.get("telegram"):
                by_channel["telegram"] = by_channel.get("telegram", 0) + 1

        return {
            "today": today_count,
            "this_week": week_count,
            "by_category": by_category,
            "by_channel": by_channel,
        }
    except Exception as e:
        log.warning("Failed to compute notification stats: %s", e)
        return {"today": 0, "this_week": 0, "by_category": {}, "by_channel": {}}


# ---------------------------------------------------------------------------
# macOS notifications
# ---------------------------------------------------------------------------

_VALID_SOUNDS = {"Basso", "Glass", "Ping", "Pop", "Purr", "Sosumi", "Tink"}


def _notify_macos(message: str, title: str) -> bool:
    """Send a macOS notification via osascript. Returns True on success."""
    try:
        prefs = load_preferences()
        sound = prefs.get("sound", "Basso")
        if sound not in _VALID_SOUNDS:
            sound = "Basso"
        # Escape double quotes to prevent osascript injection
        safe_msg = message.replace('"', '\\"')[:256]
        safe_title = title.replace('"', '\\"')[:64]
        script = (
            f'display notification "{safe_msg}" '
            f'with title "{safe_title}" sound name "{sound}"'
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


def _notify_telegram(message: str, title: str, level: str = "info",
                     priority: str = PRIORITY_NORMAL,
                     html: bool = False,
                     reply_markup: dict = None) -> bool:
    """Send a Telegram notification. Returns True on success, False otherwise.

    Args:
        message: Body text (plain text or HTML if html=True).
        title: Title shown in bold at the top.
        level: info/success/warning/error/critical — controls emoji.
        priority: urgent/normal/low — controls silent delivery.
        html: If True, message is already HTML-formatted (from TgMsg builder).
              If False, wraps in HTML with escaped content.
        reply_markup: Optional inline keyboard dict for quick-action buttons.

    Silently skips if TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID are not set.
    """
    token = _load_env_key("TELEGRAM_BOT_TOKEN")
    chat_id = _load_env_key("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        return False

    emoji = _LEVEL_EMOJI.get(level, _LEVEL_EMOJI["info"])

    if html:
        # Message is already HTML-formatted from TgMsg builder
        text = f"{emoji} <b>{_html_escape(title)}</b>\n{message}"
    else:
        # Legacy plain-text message — escape for HTML
        text = f"{emoji} <b>{_html_escape(title)}</b>\n{_html_escape(message)}"

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload_dict = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    # Low-priority notifications are sent silently (no sound/vibration)
    if priority == PRIORITY_LOW:
        payload_dict["disable_notification"] = True

    # Inline keyboard buttons (quick-action buttons)
    if reply_markup:
        payload_dict["reply_markup"] = json.dumps(reply_markup) if isinstance(reply_markup, dict) else reply_markup

    payload = json.dumps(payload_dict).encode("utf-8")

    # Retry with exponential backoff (3 attempts: 0s, 2s, 4s)
    import time as _time
    for attempt in range(3):
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
                log.warning("Telegram returned HTTP %d (attempt %d)", resp.status, attempt + 1)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                pass
            log.warning("Telegram HTTP error %d (attempt %d): %s", e.code, attempt + 1, body)
            # 429 Too Many Requests: wait longer
            if e.code == 429:
                _time.sleep(5)
                continue
        except urllib.error.URLError as e:
            log.warning("Telegram connection error (attempt %d): %s", attempt + 1, e.reason)
        except Exception as e:
            log.warning("Telegram notification failed (attempt %d): %s", attempt + 1, e)

        if attempt < 2:
            _time.sleep(2 ** attempt)  # 1s, 2s backoff
            continue

    log.error("Telegram notification failed after 3 attempts")
    return False


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def notify(message: str, title: str = "SerieAI", level: str = "info",
           category: str = "system", priority: str = "",
           tg_html: str = "", tg_reply_markup: dict = None) -> dict:
    """Send notification via configured channels, respecting category preferences.

    Args:
        message: Notification body text (plain text, used for macOS + history).
        title: Short title / subject line.
        level: One of "info", "success", "warning", "error", "critical".
        category: One of "system", "betting", "live", "retrain", "alert".
        priority: "urgent", "normal", or "low". Auto-derived if empty.
        tg_html: If provided, sends this as rich HTML to Telegram instead of
                 plain-text message. Allows different formatting per channel.
        tg_reply_markup: Optional Telegram inline keyboard markup dict for
                         quick-action buttons (e.g., "Place Bet" / "Skip").

    Returns:
        Dict with channel results, e.g. {"macos": True, "telegram": True}.
    """
    if category not in VALID_CATEGORIES:
        category = "system"
    if not priority:
        priority = _get_priority(category, level)

    results = {}

    # macOS — always plain text (256 char limit handled by _notify_macos)
    if _should_send("macos", category):
        results["macos"] = _notify_macos(message, title)
    else:
        results["macos"] = False

    # Telegram — prefer rich HTML if provided, else plain text
    if _should_send("telegram", category):
        if tg_html:
            results["telegram"] = _notify_telegram(
                tg_html, title, level, priority=priority, html=True,
                reply_markup=tg_reply_markup)
        else:
            results["telegram"] = _notify_telegram(
                message, title, level, priority=priority,
                reply_markup=tg_reply_markup)
    else:
        results["telegram"] = False

    # Record to history (non-blocking, uses plain text for readability)
    try:
        _record_history(title, message, level, category, results)
    except Exception:
        pass

    return results


def notify_status() -> dict:
    """Return configuration status for each notification channel."""
    tg_info: dict = {
        "configured": _telegram_configured(),
        "detail": (
            "Ready" if _telegram_configured()
            else "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env"
        ),
    }

    # Enrich Telegram status with bot username via getMe
    if _telegram_configured():
        try:
            token = _load_env_key("TELEGRAM_BOT_TOKEN")
            url = f"https://api.telegram.org/bot{token}/getMe"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("ok"):
                    bot = data.get("result", {})
                    tg_info["bot_username"] = bot.get("username", "")
                    tg_info["connection"] = "ok"
                else:
                    tg_info["connection"] = "error"
        except Exception:
            tg_info["connection"] = "unreachable"

        # Find last successful Telegram send from history
        try:
            recent = get_notification_history(limit=50)
            for entry in recent:
                if entry.get("channels", {}).get("telegram"):
                    tg_info["last_send"] = entry.get("timestamp", "")
                    break
        except Exception:
            pass

    return {
        "macos": {"configured": True, "detail": "Always available on macOS"},
        "telegram": tg_info,
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


# ---------------------------------------------------------------------------
# Context-aware opener + CTA system
# ---------------------------------------------------------------------------

def _get_bankroll_context() -> dict:
    """Load bankroll state for contextual messaging."""
    try:
        state_path = DATA_DIR / "bankroll" / "state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            current = state.get("current_bankroll", 0)
            initial = state.get("initial_bankroll", 1000)
            peak = state.get("peak_bankroll", current)
            return {
                "current": current,
                "initial": initial,
                "peak": peak,
                "roi_pct": ((current - initial) / initial * 100) if initial else 0,
                "drawdown_pct": ((peak - current) / peak * 100) if peak else 0,
                "is_up": current > initial,
                "is_at_peak": current >= peak * 0.98,
            }
    except Exception:
        pass
    return {"current": 0, "initial": 1000, "peak": 0, "roi_pct": 0,
            "drawdown_pct": 0, "is_up": False, "is_at_peak": False}


def _get_streak() -> int:
    """Get current W/L streak from bet journal. Positive = wins, negative = losses."""
    try:
        jpath = DATA_DIR / "betting" / "bet_journal.json"
        if not jpath.exists():
            return 0
        with open(jpath) as f:
            journal = json.load(f)
        bets = journal.get("bets", {})
        if isinstance(bets, dict):
            bets = list(bets.values())
        settled = sorted(
            [b for b in bets if b.get("status") in ("won", "lost")],
            key=lambda b: b.get("settled_at", ""),
            reverse=True,
        )
        if not settled:
            return 0
        first_status = settled[0]["status"]
        count = 0
        for b in settled:
            if b["status"] == first_status:
                count += 1
            else:
                break
        return count if first_status == "won" else -count
    except Exception:
        return 0


def _smart_opener(kind: str, **ctx) -> str:
    """Pick an opener that matches the emotional context.

    Args:
        kind: 'settle_win', 'settle_loss', 'value', 'parlay', 'goal',
              'ft_win', 'ft_loss', 'morning'
        ctx: contextual data (profit, streak, edge_pct, etc.)
    """
    streak = ctx.get("streak", _get_streak())
    profit = ctx.get("profit", 0)
    br = ctx.get("bankroll_ctx") or _get_bankroll_context()

    if kind == "settle_win":
        if profit > 50:
            return "Big day. That's a serious payday."
        elif streak >= 5:
            return f"{streak} in a row. The model is locked in."
        elif streak >= 3:
            return "Another one. Momentum is building."
        elif br.get("is_at_peak"):
            return "New peak territory. Keep it steady."
        else:
            return random.choice(_SETTLEMENT_WIN_OPENERS)

    elif kind == "settle_loss":
        if streak <= -5:
            return "Rough stretch. But the math doesn't change."
        elif streak <= -3:
            return "Cold run. Stay the course, don't size up."
        elif br.get("drawdown_pct", 0) > 15:
            return "Tough one, and we're in a drawdown. Keep stakes disciplined."
        else:
            return random.choice(_SETTLEMENT_LOSS_OPENERS)

    elif kind == "value":
        edge = ctx.get("edge_pct", 0)
        if edge > 15:
            return "Significant edge found. The market looks off here."
        elif edge > 10:
            return "Solid edge. Worth a look."
        else:
            return "Spotted a gap between the model and the market."

    elif kind == "parlay":
        if br.get("is_at_peak"):
            return "Riding momentum. Here are today's picks."
        elif streak <= -3:
            return "New picks ready. Stick to the plan."
        else:
            return "Parlay slate built."

    elif kind == "goal":
        has_bet = ctx.get("has_bet", False)
        if has_bet:
            return ""  # Goal notifications with bets lead with the bet status
        return random.choice(_GOAL_OPENERS)

    elif kind == "ft_win":
        if profit > 30:
            return "Full time. Big hit."
        return random.choice(_FT_WIN_OPENERS)

    elif kind == "ft_loss":
        return random.choice(_FT_LOSS_OPENERS)

    elif kind == "morning":
        if streak >= 3:
            return f"Morning. You're on a {streak}-bet heater."
        elif streak <= -3:
            return f"Morning. Tough stretch, but today's a fresh card."
        elif br.get("is_at_peak"):
            return "Morning. Balance at peak. Let's be selective today."
        else:
            return "Morning. Here's what's on the card."

    return ""


def _edge_label(edge_pct: float) -> str:
    """Human-readable edge magnitude."""
    if edge_pct >= 20:
        return "huge edge"
    elif edge_pct >= 12:
        return "strong edge"
    elif edge_pct >= 7:
        return "solid edge"
    elif edge_pct >= 4:
        return "modest edge"
    return "thin edge"


def _time_until_kickoff(match_date: str) -> str:
    """Return human-readable time until kickoff, or empty string."""
    # match_date is typically YYYY-MM-DD without time — just return date context
    today = datetime.now().strftime("%Y-%m-%d")
    if match_date == today:
        return "today"
    tomorrow = (datetime.now() + __import__("datetime").timedelta(days=1)).strftime("%Y-%m-%d")
    if match_date == tomorrow:
        return "tomorrow"
    return ""


def _bankroll_in_context(br_ctx: dict) -> str:
    """Format bankroll with trend context: '€1,025 (↑2.5% ROI, 20% off peak)'."""
    current = br_ctx.get("current", 0)
    roi = br_ctx.get("roi_pct", 0)
    dd = br_ctx.get("drawdown_pct", 0)

    parts = [f"\u20ac{current:,.0f}"]
    if roi != 0:
        arrow = "\u2191" if roi > 0 else "\u2193"
        parts.append(f"{arrow}{abs(roi):.1f}% ROI")
    if dd > 5:
        parts.append(f"{dd:.0f}% off peak")
    elif br_ctx.get("is_at_peak"):
        parts.append("at peak")
    return " ".join([parts[0], f"({', '.join(parts[1:])})"]) if len(parts) > 1 else parts[0]


def notify_value_bets(bets: list[dict]) -> dict:
    """Send a rich notification about new value bets found."""
    if not bets:
        return {}

    # Dedup: skip if same set of bets already notified today
    import hashlib as _hl
    today_str = datetime.now().strftime("%Y-%m-%d")
    bet_sig = _hl.md5(
        "|".join(sorted(f"{b.get('match','')}-{b.get('selection','')}" for b in bets)).encode()
    ).hexdigest()[:12]
    _vb_dedup_path = DATA_DIR / ".value_bets_dedup.json"
    try:
        if _vb_dedup_path.exists():
            with open(_vb_dedup_path) as _f:
                _vb_dedup = json.load(_f)
            if _vb_dedup.get("date") == today_str and _vb_dedup.get("sig") == bet_sig:
                log.info("Value bet notification skipped (same bets already notified today)")
                return {}
    except Exception:
        pass

    sorted_bets = sorted(bets, key=lambda b: b.get("edge_pct", 0), reverse=True)
    best = sorted_bets[0]
    match = best.get("match", "?")
    best_edge = best.get("edge_pct", 0)

    opener = _smart_opener("value", edge_pct=best_edge)

    # macOS: one decisive line
    mac_msg = f"{len(bets)} value bet{'s' if len(bets) > 1 else ''}: {match} {best.get('selection','')} @ {best.get('best_odds','')} ({_edge_label(best_edge)})"

    # Telegram
    tg = TgMsg()
    tg.line(opener)
    tg.blank()

    for b in sorted_bets[:3]:
        edge = b.get("edge_pct", 0)
        conf = b.get("confidence_tier", "")
        conf_tag = f"  [{_html_escape(conf)}]" if conf else ""
        date_ctx = _time_until_kickoff(b.get("date", ""))
        time_tag = f"  \u23f0 {date_ctx}" if date_ctx else ""

        tg.raw(f"<b>{_html_escape(b.get('match', '?'))}</b>{conf_tag}{time_tag}")
        tg.raw(f"  {_html_escape(b.get('selection', '?'))} ({_html_escape(b.get('market', '?'))}) "
               f"@ <b>{b.get('best_odds', '?')}</b>  \u2014  {_html_escape(_edge_label(edge))}")
        tg.raw(f"  Model {b.get('model_prob', 0)*100:.0f}% vs market {100/b.get('best_odds',100):.0f}%"
               f"  |  \u20ac{b.get('stake_amount', 0):.0f} stake")

        # Match-specific factors
        factors = (b.get("home_factors") or []) + (b.get("away_factors") or [])
        if factors:
            hints = [f.replace("_", " ") for f in factors[:2]]
            tg.raw(f"  <i>{_html_escape(', '.join(hints))}</i>")
        tg.blank()

    if len(bets) > 3:
        tg.raw(f"+{len(bets) - 3} more in the slip")

    # CTA
    tg.blank()
    tg.raw("<i>Tap below to act:</i>")

    # Build inline keyboard with quick-action buttons per bet
    keyboard_rows = []
    for b in sorted_bets[:3]:
        match_short = b.get("match", "?")[:30]
        sel = b.get("selection", "?")[:10]
        odds = b.get("best_odds", 0)
        # Callback data: 64-byte limit
        cb_place = f"place:{match_short}|{sel}|{odds}"[:64]
        cb_skip = f"skip:{match_short}|{sel}"[:64]
        keyboard_rows.append([
            {"text": f"\u2705 Place {sel} @{odds:.2f}", "callback_data": cb_place},
            {"text": "\u274c Skip", "callback_data": cb_skip},
        ])
    # Add a "View All" button (callback, not URL — Telegram rejects localhost URLs)
    keyboard_rows.append([
        {"text": "\U0001f4ca View All Bets", "callback_data": "view:all_bets"},
    ])
    reply_markup = {"inline_keyboard": keyboard_rows}

    result = notify(
        message=mac_msg,
        title=f"Value: {match}",
        level="info",
        category="betting",
        tg_html=tg.build(),
        tg_reply_markup=reply_markup,
    )

    # Save dedup marker
    try:
        with open(_vb_dedup_path, "w") as _f:
            json.dump({"date": today_str, "sig": bet_sig}, _f)
    except Exception:
        pass

    return result


def notify_settlement(settled: int, won: int, lost: int, push: int = 0,
                      profit: float = 0, balance: float = 0,
                      best_win: dict | None = None, worst_loss: dict | None = None) -> dict:
    """Send a rich settlement notification with story and context."""
    if settled == 0:
        return {}

    is_positive = profit >= 0
    streak = _get_streak()
    br_ctx = _get_bankroll_context()
    opener = _smart_opener("settle_win" if is_positive else "settle_loss",
                           profit=profit, streak=streak, bankroll_ctx=br_ctx)
    level = "success" if is_positive else "warning"

    record = f"{won}W-{lost}L" + (f"-{push}P" if push else "")
    sign = "+" if profit >= 0 else ""

    # macOS: the one number that matters
    mac_msg = f"{record} | {sign}\u20ac{profit:.2f} | Balance: {_bankroll_in_context(br_ctx)}"

    # Telegram
    tg = TgMsg()
    tg.line(opener)
    tg.blank()

    total = won + lost + push
    tg.raw(f"<b>{_html_escape(record)}</b>")
    if total > 0:
        tg.progress_bar(won, total)

    tg.blank()
    tg.pnl(profit)
    tg.raw(f"Balance: {_html_escape(_bankroll_in_context(br_ctx))}")

    # Best/worst — tell the story, not just the number
    if best_win and is_positive:
        bw = best_win
        tg.blank()
        tg.raw(f"Best pick: {_html_escape(bw.get('match',''))} "
               f"{_html_escape(bw.get('selection',''))} "
               f"\u2192 <b>+\u20ac{bw.get('profit',0):.2f}</b>")

    if worst_loss and not is_positive:
        wl = worst_loss
        tg.raw(f"Biggest miss: {_html_escape(wl.get('match',''))} "
               f"{_html_escape(wl.get('selection',''))}")

    return notify(
        message=mac_msg,
        title=f"Settled: {sign}\u20ac{profit:.2f}",
        level=level,
        category="betting",
        tg_html=tg.build(),
    )


def notify_goal(match_key: str, scorer: str, team: str,
                home_score: int, away_score: int, minute: int,
                is_home: bool, has_bet: bool = False, bet_selection: str = "",
                bet_context: dict = None) -> dict:
    """Send a goal notification where the BET STATUS is the hero, not the goal."""

    has_active_bet = (bet_context and bet_context.get("has_bets")) or has_bet
    remaining_min = max(0, 90 - minute)
    total = home_score + away_score

    # macOS: bet-first if active
    if has_active_bet:
        mac_msg = f"{scorer} {minute}' \u2014 {match_key} {home_score}-{away_score}"
    else:
        mac_msg = f"\u26bd {scorer} ({team}) {minute}' \u2014 {match_key} {home_score}-{away_score}"

    # Telegram: structured, bet-status-first
    tg = TgMsg()

    # Score line — compact
    tg.raw(f"\u26bd <b>{_html_escape(scorer)}</b> ({_html_escape(team)}) {minute}'")
    tg.raw(f"<b>{_html_escape(match_key)}</b>  <code>{home_score} - {away_score}</code>")

    if bet_context and bet_context.get("has_bets"):
        tg.blank()
        for b in bet_context["bets"]:
            sel = b.get("selection", "")
            won = b.get("is_winning")
            commentary = b.get("commentary", "")

            # Build bet-specific status line with context
            if won is True:
                tg.raw(f"\u2705 <b>{_html_escape(sel)}</b>: {_html_escape(commentary)}")
            elif won is False:
                tg.raw(f"\u274c <b>{_html_escape(sel)}</b>: {_html_escape(commentary)}")
            else:
                # In progress — add time context
                time_ctx = f" ({remaining_min} min left)" if remaining_min > 0 else ""
                tg.raw(f"\u23f3 <b>{_html_escape(sel)}</b>: {_html_escape(commentary)}{time_ctx}")

            # Parlay tracking
            for pl in b.get("parlay_legs", []):
                tg.raw(f"   \u2514 {_html_escape(pl['parlay_id'])} "
                       f"leg {pl['leg_index']+1}/{pl['total_legs']}")

    elif has_bet:
        tg.blank()
        if "over" in bet_selection.lower():
            try:
                line = float(bet_selection.lower().replace("over", "").strip())
                if total > line:
                    tg.raw(f"\u2705 <b>{_html_escape(bet_selection)}</b> \u2014 hit!")
                else:
                    need = line - total + 1
                    tg.raw(f"\u23f3 <b>{_html_escape(bet_selection)}</b> \u2014 "
                           f"need {need:.0f} more ({remaining_min} min left)")
            except ValueError:
                tg.raw(f"\u23f3 <b>{_html_escape(bet_selection)}</b>")
        elif "home" in bet_selection.lower():
            if home_score > away_score:
                tg.raw(f"\u2705 <b>{_html_escape(bet_selection)}</b> \u2014 winning")
            else:
                tg.raw(f"\u23f3 <b>{_html_escape(bet_selection)}</b> \u2014 need the turnaround")
        else:
            tg.raw(f"\u23f3 <b>{_html_escape(bet_selection)}</b>")

    priority = PRIORITY_URGENT if has_active_bet else PRIORITY_NORMAL

    return notify(
        message=mac_msg,
        title=f"\u26bd {home_score}-{away_score}",
        level="info",
        category="live",
        priority=priority,
        tg_html=tg.build(),
    )


def notify_full_time(match_key: str, home_score: int, away_score: int,
                     had_bet: bool = False, bet_won: bool | None = None,
                     bet_profit: float = 0, bet_context: dict = None) -> dict:
    """Send a rich full-time notification with per-bet P&L and parlay tracking."""

    has_bets = (bet_context and bet_context.get("has_bets")) or had_bet

    if bet_context and bet_context.get("has_bets"):
        net = 0.0
        any_won = any_lost = False
        parlay_outcomes = {}

        tg = TgMsg()

        # Determine outcome for opener
        for b in bet_context["bets"]:
            won = b.get("is_winning")
            if won is True:
                any_won = True
                net += round(b["stake"] * (b["odds"] - 1), 2)
            elif won is False:
                any_lost = True
                net -= b["stake"]
            for pl in b.get("parlay_legs", []):
                pid = pl["parlay_id"]
                if pid not in parlay_outcomes:
                    parlay_outcomes[pid] = {"cat": pl["category"], "won": 0, "lost": 0, "total": pl["total_legs"]}
                if won is True:
                    parlay_outcomes[pid]["won"] += 1
                elif won is False:
                    parlay_outcomes[pid]["lost"] += 1

        if any_won and not any_lost:
            opener = random.choice(_FT_WIN_OPENERS)
            level = "success"
        elif any_lost and not any_won:
            opener = random.choice(_FT_LOSS_OPENERS)
            level = "warning"
        else:
            opener = "Full time. Mixed bag." if any_won else "Full time."
            level = "info"

        tg.line(opener)
        tg.raw(f"\n<b>{_html_escape(match_key)}</b>  <code>{home_score} - {away_score}</code>")
        tg.blank()

        for b in bet_context["bets"]:
            won = b.get("is_winning")
            if won is True:
                p = round(b["stake"] * (b["odds"] - 1), 2)
                tg.raw(f"  \u2705 {_html_escape(b['selection'])} <b>WON</b> \u2192 +\u20ac{p:.2f}")
            elif won is False:
                tg.raw(f"  \u274c {_html_escape(b['selection'])} <b>LOST</b> \u2192 -\u20ac{b['stake']:.2f}")
            else:
                tg.raw(f"  \u2796 {_html_escape(b['selection'])}: {_html_escape(b.get('commentary', 'undecided'))}")

        tg.blank()
        tg.pnl(net, label="Match P&L")

        # Parlay impact
        for pid, po in parlay_outcomes.items():
            if po["lost"] > 0:
                tg.raw(f"\n\U0001f4a5 <b>{_html_escape(pid)}</b>: leg lost \u2192 <b>parlay busted</b>")
            elif po["won"] > 0:
                remaining = po["total"] - po["won"]
                if remaining == 0:
                    tg.raw(f"\n\U0001f389 <b>{_html_escape(pid)}</b>: all legs hit \u2192 <b>PARLAY WON!</b>")
                else:
                    tg.raw(f"\n\u2705 <b>{_html_escape(pid)}</b>: leg hit \u2192 {remaining} leg{'s' if remaining > 1 else ''} remaining")

        mac_msg = f"{opener} {match_key} {home_score}-{away_score} | {'+' if net >= 0 else ''}\u20ac{net:.2f}"

        return notify(
            message=mac_msg,
            title=f"\U0001f3c1 FT {home_score}-{away_score}",
            level=level,
            category="live",
            priority=PRIORITY_URGENT if has_bets else PRIORITY_NORMAL,
            tg_html=tg.build(),
        )

    # Legacy / no-bet fallback
    if had_bet and bet_won is not None:
        if bet_won:
            opener = random.choice(_FT_WIN_OPENERS)
            msg = f"{opener} {match_key} {home_score}-{away_score}"
            if bet_profit:
                msg += f" | +\u20ac{bet_profit:.2f}"
            level = "success"
        else:
            opener = random.choice(_FT_LOSS_OPENERS)
            msg = f"{opener} {match_key} {home_score}-{away_score}"
            level = "warning"
    else:
        msg = f"Full time: {match_key} {home_score}-{away_score}"
        level = "info"

    return notify(msg, title=f"\U0001f3c1 FT {home_score}-{away_score}", level=level, category="live")


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


_DIGEST_POSITIVE_CLOSERS = [
    "Stay sharp.",
    "Keep trusting the process.",
    "Discipline wins long-term.",
    "Momentum is on your side.",
]

_DIGEST_NEGATIVE_CLOSERS = [
    "Variance happens. Stay disciplined.",
    "Bad days don't break good systems.",
    "Trust the edge. Tomorrow's a new day.",
    "The model's edge is still there long-term.",
]

_DIGEST_QUIET_CLOSERS = [
    "Quiet day. Rest up for the next card.",
    "No action today. The best bet is sometimes no bet.",
    "Markets are closed. Recharge.",
]


def notify_daily_digest() -> dict:
    """Generate and send end-of-day betting summary.

    Includes:
    - Today's results: matches played, bets settled, P&L
    - Bankroll status: current, ROI, drawdown
    - Tomorrow's preview: upcoming matches, value bets count, best edge
    - Streak info: current W/L streak

    Returns:
        Dict with channel results from notify().
    """
    from datetime import timedelta as _td
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + _td(days=1)).strftime("%Y-%m-%d")

    # --- Load data files gracefully ---
    def _load_json(path: Path, default=None):
        try:
            if path.exists():
                with open(path) as f:
                    return json.load(f)
        except Exception as e:
            log.debug("Digest: failed to load %s: %s", path, e)
        return default if default is not None else {}

    history = _load_json(DATA_DIR / "betting" / "history.json", default=[])
    journal_data = _load_json(DATA_DIR / "betting" / "bet_journal.json")
    bankroll = _load_json(DATA_DIR / "bankroll" / "state.json")
    predictions = _load_json(DATA_DIR / "upcoming" / "predictions.json")
    bet_slip = _load_json(DATA_DIR / "upcoming" / "unified_bet_slip.json")

    # --- Today's results from history.json ---
    today_bets = [b for b in history if b.get("settled_at", "").startswith(today)]

    # Also check bet_journal for today's settlements
    if not today_bets and journal_data.get("bets"):
        today_bets_journal = [
            b for b in journal_data["bets"].values()
            if (b.get("settled_at") or "").startswith(today) and b.get("status") in ("won", "lost", "push")
        ]
        # Convert journal format to history-like format
        today_bets = []
        for b in today_bets_journal:
            today_bets.append({
                "match": b.get("match", ""),
                "selection": b.get("selection", ""),
                "odds": b.get("odds", 0),
                "status": b.get("status", ""),
                "profit": b.get("profit", 0),
                "settled_at": b.get("settled_at", ""),
            })

    won = sum(1 for b in today_bets if b.get("status") == "won")
    lost = sum(1 for b in today_bets if b.get("status") == "lost")
    push = sum(1 for b in today_bets if b.get("status") == "push")
    total_settled = won + lost + push
    day_pnl = sum(b.get("profit", 0) for b in today_bets)

    # Best pick today
    best_pick = None
    if today_bets:
        winning_bets = [b for b in today_bets if b.get("status") == "won"]
        if winning_bets:
            best_pick = max(winning_bets, key=lambda b: b.get("profit", 0))

    # --- Bankroll status ---
    current_bankroll = bankroll.get("current_bankroll", 0)
    initial_bankroll = bankroll.get("initial_bankroll", 1000)
    peak_bankroll = bankroll.get("peak_bankroll", current_bankroll)
    roi = ((current_bankroll - initial_bankroll) / initial_bankroll * 100) if initial_bankroll else 0
    drawdown_pct = ((peak_bankroll - current_bankroll) / peak_bankroll * 100) if peak_bankroll else 0

    # --- Streak from bet_journal ---
    streak = bankroll.get("current_streak", 0)
    if not streak and journal_data.get("bets"):
        # Calculate streak from journal
        settled_bets = sorted(
            [b for b in journal_data["bets"].values() if b.get("status") in ("won", "lost")],
            key=lambda b: b.get("settled_at", ""),
            reverse=True,
        )
        if settled_bets:
            streak_status = settled_bets[0].get("status")
            streak = 0
            for b in settled_bets:
                if b.get("status") == streak_status:
                    streak += 1
                else:
                    break
            if streak_status == "lost":
                streak = -streak

    # --- Tomorrow's preview ---
    pred_list = predictions.get("predictions", [])
    tomorrow_matches = [p for p in pred_list if p.get("date", "").startswith(tomorrow)]

    slip_bets = bet_slip.get("selected_bets", [])
    tomorrow_value_bets = [b for b in slip_bets if b.get("date", "").startswith(tomorrow)]
    if not tomorrow_value_bets:
        tomorrow_value_bets = slip_bets

    # Skip digest on truly quiet days: nothing settled, nothing tomorrow
    if total_settled == 0 and not tomorrow_matches and not slip_bets:
        return {}

    best_edge_bet = None
    if tomorrow_value_bets:
        best_edge_bet = max(tomorrow_value_bets, key=lambda b: b.get("edge_pct", 0))

    # --- Compose the messages ---
    level = "success" if day_pnl >= 0 else "warning" if total_settled > 0 else "info"

    # macOS: one-liner
    if total_settled > 0:
        sign = "+" if day_pnl >= 0 else ""
        mac_msg = f"Daily: {won}W-{lost}L | {sign}\u20ac{day_pnl:.2f} | Balance: \u20ac{current_bankroll:,.0f}"
    else:
        mac_msg = f"Daily: quiet day. Balance: \u20ac{current_bankroll:,.0f}"

    # Telegram: structured visual card
    tg = TgMsg()
    tg.raw("<b>\U0001f4ca Daily Wrap-Up</b>")
    tg.sep()

    # --- Today's Results ---
    if total_settled > 0:
        record = f"{won}W-{lost}L" + (f"-{push}P" if push else "")
        tg.raw(f"\n\U0001f3af <b>Today:</b> {_html_escape(record)}")
        tg.progress_bar(won, total_settled)
        tg.pnl(day_pnl, label="Day P&L")
        if best_pick:
            bp = best_pick
            tg.raw(f"\u2b50 Best: {_html_escape(bp.get('match',''))} "
                   f"{_html_escape(bp.get('selection',''))} (+\u20ac{bp.get('profit',0):.2f})")
    else:
        tg.raw("\n\U0001f3af <b>Today:</b> No bets settled. Quiet day.")

    # --- Bankroll ---
    tg.blank()
    tg.mini_sep()
    tg.raw(f"\n\U0001f4b0 <b>Bankroll:</b> \u20ac{current_bankroll:,.2f}")
    roi_str = f"{'+' if roi >= 0 else ''}{roi:.1f}%"
    streak_str = ""
    if streak > 0:
        streak_str = f"\U0001f525 {streak}-win streak"
    elif streak < 0:
        streak_str = f"\u2744\ufe0f {abs(streak)}-loss streak"
    tg.stat_row(("ROI", roi_str), ("Streak", streak_str or "flat"))

    if drawdown_pct > 5:
        tg.raw(f"  \u26a0\ufe0f Drawdown: {drawdown_pct:.0f}% from peak (\u20ac{peak_bankroll:,.0f})")

    # --- Tomorrow's Preview ---
    tg.blank()
    tg.mini_sep()
    if tomorrow_matches:
        tg.raw(f"\n\U0001f4c5 <b>Tomorrow:</b> {len(tomorrow_matches)} match{'es' if len(tomorrow_matches) > 1 else ''}")
        if best_edge_bet:
            tg.raw(f"\u26a1 Biggest edge: {_html_escape(best_edge_bet.get('match',''))} "
                   f"{_html_escape(best_edge_bet.get('selection',''))} "
                   f"@ {best_edge_bet.get('best_odds','?')} "
                   f"(<b>{best_edge_bet.get('edge_pct',0):.0f}%</b>)")
        if tomorrow_value_bets:
            tg.raw(f"\U0001f4cb {len(tomorrow_value_bets)} value bet{'s' if len(tomorrow_value_bets) > 1 else ''} in the slip")
    elif slip_bets:
        next_date = min(b.get("date", "") for b in slip_bets if b.get("date", ""))
        tg.raw(f"\n\U0001f4c5 <b>Next card:</b> {_html_escape(next_date)}")
        tg.raw(f"\U0001f4cb {len(slip_bets)} value bet{'s' if len(slip_bets) > 1 else ''} queued")
    else:
        tg.raw(f"\n\U0001f4c5 No upcoming matches on the radar.")

    # Closer
    tg.blank()
    if total_settled > 0 and day_pnl >= 0:
        tg.italic(random.choice(_DIGEST_POSITIVE_CLOSERS))
    elif total_settled > 0:
        tg.italic(random.choice(_DIGEST_NEGATIVE_CLOSERS))
    else:
        tg.italic(random.choice(_DIGEST_QUIET_CLOSERS))

    return notify(
        message=mac_msg,
        title="Daily Digest",
        level=level,
        category="betting",
        tg_html=tg.build(),
    )


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
# Morning Briefing
# ---------------------------------------------------------------------------

_MORNING_OPENERS = [
    "Good morning. Here's what's on the card today.",
    "Rise and grind. Match day briefing incoming.",
    "Morning. Let's see what the markets are offering.",
]


def notify_morning_briefing() -> dict:
    """Send a pre-match-day briefing: today's matches, active bets, bankroll.

    Designed for the bettor checking their phone over morning coffee.
    Should be triggered by scheduler ~2 hours before first kickoff.
    """
    from datetime import timedelta as _td
    today = datetime.now().strftime("%Y-%m-%d")

    def _load(path, default=None):
        try:
            if path.exists():
                with open(path) as f:
                    return json.load(f)
        except Exception:
            pass
        return default if default is not None else {}

    # Load data
    predictions = _load(DATA_DIR / "upcoming" / "predictions.json")
    bet_slip = _load(DATA_DIR / "upcoming" / "unified_bet_slip.json")
    journal = _load(DATA_DIR / "betting" / "bet_journal.json")
    bankroll_state = _load(DATA_DIR / "bankroll" / "state.json")
    parlay_report = _load(DATA_DIR / "betting" / "parlay_report.json")
    sentiment = _load(DATA_DIR / "upcoming" / "sentiment_analysis.json")

    # Today's matches
    pred_list = predictions.get("predictions", [])
    today_matches = [p for p in pred_list if p.get("date", "").startswith(today)]

    # Active bets (pending from journal)
    journal_bets = journal.get("bets", {})
    if isinstance(journal_bets, dict):
        journal_bets = list(journal_bets.values())
    pending = [b for b in journal_bets if b.get("status") == "pending"]

    # Value bets in slip
    slip_bets = bet_slip.get("selected_bets", [])

    # Bankroll
    br = bankroll_state.get("current_bankroll", 0)
    initial = bankroll_state.get("initial_bankroll", 1000)
    roi = ((br - initial) / initial * 100) if initial else 0

    # Top parlays
    top_picks = parlay_report.get("top_picks", []) if parlay_report else []

    # Injuries from sentiment
    injury_alerts = []
    sent_matches = sentiment.get("matches", [])
    if isinstance(sent_matches, list):
        for s in sent_matches:
            if s.get("home_injury_impact", 0) < -40:
                injury_alerts.append(f"{s.get('home_team', '?')}: major injury crisis")
            if s.get("away_injury_impact", 0) < -40:
                injury_alerts.append(f"{s.get('away_team', '?')}: major injury crisis")

    # Don't send on no-match days with no active bets
    n_matches = len(today_matches)
    n_bets = len(pending)
    if n_matches == 0 and n_bets == 0:
        return {}

    br_ctx = {"current": br, "initial": initial,
              "peak": bankroll_state.get("peak_bankroll", br),
              "roi_pct": roi,
              "drawdown_pct": ((bankroll_state.get("peak_bankroll", br) - br) / bankroll_state.get("peak_bankroll", br) * 100) if bankroll_state.get("peak_bankroll", br) else 0,
              "is_up": br > initial,
              "is_at_peak": br >= bankroll_state.get("peak_bankroll", br) * 0.98}
    streak = _get_streak()
    opener = _smart_opener("morning", streak=streak, bankroll_ctx=br_ctx)

    # macOS: one headline
    mac_msg = f"{n_matches} games today, {n_bets} active bets. {_bankroll_in_context(br_ctx)}"

    # Telegram: concise briefing - headline + key items + CTA
    tg = TgMsg()
    tg.raw(f"<b>{_html_escape(opener)}</b>")
    tg.blank()

    # One-line summary
    tg.raw(f"{n_matches} match{'es' if n_matches != 1 else ''} today  |  "
           f"Balance: {_html_escape(_bankroll_in_context(br_ctx))}")

    # Active bets - compact, max 3
    if pending:
        total_at_risk = sum(b.get("stake", 0) for b in pending)
        tg.blank()
        tg.raw(f"<b>{len(pending)} active bet{'s' if len(pending) > 1 else ''}</b> "
               f"(\u20ac{total_at_risk:.0f} at risk)")
        for b in pending[:3]:
            tg.raw(f"  {_html_escape(b.get('match', '?'))} \u2014 "
                   f"{_html_escape(b.get('selection', '?'))} @ {b.get('odds', 0):.2f}")
        if len(pending) > 3:
            tg.raw(f"  <i>+{len(pending) - 3} more</i>")

    # Best edge - one line
    if slip_bets:
        best_edge = max(slip_bets, key=lambda b: b.get("edge_pct", 0))
        edge_pct = best_edge.get("edge_pct", 0)
        tg.blank()
        tg.raw(f"Top edge: {_html_escape(best_edge.get('match', '?'))} "
               f"{_html_escape(best_edge.get('selection', '?'))} "
               f"@ {best_edge.get('best_odds', '?')} "
               f"({_html_escape(_edge_label(edge_pct))})")

    # Injury watch - only critical
    if injury_alerts:
        tg.blank()
        for alert in injury_alerts[:2]:
            tg.raw(f"\u26a0\ufe0f {_html_escape(alert)}")

    # CTA
    tg.blank()
    tg.raw("<i>/bets for full list  |  /bankroll for details</i>")

    return notify(
        message=mac_msg,
        title="\u2600\ufe0f Morning Briefing",
        level="info",
        category="betting",
        tg_html=tg.build(),
    )



# ---------------------------------------------------------------------------
# Live Matchday P&L Summary
# ---------------------------------------------------------------------------

def notify_matchday_update() -> dict:
    """Send a periodic matchday summary: running P&L, active bets, parlay status.

    Call every 90 minutes during match day, or at natural breaks (HT/FT of
    last match). Quick at-a-glance "how's my day going" message.
    """
    today = datetime.now().strftime("%Y-%m-%d")

    def _load(path, default=None):
        try:
            if path.exists():
                with open(path) as f:
                    return json.load(f)
        except Exception:
            pass
        return default if default is not None else {}

    journal = _load(DATA_DIR / "betting" / "bet_journal.json")
    bankroll_state = _load(DATA_DIR / "bankroll" / "state.json")

    journal_bets = journal.get("bets", {})
    if isinstance(journal_bets, dict):
        journal_bets = list(journal_bets.values())

    # Today's settled
    settled_today = [b for b in journal_bets
                     if b.get("status") in ("won", "lost", "push")
                     and (b.get("settled_at") or "").startswith(today)]
    pending = [b for b in journal_bets if b.get("status") == "pending"]

    won = sum(1 for b in settled_today if b["status"] == "won")
    lost = sum(1 for b in settled_today if b["status"] == "lost")
    day_pnl = sum(b.get("profit", 0) for b in settled_today)
    pending_risk = sum(b.get("stake", 0) for b in pending)

    br = bankroll_state.get("current_bankroll", 0)

    if not settled_today and not pending:
        return {}  # Nothing to report

    # macOS
    record = f"{won}W-{lost}L" if settled_today else "0 settled"
    sign = "+" if day_pnl >= 0 else ""
    mac_msg = f"Matchday: {record} | {sign}\u20ac{day_pnl:.2f} | {len(pending)} bets in play"

    # Telegram
    tg = TgMsg()
    tg.raw("<b>\U0001f4ca Matchday Update</b>")
    tg.mini_sep()

    if settled_today:
        tg.raw(f"\n\u2705 Settled: <b>{won}W-{lost}L</b>")
        tg.pnl(day_pnl, label="Running P&L")

    if pending:
        tg.blank()
        tg.raw(f"\u23f3 <b>Still in play:</b> {len(pending)} bet{'s' if len(pending) > 1 else ''}")
        for b in pending[:4]:
            tg.raw(f"  \u2022 {_html_escape(b.get('match', '?'))} \u2014 "
                   f"{_html_escape(b.get('selection', '?'))} "
                   f"@ {b.get('odds', 0):.2f}")
        if len(pending) > 4:
            tg.raw(f"  <i>+{len(pending) - 4} more</i>")
        tg.kv("At risk", f"\u20ac{pending_risk:.0f}")

    tg.blank()
    tg.raw(f"\U0001f4b0 Balance: \u20ac{br:,.2f}")

    return notify(
        message=mac_msg,
        title="\U0001f4ca Matchday Update",
        level="info",
        category="betting",
        priority=PRIORITY_LOW,
        tg_html=tg.build(),
    )


# ---------------------------------------------------------------------------
# Pipeline lifecycle narratives
# ---------------------------------------------------------------------------

_PIPELINE_START_OPENERS = [
    "Firing up the engine.",
    "Time to scan the markets.",
    "Pipeline is spinning up — let's see what the data says.",
    "Running the full sweep.",
]

_PIPELINE_DONE_OPENERS = [
    "All done. Here's the picture.",
    "Pipeline wrapped up.",
    "Data is fresh, predictions are in.",
    "Full scan complete.",
]

_ODDS_UPDATED_OPENERS = [
    "Odds snapshot locked in.",
    "Market picture updated.",
    "Fresh odds captured.",
]

_LINEUP_OPENERS = [
    "Lineups are in.",
    "Confirmed XIs just dropped.",
    "We've got the real starting lineups now.",
]

_PREDICTIONS_READY_OPENERS = [
    "Predictions refreshed with the latest intel.",
    "Model has re-run with confirmed data.",
    "Fresh predictions ready to go.",
]

_KICKOFF_OPENERS = [
    "We're underway!",
    "Kick off!",
    "The whistle's gone.",
]

_HALFTIME_OPENERS = [
    "Half-time.",
    "Breather.",
    "That's the interval.",
]

_PARLAY_OPENERS = [
    "Parlays are cooked.",
    "Multi-leg combos ready.",
    "Parlay slate built.",
]

_BANKROLL_MILESTONE_UP = [
    "Bankroll milestone hit.",
    "We're climbing.",
    "The grind is paying off.",
]

_BANKROLL_MILESTONE_DOWN = [
    "Bankroll dropped through a level.",
    "Rough stretch.",
    "We've given some back.",
]


def notify_pipeline_start() -> dict:
    """Pipeline start is routine — log only, don't interrupt the user.

    The user cares about the RESULTS (value bets, parlays), not that
    the pipeline started running. They'll get notified when it's done.
    """
    log.info("Pipeline starting (no notification — routine)")
    return {}


def notify_pipeline_done(n_predictions: int = 0, n_value_bets: int = 0,
                         elapsed_sec: float = 0) -> dict:
    """Send a coaching-style notification when the pipeline finishes."""
    opener = random.choice(_PIPELINE_DONE_OPENERS)
    parts = []
    if n_predictions:
        parts.append(f"{n_predictions} predictions generated")
    if n_value_bets:
        parts.append(f"{n_value_bets} value bets found")
    if elapsed_sec:
        parts.append(f"completed in {elapsed_sec:.0f}s")
    detail = ". ".join(parts) + "." if parts else ""
    msg = f"{opener}\n\n{detail}"
    return notify(msg, title="Pipeline Done", level="success", category="system")


def notify_odds_snapshot(n_matches: int = 0, n_bookmakers: int = 0) -> dict:
    """Odds snapshots are routine — log only, don't notify.

    Odds refresh every few hours. Notifying on each one is noise.
    The user cares about the VALUE BETS found from the odds, not the fetch itself.
    """
    log.info("Odds snapshot: %d matches, %d bookmakers (no notification — routine)",
             n_matches, n_bookmakers)
    return {}


def notify_lineups_confirmed(matches: str = "", changes: str = "") -> dict:
    """Send a coaching-style notification when lineups are confirmed."""
    opener = random.choice(_LINEUP_OPENERS)
    msg = f"{opener}\n\n"
    if matches:
        msg += f"Confirmed for: {matches}"
    if changes:
        msg += f"\nKey changes: {changes}"
    return notify(msg, title="Lineups Confirmed", level="info", category="system")


def notify_predictions_ready(n_matches: int = 0) -> dict:
    """Send a coaching-style notification when predictions are refreshed."""
    opener = random.choice(_PREDICTIONS_READY_OPENERS)
    detail = f" {n_matches} matches updated." if n_matches else ""
    msg = f"{opener}{detail}"
    return notify(msg, title="Predictions Ready", level="info", category="system")


def notify_kickoff(match_key: str, bet_context: dict = None) -> dict:
    """Send a coaching-style notification when a match kicks off."""
    opener = random.choice(_KICKOFF_OPENERS)
    msg = f"{opener} {match_key}"
    if bet_context and bet_context.get("has_bets"):
        msg += "\n\nYour bets:"
        for b in bet_context["bets"]:
            line = f"  \u00b7 {b['selection']} @ {b['odds']:.2f} (\u20ac{b['stake']:.0f})"
            if b.get("parlay_legs"):
                for pl in b["parlay_legs"]:
                    line += f"\n    \u2514 Leg {pl['leg_index']+1} of {pl['parlay_id']} ({pl['category']})"
            msg += f"\n{line}"
        msg += f"\n\nTotal exposure: \u20ac{bet_context['total_stake']:.0f}"
    return notify(msg, title="Match Started", level="info", category="live")


def notify_halftime(match_key: str, home_score: int, away_score: int,
                    bet_context: dict = None) -> dict:
    """Send a coaching-style notification at half-time."""
    opener = random.choice(_HALFTIME_OPENERS)
    msg = f"{opener} {match_key} {home_score}-{away_score}"
    if bet_context and bet_context.get("has_bets"):
        msg += "\n"
        for b in bet_context["bets"]:
            msg += f"\n  \u00b7 {b['selection']}: {b['commentary']}"
            if b.get("parlay_legs"):
                for pl in b["parlay_legs"]:
                    status = "undecided"
                    if b.get("is_winning") is True:
                        status = "looking good"
                    elif b.get("is_winning") is False:
                        status = "in danger"
                    msg += f"\n    \u2514 {pl['parlay_id']}: leg {pl['leg_index']+1} {status}, {pl['total_legs']} legs total"
    return notify(msg, title=f"HT {home_score}-{away_score}", level="info", category="live")


_PRE_KICKOFF_OPENERS = [
    "Match day briefing.",
    "Here's your pre-match rundown.",
    "Time to focus up.",
]


def notify_pre_kickoff_bets(match_key: str, bet_context: dict) -> dict:
    """Send a T-30 bet briefing before kickoff.

    Summarises all bets + parlay involvement for a match.
    Only call this when bet_context['has_bets'] is True.
    """
    opener = random.choice(_PRE_KICKOFF_OPENERS)
    msg = f"{opener}\n\n{match_key} kicks off in ~30 min.\n\nYour bets:"
    parlay_summary = {}
    for b in bet_context.get("bets", []):
        conf_tag = f" [{b['confidence']}]" if b.get("confidence") else ""
        msg += f"\n  \u00b7 {b['selection']} @ {b['odds']:.2f} (\u20ac{b['stake']:.0f}){conf_tag}"
        for pl in b.get("parlay_legs", []):
            pid = pl["parlay_id"]
            if pid not in parlay_summary:
                parlay_summary[pid] = {
                    "category": pl["category"],
                    "total_legs": pl["total_legs"],
                    "combined_odds": pl.get("combined_odds", 0),
                }

    if parlay_summary:
        msg += "\n"
        for pid, ps in parlay_summary.items():
            odds_str = f", {ps['combined_odds']:.2f}x" if ps["combined_odds"] else ""
            msg += f"\n  Parlay: {pid} ({ps['category']}, {ps['total_legs']} legs{odds_str})"

    msg += f"\n\nTotal exposure: \u20ac{bet_context['total_stake']:.0f}"
    return notify(msg, title=f"Pre-match: {match_key}", level="info", category="live")


def notify_parlays_ready(n_parlays: int = 0, best_odds: float = 0,
                         best_prob: float = 0,
                         parlay_report: dict | None = None) -> dict:
    """Send a rich notification when parlays are generated.

    Telegram gets a visual card per pick. macOS gets a brief summary.
    Only sends if the report was regenerated (not cached/stale).
    """
    if parlay_report:
        n_parlays = parlay_report.get("total_parlays", n_parlays)
        if not parlay_report.get("regenerated", True):
            log.info("Parlay notification skipped (report not regenerated)")
            return {"macos": False, "telegram": False, "skipped": True}

    # Dedup
    today_str = datetime.now().strftime("%Y-%m-%d")
    # Dedup: hash the top picks content, not just count
    import hashlib as _hl
    picks_sig = ""
    if top_picks:
        picks_sig = _hl.md5(
            "|".join(p.get("parlay", {}).get("id", "") for p in top_picks).encode()
        ).hexdigest()[:12]
    else:
        picks_sig = str(n_parlays)

    dedup_file = DATA_DIR / ".parlay_notify_dedup.json"
    try:
        if dedup_file.exists():
            with open(dedup_file) as f:
                dedup = json.load(f)
            if dedup.get("date") == today_str and dedup.get("sig") == picks_sig:
                log.info("Parlay notification skipped (same picks already sent today)")
                return {"macos": False, "telegram": False, "skipped": True}
    except Exception:
        pass

    top_picks = parlay_report.get("top_picks", []) if parlay_report else []
    title = "\U0001f3b0 Parlay Picks"

    # --- Build rich Telegram HTML ---
    tg = TgMsg()
    if top_picks:
        tg.line(f"{_smart_opener('parlay')} {n_parlays} parlays scanned.")
        tg.blank()

        for pick in top_picks[:3]:
            p = pick.get("parlay", {})
            legs = p.get("legs", [])
            cat_label = pick.get("category", "").replace("_", " ").title()
            quality = p.get("parlay_quality", 0)
            combined = p.get("combined_odds", 0)
            hp = p.get("hit_probability", {})
            hit_pct = hp.get("median", hp.get("copula_adjusted", 0))
            if hit_pct <= 1:
                hit_pct *= 100
            stake = p.get("stake", 0)

            rank_emoji = ["\U0001f947", "\U0001f948", "\U0001f949"][pick.get("rank", 1) - 1] if pick.get("rank", 1) <= 3 else "\u25b8"
            tg.raw(f"{rank_emoji} <b>{_html_escape(cat_label)}</b>  <i>q={quality:.0f}</i>")

            for leg_data in legs:
                mkt = leg_data.get("market", "")
                mkt_tag = ""
                if mkt == "double_chance":
                    mkt_tag = "DC"
                elif mkt == "btts":
                    mkt_tag = "BTTS"
                elif mkt == "draw_no_bet":
                    mkt_tag = "DNB"
                tg.leg(leg_data.get("match", "?"),
                       leg_data.get("selection", "?"),
                       leg_data.get("odds", 0),
                       market=mkt_tag)

            tg.stat_row(
                ("Odds", f"{combined:.2f}x"),
                ("Hit", f"{hit_pct:.0f}%"),
                ("Stake", f"\u20ac{stake:.0f}"),
            )

            why = pick.get("why", [])
            if why:
                tg.raw(f"  <i>\u2192 {_html_escape(why[0])}</i>")
            tg.blank()

        # Category summary bar
        if parlay_report:
            cats = parlay_report.get("categories", {})
            summary_parts = []
            for cname, citems in cats.items():
                if isinstance(citems, list) and citems:
                    summary_parts.append(f"{cname.replace('_',' ').title()}: {len(citems)}")
            if summary_parts:
                tg.mini_sep()
                tg.raw(f"<i>{_html_escape(' | '.join(summary_parts))}</i>")

        # CTA
        tg.blank()
        tg.raw("<i>Review on dashboard \u2192 Betting tab \u2192 Parlays</i>")

        tg_html = tg.build()
    else:
        tg_html = ""

    # macOS: brief
    n_picks = len(top_picks)
    mac_msg = f"{n_picks} parlay picks ready." if n_picks else f"{n_parlays} parlays generated."

    result = notify(
        message=mac_msg,
        title="Parlays Ready",
        level="info",
        category="betting",
        tg_html=tg_html if tg_html else "",
    )

    # Update dedup marker
    try:
        with open(dedup_file, "w") as f:
            json.dump({"date": today_str, "sig": picks_sig}, f)
    except Exception:
        pass

    return result


def notify_bankroll_milestone(old_balance: float, new_balance: float) -> dict:
    """Send a coaching-style notification when bankroll crosses a milestone.

    Milestones: every $100 going up, every $500 drop going down.
    Returns empty dict if no milestone was crossed.
    """
    # Check upward milestones (every $100)
    old_hundred = int(old_balance // 100)
    new_hundred = int(new_balance // 100)

    if new_hundred > old_hundred and new_balance > old_balance:
        milestone = new_hundred * 100
        opener = random.choice(_BANKROLL_MILESTONE_UP)
        msg = f"{opener}\n\nBankroll crossed ${milestone:,.0f} (now ${new_balance:,.2f})."
        msg += "\nStay disciplined — the edge compounds."
        return notify(msg, title=f"Milestone: ${milestone:,.0f}", level="success", category="betting")

    # Check downward milestones (every $500)
    old_five = int(old_balance // 500)
    new_five = int(new_balance // 500)

    if new_five < old_five and new_balance < old_balance:
        milestone = (new_five + 1) * 500
        opener = random.choice(_BANKROLL_MILESTONE_DOWN)
        msg = f"{opener}\n\nBankroll dropped below ${milestone:,.0f} (now ${new_balance:,.2f})."
        msg += "\nStick to the system. Variance is part of the game."
        return notify(msg, title=f"Below ${milestone:,.0f}", level="warning", category="betting")

    return {}


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
    parser.add_argument("--digest", action="store_true", help="Send daily digest notification")
    args = parser.parse_args()

    # Setup logging only when run as __main__ (don't hijack parent logger)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.digest:
        print("Sending daily digest...")
        results = notify_daily_digest()
        print()
        print("Results:")
        for channel, ok in results.items():
            icon = "OK" if ok else "SKIP"
            print(f"  [{icon}] {channel}")
        return

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


# ---------------------------------------------------------------------------
# NEW NOTIFICATIONS (Mar 24 2026) — Critical missing functionality
# ---------------------------------------------------------------------------

def notify_bet_settled(bet: dict, result_score: str = "") -> dict:
    """Per-bet settlement notification — immediate feedback on each bet.

    Fires for each individual bet when it settles, not just the batch summary.
    """
    match = bet.get("match", "?")
    selection = bet.get("selection", "?")
    odds = bet.get("odds", 0)
    stake = bet.get("stake", 0)
    status = bet.get("status", "unknown")
    profit = bet.get("profit", 0)

    if status == "won":
        emoji = "\u2705"
        level = "success"
        pnl_str = f"+\u20ac{profit:.2f}"
    elif status == "lost":
        emoji = "\u274c"
        level = "warning"
        pnl_str = f"-\u20ac{stake:.2f}"
        profit = -stake
    else:
        emoji = "\u2796"
        level = "info"
        pnl_str = "PUSH"

    mac_msg = f"{emoji} {match} | {selection} @{odds:.2f} | {pnl_str}"

    tg = TgMsg()
    tg.raw(f"{emoji} <b>{_html_escape(match)}</b>")
    tg.raw(f"{_html_escape(selection)} @{odds:.2f} | "
           f"Stake \u20ac{stake:.2f}")
    if result_score:
        tg.raw(f"Score: <b>{_html_escape(result_score)}</b>")
    tg.pnl(profit)

    return notify(
        message=mac_msg,
        title=f"Bet {'WON' if status == 'won' else 'LOST' if status == 'lost' else 'PUSH'}",
        level=level,
        category="betting",
        tg_html=tg.build(),
        priority=PRIORITY_NORMAL,
    )


def notify_loss_streak(streak_count: int, total_loss: float = 0,
                       recent_bets: list = None) -> dict:
    """Alert when on a losing streak — psychological protection.

    Fires when streak reaches 3, 5, 7+ consecutive losses.
    """
    if streak_count < 3:
        return {}

    if streak_count >= 7:
        severity = "critical"
        coaching = ("Significant losing streak. Pause betting for today. "
                    "Review recent bets for systematic issues. "
                    "The model may need recalibration, or this is just variance.")
    elif streak_count >= 5:
        severity = "warning"
        coaching = ("5+ losses in a row. Consider reducing stakes to half "
                    "until the streak breaks. Trust the edge, but protect the bankroll.")
    else:
        severity = "info"
        coaching = ("3-loss streak. Normal variance — happens to every system. "
                    "Stay disciplined. Don't chase losses.")

    mac_msg = (f"{'🔴' * min(streak_count, 5)} {streak_count}-loss streak | "
               f"\u20ac{total_loss:.0f} lost | {coaching[:60]}...")

    tg = TgMsg()
    tg.title(f"{streak_count}-Loss Streak", emoji="\u26a0\ufe0f")
    tg.blank()
    tg.raw(f"{'🔴 ' * min(streak_count, 7)}")
    if total_loss > 0:
        tg.raw(f"Total lost: <b>-\u20ac{total_loss:.2f}</b>")
    tg.blank()
    tg.italic(coaching)

    if recent_bets:
        tg.blank()
        tg.bold("Recent losses:")
        for b in recent_bets[-3:]:
            tg.raw(f"  \u274c {_html_escape(b.get('match',''))} "
                   f"{_html_escape(b.get('selection',''))} @{b.get('odds',0):.2f}")

    return notify(
        message=mac_msg,
        title=f"Loss Streak: {streak_count} in a row",
        level="warning" if streak_count < 5 else "error",
        category="alert",
        tg_html=tg.build(),
        priority=PRIORITY_URGENT if streak_count >= 5 else PRIORITY_NORMAL,
    )


def notify_weekly_accuracy(correct: int, total: int,
                           accuracy_pct: float,
                           best_prediction: dict = None,
                           worst_miss: dict = None,
                           roi_pct: float = 0) -> dict:
    """Weekly model accuracy report — feedback loop for confidence.

    Should fire Monday morning or after last matchweek match settles.
    """
    if total == 0:
        return {}

    if accuracy_pct >= 65:
        mood = "Excellent week. Model is sharp."
    elif accuracy_pct >= 55:
        mood = "Solid week. Edge is holding."
    elif accuracy_pct >= 45:
        mood = "Below average. Monitor for drift."
    else:
        mood = "Rough week. Check if model needs recalibration."

    mac_msg = (f"Weekly: {correct}/{total} ({accuracy_pct:.0f}%) | "
               f"ROI: {roi_pct:+.1f}% | {mood}")

    tg = TgMsg()
    tg.title("Weekly Accuracy Report", emoji="\U0001f4ca")
    tg.blank()
    tg.raw(f"<b>{correct}/{total}</b> correct ({accuracy_pct:.1f}%)")
    tg.progress_bar(correct, total)
    if roi_pct != 0:
        tg.pnl(roi_pct)  # shows as +/-
    tg.blank()
    tg.italic(mood)

    if best_prediction:
        tg.blank()
        tg.raw(f"\u2b50 Best call: {_html_escape(best_prediction.get('match',''))}")
    if worst_miss:
        tg.raw(f"\U0001f4a5 Biggest miss: {_html_escape(worst_miss.get('match',''))}")

    return notify(
        message=mac_msg,
        title=f"Weekly: {accuracy_pct:.0f}% accuracy",
        level="success" if accuracy_pct >= 55 else "warning",
        category="betting",
        tg_html=tg.build(),
    )


def notify_lineup_impact(match: str, old_pred: dict = None, new_pred: dict = None,
                         missing_players: list = None) -> dict:
    """Enhanced lineup confirmation — shows how prediction shifted.

    Fires after lineups are confirmed AND predictions re-run.
    """
    if not old_pred or not new_pred:
        return {}

    old_h = old_pred.get("prob_H", 0)
    new_h = new_pred.get("prob_H", 0)
    old_d = old_pred.get("prob_D", 0)
    new_d = new_pred.get("prob_D", 0)
    old_a = old_pred.get("prob_A", 0)
    new_a = new_pred.get("prob_A", 0)

    max_shift = max(abs(new_h - old_h), abs(new_d - old_d), abs(new_a - old_a))
    if max_shift < 0.02:  # <2% shift — not worth notifying
        return {}

    def _arrow(old, new):
        diff = new - old
        if abs(diff) < 0.01:
            return "\u2192"
        return "\u2b06\ufe0f" if diff > 0 else "\u2b07\ufe0f"

    mac_msg = (f"Lineup impact: {match} | "
               f"H {old_h:.0%}{_arrow(old_h,new_h)}{new_h:.0%} "
               f"D {old_d:.0%}{_arrow(old_d,new_d)}{new_d:.0%} "
               f"A {old_a:.0%}{_arrow(old_a,new_a)}{new_a:.0%}")

    tg = TgMsg()
    tg.title(f"Lineup Impact: {match}", emoji="\U0001f4cb")
    tg.blank()
    tg.raw(f"H: {old_h:.1%} {_arrow(old_h,new_h)} <b>{new_h:.1%}</b>")
    tg.raw(f"D: {old_d:.1%} {_arrow(old_d,new_d)} <b>{new_d:.1%}</b>")
    tg.raw(f"A: {old_a:.1%} {_arrow(old_a,new_a)} <b>{new_a:.1%}</b>")

    if missing_players:
        tg.blank()
        tg.bold("Key absences:")
        for p in missing_players[:5]:
            tg.raw(f"  \u274c {_html_escape(p)}")

    if max_shift > 0.05:
        tg.blank()
        tg.italic("Significant shift — review your bets on this match.")

    return notify(
        message=mac_msg,
        title=f"Lineup shift: {match}",
        level="warning" if max_shift > 0.05 else "info",
        category="betting",
        tg_html=tg.build(),
    )


def notify_clv_degradation(current_clv: float, previous_clv: float,
                           period: str = "2 weeks",
                           markets_at_risk: list = None) -> dict:
    """Alert when CLV edge is declining — early warning for model staleness.

    Fires when rolling CLV drops significantly between periods.
    """
    drop = previous_clv - current_clv
    if drop < 1.5:  # Less than 1.5pp drop — not worth alerting
        return {}

    if current_clv < 1.0:
        severity = "critical"
        coaching = ("CLV near zero — the model's edge over closing lines is almost gone. "
                    "Consider pausing high-stakes bets until next retrain.")
    elif drop > 3.0:
        severity = "warning"
        coaching = (f"CLV dropped {drop:.1f}pp in {period}. Edge is narrowing. "
                    "Market may be adapting. Monitor closely.")
    else:
        severity = "info"
        coaching = f"CLV eased {drop:.1f}pp. Normal fluctuation if within {period}."

    mac_msg = (f"CLV Alert: {current_clv:+.1f}% (was {previous_clv:+.1f}%) | "
               f"Drop: {drop:.1f}pp | {coaching[:60]}")

    tg = TgMsg()
    tg.title("CLV Edge Degradation", emoji="\U0001f4c9")
    tg.blank()
    tg.raw(f"Current: <b>{current_clv:+.2f}%</b> (was {previous_clv:+.2f}%)")
    tg.raw(f"Drop: <b>{drop:.1f}pp</b> over {_html_escape(period)}")
    tg.blank()
    tg.italic(coaching)

    if markets_at_risk:
        tg.blank()
        tg.bold("Markets losing edge:")
        for m in markets_at_risk[:5]:
            tg.raw(f"  \u26a0\ufe0f {_html_escape(m)}")

    return notify(
        message=mac_msg,
        title=f"CLV Drop: {drop:.1f}pp",
        level="warning" if severity == "warning" else "info",
        category="alert",
        tg_html=tg.build(),
        priority=PRIORITY_URGENT if severity == "critical" else PRIORITY_NORMAL,
    )


def notify_matchweek_summary(matchweek: int = 0) -> dict:
    """End-of-matchweek summary — all bets placed, results, P&L.

    Fires after the last match of a matchweek settles.
    Shows every bet from that matchweek with result and profit.
    """
    try:
        import json as _json
        from pathlib import Path as _Path

        journal_path = _Path(__file__).parent.parent.parent / "data" / "betting" / "bet_journal.json"
        if not journal_path.exists():
            return {}

        journal = _json.load(open(journal_path))
        bets = journal.get("bets", {})

        # Get bets from the last 7 days (approximate matchweek window)
        from datetime import datetime, timedelta
        cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        week_bets = []
        for bet_id, bet in bets.items():
            if bet.get("status") not in ("won", "lost", "push"):
                continue
            bet_date = bet.get("date", "")
            if bet_date >= cutoff:
                week_bets.append(bet)

        if not week_bets:
            return {}

        # Sort by date
        week_bets.sort(key=lambda b: b.get("date", ""))

        # Compute totals
        total_won = sum(1 for b in week_bets if b["status"] == "won")
        total_lost = sum(1 for b in week_bets if b["status"] == "lost")
        total_push = sum(1 for b in week_bets if b["status"] == "push")
        total_staked = sum(b.get("stake", 0) for b in week_bets)
        total_profit = 0
        for b in week_bets:
            if b["status"] == "won":
                total_profit += b.get("profit", 0)
            elif b["status"] == "lost":
                total_profit -= b.get("stake", 0)

        roi = (total_profit / total_staked * 100) if total_staked > 0 else 0

        # Market name mapping
        MKT_NAMES = {
            "h2h": "Match Result", "1X2": "Match Result",
            "totals": "Goals", "O/U": "Goals",
            "double_chance": "Double Chance", "DC": "Double Chance",
        }

        # Build message
        mw_label = f"Matchweek {matchweek}" if matchweek else "This Week"
        tg = TgMsg()
        tg.raw(f"\U0001f4ca <b>{mw_label} Summary</b>")
        tg.raw(f"   {total_won}W - {total_lost}L" +
               (f" - {total_push}P" if total_push else ""))
        tg.blank()

        # Each bet as a card
        for b in week_bets:
            match = b.get("match", "?")
            sel = b.get("selection", "?")
            market_raw = b.get("market", "")
            market = MKT_NAMES.get(market_raw, market_raw)
            odds = b.get("odds", 0)
            stake = b.get("stake", 0)
            status = b.get("status", "")
            score = b.get("result_score", "")

            if status == "won":
                profit = b.get("profit", 0)
                icon = "\u2705"
                result_str = f"<b>+\u20ac{profit:.2f}</b>"
            elif status == "lost":
                icon = "\u274c"
                result_str = f"-\u20ac{stake:.2f}"
            else:
                icon = "\u2796"
                result_str = "Push (stake returned)"

            tg.raw(f"{icon} <b>{_html_escape(match)}</b>"
                   + (f"  ({_html_escape(score)})" if score else ""))
            tg.raw(f"   {_html_escape(market)}: {_html_escape(sel)} @{odds:.2f}")
            tg.raw(f"   {result_str}")
            tg.blank()

        # Summary footer
        tg.raw("\u2500" * 20)
        sign = "+" if total_profit >= 0 else ""
        emoji = "\U0001f4b0" if total_profit >= 0 else "\U0001f4b8"
        tg.raw(f"{emoji} <b>Week P&amp;L: {sign}\u20ac{total_profit:.2f}</b> "
               f"(ROI: {roi:+.1f}%)")
        tg.raw(f"   Staked: \u20ac{total_staked:.2f} across {len(week_bets)} bets")

        # Closing thought
        if total_profit > 0:
            tg.blank()
            tg.italic("Good week. Stay disciplined.")
        elif total_profit < -20:
            tg.blank()
            tg.italic("Tough week. The edge is still there long-term.")
        else:
            tg.blank()
            tg.italic("Break-even. Consistency is key.")

        return notify(
            message=f"Matchweek: {total_won}W-{total_lost}L, {sign}\u20ac{total_profit:.2f}",
            title=f"{mw_label}: {sign}\u20ac{total_profit:.2f}",
            level="success" if total_profit >= 0 else "warning",
            category="betting",
            tg_html=tg.build(),
        )
    except Exception as e:
        log.warning("Matchweek summary failed: %s", e)
        return {}


def notify_entry_timing(match: str, selection: str, action: str,
                        odds: float = 0, bookmaker: str = "",
                        reason: str = "", confidence: float = 0) -> dict:
    """Alert when optimal entry timing is detected for a bet.

    Fires when sharp money confirms our bet direction and soft books
    haven't caught up — the window to get best odds.
    """
    if action != "ENTER_NOW" or confidence < 0.7:
        return {}  # Only notify on high-confidence ENTER_NOW

    mac_msg = (f"\u23f0 ENTER NOW: {match} | {selection} @{odds:.2f} ({bookmaker}) "
               f"| {reason[:60]}")

    tg = TgMsg()
    tg.title(f"Entry Window: {match}", emoji="\u23f0")
    tg.blank()
    tg.raw(f"<b>{_html_escape(selection)}</b> @{odds:.2f}")
    tg.raw(f"Book: {_html_escape(bookmaker)}")
    tg.blank()
    tg.italic(reason)
    tg.blank()
    tg.raw(f"Confidence: {confidence:.0%}")

    return notify(
        message=mac_msg,
        title=f"Entry: {match} {selection}",
        level="info",
        category="betting",
        tg_html=tg.build(),
        priority=PRIORITY_NORMAL,
    )


if __name__ == "__main__":
    _test()
