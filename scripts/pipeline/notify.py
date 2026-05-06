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

def _resolve_league(item: dict) -> str:
    """Return the league for a bet/match dict, inferring from team names if absent."""
    lg = item.get("league")
    if lg:
        return lg
    home = item.get("home_team")
    away = item.get("away_team")
    if (not home or not away) and item.get("match"):
        try:
            home, away = item["match"].split(" vs ", 1)
        except ValueError:
            pass
    try:
        from config.leagues import infer_league
        return infer_league(home, away)
    except Exception:
        return "serie_a"


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

    def shutdown(self):
        """Cancel all pending timers and flush remaining items on exit."""
        with self._lock:
            for cat, timer in list(self._timers.items()):
                timer.cancel()
            # Collect all pending items for logging (can't flush without flush_fn)
            pending_count = sum(len(v) for v in self._pending.values())
            if pending_count:
                log.warning("Batcher shutdown: %d buffered notifications lost", pending_count)
            self._pending.clear()
            self._timers.clear()


# Global batcher instance
_batcher = _NotificationBatcher(window_sec=45.0)

import atexit
atexit.register(_batcher.shutdown)

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
            # Write back atomically (crash between truncate and write would lose history)
            import tempfile as _th
            _fd, _tmp = _th.mkstemp(dir=_HISTORY_PATH.parent, suffix=".tmp")
            try:
                with open(_fd, "w") as f:
                    f.writelines(lines)
                Path(_tmp).rename(_HISTORY_PATH)
            except BaseException:
                Path(_tmp).unlink(missing_ok=True)
                raise
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
    """Send a macOS notification via osascript. Returns True on success.

    Skips silently on empty message (caller's way of saying "Telegram only,
    don't buzz the Mac"). Avoids the classic duplicate-buzz pattern where
    routine events pop desktop banners.
    """
    if not message or not message.strip():
        return False  # Telegram-only request from caller
    try:
        prefs = load_preferences()
        sound = prefs.get("sound", "Basso")
        if sound not in _VALID_SOUNDS:
            sound = "Basso"
        safe_msg = message.replace("\\", "\\\\").replace('"', '\\"')[:256]
        safe_title = title.replace("\\", "\\\\").replace('"', '\\"')[:64]
        script = (
            f'display notification "{safe_msg}" '
            f'with title "{safe_title}" sound name "{sound}"'
        )
        subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            timeout=15,
        )
        return True
    except subprocess.TimeoutExpired:
        log.debug("macOS notification timed out (osascript slow); Telegram path still fires")
        return False
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

    # Split long messages into chunks (Telegram limit: 4096 chars)
    MAX_TG_LEN = 4096
    if len(text) > MAX_TG_LEN:
        chunks = []
        while text:
            if len(text) <= MAX_TG_LEN:
                chunks.append(text)
                break
            # Try to split at a newline near the limit
            split_at = text.rfind("\n", 0, MAX_TG_LEN)
            if split_at < MAX_TG_LEN // 2:
                split_at = MAX_TG_LEN  # no good newline, hard split
            chunks.append(text[:split_at])
            text = text[split_at:].lstrip("\n")
        log.info("Telegram message split into %d chunks (%d chars total)",
                 len(chunks), sum(len(c) for c in chunks))
    else:
        chunks = [text]

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    import time as _time
    all_ok = True

    for chunk_idx, chunk_text in enumerate(chunks):
        payload_dict = {
            "chat_id": chat_id,
            "text": chunk_text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        if priority == PRIORITY_LOW:
            payload_dict["disable_notification"] = True
        # Attach keyboard only to the last chunk
        if reply_markup and chunk_idx == len(chunks) - 1:
            payload_dict["reply_markup"] = json.dumps(reply_markup) if isinstance(reply_markup, dict) else reply_markup

        payload = json.dumps(payload_dict).encode("utf-8")

        # Retry with exponential backoff (3 attempts: 0s, 2s, 4s)
        chunk_ok = False
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
                        chunk_ok = True
                        break
                    log.warning("Telegram returned HTTP %d (attempt %d)", resp.status, attempt + 1)
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8", errors="replace")[:200]
                except Exception:
                    pass
                log.warning("Telegram HTTP error %d (attempt %d): %s", e.code, attempt + 1, body)
                if e.code == 429:
                    _time.sleep(5)
                    continue
            except urllib.error.URLError as e:
                log.warning("Telegram connection error (attempt %d): %s", attempt + 1, e.reason)
            except Exception as e:
                log.warning("Telegram notification failed (attempt %d): %s", attempt + 1, e)

            if attempt < 2:
                _time.sleep(2 ** attempt)

        if not chunk_ok:
            # Retry once without HTML parsing (broken tags from mid-split)
            payload_dict.pop("parse_mode", None)
            payload = json.dumps(payload_dict).encode("utf-8")
            try:
                req = urllib.request.Request(url, data=payload,
                                            headers={"Content-Type": "application/json"},
                                            method="POST")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        chunk_ok = True
                        log.info("Telegram chunk %d sent without HTML (fallback)", chunk_idx + 1)
            except Exception:
                pass
            if not chunk_ok:
                log.error("Telegram chunk %d/%d failed after all attempts", chunk_idx + 1, len(chunks))
                all_ok = False

    if all_ok:
        log.debug("Telegram notification sent (%d chunks)", len(chunks))
    return all_ok


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

    # Fallback: if Telegram failed on critical/error messages, ensure macOS is sent
    if not results.get("telegram") and level in ("critical", "error") and not results.get("macos"):
        log.warning("Telegram failed for %s message — sending macOS fallback", level)
        results["macos_fallback"] = _notify_macos(f"[TG DOWN] {message}", title)

    # Emergency: if ALL channels failed for critical messages, write to emergency log
    if level in ("critical", "error"):
        any_sent = any(v for v in results.values() if v is True)
        if not any_sent:
            try:
                emergency_log = PROJECT_ROOT / "logs" / "emergency_alerts.log"
                emergency_log.parent.mkdir(parents=True, exist_ok=True)
                with open(emergency_log, "a") as f:
                    f.write(f"[{datetime.now().isoformat()}] [{level.upper()}] {title}: {message}\n")
            except Exception:
                pass

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
    """Return canonical bankroll state via scripts.betting.ledger.

    The ledger is the single source of truth: it reads bet_journal.json
    (authoritative) and computes balance / peak / lowest from active bets only
    (superseded bets are excluded from P&L). Falls back to raw file reads if
    the ledger import fails.
    """
    try:
        from scripts.betting import ledger
        current = ledger.get_balance()
        peak = ledger.get_peak()
        lowest = ledger.get_lowest()
        initial = ledger.get_initial_bankroll()
        return {
            "current": current,
            "initial": initial,
            "peak": peak,
            "lowest": lowest,
            "roi_pct": round((current - initial) / initial * 100, 1) if initial else 0,
            "drawdown_pct": round((peak - current) / peak * 100, 1) if peak else 0,
            "is_up": current > initial,
            "is_at_peak": current >= peak * 0.98 if peak else True,
        }
    except Exception as e:
        log.warning("ledger-based bankroll lookup failed, falling back: %s", e)
    # Fallback: old behavior from journal directly
    try:
        journal_path = DATA_DIR / "betting" / "bet_journal.json"
        if journal_path.exists():
            with open(journal_path) as f:
                journal = json.load(f)
            bets_raw = journal.get("bets", {})
            bets = list(bets_raw.values()) if isinstance(bets_raw, dict) else bets_raw
            initial = journal.get("initial_bankroll", journal.get("metadata", {}).get("initial_bankroll", 1000))
            settled = [b for b in bets if b.get("status") in ("won", "lost", "push", "voided")]
            total_profit = sum((b.get("profit") or 0) for b in settled)
            current = initial + total_profit
            peak = initial
            running = initial
            settled_sorted = sorted(settled, key=lambda b: b.get("settled_at") or "")
            for b in settled_sorted:
                running += (b.get("profit") or 0)
                if running > peak:
                    peak = running
            return {
                "current": round(current, 2),
                "initial": initial,
                "peak": round(peak, 2),
                "lowest": round(initial, 2),
                "roi_pct": round((current - initial) / initial * 100, 1) if initial else 0,
                "drawdown_pct": round((peak - current) / peak * 100, 1) if peak else 0,
                "is_up": current > initial,
                "is_at_peak": current >= peak * 0.98,
            }
    except Exception:
        pass
    return {"current": 0, "initial": 1000, "peak": 0, "lowest": 0, "roi_pct": 0,
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


def _detect_league_name(match_dict: dict) -> str:
    """Detect the league display name for a match/bet dict.

    Returns a display name like 'Serie A' or 'Premier League',
    or empty string if undetermined.
    """
    explicit = match_dict.get("league", "")
    if explicit:
        try:
            from config.leagues import LEAGUE_REGISTRY
            for key, cfg in LEAGUE_REGISTRY.items():
                if key == explicit or cfg.name.lower() == explicit.lower():
                    return cfg.name
        except Exception:
            pass
        return explicit

    # Try to detect from team name
    home = match_dict.get("home_team", "")
    if not home:
        mk = match_dict.get("match", "")
        if " vs " in mk:
            home = mk.split(" vs ", 1)[0].strip()
    if not home:
        return ""

    try:
        from config.team_names import normalize_team, SERIE_A_NAMES, PREMIER_LEAGUE_NAMES
        canonical = normalize_team(home)
        if canonical in SERIE_A_NAMES.values() or canonical in SERIE_A_NAMES:
            return "Serie A"
        if canonical in PREMIER_LEAGUE_NAMES.values() or canonical in PREMIER_LEAGUE_NAMES:
            return "Premier League"
    except Exception:
        pass
    return ""


def _league_badge(match_or_bet: dict = None, match_key: str = "") -> str:
    """Return flag emoji for the match's league. 🇮🇹 Serie A, 🏴󠁧󠁢󠁥󠁮󠁧󠁿 EPL."""
    if match_or_bet:
        league = _detect_league_name(match_or_bet)
    elif match_key:
        league = _detect_league_name({"match": match_key})
    else:
        return ""
    if "premier" in league.lower() or "epl" in league.lower():
        return "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f"
    if league:
        return "\U0001f1ee\U0001f1f9"
    return ""



_LEAGUE_ORDER = ("serie_a", "premier_league")
_LEAGUE_BADGE_MAP = {
    "serie_a": "\U0001f1ee\U0001f1f9",  # 🇮🇹
    "premier_league": "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f",  # 🏴󠁧󠁢󠁥󠁮󠁧󠁿
}


_KICKOFF_MAP_CACHE: dict = {"mtime": None, "data": None}


def _load_kickoff_map() -> dict:
    """Build {(home_team, away_team): commence_time_iso} from upcoming schedules.

    Both SA and EPL come from the same shape (odds_full*.json) so they parse
    identically. The legacy upcoming/matches.json was stale (MW24 matches from
    April 17 stuck in the file long after they played) and produced wrong
    kickoff times for SA in the digest.

    Cached by mtime so repeated digest renders don't re-parse.
    """
    sources = [
        DATA_DIR / "upcoming" / "odds_full.json",
        DATA_DIR / "upcoming" / "odds_full_premier_league.json",
    ]

    mtimes = []
    for p in sources:
        try:
            mtimes.append(p.stat().st_mtime)
        except FileNotFoundError:
            mtimes.append(0)
    sig = tuple(mtimes)
    if _KICKOFF_MAP_CACHE["mtime"] == sig and _KICKOFF_MAP_CACHE["data"] is not None:
        return _KICKOFF_MAP_CACHE["data"]

    kmap: dict = {}
    for p in sources:
        try:
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            for match_key, obj in (d.get("matches", {}) or {}).items():
                if isinstance(obj, dict) and obj.get("commence_time"):
                    h = obj.get("home_team") or (match_key.split(" vs ")[0] if " vs " in match_key else "")
                    a = obj.get("away_team") or (match_key.split(" vs ")[1] if " vs " in match_key else "")
                    if h and a:
                        kmap[(h, a)] = obj["commence_time"]
        except Exception as e:
            log.debug("kickoff-map %s load failed: %s", p.name, e)

    _KICKOFF_MAP_CACHE["mtime"] = sig
    _KICKOFF_MAP_CACHE["data"] = kmap
    return kmap


def _kickoff_display(commence_iso: str) -> str:
    """Convert ISO UTC kickoff ('2026-04-24T18:45:00Z') to 'HH:MM' in Europe/Rome.

    Falls back to UTC HH:MM if zoneinfo isn't available. Returns empty string
    on parse failure.
    """
    if not commence_iso:
        return ""
    try:
        raw = commence_iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(raw)
        try:
            from zoneinfo import ZoneInfo
            dt = dt.astimezone(ZoneInfo("Europe/Rome"))
        except Exception:
            # Fallback: no tz conversion, show UTC (rare on macOS/linux)
            pass
        return dt.strftime("%H:%M")
    except (ValueError, TypeError):
        return ""


def _render_matches_grouped_by_league(tg: "TgMsg", matches: list) -> None:
    """Render matches in strict chronological order across leagues.

    Each row keeps its league flag. Kickoff time resolved from:
      1. The match dict's own `kickoff_time` field, if populated
      2. The unified kickoff map (upcoming/matches.json +
         odds_full_premier_league.json) looked up by (home_team, away_team)

    Time is rendered in BOLD + monospace (<code>) inside square brackets and
    non-breaking spaces so the time stays anchored to the match row, never
    wrapping to a new line on narrow screens. Example: [ 20:45 ]   🇮🇹 …
    """
    if not matches:
        return

    kickoff_map = _load_kickoff_map()
    NBSP = " "  # non-breaking space — Telegram won't break the line here

    def _resolve_kickoff(p) -> str:
        own = p.get("kickoff_time") or ""
        if own and ("T" in str(own) or ":" in str(own)):
            return str(own)
        ht = p.get("home_team") or ""
        at = p.get("away_team") or ""
        return kickoff_map.get((ht, at), "")

    def _sort_key(p):
        k = _resolve_kickoff(p)
        if "T" in k:
            return (0, k)
        if ":" in k:
            return (0, k)
        return (1, p.get("home_team", "") or "")

    for p in sorted(matches, key=_sort_key):
        ht = (
            p.get("home_team")
            or (p.get("match", "").split(" vs ")[0] if " vs " in (p.get("match") or "") else "?")
        )
        at = (
            p.get("away_team")
            or (p.get("match", "").split(" vs ")[1] if " vs " in (p.get("match") or "") else "?")
        )
        league = p.get("league", "") or ""
        badge = _LEAGUE_BADGE_MAP.get(league, "")

        kick_iso = _resolve_kickoff(p)
        # Build a fixed-width bold prefix. Empty slot (for matches with no
        # resolvable kickoff) preserves column alignment.
        hhmm = ""
        if kick_iso:
            hhmm = _kickoff_display(kick_iso) or (kick_iso[:5] if ":" in kick_iso and "T" not in kick_iso else "")

        if hhmm:
            time_chip = f"<b>[{NBSP}{_html_escape(hhmm)}{NBSP}]</b>"
        else:
            # Keep column alignment when kickoff is unknown
            time_chip = f"<b>[{NBSP}{NBSP}—{NBSP}{NBSP}{NBSP}{NBSP}]</b>"

        tg.raw(f"  {time_chip}{NBSP}{NBSP}{badge}{NBSP}{_html_escape(ht)} – {_html_escape(at)}")


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

    sorted_bets = sorted(bets, key=lambda b: b.get("edge_pct") or 0, reverse=True)
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

    # Detect unique leagues for header
    league_names = set()
    for b in sorted_bets[:3]:
        ln = _detect_league_name(b)
        if ln:
            league_names.add(ln)
    if len(league_names) > 1:
        tg.raw(f"<i>Across {_html_escape(', '.join(sorted(league_names)))}</i>")
        tg.blank()

    for b in sorted_bets[:3]:
        edge = b.get("edge_pct", 0)
        conf = b.get("confidence_tier", "")
        conf_tag = f"  [{_html_escape(conf)}]" if conf else ""
        date_ctx = _time_until_kickoff(b.get("date", ""))
        time_tag = f"  \u23f0 {date_ctx}" if date_ctx else ""
        league_tag = ""
        ln = _detect_league_name(b)
        if ln and len(league_names) > 1:
            league_tag = f"  [{_html_escape(ln)}]"

        tg.raw(f"<b>{_html_escape(b.get('match', '?'))}</b>{conf_tag}{time_tag}{league_tag}")
        tg.raw(f"  {_html_escape(b.get('selection', '?'))} ({_html_escape(b.get('market', '?'))}) "
               f"@ <b>{b.get('best_odds', '?')}</b>  \u2014  {_html_escape(_edge_label(edge))}")
        market_pct = b.get('sharp_implied_prob', 0) * 100 if b.get('sharp_implied_prob') else (100 / b.get('best_odds', 1))
        tg.raw(f"  Model {b.get('model_prob', 0)*100:.0f}% vs sharp {market_pct:.0f}%"
               f"  |  \u20ac{b.get('stake', 0):.0f} stake")

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
                      best_win: dict | None = None, worst_loss: dict | None = None,
                      settled_bets: list[dict] | None = None) -> dict:
    """Send a rich settlement notification with story and context.

    Args:
        settled_bets: Optional list of individual settled bet dicts. When provided,
                      results are grouped by league in the Telegram message.
    """
    if settled == 0:
        return {}

    is_positive = profit >= 0
    streak = _get_streak()
    br_ctx = _get_bankroll_context()
    # Use passed balance if available (from settlement engine) instead of stale state
    if balance > 0:
        br_ctx["current"] = balance
        br_ctx["roi_pct"] = round((balance - br_ctx["initial"]) / br_ctx["initial"] * 100, 1) if br_ctx["initial"] else 0
    opener = _smart_opener("settle_win" if is_positive else "settle_loss",
                           profit=profit, streak=streak, bankroll_ctx=br_ctx)
    level = "success" if is_positive else "warning"

    record = f"{won}W-{lost}L" + (f"-{push}P" if push else "")
    sign = "+" if profit >= 0 else ""

    # macOS: the one number that matters
    mac_msg = f"{record} | {sign}\u20ac{profit:.2f} | Balance: {_bankroll_in_context(br_ctx)}"

    # Telegram — rich per-bet summary
    tg = TgMsg()
    tg.line(opener)
    tg.blank()

    # Show each settled bet with details
    if settled_bets:
        tg.raw("<b>\U0001f4ca Match Day Results</b>")
        tg.blank()
        for b in settled_bets:
            status = b.get("status", "")
            icon = "\u2705" if status == "won" else "\u274c" if status == "lost" else "\u21a9\ufe0f"
            match = _html_escape(b.get("match", "?"))
            market = _html_escape(b.get("market", ""))
            selection = _html_escape(b.get("selection", ""))
            odds = b.get("odds", 0)
            stake = b.get("stake", 0)
            bet_profit = b.get("profit") or 0
            score = _html_escape(b.get("result_score", ""))

            badge = _league_badge(b)
            tg.raw(f"{badge} {icon} <b>{match}</b>{f' ({score})' if score else ''}")
            tg.raw(f"   {market} {selection} @ {odds:.2f}")
            if status == "won":
                tg.raw(f"   \u20ac{stake:.0f} \u2192 Won \u20ac{stake + bet_profit:.0f} (+\u20ac{bet_profit:.0f})")
            elif status == "lost":
                tg.raw(f"   \u20ac{stake:.0f} \u2192 Lost (-\u20ac{stake:.0f})")
            else:
                tg.raw(f"   \u20ac{stake:.0f} \u2192 Push (refunded)")
            tg.raw("")

    # Daily totals
    tg.raw("\u2501" * 20)
    total = won + lost + push
    tg.raw(f"<b>Today:</b> {record} | {sign}\u20ac{profit:.2f}")
    if settled_bets:
        total_staked = sum(b.get("stake", 0) for b in settled_bets)
        if total_staked > 0:
            tg.raw(f"Staked: \u20ac{total_staked:.0f} | ROI: {profit / total_staked * 100:+.1f}%")
    if total > 0:
        tg.progress_bar(won, total)
    tg.blank()
    tg.raw(f"Balance: \u20ac{br_ctx['current']:,.0f} ({_html_escape(_bankroll_in_context(br_ctx))})")

    # Inline drawdown warning (replaces separate drawdown notification)
    dd = br_ctx.get("drawdown_pct", 0)
    if dd > 15:
        tg.raw(f"\u26a0\ufe0f <b>Drawdown: {dd:.0f}%</b> from peak \u20ac{br_ctx.get('peak', 0):,.0f}")
        if dd > 25:
            tg.italic("Consider reducing stakes until momentum turns.")

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
    """Rich goal notification: score, bet impact, time context, league badge."""

    has_active_bet = (bet_context and bet_context.get("has_bets")) or has_bet
    remaining_min = max(0, 90 - minute)
    added_time = minute > 90
    total = home_score + away_score
    badge = _league_badge(match_key=match_key)

    # Time context
    if added_time:
        time_str = f"{minute}' (added time)"
    elif remaining_min <= 10:
        time_str = f"{minute}' ({remaining_min} min left)"
    else:
        time_str = f"{minute}'"

    # macOS: compact, bet-aware
    if has_active_bet:
        mac_msg = f"{scorer} {time_str} \u2014 {match_key} {home_score}-{away_score}"
    else:
        mac_msg = f"\u26bd {scorer} ({team}) {time_str} \u2014 {match_key} {home_score}-{away_score}"

    # Telegram: rich, structured
    tg = TgMsg()

    # Header: badge + goal + score
    tg.raw(f"{badge} \u26bd <b>{_html_escape(scorer)}</b> ({_html_escape(team)}) {time_str}")
    tg.raw(f"<b>{_html_escape(match_key)}</b>  <code>{home_score} - {away_score}</code>")

    # Bet impact — the main event for bettors
    if bet_context and bet_context.get("has_bets"):
        tg.blank()
        for b in bet_context["bets"]:
            sel = b.get("selection", "")
            won = b.get("is_winning")
            commentary = b.get("commentary", "")
            odds = b.get("odds", 0)
            stake = b.get("stake", 0)

            if won is True:
                potential = round(stake * (odds - 1), 2) if odds > 1 else 0
                tg.raw(f"\u2705 <b>{_html_escape(sel)}</b> @ {odds:.2f} \u2014 winning (+\u20ac{potential:.0f})")
                if commentary:
                    tg.raw(f"   {_html_escape(commentary)}")
            elif won is False:
                tg.raw(f"\u274c <b>{_html_escape(sel)}</b> @ {odds:.2f} \u2014 lost (-\u20ac{stake:.0f})")
                if commentary:
                    tg.raw(f"   {_html_escape(commentary)}")
            else:
                # In progress — contextual status
                time_ctx = f" \u2014 {remaining_min} min left" if remaining_min > 0 else ""
                tg.raw(f"\u23f3 <b>{_html_escape(sel)}</b> @ {odds:.2f}{time_ctx}")
                if commentary:
                    tg.raw(f"   {_html_escape(commentary)}")

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
                    tg.raw(f"\u2705 <b>{_html_escape(bet_selection)}</b> \u2014 hit! \U0001f389")
                else:
                    need = line - total + 1
                    urgency = "\u26a0\ufe0f" if remaining_min < 15 else ""
                    tg.raw(f"\u23f3 <b>{_html_escape(bet_selection)}</b> \u2014 "
                           f"need {need:.0f} more goal{'s' if need > 1 else ''} in {remaining_min} min {urgency}")
            except ValueError:
                tg.raw(f"\u23f3 <b>{_html_escape(bet_selection)}</b>")
        elif "home" in bet_selection.lower():
            if home_score > away_score:
                tg.raw(f"\u2705 <b>{_html_escape(bet_selection)}</b> \u2014 winning \U0001f4aa")
            elif home_score == away_score:
                tg.raw(f"\u23f3 <b>{_html_escape(bet_selection)}</b> \u2014 level, need another ({remaining_min} min)")
            else:
                tg.raw(f"\u274c <b>{_html_escape(bet_selection)}</b> \u2014 behind, need comeback ({remaining_min} min)")
        elif "away" in bet_selection.lower():
            if away_score > home_score:
                tg.raw(f"\u2705 <b>{_html_escape(bet_selection)}</b> \u2014 winning \U0001f4aa")
            elif home_score == away_score:
                tg.raw(f"\u23f3 <b>{_html_escape(bet_selection)}</b> \u2014 level, need another ({remaining_min} min)")
            else:
                tg.raw(f"\u274c <b>{_html_escape(bet_selection)}</b> \u2014 behind ({remaining_min} min)")
        elif "draw" in bet_selection.lower():
            if home_score == away_score:
                tg.raw(f"\u2705 <b>{_html_escape(bet_selection)}</b> \u2014 holding \U0001f91e ({remaining_min} min)")
            else:
                tg.raw(f"\u274c <b>{_html_escape(bet_selection)}</b> \u2014 broken by this goal")
        else:
            tg.raw(f"\u23f3 <b>{_html_escape(bet_selection)}</b>")

    priority = PRIORITY_URGENT if has_active_bet else PRIORITY_NORMAL

    return notify(
        message=mac_msg,
        title=f"{badge} \u26bd {home_score}-{away_score}",
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
            if bet_profit and bet_profit > 0:
                msg += f" | +\u20ac{bet_profit:.2f}"
            level = "success"
        else:
            opener = random.choice(_FT_LOSS_OPENERS)
            msg = f"{opener} {match_key} {home_score}-{away_score}"
            if bet_profit and bet_profit < 0:
                msg += f" | -\u20ac{abs(bet_profit):.2f}"
            level = "warning"
    else:
        msg = f"Full time: {match_key} {home_score}-{away_score}"
        level = "info"

    badge = _league_badge(match_key=match_key)
    return notify(msg, title=f"{badge} \U0001f3c1 FT {home_score}-{away_score}", level=level, category="live")


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
    """DEPRECATED (2026-04-24): stale-data alerts are emitted by the
    health-state change detector (with proper issue-key dedup). Silent.
    """
    log.debug("notify_stale_data: silent (deprecated) — %s aged %.1fh", source, age_hours)
    return {}


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

    Every datum is traced back to its authoritative source:
      - Today's settled W/L/P&L  <- betting/history.json (list of settled bets)
      - Bankroll current balance <- betting/bankroll.json (.current_balance)
                                    Falls back to bet_journal running sum.
      - Bankroll peak            <- betting/bankroll.json (.peak_balance)
      - Per-day record + recent-days streak <- history.json grouped by date
      - Rolling ROI (last 50 settled) <- history.json sorted by settled_at
      - Tomorrow's matches       <- upcoming/predictions.json (Serie A)
                                    + upcoming/predictions_premier_league.json
      - Tomorrow's value bets    <- betting/betting_slip.json .recommended_singles
      - Was today a match day?   <- parsed/matches.parquet + matches_epl.parquet
      - Systems block            <- monitoring/scheduler_state.json,
                                    monitoring/health_status.json
    """
    from datetime import timedelta as _td
    # Use Europe/Rome local date — matches.parquet, predictions, and the user's
    # mental model all live in Italy local time. Falling back to wall-clock
    # naive datetime.now() works on a Rome-tz host, but if the host clock ever
    # drifted to UTC the digest would mis-bucket midnight matches.
    try:
        from zoneinfo import ZoneInfo
        _now_local = datetime.now(ZoneInfo("Europe/Rome"))
    except Exception:
        _now_local = datetime.now()
    today_str = _now_local.strftime("%Y-%m-%d")
    tomorrow_str = (_now_local + _td(days=1)).strftime("%Y-%m-%d")

    def _load_json(path: Path, default=None):
        try:
            if path.exists():
                with open(path) as f:
                    return json.load(f)
        except Exception as e:
            log.debug("Digest: failed to load %s: %s", path, e)
        return default if default is not None else {}

    # ---------- Sources ----------
    # History is now a derived view from the ledger — read through the ledger
    # directly instead of the on-disk cache (which can temporarily lag).
    try:
        from scripts.betting import ledger
        history = ledger.get_history_view()
    except Exception as e:
        log.warning("ledger.get_history_view failed, falling back to file: %s", e)
        history = _load_json(DATA_DIR / "betting" / "history.json", default=[])
    predictions_sa = _load_json(DATA_DIR / "upcoming" / "predictions.json")
    predictions_epl = _load_json(DATA_DIR / "upcoming" / "predictions_premier_league.json")
    # Prefer the unified slip (refreshed every morning by the pipeline) over the
    # legacy betting_slip.json which can sit stale for weeks. Normalize the
    # picks list under `recommended_singles` so the rest of the digest works
    # without further changes.
    unified_slip_path = DATA_DIR / "upcoming" / "unified_bet_slip.json"
    legacy_slip_path = DATA_DIR / "betting" / "betting_slip.json"
    slip_unified = _load_json(unified_slip_path) if unified_slip_path.exists() else {}
    slip_legacy = _load_json(legacy_slip_path) if legacy_slip_path.exists() else {}

    def _slip_age_h(s):
        gen = (s.get("generated_at") if isinstance(s, dict) else None) or ""
        if not gen:
            return float("inf")
        try:
            from datetime import timezone as _tz
            dt = datetime.fromisoformat(str(gen).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_tz.utc)
            return (datetime.now(_tz.utc) - dt).total_seconds() / 3600
        except Exception:
            return float("inf")

    # Pick the freshest slip
    if _slip_age_h(slip_unified) <= _slip_age_h(slip_legacy):
        slip = dict(slip_unified)
        # Unified file uses `selected_bets`; alias to `recommended_singles`
        if "recommended_singles" not in slip and "selected_bets" in slip:
            slip["recommended_singles"] = slip["selected_bets"]
    else:
        slip = slip_legacy

    # ---------- Bankroll (ledger is the single source of truth) ----------
    _br = _get_bankroll_context()
    initial = _br["initial"]
    current = _br["current"]
    peak = _br["peak"]
    lowest = _br.get("lowest", initial)
    roi_alltime = _br["roi_pct"]
    drawdown_pct = _br["drawdown_pct"]

    # ---------- Rolling ROI (last 50 settled bets via ledger) ----------
    settled_wl = [b for b in history if b.get("status") in ("won", "lost", "push")]
    settled_sorted = sorted(settled_wl, key=lambda b: b.get("settled_at") or "", reverse=True)
    try:
        from scripts.betting import ledger as _ledger
        roi_rolling = _ledger.get_roi(window=50)
        roll_n = min(50, len(settled_sorted))
    except Exception:
        roll_n = min(50, len(settled_sorted))
        rolling_slice = settled_sorted[:roll_n]
        roll_staked = sum(float(b.get("stake") or 0) for b in rolling_slice)
        roll_profit = sum(float(b.get("profit") or 0) for b in rolling_slice)
        roi_rolling = (roll_profit / roll_staked * 100) if roll_staked > 0 else None

    # ---------- Today's results (from history.json) ----------
    # Bucket by MATCH date, not settled_at. settled_at is UTC and can be filled
    # asynchronously days after a match plays — using it caused yesterday's
    # backfilled settlements to appear as "today" the morning after.
    today_bets = [
        b for b in history
        if (b.get("date") or "").startswith(today_str)
        and b.get("status") in ("won", "lost", "push", "void")
    ]
    won_today = sum(1 for b in today_bets if b.get("status") == "won")
    lost_today = sum(1 for b in today_bets if b.get("status") == "lost")
    push_today = sum(1 for b in today_bets if b.get("status") == "push")
    total_today = won_today + lost_today + push_today
    pnl_today = sum(float(b.get("profit") or 0) for b in today_bets)

    # Was today a MATCH day? Check the FUTURE-fixtures sources:
    #   - predictions.json (Serie A) + predictions_premier_league.json (EPL)
    #   - upcoming/matches.json (raw Odds API schedule) as secondary
    # NOT parsed/matches.parquet — that's PAST-only and lags by days.
    today_matches: list[dict] = []
    seen_keys: set = set()  # dedup across sources
    for src, league_tag in (
        (predictions_sa, "serie_a"),
        (predictions_epl, "premier_league"),
    ):
        for p in (src.get("predictions", []) if isinstance(src, dict) else []):
            if (p.get("date") or "").startswith(today_str):
                key = (p.get("home_team", ""), p.get("away_team", ""))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                today_matches.append({**p, "league": p.get("league", league_tag)})
    # Fallback: upcoming/matches.json raw schedule (covers leagues without
    # per-league prediction files)
    try:
        raw_schedule = _load_json(DATA_DIR / "upcoming" / "matches.json")
        raw_matches = raw_schedule if isinstance(raw_schedule, list) else raw_schedule.get("matches", [])
        for mm in raw_matches or []:
            ct = mm.get("commence_time", "")
            if ct.startswith(today_str):
                home = mm.get("home_team", "")
                away = mm.get("away_team", "")
                key = (home, away)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                today_matches.append({
                    "home_team": home,
                    "away_team": away,
                    "date": today_str,
                    "kickoff_time": ct,
                    "league": mm.get("league") or "unknown",
                })
    except Exception as e:
        log.debug("Digest upcoming/matches.json lookup failed: %s", e)
    matches_today_count = len(today_matches)

    best_pick = None
    if today_bets:
        winners = [b for b in today_bets if b.get("status") == "won"]
        if winners:
            best_pick = max(winners, key=lambda b: float(b.get("profit") or 0))

    # ---------- Streak computed per-DAY (timestamps within a batch are tied) ----------
    # A day "wins" if its net P&L is positive. Key by MATCH date (when the bet
    # was actually contested), not settled_at — settled_at can be filled
    # asynchronously days later and would mis-attribute the bet to the wrong
    # day in the streak chart.
    per_day: dict[str, dict] = {}
    for b in settled_wl:
        d = (b.get("date") or b.get("settled_at") or "")[:10]
        if not d:
            continue
        agg = per_day.setdefault(d, {"won": 0, "lost": 0, "push": 0, "pnl": 0.0, "n": 0})
        s = b.get("status", "")
        if s in ("won", "lost", "push"):
            agg[s] += 1
        agg["pnl"] += float(b.get("profit") or 0)
        agg["n"] += 1

    day_dates_desc = sorted(per_day.keys(), reverse=True)
    winning_days_streak = 0
    losing_days_streak = 0
    if day_dates_desc:
        first_sign = per_day[day_dates_desc[0]]["pnl"]
        if first_sign > 0:
            for d in day_dates_desc:
                if per_day[d]["pnl"] > 0:
                    winning_days_streak += 1
                else:
                    break
        elif first_sign < 0:
            for d in day_dates_desc:
                if per_day[d]["pnl"] < 0:
                    losing_days_streak += 1
                else:
                    break
    # "day streak" int: positive = winning days, negative = losing days, 0 = flat/breakeven
    day_streak = winning_days_streak if winning_days_streak else -losing_days_streak

    # Last 7 betting days pattern (emoji strip: 🟢 🔴 ⚪)
    last7 = []
    for d in day_dates_desc[:7]:
        p = per_day[d]["pnl"]
        last7.append("\U0001f7e2" if p > 0 else "\U0001f534" if p < 0 else "⚪")
    last7_str = " ".join(reversed(last7)) if last7 else ""

    # ---------- Tomorrow's matches (SA + EPL merged) ----------
    tomorrow_matches: list[dict] = []
    for src, league_tag in ((predictions_sa, "serie_a"), (predictions_epl, "premier_league")):
        for p in (src.get("predictions", []) if isinstance(src, dict) else []):
            if (p.get("date") or "").startswith(tomorrow_str):
                tomorrow_matches.append({**p, "league": p.get("league", league_tag)})

    # Tomorrow's value bets from betting_slip.json (recommended_singles is authoritative)
    slip_rec = slip.get("recommended_singles", []) if isinstance(slip, dict) else []
    # Slip staleness: compute age of the last slip write. >48h = show warning
    # instead of pretending entries are current (matters when the pipeline
    # can't refresh, e.g. odds API quota exhausted).
    slip_age_hours: float | None = None
    slip_is_stale = False
    gen_at = (slip.get("generated_at") if isinstance(slip, dict) else None) or ""
    if gen_at:
        try:
            from datetime import timezone as _tz
            gen_dt = datetime.fromisoformat(str(gen_at).replace("Z", "+00:00"))
            if gen_dt.tzinfo is None:
                gen_dt = gen_dt.replace(tzinfo=_tz.utc)
            slip_age_hours = (datetime.now(_tz.utc) - gen_dt).total_seconds() / 3600
            slip_is_stale = slip_age_hours > 48
        except (ValueError, TypeError):
            pass
    tomorrow_value_bets = [
        b for b in slip_rec if (b.get("date") or "").startswith(tomorrow_str)
    ]
    # Suppress stale slip entries entirely — they are lies, not data.
    if slip_is_stale:
        tomorrow_value_bets = []
    slip_has_any_future = (not slip_is_stale) and any(
        (b.get("date") or "") > today_str for b in slip_rec
    )
    next_bet_date = None
    if not tomorrow_value_bets and slip_has_any_future:
        future_dates = sorted({b.get("date") for b in slip_rec if (b.get("date") or "") > today_str})
        if future_dates:
            next_bet_date = future_dates[0]
            tomorrow_value_bets = [b for b in slip_rec if b.get("date") == next_bet_date]

    # Skip if literally nothing to report
    if total_today == 0 and matches_today_count == 0 and not tomorrow_matches and not slip_rec:
        return {}

    best_edge_bet = None
    if tomorrow_value_bets:
        def _edge(b):
            return float(b.get("edge_pct") or b.get("value_pct") or 0)
        best_edge_bet = max(tomorrow_value_bets, key=_edge)

    # ---------- Compose ----------
    level = "success" if pnl_today > 0 else "warning" if pnl_today < 0 else "info"

    # macOS one-liner
    if total_today > 0:
        sign = "+" if pnl_today >= 0 else ""
        mac_msg = (f"Daily: {won_today}W-{lost_today}L | {sign}€{pnl_today:.2f} "
                   f"| Balance: €{current:,.0f}")
    elif matches_today_count > 0:
        mac_msg = (f"Daily: {matches_today_count} match(es) today, none settled yet. "
                   f"Balance: €{current:,.0f}")
    else:
        mac_msg = f"Daily: no matches today. Balance: €{current:,.0f}"

    tg = TgMsg()
    tg.raw("<b>\U0001f4ca Daily Wrap-Up</b>")
    tg.raw(f"<i>{_html_escape(today_str)}</i>")
    tg.sep()

    # --- Today ---
    if total_today > 0:
        record = f"{won_today}W-{lost_today}L" + (f"-{push_today}P" if push_today else "")
        tg.raw(f"\n\U0001f3af <b>Today:</b> {_html_escape(record)}")
        tg.progress_bar(won_today, total_today)
        tg.pnl(pnl_today, label="Day P&L")
        if best_pick:
            bp = best_pick
            tg.raw(
                f"⭐ Best: {_html_escape(bp.get('match',''))} "
                f"{_html_escape(bp.get('selection',''))} "
                f"(+€{float(bp.get('profit') or 0):.2f})"
            )
    elif matches_today_count > 0:
        tg.raw(
            f"\n\U0001f3af <b>Today:</b> "
            f"{matches_today_count} match{'es' if matches_today_count != 1 else ''} "
            f"({_html_escape(today_str)})"
        )
        _render_matches_grouped_by_league(tg, today_matches)
    else:
        tg.raw("\n\U0001f3af <b>Today:</b> No fixtures. Rest day.")

    # --- Bankroll ---
    tg.blank()
    tg.mini_sep()
    tg.raw(f"\n\U0001f4b0 <b>Bankroll:</b> €{current:,.2f}")

    roi_alltime_str = f"{'+' if roi_alltime >= 0 else ''}{roi_alltime:.1f}%"
    if roi_rolling is not None:
        roi_roll_str = f"{'+' if roi_rolling >= 0 else ''}{roi_rolling:.1f}%"
        tg.stat_row(
            ("ROI all-time", roi_alltime_str),
            (f"ROI last {roll_n}", roi_roll_str),
        )
    else:
        tg.stat_row(("ROI all-time", roi_alltime_str))

    # Day-streak (meaningful, timestamp-robust)
    if day_streak > 1:
        tg.raw(f"\U0001f525 <b>Winning days:</b> {day_streak} in a row")
    elif day_streak < -1:
        tg.raw(f"❄️ <b>Losing days:</b> {abs(day_streak)} in a row")
    elif day_dates_desc:
        last = per_day[day_dates_desc[0]]
        sign_day = "+" if last["pnl"] >= 0 else ""
        tg.raw(
            f"\U0001f4c6 <b>Last betting day:</b> {day_dates_desc[0]} "
            f"({last['won']}W-{last['lost']}L, {sign_day}€{last['pnl']:.2f})"
        )
    if last7_str:
        tg.raw(f"  <i>Last {len(last7)} days:</i> {last7_str}")

    if drawdown_pct > 5:
        tg.raw(
            f"  ⚠️ Drawdown: {drawdown_pct:.1f}% from peak "
            f"(€{peak:,.0f})"
        )

    # --- Tomorrow ---
    tg.blank()
    tg.mini_sep()
    if tomorrow_matches:
        n = len(tomorrow_matches)
        tg.raw(
            f"\n\U0001f4c5 <b>Tomorrow:</b> "
            f"{n} match{'es' if n != 1 else ''} ({_html_escape(tomorrow_str)})"
        )
        _render_matches_grouped_by_league(tg, tomorrow_matches)

        if best_edge_bet:
            be = best_edge_bet
            edge = float(be.get("edge_pct") or be.get("value_pct") or 0)
            odds_val = be.get("odds") or be.get("best_odds") or "?"
            tg.raw(
                f"⚡ <b>Best edge:</b> {_html_escape(be.get('match',''))} "
                f"{_html_escape(be.get('selection',''))} "
                f"@ {odds_val} (<b>{edge:.0f}%</b>)"
            )
        if tomorrow_value_bets:
            tg.raw(
                f"\U0001f4cb {len(tomorrow_value_bets)} value bet"
                f"{'s' if len(tomorrow_value_bets) != 1 else ''} in the slip"
            )
    elif next_bet_date:
        tg.raw(
            f"\n\U0001f4c5 <b>Next card:</b> {_html_escape(next_bet_date)} "
            f"({len(tomorrow_value_bets)} value bet{'s' if len(tomorrow_value_bets) != 1 else ''})"
        )
        for b in tomorrow_value_bets[:3]:
            edge = float(b.get("edge_pct") or b.get("value_pct") or 0)
            odds_val = b.get("odds") or b.get("best_odds") or "?"
            tg.raw(
                f"  ▪ {_html_escape(b.get('match',''))} "
                f"{_html_escape(b.get('selection',''))} "
                f"@ {odds_val} (<b>{edge:.0f}%</b>)"
            )
    elif slip_rec and not slip_is_stale:
        tg.raw(f"\n\U0001f4c5 <b>Slip:</b> {len(slip_rec)} bet(s) queued (no upcoming date matched)")
    else:
        tg.raw("\n\U0001f4c5 No upcoming matches on the radar.")

    # Slip staleness warning — surface it explicitly when the slip was last
    # refreshed more than 48h ago. Tells you WHY there's no fresh pick.
    if slip_is_stale and slip_age_hours is not None:
        age_label = (
            f"{slip_age_hours:.0f}h" if slip_age_hours < 72
            else f"{slip_age_hours/24:.0f}d"
        )
        tg.raw(
            f"  ⚠️ <i>Slip is stale: last refresh {age_label} ago. "
            f"Pipeline may be blocked.</i>"
        )

    # --- Systems ---
    try:
        sched_state = _load_json_safe(DATA_DIR / "monitoring" / "scheduler_state.json", {})
        health = _load_json_safe(DATA_DIR / "monitoring" / "health_status.json", {})
        if sched_state or health:
            tg.blank()
            tg.mini_sep()
            tg.raw("\n⚙️ <b>Systems</b>")

            ran_today = [
                (name, entry) for name, entry in sched_state.items()
                if (entry.get("last_run") or "").startswith(today_str)
            ]
            if ran_today:
                for name, entry in sorted(ran_today, key=lambda x: x[1].get("last_run") or ""):
                    emoji, label = _SCHEDULER_BADGE.get(name, ("⚙️", name))
                    icon = _STATUS_ICON.get(entry.get("status", "ok"), "ℹ️")
                    hhmm = (entry.get("last_run") or "")[11:16]
                    tg.raw(f"  {icon} {emoji} {_html_escape(label)}  <i>{hhmm}</i>")
            else:
                tg.raw("  <i>No scheduler runs recorded today.</i>")

            overall = health.get("overall_status")
            if overall:
                icon = {"HEALTHY": "✅", "OK": "✅",
                        "WARNING": "⚠️",
                        "CRITICAL": "\U0001f6a8"}.get(overall, "ℹ️")
                n_issues = len(health.get("issues", []))
                suffix = f" ({n_issues} issue{'s' if n_issues != 1 else ''})" if n_issues else ""
                tg.raw(f"  {icon} <b>Health:</b> {_html_escape(overall)}{suffix}")
    except Exception as e:
        log.debug("Digest systems section failed: %s", e)

    # Closer
    tg.blank()
    if total_today > 0 and pnl_today >= 0:
        tg.italic(random.choice(_DIGEST_POSITIVE_CLOSERS))
    elif total_today > 0:
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
    """DEPRECATED (2026-04-24): drawdown is surfaced in the daily digest
    bankroll section. Standalone alerts caused duplicate pings. Silent.
    """
    log.debug(
        "notify_drawdown: silent (deprecated) — current=%s peak=%s dd=%.1f%%",
        current, peak, drawdown_pct,
    )
    return {}


# ---------------------------------------------------------------------------
# Morning Briefing
# ---------------------------------------------------------------------------

_MORNING_OPENERS = [
    "Good morning. Here's what's on the card today.",
    "Rise and grind. Match day briefing incoming.",
    "Morning. Let's see what the markets are offering.",
]


def notify_morning_briefing() -> dict:
    """DEPRECATED (2026-04-24): Morning briefing was redundant with
    notify_scheduler_run('morning') + notify_daily_digest. Converted to
    silent no-op to stop double-pinging. Call sites kept for backward
    compat.
    """
    log.debug("notify_morning_briefing: silent (deprecated)")
    return {}



# ---------------------------------------------------------------------------
# Live Matchday P&L Summary
# ---------------------------------------------------------------------------

def notify_matchday_update() -> dict:
    """DEPRECATED (2026-04-24): matchday recap is covered by the daily-digest
    + per-bet notify_settlement. Silent no-op; call sites kept for compat.
    """
    log.debug("notify_matchday_update: silent (deprecated)")
    return {}


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
    """DEPRECATED (2026-04-24): pipeline completion is now surfaced by
    notify_scheduler_run (morning/evening cards) with richer context.
    Silent no-op; callers don't need to be updated.
    """
    log.debug(
        "notify_pipeline_done: silent (deprecated) — %d preds, %d vbets, %.1fs",
        n_predictions, n_value_bets, elapsed_sec,
    )
    return {}


def notify_odds_snapshot(n_matches: int = 0, n_bookmakers: int = 0) -> dict:
    """Odds snapshots are routine — log only, don't notify.

    Odds refresh every few hours. Notifying on each one is noise.
    The user cares about the VALUE BETS found from the odds, not the fetch itself.
    """
    log.info("Odds snapshot: %d matches, %d bookmakers (no notification — routine)",
             n_matches, n_bookmakers)
    return {}


def _load_prediction_for(mk: str, home_team: str, away_team: str) -> dict | None:
    """Find the prediction record for this match across both league files."""
    for fname in ("predictions.json", "predictions_premier_league.json"):
        p = DATA_DIR / "upcoming" / fname
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
            for pred in data.get("predictions", []) or []:
                if pred.get("match") == mk:
                    return pred
                if (pred.get("home_team", "").strip() == home_team.strip()
                        and pred.get("away_team", "").strip() == away_team.strip()):
                    return pred
        except Exception:
            pass
    return None


def _build_match_preview(pred: dict, home_team: str, away_team: str,
                          home_xi: list[str], away_xi: list[str],
                          fb_map: dict, us_map: dict, _tk, _team_match) -> str:
    """Generate a pundit-style match preview in 2-4 sentences.

    Uses prediction + form + injury + player-xG data to build coherent
    narrative. Deterministic (no LLM) — rules-based with a priority list of
    angles; picks the 2-3 most salient.
    """
    if not pred:
        return ""

    parts: list[str] = []

    # --- 1. Model verdict ---
    probs = pred.get("probabilities", {}) or {}
    p_h = float(probs.get("home", 0))
    p_d = float(probs.get("draw", 0))
    p_a = float(probs.get("away", 0))
    winner = None
    if p_h > p_a + 0.10 and p_h > p_d + 0.05:
        winner = "home"
    elif p_a > p_h + 0.10 and p_a > p_d + 0.05:
        winner = "away"
    elif abs(p_h - p_a) < 0.08:
        winner = "coinflip"

    confidence = pred.get("confidence", 0)
    if winner == "home":
        parts.append(
            f"Model favors <b>{_html_escape(home_team)}</b> at "
            f"<b>{p_h*100:.0f}%</b> vs <b>{p_a*100:.0f}%</b>."
        )
    elif winner == "away":
        parts.append(
            f"Model favors <b>{_html_escape(away_team)}</b> at "
            f"<b>{p_a*100:.0f}%</b> vs <b>{p_h*100:.0f}%</b>."
        )
    elif winner == "coinflip":
        parts.append(
            f"Tight call — {_html_escape(home_team)} {p_h*100:.0f}% vs "
            f"{_html_escape(away_team)} {p_a*100:.0f}% "
            f"(draw {p_d*100:.0f}%)."
        )

    # --- 2. Form contrast ---
    hf = pred.get("home_form") or {}
    af = pred.get("away_form") or {}
    hf_ppg = hf.get("ppg")
    af_ppg = af.get("ppg")
    hf_status = hf.get("form_status", "normal")
    af_status = af.get("form_status", "normal")
    if hf_ppg is not None and af_ppg is not None and abs(hf_ppg - af_ppg) >= 0.6:
        if hf_ppg > af_ppg:
            parts.append(
                f"Form swing: {_html_escape(home_team)} {hf_ppg:.1f} ppg "
                f"vs {_html_escape(away_team)} {af_ppg:.1f} ppg "
                f"in last 5."
            )
        else:
            parts.append(
                f"Form swing: {_html_escape(away_team)} {af_ppg:.1f} ppg "
                f"vs {_html_escape(home_team)} {hf_ppg:.1f} ppg "
                f"in last 5."
            )
    elif hf_status == "cold" and af_status == "hot":
        parts.append(
            f"<b>{_html_escape(away_team)}</b> on a hot run; "
            f"<b>{_html_escape(home_team)}</b> cold."
        )
    elif hf_status == "hot" and af_status == "cold":
        parts.append(
            f"<b>{_html_escape(home_team)}</b> on a hot run; "
            f"<b>{_html_escape(away_team)}</b> cold."
        )

    # --- 3. Defensive vulnerability (goals_conceded in last 5) ---
    hf_gc = hf.get("goals_conceded")
    af_gc = af.get("goals_conceded")
    hf_games = hf.get("total_matches") or 5
    af_games = af.get("total_matches") or 5
    if hf_gc is not None and hf_games:
        hf_gcpg = hf_gc / hf_games
        if hf_gcpg >= 1.6:
            parts.append(
                f"<b>{_html_escape(home_team)}</b>'s defense leaky "
                f"({hf_gc} goals in last {hf_games})."
            )
    if af_gc is not None and af_games:
        af_gcpg = af_gc / af_games
        if af_gcpg >= 1.6:
            parts.append(
                f"<b>{_html_escape(away_team)}</b>'s defense leaky "
                f"({af_gc} goals in last {af_games})."
            )

    # --- 4. Key player in lineup — highest xG player on either side ---
    def _top_xg_player(team: str, xi: list[str]) -> tuple[str, float] | None:
        best_name = None
        best_xg = 0.0
        for name in xi:
            # Use team-aware lookup (same logic as _lookup_xg)
            tk = _tk(team); pk = name.strip()
            xg = None
            if (tk, pk) in us_map:
                xg = us_map[(tk, pk)]["xg"]
            else:
                for (t, n), rec in us_map.items():
                    if n == pk and _team_match(team, t):
                        xg = rec["xg"]; break
                else:
                    surname = pk.split()[-1].lower()
                    for (t, n), rec in us_map.items():
                        if _team_match(team, t) and n.split()[-1].lower() == surname:
                            xg = rec["xg"]; break
            if xg and xg > best_xg:
                best_xg = float(xg)
                best_name = pk
        return (best_name, best_xg) if best_name else None

    home_star = _top_xg_player(home_team, home_xi)
    away_star = _top_xg_player(away_team, away_xi)
    if home_star and home_star[1] >= 3.0:
        parts.append(
            f"<b>{_html_escape(home_star[0])}</b> leads the home attack "
            f"(xG {home_star[1]:.1f} season)."
        )
    if away_star and away_star[1] >= 3.0:
        parts.append(
            f"<b>{_html_escape(away_star[0])}</b> the away threat "
            f"(xG {away_star[1]:.1f} season)."
        )

    # --- 5. Lineup xG edge (from player_xg component) ---
    comp = pred.get("component_predictions", {}) or {}
    plxg = comp.get("player_xg_details") or {}
    home_lxg = plxg.get("home_lineup_xg")
    away_lxg = plxg.get("away_lineup_xg")
    if home_lxg is not None and away_lxg is not None:
        diff = home_lxg - away_lxg
        if abs(diff) >= 0.5:
            stronger = home_team if diff > 0 else away_team
            weaker = away_team if diff > 0 else home_team
            strong_xg = home_lxg if diff > 0 else away_lxg
            weak_xg = away_lxg if diff > 0 else home_lxg
            parts.append(
                f"Today's XI: <b>{_html_escape(stronger)}</b> "
                f"{strong_xg:.2f} xG vs <b>{_html_escape(weaker)}</b> "
                f"{weak_xg:.2f} xG."
            )

    # --- 6. Injuries — only mention key missing players ---
    inj = pred.get("injury_adjustments") or {}
    home_inj = inj.get("home_injured", []) or []
    away_inj = inj.get("away_injured", []) or []
    # Mention only if 3+ key missing
    if len(home_inj) >= 3:
        parts.append(
            f"<b>{_html_escape(home_team)}</b> missing "
            f"{len(home_inj)} starters: "
            f"{_html_escape(', '.join(home_inj[:3]))}"
            f"{'…' if len(home_inj) > 3 else ''}."
        )
    if len(away_inj) >= 3:
        parts.append(
            f"<b>{_html_escape(away_team)}</b> missing "
            f"{len(away_inj)} starters: "
            f"{_html_escape(', '.join(away_inj[:3]))}"
            f"{'…' if len(away_inj) > 3 else ''}."
        )

    # --- 7. Referee bias ---
    ref = pred.get("referee")
    ref_bias = pred.get("referee_bias")
    if ref and ref_bias and ref_bias in ("home_favoring", "away_favoring"):
        bias_team = home_team if ref_bias == "home_favoring" else away_team
        parts.append(f"Ref {_html_escape(str(ref))} slants toward <b>{_html_escape(bias_team)}</b>.")

    # --- 8. Goals market lean (O/U 2.5) ---
    ou = comp.get("over_under_ml") or {}
    ou_25 = ou.get("2.5")
    if ou_25 is not None:
        if ou_25 >= 0.60:
            parts.append(f"Model leans <b>OVER 2.5</b> ({ou_25*100:.0f}%).")
        elif ou_25 <= 0.40:
            parts.append(f"Model leans <b>UNDER 2.5</b> ({(1-ou_25)*100:.0f}%).")

    # Truncate to top 4 bullets to keep the preview punchy
    parts = parts[:4]

    return "\n".join(f"  • {p}" for p in parts)


def notify_lineups_confirmed(matches: str = "", changes: str = "") -> dict:
    """Starting XIs confirmed — full lineups with shirt numbers, positions,
    and current-season stats (TEAM-FILTERED to prevent cross-team pollution).

    Sources:
      - data/upcoming/confirmed_lineups.json       — authoritative XI names
      - data/parsed/lineups.parquet                — shirt_number per (team, player)
      - data/parsed/player_stats.parquet           — position + G/A (per team)
      - data/parsed/understat_players.parquet      — xG (per team)
      - data/betting/betting_slip.json             — active bet

    `matches` / `changes` args accepted for backward compat; ignored.
    """
    import hashlib as _hl

    lineups_path = DATA_DIR / "upcoming" / "confirmed_lineups.json"
    if not lineups_path.exists():
        return {}
    try:
        data = json.loads(lineups_path.read_text())
    except Exception as e:
        log.warning("notify_lineups_confirmed: load failed: %s", e)
        return {}

    all_matches = data.get("matches", {}) if isinstance(data, dict) else {}
    confirmed = [
        (mk, md) for mk, md in all_matches.items()
        if isinstance(md.get("home_lineup"), list) and isinstance(md.get("away_lineup"), list)
        and len(md["home_lineup"]) >= 11 and len(md["away_lineup"]) >= 11
    ]
    if not confirmed:
        return {}

    # Dedup
    sig_source = "|".join(
        f"{mk}:{','.join(md.get('home_lineup', []))}:{','.join(md.get('away_lineup', []))}"
        for mk, md in confirmed
    )
    msg_sig = _hl.md5(sig_source.encode()).hexdigest()[:12]
    today_str = datetime.now().strftime("%Y-%m-%d")
    _lineup_dedup_path = DATA_DIR / ".lineups_dedup.json"
    try:
        if _lineup_dedup_path.exists():
            with open(_lineup_dedup_path) as _f:
                _dedup = json.load(_f)
            if _dedup.get("date") == today_str and _dedup.get("sig") == msg_sig:
                return {}
    except Exception:
        pass

    def _current_season() -> str:
        y = datetime.now().year
        return f"{y-1}-{y}" if datetime.now().month < 8 else f"{y}-{y+1}"

    season = _current_season()

    # Build TEAM-KEYED enrichment tables.
    # Key: (team_canonical, player_canonical) -> record dict
    # team_canonical = team name lowercased; player_canonical = full name.
    shirt_map: dict = {}  # (team, name) -> shirt_str; also (team, surname) as fallback
    fb_map: dict = {}     # (team, name) -> {position, goals, assists, minutes}
    fb_name_only: dict = {}  # name -> list of (team, record)  — for fuzzy
    us_map: dict = {}     # (team, name) -> {xg}
    us_name_only: dict = {}

    def _tk(team: str) -> str:
        return (team or "").strip().lower()

    def _pk(name: str) -> str:
        return (name or "").strip()

    try:
        import pandas as _pd
        # ---- lineups.parquet → shirt numbers (STARTER + BENCH so subs are covered) ----
        try:
            lu = _pd.read_parquet(
                DATA_DIR / "parsed" / "lineups.parquet",
                columns=["season", "team", "player_name", "shirt_number", "role"],
            )
            cur_lu = lu[lu["season"] == season]
            # Keep last-seen shirt per (team, player); order doesn't matter since
            # shirt numbers are stable within a season
            cur_lu = cur_lu.dropna(subset=["shirt_number"]).drop_duplicates(
                subset=["team", "player_name"], keep="last"
            )
            for _, row in cur_lu.iterrows():
                team = _tk(row["team"])
                name = _pk(row["player_name"])
                shirt = str(int(row["shirt_number"])) if _pd.notna(row["shirt_number"]) else None
                if shirt:
                    shirt_map[(team, name)] = shirt
                    # Also index by surname for fuzzy
                    surname = name.split()[-1].lower()
                    shirt_map.setdefault((team, "_surname_" + surname), shirt)
        except Exception as e:
            log.debug("lineups.parquet failed: %s", e)

        # ---- player_stats.parquet + player_stats_epl.parquet → fbref season aggregates PER TEAM ----
        for _ps_file in ("player_stats.parquet", "player_stats_epl.parquet"):
            try:
                ps = _pd.read_parquet(
                    DATA_DIR / "parsed" / _ps_file,
                    columns=["season", "player", "team", "position", "goals", "assists", "minutes"],
                )
                # EPL parquet stores numeric columns as object/strings — coerce.
                for _num_col in ("goals", "assists", "minutes"):
                    if _num_col in ps.columns and ps[_num_col].dtype == object:
                        ps[_num_col] = _pd.to_numeric(ps[_num_col], errors="coerce").fillna(0)
                cur_ps = ps[ps["season"] == season]
                if cur_ps.empty:
                    continue
                grouped = cur_ps.groupby(["team", "player"], as_index=False).agg(
                    position=("position", lambda s: s.dropna().mode().iloc[0] if len(s.dropna()) else None),
                    goals=("goals", "sum"),
                    assists=("assists", "sum"),
                    minutes=("minutes", "sum"),
                )
                for _, row in grouped.iterrows():
                    team = _tk(row["team"])
                    name = _pk(row["player"])
                    rec = {
                        "position": row["position"],
                        "goals": int(row["goals"] or 0),
                        "assists": int(row["assists"] or 0),
                        "minutes": int(row["minutes"] or 0),
                    }
                    # Don't overwrite if existing record has more minutes (prefer richer source)
                    existing = fb_map.get((team, name))
                    if existing is None or rec["minutes"] >= (existing.get("minutes") or 0):
                        fb_map[(team, name)] = rec
                    fb_name_only.setdefault(name, []).append((team, rec))
            except Exception as e:
                log.debug("%s failed: %s", _ps_file, e)

        # ---- understat_players.parquet → xG PER TEAM ----
        try:
            up = _pd.read_parquet(
                DATA_DIR / "parsed" / "understat_players.parquet",
                columns=["season", "team", "player", "xg"],
            )
            cur_up = up[up["season"] == season]
            for _, row in cur_up.iterrows():
                if _pd.isna(row["xg"]):
                    continue
                team = _tk(row["team"])
                name = _pk(row["player"])
                rec = {"xg": float(row["xg"])}
                # Keep highest if duplicated
                if (team, name) not in us_map or rec["xg"] > us_map[(team, name)]["xg"]:
                    us_map[(team, name)] = rec
                us_name_only.setdefault(name, []).append((team, rec))
        except Exception as e:
            log.debug("understat_players.parquet failed: %s", e)
    except ImportError:
        log.debug("pandas not available; enrichment skipped")

    # ---- Team-normalization helper — lineup team might be "Lecce", fbref "Lecce" too,
    # but fuzzy prefixes (e.g. "Manchester United" vs "Man United") happen.
    def _team_match(lineup_team: str, source_team: str) -> bool:
        lt = _tk(lineup_team)
        st = _tk(source_team)
        if lt == st:
            return True
        # Prefix / contains match
        return lt in st or st in lt

    def _lookup_shirt(team: str, name: str) -> str | None:
        tk = _tk(team); pk = _pk(name)
        if (tk, pk) in shirt_map:
            return shirt_map[(tk, pk)]
        surname = pk.split()[-1].lower()
        key = (tk, "_surname_" + surname)
        if key in shirt_map:
            return shirt_map[key]
        # Cross-team: any team that matches (handles "Nott'm Forest" vs "Nottingham Forest")
        for (t, n), shirt in shirt_map.items():
            if _team_match(team, t) and _pk(n).lower() == pk.lower():
                return shirt
        # Surname-only last resort
        for (t, n), shirt in shirt_map.items():
            if n.startswith("_surname_"):
                continue
            if _team_match(team, t) and _pk(n).split()[-1].lower() == surname:
                return shirt
        return None

    def _lookup_fb(team: str, name: str) -> dict | None:
        tk = _tk(team); pk = _pk(name)
        if (tk, pk) in fb_map:
            return fb_map[(tk, pk)]
        # Exact-name match on any matching team (handles team alias)
        for (t, n), rec in fb_map.items():
            if n == pk and _team_match(team, t):
                return rec
        # Fuzzy: same surname + matching team
        surname = pk.split()[-1].lower()
        for (t, n), rec in fb_map.items():
            if _team_match(team, t) and n.split()[-1].lower() == surname:
                return rec
        # Last resort: name match across ANY team (rare)
        if pk in fb_name_only:
            return fb_name_only[pk][0][1]
        return None

    def _lookup_xg(team: str, name: str) -> float:
        tk = _tk(team); pk = _pk(name)
        if (tk, pk) in us_map:
            return us_map[(tk, pk)]["xg"]
        for (t, n), rec in us_map.items():
            if n == pk and _team_match(team, t):
                return rec["xg"]
        surname = pk.split()[-1].lower()
        for (t, n), rec in us_map.items():
            if _team_match(team, t) and n.split()[-1].lower() == surname:
                return rec["xg"]
        return 0.0

    # ---- Auxiliary: kickoff map + active picks ----
    try:
        kickoff_map = _load_kickoff_map()
    except Exception:
        kickoff_map = {}

    slip_path = DATA_DIR / "betting" / "betting_slip.json"
    active_picks_by_match: dict = {}
    try:
        if slip_path.exists():
            slip = json.loads(slip_path.read_text())
            for pick in slip.get("recommended_singles", []) or []:
                active_picks_by_match.setdefault(pick.get("match", ""), []).append(pick)
    except Exception:
        pass

    def _resolve_kickoff(mk, md):
        ct = md.get("commence_time") or md.get("kickoff_time")
        if ct:
            return str(ct)
        home = md.get("home_team") or (mk.split(" vs ")[0] if " vs " in mk else "")
        away = md.get("away_team") or (mk.split(" vs ")[1] if " vs " in mk else "")
        return kickoff_map.get((home, away), "")

    confirmed.sort(key=lambda pair: _resolve_kickoff(pair[0], pair[1]) or "z")

    # ---- Detect EPL matches with degraded enrichment (honest-banner guard) ----
    # Trigger if any EPL match in this batch has EITHER:
    #   (a) >8/11 missing shirt numbers, OR
    #   (b) zero G+A totals summed across all outfielders
    # Empirical signals that player_stats_epl is backfill-pending for 2025-26
    # and lineups.parquet has no EPL coverage yet.
    def _is_epl_match(mk, md) -> bool:
        ht = md.get("home_team") or (mk.split(" vs ")[0] if " vs " in mk else "")
        at = md.get("away_team") or (mk.split(" vs ")[1] if " vs " in mk else "")
        epl_path = DATA_DIR / "upcoming" / "odds_full_premier_league.json"
        try:
            if epl_path.exists():
                epl = json.loads(epl_path.read_text())
                epl_matches = epl.get("matches") or {}
                if mk in epl_matches or f"{ht} vs {at}" in epl_matches:
                    return True
        except Exception:
            pass
        return False

    epl_data_gap = False
    for mk, md in confirmed:
        if not _is_epl_match(mk, md):
            continue
        ht = md.get("home_team") or (mk.split(" vs ")[0] if " vs " in mk else "")
        at = md.get("away_team") or (mk.split(" vs ")[1] if " vs " in mk else "")
        h_xi = list(md.get("home_lineup", []))[:11]
        a_xi = list(md.get("away_lineup", []))[:11]
        shirt_misses = 0
        ga_total = 0
        for team, xi in ((ht, h_xi), (at, a_xi)):
            for name in xi:
                if _lookup_shirt(team, name) is None:
                    shirt_misses += 1
                fb = _lookup_fb(team, name) or {}
                pos = (fb.get("position") or "").upper()
                if pos != "GK":
                    ga_total += int(fb.get("goals") or 0) + int(fb.get("assists") or 0)
        if shirt_misses > 8 or ga_total == 0:
            epl_data_gap = True
            break

    NBSP = " "
    now = datetime.now()
    tg = TgMsg()
    tg.raw(f"📋 <b>Lineups Confirmed</b>  <i>{now.strftime('%H:%M')}</i>")
    if epl_data_gap:
        tg.blank()
        tg.raw(
            "⚠️ <i>EPL player stats are backfill-pending for 2025-26 — "
            "shirt numbers and season totals may be missing.</i>"
        )

    def _pos_glyph(position: str | None) -> str:
        if not position:
            return "  "
        p = position.upper().split(",")[0]
        if p == "GK":
            return "🧤"
        if p in ("FW", "ST", "W", "LW", "RW", "CF"):
            return "⚽"
        if p in ("DF", "CB", "LB", "RB", "FB", "WB", "LWB", "RWB"):
            return "🛡️"
        if p in ("MF", "CM", "DM", "AM", "LM", "RM", "CDM", "CAM"):
            return "🎯"
        return "  "

    def _stat_line(name: str, team: str, position: str | None) -> str:
        fb = _lookup_fb(team, name) or {}
        goals = fb.get("goals") or 0
        assists = fb.get("assists") or 0
        xg = _lookup_xg(team, name)
        p = (position or "").upper()

        # GK: keep minimal — stats for GKs (e.g. clean sheets) aren't in scope
        if p == "GK":
            return ""

        bits = []
        if goals:
            bits.append(f"⚽{goals}")
        if assists:
            bits.append(f"🅰{assists}")
        # Always show xG when >= 0.5 (meaningful sample), for ALL outfield roles
        if xg >= 0.5:
            bits.append(f"xG {xg:.1f}")
        return " ".join(bits)

    def _render_xi(team: str, xi: list[str]):
        for name in xi:
            shirt = _lookup_shirt(team, name)
            fb = _lookup_fb(team, name) or {}
            position = fb.get("position")
            glyph = _pos_glyph(position)
            tail = _stat_line(name, team, position)

            # Shirt in monospace for column alignment
            shirt_cell = f"<code>{(shirt or '--'):>2}</code>"

            # Single format (no position text — glyph already conveys role)
            line = f"    {shirt_cell}{NBSP}{glyph}{NBSP}<b>{_html_escape(str(name))}</b>"
            if tail:
                line += f"  <i>{_html_escape(tail)}</i>"
            tg.raw(line)

    for mk, md in confirmed:
        home_team = md.get("home_team") or (mk.split(" vs ")[0] if " vs " in mk else "?")
        away_team = md.get("away_team") or (mk.split(" vs ")[1] if " vs " in mk else "?")

        kick_iso = _resolve_kickoff(mk, md)
        hhmm = _kickoff_display(kick_iso) if kick_iso else ""
        time_chip = (
            f"<b>[{NBSP}{_html_escape(hhmm)}{NBSP}]</b>"
            if hhmm else f"<b>[{NBSP}{NBSP}—{NBSP}{NBSP}{NBSP}{NBSP}]</b>"
        )

        # League badge
        league_key = ""
        epl_path = DATA_DIR / "upcoming" / "odds_full_premier_league.json"
        try:
            if epl_path.exists():
                epl = json.loads(epl_path.read_text())
                if mk in (epl.get("matches") or {}) or f"{home_team} vs {away_team}" in (epl.get("matches") or {}):
                    league_key = "premier_league"
        except Exception:
            pass
        if not league_key and kickoff_map.get((home_team, away_team)):
            league_key = "serie_a"
        badge = _LEAGUE_BADGE_MAP.get(league_key, "")

        tg.blank()
        tg.raw(
            f"{time_chip}{NBSP}{NBSP}{badge}{NBSP}"
            f"<b>{_html_escape(home_team)} – {_html_escape(away_team)}</b>"
        )
        home_fmt = md.get("home_formation") or ""
        away_fmt = md.get("away_formation") or ""
        if home_fmt or away_fmt:
            tg.raw(f"  <i>{_html_escape(home_fmt or '?')}  vs  {_html_escape(away_fmt or '?')}</i>")

        tg.blank()
        tg.raw(f"  <b>🏠 {_html_escape(home_team)}</b>")
        _render_xi(home_team, list(md.get("home_lineup", []))[:11])
        tg.blank()
        tg.raw(f"  <b>✈️ {_html_escape(away_team)}</b>")
        _render_xi(away_team, list(md.get("away_lineup", []))[:11])

        # ---- Match preview (pundit-style narrative) ----
        try:
            pred = _load_prediction_for(mk, home_team, away_team)
            preview = _build_match_preview(
                pred, home_team, away_team,
                list(md.get("home_lineup", [])), list(md.get("away_lineup", [])),
                fb_map, us_map, _tk, _team_match,
            )
            if preview:
                tg.blank()
                tg.raw("  <b>🧠 Preview</b>")
                tg.raw(preview)
        except Exception as e:
            log.debug("match preview build failed: %s", e)

        picks = active_picks_by_match.get(mk, []) or active_picks_by_match.get(
            f"{home_team} vs {away_team}", []
        )
        for pick in picks:
            sel = pick.get("selection") or pick.get("prediction") or ""
            odds = pick.get("odds") or pick.get("best_odds")
            edge = pick.get("value_pct") or pick.get("edge_pct") or 0
            bits = []
            if sel:
                bits.append(f"<b>{_html_escape(str(sel))}</b>")
            if odds:
                try:
                    bits.append(f"@{float(odds):.2f}")
                except Exception:
                    bits.append(f"@{odds}")
            try:
                edge_v = float(edge)
                if edge_v:
                    bits.append(f"<i>edge {'+' if edge_v >= 0 else ''}{edge_v:.0f}%</i>")
            except Exception:
                pass
            tg.blank()
            tg.raw(f"  🎯 <b>Active bet:</b> {' '.join(bits)}")

    teams_text = ", ".join(
        f"{md.get('home_team','?')}–{md.get('away_team','?')}" for _, md in confirmed
    )
    mac_msg = f"Lineups in: {teams_text}"[:200]

    result = notify(
        mac_msg,
        title="📋 Lineups Confirmed",
        level="info",
        category="betting",
        tg_html=tg.build(),
    )
    try:
        with open(_lineup_dedup_path, "w") as _f:
            json.dump({"date": today_str, "sig": msg_sig}, _f)
    except Exception:
        pass
    return result


def notify_predictions_ready(n_matches: int = 0) -> dict:
    """Send a coaching-style notification when predictions are refreshed."""
    opener = random.choice(_PREDICTIONS_READY_OPENERS)
    detail = f" {n_matches} matches updated." if n_matches else ""
    msg = f"{opener}{detail}"
    return notify(msg, title="Predictions Ready", level="info", category="betting")


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

    # Extract top picks BEFORE dedup check (was causing NameError)
    top_picks = parlay_report.get("top_picks", []) if parlay_report else []

    # Dedup
    today_str = datetime.now().strftime("%Y-%m-%d")
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
        msg = f"{opener}\n\nBankroll crossed \u20ac{milestone:,.0f} (now \u20ac{new_balance:,.2f})."
        msg += "\nStay disciplined — the edge compounds."
        return notify(msg, title=f"Milestone: \u20ac{milestone:,.0f}", level="success", category="betting")

    # Check downward milestones (every $500)
    old_five = int(old_balance // 500)
    new_five = int(new_balance // 500)

    if new_five < old_five and new_balance < old_balance:
        milestone = (new_five + 1) * 500
        opener = random.choice(_BANKROLL_MILESTONE_DOWN)
        msg = f"{opener}\n\nBankroll dropped below \u20ac{milestone:,.0f} (now \u20ac{new_balance:,.2f})."
        msg += "\nStick to the system. Variance is part of the game."
        return notify(msg, title=f"Below \u20ac{milestone:,.0f}", level="warning", category="betting")

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

    # Check if this is a post-improvement bet
    placed_at = (bet.get("placed_at") or "")[:10]
    is_new_method = placed_at >= "2026-04-10"
    new_tag = " \U0001f195" if is_new_method else ""  # 🆕

    tg = TgMsg()
    tg.raw(f"{emoji} <b>{_html_escape(match)}</b>{new_tag}")
    tg.raw(f"{_html_escape(selection)} @{odds:.2f} | "
           f"Stake \u20ac{stake:.2f}")
    if result_score:
        tg.raw(f"Score: <b>{_html_escape(result_score)}</b>")
    tg.pnl(profit)

    # CLV (Closing Line Value) — did we beat the market?
    clv = bet.get("clv_pct")
    if clv is not None:
        clv_emoji = "\U0001f4c8" if clv > 0 else "\U0001f4c9"  # 📈 or 📉
        tg.raw(f"{clv_emoji} CLV: {clv:+.1f}% {'(beat closing line)' if clv > 0 else '(below closing line)'}")

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

    if accuracy_pct >= 60:
        mood = "Excellent week. Model is sharp."
    elif accuracy_pct >= 51:
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
        level="success" if accuracy_pct >= 51 else "warning",
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

    # Dedup: don't send same lineup impact twice
    import hashlib as _hl
    today_str = datetime.now().strftime("%Y-%m-%d")
    impact_sig = _hl.md5(f"{match}|{new_pred}".encode()).hexdigest()[:12]
    _impact_dedup_path = DATA_DIR / ".lineup_impact_dedup.json"
    try:
        if _impact_dedup_path.exists():
            with open(_impact_dedup_path) as _f:
                _dedup = json.load(_f)
            if _dedup.get("date") == today_str and impact_sig in _dedup.get("sigs", []):
                log.info("Lineup impact skipped (already sent for %s today)", match)
                return {}
    except Exception:
        pass

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

    result = notify(
        message=mac_msg,
        title=f"Lineup shift: {match}",
        level="warning" if max_shift > 0.05 else "info",
        category="betting",
        tg_html=tg.build(),
    )
    # Save dedup marker
    try:
        existing_sigs = []
        if _impact_dedup_path.exists():
            with open(_impact_dedup_path) as _f:
                _d = json.load(_f)
            if _d.get("date") == today_str:
                existing_sigs = _d.get("sigs", [])
        existing_sigs.append(impact_sig)
        with open(_impact_dedup_path, "w") as _f:
            json.dump({"date": today_str, "sigs": existing_sigs}, _f)
    except Exception:
        pass
    return result


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

        with open(journal_path) as _jf:
            journal = _json.load(_jf)
        bets = journal.get("bets", {})

        # Detect current matchweek from matches data
        if matchweek == 0:
            try:
                import pandas as pd
                matches_path = _Path(__file__).parent.parent.parent / "data" / "parsed" / "matches.parquet"
                if matches_path.exists():
                    mdf = pd.read_parquet(matches_path)
                    current = mdf[mdf["season"] == "2025-2026"]
                    # Filter to single league to avoid inflated MW numbers
                    if "league" in current.columns:
                        current = current[current["league"] == "serie_a"]
                    if "matchweek" in current.columns:
                        # Find the latest matchweek that has at least 5 matches
                        mw_counts = current.groupby("matchweek").size()
                        for mw in sorted(mw_counts.index, reverse=True):
                            if mw_counts[mw] >= 5:
                                matchweek = int(mw)
                                break
            except Exception:
                pass

        # Get bets from the last 10 days (covers split matchweeks)
        from datetime import datetime, timedelta
        cutoff = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")

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
        total_profit = sum(
            (b.get("profit") or 0)
            for b in week_bets
            if b.get("status") in ("won", "lost", "push")
        )

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

        # Group bets by league for multi-league display
        leagues_in_bets = set(_resolve_league(b) for b in week_bets)
        show_league_headers = len(leagues_in_bets) > 1

        LEAGUE_NAMES = {
            "serie_a": "Serie A",
            "epl": "Premier League",
            "premier_league": "Premier League",
            "la_liga": "La Liga",
            "bundesliga": "Bundesliga",
            "ligue_1": "Ligue 1",
        }

        if show_league_headers:
            # Sort bets by league then date
            week_bets.sort(key=lambda b: (_resolve_league(b), b.get("date", "")))

        current_league = None
        for b in week_bets:
            bet_league = _resolve_league(b)

            # Show league header when switching leagues
            if show_league_headers and bet_league != current_league:
                current_league = bet_league
                league_display = LEAGUE_NAMES.get(bet_league, bet_league.replace("_", " ").title())
                tg.raw(f"\U0001f3c6 <b>{_html_escape(league_display)}</b>")
                tg.blank()

            match = b.get("match", "?")
            sel = b.get("selection", "?")
            market_raw = b.get("market", "")
            market = MKT_NAMES.get(market_raw, market_raw)
            odds = b.get("odds", 0)
            stake = b.get("stake", 0)
            status = b.get("status", "")
            score = b.get("result_score", "")
            bet_date = b.get("date", "")

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

            # Date + match header
            date_str = f"  {_html_escape(bet_date)}" if bet_date else ""
            score_str = f"  ({_html_escape(score)})" if score else ""
            tg.raw(f"{icon} <b>{_html_escape(match)}</b>{score_str}")
            tg.raw(f"   {_html_escape(bet_date)}")
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


# ---------------------------------------------------------------------------
# Scheduler-run cards + health-state alerts
# ---------------------------------------------------------------------------

_SCHEDULER_STATE_PATH = DATA_DIR / "monitoring" / "scheduler_state.json"
_HEALTH_STATE_PATH = DATA_DIR / "monitoring" / "health_state.json"

_SCHEDULER_BADGE = {
    "morning":              ("☀️", "Morning Pipeline"),
    "evening":              ("\U0001f30c", "Evening Pipeline"),
    "settlement":           ("\U0001f9fe", "Settlement Run"),
    "odds-refresh":         ("\U0001f4b1", "Odds Refresh"),
    "health-monitor":       ("\U0001fa7a", "Health Monitor"),
    "pre-kickoff-monitor":  ("⏰", "Pre-Kickoff Watch"),
    "weekly-data-refresh":  ("\U0001f4e6", "Weekly Data Refresh"),
    "weekly-monitor":       ("\U0001f9ea", "Weekly Monitor"),
    "matchweek-retrain":    ("\U0001f501", "Matchweek Retrain"),
    "sunday-retrain":       ("\U0001f501", "Sunday Retrain"),
    "monthly-retrain":      ("\U0001f3af", "Monthly Retrain"),
    "daily-digest":         ("\U0001f4ca", "Daily Digest"),
}

_STATUS_ICON = {
    "ok":      "✅",
    "success": "✅",
    "skipped": "⏩",
    "warn":    "⚠️",
    "warning": "⚠️",
    "fail":    "❌",
    "failed":  "❌",
    "error":   "❌",
}


def _load_json_safe(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception as e:
        log.debug("Failed to load %s: %s", path, e)
    return default


def _save_json_safe(path: Path, data) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str))
    except Exception as e:
        log.warning("Failed to save %s: %s", path, e)


def notify_pipeline_run_with_picks(
    name: str,
    status: str = "success",
    duration_sec: float | None = None,
    error: str | None = None,
    max_picks: int = 6,
) -> dict:
    """Specialized morning/evening pipeline card — leads with the VALUE BETS,
    not run metadata. Reads the current slip + predictions; shows up to
    `max_picks` value bets with edge + kickoff, or a single dry line when
    nothing was found.

    Falls through to notify_scheduler_run on failure so the failure UI is
    consistent with other schedulers.
    """
    import json as _json
    from pathlib import Path as _Path

    status_l = (status or "ok").lower()
    is_failure = status_l in ("fail", "failed", "error")

    # Failure path — use the standard scheduler card (with error)
    if is_failure:
        return notify_scheduler_run(
            name=name,
            status=status,
            duration_sec=duration_sec,
            details=None,
            error=error,
        )

    # ---- Persist state so the digest's Systems block still shows the run ----
    now = datetime.now()
    state = _load_json_safe(_SCHEDULER_STATE_PATH, {})
    state[name] = {
        "status": status_l,
        "last_run": now.isoformat(timespec="seconds"),
        "duration_sec": round(duration_sec, 1) if duration_sec else None,
        "details": {},
        "error": None,
    }
    _save_json_safe(_SCHEDULER_STATE_PATH, state)

    emoji, label = _SCHEDULER_BADGE.get(name, ("⚙️", name.replace("-", " ").title()))
    when = now.strftime("%H:%M")

    # ---- Read the current betting slip ----
    slip_path = DATA_DIR / "betting" / "betting_slip.json"
    slip: dict = {}
    try:
        if slip_path.exists():
            slip = _json.loads(slip_path.read_text())
    except Exception as e:
        log.debug("pipeline card: slip load failed: %s", e)

    picks = slip.get("recommended_singles", []) if isinstance(slip, dict) else []

    # Slip staleness — if >24h old, don't pretend it's today's picks
    slip_is_stale = False
    try:
        gen_at = slip.get("generated_at") if isinstance(slip, dict) else None
        if gen_at:
            gen_dt = datetime.fromisoformat(gen_at)
            age_h = (now - gen_dt).total_seconds() / 3600
            slip_is_stale = age_h > 24
    except Exception:
        pass

    # Kickoff map (for picks that lack kickoff_time)
    try:
        kmap = _load_kickoff_map()
    except Exception:
        kmap = {}

    # ---- Build the Telegram card ----
    tg = TgMsg()
    header = f"✅ <b>{_html_escape(label)}</b>  <i>{_html_escape(when)}</i>"
    if duration_sec is not None and duration_sec > 900:  # only show duration if slow (>15m)
        dur_str = f"{duration_sec/60:.0f}m"
        header += f"  <i>⏱️ took {dur_str}</i>"
    tg.raw(header)

    if slip_is_stale:
        tg.blank()
        tg.raw("⚠️ <i>Slip is stale — pipeline couldn't refresh picks today.</i>")
    elif not picks:
        tg.blank()
        tg.raw("<i>No value bets today.</i>")
    else:
        tg.blank()
        n = len(picks)
        shown = picks[:max_picks]
        tg.raw(f"🎯 <b>{n} value bet{'s' if n != 1 else ''}</b>")

        NBSP = " "

        def _kickoff_for(pick):
            kt = pick.get("kickoff_time") or ""
            if kt and ("T" in str(kt) or ":" in str(kt)):
                return str(kt)
            m = pick.get("match") or ""
            if " vs " in m:
                h, a = m.split(" vs ", 1)
                return kmap.get((h.strip(), a.strip()), "")
            return ""

        def _league_for(pick):
            # Heuristic: if both teams are in matches_epl, tag EPL
            m = pick.get("match") or ""
            # Prefer explicit league tag; else look up from the pick
            lg = pick.get("league") or ""
            if lg:
                return lg
            # Fall back to kickoff map hit: we put EPL commence_time from the
            # EPL odds file into kmap, so a hit from that source implies EPL.
            # (This is a best-effort heuristic; the badge is cosmetic.)
            return ""

        def _sort_key(pick):
            k = _kickoff_for(pick)
            if "T" in k:
                return (0, k)
            return (1, pick.get("match", ""))

        for pick in sorted(shown, key=_sort_key):
            match = pick.get("match", "")
            sel = pick.get("selection") or pick.get("prediction") or ""
            odds = pick.get("odds") or pick.get("best_odds")
            edge = pick.get("value_pct") or pick.get("edge_pct") or 0

            kick_iso = _kickoff_for(pick)
            hhmm = _kickoff_display(kick_iso) if kick_iso else ""
            if hhmm:
                time_chip = f"<b>[{NBSP}{_html_escape(hhmm)}{NBSP}]</b>"
            else:
                time_chip = f"<b>[{NBSP}{NBSP}—{NBSP}{NBSP}{NBSP}{NBSP}]</b>"

            # Compact edge/odds line: "HOME @2.95 edge +18%"
            bits = []
            if sel:
                bits.append(f"<b>{_html_escape(str(sel))}</b>")
            if odds:
                try:
                    bits.append(f"@{float(odds):.2f}")
                except (TypeError, ValueError):
                    bits.append(f"@{odds}")
            if edge:
                try:
                    edge_v = float(edge)
                    sign = "+" if edge_v >= 0 else ""
                    bits.append(f"<i>edge {sign}{edge_v:.0f}%</i>")
                except (TypeError, ValueError):
                    pass
            pick_line = " ".join(bits)

            tg.raw(f"  {time_chip}{NBSP}{NBSP}{_html_escape(match)} — {pick_line}")

        if n > max_picks:
            tg.raw(f"  <i>… and {n - max_picks} more</i>")

    # ---- macOS (silent on success except when slow) ----
    macos_msg = ""
    if duration_sec is not None and duration_sec > 900:
        macos_msg = f"{label}: done ({duration_sec/60:.0f}m)"
    elif not picks and not slip_is_stale:
        # Silent — nothing actionable
        macos_msg = ""
    # else: still silent; picks are in Telegram

    return notify(
        macos_msg,
        title=label,
        level="success",
        category="system",
        priority=PRIORITY_LOW,
        tg_html=tg.build(),
    )


def notify_scheduler_run(
    name: str,
    status: str = "ok",
    duration_sec: float | None = None,
    details: dict | None = None,
    error: str | None = None,
) -> dict:
    """Post a run card for a scheduled launchd job.

    UI principles:
      - ONE title line only (no echoed emoji+title in body)
      - FAIL suppresses stale detail numbers (they lie on a failed run)
      - Low-priority successful runs are silent on macOS (Telegram only)
      - Errors shown without "see log" boilerplate
    """
    emoji, label = _SCHEDULER_BADGE.get(name, ("⚙️", name.replace("-", " ").title()))
    status_l = (status or "ok").lower()
    icon = _STATUS_ICON.get(status_l, "ℹ️")
    is_failure = status_l in ("fail", "failed", "error")
    is_success = status_l in ("ok", "success", "skipped")

    level = {
        "ok": "info", "success": "success", "skipped": "info",
        "warn": "warning", "warning": "warning",
        "fail": "error", "failed": "error", "error": "error",
    }.get(status_l, "info")

    now = datetime.now()
    when = now.strftime("%H:%M")

    # Persist last-run state (consumed by daily-digest Systems block)
    state = _load_json_safe(_SCHEDULER_STATE_PATH, {})
    state[name] = {
        "status": status_l,
        "last_run": now.isoformat(timespec="seconds"),
        "duration_sec": round(duration_sec, 1) if duration_sec else None,
        "details": details or {},
        "error": error,
    }
    _save_json_safe(_SCHEDULER_STATE_PATH, state)

    # Format duration compactly
    dur_str = ""
    if duration_sec is not None:
        dur_str = f"{duration_sec:.0f}s" if duration_sec < 60 else f"{duration_sec/60:.1f}m"

    # ---------- Telegram (rich, single title) ----------
    tg = TgMsg()
    # One header line: icon + label + status + time + duration
    header_bits = [f"{icon} <b>{_html_escape(label)}</b>"]
    if is_success and dur_str:
        header_bits.append(f"<i>{_html_escape(when)}</i>")
        header_bits.append(f"<i>{dur_str}</i>")
    elif is_failure and dur_str:
        header_bits.append(f"<i>{_html_escape(when)}</i>")
        header_bits.append(f"<i>failed after {dur_str}</i>")
    else:
        header_bits.append(f"<i>{_html_escape(when)}</i>")
        if dur_str:
            header_bits.append(f"<i>{dur_str}</i>")
    tg.raw("  ".join(header_bits))

    # On FAIL, SUPPRESS stale detail numbers (they're leftover values from
    # the previous successful run). Only show details on success/warn.
    if details and not is_failure:
        shown = [(k, v) for k, v in details.items() if v not in (None, "")]
        if shown:
            tg.blank()
            for k, v in shown:
                tg.kv(str(k), str(v))

    # Error: just the message, no "see launchd err log" boilerplate
    if error:
        err_clean = error.replace(" (see launchd err log)", "").replace(
            "Pipeline failed after retries", "Pipeline retries exhausted"
        )
        tg.blank().raw(f"⚠️ {_html_escape(err_clean[:250])}")

    # ---------- macOS (plain text, ≤ 200 chars, one-liner) ----------
    # Successful routine runs — macOS is silent (Telegram has the record)
    macos_msg = ""
    if not is_success:
        parts = [f"{label}: {status_l}"]
        if dur_str:
            parts[-1] += f" ({dur_str})"
        if error:
            err_short = (error.replace(" (see launchd err log)", "")
                         .replace("Pipeline failed after retries", "Pipeline retries exhausted"))
            parts.append(err_short[:100])
        macos_msg = " — ".join(parts)[:200]

    # Priority tiers: routine → LOW (no phone buzz); fail → NORMAL
    priority = PRIORITY_LOW if is_success else PRIORITY_NORMAL
    # Category: alerts bypass system-category suppression
    category = "alert" if is_failure else "system"

    return notify(
        macos_msg,  # empty string → macOS skipped via _notify_macos early-out
        title=f"{label}",  # single emoji in tg body, not in title
        level=level,
        category=category,
        priority=priority,
        tg_html=tg.build(),
    )


def notify_health_state_change(current: dict) -> dict:
    """Fire a Telegram+macOS alert only when health status MEANINGFULLY changes.

    Dedup strategy (fixes the thrashing we saw on 2026-04-24 where the same
    issue flapped between "7.5d ago" and "7.6d ago" every 30 min):

      - Issue IDENTITY = the bracketed category prefix + stable suffix
        (e.g. "[health_check] Data file stale: odds_data"), NOT the full
        text. The "7.5d ago" part is stripped so natural staleness progression
        doesn't trigger re-alerts.

      - Transient-suppression: if an issue appears + disappears within
        `_TRANSIENT_WINDOW_MIN` minutes (default 45), it is treated as noise
        and no alert fires. Catches Gemini 503s, temporary timeouts, etc.

      - Long-standing-issue silencing: if an issue has been present for more
        than `_QUIET_AFTER_HOURS` (default 24), no further re-alerts — you
        already know about it. Only new/resolved issues ping you.

    Silent when nothing meaningful changed.
    """
    import re as _re
    prev_state = _load_json_safe(_HEALTH_STATE_PATH, {})
    is_first_run = not prev_state

    _TRANSIENT_WINDOW_MIN = 45
    _QUIET_AFTER_HOURS = 24
    _NOW = datetime.now()

    def _issue_key(level: str, msg: str) -> str:
        """Stable identity for an issue: strip volatile suffixes like
        "7.6d ago", "(0 requests remaining)", decimal-changing %."""
        s = msg
        # Strip "(N.Nd ago)" / "(N.Nh ago)" / "N.Nd ago" tails
        s = _re.sub(r"\(\d+(?:\.\d+)?\s*[dh]\s*ago\)", "", s)
        s = _re.sub(r"\d+(?:\.\d+)?\s*[dh]\s*ago", "", s)
        # Strip floating-point numbers inside parens like "(0 requests)" etc.
        s = _re.sub(r"\(\d+(?:\.\d+)?[^)]*\)", "", s)
        # Strip percentages like "7.5% from peak"
        s = _re.sub(r"\d+(?:\.\d+)?%", "N%", s)
        # Strip trailing numeric counts like "11 issues"
        s = _re.sub(r"\d+\s*(?:bets?|items?|entries?|matches?|issues?)", "N \\g<0>", s)
        return f"{level}|{s.strip().rstrip(':').rstrip()}"

    # Load previous key-indexed issue state
    prev_keys: dict = prev_state.get("issue_keys", {})
    # {key: {'first_seen': ISO, 'last_seen': ISO, 'level': str, 'message': str, 'flap_count': int}}

    cur_issues_by_key: dict = {}
    for entry in current.get("issues", []):
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            k = _issue_key(entry[0], entry[1])
            cur_issues_by_key[k] = {"level": entry[0], "message": entry[1]}

    # Update state
    new_keys: dict = {}
    now_iso = _NOW.isoformat(timespec="seconds")
    truly_new: list[tuple[str, str]] = []  # (level, message) to announce
    truly_resolved: list[tuple[str, str]] = []

    # Handle newly-appeared keys
    for k, payload in cur_issues_by_key.items():
        if k in prev_keys:
            prev_payload = prev_keys[k]
            new_keys[k] = {
                "first_seen": prev_payload.get("first_seen", now_iso),
                "last_seen": now_iso,
                "level": payload["level"],
                "message": payload["message"],
                "flap_count": prev_payload.get("flap_count", 0),
            }
        else:
            # It wasn't there last time. Check if it existed recently (flap).
            recently_resolved = prev_state.get("recently_resolved", {}) or {}
            gone_at = recently_resolved.get(k)
            flap = 0
            if gone_at:
                try:
                    gone_dt = datetime.fromisoformat(gone_at)
                    if (_NOW - gone_dt).total_seconds() / 60 < _TRANSIENT_WINDOW_MIN:
                        flap = 1  # reappeared shortly after resolving
                except Exception:
                    pass
            new_keys[k] = {
                "first_seen": now_iso,
                "last_seen": now_iso,
                "level": payload["level"],
                "message": payload["message"],
                "flap_count": flap,
            }
            # Only announce if not a flap and not the first run
            if not is_first_run and flap == 0:
                truly_new.append((payload["level"], payload["message"]))

    # Handle just-resolved keys
    resolved_tracking: dict = dict(prev_state.get("recently_resolved", {}) or {})
    for k, prev_payload in prev_keys.items():
        if k not in cur_issues_by_key:
            # Only announce resolution if issue had been present >_TRANSIENT_WINDOW_MIN
            # (otherwise it was a transient and we didn't alert on it in the first
            # place, so don't tell the user it "resolved").
            try:
                first_seen = datetime.fromisoformat(prev_payload.get("first_seen", now_iso))
                age_min = (_NOW - first_seen).total_seconds() / 60
            except Exception:
                age_min = 0
            if age_min >= _TRANSIENT_WINDOW_MIN and not is_first_run:
                truly_resolved.append(
                    (prev_payload.get("level", "?"), prev_payload.get("message", "?"))
                )
            # Track resolution timestamp for flap detection on next cycle
            resolved_tracking[k] = now_iso

    # Prune old resolved-tracking entries (>2h old)
    trimmed = {}
    for k, iso in resolved_tracking.items():
        try:
            if (_NOW - datetime.fromisoformat(iso)).total_seconds() / 3600 < 2:
                trimmed[k] = iso
        except Exception:
            pass
    resolved_tracking = trimmed

    # Overall status — check for meaningful flips
    odds_restored = (
        (prev_state.get("odds_status") == "CRITICAL")
        and ((current.get("checks", {}).get("odds_api_key", {}) or {}).get("status") == "OK")
    )

    # Persist new state
    _save_json_safe(_HEALTH_STATE_PATH, {
        "overall_status": current.get("overall_status"),
        "odds_status": (current.get("checks", {}).get("odds_api_key", {}) or {}).get("status"),
        "issue_keys": new_keys,
        "recently_resolved": resolved_tracking,
        "updated": now_iso,
    })

    # No alert if nothing meaningful to report
    if not (truly_new or truly_resolved or odds_restored):
        log.debug("Health unchanged (dedup by issue key) — no alert")
        return {}

    overall = current.get("overall_status", "UNKNOWN")
    emoji = "\U0001f6a8" if overall == "CRITICAL" else "✅"
    tg = TgMsg().title(f"Health  •  {_NOW.strftime('%H:%M')}", emoji=emoji)
    tg.raw(f"<b>Overall:</b> {_html_escape(overall)}")
    tg.blank()

    if odds_restored:
        tg.raw("\U0001f389 <b>Odds API quota RESTORED</b>")
        tg.blank()

    if truly_new:
        tg.raw("\U0001f534 <b>New</b>")
        for lvl, msg in sorted(truly_new):
            tg.raw(f"  • <i>{_html_escape(lvl)}</i>  {_html_escape(msg[:140])}")
        tg.blank()

    if truly_resolved:
        tg.raw("\U0001f7e2 <b>Resolved</b>")
        for lvl, msg in sorted(truly_resolved):
            tg.raw(f"  • <i>{_html_escape(lvl)}</i>  {_html_escape(msg[:140])}")
        tg.blank()

    # macOS one-liner
    parts = []
    if truly_new:
        parts.append(f"{len(truly_new)} new")
    if truly_resolved:
        parts.append(f"{len(truly_resolved)} resolved")
    if odds_restored:
        parts.append("odds API back")
    macos_msg = f"Health: {overall}" + (f" — {', '.join(parts)}" if parts else "")

    # Only critical-level on NEW CRITICAL issues, not on resolutions
    has_critical_new = any(lvl == "CRITICAL" for lvl, _ in truly_new)
    level = (
        "critical" if has_critical_new else
        "success" if odds_restored or truly_resolved else
        "warning"
    )
    priority = PRIORITY_URGENT if level == "critical" else PRIORITY_NORMAL

    return notify(
        macos_msg[:250],
        title="\U0001fa7a Health",
        level=level,
        category="alert",
        priority=priority,
        tg_html=tg.build(),
    )


def notify_scheduler_failure(name: str, error: str, attempt: int | None = None,
                             max_attempts: int | None = None) -> dict:
    """Emit a failure card with retry context. Use when a scheduled job
    errors out mid-run (before notify_scheduler_run can fire at the end).
    """
    details = {}
    if attempt is not None and max_attempts is not None:
        details["Attempt"] = f"{attempt}/{max_attempts}"
    return notify_scheduler_run(
        name=name,
        status="fail",
        details=details,
        error=error,
    )


if __name__ == "__main__":
    _test()
