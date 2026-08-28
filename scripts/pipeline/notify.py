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
# Bankroll / edge context helpers
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

    # macOS: one decisive line
    mac_msg = f"{len(bets)} value bet{'s' if len(bets) > 1 else ''}: {match} {best.get('selection','')} @ {best.get('best_odds','')} ({_edge_label(best_edge)})"

    # Telegram — facts only, no opener (2026-08-27 tone cleanup)
    tg = TgMsg()

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
        book = b.get("bookmaker", "")
        book_tag = f" ({_html_escape(book)})" if book else ""
        tg.raw(f"  {_html_escape(b.get('selection', '?'))} ({_html_escape(b.get('market', '?'))}) "
               f"@ <b>{b.get('best_odds', '?')}</b>{book_tag}  \u2014  {_html_escape(_edge_label(edge))}")
        # "market", not "sharp": the comparison is best-book implied probability
        market_pct = b.get('sharp_implied_prob', 0) * 100 if b.get('sharp_implied_prob') else (100 / b.get('best_odds', 1))
        tg.raw(f"  Model {b.get('model_prob', 0)*100:.0f}% vs market {market_pct:.0f}%"
               f"  |  \u20ac{b.get('stake', 0):.0f} stake")

        # Match-specific factors
        factors = (b.get("home_factors") or []) + (b.get("away_factors") or [])
        if factors:
            hints = [f.replace("_", " ") for f in factors[:2]]
            tg.raw(f"  <i>{_html_escape(', '.join(hints))}</i>")
        tg.blank()

    if len(bets) > 3:
        tg.raw(f"+{len(bets) - 3} more in the slip")

    # No inline keyboard (2026-08-27): the Place button wrote duplicate junk
    # rows into the journal (stake 0, guessed market) and was retired; Skip
    # and View-All added nothing the bot commands don't already answer.
    result = notify(
        message=mac_msg,
        title=f"Value: {match}",
        level="info",
        category="betting",
        tg_html=tg.build(),
    )

    # Save dedup marker
    try:
        with open(_vb_dedup_path, "w") as _f:
            json.dump({"date": today_str, "sig": bet_sig}, _f)
    except Exception:
        pass

    return result


def notify_order_ticket(bets: list[dict]) -> dict:
    """T-30 ORDER TICKET — sent from run_pre_kickoff at the moment bets are
    journaled. This is the message money is placed from: every bet in the
    window, with book, stake, odds at commit, the minimum acceptable price
    (below it the edge is gone — do not place), payout, and lineup status.

    Bets are journal rows enriched by the caller with:
      _kickoff (ISO str), _xi_confirmed (bool), _floor_odds (float|None),
      _resend (bool — this match was ticketed earlier; numbers superseded),
      _num (int — day-unique ticket line number; keys the ✓/✗ confirm buttons
      and /fill, resolved to a bet_id via the T-30 marker file).
    """
    if not bets:
        return {}
    total_stake = sum(float(b.get("stake") or 0) for b in bets)
    n = len(bets)

    tg = TgMsg()
    tg.raw(f"\U0001f4b0 <b>Order Ticket</b> — {n} bet{'s' if n != 1 else ''}, "
           f"€{total_stake:.0f} total")
    if any(b.get("_resend") for b in bets):
        tg.raw("⚠️ <i>Replaces an earlier ticket — use these numbers.</i>")

    by_match: dict = {}
    for b in bets:
        by_match.setdefault(b.get("match", "?"), []).append(b)

    for mk, rows in by_match.items():
        tg.blank()
        ko = rows[0].get("_kickoff", "") or ""
        ko_str = _kickoff_display(ko) if ko else ""
        mins = ""
        try:
            if ko:
                dt = datetime.fromisoformat(str(ko).replace("Z", "+00:00"))
                delta = int((dt - datetime.now(dt.tzinfo)).total_seconds() // 60)
                if 0 < delta < 240:
                    mins = f" (in {delta} min)"
        except (ValueError, TypeError):
            pass
        xi = "  ·  XI ✓" if rows[0].get("_xi_confirmed") else ""
        head = f"<b>{_html_escape(mk)}</b>"
        if ko_str:
            head += f" — {_html_escape(ko_str)}{_html_escape(mins)}"
        tg.raw(head + xi)
        for b in rows:
            conf = f"  [{_html_escape(str(b.get('confidence')))}]" if b.get("confidence") else ""
            book = b.get("bookmaker") or "best book"
            odds = float(b.get("odds") or 0)
            stake = float(b.get("stake") or 0)
            payout = stake * odds if odds else 0
            edge = b.get("edge_pct")
            floor = b.get("_floor_odds")
            num = b.get("_num")
            num_tag = f"{num}\u00b7 " if num else ""
            tg.raw(f"  {num_tag}{_html_escape(str(b.get('selection', '?')))} "
                   f"({_html_escape(str(b.get('market', '?')))}){conf}")
            tg.raw(f"  €{stake:.0f} @ <b>{odds:.2f}</b> — {_html_escape(str(book))}")
            bits = []
            if floor:
                bits.append(f"min price <b>{floor:.2f}</b>")
            if payout:
                bits.append(f"payout €{payout:.2f}")
            if edge is not None:
                try:
                    bits.append(f"edge {float(edge):+.1f}%")
                except (TypeError, ValueError):
                    pass
            if bits:
                tg.raw("  " + "  ·  ".join(bits))

    tg.blank()
    tg.raw("<i>Below min price: don't place — the edge is gone.</i>")

    # Confirm buttons: one row per numbered bet, writing to the EXISTING
    # journal row (fill tier) — never creating one. Different price: /fill.
    keyboard = None
    numbered = [b for b in bets if b.get("_num")]
    if numbered:
        rows = []
        for b in numbered:
            num = b["_num"]
            odds = float(b.get("odds") or 0)
            rows.append([
                {"text": f"\u2713 {num}\u00b7 Placed @ {odds:.2f}",
                 "callback_data": f"fill:{num}"},
                {"text": f"\u2717 {num}\u00b7 Missed",
                 "callback_data": f"miss:{num}"},
            ])
        keyboard = {"inline_keyboard": rows}
        tg.raw("<i>Got a different price? /fill &lt;n&gt; &lt;odds&gt;</i>")

    first = bets[0]
    mac = (f"ORDER: {n} bet{'s' if n != 1 else ''} €{total_stake:.0f} — "
           f"{first.get('match', '?')}"
           + (f" +{len(by_match) - 1}" if len(by_match) > 1 else ""))
    return notify(
        mac[:250],
        title="Order Ticket",
        level="info",
        category="live",  # bypasses quiet hours — this is the money message
        priority=PRIORITY_URGENT,
        tg_html=tg.build(),
        tg_reply_markup=keyboard,
    )


def notify_no_action(matches: list[str]) -> dict:
    """T-30 ran for these imminent matches and journaled nothing — say so.

    Silence is indistinguishable from a dead chain; this one line is the
    difference between trust and doubt at 20:15 on a match night.
    """
    if not matches:
        return {}
    listed = ", ".join(matches[:4]) + (f" +{len(matches) - 4}" if len(matches) > 4 else "")
    msg = f"T-30 ran for {listed}: no edge cleared the bar. No bets."
    return notify(msg, title="T-30: no bets", level="info", category="betting",
                  priority=PRIORITY_NORMAL)


def notify_fill_nudge(match: str, count: int, minutes: int = 10) -> dict:
    """T-10: ticket lines with no \u2713/\u2717 answer, kickoff imminent.

    One line, urgent. At kickoff the sweep flags unanswered lines
    "unverified" and they drop out of verified ROI/CLV.
    """
    if count <= 0:
        return {}
    msg = (f"\u23f1 {count} bet{'s' if count != 1 else ''} unconfirmed \u2014 "
           f"{match} kicks off in ~{max(int(minutes), 0)} min. "
           f"Tap \u2713/\u2717 on the ticket or /fill <n> <odds>.")
    tg = TgMsg()
    tg.raw(f"\u23f1 <b>{count} bet{'s' if count != 1 else ''} unconfirmed</b> \u2014 "
           f"{_html_escape(match)} kicks off in ~{max(int(minutes), 0)} min.")
    tg.raw("Tap \u2713/\u2717 on the ticket or <code>/fill &lt;n&gt; &lt;odds&gt;</code>. "
           "Unanswered lines go <i>unverified</i> at kickoff.")
    return notify(msg[:250], title="Unconfirmed fills", level="warning",
                  category="live", priority=PRIORITY_URGENT, tg_html=tg.build())


def notify_day_wrap(settled_bets: list, balance: float = 0) -> dict:
    """RECONCILIATION: one card when the day's last bet settles.

    Record, P&L, balance, drawdown, per-bet fill flags. Replaces the
    per-poll settlement cards on multi-match days (FT cards already carried
    each match's P&L in real time).
    """
    if not settled_bets:
        return {}
    won = sum(1 for b in settled_bets if b.get("status") == "won")
    lost = sum(1 for b in settled_bets if b.get("status") == "lost")
    push = sum(1 for b in settled_bets if b.get("status") in ("push", "void", "voided"))
    profit = sum(float(b.get("profit") or 0) for b in settled_bets)
    record = f"{won}W-{lost}L" + (f"-{push}P" if push else "")
    sign = "+" if profit >= 0 else ""

    br_ctx = _get_bankroll_context()
    if balance > 0:
        br_ctx["current"] = balance
        if br_ctx.get("initial"):
            br_ctx["roi_pct"] = round(
                (balance - br_ctx["initial"]) / br_ctx["initial"] * 100, 1)

    _fill_icon = {"placed": "\u2713", "missed": "\u2717", "unverified": "\u26a0"}
    tg = TgMsg()
    tg.raw(f"\U0001f3c1 <b>Day Wrap</b> \u2014 {record}")
    tg.blank()
    for b in settled_bets:
        icon = "\u2705" if b.get("status") == "won" else (
            "\u274c" if b.get("status") == "lost" else "\u21a9\ufe0f")
        fill = b.get("fill_status")
        fill_tag = f"  {_fill_icon[fill]} {fill}" if fill in _fill_icon else ""
        pl = float(b.get("profit") or 0)
        tg.raw(f"  {icon} {_html_escape(str(b.get('match', '?')))} \u2014 "
               f"{_html_escape(str(b.get('selection', '')))} "
               f"@{float(b.get('odds') or 0):.2f} \u2192 "
               f"{'+' if pl >= 0 else ''}\u20ac{pl:.2f}{fill_tag}")
    tg.blank()
    tg.pnl(profit, label="Day P&L")
    tg.raw(f"Balance: {_html_escape(_bankroll_in_context(br_ctx))}")
    unverified = sum(1 for b in settled_bets if b.get("fill_status") == "unverified")
    if unverified:
        tg.raw(f"\u26a0 {unverified} bet{'s' if unverified != 1 else ''} unverified \u2014 "
               "settled in the journal but excluded from verified ROI.")

    mac = (f"WRAP {record} | {sign}\u20ac{profit:.2f} | "
           f"Balance: {_bankroll_in_context(br_ctx)}")
    return notify(mac[:250], title=f"Day Wrap: {record}",
                  level="success" if profit >= 0 else "warning",
                  category="betting", priority=PRIORITY_NORMAL, tg_html=tg.build())


def notify_proof_of_edge(days: int = 7) -> dict:
    """Sunday 22:00: the one message that answers "is this real".

    Average CLV with sample size (overall and per market), verified-fill
    rate, and ROI on verified fills vs the whole journal \u2014 over the last
    `days` days of settled bets. CLV is the only edge signal that does not
    need thousands of bets; everything else here is honesty accounting.
    Sends nothing on a week with no settled bets.
    """
    try:
        from scripts.betting.bet_journal import _load_journal
        from datetime import timedelta as _td
        cutoff = (datetime.now() - _td(days=days - 1)).strftime("%Y-%m-%d")
        rows = [b for b in _load_journal()["bets"].values()
                if b.get("status") in ("won", "lost", "push", "void", "voided")
                and (b.get("date") or "") >= cutoff]
    except Exception as e:
        log.warning("Proof-of-edge: journal read failed: %s", e)
        return {}
    if not rows:
        return {}

    decisive = [b for b in rows if b.get("status") in ("won", "lost")]
    clvs = [float(b["clv_pct"]) for b in rows if b.get("clv_pct") is not None]
    avg_clv = sum(clvs) / len(clvs) if clvs else None

    by_market: dict = {}
    for b in rows:
        if b.get("clv_pct") is None:
            continue
        m = str(b.get("market") or "?")
        by_market.setdefault(m, []).append(float(b["clv_pct"]))

    staked = sum(float(b.get("stake") or 0) for b in rows)
    profit = sum(float(b.get("profit") or 0) for b in rows)
    roi_journal = (profit / staked * 100) if staked > 0 else 0.0

    # Verified tier: explicit fills only, P&L recomputed at the FILLED price.
    with_fill = [b for b in rows if b.get("fill_status")]
    placed = [b for b in with_fill if b.get("fill_status") == "placed"]
    fill_rate = (len(placed) / len(with_fill) * 100) if with_fill else None
    v_staked = v_profit = 0.0
    for b in placed:
        stake = float(b.get("stake") or 0)
        odds = float(b.get("filled_odds") or b.get("odds") or 0)
        v_staked += stake
        if b.get("status") == "won":
            v_profit += stake * (odds - 1)
        elif b.get("status") == "lost":
            v_profit -= stake
    roi_verified = (v_profit / v_staked * 100) if v_staked > 0 else None

    tg = TgMsg()
    tg.raw(f"\U0001f4d0 <b>Proof of Edge</b> \u2014 last {days} days, "
           f"{len(rows)} settled ({len(decisive)} decisive)")
    tg.blank()
    if avg_clv is not None:
        tg.raw(f"CLV: <b>{avg_clv:+.2f}%</b> avg on {len(clvs)} bets "
               f"{'\u2014 beating the close' if avg_clv > 0 else '\u2014 behind the close'}")
        for m, vals in sorted(by_market.items()):
            tg.raw(f"  {_html_escape(m)}: {sum(vals) / len(vals):+.2f}% (n={len(vals)})")
    else:
        tg.raw("CLV: no closing lines captured this week.")
    tg.blank()
    tg.raw(f"ROI (journal): <b>{roi_journal:+.1f}%</b> on \u20ac{staked:.0f} staked")
    if roi_verified is not None:
        tg.raw(f"ROI (verified fills): <b>{roi_verified:+.1f}%</b> "
               f"on \u20ac{v_staked:.0f}")
    if fill_rate is not None:
        tg.raw(f"Fill rate: {fill_rate:.0f}% confirmed placed "
               f"({len(placed)}/{len(with_fill)})")
    elif rows:
        tg.raw("<i>No fill confirmations this week \u2014 verified ROI unavailable.</i>")

    clv_s = f"CLV {avg_clv:+.1f}%" if avg_clv is not None else "CLV n/a"
    mac = f"EDGE: {clv_s} (n={len(clvs)}) | ROI {roi_journal:+.1f}% | {len(rows)} bets"
    return notify(mac[:250], title="Proof of Edge", level="info",
                  category="betting", priority=PRIORITY_NORMAL, tg_html=tg.build())


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
    br_ctx = _get_bankroll_context()
    # Use passed balance if available (from settlement engine) instead of stale state
    if balance > 0:
        br_ctx["current"] = balance
        br_ctx["roi_pct"] = round((balance - br_ctx["initial"]) / br_ctx["initial"] * 100, 1) if br_ctx["initial"] else 0
    level = "success" if is_positive else "warning"

    record = f"{won}W-{lost}L" + (f"-{push}P" if push else "")
    sign = "+" if profit >= 0 else ""

    # macOS: the one number that matters
    mac_msg = f"{record} | {sign}\u20ac{profit:.2f} | Balance: {_bankroll_in_context(br_ctx)}"

    # Telegram — rich per-bet summary
    tg = TgMsg()

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
    # _bankroll_in_context already includes the \u20ac figure \u2014 don't wrap it again
    # (rendered as "\u20ac1,052 (\u20ac1,052 (\u21915.2% ROI\u2026))" before 2026-08-27)
    tg.raw(f"Balance: {_html_escape(_bankroll_in_context(br_ctx))}")

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
            opener = "Full time \u2014 bets won."
            level = "success"
        elif any_lost and not any_won:
            opener = "Full time \u2014 bets lost."
            level = "warning"
        else:
            opener = "Full time \u2014 mixed." if any_won else "Full time."
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
            opener = "Full time \u2014 bet won."
            msg = f"{opener} {match_key} {home_score}-{away_score}"
            if bet_profit and bet_profit > 0:
                msg += f" | +\u20ac{bet_profit:.2f}"
            level = "success"
        else:
            opener = "Full time \u2014 bet lost."
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




def _chain_armed_check() -> list:
    """Match-day dead-man's switch: is the T-30 chain actually armed?

    Checks the two launchd jobs that carry the money path and the odds key
    (via the age of the last SUCCESSFUL paid fetch \u2014 a stored credit
    number proves nothing, a completed fetch does). Returns (state, label)
    tuples, state in {"ok", "fail", "warn"}. Never raises; unknown = warn.
    """
    checks = []
    try:
        out = subprocess.run(["launchctl", "list"], capture_output=True,
                             text=True, timeout=10).stdout
        for label, job in (("T-30 monitor", "com.seriea-pipeline.pre-kickoff-monitor"),
                           ("settlement", "com.seriea-pipeline.settlement"),
                           ("bot", "com.seriea-pipeline.telegram-bot")):
            if job in out:
                checks.append(("ok", f"{label} loaded"))
            else:
                checks.append(("fail", f"{label} NOT loaded"))
    except Exception as e:
        log.debug("Arm check: launchctl unavailable: %s", e)
        checks.append(("warn", "launchd state unknown"))
    try:
        state = json.loads((DATA_DIR / "pipeline_state.json").read_text())
        raw = state.get("last_odds_fetch") or ""
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
        if age_h < 26:
            checks.append(("ok", f"odds key (fetched {age_h:.0f}h ago)"))
        else:
            checks.append(("fail", f"odds: no successful fetch in {age_h:.0f}h"))
    except Exception:
        checks.append(("warn", "odds fetch age unknown"))
    return checks


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

    # --- Match-day battle plan + chain-armed check ---
    # On a match day the digest leads with the plan: which matches can
    # produce bets, when the tickets arrive, and whether the T-30 chain is
    # actually armed. Candidates are a COUNT only \u2014 selections with odds
    # appear exclusively on the order ticket (the \u22125% early-bet path).
    if today_matches:
        try:
            from datetime import timedelta as _td30
            tg.raw(f"\n\u2694\ufe0f <b>Plan:</b> {matches_today_count} "
                   f"match{'es' if matches_today_count != 1 else ''} today")
            _kmap = _load_kickoff_map()
            for m in today_matches[:6]:
                ko_iso = (m.get("kickoff_time")
                          or _kmap.get((m.get("home_team", ""), m.get("away_team", "")), ""))
                line = f"  {_html_escape(m.get('home_team', '?'))} vs {_html_escape(m.get('away_team', '?'))}"
                if ko_iso:
                    try:
                        _dt = datetime.fromisoformat(str(ko_iso).replace("Z", "+00:00"))
                        line += (f" \u2014 KO {_html_escape(_kickoff_display(ko_iso))}"
                                 f" \u00b7 ticket ~{_html_escape(_kickoff_display((_dt - _td30(minutes=30)).isoformat()))}")
                    except (ValueError, TypeError):
                        pass
                tg.raw(line)
            # Candidates live in betting_candidates.json (written by the
            # morning candidate-only Step 24) -- NOT the legacy slip files,
            # which have been dead since June. Count only, never selections.
            cand = _load_json(DATA_DIR / "upcoming" / "betting_candidates.json")
            cand_line = "  Candidates queued: unknown"
            if isinstance(cand, dict) and cand.get("generated_at"):
                try:
                    gen = datetime.fromisoformat(str(cand["generated_at"]))
                    age_h = (datetime.now() - gen).total_seconds() / 3600
                    if age_h < 28:
                        n_cand = len(cand.get("candidates") or [])
                        cand_line = (f"  Candidates queued: {n_cand} "
                                     f"<i>(T-30 re-prices and commits)</i>")
                    else:
                        cand_line = (f"  Candidates: stale ({age_h:.0f}h) "
                                     f"\u2014 morning run may have failed")
                except (ValueError, TypeError):
                    pass
            tg.raw(cand_line)
            _icons = {"ok": "\u2705", "fail": "\u274c", "warn": "\u26a0\ufe0f"}
            _checks = _chain_armed_check()
            tg.raw("  Chain: " + " \u00b7 ".join(
                f"{_icons[s]} {_html_escape(lbl)}" for s, lbl in _checks))
            if any(s == "fail" for s, _ in _checks):
                level = "error"
                mac_msg = "\u26a0 CHAIN NOT ARMED \u2014 " + mac_msg
        except Exception as e:
            log.debug("Digest battle plan failed: %s", e)
        tg.blank()
        tg.mini_sep()

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
                health_issues = health.get("issues", [])
                n_issues = len(health_issues)
                suffix = f" ({n_issues} issue{'s' if n_issues != 1 else ''})" if n_issues else ""
                tg.raw(f"  {icon} <b>Health:</b> {_html_escape(overall)}{suffix}")
                # List the actual CRITICAL/WARNING messages, not just the count —
                # a bare "(6 issues)" let a standing CRITICAL (negative ROI) go
                # unread for weeks. The 30-min change-alerter silences anything
                # >24h old by design, so this daily digest is the one place a
                # PERSISTENT issue must surface in full. CRITICAL first.
                _lvl_rank = {"CRITICAL": 0, "WARNING": 1}
                _lvl_icon = {"CRITICAL": "🚨", "WARNING": "⚠️"}
                ranked = sorted(
                    (i for i in health_issues
                     if isinstance(i, (list, tuple)) and len(i) == 2
                     and i[0] in ("CRITICAL", "WARNING")),
                    key=lambda i: _lvl_rank.get(i[0], 9),
                )
                for lvl, msg in ranked[:8]:  # cap to keep the digest readable
                    tg.raw(f"    {_lvl_icon.get(lvl, 'ℹ️')} {_html_escape(str(msg))}")
                if len(ranked) > 8:
                    tg.raw(f"    <i>…and {len(ranked) - 8} more</i>")
    except Exception as e:
        log.debug("Digest systems section failed: %s", e)

    return notify(
        message=mac_msg,
        title="Daily Digest",
        level=level,
        category="betting",
        tg_html=tg.build(),
    )


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

def notify_loss_streak(streak_count: int, total_loss: float = 0,
                       recent_bets: list = None) -> dict:
    """Alert on a losing streak: the facts plus a CLV verdict.

    No canned coaching. The one computed judgment is the CLV read on the
    streak's own bets: if we still beat the close, the process held and
    the streak is variance; if we didn't, stop and review.
    """
    if streak_count < 3:
        return {}

    # Callers pass abs(); tolerate a signed loss so the amount never
    # silently vanishes behind the `> 0` display gate.
    total_loss = abs(total_loss)
    recent_bets = recent_bets or []

    clvs = [b.get("clv_pct") for b in recent_bets if b.get("clv_pct") is not None]
    if clvs:
        avg_clv = sum(clvs) / len(clvs)
        if avg_clv >= 0:
            verdict = (f"CLV intact on these bets (avg {avg_clv:+.1f}%) \u2014 "
                       "closing prices were beaten. Variance, not model failure.")
        else:
            verdict = (f"CLV negative on these bets (avg {avg_clv:+.1f}%) \u2014 "
                       "the market moved against these positions after placement. "
                       "Stop and review before the next card.")
    else:
        avg_clv = None
        verdict = "No CLV data on these bets \u2014 no verdict."

    severe = streak_count >= 5
    mac_clv = f"CLV {avg_clv:+.1f}%" if avg_clv is not None else "CLV n/a"
    mac_msg = f"{streak_count}-loss streak | \u20ac{total_loss:.0f} lost | {mac_clv}"

    tg = TgMsg()
    tg.title(f"{streak_count}-Loss Streak", emoji="\u26a0\ufe0f")
    tg.blank()
    if total_loss > 0:
        tg.raw(f"Total lost: <b>-\u20ac{total_loss:.2f}</b>")
    if recent_bets:
        tg.blank()
        for b in recent_bets[-5:]:
            clv = b.get("clv_pct")
            clv_s = f"  \u00b7 CLV {clv:+.1f}%" if clv is not None else ""
            tg.raw(f"  \u274c {_html_escape(b.get('match', ''))} \u2014 "
                   f"{_html_escape(b.get('selection', ''))} @{b.get('odds', 0):.2f}"
                   f"{clv_s}")
    tg.blank()
    tg.raw(f"<b>{verdict}</b>")

    return notify(
        message=mac_msg,
        title=f"Loss Streak: {streak_count} in a row",
        level="error" if severe else "warning",
        category="alert",
        tg_html=tg.build(),
        priority=PRIORITY_URGENT if severe else PRIORITY_NORMAL,
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
                    # Latest season with results, not the calendar season — see
                    # config.settings.latest_season_with_results.
                    from config.settings import latest_season_with_results
                    _season = latest_season_with_results(mdf)
                    current = mdf[mdf["season"] == _season] if _season else mdf.iloc[0:0]
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

    # Routine success is log material, not a message. State is persisted for
    # the digest's Systems block and launchd keeps the run log; only warn and
    # fail still send (2026-08-27 cleanup — these one-line "✅ done" cards
    # were ~2/day of pure chat clutter).
    if is_success and not error:
        log.info("scheduler card: %s %s — routine success, not sending", name, status_l)
        return {}

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
        `_TRANSIENT_WINDOW_MIN` minutes (default 120 = 4 monitor cycles), it
        is treated as noise and no alert fires — and a resolve-then-reappear
        inside the same window is a flap, also silent. Catches Gemini 503s,
        temporary timeouts, hourly-flapping checks.

      - Standing issues never re-alert: identity is key-stable, so an issue
        already announced stays silent until it resolves. (An earlier
        `_QUIET_AFTER_HOURS` constant promised this in prose but was never
        used — with stable keys the property is inherent.)

    Silent when nothing meaningful changed.
    """
    import re as _re
    prev_state = _load_json_safe(_HEALTH_STATE_PATH, {})
    is_first_run = not prev_state

    _TRANSIENT_WINDOW_MIN = 120
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
        # Catch-all (2026-08-27): collapse EVERY remaining number. Changing
        # counts the rules above missed (e.g. "missing 18/32") minted a fresh
        # identity each cycle and re-alerted every 30 min — 77 health sends
        # in 18 days, most of them the same issue with a moving number.
        s = _re.sub(r"\d+(?:[./]\d+)*", "N", s)
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

    # Prune old resolved-tracking entries (>4h old — must outlive the 120-min
    # transient window or flap detection forgets too early)
    trimmed = {}
    for k, iso in resolved_tracking.items():
        try:
            if (_NOW - datetime.fromisoformat(iso)).total_seconds() / 3600 < 4:
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
    # Name the first new issue — "1 new" alone is unactionable on a lock screen
    if truly_new:
        macos_msg += f": {truly_new[0][1]}"

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
