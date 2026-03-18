#!/usr/bin/env python3
"""Unified notification system -- macOS + Telegram.

Sends notifications via all configured channels. Telegram is optional:
if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are not set in .env, only
macOS notifications are sent.

Usage as module:
    from scripts.pipeline.notify import notify
    notify("Pipeline complete", title="SerieAI", level="success")

Usage as CLI (test mode):
    python -m scripts.pipeline.notify --test
    python -m scripts.pipeline.notify --test --message "Custom message"
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

log = logging.getLogger("notify")

# Level -> emoji mapping for Telegram
_LEVEL_EMOJI = {
    "info": "\u2139\ufe0f",       # info
    "success": "\u2705",           # check mark
    "warning": "\u26a0\ufe0f",    # warning
    "error": "\u274c",             # cross mark
    "critical": "\u274c",          # cross mark
}


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

def notify(message: str, title: str = "SerieAI", level: str = "info") -> dict:
    """Send notification via all configured channels.

    Args:
        message: Notification body text.
        title: Short title / subject line.
        level: One of "info", "success", "warning", "error", "critical".

    Returns:
        Dict with channel results, e.g. {"macos": True, "telegram": True}.
    """
    results = {}
    results["macos"] = _notify_macos(message, title)
    results["telegram"] = _notify_telegram(message, title, level)
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
