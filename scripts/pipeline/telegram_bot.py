#!/usr/bin/env python3
"""Two-way Telegram chat bot — SerieAI betting mental coach on your phone.

Uses the same AI advisor brain as the web app (same system prompt, same tools,
same coaching personality). Long-polls Telegram for messages and responds via
Claude API with full tool-use support.

Usage:
    python -m scripts.pipeline.telegram_bot

Environment variables (from .env):
    TELEGRAM_BOT_TOKEN  — Bot token from @BotFather
    TELEGRAM_CHAT_ID    — Authorized chat ID (only responds to this user)
    ANTHROPIC_API_KEY   — Claude API key
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("telegram-bot")
log.setLevel(logging.INFO)

# File handler
_fh = logging.FileHandler(LOG_DIR / "telegram-bot.log")
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
log.addHandler(_fh)

# Console handler (for interactive use)
_ch = logging.StreamHandler()
_ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
log.addHandler(_ch)

# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _load_env() -> dict[str, str]:
    """Load all env vars from .env file."""
    env = {}
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    env[k.strip()] = v.strip()
    return env


def _get_env(name: str) -> str:
    """Get env var from os.environ or .env file."""
    val = os.environ.get(name)
    if val:
        return val
    return _load_env().get(name, "")


# ---------------------------------------------------------------------------
# Telegram API helpers (pure urllib, no external deps)
# ---------------------------------------------------------------------------

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
TELEGRAM_MAX_MESSAGE_LENGTH = 4096


def _tg_request(token: str, method: str, params: dict | None = None,
                timeout: int = 35) -> dict | None:
    """Make a Telegram Bot API request. Returns parsed JSON or None on error."""
    url = TELEGRAM_API.format(token=token, method=method)
    payload = json.dumps(params or {}).encode("utf-8")

    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("ok"):
                return data.get("result")
            log.warning("Telegram API error: %s", data.get("description"))
            return None
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            pass
        log.warning("Telegram HTTP %d: %s", e.code, body)
        return None
    except urllib.error.URLError as e:
        log.warning("Telegram connection error: %s", e.reason)
        return None
    except Exception as e:
        log.warning("Telegram request failed: %s", e)
        return None


def _tg_send_message(token: str, chat_id: str, text: str,
                     parse_mode: str = "Markdown") -> bool:
    """Send a message to Telegram, splitting if over 4096 chars."""
    chunks = _split_message(text)
    for chunk in chunks:
        result = _tg_request(token, "sendMessage", {
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        })
        if result is None:
            # Retry without parse_mode (Markdown can fail on special chars)
            result = _tg_request(token, "sendMessage", {
                "chat_id": chat_id,
                "text": chunk,
                "disable_web_page_preview": True,
            })
            if result is None:
                return False
    return True


def _tg_send_typing(token: str, chat_id: str):
    """Send 'typing' action to show the bot is working."""
    _tg_request(token, "sendChatAction", {
        "chat_id": chat_id,
        "action": "typing",
    }, timeout=5)


def _split_message(text: str) -> list[str]:
    """Split a long message into chunks respecting Telegram's 4096 char limit.

    Tries to split at paragraph boundaries, then line boundaries, then hard-cut.
    """
    if len(text) <= TELEGRAM_MAX_MESSAGE_LENGTH:
        return [text]

    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= TELEGRAM_MAX_MESSAGE_LENGTH:
            chunks.append(remaining)
            break

        # Try to find a good split point
        limit = TELEGRAM_MAX_MESSAGE_LENGTH
        split_at = None

        # Prefer splitting at double newline (paragraph)
        idx = remaining.rfind("\n\n", 0, limit)
        if idx > limit // 3:
            split_at = idx + 2
        else:
            # Try single newline
            idx = remaining.rfind("\n", 0, limit)
            if idx > limit // 3:
                split_at = idx + 1
            else:
                # Hard cut at space
                idx = remaining.rfind(" ", 0, limit)
                if idx > limit // 3:
                    split_at = idx + 1
                else:
                    split_at = limit

        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:]

    return chunks


# ---------------------------------------------------------------------------
# Conversation history (per-session, in-memory)
# ---------------------------------------------------------------------------

MAX_CONVERSATION_MESSAGES = 20  # Keep last 10 user+assistant pairs


class ConversationManager:
    """Manages conversation history for the Telegram bot."""

    def __init__(self):
        self._history: list[dict] = []

    def add_user_message(self, text: str):
        self._history.append({"role": "user", "content": text})
        self._trim()

    def add_assistant_message(self, text: str):
        self._history.append({"role": "assistant", "content": text})
        self._trim()

    def add_tool_exchange(self, assistant_content: list, tool_results: list):
        """Add a tool call + result pair to history."""
        self._history.append({"role": "assistant", "content": assistant_content})
        self._history.append({"role": "user", "content": tool_results})
        self._trim()

    def get_messages(self) -> list[dict]:
        return list(self._history)

    def clear(self):
        self._history.clear()

    def _trim(self):
        if len(self._history) > MAX_CONVERSATION_MESSAGES:
            self._history[:] = self._history[-MAX_CONVERSATION_MESSAGES:]


# ---------------------------------------------------------------------------
# Reuse advisor tools and system prompt from web app
# ---------------------------------------------------------------------------

from web.advisor import (
    TOOL_DEFINITIONS,
    TOOL_HANDLERS,
    _build_system_prompt,
    _get_bankroll,
    _load_json,
    _tool_get_value_bets,
    _tool_get_bankroll_status,
    _tool_get_live_matches,
)

# Models — use the same routing logic but simplified for Telegram
_MODEL_SONNET = "claude-sonnet-4-5-20250514"
_MODEL_HAIKU = "claude-haiku-4-5-20251001"

_SONNET_PATTERNS = {
    "analyz", "breakdown", "break down", "deep dive", "full analysis",
    "should i bet", "worth betting", "place all", "parlay", "accumulator",
    "strategy", "allocat", "bankroll review",
    "compare", "who will win", "who is better", "best player",
    "end of season", "who finishes", "predict the", "projection",
    "injury impact", "form analysis", "why is",
    "all matches", "this weekend", "best bets today",
}

_HAIKU_PATTERNS = {
    "who plays", "today's match", "what time", "kickoff",
    "score", "result", "standings", "table",
    "bankroll", "balance", "roi",
    "odds", "what are the odds",
    "place it", "place the bet", "yes", "no",
    "settle", "pending bets", "my bets",
    "live", "cancel",
    "hello", "hi", "hey", "thanks", "thank you",
}


def _select_model(message: str) -> str:
    msg_lower = message.lower().strip()
    if len(msg_lower) < 15:
        return _MODEL_HAIKU
    for pattern in _SONNET_PATTERNS:
        if pattern in msg_lower:
            return _MODEL_SONNET
    for pattern in _HAIKU_PATTERNS:
        if pattern in msg_lower:
            return _MODEL_HAIKU
    if len(msg_lower) > 80 or ("?" in msg_lower and len(msg_lower) > 40):
        return _MODEL_SONNET
    return _MODEL_HAIKU


# ---------------------------------------------------------------------------
# Claude API call (non-streaming, with tool loop)
# ---------------------------------------------------------------------------

MAX_TOOL_RESULT_CHARS = 6000
MAX_TOOL_ROUNDS = 5


def _truncate_tool_result(result_str: str, max_chars: int = MAX_TOOL_RESULT_CHARS) -> str:
    if len(result_str) <= max_chars:
        return result_str
    return result_str[:max_chars] + "\n... [truncated]"


def _call_claude(user_message: str, conversation: ConversationManager,
                 token: str, chat_id: str) -> str:
    """Call Claude with tool loop. Returns the final text response.

    Sends typing indicators during tool calls so the user knows the bot is working.
    """
    import anthropic

    api_key = _get_env("ANTHROPIC_API_KEY")
    if not api_key:
        return "ANTHROPIC_API_KEY not configured. Cannot respond."

    client = anthropic.Anthropic(api_key=api_key)
    system_prompt = _build_system_prompt()

    conversation.add_user_message(user_message)
    messages = conversation.get_messages()

    model = _select_model(user_message)
    max_tokens = 4096 if model == _MODEL_SONNET else 2048

    for round_num in range(MAX_TOOL_ROUNDS):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=[{
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }],
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )
        except anthropic.APIError as e:
            log.error("Claude API error: %s", e)
            return f"API error: {e}"
        except Exception as e:
            log.error("Claude call failed: %s", e)
            return f"Error calling Claude: {e}"

        # Extract text and tool uses from response
        full_text = ""
        tool_uses = []

        for block in response.content:
            if block.type == "text":
                full_text += block.text
            elif block.type == "tool_use":
                tool_uses.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        # If no tool use, we're done
        if response.stop_reason != "tool_use" or not tool_uses:
            conversation.add_assistant_message(full_text)
            return full_text

        # Execute tools
        assistant_content = []
        if full_text:
            assistant_content.append({"type": "text", "text": full_text})
        for tu in tool_uses:
            assistant_content.append({
                "type": "tool_use",
                "id": tu["id"],
                "name": tu["name"],
                "input": tu["input"],
            })

        tool_results = []
        for tu in tool_uses:
            log.info("Tool call: %s(%s)", tu["name"], json.dumps(tu["input"])[:100])
            # Send typing indicator while processing tools
            _tg_send_typing(token, chat_id)

            handler = TOOL_HANDLERS.get(tu["name"])
            if handler:
                try:
                    result_str = handler(tu["input"])
                    result_str = _truncate_tool_result(result_str)
                except Exception as e:
                    log.warning("Tool %s failed: %s", tu["name"], e)
                    result_str = json.dumps({"error": str(e)})
            else:
                result_str = json.dumps({"error": f"Unknown tool: {tu['name']}"})

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": result_str,
            })

        # Add to conversation and continue loop
        conversation.add_tool_exchange(assistant_content, tool_results)
        messages = conversation.get_messages()

    return full_text or "I ran out of processing rounds. Try a simpler question."


# ---------------------------------------------------------------------------
# Quick command handlers (no Claude API call needed)
# ---------------------------------------------------------------------------

def _handle_start() -> str:
    """Handle /start command — greeting message."""
    return (
        "*SerieAI Advisor* -- your betting mental coach, now on Telegram.\n\n"
        "Ask me anything about Serie A:\n"
        "- Match predictions and analysis\n"
        "- Value bets and bankroll status\n"
        "- Player stats and form\n"
        "- Live scores\n"
        "- Place and manage bets\n\n"
        "Quick commands:\n"
        "/bets -- current value bets\n"
        "/bankroll -- bankroll status\n"
        "/live -- live match scores\n\n"
        "Or just chat naturally -- I understand football."
    )


def _handle_bets() -> str:
    """Handle /bets command — quick value bets summary."""
    try:
        result = json.loads(_tool_get_value_bets({}))
        bets = result.get("bets", [])
        if not bets:
            return "No value bets in the current slip."

        lines = [f"*Value Bets* ({len(bets)} picks)\n"]
        for b in bets:
            edge = b.get("edge_pct", 0)
            lines.append(
                f"  {b.get('match', '?')}\n"
                f"  {b.get('market', '?')}: {b.get('selection', '?')} "
                f"@ {b.get('odds', '?')} (edge: {edge:.1f}%)\n"
            )

        # Add handicap bets if present
        hcap = result.get("handicap_bets", [])
        if hcap:
            lines.append(f"\n*Handicap Bets* ({len(hcap)})\n")
            for b in hcap[:5]:
                lines.append(
                    f"  {b.get('match', '?')}: {b.get('bet', '?')} "
                    f"@ {b.get('odds', '?')} ({b.get('value_pct', 0):.1f}% value)\n"
                )

        # Add O/U bets if present
        ou = result.get("over_under_bets", [])
        if ou:
            lines.append(f"\n*Over/Under Bets* ({len(ou)})\n")
            for b in ou[:5]:
                lines.append(
                    f"  {b.get('match', '?')}: {b.get('bet', '?')} "
                    f"@ {b.get('odds', '?')} ({b.get('value_pct', 0):.1f}% value)\n"
                )

        return "\n".join(lines)
    except Exception as e:
        log.warning("/bets failed: %s", e)
        return f"Failed to load bets: {e}"


def _handle_bankroll() -> str:
    """Handle /bankroll command — quick bankroll summary."""
    try:
        result = json.loads(_tool_get_bankroll_status({}))
        current = result.get("current_bankroll", "?")
        initial = result.get("initial_bankroll", 1000)
        roi = result.get("roi_pct", 0)
        peak = result.get("peak_bankroll", "?")
        streak = result.get("current_streak", 0)

        lines = [
            f"*Bankroll Status*\n",
            f"Balance: ${current:,.2f}" if isinstance(current, (int, float)) else f"Balance: {current}",
            f"ROI: {roi:+.1f}%",
            f"Peak: ${peak:,.2f}" if isinstance(peak, (int, float)) else f"Peak: {peak}",
        ]
        if streak:
            lines.append(f"Streak: {streak}")

        # Market breakdown
        mkt = result.get("market_breakdown", {})
        if mkt:
            lines.append("\n*By Market:*")
            for name, stats in mkt.items():
                wr = stats.get("win_rate", 0)
                profit = stats.get("profit", 0)
                lines.append(f"  {name}: {wr:.0f}% WR, ${profit:+.2f}")

        return "\n".join(lines)
    except Exception as e:
        log.warning("/bankroll failed: %s", e)
        return f"Failed to load bankroll: {e}"


def _handle_live() -> str:
    """Handle /live command — quick live scores."""
    try:
        result = json.loads(_tool_get_live_matches({}))
        matches = result.get("matches", [])
        if not matches:
            return result.get("status", "No live matches right now.")

        lines = [f"*Live Matches* ({result.get('date', 'today')})\n"]
        for m in matches:
            status = m.get("status", "?")
            score = m.get("score", "?")
            minute = m.get("minute", "")
            min_str = f" ({minute}')" if minute else ""
            lines.append(f"  {m.get('match', '?')}: {score} [{status}{min_str}]")

            # Show bets on this match
            bets = m.get("bets", [])
            for b in bets:
                lines.append(f"    Your bet: {b.get('selection', '?')} @ {b.get('odds', '?')} [{b.get('status', '?')}]")

        return "\n".join(lines)
    except Exception as e:
        log.warning("/live failed: %s", e)
        return f"Failed to load live matches: {e}"


# ---------------------------------------------------------------------------
# Main bot loop
# ---------------------------------------------------------------------------

_running = True


def _signal_handler(sig, frame):
    global _running
    log.info("Received signal %s, shutting down...", sig)
    _running = False


_PID_FILE = PROJECT_ROOT / "logs" / "telegram-bot.pid"


def _acquire_lock():
    """Ensure only one bot instance runs. Kill old one if needed."""
    if _PID_FILE.exists():
        try:
            old_pid = int(_PID_FILE.read_text().strip())
            os.kill(old_pid, 0)  # Check if alive
            log.info("Killing previous bot instance (PID %d)", old_pid)
            os.kill(old_pid, signal.SIGTERM)
            time.sleep(3)
            try:
                os.kill(old_pid, 0)
                os.kill(old_pid, 9)
                time.sleep(1)
            except ProcessLookupError:
                pass
        except (ProcessLookupError, ValueError):
            pass
        except PermissionError:
            log.warning("Cannot kill old bot — may get 409 conflicts")
    _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PID_FILE.write_text(str(os.getpid()))
    log.info("Acquired lock (PID %d)", os.getpid())


def _release_lock():
    try:
        if _PID_FILE.exists() and _PID_FILE.read_text().strip() == str(os.getpid()):
            _PID_FILE.unlink()
    except Exception:
        pass


def run_bot():
    """Main bot loop — long-polls Telegram for messages."""
    global _running

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    _acquire_lock()

    token = _get_env("TELEGRAM_BOT_TOKEN")
    chat_id = _get_env("TELEGRAM_CHAT_ID")
    api_key = _get_env("ANTHROPIC_API_KEY")

    if not token:
        log.error("TELEGRAM_BOT_TOKEN not set in .env")
        sys.exit(1)
    if not chat_id:
        log.error("TELEGRAM_CHAT_ID not set in .env")
        sys.exit(1)
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set in .env")
        sys.exit(1)

    log.info("SerieAI Telegram bot starting...")
    log.info("Authorized chat ID: %s", chat_id)

    # Set ANTHROPIC_API_KEY in environment for the anthropic SDK
    os.environ["ANTHROPIC_API_KEY"] = api_key

    # Conversation manager (single user, single session)
    conversation = ConversationManager()

    # Track the last update_id to avoid processing old messages
    offset = 0

    # On startup, get current update_id to skip any queued messages
    log.info("Clearing old messages...")
    result = _tg_request(token, "getUpdates", {"timeout": 0, "limit": 1, "offset": -1})
    if result and len(result) > 0:
        offset = result[-1]["update_id"] + 1
        log.info("Starting from update_id %d", offset)

    log.info("Bot is running. Listening for messages...")

    # Retry backoff for network errors
    backoff = 1
    max_backoff = 60

    while _running:
        try:
            updates = _tg_request(token, "getUpdates", {
                "offset": offset,
                "timeout": 30,
                "limit": 10,
            }, timeout=35)

            if updates is None:
                # Network error — backoff and retry
                log.warning("getUpdates failed, retrying in %ds...", backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
                continue

            # Reset backoff on success
            backoff = 1

            for update in updates:
                update_id = update.get("update_id", 0)
                offset = update_id + 1

                message = update.get("message")
                if not message:
                    continue

                # Check authorization
                msg_chat_id = str(message.get("chat", {}).get("id", ""))
                if msg_chat_id != str(chat_id):
                    log.warning("Unauthorized message from chat_id=%s, ignoring", msg_chat_id)
                    continue

                text = (message.get("text") or "").strip()
                if not text:
                    continue

                user_name = message.get("from", {}).get("first_name", "User")
                log.info("Message from %s: %s", user_name, text[:100])

                # Handle commands
                cmd = text.split()[0].lower() if text.startswith("/") else None

                if cmd == "/start":
                    response_text = _handle_start()
                elif cmd == "/bets":
                    _tg_send_typing(token, chat_id)
                    response_text = _handle_bets()
                elif cmd == "/bankroll":
                    _tg_send_typing(token, chat_id)
                    response_text = _handle_bankroll()
                elif cmd == "/live":
                    _tg_send_typing(token, chat_id)
                    response_text = _handle_live()
                elif cmd == "/clear":
                    conversation.clear()
                    response_text = "Conversation cleared. Fresh start."
                else:
                    # Full AI response via Claude
                    _tg_send_typing(token, chat_id)
                    response_text = _call_claude(text, conversation, token, chat_id)

                # Send response
                if response_text:
                    success = _tg_send_message(token, chat_id, response_text)
                    if success:
                        log.info("Response sent (%d chars)", len(response_text))
                    else:
                        log.warning("Failed to send response")

        except KeyboardInterrupt:
            break
        except Exception as e:
            log.exception("Bot loop error: %s", e)
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)

    _release_lock()
    log.info("Bot stopped.")


if __name__ == "__main__":
    run_bot()
