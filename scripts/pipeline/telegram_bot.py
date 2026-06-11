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
import threading
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
                     parse_mode: str = "HTML",
                     reply_markup: dict | None = None) -> bool:
    """Send a message to Telegram, splitting if over 4096 chars.

    Uses HTML parse_mode by default for consistent formatting with
    the notification system. Falls back to plain text if HTML fails.
    """
    chunks = _split_message(text)
    for i, chunk in enumerate(chunks):
        params = {
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }
        # Only attach reply_markup to the last chunk
        if reply_markup and i == len(chunks) - 1:
            params["reply_markup"] = reply_markup
        result = _tg_request(token, "sendMessage", params)
        if result is None:
            # Retry without parse_mode (formatting can fail on special chars)
            params.pop("parse_mode", None)
            result = _tg_request(token, "sendMessage", params)
            if result is None:
                return False
    return True






def _tg_send_typing(token: str, chat_id: str):
    """Send 'typing' action to show the bot is working."""
    _tg_request(token, "sendChatAction", {
        "chat_id": chat_id,
        "action": "typing",
    }, timeout=5)


class _TypingKeepAlive:
    """Refreshes the typing indicator every 4 seconds while Claude is thinking.

    Telegram's typing indicator expires after 5 seconds. For API calls
    that take 10-30s, the user sees typing stop and thinks the bot crashed.
    """

    def __init__(self, token: str, chat_id: str):
        self._token = token
        self._chat_id = chat_id
        self._active = False
        self._thread: threading.Thread | None = None

    def start(self):
        self._active = True
        _tg_send_typing(self._token, self._chat_id)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._active = False
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self):
        while self._active:
            time.sleep(4)
            if self._active:
                _tg_send_typing(self._token, self._chat_id)


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
_CONVERSATION_FILE = PROJECT_ROOT / "data" / ".telegram_conversation.json"


_LEAGUE_PREFS_FILE = PROJECT_ROOT / "data" / ".telegram_league_pref.json"


class ConversationManager:
    """Manages conversation history with disk persistence.

    History survives bot restarts so the user doesn't lose context.
    Persists after each message to data/.telegram_conversation.json.
    """

    def __init__(self):
        self._history: list[dict] = []
        self._league_filter: str | None = None  # e.g. "premier_league" or None for all
        self._load()
        self._load_league_pref()

    @property
    def league_filter(self) -> str | None:
        """Current league filter preference (None = all leagues)."""
        return self._league_filter

    def set_league_filter(self, league_key: str | None):
        """Set league filter preference. None means all leagues."""
        self._league_filter = league_key
        self._save_league_pref()

    def add_user_message(self, text: str):
        self._history.append({"role": "user", "content": text})
        self._trim()
        self._save()

    def add_assistant_message(self, text: str):
        self._history.append({"role": "assistant", "content": text})
        self._trim()
        self._save()

    def add_tool_exchange(self, assistant_content: list, tool_results: list):
        """Add a tool call + result pair to history."""
        self._history.append({"role": "assistant", "content": assistant_content})
        self._history.append({"role": "user", "content": tool_results})
        self._trim()
        self._save()

    def get_messages(self) -> list[dict]:
        return list(self._history)

    def clear(self):
        self._history.clear()
        # Preserve league_filter across clear
        self._save()

    def _trim(self):
        if len(self._history) > MAX_CONVERSATION_MESSAGES:
            self._history[:] = self._history[-MAX_CONVERSATION_MESSAGES:]

    def _save(self):
        try:
            _CONVERSATION_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(_CONVERSATION_FILE, "w") as f:
                json.dump(self._history, f)
        except Exception as e:
            log.debug("Failed to save conversation: %s", e)

    def _save_league_pref(self):
        try:
            _LEAGUE_PREFS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(_LEAGUE_PREFS_FILE, "w") as f:
                json.dump({"league_filter": self._league_filter}, f)
        except Exception as e:
            log.debug("Failed to save league pref: %s", e)

    def _load_league_pref(self):
        try:
            if _LEAGUE_PREFS_FILE.exists():
                with open(_LEAGUE_PREFS_FILE) as f:
                    data = json.load(f)
                self._league_filter = data.get("league_filter")
        except Exception as e:
            log.debug("Failed to load league pref: %s", e)

    def _load(self):
        try:
            if _CONVERSATION_FILE.exists():
                with open(_CONVERSATION_FILE) as f:
                    self._history = json.load(f)
                # Only keep text messages on reload (tool exchanges don't deserialize well)
                self._history = [
                    m for m in self._history
                    if isinstance(m.get("content"), str)
                ]
                self._trim()
        except Exception as e:
            log.debug("Failed to load conversation: %s", e)


# ---------------------------------------------------------------------------
# Markdown → Telegram HTML converter
# ---------------------------------------------------------------------------

import re as _re


def _md_to_html(text: str) -> str:
    """Convert Claude's output to Telegram-safe HTML.

    Claude may output either:
    1. HTML (because we told it to) — pass through, only escape bare &
    2. Markdown (its natural format) — convert to HTML

    Detection: if text contains <b>, <i>, <code>, or <pre> tags, treat as HTML.
    """
    # Detect if Claude already used HTML tags
    has_html = bool(_re.search(r'<(b|i|code|pre|u|s|a\s)[ >/]', text))

    if has_html:
        # Claude used HTML — only escape bare ampersands (not already part of entities)
        text = _re.sub(r'&(?!amp;|lt;|gt;|quot;|#)', '&amp;', text)
        # Convert any remaining Markdown that Claude might have mixed in
        text = _re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        text = _re.sub(r'(?<!\w)\*(.+?)\*(?!\w)', r'<i>\1</i>', text)
        text = _re.sub(r'^[-\u2022]\s+', '\u2022 ', text, flags=_re.MULTILINE)
        return text

    # Pure Markdown — full conversion
    # Escape HTML entities first
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Code blocks (``` ... ```)  — must be before inline backtick
    text = _re.sub(r'```(\w*)\n(.*?)```', r'<pre>\2</pre>', text, flags=_re.DOTALL)

    # Inline code (`...`)
    text = _re.sub(r'`([^`]+)`', r'<code>\1</code>', text)

    # Bold (**...**) — must be before italic
    text = _re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)

    # Italic (*...*)
    text = _re.sub(r'(?<!\w)\*(.+?)\*(?!\w)', r'<i>\1</i>', text)

    # Headers (### Header) → bold line
    text = _re.sub(r'^#{1,3}\s+(.+)$', r'<b>\1</b>', text, flags=_re.MULTILINE)

    # Bullet points (- item) → bullet character
    text = _re.sub(r'^[-\u2022]\s+', '\u2022 ', text, flags=_re.MULTILINE)

    return text


# ---------------------------------------------------------------------------
# Inline keyboard helpers
# ---------------------------------------------------------------------------

def _inline_keyboard(buttons: list[list[dict]]) -> dict:
    """Build Telegram inline keyboard markup.

    buttons: [[{"text": "Label", "callback_data": "action"}], ...]
    Each inner list is a row of buttons.
    """
    return {"inline_keyboard": buttons}


def _button(text: str, callback_data: str) -> dict:
    return {"text": text, "callback_data": callback_data}


# ---------------------------------------------------------------------------
# Persistent reply keyboard (bottom buttons)
# ---------------------------------------------------------------------------

# Map button labels → slash commands
_REPLY_BUTTON_MAP: dict[str, str] = {
    "💰 Bets": "/bets",
    "📊 Bankroll": "/bankroll",
    "⚽ Today": "/today",
    "🔴 Live": "/live",
    "🎯 Match": "/match",
    "🏆 Parlays": "/parlays",
    "📋 Summary": "/summary",
    "📰 Digest": "/digest",
    "🌍 World Cup": "/wc",
}


def _reply_keyboard() -> dict:
    """Build a persistent reply keyboard shown at the bottom of the chat."""
    return {
        "keyboard": [
            [{"text": "💰 Bets"}, {"text": "📊 Bankroll"}, {"text": "⚽ Today"}],
            [{"text": "🔴 Live"}, {"text": "🎯 Match"}, {"text": "🏆 Parlays"}],
            [{"text": "📋 Summary"}, {"text": "📰 Digest"}, {"text": "🌍 World Cup"}],
        ],
        "resize_keyboard": True,
        "is_persistent": True,
    }


def _register_commands(token: str):
    """Clear the slash command menu — commands are handled via inline buttons instead."""
    # Send empty list to remove the ☰ menu commands
    result = _tg_request(token, "setMyCommands", {"commands": []}, timeout=10)
    if result is not None:
        log.info("Cleared bot command menu (using inline buttons instead)")
    else:
        log.warning("Failed to clear bot command menu")


def _edit_message(token: str, chat_id: str, message_id: int, text: str,
                  reply_markup: dict = None) -> bool:
    """Edit an existing message (keeps chat clean after button presses)."""
    params = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": text,
        "parse_mode": "HTML",
    }
    if reply_markup:
        params["reply_markup"] = reply_markup
    result = _tg_request(token, "editMessageText", params, timeout=10)
    return result is not None


# ---------------------------------------------------------------------------
# Reuse advisor tools and system prompt from web app
# ---------------------------------------------------------------------------

from web.advisor import (
    TOOL_DEFINITIONS,
    TOOL_HANDLERS,
    _build_system_prompt,
    _get_bankroll,
    _tool_get_value_bets,
    _tool_get_bankroll_status,
    _tool_get_live_matches,
)
# advisor.py uses load_json_safe; expose it under the legacy _load_json name
# that telegram_bot was originally written against.
from scripts.utils.json_utils import load_json_safe as _load_json


# Telegram-specific system prompt extension
_TELEGRAM_SYSTEM_ADDON = """

## TELEGRAM-SPECIFIC RULES
You are responding on Telegram (mobile phone). Adjust accordingly:
- Keep responses SHORT. The user is on a small screen. Max 15-20 lines.
- NO tables — they don't render on Telegram. Use compact key: value lines.
- NO long disclaimers. The user is experienced. Get to the point.
- Use line breaks for readability. Dense paragraphs are hard on mobile.
- Reference slash commands when relevant: "type /parlays to see today's picks"
- The user already receives notifications about goals, settlements, value bets.
  Don't repeat what they've already seen — add the INSIGHT they didn't get.
- When recommending a bet, end with a clear verdict in one line.
- Format with Telegram HTML: <b>bold</b>, <i>italic</i>, <code>code</code>.
  Do NOT use Markdown formatting (*bold*, _italic_). Use HTML tags only.

## CRITICAL DATA INTEGRITY RULES (TELEGRAM)
- EVERY number you cite (probability, PPG, edge, xG) must come from a tool call.
  If a tool returned "prob_H: 0.62", say "62%" — NOT "78%", NOT "about 80%".
- When quoting handicap data, CHECK which team is HOME. The match format is
  "HomeTeam vs AwayTeam". If the user says "Pisa +1 vs Como" but the match
  is "Como vs Pisa", Pisa is AWAY. Don't confuse home/away probabilities.
- NEVER invent form stats like "3.0 PPG last 5". Call get_team_detail or
  query_history to get real form data. If you don't have it, say so.
- If the user asks "what should I bet to recover my losses", your answer is
  "Don't chase. Stick to normal stakes." NEVER suggest recovery parlays.
"""


def _build_telegram_prompt() -> str:
    """Build system prompt with Telegram-specific addon."""
    return _build_system_prompt() + _TELEGRAM_SYSTEM_ADDON


# Models — use the same routing logic but simplified for Telegram
_MODEL_SONNET = "claude-sonnet-4-6"
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


def _select_model(message: str, conversation: ConversationManager | None = None) -> str:
    """Select model based on message content and conversation context."""
    msg_lower = message.lower().strip()

    # If in an ongoing deep conversation, stay on Sonnet
    if conversation:
        recent = conversation.get_messages()
        # If the last assistant message was from a Sonnet-level analysis,
        # follow-up questions should also use Sonnet for coherence
        if len(recent) >= 2 and len(msg_lower) < 30:
            last_assistant = next((m for m in reversed(recent) if m["role"] == "assistant"), None)
            if last_assistant and isinstance(last_assistant.get("content"), str):
                if len(last_assistant["content"]) > 500:
                    return _MODEL_SONNET

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
    from web.advisor import _truncate_tool_result as _impl

    return _impl(result_str, max_chars=max_chars)


# Tool name → user-friendly status message
_TOOL_STATUS = {
    "get_match_prediction": "Analyzing match prediction",
    "get_match_context": "Loading match context",
    "get_team_detail": "Looking up team stats",
    "get_player_stats": "Checking player data",
    "get_h2h": "Pulling head-to-head history",
    "get_value_bets": "Scanning value bets",
    "get_match_players": "Loading squad details",
    "get_live_matches": "Checking live scores",
    "get_results": "Fetching results",
    "get_bankroll_status": "Loading bankroll",
    "get_betting_performance": "Analyzing your track record",
    "get_match_context": "Loading match context",
    "get_match_scorers": "Checking goalscorer odds",
    "place_bet": "Placing bet",
    "manage_bets": "Managing bets",
    "settle_bets": "Settling bets",
    "query_history": "Searching 21 seasons of data",
    "build_parlay": "Building parlay",
}


def _call_claude(user_message: str, conversation: ConversationManager,
                 token: str, chat_id: str) -> str:
    """Call Claude with tool loop. Returns HTML-formatted response.

    Sends status messages during tool calls so the user knows what's happening.
    Converts Claude's Markdown output to Telegram-safe HTML.
    """
    import anthropic

    api_key = _get_env("ANTHROPIC_API_KEY")
    if not api_key:
        return "ANTHROPIC_API_KEY not configured."

    client = anthropic.Anthropic(api_key=api_key)
    system_prompt = _build_telegram_prompt()

    conversation.add_user_message(user_message)
    messages = conversation.get_messages()

    model = _select_model(user_message, conversation)
    max_tokens = 4096 if model == _MODEL_SONNET else 2048

    # Track if we've sent a status message (to edit/delete later)
    status_msg_id = None
    # Track tool usage for fact-checking
    all_tool_results: list[str] = []
    used_any_tools = False

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
            _delete_message(token, chat_id, status_msg_id)
            if "404" in str(e) or "not_found" in str(e):
                return "AI model temporarily unavailable. Try again in a moment."
            if "overloaded" in str(e).lower() or "529" in str(e):
                return "AI is overloaded right now. Try again in 30 seconds."
            if "rate_limit" in str(e).lower() or "429" in str(e):
                return "Too many requests. Wait a minute and try again."
            return "Something went wrong with the AI. Try again."
        except Exception as e:
            log.error("Claude call failed: %s", e)
            _delete_message(token, chat_id, status_msg_id)
            return "Something went wrong. Try again."

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

        # If no tool use, we're done — fact-check before sending
        if response.stop_reason != "tool_use" or not tool_uses:
            conversation.add_assistant_message(full_text)
            _delete_message(token, chat_id, status_msg_id)
            checked = _fact_check_response(full_text, all_tool_results, used_any_tools)
            return _md_to_html(checked)

        # Execute tools — send status message showing what we're doing
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
            tool_name = tu["name"]
            log.info("Tool call: %s(%s)", tool_name, json.dumps(tu["input"])[:100])

            # Send/update status message
            status_text = _TOOL_STATUS.get(tool_name, f"Working on {tool_name}")
            match_hint = tu["input"].get("match", tu["input"].get("team", ""))
            if match_hint:
                status_text += f": {match_hint}"
            status_text = f"\u23f3 <i>{status_text}...</i>"

            if status_msg_id:
                _edit_message_status(token, chat_id, status_msg_id, status_text)
            else:
                status_msg_id = _send_status(token, chat_id, status_text)

            handler = TOOL_HANDLERS.get(tool_name)
            if handler:
                try:
                    result_str = handler(tu["input"])
                    result_str = _truncate_tool_result(result_str)
                    used_any_tools = True
                    all_tool_results.append(result_str)
                except Exception as e:
                    log.warning("Tool %s failed: %s", tool_name, e)
                    result_str = json.dumps({"error": str(e)})
            else:
                result_str = json.dumps({"error": f"Unknown tool: {tool_name}"})

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": result_str,
            })

        conversation.add_tool_exchange(assistant_content, tool_results)
        messages = conversation.get_messages()

    _delete_message(token, chat_id, status_msg_id)
    if full_text:
        checked = _fact_check_response(full_text, all_tool_results, used_any_tools)
        return _md_to_html(checked)
    return "I ran out of processing rounds. Try a simpler question."


# ---------------------------------------------------------------------------
# Response fact-checker — structural guarantee against fabrication
# ---------------------------------------------------------------------------

# Collect tool results during a conversation turn so we can verify the response
_last_tool_results: list[str] = []


def _fact_check_response(response_text: str, tool_results: list[str],
                         used_tools: bool) -> str:
    """Post-process Claude's response to catch fabrication.

    1. If Claude answered a data question without calling any tools, flag it.
    2. Extract percentage numbers from the response and check if they
       appear in any tool output. If a probability is off by >10pp from
       any tool number, append a warning.
    3. Detect "recovery parlay" / "chase" language and append a warning.
    """
    warnings = []

    # Check 1: Did Claude skip tools on a data question?
    data_keywords = ["probability", "chance", "edge", "ppg", "form",
                     "xg", "expected goals", "win rate", "roi"]
    resp_lower = response_text.lower()
    looks_like_data_answer = any(kw in resp_lower for kw in data_keywords)
    if looks_like_data_answer and not used_tools:
        warnings.append(
            "\u26a0\ufe0f <i>This response was generated without calling data tools. "
            "Numbers may not reflect current model data. "
            "Ask me to check again if something looks off.</i>"
        )

    # Check 2: Detect chasing/recovery language
    chase_phrases = ["recovery parlay", "recover your loss", "win it back",
                     "make up for", "chase", "double down after",
                     "pick for recovery", "recovery play", "get it back",
                     "recoup", "revenge bet"]
    if any(phrase in resp_lower for phrase in chase_phrases):
        warnings.append(
            "\u26a0\ufe0f <i>Reminder: never size up after a loss. "
            "Stick to normal stakes — the edge is long-term.</i>"
        )

    # Check 2b: Detect "gut feeling" / emotional language
    gut_phrases = ["hits your gut", "gut feeling", "gut says",
                   "feel lucky", "just feels right"]
    if any(phrase in resp_lower for phrase in gut_phrases):
        warnings.append(
            "\u26a0\ufe0f <i>Betting decisions should be based on edge and probability, "
            "not gut feelings.</i>"
        )

    # Check 3: Extract percentages and cross-reference with tool output
    import re
    cited_pcts = re.findall(r'(\d{2,3})%', response_text)
    if cited_pcts and tool_results:
        all_tool_text = " ".join(tool_results)
        # Find all numbers in tool output
        tool_numbers = set()
        for match in re.finditer(r'(\d+\.?\d*)', all_tool_text):
            try:
                val = float(match.group(1))
                if 0 < val <= 100:
                    tool_numbers.add(round(val))
                if 0 < val < 1:
                    tool_numbers.add(round(val * 100))
            except ValueError:
                pass

        # Check each cited percentage
        suspicious = []
        for pct_str in cited_pcts:
            pct = int(pct_str)
            if 20 <= pct <= 95:  # Only check plausible probability range
                # Is this number within 5pp of any tool number?
                close_enough = any(abs(pct - tn) <= 5 for tn in tool_numbers)
                if not close_enough and tool_numbers:
                    suspicious.append(pct_str)

        if suspicious:
            warnings.append(
                f"\u26a0\ufe0f <i>Some numbers ({', '.join(suspicious[:3])}%) "
                f"may not match current model data. Verify on the dashboard.</i>"
            )

    if warnings:
        return response_text + "\n\n" + "\n".join(warnings)
    return response_text


def _send_status(token: str, chat_id: str, text: str) -> int | None:
    """Send a status message and return its message_id for later editing/deletion."""
    result = _tg_request(token, "sendMessage", {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_notification": True,
    })
    if result:
        return result.get("message_id")
    return None


def _edit_message_status(token: str, chat_id: str, message_id: int | None, text: str):
    """Edit an existing message (for updating status indicators)."""
    if not message_id:
        return
    _tg_request(token, "editMessageText", {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": text,
        "parse_mode": "HTML",
    }, timeout=5)


def _delete_message(token: str, chat_id: str, message_id: int | None):
    """Delete a message (clean up status messages after response is ready)."""
    if not message_id:
        return
    _tg_request(token, "deleteMessage", {
        "chat_id": chat_id,
        "message_id": message_id,
    }, timeout=5)


# ---------------------------------------------------------------------------
# Quick command handlers (no Claude API call needed)
# ---------------------------------------------------------------------------

def _handle_start() -> str:
    """Handle /start — contextual greeting showing current state."""
    from scripts.pipeline.notify import TgMsg, _html_escape, _bankroll_in_context, _get_bankroll_context

    br_ctx = _get_bankroll_context()
    tg = TgMsg()
    tg.raw("<b>SerieAI Advisor</b>")
    tg.line("Your betting mental coach, on Telegram.")
    tg.blank()

    # Show current state
    tg.raw(f"Balance: {_html_escape(_bankroll_in_context(br_ctx))}")

    # Pending bets count
    try:
        from config.settings import DATA_DIR
        jpath = DATA_DIR / "betting" / "bet_journal.json"
        if jpath.exists():
            with open(jpath) as f:
                journal = json.load(f)
            bets = journal.get("bets", {})
            if isinstance(bets, dict):
                bets = list(bets.values())
            pending = sum(1 for b in bets if b.get("status") == "pending")
            if pending:
                tg.raw(f"{pending} active bet{'s' if pending > 1 else ''} in play")
    except Exception:
        pass

    tg.blank()
    tg.raw("<b>Commands:</b>")
    tg.raw("  /bets \u2014 value bets in the slip")
    tg.raw("  /parlays \u2014 top parlay picks")
    tg.raw("  /bankroll \u2014 balance, ROI, streak")
    tg.raw("  /today \u2014 today's matches")
    tg.raw("  /live \u2014 live scores + your bets")
    tg.raw("  /league \u2014 filter by league")
    tg.raw("  /digest \u2014 daily summary")
    tg.raw("  /help \u2014 all commands")
    tg.blank()
    tg.italic("Or just ask me anything about football.")
    return tg.build()


def _handle_bets() -> str:
    """Handle /bets — value bets with clear formatting, grouped by league."""
    from scripts.pipeline.notify import TgMsg, _html_escape

    MKT_NAMES = {
        "h2h": "Match Result", "1X2": "Match Result",
        "totals": "Goals", "O/U": "Goals",
        "double_chance": "Double Chance", "DC": "Double Chance",
        "spreads": "Handicap", "AH": "Handicap",
        "btts": "Both Teams Score", "draw_no_bet": "Draw No Bet",
    }

    LEAGUE_HEADERS = {
        "serie_a": "Serie A",
        "premier_league": "Premier League",
        "la_liga": "La Liga",
        "bundesliga": "Bundesliga",
        "ligue_1": "Ligue 1",
    }

    try:
        result = json.loads(_tool_get_value_bets({}))
        bets = result.get("bets", [])
        if not bets:
            return "\U0001f4ad No value bets right now. Market is efficient today."

        # Check if multiple leagues are present
        leagues_in_bets = set(_resolve_league(b) for b in bets)
        multi_league = len(leagues_in_bets) > 1

        tg = TgMsg()
        tg.raw(f"\U0001f4b0 <b>Value Bets</b> ({len(bets)} picks)")
        tg.blank()

        if multi_league:
            # Group bets by league
            bets_by_league = {}
            for b in bets:
                league = _resolve_league(b)
                bets_by_league.setdefault(league, []).append(b)

            for league, league_bets in bets_by_league.items():
                header = LEAGUE_HEADERS.get(league, league.replace("_", " ").title())
                tg.raw(f"\U0001f3c6 <b>{_html_escape(header)}</b>")
                tg.blank()

                for b in league_bets:
                    _format_bet_line(tg, b, MKT_NAMES, _html_escape)
        else:
            for b in bets:
                _format_bet_line(tg, b, MKT_NAMES, _html_escape)

        # Removed: Handicap/Goals/BTTS raw predictions were showing disabled markets.
        # Only show bets from unified_bet_slip (already filtered by market_rules).

        tg.italic("Ask me about any pick for deeper analysis.")
        return tg.build()
    except Exception as e:
        log.warning("/bets failed: %s", e)
        return f"Failed to load bets: {e}"


_BOOKMAKER_DISPLAY = {
    "Alt totals book": "Alt Totals",
    "DC bookmaker (real)": "DC Book",
    "DC bookmaker": "DC Book",
    "alt_totals": "Alt Totals",
}


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


def _format_bet_line(tg, b: dict, mkt_names: dict, escape_fn):
    """Format a single bet line for Telegram output."""
    edge = b.get("edge_pct", 0)
    market_raw = b.get("market", "?")
    market = mkt_names.get(market_raw, market_raw)
    stake = b.get("stake", b.get("stake_amount", 0))
    odds = b.get("odds", b.get("best_odds", 0))
    bm_raw = b.get("bookmaker", b.get("best_bookmaker", ""))
    bm = _BOOKMAKER_DISPLAY.get(bm_raw, bm_raw)

    tg.raw(f"\u26bd <b>{escape_fn(b.get('match', '?'))}</b>")
    tg.raw(f"   {escape_fn(market)}: <b>{escape_fn(b.get('selection', '?'))}</b>")
    tg.raw(f"   Odds: <b>{odds}</b>"
           + (f" ({escape_fn(bm)})" if bm else "")
           + f"  |  Edge: <b>{edge:+.1f}%</b>")
    if stake:
        tg.raw(f"   Stake: \u20ac{stake:.2f}")
    tg.blank()


def _handle_bankroll() -> str:
    """Handle /bankroll — clear bankroll status."""
    from scripts.pipeline.notify import TgMsg, _html_escape

    try:
        result = json.loads(_tool_get_bankroll_status({}))
        current = result.get("current_bankroll", 0)
        initial = result.get("initial_bankroll", 1000)
        roi = result.get("roi_pct", 0)
        peak = result.get("peak_bankroll", current)
        streak = result.get("current_streak", 0)

        tg = TgMsg()
        tg.raw("\U0001f4b3 <b>Bankroll</b>")
        tg.blank()

        tg.raw(f"   Balance: <b>\u20ac{current:,.2f}</b>" if isinstance(current, (int, float)) else f"   Balance: {current}")
        tg.raw(f"   Started: \u20ac{initial:,.0f}" if isinstance(initial, (int, float)) else "")
        tg.raw(f"   Return: <b>{roi:+.1f}%</b>" if roi else "")
        tg.raw(f"   Peak: \u20ac{peak:,.0f}" if isinstance(peak, (int, float)) else "")
        tg.blank()

        if streak:
            if streak > 0:
                tg.raw(f"   \U0001f525 {streak}-bet winning streak")
            else:
                tg.raw(f"   \u2744\ufe0f {abs(streak)}-bet losing streak")

        # Drawdown
        if isinstance(current, (int, float)) and isinstance(peak, (int, float)) and peak > 0:
            dd = (peak - current) / peak * 100
            if dd > 2:
                tg.raw(f"   \u26a0\ufe0f {dd:.0f}% below peak balance")

        # Market breakdown — consolidated and sorted by profitability
        mkt = result.get("market_breakdown", {})
        if mkt:
            # Consolidate similar markets
            _MARKET_GROUPS = {
                "1X2": "Match Result",
                "DC": "Double Chance",
                "DNB": "Draw No Bet",
                "BTTS": "BTTS",
            }
            consolidated: dict = {}
            for name, stats in mkt.items():
                # Group AH variants → "Asian Handicap", O/U variants → "Over/Under"
                if name.startswith(("AH", "spreads")):
                    group = "Asian Handicap"
                elif name.startswith(("O/U", "totals")):
                    group = "Over/Under"
                else:
                    group = _MARKET_GROUPS.get(name, name)

                g = consolidated.setdefault(group, {"wins": 0, "losses": 0, "profit": 0.0, "total": 0})
                g["wins"] += stats.get("wins", 0)
                g["losses"] += stats.get("losses", 0)
                g["profit"] += stats.get("profit", 0)
                g["total"] += stats.get("total", stats.get("wins", 0) + stats.get("losses", 0))

            # Sort: profitable first, then by total bets
            sorted_markets = sorted(
                consolidated.items(),
                key=lambda x: (-1 if x[1]["profit"] >= 0 else 1, -x[1]["total"]),
            )

            tg.blank()
            tg.raw("<b>Performance by Market:</b>")
            for name, stats in sorted_markets:
                if stats["total"] == 0:
                    continue
                wr = stats["wins"] / max(stats["total"], 1) * 100
                profit = stats["profit"]
                sign = "+" if profit >= 0 else ""
                emoji = "\u2705" if profit >= 0 else "\u274c"
                tg.raw(f"   {emoji} {_html_escape(name)}: "
                       f"{wr:.0f}% WR ({stats['total']} bets), "
                       f"{sign}\u20ac{profit:.2f}")

        # Post-improvement tracking
        try:
            from scripts.betting.benchmark_tracker import get_benchmark_report
            bench = get_benchmark_report()
            after = bench.get("after", {})
            progress = bench.get("progress_pct", 0)
            if after.get("count", 0) > 0 or progress > 0:
                tg.blank()
                tg.raw("<b>Since Improvements (Apr 10):</b>")
                if after["count"] > 0:
                    tg.raw(f"   Record: {after['wins']}W-{after['losses']}L"
                           f" ({after['win_rate']:.0f}% WR)")
                    tg.raw(f"   P&L: {'+'if after['total_profit']>=0 else ''}"
                           f"\u20ac{after['total_profit']:.2f}"
                           f" ({after['roi_pct']:+.1f}% ROI)")
                tg.raw(f"   Progress: {after['count']}/{bench.get('target_bets', 50)} "
                       f"bets ({progress:.0f}%)")
        except Exception:
            pass

        return tg.build()
    except Exception as e:
        log.warning("/bankroll failed: %s", e)
        return f"Failed to load bankroll: {e}"


def _handle_live() -> str:
    """Handle /live — live scores with bet status."""
    from scripts.pipeline.notify import TgMsg, _html_escape

    STATUS_NAMES = {
        "first_half": "1st Half",
        "second_half": "2nd Half",
        "half_time": "Half Time",
        "completed": "Full Time",
        "not_started": "Not Started",
        "pre_match": "Pre-Match",
        "in_play": "In Play",
    }

    try:
        result = json.loads(_tool_get_live_matches({}))
        matches = result.get("matches", [])
        if not matches:
            return "\u26bd No live matches right now. Check /today for upcoming."

        tg = TgMsg()
        tg.raw(f"\U0001f534 <b>Live Matches</b>")
        tg.blank()

        for m in matches:
            status_raw = m.get("status", "?")
            status = STATUS_NAMES.get(status_raw, status_raw)
            score = m.get("score", "?")
            minute = m.get("minute", "")
            min_str = f" ({minute}')" if minute else ""

            tg.raw(f"\u26bd <b>{_html_escape(m.get('match', '?'))}</b>")
            tg.raw(f"   Score: <b>{_html_escape(str(score))}</b>  |  "
                   f"{_html_escape(status)}{min_str}")

            bets = m.get("bets", [])
            if bets:
                tg.raw("   <b>Your bets:</b>")
                for b in bets:
                    if b.get("status") == "winning":
                        icon = "\u2705"
                        label = "Winning"
                    elif b.get("status") == "losing":
                        icon = "\u274c"
                        label = "Losing"
                    else:
                        icon = "\u23f3"
                        label = "Pending"
                    tg.raw(f"   {icon} {_html_escape(b.get('selection', '?'))} "
                           f"@{b.get('odds', '?')} — {label}")
            tg.blank()

        return tg.build()
    except Exception as e:
        log.warning("/live failed: %s", e)
        return f"Failed to load live matches: {e}"


def _handle_parlays() -> str:
    """Handle /parlays — today's top parlay picks."""
    from scripts.pipeline.notify import TgMsg, _html_escape

    # Human-readable category names
    CAT_NAMES = {
        "banker_combos": "\U0001f3e6 Safe Combo",
        "draw_specials": "\U0001f3af Draw Parlay",
        "safe_doubles": "\U0001f91d Safe Double",
        "value_trebles": "\U0001f4b0 Value Treble",
        "sharp_specials": "\U0001f9e0 Sharp Pick",
        "long_shots": "\U0001f680 Long Shot",
        "same_game": "\u26bd Same Game",
    }

    # Human-readable market names
    MKT_NAMES = {
        "double_chance": "Double Chance",
        "btts": "Both Teams Score",
        "draw_no_bet": "Draw No Bet",
        "h2h": "Match Result",
        "totals": "Goals",
        "spreads": "Handicap",
    }

    # Translate internal "why" reasons to clear language
    def _humanize_why(reasons: list) -> str:
        result = []
        for r in reasons[:2]:
            r = r.replace("DC anchor", "Double Chance anchor")
            r = r.replace("hist WR", "historical win rate")
            r = r.replace("high-prob legs", "high-probability legs")
            r = r.replace(">70%", "over 70% each")
            result.append(r)
        return " | ".join(result)

    try:
        from config.settings import DATA_DIR
        report_path = DATA_DIR / "betting" / "parlay_report.json"
        if not report_path.exists():
            return "No parlay report available. Run the pipeline first."

        with open(report_path) as f:
            report = json.load(f)

        top_picks = report.get("top_picks", [])
        if not top_picks:
            return "No top picks selected. Check the dashboard for all parlays."

        tg = TgMsg()
        tg.raw(f"\U0001f3af <b>Top Parlay Picks</b>")
        tg.blank()

        rank_emojis = ["\U0001f947", "\U0001f948", "\U0001f949"]
        for idx, pick in enumerate(top_picks[:3]):
            p = pick.get("parlay", {})
            cat_raw = pick.get("category", "")
            cat_name = CAT_NAMES.get(cat_raw, cat_raw.replace("_", " ").title())
            combined = p.get("combined_odds", 0)
            hp = p.get("hit_probability", {})
            hit_pct = hp.get("median", hp.get("copula_adjusted", 0))
            if hit_pct <= 1:
                hit_pct *= 100
            stake = p.get("stake", 0)
            n_legs = p.get("n_legs", len(p.get("legs", [])))

            rank_emoji = rank_emojis[idx] if idx < 3 else ""
            tg.raw(f"{rank_emoji} <b>{_html_escape(cat_name)}</b>")
            tg.raw(f"   Combined odds: <b>{combined:.2f}</b>  |  {n_legs} legs")
            tg.blank()

            for leg in p.get("legs", []):
                mkt = leg.get("market", "")
                mkt_name = MKT_NAMES.get(mkt, mkt.replace("_", " ").title())
                match = leg.get("match", "?")
                sel = leg.get("selection", "?")
                odds = leg.get("odds", 0)
                tg.raw(f"   \u2022 {_html_escape(match)}")
                tg.raw(f"     {_html_escape(mkt_name)}: <b>{_html_escape(sel)}</b> @{odds:.2f}")

            tg.blank()
            tg.raw(f"   \U0001f4b5 Stake: <b>\u20ac{stake:.2f}</b>  |  "
                   f"\U0001f3b2 Hit chance: <b>{hit_pct:.0f}%</b>")

            if stake > 0 and combined > 1:
                potential_win = stake * combined - stake
                tg.raw(f"   \U0001f4b0 Win: <b>\u20ac{potential_win:.2f}</b>  |  "
                       f"Lose: \u20ac{stake:.2f}")

            why = pick.get("why", [])
            if why:
                tg.raw(f"   <i>\U0001f4a1 {_html_escape(_humanize_why(why))}</i>")
            tg.blank()
            tg.raw("\u2500" * 20)
            tg.blank()

        tg.italic("Ask me about any pick for deeper analysis.")
        return tg.build()
    except Exception as e:
        log.warning("/parlays failed: %s", e)
        return f"Failed to load parlays: {e}"


def _handle_player_lookup(query: str) -> str:
    """Look up a player's team history, nationality, and market value."""
    from scripts.pipeline.notify import TgMsg, _html_escape

    try:
        from scripts.analysis.player_history import build_player_history, get_player_profile
        history = build_player_history()

        # Fuzzy match: find players whose name contains the query
        query_lower = query.lower()

        # Search in history (multi-team players)
        matches = [(name, teams) for name, teams in history.items()
                   if query_lower in name.lower()]

        # Also search in market values for single-team players
        if not matches:
            import pandas as pd
            from pathlib import Path as _P
            for season in ["2025_2026", "2024_2025"]:
                mv_path = _P("data/external/transfermarkt") / f"market_values_{season}.parquet"
                if mv_path.exists():
                    mv = pd.read_parquet(mv_path)
                    found = mv[mv["player_name"].str.lower().str.contains(query_lower, na=False)]
                    for _, row in found.iterrows():
                        pname = row["player_name"]
                        if not any(m[0] == pname for m in matches):
                            matches.append((pname, [{"team": row["team"], "seasons": [season.replace("_","-")],
                                                     "total_matches": 0, "first_season": "", "last_season": ""}]))
                    if matches:
                        break

        if not matches:
            return f"No player found matching \"{query}\". Try a last name like <code>/player Belotti</code>"

        # Sort by closest match
        matches.sort(key=lambda x: (
            0 if x[0].lower() == query_lower else
            1 if x[0].lower().startswith(query_lower) else
            2 if query_lower in x[0].split()[-1].lower() else 3
        ))

        FLAG_MAP = {
            "Italy": "\U0001f1ee\U0001f1f9", "Argentina": "\U0001f1e6\U0001f1f7",
            "Brazil": "\U0001f1e7\U0001f1f7", "France": "\U0001f1eb\U0001f1f7",
            "Spain": "\U0001f1ea\U0001f1f8", "Portugal": "\U0001f1f5\U0001f1f9",
            "Germany": "\U0001f1e9\U0001f1ea", "Netherlands": "\U0001f1f3\U0001f1f1",
            "Belgium": "\U0001f1e7\U0001f1ea", "Croatia": "\U0001f1ed\U0001f1f7",
            "Serbia": "\U0001f1f7\U0001f1f8", "Nigeria": "\U0001f1f3\U0001f1ec",
            "Ghana": "\U0001f1ec\U0001f1ed", "Colombia": "\U0001f1e8\U0001f1f4",
            "Uruguay": "\U0001f1fa\U0001f1fe", "Turkey": "\U0001f1f9\U0001f1f7",
            "Poland": "\U0001f1f5\U0001f1f1", "Scotland": "\U0001f3f4\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f",
            "Norway": "\U0001f1f3\U0001f1f4", "Sweden": "\U0001f1f8\U0001f1ea",
            "Denmark": "\U0001f1e9\U0001f1f0", "Switzerland": "\U0001f1e8\U0001f1ed",
            "Cameroon": "\U0001f1e8\U0001f1f2", "Ivory Coast": "\U0001f1e8\U0001f1ee",
            "Senegal": "\U0001f1f8\U0001f1f3", "Japan": "\U0001f1ef\U0001f1f5",
            "South Korea": "\U0001f1f0\U0001f1f7", "Mexico": "\U0001f1f2\U0001f1fd",
            "United States": "\U0001f1fa\U0001f1f8", "Canada": "\U0001f1e8\U0001f1e6",
            "Austria": "\U0001f1e6\U0001f1f9", "Czech Republic": "\U0001f1e8\U0001f1ff",
            "Albania": "\U0001f1e6\U0001f1f1", "Romania": "\U0001f1f7\U0001f1f4",
            "Greece": "\U0001f1ec\U0001f1f7", "Morocco": "\U0001f1f2\U0001f1e6",
            "Tunisia": "\U0001f1f9\U0001f1f3", "Algeria": "\U0001f1e9\U0001f1ff",
            "England": "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f",
        }

        tg = TgMsg()
        for player_name, teams in matches[:3]:
            profile = get_player_profile(player_name, history)

            tg.raw(f"\U0001f464 <b>{_html_escape(player_name)}</b>")

            # Nationality + market value
            if profile:
                nat = profile.get("nationality")
                mv = profile.get("market_value_eur")
                fee = profile.get("transfer_fee_eur")

                info_parts = []
                if nat:
                    flag = FLAG_MAP.get(nat, "\U0001f30d")
                    info_parts.append(f"{flag} {nat}")
                if mv:
                    if mv >= 1_000_000:
                        info_parts.append(f"Value: \u20ac{mv/1_000_000:.1f}M")
                    else:
                        info_parts.append(f"Value: \u20ac{mv/1_000:.0f}K")
                if fee:
                    if fee >= 1_000_000:
                        info_parts.append(f"Fee: \u20ac{fee/1_000_000:.1f}M")

                if info_parts:
                    tg.raw(f"   {' | '.join(info_parts)}")

            tg.blank()
            tg.raw("   <b>Career:</b>")

            for i, t in enumerate(reversed(teams)):
                if i == 0:
                    marker = "\u25b6\ufe0f"  # Current team
                else:
                    marker = "\u25aa\ufe0f"  # Past team
                seasons = ", ".join(t["seasons"][-3:])
                if len(t["seasons"]) > 3:
                    seasons = f"{t['seasons'][0]}...{t['seasons'][-1]}"
                matches_str = f" ({t['total_matches']} matches)" if t['total_matches'] > 0 else ""
                tg.raw(f"   {marker} <b>{_html_escape(t['team'])}</b>  {seasons}{matches_str}")

            tg.blank()

        if len(matches) > 3:
            tg.italic(f"Showing 3 of {len(matches)} results. Be more specific.")

        return tg.build()
    except Exception as e:
        return f"Failed: {e}"


def _get_weekly_bets() -> dict:
    """Load settled bets grouped by week."""
    from collections import defaultdict
    from config.settings import DATA_DIR

    journal_path = DATA_DIR / "betting" / "bet_journal.json"
    if not journal_path.exists():
        return {}

    with open(journal_path) as f:
        journal = json.load(f)

    by_week = defaultdict(list)
    for bet_id, bet in journal.get("bets", {}).items():
        if bet.get("status") not in ("won", "lost", "push"):
            continue
        d = bet.get("date", "")
        if d:
            try:
                dt = datetime.strptime(d, "%Y-%m-%d")
                week_key = f"{dt.year}-W{dt.isocalendar()[1]:02d}"
                by_week[week_key].append(bet)
            except ValueError:
                pass
    return dict(by_week)


def _handle_summary_menu(token: str, chat_id: str) -> str | None:
    """Show week selector with inline keyboard buttons."""
    from scripts.pipeline.notify import TgMsg, _html_escape

    by_week = _get_weekly_bets()
    if not by_week:
        return "No settled bets yet."

    tg = TgMsg()
    tg.raw("\U0001f4c5 <b>Weekly History</b>")
    tg.raw("Tap a week to see all bets:")
    tg.blank()

    # Build summary + buttons
    rows = []
    for week_key in sorted(by_week.keys(), reverse=True):
        bets = by_week[week_key]
        won = sum(1 for b in bets if b["status"] == "won")
        lost = sum(1 for b in bets if b["status"] == "lost")
        profit = sum(
            b.get("profit", 0) if b["status"] == "won" else -(b.get("stake", 0))
            for b in bets
        )
        sign = "+" if profit >= 0 else ""
        emoji = "\u2705" if profit >= 0 else "\u274c"

        # Week label: "W12 (Mar 16-22)"
        dates = sorted(b.get("date", "") for b in bets if b.get("date"))
        if dates:
            first = datetime.strptime(dates[0], "%Y-%m-%d")
            last = datetime.strptime(dates[-1], "%Y-%m-%d")
            date_range = f"{first.strftime('%b %d')}-{last.strftime('%d')}"
        else:
            date_range = ""

        week_num = week_key.split("-W")[1]
        label = f"{emoji} Week {week_num} ({date_range}): {won}W-{lost}L {sign}\u20ac{profit:.0f}"

        rows.append([_button(label, f"week:{week_key}")])

    keyboard = _inline_keyboard(rows)
    _tg_send_message(token, chat_id, tg.build(), reply_markup=keyboard)
    return None  # Already sent


def _handle_week_detail(week_key: str) -> str:
    """Show all bets for a specific week."""
    from scripts.pipeline.notify import TgMsg, _html_escape

    MKT_NAMES = {
        "h2h": "Match Result", "1X2": "Match Result",
        "totals": "Goals", "O/U": "Goals",
        "double_chance": "Double Chance", "DC": "Double Chance",
    }

    by_week = _get_weekly_bets()
    bets = by_week.get(week_key, [])
    if not bets:
        return f"No bets found for {week_key}."

    # Sort by date
    bets.sort(key=lambda b: b.get("date", ""))

    won = sum(1 for b in bets if b["status"] == "won")
    lost = sum(1 for b in bets if b["status"] == "lost")
    push = sum(1 for b in bets if b["status"] == "push")
    total_staked = sum(b.get("stake", 0) for b in bets)
    total_profit = sum(
        b.get("profit", 0) if b["status"] == "won" else -(b.get("stake", 0))
        for b in bets
    )
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0

    week_num = week_key.split("-W")[1]
    dates = sorted(b.get("date", "") for b in bets if b.get("date"))
    date_range = ""
    if dates:
        first = datetime.strptime(dates[0], "%Y-%m-%d")
        last = datetime.strptime(dates[-1], "%Y-%m-%d")
        date_range = f" ({first.strftime('%b %d')} - {last.strftime('%b %d')})"

    tg = TgMsg()
    tg.raw(f"\U0001f4ca <b>Week {week_num}{date_range}</b>")
    tg.raw(f"   {won}W - {lost}L" + (f" - {push}P" if push else ""))
    tg.blank()

    for b in bets:
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
            result_str = "Push"

        score_str = f"  ({_html_escape(score)})" if score else ""
        tg.raw(f"{icon} <b>{_html_escape(match)}</b>{score_str}")
        tg.raw(f"   {_html_escape(bet_date)}")
        tg.raw(f"   {_html_escape(market)}: {_html_escape(sel)} @{odds:.2f}")
        tg.raw(f"   {result_str}")
        tg.blank()

    tg.raw("\u2500" * 20)
    sign = "+" if total_profit >= 0 else ""
    emoji = "\U0001f4b0" if total_profit >= 0 else "\U0001f4b8"
    tg.raw(f"{emoji} <b>Week P&amp;L: {sign}\u20ac{total_profit:.2f}</b> "
           f"(ROI: {roi:+.1f}%)")
    tg.raw(f"   Staked: \u20ac{total_staked:.2f} across {len(bets)} bets")

    return tg.build()


def _handle_today(token: str | None = None, chat_id: str | None = None) -> str:
    """Handle /today — today's matches with predictions, grouped by league.

    If `token` and `chat_id` are provided, also generate one Excel-style PNG
    per Serie A match with walkforward intelligence and send it as a photo
    attachment after the text digest. Silently degrades if Pillow is missing
    or if a per-match render fails.
    """
    from scripts.pipeline.notify import TgMsg, _html_escape

    CONF_EMOJI = {
        "VERY HIGH": "\U0001f7e2",  # green circle
        "HIGH": "\U0001f7e1",       # yellow circle
        "MEDIUM-HIGH": "\U0001f7e0", # orange circle
        "MEDIUM": "\u26aa",          # white circle
    }

    LEAGUE_HEADERS = {
        "serie_a": "Serie A",
        "premier_league": "Premier League",
        "la_liga": "La Liga",
        "bundesliga": "Bundesliga",
        "ligue_1": "Ligue 1",
    }

    try:
        from config.settings import DATA_DIR
        today = datetime.now().strftime("%Y-%m-%d")

        # Load predictions from all league files
        all_preds = []
        preds_path = DATA_DIR / "upcoming" / "predictions.json"
        if preds_path.exists():
            with open(preds_path) as f:
                preds = json.load(f)
            for p in preds.get("predictions", []):
                p.setdefault("league", _resolve_league(p))
                all_preds.append(p)

        # Load extra league predictions
        for league_key in LEAGUE_HEADERS:
            if league_key == "serie_a":
                continue
            extra_path = DATA_DIR / "upcoming" / f"predictions_{league_key}.json"
            if extra_path.exists():
                try:
                    with open(extra_path) as f:
                        extra = json.load(f)
                    for p in extra.get("predictions", []):
                        p.setdefault("league", league_key)
                        all_preds.append(p)
                except Exception:
                    pass

        today_matches = [p for p in all_preds if p.get("date", "").startswith(today)]

        # Load supplementary intelligence files (corners, cards, scorers, reasoning).
        # Indexed by (match, date) so per-match lookup is O(1). Missing files
        # degrade gracefully — only the available fields render.
        def _load_index(name: str) -> dict:
            path = DATA_DIR / "upcoming" / name
            if not path.exists():
                return {}
            try:
                d = json.load(open(path))
            except Exception:
                return {}
            preds = d.get("predictions", []) if isinstance(d, dict) else d
            return {(p.get("match", ""), p.get("date", "")): p
                    for p in preds if isinstance(p, dict)}
        # corners_idx + cards_idx loads removed 2026-05-06 — backtest showed
        # those models had skill score ≤ 0. See CLAUDE.md.
        # scorers JSON often lacks a date — index by match name only as a
        # second-tier lookup (covers stale-snapshot cases gracefully).
        _scorers_raw = json.load(open(DATA_DIR / "upcoming" / "scorers_predictions.json")) \
            if (DATA_DIR / "upcoming" / "scorers_predictions.json").exists() else {}
        _scorers_list = (_scorers_raw.get("predictions", [])
                         if isinstance(_scorers_raw, dict) else _scorers_raw) or []
        scorers_idx = {p.get("match"): p for p in _scorers_list
                       if isinstance(p, dict) and p.get("match")}
        reasoning_idx = _load_index("match_reasoning.json")

        if not today_matches:
            future = sorted([p for p in all_preds if p.get("date", "") > today],
                           key=lambda p: p.get("date", ""))
            if future:
                next_date = future[0].get("date", "?")
                next_matches = [p for p in all_preds if p.get("date", "").startswith(next_date)]
                tg = TgMsg()
                tg.raw(f"\U0001f4c5 No matches today.")
                tg.raw(f"Next matchday: <b>{_html_escape(next_date)}</b> ({len(next_matches)} matches)")
                tg.blank()
                for m in next_matches[:8]:
                    tg.raw(f"   \u26bd {_html_escape(m.get('match', '?'))}")
                return tg.build()
            return "\U0001f4c5 No upcoming matches found."

        # Group by league
        leagues_present = []
        matches_by_league = {}
        for m in today_matches:
            league = _resolve_league(m)
            matches_by_league.setdefault(league, []).append(m)
            if league not in leagues_present:
                leagues_present.append(league)

        multi_league = len(leagues_present) > 1

        tg = TgMsg()
        tg.raw(f"\U0001f4c5 <b>Today's Matches</b> ({len(today_matches)})")
        tg.blank()

        for league in leagues_present:
            league_matches = matches_by_league[league]

            # Show league header when multiple leagues are active
            if multi_league:
                header = LEAGUE_HEADERS.get(league, league.replace("_", " ").title())
                tg.raw(f"\U0001f3c6 <b>{_html_escape(header)}</b> ({len(league_matches)})")
                tg.blank()

            for m in league_matches:
                conf = m.get("confidence_level", "")
                prediction = m.get("predicted_result", "")
                kickoff = m.get("time", "")

                conf_dot = CONF_EMOJI.get(conf, "\u26aa")
                time_str = f"  {_html_escape(kickoff)}" if kickoff else ""

                tg.raw(f"{conf_dot} <b>{_html_escape(m.get('match', '?'))}</b>{time_str}")

                # Show prediction + probabilities
                probs = m.get("probabilities", m.get("betting_probabilities", {}))
                if isinstance(probs, dict) and probs:
                    h = probs.get("home", 0)
                    d = probs.get("draw", 0)
                    a = probs.get("away", 0)
                    tg.raw(f"   Home {h:.0%} | Draw {d:.0%} | Away {a:.0%}")
                    if prediction:
                        tg.raw(f"   Prediction: <b>{_html_escape(prediction)}</b>")

                # Match-intelligence bundle (scorers + reasoning only).
                # Corners + cards walkforward predictions were dropped 2026-05-04
                # after held-out 2024-25 backtest showed skill score ≤ 0 and
                # AUC ≈ 0.51-0.60 for all six lines (8.5/9.5/10.5 corners,
                # 3.5/4.5/5.5 cards). Predictions were base-rate ± noise,
                # not worth showing in a "betting intelligence" feed.
                key = (m.get("match", ""), m.get("date", ""))
                sc = scorers_idx.get(m.get("match", "")) or {}
                rs = reasoning_idx.get(key) or {}

                home_top = sc.get("home_top_scorers") or []
                away_top = sc.get("away_top_scorers") or []
                if home_top:
                    best = home_top[0]
                    tg.raw(f"   Top home scorer: <b>{_html_escape(best['player'])}</b> ({best['goal_prob']:.0%})")
                if away_top:
                    best = away_top[0]
                    tg.raw(f"   Top away scorer: <b>{_html_escape(best['player'])}</b> ({best['goal_prob']:.0%})")

                if rs.get("reasoning"):
                    tg.italic(f"   {_html_escape(rs['reasoning'])}")

                tg.blank()

        tg.italic("Tap /match to analyze any match in detail.")
        digest_text = tg.build()

        # PNG attachment path (corners/cards intel image) was removed
        # 2026-05-04 — those models had skill score ≤ 0 on held-out 2024-25
        # backtest, so the image had nothing real to show. Scorers + AI
        # reasoning are in the text digest above.

        return digest_text
    except Exception as e:
        log.warning("/today failed: %s", e)
        return f"Failed to load matches: {e}"


def _handle_digest() -> str:
    """Handle /digest — trigger daily digest."""
    try:
        from scripts.pipeline.notify import notify_daily_digest
        result = notify_daily_digest()
        if not result:
            return "Nothing to report today."
        return "Digest sent."
    except Exception as e:
        log.warning("/digest failed: %s", e)
        return f"Failed to generate digest: {e}"


def _handle_worldcup() -> str:
    """World Cup digest: today's slate + the three best-combo tiers.

    Same sources as /worldcup on the dashboard (predictions.json +
    market_odds.json via scripts.worldcup.combos — who-wins markets only).
    Times shown in Italy time.
    """
    from datetime import UTC, datetime
    from zoneinfo import ZoneInfo

    from scripts.worldcup.combos import (
        MARKET_ODDS_JSON,
        PREDICTIONS_JSON,
        build_best_combos,
    )
    from scripts.worldcup.engine import read_json_safe

    rome = ZoneInfo("Europe/Rome")
    now = datetime.now(UTC)
    doc = read_json_safe(PREDICTIONS_JSON, {})
    preds = doc.get("predictions", []) if isinstance(doc, dict) else []
    if not preds:
        return "🌍 No World Cup predictions on disk — run scripts.worldcup.refresh."
    market = dict(read_json_safe(MARKET_ODDS_JSON, {}))

    pick_sym = {"home": "1", "draw": "X", "away": "2"}
    today_rome = now.astimezone(rome).date()
    todays = []
    for p in preds:
        try:
            ko = datetime.fromisoformat(p.get("kickoff_utc") or "")
        except (TypeError, ValueError):
            continue
        if ko.astimezone(rome).date() == today_rome:
            todays.append((ko, p))
    todays.sort(key=lambda x: x[0])

    lines = [f"🌍 <b>World Cup 2026 — {today_rome.strftime('%A %d %B')}</b>", ""]
    if todays:
        lines.append("⚽ <b>Today</b> (Italy time)")
        for ko, p in todays:
            probs = p.get("probabilities") or {}
            pick = max(probs, key=probs.get) if probs else None
            played = ko <= now
            mark = "✔️ played" if played else (
                f"<b>{pick_sym.get(pick, '?')}</b> {probs.get(pick, 0) * 100:.0f}%")
            lines.append(
                f"{ko.astimezone(rome).strftime('%H:%M')}  "
                f"{p.get('home_team', '?')}–{p.get('away_team', '?')} — {mark}")
        lines.append("")
    else:
        lines.append("⚽ No matches today.")
        lines.append("")

    best = build_best_combos(preds, market)
    icons = {"safe": "🔒", "favorites": "⭐", "value": "💎"}
    if best.get("combos"):
        lines.append("🎯 <b>Best combos — next 48h</b>")
        for c in best["combos"]:
            cm = c["combined"]
            head = (f"{icons.get(c['key'], '🎯')} <b>{c['key'].capitalize()}</b> — "
                    f"{cm['prob'] * 100:.0f}%")
            if cm.get("market_odds"):
                ev = cm.get("ev")
                head += (f" @ {cm['market_odds']:.2f}"
                         f" (EV {'+' if ev >= 0 else ''}{ev * 100:.0f}%)")
            lines.append(head)
            for leg in c["legs"]:
                odds = f" @ {leg['market_odds']:.2f}" if leg.get("market_odds") else ""
                lines.append(f" • {leg['pick_label']} ({leg['prob'] * 100:.0f}%{odds})"
                             f" — {leg['match']}")
            lines.append("")
    else:
        lines.append("🎯 No combos — no upcoming matches in the next 48h.")
        lines.append("")
    lines.append("<i>Model probabilities, Sofascore-proxy prices — insight, not bets.</i>")
    return "\n".join(lines)


def send_worldcup_digest() -> bool:
    """One-shot morning push (refresh loop runs `--wc-digest`); independent
    of the polling loop — no lock, no signal handlers, send and exit."""
    token = _get_env("TELEGRAM_BOT_TOKEN")
    chat_id = _get_env("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        log.error("wc-digest: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set")
        return False
    text = _handle_worldcup()
    result = _tg_request(token, "sendMessage", {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
    })
    ok = result is not None
    log.info("wc-digest: %s", "sent" if ok else "send FAILED")
    return ok


def _handle_league(args: str, conversation: ConversationManager) -> str:
    """Handle /league — show, set, or clear league filter.

    /league         -> show available leagues with match counts + current filter
    /league epl     -> set filter to Premier League
    /league all     -> remove filter (show all leagues)
    """
    from scripts.pipeline.notify import TgMsg, _html_escape
    from config.leagues import LEAGUE_REGISTRY

    # League aliases for user-friendly input
    _ALIASES: dict[str, str] = {
        "epl": "premier_league", "pl": "premier_league", "eng": "premier_league",
        "premier_league": "premier_league", "premierleague": "premier_league",
        "serie_a": "serie_a", "seriea": "serie_a", "ita": "serie_a",
        "la_liga": "la_liga", "laliga": "la_liga", "esp": "la_liga",
        "bundesliga": "bundesliga", "ger": "bundesliga",
        "ligue_1": "ligue_1", "ligue1": "ligue_1", "fra": "ligue_1",
    }

    from config.leagues import ACTIVE_LEAGUES as ACTIVE_KEYS

    arg = args.strip().lower().replace("-", "_").replace(" ", "_")

    if arg == "all" or arg == "reset" or arg == "clear":
        conversation.set_league_filter(None)
        return "\U0001f30d League filter removed. Showing <b>all leagues</b>."

    if arg:
        resolved = _ALIASES.get(arg)
        if not resolved:
            return (f"Unknown league '<b>{_html_escape(arg)}</b>'. "
                    f"Try: epl, serie_a, la_liga, bundesliga, ligue_1")
        cfg = LEAGUE_REGISTRY.get(resolved)
        if not cfg:
            return f"League config not found for '{arg}'."
        conversation.set_league_filter(resolved)
        return (f"\U0001f3c6 League filter set to <b>{_html_escape(cfg.name)}</b>.\n"
                f"Commands like /bets, /today will show {_html_escape(cfg.name)} only.\n"
                f"Use /league all to show all leagues again.")

    # No args: show available leagues + current filter + match counts
    tg = TgMsg()
    tg.raw("<b>League Filter</b>")
    tg.blank()

    current = conversation.league_filter
    if current:
        cfg = LEAGUE_REGISTRY.get(current)
        name = cfg.name if cfg else current
        tg.raw(f"Currently showing: <b>{_html_escape(name)}</b>")
    else:
        tg.raw("Currently showing: <b>All leagues</b>")
    tg.blank()

    # Load match counts per league
    try:
        from config.settings import DATA_DIR
        preds_path = DATA_DIR / "upcoming" / "predictions.json"
        if preds_path.exists():
            with open(preds_path) as f:
                preds = json.load(f)
            pred_list = preds.get("predictions", [])
            today = datetime.now().strftime("%Y-%m-%d")
            today_matches = [p for p in pred_list if p.get("date", "").startswith(today)]
            all_upcoming = pred_list
        else:
            today_matches = []
            all_upcoming = []
    except Exception:
        today_matches = []
        all_upcoming = []

    tg.raw("<b>Available leagues:</b>")
    for key in ACTIVE_KEYS:
        cfg = LEAGUE_REGISTRY.get(key)
        if not cfg:
            continue
        # Count matches in this league
        from scripts.pipeline.notify import _detect_league_name
        league_today = sum(1 for m in today_matches if _detect_league_name(m) == cfg.name)
        league_total = sum(1 for m in all_upcoming if _detect_league_name(m) == cfg.name)
        active_marker = " \u2705" if current == key else ""
        counts = []
        if league_today:
            counts.append(f"{league_today} today")
        if league_total:
            counts.append(f"{league_total} upcoming")
        count_str = f" ({', '.join(counts)})" if counts else ""
        tg.raw(f"  {_html_escape(cfg.name)}{count_str}{active_marker}")

    tg.blank()
    tg.raw("<b>Set filter:</b>")
    tg.raw("  /league epl \u2014 Premier League only")
    tg.raw("  /league serie_a \u2014 Serie A only")
    tg.raw("  /league all \u2014 show all leagues")

    return tg.build()


def _handle_help() -> str:
    """Handle /help — all available commands."""
    from scripts.pipeline.notify import TgMsg

    tg = TgMsg()
    tg.raw("<b>SerieAI Commands</b>")
    tg.blank()
    tg.raw("<b>Quick data:</b>")
    tg.raw("  /bets \u2014 value bets in the slip")
    tg.raw("  /parlays \u2014 top parlay picks")
    tg.raw("  /bankroll \u2014 balance, ROI, streak")
    tg.raw("  /today \u2014 today's matches + predictions")
    tg.raw("  /match \u2014 tap a match for full analysis")
    tg.raw("  /live \u2014 live scores + your bets")
    tg.raw("  /league \u2014 filter by league (EPL, Serie A)")
    tg.blank()
    tg.raw("<b>Reports:</b>")
    tg.raw("  /digest \u2014 daily summary")
    tg.raw("  /wc — World Cup: today's slate + best combos")
    tg.blank()
    tg.raw("<b>Session:</b>")
    tg.raw("  /clear \u2014 reset conversation")
    tg.blank()
    tg.raw("<b>Natural language:</b>")
    tg.italic("Just ask anything \u2014 'analyze Milan vs Torino',")
    tg.italic("'should I bet on this?', 'best bets today'")
    return tg.build()


def _handle_match(token: str, chat_id: str) -> bool:
    """Handle /match — show inline keyboard of today's matches to tap.

    Returns True if sent successfully (caller should not send another response).
    """
    try:
        from config.settings import DATA_DIR
        today = datetime.now().strftime("%Y-%m-%d")

        # Load predictions from ALL leagues
        pred_list = []
        for pred_file in ["predictions.json", "predictions_premier_league.json"]:
            preds_path = DATA_DIR / "upcoming" / pred_file
            if preds_path.exists():
                with open(preds_path) as f:
                    preds = json.load(f)
                for p in preds.get("predictions", []):
                    pred_list.append(p)

        matches = [p for p in pred_list if p.get("date", "").startswith(today)]
        if not matches:
            matches = pred_list[:10]

        if not matches:
            return False

        # Build inline keyboard — one button per match
        rows = []
        for m in matches[:8]:
            match_name = m.get("match", "?")
            # Callback data has 64-byte limit — use shortened match key
            cb_data = f"analyze:{match_name[:50]}"
            rows.append([_button(match_name, cb_data)])

        keyboard = _inline_keyboard(rows)
        _tg_send_message(token, chat_id,
                         "<b>Tap a match for full analysis:</b>",
                         reply_markup=keyboard)
        return True
    except Exception as e:
        log.warning("/match failed: %s", e)
        return False


def _handle_callback_query(token: str, chat_id: str, callback_query: dict,
                           conversation: ConversationManager) -> str | None:
    """Handle inline keyboard button presses.

    Returns response text, or None if handled internally.
    """
    query_id = callback_query.get("id", "")
    data = callback_query.get("data", "")
    message_id = callback_query.get("message", {}).get("message_id", 0)

    # Default: silent answer to remove loading spinner
    answer_text = ""
    show_alert = False

    if data.startswith("analyze:"):
        match_name = data[len("analyze:"):]
        _tg_request(token, "answerCallbackQuery", {
            "callback_query_id": query_id,
            "text": f"Loading {match_name}...",
        }, timeout=5)

        # Add ex-player context to the analysis prompt
        ex_context = ""
        try:
            from scripts.analysis.player_history import get_match_context
            ctx = get_match_context(match_name)
            ex_home = ctx.get("home_vs_former", [])
            ex_away = ctx.get("away_vs_former", [])
            if ex_home or ex_away:
                parts = []
                for p in ex_home:
                    parts.append(f"{p['player']} (now {p['current_team']}, ex-{p['former_team']})")
                for p in ex_away:
                    parts.append(f"{p['player']} (now {p['current_team']}, ex-{p['former_team']})")
                ex_context = f"\n\nPlayers facing former team: {', '.join(parts)}. Mention this in your analysis."
        except Exception:
            pass

        return f"Analyze {match_name} — full prediction, value assessment, should I bet?{ex_context}"

    if data.startswith("place:"):
        # Quick-place bet from notification button
        parts = data[len("place:"):].split("|")
        if len(parts) >= 3:
            match, selection, odds = parts[0], parts[1], parts[2]
            try:
                from scripts.betting.bet_journal import add_bet
                bet_data = {
                    "match": match,
                    "selection": selection,
                    "odds": float(odds),
                    "market": "1X2" if selection in ("Home", "Draw", "Away") else "O/U",
                    "date": "",
                    "stake": 0,
                    "placed_at": __import__("datetime").datetime.now().isoformat(),
                }
                add_bet(bet_data)
                # Show confirmation popup (user must dismiss)
                _tg_request(token, "answerCallbackQuery", {
                    "callback_query_id": query_id,
                    "text": f"\u2705 Bet recorded: {selection} @{odds}",
                    "show_alert": True,
                }, timeout=5)
                # Edit the original message to mark this bet as placed
                if message_id:
                    _edit_message(token, chat_id, message_id,
                                  f"\u2705 <b>Bet placed:</b> {match}\n"
                                  f"{selection} @{odds}\n\n"
                                  f"<i>Recorded in journal. Confirm stake on dashboard.</i>")
                return None  # Already handled
            except Exception as e:
                _tg_request(token, "answerCallbackQuery", {
                    "callback_query_id": query_id,
                    "text": f"\u274c Failed: {str(e)[:50]}",
                    "show_alert": True,
                }, timeout=5)
                return None
        _tg_request(token, "answerCallbackQuery", {"callback_query_id": query_id}, timeout=5)
        return "\u274c Invalid bet data"

    if data.startswith("skip:"):
        parts = data[len("skip:"):].split("|")
        match = parts[0] if parts else "?"
        selection = parts[1] if len(parts) > 1 else "?"
        _tg_request(token, "answerCallbackQuery", {
            "callback_query_id": query_id,
            "text": f"Skipped {selection}",
        }, timeout=5)
        return f"\u274c Skipped: {match} {selection}. Good discipline \u2014 only bet when you're sure."

    if data.startswith("week:"):
        week_key = data[len("week:"):]
        _tg_request(token, "answerCallbackQuery", {
            "callback_query_id": query_id,
            "text": f"Loading week {week_key.split('-W')[1]}...",
        }, timeout=5)
        return _handle_week_detail(week_key)

    if data == "view:all_bets":
        _tg_request(token, "answerCallbackQuery", {
            "callback_query_id": query_id, "text": "Loading bets..."
        }, timeout=5)
        # Respond directly with merged bet list from ALL sources (instant)
        try:
            import json as _json
            from pathlib import Path as _Path
            _data = _Path(__file__).parent.parent.parent / "data"

            all_bets = []
            seen = set()

            # Source 1: Pipeline unified report (has O/U, DC, 1X2)
            report_path = _data / "betting" / "unified_report.json"
            if report_path.exists():
                report = _json.load(open(report_path))
                for b in report.get("bets", []):
                    key = f"{b.get('match','')}_{b.get('selection','')}"
                    if key not in seen:
                        seen.add(key)
                        all_bets.append({
                            "match": b.get("match", "?"),
                            "sel": b.get("selection", "?"),
                            "market": b.get("market", "?"),
                            "odds": b.get("best_odds", b.get("odds", 0)),
                            "edge": b.get("edge_pct", 0),
                            "source": "pipeline",
                        })

            # Source 2: Edge monitor (may find newer draws)
            scan_path = _data / "betting" / "edge_scan_latest.json"
            if scan_path.exists():
                scan = _json.load(open(scan_path))
                for b in scan.get("value_bets", []):
                    key = f"{b.get('match','')}_{b.get('selection','')}"
                    if key not in seen:
                        seen.add(key)
                        all_bets.append({
                            "match": b.get("match", "?"),
                            "sel": b.get("selection", "?"),
                            "market": b.get("market", "?"),
                            "odds": b.get("best_odds", 0),
                            "edge": b.get("edge_pct", 0),
                            "source": "scan",
                        })

            if not all_bets:
                return "\U0001f4ad No value bets right now — market is efficient today."

            all_bets.sort(key=lambda x: x["edge"], reverse=True)

            lines = [f"\U0001f3af <b>All Value Bets ({len(all_bets)})</b>\n"]
            for i, b in enumerate(all_bets[:10], 1):
                emoji = "\U0001f534" if "Draw" in b["sel"] else "\u26bd" if "Over" in b["sel"] else "\U0001f7e2"
                lines.append(f"{i}. {emoji} <b>{b['match']}</b>")
                lines.append(f"   {b['market']} {b['sel']} @{b['odds']:.2f} | edge {b['edge']:+.1f}%\n")
            return "\n".join(lines)
        except Exception as e:
            return f"Failed to load bets: {e}"

    # Default: answer the callback to remove loading spinner (unrecognized action)
    _tg_request(token, "answerCallbackQuery", {"callback_query_id": query_id}, timeout=5)
    return None


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Simple per-user rate limiter. Prevents API abuse from message spam."""

    def __init__(self, max_per_minute: int = 8):
        self._max = max_per_minute
        self._timestamps: list[float] = []
        self._lock = threading.Lock()

    def allow(self) -> bool:
        """Check if a request is allowed. Returns False if rate limited."""
        now = time.time()
        with self._lock:
            # Remove timestamps older than 60 seconds
            self._timestamps = [t for t in self._timestamps if now - t < 60]
            if len(self._timestamps) >= self._max:
                return False
            self._timestamps.append(now)
            return True


_rate_limiter = _RateLimiter(max_per_minute=8)


# ---------------------------------------------------------------------------
# Photo / Image handling
# ---------------------------------------------------------------------------

def _extract_photo(message: dict, token: str) -> dict | None:
    """Extract photo from a Telegram message, download it, return as base64.

    Telegram sends photos in multiple sizes. We pick the largest one
    (last in the array) for best quality, but cap at ~1MB.

    Returns:
        Dict with {"base64": str, "media_type": str} or None if no photo.
    """
    import base64

    photos = message.get("photo")
    if not photos:
        # Also check for documents that are images
        doc = message.get("document", {})
        if doc.get("mime_type", "").startswith("image/"):
            file_id = doc.get("file_id")
            media_type = doc.get("mime_type", "image/jpeg")
        else:
            return None
    else:
        # Pick the largest photo (last in array), but not too large
        # Telegram provides sizes like 90px, 320px, 800px, 1280px
        best = photos[-1]
        # If file is >1MB, use a smaller version
        if best.get("file_size", 0) > 1_000_000 and len(photos) > 1:
            best = photos[-2]
        file_id = best.get("file_id")
        media_type = "image/jpeg"

    if not file_id:
        return None

    # Step 1: Get file path from Telegram
    file_info = _tg_request(token, "getFile", {"file_id": file_id}, timeout=10)
    if not file_info:
        log.warning("Failed to get file info for photo")
        return None

    file_path = file_info.get("file_path", "")
    if not file_path:
        return None

    # Step 2: Download the file
    download_url = f"https://api.telegram.org/file/bot{token}/{file_path}"
    try:
        req = urllib.request.Request(download_url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            image_bytes = resp.read()
    except Exception as e:
        log.warning("Failed to download photo: %s", e)
        return None

    # Detect media type from file extension
    if file_path.endswith(".png"):
        media_type = "image/png"
    elif file_path.endswith(".webp"):
        media_type = "image/webp"
    elif file_path.endswith(".gif"):
        media_type = "image/gif"

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    log.info("Photo downloaded: %d bytes, %s", len(image_bytes), media_type)

    return {"base64": b64, "media_type": media_type}


def _call_claude_with_image(user_text: str, photo: dict,
                            conversation: ConversationManager,
                            token: str, chat_id: str) -> str:
    """Call Claude with an image (vision) + text. Returns HTML-formatted response.

    Uses the same system prompt and tools as regular calls, but includes
    the image in the user message content.
    """
    import anthropic

    api_key = _get_env("ANTHROPIC_API_KEY")
    if not api_key:
        return "ANTHROPIC_API_KEY not configured."

    client = anthropic.Anthropic(api_key=api_key)
    system_prompt = _build_telegram_prompt()

    # Build multimodal user message: image + text
    user_content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": photo["media_type"],
                "data": photo["base64"],
            },
        },
        {
            "type": "text",
            "text": user_text,
        },
    ]

    conversation.add_user_message(f"[Photo sent] {user_text}")
    messages = conversation.get_messages()

    # Replace the last message's content with the multimodal version
    # (conversation stores text-only, but we send multimodal to Claude)
    api_messages = list(messages)
    if api_messages and api_messages[-1]["role"] == "user":
        api_messages[-1] = {"role": "user", "content": user_content}

    # Always use Sonnet for vision (Haiku's vision is weaker)
    model = _MODEL_SONNET
    status_msg_id = _send_status(token, chat_id,
                                  "\u23f3 <i>Analyzing image...</i>")
    all_tool_results = []
    used_any_tools = False

    try:
        for round_num in range(MAX_TOOL_ROUNDS):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=[{
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }],
                    tools=TOOL_DEFINITIONS,
                    messages=api_messages,
                )
            except anthropic.APIError as e:
                log.error("Claude vision API error: %s", e)
                _delete_message(token, chat_id, status_msg_id)
                return "Something went wrong analyzing the image. Try again."
            except Exception as e:
                log.error("Claude vision call failed: %s", e)
                _delete_message(token, chat_id, status_msg_id)
                return "Something went wrong. Try again."

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

            if response.stop_reason != "tool_use" or not tool_uses:
                conversation.add_assistant_message(full_text)
                _delete_message(token, chat_id, status_msg_id)
                checked = _fact_check_response(full_text, all_tool_results, used_any_tools)
                return _md_to_html(checked)

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
                tool_name = tu["name"]
                log.info("Vision tool call: %s(%s)", tool_name, json.dumps(tu["input"])[:100])

                status_text = _TOOL_STATUS.get(tool_name, f"Working on {tool_name}")
                _edit_message_status(token, chat_id, status_msg_id,
                                     f"\u23f3 <i>{status_text}...</i>")

                handler = TOOL_HANDLERS.get(tool_name)
                if handler:
                    try:
                        result_str = handler(tu["input"])
                        result_str = _truncate_tool_result(result_str)
                        used_any_tools = True
                        all_tool_results.append(result_str)
                    except Exception as e:
                        log.warning("Tool %s failed: %s", tool_name, e)
                        result_str = json.dumps({"error": str(e)})
                else:
                    result_str = json.dumps({"error": f"Unknown tool: {tool_name}"})

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": result_str,
                })

            # Add to conversation and continue
            conversation.add_tool_exchange(assistant_content, tool_results)
            api_messages = conversation.get_messages()

        _delete_message(token, chat_id, status_msg_id)
        if full_text:
            checked = _fact_check_response(full_text, all_tool_results, used_any_tools)
            return _md_to_html(checked)
        return "Couldn't fully analyze the image. Try asking a specific question about it."

    except Exception as e:
        log.error("Vision processing failed: %s", e)
        _delete_message(token, chat_id, status_msg_id)
        return "Failed to process the image. Try again."


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

    _acquire_lock()
    # Register signal handlers AFTER acquiring lock to avoid the old
    # process's SIGTERM death setting _running=False in the new process
    # during the kill-and-wait window.
    _running = True
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

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

    # Register slash commands with Telegram (appears in / menu)
    _register_commands(token)

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
                "allowed_updates": ["message", "callback_query"],
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

                # --- Handle callback queries (inline button presses) ---
                callback_query = update.get("callback_query")
                if callback_query:
                    cb_chat_id = str(callback_query.get("message", {}).get("chat", {}).get("id", ""))
                    if cb_chat_id != str(chat_id):
                        continue
                    try:
                        cb_result = _handle_callback_query(token, chat_id, callback_query, conversation)
                    except Exception as e:
                        log.exception("Callback handler error: %s", e)
                        query_id = callback_query.get("id", "")
                        if query_id:
                            _tg_request(token, "answerCallbackQuery", {
                                "callback_query_id": query_id,
                                "text": f"Error: {str(e)[:50]}",
                                "show_alert": True,
                            }, timeout=5)
                        cb_result = None
                    if cb_result:
                        # Feed the callback result as a user message to Claude
                        typing = _TypingKeepAlive(token, chat_id)
                        typing.start()
                        try:
                            response_text = _call_claude(cb_result, conversation, token, chat_id)
                        finally:
                            typing.stop()
                        if response_text:
                            _tg_send_message(token, chat_id, response_text)
                    continue

                # --- Handle regular messages ---
                message = update.get("message")
                if not message:
                    continue

                msg_chat_id = str(message.get("chat", {}).get("id", ""))
                if msg_chat_id != str(chat_id):
                    log.warning("Unauthorized message from chat_id=%s", msg_chat_id)
                    continue

                # Handle photos — download and send to Claude with vision
                photo_data = _extract_photo(message, token)
                text = (message.get("text") or message.get("caption") or "").strip()
                if not text and not photo_data:
                    continue

                user_name = message.get("from", {}).get("first_name", "User")
                log.info("Message from %s: %s", user_name, text[:100])

                # Rate limiting
                if not _rate_limiter.allow():
                    _tg_send_message(token, chat_id,
                                     "<i>Slow down \u2014 max 8 messages per minute.</i>")
                    continue

                # Handle commands
                cmd = text.split()[0].lower() if text.startswith("/") else None
                response_text = None

                # Map persistent reply keyboard button taps to commands
                if not cmd and text in _REPLY_BUTTON_MAP:
                    text = _REPLY_BUTTON_MAP[text]
                    cmd = text.split()[0].lower()

                if cmd == "/start":
                    response_text = _handle_start()
                elif cmd == "/bets":
                    _tg_send_typing(token, chat_id)
                    response_text = _handle_bets()
                elif cmd == "/parlays":
                    _tg_send_typing(token, chat_id)
                    response_text = _handle_parlays()
                elif cmd == "/bankroll":
                    _tg_send_typing(token, chat_id)
                    response_text = _handle_bankroll()
                elif cmd == "/today" or cmd == "/matches":
                    _tg_send_typing(token, chat_id)
                    response_text = _handle_today(token=token, chat_id=chat_id)
                elif cmd == "/match":
                    if _handle_match(token, chat_id):
                        continue  # Match selector sent via inline keyboard
                    response_text = "No matches found."
                elif cmd == "/live":
                    _tg_send_typing(token, chat_id)
                    response_text = _handle_live()
                elif cmd == "/digest":
                    _tg_send_typing(token, chat_id)
                    response_text = _handle_digest()
                elif cmd in ("/wc", "/worldcup"):
                    _tg_send_typing(token, chat_id)
                    response_text = _handle_worldcup()
                elif cmd and cmd.startswith("/player"):
                    _tg_send_typing(token, chat_id)
                    player_name = cmd.replace("/player", "").strip()
                    if not player_name:
                        response_text = "Usage: <code>/player Dzeko</code> — shows all teams a player has played for."
                    else:
                        response_text = _handle_player_lookup(player_name)
                elif cmd == "/summary":
                    _tg_send_typing(token, chat_id)
                    response_text = _handle_summary_menu(token, chat_id)
                    if response_text is None:
                        continue  # Already sent via inline keyboard
                elif cmd == "/league":
                    league_args = text[len("/league"):].strip()
                    response_text = _handle_league(league_args, conversation)
                elif cmd == "/help":
                    response_text = _handle_help()
                elif cmd == "/clear":
                    conversation.clear()
                    response_text = "Conversation cleared."
                else:
                    # Full AI response via Claude — with typing keepalive
                    typing = _TypingKeepAlive(token, chat_id)
                    typing.start()
                    try:
                        if photo_data:
                            # Vision: send image + text to Claude
                            prompt = text or "What do you see in this image? Analyze it in the context of Serie A betting."
                            response_text = _call_claude_with_image(
                                prompt, photo_data, conversation, token, chat_id)
                        else:
                            response_text = _call_claude(text, conversation, token, chat_id)
                    finally:
                        typing.stop()

                # Send response (always attach persistent keyboard)
                if response_text:
                    success = _tg_send_message(token, chat_id, response_text,
                                               reply_markup=_reply_keyboard())
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
    if "--wc-digest" in sys.argv:
        sys.exit(0 if send_worldcup_digest() else 1)
    run_bot()
