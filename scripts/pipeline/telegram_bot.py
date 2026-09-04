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
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

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
    "🪜 Ladder": "/ladder",
    "🎲 Risk": "/ladder risk",
    "🎫 My bets": "/mybets",
    "💰 Balance": "/balance",
    "⚽ XI": "/xi",
    "🆚 Sfide": "/sfide",
    "🎯 Picks": "/picks",
    "📸 Formazioni": "/formazioni",
}


def _reply_keyboard() -> dict:
    """Button grid, COLLAPSED by default (Nicola 2026-06-12: the old
    persistent 9-button grid ate half the screen; full removal killed the
    toggle too). one_time_keyboard + not persistent = the grid hides after
    use and lives behind the keyboard toggle icon in the input bar — pops
    up only when summoned. Slash commands also sit in the left ☰ menu.
    2026-09-03: WC-era buttons replaced by the Serie A + fantacalcio set."""
    return {
        "keyboard": [
            [{"text": "📸 Formazioni"}, {"text": "⚽ XI"}, {"text": "🆚 Sfide"}],
            [{"text": "🎯 Picks"}, {"text": "⚽ Today"}, {"text": "💰 Bets"}],
        ],
        "resize_keyboard": True,
        "one_time_keyboard": True,
        "is_persistent": False,
    }


# WC-season command menu — populates Telegram's ☰ menu button at input-left.
_MENU_COMMANDS = [
    {"command": "formazioni", "description": "📸 Manda la formazione avversaria"},
    {"command": "xi", "description": "⚽ Formazione fantacalcio consigliata"},
    {"command": "sfide", "description": "🆚 Pronostico H2H prossimi avversari"},
    {"command": "picks", "description": "🎯 Miglior angolo per ogni partita"},
    {"command": "today", "description": "📅 Partite di oggi + previsioni"},
    {"command": "bets", "description": "🎫 Value bet nello slip"},
    {"command": "live", "description": "🔴 Risultati live"},
    {"command": "bankroll", "description": "💰 Bilancio, ROI, streak"},
    {"command": "player", "description": "👤 Scheda giocatore: /player Dzeko"},
    {"command": "digest", "description": "📰 Riassunto del giorno"},
    {"command": "help", "description": "❓ Tutti i comandi"},
]


def _register_commands(token: str):
    """Register the command menu → Telegram shows it in the left ☰ button."""
    result = _tg_request(token, "setMyCommands", {"commands": _MENU_COMMANDS}, timeout=10)
    if result is not None:
        log.info("Registered %d commands in the menu button", len(_MENU_COMMANDS))
    else:
        log.warning("Failed to register command menu")


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

# advisor.py uses load_json_safe; expose it under the legacy _load_json name
# that telegram_bot was originally written against.
from web.advisor import (
    TOOL_DEFINITIONS,
    TOOL_HANDLERS,
    _build_system_prompt,
    _tool_get_bankroll_status,
    _tool_get_live_matches,
    _tool_get_value_bets,
)

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


# Fantacalcio vision tool — opponent lineups read from a screenshot get
# valued with the real levels machinery, never estimated by eye.
_FANTA_XI_TOOL = {
    "name": "score_opponent_xi",
    "description": (
        "Value an opponent's actually-fielded fantacalcio XI (names read "
        "from a Leghe screenshot) against Nicola's current board: their "
        "expected total from live levels, his P(win) vs THIS exact lineup, "
        "and whether a tilted module/XI of his beats the base against it. "
        "ALWAYS call this after reading an opponent lineup from a photo — "
        "never estimate totals by eye. Pass the bench too when visible."),
    "input_schema": {
        "type": "object",
        "properties": {
            "team": {"type": "string",
                     "description": "League team name as shown"},
            "players": {"type": "array", "items": {"type": "string"},
                        "description": "The 11 fielded names, as written"},
            "module": {"type": "string",
                       "description": "Module like 3-4-3 if visible"},
            "bench": {"type": "array", "items": {"type": "string"},
                      "description": "Bench names IN LISTED ORDER if visible"},
        },
        "required": ["team", "players"],
    },
}


def _tool_score_opponent_xi(tool_input: dict) -> str:
    from scripts.fantacalcio.xi_advisor import score_observed_xi
    res = score_observed_xi(tool_input.get("team", ""),
                            tool_input.get("players") or [],
                            module=tool_input.get("module"),
                            bench_names=tool_input.get("bench"))
    return json.dumps(res, ensure_ascii=False, default=str)


_TG_TOOLS = list(TOOL_DEFINITIONS) + [_FANTA_XI_TOOL]
_TG_TOOL_HANDLERS = {**TOOL_HANDLERS, "score_opponent_xi": _tool_score_opponent_xi}


def _fantacalcio_context() -> str:
    """Compact fantacalcio block for the chat/vision prompt — current advice,
    rival matrix, my roster. Lets a Leghe screenshot sent ~1h before kickoff
    get answered with the exact formation to field. Best-effort: any missing
    artifact just shrinks the block."""
    import json as _json
    from pathlib import Path as _Path
    base = _Path(__file__).resolve().parents[2] / "data" / "fantacalcio"
    lines = ["", "FANTACALCIO (la mia lega — Whisky Palermo):"]
    try:
        adv = _json.loads((base / "xi_advice.json").read_text())
        xi = ", ".join(f"{x['R']} {x['nome']}" for x in adv.get("xi", []))
        bench = ", ".join(x["nome"] for x in adv.get("bench", []))
        lines.append(f"Giornata {adv.get('round')} consigliata: "
                     f"{adv.get('module')} attesi {adv.get('total')}")
        lines.append(f"XI: {xi}")
        lines.append(f"Panchina (in ordine): {bench}")
    except (OSError, ValueError):
        pass
    try:
        riv = _json.loads((base / "rivals.json").read_text())
        rl = riv.get("rivals") or []
        if isinstance(rl, dict):
            rl = list(rl.values())
        lines.append("Sfide: " + "; ".join(
            f"{r['team']} {r.get('module')} attesi {r.get('total')} "
            f"p(vittoria mia) {r.get('p_win'):.0%}" for r in rl))
    except (OSError, ValueError, TypeError):
        pass
    lines.append(
        "Se arriva una FOTO di una schermata Leghe Fantacalcio: estrai "
        "squadra, modulo, gli 11 titolari e la panchina IN ORDINE. Se è "
        "l'XI AVVERSARIO, CHIAMA SEMPRE il tool score_opponent_xi con "
        "quei dati — mai stimare i totali a occhio — e rispondi coi suoi "
        "numeri: atteso loro, atteso mio, P(vittoria), e se il campo 'alt' "
        "propone un cambio modulo/XI dillo come mossa concreta (dentro X, "
        "fuori Y). Poi la formazione ESATTA da schierare (modulo + 11 + "
        "ordine panchina) e chiudi con 'Modulo avversario osservato: "
        "<squadra> <modulo>' su una riga a sé. Se è la MIA schermata di "
        "inserimento, confrontala con l'XI consigliato sopra e correggi.")
    return "\n".join(lines)


def _build_telegram_prompt() -> str:
    """Build system prompt with Telegram-specific addon."""
    return _build_system_prompt() + _TELEGRAM_SYSTEM_ADDON + _fantacalcio_context()


# Models — use the same routing logic but simplified for Telegram
_MODEL_SONNET = "claude-sonnet-5"
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


_AI_USAGE_FILE = PROJECT_ROOT / "data" / "monitoring" / "tg_ai_usage.json"
_AI_DAILY_CALLS = int(os.environ.get("TG_AI_DAILY_CALLS", "300"))


def _ai_budget_ok() -> bool:
    """Daily call cap on the bot's Claude usage — a runaway loop or a
    flooded chat must not run an unbounded API bill. 300 calls/day is far
    above any human usage; env TG_AI_DAILY_CALLS overrides."""
    from datetime import date
    today = date.today().isoformat()
    try:
        st = json.loads(_AI_USAGE_FILE.read_text())
    except (OSError, ValueError):
        st = {}
    if st.get("date") != today:
        st = {"date": today, "calls": 0}
    if st["calls"] >= _AI_DAILY_CALLS:
        return False
    st["calls"] += 1
    try:
        _AI_USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _AI_USAGE_FILE.write_text(json.dumps(st))
    except OSError:
        pass
    return True


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

    if not _ai_budget_ok():
        return ("Limite giornaliero AI raggiunto "
                f"({_AI_DAILY_CALLS} chiamate). Riparte a mezzanotte.")

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
                tools=_TG_TOOLS,
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

            handler = _TG_TOOL_HANDLERS.get(tool_name)
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
    from scripts.pipeline.notify import TgMsg, _bankroll_in_context, _get_bankroll_context, _html_escape

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
        growth = result.get("bankroll_growth_pct", 0)
        peak = result.get("peak_bankroll", current)
        streak = result.get("current_streak", 0)

        tg = TgMsg()
        tg.raw("\U0001f4b3 <b>Bankroll</b>")
        tg.blank()

        tg.raw(f"   Balance: <b>\u20ac{current:,.2f}</b>" if isinstance(current, (int, float)) else f"   Balance: {current}")
        tg.raw(f"   Started: \u20ac{initial:,.0f}" if isinstance(initial, (int, float)) else "")
        tg.raw(f"   ROI on stake: <b>{roi:+.1f}%</b>" if roi else "")
        tg.raw(f"   Growth: {growth:+.1f}%" if growth else "")
        tg.raw(f"   Peak: \u20ac{peak:,.0f}" if isinstance(peak, (int, float)) else "")
        tg.blank()

        if streak:
            if streak > 0:
                tg.raw(f"   \U0001f525 {streak}-bet winning streak")
            else:
                tg.raw(f"   \u2744\ufe0f {abs(streak)}-bet losing streak")

        # Drawdown — from the payload, not recomputed
        dd = result.get("drawdown_pct", 0) or 0
        if isinstance(dd, (int, float)) and dd > 2:
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
            from pathlib import Path as _P

            import pandas as pd
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
    from scripts.pipeline.notify import TgMsg

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


def _fanta_json(name: str) -> dict:
    try:
        return json.loads((PROJECT_ROOT / "data" / "fantacalcio"
                           / name).read_text())
    except (OSError, ValueError):
        return {}


def _fanta_age_note(iso: str | None) -> str:
    """'aggiornata Xh fa' footer so an on-demand read is honest about age."""
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        age_h = (datetime.now(UTC) - dt).total_seconds() / 3600
        return (f"\n<i>aggiornata {age_h * 60:.0f}min fa</i>" if age_h < 1
                else f"\n<i>aggiornata {age_h:.1f}h fa</i>")
    except (ValueError, TypeError):
        return ""


def _handle_xi() -> str:
    """/xi — the tracker's latest XI board, on demand (no rebuild: the
    tracker refreshes it 3x/day + T-6h, so serving the file is honest as
    long as the age footer says how old it is)."""
    adv = _fanta_json("xi_advice.json")
    if not adv.get("xi"):
        return ("Nessuna formazione consigliata sul disco — il tracker "
                "non ha ancora costruito la giornata.")
    riv = _fanta_json("rivals.json") or None
    try:
        from scripts.fantacalcio.tracker import render_xi
        _, tg = render_xi(adv, riv)
    except Exception as e:
        log.warning("/xi render failed: %s", e)
        return f"Formazione non renderizzabile: {e}"
    return tg + _fanta_age_note(adv.get("generated_at"))


def _handle_sfide() -> str:
    """/sfide — next-round H2H forecast vs each opponent, both competitions."""
    riv = _fanta_json("rivals.json")
    if not riv.get("next_opponents"):
        return "Nessuna sfida in calendario nel file rivali."
    try:
        from scripts.fantacalcio.tracker import _vs_block
        _, vs_tg, _ = _vs_block(riv, _fanta_json("xi_advice.json") or None)
    except Exception as e:
        log.warning("/sfide render failed: %s", e)
        return f"Sfide non renderizzabili: {e}"
    if not vs_tg:
        return "Nessuna sfida in calendario nel file rivali."
    rnd = riv.get("round")
    head = f"<b>🆚 Sfide giornata {rnd}</b>\n" if rnd else "<b>🆚 Sfide</b>\n"
    return head + vs_tg + _fanta_age_note(riv.get("generated_at"))


def _handle_formazioni() -> str:
    """/formazioni — guided flow: ask for the opponent-lineup screenshot,
    which the photo path then reads and scores with score_opponent_xi."""
    try:
        adv = _fanta_json("xi_advice.json")
        rnd = adv.get("round")
        head = (f"<b>📸 Formazione avversaria — giornata {rnd}</b>\n"
                if rnd else "<b>📸 Formazione avversaria</b>\n")
    except Exception:
        head = "<b>📸 Formazione avversaria</b>\n"
    return (head
            + "Mandami lo screenshot della formazione schierata dal tuo "
              "avversario (Leghe → la sfida → formazioni).\n"
              "La leggo, la valuto coi livelli reali e ti rispondo con: "
              "atteso suo, atteso tuo, P(vittoria) e l'XI giusto da "
              "schierare — cambio modulo incluso se conviene.\n"
              "<i>Includi la panchina nello screenshot se puoi: l'ordine "
              "conta per i cambi automatici.</i>")


def _handle_picks() -> str:
    """/picks — the best-priced angle for EVERY upcoming match, edge-ranked.

    Serves best_picks from the unified bet slip (written by the betting
    engine each run). Advisory: real-money bets remain the edge-gated slip
    ('/bets'); this shows the top credible angle per game even when nothing
    clears the betting bar."""
    try:
        slip = json.loads((PROJECT_ROOT / "data" / "upcoming"
                           / "unified_bet_slip.json").read_text())
    except (OSError, ValueError):
        return "Nessuno slip su disco — il motore scommesse non ha ancora girato."
    picks = slip.get("best_picks") or []
    if not picks:
        return ("Lo slip attuale non ha ancora la sezione best-pick "
                "(arriva col prossimo giro del motore).")
    n_bets = len(slip.get("selected_bets") or [])
    lines = [f"<b>🎯 Miglior angolo per partita</b> "
             f"(bet reali in slip: {n_bets})"]
    for pk in picks[:15]:
        b = pk.get("best") or {}
        flag = "✅" if b.get("in_band") else "⚠️"
        lines.append(
            f"{flag} {pk.get('date', '?')[5:]} <b>{pk.get('match')}</b>\n"
            f"     {b.get('market')} {b.get('selection')} @ {b.get('odds')} "
            f"— edge {b.get('edge_pct', 0):+.1f}%, p={b.get('model_prob', 0):.0%} "
            f"<i>[{b.get('tier')}]</i>")
    if any(not (pk.get("best") or {}).get("in_band") for pk in picks[:15]):
        lines.append("<i>⚠️ = edge fuori banda 2-12%: storicamente "
                     "overconfidence del modello, non value.</i>")
    return "\n".join(lines) + _fanta_age_note(slip.get("generated_at"))


def _handle_worldcup() -> str:
    """World Cup digest: today's slate + the three best-combo tiers.

    Same sources as /worldcup on the dashboard (predictions.json +
    market_odds.json via scripts.worldcup.combos — who-wins markets only).
    Times shown in the WC display timezone (Miami by default).
    """
    from datetime import UTC, datetime

    from scripts.worldcup.combos import (
        MARKET_ODDS_JSON,
        PREDICTIONS_JSON,
        build_best_combos,
        build_fun_combos,
    )
    from scripts.worldcup.engine import read_json_safe

    rome = _wc_tz()
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
        lines.append(f"⚽ <b>Today</b> ({WC_TZ_LABEL} time)")
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
        lines.append("🎯 <b>Edge combos — next 48h</b> (who-wins, model-backed)")
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
        lines.append("🎯 No edge combos — no upcoming matches in the next 48h.")
        lines.append("")

    # Fun combos — goal-prop multiplas like Nicola's played slip. NO model
    # edge (goal props are noise on internationals); labeled as such so they
    # are never mistaken for the edge combos above.
    fun = build_fun_combos(preds)
    if fun.get("combos"):
        lines.append("🎲 <b>Fun combos — no model edge</b> (Over-goals, like a played multipla)")
        for c in fun["combos"]:
            cm = c["combined"]
            lines.append(f"{c['title']} — {cm['prob'] * 100:.0f}% "
                         f"(fair {cm['fair_odds']:.1f}×, {len(c['legs'])} legs)")
            for leg in c["legs"]:
                lines.append(f" • {leg['pick_label']} ({leg['prob'] * 100:.0f}%)"
                             f" — {leg['match']}")
            lines.append("")
        lines.append("<i>🎲 Fun = entertainment only. Goal props don't beat the "
                     "book on internationals — bet these for fun, not for edge.</i>")

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
    from config.leagues import LEAGUE_REGISTRY
    from scripts.pipeline.notify import TgMsg, _html_escape

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
    """Handle /help — all available commands, grouped by what they serve."""
    from scripts.pipeline.notify import TgMsg

    tg = TgMsg()
    tg.raw("<b>SerieAI Commands</b>")
    tg.blank()
    tg.raw("<b>Fantacalcio:</b>")
    tg.raw("  /xi \u2014 formazione consigliata della giornata")
    tg.raw("  /sfide \u2014 pronostico H2H vs i prossimi avversari")
    tg.raw("  /formazioni \u2014 foto della formazione avversaria \u2192 XI corretto")
    tg.blank()
    tg.raw("<b>Serie A betting:</b>")
    tg.raw("  /picks \u2014 miglior angolo per OGNI partita (tutti i mercati)")
    tg.raw("  /bets \u2014 value bet nello slip (edge-gated, soldi veri)")
    tg.raw("  /today \u2014 partite di oggi + previsioni")
    tg.raw("  /match \u2014 tocca una partita per l'analisi completa")
    tg.raw("  /live \u2014 risultati live + le tue bet")
    tg.raw("  /bankroll \u2014 bilancio, ROI, streak")
    tg.raw("  /player \u2014 scheda giocatore: /player Dzeko")
    tg.raw("  /digest \u2014 riassunto del giorno")
    tg.raw("  /league \u2014 filtro per lega (EPL, Serie A)")
    tg.raw("  /fill \u2014 conferma una giocata (/fill 2 1.95)")
    tg.blank()
    tg.raw("<b>🤖 AI chat \u2014 scrivimi e basta:</b>")
    tg.italic("Niente comando: qualsiasi messaggio va all'AI con accesso a")
    tg.italic("previsioni, quote, bankroll, storico 21 stagioni, fantacalcio.")
    tg.italic("Es: 'analizza Genoa-Como', 'come sta andando il bankroll?',")
    tg.italic("'chi schiero in porta?' \u2014 anche screenshot di formazioni.")
    tg.blank()
    tg.raw("<b>Legacy World Cup:</b> /wc /ladder /mybets /bet /settle /balance /guard /lossstop")
    tg.raw("<b>Session:</b> /clear \u2014 reset conversazione")
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


def _resolve_ticket_num(num: str):
    """Resolve a T-30 ticket line number to (bet_id, match).

    The order ticket's \u2713/\u2717 buttons and /fill carry the day-unique
    line number, mapped to bet_ids in data/pipeline/t30_ticket_state.json.
    Returns (None, reason) for a stale/unknown number -- old buttons tapped
    on a later day must fail safely, never touch the wrong bet.
    """
    import json as _json
    from datetime import datetime as _dt
    from pathlib import Path as _Path
    marker = _Path(__file__).parent.parent.parent / "data" / "pipeline" / "t30_ticket_state.json"
    try:
        st = _json.loads(marker.read_text())
    except (OSError, ValueError):
        return None, "No ticket on record today."
    if st.get("date") != _dt.now().strftime("%Y-%m-%d"):
        return None, "That ticket expired \u2014 numbers reset daily."
    entry = (st.get("bets") or {}).get(str(num))
    if isinstance(entry, dict) and entry.get("bet_id"):
        return entry["bet_id"], entry.get("match", "?")
    return None, f"No ticket line {num} today."


def _record_fill(num: str, placed: bool, odds: float | None = None) -> str:
    """Mark a ticket line placed/missed on its EXISTING journal row.

    Annotation only -- mark_bet_fill never creates rows, never touches
    stake/odds/status. Returns the human confirmation string.
    """
    bet_id, info = _resolve_ticket_num(num)
    if not bet_id:
        return info
    try:
        from scripts.betting.bet_journal import mark_bet_fill
        r = mark_bet_fill(bet_id, "placed" if placed else "missed",
                          filled_odds=odds)
    except (OSError, ValueError, KeyError) as e:
        log.warning("Fill record failed for %s: %s", bet_id, e)
        return "Could not write the fill \u2014 try again or check the journal."
    if not r.get("ok"):
        return r.get("error", "Fill not recorded.")
    if placed:
        at = r.get("filled_odds")
        return (f"\u2713 {num}\u00b7 placed @ {at:.2f} \u2014 {info}" if at
                else f"\u2713 {num}\u00b7 placed \u2014 {info}")
    return f"\u2717 {num}\u00b7 missed \u2014 {info}. Stays in the journal, drops from verified ROI."


def _handle_fill_command(text: str) -> str:
    """/fill <n> <odds> -- confirm a ticket line at the price actually got.
    /fill alone lists today's ticket lines and their fill states."""
    parts = text.split()
    if len(parts) >= 2:
        num = parts[1].strip()
        odds = None
        if len(parts) >= 3:
            try:
                odds = float(parts[2].replace(",", "."))
            except ValueError:
                return "Usage: <code>/fill 2 1.95</code> (line number, then odds)"
            if not 1.01 <= odds <= 50:
                return f"Odds {odds} out of range \u2014 expected 1.01\u201350."
        return _record_fill(num, placed=True, odds=odds)
    # Bare /fill: show today's ticket state
    import json as _json
    from datetime import datetime as _dt
    from pathlib import Path as _Path
    marker = _Path(__file__).parent.parent.parent / "data" / "pipeline" / "t30_ticket_state.json"
    try:
        st = _json.loads(marker.read_text())
        assert st.get("date") == _dt.now().strftime("%Y-%m-%d")
        entries = st.get("bets") or {}
        assert entries
    except (OSError, ValueError, AssertionError):
        return "No ticket today. Lines appear here once the T-30 ticket fires."
    try:
        from scripts.betting.bet_journal import get_pending_bets
        rows = {b.get("bet_id"): b for b in get_pending_bets(include_superseded=False)}
    except Exception:
        rows = {}
    lines = ["<b>Today's ticket lines</b>"]
    icon = {"placed": "\u2713", "missed": "\u2717", "unverified": "\u26a0"}
    for num in sorted(entries, key=lambda x: int(x) if x.isdigit() else 0):
        e = entries[num]
        if not isinstance(e, dict):
            continue
        b = rows.get(e.get("bet_id"), {})
        fs = b.get("fill_status")
        tag = f" {icon.get(fs, '')} {fs}" if fs else " \u00b7 unconfirmed"
        sel = b.get("selection", "")
        lines.append(f"{num}\u00b7 {e.get('match', '?')} \u2014 {sel}{tag}")
    lines.append("\nConfirm: tap the ticket buttons, or <code>/fill &lt;n&gt; &lt;odds&gt;</code>.")
    return "\n".join(lines)


def _handle_callback_query(token: str, chat_id: str, callback_query: dict,
                           conversation: ConversationManager) -> str | None:
    """Handle inline keyboard button presses.

    Returns response text, or None if handled internally.
    """
    query_id = callback_query.get("id", "")
    data = callback_query.get("data", "")
    _message_id = callback_query.get("message", {}).get("message_id", 0)  # noqa: F841 — kept for future message edits

    # Default: silent answer to remove loading spinner
    answer_text = ""
    show_alert = False

    if data.startswith("wcbeto:"):
        # One-tap from an odds-sheet scan: leg AND price already known —
        # jump straight to the stake calculation. Verbatim send, never Claude.
        payload = data[len("wcbeto:"):]
        mn, key, odds_s = payload.split("|", 2)
        odds = float(odds_s)
        _tg_request(token, "answerCallbackQuery", {"callback_query_id": query_id},
                    timeout=5)
        p = _wc_pred_for_match(mn)
        label, prob = key, None
        if p and key.startswith("CS:"):
            _, sh, sa = key.split(":")
            lh, la = float(p.get("home_xg") or 0), float(p.get("away_xg") or 0)
            prob = _wc_grid(lh, la).get((int(sh), int(sa))) if lh and la else None
            label = f"Exact score {sh}-{sa}"
        elif p:
            for pr, lbl, k in _wc_ladder_legs(p):
                if k == key:
                    label, prob = lbl, pr
                    break
        pend = {"match_number": int(mn), "leg_key": key, "label": label,
                "prob": prob, "odds": odds,
                "match": f"{p.get('home_team')} vs {p.get('away_team')}" if p else f"match {mn}"}
        suggested, msg = _wc_stake_suggestion(odds, prob)
        pend["suggested"] = suggested
        pend["at"] = time.time()
        _WC_PENDING_BET[chat_id] = pend
        _tg_send_message(token, chat_id,
                         f"🎫 <b>{label}</b> @ {odds} ({pend['match']})\n{msg}")
        return None

    if data.startswith("wcbet:"):
        # Money flow: send VERBATIM and return None — a returned string gets
        # fed to Claude as a user message (the analyze-button pattern) and
        # comes back paraphrased/footered/refused. Found live 2026-06-12.
        _, mn, key = data.split(":", 2)
        _tg_request(token, "answerCallbackQuery", {"callback_query_id": query_id},
                    timeout=5)
        if key == "other":
            _WC_PENDING_BET[chat_id] = {"match_number": int(mn), "leg_key": None,
                                        "label": None, "at": time.time()}
            _tg_send_message(token, chat_id,
                "📝 Type the bet as: <code>STAKE @ ODDS description</code>\n"
                "e.g. <code>60 @ 1.80 Canada win</code>\n"
                "(free-text legs settle via /settle won|lost · /cancel to abort)")
            return None
        legs = {}
        match_name = f"match {mn}"
        p = _wc_pred_for_match(mn)
        if p:
            legs = {k: (lbl, pr) for pr, lbl, k in _wc_ladder_legs(p)}
            match_name = f"{p.get('home_team')} vs {p.get('away_team')}"
        label, prob = legs.get(key, (key, None))
        _WC_PENDING_BET[chat_id] = {"match_number": int(mn), "leg_key": key,
                                    "label": label, "match": match_name,
                                    "prob": prob, "at": time.time()}
        _tg_send_message(token, chat_id,
            f"🎫 Logging <b>{label}</b> ({match_name}).\n"
            f"Send the SISAL odds (e.g. <code>1.80</code>) and I'll calculate "
            f"the stake — or stake and odds together: <code>60 @ 1.80</code> (/cancel)")
        return None

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
        # RETIRED 2026-08-27. This button used to call add_bet() with stake 0,
        # an empty date, and a GUESSED market \u2014 writing a duplicate junk row
        # into bet_journal.json (the ledger source of truth) from a phone tap.
        # The T-30 chain journals the real bet; the order ticket is the record.
        # Handler kept only so taps on old messages fail safely.
        _tg_request(token, "answerCallbackQuery", {
            "callback_query_id": query_id,
            "text": "Button retired \u2014 the T-30 order ticket is the record.",
            "show_alert": True,
        }, timeout=5)
        return None

    if data.startswith("fill:") or data.startswith("miss:"):
        placed = data.startswith("fill:")
        num = data.split(":", 1)[1]
        result = _record_fill(num, placed=placed)
        _tg_request(token, "answerCallbackQuery", {
            "callback_query_id": query_id,
            "text": result[:190],
            "show_alert": not result.startswith(("\u2713", "\u2717")),
        }, timeout=5)
        # Confirmations land in the chat too -- the ticket thread is the record.
        return result

    if data.startswith("skip:"):
        parts = data[len("skip:"):].split("|")
        match = parts[0] if parts else "?"
        selection = parts[1] if len(parts) > 1 else "?"
        _tg_request(token, "answerCallbackQuery", {
            "callback_query_id": query_id,
            "text": f"Skipped {selection}",
        }, timeout=5)
        return f"Noted \u2014 skipped {match} {selection}."

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
            report_path = _data / "upcoming" / "unified_bet_slip.json"
            if report_path.exists():
                report = _json.load(open(report_path))
                for b in report.get("selected_bets", []):
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


def _record_observed_module(text: str) -> None:
    """Auto-record 'Modulo avversario osservato: <squadra> <modulo>' lines
    from vision replies into the rival-modules ledger. Best-effort."""
    try:
        import re as _re
        m = _re.search(r"Modulo avversario osservato:\s*(.+?)\s+(\d-\d-\d)",
                       text)
        if not m:
            return
        import json as _json
        from pathlib import Path as _Path

        from scripts.fantacalcio.xi_advisor import record_fielded
        base = _Path(__file__).resolve().parents[2] / "data" / "fantacalcio"
        rnd = _json.loads((base / "xi_advice.json").read_text()).get("round")
        if rnd:
            record_fielded(m.group(1).strip(), m.group(2), int(rnd))
            log.info("Recorded observed module: %s %s (round %s)",
                     m.group(1), m.group(2), rnd)
            # Rebuild the rival matrix on the spot so the observation feeds
            # the next answer/push instead of waiting for the tracker run.
            from scripts.fantacalcio.xi_advisor import build_rivals
            riv = build_rivals()
            (base / "rivals.json").write_text(
                _json.dumps(riv, ensure_ascii=False, indent=1))
            log.info("rivals.json rebuilt with the observed module")
    except Exception as e:
        log.warning("observed-module record failed: %s", e)


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

    if not _ai_budget_ok():
        return ("Limite giornaliero AI raggiunto "
                f"({_AI_DAILY_CALLS} chiamate). Riparte a mezzanotte.")

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
                    tools=_TG_TOOLS,
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
                _record_observed_module(full_text)
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

                handler = _TG_TOOL_HANDLERS.get(tool_name)
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


# =============================================================================
# WC BET TRACKER — Nicola's manual SISAL bets, logged via buttons/text,
# auto-settled from real results, bankroll-aware (added 2026-06-12)
# =============================================================================
# Display timezone for all WC times (Nicola is in Miami for the tournament).
# Override via WC_DISPLAY_TZ / WC_TZ_LABEL in .env if he travels.
WC_DISPLAY_TZ = os.environ.get("WC_DISPLAY_TZ", "America/New_York")
WC_TZ_LABEL = os.environ.get("WC_TZ_LABEL", "Miami")


def _wc_tz():
    from zoneinfo import ZoneInfo
    return ZoneInfo(WC_DISPLAY_TZ)


WC_MYBETS_JSON = PROJECT_ROOT / "data" / "worldcup" / "my_bets.json"
WC_BANKROLL_JSON = PROJECT_ROOT / "data" / "worldcup" / "my_bankroll.json"
WC_RUNG_FRACTION = 0.47          # ladder rung ≈ this share of balance, floor keeps the rest

# ── Responsible-betting guardrail (added 2026-06-15 after a EUR50 8-leg
# parlay on Brazil Serie B lost on one leg; EUR151 deposited -> EUR0.40).
# The bot REFUSES to log bets that violate these. Override per-bet only with
# an explicit "force" word; change the limits here.
WC_GUARD = {
    "max_parlay_legs": 3,        # no accumulators beyond this (8-folds are -EV by construction)
    "max_stake_pct": 0.50,       # one bet ≤ 50% of current balance
    "wc_matches_only": True,     # refuse legs the model can't price (non-WC leagues)
    "loss_stop": True,           # hard stop when down past the limit below
}
_WC_PENDING_BET: dict = {}       # chat_id -> {"match_number", "leg_key", "label", "match"}
_WC_SETTLE_LAST = {"t": 0.0}

# Structured legs the bot can settle ALONE from a final score (h, a).
WC_LEG_EVAL = {
    "1":      lambda h, a: h > a,
    "X":      lambda h, a: h == a,
    "2":      lambda h, a: a > h,
    "1X":     lambda h, a: h >= a,
    "X2":     lambda h, a: a >= h,
    "12":     lambda h, a: h != a,
    "O15":    lambda h, a: h + a > 1.5,
    "U25":    lambda h, a: h + a < 2.5,
    "O25":    lambda h, a: h + a > 2.5,
    "U35":    lambda h, a: h + a < 3.5,
    "BTTSY":  lambda h, a: h > 0 and a > 0,
    "BTTSN":  lambda h, a: h == 0 or a == 0,
    "1U35":   lambda h, a: h > a and h + a < 3.5,
    "1O15":   lambda h, a: h > a and h + a > 1.5,
    "2U35":   lambda h, a: a > h and h + a < 3.5,
    "1orO25": lambda h, a: h > a or h + a > 2.5,
}

# Half-based markets — keyed on (ht_h, ht_a) for first-half legs, or on BOTH
# halves for combo legs. Evaluated by _wc_eval_leg only when HT data exists;
# otherwise the leg is undecidable (returns None) and flagged for manual entry.
#   ht  = (home, away) goals at half-time
#   fh  = first-half goals only; sh = second-half = ft - ht
WC_HALF_EVAL = {
    # First-half totals/results (1T)
    "1T_O05": lambda ht, sh: ht[0] + ht[1] > 0.5,
    "1T_O15": lambda ht, sh: ht[0] + ht[1] > 1.5,
    "1T_U15": lambda ht, sh: ht[0] + ht[1] < 1.5,
    "1T_1":   lambda ht, sh: ht[0] > ht[1],
    "1T_X":   lambda ht, sh: ht[0] == ht[1],
    "1T_2":   lambda ht, sh: ht[1] > ht[0],
    # Second-half totals (2T) — second-half goals = full-time minus half-time
    "2T_O05": lambda ht, sh: sh[0] + sh[1] > 0.5,
    "2T_O15": lambda ht, sh: sh[0] + sh[1] > 1.5,
    "2T_U05": lambda ht, sh: sh[0] + sh[1] < 0.5,
    # Combo: the leg Nicola played — Over 1.5 in 1T AND Over 0.5 in 2T.
    "1TO15_2TO05": lambda ht, sh: (ht[0] + ht[1] > 1.5) and (sh[0] + sh[1] > 0.5),
}


def _wc_eval_leg(leg: dict, ft: tuple[int, int] | None,
                 ht: tuple[int, int] | None) -> bool | None:
    """Evaluate ONE parlay/structured leg against a final score (ft) and an
    optional half-time score (ht).

    Returns True (won), False (lost), or None (undecidable — settle manually
    or wait). Undecidable when:
      • feed != "wc" (e.g. a Brazilian leg with no result source), or
      • the full-time score is missing, or
      • the leg is half-based but no HT score is available (the defensive,
        unverified HT path — degrades to manual, never guesses).
    """
    if leg.get("feed") and leg["feed"] != "wc":
        return None  # manual feed — only a human (or /leg) settles it
    key = leg.get("leg_key")
    if not key:
        return None
    # Exact-score legs: "CS:h:a"
    if isinstance(key, str) and key.startswith("CS:"):
        if ft is None:
            return None
        _, sh, sa = key.split(":")
        return ft[0] == int(sh) and ft[1] == int(sa)
    # Half-based legs need HT (and full-time, to derive the second half).
    if key in WC_HALF_EVAL:
        if ft is None or ht is None:
            return None
        second = (ft[0] - ht[0], ft[1] - ht[1])
        return bool(WC_HALF_EVAL[key](ht, second))
    # Full-time legs.
    if key in WC_LEG_EVAL:
        if ft is None:
            return None
        return bool(WC_LEG_EVAL[key](ft[0], ft[1]))
    return None  # unknown market — manual


def _wc_json_load(path: Path, default):
    import json as _json
    try:
        return _json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _wc_json_save(path: Path, data) -> None:
    import json as _json
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(_json.dumps(data, indent=1))
    tmp.replace(path)


def _wc_bankroll() -> dict:
    return _wc_json_load(WC_BANKROLL_JSON, {"balance": None, "history": []})


def _wc_bankroll_apply(delta: float, note: str) -> dict:
    bk = _wc_bankroll()
    if bk.get("balance") is None:
        return bk
    bk["balance"] = round(bk["balance"] + delta, 2)
    bk["history"] = (bk.get("history") or [])[-49:] + [
        {"at": datetime.now(UTC).isoformat(), "delta": round(delta, 2), "note": note}]
    _wc_json_save(WC_BANKROLL_JSON, bk)
    return bk


def _wc_guard_check(stake: float, n_legs: int = 1, is_wc: bool = True,
                    force: bool = False) -> str | None:
    """Return a refusal message if this bet violates the guardrail, else None.
    `force` (user typed 'force') bypasses the soft limits — but the loss-stop
    is never silently bypassable: it nudges to /deposit or a cool-off."""
    if force:
        return None
    bk = _wc_bankroll()
    bal = bk.get("balance")
    # Loss-stop: down past the limit → hard stop.
    if WC_GUARD["loss_stop"] and bk.get("deposited"):
        limit = bk.get("loss_stop_eur")
        net = (bal or 0) - bk["deposited"]
        if limit and net <= -abs(limit):
            return (f"🛑 <b>Loss-stop hit.</b> You're down €{abs(net):.2f} of your "
                    f"€{abs(limit):.0f} limit. No more bets this session — that was "
                    f"the deal you set. Take a break; the model's calls keep grading "
                    f"on /mybets with no money at risk.")
    if WC_GUARD["wc_matches_only"] and not is_wc:
        return ("🚫 That's not a World Cup match — the model can't price it, so I "
                "won't log it. WC 2026 fixtures only (the Brazil Série B legs are "
                "exactly the blind bets that cost €150). Type <b>force</b> to override.")
    if n_legs > WC_GUARD["max_parlay_legs"]:
        return (f"🚫 <b>{n_legs}-leg parlay blocked.</b> Max {WC_GUARD['max_parlay_legs']} "
                f"legs — an 8-fold is ~6% to hit and negative-EV by construction "
                f"(it's how you lost €50 on one leg). Singles or ≤3-leg combos. "
                f"Type <b>force</b> to override.")
    if bal and stake > bal * WC_GUARD["max_stake_pct"] + 0.01:
        cap = bal * WC_GUARD["max_stake_pct"]
        return (f"🚫 €{stake:.2f} is over half your €{bal:.2f} balance. Max bet right "
                f"now: €{cap:.2f}. One bet shouldn't be able to halve you. "
                f"Type <b>force</b> to override.")
    return None


def _wc_log_bet(match_number, match: str, leg_key: str | None, label: str,
                stake: float, odds: float) -> dict:
    bets = _wc_json_load(WC_MYBETS_JSON, [])
    bet = {
        "id": len(bets) + 1, "match_number": match_number, "match": match,
        "leg_key": leg_key, "label": label, "stake": round(stake, 2),
        "odds": round(odds, 2), "placed_at": datetime.now(UTC).isoformat(),
        "status": "open",
    }
    bets.append(bet)
    _wc_json_save(WC_MYBETS_JSON, bets)
    _wc_bankroll_apply(-stake, f"bet #{bet['id']} {label} @ {odds}")
    return bet


def _wc_result_for_match(match_number) -> tuple[int, int] | None:
    """Final score for a fixture, via fixtures.json names ↔ same-night results."""
    fx = _wc_json_load(PROJECT_ROOT / "data" / "worldcup" / "fixtures.json", {})
    fixtures = fx.get("fixtures", fx) if isinstance(fx, dict) else fx
    home = away = None
    for f in fixtures or []:
        if f.get("match_number") == int(match_number):
            home, away = f.get("home"), f.get("away")
            break
    if not home:
        return None
    res = _wc_json_load(PROJECT_ROOT / "data" / "worldcup" / "sofascore_results.json", {})
    rows = res.get("results", res) if isinstance(res, dict) else res
    for r in rows or []:
        if r.get("home") == home and r.get("away") == away:
            return int(r["home_score"]), int(r["away_score"])
        if r.get("home") == away and r.get("away") == home:   # orientation flip
            return int(r["away_score"]), int(r["home_score"])
    return None


def _wc_ht_for_match(match_number) -> tuple[int, int] | None:
    """Half-time score for a fixture, oriented home-away like _wc_result_for_match.

    Returns None when no HT data is stored (the UNVERIFIED HT path — Sofascore
    `period1` may be absent during/after a ban). Half-based legs degrade to
    manual settle in that case; they never settle on a guess."""
    fx = _wc_json_load(PROJECT_ROOT / "data" / "worldcup" / "fixtures.json", {})
    fixtures = fx.get("fixtures", fx) if isinstance(fx, dict) else fx
    home = away = None
    for f in fixtures or []:
        if f.get("match_number") == int(match_number):
            home, away = f.get("home"), f.get("away")
            break
    if not home:
        return None
    res = _wc_json_load(PROJECT_ROOT / "data" / "worldcup" / "sofascore_results.json", {})
    rows = res.get("results", res) if isinstance(res, dict) else res
    for r in rows or []:
        if r.get("ht_home") is None or r.get("ht_away") is None:
            continue
        if r.get("home") == home and r.get("away") == away:
            return int(r["ht_home"]), int(r["ht_away"])
        if r.get("home") == away and r.get("away") == home:   # orientation flip
            return int(r["ht_away"]), int(r["ht_home"])
    return None


def _wc_settle_parlay(b: dict, token: str, chat_id: str) -> bool:
    """Settle one open PARLAY (a bet carrying a `legs` array) leg-by-leg.

    Each leg with feed=="wc" is evaluated as its match produces a result (and
    HT, for half-based legs). Rules:
      • ANY leg lost  → the whole ticket is lost immediately (no need to wait
        for the remaining legs — a parlay needs every leg).
      • ALL legs won  → ticket won, return = stake × odds.
      • otherwise (some legs still open, or undecidable/manual legs pending)
        → ticket stays open; only the per-leg statuses advance.
    Messages each newly-settled leg and the final ticket grade. Returns True
    if anything changed (so the caller persists)."""
    legs = b.get("legs") or []
    changed = False
    for lg in legs:
        if lg.get("status") and lg["status"] != "open":
            continue  # already settled (auto or via /leg)
        mn = lg.get("match_number")
        ft = _wc_result_for_match(mn) if mn is not None else None
        ht = _wc_ht_for_match(mn) if mn is not None else None
        verdict = _wc_eval_leg(lg, ft, ht)
        if verdict is None:
            continue  # undecidable yet (no result / no HT / manual feed)
        lg["status"] = "won" if verdict else "lost"
        if ft is not None:
            lg["score"] = f"{ft[0]}-{ft[1]}"
        lg["settled_at"] = datetime.now(UTC).isoformat()
        changed = True
        _tg_send_message(token, chat_id,
            f"{'✅' if verdict else '❌'} <b>Leg {lg.get('n', '?')} "
            f"{'won' if verdict else 'LOST'}</b> — {lg.get('match', '?')}: "
            f"{lg.get('label') or lg.get('leg_key')} "
            + (f"({lg['score']})" if lg.get("score") else ""))

    statuses = [lg.get("status", "open") for lg in legs]
    manual_pending = any(
        (lg.get("feed") == "manual") and lg.get("status", "open") == "open"
        for lg in legs)

    if "lost" in statuses:
        # Early-loss: ticket dead the moment one leg loses.
        b["status"] = "lost"
        b["settled_at"] = datetime.now(UTC).isoformat()
        bk = _wc_bankroll()
        dead = next(lg for lg in legs if lg.get("status") == "lost")
        _tg_send_message(token, chat_id,
            f"❌ <b>Parlay LOST</b> — {b['label']} × €{b['stake']:.2f}"
            f"\nKilled by leg {dead.get('n', '?')}: {dead.get('match', '?')}"
            + (f"\n💰 Balance: <b>€{bk['balance']:.2f}</b> · {_wc_net_line()}"
               if bk.get("balance") is not None else ""))
        return True
    if statuses and all(s == "won" for s in statuses):
        # Every leg won — pay the full ticket.
        b["status"] = "won"
        b["settled_at"] = datetime.now(UTC).isoformat()
        ret = b["stake"] * b["odds"]
        b["return"] = round(ret, 2)
        bk = _wc_bankroll_apply(ret, f"bet #{b['id']} parlay WON")
        _tg_send_message(token, chat_id,
            f"✅ <b>Parlay WON</b> — {b['label']} @ {b['odds']} × €{b['stake']:.2f}"
            f"\nAll {len(legs)} legs landed → return <b>€{ret:.2f}</b>"
            + (f"\n💰 Balance: <b>€{bk['balance']:.2f}</b> · {_wc_net_line()}"
               if bk.get("balance") is not None else ""))
        return True
    if changed and manual_pending:
        # Some auto legs settled, but manual legs (e.g. BRA) still block the
        # grade — nudge once so the user knows to confirm them.
        n_open = sum(1 for s in statuses if s == "open")
        _tg_send_message(token, chat_id,
            f"⏳ Parlay {b['label']}: {sum(1 for s in statuses if s == 'won')} legs won, "
            f"{n_open} still open — confirm manual legs with "
            f"<code>/leg {b['id']} &lt;n&gt; w|l</code> when they finish.")
    return changed


def _wc_check_my_bets(token: str, chat_id: str) -> None:
    """Settle open structured bets against real results; message each verdict.

    Dispatches per bet: a bet with a `legs` array is a PARLAY (per-leg settle,
    see _wc_settle_parlay); a flat bet with a top-level `leg_key` is a SINGLE
    (the original path, unchanged)."""
    if time.time() - _WC_SETTLE_LAST["t"] < 300:
        return
    _WC_SETTLE_LAST["t"] = time.time()
    try:
        bets = _wc_json_load(WC_MYBETS_JSON, [])
        changed = False
        for b in bets:
            if b.get("status") != "open":
                continue
            if b.get("legs"):
                changed = _wc_settle_parlay(b, token, chat_id) or changed
                continue
            if not b.get("leg_key"):
                continue
            score = _wc_result_for_match(b.get("match_number"))
            if not score:
                continue
            h, a = score
            key = b["leg_key"]
            if key.startswith("CS:"):
                _, sh, sa = key.split(":")
                won = (h == int(sh) and a == int(sa))
            elif key in WC_LEG_EVAL:
                won = WC_LEG_EVAL[key](h, a)
            else:
                continue  # unknown key — leave open for manual /settle
            b["status"] = "won" if won else "lost"
            b["settled_at"] = datetime.now(UTC).isoformat()
            b["score"] = f"{h}-{a}"
            changed = True
            st_l = _wc_ladder_on_settle(b, won)
            if won:
                ret = b["stake"] * b["odds"]
                b["return"] = round(ret, 2)
                bk = _wc_bankroll_apply(ret, f"bet #{b['id']} WON")
                # rung can be None (derived ladder unseeded) — guard the format
                # so a missing rung never aborts the whole settle (NoneType
                # format crash was swallowing the settlement).
                rung = st_l.get("rung")
                rung_txt = (f"next rung: <b>€{rung:.2f}</b> " if rung is not None
                            else "next rung sizes from the floor ")
                _tg_send_message(token, chat_id,
                    f"✅ <b>Bet WON</b> — {b['label']} @ {b['odds']} × €{b['stake']:.2f}"
                    f"\n{b['match']} finished {h}-{a} → return <b>€{ret:.2f}</b>"
                    + (f"\n💰 Balance: <b>€{bk['balance']:.2f}</b> · {_wc_net_line()}" if bk.get("balance") is not None else "")
                    + f"\n🪜 Ladder continues — {rung_txt}"
                      f"(streak {st_l.get('streak', 0)}). Banking is always allowed.")
            else:
                bk = _wc_bankroll()
                _tg_send_message(token, chat_id,
                    f"❌ <b>Bet lost</b> — {b['label']} @ {b['odds']} × €{b['stake']:.2f}"
                    f"\n{b['match']} finished {h}-{a}"
                    + (f"\n💰 Balance: <b>€{bk['balance']:.2f}</b> · {_wc_net_line()}" if bk.get("balance") is not None else "")
                    + "\n🪜 Ladder reset — next bet sizes from the floor again.")
        if changed:
            _wc_json_save(WC_MYBETS_JSON, bets)
    except Exception as e:  # noqa: BLE001 — settling must never kill the loop
        log.warning("my-bets settle check failed: %s", e)


def _wc_net_line() -> str:
    """Real P/L vs deposits: balance + money riding − total deposited."""
    bk = _wc_bankroll()
    bal, dep = bk.get("balance"), bk.get("deposited")
    if bal is None or not dep:
        return ""
    bets = _wc_json_load(WC_MYBETS_JSON, [])
    riding = sum(b["stake"] for b in bets if b.get("status") == "open")
    net = bal - dep
    riding_txt = f" · €{riding:.0f} riding" if riding else ""
    sign = "🟢 up" if net > 0 else ("🔴 down" if net < 0 else "⚪ even")
    return f"deposited €{dep:.0f}{riding_txt} → {sign} <b>€{abs(net):.2f}</b> real"


def _wc_money_footer() -> str:
    """Personalized footer: balance + real P/L + how the last bet went."""
    bk = _wc_bankroll()
    bets = _wc_json_load(WC_MYBETS_JSON, [])
    parts = []
    if bk.get("balance") is not None:
        parts.append(f"💰 Balance <b>€{bk['balance']:.2f}</b>")
        nl = _wc_net_line()
        if nl:
            parts.append(nl)
    else:
        parts.append("💰 Set your balance: /balance 128.56")
    settled = [b for b in bets if b.get("status") in ("won", "lost")]
    if settled:
        b = settled[-1]
        if b["status"] == "won":
            parts.append(f"last bet ✅ {b['label']} @ {b['odds']} → +€{b['return'] - b['stake']:.2f}")
        else:
            parts.append(f"last bet ❌ {b['label']} @ {b['odds']} → −€{b['stake']:.2f}")
    open_n = sum(1 for b in bets if b.get("status") == "open")
    if open_n:
        parts.append(f"{open_n} open")
    return " · ".join(parts)


def _wc_parse_bet_text(text: str, pend: dict) -> tuple[float | None, float | None, str]:
    """Human-friendly stake/odds extraction for the pending-bet flow.

    Accepts "60 @ 1.80", "60 at 1.80", "I bet Canada winning at 1.80" (odds
    only — stake asked next), bare "60" (stake only — odds asked next), in
    any order across messages. Returns (stake, odds, leftover_words); either
    number may be None — the caller asks for what's missing.
    Heuristics: two numbers = positional stake, odds — swapped only when the
    first looks like odds (decimal ≤ 20) and the second like a stake (whole
    number ≥ 21). One number = odds if it has decimals in 1.01–20 (or stake
    is already known), else stake.
    """
    # Market tokens first ("X2", "over 2.5") — their digits are NOT money.
    cleaned = re.sub(r"(?i)\b(over|under)\s*\d[.,]5\b", " ", text)
    cleaned = re.sub(r"(?i)(?<![a-z0-9])(1x|x2|12)(?![a-z0-9.,])", " ", cleaned)
    nums = [float(n.replace(",", ".")) for n in
            re.findall(r"(?<![A-Za-z])\d+(?:[.,]\d+)?", cleaned)]
    words = re.sub(r"\d+(?:[.,]\d+)?", " ", text)
    words = re.sub(r"(?i)\b(at|bet|i|on|the|per|euro|eur|want|to|and|odds|are|is|in|vs)\b|[@×€$]",
                   " ", words)
    words = " ".join(words.split()).strip(" .,!?")
    stake, odds = pend.get("stake"), pend.get("odds")
    if len(nums) >= 2:
        stake, odds = nums[0], nums[1]
        if (stake != int(stake) and stake <= 20
                and odds == int(odds) and odds >= 21):
            stake, odds = odds, stake          # "1.80 ... 60" word order
    elif len(nums) == 1:
        n = nums[0]
        looks_like_odds = n != int(n) and 1.01 <= n <= 20
        if odds is None and (looks_like_odds or stake is not None):
            odds = n
        else:
            stake = n
    return stake, odds, words


WC_LADDER_STATE_JSON = PROJECT_ROOT / "data" / "worldcup" / "ladder_state.json"


def _wc_ladder_state() -> dict:
    # Always derive from the settled journal (self-healing — the stored file is
    # just a cache/audit trail). Guarantees the rung the user sees matches their
    # real win/loss record, never a stale incremental counter.
    return _wc_ladder_derive()


def _wc_ladder_derive() -> dict:
    """Derive the ladder from the REAL settled journal — single source of truth.

    The trailing streak = consecutive WINS counting back from the most recent
    settled bet; the rung = that latest win's full return; any loss (or no
    settled bets) resets to base. Voids are skipped (they don't break a streak
    or continue it). Derived, not incremented, so it can NEVER drift again
    (the streak-7 bug on 2026-06-15 came from settle paths that bypassed the
    old incremental counter)."""
    bets = _wc_json_load(WC_MYBETS_JSON, [])
    settled = [b for b in bets if b.get("status") in ("won", "lost")]
    settled.sort(key=lambda b: b.get("settled_at") or b.get("placed_at") or "")
    streak, rung = 0, None
    for b in reversed(settled):
        if b["status"] == "won":
            streak += 1
            if rung is None:
                rung = round(b["stake"] * b["odds"], 2)
        else:
            break
    return {"rung": rung, "streak": streak,
            "updated_at": datetime.now(UTC).isoformat(),
            "note": "derived from settled journal"}


def _wc_ladder_on_settle(bet: dict, won: bool) -> dict:
    """Recompute the ladder from the journal after a settle. (bet/won kept for
    call-site compatibility; the journal is now the source of truth.)"""
    st = _wc_ladder_derive()
    _wc_json_save(WC_LADDER_STATE_JSON, st)
    return st


def _wc_pred_for_match(match_number) -> dict | None:
    import json as _json
    try:
        doc = _json.loads(_WC_PREDICTIONS_JSON.read_text())
        return next(x for x in doc.get("predictions", [])
                    if str(x.get("match_number")) == str(match_number))
    except (OSError, ValueError, StopIteration):
        return None


def _wc_guess_leg(p: dict, words: str) -> tuple[str, str, float] | None:
    """Map free text like 'Canada winning' / 'over 2.5' / 'no goal' to a
    structured leg of THIS match → (key, label, our_prob). None = no guess."""
    w = f" {words.lower()} "
    home = (p.get("home_team") or "").lower()
    away = (p.get("away_team") or "").lower()

    def team_in(t: str) -> bool:
        return bool(t) and any(tok in w for tok in t.split() if len(tok) > 3)

    key = None
    over = re.search(r"over\s*(\d[.,]5)", w)
    under = re.search(r"under\s*(\d[.,]5)", w)
    sym = re.search(r"(?<![a-z0-9])(1x|x2|12)(?![a-z0-9.,])", w)
    if sym:
        key = {"1x": "1X", "x2": "X2", "12": "12"}[sym.group(1)]
    elif over:
        key = {"1.5": "O15", "2.5": "O25"}.get(over.group(1).replace(",", "."))
    elif under:
        key = {"2.5": "U25", "3.5": "U35"}.get(under.group(1).replace(",", "."))
    elif "no goal" in w or "nogoal" in w or "btts no" in w:
        key = "BTTSN"
    elif "btts" in w or "goal goal" in w:
        key = "BTTSY"
    elif "draw" in w or "pareggio" in w:
        key = "X"
    elif team_in(home):
        key = "2" if ("lose" in w or "perde" in w) else "1"
    elif team_in(away):
        key = "1" if ("lose" in w or "perde" in w) else "2"
    if not key:
        return None
    for prob, label, k in _wc_ladder_legs(p):
        if k == key:
            return key, label, prob
    return None


def _wc_stake_suggestion(odds: float, prob: float | None) -> tuple[float | None, str]:
    """(suggested_stake, message) — ladder rung from the REAL balance plus a
    Kelly line when we know our probability for the leg."""
    bk = _wc_bankroll()
    bal = bk.get("balance")
    if not bal:
        return None, "Set your balance first: /balance 128.56 — then I can size bets."
    st = _wc_ladder_state()
    if st.get("streak", 0) >= 1 and st.get("rung"):
        # Ladder memory: the rung is what the last win returned. Win big by
        # letting it ride — but never suggest more than the account holds.
        nominal = float(st["rung"])
        rung = min(nominal, bal)
        surv = 0.6 ** (st["streak"] + 1)  # rough: rungs run ~55-65% legs
        cap = (f" (your last win returned €{nominal:.2f}; account holds €{bal:.2f})"
               if rung < nominal else " — your last win's full return")
        lines = [f"🪜 <b>Ladder rung {st['streak'] + 1}</b> — bet <b>€{rung:.2f}</b>{cap}. "
                 f"Streak {st['streak']}; roughly {surv:.0%} of ladders survive "
                 f"this deep — banking is always allowed."]
        if rung > 0.75 * bal:
            half = max(5.0, round(rung / 2))
            lines.append(f"⚖️ That rung is {rung / bal:.0%} of your account. "
                         f"Half-ride: reply <b>{half:.0f}</b> to bet €{half:.0f} "
                         f"and keep €{bal - half:.2f} on the floor.")
    else:
        rung = max(5.0, round(bal * WC_RUNG_FRACTION))
        lines = [f"💡 Suggested stake: <b>€{rung:.0f}</b> (base rung — "
                 f"{WC_RUNG_FRACTION:.0%} of €{bal:.2f}, floor €{bal - rung:.2f} stays)"]
    if prob:
        fair = 1.0 / prob
        b = odds - 1.0
        kelly = max(0.0, (prob * b - (1 - prob)) / b) if b > 0 else 0.0
        if odds > fair:
            lines.append(f"📐 Our {prob:.0%} → fair {fair:.2f} → at {odds} this is "
                         f"<b>+EV ✅</b> · Kelly-optimal would be €{kelly * bal:.0f}")
        else:
            # Below our fair price: no stake suggestion AT ALL — the model
            # says skip, and a suggested number reads as endorsement.
            return None, (
                f"📐 Our {prob:.0%} → fair <b>{fair:.2f}</b> — at {odds} this is "
                f"<b>below our fair price ⚠️</b>. The model says SKIP: you'd "
                f"need ≥ {fair:.2f} for this leg to be value.\n"
                f"Send a stake anyway to override, or /cancel.")
    lines.append("Reply <b>ok</b> to log the suggested stake, or send your own number.")
    return rung, "\n".join(lines)


# ── odds-sheet scanner: paste a menu of prices, get the EV ranking ───────────

def _wc_match_from_text(text: str, require_mention: bool = False) -> dict | None:
    """Resolve which upcoming fixture (next 48h) a message refers to, by team
    mentions; fallback = the next kickoff (sheet senders often omit the team)."""
    import json as _json
    w = f" {text.lower()} "
    try:
        doc = _json.loads(_WC_PREDICTIONS_JSON.read_text())
    except (OSError, ValueError):
        return None
    now = datetime.now(UTC)
    best, soonest = None, None
    for p in doc.get("predictions", []):
        try:
            ko = datetime.fromisoformat(p["kickoff_utc"])
        except (KeyError, ValueError, TypeError):
            continue
        if not (-3 <= (ko - now).total_seconds() / 3600 <= 48):
            continue
        if soonest is None or ko < soonest[0]:
            soonest = (ko, p)
        hits = sum(1 for team in (p.get("home_team", ""), p.get("away_team", ""))
                   for tok in team.lower().split() if len(tok) > 3 and tok in w)
        if hits and (best is None or hits > best[0]):
            best = (hits, p)
    if best:
        return best[1]
    if require_mention:
        return None
    return soonest[1] if soonest else None


def _wc_resolve_market(p: dict, phrase: str) -> tuple[str, str, float] | None:
    """Segment-level market resolution — bare 1/X/2 are safe here because the
    phrase is already isolated from the odds number."""
    w = f" {phrase.lower().strip()} "
    cs = re.search(r"(\d)\s*[-:]\s*(\d)", w)
    if cs:
        h, a = int(cs.group(1)), int(cs.group(2))
        lh, la = float(p.get("home_xg") or 0), float(p.get("away_xg") or 0)
        if lh and la:
            prob = _wc_grid(lh, la).get((h, a), 0.0)
            return f"CS:{h}:{a}", f"Exact score {h}-{a}", prob
        return None
    guess = _wc_guess_leg(p, w)
    if guess:
        return guess
    sym = re.search(r"(?<![a-z0-9])(1x|x2|12|[1x2])(?![a-z0-9.,])", w)
    if sym:
        key = {"1x": "1X", "x2": "X2", "12": "12",
               "1": "1", "x": "X", "2": "2"}[sym.group(1)]
        for prob, label, k in _wc_ladder_legs(p):
            if k == key:
                return key, label, prob
    return None


def _wc_parse_odds_sheet(p: dict, text: str) -> list[tuple[str, str, float, float]]:
    """'X2 at 1.50, 1 at 2.00, 2 at 3.00' → [(key, label, our_prob, odds)].
    One segment = one market + one price; the LAST plausible decimal in the
    segment is the price, everything before it is the market phrase."""
    out, seen = [], set()
    for seg in re.split(r"[,\n;]+|\band\b", text, flags=re.IGNORECASE):
        nums = list(re.finditer(r"(?<![A-Za-z])\d+(?:[.,]\d+)?", seg))
        if not nums:
            continue
        m = nums[-1]
        odds = float(m.group(0).replace(",", "."))
        if not (1.01 <= odds <= 100):
            continue
        resolved = _wc_resolve_market(p, seg[:m.start()])
        if resolved and resolved[0] not in seen:
            seen.add(resolved[0])
            out.append((*resolved, odds))
    return out


def _wc_scan_odds_sheet(p: dict, pairs: list) -> tuple[str, dict | None]:
    """Rank a pasted odds menu by EV against our model → (message, keyboard).
    Keyboard offers one-tap logging for the +EV legs only."""
    bal = _wc_bankroll().get("balance")
    rows, buttons = [], []
    ranked = sorted(pairs, key=lambda t: t[2] * t[3], reverse=True)
    for key, label, prob, odds in ranked:
        if prob <= 0:
            continue
        fair = 1.0 / prob
        ev = prob * odds - 1.0
        if ev > 0:
            b = odds - 1.0
            kelly = max(0.0, (prob * b - (1 - prob)) / b)
            kelly_txt = f" · Kelly €{kelly * bal:.0f}" if bal else ""
            rows.append(f"✅ <b>{label}</b> @ {odds} — our {prob:.0%} "
                        f"(fair {fair:.2f}) → <b>EV {ev:+.0%}</b>{kelly_txt}")
            if len(buttons) < 3:
                buttons.append([_button(f"🎫 {label} @ {odds}",
                                        f"wcbeto:{p.get('match_number')}|{key}|{odds}")])
        else:
            rows.append(f"❌ {label} @ {odds} — our {prob:.0%} "
                        f"(fair {fair:.2f}) → EV {ev:+.0%}")
    head = (f"🧮 <b>{p.get('home_team')} vs {p.get('away_team')}</b> — "
            f"your prices vs our model\n\n")
    if buttons:
        best = ranked[0]
        tail = (f"\n👉 <b>Best of this menu: {best[1]} @ {best[3]}</b> — "
                f"tap to log it and I'll size the stake.")
    else:
        tail = ("\n👉 <b>Nothing here beats our fair lines — skip this menu.</b> "
                "The book is charging more than every leg is worth.")
    return head + "\n".join(rows) + tail, (_inline_keyboard(buttons) if buttons else None)


def _wc_spontaneous_bet(text: str) -> dict | None:
    """Detect 'I want to bet on X2 in Canada vs Bosnia, odds are 1.95' typed
    out of nowhere — find the fixture (next 48h) by team mentions, resolve
    the leg, and return a pending-bet dict. None = not a bet statement.
    Deterministic on purpose: money intent must NEVER reach the chat AI,
    which paraphrases or refuses (observed live 2026-06-12, twice)."""
    w = f" {text.lower()} "
    if not re.search(r"\b(bet|betting|punto|scommetto|gioco)\b", w):
        return None
    import json as _json
    try:
        doc = _json.loads(_WC_PREDICTIONS_JSON.read_text())
    except (OSError, ValueError):
        return None
    now = datetime.now(UTC)
    best = None
    for p in doc.get("predictions", []):
        try:
            ko = datetime.fromisoformat(p["kickoff_utc"])
        except (KeyError, ValueError, TypeError):
            continue
        if not (-3 <= (ko - now).total_seconds() / 3600 <= 48):
            continue
        hits = sum(1 for team in (p.get("home_team", ""), p.get("away_team", ""))
                   for tok in team.lower().split() if len(tok) > 3 and tok in w)
        if hits and (best is None or hits > best[0]):
            best = (hits, p)
    if not best:
        return None
    p = best[1]
    pend = {"match_number": p.get("match_number"),
            "match": f"{p.get('home_team')} vs {p.get('away_team')}",
            "leg_key": None, "label": None}
    guess = _wc_guess_leg(p, text)
    if guess:
        pend["leg_key"], pend["label"], pend["prob"] = guess
    return pend


def _wc_bet_keyboard(p: dict) -> dict | None:
    """Big one-per-row buttons for logging what Nicola actually bet."""
    legs = _wc_ladder_legs(p)
    mn = p.get("match_number")
    rows = []
    for prob, label, key in legs:
        if key and 0.35 <= prob <= 0.90 and len(rows) < 4:
            rows.append([_button(f"🎫 I bet: {label} ({prob:.0%}, min {1 / prob:.2f})",
                                 f"wcbet:{mn}:{key}")])
    if not rows:
        return None
    rows.append([_button("📝 something else — I'll type it", f"wcbet:{mn}:other")])
    return _inline_keyboard(rows)


# =============================================================================
# WORLD CUP PRE-MATCH ALERTS (proactive, added 2026-06-11)
# =============================================================================
# Unprompted Telegram message when a WC fixture is WC_ALERT_WINDOW minutes
# from kickoff: our call, the market family derived from the lambdas, lineup
# status, scorer propensities (labeled lower-confidence), and the WHY (Elo
# gap, model-vs-market agreement, availability). One alert per match, state
# on disk so restarts don't re-send. Reads the same artifacts as /worldcup —
# never edits anything under scripts/worldcup/.

WC_ALERT_STATE = PROJECT_ROOT / "data" / "worldcup" / "prematch_alerts_sent.json"
WC_ALERT_WINDOW_MIN = (0, 150)        # consider fixtures 0–150 min from kickoff
WC_ALERT_LAST_CALL_MIN = 25           # ≤ this: send with whatever we have
WC_REFRESH_WAIT_MAX_S = 12 * 60       # re-pull cadence while XIs unconfirmed
_WC_ALERT_LAST_CHECK = {"t": 0.0}
_WC_PREDICTIONS_JSON = PROJECT_ROOT / "data" / "worldcup" / "predictions.json"


def _wc_lineups_confirmed(match_number) -> bool:
    import json as _json
    try:
        cl = _json.loads((PROJECT_ROOT / "data" / "worldcup" / "confirmed_lineups.json").read_text())
        return bool((cl.get(str(match_number)) or {}).get("confirmed"))
    except (OSError, ValueError):
        return False


def _wc_spawn_refresh(state: dict) -> None:
    """Kick the full WC pipeline (lineups → availability → predictions) in a
    detached subprocess so the briefing is built from CONFIRMED-XI numbers.
    Globally throttled; output appended to logs/wc-prematch-refresh.log."""
    import subprocess
    last = state.get("_refresh_spawned_at", 0.0)
    if time.time() - last < 600:        # one spawn per 10 min, globally
        return
    state["_refresh_spawned_at"] = time.time()
    with open(PROJECT_ROOT / "logs" / "wc-prematch-refresh.log", "ab") as logf:
        subprocess.Popen(  # noqa: S603 — static argv, no user input
            [sys.executable, "-m", "scripts.worldcup.refresh"],
            cwd=str(PROJECT_ROOT), stdout=logf, stderr=subprocess.STDOUT,
            start_new_session=True,
        )   # child holds its own dup of the fd; parent copy closes here
    log.info("Pre-match: spawned WC refresh for confirmed-lineup predictions")


def _wc_grid(lh: float, la: float) -> dict:
    """Joint score grid P(h, a) from the two lambdas — the base object every
    derived market (and every same-match COMBO leg) is computed from, so
    correlated legs get their TRUE joint probability, never a naive product."""
    import math

    def P(lam: float, k: int) -> float:
        return math.exp(-lam) * lam ** k / math.factorial(k)

    ph = [P(lh, k) for k in range(9)]
    pa = [P(la, k) for k in range(9)]
    return {(h, a): ph[h] * pa[a] for h in range(9) for a in range(9)}


def _wc_poisson_family(lh: float, la: float) -> dict:
    """Score grid markets from the two lambdas (mirrors the /worldcup page)."""
    grid = _wc_grid(lh, la)
    top = sorted(grid.items(), key=lambda kv: -kv[1])[:3]
    btts_no = sum(p for (h, a), p in grid.items() if h == 0 or a == 0)
    o25 = sum(p for (h, a), p in grid.items() if h + a > 2.5)
    margins = {
        "home by 1": sum(p for (h, a), p in grid.items() if h - a == 1),
        "home by 2+": sum(p for (h, a), p in grid.items() if h - a >= 2),
        "away by 1": sum(p for (h, a), p in grid.items() if a - h == 1),
        "away by 2+": sum(p for (h, a), p in grid.items() if a - h >= 2),
    }
    return {"top_scores": top, "btts_no": btts_no, "over25": o25, "margins": margins}


def _wc_scorer_shortlist(players_doc: dict, match_number) -> list[str]:
    """Top anytime-scorer propensities for one match.

    per_match shape: {match_number: {"home": [{name, prob, fair_odds, ...}],
    "away": [...]}} — names + our prob + our fair odds, so the user can spot
    when a book price is above/below our number.
    """
    entry = ((players_doc or {}).get("per_match") or {}).get(str(match_number)) or {}
    out = []
    for side in ("home", "away"):
        rows = entry.get(side, []) or []
        # once lineups are confirmed upstream, only confirmed starters count
        confirmed_rows = [r for r in rows if r.get("confirmed_xi")]
        for r in (confirmed_rows or rows):
            prob = r.get("prob")
            if prob:
                out.append((float(prob),
                            f"{r.get('name', '?')} {float(prob):.0%}"
                            f" (fair {r.get('fair_odds', '?')})"))
    out.sort(reverse=True)
    return [s for _, s in out[:3]]


def _build_prematch_alert(p: dict, minutes_to_ko: float) -> str:
    """One match's pre-kickoff briefing (Telegram HTML)."""
    import json as _json

    home, away = p.get("home_team", "?"), p.get("away_team", "?")
    probs = p.get("probabilities") or {}
    pure = p.get("probabilities_pure_model") or {}
    lh, la = float(p.get("home_xg") or 0), float(p.get("away_xg") or 0)
    ko_local = ""
    try:
        ko = datetime.fromisoformat(p["kickoff_utc"])
        ko_local = ko.astimezone(_wc_tz()).strftime("%H:%M")
    except (KeyError, ValueError, TypeError):
        pass

    pick = max(probs, key=probs.get) if probs else "?"
    sym = {"home": "1", "draw": "X", "away": "2"}.get(pick, "?")
    fam = _wc_poisson_family(lh, la) if lh and la else None

    lines = [
        f"⏰ <b>{home} vs {away}</b> — kickoff ~{int(minutes_to_ko)} min "
        f"({ko_local} {WC_TZ_LABEL})",
        "",
        f"🎯 <b>Our call: {sym}</b> — {home} {probs.get('home', 0):.0%} / "
        f"X {probs.get('draw', 0):.0%} / {away} {probs.get('away', 0):.0%}",
    ]
    if fam:
        ts = " · ".join(f"{h}-{a} {pr:.0%} (min {1 / pr:.1f})"
                        for (h, a), pr in fam["top_scores"] if pr > 0)
        lines.append(f"📊 Top scores: {ts}")
        lines.append(
            f"⚽ BTTS No {fam['btts_no']:.0%} · Over 2.5: {fam['over25']:.0%}"
            + ("  ⚠️ coin-flip" if 0.45 <= fam["over25"] <= 0.55 else "")
        )
        best_margin = max(fam["margins"], key=fam["margins"].get)
        lines.append(f"📏 Margin lean: {best_margin.replace('home', home).replace('away', away)} "
                     f"{fam['margins'][best_margin]:.0%}")

    # WHY — elo gap, model vs market agreement, availability
    why = []
    eh, ea = p.get("elo_home"), p.get("elo_away")
    if eh and ea:
        why.append(f"Elo {eh:.0f} vs {ea:.0f} ({'+' if eh >= ea else '−'}{abs(eh - ea):.0f})")
    if pure and probs:
        d = abs(pure.get(pick, 0) - probs.get(pick, 0))
        if p.get("market_informed"):
            why.append(f"market {'agrees' if d < 0.05 else 'tempers us'} "
                       f"(pure model {pure.get(pick, 0):.0%})")
    if p.get("availability_adjusted"):
        why.append("λ adjusted for team news")
    if why:
        lines.append(f"🧠 Why: {'; '.join(why)}")

    # Lineups — confirmed or not (status honesty)
    try:
        cl = _json.loads((PROJECT_ROOT / "data" / "worldcup" / "confirmed_lineups.json").read_text())
        entry = cl.get(str(p.get("match_number"))) or {}
        if entry.get("confirmed"):
            miss = entry.get("missing") or []
            mtxt = f" · out: {', '.join(m.get('name', '?') for m in miss[:4])}" if miss else ""
            lines.append(f"📋 Lineups CONFIRMED{mtxt}")
        else:
            lines.append("📋 Lineups not confirmed yet — based on expected XI")
    except (OSError, ValueError):
        pass

    # Scorer propensities — honest label, this family is NOT betting-validated
    try:
        players_doc = _json.loads((PROJECT_ROOT / "data" / "worldcup" / "player_predictions.json").read_text())
        sc = _wc_scorer_shortlist(players_doc, p.get("match_number"))
        if sc:
            lines.append(f"👟 Scorer propensity (lower confidence): {' · '.join(sc)}")
    except (OSError, ValueError):
        pass

    lines.append("")
    lines.append(_wc_money_footer())
    return "\n".join(lines)


def _wc_ladder_legs(p: dict) -> list[tuple[float, str]]:
    """Candidate legs for one match across the WHOLE market family.

    Returns [(prob, label)] sorted by prob desc. Who-wins legs use the
    deployed (market-informed) probabilities; totals/joints come from the
    score grid. SISAL-style combo legs (e.g. "1 + Under 3.5") are exact
    joint probabilities from the grid.
    """
    probs = p.get("probabilities") or {}
    home, away = p.get("home_team", "?"), p.get("away_team", "?")
    lh, la = float(p.get("home_xg") or 0), float(p.get("away_xg") or 0)
    legs: list[tuple[float, str, str]] = []          # (prob, label, settle-key)
    if probs:
        ph, pd, pa = probs.get("home", 0), probs.get("draw", 0), probs.get("away", 0)
        legs += [(ph, f"1 ({home} win)", "1"), (pa, f"2 ({away} win)", "2")]
        legs += [(ph + pd, f"1X ({home} or draw)", "1X"),
                 (pa + pd, f"X2 ({away} or draw)", "X2"),
                 (ph + pa, "12 (no draw)", "12")]
    if lh and la:
        g = _wc_grid(lh, la)
        tot = lambda f: sum(v for (h, a), v in g.items() if f(h, a))  # noqa: E731
        legs += [
            (tot(lambda h, a: h + a > 1.5), "Over 1.5", "O15"),
            (tot(lambda h, a: h + a < 2.5), "Under 2.5", "U25"),
            (tot(lambda h, a: h + a > 2.5), "Over 2.5", "O25"),
            (tot(lambda h, a: h + a < 3.5), "Under 3.5", "U35"),
            (tot(lambda h, a: h > 0 and a > 0), "BTTS Yes", "BTTSY"),
            (tot(lambda h, a: h == 0 or a == 0), "BTTS No", "BTTSN"),
            # SISAL "Combo" markets — exact joints, correlation priced in
            (tot(lambda h, a: h > a and h + a < 3.5), f"1 + Under 3.5 ({home})", "1U35"),
            (tot(lambda h, a: h > a and h + a > 1.5), f"1 + Over 1.5 ({home})", "1O15"),
            (tot(lambda h, a: a > h and h + a < 3.5), f"2 + Under 3.5 ({away})", "2U35"),
            (tot(lambda h, a: h > a or h + a > 2.5), f"1 or Over 2.5 ({home})", "1orO25"),
        ]
    legs.sort(reverse=True)
    return legs


# Ladder tiers. "safe" takes the HIGHEST-prob leg inside its band (steady
# compounding); "risk" takes the LOWEST-prob leg inside its band — longest
# odds that still clear the 40% floor, so each rung roughly doubles. Same
# honest math either way: min-odds gate + cumulative survival shown.
WC_LADDER_TIERS = {
    "safe": {"band": (0.55, 0.85), "pick": "safest", "title": "🪜 Safe ladder"},
    "risk": {"band": (0.40, 0.70), "pick": "longest", "title": "🎲 Risk ladder"},
}


def _build_daily_ladder(stake: float = 10.0, tier: str = "safe") -> str:
    """Kickoff-ordered ladder over today's slate: one band-qualifying leg per
    match, each rung staking the full return of the previous one. All payouts
    shown at OUR FAIR ODDS (= 1/prob): that is the MINIMUM SISAL price to
    accept — if the book pays less than the rung's 'min odds', skip the rung."""
    import json as _json

    try:
        doc = _json.loads(_WC_PREDICTIONS_JSON.read_text())
    except (OSError, ValueError):
        return "🪜 No predictions on disk."
    preds = doc.get("predictions", []) if isinstance(doc, dict) else []
    tz = _wc_tz()
    now = datetime.now(UTC)
    today = now.astimezone(tz).date()

    slate = []
    for p in preds:
        try:
            ko = datetime.fromisoformat(p["kickoff_utc"])
        except (KeyError, ValueError, TypeError):
            continue
        if ko.astimezone(tz).date() == today and ko > now:
            slate.append((ko, p))
    slate.sort(key=lambda x: x[0])
    if not slate:
        return "🪜 No remaining fixtures today — ladder starts with tomorrow's slate."

    cfg = WC_LADDER_TIERS.get(tier, WC_LADDER_TIERS["safe"])
    lo, hi = cfg["band"]
    # Bankroll-aware staking: rung = WC_RUNG_FRACTION of the real balance,
    # the floor stays untouched — the split Nicola actually plays (60/128.56).
    bk = _wc_bankroll()
    floor_line = ""
    if bk.get("balance"):
        bal = float(bk["balance"])
        stake = max(5.0, round(bal * WC_RUNG_FRACTION))
        floor_line = (f"💰 Balance €{bal:.2f} → rung <b>€{stake:.0f}</b>, "
                      f"floor <b>€{bal - stake:.2f}</b> stays in the account")
    lines = [f"{cfg['title']} — <b>{today.strftime('%A %d %B')}</b> (start €{stake:.0f}, "
             "each rung bets the previous return)"]
    if floor_line:
        lines.append(floor_line)
    lines.append("")
    bank = stake
    surv = 1.0
    rung = 0
    for ko, p in slate:
        in_band = [leg for leg in _wc_ladder_legs(p) if lo <= leg[0] <= hi]
        # safest = highest prob in band; longest = lowest prob in band
        pick = (in_band[0] if cfg["pick"] == "safest" else in_band[-1]) if in_band else None
        ko_txt = ko.astimezone(tz).strftime("%H:%M")
        match = f"{p.get('home_team', '?')}–{p.get('away_team', '?')}"
        if not pick:
            lines.append(f"⏭ {ko_txt} {match}: no leg in the {lo:.0%}–{hi:.0%} band — skip")
            continue
        prob, label, _key = pick
        rung += 1
        fair = 1.0 / prob
        ret = bank * fair
        surv *= prob
        lines.append(
            f"<b>Rung {rung}</b> · {ko_txt} {match}\n"
            f"   {label} — our {prob:.0%} · <b>min odds {fair:.2f}</b>\n"
            f"   stake €{bank:.2f} → €{ret:.2f} if it lands "
            f"(survival so far {surv:.0%})"
        )
        bank = ret
    if rung == 0:
        return (f"{cfg['title']}: nothing in the {lo:.0%}–{hi:.0%} band today — "
                "no ladder, that's the discipline.")
    lines += [
        "",
        f"💰 Full ladder at min odds: €{stake:.0f} → €{bank:.2f} "
        f"(x{bank / stake:.1f}) with {surv:.0%} survival",
        "⚠️ Rules: take a rung ONLY if SISAL pays ≥ the min odds (that's our "
        "fair price — below it the rung is -EV by our model). Real book odds "
        "above min raise the payout. Stop any time; banking a rung is always "
        "allowed; never chase a broken ladder.",
    ]
    if tier == "safe":
        lines.append("🎲 Spicier version: /ladder risk")
    if tier == "risk":
        # Score shot of the day: the slate's strongest exact-score conviction.
        # SINGLE bet, not a rung — laddering 14% legs is a lottery (2% over two
        # rungs); a top-conviction score at min odds is a longshot WITH a thesis.
        best = None
        for _ko, p in slate:
            lh, la = float(p.get("home_xg") or 0), float(p.get("away_xg") or 0)
            if not (lh and la):
                continue
            g = _wc_grid(lh, la)
            (h, a), pr = max(g.items(), key=lambda kv: kv[1])
            if best is None or pr > best[0]:
                nxt = sorted(g.items(), key=lambda kv: -kv[1])[1]
                best = (pr, h, a, nxt, p)
        if best:
            pr, h, a, ((h2, a2), pr2), p = best
            m = f"{p.get('home_team', '?')}–{p.get('away_team', '?')}"
            lines += [
                "",
                f"🎯 <b>Score shot of the day</b>: {m} → <b>{h}-{a}</b> "
                f"(our {pr:.0%}, <b>min odds {1 / pr:.1f}</b>)",
                f"   cover: add {h2}-{a2} ({pr2:.0%}, min {1 / pr2:.1f}) — "
                f"together {pr + pr2:.0%} that one of the two lands",
                "   single bet, flat stake — exact scores do NOT ladder.",
            ]
    return "\n".join(lines)


_WC_POSTMATCH_LAST = {"t": 0.0}


def _wc_postmatch_check(token: str, chat_id: str) -> None:
    """After an alerted match should have ENDED (kickoff + 110 min), keep
    spawning the WC refresh until its final score is on disk — so settlement,
    result-pinning and grading happen minutes after the whistle instead of at
    the next 2-hourly tick. (Found live 2026-06-12: Canada FT 1-1, Nicola's
    settlement message sat in the gap between refresh cycles.)"""
    import json as _json
    if time.time() - _WC_POSTMATCH_LAST["t"] < 120:
        return
    _WC_POSTMATCH_LAST["t"] = time.time()
    try:
        try:
            sent = _json.loads(WC_ALERT_STATE.read_text())
        except (OSError, ValueError):
            return
        now = datetime.now(UTC)
        changed = False
        for mn, st in list(sent.items()):
            if not isinstance(st, dict) or not st.get("sent") or st.get("result_seen"):
                continue
            p = _wc_pred_for_match(mn)
            if not p:
                continue
            try:
                ko = datetime.fromisoformat(p["kickoff_utc"])
            except (KeyError, ValueError, TypeError):
                continue
            mins_since_ko = (now - ko).total_seconds() / 60
            if mins_since_ko < 110:
                continue                      # match still running
            if _wc_result_for_match(mn):
                st["result_seen"] = now.isoformat()
                changed = True                # settle check picks it up ≤5 min
            elif mins_since_ko < 240:
                _wc_spawn_refresh(sent)       # globally throttled to 1/10min
                changed = True
        if changed:
            WC_ALERT_STATE.write_text(_json.dumps(sent, indent=1))
    except Exception as e:  # noqa: BLE001 — must never kill the bot loop
        log.warning("postmatch check failed: %s", e)


def _check_prematch_alerts(token: str, chat_id: str) -> None:
    """Fire pre-kickoff briefings for fixtures entering the alert window."""
    import json as _json

    now_ts = time.time()
    if now_ts - _WC_ALERT_LAST_CHECK["t"] < 120:    # throttle: every 2 min
        return
    _WC_ALERT_LAST_CHECK["t"] = now_ts
    try:
        doc = _json.loads(_WC_PREDICTIONS_JSON.read_text())
        preds = doc.get("predictions", []) if isinstance(doc, dict) else []
        if not preds:
            return
        try:
            sent = _json.loads(WC_ALERT_STATE.read_text())
        except (OSError, ValueError):
            sent = {}
        now = datetime.now(UTC)
        lo, hi = WC_ALERT_WINDOW_MIN
        changed = False
        for p in preds:
            mn = str(p.get("match_number"))
            entry = sent.get(mn)
            if isinstance(entry, str) or (isinstance(entry, dict) and entry.get("sent")):
                continue                     # already alerted (old or new format)
            try:
                ko = datetime.fromisoformat(p["kickoff_utc"])
            except (KeyError, ValueError, TypeError):
                continue
            mins = (ko - now).total_seconds() / 60
            if not (lo <= mins <= hi):
                continue

            # ── lineup-gated, pipeline-fresh send ────────────────────────
            # Goal: send ONCE, built from a predictions.json regenerated
            # AFTER lineups confirmed. Last-call: never miss a match.
            st = entry if isinstance(entry, dict) else {}
            confirmed = _wc_lineups_confirmed(mn)
            try:
                preds_mtime = _WC_PREDICTIONS_JSON.stat().st_mtime
            except OSError:
                preds_mtime = 0.0
            refresh_at = st.get("refresh_at", 0.0)
            fresh = refresh_at and preds_mtime > refresh_at
            last_call = mins <= WC_ALERT_LAST_CALL_MIN

            # Send ONLY on confirmed lineups + pipeline rebuilt after the spawn
            # that confirmed them — or at last call (a ban never silences a
            # match, but it also never tricks us into an early unconfirmed
            # send: Sofascore publishes XIs ~T-60, so we keep refreshing).
            ready = (confirmed and fresh) or last_call
            if not ready:
                refresh_stale = (not refresh_at
                                 or time.time() - refresh_at > WC_REFRESH_WAIT_MAX_S)
                if refresh_stale and mins <= 90:
                    # pull the whole pipeline (lineups → availability →
                    # predictions); re-pulls every ~12 min until XIs confirm.
                    _wc_spawn_refresh(sent)
                    st["refresh_at"] = time.time()
                    sent[mn] = st
                    changed = True
                continue

            msg = _build_prematch_alert(p, mins)
            _tg_send_message(token, chat_id, msg, reply_markup=_wc_bet_keyboard(p))
            st["sent"] = now.isoformat()
            st["lineups_confirmed"] = confirmed
            st["pipeline_fresh"] = bool(fresh)
            sent[mn] = st
            changed = True
            log.info("Pre-match alert sent: %s vs %s (T-%dmin, lineups=%s, fresh=%s)",
                     p.get("home_team"), p.get("away_team"), int(mins),
                     confirmed, bool(fresh))
            # First alert of the (Rome) day also carries the day's ladders
            day_key = f"ladder_{now.astimezone(_wc_tz()).date()}"
            if day_key not in sent:
                _tg_send_message(token, chat_id, _build_daily_ladder(tier="safe"))
                _tg_send_message(token, chat_id, _build_daily_ladder(tier="risk"))
                sent[day_key] = now.isoformat()
        if changed:
            WC_ALERT_STATE.write_text(_json.dumps(sent, indent=1))
    except Exception as e:  # noqa: BLE001 — alerts must never kill the bot loop
        log.warning("prematch alert check failed: %s", e)


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

            # Proactive WC pre-match briefings (throttled internally)
            _check_prematch_alerts(token, chat_id)
            # Post-match: hunt the final score until it lands on disk
            _wc_postmatch_check(token, chat_id)
            # Auto-settle Nicola's logged bets against real results
            _wc_check_my_bets(token, chat_id)

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

                # Pending bet completion: "60 @ 1.80" / "60 at 1.80" (+ free
                # text). While a bet is pending this flow OWNS the chat —
                # nothing falls through to Claude (which paraphrases, refuses,
                # or claims SA/EPL-only — observed live 2026-06-12).
                # Spontaneous bet statement with no pending flow: detect it
                # deterministically and open the calculation flow — the chat
                # AI never sees money intent.
                if not cmd and chat_id not in _WC_PENDING_BET:
                    # Pasted odds menu ("X2 at 1.5, 1 at 2, 2 at 3") → rank
                    # the whole sheet by EV and offer one-tap logging.
                    p_sheet = _wc_match_from_text(text)
                    pairs = _wc_parse_odds_sheet(p_sheet, text) if p_sheet else []
                    if len(pairs) >= 2:
                        sheet_msg, sheet_kb = _wc_scan_odds_sheet(p_sheet, pairs)
                        _tg_send_message(token, chat_id, sheet_msg,
                                         reply_markup=sheet_kb)
                        continue
                    spont = _wc_spontaneous_bet(text)
                    if spont:
                        spont["at"] = time.time()
                        _WC_PENDING_BET[chat_id] = spont
                        if spont.get("label"):
                            _tg_send_message(token, chat_id,
                                f"🎫 <b>{spont['label']}</b> ({spont['match']}) — got it.")
                        # fall through to the pending handler below, which
                        # parses odds/stake from THIS same message

                # Stale pending (>15 min) expires silently — the message
                # gets processed fresh instead of feeding a dead flow.
                if (chat_id in _WC_PENDING_BET
                        and time.time() - _WC_PENDING_BET[chat_id].get("at", 0) > 900):
                    _WC_PENDING_BET.pop(chat_id, None)

                if not cmd and chat_id in _WC_PENDING_BET:
                    if text.strip().lower() in ("/cancel", "cancel", "annulla"):
                        _WC_PENDING_BET.pop(chat_id, None)
                        _tg_send_message(token, chat_id, "🚫 Bet logging cancelled.")
                        continue
                    pend = _WC_PENDING_BET[chat_id]
                    # "ok" accepts the suggested (calculated) stake
                    if (text.strip().lower() in ("ok", "okay", "yes", "si", "sì", "va bene", "👍")
                            and pend.get("odds") and pend.get("suggested")):
                        stake, odds, words = pend["suggested"], pend["odds"], ""
                    else:
                        stake, odds, words = _wc_parse_bet_text(text, pend)
                    if not pend.get("label") and len(words) > 2:
                        pend["label"] = words            # "Canada winning" etc.
                        # try to resolve free text to a structured leg of this
                        # match → enables auto-settle + the EV check
                        mp = _wc_pred_for_match(pend.get("match_number"))
                        guess = _wc_guess_leg(mp, words) if mp else None
                        if guess:
                            pend["leg_key"], pend["label"], pend["prob"] = guess
                    if stake and odds and odds >= 1.01:
                        _force = "force" in text.lower()
                        _guard = _wc_guard_check(stake, n_legs=1, is_wc=True, force=_force)
                        if _guard:
                            _tg_send_message(token, chat_id, _guard)
                            continue
                        _WC_PENDING_BET.pop(chat_id, None)
                        label = pend.get("label") or "custom bet"
                        bet = _wc_log_bet(pend["match_number"],
                                          pend.get("match", f"match {pend['match_number']}"),
                                          pend.get("leg_key"), label, stake, odds)
                        bk = _wc_bankroll()
                        bal_txt = (f"\n💰 Balance: <b>€{bk['balance']:.2f}</b>"
                                   if bk.get("balance") is not None else "")
                        auto = ("settles automatically at the final whistle"
                                if bet["leg_key"] else "free-text — settle with /settle won|lost")
                        ev_line = ""
                        prob = pend.get("prob")
                        if prob:
                            fair = 1.0 / prob
                            ev_line = (f"\n📐 Our {prob:.0%} → fair {fair:.2f} → "
                                       + ("<b>+EV ✅</b>" if odds > fair
                                          else "<b>below fair ⚠️</b>"))
                        _tg_send_message(token, chat_id,
                            f"🎫 Logged: <b>{label}</b> @ {odds} × €{stake:.2f} "
                            f"(returns €{stake * odds:.2f}){ev_line}\n{auto}{bal_txt}")
                    else:
                        pend["stake"], pend["odds"] = stake, odds
                        if odds and not stake:
                            # THE calculation he asked for: size it from the
                            # real balance + judge the price when leg is known
                            suggested, ask = _wc_stake_suggestion(odds, pend.get("prob"))
                            pend["suggested"] = suggested
                        elif stake and not odds:
                            ask = f"Got it — €<b>{stake:.0f}</b>. At what odds? (e.g. 1.80)"
                        else:
                            ask = ("Tell me the bet with odds — e.g. "
                                   "<code>Canada win at 1.80</code> — and I'll "
                                   "calculate the stake. (/cancel to abort)")
                        _tg_send_message(token, chat_id, ask)
                    continue

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
                elif cmd == "/fill":
                    response_text = _handle_fill_command(text)
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
                elif cmd == "/xi":
                    _tg_send_typing(token, chat_id)
                    response_text = _handle_xi()
                elif cmd in ("/sfide", "/h2h"):
                    _tg_send_typing(token, chat_id)
                    response_text = _handle_sfide()
                elif cmd in ("/picks", "/angoli"):
                    _tg_send_typing(token, chat_id)
                    response_text = _handle_picks()
                elif cmd in ("/formazioni", "/avversario"):
                    response_text = _handle_formazioni()
                elif cmd in ("/wc", "/worldcup"):
                    _tg_send_typing(token, chat_id)
                    response_text = _handle_worldcup()
                elif cmd in ("/ladder", "/scala"):
                    _tg_send_typing(token, chat_id)
                    _tier = "risk" if any(w in text.lower() for w in ("risk", "rischio")) else "safe"
                    response_text = _build_daily_ladder(tier=_tier)
                elif cmd == "/lossstop":
                    arg = text.replace("/lossstop", "").strip().replace(",", ".")
                    bk = _wc_bankroll()
                    if arg:
                        try:
                            bk["loss_stop_eur"] = round(abs(float(arg)), 2)
                            _wc_json_save(WC_BANKROLL_JSON, bk)
                            response_text = (f"🛑 Loss-stop set: the bot refuses bets once "
                                             f"you're down <b>€{bk['loss_stop_eur']:.0f}</b> "
                                             f"vs deposits. This is the deal — keep it.")
                        except ValueError:
                            response_text = "Usage: <code>/lossstop 50</code>"
                    else:
                        cur = bk.get("loss_stop_eur")
                        response_text = (f"🛑 Loss-stop: €{cur:.0f}" if cur
                                         else "No loss-stop set. <code>/lossstop 50</code> to set one.")
                elif cmd == "/guard":
                    g = WC_GUARD
                    bk = _wc_bankroll()
                    response_text = (
                        "🛡 <b>Guardrail</b> (protects against the parlay/blind-bet pattern):\n"
                        f"• Max {g['max_parlay_legs']} legs per ticket\n"
                        f"• Max bet ≤ {g['max_stake_pct']:.0%} of balance\n"
                        f"• World Cup matches only\n"
                        f"• Loss-stop: {('€' + format(bk['loss_stop_eur'], '.0f')) if bk.get('loss_stop_eur') else 'not set (/lossstop 50)'}\n"
                        "Type <b>force</b> in a bet to override the soft limits.")
                elif cmd == "/deposit":
                    arg = text.replace("/deposit", "").strip().replace(",", ".")
                    try:
                        amt = float(arg)
                        bk = _wc_bankroll()
                        bk["deposited"] = round(float(bk.get("deposited") or 0) + amt, 2)
                        bk["balance"] = round(float(bk.get("balance") or 0) + amt, 2)
                        bk["history"] = (bk.get("history") or [])[-49:] + [
                            {"at": datetime.now(UTC).isoformat(), "delta": amt,
                             "note": "deposit"}]
                        _wc_json_save(WC_BANKROLL_JSON, bk)
                        response_text = (f"🏦 Deposit €{amt:.2f} recorded. "
                                         f"Balance €{bk['balance']:.2f} · "
                                         f"total deposited €{bk['deposited']:.2f}.")
                    except ValueError:
                        response_text = "Usage: <code>/deposit 20</code>"
                elif cmd == "/cancel":
                    if _WC_PENDING_BET.pop(chat_id, None):
                        response_text = "🚫 Bet logging cancelled."
                    else:
                        response_text = "Nothing pending — all clear."
                elif cmd == "/balance":
                    arg = text.replace("/balance", "").strip().replace(",", ".")
                    if arg:
                        try:
                            bk = _wc_bankroll()
                            bk["balance"] = round(float(arg), 2)
                            _wc_json_save(WC_BANKROLL_JSON, bk)
                            response_text = f"💰 Balance set: <b>€{bk['balance']:.2f}</b>"
                        except ValueError:
                            response_text = "Usage: <code>/balance 128.56</code>"
                    else:
                        response_text = _wc_money_footer()
                elif cmd == "/bet":
                    bm = re.match(
                        r"^/bet\s+(\d+(?:[.,]\d+)?)\s*(?:@|at)\s*(\d+(?:[.,]\d+)?)\s*(.*)$",
                        text.strip(), re.IGNORECASE)
                    if not bm:
                        response_text = ("Usage: <code>/bet 60 @ 1.80 Canada win</code> "
                                         "(or tap a 🎫 button on an alert)")
                    else:
                        stake = float(bm.group(1).replace(",", "."))
                        odds = float(bm.group(2).replace(",", "."))
                        label = bm.group(3).strip() or "custom bet"
                        _force = "force" in label.lower()
                        # an odds product that looks like a multi-leg combo
                        _legs = 4 if odds >= 6.0 else (2 if odds >= 3.0 else 1)
                        _guard = _wc_guard_check(stake, n_legs=_legs, is_wc=False,
                                                 force=_force)
                        if _guard:
                            response_text = _guard
                        else:
                            bet = _wc_log_bet(None, "manual", None, label, stake, odds)
                            bk = _wc_bankroll()
                            response_text = (
                                f"🎫 Logged #{bet['id']}: <b>{label}</b> @ {odds} × €{stake:.2f}. "
                                f"Settle with /settle won|lost."
                                + (f"\n💰 €{bk['balance']:.2f}" if bk.get("balance") is not None else ""))
                elif cmd == "/settle":
                    arg = text.replace("/settle", "").strip().lower()
                    bets = _wc_json_load(WC_MYBETS_JSON, [])
                    open_manual = [b for b in bets
                                   if b.get("status") == "open" and not b.get("leg_key")]
                    if arg not in ("won", "lost", "void") or not open_manual:
                        response_text = ("Usage: /settle won|lost|void — settles your "
                                         f"latest free-text bet ({len(open_manual)} open)")
                    else:
                        b = open_manual[-1]
                        b["status"] = arg
                        b["settled_at"] = datetime.now(UTC).isoformat()
                        if arg == "won":
                            b["return"] = round(b["stake"] * b["odds"], 2)
                            _wc_bankroll_apply(b["return"], f"bet #{b['id']} WON (manual)")
                            _wc_ladder_on_settle(b, True)
                        elif arg == "void":
                            _wc_bankroll_apply(b["stake"], f"bet #{b['id']} void")
                        else:
                            _wc_ladder_on_settle(b, False)
                        _wc_json_save(WC_MYBETS_JSON, bets)
                        bk = _wc_bankroll()
                        response_text = (f"{'✅' if arg == 'won' else '⚪' if arg == 'void' else '❌'} "
                                         f"#{b['id']} {b['label']} settled {arg}."
                                         + (f" 💰 €{bk['balance']:.2f}" if bk.get("balance") is not None else ""))
                elif cmd == "/leg":
                    # /leg <bet_id> <leg_n> w|l — settle a manual parlay leg
                    # (e.g. the Brazilian legs with no result feed), then
                    # re-grade the ticket (may complete or early-kill it).
                    parts = text.replace("/leg", "").strip().split()
                    bets = _wc_json_load(WC_MYBETS_JSON, [])
                    ok = False
                    if len(parts) == 3 and parts[2].lower() in ("w", "l", "won", "lost"):
                        try:
                            bid, ln = int(parts[0]), int(parts[1])
                        except ValueError:
                            bid = ln = None
                        won = parts[2].lower() in ("w", "won")
                        bet = next((x for x in bets if x.get("id") == bid
                                    and x.get("status") == "open" and x.get("legs")), None)
                        leg = next((lg for lg in (bet or {}).get("legs", [])
                                    if lg.get("n") == ln), None) if bet else None
                        if leg and leg.get("status", "open") == "open":
                            leg["status"] = "won" if won else "lost"
                            leg["settled_at"] = datetime.now(UTC).isoformat()
                            leg["manual"] = True
                            _wc_json_save(WC_MYBETS_JSON, bets)
                            # Re-grade the whole ticket now this leg is in.
                            _wc_settle_parlay(bet, token, chat_id)
                            _wc_json_save(WC_MYBETS_JSON, bets)
                            ok = True
                            response_text = (
                                f"{'✅' if won else '❌'} Leg {ln} of #{bid} "
                                f"settled {'won' if won else 'lost'} "
                                f"[ticket: {bet['status']}].")
                    if not ok:
                        open_parlays = [b for b in bets
                                        if b.get("status") == "open" and b.get("legs")]
                        hint = ""
                        if open_parlays:
                            b = open_parlays[-1]
                            pend = [str(lg["n"]) for lg in b["legs"]
                                    if lg.get("status", "open") == "open"]
                            hint = (f" Open parlay #{b['id']} legs awaiting: "
                                    f"{', '.join(pend) or 'none'}.")
                        response_text = ("Usage: <code>/leg &lt;bet_id&gt; &lt;leg_n&gt; w|l</code>"
                                         f" — settle one manual parlay leg.{hint}")
                elif cmd == "/mybets":
                    bets = _wc_json_load(WC_MYBETS_JSON, [])
                    if not bets:
                        response_text = "No bets logged yet — tap 🎫 on an alert."
                    else:
                        rows = [_wc_money_footer(), ""]
                        leg_icon = {"open": "⏳", "won": "✅", "lost": "❌"}
                        for b in bets[-10:]:
                            icon = {"open": "⏳", "won": "✅", "lost": "❌",
                                    "void": "⚪"}.get(b["status"], "?")
                            rows.append(f"{icon} #{b['id']} {b['label']} @ {b['odds']} "
                                        f"× €{b['stake']:.2f} [{b['status']}]")
                            # Parlay: show per-leg status so pending legs are visible.
                            for lg in (b.get("legs") or []):
                                li = leg_icon.get(lg.get("status", "open"), "?")
                                tag = " ✋" if lg.get("feed") == "manual" else ""
                                rows.append(f"     {li} L{lg.get('n', '?')} "
                                            f"{lg.get('match', '?')}: "
                                            f"{lg.get('label') or lg.get('leg_key')}{tag}")
                        response_text = "\n".join(rows)
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
