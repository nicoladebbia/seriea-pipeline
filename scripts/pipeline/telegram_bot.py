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
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import atomic_write_json
from scripts.utils.match_timing import now_local, timedelta

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
# Set by _tg_request: "ok" | "http:<code>" | "network" | "api" — so a sender can
# retry a network blip WITH formatting instead of falling back to raw text
# (2026-09-05: a connection reset on chunk 2 of /picks resent it without
# parse_mode and the reader saw literal <b> tags).
_LAST_TG_STATUS = "ok"


def _tg_request(token: str, method: str, params: dict | None = None,
                timeout: int = 35) -> dict | None:
    """Make a Telegram Bot API request. Returns parsed JSON or None on error."""
    global _LAST_TG_STATUS
    url = TELEGRAM_API.format(token=token, method=method)
    payload = json.dumps(params or {}).encode("utf-8")
    _LAST_TG_STATUS = "ok"

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
            _LAST_TG_STATUS = "api"
            return None
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            pass
        log.warning("Telegram HTTP %d: %s", e.code, body)
        _LAST_TG_STATUS = f"http:{e.code}"
        return None
    except urllib.error.URLError as e:
        log.warning("Telegram connection error: %s", e.reason)
        _LAST_TG_STATUS = "network"
        return None
    except Exception as e:
        log.warning("Telegram request failed: %s", e)
        _LAST_TG_STATUS = "network"
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
        if i:
            time.sleep(0.6)  # back-to-back POSTs got "connection reset" on chunk 2 twice (2026-09-05)
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
        if result is None and _LAST_TG_STATUS == "network":
            # a connection blip is not a formatting error: same params once more
            time.sleep(1.5)
            result = _tg_request(token, "sendMessage", params)
        if result is None:
            # Retry without parse_mode only when Telegram rejected the markup
            # (HTTP 400); a chunk resent raw shows literal <b> tags
            if not _LAST_TG_STATUS.startswith("http:4"):
                return False
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
            atomic_write_json(_CONVERSATION_FILE, self._history)
        except Exception as e:
            log.debug("Failed to save conversation: %s", e)

    def _save_league_pref(self):
        try:
            _LEAGUE_PREFS_FILE.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(_LEAGUE_PREFS_FILE, {"league_filter": self._league_filter})
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
    "⚽ XI": "/xi",
    "🆚 Sfide": "/sfide",
    "🎯 Picks": "/picks",
    "📊 Record": "/record",
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
# ---------------------------------------------------------------------------
# Command registry — ONE definition, three consumers
# ---------------------------------------------------------------------------
# Until 2026-09-07 the router handled ~27 commands, `_MENU_COMMANDS` registered
# 12 and `/help` listed ~22, all maintained by hand. They disagreed: `/record`
# was in the menu but absent from `/help`; `/match`, `/fill`, `/league`,
# `/parlays`, `/summary` and `/clear` were reachable but advertised nowhere.
#
# Fields: (command, menu_description, help_line, group, in_menu)
#   in_menu=True  → registered with Telegram's ☰ button (max ~12 reads well)
#   in_menu=False → reachable, and still listed in /help. Fantacalcio and
#                   Serie A betting get a line each; Session and Legacy get a
#                   compact trailing line. Every row in this table appears in
#                   /help — that is what test_every_registered_command... pins.
# `tests/test_telegram_commands.py` asserts this registry and the router's
# dispatch branches name exactly the same set, so they cannot drift again.

_GROUP_FANTA = "Fantacalcio"
_GROUP_BETTING = "Serie A betting"
_GROUP_SESSION = "Session"

_COMMANDS: list[dict] = [
    # ── Fantacalcio ──
    {"command": "xi", "menu": "⚽ Formazione fantacalcio consigliata",
     "help": "formazione consigliata della giornata",
     "group": _GROUP_FANTA, "in_menu": True},
    {"command": "sfide", "menu": "🆚 Pronostico H2H prossimi avversari",
     "help": "pronostico H2H vs i prossimi avversari",
     "group": _GROUP_FANTA, "in_menu": True, "aliases": ["h2h"]},
    {"command": "formazioni", "menu": "📸 Manda la formazione avversaria",
     "help": "foto della formazione avversaria → XI corretto",
     "group": _GROUP_FANTA, "in_menu": True, "aliases": ["avversario"]},
    # ── Serie A betting ──
    {"command": "picks", "menu": "🎯 Miglior angolo per ogni partita",
     "help": "miglior angolo per OGNI partita (tutti i mercati)",
     "group": _GROUP_BETTING, "in_menu": True, "aliases": ["angoli"]},
    {"command": "bets", "menu": "🎫 Value bet nello slip",
     "help": "value bet nello slip (edge-gated, soldi veri)",
     "group": _GROUP_BETTING, "in_menu": True},
    {"command": "record", "menu": "📊 Record mercati: chi ha guadagnato la puntata vera",
     "help": "record per mercato: chi ha guadagnato la puntata vera",
     "group": _GROUP_BETTING, "in_menu": True, "aliases": ["storico"]},
    {"command": "today", "menu": "📅 Partite di oggi + previsioni",
     "help": "partite di oggi + previsioni",
     "group": _GROUP_BETTING, "in_menu": True, "aliases": ["matches"]},
    {"command": "match", "menu": "", "help": "tocca una partita per l'analisi completa",
     "group": _GROUP_BETTING, "in_menu": False},
    {"command": "live", "menu": "🔴 Risultati live",
     "help": "risultati live + le tue bet",
     "group": _GROUP_BETTING, "in_menu": True},
    {"command": "bankroll", "menu": "💰 Bilancio, ROI, streak",
     "help": "bilancio, ROI, streak",
     "group": _GROUP_BETTING, "in_menu": True},
    {"command": "player", "menu": "👤 Scheda giocatore: /player Dzeko",
     "help": "scheda giocatore: /player Dzeko",
     "group": _GROUP_BETTING, "in_menu": True},
    {"command": "digest", "menu": "📰 Riassunto del giorno",
     "help": "riassunto del giorno",
     "group": _GROUP_BETTING, "in_menu": True},
    {"command": "parlays", "menu": "", "help": "multiple (guardrail: max 3 legs)",
     "group": _GROUP_BETTING, "in_menu": False},
    {"command": "fill", "menu": "", "help": "conferma una giocata (/fill 2 1.95)",
     "group": _GROUP_BETTING, "in_menu": False},
    {"command": "league", "menu": "", "help": "filtro per lega (EPL, Serie A)",
     "group": _GROUP_BETTING, "in_menu": False},
    {"command": "summary", "menu": "", "help": "riepilogo settimanale a bottoni",
     "group": _GROUP_BETTING, "in_menu": False},
    # ── Session ──
    {"command": "start", "menu": "", "help": "benvenuto e stato del sistema",
     "group": _GROUP_SESSION, "in_menu": False},
    {"command": "clear", "menu": "", "help": "reset conversazione",
     "group": _GROUP_SESSION, "in_menu": False},
    {"command": "help", "menu": "❓ Tutti i comandi", "help": "tutti i comandi",
     "group": _GROUP_SESSION, "in_menu": True},
]

# Every name the router must dispatch: primary commands plus their aliases.
ALL_COMMAND_NAMES: set[str] = {c["command"] for c in _COMMANDS} | {
    a for c in _COMMANDS for a in c.get("aliases", [])
}

_MENU_COMMANDS = [
    {"command": c["command"], "description": c["menu"]}
    for c in _COMMANDS if c.get("in_menu") and c.get("menu")
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
        atomic_write_json(_AI_USAGE_FILE, st)
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
        tg.raw("\U0001f534 <b>Live Matches</b>")
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
        tg.raw("\U0001f3af <b>Top Parlay Picks</b>")
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
        today = now_local().strftime("%Y-%m-%d")

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
                tg.raw("\U0001f4c5 No matches today.")
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


_IT_DAYS = ("lun", "mar", "mer", "gio", "ven", "sab", "dom")
_TIER_MARK = {"A": " ✓", "C": " ~"}   # B (label, not measured) gets no mark
_ENGINE_REASON_IT = {"veto_factor": "veto fattori", "below_min_edge": "edge sotto banda",
                     "above_max_edge": "edge sopra banda", "odds_dead_zone": "quota in zona morta"}


def _human_bet(pk: dict, home: str, away: str) -> str:
    """A bet in plain Italian with the team names, no market jargon:
    'Milan o pareggio', 'Over 1.5 gol', 'Pašalić over 1.5 tiri', 'Malen segna'."""
    bt, sel = pk.get("bet_type") or "", str(pk.get("selection") or "")
    pl = str(pk.get("player") or "").strip()
    surname = pl.split(" ")[-1] if pl else ""
    side = {"1": home, "X": "pareggio", "2": away}
    low = sel.lower()
    if bt in ("O/U 1.5", "O/U 2.5", "Under/over", "O/U") or bt.startswith("O/U"):
        return f"{sel} gol"
    if bt == "1x2 finale":
        return "Pareggio" if sel == "X" else f"{side.get(sel, sel)} vince"
    if bt == "Doppia chance":
        return {"1X": f"{home} o pareggio", "X2": f"{away} o pareggio", "12": "Niente pareggio"}.get(sel, sel)
    if bt == "Goal":
        return "Gol entrambe: sì" if sel == "Sì" else "Gol entrambe: no"
    if bt == "1° tempo 1x2":
        return "1° tempo pari" if sel == "X" else f"1° tempo {side.get(sel, sel)} avanti"
    if bt == "1° tempo under/over":
        return f"1° tempo {low} gol"
    if bt == "Goal 1° tempo":
        return "1° tempo gol entrambe: sì" if sel == "Sì" else "1° tempo gol entrambe: no"
    if bt == "Doppia chance 1° tempo":
        return {"1X": f"1° tempo {home} o pari", "X2": f"1° tempo {away} o pari",
                "12": "1° tempo non pari"}.get(sel, sel)
    if bt == "2° tempo under/over":
        return f"2° tempo {low} gol"
    if bt == "Primo tempo / Finale":
        names = {"H": home, "D": "pari", "A": away}
        a, _, b = sel.partition("/")
        return f"HT/FT {names.get(a, a)} / {names.get(b, b)}"
    if bt == "Risultato esatto":
        return f"Risultato esatto {sel}"
    if bt == "Prima squadra a segnare":
        return {"Casa": f"Segna prima {home}", "Ospite": f"Segna prima {away}", "Nessuno": "Nessun gol"}.get(sel, sel)
    if bt == "Vince o quasi":
        return f"Vince o quasi {sel}"
    if bt == "Tiri totali del giocatore":
        return f"{surname} {low} tiri totali"
    if bt == "Tiri in porta":
        return f"{surname} {low} tiri in porta"
    if bt == "Giocatore marcatore":
        return f"{surname} segna"
    if bt == "Assist giocatore":
        return f"{surname} assist"
    if pl:
        return f"{surname} {bt.lower()} {sel}"
    return f"{bt} {sel}".strip()


def _bet_line(icon: str, pk: dict, home: str, away: str, note: str = "") -> str:
    odds = pk.get("odds")
    edge = pk.get("edge_pct")
    q = f" @{odds:.2f}" if isinstance(odds, int | float) else ""
    e = f" · {edge:+.1f}%" if isinstance(edge, int | float) else ""
    mark = _TIER_MARK.get(str(pk.get("tier")), "")
    one = " (1 book)" if pk.get("n_books") == 1 else ""
    xi = ""
    if pk.get("player") and pk.get("lineup") != "confirmed":
        # a player row priced off a PREDICTED (or last-match) XI, not the team
        # sheet: say so, with the predictor's start% when it has one
        # the CALIBRATED P(starts) the price used (start_prob), else the raw start%
        sp = pk.get("start_prob")
        sp = sp * 100.0 if isinstance(sp, int | float) else pk.get("start_pct")
        xi = f" · XI prob. {sp:.0f}%" if isinstance(sp, int | float) and pk.get("lineup") == "predicted" else " · XI prob."
    return f"{icon} {_human_bet(pk, home, away)}{q}{e}{mark}{one}{xi}{note}"


def _bet_family(pk: dict) -> tuple:
    """Two bets of one family on one card are redundant (Under 3.5 AND Under 4.5,
    Laurienté over 0.5 AND over 1.5 SoT, Udinese vince AND Lazio vince): the card
    keeps the higher-ranked one. Nicola's rule, 2026-09-05."""
    bt, sel = pk.get("bet_type") or "", str(pk.get("selection") or "").lower()
    if pk.get("player"):
        return ("player", pk.get("player"), bt)
    side = "under" if sel.startswith("under") else "over"
    if bt.startswith("O/U") or bt == "Under/over":
        return ("ou", side)
    if bt in ("1x2 finale", "Doppia chance", "Vince o quasi"):
        return ("result",)
    if bt == "1° tempo under/over":
        return ("1h_ou", side)
    if bt in ("1° tempo 1x2", "Doppia chance 1° tempo"):
        return ("1h_result",)
    if bt == "2° tempo under/over":
        return ("2h_ou", side)
    return (bt,)


MAX_CARD_LINES = 4


def _engine_note_it(note: str | None) -> str:
    if not note:
        return ""
    reason = note.split("rejected it: ", 1)[-1].split(" at ")[0].split(" (")[0]
    key = reason.split(":")[0]
    return f"\n   ⚙️ <i>motore: {_ENGINE_REASON_IT.get(key, reason)}</i>"


def _handle_picks(max_matches: int = 20) -> str:
    """/picks — one card per upcoming match (scripts/betting/picks.py), split
    the way the money is split. 💰 is the engine's REAL bet on that match, with
    its stake, or "nessuna vera". 📝 are the PAPER bets (€10 each): exactly what
    the T-30 run journals for the match — the headline lean, the best exotic,
    then every other angle inside the credible band — deduped one per family
    and capped at MAX_CARD_LINES so the card stays readable (Nicola's rule,
    2026-09-05). An angle below the band is journaled nowhere, so it is shown
    nowhere. ➖ = nothing beats its price. Plain Italian, team names, price and
    edge only."""
    try:
        doc = json.loads((PROJECT_ROOT / "data" / "upcoming" / "picks.json").read_text())
    except (OSError, ValueError):
        return ("Nessun picks.json su disco — il motore delle scelte gira col "
                "giro scommesse (mattina/sera e T-30).")
    picks = doc.get("picks") or []
    if not picks:
        return "Nessuna partita in programma nel file picks."
    from zoneinfo import ZoneInfo
    try:
        gen = datetime.fromisoformat(str(doc.get("generated_at") or "").replace("Z", "+00:00"))
    except ValueError:
        gen = None
    journaled_now = bool(doc.get("n_journaled"))
    blocks: list[str] = []
    n_real = sum(1 for q in picks if q.get("label") == "VALUE" and q.get("pick"))
    paper_all = [_paper_bets(q, q.get("lean") if q.get("label") == "VALUE" else (q.get("pick") if q.get("label") == "LEAN" else None))
                 for q in picks]
    n_paper, n_paper_matches = sum(len(x) for x in paper_all), sum(1 for x in paper_all if x)
    for p in picks[:max_matches]:
        home, away = p.get("home_team") or "Casa", p.get("away_team") or "Ospite"
        ko_dt = None
        try:
            ko_dt = datetime.fromisoformat((p.get("kickoff_utc") or "").replace("Z", "+00:00"))
            dt = ko_dt.astimezone(ZoneInfo("Europe/Rome"))
            ko = f"{_IT_DAYS[dt.weekday()]} {dt.strftime('%d/%m %H:%M')}"
        except (ValueError, TypeError):
            ko = (p.get("date") or "?")[5:]
        xi = " · XI ufficiali ✓" if p.get("lineup_state") == "confirmed" else ""
        block = [f"\n<b>{home.upper()} – {away.upper()}</b> · {ko}{xi}"]
        label, headline = p.get("label"), p.get("pick")

        # ---- 💰 the real bet on this match --------------------------------
        if label == "VALUE" and headline:
            stake = headline.get("stake")
            eur = f" · €{stake:.0f}" if isinstance(stake, int | float) and stake > 0 else ""
            tag = " <i>vera · si conferma a T-30</i>" if p.get("stage") == "candidate" else " <i>vera · in slip</i>"
            block.append(f"💰 <b>{_bet_line('', headline, home, away).strip()}</b>{eur}{tag}")
        else:
            block.append("💰 <i>nessuna vera</i>")

        # ---- 📝 the paper bets: what the T-30 run journals ---------------
        lean = p.get("lean") if label == "VALUE" else (headline if label == "LEAN" else None)
        paper = _paper_bets(p, lean)
        key = lambda a: (a.get("bet_type"), a.get("selection"), a.get("player"))  # noqa: E731
        seen: set = set()
        fams: set = set()
        if headline and label == "VALUE":
            seen.add(("Under/over", headline.get("selection"), None))
            fams.add(_bet_family(headline))
        shown: list[dict] = []
        for a in paper:
            k, f = key(a), _bet_family(a)
            if k in seen or f in fams:
                continue
            seen.add(k)
            fams.add(f)
            shown.append(a)
            if len(shown) >= MAX_CARD_LINES:
                break
        if paper:
            inside = ko_dt is not None and gen is not None and timedelta(0) <= (ko_dt - gen) <= timedelta(hours=3)
            when = "scritte a T-30" if journaled_now and inside else "si scrivono a T-30"
            more = f" · {len(paper) - len(shown)} altre" if len(paper) > len(shown) else ""
            block.append(f"📝 <i>carta €10 l'una · {len(paper)} · {when}{more}</i>")
            for i, a in enumerate(shown):
                note = _engine_note_it(a.get("engine_note")) if a is lean else ""
                line = _bet_line("📝", a, home, away, note)
                if i == 0:
                    line = f"📝 <b>{_bet_line('', a, home, away).strip()}</b>{note}"
                block.append(line)
        elif label != "VALUE":
            block.append("📝 <i>nessuna carta</i>")
        if not paper and label not in ("VALUE", "LEAN"):
            if p.get("most_probable"):
                block.append(_bet_line("➖", p["most_probable"], home, away, " <i>il più probabile, ma la quota lo paga già</i>"))
            else:
                block.append("➖ <i>nessuna quota ancora</i>")
        blocks.append("\n".join(block))
    out = [f"🎯 <b>Scelte</b> · {len(picks)} partite",
           f"💰 {n_real} con puntata vera · 📝 {n_paper} scommesse carta su {n_paper_matches} partite"]
    out.extend(blocks)
    if len(picks) > max_matches:
        out.append(f"\n<i>… e altre {len(picks) - max_matches} partite.</i>")
    out.append("\n<i>💰 vera = soldi veri, slip del motore (si conferma a T-30) · 📝 carta €10 = finta, "
               "per costruire lo storico che promuove un mercato a soldi veri (/record) · "
               "grassetto = la migliore · ➖ niente da giocare\n"
               "✓ backtest superato · ~ solo tasso base · % = edge sulla quota · "
               "XI prob. = giocatore atteso, formazione non ufficiale</i>")
    try:
        from scripts.betting.picks import picks_record
        rec = picks_record("serie_a")
        if rec.get("n_settled"):
            top = sorted(rec["by_market"].items(), key=lambda kv: -kv[1]["n"])[:4]
            out.append("<b>Storico carta</b>: " + " · ".join(
                f"{m} n={v['n']} ROI {v['roi_pct']:+.0f}%"
                + (f" CLV {v['mean_clv_pct']:+.1f}%" if v.get("mean_clv_pct") is not None else "")
                for m, v in top))
    except Exception:  # noqa: BLE001 - the record is a footer, never a failure
        pass
    return "\n".join(out) + _fanta_age_note(doc.get("generated_at"))


def _paper_bets(p: dict, lean: dict | None) -> list[dict]:
    """The paper bets of one match, in journal order: mirrors
    picks._journal_candidates (headline lean, best exotic, then every other
    alternative / exotic inside the band). A candidate without `in_band` (older
    files, fixtures) counts as in band; one flagged below it is dropped — the
    card must show what gets journaled, nothing that does not."""
    out: list[dict] = []
    seen: set = set()
    head = [lean, (p.get("exotic") or [None])[0]]
    rest = [c for c in (p.get("alternatives") or []) + (p.get("exotic") or []) if c.get("in_band", True)]
    for c in head + rest:
        if not c or not isinstance(c.get("odds"), int | float):
            continue
        if isinstance(c.get("edge_pct"), int | float) and c["edge_pct"] < 0:
            continue
        k = (c.get("bet_type"), c.get("selection"), c.get("player"))
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out


def _handle_record() -> str:
    """/record — per market: paper record, distance to the promotion bar, and
    the real record once promoted (scripts/betting/market_promotion.py). The
    state file is rewritten after every settle; here it is only rendered."""
    try:
        from scripts.betting.market_promotion import load_state, record_card
        return record_card(load_state(), html=True)
    except Exception as e:  # noqa: BLE001
        return f"Record non disponibile: {e}"


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
            today = now_local().strftime("%Y-%m-%d")
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
    """Handle /help — all available commands, grouped by what they serve.

    Rendered from `_COMMANDS`, the same registry that builds the ☰ menu, so a
    command can never again be reachable-but-undocumented (or the reverse).
    """
    from scripts.pipeline.notify import TgMsg

    tg = TgMsg()
    tg.raw("<b>SerieAI Commands</b>")
    for _group in (_GROUP_FANTA, _GROUP_BETTING):
        _rows = [c for c in _COMMANDS if c["group"] == _group]
        if not _rows:
            continue
        tg.blank()
        tg.raw(f"<b>{_group}:</b>")
        for _c in _rows:
            tg.raw(f"  /{_c['command']} \u2014 {_c['help']}")
    tg.blank()
    tg.raw("<b>🤖 AI chat \u2014 scrivimi e basta:</b>")
    tg.italic("Niente comando: qualsiasi messaggio va all'AI con accesso a")
    tg.italic("previsioni, quote, bankroll, storico 21 stagioni, fantacalcio.")
    tg.italic("Es: 'analizza Genoa-Como', 'come sta andando il bankroll?',")
    tg.italic("'chi schiero in porta?' \u2014 anche screenshot di formazioni.")
    tg.blank()
    _session = [c for c in _COMMANDS if c["group"] == _GROUP_SESSION and c["command"] != "help"]
    if _session:
        tg.raw("<b>Session:</b> " + " · ".join(
            f"/{c['command']} \u2014 {c['help']}" for c in _session))
    return tg.build()


def _handle_match(token: str, chat_id: str) -> bool:
    """Handle /match — show inline keyboard of today's matches to tap.

    Returns True if sent successfully (caller should not send another response).
    """
    try:
        from config.settings import DATA_DIR
        today = now_local().strftime("%Y-%m-%d")

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
    from pathlib import Path as _Path
    marker = _Path(__file__).parent.parent.parent / "data" / "pipeline" / "t30_ticket_state.json"
    try:
        st = _json.loads(marker.read_text())
    except (OSError, ValueError):
        return None, "No ticket on record today."
    if st.get("date") != now_local().strftime("%Y-%m-%d"):
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
    from pathlib import Path as _Path
    marker = _Path(__file__).parent.parent.parent / "data" / "pipeline" / "t30_ticket_state.json"
    try:
        st = _json.loads(marker.read_text())
        assert st.get("date") == now_local().strftime("%Y-%m-%d")
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
                elif cmd in ("/record", "/storico"):
                    response_text = _handle_record()
                elif cmd in ("/formazioni", "/avversario"):
                    response_text = _handle_formazioni()
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
    run_bot()
