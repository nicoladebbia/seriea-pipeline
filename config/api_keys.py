"""Centralized API key loading. Single source of truth.

All API keys come from environment variables (loaded from .env).
No JSON config fallback, no .env file parsing — just os.environ.
"""
from __future__ import annotations
import os
from dotenv import load_dotenv
load_dotenv()

def get_odds_api_key() -> str:
    """The Odds API key ($59/mo, 100K requests)."""
    return os.environ.get("ODDS_API_KEY", "")

def get_telegram_token() -> str:
    return os.environ.get("TELEGRAM_BOT_TOKEN", "")

def get_telegram_chat_id() -> str:
    return os.environ.get("TELEGRAM_CHAT_ID", "")

def get_perplexity_key() -> str:
    """Deprecated — use get_gemini_key() or get_groq_key() instead."""
    return os.environ.get("PERPLEXITY_API_KEY", "")

def get_gemini_key() -> str:
    """Google Gemini API key (primary web search sentiment)."""
    return os.environ.get("GOOGLE_GEMINI_KEY", "")

def get_groq_key() -> str:
    """Groq API key (fallback web search sentiment)."""
    return os.environ.get("GROQ_API_KEY", "")

def get_apifootball_key() -> str:
    return os.environ.get("APIFOOTBALL_KEY", "")

def get_footballdata_key() -> str:
    return os.environ.get("FOOTBALLDATA_API_KEY", "")
