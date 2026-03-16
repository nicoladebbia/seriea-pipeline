#!/usr/bin/env python3
"""
50-message stress test for the AI Advisor.
Tests every tool, edge cases, memory, accuracy, and error handling.
Validates outputs against real data where possible.
"""

import json
import re
import sys
import time
import requests
import pandas as pd
from pathlib import Path

BASE_URL = "http://127.0.0.1:5001"
DATA_DIR = Path(__file__).parent.parent / "data"
PASS = 0
FAIL = 0
WARN = 0
ISSUES = []

# Pre-load reference data for validation
sof = pd.read_parquet(DATA_DIR / "external" / "sofascore" / "player_match_stats.parquet")
sof["date"] = pd.to_datetime(sof["date"])


def chat(message: str, session_id: str = "test") -> tuple[str, list[str]]:
    """Send a chat message and return (full_text, tools_called)."""
    try:
        resp = requests.post(
            f"{BASE_URL}/api/chat",
            json={"message": message, "session_id": session_id},
            timeout=180,
            stream=True,
        )
        full = ""
        tools = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            raw = line[6:].strip()
            if not raw:
                continue
            try:
                d = json.loads(raw)
                if d.get("type") == "text":
                    full += d["content"]
                elif d.get("type") == "tool_use":
                    tools.append(d["tool"])
                elif d.get("type") == "error":
                    full += f"[ERROR: {d.get('content', '')}]"
            except json.JSONDecodeError:
                pass
        return full, tools
    except Exception as e:
        return f"[CONNECTION ERROR: {e}]", []


def check(test_num: int, name: str, message: str, validators: list, session_id: str = None):
    """Run a test and validate the output."""
    global PASS, FAIL, WARN, ISSUES
    sid = session_id or f"test_{test_num}_{int(time.time())}"
    text, tools = chat(message, sid)

    errors = []
    warnings = []

    for v in validators:
        result = v(text, tools)
        if result is None:
            continue
        level, msg = result
        if level == "FAIL":
            errors.append(msg)
        elif level == "WARN":
            warnings.append(msg)

    status = "FAIL" if errors else ("WARN" if warnings else "PASS")
    icon = {"PASS": "\u2705", "FAIL": "\u274c", "WARN": "\u26a0\ufe0f"}[status]

    print(f"{icon} #{test_num:02d} {name}")
    if errors:
        FAIL += 1
        for e in errors:
            print(f"      FAIL: {e}")
            ISSUES.append((test_num, name, "FAIL", e, text[:200]))
    elif warnings:
        WARN += 1
        for w in warnings:
            print(f"      WARN: {w}")
            ISSUES.append((test_num, name, "WARN", w, ""))
    else:
        PASS += 1

    if errors:
        # Print first 300 chars of output for debugging
        print(f"      Output: {text[:300]}...")

    # Rate limit — be nice to Haiku
    time.sleep(0.5)


# ---------------------------------------------------------------------------
# Validator helpers
# ---------------------------------------------------------------------------
def has_tool(*tool_names):
    """Check that at least one of the tool names was called."""
    def v(text, tools):
        if not any(t in tools for t in tool_names):
            return ("FAIL", f"Expected one of {list(tool_names)} but got {tools}")
    return v

def no_tool(tool_name):
    def v(text, tools):
        if tool_name in tools:
            return ("WARN", f"Unnecessary tool call '{tool_name}'")
    return v

def contains(*substrs, case_sensitive=False):
    """Check that at least one of the substrings is present."""
    def v(text, tools):
        t = text if case_sensitive else text.lower()
        for substr in substrs:
            s = substr if case_sensitive else substr.lower()
            if s in t:
                return None
        return ("FAIL", f"Missing expected text (any of): {list(substrs)}")
    return v

def not_contains(substr, case_sensitive=False):
    def v(text, tools):
        t = text if case_sensitive else text.lower()
        s = substr if case_sensitive else substr.lower()
        if s in t:
            return ("FAIL", f"Should NOT contain: '{substr}'")
    return v

def has_rating(player_name, expected_rating):
    """Check that a player's rating matches the database."""
    def v(text, tools):
        pattern = rf'{re.escape(player_name)}.*?(\d\.\d)'
        m = re.search(pattern, text, re.IGNORECASE)
        if not m:
            return ("WARN", f"Rating for {player_name} not found in output")
        actual = m.group(1)
        if actual != str(expected_rating):
            return ("FAIL", f"{player_name} rating: got {actual}, expected {expected_rating}")
    return v

def min_length(n):
    def v(text, tools):
        if len(text) < n:
            return ("FAIL", f"Response too short ({len(text)} chars, expected >={n})")
    return v

def has_table():
    def v(text, tools):
        if "|" not in text or "---" not in text:
            return ("WARN", "No markdown table found in output")
    return v

def no_error():
    def v(text, tools):
        if "[ERROR" in text or "[CONNECTION ERROR" in text:
            return ("FAIL", f"Error in response: {text[:200]}")
        # Check for Claude saying "I don't have" or "I can't" when it should have data
        if re.search(r"i (don.t|cannot|can.t) (access|find|get|retrieve)", text.lower()):
            return ("WARN", "Model claims it can't access data")
    return v

def no_hallucination_markers():
    """Check for common hallucination patterns."""
    def v(text, tools):
        # Suspiciously round numbers
        if re.search(r'rating.*\*\*10\.0\*\*', text):
            return ("WARN", "Suspiciously perfect 10.0 rating")
        # Claiming data doesn't exist when it should
        if "no data available" in text.lower() and any(t.startswith("get_") for t in tools):
            return ("WARN", "Tool was called but model says no data")
    return v


# ---------------------------------------------------------------------------
# THE 50 TESTS
# ---------------------------------------------------------------------------
def run_tests():
    print("=" * 70)
    print("ADVISOR STRESS TEST — 50 Messages")
    print("=" * 70)
    print()

    # ===== MATCH PLAYER ANALYSIS (get_match_players) =====
    print("--- MATCH PLAYER ANALYSIS ---")

    check(1, "Como vs Roma ratings",
        "Show me exact player ratings for Como vs Roma today",
        [has_tool("get_match_players"), has_rating("Valle", 7.6),
         has_rating("El Shaarawy", 7.4), has_rating("Malen", 6.8),
         has_table(), no_error()])

    check(2, "Pisa vs Cagliari best player",
        "Who was the man of the match in Pisa vs Cagliari?",
        [has_tool("get_match_players"), contains("Caracciolo"),
         has_rating("Caracciolo", 8.8), no_error()])

    check(3, "Verona Genoa analysis",
        "Give me a full analysis of the Verona vs Genoa match with all player ratings",
        [has_tool("get_match_players"), has_rating("stigård", 8.5),
         min_length(500), no_error()])

    check(4, "Sassuolo Bologna match events",
        "What happened in the Sassuolo vs Bologna match? Goals, cards, subs?",
        [has_tool("get_match_players"), contains("goal"),
         min_length(300), no_error()])

    check(5, "Lazio Milan worst player",
        "Who was the worst rated player in Lazio vs Milan today?",
        [has_tool("get_match_players"), no_error()])

    # ===== PLAYER STATS (get_player_stats) =====
    print("\n--- PLAYER STATS ---")

    check(6, "Lautaro Martinez stats",
        "How is Lautaro Martinez doing this season?",
        [has_tool("get_player_stats"), contains("martinez", "lautaro"),
         contains("inter"), min_length(300), no_error()])

    check(7, "Vlahovic injury detection",
        "Tell me about Vlahovic",
        [has_tool("get_player_stats"), contains("absent", "miss", "injur", "out"),
         min_length(300), no_error()])

    check(8, "Kenan Yildiz Turkish chars",
        "Stats for kenan yildiz",
        [has_tool("get_player_stats"), contains("ldız", "ldiz", "Yildiz", "Yıldız"),
         not_contains("not found"), not_contains("don't have"), no_error()])

    check(9, "Calhanoglu Turkish chars",
        "How is calhanoglu doing?",
        [has_tool("get_player_stats"), contains("inter", "milan"),
         not_contains("can't find"), no_error()])

    check(10, "Nico Paz form",
        "Is Nico Paz in good form? Show me his last 5 matches",
        [has_tool("get_player_stats"), contains("paz"),
         contains("como"), min_length(300), no_error()])

    check(11, "Conceicao accented name",
        "Give me Francisco Conceicao stats",
        [has_tool("get_player_stats"), contains("conceição", "conceicao"),
         no_error()])

    check(12, "Unknown player graceful fail",
        "Stats for Giovanni Fantastico",
        [no_error(), not_contains("connection error"),
         contains("don't have", "no data", "doesn't match", "not found", "no player", "doesn't exist")])

    check(13, "Player comparison context",
        "Who is better, Lautaro or Vlahovic right now?",
        [has_tool("get_player_stats"), contains("martinez", "lautaro"),
         contains("vlahovic", "vlahović"), min_length(300), no_error()])

    # ===== MATCH PREDICTIONS (get_match_prediction) =====
    print("\n--- MATCH PREDICTIONS ---")

    check(14, "Juve vs Sassuolo prediction",
        "Analyze the Juventus vs Sassuolo match for me",
        [has_tool("get_match_prediction"), contains("juventus", "juve"),
         min_length(400), no_error()])

    check(15, "Fiorentina vs Inter prediction",
        "What's your prediction for Fiorentina Inter?",
        [has_tool("get_match_prediction"), contains("fiorentina", "inter"),
         min_length(300), no_error()])

    check(16, "Roma vs Lecce deep analysis",
        "Give me a deep analysis of Roma vs Lecce with odds and value bets",
        [has_tool("get_match_prediction"), contains("roma"),
         min_length(400), no_error()])

    # ===== VALUE BETS (get_value_bets) =====
    print("\n--- VALUE BETS ---")

    check(17, "Today's best bets",
        "What are today's best value bets?",
        [has_tool("get_value_bets"), contains("edge", "value"),
         min_length(200), no_error()])

    check(18, "Safest bets",
        "Which bets have the highest probability of winning?",
        [has_tool("get_value_bets"), no_error()])

    check(19, "Highest edge bets",
        "Show me the bets with the biggest edge",
        [has_tool("get_value_bets"), contains("edge"), no_error()])

    # ===== H2H (get_h2h) =====
    print("\n--- HEAD TO HEAD ---")

    check(20, "Napoli vs Inter H2H",
        "Compare Napoli vs Inter head to head",
        [has_tool("get_h2h"), contains("napoli"), contains("inter"),
         min_length(200), no_error()])

    check(21, "Milan vs Juve H2H",
        "What's the historical record between Milan and Juventus?",
        [has_tool("get_h2h"), min_length(200), no_error()])

    # ===== BANKROLL (get_bankroll_status) =====
    print("\n--- BANKROLL ---")

    check(22, "Bankroll overview",
        "How's my bankroll doing?",
        [has_tool("get_bankroll_status"), contains("roi", "bankroll"),
         min_length(200), no_error()])

    check(23, "Market profitability",
        "Which betting markets are making me money and which are losing?",
        [has_tool("get_bankroll_status"), min_length(200), no_error()])

    # ===== MATCH CONTEXT (get_match_context) =====
    print("\n--- MATCH CONTEXT ---")

    check(24, "Match context Como vs Pisa",
        "What's the context for the Como vs Pisa match? Referee, injuries, odds movement",
        [has_tool("get_match_context"), min_length(200), no_error()])

    check(25, "Milan Torino context",
        "Tell me everything about Milan vs Torino — sharp money, weather, lineups",
        [has_tool("get_match_context", "get_match_prediction"),
         min_length(200), no_error()])

    # ===== RESULTS (get_results) =====
    print("\n--- RESULTS ---")

    check(26, "Recent results",
        "What were the most recent Serie A results?",
        [has_tool("get_results"), contains("como", "lazio", "pisa", "result"),
         min_length(100), no_error()])

    check(27, "Today's bet results",
        "How did my bets do today? Show me the P&L",
        [has_tool("get_results", "get_bankroll_status"),
         min_length(200), no_error()])

    # ===== MATCH SCORERS (get_match_scorers) =====
    print("\n--- MATCH SCORERS ---")

    check(28, "Goalscorer picks Juve Sassuolo",
        "Who will score in Juventus vs Sassuolo? Give me goalscorer picks",
        [has_tool("get_match_scorers"), contains("goal", "scorer"),
         min_length(300), no_error()])

    check(29, "Goalscorer picks Atalanta Verona",
        "Best anytime goalscorer bets for Atalanta vs Verona",
        [has_tool("get_match_scorers"), min_length(200), no_error()])

    # ===== HISTORICAL QUERIES (query_history) =====
    print("\n--- HISTORICAL QUERIES ---")

    check(30, "Juve home record",
        "How does Juventus do at home this season?",
        [has_tool("query_history"), contains("juventus", "juve"),
         min_length(200), no_error()])

    check(31, "Inter away form",
        "What's Inter's away record? Win rate, goals scored",
        [has_tool("query_history"), contains("inter"), no_error()])

    check(32, "Derby history",
        "How do derbies usually go in Serie A? More goals or fewer?",
        [has_tool("query_history"), min_length(200), no_error()])

    check(33, "Over 2.5 trend",
        "What percentage of Serie A matches go over 2.5 goals this season?",
        [has_tool("query_history"), no_error()])

    # ===== BETTING FEATURES (place_bet, manage_bets, build_parlay) =====
    print("\n--- BETTING FEATURES ---")

    check(34, "List pending bets",
        "Show me all my pending bets",
        [has_tool("manage_bets"), no_error()])

    check(35, "Build auto parlay",
        "Build me a 3-leg parlay from the best value bets",
        [has_tool("build_parlay", "get_value_bets"),
         min_length(200), no_error()])

    # ===== CONVERSATION MEMORY =====
    print("\n--- CONVERSATION MEMORY ---")

    mem_session = f"memory_{int(time.time())}"
    check(36, "Memory setup: ask about Napoli",
        "Tell me about Napoli's current form",
        [has_tool("get_team_detail"), contains("napoli"), no_error()],
        session_id=mem_session)

    check(37, "Memory test: follow-up",
        "How did they do in their last match?",
        [contains("napoli"), no_error()],
        session_id=mem_session)

    check(38, "Memory test: another follow-up",
        "And who is their top scorer?",
        [no_error()],
        session_id=mem_session)

    mem_session2 = f"memory2_{int(time.time())}"
    check(39, "Memory setup: match analysis",
        "Analyze Fiorentina vs Inter for me",
        [has_tool("get_match_prediction"), no_error()],
        session_id=mem_session2)

    check(40, "Memory: which team is favored",
        "Which team do you think wins and why?",
        [not_contains("which team"), no_error()],
        session_id=mem_session2)

    # ===== EDGE CASES =====
    print("\n--- EDGE CASES ---")

    check(41, "Typo in team name",
        "How is Juvenuts doing?",
        [contains("juventus", "juve"), no_error()])

    check(42, "Abbreviation: inter",
        "What's inter's form?",
        [has_tool("get_team_detail"), contains("inter"), no_error()])

    check(43, "Slang: juve vs sassuolo",
        "Give me the juve vs sassuolo prediction",
        [has_tool("get_match_prediction", "get_team_detail"),
         no_error()])

    check(44, "Multiple tools needed",
        "Compare my bankroll performance with today's results and suggest what to bet next",
        [no_error(), min_length(300)])

    check(45, "Very specific question",
        "What was Nico Paz's rating in the Como vs Roma match and how many key passes did he have?",
        [has_tool("get_match_players", "get_player_stats"),
         contains("paz"), no_error()])

    check(46, "Ambiguous question",
        "Is the over a good bet for the Milan match?",
        [no_error(), min_length(100)])

    check(47, "Complex multi-part",
        "Give me Atalanta vs Verona prediction, who will score, and what's the best bet",
        [min_length(300), no_error()])

    check(48, "Negative case: fake match",
        "What's the prediction for Barcelona vs Real Madrid?",
        [no_error(), not_contains("connection error")])

    check(49, "Live matches check",
        "Are there any matches being played right now?",
        [has_tool("get_live_matches"), no_error()])

    check(50, "Settlement check",
        "Have all my bets from today been settled?",
        [has_tool("get_results", "manage_bets", "get_bankroll_status"),
         no_error()])


    # ===== REPORT =====
    print()
    print("=" * 70)
    print(f"RESULTS: {PASS} PASS | {FAIL} FAIL | {WARN} WARN | {PASS + FAIL + WARN} TOTAL")
    print("=" * 70)

    if ISSUES:
        print()
        print("ISSUES FOUND:")
        print("-" * 70)
        for num, name, level, msg, snippet in ISSUES:
            print(f"  #{num:02d} [{level}] {name}")
            print(f"        {msg}")
            if snippet:
                print(f"        Output: {snippet}...")
            print()

    return FAIL


if __name__ == "__main__":
    fails = run_tests()
    sys.exit(1 if fails > 0 else 0)
