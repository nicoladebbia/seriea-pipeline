"""Which live cards send, and for which league.

2026-09-07: goal/red cards were gated by a global all/bets toggle whose default
was "all", and the full-time card had NO gate at all. In the 09-05..09-07
window 24 of 64 live cards were EPL — a league whose betting is gated, so those
cards could not flip a decision.

New default mode "serie_a": Serie A always, other leagues only when a bet is on
the match. The gate FAILS OPEN — an unknown league still pings, because
silencing a real Serie A goal costs more than one card too many.
"""
from __future__ import annotations

import pytest

from scripts.data import live_monitor as LM

SA = {"home_team": "AS Roma", "away_team": "Atalanta BC", "league": "serie_a"}
EPL = {"home_team": "Arsenal", "away_team": "Chelsea", "league": "premier_league"}
UNKNOWN = {"home_team": "Wildcats FC", "away_team": "Rovers"}


# ---------------------------------------------------------------------------
# mode "serie_a" — the new default
# ---------------------------------------------------------------------------

def test_serie_a_pings_without_a_bet():
    assert LM.live_pings_allowed("serie_a", "AS Roma vs Atalanta BC", SA, False) is True


def test_epl_is_silent_without_a_bet():
    """The 24 cards this change removes."""
    assert LM.live_pings_allowed("serie_a", "Arsenal vs Chelsea", EPL, False) is False


def test_epl_pings_when_there_is_a_bet():
    assert LM.live_pings_allowed("serie_a", "Arsenal vs Chelsea", EPL, True) is True


def test_an_unknown_league_fails_open():
    """A league we cannot identify must not be silenced by accident."""
    assert LM.live_pings_allowed("serie_a", "Wildcats FC vs Rovers", UNKNOWN, False) is True


def test_league_is_inferred_from_team_names_when_the_field_is_missing():
    """match_data does not always carry `league` — inference is the fallback,
    and it must actually work or every match would fail open to 'ping'."""
    no_field = {"home_team": "Arsenal", "away_team": "Chelsea"}
    assert LM._match_league_key("Arsenal vs Chelsea", no_field) == "premier_league"
    assert LM.live_pings_allowed("serie_a", "Arsenal vs Chelsea", no_field, False) is False


def test_league_is_inferred_from_the_match_key_alone():
    assert LM.live_pings_allowed("serie_a", "Arsenal vs Chelsea", None, False) is False


# ---------------------------------------------------------------------------
# the other two modes still behave
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("data,key", [(SA, "AS Roma vs Atalanta BC"),
                                      (EPL, "Arsenal vs Chelsea")])
def test_mode_all_pings_everything(data, key):
    assert LM.live_pings_allowed("all", key, data, False) is True


@pytest.mark.parametrize("data,key", [(SA, "AS Roma vs Atalanta BC"),
                                      (EPL, "Arsenal vs Chelsea")])
def test_mode_bets_needs_a_bet_in_either_league(data, key):
    assert LM.live_pings_allowed("bets", key, data, False) is False
    assert LM.live_pings_allowed("bets", key, data, True) is True


# ---------------------------------------------------------------------------
# mode resolution
# ---------------------------------------------------------------------------

def test_default_mode_is_serie_a_when_state_is_empty(monkeypatch):
    monkeypatch.setattr("scripts.pipeline.pipeline_state.load_state", lambda: {})
    assert LM._goal_ping_mode() == LM.GOAL_PING_DEFAULT_MODE == "serie_a"


def test_an_unknown_stored_mode_falls_back_to_the_default(monkeypatch):
    monkeypatch.setattr("scripts.pipeline.pipeline_state.load_state",
                        lambda: {LM.GOAL_PING_STATE_KEY: "banana"})
    assert LM._goal_ping_mode() == "serie_a"


def test_a_stored_mode_is_honoured(monkeypatch):
    monkeypatch.setattr("scripts.pipeline.pipeline_state.load_state",
                        lambda: {LM.GOAL_PING_STATE_KEY: "all"})
    assert LM._goal_ping_mode() == "all"


def test_state_trouble_does_not_silence_serie_a(monkeypatch):
    def boom():
        raise RuntimeError("state unreadable")
    monkeypatch.setattr("scripts.pipeline.pipeline_state.load_state", boom)
    mode = LM._goal_ping_mode()
    assert mode == "serie_a"
    assert LM.live_pings_allowed(mode, "AS Roma vs Atalanta BC", SA, False) is True


def test_all_modes_are_offered_by_the_web_config():
    """The /live dropdown and the endpoint validate against GOAL_PING_MODES —
    a mode missing from the constant is unreachable from the UI."""
    assert set(LM.GOAL_PING_MODES) == {"all", "serie_a", "bets"}
    from pathlib import Path
    html = Path("web/templates/live.html").read_text(encoding="utf-8")
    for mode in LM.GOAL_PING_MODES:
        assert f'value="{mode}"' in html, f"/live cannot select goal-ping mode {mode}"


def test_the_full_time_card_is_gated_and_latches_the_flag():
    """The FT path had no gate. It also must set _ft_notified when suppressed,
    or a gated match is re-evaluated every single poll cycle forever."""
    from pathlib import Path
    src = Path("scripts/data/live_monitor.py").read_text(encoding="utf-8")
    block = src[src.index("Full-time notifications for completed matches"):]
    block = block[:block.index("Reconciliation: cross-check")]
    assert "live_pings_allowed" in block, "the FT card is still ungated"
    gate = block[block.index("live_pings_allowed"):]
    assert '_ft_notified"] = True' in gate[:gate.index("notify_full_time")], \
        "a suppressed FT card must latch the flag or it retries every cycle"


# ---------------------------------------------------------------------------
# The two gates COMPOSED — the hole the per-file suites leave
# ---------------------------------------------------------------------------
#
# Nicola made two independent choices on 2026-09-07: "Serie A always, gate EPL
# to bets" and "only live-with-a-bet bypasses quiet hours". Each is tested on
# its own above and in test_notify_gaps.py. Their COMPOSITION is what he
# actually experiences at 23:30 on a Sunday, and nothing exercised it:
# `live_pings_allowed` lets a bet-less Serie A goal through, and then
# `notify_goal` stamps it PRIORITY_NORMAL, and then quiet hours hold it.

def _goal_priority(monkeypatch, has_bets: bool) -> str:
    """The priority notify_goal actually stamps — not a re-derivation of it."""
    from scripts.pipeline import notify as N
    sent: list[dict] = []
    monkeypatch.setattr(N, "notify", lambda *a, **k: sent.append(k) or {})
    ctx = {"has_bets": True, "bets": [{"selection": "OVER 1.5", "odds": 1.8,
                                       "stake": 10, "is_winning": True,
                                       "commentary": "cruising"}]} if has_bets else None
    N.notify_goal("AS Roma vs Atalanta BC", "Dybala", "AS Roma",
                  home_score=1, away_score=0, minute=30, is_home=True,
                  bet_context=ctx)
    return sent[0]["priority"]


@pytest.mark.parametrize("has_bets,expect_telegram", [(False, False), (True, True)])
def test_serie_a_goal_in_quiet_hours(monkeypatch, has_bets, expect_telegram):
    """Serie A passes the league gate either way; the BET decides whether it
    survives quiet hours. Without a bet it is held until 07:00 — by design."""
    from scripts.pipeline import notify as N

    assert LM.live_pings_allowed("serie_a", "AS Roma vs Atalanta BC", SA, has_bets) is True

    priority = _goal_priority(monkeypatch, has_bets)
    monkeypatch.setattr(N, "_is_quiet_hours", lambda prefs: True)
    monkeypatch.setattr(N, "load_preferences", lambda: {
        "mute_all": False,
        "channels": {"macos": True, "telegram": True},
        "categories": {c: {"macos": True, "telegram": True} for c in N.VALID_CATEGORIES},
        "quiet_hours": {"enabled": True, "start": "23:00", "end": "07:00"},
    })
    assert N._should_send("telegram", "live", priority) is expect_telegram
    # macOS is never quieted, so the goal is still on the laptop either way.
    assert N._should_send("macos", "live", priority) is True


def test_an_epl_goal_with_a_bet_survives_both_gates(monkeypatch):
    """The one EPL case that must still reach him at 01:00: real money on it."""
    from scripts.pipeline import notify as N
    assert LM.live_pings_allowed("serie_a", "Arsenal vs Chelsea", EPL, True) is True
    priority = _goal_priority(monkeypatch, True)
    monkeypatch.setattr(N, "_is_quiet_hours", lambda prefs: True)
    monkeypatch.setattr(N, "load_preferences", lambda: {
        "mute_all": False,
        "channels": {"macos": True, "telegram": True},
        "categories": {c: {"macos": True, "telegram": True} for c in N.VALID_CATEGORIES},
    })
    assert N._should_send("telegram", "live", priority) is True


def test_every_raw_live_notify_states_its_priority():
    """The three `category="live"` sites in live_monitor were correct only by
    inheriting _DEFAULT_PRIORITY[("live","info")] == URGENT. Once priority
    became the quiet-hours gate, loosening any has_bets condition above them
    would leak an 01:00 ping with no test failing. They now say it."""
    from pathlib import Path
    src = Path("scripts/data/live_monitor.py").read_text(encoding="utf-8")
    for line_no, line in enumerate(src.splitlines(), 1):
        if 'category="live"' in line:
            window = "\n".join(src.splitlines()[max(0, line_no - 6):line_no + 6])
            assert "priority=" in window, (
                f"live_monitor.py:{line_no} sends category='live' without stating "
                "a priority — it would inherit the quiet-hours bypass"
            )


# ---------------------------------------------------------------------------
# The FT gate, EXECUTED through poll_once
# ---------------------------------------------------------------------------
#
# /verify gate 3, 2026-09-07: the FT gate was pinned by a source scan, which
# reads that the call is written but cannot see it fire, and cannot see the
# `_ft_notified` latch at all. The whole poll is data-driven — empty score and
# odds payloads make every loop before the FT block a no-op — so the branch is
# reachable by stubbing the network edges and handing it one completed match.

def _drive_poll(monkeypatch, *, match_key, match_data, has_bets, mode="serie_a"):
    """Run the real poll_once over one completed match. Returns
    (ft_cards_sent, the matchday dict poll_once persisted)."""
    from scripts.pipeline import notify as N

    matchday = {
        "date": "2026-09-13", "polls": 0, "api_calls": 0,
        "matches": {match_key: dict(match_data)}, "bet_tracking": [],
    }
    saved: list[dict] = []
    sent: list[dict] = []

    monkeypatch.setattr(LM, "get_odds_api_key", lambda: "test-key")
    monkeypatch.setattr(LM, "load_matchday", lambda *a, **k: matchday)
    monkeypatch.setattr(LM, "save_matchday", lambda d: saved.append(d))
    monkeypatch.setattr(LM, "fetch_live_scores_all_leagues", lambda *a, **k: [])
    monkeypatch.setattr(LM, "fetch_live_odds_all_leagues", lambda *a, **k: [])
    monkeypatch.setattr(LM, "_fetch_footballdata_scores", lambda *a, **k: {})
    monkeypatch.setattr(LM, "_leagues_with_active_matches", lambda *a, **k: set())
    monkeypatch.setattr(LM, "_load_active_bets", lambda: [])
    monkeypatch.setattr(LM, "_get_bet_context",
                        lambda *a, **k: {"has_bets": has_bets, "bets": []})
    monkeypatch.setattr(LM, "_goal_ping_mode", lambda: mode)
    monkeypatch.setattr(N, "notify_full_time", lambda **kw: sent.append(kw) or {})

    LM.poll_once()
    return sent, matchday


_DONE = {"status": "completed", "final_score": [2, 1], "snapshots": []}
_EPL_DONE = {**_DONE, **EPL}
_SA_DONE = {**_DONE, **SA}


def test_a_finished_epl_match_with_no_bet_posts_no_ft_card(monkeypatch):
    """The card that used to post regardless: a gated league, no money on it."""
    sent, matchday = _drive_poll(monkeypatch, match_key="Arsenal vs Chelsea",
                                 match_data=_EPL_DONE, has_bets=False)
    assert sent == [], "the FT card posted for a gated league with no bet"


def test_a_suppressed_ft_card_is_not_retried_next_cycle(monkeypatch):
    """The latch. Without it the gate is re-evaluated every 60s forever, and
    the day a bet lands on that match it fires a result card for a match that
    finished hours ago."""
    sent, matchday = _drive_poll(monkeypatch, match_key="Arsenal vs Chelsea",
                                 match_data=_EPL_DONE, has_bets=False)
    # Discriminating: on the ungated version the first cycle SENDS the card and
    # latches the flag through the normal path, so the second-cycle assertion
    # below passes there too. This line is what fails without the gate.
    assert sent == []
    assert matchday["matches"]["Arsenal vs Chelsea"]["_ft_notified"] is True

    # Second cycle over the state the first one left behind.
    sent2, _ = _drive_poll(monkeypatch, match_key="Arsenal vs Chelsea",
                           match_data=matchday["matches"]["Arsenal vs Chelsea"],
                           has_bets=True)
    assert sent2 == [], "a suppressed FT card came back on the next poll"


def test_a_finished_serie_a_match_still_posts_its_ft_card(monkeypatch):
    """The gate must not silence the league he actually bets."""
    sent, matchday = _drive_poll(monkeypatch, match_key="AS Roma vs Atalanta BC",
                                 match_data=_SA_DONE, has_bets=False)
    assert len(sent) == 1
    assert sent[0]["match_key"] == "AS Roma vs Atalanta BC"
    assert (sent[0]["home_score"], sent[0]["away_score"]) == (2, 1)
    assert matchday["matches"]["AS Roma vs Atalanta BC"]["_ft_notified"] is True


def test_a_finished_epl_match_WITH_a_bet_still_posts(monkeypatch):
    """Real money on it is exactly the case the gate must let through."""
    sent, _ = _drive_poll(monkeypatch, match_key="Arsenal vs Chelsea",
                          match_data=_EPL_DONE, has_bets=True)
    assert len(sent) == 1


def test_mode_all_posts_the_epl_card_again(monkeypatch):
    """The old behaviour is still reachable — this is a default, not a removal."""
    sent, _ = _drive_poll(monkeypatch, match_key="Arsenal vs Chelsea",
                          match_data=_EPL_DONE, has_bets=False, mode="all")
    assert len(sent) == 1
