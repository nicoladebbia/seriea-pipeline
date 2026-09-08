"""The messages that were missing, and the quiet-hours gate they answer to.

Written alongside the 2026-09-07 Telegram audit. Each test pins one defect that
was live in production:

1. `betting_unified` imported `send_notification` from `notify`, where that name
   has never existed. The ImportError went into a bare `except: pass`, so the
   engine refused to bet SILENTLY.
2. The journal rejecting a selected bet pushed nothing — the shape of the
   date-blind dedup bug that ate every real bet for nine days.
3. `market_promotion` pushed nothing at all, though it decides where real money
   goes.
4. `live` (59% of all volume) bypassed quiet hours wholesale, so a scoreline on
   a match with no bet could wake you at 01:00.

Nothing here sends: `tests/conftest.py` sets NOTIFY_DISABLED=1 and an autouse
tripwire fails any test that reaches a transport.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pytest

from scripts.pipeline import notify as N

# ---------------------------------------------------------------------------
# 1. The import that could never resolve
# ---------------------------------------------------------------------------

def test_notify_does_not_export_send_notification():
    """The name lives in scheduler.py. If someone re-adds it to notify.py the
    original bug's disguise comes back — assert the absence deliberately."""
    assert not hasattr(N, "send_notification")


def test_betting_unified_imports_a_name_that_actually_exists():
    """A true-positive test: the broken version imported `send_notification`
    from notify, so this assertion would have failed on it."""
    src = Path("scripts/betting/betting_unified.py").read_text(encoding="utf-8")
    for m in re.finditer(r"from scripts\.pipeline\.notify import ([^\n]+)", src):
        for name in (n.strip() for n in m.group(1).split(",")):
            assert hasattr(N, name), (
                f"betting_unified imports notify.{name}, which does not exist — "
                "this is the silent-bankroll-failure bug"
            )


def test_the_bankroll_alert_is_an_alert_so_it_breaks_quiet_hours():
    src = Path("scripts/betting/betting_unified.py").read_text(encoding="utf-8")
    block = src[src.index("BANKROLL LOAD FAILED"):]
    block = block[:block.index("raise RuntimeError")]
    assert 'category="alert"' in block
    assert 'level="critical"' in block
    # and the failure is no longer swallowed in silence
    assert "except Exception:\n                    pass" not in block


# ---------------------------------------------------------------------------
# 2. Journal rejection
# ---------------------------------------------------------------------------

def test_journal_rejected_says_how_many_of_how_many(monkeypatch):
    sent: list[dict] = []
    monkeypatch.setattr(N, "notify", lambda *a, **k: sent.append(k) or {})
    N.notify_journal_rejected(["Lazio vs Milan OVER 1.5 -> 2026-03-15_old_id"], 3)
    assert len(sent) == 1
    assert "1 of 3" in sent[0]["title"]
    # It must break quiet hours: this is money that did not get placed.
    assert sent[0]["category"] == "alert"
    assert sent[0]["priority"] == N.PRIORITY_URGENT
    assert "Lazio vs Milan" in sent[0]["tg_html"]


def test_journal_rejected_is_silent_when_nothing_was_blocked(monkeypatch):
    sent: list = []
    monkeypatch.setattr(N, "notify", lambda *a, **k: sent.append(k) or {})
    assert N.notify_journal_rejected([], 3) == {}
    assert not sent


def test_save_bet_slip_pushes_on_rejection():
    """The WARNING existed; the push did not."""
    src = Path("scripts/betting/betting_unified.py").read_text(encoding="utf-8")
    block = src[src.index("Journal: %d of %d bets NOT recorded"):]
    block = block[:block.index("Journal: recorded %d of %d bets")]
    assert "notify_journal_rejected" in block


# ---------------------------------------------------------------------------
# 3. The promotion gate announces itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind,expect_level", [
    ("promoted", "success"),
    ("demoted", "warning"),
    ("stake_up", "success"),
    ("stake_down", "warning"),
])
def test_market_gate_card_levels(monkeypatch, kind, expect_level):
    sent: list[dict] = []
    monkeypatch.setattr(N, "notify", lambda *a, **k: sent.append(k) or {})
    N.notify_market_promotion([
        {"kind": kind, "market": "Shots O0.5", "reason": "bar cleared",
         "n": 52, "roi_pct": 4.1, "z": 2.7},
    ])
    assert sent[0]["level"] == expect_level
    assert sent[0]["category"] == "alert"
    assert "Shots O0.5" in sent[0]["tg_html"]
    assert "n=52" in sent[0]["tg_html"]


def test_market_gate_is_silent_with_no_transitions(monkeypatch):
    sent: list = []
    monkeypatch.setattr(N, "notify", lambda *a, **k: sent.append(k) or {})
    assert N.notify_market_promotion([]) == {}
    assert not sent


def test_a_demotion_anywhere_in_the_batch_makes_the_card_a_warning(monkeypatch):
    sent: list[dict] = []
    monkeypatch.setattr(N, "notify", lambda *a, **k: sent.append(k) or {})
    N.notify_market_promotion([
        {"kind": "promoted", "market": "A", "n": 51, "roi_pct": 3.0, "z": 2.6},
        {"kind": "demoted", "market": "B", "n": 31, "roi_pct": -12.0, "z": -1.4},
    ])
    assert sent[0]["level"] == "warning"


def test_promotion_push_happens_after_the_state_write():
    """A failed notification must never cost us the state change it describes.

    Anchored on the `if write:` block, not on the whole file: the name
    `notify_market_promotion` occurs more than once (import + call), and an
    index into the whole source would silently start measuring the wrong one.
    """
    src = Path("scripts/betting/market_promotion.py").read_text(encoding="utf-8")
    block = src[src.index("    if write:\n        from config.settings import atomic_write_json"):]
    block = block[:block.index("\n    return state")]
    assert block.index("atomic_write_json(path or STATE_PATH, state)") < \
        block.index("notify_market_promotion")
    # and it fires only when something actually changed
    assert "if transitions:" in block


def test_a_stake_flip_pushes_once_and_only_once(tmp_path, monkeypatch):
    """True positive first, then idempotence.

    `evaluate_promotions` runs after EVERY settlement. Seeding the state with a
    scale that differs from the computed one forces a real `stake_down`
    transition, so this test would fail on a version that never pushed; the
    second, identical run must then push nothing, because the comparison is
    against the PERSISTED scale.
    """
    import json

    from scripts.betting import market_promotion as MP

    sent: list = []
    monkeypatch.setattr(N, "notify_market_promotion", lambda t: sent.append(t) or {})
    path = tmp_path / "market_promotion.json"
    # Nothing settled anywhere -> every incumbent computes to the halved scale.
    path.write_text(json.dumps({
        "markets": {}, "incumbents": {"ou_over_1_5": {"stake_scale": 1.0}},
    }), encoding="utf-8")
    kw = dict(paper_settled=[], real_settled=[], real_all=[], path=path, write=True)

    MP.evaluate_promotions(**kw)
    assert len(sent) == 1, "a stake change pushed no card"
    kinds = [t["kind"] for t in sent[0]]
    assert kinds == ["stake_down"], kinds
    assert sent[0][0]["market"] == "ou_over_1_5"

    MP.evaluate_promotions(**kw)
    assert len(sent) == 1, "a second identical evaluation pushed another card"


def test_a_market_at_a_steady_scale_never_pushes(tmp_path, monkeypatch):
    """The everyday case: the scale is already what it computes to."""
    import json

    from scripts.betting import market_promotion as MP

    sent: list = []
    monkeypatch.setattr(N, "notify_market_promotion", lambda t: sent.append(t) or {})
    path = tmp_path / "market_promotion.json"
    path.write_text(json.dumps({"markets": {}, "incumbents": {}}), encoding="utf-8")
    kw = dict(paper_settled=[], real_settled=[], real_all=[], path=path, write=True)
    MP.evaluate_promotions(**kw)
    MP.evaluate_promotions(**kw)
    assert not sent


# ---------------------------------------------------------------------------
# 4. Quiet hours
# ---------------------------------------------------------------------------

@pytest.fixture
def _quiet(monkeypatch):
    monkeypatch.setattr(N, "_is_quiet_hours", lambda prefs: True)
    monkeypatch.setattr(N, "load_preferences", lambda: {
        "mute_all": False,
        "channels": {"macos": True, "telegram": True},
        "categories": {c: {"macos": True, "telegram": True} for c in N.VALID_CATEGORIES},
        "quiet_hours": {"enabled": True, "start": "23:00", "end": "07:00"},
    })


def test_alert_always_breaks_quiet_hours(_quiet):
    assert N._should_send("telegram", "alert", N.PRIORITY_NORMAL) is True


def test_live_with_a_bet_breaks_quiet_hours(_quiet):
    assert N._should_send("telegram", "live", N.PRIORITY_URGENT) is True


def test_live_without_a_bet_is_held(_quiet):
    """The change: 59% of volume no longer ignores the only volume control."""
    assert N._should_send("telegram", "live", N.PRIORITY_NORMAL) is False


def test_betting_is_still_held_in_quiet_hours(_quiet):
    assert N._should_send("telegram", "betting", N.PRIORITY_NORMAL) is False


def test_macos_ignores_quiet_hours_entirely(_quiet):
    assert N._should_send("macos", "live", N.PRIORITY_NORMAL) is True


def test_outside_quiet_hours_everything_sends(monkeypatch):
    monkeypatch.setattr(N, "_is_quiet_hours", lambda prefs: False)
    monkeypatch.setattr(N, "load_preferences", lambda: {
        "mute_all": False,
        "channels": {"macos": True, "telegram": True},
        "categories": {c: {"macos": True, "telegram": True} for c in N.VALID_CATEGORIES},
    })
    assert N._should_send("telegram", "live", N.PRIORITY_NORMAL) is True


def test_full_time_fallback_sets_priority_by_bet():
    """The ('live','info') default is URGENT, so the legacy FT path would have
    leaked a no-bet scoreline through quiet hours."""
    src = Path("scripts/pipeline/notify.py").read_text(encoding="utf-8")
    tail = src[src.index("def notify_full_time"):]
    tail = tail[:tail.index("def notify_retrain")]
    assert "PRIORITY_URGENT if had_bet else PRIORITY_NORMAL" in tail


def test_inplay_paper_ping_never_breaks_quiet_hours():
    src = Path("scripts/betting/inplay.py").read_text(encoding="utf-8")
    block = src[src.index('title=f"IN-PLAY (paper)') - 400:]
    block = block[:block.index("\n\n\n")] if "\n\n\n" in block else block
    assert "PRIORITY_NORMAL" in block


# ---------------------------------------------------------------------------
# 5. Every builder renders as a Telegram card
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn,args", [
    ("notify_no_action", (["Milan vs Venezia", "Roma vs Lecce"],)),
    ("notify_journal_rejected", (["Milan OVER 1.5 -> blocked"], 2)),
])
def test_builders_send_html_not_the_macos_body(monkeypatch, fn, args):
    sent: list[dict] = []
    monkeypatch.setattr(N, "notify", lambda *a, **k: sent.append(k) or {})
    getattr(N, fn)(*args)
    assert sent[0].get("tg_html"), f"{fn} still ships the plain macOS body to Telegram"
    # any Telegram tag: the point is that this is markup and not the plain
    # macOS body. NOT "<b>" specifically -- a card whose only bold line was
    # a duplicate of the header (see test_no_card_prints_its_own_title_twice)
    # is correct with none.
    assert re.search(r"</?(b|i|code|pre|a|u|s)>", sent[0]["tg_html"]), \
        f"{fn} ships no HTML markup at all"


@pytest.mark.parametrize("promoted", [True, False])
def test_retrain_card_has_html(monkeypatch, promoted):
    sent: list[dict] = []
    monkeypatch.setattr(N, "notify", lambda *a, **k: sent.append(k) or {})
    N.notify_retrain("quick", 3, promoted, old_ll=0.99, new_ll=0.97, reason="better")
    assert sent[0].get("tg_html")
    assert "MW 3" in sent[0]["tg_html"]


# ---------------------------------------------------------------------------
# 6. The two silent paths, EXECUTED
# ---------------------------------------------------------------------------
#
# /verify gate 3, 2026-09-07: defects 1 and 2 above were pinned by SOURCE SCANS
# — they prove the call is written, not that it fires. Both failure modes are
# specifically "the branch runs and silently does nothing", which is exactly
# what a source scan cannot see. These two drive the real branch.

def test_a_bankroll_failure_actually_sends_the_alert(monkeypatch):
    """The original bug executed: get_effective_bankroll raises, and the alert
    that should follow died on an ImportError inside a bare `except: pass`.

    True positive: on the broken version `notify.send_notification` did not
    exist, so `sent` stayed empty and this fails.
    """
    from scripts.betting import bankroll_loader
    from scripts.betting.betting_unified import BettingConfig

    def _boom():
        raise ValueError("journal unreadable")

    monkeypatch.setattr(bankroll_loader, "get_effective_bankroll", _boom)
    sent: list[tuple] = []
    monkeypatch.setattr(N, "notify", lambda *a, **k: sent.append((a, k)) or {})

    with pytest.raises(RuntimeError, match="Cannot load bankroll"):
        BettingConfig.from_config(bankroll_override=None)

    assert len(sent) == 1, "the engine refused to bet in silence"
    args, kw = sent[0]
    assert kw["category"] == "alert", "must break quiet hours"
    assert kw["level"] == "critical"
    body = " ".join(str(a) for a in args) + str(kw.get("message", ""))
    assert "journal unreadable" in body, "the card must name why"


def test_a_blocked_bet_actually_pushes_from_save_bet_slip(tmp_path, monkeypatch):
    """The nine-day silence executed: the journal answers with an OLDER id, so
    the bet was not recorded, and the only trace was a WARNING in a log nobody
    reads at 20:15.

    True positive: delete the `notify_journal_rejected` call and `sent` is empty.
    """
    from scripts.betting import bet_journal
    from scripts.betting import betting_unified as BU

    bet = BU.ValueBet(
        match="Lazio vs Milan", date="2026-09-13", market="O/U 1.5",
        selection="OVER 1.5", model_prob=0.78, sharp_implied_prob=0.71,
        edge_pct=7.0, raw_edge=7.0, best_odds=1.40, best_bookmaker="Pinnacle",
        avg_odds=1.38, pinnacle_odds=1.40, odds_count=5, stake_amount=10.0,
    )
    slip = BU.BetSlip(bets=[bet], total_stake=10.0, generated_at="2026-09-13T18:00:00+00:00",
                      bankroll=1000.0, n_matches=1)

    # The store keeps a LAST-SEASON id: exactly the date-blind dedup's answer.
    monkeypatch.setattr(bet_journal, "add_bet",
                        lambda payload, **kw: "2026-03-15_Lazio_vs_Milan_OU_1.5_OVER_1.5")
    monkeypatch.setattr(BU, "UPCOMING", tmp_path)
    monkeypatch.setattr(BU, "run_paper_track", lambda *a, **k: 0)
    from scripts.betting import picks
    monkeypatch.setattr(picks, "build_picks", lambda *a, **k: None)

    sent: list[tuple] = []
    monkeypatch.setattr(N, "notify_journal_rejected",
                        lambda blocked, offered: sent.append((blocked, offered)) or {})

    BU.save_bet_slip(slip, [bet], dry_run=False)

    assert len(sent) == 1, "a rejected bet pushed nothing — the nine-day silence"
    blocked, offered = sent[0]
    assert offered == 1
    assert "Lazio vs Milan" in blocked[0]
    assert "2026-03-15" in blocked[0], "the card must name the id that swallowed it"


# ---------------------------------------------------------------------------
# 7. Telegram actually accepts the HTML
# ---------------------------------------------------------------------------
#
# On 2026-09-07 every card below was rendered to its real HTTP payload and put
# through Telegram's OWN parser with `chat_id: 0` — the API parses entities
# BEFORE it validates the chat, so a malformed card answers "can't parse
# entities" and a well-formed one answers "chat not found". All 16 payloads
# answered "chat not found"; nothing was delivered.
#
# That probe is a network call and does not belong in the suite. This is the
# offline proxy for it — Telegram's documented tag whitelist plus balance —
# and it is labelled a PROXY on purpose: it pins the failure class (an
# unescaped `<` from a team name, a bet selection, or an exception repr like
# `<class 'ValueError'>`), not Telegram's exact grammar. Re-run the live probe
# after changing the transport's escaping.
#
# Worth knowing when reading this: a card Telegram REJECTS is not lost. The
# transport retries it once with `parse_mode` removed, so a broken card arrives
# with its tags showing and only an INFO line records it.

_TG_TAGS = {"b", "strong", "i", "em", "u", "ins", "s", "strike", "del",
            "span", "tg-spoiler", "a", "code", "pre", "blockquote", "tg-emoji"}

_NASTY = 'Roma <b>& "Milan" <not-a-tag> 5>3 & <i'


def _telegram_payloads(monkeypatch, transports, build, tmp_path=None) -> list[str]:
    """The exact `text` the bot would POST, with delivery made impossible.

    Goes through the REAL `_notify_telegram` — conftest's tripwire yields the
    originals precisely so a test of the transport itself can ask for them —
    with `urllib.request.urlopen` replaced, so the bytes are the production
    bytes and not one of them leaves the process.
    """
    import urllib.request

    real_telegram, _ = transports
    seen: list[dict] = []

    def _fake_urlopen(req, timeout=None):
        seen.append(json.loads(req.data.decode()))
        raise RuntimeError("captured, never delivered")

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(N, "_notify_telegram", real_telegram)
    monkeypatch.setattr(N, "_notify_macos", lambda *a, **k: False)
    monkeypatch.setattr(N, "_sending_suppressed", lambda: False)
    monkeypatch.setattr(N, "_load_env_key",
                        lambda k: "x" if str(k).startswith("TELEGRAM_") else "")
    monkeypatch.setattr(N, "_should_send", lambda ch, cat, pri="": ch == "telegram")
    monkeypatch.setattr(N, "_record_history", lambda *a, **k: None)
    import time as _t
    monkeypatch.setattr(_t, "sleep", lambda *a, **k: None)

    build(monkeypatch, tmp_path)
    return [c["text"] for c in seen if c.get("parse_mode") == "HTML"]


def _assert_telegram_parseable(text: str, label: str) -> None:
    stack: list[str] = []
    for m in re.finditer(r"<(/?)([a-zA-Z][a-zA-Z0-9-]*)([^>]*)>", text):
        closing, tag = m.group(1), m.group(2).lower()
        assert tag in _TG_TAGS, (
            f"{label}: <{tag}> is not a tag Telegram supports — this card is "
            f"rejected and silently resent with its tags showing"
        )
        if closing:
            assert stack and stack[-1] == tag, f"{label}: </{tag}> does not close {stack[-1:]}"
            stack.pop()
        else:
            stack.append(tag)
    assert not stack, f"{label}: unclosed {stack}"
    # A bare `<` that never formed a tag is the exact byte Telegram rejects.
    for m in re.finditer(r"<(?![/a-zA-Z])", text):
        raise AssertionError(f"{label}: unescaped '<' at offset {m.start()}")


def _build_matchweek_summary(monkeypatch, tmp_path):
    """The card that actually went out at 08:00 today.

    It derives its own paths from `notify.__file__`, so redirecting that is
    what puts a deterministic journal under it — the real one holds whatever
    settled in the last 10 days, and a card built from "whatever" is a card
    that tests nothing on a quiet week.
    """
    root = tmp_path / "pkg" / "scripts" / "pipeline"
    root.mkdir(parents=True)
    (tmp_path / "pkg" / "data" / "betting").mkdir(parents=True)
    today = datetime.now().strftime("%Y-%m-%d")
    (tmp_path / "pkg" / "data" / "betting" / "bet_journal.json").write_text(json.dumps({
        "bets": {
            "b1": {"status": "won", "date": today, "match": _NASTY, "league": "serie_a",
                   "market": "O/U 1.5", "selection": "Over 1.5 <x>", "odds": 1.45,
                   "stake": 10.0, "profit": 4.5},
            "b2": {"status": "lost", "date": today, "match": "Roma vs Lecce",
                   "league": "serie_a", "market": "O/U 2.5", "selection": "Over 2.5",
                   "odds": 1.9, "stake": 10.0, "profit": -10.0},
        }
    }))
    monkeypatch.setattr(N, "__file__", str(root / "notify.py"))
    return N.notify_matchweek_summary(3)


def _build_health_state_change(monkeypatch, tmp_path):
    """Silent on the first run and it WRITES its dedup state, so the state path
    is redirected (never touch the live one) and seeded with a prior cycle."""
    state = tmp_path / "health_notify_state.json"
    state.write_text(json.dumps({
        "issue_keys": {"critical|[health_check] something old": {
            "first_seen": "2026-01-01T00:00:00", "last_seen": "2026-01-01T00:00:00",
            "level": "critical", "message": "[health_check] something old"}},
    }))
    monkeypatch.setattr(N, "_HEALTH_STATE_PATH", state)
    return N.notify_health_state_change({
        "overall_status": "critical",
        "issues": [("critical", f"[health_check] {_NASTY} lineup fetch 403 <blocked>"),
                   ("warning", "[health_check] odds stale 7.5d ago")],
    })


_CARD_BUILDERS = [
    ("no_action", lambda mp, tp: N.notify_no_action([_NASTY, "Roma vs Lecce"])),
    ("journal_rejected", lambda mp, tp: N.notify_journal_rejected([f"{_NASTY} OVER 1.5 -> old"], 3)),
    ("market_promotion", lambda mp, tp: N.notify_market_promotion([
        {"kind": "promoted", "market": _NASTY, "reason": "bar cleared",
         "n": 52, "roi_pct": 4.1, "z": 2.7}])),
    ("retrain", lambda mp, tp: N.notify_retrain("quick", 3, True, old_ll=0.99, new_ll=0.97, reason=_NASTY)),
    ("goal_with_bet", lambda mp, tp: N.notify_goal(
        _NASTY, "Dybala <10>", "AS Roma & Co", 1, 0, 30, True,
        bet_context={"has_bets": True, "bets": [{"selection": "OVER 1.5 <x>", "odds": 1.8,
                                                 "stake": 10, "is_winning": True,
                                                 "commentary": "cruising & fine"}]})),
    ("goal_no_bet", lambda mp, tp: N.notify_goal(_NASTY, "Dybala", "AS Roma", 1, 0, 30, True)),
    ("full_time_no_bet", lambda mp, tp: N.notify_full_time(_NASTY, 2, 1)),
    # The realistic carrier of a stray '<': an exception repr in a failure card.
    ("scheduler_failure", lambda mp, tp: N.notify_scheduler_failure(
        _NASTY, error="boom <class 'ValueError'> & <Response [403]>")),
    # The four below were the '?' rows of the manual audit — a text comparison
    # could not resolve them, so they are measured here instead. scheduler_run
    # was in that same '?' set and WAS a duplicate; reading is not measuring.
    # "failed", not "success": a routine-success card deliberately does not
    # send (2026-08-27 volume cut), so the success variant posts no bytes at all.
    ("scheduler_run", lambda mp, tp: N.notify_scheduler_run(
        _NASTY, "failed", details={"bets": 2, "note": "fine & dandy"})),
    ("loss_streak", lambda mp, tp: N.notify_loss_streak(
        4, 120.0, [{"match": _NASTY, "selection": "Over 1.5 <x>", "odds": 1.9,
                    "stake": 10.0, "clv": -1.2}])),
    ("clv_degradation", lambda mp, tp: N.notify_clv_degradation(0.5, 4.0, "2 weeks", [_NASTY])),
    ("matchweek_summary", _build_matchweek_summary),
    ("health_state_change", _build_health_state_change),
]


@pytest.mark.parametrize("label,build", _CARD_BUILDERS)
def test_the_bytes_we_post_are_html_telegram_accepts(monkeypatch, tmp_path, label, build,
                                                    _no_real_notifications):
    payloads = _telegram_payloads(monkeypatch, _no_real_notifications, build, tmp_path)
    assert payloads, f"{label} built no Telegram payload"
    for text in payloads:
        _assert_telegram_parseable(text, label)


@pytest.mark.parametrize("label,build", _CARD_BUILDERS)
def test_no_card_prints_its_own_title_twice(monkeypatch, tmp_path, label, build,
                                            _no_real_notifications):
    """`_notify_telegram` prepends "<emoji> <b>{title}</b>" to EVERY card, so a
    builder that also opens its body with that same string renders the header
    twice. Two did — notify_market_promotion and notify_no_action — and nobody
    saw it, because the history file logs the macOS body and never the HTML:
    the duplicate exists only in the bytes actually POSTed. Caught 2026-09-08
    by capturing them. A body MAY open with its own bold line when it says
    something DIFFERENT (notify_settlement's section heading)."""
    payloads = _telegram_payloads(monkeypatch, _no_real_notifications, build, tmp_path)
    # Without this the loop body never runs for a card that builds nothing, and
    # the test goes green having checked zero bytes.
    assert payloads, f"{label} built no Telegram payload"
    for text in payloads:
        head = re.match(r"^\S+ <b>(.+?)</b>\n", text)
        assert head, f"{label}: card does not start with the standard header"
        title, body = head.group(1), text[head.end():]
        assert f"<b>{title}</b>" not in body, (
            f"{label}: the body repeats the header {title!r} — drop the "
            f"tg.title() call, _notify_telegram already writes that line"
        )
