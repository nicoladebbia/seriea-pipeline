"""Tests for the unified bet journal system."""

import json
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from scripts.betting.bet_journal import (
    JOURNAL_PATH,
    _generate_bet_id,
    _load_journal,
    _normalize_market,
    _save_journal,
    add_bet,
    get_journal_stats,
    get_pending_bets,
    get_settled_bets,
    generate_report,
    settle_bet,
    update_clv,
)


@pytest.fixture
def clean_journal(tmp_path, monkeypatch):
    """Redirect journal to a temp directory for isolated tests."""
    journal_path = tmp_path / "bet_journal.json"
    monkeypatch.setattr("scripts.betting.bet_journal.JOURNAL_PATH", journal_path)
    yield journal_path


@pytest.fixture
def sample_bet():
    """A single bet dict matching the journal schema."""
    return {
        "match": "Inter vs Milan",
        "date": "2026-02-15",
        "market": "1X2",
        "selection": "Draw",
        "model_prob": 0.35,
        "sharp_implied_prob": 0.30,
        "edge_pct": 8.5,
        "odds": 3.40,
        "bookmaker": "Bet365",
        "avg_odds": 3.30,
        "pinnacle_odds": 3.25,
        "stake": 25.0,
        "confidence": "MEDIUM",
        "factors": ["derby", "home_fav"],
        "placed_at": "2026-02-14T10:00:00",
    }


@pytest.fixture
def populated_journal(clean_journal, sample_bet):
    """Journal with a few bets already in it."""
    add_bet(sample_bet)
    add_bet({
        **sample_bet,
        "match": "Napoli vs Roma",
        "market": "DC",
        "selection": "1X (Home or Draw)",
        "odds": 1.39,
        "stake": 50.0,
        "confidence": "HIGH",
    })
    add_bet({
        **sample_bet,
        "match": "Genoa vs Napoli",
        "market": "O/U 2.5",
        "selection": "Over 2.5",
        "odds": 2.47,
        "stake": 29.4,
    })
    return clean_journal


# =============================================================================
# UNIT TESTS: bet_id generation
# =============================================================================

class TestBetIdGeneration:
    def test_deterministic(self):
        id1 = _generate_bet_id("2026-02-15", "Inter vs Milan", "1X2", "Draw")
        id2 = _generate_bet_id("2026-02-15", "Inter vs Milan", "1X2", "Draw")
        assert id1 == id2

    def test_different_bets_different_ids(self):
        id1 = _generate_bet_id("2026-02-15", "Inter vs Milan", "1X2", "Draw")
        id2 = _generate_bet_id("2026-02-15", "Inter vs Milan", "1X2", "Home")
        assert id1 != id2

    def test_spaces_normalized(self):
        bet_id = _generate_bet_id("2026-02-15", "Inter vs Milan", "O/U 2.5", "Over 2.5")
        assert " " not in bet_id

    def test_selection_uppercased(self):
        id1 = _generate_bet_id("2026-02-15", "Inter vs Milan", "1X2", "draw")
        id2 = _generate_bet_id("2026-02-15", "Inter vs Milan", "1X2", "DRAW")
        assert id1 == id2


# =============================================================================
# UNIT TESTS: market normalization
# =============================================================================

class TestMarketNormalization:
    def test_h2h_to_1x2(self):
        assert _normalize_market("h2h") == "1X2"
        assert _normalize_market("1X2") == "1X2"

    def test_totals_to_ou(self):
        assert _normalize_market("totals") == "O/U 2.5"

    def test_ou_preserved(self):
        assert _normalize_market("O/U 2.5") == "O/U 2.5"

    def test_spreads(self):
        assert _normalize_market("spreads") == "spreads"
        assert _normalize_market("handicap") == "spreads"


# =============================================================================
# CRUD TESTS
# =============================================================================

class TestAddBet:
    def test_add_new_bet(self, clean_journal, sample_bet):
        bet_id = add_bet(sample_bet)
        assert bet_id
        journal = _load_journal()
        assert bet_id in journal["bets"]
        assert journal["bets"][bet_id]["status"] == "pending"
        assert journal["bets"][bet_id]["odds"] == 3.40
        assert journal["bets"][bet_id]["match"] == "Inter vs Milan"

    def test_dedup_same_bet(self, clean_journal, sample_bet):
        id1 = add_bet(sample_bet)
        id2 = add_bet(sample_bet)
        assert id1 == id2
        journal = _load_journal()
        assert len(journal["bets"]) == 1

    def test_update_pending_updates_odds_stake(self, clean_journal, sample_bet):
        """odds/stake SHOULD be updated on re-add (pipeline re-runs propagate config changes)."""
        add_bet(sample_bet)
        updated = {**sample_bet, "odds": 3.50, "stake": 30.0, "model_prob": 0.45}
        add_bet(updated)
        journal = _load_journal()
        bet = list(journal["bets"].values())[0]
        # odds and stake updated from latest pipeline run
        assert bet["odds"] == 3.50
        assert bet["stake"] == 30.0
        # model_prob also updated
        assert bet["model_prob"] == 0.45

    def test_no_overwrite_settled_bet(self, clean_journal, sample_bet):
        bet_id = add_bet(sample_bet)
        settle_bet(bet_id, "won", "2-1", profit=60.0)
        add_bet({**sample_bet, "odds": 5.0})
        journal = _load_journal()
        assert journal["bets"][bet_id]["odds"] == 3.40  # unchanged


class TestGetPendingBets:
    def test_all_pending(self, populated_journal):
        pending = get_pending_bets()
        assert len(pending) == 3

    def test_filter_by_date(self, populated_journal):
        pending = get_pending_bets(match_date="2026-02-15")
        assert len(pending) == 3  # all bets in fixture have same date

    def test_filter_excludes_other_dates(self, populated_journal):
        pending = get_pending_bets(match_date="2026-01-01")
        assert len(pending) == 0

    def test_excludes_settled(self, populated_journal):
        pending = get_pending_bets()
        bet_id = pending[0]["bet_id"]
        settle_bet(bet_id, "won", "2-1", profit=50.0)
        pending_after = get_pending_bets()
        assert len(pending_after) == 2


class TestSettleBet:
    def test_settle_won(self, clean_journal, sample_bet):
        bet_id = add_bet(sample_bet)
        result = settle_bet(bet_id, "won", "2-1", profit=60.0)
        assert result is True
        journal = _load_journal()
        bet = journal["bets"][bet_id]
        assert bet["status"] == "won"
        assert bet["profit"] == 60.0
        assert bet["result_score"] == "2-1"
        assert bet["settled_at"] is not None

    def test_settle_lost(self, clean_journal, sample_bet):
        bet_id = add_bet(sample_bet)
        result = settle_bet(bet_id, "lost", "0-1", profit=-25.0)
        assert result is True
        journal = _load_journal()
        assert journal["bets"][bet_id]["status"] == "lost"

    def test_settle_push(self, clean_journal, sample_bet):
        bet_id = add_bet(sample_bet)
        result = settle_bet(bet_id, "push", "0-0", profit=0.0)
        assert result is True
        journal = _load_journal()
        assert journal["bets"][bet_id]["status"] == "push"

    def test_settle_invalid_status(self, clean_journal, sample_bet):
        bet_id = add_bet(sample_bet)
        result = settle_bet(bet_id, "invalid")
        assert result is False

    def test_settle_nonexistent_bet(self, clean_journal):
        result = settle_bet("fake_id", "won")
        assert result is False

    def test_no_double_settle(self, clean_journal, sample_bet):
        bet_id = add_bet(sample_bet)
        settle_bet(bet_id, "won", "2-1", profit=60.0)
        result = settle_bet(bet_id, "lost", "0-1", profit=-25.0)
        assert result is False
        journal = _load_journal()
        assert journal["bets"][bet_id]["status"] == "won"  # unchanged


class TestUpdateCLV:
    def test_update_clv_manual(self, clean_journal, sample_bet):
        bet_id = add_bet(sample_bet)
        result = update_clv(bet_id, closing_odds=3.20, clv_pct=6.25)
        assert result is True
        journal = _load_journal()
        bet = journal["bets"][bet_id]
        assert bet["closing_odds"] == 3.20
        assert bet["clv_pct"] == 6.25

    def test_update_clv_auto_compute(self, clean_journal, sample_bet):
        bet_id = add_bet(sample_bet)
        # bet odds = 3.40, closing = 3.20 → CLV = (3.40/3.20)-1 = +6.25%
        result = update_clv(bet_id, closing_odds=3.20)
        assert result is True
        journal = _load_journal()
        bet = journal["bets"][bet_id]
        assert bet["closing_odds"] == 3.20
        assert abs(bet["clv_pct"] - 6.25) < 0.01

    def test_update_clv_nonexistent(self, clean_journal):
        result = update_clv("fake_id", 3.20)
        assert result is False


class TestGetSettledBets:
    def test_empty_initially(self, clean_journal):
        assert get_settled_bets() == []

    def test_returns_settled_only(self, populated_journal):
        pending = get_pending_bets()
        settle_bet(pending[0]["bet_id"], "won", "2-1", profit=50.0)
        settled = get_settled_bets()
        assert len(settled) == 1
        assert settled[0]["status"] == "won"


# =============================================================================
# STATISTICS TESTS
# =============================================================================

class TestJournalStats:
    def test_empty_journal(self, clean_journal):
        stats = get_journal_stats()
        assert stats["total_bets"] == 0

    def test_stats_with_settled_bets(self, populated_journal):
        pending = get_pending_bets()
        # Find specific bets by match name for deterministic assertions
        inter_bet = next(b for b in pending if b["match"] == "Inter vs Milan")
        napoli_bet = next(b for b in pending if b["match"] == "Napoli vs Roma")
        settle_bet(inter_bet["bet_id"], "won", "2-1", profit=60.0)
        settle_bet(napoli_bet["bet_id"], "lost", "0-1", profit=-50.0)

        stats = get_journal_stats()
        assert stats["total_bets"] == 3
        assert stats["settled"] == 2
        assert stats["pending"] == 1
        assert stats["won"] == 1
        assert stats["lost"] == 1
        assert stats["total_profit"] == 10.0
        assert stats["total_staked"] == 75.0  # 25 + 50
        assert stats["roi_pct"] == pytest.approx(13.33, abs=0.01)

    def test_stats_by_market(self, populated_journal):
        pending = get_pending_bets()
        inter_bet = next(b for b in pending if b["match"] == "Inter vs Milan")
        settle_bet(inter_bet["bet_id"], "won", "2-1", profit=60.0)

        stats = get_journal_stats()
        assert "1X2" in stats["by_market"]
        assert stats["by_market"]["1X2"]["wins"] == 1


# =============================================================================
# REPORT TESTS
# =============================================================================

class TestReport:
    def test_empty_report(self, clean_journal):
        report = generate_report()
        assert "No bets" in report

    def test_report_contains_bankroll(self, populated_journal, monkeypatch, tmp_path):
        # Redirect DATA_DIR, don't derive the path from JOURNAL_PATH: the name
        # imported into this module at line 10 is a *copy* that clean_journal's
        # monkeypatch (which sets the attribute on scripts.betting.bet_journal)
        # does not touch. Deriving from it wrote the REAL data/betting/
        # bankroll.json and overwrote the live ledger with this fixture.
        monkeypatch.setattr("scripts.betting.bet_journal.DATA_DIR", tmp_path)
        bankroll_path = tmp_path / "betting" / "bankroll.json"
        bankroll_path.parent.mkdir(parents=True, exist_ok=True)
        with open(bankroll_path, "w") as f:
            json.dump({"initial_balance": 1000.0, "current_balance": 1050.0}, f)

        report = generate_report()
        assert "BANKROLL" in report

    def test_report_contains_sections(self, populated_journal, monkeypatch, tmp_path):
        pending = get_pending_bets()
        settle_bet(pending[0]["bet_id"], "won", "2-1", profit=60.0)

        # See test_report_contains_bankroll: redirect DATA_DIR rather than
        # deriving from the module-level JOURNAL_PATH copy.
        monkeypatch.setattr("scripts.betting.bet_journal.DATA_DIR", tmp_path)
        bankroll_path = tmp_path / "betting" / "bankroll.json"
        bankroll_path.parent.mkdir(parents=True, exist_ok=True)
        with open(bankroll_path, "w") as f:
            json.dump({"initial_balance": 1000.0, "current_balance": 1060.0}, f)

        report = generate_report(days=30)
        assert "BY MARKET:" in report
        assert "BY CONFIDENCE:" in report
        assert "BY EDGE BUCKET:" in report


# =============================================================================
# JOURNAL I/O TESTS
# =============================================================================

class TestJournalIO:
    def test_atomic_save(self, clean_journal):
        journal = _load_journal()
        journal["bets"]["test"] = {"bet_id": "test", "status": "pending"}
        _save_journal(journal)

        # Verify no .tmp file left behind
        tmp_path = clean_journal.with_suffix(".tmp")
        assert not tmp_path.exists()

        # Verify journal is readable
        reloaded = _load_journal()
        assert "test" in reloaded["bets"]

    def test_metadata_updated_on_save(self, clean_journal):
        journal = _load_journal()
        _save_journal(journal)
        reloaded = _load_journal()
        assert "updated_at" in reloaded["metadata"]
        assert reloaded["metadata"]["total_bets"] == 0

    def test_load_corrupted_file(self, clean_journal):
        """If journal is corrupted, load returns empty journal."""
        clean_journal.parent.mkdir(parents=True, exist_ok=True)
        with open(clean_journal, "w") as f:
            f.write("{invalid json")
        journal = _load_journal()
        assert journal["bets"] == {}


# =============================================================================
# CLV COMPUTATION TESTS
# =============================================================================

class TestCLVComputation:
    def test_compute_clv_positive(self):
        """Bet at 3.50, closes at 3.30 → positive CLV (we beat the market)."""
        from scripts.betting.clv_tracker import compute_clv
        clv = compute_clv(3.50, 3.30)
        assert clv > 0
        assert abs(clv - 0.0606) < 0.001

    def test_compute_clv_negative(self):
        """Bet at 3.50, closes at 3.70 → negative CLV (market moved against us)."""
        from scripts.betting.clv_tracker import compute_clv
        clv = compute_clv(3.50, 3.70)
        assert clv < 0

    def test_compute_clv_dc_positive(self):
        """DC bet at 1.39, closes at 1.29 → positive CLV."""
        from scripts.betting.clv_tracker import compute_clv
        clv = compute_clv(1.39, 1.29)
        assert clv > 0
        assert abs(clv - 0.0775) < 0.001

    def test_compute_clv_zero(self):
        from scripts.betting.clv_tracker import compute_clv
        clv = compute_clv(3.0, 3.0)
        assert clv == 0.0

    def test_compute_clv_invalid_odds(self):
        from scripts.betting.clv_tracker import compute_clv
        assert compute_clv(0.5, 3.0) == 0.0
        assert compute_clv(3.0, 0.5) == 0.0


def test_last_seasons_settled_bet_never_blocks_this_seasons(tmp_path):
    """Serie A fixtures repeat every season. Until 2026-09-05 the
    match+selection duplicate guard was date-blind: the settled March
    'Lazio vs Milan Over 1.5' swallowed the September one (the T-30 run logged
    'recorded 1 bets', the journal took nothing). Same date + same selection
    under a market-name variant is still one bet."""
    import scripts.betting.bet_journal as BJ
    jp = tmp_path / "journal.json"
    base = {"match": "Lazio vs Milan", "market": "O/U 1.5", "selection": "Over 1.5",
            "odds": 1.3, "stake": 5.0, "edge_pct": 4.0}
    old = BJ.add_bet(dict(base, date="2026-03-15"), journal_path=jp)
    j = BJ._load_journal(jp)
    j["bets"][old]["status"] = "lost"
    BJ._save_journal(j, jp)
    new = BJ.add_bet(dict(base, date="2026-09-06"), journal_path=jp)
    assert new != old and new.startswith("2026-09-06")
    assert set(BJ._load_journal(jp)["bets"]) == {old, new}
    # same fixture, same date, market spelled differently -> the guard still holds
    variant = BJ.add_bet(dict(base, date="2026-09-06", market="OU_1.5"), journal_path=jp)
    assert variant == new and len(BJ._load_journal(jp)["bets"]) == 2


# =============================================================================
# CLV: beat-the-close vs the line actually MOVING (2026-09-08)
# =============================================================================

class TestCLVIsNotTheEntryEdge:
    """Until 2026-09-08 one field, `clv_pct`, held three different quantities:
    a probability difference against the closing line, a percent return against
    the closing line, and — when no closing line existed — the entry edge
    against Pinnacle, which contains no information about how the line moved.
    The incumbent stake ladder read that field."""

    def test_no_closing_line_means_no_CLV_only_an_entry_edge(self):
        from scripts.betting.bet_journal import _compute_clv
        bet = {"odds": 2.00, "sharp_implied_prob": 0.55, "closing_odds": None}
        _compute_clv(bet)
        # the true positive: the OLD rule wrote exactly this number into clv_pct
        assert bet["entry_edge_vs_sharp_pct"] == round((0.55 - 1 / 2.00) * 100, 2) == 5.0
        assert bet.get("clv_pct") is None, "an entry-time edge is not closing-line value"
        assert bet.get("clv_move_pct") is None

    def test_the_line_move_needs_the_SAME_book_at_both_ends(self):
        from scripts.betting.bet_journal import _compute_clv
        base = {"odds": 1.41, "pinnacle_odds": 1.38, "closing_odds": 1.34,
                "entry_sharp_odds": 1.38, "entry_sharp_book": "Pinnacle"}
        for source, why in (
            (None, "captured before the source was recorded"),
            ("totals.1.5.over (9 bm) [market_mean]", "a mean of many books is a different reference"),
            ("totals.1.5.over (summary)", "a summary field names no book"),
            ("totals.1.5.over (9 bm) [Bet365]", "a soft book is not the sharp line"),
        ):
            bet = dict(base, closing_source=source)
            _compute_clv(bet)
            assert bet.get("clv_move_pct") is None, why
            assert bet["clv_pct"] is not None, "beat-the-close still stands"

        # ...and the ENTRY side needs a named book too. Every legacy row carries
        # `pinnacle_odds` that is really the alt-totals market MEAN, so a Pinnacle
        # close against it is a change of reference, not a line that moved. The
        # true positive: the rule this replaces read exactly that field.
        no_entry_book = dict(base, closing_source="totals.1.5.over (9 bm) [Pinnacle]")
        no_entry_book.pop("entry_sharp_book"), no_entry_book.pop("entry_sharp_odds")
        _compute_clv(no_entry_book)
        assert no_entry_book["pinnacle_odds"] == 1.38, "the old rule's input is right there"
        assert no_entry_book.get("clv_move_pct") is None

        mismatched = dict(base, entry_sharp_book="Matchbook",
                          closing_source="totals.1.5.over (9 bm) [Pinnacle]")
        _compute_clv(mismatched)
        assert mismatched.get("clv_move_pct") is None, "two sharp books are still two books"

        bet = dict(base, closing_source="totals.1.5.over (9 bm) [Pinnacle]")
        _compute_clv(bet)
        assert bet["clv_move_pct"] == round((1 / 1.34 - 1 / 1.38) * 100, 2)
        # and it is a DIFFERENT number from beat-the-close, which also carries
        # the same-moment spread between our 1.41 and Pinnacle's 1.38
        assert bet["clv_move_pct"] != bet["clv_pct"]

    def test_the_move_is_negative_when_the_market_goes_against_us(self):
        from scripts.betting.bet_journal import _compute_clv
        bet = {"odds": 1.41, "pinnacle_odds": 1.38, "closing_odds": 1.45,
               "entry_sharp_odds": 1.38, "entry_sharp_book": "Pinnacle",
               "closing_source": "totals.1.5.over (9 bm) [Pinnacle]"}
        _compute_clv(bet)
        assert bet["clv_move_pct"] < 0
        # beat-the-close is negative here too — the point of the pair is that on
        # the live journal it almost never was, because we shop and the close is
        # one book. This asserts the sign logic, not the live distribution.
        assert bet["clv_pct"] < 0

    def test_backfill_relocates_an_old_entry_edge_and_leaves_a_real_CLV_alone(self, clean_journal, sample_bet):
        from scripts.betting.bet_journal import _load_journal, _save_journal, add_bet, backfill_clv, settle_bet
        bid = add_bet(sample_bet)
        settle_bet(bid, "won", result_score="1-1")
        j = _load_journal()
        # a row as the old rule left it: entry-vs-sharp value sitting in clv_pct
        j["bets"][bid].update(closing_odds=None, clv_pct=1.47, clv_move_pct=None,
                              entry_edge_vs_sharp_pct=None)
        j["bets"]["real"] = dict(j["bets"][bid], bet_id="real", clv_pct=2.4,
                                 closing_odds=3.20, pinnacle_odds=3.25,
                                 entry_sharp_odds=3.25, entry_sharp_book="Pinnacle",
                                 closing_source="h2h.draw (9 bookmakers) [Pinnacle]",
                                 clv_move_pct=None, entry_edge_vs_sharp_pct=None)
        _save_journal(j)

        backfill_clv()
        out = _load_journal()["bets"]
        assert out[bid].get("clv_pct") is None
        assert out[bid]["entry_edge_vs_sharp_pct"] == 1.47, "renamed, not recomputed"
        assert out["real"]["clv_pct"] == 2.4, "a displayed metric is never rewritten in place"
        assert out["real"]["clv_move_pct"] == round((1 / 3.20 - 1 / 3.25) * 100, 2)

    def _pending_with_a_close(self, clean_journal, when, monkeypatch):
        """A journalled bet that already carries a captured close, plus a fresh
        Pinnacle price for the same fixture kicking off at `when`."""
        from datetime import timedelta
        from scripts.betting import clv_capture
        from scripts.utils.match_timing import now_utc

        add_bet({
            "match": "Inter vs Milan", "date": "2026-02-15", "market": "O/U 1.5",
            "selection": "Over 1.5", "model_prob": 0.74, "sharp_implied_prob": 0.7246,
            "edge_pct": 1.5, "odds": 1.41, "bookmaker": "Bet365", "avg_odds": 1.40,
            "pinnacle_odds": 1.38, "entry_sharp_odds": 1.38,
            "entry_sharp_book": "Pinnacle", "stake": 10.0, "confidence": "MEDIUM",
            "factors": [], "placed_at": "2026-02-14T10:00:00",
        })
        bet_id = next(iter(_load_journal()["bets"]))
        # the price the T-30 cycle read off the very snapshot the bet was priced on
        update_clv(bet_id, closing_odds=1.38, clv_pct=2.17,
                   closing_source="totals.1.5.over (9 bm) [Pinnacle]")
        ko = (now_utc() + timedelta(hours=when)).isoformat().replace("+00:00", "Z")
        odds = {"Inter vs Milan": {
            "commence_time": ko,
            "totals": [{"line": 1.5, "all_bookmakers": [
                {"bookmaker": "Bet365", "over": 1.40},
                {"bookmaker": "Pinnacle", "over": 1.34},
            ]}],
        }}
        monkeypatch.setattr(clv_capture, "_load_cached_odds", lambda: odds)
        monkeypatch.setattr(clv_capture, "_append_clv_history", lambda *a, **k: None)
        return bet_id

    def test_the_close_is_the_LAST_pre_kickoff_price_not_the_first(
        self, clean_journal, monkeypatch
    ):
        from scripts.betting.clv_capture import capture_clv
        bet_id = self._pending_with_a_close(clean_journal, when=+2, monkeypatch=monkeypatch)

        # the true positive: the rule this replaces skipped on exactly this state
        assert _load_journal()["bets"][bet_id]["clv_pct"] is not None

        summary = capture_clv(from_cache=True)
        bet = _load_journal()["bets"][bet_id]
        assert summary["captured"] == 1
        assert bet["closing_odds"] == 1.34, "kickoff is still ahead — the line moved, take it"
        assert bet["clv_move_pct"] == round((1 / 1.34 - 1 / 1.38) * 100, 2)
        assert bet["clv_pct"] == round((1.41 / 1.34 - 1) * 100, 2)

    def test_a_price_read_after_kickoff_never_overwrites_the_close(
        self, clean_journal, monkeypatch
    ):
        from scripts.betting.clv_capture import capture_clv
        bet_id = self._pending_with_a_close(clean_journal, when=-2, monkeypatch=monkeypatch)

        summary = capture_clv(from_cache=True)
        bet = _load_journal()["bets"][bet_id]
        assert summary["captured"] == 0 and summary["skipped"] == 1
        assert bet["closing_odds"] == 1.38, "in-play/stale quote is not a closing line"

    def test_an_unknown_kickoff_is_treated_as_past(self, clean_journal, monkeypatch):
        from scripts.betting.clv_capture import _is_pre_kickoff
        assert _is_pre_kickoff({}) is False
        assert _is_pre_kickoff({"commence_time": "not a date"}) is False

    def test_the_close_for_OU_1_5_comes_from_alternate_totals(self):
        """The bulk feed's `totals` carries 2.0/2.25/2.5 only. O/U 1.5 — the line
        this system actually bets — is in alternate_totals, so a matcher that reads
        `totals` alone never matched the money market at all."""
        from scripts.betting.clv_capture import _alt_line, _match_bet_to_odds
        # the live feed writes "1.0" / "2.0"; "2" would be just as valid, so the
        # lookup matches the parsed value and never the string shape
        alt = {"1.0": {"n": 1}, "1.5": {"n": 2}, "2.0": {"n": 3}}
        assert (_alt_line(alt, 1.5), _alt_line(alt, 2.0), _alt_line(alt, 3.5)) == (
            {"n": 2}, {"n": 3}, None)
        assert _alt_line({"2": {"n": 4}}, 2.0) == {"n": 4}

        bet = {"match": "Lazio vs Milan", "market": "O/U 1.5", "selection": "Over 1.5"}
        headline_only = {"Lazio vs Milan": {"totals": [{"line": 2.5, "all_bookmakers": [
            {"bookmaker": "Pinnacle", "over": 1.90}]}]}}
        assert _match_bet_to_odds(bet, headline_only) is None, "1.5 is not 2.5"

        odds = {"Lazio vs Milan": {
            "totals": [{"line": 2.5, "all_bookmakers": [{"bookmaker": "Pinnacle", "over": 1.90}]}],
            "alternate_totals": {"1.5": {
                "over": 1.36, "best_over": 1.40, "bookmakers_count": 3,
                "all_bookmakers": [{"bookmaker": "Bet365", "over": 1.40},
                                   {"bookmaker": "Pinnacle", "over": 1.34}],
            }},
        }}
        closing, source = _match_bet_to_odds(bet, odds)
        assert closing == 1.34 and source.endswith("[Pinnacle]")

        # without per-book prices the line can still be read, but never as a book
        no_books = {"Lazio vs Milan": {"totals": [], "alternate_totals": {
            "1.5": {"over": 1.36, "best_over": 1.40, "bookmakers_count": 3}}}}
        closing, source = _match_bet_to_odds(bet, no_books)
        # the market MEAN, not best_over: a max over N books is an extreme, not a
        # line, and it would make our CLV negative by construction
        assert closing == 1.36 and source.endswith("(summary)")
        from scripts.betting.bet_journal import closing_book
        assert closing_book({"closing_source": source}) is None

    def test_a_settlement_time_writer_cannot_clobber_a_tagged_close(self, clean_journal):
        add_bet({
            "match": "Inter vs Milan", "date": "2026-02-15", "market": "O/U 1.5",
            "selection": "Over 1.5", "model_prob": 0.74, "sharp_implied_prob": 0.72,
            "edge_pct": 1.5, "odds": 1.41, "bookmaker": "Bet365", "avg_odds": 1.40,
            "pinnacle_odds": 1.38, "entry_sharp_odds": 1.38,
            "entry_sharp_book": "Pinnacle", "stake": 10.0, "confidence": "MEDIUM",
            "factors": [], "placed_at": "2026-02-14T10:00:00",
        })
        bet_id = next(iter(_load_journal()["bets"]))
        update_clv(bet_id, closing_odds=1.34,
                   closing_source="totals.1.5.over (9 bm) [Pinnacle]")
        assert _load_journal()["bets"][bet_id]["clv_move_pct"] is not None

        # clv_tracker rewrites closing prices from settlement-time snapshots with no
        # book behind them. A price read after the match must not displace the tagged
        # one read before kickoff — that would wipe the movement leg on exactly the
        # settled population the stake ladder scores.
        update_clv(bet_id, closing_odds=1.36)
        bet = _load_journal()["bets"][bet_id]
        assert bet["closing_odds"] == 1.34
        assert bet["closing_source"].endswith("[Pinnacle]")
        assert bet["clv_move_pct"] == round((1 / 1.34 - 1 / 1.38) * 100, 2)

        # but an untagged price on a row that never had a tag is still taken, and it
        # yields no movement number
        add_bet({"match": "Roma vs Lazio", "date": "2026-02-16", "market": "O/U 1.5",
                 "selection": "Over 1.5", "model_prob": 0.74, "sharp_implied_prob": 0.72,
                 "edge_pct": 1.5, "odds": 1.41, "bookmaker": "Bet365", "avg_odds": 1.40,
                 "pinnacle_odds": 1.38, "entry_sharp_odds": 1.38,
            "entry_sharp_book": "Pinnacle", "stake": 10.0, "confidence": "MEDIUM",
                 "factors": [], "placed_at": "2026-02-15T10:00:00"})
        other = [b for b in _load_journal()["bets"] if "Roma" in b][0]
        update_clv(other, closing_odds=1.36)
        row = _load_journal()["bets"][other]
        assert row["closing_source"] is None and row["clv_move_pct"] is None
        assert row["clv_pct"] == round((1.41 / 1.36 - 1) * 100, 2)

    def test_capture_names_the_book_it_read_the_close_from(self):
        from scripts.betting.clv_capture import _find_sharp_odds, _match_bet_to_odds
        books = [{"bookmaker": "Bet365", "over": 1.44}, {"bookmaker": "Pinnacle", "over": 1.38}]
        assert _find_sharp_odds(books, "over") == (1.38, "Pinnacle")
        assert _find_sharp_odds([{"bookmaker": "Bet365", "over": 1.44}], "over") == (1.44, "market_mean")

        odds = {"Inter vs Milan": {"totals": [{"line": 1.5, "all_bookmakers": books}]}}
        got = _match_bet_to_odds({"match": "Inter vs Milan", "market": "O/U 1.5",
                                  "selection": "Over 1.5"}, odds)
        assert got[0] == 1.38 and got[1].endswith("[Pinnacle]")
