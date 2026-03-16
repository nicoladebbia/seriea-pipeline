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
        "edge_pct": 16.7,
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

    def test_update_pending_preserves_odds_stake(self, clean_journal, sample_bet):
        """odds/stake should NOT be overwritten on re-add (user may have adjusted)."""
        add_bet(sample_bet)
        updated = {**sample_bet, "odds": 3.50, "stake": 30.0, "model_prob": 0.45}
        add_bet(updated)
        journal = _load_journal()
        bet = list(journal["bets"].values())[0]
        # odds and stake preserved from original
        assert bet["odds"] == 3.40
        assert bet["stake"] == 25.0
        # model_prob updated (non-protected field)
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

    def test_report_contains_bankroll(self, populated_journal, monkeypatch):
        # Create a mock bankroll file
        bankroll_path = JOURNAL_PATH.parent / "bankroll.json"
        bankroll_path.parent.mkdir(parents=True, exist_ok=True)
        with open(bankroll_path, "w") as f:
            json.dump({"initial_balance": 1000.0, "current_balance": 1050.0}, f)

        report = generate_report()
        assert "BANKROLL" in report

    def test_report_contains_sections(self, populated_journal, monkeypatch):
        pending = get_pending_bets()
        settle_bet(pending[0]["bet_id"], "won", "2-1", profit=60.0)

        bankroll_path = JOURNAL_PATH.parent / "bankroll.json"
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
