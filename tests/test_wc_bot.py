"""Tests for the WC Telegram bot's bet tracker — leg settlement math and the
stake@odds parser. Pure functions, no IO."""

from __future__ import annotations

import re

import pytest

from scripts.pipeline.telegram_bot import WC_LEG_EVAL, _wc_grid


class TestLegEval:
    @pytest.mark.parametrize("key,score,expected", [
        ("1", (2, 1), True), ("1", (1, 1), False), ("1", (0, 1), False),
        ("X", (1, 1), True), ("X", (2, 1), False),
        ("2", (0, 1), True), ("2", (1, 1), False),
        ("1X", (1, 1), True), ("1X", (0, 1), False),
        ("X2", (1, 1), True), ("X2", (2, 1), False),
        ("12", (2, 1), True), ("12", (1, 1), False),
        ("O15", (1, 1), True), ("O15", (1, 0), False),
        ("U25", (1, 1), True), ("U25", (2, 1), False),
        ("O25", (2, 1), True), ("O25", (1, 1), False),
        ("U35", (2, 1), True), ("U35", (3, 1), False),
        ("BTTSY", (2, 1), True), ("BTTSY", (2, 0), False),
        ("BTTSN", (2, 0), True), ("BTTSN", (2, 1), False),
        ("1U35", (2, 0), True), ("1U35", (3, 1), False), ("1U35", (1, 1), False),
        ("1O15", (2, 1), True), ("1O15", (1, 0), False),
        ("2U35", (0, 2), True), ("2U35", (1, 3), False),
        ("1orO25", (1, 0), True), ("1orO25", (1, 2), True), ("1orO25", (0, 1), False),
    ])
    def test_settlement_truth_table(self, key, score, expected):
        assert WC_LEG_EVAL[key](*score) is expected

    def test_korea_czechia_2_1(self):
        """Real case: Korea 2-1 — his Over 2.5 won, our BTTS No lost."""
        assert WC_LEG_EVAL["O25"](2, 1) is True
        assert WC_LEG_EVAL["BTTSN"](2, 1) is False
        assert WC_LEG_EVAL["1"](2, 1) is True
        assert WC_LEG_EVAL["U35"](2, 1) is True

    def test_eval_probs_sum_with_grid(self):
        """Complementary legs must partition the grid mass (grid truncates at
        9 goals a side, so the partition target is the grid total, not 1.0)."""
        g = _wc_grid(1.78, 0.834)
        total = sum(g.values())
        assert total == pytest.approx(1.0, abs=1e-3)   # truncation is tiny
        for a_key, b_key in (("O25", "U25"), ("BTTSY", "BTTSN"), ("1X", "2")):
            pa = sum(v for (h, a), v in g.items() if WC_LEG_EVAL[a_key](h, a))
            pb = sum(v for (h, a), v in g.items() if WC_LEG_EVAL[b_key](h, a))
            assert pa + pb == pytest.approx(total, abs=1e-9)


from scripts.pipeline.telegram_bot import _wc_parse_bet_text


class TestLenientBetParse:
    """The conversational parser: Nicola types like a human, the bot fills
    in whichever piece is missing across messages."""

    @pytest.mark.parametrize("text,pend,stake,odds", [
        ("60 @ 1.80", {}, 60.0, 1.80),
        ("60 at 1.80", {}, 60.0, 1.80),
        ("I bet Canada winning at 1.80", {}, None, 1.80),     # odds only → ask stake
        ("60", {"odds": 1.80}, 60.0, 1.80),                   # stake completes it
        ("1.85", {"stake": 5.0}, 5.0, 1.85),                  # odds completes it
        ("60", {}, 60.0, None),                               # stake only → ask odds
        ("1.80 for my 60", {}, 60.0, 1.80),                   # swapped word order
        ("5 at 1.85", {}, 5.0, 1.85),                         # both small numbers
        ("put 12,50 on it at 2,43", {}, 12.5, 2.43),
    ])
    def test_extraction(self, text, pend, stake, odds):
        s, o, _w = _wc_parse_bet_text(text, dict(pend))
        assert s == stake and o == odds

    def test_words_become_label(self):
        _s, _o, words = _wc_parse_bet_text("I bet Canada winning at 1.80", {})
        assert "Canada winning" in words

    def test_market_symbol_digits_are_not_money(self):
        """'X2 ... odds are 1.95' — the 2 in X2 must not become a stake."""
        s, o, _w = _wc_parse_bet_text(
            "I want to bet on X2 in Canada vs bosnia, and the odds are 1.95", {})
        assert s is None and o == 1.95

    def test_over_line_digits_are_not_money(self):
        s, o, _w = _wc_parse_bet_text("bet over 2.5 at 2.10", {})
        assert s is None and o == 2.10


class TestLadderMemory:
    """The ladder is DERIVED from the settled journal (self-healing) — the
    trailing win-streak counts back from the most recent settled bet."""

    @pytest.fixture(autouse=True)
    def _sandbox(self, tmp_path, monkeypatch):
        import scripts.pipeline.telegram_bot as tb
        monkeypatch.setattr(tb, "WC_LADDER_STATE_JSON", tmp_path / "ladder.json")
        monkeypatch.setattr(tb, "WC_MYBETS_JSON", tmp_path / "bets.json")
        self.tb = tb

    def _journal(self, *settled):
        """settled = list of (status, stake, odds) in chronological order."""
        import json
        rows = []
        for i, (status, stake, odds) in enumerate(settled):
            rows.append({"id": i + 1, "label": "x", "stake": stake, "odds": odds,
                         "status": status, "settled_at": f"2026-06-12T{i:02d}:00:00"})
        (self.tb.WC_MYBETS_JSON).write_text(json.dumps(rows))

    def test_win_sets_next_rung_to_return(self):
        self._journal(("won", 60.0, 1.80))
        st = self.tb._wc_ladder_state()
        assert st["rung"] == pytest.approx(108.0)
        assert st["streak"] == 1

    def test_consecutive_wins_compound(self):
        self._journal(("won", 60.0, 1.80), ("won", 108.0, 1.50))
        st = self.tb._wc_ladder_state()
        assert st["rung"] == pytest.approx(162.0)
        assert st["streak"] == 2

    def test_loss_resets(self):
        self._journal(("won", 60.0, 1.80), ("lost", 108.0, 1.50))
        st = self.tb._wc_ladder_state()
        assert st["rung"] is None
        assert st["streak"] == 0

    def test_lost_won_lost_is_streak_zero(self):
        """The exact 2026-06-15 bug: lost-won-lost must derive streak 0, NOT
        carry a phantom streak from an old incremental counter."""
        self._journal(("lost", 120.0, 2.0), ("won", 8.0, 2.05), ("lost", 16.0, 1.17))
        st = self.tb._wc_ladder_state()
        assert st["streak"] == 0 and st["rung"] is None

    def test_self_heals_from_corrupt_cache(self):
        """A corrupt stored streak-7 is ignored — journal wins."""
        import json
        self.tb.WC_LADDER_STATE_JSON.write_text(json.dumps({"rung": 18.0, "streak": 7}))
        self._journal(("lost", 16.0, 1.17))
        st = self.tb._wc_ladder_state()
        assert st["streak"] == 0


FAKE_P = {
    "match_number": 3, "home_team": "Canada", "away_team": "Bosnia and Herzegovina",
    "probabilities": {"home": 0.5981, "draw": 0.2297, "away": 0.1722},
    "home_xg": 1.78, "away_xg": 0.834, "kickoff_utc": "2026-06-12T19:00:00+00:00",
}


class TestOddsSheetScanner:
    def test_parses_his_exact_example(self):
        from scripts.pipeline.telegram_bot import _wc_parse_odds_sheet
        pairs = _wc_parse_odds_sheet(FAKE_P, "X2 is at 1.50, 1 is at 2.00, 2 is at 3.00")
        keys = {k: o for k, _l, _p, o in pairs}
        assert keys == {"X2": 1.50, "1": 2.00, "2": 3.00}

    def test_ev_ranking_picks_the_value_leg(self):
        from scripts.pipeline.telegram_bot import _wc_parse_odds_sheet, _wc_scan_odds_sheet
        pairs = _wc_parse_odds_sheet(FAKE_P, "X2 at 1.50, 1 at 2.00, 2 at 3.00")
        msg, kb = _wc_scan_odds_sheet(FAKE_P, pairs)
        assert "Best of this menu: 1 (Canada win) @ 2.0" in msg
        assert msg.count("❌") == 2 and msg.count("✅") == 1
        assert kb and len(kb["inline_keyboard"]) == 1   # only the +EV leg gets a button

    def test_all_bad_menu_says_skip(self):
        from scripts.pipeline.telegram_bot import _wc_parse_odds_sheet, _wc_scan_odds_sheet
        pairs = _wc_parse_odds_sheet(FAKE_P, "X2 at 1.50, 2 at 3.00")
        msg, kb = _wc_scan_odds_sheet(FAKE_P, pairs)
        assert "skip this menu" in msg and kb is None

    def test_exact_scores_and_totals_in_sheet(self):
        from scripts.pipeline.telegram_bot import _wc_parse_odds_sheet
        pairs = _wc_parse_odds_sheet(
            FAKE_P, "1-0 at 8.0, over 2.5 at 2.10, btts no at 1.85")
        keys = [k for k, *_ in pairs]
        assert keys == ["CS:1:0", "O25", "BTTSN"]

    def test_cs_settlement(self):
        key = "CS:1:0"
        _, sh, sa = key.split(":")
        assert (1 == int(sh) and 0 == int(sa)) is True
        assert (2 == int(sh) and 0 == int(sa)) is False


STAKE_RE = r"^(\d+(?:[.,]\d+)?)\s*@\s*(\d+(?:[.,]\d+)?)\s*(.*)$"


class TestStakeParser:
    @pytest.mark.parametrize("text,stake,odds,rest", [
        ("60 @ 1.80", 60.0, 1.80, ""),
        ("60@1.8", 60.0, 1.8, ""),
        ("12,50 @ 2,43", 12.5, 2.43, ""),
        ("60 @ 1.80 Canada win", 60.0, 1.8, "Canada win"),
    ])
    def test_valid(self, text, stake, odds, rest):
        m = re.match(STAKE_RE, text.strip())
        assert m
        assert float(m.group(1).replace(",", ".")) == stake
        assert float(m.group(2).replace(",", ".")) == odds
        assert m.group(3).strip() == rest

    @pytest.mark.parametrize("text", ["hello", "@ 1.8", "60 @", "ladder risk"])
    def test_invalid(self, text):
        assert re.match(STAKE_RE, text.strip()) is None
