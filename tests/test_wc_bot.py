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
